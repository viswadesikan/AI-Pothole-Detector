"""Step 1.4 - Leak-free train/val/test split.
Near-duplicate images (including mirrored copies) are clustered and each cluster
stays inside a single split.
Run from the project root:  python src/make_splits.py
"""
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "yolo_all"
DST = ROOT / "data" / "yolo"
REPORTS = ROOT / "reports"

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
HASH_THRESHOLD = 6  # max differing bits (out of 64) to call two images near-duplicates


def dhash(im):
    """64-bit difference hash of an image."""
    gray = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def find_clusters(names):
    """Group near-duplicate images (also catches mirrored copies)."""
    hashes, flipped = [], []
    for name in tqdm(names, desc="Hashing images"):
        with Image.open(SRC / "images" / name) as im:
            im = im.convert("RGB")
            hashes.append(dhash(im))
            flipped.append(dhash(ImageOps.mirror(im)))

    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = min((hashes[i] ^ hashes[j]).bit_count(),
                    (hashes[i] ^ flipped[j]).bit_count())
            if d <= HASH_THRESHOLD:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(len(names)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def main():
    with open(SRC / "index.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["image"] for r in rows]
    n_boxes = {r["image"]: int(r["n_boxes"]) for r in rows}
    n = len(names)

    clusters = find_clusters(names)
    rng = random.Random(SEED)
    rng.shuffle(clusters)
    clusters.sort(key=len, reverse=True)  # big clusters first; ties stay shuffled

    n_val = round(n * RATIOS["val"])
    n_test = round(n * RATIOS["test"])
    targets = {"train": n - n_val - n_test, "val": n_val, "test": n_test}
    counts = Counter()
    assign = {}  # image name -> (split, cluster id)
    for cid, members in enumerate(clusters):
        split = max(targets, key=lambda s: targets[s] - counts[s])
        for i in members:
            assign[names[i]] = (split, cid)
        counts[split] += len(members)

    # fresh output: only these generated folders are cleared
    for sub in ("images", "labels"):
        shutil.rmtree(DST / sub, ignore_errors=True)
        for split in RATIOS:
            (DST / sub / split).mkdir(parents=True)

    for name, (split, _) in tqdm(assign.items(), desc="Copying files"):
        stem = Path(name).stem
        shutil.copy2(SRC / "images" / name, DST / "images" / split / name)
        shutil.copy2(SRC / "labels" / f"{stem}.txt", DST / "labels" / split / f"{stem}.txt")

    # manifest
    with open(DST / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "split", "cluster_id", "n_boxes"])
        for name, (split, cid) in sorted(assign.items()):
            writer.writerow([name, split, cid, n_boxes[name]])

    # data.yaml (absolute path avoids Ultralytics path-resolution surprises)
    config = {
        "path": DST.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "pothole"},
    }
    (DST / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    # duplicate clusters for manual review
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "duplicate_clusters.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "size", "images"])
        for cid, members in enumerate(clusters):
            if len(members) > 1:
                writer.writerow([cid, len(members), " ".join(names[i] for i in members)])

    # summary
    multi = [c for c in clusters if len(c) > 1]
    out = ["--- Split summary ---", f"Seed: {SEED}   Hash threshold: {HASH_THRESHOLD}",
           f"Images: {n}   Clusters: {len(clusters)}   "
           f"Clusters with >1 image: {len(multi)}   "
           f"Largest cluster: {max(len(c) for c in clusters)}", ""]
    for split in RATIOS:
        imgs = [nm for nm, (s, _) in assign.items() if s == split]
        boxes = sum(n_boxes[nm] for nm in imgs)
        out.append(f"{split:5s}: {len(imgs):4d} images  {boxes:5d} boxes  "
                   f"{boxes / len(imgs):.2f} boxes/image")
    text = "\n".join(out)
    (REPORTS / "split_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\nSaved: data/yolo/data.yaml, data/yolo/split_manifest.csv, "
          "reports/duplicate_clusters.csv, reports/split_summary.txt")


if __name__ == "__main__":
    main()
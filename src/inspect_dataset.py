"""Step 1.2 - Inspect the raw dataset (read-only).
Run from the project root:  python src/inspect_dataset.py
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "dataset_inspection.txt"
IMG_EXT = {".jpg", ".jpeg", ".png"}

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def head(path, n=15):
    if path.exists():
        log(f"--- {path.name} (first {n} lines) ---")
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in content[:n]:
            log("  " + line)
    else:
        log(f"!! {path.name} not found")
    log()


def main():
    log(f"RAW folder: {RAW}")
    log("Top level: " + ", ".join(sorted(p.name for p in RAW.iterdir())))
    log()
    head(RAW / "README.md")
    head(RAW / "Licence.md")

    # ---- classes from meta.json
    meta = json.loads((RAW / "meta.json").read_text(encoding="utf-8"))
    log("--- Classes in meta.json ---")
    for c in meta.get("classes", []):
        log(f"  {c.get('title')}   shape={c.get('shape')}")
    log()

    # ---- layout
    for folder in ("img", "ann"):
        base = RAW / folder
        subdirs = sorted({p.parent.relative_to(base).as_posix()
                          for p in base.rglob("*") if p.is_file()})
        log(f"{folder}/ subfolders that contain files: {subdirs}")
    log()

    # ---- files and matching
    img_files = [p for p in (RAW / "img").rglob("*") if p.suffix.lower() in IMG_EXT]
    ann_files = list((RAW / "ann").rglob("*.json"))
    log(f"Images: {len(img_files)}   Annotation files: {len(ann_files)}")

    name_counts = Counter(p.name for p in img_files)
    dups = [n for n, c in name_counts.items() if c > 1]
    log(f"Duplicate image filenames: {len(dups)} {dups[:5]}")

    img_by_name = {p.name: p for p in img_files}
    img_by_stem = {p.stem: p for p in img_files}
    matched, orphan_ann = {}, []
    for a in ann_files:
        base = a.name[:-5]  # drop ".json"
        img = img_by_name.get(base) or img_by_stem.get(Path(base).stem)
        if img:
            matched[a] = img
        else:
            orphan_ann.append(a.name)
    orphan_imgs = set(img_files) - set(matched.values())
    log(f"Matched pairs: {len(matched)}   "
        f"Annotations without image: {len(orphan_ann)}   "
        f"Images without annotation: {len(orphan_imgs)}")
    log()

    # ---- statistics
    class_counts = Counter()
    geom_counts = Counter()
    per_group = defaultdict(Counter)
    size_counts = Counter()
    empty_images = 0
    example = None

    for a, img in tqdm(matched.items(), desc="Reading annotations"):
        data = json.loads(a.read_text(encoding="utf-8"))
        objs = data.get("objects", [])
        size = data.get("size", {})
        size_counts[(size.get("width"), size.get("height"))] += 1
        group = img.stem.rsplit("_", 1)[0]  # e.g. India, Czech, China_Drone
        per_group[group]["images"] += 1
        if not objs:
            empty_images += 1
        elif example is None:
            example = data
        for o in objs:
            cls = o.get("classTitle")
            class_counts[cls] += 1
            geom_counts[(cls, o.get("geometryType"))] += 1
            per_group[group][cls] += 1

    log()
    log("--- Objects per class ---")
    for cls, n in class_counts.most_common():
        log(f"  {cls}: {n}")
    log()
    log("--- Geometry types per class ---")
    for (cls, g), n in sorted(geom_counts.items(), key=lambda x: str(x[0])):
        log(f"  {cls} / {g}: {n}")
    log()
    log(f"Images with no objects: {empty_images}")
    log("Most common image sizes (w, h): " + str(size_counts.most_common(5)))
    log()
    log("--- Per group (filename prefix) ---")
    for group, c in sorted(per_group.items()):
        log(f"  {group}: {dict(c)}")
    log()

    # ---- schema example
    if example:
        log("--- Example annotation (first with objects) ---")
        log("Top-level keys: " + str(list(example.keys())))
        log(json.dumps(example, indent=2)[:1800])
        log()

    # ---- check ann size vs real image size on a random sample
    sample = random.Random(0).sample(list(matched.items()), min(20, len(matched)))
    mismatches = 0
    for a, img in sample:
        data = json.loads(a.read_text(encoding="utf-8"))
        w, h = Image.open(img).size
        s = data.get("size", {})
        if (s.get("width"), s.get("height")) != (w, h):
            mismatches += 1
            log(f"  size mismatch: {img.name} ann={s} real={(w, h)}")
    log(f"Size check on {len(sample)} random images: {mismatches} mismatches")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nSaved to {REPORT}")


if __name__ == "__main__":
    main()
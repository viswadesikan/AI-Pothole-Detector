"""Step 1.5 - Validate data/yolo and draw sample boxes.
Run from the project root:  python src/validate_dataset.py
"""
import random
import statistics
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo"
SAMPLES = ROOT / "reports" / "samples"
REPORT = ROOT / "reports" / "validation_summary.txt"

SPLITS = ("train", "val", "test")
N_SAMPLES = 20
SEED = 0
TOL = 1e-5
SMALL_BOX = 0.05
IMG_EXT = {".jpg", ".jpeg", ".png"}


def read_labels(path):
    """Return (boxes, problems) for one label file."""
    boxes, problems = [], []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(set(lines)) != len(lines):
        problems.append(f"{path.name}: duplicate box lines")
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{path.name}:{n} expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(f"{path.name}:{n} non-numeric value")
            continue
        if cls != 0:
            problems.append(f"{path.name}:{n} class {cls} (expected 0)")
        if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
            problems.append(f"{path.name}:{n} value outside 0-1")
        if bw <= 0 or bh <= 0:
            problems.append(f"{path.name}:{n} zero-size box")
        if (xc - bw / 2 < -TOL or xc + bw / 2 > 1 + TOL
                or yc - bh / 2 < -TOL or yc + bh / 2 > 1 + TOL):
            problems.append(f"{path.name}:{n} box extends outside the image")
        boxes.append((xc, yc, bw, bh))
    return boxes, problems


def draw_samples(split, images):
    out_dir = SAMPLES / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    chosen = random.Random(SEED).sample(images, min(N_SAMPLES, len(images)))
    tiles = []
    for p in chosen:
        with Image.open(p) as im:
            im = im.convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        boxes, _ = read_labels(DATA / "labels" / split / f"{p.stem}.txt")
        for xc, yc, bw, bh in boxes:
            draw.rectangle([(xc - bw / 2) * w, (yc - bh / 2) * h,
                            (xc + bw / 2) * w, (yc + bh / 2) * h],
                           outline=(255, 0, 0), width=2)
        im.save(out_dir / f"{p.stem}.jpg", quality=92)
        tiles.append(im)

    # contact sheet: 5 columns, 256 px cells
    cols, cell = 5, 256
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (40, 40, 40))
    for i, tile in enumerate(tiles):
        thumb = tile.copy()
        thumb.thumbnail((cell - 4, cell - 4))
        sheet.paste(thumb, ((i % cols) * cell + 2, (i // cols) * cell + 2))
    sheet.save(SAMPLES / f"sheet_{split}.jpg", quality=92)


def main():
    out, problems = ["--- Validation summary ---"], []

    cfg = yaml.safe_load((DATA / "data.yaml").read_text(encoding="utf-8"))
    out.append(f"data.yaml: nc={cfg.get('nc')} names={cfg.get('names')}")
    if cfg.get("nc") != 1:
        problems.append("data.yaml: nc is not 1")

    names_by_split = {}
    for split in SPLITS:
        images = sorted(p for p in (DATA / "images" / split).iterdir()
                        if p.suffix.lower() in IMG_EXT)
        labels = sorted((DATA / "labels" / split).glob("*.txt"))
        names_by_split[split] = {p.name for p in images}

        img_stems = {p.stem for p in images}
        lab_stems = {p.stem for p in labels}
        for s in sorted(img_stems - lab_stems):
            problems.append(f"{split}: image without label: {s}")
        for s in sorted(lab_stems - img_stems):
            problems.append(f"{split}: label without image: {s}")

        widths, heights, n_boxes = [], [], 0
        for lab in tqdm(labels, desc=f"Checking {split}", leave=False):
            boxes, probs = read_labels(lab)
            problems.extend(f"{split}/{p}" for p in probs)
            if not boxes:
                problems.append(f"{split}: empty label file: {lab.name}")
            n_boxes += len(boxes)
            widths += [b[2] for b in boxes]
            heights += [b[3] for b in boxes]

        small = sum(1 for bw, bh in zip(widths, heights) if max(bw, bh) < SMALL_BOX)
        out.append(
            f"{split:5s}: {len(images):4d} images  {len(labels):4d} labels  "
            f"{n_boxes:5d} boxes  {n_boxes / max(len(images), 1):.2f} boxes/image  "
            f"median box w/h {statistics.median(widths):.3f}/{statistics.median(heights):.3f}  "
            f"small boxes {small} ({small / max(n_boxes, 1):.0%})")
        draw_samples(split, images)

    # leakage by filename
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = names_by_split[a] & names_by_split[b]
        out.append(f"Filename overlap {a}/{b}: {len(shared)}")
        if shared:
            problems.append(f"{len(shared)} filenames appear in both {a} and {b}")

    out.append("")
    if problems:
        out.append(f"RESULT: FAIL - {len(problems)} problem(s)")
        out.extend("  " + p for p in problems[:30])
    else:
        out.append("RESULT: PASS - no problems found")
    out.append("Sample images: reports/samples/ (see sheet_train.jpg, sheet_val.jpg, sheet_test.jpg)")

    text = "\n".join(out)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
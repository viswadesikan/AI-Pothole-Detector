"""Step 1.3 - Convert Supervisely JSON annotations to YOLO format (pothole only).
Run from the project root:  python src/convert_to_yolo.py
data/raw is never modified; output goes to data/yolo_all.
"""
import csv
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "yolo_all"
REPORT = ROOT / "reports" / "conversion_summary.txt"

TARGET_CLASS = "pothole"   # exact classTitle used in the annotations
CLASS_ID = 0               # YOLO class index
MIN_SIDE_PX = 2            # drop boxes thinner than this (degenerate)
SMALL_BOX = 0.05           # "small" = longest side under 5% of the image
IMG_EXT = {".jpg", ".jpeg", ".png"}


def to_yolo(points, w, h):
    """Pixel corners -> normalized (xc, yc, bw, bh); None if the box is unusable."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))
    if (x2 - x1) < MIN_SIDE_PX or (y2 - y1) < MIN_SIDE_PX:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def main():
    img_files = [p for p in (RAW / "img").rglob("*") if p.suffix.lower() in IMG_EXT]
    img_by_name = {p.name: p for p in img_files}
    img_by_stem = {p.stem: p for p in img_files}
    ann_files = sorted((RAW / "ann").rglob("*.json"))

    # fresh output: only these two generated folders are cleared
    for sub in ("images", "labels"):
        folder = OUT / sub
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    stats = Counter()
    box_w, box_h, per_image, rows = [], [], [], []

    for ann in tqdm(ann_files, desc="Converting"):
        base = ann.name[:-5]  # drop ".json"
        img_path = img_by_name.get(base) or img_by_stem.get(Path(base).stem)
        if img_path is None:
            stats["annotation_without_image"] += 1
            continue

        data = json.loads(ann.read_text(encoding="utf-8"))
        w, h = data["size"]["width"], data["size"]["height"]
        with Image.open(img_path) as im:
            if im.size != (w, h):
                stats["size_mismatch_skipped"] += 1
                continue

        lines = []
        for obj in data.get("objects", []):
            if obj.get("classTitle") != TARGET_CLASS:
                stats["other_class_skipped"] += 1
                continue
            if obj.get("geometryType") != "rectangle":
                stats["non_rectangle_skipped"] += 1
                continue
            box = to_yolo(obj["points"]["exterior"], w, h)
            if box is None:
                stats["degenerate_box_dropped"] += 1
                continue
            lines.append(f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in box))
            box_w.append(box[2])
            box_h.append(box[3])

        if not lines:
            stats["image_without_valid_box_skipped"] += 1
            continue

        shutil.copy2(img_path, OUT / "images" / img_path.name)
        label_path = OUT / "labels" / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["images_written"] += 1
        stats["boxes_written"] += len(lines)
        per_image.append(len(lines))
        rows.append((img_path.name, w, h, len(lines)))

    # manifest used by the split script
    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "width", "height", "n_boxes"])
        writer.writerows(rows)

    # summary
    out = ["--- Conversion summary ---"]
    for key, value in sorted(stats.items()):
        out.append(f"{key}: {value}")
    if box_w:
        small = sum(1 for bw, bh in zip(box_w, box_h) if max(bw, bh) < SMALL_BOX)
        out.append("")
        out.append("Box width  (normalized) min/median/max: "
                   f"{min(box_w):.3f} / {statistics.median(box_w):.3f} / {max(box_w):.3f}")
        out.append("Box height (normalized) min/median/max: "
                   f"{min(box_h):.3f} / {statistics.median(box_h):.3f} / {max(box_h):.3f}")
        out.append(f"Small boxes (longest side < {SMALL_BOX:.0%} of image): "
                   f"{small} of {len(box_w)}")
        out.append(f"Boxes per image min/median/max: {min(per_image)} / "
                   f"{statistics.median(per_image)} / {max(per_image)}")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved to {REPORT}")


if __name__ == "__main__":
    main()
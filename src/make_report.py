"""Step 1.6 - Build reports/dataset_report.md from the actual data.
Run from the project root:  python src/make_report.py
"""
import json
import statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data" / "yolo"
REPORTS = ROOT / "reports"
SPLITS = ("train", "val", "test")
SMALL_BOX = 0.05


def read_text(path, lines=None):
    if not path.exists():
        return f"(file not found: {path.name})"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[:lines] if lines else content)


def raw_counts():
    images = [p for p in (RAW / "img").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    boxes = 0
    for ann in (RAW / "ann").rglob("*.json"):
        data = json.loads(ann.read_text(encoding="utf-8"))
        boxes += sum(1 for o in data.get("objects", []) if o.get("classTitle") == "pothole")
    return len(images), boxes


def split_stats(split):
    widths, heights, n_img = [], [], 0
    for lab in (DATA / "labels" / split).glob("*.txt"):
        n_img += 1
        for line in lab.read_text(encoding="utf-8").splitlines():
            if line.strip():
                _, _, _, bw, bh = line.split()
                widths.append(float(bw))
                heights.append(float(bh))
    small = sum(1 for w, h in zip(widths, heights) if max(w, h) < SMALL_BOX)
    return {
        "images": n_img, "boxes": len(widths),
        "per_image": len(widths) / n_img,
        "w": statistics.median(widths), "h": statistics.median(heights),
        "small": small, "small_pct": small / len(widths),
    }


def main():
    raw_images, raw_boxes = raw_counts()
    stats = {s: split_stats(s) for s in SPLITS}
    total_images = sum(s["images"] for s in stats.values())
    total_boxes = sum(s["boxes"] for s in stats.values())

    clusters_file = REPORTS / "duplicate_clusters.csv"
    n_clusters = max(len(clusters_file.read_text(encoding="utf-8").splitlines()) - 1, 0) \
        if clusters_file.exists() else 0

    md = []
    md.append(f"# Dataset Report - RoadGuard AI\n\nGenerated: {date.today()}\n")
    md.append("## 1. Source and License\n")
    md.append("Dataset folder: `data/raw` (annotations in Supervisely JSON format).\n")
    md.append("**README (first lines):**\n\n```\n" + read_text(RAW / "README.md", 15) + "\n```\n")
    md.append("**LICENSE (first lines):**\n\n```\n" + read_text(RAW / "LICENSE.md", 15) + "\n```\n")
    md.append("> Check the full `LICENSE.md` and cite the dataset as it requires.\n")

    md.append("## 2. Overview\n")
    md.append("| Item | Value |\n|---|---|")
    md.append(f"| Raw images | {raw_images} |")
    md.append(f"| Raw pothole boxes | {raw_boxes} |")
    md.append(f"| Images after conversion | {total_images} |")
    md.append(f"| Boxes after conversion | {total_boxes} |")
    md.append(f"| Boxes dropped | {raw_boxes - total_boxes} (degenerate: under 2 px) |")
    md.append("| Classes | 1 (`pothole`, class id 0) |\n")

    md.append("## 3. Splits\n")
    md.append("| Split | Images | Boxes | Boxes/image | Median box w / h | Small boxes |\n|---|---|---|---|---|---|")
    for s in SPLITS:
        d = stats[s]
        md.append(f"| {s} | {d['images']} | {d['boxes']} | {d['per_image']:.2f} | "
                  f"{d['w']:.3f} / {d['h']:.3f} | {d['small']} ({d['small_pct']:.0%}) |")
    md.append("\nBox sizes are fractions of the image. Small = longest side under 5% of the image.\n")

    md.append("## 4. Leakage Control\n")
    md.append("Filenames carry no location or video information, so a location-based split was not possible. "
              "Images were fingerprinted (dHash, including mirrored versions), near-duplicates were clustered, "
              f"and each cluster was kept inside one split. Clusters with more than one image: **{n_clusters}** "
              "(see `reports/duplicate_clusters.csv`).\n")
    md.append("```\n" + read_text(REPORTS / "split_summary.txt") + "\n```\n")

    md.append("## 5. Known Limitations\n")
    md.append("- Small dataset: a few hundred images, so results will vary between splits.\n"
              "- **No negative images:** every image contains at least one pothole, so false positives on "
              "shadows, patches, and manholes cannot be measured yet.\n"
              "- Images are mostly 400x400 or 400x300 and are not dashcam footage; potholes fill a large part "
              "of each image. Real dashcam potholes will be smaller.\n"
              "- The split cannot detect different photos of the same pothole from different angles.\n"
              "- No performance numbers exist yet: nothing has been trained.\n")

    md.append("## 6. Reproduce\n\n```\n"
              "python src/inspect_dataset.py\npython src/convert_to_yolo.py\n"
              "python src/make_splits.py\npython src/validate_dataset.py\npython src/make_report.py\n```\n")

    (REPORTS / "dataset_report.md").write_text("\n".join(md), encoding="utf-8")
    print("Saved reports/dataset_report.md")
    print(f"Raw: {raw_images} images / {raw_boxes} boxes   ->   Final: {total_images} images / {total_boxes} boxes")


if __name__ == "__main__":
    main()
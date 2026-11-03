"""Run the project's one-time final evaluation on the held-out TEST split.

Run from the project root with:
  .venv\\Scripts\\python.exe src\\evaluate_test.py

This script is intentionally single-use. It refuses to run if its output folder
already exists. Use only after the candidate model and confidence threshold have
been selected from the validation split.
"""
import argparse
import gc
from pathlib import Path

import torch
from ultralytics import YOLO

from analyze_errors import IOU_T, match, predict_all, read_gt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo"
DATA_YAML = DATA / "data.yaml"
WEIGHTS = ROOT / "models" / "runs" / "baseline_yolo26n" / "weights" / "best.pt"
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "final_test_evaluation"
CONF_THRESHOLD = 0.40  # selected on validation for baseline_yolo26n
IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def validate_inputs():
    if OUT_DIR.exists():
        raise SystemExit(
            f"Refusing to evaluate the TEST split again because {OUT_DIR} already exists."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available; no TEST data was evaluated.")
    if not DATA_YAML.is_file():
        raise SystemExit(f"Missing dataset config: {DATA_YAML}")
    if not WEIGHTS.is_file():
        raise SystemExit(f"Missing selected baseline weights: {WEIGHTS}")

    image_dir = DATA / "images" / "test"
    label_dir = DATA / "labels" / "test"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit("The prepared TEST image or label folder is missing.")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit("No TEST images found; no TEST data was evaluated.")
    missing_labels = [p.stem for p in images if not (label_dir / f"{p.stem}.txt").is_file()]
    if missing_labels:
        raise SystemExit(
            f"Missing TEST labels for {len(missing_labels)} image(s); no TEST data was evaluated."
        )
    return images


def fixed_threshold_counts(images, imgsz):
    label_dir = DATA / "labels" / "test"
    ground_truth = [read_gt(label_dir / f"{image.stem}.txt") for image in images]
    predictions = predict_all(WEIGHTS, images, imgsz)

    tp = fp = fn = 0
    gt_count = 0
    for (boxes, scores), gt in zip(predictions, ground_truth):
        _, _, is_tp, gt_hit = match(boxes, scores, gt, CONF_THRESHOLD)
        tp += int(is_tp.sum())
        fp += int((~is_tp).sum())
        fn += int((~gt_hit).sum())
        gt_count += len(gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return tp, fp, fn, gt_count, precision, recall, f1


def main():
    args = parse_args()
    images = validate_inputs()

    # Create the one-shot marker before inference so a partial run cannot
    # accidentally be repeated against the held-out TEST split.
    OUT_DIR.mkdir(parents=True)
    (OUT_DIR / "evaluation_started.txt").write_text(
        "TEST evaluation started. Do not rerun this evaluation.\n", encoding="utf-8"
    )

    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=0,
        conf=0.001,
        plots=True,
        project=str(REPORTS),
        name=OUT_DIR.name,
        exist_ok=True,
    )
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    val_precision = float(metrics.box.mp)
    val_recall = float(metrics.box.mr)

    del metrics, model
    gc.collect()
    torch.cuda.empty_cache()

    tp, fp, fn, gt_count, precision, recall, f1 = fixed_threshold_counts(
        images, args.imgsz
    )
    summary = f"""# Final test evaluation

- Model: `{WEIGHTS.relative_to(ROOT)}`
- Final model copy: `models/roadguard_best.pt`
- Images: {len(images)}
- Ground-truth potholes: {gt_count}
- Image size: {args.imgsz}
- Standard detection metrics: mAP50 **{map50:.3f}**, mAP50–95 **{map50_95:.3f}**
- Ultralytics P/R at its internal operating point: precision {val_precision:.3f}, recall {val_recall:.3f}
- Fixed confidence threshold: **{CONF_THRESHOLD:.2f}** (chosen on validation)
- Fixed-threshold matching: IoU ≥ {IOU_T:.2f}; TP {tp}, FP {fp}, FN {fn}
- Fixed-threshold precision: **{precision:.3f}**; recall: **{recall:.3f}**; F1: **{f1:.3f}**

The held-out TEST split was evaluated once. Do not use these results to tune the model or threshold.
"""
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

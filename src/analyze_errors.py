"""Step 3 - Error analysis and confidence-threshold sweep.
Run from the project root:
  python src/analyze_errors.py --weights models/runs/baseline_yolo26n/weights/best.pt
The test split is blocked unless you pass --final.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo"
OUT = ROOT / "reports" / "analysis"
IOU_T = 0.5
IMG_EXT = {".jpg", ".jpeg", ".png"}
SIZE_BINS = [("small  (<1% of image area)", 0.0, 0.01),
             ("medium (1-3% of image area)", 0.01, 0.03),
             ("large  (>3% of image area)", 0.03, 1.01)]


def read_gt(path):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, xc, yc, w, h = map(float, line.split())
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(boxes, scores, gt, conf):
    """Greedy matching at IOU_T. Returns kept boxes, scores, tp flags, gt-hit flags."""
    keep = scores >= conf
    b, s = boxes[keep], scores[keep]
    order = np.argsort(-s)
    b, s = b[order], s[order]
    ious = iou_matrix(b, gt)
    gt_hit = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(b), dtype=bool)
    for i in range(len(b)):
        if len(gt) == 0:
            break
        cand = np.where(gt_hit, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= IOU_T:
            gt_hit[j] = True
            tp[i] = True
    return b, s, tp, gt_hit


def predict_all(weights, images, imgsz):
    model = YOLO(str(weights))
    results = model.predict(source=[str(p) for p in images], conf=0.001, imgsz=imgsz,
                            device=0, batch=8, stream=True, verbose=False)
    return [(r.boxes.xyxyn.cpu().numpy().astype(float).reshape(-1, 4),
             r.boxes.conf.cpu().numpy().astype(float)) for r in results]


def sweep(data, thresholds):
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        for boxes, scores, gt in data:
            _, _, is_tp, hit = match(boxes, scores, gt, t)
            tp += int(is_tp.sum())
            fp += int((~is_tp).sum())
            fn += int((~hit).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        rows.append((float(t), tp, fp, fn, p, r, 2 * p * r / max(p + r, 1e-9)))
    return rows


def draw_errors(images, data, conf, out_dir, limit):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    size_stats = {name: [0, 0] for name, _, _ in SIZE_BINS}  # [total, missed]
    bad = []
    for path, (boxes, scores, gt) in zip(images, data):
        b, s, tp, hit = match(boxes, scores, gt, conf)
        for g, matched in zip(gt, hit):
            area = (g[2] - g[0]) * (g[3] - g[1])
            for name, lo, hi in SIZE_BINS:
                if lo <= area < hi:
                    size_stats[name][0] += 1
                    size_stats[name][1] += int(not matched)
        n_fn, n_fp = int((~hit).sum()), int((~tp).sum())
        if n_fn or n_fp:
            bad.append((n_fn + n_fp, n_fn, n_fp, path, b, s, tp, gt, hit))
    bad.sort(key=lambda x: -x[0])
    for _, n_fn, n_fp, path, b, s, tp, gt, hit in bad[:limit]:
        im = Image.open(path).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for g, matched in zip(gt, hit):  # green = found, yellow = MISSED
            d.rectangle([g[0] * W, g[1] * H, g[2] * W, g[3] * H],
                        outline=(0, 200, 0) if matched else (255, 215, 0),
                        width=2 if matched else 3)
        for box, sc, ok in zip(b, s, tp):  # red = FALSE ALARM
            if not ok:
                d.rectangle([box[0] * W, box[1] * H, box[2] * W, box[3] * H],
                            outline=(255, 0, 0), width=3)
                d.text((box[0] * W + 3, box[1] * H + 3), f"{sc:.2f}", fill=(255, 0, 0))
        im.save(out_dir / f"{n_fn}FN_{n_fp}FP_{path.stem}.jpg", quality=92)
    return size_stats, len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=None,
                    help="threshold for the error images (default: best F1)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=40, help="max error images to save")
    ap.add_argument("--final", action="store_true", help="allow the test split")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        raise SystemExit("The test split is reserved for the final evaluation. "
                         "Use --final only once, on the chosen model.")

    weights = Path(a.weights)
    run = weights.parents[1].name
    images = sorted(p for p in (DATA / "images" / a.split).iterdir()
                    if p.suffix.lower() in IMG_EXT)
    gts = [read_gt(DATA / "labels" / a.split / f"{p.stem}.txt") for p in images]
    preds = predict_all(weights, images, a.imgsz)
    data = [(b, s, g) for (b, s), g in zip(preds, gts)]

    rows = sweep(data, np.round(np.arange(0.05, 0.96, 0.05), 2))
    best = max(rows, key=lambda r: r[6])
    conf = a.conf if a.conf is not None else best[0]

    out_dir = OUT / f"{run}_{a.split}"
    size_stats, n_bad = draw_errors(images, data, conf, out_dir / "errors", a.limit)

    with open(out_dir / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conf", "TP", "FP", "FN", "precision", "recall", "f1"])
        w.writerows(rows)

    lines = [f"Run: {run}   Split: {a.split}   Images: {len(images)}   IoU>={IOU_T}", "",
             "conf    TP    FP    FN   precision  recall    F1"]
    for t, tp, fp, fn, p, r, f1 in rows:
        mark = "  <- best F1" if t == best[0] else ""
        lines.append(f"{t:4.2f}  {tp:4d}  {fp:4d}  {fn:4d}   {p:7.3f}  {r:6.3f}  {f1:6.3f}{mark}")
    lines += ["", f"Error images use conf = {conf:.2f}: {n_bad} images have errors "
              f"(top {min(a.limit, n_bad)} saved to {out_dir / 'errors'})",
              "Image names read <missed>FN_<false alarms>FP_<name>.",
              "Colours: green = found, YELLOW = missed potholes, RED = false alarms (with score).", "",
              "Missed potholes by size:"]
    for name, (total, missed) in size_stats.items():
        lines.append(f"  {name}: {missed} of {total} missed"
                     f" ({missed / max(total, 1):.0%})")
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
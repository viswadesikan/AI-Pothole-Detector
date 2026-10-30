"""Step 2 - Train the baseline pothole detector.
Run from the project root:
  python src/train_baseline.py --smoke               # 2-epoch pipeline test
  python src/train_baseline.py                       # full baseline (pretrained weights)
  python src/train_baseline.py --scratch --epochs 200 --name scratch_yolo26n
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo" / "data.yaml"
RUNS = ROOT / "models" / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n", help="e.g. yolo26n, yolo26s")
    p.add_argument("--scratch", action="store_true", help="random init, no pretrained weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="lower to 8 or 4 if GPU memory runs out")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=30, help="stop if val does not improve")
    p.add_argument("--name", default=None)
    p.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Fix 2.1 before training.")
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}. Re-run src/make_splits.py.")

    name = a.name or (f"baseline_{a.model}" + ("_scratch" if a.scratch else ""))
    epochs, fraction = a.epochs, 1.0
    if a.smoke:
        name, epochs, fraction = "smoke_test", 2, 0.25

    model = YOLO(f"{a.model}.yaml" if a.scratch else f"{a.model}.pt")
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,      # keep low on Windows
        patience=a.patience,
        device=0,
        project=str(RUNS),
        name=name,
        exist_ok=a.smoke,       # smoke test may overwrite itself
        pretrained=not a.scratch,
        seed=a.seed,
        deterministic=True,
        cache="ram",            # the dataset is small, so cache it
        fraction=fraction,
        plots=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nRun folder : {save_dir}")
    print(f"Best weights: {best}")

    # score the best weights on the VALIDATION split (the test split stays untouched until Step 3)
    m = YOLO(str(best)).val(data=str(DATA), split="val", imgsz=a.imgsz,
                            batch=a.batch, workers=a.workers, plots=True,
                            project=str(RUNS), name=f"{name}_val", exist_ok=True)
    print("\n--- Validation (best.pt) ---")
    print(f"Precision {m.box.mp:.3f} | Recall {m.box.mr:.3f} | "
          f"mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f}")


if __name__ == "__main__":  # required on Windows (dataloader workers)
    main()
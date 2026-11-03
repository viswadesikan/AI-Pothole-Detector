# RoadGuard AI - Project Context (read this first)

## Goal
Dashcam pothole detection -> driver warning -> GPS-tagged, de-duplicated reports -> internal map.
Plan has 14 steps (0-13). Done: 0, 1, 2. In progress: 3 (evaluation). Next: 4 (own footage + negative images).

## Environment
- Windows 11, PowerShell, VS Code. Project folder: D:\Pyspiders Training\Machine Learning\AI Pathole detector
- Python 3.14.6, venv in .venv. ALWAYS use `.venv\Scripts\python.exe` (or activate the venv first).
- Ultralytics 8.4.193, torch 2.14.1+cu130, GPU: RTX 3050 Laptop, 4 GB VRAM.
- On Windows: dataloader workers <= 2, and training scripts need `if __name__ == "__main__":`.
- If out of GPU memory: lower --batch (8, then 4). Close other GPU apps first.

## Folder structure (simple, flat)
- data/raw/          Kaggle dataset, NEVER modify (ann/, img/, meta.json, README.md, LICENSE.md)
- data/yolo_all/     converted images + YOLO labels (all 665) + index.csv
- data/yolo/         final dataset: images/{train,val,test}, labels/{train,val,test}, data.yaml, split_manifest.csv
- src/               all scripts
- models/runs/       training runs (git-ignored); models/*.pt = weight backups
- reports/           reports, experiments.md, analysis/, samples/
- Also: .gitignore, README.md, requirements.txt, requirements-train.txt
(A controller/service/repository app design exists but is deferred until the inference pipeline, Step 5.)

## Dataset facts
- Dataset Ninja "Pothole Detection" (NOT RDD2022): 665 images, 1,740 boxes, ONE class: pothole (class id 0).
- Annotations: Supervisely JSON, one file per image, box = points.exterior [[x1,y1],[x2,y2]] in pixels.
- Images mostly 400x400 / 400x300; potholes are large in frame (median box ~19% x 12% of image).
- NO negative images (every image has >=1 pothole). Not dashcam footage -> domain gap.
- 1 degenerate box (<2 px) dropped in conversion -> 1,739 boxes.
- Split (70/15/15, near-duplicate clusters kept together via dHash, seed 42): 
  train 465 imgs / 1,214 boxes; val 100 / 288; test 100 / 237.
- License text in data/raw/LICENSE.md has not been reviewed yet: read it and cite the dataset properly.

## Scripts in src/
- inspect_dataset.py   read-only inspection of data/raw (NOTE: looks for Licence.md; the file is LICENSE.md)
- convert_to_yolo.py   JSON -> YOLO labels, pothole only -> data/yolo_all
- make_splits.py       leak-free split -> data/yolo + data.yaml (absolute path inside)
- validate_dataset.py  format checks + sample images/contact sheets in reports/samples
- make_report.py       builds reports/dataset_report.md
- train_baseline.py    args: --model --scratch --epochs --imgsz --batch --workers --patience --name --smoke
                       (includes --seed; .val() output is routed under models/runs)
- analyze_errors.py    error analysis + confidence sweep on VAL; test split blocked without --final
- evaluate_test.py     final TEST evaluation; run ONCE on the chosen model only

## Experiment results (VALIDATION split, 100 images, so about +-0.02 mAP50 is noise)
| # | Run | Init | Epochs (best) | imgsz | batch | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline_yolo26n | pretrained | 100 | 640 | 16 | 0.864 | 0.729 | 0.825 | 0.538 |
| 2 | scratch_yolo26n | random init | 200 (170) | 640 | 16 | 0.818 | 0.687 | 0.771 | 0.442 |
| 3 | yolo26n_img800 | pretrained | 80 (50) | 800 | 8 | 0.770 | 0.819 | 0.837 | 0.543 |
Notes: pretrained beats scratch (esp. mAP50-95). 800 px is a tie with 640 within noise (batch also changed).
P/R printed by Ultralytics are at one internal threshold, so compare mAP or use the sweep. Baseline ran all 100 epochs
(best epoch >= 71), so it may be undertrained. Test split has NOT been used yet.
Weights: models/runs/<run>/weights/best.pt

## Latest continuation (2026-11-04)
- The optional 300-epoch baseline run stopped after epoch 162. Its `last.pt` had `epoch=-1` and no optimizer state, so Ultralytics could not resume it exactly.
- A separate fine-tune initialized from that checkpoint ran 70 epochs and early-stopped (best epoch 20): P=0.824, R=0.722, mAP50=0.794, mAP50-95=0.512 on validation. It did not beat the original baseline. See `reports/experiments.md`.
- Keep the original baseline candidate; do not describe the fine-tune as a 300-epoch run or as an exact resume.

## Open items (in order)
1. Done: visually review reports/samples/sheet_*.jpg and read data/raw/LICENSE.md (DbCL v1.0). Add a proper dataset citation to the project README before sharing.
2. Update and review reports/experiments.md, then commit the completed evaluation task.
3. Done: analyze_errors.py and confidence sweeps exist for baseline (640) and img800 (800) at the same thresholds.
4. Check and record the existing seed runs (`baseline_yolo26n_seed1` and `baseline_yolo26n_seed1-2`) before deciding whether another seed is needed.
5. The 300-epoch run cannot be resumed because optimizer state is missing. Its fine-tune did not improve the baseline; STOP tuning and keep the original baseline candidate.
6. Run evaluate_test.py ONCE; copy weights to models/roadguard_best.pt; record the chosen confidence threshold.
7. Step 4: collect own dashcam footage + negative (no-pothole) images, label a small set, test the model on it.
8. Steps 5-13: video inference, tracking/dedup, warning UI, GPS, SQLite events, Leaflet map, reporting adapter
   (external submission ONLY via an official authorized integration), end-to-end tests, docs.

## Rules for the coding assistant
- Never modify or delete anything in data/raw. Never touch the TEST split except via evaluate_test.py (once).
- One change per experiment; log every run in reports/experiments.md; never invent numbers.
- Work on one task at a time, state a short plan first, then stop and report. Commit after each completed task.
- Scripts go in src/, use argparse, resolve paths from the project root, and are safe to re-run.
- If anything contradicts this file, stop and tell me instead of guessing.
- Ultralytics is AGPL-3.0; keep that in mind for distribution.

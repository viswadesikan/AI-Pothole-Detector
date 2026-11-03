# Experiments

All metrics below are from the validation split. The 100-image validation set is small, so differences around 0.02 mAP can be noise. The held-out test split has not been used.

| # | Name | Model | Initialization | Epochs run (best) | imgsz | Batch | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | baseline_yolo26n | yolo26n | pretrained | 100 (best epoch not recorded) | 640 | 16 | 0.864 | 0.729 | 0.825 | 0.538 | Validation split; ran all 100 epochs. |
| 2 | scratch_yolo26n | yolo26n | random initialization | 200 (best 170) | 640 | 16 | 0.818 | 0.687 | 0.771 | 0.442 | Validation split; no pretrained weights. |
| 3 | yolo26n_img800 | yolo26n | pretrained | 80 (best 50) | 800 | 8 | 0.770 | 0.819 | 0.837 | 0.543 | Validation split; higher resolution and smaller batch together, so their effects are confounded. |
| 4 | baseline_yolo26n_finetune_from162 | yolo26n | fine-tuned from the epoch-162 checkpoint | 70 (best epoch 20) | 640 | 16 | 0.824 | 0.722 | 0.794 | 0.512 | Validation split; original checkpoint lacked optimizer state, so this was a new AdamW fine-tuning run, not an exact resume. Stopped by patience 50. Lower mAP than the baseline. |

## Validation threshold analysis

`reports/analysis/baseline_yolo26n_val/` and `reports/analysis/yolo26n_img800_val/` contain error images and confidence sweeps evaluated at the same confidence thresholds. Their best F1 points were 0.793 at confidence 0.40 (640px) and 0.785 at 0.30 (800px). These thresholds are validation-only and must not be treated as test results.

## Selection

The 800px run has the highest recorded mAP50 and mAP50-95, but its different batch size and image size prevent attributing the change to resolution alone. The 640px baseline is the simpler comparison point. The fine-tune from the interrupted 300-epoch run did not improve either mAP measure. Keep the test split untouched until the evaluation script and final candidate are ready; evaluate it once.

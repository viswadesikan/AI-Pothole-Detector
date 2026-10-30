# Experiments

| # | Name             | Model   | Init       | Epochs run | imgsz | Batch | Precision | Recall | mAP50 | mAP50-95 | Notes                       |
|---|------------------|---------|------------|------------|-------|-------|-----------|--------|-------|----------|-----------------------------|
| 1 | baseline_yolo26n | yolo26n | pretrained | 100        | 640   | 16    | 0.864     | 0.729  | 0.825 | 0.538    | default settings, val split |
| 1 | baseline_yolo26n | yolo26n | pretrained | 100        | 640   | 16    | 0.864     | 0.729  | 0.825 | 0.538    | default settings, val split, ran all 100 epochs |
| 2 | scratch_yolo26n  | yolo26n |random init | 200 (best 170) | 640 | 16  | 0.818     | 0.687  | 0.771 | 0.442    | no pretrained weights, val split |
| 3 | yolo26n_img800   | yolo26n | pretrained | 80 (best 50) | 800 | 8     | 0.770     | 0.819  | 0.837 | 0.543    | imgsz 800 + batch 8; tie
+os baseline within noise |
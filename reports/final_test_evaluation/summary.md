# Final test evaluation

- Model: `models\runs\baseline_yolo26n\weights\best.pt`
- Final model copy: `models/roadguard_best.pt`
- Images: 100
- Ground-truth potholes: 237
- Image size: 640
- Standard detection metrics: mAP50 **0.798**, mAP50–95 **0.518**
- Ultralytics P/R at its internal operating point: precision 0.801, recall 0.722
- Fixed confidence threshold: **0.40** (chosen on validation)
- Fixed-threshold matching: IoU ≥ 0.50; TP 170, FP 35, FN 67
- Fixed-threshold precision: **0.829**; recall: **0.717**; F1: **0.769**

The held-out TEST split was evaluated once. Do not use these results to tune the model or threshold.

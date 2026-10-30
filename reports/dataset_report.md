# Dataset Report - RoadGuard AI

Generated: 2026-10-28

## 1. Source and License

Dataset folder: `data/raw` (annotations in Supervisely JSON format).

**README (first lines):**

```
# Pothole Detection

Pothole Detection is a dataset for object detection task.
```

**LICENSE (first lines):**

```
Pothole Detection is under [DbCL v1.0](https://opendatacommons.org/licenses/dbcl/1-0/) license.
```

> Check the full `LICENSE.md` and cite the dataset as it requires.

## 2. Overview

| Item | Value |
|---|---|
| Raw images | 665 |
| Raw pothole boxes | 1740 |
| Images after conversion | 665 |
| Boxes after conversion | 1739 |
| Boxes dropped | 1 (degenerate: under 2 px) |
| Classes | 1 (`pothole`, class id 0) |

## 3. Splits

| Split | Images | Boxes | Boxes/image | Median box w / h | Small boxes |
|---|---|---|---|---|---|
| train | 465 | 1214 | 2.61 | 0.195 / 0.120 | 34 (3%) |
| val | 100 | 288 | 2.88 | 0.174 / 0.113 | 7 (2%) |
| test | 100 | 237 | 2.37 | 0.193 / 0.120 | 7 (3%) |

Box sizes are fractions of the image. Small = longest side under 5% of the image.

## 4. Leakage Control

Filenames carry no location or video information, so a location-based split was not possible. Images were fingerprinted (dHash, including mirrored versions), near-duplicates were clustered, and each cluster was kept inside one split. Clusters with more than one image: **0** (see `reports/duplicate_clusters.csv`).

```
--- Split summary ---
Seed: 42   Hash threshold: 6
Images: 665   Clusters: 665   Clusters with >1 image: 0   Largest cluster: 1

train:  465 images   1214 boxes  2.61 boxes/image
val  :  100 images    288 boxes  2.88 boxes/image
test :  100 images    237 boxes  2.37 boxes/image
```

## 5. Known Limitations

- Small dataset: a few hundred images, so results will vary between splits.
- **No negative images:** every image contains at least one pothole, so false positives on shadows, patches, and manholes cannot be measured yet.
- Images are mostly 400x400 or 400x300 and are not dashcam footage; potholes fill a large part of each image. Real dashcam potholes will be smaller.
- The split cannot detect different photos of the same pothole from different angles.
- No performance numbers exist yet: nothing has been trained.

## 6. Reproduce

```
python src/inspect_dataset.py
python src/convert_to_yolo.py
python src/make_splits.py
python src/validate_dataset.py
python src/make_report.py
```

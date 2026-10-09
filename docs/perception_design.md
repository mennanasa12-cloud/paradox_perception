## Task 1: Dataset Preparation (LuSNAR segmentation)

### Source and subset
- Dataset: LuSNAR, 9 scenes (`Moon_1` to `Moon_9`), Hugging Face repo `JeremyLuo/LuSNAR`.
- Used: `image0/color` (RGB) and `image0/label` (semantic labels). Only these folders have labels.
  Depth, LiDAR and `image1` are not needed for perception.
- 13,014 image/label pairs in total. We keep every 5th frame, because consecutive frames
  (about 0.1 s apart) are nearly identical.

### Annotation format
- Original labels: 1024x1024 palette PNGs where each color is a class
  (see `segmentation_colors` in `configs/classes.yaml`).
- Processed masks: single-channel PNG, 512x512, pixel value = class ID.

| ID | Class | Original color |
|---|---|---|
| 0 | regolith | #BB469C |
| 1 | crater | #7800C8 |
| 2 | rock | #E8FA50 |
| 3 | mountain | #AD451F |
| 4 | sky | #22C9F8 |
| 255 | unknown color (ignored in the loss) | none found in the data |

- Images: RGB PNG, 512x512 (bilinear resize).
- Masks are resized with nearest-neighbor so class IDs are never blended.

### Splits
Split by scene, not by frame, so near-duplicate frames never appear in both train and test.

| Split | Scenes | Pairs |
|---|---|---|
| train | Moon_1 to Moon_7 | 1916 |
| val | Moon_8 | 271 |
| test | Moon_9 | 418 |

### Class balance and weights
Train pixels: regolith 68.9%, sky 24.6%, mountain 3.1%, rock 2.2%, crater 1.2%.
Rock and crater are rare, so the loss uses class weights (inverse square-root frequency, mean = 1):
`0: 0.244, 1: 1.829, 2: 1.361, 3: 1.158, 4: 0.408` (stored in `configs/classes.yaml`).
Evaluation reports per-class IoU and mIoU, not only overall accuracy.

### Augmentation strategy (train only)
Implemented in `src/perception/dataset/lusnar_seg.py`.
- Horizontal flip only. No vertical flip, because the sky is always at the top.
- Brightness/contrast and gamma changes, to mimic changing sun angle.
- Gaussian blur and noise, to mimic sensor quality.
- No hue or saturation shifts, because lunar scenes are nearly colorless.
- Validation and test use normalization only (ImageNet mean/std).

### Storage and reuse
- Processed data is not in git. It is one zip, `lusnar_processed.zip` (1.12 GB), in Google Drive
  under `paradox/`, next to `class_stats.json`.
- Layout inside the zip: `train|val|test` / `images|masks` / `Moon_<k>_<frame>.png`.
- Colab workflow: mount Drive, copy the zip to `/content/data`, unzip, then use
  `LuSNARSeg("/content/data", "train")`.
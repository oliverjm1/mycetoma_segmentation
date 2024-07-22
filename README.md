# Mycetoma Grain Segmentation

This repository contains code for segmenting bacterial and fungal mycetoma grains from histopathology images.

## Pipeline Idea So Far...

1. Visualised Data (`visualise_data.ipynb`), made a few corrections to errant files (wrong extensions, wrong mask shape etc...).
2. Dealt with duplicate images in train and validation data (`dealing_with_duplicates.ipynb`).
3. Dealt with overlapping images/masks (`combining_overlapping_masks.ipynb`).
4. Trained 2D UNet on both unaltered and altered data to compare (`plot_train_scores.ipynb`).
5. Had a go at some post-processing stages to see how this affected dice score performance (`look_at_predictions.ipynb`).

## Potential Further Steps

1. Go through train/validation data and manually remove bad examples (large areas of ground truth still not labelled).
2. Data Augmentation.
3. Domain issues (staining?).
4. Saving training/validation masks for Ben to train classifier on.

## Repo Structure

```
.
├── README.md
├── data
│   └── data.md
├── model_saves
│   └── model_saves.md
├── notebooks
│   ├── combining_overlapping_masks.ipynb
│   ├── dealing_with_duplicates.ipynb
│   ├── look_at_predictions.ipynb
│   ├── plot_train_scores.ipynb
│   ├── train_UNet.ipynb
│   └── visualise_data.ipynb
├── scripts
│   └── train_UNet.py
├── src
│   ├── UNet2D.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── postprocessing.py
│   └── utils.py
└── train_stats
    └── train_stats.md
```

# Mycetoma Grain Segmentation

This repository contains code for segmenting bacterial and fungal mycetoma grains from histopathology images.

## Pipeline Idea So Far...

1. Visualised Data (`visualise_data.ipynb`), made a few corrections to errant files (wrong extensions, wrong mask shape etc...).
2. Dealt with duplicate images in train and validation data (`dealing_with_duplicates.ipynb`).
3. Dealt with overlapping images/masks (`combining_overlapping_masks.ipynb`).
4. Trained 2D UNet (`train_UNet.py`) on both unaltered and altered data to compare (`plot_train_scores.ipynb`).
5. Had a go at some post-processing stages to see how this affected dice score performance (`look_at_predictions.ipynb`).

## Potential Further Steps

1. Go through train/validation data and manually remove bad examples (large areas of ground truth still not labelled). (DONE)
2. Data Augmentation. (DONE)
3. Domain issues (staining?). (Some colour jitter augmentation done to attempt to partially deal with this)
4. Saving training/validation masks for Ben to train classifier on. (DONE)

## Repo Structure

```
.
├── data
│   └── data.md
├── model_saves
│   └── model_saves.md
├── multitask_model_saves
├── notebooks
│   ├── combining_overlapping_masks.ipynb
│   ├── dealing_with_duplicates.ipynb
│   ├── look_at_predictions.ipynb
│   ├── overlay_edges.ipynb
│   ├── plot_train_scores.ipynb
│   ├── save_masks.ipynb
│   ├── train_UNet.ipynb
│   └── visualise_data.ipynb
├── scripts
│   ├── train_UNet.py
│   └── train_multitask.py
├── src
│   ├── UNet2D.py
│   ├── UNetMultiTask.py
│   ├── __init__.py
│   ├── datasets.py
│   ├── datasets_only_segmentation.py
│   ├── metrics.py
│   ├── postprocessing.py
│   ├── preprocess.py
│   ├── transforms.py
│   └── utils.py
├── train_stats
│   └── train_stats.md
├── README.md
├── mycetoma_env.yml
├── preprocessing.py
├── test.py
├── test1_segmentation_working.py
└── train.py
```

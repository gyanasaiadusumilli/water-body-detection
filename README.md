# Water Body Detection from Sentinel-2 and Landsat-8 Imagery

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gyanasaiadusumilli/water-body-detection/blob/main/notebooks/train_on_colab.ipynb)

Pixel-level mapping of surface water from multispectral satellite imagery, comparing classical machine learning models against a CNN on two sensors: **Sentinel-2** (10 m) and **Landsat-8** (30 m).

> Originally built during my undergraduate studies. Rebuilt and extended in 2026 after the original codebase was lost.

## Problem

Mapping lakes, reservoirs and rivers is essential for water resource management, flood monitoring and drought assessment. Simple spectral indices like NDWI confuse water with shadows, dark roofs and wet soil. This project asks: **how much do learned models improve on index thresholds, and does a CNN that sees spatial context beat pixel-wise classifiers?**

## Data

| Sensor | Collection | Resolution | Bands used |
|---|---|---|---|
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` (surface reflectance) | 10 m | B2, B3, B4, B8, B11, B12 |
| Landsat-8 | `LANDSAT/LC08/C02/T1_L2` (surface reflectance) | 30 m | SR_B2–SR_B7 |

Both sensors are mapped to a common set of bands: **blue, green, red, NIR, SWIR1, SWIR2**.

- **Imagery:** monthly cloud-masked median composites exported from Google Earth Engine (`gee/export_scenes.py`). Clouds are removed with the Sentinel-2 SCL band and the Landsat QA_PIXEL bits.
- **Labels:** [JRC Global Surface Water](https://global-surface-water.appspot.com/) Monthly History for the **same month** as the imagery (water / not water / no data). Cloud-masked pixels become "ignore".

## Features

Every model receives 10 features per pixel:

| Feature | Formula | Why |
|---|---|---|
| 6 reflectance bands | — | Raw spectral signature |
| NDWI | (Green − NIR) / (Green + NIR) | Classic water index |
| MNDWI | (Green − SWIR1) / (Green + SWIR1) | Suppresses built-up areas |
| NDVI | (NIR − Red) / (NIR + Red) | Separates vegetation |
| AWEI_sh | Blue + 2.5·Green − 1.5·(NIR + SWIR1) − 0.25·SWIR2 | Robust to shadows |

## Models

| Type | Model |
|---|---|
| Index baseline | NDWI > 0 |
| Index baseline | MNDWI with Otsu threshold |
| Classical ML (pixel-wise) | Logistic Regression |
| Classical ML (pixel-wise) | K-Nearest Neighbors |
| Classical ML (pixel-wise) | SVM (RBF kernel) |
| Classical ML (pixel-wise) | Random Forest |
| Classical ML (pixel-wise) | Gradient Boosting (HistGradientBoosting) |
| Deep learning | **CNN (U-Net)** on 128×128 tiles |

The CNN is an encoder-decoder with skip connections, trained with BCE + Dice loss (no-data pixels masked out), flip/rotation augmentation and early stopping on validation IoU.

## Evaluation design

- **Spatial split.** Scenes are cut into 128×128 tiles, and tiles are grouped into spatial blocks. Whole blocks go to train / val / test (70 / 15 / 15), so neighbouring, near-identical tiles cannot leak from training into testing.
- **Same test pixels for every model.** A fixed sample of up to 500k valid test pixels is scored for every model, including the CNN.
- **Metrics:** accuracy, precision, recall, F1, water IoU, Cohen's kappa. Because most pixels are land, accuracy alone is misleading; **IoU and F1 for the water class** are the main comparison metrics.

## Results

Eight study areas, two sensors, one fixed sample of 500,000 test pixels per sensor (same pixels for every model). Main metric: **IoU for the water class**.

![Model comparison](results/comparison_iou.png)

| Model | Sentinel-2 IoU | Sentinel-2 Acc | Landsat-8 IoU | Landsat-8 Acc |
|---|---|---|---|---|
| NDWI > 0 | 0.950 | 96.9% | 0.921 | 96.1% |
| MNDWI (Otsu) | 0.979 | 98.7% | 0.957 | 97.8% |
| Logistic Regression | 0.980 | 98.7% | 0.977 | 98.8% |
| KNN | 0.957 | 97.3% | 0.975 | 98.8% |
| SVM (RBF) | 0.981 | 98.8% | 0.978 | 98.9% |
| Random Forest | 0.967 | 98.0% | 0.976 | 98.8% |
| Gradient Boosting | 0.975 | 98.4% | **0.978** | **98.9%** |
| **U-Net CNN** | **0.984** | **99.0%** | 0.973 | 98.6% |

Full metrics (precision, recall, F1, kappa) are in [`results/comparison.md`](results/comparison.md).

### What the results show

- **On Sentinel-2 (10 m), the U-Net CNN is the best model**: water IoU 0.984 and 99.0% accuracy. Compared with the classic NDWI > 0 threshold, it cuts the mislabelled water area (1 − IoU) from 5.0% to 1.6%, about two-thirds fewer errors.
- **On Landsat-8 (30 m), the classical models edge out the CNN.** Gradient boosting reaches 0.978 against 0.973 for the U-Net. At 30 m, a 128 × 128 tile covers almost 4 km, so shorelines and small water bodies span few pixels and spatial context helps less. The JRC labels are also derived from Landsat imagery by a per-pixel classifier, so per-pixel models trained on Landsat bands can match them very closely.
- **Learned models beat fixed index thresholds on both sensors.** Even choosing the MNDWI threshold from the data (Otsu: 0.19 for Sentinel-2, 0.17 for Landsat-8, instead of 0) closes most of the gap.
- The CNN used early stopping on validation IoU: it trained for 14 epochs on Sentinel-2 and 30 on Landsat-8.

![Detected water over Hyderabad](results/maps/hyderabad_202102_cnn_unet_preview.png)

*U-Net water mask over Hyderabad (Sentinel-2, February 2021), including Hussain Sagar lake.*

These are single runs with one random seed. Differences of about 0.005 IoU between the top models are within what a different split or seed could change.

## Project structure

```
water-body-detection/
├── gee/
│   ├── export_scenes.py    # Earth Engine export to Google Drive: imagery + JRC labels
│   └── fetch_scenes.py     # Direct download of 8 study areas (no Drive export)
├── notebooks/
│   └── train_on_colab.ipynb  # One-click download, train and compare
├── src/
│   ├── features.py         # Bands and water indices
│   ├── prepare_data.py     # Tiling and spatial train/val/test split
│   ├── common.py           # Shared test sample and metrics
│   ├── train_ml.py         # Baselines + 5 classical models
│   ├── cnn.py              # U-Net architecture
│   ├── train_cnn.py        # CNN training and test evaluation
│   ├── compare.py          # Comparison table and chart
│   └── predict_scene.py    # Full-scene water mask (GeoTIFF + preview)
└── requirements.txt
```

## How to run

**Easiest: Google Colab.** Click the "Open in Colab" badge above and run the notebook top to bottom. It downloads Sentinel-2 and Landsat-8 scenes for 8 areas straight from Earth Engine, trains every model on a free GPU, and prints the comparison table.

**Locally:**

```bash
pip install -r requirements.txt

# 1. Export scenes from Earth Engine (needs a free Earth Engine account).
#    Use several areas with different water types: lakes, rivers, reservoirs, coast.
earthengine authenticate
python gee/export_scenes.py --project YOUR_GEE_PROJECT --name hyderabad \
    --bbox 78.25 17.25 78.60 17.55 --year 2021 --month 3 --sensor both
#    Download the GeoTIFFs from Google Drive into data/raw/sentinel2/ and data/raw/landsat8/

# 2. Tile and split
python -m src.prepare_data --sensor sentinel2
python -m src.prepare_data --sensor landsat8

# 3. Classical models and baselines
python -m src.train_ml --sensor sentinel2
python -m src.train_ml --sensor landsat8

# 4. CNN (GPU recommended: Google Colab or Kaggle)
python -m src.train_cnn --sensor sentinel2
python -m src.train_cnn --sensor landsat8

# 5. Compare everything
python -m src.compare

# 6. Map water across a full scene
python -m src.predict_scene data/raw/sentinel2/<scene>.tif --model models/sentinel2/cnn_unet.pt
```

## Limitations

- JRC labels are themselves derived from Landsat, so they are a strong reference but not hand-annotated ground truth. Small ponds and narrow rivers below 30 m are under-represented.
- JRC monthly history currently runs to 2021, which sets the latest usable month.

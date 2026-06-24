---
title: >-
  [Paper Note] High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data
description: >-
  [ICML 2025][Remote Sensing][LFMC] Fine-tuning the pretrained multimodal Earth observation model Galileo generates 10-meter resolution Live Fuel Moisture Content (LFMC) maps, reducing RMSE by 20%+ compared to randomly initialized models, with the pipeline's utility validated by a 2025 Los Angeles wildfire case study.
tags:
  - "ICML 2025"
  - "Remote Sensing"
  - "LFMC"
  - "Wildfire Risk"
  - "foundation model"
  - "Galileo"
date: 2026-05-08
content_hash: 858223676b53bf94
---

# High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data

**Conference**: ICML 2025  
**arXiv**: [2506.20132](https://arxiv.org/abs/2506.20132)  
**Code**: [github.com/allenai/lfmc](https://github.com/allenai/lfmc)  
**Area**: Remote Sensing  
**Keywords**: LFMC, Wildfire Risk, foundation model, Remote Sensing, Galileo

## TL;DR

Fine-tuning the pretrained multimodal Earth observation model Galileo generates 10-meter resolution Live Fuel Moisture Content (LFMC) maps, reducing RMSE by 20%+ compared to randomly initialized models, with the pipeline's utility validated by a 2025 Los Angeles wildfire case study.

## Background & Motivation

**Background**: Live Fuel Moisture Content (LFMC) is a critical metric for measuring live vegetation moisture content, directly influencing wildfire ignition probability, fuel availability, and fire spread. The LFMC calculation formula is $\text{LFMC}[\%] = \frac{W_f - W_d}{W_d} \times 100$, where $W_f$ is the fresh vegetation weight and $W_d$ is the dry weight. Lower LFMC values indicate higher wildfire risk.

**Limitations of Prior Work**: Ground LFMC sampling is highly time-consuming and labor-intensive—requiring field collection of plant samples, weighing, oven-drying for 1-2 days, and re-weighing, taking 12 hours to 4 days for a single site. Restricted by this, existing sampling sites are sparse in both space and time, failing to provide complete coverage for LFMC assessment.

**Key Challenge**: Existing machine learning-based LFMC estimation methods (e.g., Rao et al. 2020, Miller et al. 2023) employ fully supervised, randomly initialized models with spatial resolutions of only 250-500 meters, and suffer from limited generalization performance due to label imbalance and heterogeneity.

**Key Insight**: Leveraging Earth observation foundation models (Galileo) pretrained on large-scale multimodal remote sensing data can achieve high-precision, high-resolution (10-meter) LFMC predictions through fine-tuning, improving the resolution by 25 to 50 times. The prior knowledge brought by pretraining enhances robustness to missing inputs and spatiotemporal generalization.

**Core Idea**: Pretrained multimodal remote sensing foundation model + Globe-LFMC 2.0 dataset fine-tuning = an automated high-resolution LFMC mapping pipeline.

## Method

### Overall Architecture

The pipeline consists of three stages:
1. **Training Data Construction**: Filter samples from the Globe-LFMC 2.0 dataset within the CONUS (2017-2023) region, and export corresponding multimodal remote sensing data for each sample from Google Earth Engine.
2. **Model Fine-tuning**: Fine-tune the LFMC regression model using MSE loss based on Galileo-Tiny pretrained weights.
3. **Map Generation**: Given a spatiotemporal region $\rightarrow$ export remote sensing data $\rightarrow$ perform model inference $\rightarrow$ output 10-meter resolution LFMC maps.

### Key Designs

1. **Globe-LFMC 2.0 Dataset Processing**: Filtered to obtain 41,214 samples across 1,031 sites. Multiple samples taken on the same day at the same site are averaged. LFMC values are clipped at 302% (99.9th percentile) and normalized to suppress outliers. Splitting training/validation/test sets randomly with 70%/15%/15%. Data covers elevations from 15 to 3187 meters, multiple land cover types, and all four seasons.

2. **Galileo Pretrained Model**: A Vision Transformer-based multimodal remote sensing foundation model (Galileo-Tiny, 5.3M parameters) capable of handling 10 types of remote sensing products. Input modalities include:

    - Sentinel-2 multispectral optical data (visible, NIR, SWIR + NDVI)
    - Sentinel-1 SAR data (VV/VH polarization)
    - VIIRS nighttime lights
    - ERA-5 meteorological data (precipitation, temperature)
    - TerraClimate water balance data (climate water deficit, soil moisture, evapotranspiration)
    - SRTM topographic data (elevation, slope)
    - Latitude and longitude positional encoding
   
   Modality spatial resolutions vary widely (10m to tens of km), and temporal resolutions also differ (5 days to monthly). Galileo categorizes inputs based on whether they vary spatially/temporally, aggregating the temporal dimension to a monthly scale.

3. **Fine-tuning Strategy**: MSE loss, maximum of 100 epochs, early stopping (halting when the validation set shows no improvement for 5 epochs). Single H100 card training takes about 30-60 minutes. The default input shape is 32×32 pixels with 12 timesteps.

### Loss & Training

- Loss Function: Mean Squared Error (MSE), directly regressing normalized LFMC values.
- Data Preprocessing: LFMC clipped at 302%, normalized by the 99.9th percentile.
- Training Hardware: Single NVIDIA H100 GPU, training time 30-60 minutes.
- Early Stopping Strategy: Stops when the validation set has no improvement for 5 consecutive epochs.

## Key Experimental Results

### Main Results

| Model | RMSE↓ | MAE↓ | R²↑ | Note |
|------|-------|------|-----|------|
| **GalileoLFMC (Pretrained)** | **18.91** | **12.58** | **0.72** | Ours |
| Randomly Initialized Model | 23.61 | 16.33 | 0.57 | Traditional fully supervised method |
| Monthly Mean Prediction | 33.66 | 25.38 | 0.11 | Simple baseline |

The pretrained model achieves a 20% reduction in RMSE (23.61 $\rightarrow$ 18.91) and a 26% improvement in R² (0.57 $\rightarrow$ 0.72) compared to the randomly initialized counterpart.

### Ablation Study

**Input Shape Sensitivity**:

| Spatial (H×W) | Timesteps T | RMSE | R² | Note |
|------------|---------|------|-----|------|
| 32×32 | 12 | 18.91 | 0.72 | Default configuration |
| 32×32 | 3 | 19.45 | 0.70 | Reduced temporal dimension has minor impact |
| 1×1 | 12 | 20.25 | 0.68 | Point-level spatial information only |

**Robustness to Missing Inputs** (Pretrained vs. Randomly Initialized):

| Removed Input | Pretrained RMSE | Pretrained R² | Random RMSE | Random R² |
|-----------|------------|----------|----------|---------|
| None | 18.91 | 0.72 | 23.61 | 0.57 |
| TerraClimate | 19.51 | 0.70 | 25.57 | **0.49** (-14%) |
| Positional Encoding | 20.08 | 0.69 | 23.80 | 0.56 |

Removing any single modality from the pretrained model leads to minor performance fluctuations (RMSE change < 1.2), whereas removing TerraClimate from the randomly initialized model causes R² to plunge by 14%.

### Key Findings

- **Cross-Season Generalization**: Despite having the fewest training samples in winter, the model achieves the lowest RMSE (15.31) and highest R² (0.77) during winter, indicating that pretraining benefits temporal generalization.
- **Consistency Across Land Covers**: Consistent performance with RMSE ranging from 16.79 to 20.52 across different land cover types (trees, grass, shrubs, built-up, barren).
- **High-Altitude Degradation**: While R² > 0.7 below 2000 meters, R² drops to 0.32 in the 3000-3500 meter range, which correlates with the lack of high-altitude training samples (only 444 samples).
- **Spatial Autocorrelation**: Moran's I = 0.057 (p=0.001), indicating weak positive spatial autocorrelation in residuals, which suggests slight information leakage may exist with random splitting.

## Highlights & Insights

- **Resolution Leap**: Improvement from prior 250-500 meters to 10 meters, achieving a 25-50x resolution enhancement, which is highly valuable for fine-grained localized fire risk assessment.
- **Implicit Value of Pretraining**: Beyond improving accuracy, it crucially enhances robustness to missing inputs—self-supervised pretraining teaches the model complementary relationships among modalities.
- **Utility-Oriented**: Provides an end-to-end automated pipeline (data export $\rightarrow$ inference $\rightarrow$ mapping) directly serving disaster management and prescribed burn planning.
- **2025 Los Angeles Wildfires Case Study**: LFMC predictions for the Palisades and Eaton fires align with expert field observations—the LFMC for 2023-2024 was higher than for 2021-2022, reflecting vegetation growth following two consecutive wet springs, potentially leading to higher fuel loads.

## Limitations & Future Work

- The model is fine-tuned only on CONUS western data, and global generalization has not been validated.
- Currently used only for retrospective analysis; LFMC forecasting/prediction capability is not yet tested.
- The temporal resolution is on a monthly average scale, whereas actual wildfire management requires weekly or even daily updates.
- Random splitting may cause spatial information leakage (neighboring sites appearing in both training and testing sets), requiring spatial partitioning strategies in the future.
- Performance drops significantly in high-altitude regions (>3000m), necessitating the acquisition of more high-altitude labeled data.

## Related Work & Insights

- **Rao et al. (2020)**: Mapped LFMC at 250m resolution using a physics-informed RNN, validating the value of multimodal remote sensing inputs.
- **Miller et al. (2023)**: Predicted LFMC at 500m resolution with a tempCNN, incorporating forecasting capability (3-month lead time).
- **Galileo (Tseng et al., 2025)**: The pretrained Earth observation foundation model on which this work primarily relies, capable of handling 10 remote sensing products.
- **Jolly et al. (2024)**: Physical modeling approach of the National Fire Danger Rating System, using the meteorologically-driven GSI model.
- **Insight**: The core value of remote sensing foundation models in downstream tasks lies not only in accuracy improvements but also in bolstering robustness and data efficiency.

## Rating

- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RAMEN: Resolution-Adjustable Multimodal Encoder for Earth Observation](../../CVPR2026/remote_sensing/ramen_resolution-adjustable_multimodal_encoder_for_earth_observation.md)
- [\[CVPR 2026\] OlmoEarth: Stable Latent Image Modeling for Multimodal Earth Observation](../../CVPR2026/remote_sensing/olmoearth_stable_latent_image_modeling_for_multimodal_earth_observation.md)
- [\[CVPR 2026\] YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction](../../CVPR2026/remote_sensing/yieldsat_a_multimodal_benchmark_dataset_for_high-resolution_crop_yield_predictio.md)
- [\[ICML 2025\] LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth](lighthouse_fast_and_precise_distance_to_shoreline_calculations_from_anywhere_on_.md)
- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](../../NeurIPS2025/remote_sensing/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)

</div>

<!-- RELATED:END -->

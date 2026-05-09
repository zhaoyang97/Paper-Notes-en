---
title: >-
  [Paper Note] BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction
description: >-
  [AAAI 2026][LLM Evaluation][Wildfire Risk Prediction] This paper introduces BCWildfire, a multimodal wildfire risk prediction dataset covering 240 million hectares of British Columbia, Canada over a 25-year span, encompassing 38 driving factors. It conducts a systematic benchmark evaluation of time series forecasting models across four paradigms—CNN, Linear, Transformer, and Mamba—revealing the performance ceiling of current models and the key influential factors in wildfire prediction.
tags:
  - AAAI 2026
  - LLM Evaluation
  - Wildfire Risk Prediction
  - Time Series Forecasting
  - Multimodal Dataset
  - Deep Learning Benchmark
  - Boreal Forest
date: 2026-05-08
content_hash: 20a1909e758984a9
---

# BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction

**Conference**: AAAI 2026
**arXiv**: [2511.17597](https://arxiv.org/abs/2511.17597)
**Code**: [https://github.com/SynUW/BCWildfire](https://github.com/SynUW/BCWildfire)
**Area**: LLM Evaluation
**Keywords**: Wildfire Risk Prediction, Time Series Forecasting, Multimodal Dataset, Deep Learning Benchmark, Boreal Forest

## TL;DR

This paper introduces BCWildfire, a multimodal wildfire risk prediction dataset covering 240 million hectares of British Columbia, Canada over a 25-year span, encompassing 38 driving factors. It conducts a systematic benchmark evaluation of time series forecasting models across four paradigms—CNN, Linear, Transformer, and Mamba—revealing the performance ceiling of current models and the key influential factors in wildfire prediction.

## Background & Motivation

Wildfires are escalating globally in frequency, scale, and intensity, with particularly severe risks in carbon-rich boreal forests. Data-driven approaches, especially deep learning, have demonstrated promise for wildfire prediction, yet deployment remains constrained by the scarcity of high-quality datasets.

Existing datasets suffer from three major limitations:

**Limited geographic coverage**: Most datasets focus on localized fire spread modeling in the United States and Mediterranean regions, lacking coverage of carbon-intensive boreal ecosystems.

**Insufficient temporal span**: The vast majority of datasets support only short retrospective windows of 1 or 7 days, making it impossible to model large-scale wildfire risk driven by long-term factors such as fuel accumulation and prolonged drought.

**Incomplete driving factors**: Integration of heterogeneous fire-driving factors—meteorological, vegetation, topographic, and anthropogenic—remains inadequate.

The root cause lies in the fact that wildfire risk is a multi-factor interactive process that accumulates over long time scales, whereas existing datasets and models primarily focus on short-term, localized fire spread prediction. The paper's starting point is to construct a standardized dataset with long time series, broad spatial coverage, and multimodal inputs, reformulating wildfire risk prediction as a time series forecasting problem to leverage recent advances in that field.

## Method

### Overall Architecture

BCWildfire is a dataset benchmark in time series forecasting format, comprising two components: dataset construction and model evaluation. The data covers British Columbia and surrounding areas ($2782 \times 1302 \text{ km}^2$) from 2000 to 2024, at 1 km spatial resolution and daily temporal resolution, with 38 multimodal covariates.

### Key Designs

1. **Multimodal Driving Factor Design (38 variables)**:

   - **Function**: Integrates five categories of wildfire driving factors.
   - **Mechanism**: Fuel conditions (MODIS products including LAI/FPAR/NDVI/EVI), meteorological factors (ERA5-Land temperature/precipitation/wind speed/soil moisture + MODIS thermal radiance products), topographic factors (ASTER DEM slope/aspect/hillshade + distance to water bodies), human activity (MODIS land use + distance to infrastructure), and fire detection (MOD/MYD14A1 active fire products).
   - **Design Motivation**: ERA5-Land has coarse spatial resolution (~11 km) and cannot capture fine thermal anomalies; MODIS 1 km thermal radiance products are therefore incorporated as a complement. Deep-layer soil moisture (0–289 cm) is included to reflect long-term cumulative environmental moisture effects.

2. **Data Preprocessing and Class Imbalance Strategy**:

   - **Function**: Unifies coordinate systems and resolutions, and addresses cloud occlusion and class imbalance.
   - **Mechanism**: All data are harmonized to WGS84 and 1 km/daily resolution; QC bands combined with historical gap-filling handle cloud occlusion (avoiding data leakage); an undersampling strategy excludes negative samples within a 60 km/3-day buffer zone around positive samples, maintaining positive-to-negative ratios by land cover type (1:2 for training, 1:1 for testing).
   - **Design Motivation**: Wildfires are extremely rare events, and naive training leads to severe class imbalance. Random selection of negative samples risks including high-risk areas, necessitating spatial buffering for exclusion.

3. **Time Series Forecasting Benchmark Evaluation**:

   - **Function**: Evaluates the next-day wildfire risk prediction performance of six state-of-the-art time series models.
   - **Mechanism**: The preceding 10 days of driving factors serve as input to predict the probability of wildfire occurrence on the following day, framed as a binary classification time series forecasting problem.
   - **Design Motivation**: Reframes wildfire prediction from a traditional spatial CNN task to a time series forecasting paradigm, leveraging the temporal modeling capacity for long-term cumulative factors.

### Loss & Training

Binary Cross-Entropy (BCE) loss is used to train all models. Adam optimizer, two NVIDIA A6000 GPUs, batch size 128, 50 training epochs, learning rate $1 \times 10^{-5}$. Data splits: 2000–2020 for training, 2021–2022 for validation, 2023–2024 for testing.

## Key Experimental Results

### Main Results

| Model | Type | Precision | Recall | F1 | PR-AUC |
|-------|------|-----------|--------|----|--------|
| SCINet | CNN | 84.77 | 88.05 | 86.38 | 94.46 |
| TSMixer | Linear | 85.69 | 90.39 | 87.97 | 96.24 |
| CrossLinear | Linear | 88.04 | 87.59 | 87.81 | 96.07 |
| Crossformer | Transformer | 88.74 | 87.49 | 88.11 | 96.28 |
| FEDformer | Transformer | 82.95 | 91.18 | 86.87 | 94.93 |
| S_Mamba | Mamba | 84.21 | 86.44 | 85.31 | 94.83 |

### Ablation Study (Effect of Positional Encoding)

| Model | F1 w/o PE | F1 w/ PE | Recall Gain |
|-------|-----------|----------|-------------|
| Crossformer | 88.11 | 88.70 | +2.03% |
| FEDformer | 86.87 | 87.43 | −1.36% |
| S_Mamba | 85.31 | 87.46 | +3.65% |
| TSMixer | 87.97 | 88.12 | −1.10% |

### Key Findings

- **Performance Ceiling**: Even the best-performing models achieve recall below 92%, with precision confined to a narrow range of 83–88%, indicating that the inherent stochasticity of wildfires and class imbalance remain fundamental challenges.
- **Transformer Advantage**: Crossformer and FEDformer outperform CNN and Mamba models in recall and stability, reflecting their superior capacity for modeling long-range temporal dependencies.
- **SHAP Factor Analysis**: Fire detection signals are the most important feature (negative SHAP values for absence of fire signals indicate potential new ignitions), followed by 28–100 cm soil moisture and surface latent heat flux; snow cover exerts a significant negative effect (suppressing ignition).
- **Spatial Positional Encoding** effectively improves performance for most models, with S_Mamba achieving a 3.65% gain in recall.

## Highlights & Insights

- Reformulating wildfire risk prediction as a time series forecasting problem is a valuable paradigm shift, enabling direct application of a wide range of models from the time series forecasting literature.
- The dataset's scale—25 years of daily resolution data, 38 variables, and 240 million hectares—represents a significant advantage in the wildfire domain.
- SHAP analysis reveals a counterintuitive finding: ignition sites tend to exhibit slightly elevated deep-layer (100–289 cm) soil moisture, suggesting that wildfires frequently occur in ecologically moist, fuel-rich environments.

## Limitations & Future Work

- The current evaluation covers only next-day prediction, leaving the full 25-year longitudinal data underutilized for longer-term trend modeling.
- The dataset is region-specific to British Columbia; the generalizability of trained models to other boreal forest regions has not been validated.
- Only temporal information is utilized, without incorporating spatial neighborhood context (e.g., CNN-based spatial modeling), which may cause the loss of local spatial propagation patterns.
- Synthetic (CARLA) datasets could be combined with real meteorological data, though a potential domain gap between the two sources remains unaddressed.

## Related Work & Insights

- Compared to existing datasets such as SeasFire Cube, the key distinguishing feature of BCWildfire is its support for time series forecasting with long retrospective windows.
- The Mamba-based model (S_Mamba) achieves a favorable balance between accuracy and efficiency, making it suitable for practical deployment.
- Future work could explore spatiotemporal joint modeling (e.g., introducing BEV-style paradigms into wildfire prediction) and ensemble forecasting methods.

## Rating
- Novelty: ⭐⭐⭐⭐ (Substantial dataset contribution; no new model is proposed)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive evaluation, though comparisons with traditional methods are absent)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (Fills the gap in long-term wildfire prediction datasets for boreal forests)

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](../../ACL2026/llm_evaluation/sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](../../ACL2026/llm_evaluation/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/llm_evaluation/reframing_long-tailed_learning_via_loss_landscape_geometry.md)
- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery
description: >-
  [ICML 2025 (TerraBytes Workshop)][Segmentation][Satellite remote sensing] This work systematically studies the effects of fusing optical imagery with additional geographic data layers (DEM, land cover maps, temperature, wind speed, etc.) in satellite remote sensing ML tasks. It is found that multimodal inputs significantly improve model performance, with the gains being most pronounced in scenarios with limited labeled data and geographic out-of-distribution shifts. Surprisin…
tags:
  - "ICML 2025 (TerraBytes Workshop)"
  - "Segmentation"
  - "Satellite remote sensing"
  - "multimodal fusion"
  - "data-efficiency"
  - "out-of-distribution generalization"
  - "geographic covariates"
  - "semantic segmentation"
  - "land cover classification"
date: 2026-05-08
content_hash: 8fa877e8a67db31e
---

# Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery

**Conference**: ICML 2025 (TerraBytes Workshop)

**arXiv**: [2507.13385](https://arxiv.org/abs/2507.13385)

**Author**: Arjun Rao, Esther Rolf

**Area**: Segmentation (Remote Sensing Image Segmentation / Multimodal Fusion)

**Keywords**: Satellite remote sensing, multimodal fusion, data-efficiency, out-of-distribution generalization, geographic covariates, semantic segmentation, land cover classification

## TL;DR

This work systematically studies the effects of fusing optical imagery with additional geographic data layers (DEM, land cover maps, temperature, wind speed, etc.) in satellite remote sensing ML tasks. It is found that multimodal inputs significantly improve model performance, with the gains being most pronounced in scenarios with limited labeled data and geographic out-of-distribution shifts. Surprisingly, hard-coded fusion strategies outperform learned fusion strategies.

## Background & Motivation

A rich abundance of geospatial data layers exists globally: remote sensing raster data (satellite imagery, digital elevation models DEM, predicted land cover maps), human annotations, and environmental sensor data (temperature, wind speed, etc.). However, currently, the vast majority of satellite remote sensing ML (SatML) models solely use **multispectral satellite imagery** as input, underutilizing other available geographic data modalities.

Core research questions:

**Value of Multimodal Fusion**: Can fusing additional geographic data with optical imagery improve the performance of SatML models in supervised learning?

**Data-Efficiency**: Can multimodal inputs bring greater benefits when labeled data is limited?

**Out-of-Distribution Generalization**: Do multimodal inputs help models generalize to geographically out-of-distribution regions?

**Choice of Fusion Strategy**: Hard-coded fusion vs. learned fusion, which one is superior?

The answers to these questions hold significant practical value for resource-constrained remote sensing applications (such as land use monitoring in developing countries, disaster assessment, etc.).

## Method

### Overall Architecture

This work adopts a systematic benchmark evaluation framework:

1. Select multiple SatML benchmark tasks (covering classification, regression, and segmentation).
2. Append additional geographic data layers to the dataset of each task to construct "augmented" datasets.
3. Compare model performance under different fusion strategies.
4. Analyze the impact of data volume and geographic distribution on fusion gains.

### Key Design 1: Selection of Geographic Data Layers

Additional input modalities include various geographic covariates:

- **Digital Elevation Model (DEM)**: Terrain elevation information
- **Predicted Land Cover Maps**: Existing products such as ESA WorldCover
- **Temperature Data**: From meteorological sensors or reanalysis data
- **Wind Speed Data**: Environmental wind field information
- **Other Remote Sensing Indices**: Vegetation indices such as NDVI

For input $X$, the augmented multimodal input is formulated as:

$$X_{\text{aug}} = [X_{\text{optical}}, X_{\text{DEM}}, X_{\text{LC}}, X_{\text{temp}}, X_{\text{wind}}, \ldots]$$

where $[\cdot]$ denotes concatenation along the channel dimension.

### Key Design 2: Comparison of Fusion Strategies

This work systematically compares two main categories of fusion strategies:

**Hard-coded Fusion**:
- **Early Fusion**: Directly concatenate all modalities along channels and feed them into the model
  $$Z = f_\theta([X_1, X_2, \ldots, X_K])$$
- **Feature Stacking**: Directly concatenate additional modalities as extra channels

**Learned Fusion**:
- **Attention-based Fusion**: Dynamically fuse features of different modalities using learnable attention weights
- **Gated Fusion**: Learn a gating mechanism to determine the contribution ratio of each modality
  $$Z = \sum_{k=1}^{K} g_k(\{X_i\}) \cdot f_k(X_k)$$
  where $g_k$ is the gating function and $f_k$ is the modality-specific encoder

### Key Design 3: Data-Efficiency and OOD Experimental Design

**Data-Efficiency Experiments**: Systematically reduce the training set scale (e.g., using 10%, 25%, 50%, 100% of the training data) to observe how the gains of multimodal fusion change under different data volumes.

**Geographic OOD Experiments**: Divide data into training and test sets by geographic regions, where the test set comes from geographic regions not covered by the training set. This constructs two evaluation settings: "in-distribution" (geographically consistent) and "out-of-distribution" (geographically different).

### Loss & Training

Standard loss functions are used according to the different tasks:

- **Classification Tasks**: Cross-entropy loss $\mathcal{L}_{\text{cls}} = -\sum_c y_c \log \hat{y}_c$
- **Regression Tasks**: Mean squared error $\mathcal{L}_{\text{reg}} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$
- **Segmentation Tasks**: Pixel-level cross-entropy loss $\mathcal{L}_{\text{seg}} = -\frac{1}{HW}\sum_{h,w}\sum_c y_{h,w,c} \log \hat{y}_{h,w,c}$

## Key Experimental Results

### Main Results: Impact of Multimodal Fusion on Different Tasks

| Task Type | Dataset | Optical Only | Multimodal Fusion | Gain |
|---------|-------|-------|----------|------|
| Classification | SatML Benchmark A | baseline | **↑ Significant** | Classification accuracy improved |
| Regression | SatML Benchmark B | baseline | **↑ Significant** | Regression error reduced |
| Segmentation | SatML Benchmark C | baseline | **↑ Significant** | Segmentation IoU improved |

**Key Findings**: On all three task types, fusing additional geographic data layers significantly improves model performance.

### Data-Efficiency Experiments: Labeled Data Volume vs. Multimodal Gain

| Training Data Ratio | Optical Only | Multimodal Fusion | Relative Multimodal Gain |
|-------------|-------|----------|-------------|
| 10% | Low | Medium | **Largest Gain** |
| 25% | Medium-Low | Medium-High | **Significant Gain** |
| 50% | Medium | Medium-High | Moderate Gain |
| 100% | High | Higher | Small Gain |

**Key Findings**: The benefits of multimodal fusion are most pronounced when labeled data is limited. With only 10%-25% of training data, the relative performance improvement of the multimodal model compared to the optical-only model is much larger than that observed when using the full dataset.

### Comparison of Fusion Strategies

| Fusion Strategy | Classification Performance | Regression Performance | Segmentation Performance | Average Rank |
|---------|---------|---------|---------|---------|
| Optical Only (No Fusion) | Low | Low | Low | 4 |
| Hard-coded Early Fusion | **High** | **High** | **High** | **1** |
| Learned Attention Fusion | Medium | Medium | Medium | 3 |
| Learned Gated Fusion | Medium-High | Medium | Medium-High | 2 |

**Key Findings**: Hard-coded fusion strategies (simple channel concatenation) surprisingly outperform learned fusion variants (attention/gated fusion). This contradicts the common assumption in most multimodal learning literature that "learned fusion is superior."

### Geographic OOD Generalization Experiments

| Evaluation Setting | Optical Only | Multimodal Fusion | Relative Multimodal Gain |
|---------|-------|----------|-------------|
| In-Distribution (Geographically ID) | Relatively High | High | Moderate Gain |
| Out-of-Distribution (Geographically OOD) | Relatively Low | Medium-High | **Largest Gain** |

**Key Findings**: Multimodal inputs assist OOD generalization more than in ID scenarios, as additional geographic data layers provide auxiliary information (such as elevation and climate) that is directly related to the target variable but does not vary drastically across geographic regions.

## Highlights & Insights

1. **Hard-coded Fusion Outperforms Learned Fusion**: This is the most surprising finding of this study—simple feature concatenation outperforms elaborately designed learned fusion. Possible reasons include: (a) the auxiliary data layers already contain directly useful information that does not require complex transformations; (b) learned fusion introduces additional parameters, making it more prone to overfitting when data is limited.
2. **Data-Efficiency Multiplier**: The "marginal value" of multimodal inputs is maximized under label scarcity, which has significant practical implications for remote sensing applications where annotation costs are high—instead of investing in more annotations, introducing freely available geographic data layers is more beneficial.
3. **Enhanced OOD Robustness**: Geographic covariates (such as DEM, climate, etc.) exhibit consistency across regions, providing "anchored" information for the model and mitigating distribution shifts in optical imagery caused by changes in geographic locations.
4. **Systematic Benchmark Evaluation**: The study covers three types of tasks (classification, regression, and segmentation), providing a comprehensive picture of the effects of multimodal fusion.

## Limitations & Future Work

1. **Limited Experimental Scale**: As a workshop paper (17 pages), the dataset and model scales may not be sufficient to support strong conclusions.
2. **Accessibility of Geographic Data Layers**: Some additional data layers (e.g., high-resolution DEM, real-time meteorological data) might be unavailable or have low accuracy in certain regions.
3. **Insufficient Exploration of Fusion Strategies**: Only a limited number of fusion variants were compared, without involving more advanced multimodal fusion methods (such as Transformer-level cross-modal attention).
4. **Missing Temporal Dimension**: Temporal dynamics were not considered—the complementarity between geographical data layers and optical imagery might vary across different seasons.
5. **Single Model Architecture**: The generalizability of the findings across different backbones (ViT, Swin, etc.) has not been verified.

## Related Work & Insights

- **GFM (Geospatial Foundation Model)**: Large-scale pretrained geospatial foundation models, which typically use only optical modalities.
- **SatMAE**: A self-supervised pretraining method for satellite imagery.
- **Multimodal Remote Sensing Fusion**: Traditional methods mostly focus on optical+SAR or optical+LiDAR fusion, rarely exploring "non-remote sensing" data layers such as meteorology.
- **Data-Efficient Learning**: Few-shot learning and semi-supervised methods also aim to improve annotation efficiency; this paper provides an orthogonally complementary "data augmentation" perspective.

**Insights**: The core takeaway of this paper is "do not ignore free auxiliary data." For remote sensing ML systems, concatenating publicly available geographic data layers (DEM, climate, land cover, etc.) as additional channels into the input is an extremely simple yet effective enhancement strategy, especially for scenarios with scarce annotations and cross-region deployment. The finding that hard-coded fusion outperforms learned fusion also serves as a reminder that simple methods can be more robust when data is limited.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 3 | Clear research questions but limited methodological novelty, primarily a systematic empirical study |
| Technical Depth | 3 | Reasonable experimental design but lacks in-depth theoretical analysis |
| Experimental Thoroughness | 4 | Covers classification/regression/segmentation tasks; both data-efficiency and OOD experimental designs are sound |
| Writing Quality | 4 | 17 pages + 9 figures + 7 tables, clearly structured with explicit conclusions |
| Value | 4 | Directly applicable conclusion: concatenating free geographic data layers can boost performance |
| Overall Rating | 3.6 | Solid empirical workshop paper with highly practical conclusions but limited depth |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery](alberta_wells_dataset_pinpointing_oil_and_gas_wells_from_satellite_imagery.md)
- [\[ICCV 2025\] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [\[AAAI 2026\] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts](../../AAAI2026/segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)
- [\[CVPR 2026\] The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation](../../CVPR2026/segmentation/the_golden_subspace_where_efficiency_meets_generalization_in_continual_test-time.md)
- [\[ICML 2025\] Separating Knowledge and Perception with Procedural Data](separating_knowledge_and_perception_with_procedural_data.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] WildSAT: Learning Satellite Image Representations from Wildlife Observations
description: >-
  [ICCV 2025][Remote Sensing][remote sensing representation learning] This paper proposes WildSAT, which leverages millions of geotagged wildlife observations from citizen science platforms to align satellite images…
tags:
  - "ICCV 2025"
  - "Remote Sensing"
  - "remote sensing representation learning"
  - "contrastive learning"
  - "wildlife observations"
  - "cross-modal"
  - "satellite imagery"
date: 2026-05-08
content_hash: c5cd7f27fe550266
---

# WildSAT: Learning Satellite Image Representations from Wildlife Observations

**Conference**: ICCV 2025
**arXiv**: [2412.14428](https://arxiv.org/abs/2412.14428)  
**Code**: [https://github.com/cvl-umass/wildsat](https://github.com/cvl-umass/wildsat)  
**Area**: Remote Sensing / Representation Learning
**Keywords**: remote sensing representation learning, contrastive learning, wildlife observations, cross-modal, satellite imagery

## TL;DR

This paper proposes WildSAT, which leverages millions of geotagged wildlife observations from citizen science platforms to align satellite images, species locations, and textual descriptions via contrastive learning, substantially improving remote sensing representation quality and enabling zero-shot text-based retrieval.

## Background & Motivation

A core challenge in remote sensing representation learning is obtaining supervision signals. Existing approaches include:
- **Self-supervised learning** (SeCo, Prithvi): exploiting spatiotemporal invariances or masked autoencoders, but lacking semantic supervision
- **Supervised learning** (SatlasPretrain): large-scale multi-task labels, but with high annotation costs
- **Cross-modal learning** (GRAFT, TaxaBind, RemoteCLIP): aligning ground-level images or text, but primarily targeting anthropogenic features (roads, buildings)

The key insight of this work is that **species distributions encode rich ecological and environmental information**. For instance, mountain goats inhabit rugged terrain, and cactus wrens nest in desert cacti — species habitat preferences directly reflect local natural environment characteristics. Such information, freely available and globally distributed from platforms like iNaturalist, amounts to hundreds of millions of observations. Nevertheless, the potential of wildlife observations for improving remote sensing representations has remained largely unexplored.

## Method

### Overall Architecture

WildSAT adopts a multimodal contrastive learning framework, jointly training on three types of signals:
1. **Satellite images**: Sentinel-2 images of the same location at different times provide temporal augmentation
2. **Species locations**: latitude/longitude coordinates encoded into location vectors via the SINR model, incorporating environmental covariates (climate data)
3. **Text descriptions**: habitat and behavior descriptions from Wikipedia pages of the corresponding species, encoded via GritLM

The image encoder $f_\theta$ can be any architecture (ResNet50, ViT-B/16, etc.), with three separate linear projection heads producing embeddings for the image, text, and location modalities respectively.

### Key Designs

**Three-way contrastive learning**:
- $\mathcal{L}_{img}$: satellite images of the same location at different times serve as positive pairs (with geometric augmentation)
- $\mathcal{L}_{txt}$: satellite image embeddings are aligned with Wikipedia text embeddings
- $\mathcal{L}_{loc}$: satellite image embeddings are aligned with SINR location embeddings

All losses are based on InfoNCE; the overall objective is their sum.

**Parameter-efficient fine-tuning strategy**:
- Out-of-domain pretrained models (e.g., ImageNet): ResNet50 uses Scale and Shift Fine-tuning (BatchNorm parameters only); ViT uses DoRa (attention parameters only)
- Randomly initialized or in-domain pretrained models: full fine-tuning
- This ensures that existing domain knowledge is preserved

**Data construction**:
- The iNaturalist dataset provides 35.5 million observations across 47,375 species
- Corresponding Sentinel-2 satellite images (10 m/pixel, 512×512)
- Wikipedia text covers 127,484 paragraphs for 37,889 species
- Total of 980,376 training samples

### Loss & Training

$$\min_\theta [\mathcal{L}_{img} + \mathcal{L}_{txt} + \mathcal{L}_{loc}]$$

Each contrastive loss adopts standard InfoNCE:

$$\mathcal{L}_{con}(\mathbf{z}_i, \mathbf{e}_{1,...,n}) = -\log \frac{\exp(\mathbf{z}_i \cdot \mathbf{e}_i / \tau)}{\sum_j \exp(\mathbf{z}_i \cdot \mathbf{e}_j / \tau)}$$

During training, one text paragraph is randomly sampled per image–location pair.

## Key Experimental Results

### Main Results

Linear probing performance is evaluated on 7 downstream classification datasets and 2 segmentation datasets across 20 baseline models:

| Dataset | Base Avg. | +WildSAT Avg. | Gain |
|--------|----------|--------------|------|
| AID | 72.7 | 79.4 | +6.7 |
| EuroSAT | 88.9 | 94.3 | +5.4 |
| RESISC45 | 77.8 | 83.5 | +5.7 |
| So2Sat20k | 37.9 | 48.2 | +10.3 |
| UCM | 81.8 | 87.9 | +6.1 |
| BEN20k | 45.7 | 53.4 | +7.7 |

WildSAT achieves improvements in 108 out of 115 configurations, with an average gain of 4.3%–10.4%.

Comparison with CLIP-based methods (ViT-B/16):

| Method | Avg. Classification Performance |
|------|------------|
| TaxaBind | 59.8% |
| GRAFT | 65.0% |
| RemoteCLIP | 71.0% |
| CLIP | 71.6% |
| **WildSAT** | **76.6%** |

### Ablation Study

Ablation of modality contributions (Random ResNet50 → ImageNet ResNet50):

| loc | env | text | img-a | Random R50 | ImageNet R50 | Random ViT | ImageNet ViT |
|:---:|:---:|:----:|:-----:|-----------|-------------|-----------|-------------|
| | | | | 24.3% | 93.2% | 25.2% | 84.4% |
| ✓ | | | | 44.2% | 95.0% | 41.6% | — |

- Location signal alone brings a substantial +20% improvement to randomly initialized models
- The full four-modality combination yields the best performance

Segmentation results:

| Model | Cashew1k IoU | SAcrop3k IoU |
|------|-------------|-------------|
| Random | 40.1% → 72.6% | 18.0% → 20.3% |
| SatlasNet | 55.2% → 71.0% | 19.4% → 20.5% |

### Key Findings

1. **Remote sensing pretrained models benefit the most**: SeCo, SatlasNet, and similar models achieve gains of up to 10%, as WildSAT supplements habitat-relevant information
2. **ViT benefits more than CNN**: the flexible attention mechanism of Transformers more readily adapts to multimodal fusion
3. **WildSAT reduces false positives for habitat-related categories**: confusion matrix analysis on So2Sat20k shows improved true positive rates across all classes, primarily through reduced false positives for habitat categories such as "Scattered trees" and "Dense trees"
4. Zero-shot text-based retrieval is supported (e.g., querying "desert" or "ibex" retrieves satellite images of corresponding landscapes)

## Highlights & Insights

1. **A unique supervision signal**: wildlife observation data constitute free, globally distributed, naturally generated ecological labels that are complementary to anthropogenic features
2. **General-purpose framework**: WildSAT can serve as a continual pretraining stage to enhance existing models (SatlasNet, SeCo, and Prithvi all benefit)
3. **Zero-shot capability**: text alignment enables semantic retrieval of geographic locations, a capability absent from prior remote sensing representation methods
4. Complementarity with anthropogenic feature methods (WikiSatNet) — natural environment information and built-structure information together yield a more comprehensive understanding of Earth's surface

## Limitations & Future Work

1. Species observation data exhibit geographic bias (high density in Europe and North America, sparse in Africa and Asia), potentially limiting global generalization
2. Only RGB three-channel imagery is used; the multispectral advantages of Sentinel-2 are not fully exploited (preliminary multispectral experiments are included in the appendix)
3. Wikipedia text quality is uneven; descriptions of some species may be inaccurate or missing
4. Linear probing evaluation may underestimate the full representational capacity; results under full fine-tuning are not reported

## Related Work & Insights

- **SatlasPretrain**: large-scale supervised remote sensing pretraining; WildSAT demonstrates that its representations can be further improved
- **GRAFT**: ground-level image–satellite image alignment, but primarily targeting anthropogenic features
- **TaxaBind**: the first multimodal method to use species location and satellite imagery, but focused on ecological tasks rather than remote sensing
- Takeaway: citizen science data (eBird, iNaturalist) are an underutilized source of supervision signals, with potential applicability to a broader range of Earth observation tasks

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 4.5 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4.5 |
| Value | 4.5 |
| Overall | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AstroLoc: Robust Space to Ground Image Localizer](astroloc_robust_space_to_ground_image_localizer.md)
- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)
- [\[ICCV 2025\] Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening](pan-crafter_learning_modality-consistent_alignment_for_pan-sharpening.md)
- [\[NeurIPS 2025\] Scaling Image Geo-Localization to Continent Level](../../NeurIPS2025/remote_sensing/scaling_image_geo-localization_to_continent_level.md)
- [\[ICLR 2026\] Measuring the Intrinsic Dimension of Earth Representations](../../ICLR2026/remote_sensing/measuring_the_intrinsic_dimension_of_earth_representations.md)

</div>

<!-- RELATED:END -->

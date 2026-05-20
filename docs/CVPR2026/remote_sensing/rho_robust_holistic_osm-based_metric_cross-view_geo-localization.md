---
title: >-
  [Paper Note] RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization
description: >-
  [CVPR 2026][Remote Sensing][Cross-View Geo-Localization] This paper introduces CV-RHO, the first OSM-based metric cross-view geo-localization benchmark targeting adverse weather and sensor noise (2.72M+ images)…
tags:
  - "CVPR 2026"
  - "Remote Sensing"
  - "Cross-View Geo-Localization"
  - "OpenStreetMap"
  - "Panorama"
  - "Robustness"
  - "BEV"
date: 2026-05-08
content_hash: 85d3c496f7226c29
---

# RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization

**Conference**: CVPR 2026
**arXiv**: [2603.27758](https://arxiv.org/abs/2603.27758)  
**Code**: [https://github.com/InSAI-Lab/RHO](https://github.com/InSAI-Lab/RHO)  
**Area**: Remote Sensing / Visual Localization
**Keywords**: Cross-View Geo-Localization, OpenStreetMap, Panorama, Robustness, BEV

## TL;DR

This paper introduces CV-RHO, the first OSM-based metric cross-view geo-localization benchmark targeting adverse weather and sensor noise (2.72M+ images), and proposes RHO, a dual-branch Pin-Pan architecture integrating panoramic undistortion (SUM) and position-orientation fusion (POF) mechanisms, achieving up to 20% localization improvement under diverse degradation conditions.

## Background & Motivation

Cross-view geo-localization (CVGL) is a fundamental task in computer vision, encompassing large-scale retrieval (LCVGL) and metric-level fine localization (MCVGL). MCVGL estimates meter-level position and orientation by matching ground-to-satellite imagery from a coarse GPS prior, with significant applications in autonomous driving and remote sensing.

However, three critical gaps exist in prior work:

**Lack of robustness**: Existing MCVGL methods almost universally assume ideal illumination and weather, while real-world scenarios routinely involve rain, snow, fog, nighttime, and other degradation conditions. The authors demonstrate that OrienterNet trained under clean conditions suffers a substantial drop in Position Recall under degraded conditions (average $-8.22\%$ @1m).

**Underutilization of panoramic information**: Compared to pinhole images, 360° panoramas provide richer visual context beneficial for position and orientation estimation, yet naive panoramic input introduces severe distortion artifacts.

**Underexploited advantages of OSM**: OpenStreetMap is updated more frequently than satellite imagery and requires only 1/15 the storage overhead (4.8 MB/km² vs. 75 MB/km²), yet no large-scale robust OSM-MCVGL benchmark exists.

## Method

### Overall Architecture

RHO adopts a dual-branch Pin-Pan architecture (Figure 3). The overall pipeline is as follows:

- **Panoramic branch**: 360° panorama → SUM module (undistortion) → ViT encoder → BEV projection → matching with OSM feature map → 3D probability volume $(u, v, \theta)$
- **Pinhole branch**: 120° pinhole image → encoder → BEV projection → matching with OSM feature map → 3D probability volume
- **POF module**: fuses the two probability volumes to output the final 3-DoF camera pose

### Key Designs

1. **Split-Undistort-Merge (SUM) Module**: Addresses severe distortion caused by equirectangular panoramic projection.

    - Splits one panorama into three 120° pinhole images (covering 0°–120°, 120°–240°, 240°–360°)
    - Applies undistortion to each pinhole image
    - Passes each through the image encoder and BEV projection to obtain three BEV feature maps
    - Merges them in BEV space into a complete 360° BEV feature map
    - **Design Motivation**: Training OrienterNet directly on 360° panoramas yields only 3.79% PR@1m, far below the 21.83% achieved with the pinhole variant

2. **Position-Orientation Fusion (POF) Module**: Exploits the complementary information provided by panoramic and pinhole views.

    - **Stage 1**: Marginalizes the panoramic probability volume along the orientation dimension via LogSumExp to obtain a 2D spatial prior, then uses this prior to weight and enhance the positional information in the pinhole probability volume (Equations 2–5)
    - **Stage 2**: Marginalizes the enhanced pinhole probability volume along the spatial dimensions to obtain an orientation prior, then uses this prior to weight and enhance the orientation information in the panoramic probability volume (Equations 6–8)
    - **Core Idea** grounded in information theory: the Shannon entropy of panoramic images is invariant to rotation → suitable for position estimation; the Shannon entropy of pinhole images varies with rotation → suitable for orientation estimation
    - $\alpha$ and $\beta$ are learnable hyperparameters

3. **CV-RHO Dataset Construction**: The first large-scale robust OSM-MCVGL benchmark.

    - Covers 7 cities across the US, Germany, and France; 114K panoramas and 342K pinhole images
    - 8 degradation variants: rain, snow, fog, nighttime (generated via FLUX.1 Kontext, consuming 30.7K A100 GPU hours); overexposure, underexposure, and motion blur (generated via OpenCV)
    - 2.72M+ images in total
    - Additional cross-region test set (Mount Vernon) and Sim2Real test set

### Loss & Training

- Training objective: maximize the probability estimate of the 3-DoF camera pose using NLL loss
- Training resources: 12× A100 GPUs
- Rotation angle sampling: 64 during training, 256 during evaluation
- Batch size 36, learning rate $2\times10^{-5}$, Adam optimizer with ReduceLROnPlateau scheduler
- Best model checkpoint appears at 2–4 epochs (~20K–40K steps)

## Key Experimental Results

### Main Results

**Clean Conditions** (Table 3)

| Method | FoV | PR@1m | PR@3m | PR@5m | OR@1° | OR@3° | OR@5° |
|--------|-----|-------|-------|-------|-------|-------|-------|
| OrienterNet | 90° | 18.02 | 58.37 | 71.04 | 27.72 | 63.86 | 77.50 |
| OrienterNet | 120° | 21.83 | 66.16 | 78.03 | 35.02 | 74.89 | 85.62 |
| OrienterNet | 360° | 3.79 | 19.35 | 28.78 | 10.29 | 28.43 | 36.87 |
| **RHO** | **360°** | **24.59** | **73.55** | **84.36** | **43.46** | **83.61** | **90.44** |

**Robustness under Degradation** (Table 4, trained on Clean, tested on degraded conditions)

| Method | Train→Test / Avg. Degradation | PR@1m Drop | OR@1° Drop |
|--------|-------------------------------|------------|------------|
| OrienterNet | Clean→AV | −8.22 | −10.98 |
| **RHO** | **Clean→AV** | **−5.97** | **−9.95** |

**Trained and Tested on Degraded Conditions** (Table 4, lower section)

| Method | Matched Train→Test / Avg. Degradation | PR@1m | OR@1° |
|--------|---------------------------------------|-------|-------|
| OrienterNet | AV→AV | −2.04 | −1.82 |
| **RHO** | **AV→AV** | **+0.03** | **−1.10** |

Under matched training and testing conditions, RHO exhibits virtually no performance degradation, with PR@1m even showing a marginal improvement.

### Ablation Study

| Configuration | PR@3m | OR@3° | Notes |
|---------------|-------|-------|-------|
| Pinhole branch only | ~66 | ~75 | Baseline: OrienterNet 120° |
| Panorama only (w/o SUM) | ~19 | ~28 | Severe distortion |
| Panorama + SUM | ~70 | ~80 | SUM effectively resolves distortion |
| Panorama + SUM + POF | **73.55** | **83.61** | POF further improves +3.5/+3.6 |

### Key Findings

- Directly training on 360° panoramas yields extremely poor results (PR@1m: 3.79%); the SUM module is critical for the panoramic branch to function effectively.
- POF's two-stage mutual injection significantly outperforms naive probability volume concatenation or single-branch approaches.
- After training under matched degradation conditions, RHO's average performance drop approaches zero (PR@1m: +0.03), demonstrating strong robustness.
- Motion blur is the most challenging degradation type, producing the largest performance drop even for RHO.

## Highlights & Insights

1. **Information-theoretic architecture design**: Shannon entropy analysis of panoramic and pinhole images provides theoretical justification for their complementary roles in position and orientation estimation, motivating the dual-branch design.
2. **Lightweight distortion handling**: The SUM module requires no additional training and leverages standard panorama-to-pinhole projection to resolve distortion.
3. **First OSM-MCVGL robustness benchmark**: CV-RHO fills a critical data gap in the field and makes a significant contribution to advancing robust localization research.
4. **Sim2Real feasibility**: Models trained on degraded images generated by FLUX.1 Kontext also exhibit favorable zero-shot performance on real degraded scenes.

## Limitations & Future Work

- Motion blur induces the largest performance drop, suggesting the need for targeted data augmentation or blur-robust feature design.
- The fixed decomposition of panoramas into three 120° views in SUM could be replaced by an adaptive splitting strategy.
- A domain gap persists between synthetic and real degradation; additional domain adaptation techniques could be explored.
- Computational overhead is not discussed; the dual-branch architecture may limit real-time applicability.
- Although OSM data is updated frequently, it is not real-time, and may still fail in rapidly changing environments such as construction zones.

## Related Work & Insights

- RHO extends OrienterNet with robustness and panoramic capability, evolving from a single-branch pinhole design to a dual-branch panorama-plus-pinhole architecture.
- The two-stage mutual injection strategy in POF is transferable to other multi-modal probabilistic fusion scenarios.
- The CV-RHO dataset construction pipeline (FLUX.1 Kontext + OpenCV-simulated degradation) can serve as a reference for building other visual robustness benchmarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-branch architecture and POF fusion design are novel, though the core method remains grounded in the BEV matching framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage across multiple conditions, settings, cross-region, and Sim2Real experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — The dataset and robustness analysis make substantial contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization](../../AAAI2026/remote_sensing/uniabg_unified_adversarial_view_bridging_and_graph_correspondence_for_unsupervis.md)
- [\[CVPR 2026\] GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](geoflow_real-time_fine-grained_cross-view_geolocalization.md)
- [\[ICCV 2025\] GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization](../../ICCV2025/remote_sensing/geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo.md)
- [\[NeurIPS 2025\] Scaling Image Geo-Localization to Continent Level](../../NeurIPS2025/remote_sensing/scaling_image_geo-localization_to_continent_level.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)

</div>

<!-- RELATED:END -->

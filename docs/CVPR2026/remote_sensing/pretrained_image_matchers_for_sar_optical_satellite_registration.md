---
title: >-
  [Paper Note] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?
description: >-
  [CVPR 2026][Remote Sensing][SAR-optical registration] This paper evaluates 24 families of pretrained image matchers on SAR-optical satellite registration under a zero-shot setting, finding that deployment protocol choices (geometric model, tile size, etc.) can affect accuracy by up to 33×, sometimes surpassing the effect of switching the matcher itself.
tags:
  - CVPR 2026
  - Remote Sensing
  - SAR-optical registration
  - image matching
  - cross-modal
  - zero-shot transfer
  - satellite imagery
date: 2026-05-08
content_hash: c834374275cc568b
---

# Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?

**Conference**: CVPR 2026
**arXiv**: [2604.10217](https://arxiv.org/abs/2604.10217)
**Code**: None
**Area**: Remote Sensing Imagery
**Keywords**: SAR-optical registration, image matching, cross-modal, zero-shot transfer, satellite imagery

## TL;DR

This paper evaluates 24 families of pretrained image matchers on SAR-optical satellite registration under a zero-shot setting, finding that deployment protocol choices (geometric model, tile size, etc.) can affect accuracy by up to 33×, sometimes surpassing the effect of switching the matcher itself.

## Background & Motivation

**Background**: Cloud cover during disaster response frequently renders optical imagery unavailable, necessitating the registration of SAR images to optical basemaps for generating georeferenced damage assessments. However, state-of-the-art image matchers are designed for indoor, urban, or natural scene imagery.

**Limitations of Prior Work**: Optical and SAR sensors observe the same scene through fundamentally different physical mechanisms — optical sensors capture reflected light (texture-rich), while SAR captures radar backscatter (speckle noise, layover, radiometric inversion). Whether pretrained matchers can function under such extreme domain shift remains unclear.

**Key Challenge**: Cross-modal matching requires modality-invariant feature representations, yet pretrained data contains virtually no satellite or SAR imagery.

**Goal**: Evaluate the cross-modal satellite registration performance of 24 matcher families in a purely zero-shot setting, without any fine-tuning or domain adaptation.

**Key Insight**: A unified, deterministic evaluation protocol encompassing large-image tile-based inference, robust geometric filtering, and tie-point-anchored metrics.

**Core Idea**: Cross-modal transfer is asymmetric — explicit cross-modal training does not consistently outperform training on natural images alone, and foundation model features may partially substitute for cross-modal supervision.

## Method

### Overall Architecture

24 matcher families × unified evaluation protocol (tile-based inference + RANSAC geometric filtering + tie-point metrics) × SpaceNet9 and two additional cross-modal benchmarks. Deployment protocol choices (geometric model, tile size, inlier gating) are systematically ablated.

### Key Designs

1. **Unified Evaluation Protocol**:

    - **Function**: Ensures all 24 matchers are evaluated under fully comparable conditions.
    - **Mechanism**: Tile-based inference for handling the very high resolution of satellite imagery, affine RANSAC or fundamental matrix RANSAC for geometric filtering, and tie-point-anchored reprojection error metrics.
    - **Design Motivation**: Different matcher papers employ different evaluation setups, making direct comparison meaningless without standardization.

2. **Deployment Protocol Sensitivity Analysis**:

    - **Function**: Quantifies the impact of hyperparameter choices on registration accuracy.
    - **Mechanism**: Systematic sweep over protocol parameters including geometric model (affine/fundamental matrix/homography), tile size, and inlier gating threshold. Using affine geometry alone reduces mean error from 12.34 px to 9.74 px. A single matcher can exhibit up to 33× accuracy variation across different protocols.
    - **Design Motivation**: In practical deployment, protocol selection may matter more than matcher selection.

3. **Cross-Modal Transfer Asymmetry Finding**:

    - **Function**: Reveals that explicit cross-modal training is not a prerequisite for strong performance.
    - **Mechanism**: XoFTR (trained on visible-thermal pairs) and RoMa (no cross-modal training) both achieve the lowest mean error of 3.0 px. MatchAnything-ELoFTR (trained on synthetic cross-modal pairs) follows closely at 3.4 px. DINOv2 foundation model features may provide partial modality invariance.
    - **Design Motivation**: Challenges the intuition that cross-modal tasks necessarily require cross-modal training.

### Loss & Training

This is a purely zero-shot evaluation study; no training is involved. All matchers use their official pretrained weights.

## Key Experimental Results

### Main Results

| Matcher | SpaceNet9 Mean Error (px) | Cross-Modal Training |
|--------|--------------------------|---------------------|
| XoFTR | 3.0 | Yes (visible–thermal) |
| RoMa | 3.0 | No |
| MatchAnything-ELoFTR | 3.4 | Yes (synthetic cross-modal) |
| MASt3R/DUSt3R | Protocol-sensitive | No (3D reconstruction) |

### Ablation Study

| Protocol Choice | Mean Error Change | Note |
|----------------|------------------|------|
| Affine geometry vs. others | 12.34→9.74 px | 21% reduction |
| Tile size variation | Up to 33× difference | For individual matchers |
| Inlier gating variation | Significant impact | Both overly strict and overly loose settings degrade performance |

### Key Findings

- The impact of deployment protocol choices can exceed that of switching matchers — affine geometry alone reduces error by 21%.
- 3D reconstruction-based matchers (MASt3R/DUSt3R) are highly fragile under default settings and are heavily dependent on protocol configuration.
- DINOv2 foundation model features may provide a form of implicit modality invariance.

## Highlights & Insights

- **"Protocol over algorithm" finding**: For practitioners, optimizing the deployment protocol may be more effective than replacing the matcher.
- **Cross-modal transfer asymmetry**: Counterintuitively, RoMa achieves the lowest error without any cross-modal training.
- **Hypothesis of implicit modality invariance in DINOv2**: Foundation model features may inherently generalize across modalities.

## Limitations & Future Work

- Only zero-shot performance is evaluated; the effect of few-shot fine-tuning remains unexplored.
- Scene coverage in SpaceNet9 may be limited (primarily urban areas).
- The hypothesis of implicit modality invariance in DINOv2 requires deeper mechanistic analysis.

## Related Work & Insights

- **vs. RemoteCLIP**: RemoteCLIP achieves domain adaptation through large-scale remote sensing pretraining, whereas this paper demonstrates that zero-shot transfer may already be sufficient.
- **vs. LoFTR/ELoFTR**: Standard natural-image matchers exhibit inconsistent performance on cross-modal satellite registration; deployment protocol is the critical factor.

## Rating

- Novelty: ⭐⭐⭐ Empirical study, but findings are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 24 matcher families × multiple protocols × multiple benchmarks — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ In-depth analysis with a practical orientation.
- Value: ⭐⭐⭐⭐ Provides direct guidance for real-world deployment in applications such as disaster response.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](sdfnet_structureaware_disentangled_feature_learnin.md)
- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](../../ICCV2025/remote_sensing/wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)
- [\[CVPR 2026\] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)
- [\[AAAI 2026\] Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification](../../AAAI2026/remote_sensing/perceive_act_and_correct_confidence_is_not_enough_for_hyperspectral_classificati.md)

<!-- RELATED:END -->

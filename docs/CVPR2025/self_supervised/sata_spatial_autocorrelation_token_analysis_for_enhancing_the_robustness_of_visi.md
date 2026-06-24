---
title: >-
  [Paper Note] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers
description: >-
  [CVPR 2025][Self-Supervised Learning][Vision Transformer] This paper proposes SATA (Spatial Autocorrelation Token Analysis), a training-free robustness enhancement method for ViTs. By grouping tokens based on spatial correlation patterns via spatial autocorrelation analysis and reweighting token representations according to the grouping information, SATA improves ViT robustness under distribution shifts and adversarial attacks without compromising clean performance.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Vision Transformer"
  - "spatial autocorrelation"
  - "token analysis"
  - "robustness"
  - "training-free"
date: 2026-05-08
content_hash: 4a6e6032d233e3bd
---

# SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers

**Conference**: CVPR 2025  
**Code**: None  
**Area**: ViT Robustness  
**Keywords**: Vision Transformer, spatial autocorrelation, token analysis, robustness, training-free

## TL;DR

This paper proposes SATA (Spatial Autocorrelation Token Analysis), a training-free robustness enhancement method for ViTs. By grouping tokens based on spatial correlation patterns via spatial autocorrelation analysis and reweighting token representations according to the grouping information, SATA improves ViT robustness under distribution shifts and adversarial attacks without compromising clean performance.

## Background & Motivation

### Background

**Background**: Vision Transformers (ViTs) have demonstrated outstanding performance across various visual recognition tasks, yet their robustness under distribution shifts (corruptions, domain shifts) and adversarial attacks still requires improvement. Existing methods for enhancing ViT robustness include adversarial training, patch augmentation, and architectural modifications.

**Limitations of Prior Work**: (1) High training costs: Adversarial training and data augmentation strategies require retraining from scratch or long training times, which are computationally expensive. (2) Clean performance degradation: Many robustness enhancement methods compromise accuracy on clean samples while improving robustness (the robustness-accuracy trade-off). (3) Neglecting spatial relations between tokens: Existing methods primarily enhance robustness from the perspective of data or network architecture, failing to fully utilize the spatial distribution characteristics of token features in ViTs.

**Key Challenge**: ViTs partition an image into patch tokens and process them with self-attention. However, when inputs are perturbed, features of certain tokens deviate from the normal distribution. Identifying and correcting these "deviated" tokens during inference (without retraining) is a key challenge.

**Goal** How to leverage spatial relations among tokens to enhance ViT robustness without retraining?

**Key Insight**: Drawing inspiration from the concept of Spatial Autocorrelation in geography—Tobler's First Law of Geography stating that "near things are more related than distant things" also applies to the feature relationships of spatially adjacent tokens in images. Perturbed tokens disrupt this spatial autocorrelation pattern.

**Core Idea**: Analyze spatial clustering patterns of token features using spatial autocorrelation statistics such as Moran's I, thereby identifying abnormal tokens and reweighting them to enhance robustness.

## Method

### Overall Architecture

SATA is applied as a plug-and-play module during the inference stage of pre-trained ViTs: (1) Extract token features and their spatial positions from intermediate layers of the ViT. (2) Compute spatial autocorrelation statistics (Moran's I) of token features to measure the feature similarity of spatially adjacent tokens. (3) Categorize tokens into spatial clusters such as "High-High" (high-value clustering) and "Low-Low" (low-value clustering) based on spatial autocorrelation patterns. (4) Revise features of abnormal tokens that deviate from the school patterns (by smoothing towards neighborhood mean) or decrease their attention weights.

### Key Designs

1. **Spatial Autocorrelation Token Analysis**:
    - Function: Quantify the feature consistency of each token with its spatial neighbors in the ViT.
    - Mechanism: For token features $\{z_i^l\}_{i=1}^N$ in the $l$-th layer of a ViT, construct a spatial weight matrix $W$ (based on 2D grid positions of tokens, where weights of adjacent tokens are set to 1, and 0 otherwise). Compute the local Moran's I statistic: $I_i = \frac{(z_i - \bar{z})}{\sigma^2} \sum_j w_{ij} (z_j - \bar{z})$, where a positive value indicates feature consistency between the token and its neighbors (spatial clustering), and a negative value indicates an abnormal feature (spatial outlier). Tokens are grouped into four categories—HH, HL, LH, and LL—using the LISA (Local Indicator of Spatial Association) method.
    - Design Motivation: Adjacent regions in natural images typically share similar semantics, so token features should exhibit positive spatial autocorrelation. Perturbations (noise, adversarial attacks) destroy this local consistency, producing HL/LH-type abnormal tokens.

2. **Spatial Clustering-Based Token Reweight**:
    - Function: Correct the feature representation of abnormal tokens.
    - Mechanism: For spatial outlier tokens (HL or LH types), their features are smoothed using a weighted average of their spatial neighbors' features, with the smoothing intensity determined by the degree of deviation in the autocorrelation statistic. Spatially clustered tokens (HH, LL types) remain unchanged. This is equivalent to enhancing the weights of spatially consistent tokens and reducing the weights of abnormal tokens during attention computation.
    - Design Motivation: Abnormal tokens contain noise or adversarial signals. Correcting them toward neighborhood consistency reduces the impact of perturbations, resembling spatial filtering but operating in the token feature space.

3. **Multi-layer Adaptive Application Strategy**:
    - Function: Determine in which layers of the ViT to apply SATA.
    - Mechanism: Analysis reveals that different layers have varying sensitivity to perturbations. Shallow-layer tokens are closer to raw pixel features, where perturbations are directly reflected; deep-layer tokens have partially alleviated perturbations after passing through multiple attention layers. Therefore, applying SATA in intermediate layers (e.g., layers 3-6 in a 12-layer ViT) yields the best performance. Additionally, the smoothing intensity is layer-adaptive—stronger smoothing is applied in shallower layers.
    - Design Motivation: Applying the module in all layers incurs high overhead and might over-smooth, damaging semantic features.

### Loss & Training
The model is trained end-to-end, with the optimization objective comprehensively considering both task loss and regularization terms.


## Key Experimental Results

### Key Findings

- On ImageNet-C (15 types of corruptions), SATA improves the robust accuracy of DeiT-B by approximately 3-5% without requiring any retraining.
- Accuracy on clean samples suffers almost no loss (<0.3% drop), resolving the robustness-accuracy trade-off.
- Robustness against adversarial attacks (PGD, AutoAttack) is also improved by around 2-3%.
- The method is effective across various ViT variants (DeiT, Swin, CaiT, etc.).
- It is complementary to methods like adversarial training—applying them together further boosts robustness.
- Spatial autocorrelation statistics decrease significantly on attacked samples, verifying the methodological hypothesis.

## Highlights & Insights

- **Cross-disciplinary Innovation**: Introduces mature spatial statistical theories from geography into the study of ViT robustness.
- **Training-free and Plug-and-play**: Does not modify model parameters or require additional training, rendering deployment costs extremely low.
- **Strong Theoretical Support**: The analytical framework of spatial autocorrelation provides a new perspective for understanding perturbation propagation in ViTs.
- **No Clean Performance Damage**: Avoids the accuracy-robustness trade-off characteristic of traditional robustness methods.

## Limitations & Future Work

- Insensitive to large-scale global perturbations (such as full-image brightness shifts) via spatial autocorrelation, as all tokens shift consistently.
- Incurs additional inference overhead (around 5-10%) for computing spatial autocorrelation statistics.
- Overhead may be more significant for high-resolution inputs (which contain a large number of tokens).
- Future work could integrate frequency-domain analysis to further enhance abnormal token detection.


## Related Work & Insights
- **vs Representative Methods in the Same Field**: This work makes unique contributions to method design and is complementary to existing methods.
- **vs Traditional Methods**: Compared to traditional approaches, the proposed method achieves significant improvements in key metrics.
- **Insights**: The technical route of this work offers valuable references for subsequent research.


## Rating
- Novelty: ⭐⭐⭐⭐ Unique contributions to method design
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple datasets
- Writing Quality: ⭐⭐⭐⭐ Well-structured and clear
- Value: ⭐⭐⭐⭐ Facilitates development in the field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vision Transformers Need More Than Registers](../../CVPR2026/self_supervised/vision_transformers_need_more_than_registers.md)
- [\[CVPR 2025\] Transformers without Normalization](transformers_without_normalization.md)
- [\[CVPR 2025\] SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)
- [\[CVPR 2025\] Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](breaking_the_tuning_barrier_zero-hyperparameters_yield_multi-corner_analysis_via.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)

</div>

<!-- RELATED:END -->

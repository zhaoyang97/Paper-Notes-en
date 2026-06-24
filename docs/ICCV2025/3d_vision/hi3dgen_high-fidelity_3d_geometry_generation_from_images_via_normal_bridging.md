---
title: >-
  [Paper Note] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging
description: >-
  [ICCV 2025][3D Vision][3D generation] This paper proposes Hi3DGen, a framework that uses normal maps as an intermediate representation to bridge 2D images and 3D geometry. Through two core components — a Noise-injected Regressive Normal Estimation (NiRNE) module and Normal-Regularized Latent Diffusion (NoRLD) — the framework significantly improves the geometric detail fidelity of generated 3D models.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D generation"
  - "normal map bridging"
  - "noise-injected regression"
  - "dual-stream architecture"
  - "latent diffusion regularization"
  - "high-fidelity geometry"
date: 2026-05-08
content_hash: edfbc1b02bd495d7
---

# Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging

**Conference**: ICCV 2025
**arXiv**: [2503.22236](https://arxiv.org/abs/2503.22236)  
**Area**: Other
**Keywords**: 3D generation, normal map bridging, noise-injected regression, dual-stream architecture, latent diffusion regularization, high-fidelity geometry

## TL;DR

This paper proposes Hi3DGen, a framework that uses normal maps as an intermediate representation to bridge 2D images and 3D geometry. Through two core components — a Noise-injected Regressive Normal Estimation (NiRNE) module and Normal-Regularized Latent Diffusion (NoRLD) — the framework significantly improves the geometric detail fidelity of generated 3D models.

## Background & Motivation

Generating high-fidelity 3D models from 2D images is a fundamental task in computer vision. However, existing methods still face three major challenges in recovering fine-grained geometric details:

**Scarcity of high-quality 3D training data**: Datasets such as Objaverse contain insufficient high-quality 3D assets with complex geometric details, leading to overly simplified generated models.

**Domain gap**: Training data typically comes from synthetic renderings, which differ substantially in style from real-world images, causing notable performance degradation at inference time.

**Inherent ambiguity in RGB images**: Lighting, shading, and complex textures make it difficult to extract accurate geometric information from RGB inputs.

Existing direct RGB-to-3D methods (e.g., CRM, InstantMesh, CraftsMan) fail to adequately preserve fine-grained geometric features from input images. Normal maps, as a 2.5D representation encoding surface orientation, provide inherently clearer geometric cues and can leverage strong 2D priors to mitigate the domain gap.

**Core Insight**: Decomposing image-to-3D into two steps — image-to-normal and normal-to-geometry — and using normal maps as a bridge (normal bridging) can simultaneously alleviate both the domain gap and geometric ambiguity.

## Method

### Overall Architecture

Hi3DGen consists of three core components:
- **NiRNE** (Noise-injected Regressive Normal Estimation): estimates high-quality normal maps from images.
- **NoRLD** (Normal-Regularized Latent Diffusion): leverages normal regularization to enhance 3D latent diffusion learning.
- **DetailVerse Dataset**: a synthesized high-quality 3D dataset providing training data with rich geometric details.

### Key Design 1: Noise-injected Regressive Normal Estimation (NiRNE)

**Problem Analysis**: Diffusion-based methods produce sharper but unstable normals with spurious details; regression-based methods are stable but lack sharpness. The authors analyze the fundamental reason diffusion methods yield sharper results from a frequency-domain perspective.

**Frequency-Domain Analysis**: In the diffusion process $x_t = x_0 + \int_0^t g(s) dw_t$, because natural images exhibit low-pass characteristics $|\hat{x}_0(\omega)|^2 \propto |\omega|^{-\alpha}$, high-frequency components experience faster SNR decay. This means diffusion models receive stronger supervision signals in the high-frequency regime, driving them to focus on sharp details.

**Noise Injection**: Motivated by this analysis, noise injection is integrated into the regression framework to enhance sensitivity to high-frequency information, achieving both sharpness and stability.

**Dual-Stream Architecture**:
- **Clean Stream**: processes the original noise-free image to robustly capture low-frequency information (overall structural content).
- **Noisy Stream**: processes the noise-injected image to focus on learning high-frequency details (edges and fine-grained textures).
- Features from both streams are concatenated in a ControlNet-style manner before being fed into the decoder for regression prediction.

**Domain-Specific Training Strategy**:
- **Stage 1**: The full network is trained on real-domain data to learn low-frequency information for strong generalization.
- **Stage 2**: The Clean Stream is frozen; only the Noisy Stream is fine-tuned on synthetic-domain data to learn high-frequency details as residuals.

This design elegantly exploits the complementary strengths of real data (good generalization but noisy high-frequency labels) and synthetic data (precise high-frequency labels but domain gap).

### Key Design 2: Normal-Regularized Latent Diffusion (NoRLD)

Existing 3D latent diffusion methods (e.g., Trellis, CRM) apply supervision only in heavily compressed latent spaces, making geometric detail easy to lose.

**Core Idea**: During diffusion training, normal-map regularization is applied online to the predicted latent codes to provide explicit 3D geometry supervision:

$$\mathcal{L}_{\text{NoRLD}} = \mathcal{L}_{\text{LDM}} + \lambda \cdot \mathcal{R}_{\text{Normal}}(\hat{x}_0)$$

where the normal regularization term is:

$$\mathcal{R}_{\text{Normal}}(\hat{x}_0) = \mathbb{E}_v \left[ \| R_v(D(\hat{x}_0)) - N_v \|^2 \right]$$

This decodes the predicted latent code $\hat{x}_0$ into 3D geometry, renders normal maps from viewpoint $v$, and compares them against ground-truth normal maps. This regularization is performed **online** during diffusion training (not as post-processing), actively guiding the diffusion network to learn distributions rich in geometric detail.

### Loss & Training

- **NiRNE**: Standard regression loss (L2) with domain-specific two-stage training.
- **NoRLD**: Flow matching loss + online normal rendering regularization loss.
- $\mathcal{L}_{\text{LDM}}$: Standard velocity field matching loss.
- $\lambda \cdot \mathcal{R}_{\text{Normal}}$: Online normal regularization.

### DetailVerse Dataset

A high-quality dataset constructed via a 3D data synthesis pipeline to compensate for the shortage of high-quality assets in Objaverse:
- Features high semantic diversity, geometric structural diversity, and rich surface detail.
- Provides clean normal labels for NiRNE and high-fidelity 3D training data for NoRLD.

## Key Experimental Results

### Main Results

The paper presents comparisons against multiple state-of-the-art methods (the experimental section is truncated in the cache; the following is summarized from the paper description):

- Hi3DGen surpasses all SOTA methods in geometric detail fidelity of generated models.
- Baselines include CRM, InstantMesh, CraftsMan, and Trellis.
- Performance on real-world input images is particularly strong, benefiting from the mitigation of the domain gap.

### Normal Estimation Comparison

NiRNE simultaneously achieves the sharpness of diffusion-based methods and the stability of regression-based methods:
- Compared to StableNormal: more stable, without spurious details.
- Compared to Marigold: comparable sharpness, but faster inference (single-step regression vs. multi-step diffusion).
- Compared to traditional regression methods (e.g., DSINE): richer and sharper details.

### Key Findings

- The normal bridging strategy effectively mitigates the domain gap, enabling high-fidelity geometry generation across diverse input image styles.
- Online normal regularization significantly improves geometric detail retention in latent diffusion learning.
- The introduction of the DetailVerse dataset is critical for learning high-frequency details.

## Highlights & Insights

1. **Frequency-domain analysis is highly insightful**: Explaining why diffusion models excel at high frequencies through the lens of SNR decay, and using this insight to introduce noise injection into a regression framework, represents an elegant integration of theory and practice.
2. **Dual-stream decoupling is well-designed**: Separating low-frequency generalization and high-frequency detail capability into two independent streams, combined with a domain-specific training strategy, fully leverages the advantages of different data sources.
3. **Online normal regularization is a key innovation**: Unlike CraftsMan's post-processing regularization or Trellis's use of normal loss solely during VAE training, NoRLD introduces normal supervision online during diffusion training, making it more direct and effective.
4. **Problem decomposition is a valuable paradigm**: Decomposing the challenging image-to-3D task into two relatively tractable sub-problems and using an intermediate representation (normal maps) reduces overall learning difficulty.
5. **Data supplementation strategy is practical**: Synthesizing the DetailVerse dataset to compensate for deficiencies in existing datasets represents a viable solution to the scarcity of high-quality 3D data.

## Limitations & Future Work

1. Cache truncation prevents access to complete quantitative experiments and ablation study results.
2. The dual-stream architecture combined with VAE decoding introduces substantial computational overhead for normal regularization.
3. Performance may be limited for categories not well covered by the DetailVerse dataset, given its dependence on data quality and diversity.
4. While normal maps as an intermediate representation alleviate ambiguity, they cannot fully resolve depth ambiguity (normals encode orientation rather than distance).
5. Errors may accumulate across the two-stage pipeline (normal estimation followed by 3D generation).

## Related Work & Insights

- **3D Generation**: Direct image-to-3D methods including CRM, InstantMesh, CraftsMan, Trellis, and Unique3D.
- **Normal Estimation**: Diffusion-based (Marigold, GeoWizard, StableNormal) vs. regression-based (DSINE, Metric3D).
- **Normal Maps in 3D**: Normal rendering losses in SDS optimization; multi-view normal fusion.
- **3D Datasets**: Objaverse, Objaverse-XL, MVImgNet.

## Rating

- **Novelty**: ★★★★☆ — Normal bridging is not entirely novel, but noise-injected regression and online normal regularization represent significant innovations.
- **Technical Depth**: ★★★★★ — The frequency-domain analysis is rigorous; the dual-stream design and domain-specific training strategy are well-grounded theoretically.
- **Experimental Quality**: ★★★☆☆ — Cannot be fully assessed due to cache truncation, though the experimental design appears sound based on the method description.
- **Practicality**: ★★★★☆ — High-fidelity generated 3D models have direct application value in gaming, film production, 3D printing, and related fields.
- **Writing Clarity**: ★★★★★ — The paper is well-structured, with thorough explanations of the frequency-domain analysis and method motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_high-qualit.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] LATTICE: Democratize High-Fidelity 3D Generation at Scale](../../CVPR2026/3d_vision/lattice_democratize_high-fidelity_3d_generation_at_scale.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[CVPR 2026\] Skullptor: High Fidelity 3D Head Reconstruction in Seconds with Multi-View Normal Prediction](../../CVPR2026/3d_vision/skullptor_high_fidelity_3d_head_reconstruction_in_seconds_with_multi-view_normal.md)

</div>

<!-- RELATED:END -->

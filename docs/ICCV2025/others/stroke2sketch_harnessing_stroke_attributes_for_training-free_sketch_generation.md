---
title: >-
  [Paper Note] Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation
description: >-
  [ICCV 2025][sketch generation] This paper proposes Stroke2Sketch, a training-free reference-guided sketch generation framework that achieves fine-grained stroke attribute transfer while preserving content structure withi…
tags:
  - "ICCV 2025"
  - "sketch generation"
  - "stroke attribute transfer"
  - "training-free"
  - "diffusion models"
  - "cross-image attention"
date: 2026-05-08
content_hash: cb71d67e67c01495
---

# Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation

**Conference**: ICCV 2025
**arXiv**: [2510.16319](https://arxiv.org/abs/2510.16319)  
**Code**: [https://github.com/rane7/Stroke2Sketch](https://github.com/rane7/Stroke2Sketch)  
**Area**: Sketch Generation / Style Transfer
**Keywords**: sketch generation, stroke attribute transfer, training-free, diffusion models, cross-image attention

## TL;DR

This paper proposes Stroke2Sketch, a training-free reference-guided sketch generation framework that achieves fine-grained stroke attribute transfer while preserving content structure within a pretrained diffusion model, via three collaborative modules: Cross-image Stroke Attention (CSA), Directive Attention Module (DAM), and Semantic Preservation Module (SPM).

## Background & Motivation

Reference-guided sketch generation aims to produce a sketch that preserves the content structure of a given content image while adopting the stroke style (line thickness, curvature, texture density, etc.) of a reference sketch. This task poses three fundamental challenges:

**Semantics-aware stroke transfer**: Stroke attributes from the reference must be mapped precisely to semantically corresponding content regions, rather than through simple global style blending.

**Foreground prioritization**: Human artists naturally emphasize foreground subjects with rich strokes and simplify backgrounds, yet existing methods apply uniform stylization across all regions.

**Content–style balance**: Sketches encode content through line structures, and content leakage can destroy critical edge configurations.

**Limitations of prior work**:
- **Training-based methods** (Ref2Sketch, Semi-Ref2Sketch): fail to generalize to unseen styles due to catastrophic forgetting.
- **IP-Adapter / InstantStyle**: excel at texture transfer but suffer from poor structural integrity due to content leakage in cross-attention.
- **ControlNet-augmented methods**: preserve structure too rigidly, sacrificing style flexibility.
- **Progressive stroke methods** (RB-Modulation): uniform strokes lead to semantic inconsistency.

**Core Idea**: Stroke attributes (line thickness, curvature, texture density) are intrinsically encoded in the self-attention and cross-attention relationships of pretrained diffusion models. By dynamically aligning attention patterns between content and reference features, style transfer can be achieved without compromising structural integrity.

## Method

### Overall Architecture

**Inputs**: a content image $I^{cnt}$ and a reference sketch $I^{ref}$. Latent representations of the content, reference, and contour (extracted via the TEED edge detector) are obtained through DDPM inversion. During denoising, three modules collaborate to generate the stylized sketch.

### Key Designs

1. **Cross-image Stroke Attention (CSA)**: Performs Key-Value substitution within the self-attention layers of the diffusion model. The K/V features of the reference sketch are blended with those of the content image and injected into the generation process:

    $K^{ske}_t = K^{ref}_t + \alpha K^{cnt}_t, \quad V^{ske}_t = V^{ref}_t + \alpha V^{cnt}_t$

   where $\alpha$ controls the mixing ratio between reference and content. Unlike direct feature blending (e.g., InstantStyle), this approach leverages the attention mechanism to naturally map stroke features to semantically corresponding regions. However, direct K-V substitution may distort certain structural elements (e.g., curves), necessitating the complementary modules described below.

2. **Directive Attention Module (DAM)**: Addresses the problem of non-uniform stylization across foreground and background. The procedure is as follows:

    - Self-attention feature maps $F_{SA}$ at $32\times32$ resolution are extracted and aggregated via channel averaging.
    - KMeans clustering is applied to obtain segmentation masks $M_j$.
    - Cross-attention maps $A_n$ for nouns extracted by BLIP are used to compute a foreground relevance score for each cluster: $r(j,n) = \frac{\sum M_j \cdot A_n}{\sum M_j + \delta}$
    - Clusters with relevance $> 0.35$ are labeled as foreground; style transfer is suppressed in background regions.

3. **Semantic Preservation Module (SPM)**: Addresses noise and misalignment arising from semantic mismatch between the reference sketch and content image. Dual guidance is applied:

    - **Text guidance**: High-level semantics are preserved via a CLIP loss $L_{sem} = \lambda \cdot \text{CLIP}(I^{ske}, T^{cnt})$.
    - **Contour guidance**: Query features cached from the DDPM inversion of the contour are injected into the generation queries: $Q^{ske}_{i+1} = \gamma Q^{cont}_i + (1-\gamma) Q^{ske}_i$ (default $\gamma=0.25$). The contour serves as a soft constraint rather than the rigid constraint imposed by ControlNet.

4. **Stroke Detail Propagation Enhancement (SDPE)**: Low-contrast noise is suppressed via adaptive contrast enhancement $\text{Enhance}(A) = (A - \mu(A))\zeta(\sigma(A)) + \mu(A)$. A parallel dual-channel CFG is employed: one branch uses cross-image attention to capture stroke features, and the other uses text guidance to preserve semantics. The final noise prediction is:

    $\epsilon^t = \epsilon^{self} + \beta_{sg}(\epsilon^{\times}_{stroke} - \epsilon^{self}) + \beta_{text}(\epsilon^{\times}_{text} - \epsilon^{self})$

### Loss & Training

Entirely training-free — built upon pretrained Stable Diffusion v2.1-base, with DDPM inversion for image reversal and DDIM 50-step denoising. All modules operate by manipulating attention layers without modifying network parameters.

## Key Experimental Results

### Main Results (Stroke2Sketch-dataset)

| Method | ArtFID ↓ | LPIPS ↓ | FID ↓ |
|--------|---------|---------|-------|
| Ref2sketch | 45.292 | 0.6982 | 34.650 |
| Semi-ref | 33.242 | 0.5306 | 24.359 |
| IP-Adapter | 33.457 | 0.6634 | 24.068 |
| InstantStyle | 32.532 | 0.5432 | 23.940 |
| InstantStyle+ | 37.656 | 0.6532 | 26.632 |
| StyleID | 35.727 | 0.5426 | 25.658 |
| **Ours** | **32.455** | **0.5315** | **22.435** |

### Ablation Study

| Configuration | ArtFID ↓ | FID ↓ | LPIPS ↓ |
|--------------|---------|-------|---------|
| A: Full (Ours) | **32.45** | **22.43** | **0.530** |
| B: − DAM | 38.67 | 26.53 | 0.672 |
| C: − SPM | 36.89 | 30.47 | 0.637 |
| D: − SDPE | 40.53 | 32.44 | 0.598 |

Removing SDPE causes the most severe degradation (ArtFID rising from 32.45 to 40.53), indicating that stroke detail propagation enhancement is critical to final quality.

### Key Findings

- **User study** (2,000 votes / 100 users): Stroke2Sketch achieves the highest preference across all three dimensions — content fidelity, stroke stylization, and overall preference.
- On the FS2K face sketch dataset, the method achieves the lowest FID (128.84 vs. 185.26 for the runner-up) and LPIPS (0.4057 vs. 0.4540).
- Colored sketch generation is supported (Fig. 9), preserving reference stroke characteristics and artistic style.
- Hyperparameters $\gamma$ (contour weight), $\beta_{sg}$ (stroke guidance scale), and $\zeta$ (contrast intensity) provide flexible user control.

## Highlights & Insights

- **Precise problem formulation**: The paper explicitly distinguishes "stroke attribute transfer" from general "style transfer," as the former requires finer-grained semantic correspondence.
- **Complementary three-module design**: CSA handles stroke injection, DAM handles region selection, and SPM handles structural constraint — with a clean decoupling of responsibilities.
- Unsupervised foreground segmentation is achieved by clustering diffusion model self-attention features, without requiring an additional segmentation model — an elegant design choice.
- Fully training-free: requires no sketch datasets, no fine-tuning, and is applicable to any reference style.

## Limitations & Future Work

- Performance degrades on overly minimalist references (e.g., single continuous-line drawings) or highly complex ones (dense fine strokes).
- Complete disentanglement of semantic information and stroke attributes remains unsolved — semantic leakage persists in certain cases.
- The method depends on the quality of BLIP text extraction and TEED edge detection.
- Multiple hyperparameters ($\gamma$, $\beta_{sg}$, $\zeta$) require manual tuning according to style type.

## Related Work & Insights

- **Cross-Image Attention (CIA)** and **StyleAligned** demonstrate that self-attention encodes critical style information.
- The CLIP-space style subtraction strategy of **InstantStyle**, while conceptually simple, suffers from severe content leakage in the sketch domain.
- The proposed approach may generalize to other reference-guided artistic generation tasks, such as ink wash painting and oil painting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Cross-image stroke attention mechanism combined with unsupervised foreground focusing constitutes a novel design.
- **Theoretical Depth**: ⭐⭐⭐ — Primarily engineering-driven; formal theoretical analysis is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-baseline comparisons, complete ablations, and a convincing user study.
- **Practicality**: ⭐⭐⭐⭐ — Ready to use without training, though hyperparameter tuning presents a non-trivial barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [\[NeurIPS 2025\] Optimized Learned Count-Min Sketch](../../NeurIPS2025/others/optimized_learned_count-min_sketch.md)
- [\[ICML 2026\] SORA: Free Second-Order Attacks in Fast Adversarial Training](../../ICML2026/others/sora_free_second-order_attacks_in_fast_adversarial_training.md)
- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)
- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)

</div>

<!-- RELATED:END -->

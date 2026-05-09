---
title: >-
  [Paper Note] ExpPortrait: Expressive Portrait Generation via Personalized Representation
description: >-
  [CVPR 2026][Image Generation][Portrait Animation] This paper proposes a high-fidelity personalized head representation (static identity offset + dynamic expression offset) to address the limited expressiveness of parametric models such as SMPL-X. Combined with an identity-adaptive expression transfer module and a DiT-based generator, the method achieves state-of-the-art performance on both self-driven portrait video animation and cross-identity reenactment tasks.
tags:
  - CVPR 2026
  - Image Generation
  - Portrait Animation
  - Personalized Head Representation
  - Expression Transfer
  - Diffusion Transformer
  - SMPL-X
date: 2026-05-08
content_hash: a74101ff5012dc01
---

# ExpPortrait: Expressive Portrait Generation via Personalized Representation

**Conference**: CVPR 2026
**arXiv**: [2602.19900](https://arxiv.org/abs/2602.19900)
**Code**: None
**Area**: Portrait Generation / Face Reenactment
**Keywords**: Portrait Animation, Personalized Head Representation, Expression Transfer, Diffusion Transformer, SMPL-X

## TL;DR

This paper proposes a high-fidelity personalized head representation (static identity offset + dynamic expression offset) to address the limited expressiveness of parametric models such as SMPL-X. Combined with an identity-adaptive expression transfer module and a DiT-based generator, the method achieves state-of-the-art performance on both self-driven portrait video animation and cross-identity reenactment tasks.

## Background & Motivation

The core challenge in portrait video generation is achieving a balance between fine-grained expression control and identity consistency. Existing intermediate representations suffer from fundamental limitations:

**2D Keypoints**: Sparse signals lacking geometric detail, unstable under large poses.

**3D Parametric Models (SMPL-X/FLAME)**: Low-rank linear approximations with predefined blendshapes that cannot model high-frequency nonlinear dynamics (e.g., wrinkles), leading to severe entanglement between identity and expression.

**Implicit Motion Features**: Weakly controllable with insufficient disentanglement, prone to identity leakage.

Root Cause: The low-dimensional template subspace of parametric models limits expressiveness, making it impossible to capture subject-specific anatomical structures and dynamic wrinkles, and hindering the simultaneous preservation of identity and richness of expression.

## Method

### Overall Architecture

A three-stage pipeline: (1) construct a personalized high-detail head representation based on SMPL-X → (2) apply an identity-adaptive expression transfer module for cross-identity expression transfer → (3) train a DiT video generator conditioned on personalized normal maps.

### Key Designs

1. **Personalized Head Representation**: Two complementary offset fields are superimposed on the coarse SMPL-X base:

    - **Static Global Offset $\Delta_g^s \in \mathbb{R}^{N_s \times 3}$**: Captures expression-independent personalized geometric details (e.g., face shape, hairline, shoulder contour), constrained to non-facial regions.
    - **Dynamic Per-Frame Offset $\Delta_f^s(i) \in \mathbb{R}^{N_s \times 3}$**: Captures expression-related dynamics per frame (e.g., wrinkles, micro-expressions), constrained to facial regions.

   The detailed mesh is formulated as: $\widetilde{V}^s(i) = V^s + \Delta_g^s + \Delta_f^s(i)$

   where $V^s = \mathcal{B}(V) \in \mathbb{R}^{N_s \times 3}$ is a high-resolution mesh obtained via barycentric interpolation upsampling ($N_s \gg N$). Disentanglement is achieved through spatial constraints (facial vs. non-facial regions) and temporal regularization (minimum magnitude penalty + Laplacian smoothing).

   **Optimization objectives** include:
    - Sparse landmark loss: $\mathcal{L}_{\text{ldmk}} = \|\Pi(L_{\text{3D}}(i), \mathbf{c}) - L_{\text{2D}}(i)\|_2^2$
    - Dense normal/depth supervision: $\mathcal{L}_{\text{normal}} = \|\hat{N}_i - N_i\|_1, \mathcal{L}_{\text{depth}} = \|\hat{D}_i - D_i\|_1$
    - Expression coefficient regularization + displacement magnitude penalty + Laplacian smoothing

2. **Identity-Adaptive Expression Transfer Module**: Addresses the incompatibility of cross-identity expression offsets (e.g., a child should not inherit the deep wrinkle patterns of an elderly person):

    - **Driving Signal Encoder**: Encodes expression coefficients $\boldsymbol{\psi} \in \mathbb{R}^{F \times 100}$ and jaw pose $\boldsymbol{\omega} \in \mathbb{R}^{F \times 3}$ into per-frame condition codes $Q = \mathcal{E}(\boldsymbol{\psi}, \boldsymbol{\omega}) \in \mathbb{R}^{F \times D}$.
    - **Vertex-Level MLP**: Conditioned on the target identity's neutral mesh $V_{\text{neutral}} = V^s + \Delta_g^s$ and the driving code $q_i$, it predicts personalized dynamic offsets:

    $\Delta_f^s(i) = \mathcal{G}(V_{\text{neutral}}, q_i) \in \mathbb{R}^{N_s \times 3}$

   The conditioning design ensures that transferred expressions are adapted to the anatomical structure of the target identity.

3. **DiT Video Generator**: Fine-tunes a pretrained video generation model (DiT within the LDM framework):

    - **Control signals**: Reference frame normal map $N^R$ + driving sequence normal maps $N_{1:F}^D$.
    - A 3D convolutional pose encoder extracts spatio-temporal features; a 2D convolutional reference encoder extracts appearance cues.
    - Standard noise prediction loss:

    $\mathcal{L}_{\text{ldm}} = \mathbb{E}_{z_0, \epsilon, t}[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2]$

### Loss & Training

- Data: VFHQ + CelebV-HQ + HDTF, totaling ~4,000 videos (~10 hours), at 512×512 resolution.
- SMPL-X reconstruction and joint geometric optimization are first performed to obtain the personalized head model.
- The expression transfer module is frozen before training the diffusion model.
- 30 epochs, 4×A800 GPUs, batch size 1/GPU, learning rate $10^{-4}$.
- Evaluation datasets: RAVDESS (20 videos) + NeRSemble (80 videos).

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | L1↓ | AED↓ | APD↓ | CSIM↑ |
|--------|-------|-------|--------|-----|------|------|-------|
| LivePortrait | 23.29 | 0.830 | 0.373 | 0.046 | 0.129 | 0.021 | 0.830 |
| Follow-Your-Emoji | 25.69 | 0.841 | 0.236 | 0.029 | 0.147 | 0.015 | 0.803 |
| X-NeMo | 21.56 | 0.781 | 0.324 | 0.048 | 0.137 | 0.018 | 0.830 |
| **Ours** | **26.55** | **0.859** | **0.184** | **0.022** | **0.132** | **0.009** | **0.835** |

The proposed method substantially outperforms all baselines on the self-driven task: PSNR +0.86 (vs. F-Y-E), LPIPS −0.052, APD as low as 0.009.

### Cross-Identity Reenactment

| Method | AED↓ | APD↓ | CSIM↑ |
|--------|------|------|-------|
| LivePortrait | 0.286 | 0.230 | 0.729 |
| X-NeMo | 0.171 | 0.021 | 0.722 |
| **Ours** | **0.211** | **0.013** | **0.729** |

The proposed method achieves the best trade-off between expression accuracy (AED/APD) and identity preservation (CSIM).

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| SMPL-X baseline | Stiff expressions, limited facial dynamics | Expressiveness ceiling of standard parametric models |
| Direct offset transfer | Weakened and unnatural expressions | Incompatibility of offsets across identities |
| **Full model** | Rich expressions + high fidelity | Personalized representation + adaptive transfer |

### Key Findings

- The personalized head representation significantly outperforms standard SMPL-X, improving both expression richness and identity fidelity simultaneously.
- Implicit motion methods (Hunyuan Portrait, X-NeMo) can produce realistic results but suffer from severe identity leakage.
- Explicit control methods (AniPortrait, F-Y-E) are limited by sparse or low-rank signals, resulting in insufficient expressiveness.
- The expression transfer module generates more vivid and natural expressions compared to direct offset transfer.

## Highlights & Insights

- **The ceiling of intermediate representations determines generation quality**: Rather than improving the generator, it is more effective to increase the information density and controllability of control signals.
- **Static + dynamic disentanglement design**: Achieved through spatial constraints (facial/non-facial regions) and temporal regularization (zero-mean dynamic offsets), without requiring additional annotations.
- **Identity-adaptive mechanism**: Conditional prediction avoids a one-size-fits-all approach to expression transfer and is consistent with anatomical differences across individuals.

## Limitations & Future Work

- **Intra-oral regions are not modeled**: Details such as the tongue cannot be precisely generated.
- **Eye movement is insufficiently refined**: Fine-grained eye motion capture is lacking.
- The limited amount of training data (~10 hours) may constrain generalization to extreme poses and expressions.
- The approach has not been extended to full-body animation scenarios.

## Related Work & Insights

- Compared to the implicit keypoint approach of LivePortrait, the explicit 3D representation proposed here is more controllable and does not suffer from identity leakage.
- Compared to the FLAME-driven approach of Follow-Your-Emoji, the personalized offset field captures more high-frequency details.
- Insight: The idea of personalized representation can be generalized to hand and full-body animation scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of personalized offset fields combined with identity-adaptive transfer is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation covering both self-driven and cross-driven settings, with clear ablations and convincing qualitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear exposition with well-structured pipeline diagrams and formalized notation.
- Value: ⭐⭐⭐⭐ Provides a superior intermediate representation scheme for high-fidelity portrait animation with strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[CVPR 2026\] Unified Vector Floorplan Generation via Markup Representation](unified_vector_floorplan_generation_via_markup_representation.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[ICLR 2026\] Directional Textual Inversion for Personalized Text-to-Image Generation](../../ICLR2026/image_generation/directional_textual_inversion_for_personalized_text-to-image_generation.md)

</div>

<!-- RELATED:END -->

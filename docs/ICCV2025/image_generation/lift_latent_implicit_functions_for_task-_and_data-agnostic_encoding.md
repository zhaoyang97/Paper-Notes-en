---
title: >-
  [Paper Note] LIFT: Latent Implicit Functions for Task- and Data-Agnostic Encoding
description: >-
  [ICCV 2025][Image Generation][implicit neural representations] LIFT proposes a meta-learning-based multi-scale implicit neural representation framework that achieves unified encoding across tasks (generation…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "implicit neural representations"
  - "meta-learning"
  - "multi-scale latent variables"
  - "classification"
  - "generative modeling"
date: 2026-05-08
content_hash: d5f81d14a6da20d2
---

# LIFT: Latent Implicit Functions for Task- and Data-Agnostic Encoding

**Conference**: ICCV 2025
**arXiv**: [2503.15420](https://arxiv.org/abs/2503.15420)
**Code**: [GitHub](https://amirhossein-kz.github.io/lift/)
**Area**: Implicit Neural Representations / Generative Models
**Keywords**: implicit neural representations, meta-learning, multi-scale latent variables, classification, generative modeling

## TL;DR
LIFT proposes a meta-learning-based multi-scale implicit neural representation framework that achieves unified encoding across tasks (generation, classification) and data modalities (2D images, 3D voxels) via parallel local implicit functions and a hierarchical latent generator, attaining state-of-the-art performance on both reconstruction and generation tasks while substantially reducing computational cost.

## Background & Motivation

Implicit neural representations (INRs) map coordinates to signal values via neural networks, providing continuous, resolution-agnostic representations for diverse data modalities. Existing INR frameworks suffer from several core issues:

**Limitations of global latent vectors**: Methods such as Functa use a single global latent vector to represent an entire data point, failing to capture fine-grained local details and exhibiting limited performance on downstream tasks such as generation and classification.

**Computational inefficiency**: SpatialFuncta employs spatially distributed latent representations but requires a large MLP with depth 6 and width 256, consuming 0.271 GFLOPs on CIFAR-10 alone.

**Strong modality dependence**: Conventional deep learning models are typically modality-specific, requiring customized architectures and objective functions for different signal types.

The root cause lies in the following tension: how can one simultaneously maintain computational efficiency, capture local details, preserve global context, and generalize across tasks and modalities?

The paper's starting point is to partition the domain into multiple local regions, each processed by an independent small MLP, and then fuse global, intermediate, and local scale features through a hierarchical latent generator. The core idea is: **multi-scale hierarchical latent variable modulation + parallel local implicit functions = efficient unified representation**.

## Method

### Overall Architecture

LIFT is a two-stage framework:
- **Stage 1 (Context Adaptation)**: Generates a multi-scale latent variable-modulated dataset via meta-learning.
- **Stage 2 (Task-Driven Generalization)**: Leverages the latent variables for downstream tasks (DDPM/DDIM for generation, VMamba for classification).

The core architecture consists of parallel local implicit functions (P-MLP) and a hierarchical latent generator (HLG), which jointly produce a unified multi-scale latent representation.

### Key Designs

1. **Parallel Local Implicit Functions (P-MLP)**:

    - Function: Partitions the input domain $[0,1]^D$ into $M^D$ regions, each assigned an independent small MLP.
    - Mechanism: The overall function is expressed as a weighted sum of sub-functions:
    $\mathcal{F}_\theta(\mathbf{x}) = \sum_{m=1}^{M^D} f_m(\mathbf{x}) \cdot \mathbb{1}_m(\mathbf{x})$
      where $\mathbb{1}_m$ is the indicator function and $f_m$ is the local MLP for the corresponding region.
    - Design Motivation: Localized learning enables each sub-network to specialize in signal modeling within a small region, enhancing representational capacity while reducing the complexity of individual MLPs.

2. **Hierarchical Latent Generator (HLG)**:

    - Function: Produces a composite latent variable $Z^\alpha$ that fuses information across global, intermediate, and local scales.
    - Mechanism: Three levels of latent variables are defined:
        - Global latent variable $\mathbf{Z}^\dagger \in \mathbb{R}^{1 \times 1 \times d_g}$
        - Intermediate latent variable $\mathbf{Z}^\star \in \mathbb{R}^{P_i \times P_i \times d_i}$
        - Local latent variable $\mathbf{Z} \in \mathbb{R}^{P \times P \times d_l}$

      The fusion is formulated as:
    $\mathbf{Z}' = \text{Linear}_1(\text{Concat}(\text{Upsample}(\mathbf{Z}^\dagger, P_i, P_i), \mathbf{Z}^\star))$
    $\mathbf{Z}^\alpha = \text{Linear}_2(\text{Concat}(\text{Upsample}(\mathbf{Z}', P, P), \mathbf{Z}))$
    - Design Motivation: Adjacent regions obtain consistent high-level representations through shared intermediate and global latent variables, enabling smooth inter-region transitions and eliminating patch boundary discontinuities.

3. **ReLIFT Variant**:

    - Function: Introduces residual connections and a frequency scaling factor on top of SIREN activations.
    - Mechanism:
    $\mathbf{z}^{(0)} = \sin(\gamma \Omega \mathbf{r})$
    $\mathbf{z}^{(1)} = \sin(\mathbf{W}^{(1)} \sin(\gamma \Omega \mathbf{r})) + \sin(\gamma \Omega \mathbf{r})$
      When $\gamma > 1$, the frequency $\gamma \sum s_t \boldsymbol{\omega}_t$ scales proportionally, expanding the network's capacity to model high-frequency components. Residual connections ensure that base-frequency components are preserved.
    - Design Motivation: Standard SIREN exhibits an implicit bias toward low-frequency signals due to Bessel function properties (the convergence-capacity gap). ReLIFT addresses this via frequency scaling to enhance high-frequency capacity, with residual connections balancing high- and low-frequency representations.

### Loss & Training

A CAVIA-style meta-learning scheme is adopted:
- **Inner loop**: Updates multi-scale latent variables $Z^\dagger, Z^\star, Z$ via SGD ($T_\text{inner}$ steps).
- **Outer loop**: Updates network weights via Adam.

The total loss function is:
$$\mathcal{L}_\text{Total} = \mathcal{L}_\text{Rec} + \lambda \mathcal{L}_\text{Smoothness}$$

where the reconstruction loss is MSE, and the smoothness loss encourages consistency between neighboring latent variables:
$$\mathcal{L}_\text{Smoothness}(Z_m^\alpha) = \frac{1}{K} \sum_{k=1}^{K} \|Z_m^\alpha - Z_k^\alpha\|_2^2$$

## Key Experimental Results

### Main Results

| Dataset | Metric | LIFT | Prev. SOTA (mNIF-L) | Gain |
|--------|------|------|----------|------|
| CelebA-HQ 64² | PSNR↑ | **39.4** | 34.5 | +4.9 dB |
| CelebA-HQ 64² | rFID↓ | **2.6** | 5.8 | -3.2 |
| CelebA-HQ 64² | FID↓ | **10.0** | 13.2 | -3.2 |
| CelebA-HQ 64² | F1↑ | **0.742** | 0.679 | +0.063 |
| ShapeNet 64³ | MSE↓ | **0.00053** | 0.0153 | **28×** improvement |
| ShapeNet 64³ | PSNR↑ | **35.2** | 21.3 | +13.9 dB |
| CIFAR-10 | Top-1 Acc | **95.47%** | 90.30% | +5.17% |
| CelebA-HQ 64² | FLOPs↓ | **54.52M** | 340M | **6.2×** more efficient |

### Ablation Study

| Configuration (Z†×Z⋆×Z) | Test PSNR | rFID | Notes |
|------|---------|------|------|
| 1×1×64, 4×4×32, 8×8×16 | 29.00 | 23.51 | Minimal configuration, limited performance |
| 1×1×128, 4×4×64, 8×8×32 | 34.38 | 7.87 | Doubled channels, significant improvement |
| 1×1×256, 4×4×128, 8×8×64 | 40.91 | 2.22 | Default configuration |
| 1×1×512, 4×4×256, 8×8×128 | **49.86** | **0.40** | Largest configuration, best performance |
| 1×1×256, 2×2×128, 4×4×64 | 30.27 | 21.93 | Reduced local spatial size → substantial degradation |

### Key Findings

- The spatial dimensionality of local latent variables (4×4→8×8) has the greatest impact on reconstruction quality, with PSNR improving from 30.27 to 40.42.
- Doubling channel capacity yields a substantial PSNR gain of +8.95.
- On CIFAR-10, only 5 data augmentation operations are needed to achieve 95.30% accuracy, surpassing ResNet-50 with MixUp+CutMix.
- In 3D voxel experiments, LIFT uses **47×** fewer FLOPs than GEM while achieving an order-of-magnitude improvement in reconstruction quality.

## Highlights & Insights

1. **The necessity of multi-scale design is clearly validated by ablation**: purely local (SpatialFuncta) → adequate generation but poor classification; purely global (Functa) → 68.30% classification accuracy; LIFT multi-scale → 95.47%.
2. **The frequency analysis in ReLIFT is theoretically elegant**: frequency scaling and residual connections are derived from Fourier–Bessel expansions with theoretical guarantees for high-frequency learning.
3. **Extreme efficiency**: 0.915M parameters and 54.52M FLOPs suffice to outperform competing methods with millions of parameters.
4. **High-quality latent space interpolation**: interpolation experiments in 2D and 3D demonstrate manifold smoothness and structural plausibility.

## Limitations & Future Work

- Relies on regular grid partitioning, which may lack flexibility for irregular or sparse signals.
- The two-stage meta-learning training pipeline may increase overall training complexity.
- Validation is limited to relatively low resolutions (64², 256²); scalability to higher resolutions remains to be explored.
- Although generation quality is leading among INR-based methods, a gap remains compared to dedicated generative models such as the StyleGAN family.

## Related Work & Insights

- The multi-scale unification of the Functa family (global modulation) and SpatialFuncta (spatial modulation) represents a natural and effective line of progression.
- The analysis of SIREN frequency properties in ReLIFT can be generalized to other implicit representation-based methods.
- The domain partitioning strategy of parallel local MLPs can be adapted to fields such as NeRF.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-scale hierarchical modulation and the frequency analysis in ReLIFT are innovative, though the overall framework is a composition of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 2D/3D reconstruction, generation, classification, interpolation, and ablation — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are detailed, but the dense notation raises the reading barrier.
- Value: ⭐⭐⭐⭐ Provides an efficient and powerful framework for task-agnostic encoding with practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis](infgen_a_resolution-agnostic_paradigm_for_scalable_image_synthesis.md)
- [\[NeurIPS 2025\] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](../../NeurIPS2025/image_generation/utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)
- [\[ICCV 2025\] The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation](the_silent_assistant_noisequery_as_implicit_guidance_for_goal-driven_image_gener.md)
- [\[NeurIPS 2025\] Energy Loss Functions for Physical Systems](../../NeurIPS2025/image_generation/energy_loss_functions_for_physical_systems.md)
- [\[ICCV 2025\] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)

</div>

<!-- RELATED:END -->

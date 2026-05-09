---
title: >-
  [Paper Note] Detecting Generated Images by Fitting Natural Image Distributions
description: >-
  [Image Generation] This paper proposes ConV, a consistency verification framework that exploits the geometric discrepancy between the natural image manifold and generated images. By constructing two gradient-orthogonal functions, ConV achieves training-free generated image detection. An enhanced variant, F-ConV, further amplifies manifold deviation via Normalizing Flows.
tags:
  - Image Generation
date: 2026-05-08
content_hash: e5137f4bdb8770f1
---

# Detecting Generated Images by Fitting Natural Image Distributions

## Basic Information
- **arXiv**: 2511.01293
- **Conference**: NeurIPS 2025
- **Authors**: Yonggang Zhang, Jun Nie, Xinmei Tian, Mingming Gong, Kun Zhang, Bo Han
- **Institutions**: HKUST, HKBU, USTC, University of Melbourne, CMU, MBZUAI
- **Code**: https://github.com/tmlr-group/ConV

## TL;DR
This paper proposes ConV, a consistency verification framework that exploits the geometric discrepancy between the natural image manifold and generated images. By constructing two gradient-orthogonal functions, ConV achieves training-free generated image detection. An enhanced variant, F-ConV, further amplifies manifold deviation via Normalizing Flows.

## Background & Motivation
Generative models (e.g., Stable Diffusion, Sora) produce increasingly photorealistic images, making robust detection methods urgently needed. The core issues with existing approaches are:

1. **Reliance on binary classifiers**: Require large amounts of both natural and generated images for training, with generalization limited to the generator types seen during training.
2. **Poor cross-model generalization**: Detectors trained on diffusion model outputs may fail to identify images from GANs or Sora.
3. **High cost of continuous data collection**: Each new generative model necessitates updating the training data.

**Key Insight**: Can one build a detector that relies solely on the natural image distribution, without requiring any generated images?

## Core Problem
How to leverage models fitted exclusively on natural images to distinguish natural images from generated ones, without training a detection classifier on generated images?

## Method

### 1. Manifold-Perspective Motivation
- Natural images lie on a data manifold $\mathcal{M}$; generated images $\mathbf{x}_g$ deviate from $\mathcal{M}$.
- The projection of a generated image onto the manifold: $\mathbf{x}_{\mathcal{M}}(\mathbf{x}_g) = \arg\min_{\mathbf{x}' \in \mathcal{M}} d(\mathbf{x}', \mathbf{x}_g)$
- The deviation vector $\mathbf{p} = \mathbf{x}_g - \mathbf{x}_{\mathcal{M}}$ is orthogonal to the tangent space $\mathcal{T}(\mathbf{x}_{\mathcal{M}})$:
$$\mathbf{v}^\top (\mathbf{x}_{\mathcal{M}}(\mathbf{x}_g) - \mathbf{x}_g) = 0, \quad \mathbf{v} \in \mathcal{T}(\mathbf{x}_{\mathcal{M}})$$

### 2. Consistency Verification Objective
Two functions $f_1, f_2$ are designed to satisfy:
- **Consistency on natural images**: $\delta(\mathbf{x}_{\mathcal{M}}) = |f_1(\mathbf{x}_{\mathcal{M}}) - f_2(\mathbf{x}_{\mathcal{M}})| = 0$
- **Inconsistency on generated images**: $\delta(\mathbf{x}_g) > 0$

**Orthogonality Principle** (core design guideline):
$$\nabla f_1(\mathbf{x}_{\mathcal{M}}) \in \mathcal{O}(\mathbf{x}_{\mathcal{M}}), \quad \nabla f_2(\mathbf{x}_{\mathcal{M}}) \in \mathcal{T}(\mathbf{x}_{\mathcal{M}}), \quad f_1(\mathbf{x}_{\mathcal{M}}) = f_2(\mathbf{x}_{\mathcal{M}})$$

The gradients of the two functions reside in the normal and tangent spaces, respectively, ensuring:
$$\delta(\mathbf{x}_g) \geq |\nabla f_1(\mathbf{x}_{\mathcal{M}})^\top \mathbf{p}| > 0 = \delta(\mathbf{x}_{\mathcal{M}})$$

### 3. Training-Free Implementation
- **$f_1$**: The loss function $\ell(\cdot)$ of a pretrained self-supervised model (e.g., DINOv2).
    - A well-trained model is insensitive to on-manifold transformations, so $\frac{\partial \ell}{\partial \mathbf{x}_{\mathcal{M}}} \perp \mathcal{T}(\mathbf{x}_{\mathcal{M}})$.
- **$f_2 = f_1 \circ h$**: A data transformation $h$ (e.g., affine transformation) is composed with $f_1$.
    - $h$ models transformations along the tangent space of the manifold, with its Jacobian $\mathbf{J}_h$ spanning the tangent space.

The resulting detection criterion is:
$$\delta(\mathbf{x}) = |f_1(\mathbf{x}) - f_1(h(\mathbf{x}))| \begin{cases} = 0, & \mathbf{x} \in \mathcal{M} \\ > 0, & \mathbf{x} \notin \mathcal{M} \end{cases}$$

In practice, feature cosine similarity $\mathbf{r}^\top \mathbf{r}_h$ is used in place of loss values to avoid negative sample computation.

### 4. F-ConV: Manifold Extrusion via Normalizing Flows
When advanced generative models produce images with small manifold deviation, a Normalizing Flow (NF) is introduced to actively amplify the discrepancy:
- An invertible transformation $f$ maps the natural image distribution to a Gaussian: $z = f(v), \; z \sim \mathcal{N}(0, I)$
- The training objective comprises two terms:
$$\mathcal{L} = \underbrace{-\mathbb{E}_{v \sim \mathcal{D}_n} \log p(v) + \mathbb{E}_{v \sim \mathcal{D}_g} \log p(v)}_{\text{Shaping Loss}} \underbrace{- \mathbb{E}_{v \sim \mathcal{D}_n} \cos(f(v), f(T(v))) + \mathbb{E}_{v \sim \mathcal{D}_g} \cos(f(v), f(T(v)))}_{\text{Consistency Loss}}$$
- The Shaping Loss pushes generated images away from the natural manifold; the Consistency Loss amplifies the consistency discrepancy.

## Key Experimental Results

### ImageNet Detection (AUROC↑ / AP↑, averaged over 9 generative models)

| Method | Category | AUROC | AP |
|--------|----------|-------|-----|
| CNNspot | Training-based | 67.04 | 66.78 |
| Ojha | Training-based | 85.35 | 84.25 |
| NPR | Training-based | 86.00 | 80.84 |
| FatFormer | Training-based | 93.68 | 93.11 |
| **F-ConV** | **Training-based** | **93.77** | **93.38** |
| AEROBLADA | Training-free | 57.87 | 57.85 |
| **ConV** | **Training-free** | **87.13** | **85.15** |

### Sora / OpenSora Detection

| Method | Sora AUROC | OpenSora AUROC |
|--------|-----------|---------------|
| CNNspot | 52.85 | 50.14 |
| DRCT | 82.53 | 81.79 |
| FatFormer | 89.95 | 88.76 |
| **F-ConV** | **91.74** | **90.16** |
| **ConV** | **87.74** | **82.84** |

- As a training-free method, ConV substantially outperforms most training-based methods on Sora detection.
- F-ConV approaches or achieves state-of-the-art performance across all benchmarks.

## Highlights & Insights
1. **Theoretical elegance**: The orthogonality principle provides clear design guidance, linking generated image detection to manifold geometry.
2. **Training-free**: ConV requires only a pretrained self-supervised model and can be deployed at zero training cost.
3. **Robustness to unseen models**: By not relying on the generated image distribution, ConV remains effective against novel generators such as Sora.
4. **Manifold extrusion in F-ConV**: Actively amplifying deviation via NF is a forward-looking design strategy to counter the progress of generative models.
5. **Practical utility**: Majority voting over multiple random transformations can further improve detection accuracy.

## Limitations & Future Work
1. **Theoretical assumption**: The framework relies on the premise that generated images deviate from the natural manifold — ultra-high-quality generative models may challenge this assumption.
2. **F-ConV requires a small number of generated images**: The Shaping Loss necessitates generated image samples, so F-ConV is not entirely training-free.
3. **Sensitivity to the choice of transformation $h$**: The method depends on data augmentations used in self-supervised training; robustness may vary with augmentation strategies.
4. **Computational overhead**: Multiple transformations combined with forward passes may limit applicability in real-time scenarios.

## Related Work & Insights
- **vs. Ojha et al. 2023 (CLIP-based)**: Both leverage pretrained model features, but Ojha requires training a classifier head, whereas ConV is entirely training-free.
- **vs. DIRE (Diffusion-based)**: DIRE detects generated images via diffusion model reconstruction error but exhibits poor generalization (AUROC ~52%).
- **vs. FatFormer**: The training-based state-of-the-art; ConV approaches its performance under the training-free setting.
- **vs. AEROBLADA**: Both are training-free, but AEROBLADA relies solely on reconstruction error, while ConV is grounded in manifold geometric theory.

**Broader connections**:
- **Byproduct value of self-supervised models**: The invariance of models such as DINOv2 to on-manifold transformations constitutes a powerful detection signal.
- **Manifold geometry × AI safety**: The orthogonality principle provides an elegant geometric interpretation for generated image detection.
- **Connection to adversarial example detection**: Adversarial examples also deviate from the natural manifold; the ConV framework may be applicable to adversarial example detection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Deriving a training-free detection criterion from manifold geometry represents an outstanding theoretical contribution.
- Technical Depth: ⭐⭐⭐⭐⭐ — The orthogonality principle, NF-based manifold extrusion, and training-free implementation form a coherent and rigorous whole.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 9 generative models, Sora, and multiple benchmarks.
- Writing Quality: ⭐⭐⭐⭐☆ — Theoretical derivations are clear, though the dense notation requires careful reading.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[ICCV 2025\] FlowTok: Flowing Seamlessly Across Text and Image Tokens](../../ICCV2025/image_generation/flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](../../ICCV2025/image_generation/timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](../../ICCV2025/image_generation/eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)

<!-- RELATED:END -->

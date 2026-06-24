---
title: >-
  [Paper Note] ∞-Brush: Controllable Large Image Synthesis with Diffusion Models in Infinite Dimensions
description: >-
  [ECCV 2024][Image Generation][Infinite-dimensional diffusion models] Proposes ∞-Brush, the first conditional diffusion model in infinite-dimensional function space. By introducing a cross-attention neural operator, it achieves controllable conditional generation. Trained on only 0.4% of pixels, it can generate large images maintaining global layout consistency at arbitrary resolutions up to 4096×4096.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Infinite-dimensional diffusion models"
  - "Function space"
  - "Large image synthesis"
  - "Cross-attention neural operators"
  - "Controllable generation"
date: 2026-05-08
content_hash: cd713986fc65aa02
---

# ∞-Brush: Controllable Large Image Synthesis with Diffusion Models in Infinite Dimensions

**Conference**: ECCV 2024  
**arXiv**: [2407.14709](https://arxiv.org/abs/2407.14709)  
**Code**: [https://github.com/cvlab-stonybrook/infinity-brush](https://github.com/cvlab-stonybrook/infinity-brush)  
**Area**: Image Generation / Diffusion Models / Function Space Generation  
**Keywords**: [Infinite-dimensional diffusion models, Function space, Large image synthesis, Cross-attention neural operators, Controllable generation]  

## TL;DR

Proposes ∞-Brush, the first conditional diffusion model in infinite-dimensional function space. By introducing a cross-attention neural operator, it achieves controllable conditional generation. Trained on only 0.4% of pixels, it can generate large images maintaining global layout consistency at arbitrary resolutions up to 4096×4096.

## Background & Motivation

High-resolution large image synthesis is in high demand in fields such as digital pathology and remote sensing, but existing methods face two major bottlenecks: (1) pixel/latent-space diffusion models (e.g., SDXL) cannot generalize beyond their training resolution, and their computational complexity scales quadratically with resolution; (2) patch-based methods (e.g., MultiDiffusion, Graikos et al.), while computationally efficient, are constrained to local information and fail to capture long-range spatial dependencies. Although existing infinite-dimensional diffusion models (∞-Diff) solve the resolution limitation, they lack support for conditional control.

## Core Problem

How to formulate conditional diffusion models in function spaces such that they can generate large images at arbitrary resolutions while preserving global structural consistency, and achieve controllable generation via external conditions (such as class labels or embedding vectors)?

## Method

### Overall Architecture

∞-Brush models images as continuous functions in a Hilbert space $\mathcal{H} = L^2(\mathcal{X}, \mu)$ instead of fixed-resolution pixel grids. The key components include:

1. **Conditional Diffusion Process**: Defines the forward noising and conditional reverse denoising processes in function space, where the conditioning information $\mathbf{e}$ can be a finite-dimensional label or embedding vector.
2. **Cross-Attention Neural Operator (CANO)**: Achieves cross-modal feature fusion with linear complexity in the sparse layers.
3. **Hierarchical Denoiser**: Sparse layers (fine-grained details) + grid layers (global information).

### Key Designs

**Cross-Attention Neural Operator (CANO)**:
- Modifies the computation order of vanilla attention: first computes the element-wise product of $\tilde{\mathbf{k}}_i^l \odot \mathbf{v}_i^l$, and then performs the dot product with $\tilde{\mathbf{q}}_t$.
- Reduces computational complexity from $\mathcal{O}(N^2 d)$ to $\mathcal{O}((N + \sum_l N_l) d^2)$, which is linear with respect to the number of query points $N$.
- Simultaneously fuses three types of conditioning embeddings: timestep embedding $\mathbf{t}$, conditional embedding $\mathbf{e}$, and coordinate embedding $\mathbf{c}$.

**Hierarchical Denoiser Architecture**:
- **Sparse Layers** (Blue): Sequentially applies sparse neural operators → CANO → self-attention operators to handle randomly sampled sparse coordinate points and capture fine-grained details.
- **Grid Layers** (Pink): Transforms sparse points into regular grids using k-NN linear interpolation, feeds them into a UNO (UNet-like Neural Operator) to aggregate global information, and applies vanilla cross-attention at the bottleneck layer.

**Efficient Training Strategy**:
- Randomly samples only $256 \times 256 = 65536$ pixels per iteration (accounting for 0.4% of a 4096×4096 image), whereas ∞-Diff requires 25%.
- The coordinate embeddings in CANO serve as positional encodings, enabling the model to reconstruct full images even when trained on extremely sparse pixels.

**Smoothing Operator**: Approximates discrete pixels as smooth functions using a truncated Gaussian kernel $\mathbf{A}: \mathcal{H} \to \mathcal{H}$, ensuring the mathematical validity of representations in the function space.

### Loss & Training

The training objective stems from minimizing the variational upper bound of the infinite-dimensional conditional diffusion, ultimately simplified to:

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathbf{u}_0 \sim \mathbb{Q}_{\text{data}}, t} \lambda_t \|\mathbf{C}^{-1/2}(\mathbf{A}\boldsymbol{\xi} - \boldsymbol{\xi}_\theta(\sqrt{\bar\alpha_t}\mathbf{A}\mathbf{u}_0 + \sqrt{1-\bar\alpha_t}\mathbf{A}\boldsymbol{\xi}, \mathbf{e}, t))\|_{\mathcal{H}}^2$$

where $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{C})$ is Gaussian noise governed by the covariance operator $\mathbf{C}$, and $\mathbf{A}$ is the smoothing operator. The core theoretical foundation is the Feldman-Hájek theorem, which guarantees the equivalence of two Gaussian measures, making the KL divergence computable.

- Optimizer: Adam, lr=$5 \times 10^{-5}$, $\beta_1=0.9$, $\beta_2=0.99$
- EMA rate: 0.995
- Inference: DDIM with 50 steps
- Hardware: 4× NVIDIA A100, batch size of 20 per GPU
- FlashAttention-2 to accelerate CANO

## Key Experimental Results

| Dataset | Resolution | Method | CLIP FID ↓ | Crop FID ↓ |
|--------|--------|------|-----------|-----------|
| CelebA-HQ | 1024² | ∞-Diff (unconditional) | 9.44 | - |
| CelebA-HQ | 1024² | **∞-Brush** | **8.38** | - |
| BRCA | 4096² | Graikos et al. | 2.75 | **11.30** |
| BRCA | 4096² | **∞-Brush** | **2.63** | 14.76 |
| BRCA 5× | 1024² | SDXL | 6.64 | **6.98** |
| BRCA 5× | 1024² | Graikos et al. | 7.43 | 15.51 |
| BRCA 5× | 1024² | **∞-Brush** | **3.74** | 17.87 |
| NAIP | 1024² | SDXL | 10.90 | **11.50** |
| NAIP | 1024² | Graikos et al. | 6.86 | 43.76 |
| NAIP | 1024² | **∞-Brush** | **6.32** | 48.65 |

| Method | Parameters | 1024² Epoch Time | 4096² Epoch Time |
|------|--------|----------------|----------------|
| SDXL | ~2.6B | ~300 hr | OOM / Infeasible |
| Graikos et al. | ~400M | ~140 hr | ~140 hr |
| **∞-Brush** | **~78M** | **~12 hr** | **~12 hr** |

Downstream Applications: Accuracy on the BACH test set improved from 79% (real data only) to 83% (real + synthetic data).

### Ablation Study

1. **Role of CANO** (BRCA 4096²): With CANO → CLIP FID 2.63 / Crop FID 14.76; Without CANO → 3.81 / 16.28. Removing CANO and relying only on vanilla cross-attention in the UNet bottleneck layer leads to a significant degradation in both structural and fine-grained details.
2. **Training Pixel Ratio**: 0.4% vs 1.6% pixels; while more pixels improve generation quality, 0.4% already yields satisfactory results.
3. **Zero-Shot Classification**: Confusion matrices indicate that the generated images are semantically consistent with the text prompts (benign/in-situ/invasive/normal).

## Highlights & Insights

1. **Solid Theoretical Foundation**: Rigorously derives the variational objective for conditional diffusion models in function space, utilizing functional analysis tools like the Feldman-Hájek theorem and Radon-Nikodym derivatives to guarantee the mathematical correctness of the training objective.
2. **Extreme Training Efficiency**: Performs training with only 0.4% pixel sampling. The 78M parameter size is only 1/33 of SDXL's, and the training time is 1/25, remaining constant across different resolutions.
3. **Resolution Agnosticism**: The same model can perform inference at arbitrary resolutions (from 256² to 4096²), enabling true "infinite-dimensional" generation.
4. **Ingenious CANO Design**: Achieves linear complexity by rearranging the attention computation order and utilizes coordinate embeddings as implicit positional encodings.

## Limitations & Future Work

1. **Insufficient Local Details**: Crop FID is systematically worse than patch-based methods and SDXL; while global layout is superior, fine-grained texture lacks fidelity.
2. **Limited Parameters**: The 78M parameters are far fewer than competitors (such as SDXL's 2.6B), leading to insufficient model capacity.
3. **Training from Scratch**: Cannot leverage existing large-scale pre-trained models (e.g., SDXL pre-trained on LAION-5B), suffering from a clear cold-start disadvantage.
4. **Coarse-Grained Conditioning**: Relies only on a single global embedding vector as the conditioning input (extracted by downsampling 4096² to 256² then extracting), which discards significant spatial details, whereas patch-based methods use up to 16 local conditions.
5. **Domain Specificity**: Validated only on pathology and remote sensing datasets, without testing on open-domain natural images.

## Related Work & Insights

| Dimension | ∞-Brush | ∞-Diff | SDXL | Patch-based (Graikos et al.) |
|------|---------|--------|------|--------------------------|
| Function Space | ✅ | ✅ | ❌ | ❌ |
| Conditioning Control | ✅ | ❌ | ✅ | ✅ |
| Inference Beyond Training Resolution | ✅ | ✅ | ❌ | Limited |
| Global Layout Preservation | ✅ Best | - | Medium | Weak (Poor long-range dependencies) |
| Local Details | Weaker | - | Strong | Stronger |
| Pre-training Utilization | ❌ | ❌ | ✅ (LAION-5B) | ✅ |
| Maximum Resolution | 4096² | Theoretically Arbitrary | 1024² | ~2048² |

## Inspirations & Connections

1. **Function Space Diffusion → Video/3D**: Extending function space representations to the temporal dimension (videos as spatio-temporal functions) or 3D scenes (NeRF-like continuous representations) could enable generation at arbitrary lengths or resolutions.
2. **Knowledge Transfer**: The future work mentioned in the paper—transferring knowledge from finite-dimensional pre-trained models to infinite-dimensional ones—could leverage techniques like distillation or adapters.
3. **Multi-Scale Conditioning**: The current reliance on a single global embedding as the condition acts as a performance bottleneck. Designing hierarchical conditions (global + regional + local), similar to ControlNet's multi-resolution condition injection, would be beneficial.
4. **Crossover with the Neural Operator Community**: The core design of CANO can be reversely applied to conditional control in scientific computing scenarios, such as PDE solving.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The first conditional diffusion model in function space, with rigorous theoretical proofs)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Extensive multi-dataset validation and thorough ablations, but lacks experiments on natural images)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear theoretical derivations, structured layout, and comprehensive supplementary materials)
- Value: ⭐⭐⭐⭐ (Introduces a pioneering paradigm, though the lack of local detail limits immediate utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Editable Image Elements for Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)
- [\[ECCV 2024\] DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)

</div>

<!-- RELATED:END -->

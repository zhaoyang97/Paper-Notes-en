---
title: >-
  [Paper Note] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming
description: >-
  [NeurIPS 2025][3D Vision][3D generation] This paper proposes TRIM (Trajectory Reduction and Instance Mask denoising), a post-training framework that accelerates 3D Gaussian diffusion model inference while improving gener…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D generation"
  - "Gaussian diffusion"
  - "inference acceleration"
  - "inference-time scaling"
  - "post-training optimization"
date: 2026-05-08
content_hash: 424ce364192b003f
---

# TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming

**Conference**: NeurIPS 2025
**arXiv**: [2511.16642](https://arxiv.org/abs/2511.16642)  
**Code**: Available (link in paper)  
**Area**: 3D Vision
**Keywords**: 3D generation, Gaussian diffusion, inference acceleration, inference-time scaling, post-training optimization

## TL;DR

This paper proposes TRIM (Trajectory Reduction and Instance Mask denoising), a post-training framework that accelerates 3D Gaussian diffusion model inference while improving generation quality through temporal trajectory pre-selection and spatial background token pruning. TRIM outperforms baselines such as DiffSplat on both T3Bench text-to-3D and GSO image-to-3D benchmarks.

## Background & Motivation

Text/image-to-3D generation has seen remarkable progress recently, with methods such as DiffSplat repurposing 2D image diffusion models to generate 3D Gaussian representations. However, transferring the various post-training acceleration techniques from 2D diffusion (distillation, compression, inference-time scaling, etc.) to 3D faces two core challenges:

**Unstructured nature of 3D Gaussian representations**: A large number of Gaussian primitives are scattered throughout 3D space, making structured compression difficult.

**Heavy three-stage computation pipeline**: The Recon-Gen-Render process (reconstruction → generation → rendering) incurs significantly higher computational cost than simple 2D denoising.

The paper identifies two key efficiency bottlenecks:
- **Trajectory level**: Increasing the number of sampling trajectories (via different random seeds) substantially improves generation quality, but the denoising cost per trajectory in 3D diffusion is extremely high.
- **Token level**: Gaussian primitives in large transparent background regions are denoised unnecessarily, wasting computational resources.

**Core Idea**: Efficiency and quality are improved simultaneously by (1) using a lightweight selector to pre-screen high-quality trajectories along the temporal dimension (retaining only one), and (2) detecting and discarding background tokens via instance masks along the spatial dimension.

## Method

### Overall Architecture

TRIM is a three-stage inference framework: (1) In the first stage, multiple trajectories are denoised in parallel to an intermediate timestep, after which a Latent Selector identifies the best trajectory and the rest are terminated. (2) In the second stage, the selected trajectory continues denoising, with background tokens progressively pruned via instance masks to reduce transformer computation. (3) In the third stage, the detected mask is used to correct the denoised Gaussian primitive parameters by zeroing the opacity of background-region Gaussians, eliminating rendering artifacts.

### Key Designs

1. **Trajectory Reduction**: A lightweight Latent Selector is trained to predict which denoising trajectory will ultimately yield a high-quality 3D asset. An offline knowledge distillation strategy is employed in two steps: first, data synthesis — 100 text prompts are each used to generate 64 trajectories, and CLIP scores on the final rendered images produce {trajectory, score} data pairs; then, selector training — the task is formulated as pairwise comparison, where given the intermediate latents of two trajectories, the selector predicts which one has a higher score. The selector architecture consists of a single-layer CNN followed by a two-layer MLP, which is compact and introduces negligible additional latency. At inference time, a tournament selection strategy is applied: at 50% denoising progress, $N$ trajectories are compared pairwise and one is selected to continue. This reduces the total number of denoising steps from $NT$ to $NT-(N-1)t$, and also reduces VAE decoding and rendering by a factor of $N$.

2. **Instance Mask Denoising**: A training-free background detection and pruning mechanism. Based on the observation that the four corners of the latent feature map typically correspond to transparent background regions, corner features are aggregated to form a [REF] reference token. The cosine similarity between each patch and [REF] is computed, and low-similarity regions are treated as foreground instances. A **progressive mask expansion scheduler** is adopted: the denoising process is divided into four stages, and the mask is gradually expanded from the outermost 2 rows/columns of patches to 4, 6, and 8 rows/columns (full grid), avoiding artifacts caused by hard masking at early steps. Detected background tokens are merged into a single [BG] token via Token Merging, concatenated with the foreground token sequence, and fed into the transformer. After denoising, the [BG] token is restored to its original spatial positions.

3. **Post-denoising Correction**: Since the original model was not trained with masks or [BG] tokens, the denoised [BG] token does not correspond to fully transparent background. Therefore, in the final step, the detected mask is used to zero out the opacity of background-region Gaussian primitives, effectively eliminating rendering artifacts.

### Loss & Training

The selector is trained with a BCE loss: $L_{\text{BCE}} = -(y \cdot \log \sigma(\hat{y}) + (1-y) \cdot \log(1-\sigma(\hat{y})))$, where $y = \mathbf{1}(s_1 > s_2)$. Training uses AdamW (lr=0.001, wd=0.01) with cosine scheduling, batch size 64, for 20 epochs. Instance Mask Denoising is entirely training-free and can be directly inserted into existing 3D diffusion models.

## Key Experimental Results

### Main Results

**Text-to-3D (T3Bench)**:

| Category | Metric | TRIM (Ours) | DiffSplat | LGM | GVGEN |
|----------|--------|-------------|-----------|-----|-------|
| Single Object | CLIP Sim.% | **31.58** | 30.95 | 29.96 | 23.66 |
| Single Object | ImageReward | **0.12** | -0.49 | -0.72 | -2.15 |
| Single w/ Sur. | CLIP Sim.% | **31.48** | 30.20 | 27.79 | 22.65 |
| Single w/ Sur. | CLIP R-Prec.% | **88.25** | 80.75 | 55.00 | 26.75 |
| Multiple Objects | CLIP Sim.% | **30.11** | 29.46 | 27.07 | 21.48 |
| Multiple Objects | ImageReward | **-0.24** | -0.84 | -1.73 | -2.27 |

TRIM is the only method to achieve a positive ImageReward score (0.12), indicating that generation quality reaches the level of human preference.

**Image-to-3D (GSO dataset)**:

| Metric | TRIM (Ours) | DiffSplat | InstantMesh | LGM |
|--------|-------------|-----------|-------------|-----|
| PSNR ↑ | **16.78** | 16.20 | 15.53 | 14.90 |
| SSIM ↑ | **0.82** | 0.79 | 0.77 | 0.71 |
| LPIPS ↓ | **0.17** | 0.19 | 0.22 | 0.25 |

### Ablation Study

| Configuration | FLOPs (T) ↓ | Throughput (step/s) ↑ | Runtime (s) ↓ | Note |
|---------------|-------------|----------------------|--------------|------|
| DiffSplat (baseline) | 195.68 | 13.18 | 8.64 | — |
| + Instance Masking | 165.60 | 18.09 | — | 15.7% FLOPs reduction |
| + Trajectory Reduction | 110.07 | 13.18 | — | Fewer total denoising steps |
| + TRIM (both) | **106.31** | **18.09** | **~5** | Inference time reduced from 8s to 5s |

Ablation over selector architectures shows that a single-layer CNN plus two-layer MLP is optimal (pairwise accuracy 74.18%), with more complex architectures yielding no additional gain.

### Key Findings

- **Trajectory diversity is more effective than increasing step count**: With the total number of denoising steps fixed at 80, TRIM using 10 steps × 8 trajectories + selection substantially outperforms DiffSplat using 80 steps × 1 trajectory in both CLIP and ImageReward, while the latter exhibits semantic drift and over-smoothing at high step counts.
- **Selector performs best at 50% denoising progress**: Applying selection too early renders trajectories indistinguishable due to high noise levels; applying it too late reduces efficiency gains.
- **Trajectory reduction slightly decreases output diversity**: The output distribution shifts from low-quality–high-variance to high-quality–low-variance, consistent with the design intent.
- **Instance Masking primarily reduces runtime**, while Trajectory Reduction primarily improves quality; the two components are complementary.

## Highlights & Insights

- This work is the first to introduce inference-time scaling into 3D diffusion models, establishing that trajectory diversity is more effective than increasing denoising steps.
- The offline distillation with pairwise comparison training paradigm for the Latent Selector is concise and effective, requiring only 100 prompts × 64 trajectories to train a reliable selector.
- The corner-reference attention mechanism for instance mask detection requires no training and is plug-and-play, cleverly exploiting the prior that corners correspond to background in 3D generation.
- The progressive mask expansion schedule avoids artifacts from hard pruning at early timesteps, representing a practical engineering contribution.
- Zeroing background Gaussian opacity via the detected mask in the post-processing stage provides a simple yet effective remedy for artifacts introduced by training-free insertion.

## Limitations & Future Work

- The current 3D diffusion pipeline relies heavily on repurposed 2D backbones, restricting spatial pruning to the denoising transformer blocks and preventing extension to the full generation pipeline.
- The Latent Selector requires training on synthesized data, necessitating data re-synthesis and retraining for new backbone architectures.
- Instance Mask Denoising depends on the assumption that the four corners correspond to background, which may fail when objects are off-center.
- Validation is limited to a single 3D diffusion model (DiffSplat); while model-agnosticism is claimed, this has not been demonstrated on other models.

## Related Work & Insights

- Inference-time scaling in 2D (SANA-1.5, Inference Scaling for Diffusion, etc.) motivates the trajectory diversity strategy.
- Token Merging (ToMe) is a token pruning method for 2D diffusion that requires retraining, whereas TRIM's approach is training-free.
- DiffSplat and Gaussian Atlas are representative works that repurpose 2D diffusion models for 3D Gaussian generation.
- Visualization of patch attention via the [CLS] token in DINO inspires the design of corner-reference attention.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](../../ICCV2025/3d_vision/spatial-temporal_aware_visuomotor_diffusion_policy_learning.md)
- [\[NeurIPS 2025\] MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference](materialrefgs_reflective_gaussian_splatting_with_multi-view_consistent_material_.md)
- [\[NeurIPS 2025\] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [\[ICCV 2025\] 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](../../ICCV2025/3d_vision/7dgs_unified_spatial-temporal-angular_gaussian_splatting.md)
- [\[ICCV 2025\] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](../../ICCV2025/3d_vision/towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)

</div>

<!-- RELATED:END -->

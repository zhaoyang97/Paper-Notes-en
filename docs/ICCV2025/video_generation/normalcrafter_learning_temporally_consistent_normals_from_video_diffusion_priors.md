---
title: >-
  [Paper Note] NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors
description: >-
  [ICCV 2025][Video Generation][surface normal estimation] NormalCrafter proposes a video normal estimation method built upon Stable Video Diffusion (SVD). By incorporating Semantic Feature Regularization (SFR) and a two-s…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "surface normal estimation"
  - "temporal consistency"
  - "video diffusion model"
  - "semantic feature regularization"
  - "SVD"
date: 2026-05-08
content_hash: bd3d3af080a443b1
---

# NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors

**Conference**: ICCV 2025
**arXiv**: [2504.11427](https://arxiv.org/abs/2504.11427)  
**Code**: [https://github.com/NormalCrafter](https://github.com/NormalCrafter)  
**Area**: Surface Normal Estimation / Video Understanding
**Keywords**: surface normal estimation, temporal consistency, video diffusion model, semantic feature regularization, SVD

## TL;DR

NormalCrafter proposes a video normal estimation method built upon Stable Video Diffusion (SVD). By incorporating Semantic Feature Regularization (SFR) and a two-stage training strategy, the method generates normal sequences with fine-grained details and temporal consistency, substantially outperforming existing per-frame methods on video benchmarks.

## Background & Motivation

Surface normal estimation is a cornerstone of applications such as 3D reconstruction, relighting, and video editing. Although single-frame normal estimation has advanced considerably—through both discriminative approaches (e.g., DSINE) and generative ones (e.g., Marigold)—applying these methods to video introduces severe **temporal inconsistency** (flickering):

**Discriminative methods** (DSINE, Omnidata v2) are limited by training data scale and quality, resulting in limited zero-shot generalization.

**Diffusion-based methods** (Marigold-E2E-FT, StableNormal) leverage pretrained diffusion priors to achieve state-of-the-art single-frame performance, but entirely ignore temporal information, causing flickering when applied frame-by-frame to video.

**Naively adding temporal modules** is suboptimal: augmenting image models with temporal layers (e.g., BufferAnytime) relies on optical flow supervision, which cannot guarantee correct normal correspondences as it neglects camera motion and scene dynamics.

The authors' core insight is that video diffusion models (e.g., SVD) already encode rich spatiotemporal priors. However, directly applying SVD to normal estimation produces overly smooth results, since SVD's intermediate features lack sufficient high-level semantic information. By aligning diffusion features with DINO semantic features, the model can be guided to attend to the intrinsic semantics of the scene, yielding fine-grained and accurate normal predictions.

## Method

### Overall Architecture

NormalCrafter is built upon SVD (Stable Video Diffusion). Given an input video $c \in \mathbb{R}^{F \times W \times H \times 3}$, a conditional diffusion process generates a normal sequence $n \in \mathbb{R}^{F \times W \times H \times 3}$.

The core modification replaces SVD's image input with per-frame concatenated noisy normal latents $z_t^n$ and conditional video latents $z^c$. The VAE decoder is fine-tuned on normal data to improve reconstruction quality.

### Key Designs

**1. Semantic Feature Regularization (SFR)**

This is the paper's primary contribution. Two observations motivate the design:

- SVD's intermediate features exhibit **semantic ambiguity**—background regions are excessively blurred, losing fine geometric detail.
- DINO encoder features are highly correlated with geometric structure, accurately distinguishing different regions such as rocks and vegetation.

SFR extracts DINO features $h_\text{dino}$ from input video frames and intermediate features $h_l$ from the second upsampling block of the diffusion U-Net decoder. A learnable MLP $h_\phi$ projects $h_l$ into the DINO feature space, and patch-wise cosine similarity serves as the regularization objective:

$$\mathcal{L}_\text{reg} = -\mathbb{E}\!\left[\frac{1}{N} \sum \text{cossim}\!\left(h_\text{dino}^{[n]},\, h_\phi(h_l^{[n]})\right)\right]$$

A key advantage of SFR is that **it introduces overhead only during training**; the DINO encoder is not required at inference, incurring zero additional inference cost.

**2. Two-Stage Training Strategy**

- **Stage 1 (Latent-Space Training)**: The entire U-Net is trained with loss $\mathcal{L} = \mathcal{L}_\text{DSM} + \mathcal{L}_\text{reg}$. Sequence length is randomly sampled from $[1, 14]$ frames to learn long-range temporal relationships. Training runs for 20,000 steps.
- **Stage 2 (Pixel-Space Fine-Tuning)**: Only spatial layers are fine-tuned; latents are decoded to pixel space and supervised with an angular loss and $\mathcal{L}_\text{reg}$. Sequence length is reduced to $[1, 4]$ frames to reduce memory consumption. Training runs for 10,000 steps.

This design is complementary: Stage 1 learns temporal priors from longer sequences in latent space, while Stage 2 improves spatial accuracy in pixel space. Since Stage 2 fine-tunes only spatial layers, the temporal capabilities acquired in Stage 1 are preserved.

### Loss & Training

- $\mathcal{L}_\text{DSM}$: denoising score matching loss with noise-level weighting $\lambda(\sigma_t) = (1 + \sigma_t^2)\sigma_t^{-2}$
- $\mathcal{L}_\text{reg}$: DINO semantic feature alignment loss (cosine similarity)
- $\mathcal{L}_\text{angular}$: pixel-space angular loss, $\arccos\!\left(\frac{n^* \cdot \hat{n}}{\|n^*\| \cdot \|\hat{n}\|}\right)$

Training data comprises five synthetic datasets (Replica, 3D Ken Burns, Hypersim, MatrixCity, Objaverse), covering indoor/outdoor scenes and object sequences. The AdamW optimizer is used with exponential decay and a 100-step warm-up, on 8 GPUs with batch size 8; U-Net training takes approximately 1.5 days.

## Key Experimental Results

### Main Results

**Single-frame and video normal estimation benchmarks** (angular error mean↓/med↓, threshold ratios at 11.25°/22.5°/30°↑):

| Method | NYUv2 mean↓ | ScanNet mean↓ | Sintel mean↓ | Sintel 22.5°↑ |
|--------|:-----------:|:-------------:|:------------:|:-------------:|
| DSINE | 16.4 | 15.5 | 34.9 | 41.5 |
| Marigold-E2E-FT | 16.2 | 14.1 | 33.5 | 43.0 |
| Lotus-D | 16.2 | 14.3 | 32.3 | 44.9 |
| **NormalCrafter** | **15.4** | **13.3** | **30.7** | **47.5** |

On Sintel, NormalCrafter reduces mean angular error by 1.6° compared to the second-best method (Lotus-D) and improves the 22.5° threshold ratio by 2.6 percentage points.

### Ablation Study

**Contribution of each component** (on ScanNet and Sintel video benchmarks):

| Setting | ScanNet mean↓ | Sintel mean↓ | Note |
|---------|:-------------:|:------------:|------|
| w/o SFR | higher | higher | over-smoothed |
| w/o Stage 1 | medium | medium | lacks long-range temporal modeling |
| w/o Stage 2 | medium | medium | lacks spatial accuracy |
| w/o VAE-FT | medium | medium | poor normal reconstruction |
| **Full Model** | **13.3** | **30.7** | best |

SFR contributes most significantly; its removal causes a marked drop in normal quality due to over-smoothing. Omitting either training stage also degrades performance.

### Key Findings

1. **SFR is critical**: DINO semantic alignment substantially improves normal detail, resolving the over-smoothing caused by semantic ambiguity in SVD features.
2. **Largest gains on video benchmarks**: Improvements are most pronounced on Sintel (large motion, dynamic objects), demonstrating the value of video diffusion priors.
3. **Applicable to single frames**: Setting sequence length to 1 enables single-frame estimation, achieving a mean error of 15.4° on NYUv2—surpassing all prior single-frame methods.
4. **Temporal consistency**: Space-time ($y$-$t$) profile visualizations clearly show smooth outputs from NormalCrafter, in contrast to the pronounced flickering of Marigold-E2E-FT.

## Highlights & Insights

1. **Elegant SFR design**: DINO alignment is applied only during training, incurring no inference overhead—essentially a form of knowledge distillation.
2. **Well-balanced two-stage training**: The progressive transition from long-sequence latent-space training to short-sequence pixel-space fine-tuning is practically effective.
3. **Insightful problem analysis**: PCA visualizations clearly illustrate the semantic gap between SVD and DINO features.
4. **Arbitrary-length video support**: Sliding-window inference removes the constraint imposed by the training sequence length.

## Limitations & Future Work

1. **Synthetic-only training data**: All training data originate from synthetic environments; performance on complex real-world scenes (strong reflections, translucent materials) may be limited.
2. **Inference speed**: Multi-step denoising inference is slow, precluding real-time applications.
3. **VAE reconstruction bottleneck**: High-frequency normal details may be constrained by the precision of VAE encoding and decoding.
4. **DINO choice not ablated**: Only DINO is validated as the semantic encoder; alternative foundation models (CLIP, DINOv2, SAM) may also be effective.
5. **Limited real-world GT evaluation**: Ground-truth normals in the primary benchmarks (NYUv2, ScanNet, Sintel) are of limited accuracy.

## Related Work & Insights

- **Marigold / Marigold-E2E-FT**: State-of-the-art single-frame diffusion-based normal estimation; this work extends it by incorporating video priors.
- **SVD (Stable Video Diffusion)**: Serves as the video diffusion backbone, providing spatiotemporal priors.
- **DSINE**: State-of-the-art discriminative normal estimator exploiting ray directions and neighborhood normal relationships.
- **REPA**: Inspires SFR; proposes aligning diffusion features with external representations during training.
- **DepthCrafter**: A concurrent video depth estimation work following a similar SVD adaptation paradigm.
- **Takeaway**: Video diffusion models encode rich geometric priors that can be effectively activated through semantic feature alignment.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](../../CVPR2026/video_generation/orbital_video_3d_foundation_priors.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[ICCV 2025\] Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)

</div>

<!-- RELATED:END -->

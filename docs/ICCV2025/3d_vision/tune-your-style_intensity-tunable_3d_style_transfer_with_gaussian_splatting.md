---
title: >-
  [Paper Note] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting
description: >-
  [3D Vision] This paper proposes Tune-Your-Style, the first intensity-tunable 3D style transfer paradigm, which explicitly models style intensity via Gaussian neurons and parameterizes a learnable style tuner. Combined wi…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: c0883aec20de60c5
---

# Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2602.00618](https://arxiv.org/abs/2602.00618)
- **Authors**: Yian Zhao, Rushi Ye, Ruochong Zheng, Zesen Cheng, Chaoran Feng, Jiashu Yang, Pengchong Qiao, Chang Liu, Jie Chen
- **Institutions**: Peking University Shenzhen Graduate School, Tsinghua University, Dalian University of Technology
- **Code**: [Project Page](https://zhao-yian.github.io/TuneStyle)
- **Area**: 3D Vision / 3D Style Transfer
- **Keywords**: 3D Gaussian Splatting, style transfer, intensity-tunable, diffusion model prior, cross-view consistency

## TL;DR
This paper proposes Tune-Your-Style, the first intensity-tunable 3D style transfer paradigm, which explicitly models style intensity via Gaussian neurons and parameterizes a learnable style tuner. Combined with a two-stage optimization strategy, the method enables users to freely adjust the degree of style injection without retraining.

## Background & Motivation

### Problem Definition
3D style transfer aims to transfer the artistic effect of a reference style image onto a 3D scene while preserving content consistency and multi-view consistency. The core challenge lies in **balancing content preservation and style injection**.

### Limitations of Prior Work
Existing mainstream methods (StyleGaussian, G-Style, InstantStyleGaussian, etc.) all adopt a **fixed-output paradigm**:
- Each training run produces a stylized result at a single fixed content–style balance point.
- Different users have varying needs for this balance, yet no flexible adjustment is possible.
- If a user finds the style too strong or too weak, the only recourse is to retrain the model or manually tune hyperparameters.

### Core Idea
An **intensity-tunable paradigm** is introduced: after a single training run, users can freely adjust style injection intensity via a style tuner (range 0%–100%) without retraining. The key challenge is how to explicitly model the concept of "style intensity."

## Method

### Overall Architecture
Tune-Your-Style comprises two core components:
1. **Intensity-tunable Style Injection (ISI)**: explicit modeling of style intensity with a learnable style tuner.
2. **Tunable Stylization Guidance (TSG)**: diffusion-model-generated multi-view consistent stylization guidance with a two-stage optimization strategy.

### 1. Intensity-Tunable Style Injection (ISI)

**Gaussian neuron modeling of style intensity**: A learnable neuron is assigned to each Gaussian primitive to predict offsets for all its attributes (position, scale, rotation, opacity, color):

$$\mathcal{G}(S_k, \Theta) = \{\Delta_k \boldsymbol{\mu}_i, \Delta_k \boldsymbol{S}_i, \Delta_k \boldsymbol{R}_i, \Delta_k \sigma_i, \Delta_k \boldsymbol{c}_i\}_{i=1}^{N}$$

The stylized scene is the original scene plus these offsets: $\hat{\Theta}_k = \Theta + \mathcal{G}(S_k, \Theta)$.

**Style tuner parameterization**: A step function $\mathcal{H}(\beta)$ is introduced to quantize the continuous style control signal into $Z=10$ discrete levels, and a bijective mapping to learnable embeddings is established: $\mathcal{V}_\beta = f(\mathcal{H}(\beta))$:

$$\hat{\Theta}_k^{\beta} = \Theta + \mathcal{V}_\beta \odot \mathcal{G}(S_k, \Theta)$$

When $\beta = 0\%$, $\mathcal{V}_\beta$ suppresses the offsets to achieve zero style; when $\beta = 100\%$, the full offset is retained.

**3D Gaussian filter**: An importance score $\psi_i$ is computed for each Gaussian across all training views, and the least important 50% are filtered out to avoid artifacts introduced by redundant Gaussians during stylized rendering:

$$\psi_i = \sum^{C} \sum^{H \times W} \kappa(\Theta_i) \cdot \sigma_i \cdot \prod_{j=1}^{i-1}(1-\sigma_j)$$

### 2. Tunable Stylization Guidance (TSG)

**Diffusion model stylization guidance**: IP-Adapter-SDXL is used as the 2D diffusion model to perform style transfer on rendered views, generating stylized views $\mathcal{I}_v^k$.

**Cross-view style alignment**: To address the lack of 3D consistency across multi-view outputs from the diffusion model:
- A random anchor view is selected, and its features are injected into the self-attention layers of other views.
- Cross-view feature matching is performed for content calibration (back-projecting anchor view content features into 3D and re-projecting them onto target views).
- Mutual self-attention is applied:

$$\mathcal{A}_{v_c}^{t} = \text{softmax}\left(\frac{Q_{v_c}^{t} \cdot ([K_{v_a \rightarrow v_c}^{t}, K_{v_c}^{t}])^{T}}{\sqrt{d}}\right) \cdot [V_{v_a \rightarrow v_c}^{t}, V_{v_f}^{t}]$$

**Two-stage optimization**:

**Stage 1 (full-style guidance, 2000 steps)**:
- Only the Gaussian neurons $\mathcal{G}$ and the full-offset embedding $\mathcal{V}_{full}$ are optimized.
- Loss: $\mathcal{L}_{full}^{s_1} = \mathcal{L}_1(\mathcal{I}_v^{\tilde{t}_1}, \mathcal{I}_v^k) + \mathcal{L}_{lpips}(\mathcal{I}_v^{\tilde{t}_1}, \mathcal{I}_v^k)$

**Stage 2 (tunable guidance, 2000 steps)**:
- The neurons and the full-offset embedding are frozen; only the remaining level embeddings are optimized.
- Intermediate $\beta$ values are randomly sampled, and zero-style and full-style guidance are mixed with a weighted combination:

$$\mathcal{L}_{tunable} = (1-\beta_{\tilde{t}_2}) \cdot \mathcal{L}_{zero} + \beta_{\tilde{t}_2} \cdot \mathcal{L}_{full}^{s_2}$$

### Loss & Training
The overall training objective is a weighted combination of L1 loss and LPIPS perceptual loss, applied respectively under zero-style guidance (compared against the original rendering) and full-style guidance (compared against the stylized views).

## Key Experimental Results

### Main Results: Comparison with 3DGS Style Transfer Methods

| Method | Short-range Consistency↓ (LPIPS/RMSE) | Long-range Consistency↓ (LPIPS/RMSE) | CLIP_S↑ | CLIP_Sdir↑ | User Study↑ |
|--------|---------------------------------------|---------------------------------------|---------|-----------|------------|
| StyleGaussian | 0.067 / 0.070 | 0.126 / 0.108 | 0.2134 | 0.2223 | 2.79±0.16 |
| G-Style | 0.044 / 0.059 | 0.093 / 0.096 | 0.2406 | 0.2391 | 3.10±0.40 |
| InstantStyleGaussian | 0.053 / 0.062 | 0.108 / 0.113 | 0.2204 | 0.2160 | 2.06±0.22 |
| **Ours** | **0.033 / 0.035** | **0.062 / 0.067** | **0.2619** | **0.2881** | **3.97±0.13** |

- Multi-view consistency (both short-range and long-range) is comprehensively superior, far outperforming competing methods.
- CLIP similarity and directional similarity scores are both highest, indicating superior style fidelity.
- User study score of 3.97/5.0, significantly ahead of all baselines.

### Ablation Study

**Two-stage optimization ablation**:
- Removing two-stage optimization (jointly training zero-style and full-style) leads to a severe degradation in stylization quality (over-stylization) and renders the style tuner ineffective.
- Root cause: random mixing of zero-style and full-style guidance produces unstable supervision signals.

**Cross-view style alignment ablation**:

| Configuration | Effect |
|---------------|--------|
| Raw diffusion output | Visually plausible but lacks 3D consistency |
| + Feature injection | Good consistency near anchor views; content distortion in distant views |
| + Content calibration | Content and layout well preserved in distant views; style textures remain consistent |

### Key Findings
1. **Intensity tunability is genuinely effective**: the style tuner smoothly controls style injection intensity from 0% to 100%.
2. **Multi-style composition**: combined with SAM segmentation, different styles can be applied to different scene regions and adjusted independently.
3. **Two-stage training is critical**: learning the full-style offset stably in the first stage before learning intermediate-level embeddings in the second stage is essential.
4. **Training efficiency**: approximately 20 minutes per scene on a single V100 GPU.

## Highlights & Insights

1. **Paradigm innovation**: the first intensity-tunable 3D style transfer paradigm, representing a qualitative leap from "fixed output" to "continuously adjustable."
2. **Explicit modeling of style intensity**: the three-layer design—Gaussian neuron attribute offset prediction, step-function quantization, and learnable embeddings—achieves elegant intensity control.
3. **Cross-view style alignment**: the depth-based back-projection–reprojection content calibration scheme effectively resolves multi-view inconsistency in diffusion-generated results.
4. **Full-attribute offsets**: offsets are predicted not only for color but also for position, scale, rotation, and opacity, enabling simultaneous transfer of geometric and appearance styles.

## Limitations & Future Work

- The quantization level $Z=10$ is manually set and may not generalize optimally to all scenes.
- Generating stylization guidance via the diffusion model increases preprocessing time.
- The Gaussian filter removes 50% of primitives, which may cause information loss in certain scenes.
- Fine-grained control for local hierarchical editing is not yet supported.

## Related Work & Insights

- **vs. StyleGaussian/G-Style**: the proposed method shifts from VGG feature alignment to diffusion prior guidance, avoiding costly encoder–decoder training.
- **vs. SDS/IDU**: the proposed cross-view style alignment is better suited to handling fine style textures than SDS loss and iterative data updates.
- **Insight**: the Gaussian neuron + style tuner design is extensible to other 3D attribute editing tasks (e.g., lighting, material).
- **Multi-style composition** capability has direct application value in the gaming and film industries.

## Rating ⭐⭐⭐⭐
Strong novelty (first intensity-tunable 3D style transfer paradigm), comprehensive experiments, and substantial quantitative and user study advantages over baselines. The cross-view style alignment design is elegant. However, some design choices (quantization level, filter ratio) lack sufficient empirical justification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)

</div>

<!-- RELATED:END -->

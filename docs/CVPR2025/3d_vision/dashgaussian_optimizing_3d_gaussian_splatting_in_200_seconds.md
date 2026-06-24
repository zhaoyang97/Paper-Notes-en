---
title: >-
  [Paper Note] DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds
description: >-
  [CVPR 2025][3D Vision][3D Gaussians] DashGaussian is proposed, offering a joint framework for scheduling rendering resolution and Gaussian primitive count based on frequency analysis. It reformulates 3DGS optimization as a progressive fitting of high-frequency components, achieving an average acceleration of 45.7% without compromising rendering quality.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussians"
  - "Training Acceleration"
  - "Frequency Scheduling"
  - "Resolution Scheduling"
  - "Adaptive Densification"
date: 2026-05-08
content_hash: e655e95c0eda3e7c
---

# DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds

**Conference**: CVPR 2025  
**arXiv**: [2503.18402](https://arxiv.org/abs/2503.18402)  
**Code**: [dashgaussian.github.io](https://dashgaussian.github.io)  
**Area**: 3D Vision / 3DGS Acceleration  
**Keywords**: 3D Gaussians, Training Acceleration, Frequency Scheduling, Resolution Scheduling, Adaptive Densification

## TL;DR

DashGaussian is proposed, offering a joint framework for scheduling rendering resolution and Gaussian primitive count based on frequency analysis. It reformulates 3DGS optimization as a progressive fitting of high-frequency components, achieving an average acceleration of 45.7% without compromising rendering quality.

## Background & Motivation

- Although 3DGS is significantly faster than NeRF (tens of minutes vs. days), further acceleration is still required on resource-constrained devices and for large-scale scene reconstruction.
- Existing acceleration methods fall into two categories: (1) engineering optimizations (e.g., efficient forward/backward implementations); (2) algorithmic optimizations (e.g., pruning redundant Gaussians), where the latter often sacrifices rendering quality.
- Three key observations: a) computational overhead is primarily determined by rendering resolution and Gaussian count; b) at the beginning of optimization, Gaussians are sparse, making high-resolution rendering wasteful; c) in the late stages of optimization, a surge in Gaussian count brings massive computation with limited quality improvement.
- **Key Insight**: Allocating computational resources reasonably is more efficient than simply pruning parameters.

## Method

### Overall Architecture

The optimization of 3DGS is reformulated as a progressive process of fitting higher-frequency components in training views. Based on this, a resolution scheduler and a primitive scheduler are proposed to jointly control the optimization complexity.

### Key Designs

1. **Frequency-Guided Resolution Scheduler**:
    - **Function**: Adaptively increases the rendering resolution step-by-step during optimization.
    - **Mechanism**: Formulating image downsampling as removing high-frequency components, a resolution saliency function is defined as $\mathcal{X}(\mathbf{F}) = \frac{1}{N}\sum_{n=1}^N \sum_{i,j} \|\mathbf{F}^n(i,j)\|_2$. Iteration steps for high/low resolutions are allocated using the frequency energy ratio $f(\mathbf{F}, \mathbf{F}_r) = \frac{\mathcal{X}(\mathbf{F}) - \mathcal{X}(\mathbf{F}_r)}{\mathcal{X}(\mathbf{F})}$.
    - **Design Motivation**: Low-frequency components contain core structural information and require less computation, so they should be fitted first. High-frequency details require more primitives to be effectively fitted and should be addressed after sufficient primitives are present.
    - **Switching Point**: Switches from low resolution to high resolution at the $s_r = S \cdot \mathcal{X}(\mathbf{F}_r) / \mathcal{X}(\mathbf{F})$ iteration.

2. **Resolution-Guided Primitive Scheduler**:
    - **Function**: Controls the growth of Gaussian primitives in synchronization with the resolution.
    - **Mechanism**: $P_i = P_{\text{init}} + (P_{\text{fin}} - P_{\text{init}}) / (r^{(i)})^{2-i/S}$, where the power factor decays linearly from 2 to 1, producing a concave growth curve (suppressed in the early stage, encouraged in the middle stage).
    - **Design Motivation**: A specific resolution should match an appropriate number of primitives. Too many primitives in low-resolution stages are wasteful and may lead to over-densification. The concave curve ensures primitives grow rapidly only when truly needed.

3. **Momentum-Based Primitive Budget Estimation**:
    - **Function**: Adaptively determines the final primitive count $P_{\text{fin}}$ without relying on dataset priors.
    - **Mechanism**: Treating $P_{\text{fin}}$ as momentum and the natural densification amount at each step $P_{\text{add}}$ as force: $P_{\text{fin}}^{(i)} = \max(P_{\text{fin}}^{(i-1)}, \gamma P_{\text{fin}}^{(i-1)} + \eta P_{\text{add}}^{(i)})$.
    - **Design Motivation**: Bypasses artificially set upper bounds, dynamically adjusting the primitive budget based on actual scene requirements.

### Loss & Training

- Uses anti-aliasing downsampling (frequency domain center cropping + DFT/IDFT) to avoid 2D aliasing.
- Discretizes the resolution factor using floor functions to encourage primitives to grow faster than under continuous scheduling.
- Hyperparameters: $a=4$ (maximum downsampling corresponding to $1/4$ of frequency energy), $\gamma=0.98$, $\eta=1$, $P_{\text{fin}}^{(0)} = 5 \cdot P_{\text{init}}$.
- Maintains the original hyperparameters unchanged when incorporating various 3DGS backbones.

## Key Experimental Results

### Main Results (Comparison with Fast Optimization Methods)

| Method | Mip-NeRF360 PSNR | Time (min) | Deep Blending PSNR | Time (min) | T&T PSNR | Time (min) |
|------|-----------------|-----------|-------------------|-----------|----------|-----------|
| 3DGS | 27.72 | 18.31 | 29.50 | 17.27 | 23.62 | 10.59 |
| Reduced-3DGS | 27.28 | 15.52 | 29.78 | 13.74 | 23.59 | 8.31 |
| Taming-3DGS | 27.61 | 5.51 | 29.69 | 4.52 | 23.62 | 4.02 |
| **DashGaussian** | **27.92** | **3.23** | **30.02** | **2.20** | **23.97** | **2.62** |

### Speedup Effects on Various Backbones

| Backbone | Original Time | +DashGaussian | Speedup Ratio | PSNR Change |
|------|---------|---------------|--------|---------|
| 3DGS | 18.31 min | 10.16 min | 44.5% | +0.09 |
| Mip-Splatting | 25.83 min | 12.60 min | 51.2% | +0.08 |
| Revising-3DGS | 5.73 min | 3.46 min | 39.6% | +0.20 |
| Taming-3DGS | 5.51 min | 3.23 min | 41.4% | +0.31 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Resolution scheduling only | Significant speedup but slight quality drop | Lack of primitive coordination leads to underfitting |
| Primitive scheduling only | Moderate speedup | High-resolution rendering remains the bottleneck |
| Joint scheduling (Full) | **Maximum speedup with maintained quality** | Optimal coordination between resolution and primitives |
| Fixed $P_{\text{fin}}$ | Over/under-densification in some scenes | Momentum estimation yields better adaptivity |

### Key Findings

- Accelerates optimization by 45.7% on average with widespread improvements in PSNR (speed is not gained at the cost of quality).
- Integrates seamlessly with arbitrary 3DGS backbones as a plug-and-play accelerator.
- The concave primitive growth curve (fewer in early stages, more in late stages) outperforms existing convex or front-heavy strategies.
- Compresses the optimization time of Taming-3DGS on the Mip-NeRF 360 dataset to approximately 3 minutes (~200 seconds level).

## Highlights & Insights

- Unifies the theoretical foundation of resolution scheduling from the perspective of frequency analysis, reformulating 3DGS optimization as a progressive frequency fitting process.
- The philosophy of "allocating computational resources reasonably rather than pruning parameters" is more elegant than aggressive pruning and avoids quality loss.
- Momentum-based primitive budgeting is a truly adaptive solution, escaping the limitations of manually defined upper bounds.
- Experiments demonstrate that reasonable allocation of computational resources can even enhance quality (the low-resolution stage acts as implicit regularization).

## Limitations & Future Work

- Discretizing the resolution factor (using floor) is an engineering approximation, leaving the theoretical optimality of continuous scheduling not fully realized.
- The frequency energy ratio $f$ assumes perfect anti-aliased downsampling, whereas residual aliasing may exist in practice.
- The selection of hyperparameter $a=4$ is not extensively ablated.
- Not fully validated on ultra-large-scale scenes (e.g., city-scale).

## Related Work & Insights

- 3DGS $\to$ Base representation; Mip-Splatting $\to$ Multi-scale rendering and anti-aliasing.
- Taming-3DGS $\to$ Engineering acceleration baseline; Mini-Splatting $\to$ Algorithmic pruning baseline.
- Nyquist Shannon sampling theorem + DFT $\to$ Theoretical foundation for resolution scheduling.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Frequency-guided resolution scheduling offers a novel perspective, and the joint scheduling scheme is designed cleverly.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High thoroughness with three datasets, four backbones, and detailed ablations/comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivations; the frequency analysis perspective is highly educational.
- **Value**: ⭐⭐⭐⭐⭐ Extremely high practical value as a plug-and-play general accelerator.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](../../CVPR2026/3d_vision/fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](../../AAAI2026/3d_vision/opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)
- [\[CVPR 2025\] 3D-HGS: 3D Half-Gaussian Splatting](3d-hgs_3d_half-gaussian_splatting.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] MV-DUSt3R(+): Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds](mv-dust3r_single-stage_scene_reconstruction_from_sparse_views_in_2_seconds.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] An adaptive reconstruction-aware pruning scheduler (RPS) and 3D DoG primitives are proposed to achieve 90% Gaussian pruning while preserving rendering quality.
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "model pruning"
  - "Difference-of-Gaussians"
  - "compact representation"
  - "novel view synthesis"
date: 2026-05-08
content_hash: c09f094850ef8318
---

# Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives

**Conference**: CVPR 2026
**arXiv**: [2602.24136](https://arxiv.org/abs/2602.24136)
**Authors**: Haoran Wang, Guoxi Huang, Fan Zhang, David Bull, Nantheera Anantrasirichai (University of Bristol)
**Code**: Coming soon
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, model pruning, Difference-of-Gaussians, compact representation, novel view synthesis

## TL;DR

An adaptive reconstruction-aware pruning scheduler (RPS) and 3D DoG primitives are proposed to achieve 90% Gaussian pruning while preserving rendering quality.

## Background & Motivation

3D Gaussian Splatting (3DGS) enables real-time, high-fidelity rendering but typically requires a large number of Gaussian primitives, resulting in redundant representations and high resource consumption. Existing pruning methods apply pruning at fixed iterations with uniform densification intervals, ignoring the dynamic nature of the reconstruction process. This leads to optimization instability: premature pruning removes necessary primitives, while late pruning yields marginal benefit. Furthermore, smooth Gaussian kernels struggle to capture fine details under compact configurations.

## Core Problem

1. **When to prune**: Fixed pruning schedules do not account for varying reconstruction difficulty across scenes.
2. **How much to prune**: Fixed pruning ratios overlook the fact that redundancy changes throughout training.
3. **How to assess importance**: Scoring based solely on the spatial domain neglects frequency-domain information critical to edges and textures.
4. **How to preserve details**: The smoothness of standard Gaussian primitives limits detail representation after aggressive pruning.

## Method

### 3.1 Reconstruction-aware Pruning Scheduler (RPS)

RPS consists of three key components:

**Refinement Interval Regulator (RIR)**: Uses the L1 reconstruction loss as a quality indicator to adaptively determine pruning timing. The pruning condition is defined as:

$$L_1^{(t)} \leq \beta \cdot L_1^{(t-1)}, \quad \beta = 0.95$$

- Condition satisfied → reconstruction quality is improving; proceed to the next pruning round.
- Condition not satisfied → continue refinement until the condition is met or the maximum interval $Iter_{\max} = 2000$ is reached.
- Checked every 500 iterations.

**Dynamic Pruning Ratio Adjustment (DPRA)**: Progressively reduces the pruning volume per round as redundancy decreases. The target count and pruning ratio at round $t$ are:

$$N^{(t)} = N_{\text{current}} - (N_0 - N_{\text{target}}) \cdot \frac{1}{2^t}$$

$$R^{(t)} = \frac{N_{\text{current}} - N^{(t)}}{N_{\text{current}}}$$

Aggressive pruning is applied early and conservative pruning later, balancing efficiency and quality.

**Spatio-spectral Pruning Score (SPS)**: Evaluates the importance of each Gaussian primitive by combining spatial-domain and frequency-domain gradient information:

$$\tilde{U}_i^* = \lambda_s \cdot \frac{(\nabla_{g_i} I_{\mathcal{G}})^2}{\|\tilde{U}\|_2} + \lambda_f \cdot \frac{(\nabla_{g_i} \text{FFT}(I_{\mathcal{G}}))^2}{\|\tilde{U}^f\|_2}$$

The frequency-domain score applies radial frequency weighting $w(\omega) = (\|\omega\| / \omega_{\max})^{\gamma_f}$ to emphasize high-frequency information, ensuring that primitives critical to sharp structures are not erroneously pruned.

### 3.2 3D Difference-of-Gaussians (3D-DoG) Primitives

To address detail loss after pruning, a novel primitive capable of modeling both positive and negative density is proposed:

$$\text{DoG}(x) = G(x) - G_p(x)$$

- The pseudo-Gaussian $G_p$ shares the center coordinates and rotation with the primary Gaussian but has independent opacity factor $f^\alpha$ and scaling factors $[f_x^s, f_y^s, f_z^s]$.
- Only 4 additional learnable parameters are introduced.
- A positive density peak combined with a negative density ring yields intrinsic contrast enhancement, making the primitive more sensitive to edges and textures.
- All factors are constrained to be $< 1.0$, ensuring the positive peak remains dominant in radiance.

**3D-DoG Density Control**: DoG primitives are activated upon completion of pruning. The pseudo-Gaussian opacity $\alpha_p$ is iteratively monitored; when $\alpha_p$ falls below a threshold, the corresponding DoG degenerates into a standard Gaussian, adaptively balancing the ratio of the two primitive types.

### Overall Architecture

1. First 15k iterations: standard 3DGS training with densification.
2. After 15k: RPS is activated for progressive pruning–refinement cycles, up to 25k iterations.
3. After pruning completes: 3D-DoG primitives are introduced and the mixture ratio is optimized.

## Key Experimental Results

| Method | Size (MB) ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Time ↓ |
|--------|------------|--------|--------|---------|----------------|
| 3DGS | 645.2 | 27.47 | 0.826 | 0.201 | 17m1s |
| MaskGaussian | 280.7 | 27.43 | 0.811 | 0.227 | 24m11s |
| PuP-3DGS | 90.6 | 26.67 | 0.786 | 0.271 | - |
| Speedy-Splat | 73.9 | 26.84 | 0.782 | 0.296 | 16m30s |
| **Ours** | **65.3** | **27.16** | **0.789** | **0.285** | **13m48s** |

*Mip-NeRF 360 dataset, 90% pruning ratio*

| Ablation Variant | RIR | DPRA | SPS | DoG | PSNR | SSIM | FPS |
|-----------------|-----|------|-----|-----|------|------|-----|
| 3DGS (100%) | ✗ | ✗ | ✗ | ✗ | 27.47 | 0.826 | 143.5 |
| V1 | ✓ | ✗ | ✗ | ✗ | 26.03 | 0.742 | 362.4 |
| V2 | ✓ | ✓ | ✗ | ✗ | 26.17 | 0.751 | 363.2 |
| V3 | ✓ | ✓ | ✓ | ✗ | 26.99 | 0.771 | 361.9 |
| **Full** | ✓ | ✓ | ✓ | ✓ | **27.16** | **0.789** | 289.0 |

## Highlights & Insights

- Adaptive pruning timing combined with dynamic ratios eliminates manual hyperparameter tuning and accommodates varying scene complexity.
- SPS frequency-domain scoring is the first to incorporate FFT gradients into importance evaluation for 3DGS.
- The negative density ring of 3D-DoG cleverly exploits the edge-enhancement property of DoG to recover fine details lost after pruning.
- Near-original quality is maintained after 90% pruning, with a 1.23× training speedup and 2× inference FPS improvement.

## Limitations & Future Work

- Mild performance degradation persists in complex scenes such as Bicycle at the 90% pruning target.
- A brief PSNR fluctuation occurs upon DoG activation (at iteration 25k), requiring subsequent iterations to recover.
- Validation on dynamic or large-scale scenes has not been conducted.
- The additional FFT computation and DoG primitives incur a moderate FPS overhead (289 vs. 362 FPS).

## Related Work & Insights

- vs. **Mini-Splatting**: Mini-Splatting aggregates blending weights as pruning scores; the proposed SPS additionally incorporates frequency-domain information.
- vs. **PuP-3DGS**: PuP evaluates spatial sensitivity but relies on a fixed pruning schedule; this work adopts adaptive scheduling.
- vs. **MaskGaussian**: MaskGaussian learns adaptive masks but produces models 4× larger than the proposed method.
- vs. **LightGaussian**: LightGaussian scores based on 2D projected area × opacity; the proposed approach is more robust in the frequency domain.

- DoG is the classical inspiration for edge detection in image processing (as a LoG approximation); introducing it into 3DGS primitive design represents a cross-domain innovation.
- Frequency-domain pruning scores are generalizable to other point-based representations (e.g., point cloud compression).
- The adaptive pruning strategy can be combined with model quantization and encoding methods for further storage compression.

## Rating

- Novelty: ⭐⭐⭐⭐ — DoG primitives and frequency-domain scoring constitute interesting designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three standard datasets with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — Practically significant contribution to 3DGS compactification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CGHair: Compact Gaussian Hair Reconstruction with Card Clustering](cghair_compact_gaussian_hair_reconstruction_with_card_clustering.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](../../AAAI2026/3d_vision/opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)
- [\[ICLR 2026\] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](../../ICLR2026/3d_vision/megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)

</div>

<!-- RELATED:END -->

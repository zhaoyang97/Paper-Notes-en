---
title: >-
  [Paper Note] Faster and Better 3D Splatting via Group Training
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes a **Group Training** strategy that accelerates 3DGS training by periodically partitioning Gaussian primitives into an "under-training group" and a "cached group," combined with **Opacity-based Priority Sampling** (OPS). Across four standard benchmarks, the method achieves approximately **30% training speedup** while simultaneously **improving rendering quality** and **reducing model size**, and can be applied as a plug-and-play module to 3DGS and Mip-Splatting frameworks.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Training Acceleration
  - Group Training
  - Opacity-based Sampling
  - Novel View Synthesis
date: 2026-05-08
content_hash: 51193f0f97a34503
---

# Faster and Better 3D Splatting via Group Training

**Conference**: ICCV 2025
**arXiv**: [2412.07608](https://arxiv.org/abs/2412.07608)
**Code**: [Project Page](https://chengbo-wang.github.io/3DGS-with-Group-Training/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Training Acceleration, Group Training, Opacity-based Sampling, Novel View Synthesis

## TL;DR

This paper proposes a **Group Training** strategy that accelerates 3DGS training by periodically partitioning Gaussian primitives into an "under-training group" and a "cached group," combined with **Opacity-based Priority Sampling** (OPS). Across four standard benchmarks, the method achieves approximately **30% training speedup** while simultaneously **improving rendering quality** and **reducing model size**, and can be applied as a plug-and-play module to 3DGS and Mip-Splatting frameworks.

## Background & Motivation

### Core Problem

3D Gaussian Splatting (3DGS) demonstrates remarkable real-time, high-quality rendering for novel view synthesis, but training efficiency is bottlenecked by the **exponentially growing number of Gaussian primitives** (typically reaching millions), which substantially increases the training burden.

### Limitations of Prior Work

The most straightforward acceleration strategy is to **periodically prune low-opacity Gaussians** to reduce primitive count:
- A conservative threshold $\epsilon_\alpha$ → negligible speedup
- An aggressive threshold → severe degradation in rendering quality
- Extremely high parameter sensitivity (see Figure 2 left), making a good trade-off difficult to find

### Key Intuition

**Cache rather than prune** — temporarily exclude some Gaussians from training while retaining them, and periodically resample to rotate cached Gaussians back into training. This reduces per-iteration computation without discarding "important" primitives.

## Method

### Overall Architecture

Group Training periodically partitions all Gaussian primitives into two groups during training:

$$G_{\text{Under-training}} = \{g_i | g_i \in G, i \in I, I \subseteq \{1,2,...,|G|\}\}$$
$$G_{\text{Cached}} = G \setminus G_{\text{Under-training}}$$

- **Under-training group**: participates in rendering and gradient optimization
- **Cached group**: excluded from all computation temporarily
- Groups are merged and resampled every 500 iterations
- The under-training ratio (UTR) defaults to 0.6

### Training Schedule

- Densification phase (0–15K iter): Group Training operates normally; a **global densification** (merging all groups) is performed at 14.5K
- Optimization phase (15K–30K iter): Group Training continues; a **global optimization** step is performed at 29K
- Group Training begins at iteration 500 to protect the importance of initial Gaussians

### Key Designs: Sampling Strategies

#### Random Sampling (RS)

The simplest approach — uniform random sampling. Experiments show RS already accelerates training, but may lead to excessive redundant Gaussians.

#### Opacity-based Priority Sampling (OPS)

OPS is the core contribution of this paper. The sampling probability is defined as:

$$p_i = \frac{\alpha_i}{\sum_{i=1}^{N} \alpha_i}$$

That is, Gaussians with higher opacity are preferentially selected into the under-training group. This is supported by two mathematical propositions:

**Proposition 1: Higher Opacity Promotes Effective Densification**

The gradient of the loss with respect to Gaussian position is:

$$\frac{\partial L}{\partial x_m} = o_m \sum_{\text{pixel}} \frac{\partial L}{\partial \hat{C}} \frac{\partial \hat{C}}{\partial \alpha_m} \frac{\partial G_m^{2D}}{\partial \Delta x} \frac{\partial \Delta x}{\partial x_m}$$

- The gradient is proportional to opacity $o_m$
- $\frac{\partial \hat{C}}{\partial \alpha_m}$ increases as the expected $o_m$ increases
- Therefore, high-opacity Gaussians are more likely to satisfy the densification threshold $\tau_{\text{grad}}$ and are the primary contributors to densification

**Proposition 2: Higher Opacity Accelerates Rendering**

The termination condition of α-blending depends on transmittance $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ reaching a saturation threshold:

$$\mathbb{E}[T_N] = (1 - \mathbb{E}[o_i] \cdot \mathbb{E}[G_i^{2D}])^N$$

- Higher opacity → faster α saturation → fewer blending steps $N$ → faster rendering
- Experimentally verified: as $\mu_o$ increases, rendering time decreases by approximately 40%

### Design Summary

Caching low-opacity Gaussians (retaining high-opacity ones in the under-training group) yields three simultaneous benefits: promoting effective densification, reducing redundant primitives, and accelerating rendering.

## Key Experimental Results

### Main Results: 3DGS Reconstruction Efficiency and Quality (Tables 1 & 2)

| Method | Config | Mip-NeRF360 PSNR↑ | Time (min)↓ | Model Size (MB)↓ |
|--------|--------|-------------------|------------|-----------------|
| 3DGS* | baseline | 27.445 | 26.7 | 792 |
| + GT w/ RS | 0~30K | 27.537 | 22.6 | 902 |
| + GT w/ OPS | 0~15K | 27.582 | **22.5** | **678** |
| + GT w/ OPS | 0~30K | 27.564 | **19.6** | 679 |

| Method | T&T PSNR↑ | Time↓ | Deep Blending PSNR↑ | Time↓ |
|--------|-----------|------|---------------------|------|
| 3DGS* | 23.697 | 15.0 | 29.586 | 23.9 |
| + GT w/ OPS (0~30K) | **23.853** | **11.0** | **29.746** | **16.9** |

**Core result**: OPS achieves ~27–30% training speedup across all scenes, with improved quality and 10–40% reduction in model size.

### Validation on Mip-Splatting (Table 3)

| Method | T&T PSNR↑ | Time↓ | Deep Blending PSNR↑ | Time↓ |
|--------|-----------|------|---------------------|------|
| Mip-Splatting* | 23.749 | 23.0 | 29.358 | 35.1 |
| + GT w/ OPS (0~30K) | **24.156** | **18.2** | **29.788** | **24.0** |

**Speedup**: 21% faster on T&T, 32% faster on Deep Blending, with consistent quality improvements.

### Ablation Study (Table 5, Tanks & Temples)

| Periodic Resampling | Global Densification | Global Optimization | PSNR↑ | Model Size (MB)↓ | Time (min)↓ |
|--------------------|---------------------|--------------------|----|----------------|-----------|
| - | - | - | 23.697 | 434 | 15.0 |
| ✓ | - | - | 23.866 | 292 | 11.8 |
| ✓ | ✓ | - | 23.769 | **231** | **11.0** |
| ✓ | - | ✓ | 23.835 | 485 | 11.8 |
| ✓ | ✓ | ✓ | **23.853** | 384 | **11.0** |

**Key Findings**:
- **Periodic resampling** contributes the largest speedup (15.0 → 11.8 min)
- **Global densification** further reduces model size (292 → 231 MB)
- **Global optimization** improves quality when combined with global densification (23.769 → 23.853)

### Counter-Intuitive Findings

1. **RS increases model size yet still accelerates training**: Group Training w/ RS produces more primitives, yet training is still faster — indicating that training dynamics rather than model size determine efficiency.
2. **OPS yields both higher quality and smaller models**: The simultaneous improvement in quality and reduction in model size suggests that performance gains stem from the optimization process rather than model capacity.

## Highlights & Insights

1. **Simplicity with rigor**: The core idea is extremely straightforward (grouping + rotation), yet the paper provides a thorough mathematical analysis (complete proofs of two propositions) explaining the underlying reasons for its effectiveness.
2. **Plug-and-play**: The method integrates seamlessly into 3DGS and Mip-Splatting as a training strategy without any architectural modifications.
3. **Dual role of opacity**: This work is the first to reveal that opacity simultaneously governs the effectiveness of densification and rendering speed — an underappreciated yet critical property in 3DGS.
4. **Hyperparameter robustness**: In contrast to the extreme sensitivity of pruning thresholds, the Group Training cache ratio maintains stable performance over a wide range of values (Figure 2 right).

## Limitations & Future Work

1. **Only validated at 30K iterations**: Performance under longer training schedules or very large-scale scenes remains unexplored.
2. **GPU memory overhead**: Merging all groups may increase peak memory consumption, partially mitigated by fixing spherical harmonic coefficients.
3. **Independence assumption in OPS**: The proofs assume Gaussian attributes are mutually independent, which may not hold in practice.
4. **Only two frameworks tested**: Generalizability to newer variants such as 2DGS and InstantSplat has not been verified.

## Related Work & Insights

- **Fundamental distinction from pruning**: Pruning permanently removes low-opacity Gaussians, whereas Group Training only temporarily caches them with periodic rotation — preserving the opportunity for important primitives to contribute.
- **Relationship to model compression**: OPS naturally produces more compact models (fewer redundant Gaussians) and is complementary to post-training compression methods.
- **Implications for 3DGS training**: The opacity distribution serves as a key entry point for understanding and optimizing 3DGS behavior; future work could explore dynamically adjusting UTR or employing adaptive sampling strategies.

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐ — A simple idea backed by rigorous theoretical analysis, yielding a surprising quality–efficiency win
**Practicality**: ⭐⭐⭐⭐⭐ — Plug-and-play, no hyperparameter sensitivity, engineering-friendly
**Experimental Thoroughness**: ⭐⭐⭐⭐ — 4 datasets × 2 frameworks, with complete ablations and sampling strategy comparisons
**Writing Quality**: ⭐⭐⭐⭐ — Proposition proofs are clearly presented; experimental organization is systematic

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgs-lm_faster_gaussian-splatting_optimization_with_levenberg-marquardt.md)
- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](../../CVPR2026/3d_vision/fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[ICLR 2026\] Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention](../../ICLR2026/3d_vision/efficient-lvsm_faster_cheaper_and_better_large_view_synthesis_model_via_decouple.md)
- [\[ICCV 2025\] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)

<!-- RELATED:END -->

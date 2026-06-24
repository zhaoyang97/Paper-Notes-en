---
title: >-
  [Paper Note] NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis
description: >-
  [ECCV 2024][3D Vision][NeRF] NGP-RT is proposed to replace the per-point MLP by aggregating multi-level explicit hash features using a lightweight attention mechanism, and to introduce an occupancy distance grid to reduce memory access during ray marching, achieving real-time NeRF rendering at 1080p 108fps on the Mip-NeRF 360 dataset.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "NeRF"
  - "Real-Time Rendering"
  - "Hash Features"
  - "Attention Mechanism"
  - "Occupancy Distance Grid"
date: 2026-05-08
content_hash: 1921e19239ced7ca
---

# NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2407.10482](https://arxiv.org/abs/2407.10482)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: NeRF, Real-Time Rendering, Hash Features, Attention Mechanism, Occupancy Distance Grid

## TL;DR

NGP-RT is proposed to replace the per-point MLP by aggregating multi-level explicit hash features using a lightweight attention mechanism, and to introduce an occupancy distance grid to reduce memory access during ray marching, achieving real-time NeRF rendering at 1080p 108fps on the Mip-NeRF 360 dataset.

## Background & Motivation

**Background**: Instant-NGP achieves fast training and high-quality rendering by storing implicit features in multi-level hash grids and decoding them with a shallow MLP. Deferred NeRF architectures (e.g., SNeRG, MERF) achieve real-time rendering by storing explicit color/density and reducing the MLP from per-point to per-ray execution.

**Limitations of Prior Work**: (a) The rendering bottleneck of Instant-NGP lies in the **per-point MLP execution**—each sampled point requires the MLP to decode implicit features into color and density, which restricts the rendering speed (~10fps@1080p); (b) SNeRG's low-resolution sparse grids fail to represent the fine details of Instant-NGP's high-resolution features; (c) MERF's simple summation aggregation is not flexible enough to resolve hash collisions in high-resolution NGP features.

**Key Challenge**: Multi-level hash features possess strong expressive power but rely heavily on MLP aggregation. Removing the MLP results in the **inability to disambiguate hash collisions**—where multiple distinct 3D positions map to the same hash table entry, requiring the mask function implicitly learned by the MLP to differentiate them.

**Goal**: To eliminate the computational burden of per-point MLPs while preserving the strong representation capabilities of multi-level hash features, thereby achieving real-time NeRF rendering at >100fps.

**Key Insight**: To make the implicit masked function of the MLP explicit as **learnable attention parameters**, replacing the MLP with an extremely lightweight channel-wise weighted sum, while introducing an occupancy distance grid to reduce ray marching overhead.

**Core Idea**: The central role of the MLP in NGP is to assign different importance to different locations of hash collisions, which can be replaced with spatially-varying, lightweight attention parameters.

## Method

### Overall Architecture

NGP-RT splits multi-level hash features into two components: coarse-grained (low-resolution) and fine-grained (high-resolution). The coarse-grained features $\tilde{\mathbf{f}}$ are decoded by an auxiliary NGP model and then baked into a low-resolution sparse voxel grid $\tilde{\mathcal{F}}$. The fine-grained features $\hat{\mathbf{f}}$ are stored as $L$ levels of explicit hash features, which are aggregated through a lightweight attention mechanism. The final features $\mathbf{f} = \tilde{\mathbf{f}} + \hat{\mathbf{f}}$ are fed into a deferred NeRF volume rendering pipeline: volume accumulation is first performed on density and diffuse colors, and one final tiny MLP is executed only once per ray.

### Key Designs

1. **Lightweight Attention Mechanism**: The core idea is to learn two spatially-varying attention parameters $\omega^l$ (for density) and $\beta^l$ (for color features) for each fine-grained level $l$, aggregating multi-level features via a channel-wise weighted sum:

$$\hat{\mathbf{f}} = \text{Att}(\hat{\mathbf{f}}^1, \ldots, \hat{\mathbf{f}}^L; \mathbf{a}) = \left[\sum_{l=1}^{L} \omega^l \cdot \hat{\sigma}^l, \quad \sum_{l=1}^{L} \beta^l \cdot \hat{\mathbf{c}}_d^l, \quad \sum_{l=1}^{L} \beta^l \cdot \hat{\mathbf{v}}_s^l\right]$$

where $\mathbf{a} = [\omega^1, \beta^1, \ldots, \omega^L, \beta^L]$. **Design Motivation**: (a) Density and color represent different modalities and should be weighted independently; (b) the spatial variation of the parameters mimics the implicit masking function of the MLP—different spatial locations subjected to hash collisions are disambiguated through different attention weights.

2. **Decoupled Training and Inference of Attention Parameters**: During training, an auxiliary NGP model (implicit hash features + shallow MLP) is used to decode the attention parameters $\mathbf{a}$ and coarse features $\tilde{\mathbf{f}}$, which are evaluated at voxel corner resolution $L_C$ and then trilinearly interpolated. After training, the auxiliary NGP is discarded, and the parameters are baked into a sparse grid $\mathcal{A}$. During inference, values are retrieved directly via lookups and interpolation:

$$\tilde{\mathbf{f}} = \text{Interp}(\tilde{\mathcal{F}}, \mathbf{x}), \quad \mathbf{a} = \text{Interp}(\mathcal{A}, \mathbf{x})$$

3. **Occupancy Distance Grid**: Existing methods check occupancy status voxel-by-voxel, leading to substantial global memory accesses. NGP-RT precomputes a $256^3$ distance grid $\mathcal{G}$ that stores the distance from each position to the nearest occupied voxel (as a uint8 integer, in units of voxel size), allowing empty regions to be skipped directly:

$$s_{\mathbf{p}} = \begin{cases} v \cdot \mathcal{G}_{\mathbf{p}}, & \text{if } \mathcal{G}_{\mathbf{p}} > 0 \\ s_{\mathcal{O}}, & \text{otherwise} \end{cases}$$

The grid $\mathcal{G}$ is queried only when the position is unoccupied and the ray-marching step size is smaller than the resolution of $\mathcal{G}$, reducing redundant marching points by over 40%.

### Loss & Training

Standard multi-view photometric reconstruction loss of NeRF is used. Training is divided into two stages: (1) end-to-end optimization of the multi-level explicit hash features, implicit features and MLP parameters of the auxiliary NGP, and the tiny MLP of the deferred NeRF; (2) baking the coarse-grained features and attention parameters into grids after training, and discarding the auxiliary NGP.

## Key Experimental Results

### Main Results

**Mip-NeRF 360 Full Scene (1080p)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|------|-------|-------|--------|------|
| Instant-NGP | 25.62 | 0.703 | 0.301 | 10.4 |
| MERF | 25.24 | 0.722 | 0.311 | 119 |
| BakedSDF | 24.51 | 0.697 | 0.309 | >60 |
| Gaussian-7K | 25.91 | 0.766 | 0.288 | 107 |
| **NGP-RT** | **25.64** | **0.737** | **0.299** | **108** |

NGP-RT achieves comparable quality to Instant-NGP with a ~10-fold speedup. It performs significantly better than MERF in indoor scenes (29.25 vs 27.80 PSNR).

### Ablation Study

**Comparison of Feature Aggregation Methods (L=4)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|------|-------|-------|--------|------|
| MLP | 26.22 | 0.764 | 0.268 | 14.7 |
| SUM | 25.51 | 0.719 | 0.315 | 66.9 |
| Shared-Att (Inv) | 25.69 | 0.738 | 0.295 | 65.9 |
| Separate-Att (V) | **26.05** | **0.753** | **0.280** | 61.8 |

Separate-Att (V) is close to MLP in quality but **more than 4x faster**, proving that lightweight attention is an excellent substitute for MLP.

**Impact of Fine-Grained Level Number $L$**:

| L | PSNR↑ | FPS↑ |
|---|-------|------|
| 2 | 25.64 | 108 |
| 3 | 25.93 | 79.7 |
| 4 | 26.05 | 61.8 |

**Effect of Occupancy Distance Grid**:

| Configuration | # Marching Points↓ | Time (ms)↓ |
|------|----------|-----------|
| W/o Distance Grid | 85.1 | 9.98 |
| W/ Distance Grid | **46.7** | **9.26** |

Marching points are reduced by 45%, resulting in a 7-10% speedup.

### Key Findings

- The core value of lightweight attention lies in **spatial variability**—spatially-invariant attention (Inv) performs significantly worse than spatially-varying attention (V), as hash collisions are inherently spatially dependent.
- Separated attention for density and color (Separate-Att) outperforms shared attention (Shared-Att), confirming that different modalities require different assignment of importance.
- Visualization reveals that attention effectively distributes texture details to different hash levels, and most collision locations are assigned small weights to avoid gradient interference.
- Raising $L=4$ over $L=3$ yields limited improvement but incurs a significant speed penalty; $L=2$ provides the optimal trade-off between speed and quality.

## Highlights & Insights

- **Precise Analysis of MLP Functions**: A deep analysis is provided on the true role of the MLP in NGP—which is not a simple feature transformation, but rather providing an implicit mask for hash collisions. This insight guides the design of the lightweight alternative.
- **Over 90% MAC Operation Reduction**: Lightweight attention reduces MAC (multiply-accumulate) operations by >90% compared to typical shallow MLPs.
- **Global Memory Access Optimization**: The concept of the occupancy distance grid is simple and practical, requiring only 16MB ($256^3 \times$ uint8) of additional storage to achieve a 7-10% speedup.
- **NeRF-Native Real-Time Scheme**: It demonstrates that the NeRF volume rendering framework itself can achieve 100+ fps without resorting to 3DGS.

## Limitations & Future Work

- The PSNR in outdoor scenes is slightly lower than that of MERF, indicating that the capability to model high-frequency fine structures is still insufficient.
- The quality is still lower than offline high-quality methods such as Zip-NeRF (25.64 vs 28.54 PSNR).
- The $256^3$ occupancy distance grid might require a more refined multi-scale design for extremely large-scale scenes.
- The baking process introduces discretization errors, which may lead to loss of detail in extremely fine structures.
- Compared to 3DGS, there remains a gap in storage efficiency and flexibility under comparable visual quality.

## Related Work & Insights

- **Instant-NGP**: The direct baseline of NGP-RT, inheriting its strong representation capability of multi-level hash features.
- **SNeRG**: Pioneered the deferred NeRF architecture, reducing the MLP execution to a per-ray basis.
- **MERF**: Improved upon the storage scheme of SNeRG (tri-planes + sparse grid). NGP-RT further enhances representation capabilities using multi-level hash features.
- **3DGS**: Represents another direction for rendering efficiency (rasterization-based vs. volume rendering). This work proves that the NeRF pipeline can also achieve real-time rates.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The insight of utilizing lightweight attention instead of an MLP to resolve hash collisions is novel, and the concept of the occupancy distance grid is simple and efficient.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Fully evaluated on the standard Mip-NeRF 360 dataset with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear analysis of the MLP roles, and the comparison diagrams with SNeRG/MERF are highly intuitive.
- **Value**: ⭐⭐⭐⭐ — Promotes the engineering practicality of real-time NeRF rendering, though it faces strong competition from 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)
- [\[ECCV 2024\] MegaScenes: Scene-Level View Synthesis at Scale](megascenes_scene-level_view_synthesis_at_scale.md)
- [\[CVPR 2026\] LagerNVS: Latent Geometry for Fully Neural Real-time Novel View Synthesis](../../CVPR2026/3d_vision/lagernvs_latent_geometry_for_fully_neural_real-time_novel_view_synthesis.md)
- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)
- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)

</div>

<!-- RELATED:END -->

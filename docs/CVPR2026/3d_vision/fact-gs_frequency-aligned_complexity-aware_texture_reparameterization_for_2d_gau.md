---
title: >-
  [Paper Note] FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][2D Gaussian Splatting] FACT-GS reframes texture parameterization as a sampling density allocation problem, employing a learnable deformation field to achieve frequency-adaptive non-uniform texture…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "2D Gaussian Splatting"
  - "texture parameterization"
  - "adaptive sampling"
  - "frequency alignment"
  - "deformation field"
date: 2026-05-08
content_hash: d181f84668195da5
---

# FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting

**Conference**: CVPR 2026 Findings  
**arXiv**: [2511.23292](https://arxiv.org/abs/2511.23292)  
**Code**: Available (based on gsplat framework)  
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: 2D Gaussian Splatting, texture parameterization, adaptive sampling, frequency alignment, deformation field

## TL;DR

FACT-GS reframes texture parameterization as a sampling density allocation problem, employing a learnable deformation field to achieve frequency-adaptive non-uniform texture sampling, substantially improving high-frequency detail recovery under a fixed parameter budget.

## Background & Motivation

**Background**: 3DGS/2DGS models appearance via spherical harmonics, which lacks spatial color variation within each primitive. Texture extension methods attach learnable texture maps to each Gaussian but rely on uniform sampling grids.

**Limitations of Prior Work**: Uniform texture parameterization leads to a **sampling–complexity mismatch**:
   - High-frequency regions (sharp edges, fine textures) are allocated insufficient texture capacity, causing detail loss.
   - Large smooth regions waste parameters to represent nearly uniform color.
   - Each Gaussian stores only low-resolution textures (e.g., $4\times4\times4$), further exacerbating the representational limitations of uniform sampling.

**Key Challenge**: Increasing texture resolution incurs quadratic growth in memory and bandwidth with minimal benefit, since the sampling pattern remains uniform; neural texture fields can resolve this but at the cost of real-time rendering performance.

**Goal**: Under a fixed texture resolution, allocate more sampling capacity to high-frequency regions.

**Key Insight**: Drawing from adaptive sampling theory, texture parameterization is treated as a sampling density allocation problem.

**Core Idea**: A learnable deformation field is introduced to modulate local sampling density via the Jacobian determinant, naturally directing texture capacity toward regions of high visual detail.

## Method

### Overall Architecture

FACT-GS builds upon 2D Gaussian Splatting with a two-stage optimization:
1. **Stage 1** (30k iterations): Standard 2DGS training.
2. **Stage 2** (30k iterations): Joint optimization of textures and the deformation field.

### Key Designs

1. **Frequency-Aligned Deformation Field**

   **Function**: Learns a deformation field $\mathbf{D}_i \in \mathbb{R}^{\tau \times \tau \times 2}$ per Gaussian that predicts per-texel displacements.

   **Mechanism**: Deformed coordinates are obtained in residual form:
   $$(u', v') = (u, v) + \lambda \mathbf{D}_i(u, v)$$
   This defines a continuous mapping $\Phi_i$, whose Jacobian determinant $|\det J_{\Phi_i}(u,v)|$ governs the local sampling density. A larger $|\det J_\Phi|$ corresponds to higher local texture capacity.

   **Design Motivation**: According to adaptive sampling theory, sampling density should increase with the local frequency of the signal. The target density is:
   $$\rho^\star(u,v) \propto (\|\nabla C(u,v)\| + \epsilon)^\alpha$$
   The deformation field is trained end-to-end via photometric loss, whose gradients naturally concentrate in high-frequency regions, driving $\Phi$ to expand sampling there and causing $|\det J_\Phi|$ to naturally approximate $\rho^\star$.

2. **Gradient Modulation**

   **Function**: The deformation field automatically modulates the local gradient field of the texture.

   **Mechanism**: By the chain rule, the gradient of the deformed texture is:
   $$\nabla c_i^{\text{tex}}(u,v) = J_{\Phi_i}(u,v)^\top \nabla \mathbf{T}_i^{\text{RGB}}(\Phi_i(u,v))$$
   The singular values of $J_\Phi$ amplify or attenuate the original texture gradient components, enabling the deformed texture to faithfully respond to high-frequency variations in the ground-truth color function.

   **Design Motivation**: This establishes a mathematical connection between geometric reparameterization and frequency-aligned density allocation—the deformation field acts not merely as a spatial transform, but as an active modulator of frequency content.

3. **Parameter-Efficient Design**

   **Function**: The deformation field shares resolution with the original texture, adding only 2 extra channels.

   **Mechanism**: For each Gaussian, $\mathbf{D}_i \in \mathbb{R}^{\tau \times \tau \times 2}$ adds only 50% more parameters over the base $\tau \times \tau \times 4$ texture, yet achieves quality gains far exceeding those of a direct resolution increase through non-uniform sampling.

   **Design Motivation**: Directly increasing texture resolution incurs quadratic memory overhead, whereas the deformation field scales linearly. Moreover, it modifies the sampling pattern rather than the data volume, preserving real-time rendering performance.

### Loss & Training

Standard GS loss combination:
$$\mathcal{L} = \eta \mathcal{L}_1 + (1-\eta)\mathcal{L}_{\text{SSIM}} + \mathcal{L}_\alpha, \quad \eta = 0.2$$

Learning rates for the texture and deformation field are $2.5 \times 10^{-3}$ and $1 \times 10^{-3}$, respectively. No pruning or densification is performed in Stage 2.

## Key Experimental Results

### Main Results (Five Standard Benchmarks, 100% Primitive Budget)

| Method | NeRF Synthetic PSNR↑ | MipNeRF360 PSNR↑ | DTU PSNR↑ | T&T PSNR↑ | LLFF PSNR↑ |
|---|---|---|---|---|---|
| 2D GS | 33.38 | 28.96 | 27.85 | 22.79 | 27.32 |
| Textured GS | 33.91 | 29.30 | 28.76 | 23.56 | 29.04 |
| **FACT-GS** | **34.02** | **29.34** | **28.76** | 23.53 | 28.99 |

### Ablation Study (Advantage Under Low Primitive Budget)

| Method | Primitive Ratio | NeRF Synthetic PSNR | LPIPS |
|---|---|---|---|
| 2D GS | 10% | 30.35 | - |
| Textured GS | 10% | - | - |
| **FACT-GS** | **10%** | **Significantly outperforms the above** | - |

### Key Findings

- Under the full parameter budget, FACT-GS consistently outperforms Textured GS across all benchmarks (up to 0.11 dB PSNR gain, with more pronounced LPIPS improvement).
- **The advantage is substantially amplified under low parameter budgets (10%/1%)**: non-uniform sampling proves especially valuable when parameters are constrained.
- Inference overhead introduced by the deformation field is negligible, maintaining real-time rendering.
- Compared with directly increasing texture resolution, FACT-GS achieves superior quality under the same parameter budget.

## Highlights & Insights

- **Theoretical Elegance**: Reframing texture parameterization as an adaptive sampling problem establishes a complete theoretical framework (target density → deformation field → Jacobian → sampling density).
- **Generality**: Although instantiated for Gaussian textures, the principle is applicable in principle to any spatially parameterized appearance representation (feature planes, neural textures, voxel grids).
- **Real-Time Friendly**: Only the texture parameterization is modified; the rendering pipeline remains entirely unchanged.
- **From "Geometry-Driven" to "Information-Driven"**: Texture layout transitions from static spatial uniformity to dynamic frequency awareness.

## Limitations & Future Work

- Gains under the full parameter budget are relatively modest (0.1–0.4 dB PSNR); the primary advantage manifests in parameter-constrained settings.
- The deformation field itself requires learning and storage; scalability to very large scenes warrants further investigation.
- Only static scenes are currently supported.
- The fixed two-stage training strategy may not be optimal.

## Related Work & Insights

- **Textured Gaussians**: The direct foundation of FACT-GS, attaching RGBA textures to each Gaussian.
- **Mip-Splatting**: Frequency-consistent parameterization at the global/feature-map level, without per-primitive adaptive adjustment.
- **Inspiration**: The "frequency alignment" principle can be extended to dynamic scenes (frequency adaptation along the temporal dimension) and large-scale scenes (frequency adaptation in spatial LoD).

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel theoretical perspective; deformation field implementation is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five benchmarks with complete ablations; low-budget comparisons are convincing.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear; the introduction of adaptive sampling theory is highly natural.
- Value: ⭐⭐⭐⭐ — Practically significant for parameter-constrained settings; the underlying principle has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](../../AAAI2026/3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2026\] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](directfisheye-gs_enabling_native_fisheye_input_in_gaussian_splatting_with_cross-.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes OceanSplat, which achieves high-fidelity underwater 3D Gaussian Splatting scene reconstruction under scattering media through trinocular view consistency…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "underwater scene reconstruction"
  - "trinocular view consistency"
  - "depth regularization"
  - "scattering medium"
date: 2026-05-08
content_hash: 8e8d4bfa9456c899
---

# OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2601.04984](https://arxiv.org/abs/2601.04984)  
**Code**: [oceansplat.github.io](https://oceansplat.github.io)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, underwater scene reconstruction, trinocular view consistency, depth regularization, scattering medium

## TL;DR

This paper proposes OceanSplat, which achieves high-fidelity underwater 3D Gaussian Splatting scene reconstruction under scattering media through trinocular view consistency constraints, synthetic epipolar depth priors, and depth-aware alpha adjustment, significantly reducing floating artifacts and surpassing existing methods.

## Background & Motivation

Underwater scene reconstruction is essential for marine robotics tasks such as seabed mapping, ecological monitoring, and underwater infrastructure inspection. However, the optical properties of underwater environments — wavelength-dependent attenuation, scattering, and low illumination — severely degrade visual cues, posing significant challenges for vision-based scene reconstruction.

**Limitations of Prior Work**:

**NeRF-based methods** (SeaThru-NeRF, etc.): embed underwater physical models into volumetric rendering, but implicit representations hinder precise geometric understanding and suffer from slow rendering speeds.

**3DGS-based methods** (SeaSplat, WaterSplatting, etc.): while rendering is fast, **medium intensity is often absorbed into the 3D Gaussians**, leading to extensive floating artifacts, entanglement between 3D Gaussians and the scattering medium, and degraded reconstruction quality.

**Key Challenge**: In scattering media, view-dependent sampling in alpha-blending leads to multi-view inconsistency, causing 3D Gaussians to erroneously represent the water volume itself rather than scene objects, producing floating artifacts.

**Key Insight**:
- Drawing on the advantage of multi-baseline stereo over single-baseline, the paper extends binocular consistency to **trinocular consistency** (horizontal + vertical virtual viewpoints), providing orthogonal constraints.
- Self-supervised depth priors are generated via triangulation between virtual viewpoints.
- **Depth-aware alpha adjustment** suppresses 3D Gaussians in medium regions during early training.

## Method

### Overall Architecture

OceanSplat builds upon the 3DGS framework, initializing 3D Gaussians with SfM and modeling underwater medium properties (attenuation, backscattering, medium color) via an MLP. Four key modules are introduced during training: trinocular view consistency, synthetic epipolar depth prior, depth residual loss, and depth-aware alpha adjustment.

The underwater image formation model decomposes the observed image into attenuated object color and backscattering:
$$C = C^{obj} \cdot e^{-\sigma^{attn} \cdot z} + C^{\infty} \cdot (1 - e^{-\sigma^{bs} \cdot z})$$

Object and medium rendering are accumulated separately via alpha-blending, enabling object–medium disentanglement.

### Key Designs

1. **Trinocular View Consistency**

   **Mechanism**: Two virtual viewpoints $P_h$ and $P_v$ (horizontal and vertical) are generated from the original camera pose $P_c$, and consistency across all three views is enforced to regularize the spatial positions of 3D Gaussians.

   Virtual viewpoints are constructed via translation:
    $P_h = \begin{bmatrix} \mathbb{I} & \mathbf{t}_h \\ \mathbf{0}^\top & 1 \end{bmatrix} P_c, \quad P_v = \begin{bmatrix} \mathbb{I} & \mathbf{t}_v \\ \mathbf{0}^\top & 1 \end{bmatrix} P_c$
   where $\mathbf{t}_h = (b_h, 0, 0)^\top$, $\mathbf{t}_v = (0, b_v, 0)^\top$.

   After rendering images from virtual viewpoints, disparity maps are computed from depth maps to perform inverse warping, aligning virtual-viewpoint images to the center view:
    $d_h(x,y) = \frac{f_h \cdot b_h}{D_c(x,y)}, \quad d_v(x,y) = \frac{f_v \cdot b_v}{D_c(x,y)}$

   The consistency loss comprises three components:
    - **Object stereo consistency**: $L_{obj\text{-}stereo}$, R-L1 loss between the warped object image and the center-view object image.
    - **Full stereo consistency**: $L_{full\text{-}stereo}$, R-L1 loss between the synthesized full image and the ground truth.
    - **Disparity smoothness**: $L_{smooth}$, edge-aware disparity regularization.

   **Design Motivation**: Single-baseline stereo provides constraints in only one direction. Orthogonal horizontal and vertical baselines yield stronger spatial constraints and better resolve geometric ambiguities in scattering media. $b_v$ is sampled from $[-0.4, 0.4]$ and $b_h = 1.5 b_v$, using unequal baselines to increase constraint diversity.

2. **Synthetic Epipolar Depth Prior**

   **Mechanism**: Self-supervised depth priors $D_{epi}$ are derived via triangulation between virtual viewpoints, requiring no external depth supervision.

   Specific steps:
    - Select 3D Gaussians within the trinocular view frustum intersection with opacity $> \tau_\alpha$.
    - Project selected Gaussians onto the image planes of $P_h$ and $P_v$.
    - Establish a linear system $\mathbf{A}_i \tilde{\mathbf{X}}_i = \mathbf{0}$ via epipolar geometry.
    - Solve for triangulated points via least squares, transform to the center camera coordinate system, and take the z-component as the depth prior.

   An edge-aware Log-L1 loss is applied:
    $L_{epi} = \frac{1}{HW}\sum_{x,y}\sum_{k}\log(1 + |D_c' - D_{epi}|) \cdot e^{-|\nabla_k I_c|}$

   **Design Motivation**: Geometric cues in underwater scenes are limited and external depth models may be inaccurate. Using geometric relationships between self-generated virtual viewpoints provides a self-consistent depth constraint, avoiding external dependencies.

3. **Depth Residual Loss**

   Constrains the z-component of each 3D Gaussian to be consistent with the alpha-blending rendered depth:
    $L_{res} = \frac{1}{N'}\sum_{i=1}^{N'}|D_c(\mathbf{x}_i) - z_i|$

   This prevents 3D Gaussians from spreading excessively along rays, reducing floating artifacts.

4. **Depth-aware Alpha Adjustment**

   During early training ($t < t_\alpha$), an MLP adjusts the opacity of each 3D Gaussian based on depth and viewing direction:
    $\alpha_i' = (1-w)\alpha_i + w \cdot \phi_\alpha(\alpha_i, z_i, \vec{\mathbf{v}}_i)$

   After the transition step $t_\alpha$, the weight $w$ decays to zero, eliminating inference overhead.

   **Design Motivation**: In scattering media, misplaced 3D Gaussians absorb medium color contributions. Suppressing the opacity of such Gaussians during early training encourages their pruning, preventing medium-induced artifacts at the source.

### Loss & Training

$$L_{total} = L_{photo} + \lambda_{tri} L_{tri} + \lambda_{epi} L_{epi} + \lambda_{res} L_{res}$$

- $L_{photo}$: weighted R-L1 + R-SSIM ($\lambda_s = 0.2$)
- $\lambda_{tri} = 0.1$, $\lambda_{res} = 0.01$
- $\lambda_{epi}$ annealed from 0.4 to 0.2
- Training steps: 7K/3K (densification/fine-tuning) for SeaThru-NeRF data; 10K/5K for In-the-Wild data
- Progressive resolution training: $1/4 \to 1/2 \to$ full resolution

## Key Experimental Results

### Main Results

**Real underwater scenes (SeaThru-NeRF + In-the-Wild)**:

| Dataset | Metric | OceanSplat | WaterSplatting | SeaSplat | Gain |
|---------|--------|------------|----------------|----------|------|
| Curaçao | PSNR | **34.56** | 32.32 | 29.77 | +2.24 |
| Panama | PSNR | **32.74** | 31.71 | 28.65 | +1.03 |
| J.G-Redsea | PSNR | **25.35** | 24.77 | 23.07 | +0.58 |
| IUI3-Redsea | PSNR | **30.17** | 29.84 | 27.23 | +0.33 |
| Coral | PSNR | **29.15** | 28.19 | 28.41 | +0.96 |
| Composite | PSNR | 26.39 | 25.47 | 26.22 | +0.92 |

Average PSNR surpasses WaterSplatting by 1.05 dB and SeaThru-NeRF-NS by 2.88 dB.

**Simulated scattering scenes (underwater + fog)**:

| Scene | Metric | OceanSplat | WaterSplatting | SeaSplat |
|-------|--------|------------|----------------|----------|
| Underwater-NVS | PSNR | **28.80** | 28.12 | 15.62 |
| Fog-NVS | PSNR | **29.12** | 28.45 | 27.52 |
| Underwater-Restoration | SSIM | **0.768** | 0.748 | 0.719 |
| Fog-Restoration | SSIM | **0.791** | 0.770 | 0.744 |

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS | Note |
|---------------|------|------|-------|------|
| Full Model | **34.56** | **0.961** | **0.113** | Complete model |
| w/o $L_{res}$ | 34.30 | 0.960 | 0.115 | Depth residual loss is effective |
| w/o $L_{epi}$ | 33.82 | 0.959 | 0.120 | Epipolar depth prior contributes significantly |
| w/o $L_{tri}$ | 33.20 | 0.957 | 0.115 | Trinocular consistency contributes most (−1.36 dB) |
| w/o $\alpha^d$ | 33.90 | 0.960 | 0.116 | Depth-aware alpha adjustment is effective |

Efficiency comparison: training in 19 minutes (vs. SeaThru-NeRF at 18h25m), inference at 85.67 FPS, GPU memory 7.6 GB.

### Key Findings

- Trinocular consistency is the most important component (PSNR drops 1.36 dB upon removal).
- The epipolar depth prior ranks second in contribution (−0.74 dB).
- Depth-aware alpha adjustment is notably effective in suppressing medium artifacts.
- All above components are self-supervised, requiring no external depth ground truth or annotations.

## Highlights & Insights

1. **Well-motivated geometric extension to trinocular**: Compared to binocular methods that provide only horizontal constraints, adding a vertical virtual viewpoint introduces orthogonal constraints with a solid theoretical basis in stereo geometry.
2. **Fully self-supervised depth regularization**: The synthetic epipolar depth prior is derived from triangulation between the model's own virtual viewpoints, relying on no external depth model and achieving a self-consistent geometric constraint.
3. **Object–medium disentanglement**: Effective geometric constraints promote separation between 3D Gaussians and the scattering medium, improving both reconstruction quality and scene restoration (dewatering/defogging).
4. **Preventive strategy via early alpha adjustment**: Rather than correcting artifacts after they emerge, this approach suppresses potentially problematic 3D Gaussians from the outset of training.

## Limitations & Future Work

- Each iteration requires additional rasterization (virtual viewpoint rendering) and least-squares solving, making training slightly longer than WaterSplatting (19 min vs. 10 min).
- The virtual viewpoint baseline lengths $b_h, b_v$ are empirically determined and may be sensitive to varying scene scales.
- Validation is currently limited to static underwater scenes; dynamic scenarios (water currents, bubbles) are not addressed.
- The scattering model remains simplified and does not account for complex wavelength-dependent scattering effects.

## Related Work & Insights

- **WaterSplatting (2024)**: A hybrid method combining implicit medium with explicit objects; the primary comparison baseline for this paper.
- **SeaSplat (2024)**: Incorporates underwater physics into 3D Gaussian Splatting but lacks sufficient geometric constraints.
- **StereoGS (2024, Han et al.)**: Regularizes 3DGS with binocular stereo consistency; this paper extends the idea to trinocular.
- **Insight**: The paradigm of constructing constraints via virtual viewpoints can be generalized to 3D reconstruction in other degraded scenes (fog, smoke, dust).

## Rating

- Novelty: ⭐⭐⭐⭐ (Trinocular extension is well-motivated; self-supervised depth prior is cleverly designed)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Real + simulated, NVS + restoration, complete ablations, detailed efficiency comparison)
- Writing Quality: ⭐⭐⭐⭐⭐ (Complete mathematical derivations, clear illustrations, thorough explanation of physical motivation)
- Value: ⭐⭐⭐⭐ (Significant advancement in underwater scene reconstruction; self-supervised design offers strong practicality)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction](splat-sap_feed-forward_gaussian_splatting_for_human-centered_scene_with_scale-aw.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[AAAI 2026\] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring
description: >-
  [CVPR 2026][Image Restoration][3D human avatar] The first method to directly reconstruct sharp, drivable 3D Gaussian human avatars from blurry video: proposes a 3D-aware physical blur formation model (decomposing blur in…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "3D human avatar"
  - "Gaussian splatting"
  - "motion blur"
  - "SMPL"
  - "deblurring"
date: 2026-05-08
content_hash: 84394b0d2cd20241
---

# MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring

**Conference**: CVPR 2026
**arXiv**: [2411.16758](https://arxiv.org/abs/2411.16758)  
**Code**: [GitHub](https://github.com/MyNiuuu/MAD-Avatar)  
**Area**: 3D Vision / Human Body Reconstruction / Deblurring
**Keywords**: 3D human avatar, Gaussian splatting, motion blur, SMPL, deblurring

## TL;DR
The first method to directly reconstruct sharp, drivable 3D Gaussian human avatars from blurry video: proposes a 3D-aware physical blur formation model (decomposing blur into sub-frame SMPL motion and canonical 3DGS), models sub-frame motion via B-spline interpolation and a pose deformation network, resolves motion direction ambiguity with inter-frame regularization, and substantially outperforms two-stage "2D deblurring + 3DGS" pipelines on both synthetic and real datasets (~2.5 dB PSNR gain).

## Background & Motivation
3D human avatar reconstruction methods (e.g., GauHuman) rely on sharp multi-view video inputs, yet in practice human motion inevitably introduces motion blur, leading to: (1) 3DGS learning degenerate 3D representations (the ambiguity of blur means a single blurry image can correspond to multiple distinct motions); and (2) inaccurate SMPL parameter estimation from blurry frames. Existing two-stage pipelines (first 2D deblurring, then 3DGS training) are inadequate: 2D deblurring lacks 3D structural information, producing multi-view inconsistencies that ultimately limit 3DGS reconstruction quality.

## Core Problem
How to directly reconstruct sharp, animatable 3D human avatars from multi-view blurry video? The key challenges are motion ambiguity introduced by blur (the same blur effect can arise from multiple distinct motions) and SMPL initialization errors.

## Method

### Overall Architecture
**Input**: Multi-view blurry video + coarsely estimated SMPL parameters from blurry frames. The model jointly optimizes two objectives: (1) a sharp 3DGS avatar in canonical space; and (2) a sub-frame SMPL motion sequence within each frame's exposure interval. The canonical 3DGS is warped to observation space according to the estimated sub-frame motions, rendered into $T$ virtual sharp frames, and averaged to synthesize a simulated blurry frame, which is then compared to the observed blurry frame via an L1 loss.

### Key Designs
1. **3D Blur Formation Model**: Extends the classical 2D blur formation equation (exposure integration) to 3D: blurry frame $= \frac{1}{T}\sum R(W(G_\text{canonical}, S_t), R, K)$. Rather than convolving a blur kernel at the pixel level, the model renders SMPL-driven deformations of 3DGS in 3D space and averages the results. This naturally leverages 3D structure and multi-view consistency for deblurring.

2. **Sub-frame Motion Model**: Comprises two components — (a) B-spline rigid pose interpolation: parameterizes the continuous rotation trajectories of 24 SMPL joints over the exposure interval using $P$ control knots, enforcing motion smoothness; (b) pose deformation network $G_\text{disp}$: a CNN that predicts residual displacements per joint per timestep, capturing high-frequency non-rigid variations that B-splines cannot represent.

3. **Inter-frame Motion Regularization**: Addresses motion direction ambiguity (illustrated in Fig. 1(c) — two symmetric motions can produce similar blur). The pose at the last timestep of the current frame is constrained to be close (in geodesic distance) to the pose at the first timestep of the next frame, exploiting inter-frame temporal continuity to break the symmetry ambiguity.

4. **Joint SMPL Parameter Optimization**: Shape $\beta$, LBS weights (base + CNN offsets), and per-frame sub-frame poses are all treated as learnable parameters and jointly optimized, without relying on accurate initial SMPL estimates.

### Loss & Training
$\mathcal{L} = \mathcal{L}_1(\text{synthesized blurry frame},\ \text{observed blurry frame}) + \mathcal{L}_\text{reg}(\text{inter-frame pose continuity regularization})$. Adam optimizer; learning rate and decay schedule follow the original 3DGS settings. Input resolution: $512\times512$ (synthetic) / $612\times512$ (real); single RTX 4090 GPU.

## Key Experimental Results

### Synthetic Dataset (ZJU-MoCap, $K_\text{blur}=5$)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| GauHuman (blurry input) | 23.08 | 0.766 | 0.228 |
| BSST + GauHuman (best two-stage) | 23.08 | 0.770 | 0.221 |
| **Ours** | **25.55** | **0.829** | **0.148** |

### Real Dataset (360° Mixed-Exposure Camera Rig)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| BSST + GauHuman | 25.57 | 0.807 | 0.234 |
| **Ours** | **27.01** | **0.827** | **0.167** |

### Ablation Study
- **Removing B-spline interpolation (independently optimizing poses per timestep)**: PSNR drops by 1.5 dB, as unconstrained per-timestep pose optimization leads to disordered motion estimation.
- **Removing the pose deformation network**: PSNR drops by 0.25 dB; B-spline alone is insufficient to capture complex motion details.
- **Removing inter-frame regularization**: Performance at the middle timestep ($t=0.5$) is nearly unaffected, but non-middle timesteps degrade significantly (~1 dB PSNR drop) due to motion direction misjudgment.
- **Removing SMPL optimization**: PSNR drops by 3.9 dB (synthetic) and 1.9 dB (real), indicating that coarse SMPL estimates from blurry frames are highly inaccurate and joint optimization is essential.
- **B-spline vs. linear vs. Slerp interpolation**: Differences are marginal (B-spline slightly best), as the pose deformation network compensates for interpolation accuracy differences.
- **Robustness to SMPL initialization perturbations**: Even under large random perturbations ($\xi=0.4$), PSNR drops by only 0.4 dB, demonstrating independence from accurate initialization.
- **Different blur magnitudes**: The method substantially outperforms baselines across $K_\text{blur} \in \{5,7,9,11\}$, confirming robustness to varying degrees of blur.

## Highlights & Insights
- **"3D-aware blur formation" paradigm**: Rather than performing 2D deblurring, the method models the blur formation process in 3D space, allowing deblurring and 3D reconstruction to mutually reinforce each other. This idea is transferable to other dynamic 3D reconstruction tasks.
- **Elegant resolution of motion direction ambiguity**: Inter-frame continuity regularization is a simple yet critical design — without it, middle-timestep frames show almost no degradation while non-middle-timestep frames collapse entirely, confirming that directional ambiguity is a genuine bottleneck.
- **360° mixed-exposure camera benchmark**: A real benchmark comprising 12 synchronized cameras (4 blurry + 8 sharp) was physically constructed, offering lasting value to the research community.
- **iPhone demo generalization**: The method also works on monocular iPhone video with SMPL estimates from TRAM, demonstrating practical applicability.

## Limitations & Future Work
- SMPL-based representation cannot handle motion blur from hand-held objects or loose clothing.
- Averaging is performed in sRGB space rather than linear radiance space, introducing physical inaccuracies in high-contrast regions.
- Geometry (normals / BRDF) cannot be recovered due to the 3DGS representation.
- Training overhead is not discussed in detail; rendering and averaging multiple sub-frames may be slow.

## Related Work & Insights
- **vs. BAD-NeRF / Deblur-NeRF**: These methods address camera-motion blur or defocus blur in static scenes and are not applicable to motion blur from dynamic human body movement.
- **vs. DyBluRF / BARD-GS**: Handle blur in dynamic scenes but cannot produce drivable avatars.
- **vs. GauHuman / 3DGS-Avatar**: Avatar methods designed for sharp inputs that degrade severely under blurry inputs.
- The "3D-aware blur formation" methodology may be useful in video understanding for improving robustness to real-world scenes through physics-based blur modeling, though human avatar reconstruction is not a core research focus.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to formulate the "blurry video → sharp drivable avatar" problem; the 3D blur formation model is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Synthetic + real datasets, 10+ ablation variants, robustness tests across perturbations / blur magnitudes / number of views / mask methods, plus an iPhone demo.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical structure, information-rich figures and tables, well-motivated problem setup.
- **Value**: ⭐⭐⭐ The 3D blur formation methodology is worth borrowing; human avatar reconstruction itself is not a core focus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](selfhvd_self-supervised_handheld_video_deblurring.md)
- [\[CVPR 2026\] BluRef: Unsupervised Image Deblurring with Dense-Matching References](bluref_unsupervised_image_deblurring_with_dense-matching_references.md)
- [\[CVPR 2026\] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[NeurIPS 2025\] The Effect of Optimal Self-Distillation in Noisy Gaussian Mixture Model](../../NeurIPS2025/image_restoration/the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)

</div>

<!-- RELATED:END -->

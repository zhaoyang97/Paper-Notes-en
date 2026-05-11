---
title: >-
  [Paper Note] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation
description: >-
  [ICCV2025][3D Vision][dual-pixel] Sdirt proposes a ray-tracing-based dual-pixel (DP) image simulation framework that computes spatially varying DP PSFs incorporating lens aberrations and phase-splitting characteristics…
tags:
  - "ICCV2025"
  - "3D Vision"
  - "dual-pixel"
  - "depth estimation"
  - "ray tracing"
  - "PSF simulation"
  - "domain gap"
date: 2026-05-08
content_hash: 13c4442df9b22f3c
---

# Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation

**Conference**: ICCV2025
**arXiv**: [2503.11213](https://arxiv.org/abs/2503.11213)
**Code**: [GitHub](https://github.com/LinYark/Sdirt)
**Area**: 3D Vision
**Keywords**: dual-pixel, depth estimation, ray tracing, PSF simulation, domain gap

## TL;DR

Sdirt proposes a ray-tracing-based dual-pixel (DP) image simulation framework that computes spatially varying DP PSFs incorporating lens aberrations and phase-splitting characteristics, thereby bridging the domain gap between simulated and real DP data and improving the generalization of depth estimation models on real DP images.

## Background & Motivation

- **Problem Definition**: DP sensors split each pixel into left and right sub-pixels, leveraging microlenses to achieve phase splitting and capturing a pair of DP images in a single shot. DP images can be used for depth-from-dual-pixel (DfDP) estimation, yet paired DP-depth data is extremely scarce.
- **Limitations of Prior Work**:
    - **Calibration-based simulators** (Xin et al., Li et al.): require extensive real-camera calibration time, suffer from interpolation errors at discrete calibration points, and generalize poorly to other lenses.
    - **Model-based simulators** (DDDNet, Pan et al., Punnappurath et al.): compute DP PSFs using ideal optical models but neglect lens aberrations and sensor phase-splitting properties.
    - As illustrated in the paper, the CoC-simulated DP PSF from the ideal thin-lens model exhibits a significant domain gap relative to real PSFs.
- **Key Challenge**: Existing model-based simulators violate real optical propagation laws, causing models trained on simulated DP images to generalize poorly to real DP data.
- **Design Motivation**: Accurately simulate DP PSFs that encode aberrations and phase information via ray tracing, so as to fundamentally reduce the sim-to-real domain gap.

## Method

### Overall Architecture

Sdirt consists of three modules:
1. **Ray-tracing DP PSF simulator**: computes spatially varying DP PSFs.
2. **DP PSF prediction network**: an MLP that predicts the DP PSF for each pixel to reduce computational overhead.
3. **Per-pixel DP image rendering module**: generates simulated DP images via convolution.

### Key Design 1: Ray-Tracing DP PSF Simulator

**Lens ray tracing**: Starting from an object point $p$, $n$ rays are densely sampled on the entrance pupil. Each ray updates its position and direction at every lens surface according to Snell's law and the lens parameters, ultimately arriving at the sensor plane to yield landing point $O$ and direction $D$.

**DP sensor ray tracing**: The DP pixel structure is simplified by modeling each microlens as a thin lens (radius $r$, focal length $f$), with sub-pixel width $w$, microlens-to-sub-pixel distance $h$, and pixel size $ps$.

**Two cases determine which sub-pixel a ray enters:**

When a ray lands within the microlens aperture, it is refracted by the microlens and directed into a sub-pixel according to the boundary lines:

$$x_{L1} = x_i + w - (f \cdot \tan\theta - w) \cdot h / (f - h)$$
$$x_{M1} = x_i - (f \cdot \tan\theta) \cdot h / (f - h)$$
$$x_{R1} = x_i - w - (f \cdot \tan\theta + w) \cdot h / (f - h)$$

When a ray lands outside the microlens aperture, it enters a sub-pixel directly:

$$x_{L2} = x_i + w - h \cdot \tan\theta, \quad x_{M2} = x_i - h \cdot \tan\theta$$

**Left PSF computation** (integrating energy contributions of all $n$ rays over each left sub-pixel):

$$PSF_L(i,j) = \sum_{k=1}^{n} A_k \cdot \delta_{L,k}(i,j)$$

### Key Design 2: MLP PSF Prediction Network

To reduce the computational cost of ray tracing, an MLP is trained to predict DP PSFs:
- **Input**: normalized coordinates $p$ within the effective imaging volume (a frustum defined by the field of view and sensor).
- **Network**: 5 hidden layers × 512 neurons + output layer with $2 \times ks^2$ neurons.
- **Training loss**:

$$Loss = L_2(\widehat{PSF_L}, PSF_L) + L_2(\widehat{PSF_R}, PSF_R)$$

- Max-value normalization is used during training; sum normalization is used at inference (approximating the camera's vignetting compensation).

### Key Design 3: Per-Pixel DP Image Rendering

Each pixel in the depth map is treated as an object point. The trained MLP predicts its DP PSF, which is then applied via per-pixel local convolution with the all-in-focus RGB image to render simulated DP images that encode aberrations and phase information.

### DfDP Model Adaptation

AANet is adopted as the DfDP backbone. A key modification introduces reverse disparity in the cost volume construction step (blue arrows in the paper), since the disparity direction for objects in front of and behind the focal plane is reversed in DP images—unlike the unidirectional disparity in stereo images.

Training loss: $Loss = L_1(\hat{I_D}, I_D)$

## Key Experimental Results

### Main Results: DP PSF Simulation Accuracy

| Method | NCC ↑ | NSD ↓ |
|--------|-------|-------|
| DDDNet | 0.589 | 0.625 |
| L2R | 0.638 | 0.523 |
| CoC | 0.672 | 0.448 |
| Modeling | 0.707 | 0.423 |
| **Sdirt** | **0.915** | **0.133** |

NCC (Normalized Cross-Correlation) measures similarity; NSD (Normalized Squared Difference) measures error. Sdirt achieves substantially higher PSF fidelity than all baselines.

### Depth Estimation Results (DP119 Test Set)

| Scene | Method | MAE ↓ | MSE ↓ | Acc-1 ↑ | Acc-2 ↑ |
|-------|--------|-------|-------|---------|---------|
| Planar | **Sdirt** | **0.0845** | **0.0109** | **0.9849** | **0.9997** |
| Planar | CoC | 0.2085 | 0.1001 | 0.6670 | 0.8990 |
| Box | **Sdirt** | **0.1197** | **0.0339** | **0.9474** | **0.9812** |
| Box | CoC | 0.3375 | 0.1804 | 0.4412 | 0.8277 |
| Casual | **Sdirt** | **0.2702** | **0.2294** | **0.8236** | **0.9314** |
| Casual | CoC | 0.7925 | 1.8579 | 0.3318 | 0.6103 |

### Ablation Study / Key Findings

1. **Large PSF accuracy gap**: Sdirt NCC=0.915 vs. the next best Modeling=0.707; other methods exhibit significant domain gaps due to their neglect of aberrations and phase splitting.
2. **Depth estimation generalization**: Acc-1 improves from 0.6670 (CoC) to 0.9849 (Sdirt) on the Planar scene, a margin of approximately 30 percentage points.
3. **Robustness on Casual scenes**: Even with textureless regions, Sdirt achieves Acc-1=0.8236, far exceeding the next best score of 0.3318.
4. **Key physical phenomenon**: As the image field angle increases, real PSF phase asymmetry and aberrations grow substantially—properties entirely ignored by existing simulators.
5. **Quantitative evaluation of simulated DP images** (PSNR/SSIM): Sdirt achieves the best performance across all depth ranges, with average PSNR=37.20 and SSIM=0.9845.

## Highlights & Insights

1. **Return to physics-based modeling**: In the deep learning era, this work addresses the domain gap by returning to rigorous physical optics (ray tracing) rather than relying on simplified idealized models.
2. **Elegant MLP-accelerated ray tracing**: The MLP approximates ray-tracing results—ground-truth PSFs are computed offline during training, enabling fast inference.
3. **New benchmark DP119**: 119 scenes (planar/box/casual) with known lens structures and fixed focus distances, filling a gap in evaluation resources.
4. **Physics-aware cost volume**: Adding reverse disparity to capture the direction reversal of DP defocus cues across the focal plane is a simple yet critical adaptation.
5. **Thorough domain gap analysis**: The paper systematically analyzes how PSF-level discrepancies propagate to image-level and depth-estimation-level performance degradation.

## Limitations & Future Work

1. Applicable only to fixed-focus lenses with known optical prescriptions paired with DP sensors; currently only Canon camera systems satisfy this requirement.
2. Camera manufacturers do not disclose DP pixel structural parameters, necessitating simplified modeling assumptions.
3. Chromatic aberration is assumed to be corrected (single wavelength 550 nm); accuracy may degrade for systems with significant chromatic aberration.
4. At F/1.8, excessively large defocus kernels cause GPU memory overflow; experiments are therefore restricted to F/4.
5. The DP119 dataset is relatively small in scale with limited scene diversity.

## Related Work & Insights

- **Evolution of DP simulators**: calibration-based (accurate but time-consuming) → model-based (fast but inaccurate) → **ray-tracing-based** (this work: accurate and transferable).
- **Sim-to-real domain gap**: This work provides a systematic study of simulation-to-real discrepancies in DP imaging; the methodology is transferable to other computational imaging tasks.
- **MLP as a physical field approximator**: Consistent with the NeRF paradigm, a lightweight MLP approximates a spatially varying physical quantity (here, the PSF).
- **Broader applications of DP data**: Beyond depth estimation, the framework is relevant to deblurring, refocusing, raindrop removal, and reflection removal.

## Rating ⭐⭐⭐⭐

The problem is clearly defined (DP simulation domain gap), the method is grounded in rigorous physical optics modeling, and the experimental design and evaluation metrics are carefully constructed. Depth estimation performance improves substantially. The DP119 dataset constitutes an independent contribution. The primary limitation is the narrow applicability (Canon cameras with known lens prescriptions), but within that constraint the work is thorough and complete.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[NeurIPS 2025\] DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints](../../NeurIPS2025/3d_vision/dualfocus_depth_from_focus_with_spatio-focal_dual_variational_constraints.md)
- [\[NeurIPS 2025\] 3D Visual Illusion Depth Estimation](../../NeurIPS2025/3d_vision/3d_visual_illusion_depth_estimation.md)

</div>

<!-- RELATED:END -->

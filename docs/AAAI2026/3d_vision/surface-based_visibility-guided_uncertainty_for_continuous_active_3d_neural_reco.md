---
title: >-
  [Paper Note] Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction
description: >-
  [AAAI 2026][3D Vision][Active 3D reconstruction] This paper proposes a Surface-Based Visibility field (SBV) that derives surface confidence from SDF sign changes and updates it via a voxel grid, enabling accurate visibility-aware uncertainty estimation during continuous active learning for Next-Best View selection. SBV achieves up to 11.6% improvement in image rendering quality across four benchmarks: DTU, Blender, TanksAndTemples, and BlendedMVS.
tags:
  - AAAI 2026
  - 3D Vision
  - Active 3D reconstruction
  - neural implicit surface
  - uncertainty estimation
  - Next-Best View
  - visibility reasoning
date: 2026-05-08
content_hash: e7f27747d0b0d56c
---

# Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2405.02568](https://arxiv.org/abs/2405.02568)
**Code**: [https://github.com/hskAlena/Surface-Based-Visibility](https://github.com/hskAlena/Surface-Based-Visibility)
**Area**: 3D Vision
**Keywords**: Active 3D reconstruction, neural implicit surface, uncertainty estimation, Next-Best View, visibility reasoning

## TL;DR
This paper proposes a Surface-Based Visibility field (SBV) that derives surface confidence from SDF sign changes and updates it via a voxel grid, enabling accurate visibility-aware uncertainty estimation during continuous active learning for Next-Best View selection. SBV achieves up to 11.6% improvement in image rendering quality across four benchmarks: DTU, Blender, TanksAndTemples, and BlendedMVS.

## Background & Motivation
**State of the Field**: Active 3D neural reconstruction selects the Next-Best View (NBV) by evaluating scene uncertainty, reducing data acquisition and computational costs. Existing methods quantify uncertainty using the variance of volumetric density distributions (ActiveNeRF), variance or entropy of color distributions (ActiveRMAP), or Fisher information (FisherRF).

**Limitations of Prior Work**: (1) Visibility estimation based on volumetric density is severely inaccurate when the model is underfitted—high uncertainty in low-density regions is suppressed by the volume rendering process, causing informative candidate views to be excluded. (2) Existing visibility-aware methods (NVF) can only estimate visibility after full convergence within a training sub-phase, making them inapplicable to continuous active learning scenarios.

**Root Cause**: Volume rendering cannot reliably determine the visibility of uncertain regions in early training stages—uncertainty from low-density floaters and underfitted surfaces is systematically underestimated. Yet continuous active learning requires real-time assessment of information gain for view selection.

**Paper Goals**: During continuous active learning (where new views are selected while the model trains), how can one accurately estimate the visibility of uncertainty in different regions—i.e., distinguish "uncertainty on surfaces" from "uncertainty in empty space"?

**Starting Point**: Surface confidence is derived directly from NeuS SDF values via sign-change detection, bypassing the unreliability of volumetric density. A voxel grid stores and robustly updates surface confidence, enabling fast and stable visibility reasoning.

**Core Idea**: Rather than relying on volumetric density, surface detection is performed via SDF sign-change detection. Surface confidence is robustly updated through a voxel grid, allowing accurate determination of whether uncertainty lies on a visible surface throughout continuous training.

## Method

### Overall Architecture
Building upon the NeuS neural implicit surface network, the color prediction is extended to a Gaussian distribution $c(\mathbf{r}(t)) \sim \mathcal{N}(\bar{c}, \beta^2)$, where the variance $\beta^2$ represents rendering uncertainty. A voxel grid is maintained to store two types of information: (1) per-voxel uncertainty (color variance $\beta^2$); and (2) per-voxel surface confidence (SDF sign-detection score). Information gain is computed based on surface-guided visibility: when a ray passes through a surface voxel, only the entropy of that surface voxel is counted; when a ray does not encounter a surface, the entropy of all traversed voxels is accumulated.

### Key Designs

1. **Rendering Uncertainty Estimation**:

    - Function: Estimate the color uncertainty at every 3D point in the scene.
    - Mechanism: The NeuS network is extended to $F_\Theta: (\mathbf{x}, \mathbf{d}) \rightarrow (g(\mathbf{x}), \bar{c}, \beta^2)$, where the color variance $\beta^2$ is view-direction-independent. The uncertainty loss is the negative log-likelihood: $\mathcal{L}_u = \frac{1}{M}\sum_i (\frac{\|\bar{\mathcal{C}}(\mathbf{r}_i) - C(\mathbf{r}_i)\|^2}{2\mathcal{B}^2(\mathbf{r}_i)} + \frac{\log \mathcal{B}^2(\mathbf{r}_i)}{2})$
    - Design Motivation: Modeling color as a Gaussian distribution allows the variance to naturally reflect reconstruction quality at each point—high variance indicates multi-view color inconsistency or insufficient training.

2. **SDF-Based Surface Confidence**:

    - Function: Detect whether a surface exists within each voxel and robustly update the confidence score.
    - Mechanism: Three points are sampled along the ray within each voxel—at the voxel center and at offsets of ±$1/s$ (NeuS step size). If the SDF values of the two offset points have opposite signs (negative product), a surface is detected. Confidence is binary (1 = surface present, 0 = absent), updated as $\max(\text{previous} \times 0.95, \text{current})$; voxels with confidence above 0.8 are treated as surface voxels. The decay rate of 0.95 compensates for sampling noise.
    - Design Motivation: Volumetric density is highly unreliable in early training (low density in underfitted regions), whereas SDF sign changes provide a more stable surface indicator. Progressive voxel grid updates allow surface estimation to improve continuously throughout training.

3. **SBV-Guided Information Gain Computation**:

    - Function: Differentiate between ray types based on surface visibility to accurately compute information gain for candidate views.
    - Mechanism: Color entropy is defined as $H(c) = \frac{1}{2}\log(2\pi\beta^2) + \frac{1}{2}$. For rays that pass through surface voxels, information gain is computed only from surface voxel entropy; for rays that do not encounter a surface (empty or ambiguous regions), entropy is accumulated over all traversed voxels. The formulation is $G_s(v) = \frac{1}{N}\sum_{r \in \mathcal{R}_v}\sum_{x \in \tilde{\mathcal{X}}_r} H(c(x))$, where $\tilde{\mathcal{X}}_r = \mathcal{X}_r \cap \mathcal{S}$ (when a surface is encountered) or $\tilde{\mathcal{X}}_r = \mathcal{X}_r$ (otherwise).
    - Design Motivation: Uncertainty on surfaces directly affects rendering quality and should be prioritized. Uncertainty in empty regions, though potentially high, has little impact on output. SBV naturally implements this distinction through surface confidence.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_s + \omega\mathcal{L}_u$, where $\mathcal{L}_s$ combines an L1 color loss and Eikonal regularization, and $\omega$ balances the two terms. A two-stage multi-view selection strategy selects 2 or 10 NBVs per round using a distance-aware scheme to ensure diversity—candidate views must be farther than threshold $\tau$ (initialized at 1.732, decayed by 0.95) from all already-selected views, and the candidate with the highest information gain among those satisfying the constraint is chosen. Frequency regularization is applied during training to prevent overfitting under sparse views, and a warm-up phase stabilizes early training.

## Key Experimental Results

### Main Results (Four Benchmarks)

| Method | DTU-PSNR↑ | Blender-PSNR↑ | TNT-PSNR↑ | BlendedMVS-PSNR↑ |
|--------|-----------|--------------|-----------|-----------------|
| Random | 27.69 | 16.24 | 17.17 | 26.21 |
| FVS | 27.09 | 20.07 | 17.94 | 25.49 |
| Entropy | 24.21 | 15.41 | 16.70 | 25.72 |
| ActiveNeRF* | 26.30 | 19.25 | 18.62 | 26.57 |
| FisherRF* | 27.78 | 20.48 | 18.44 | 25.71 |
| **Ours (SBV)** | **28.19** | **21.22** | **20.49** | **26.80** |

### Ablation Study & New Dataset

| Configuration/Method | DTU-PSNR↑ | DTU-Chamfer↓ | ImBView-PSNR↑ |
|----------------------|-----------|-------------|--------------|
| SBV (with surface guidance) | **28.19** | **2.002** | **32.23** |
| Without surface guidance (all voxels accumulated) | 27.18 | 2.251 | - |
| Random | 27.69 | 2.920 | 27.88 |
| ActiveNeRF* | 26.30 | 2.395 | 29.60 |
| FisherRF* | 27.78 | 3.476 | 27.77 |

### Key Findings
- SBV consistently outperforms all baselines across all four benchmarks, with particularly large gains on the large-scale TanksAndTemples dataset—PSNR leads the second-best method by 1.87 (20.49 vs. 18.62), an improvement of 11.6%.
- Surface confidence guidance is critical: removing it reduces PSNR on DTU from 28.19 to 27.18 and increases Chamfer distance from 2.002 to 2.251.
- On the imbalanced-viewpoint ImBView dataset, SBV surpasses FisherRF* by 4.46 PSNR (32.23 vs. 27.77), as FisherRF tends to select views at the extremities while SBV selects diverse views with broader angular coverage.
- NBV selection speed: SBV requires only 0.8 seconds, substantially faster than ActiveNeRF* (10.8 s) and FisherRF* (13.5 s), and comparable in efficiency to the voxel-based Entropy method (0.5 s).
- SBV also achieves the best mesh reconstruction on DTU (Chamfer distance 2.002 vs. 2.395 for the second-best ActiveNeRF*).

## Highlights & Insights
- **Surface detection via SDF sign changes**: This approach circumvents the fundamental unreliability of volumetric density in early training. Zero-crossings of the SDF naturally mark surface locations and provide useful signals even before network convergence. This idea is transferable to any task requiring geometric estimation from intermediate training states.
- **Robust voxel grid update strategy**: The $\max(\text{previous} \times 0.95, \text{current})$ design—slow decay with instant update—ensures surface confidence converges monotonically without oscillating due to training fluctuations. Simple yet effective.
- **View diversity guarantee**: The decaying distance threshold $\tau \times 0.95$ elegantly balances information gain and view distribution—preferring distant high-IG views and progressively relaxing the constraint when no qualifying candidate exists.
- **ImBView dataset design**: By deliberately introducing viewpoint imbalance (75% normal / 12.5% high-angle / 12.5% low-angle), the dataset cleanly exposes selection bias in different NBV strategies.

## Limitations & Future Work
- **Motion cost between views not considered**: In robotic active reconstruction, the travel cost between consecutive views is a critical constraint, which this work does not address.
- **Architectural constraints**: NeuS is used rather than NeuS2 or Neuralangelo, because the locality of multi-resolution hash encoding adversely affects the Eikonal loss and surface accuracy.
- **Foreground objects only**: Object masks are required, making the method unsuitable for indoor scenes or depth-dependent environments.
- **Background noise**: SBV information gain visualizations exhibit persistent background noise patterns caused by NerfAcc grid sampling.

## Related Work & Insights
- **vs. NVF**: NVF estimates visibility-guided uncertainty only after full convergence within a training sub-phase, following a staged train–evaluate–select paradigm that is incompatible with continuous active learning. SBV decouples surface confidence updates via a voxel grid, enabling online visibility estimation throughout training.
- **vs. ActiveNeRF**: ActiveNeRF uses color variance without accounting for visibility. On DTU fruit scenes, it cannot distinguish surface from non-surface uncertainty, failing to select views that cover regions with complex occlusions.
- **vs. FisherRF**: FisherRF quantifies information gain using the Fisher information matrix but is computationally slow (13.5 s vs. SBV's 0.8 s). ImBView experiments reveal severe view selection bias (preference for extreme-angle views), whereas SBV's surface guidance yields more balanced coverage.
- **vs. Entropy method**: The Entropy method uses occupancy probability entropy without accounting for surface occlusion, causing uncertain regions behind occluders to be counted toward information gain. Its DTU mesh reconstruction Chamfer distance (3.644) is far worse than SBV's (2.002).
- **Inspiration from voxel methods**: Traditional voxel occupancy probability methods (Isler et al.) update stably but are incompatible with neural representations. SBV replaces occupancy probability with SDF, achieving the best of both worlds.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of replacing volumetric density with SDF sign-change detection for surface localization is concise and effective, though the overall framework is an incremental combination of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four major benchmarks plus a custom dataset, five baseline comparisons, full coverage of 1/2/4/10-image settings, and runtime comparisons are all included.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear (the volumetric density visualization in Fig. 2 is highly convincing), methodology is rigorously presented, and the appendix is thorough.
- Value: ⭐⭐⭐⭐ Addresses a practical bottleneck in continuous active learning—visibility estimation—with significant improvements across multiple benchmarks and fast NBV selection (0.8 s).

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](../../ICLR2026/3d_vision/peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](../../CVPR2026/3d_vision/neural_gabor_splatting.md)

<!-- RELATED:END -->

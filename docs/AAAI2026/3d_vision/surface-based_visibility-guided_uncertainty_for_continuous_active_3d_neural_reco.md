---
title: >-
  [Paper Note] Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction
description: >-
  [AAAI 2026][3D Vision][Active 3D reconstruction] Proposes Surface-Based Visibility (SBV), which utilizes SDF-derived surface confidence and a voxel grid update mechanism to accurately estimate visibility of uncertainty during continuous active learning. Guided by SBV, Next-Best View selection achieves an image rendering quality improvement of up to 11.6% across four benchmarks: DTU, Blender, TanksAndTemples, and BlendedMVS.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Active 3D reconstruction"
  - "neural implicit surface"
  - "uncertainty estimation"
  - "Next-Best View"
  - "visibility reasoning"
date: 2026-05-08
content_hash: 47f1b5c90b14b8fd
---

# Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction

**Conference**: AAAI 2026  
**arXiv**: [2405.02568](https://arxiv.org/abs/2405.02568)  
**Code**: [https://github.com/hskAlena/Surface-Based-Visibility](https://github.com/hskAlena/Surface-Based-Visibility)  
**Area**: 3D Vision  
**Keywords**: Active 3D reconstruction, neural implicit surface, uncertainty estimation, Next-Best View, visibility reasoning

## TL;DR
Proposes Surface-Based Visibility (SBV), which utilizes SDF-derived surface confidence and a voxel grid update mechanism to accurately estimate visibility of uncertainty during continuous active learning. Guided by SBV, Next-Best View selection achieves an image rendering quality improvement of up to 11.6% across four benchmarks: DTU, Blender, TanksAndTemples, and BlendedMVS.

## Background & Motivation
**Background**: Active 3D neural reconstruction selects the Next-Best View (NBV) by evaluating scene uncertainty, thereby reducing data acquisition and computational costs. Existing methods quantify uncertainty using the variance of volume density distribution (ActiveNeRF), variance of color distribution, entropy (ActiveRMAP), or Fisher information (FisherRF).

**Limitations of Prior Work**: (1) Visibility estimation based on volume density is highly inaccurate when the model is underfit—high uncertainty in low-density regions is ignored by the volume rendering process, leading to the exclusion of highly informative candidate views. (2) Existing visibility-aware methods (such as NVF) can only estimate visibility after full convergence in sub-training phases, making them inapplicable to continuous active learning scenarios.

**Key Challenge**: Volume rendering cannot reliably determine the visibility of uncertain regions in the early training phases—uncertainties associated with low-density floaters and underfit surfaces are systematically underestimated. However, continuous active learning requires real-time evaluation of information gain for view selection.

**Goal**: In continuous active learning (where the model continues training while selecting new views), how can the visibility of uncertainties in different regions be accurately estimated? More specifically, how to distinguish "uncertainty on the surface" from "uncertainty in empty space"?

**Key Insight**: Leveraging the SDF values from NeuS to directly derive surface confidence (via SDF sign change detection), bypassing the unreliability of volume density. Surface confidence is stored and robustly updated using a voxel grid to achieve fast and stable visibility reasoning.

**Core Idea**: Detecting surfaces using SDF sign changes instead of relying on volume density, and robustly updating surface confidence via a voxel grid, thereby accurately determining whether uncertainty lies on a visible surface during continuous training.

## Method

### Overall Architecture
Based on the NeuS neural implicit surface network, color prediction is expanded to a Gaussian distribution $c(\mathbf{r}(t)) \sim \mathcal{N}(\bar{c}, \beta^2)$, where the variance $\beta^2$ represents the rendered uncertainty. Concurrently, a voxel grid is maintained to store two types of information: (1) the uncertainty of each voxel (color variance $\beta^2$); (2) the surface confidence of each voxel (SDF sign detection score). Information gain computation is based on surface-guided visibility: when a ray passes through surface voxels, only the entropy of the surface voxels is calculated; when the ray does not encounter any surface, the entropy of all traversed voxels is aggregated.

### Key Designs

1. **Rendered Uncertainty Estimation**:

    - **Function**: Estimates the color uncertainty for every 3D point in the scene.
    - **Mechanism**: Extends the NeuS network to $F_\Theta: (\mathbf{x}, \mathbf{d}) \rightarrow (g(\mathbf{x}), \bar{c}, \beta^2)$, where the color variance $\beta^2$ is independent of the viewing direction. The uncertainty loss is defined as a negative log-likelihood: $\mathcal{L}_u = \frac{1}{M}\sum_i (\frac{\|\bar{\mathcal{C}}(\mathbf{r}_i) - C(\mathbf{r}_i)\|^2}{2\mathcal{B}^2(\mathbf{r}_i)} + \frac{\log \mathcal{B}^2(\mathbf{r}_i)}{2})$
    - **Design Motivation**: By modeling color with a Gaussian distribution, the variance naturally reflects the reconstruction quality of that point—high variance indicates view-inconsistent colors or insufficient training.

2. **SDF-Based Surface Confidence**:

    - **Function**: Detects the presence of a surface in each voxel and robustly updates its confidence.
    - **Mechanism**: Samples three points along the ray (at the voxel center and its adjacent step $1/s$ forward/backward). If the SDF values at the outer two points have opposite signs (their product is negative), a surface is detected. Surface confidence is binary (1 = surface, 0 = no surface), and the update strategy is defined as $\max(\text{previous} \times 0.95, \text{current})$, where a threshold above 0.8 identifies a surface voxel. The decay rate of 0.95 compensates for sampling noise.
    - **Design Motivation**: While volume density is highly unreliable during early training (low density in underfitted regions), SDF sign changes serve as a more stable indicator of surfaces. Progressive updates of the voxel grid allow surface estimation to improve continuously throughout training.

3. **SBV-Guided Information Gain (SBV-Guided IG)**:

    - **Function**: Distinguishes different types of rays according to surface visibility to accurately compute the information gain of candidate views.
    - **Mechanism**: The color entropy is $H(c) = \frac{1}{2}\log(2\pi\beta^2) + \frac{1}{2}$. For rays intersecting surface voxels, IG only accumulates the entropy of the surface voxels; for rays not hitting any surfaces (empty space or blurry regions), IG aggregates the entropy of all traversed voxels. The formula is $G_s(v) = \frac{1}{N}\sum_{r \in \mathcal{R}_v}\sum_{x \in \tilde{\mathcal{X}}_r} H(c(x))$, where $\tilde{\mathcal{X}}_r = \mathcal{X}_r \cap \mathcal{S}$ (when the ray hits a surface) or $\tilde{\mathcal{X}}_r = \mathcal{X}_r$ (when it does not hit a surface).
    - **Design Motivation**: Uncertainty on surfaces directly affects rendering quality and should be prioritized. Although uncertainty in empty regions can be high, it has negligible impact on the reconstructed output; SBV naturally separates these behaviors using surface confidence.

### Loss & Training
The total loss is formulated as $\mathcal{L} = \mathcal{L}_s + \omega\mathcal{L}_u$, where $\mathcal{L}_s$ includes the color L1 loss and Eikonal regularization, and $\omega$ balances the two terms. A two-stage multi-view selection strategy is adopted: during each iteration, 2 or 10 NBVs are selected. A distance-aware strategy ensures diversity—candidate views must maintain a distance larger than a threshold $\tau$ (initially 1.732, with a 0.95 decay rate) from all selected views, and the candidate with the highest IG among the eligible ones is chosen. Frequency regularization is used during training to prevent few-shot overfitting, alongside a warm-up phase to stabilize early training.

## Key Experimental Results

### Main Results (Four Benchmark Datasets)

| Method | DTU-PSNR↑ | Blender-PSNR↑ | TNT-PSNR↑ | BlendedMVS-PSNR↑ |
|------|-----------|--------------|-----------|-----------------|
| Random | 27.69 | 16.24 | 17.17 | 26.21 |
| FVS | 27.09 | 20.07 | 17.94 | 25.49 |
| Entropy | 24.21 | 15.41 | 16.70 | 25.72 |
| ActiveNeRF* | 26.30 | 19.25 | 18.62 | 26.57 |
| FisherRF* | 27.78 | 20.48 | 18.44 | 25.71 |
| **Ours (SBV)** | **28.19** | **21.22** | **20.49** | **26.80** |

### Ablation Study

| Configuration/Method | DTU-PSNR↑ | DTU-Chamfer↓ | ImBView-PSNR↑ |
|-----------|-----------|-------------|--------------|
| SBV (with surface guidance) | **28.19** | **2.002** | **32.23** |
| w/o surface guidance (aggregate all voxels) | 27.18 | 2.251 | - |
| Random | 27.69 | 2.920 | 27.88 |
| ActiveNeRF* | 26.30 | 2.395 | 29.60 |
| FisherRF* | 27.78 | 3.476 | 27.77 |

### Key Findings
- SBV consistently outperforms all baseline methods across four major benchmarks. Notably, on large-scale scenes from TanksAndTemples, it leads the runner-up by 1.87 in PSNR (20.49 vs. 18.62), achieving an 11.6% improvement.
- Surface confidence guidance is critical: removing surface guidance drops the PSNR on DTU from 28.19 to 27.18 and increases the Chamfer distance from 2.002 to 2.251.
- On the ImBView imbalanced-view dataset, SBV exceeds FisherRF* by 4.46 PSNR (32.23 vs. 27.77). This is because FisherRF tends to select extreme end-views, whereas SBV selects diverse views covering various viewpoints.
- NBV selection efficiency: SBV requires only 0.8 seconds, which is significantly faster than ActiveNeRF* (10.8 seconds) and FisherRF* (13.5 seconds), achieving comparable efficiency to the voxel-based Entropy method (0.5 seconds).
- SBV also delivers optimal mesh reconstruction quality (DTU Chamfer distance of 2.002 vs. 2.395 for the runner-up ActiveNeRF*).

## Highlights & Insights
- **SDF Sign Change for Surface Detection**: Bypasses the fundamental limitation of unreliable volume density in early training. Zero-crossings of SDF naturally locate surface positions and provide useful signals even under unconverged states. This idea is generic and transferable to other tasks requiring geometry estimation at intermediate training stages.
- **Robust Voxel Grid Update Strategy**: The design of $\max(\text{previous} \times 0.95, \text{current})$ (slow decay + instant update) ensures that the surface confidence converges monotonically to stability without oscillating under training fluctuations—simple yet highly effective.
- **Guaranteed View Diversity**: The distance threshold decay strategy $\tau \times 0.95$ elegantly balances information gain and view distribution, prioritizing far-distanced high-IG views and gradually relaxing the constraint if no candidates satisfy the requirement.
- **ImBView Dataset Construction**: By intentionally introducing view imbalances (75% normal, 12.5% high-angle, and 12.5% low-angle views), the evaluation clearly exposes bias and preference characteristics of different NBV selection strategies.

## Limitations & Future Work
- **Omission of Inter-view Movement Cost**: In robotic active reconstruction, the locomotion cost between adjacent viewpoints is a critical constraint, which is not incorporated in this work.
- **Architecture Limitations**: Relies on NeuS instead of faster frameworks like NeuS2 or Neuralangelo, because the locality issue of multi-resolution hash encoding degrades the Eikonal loss and surface precision.
- **Only Foregrounds Supported**: Requires object masks and remains unsuitable for indoor scenes or environments requiring depth priors.
- **Background Noise**: A persistent background noise pattern is present in the SBV IG visualization due to grid sampling artifacts from NerfAcc.

## Related Work & Insights
- **vs NVF**: NVF estimates visibility-guided uncertainty only after full convergence in sub-training phases, adopting a staged train-evaluate-select paradigm that prevents its application in continuous active learning. SBV decouples surface confidence updates via a voxel grid, enabling on-the-fly online visibility estimation during training.
- **vs ActiveNeRF**: ActiveNeRF is based on color variance but ignores visibility. In the DTU fruit scenes, it struggles to distinguish between uncertainty on and off the surface, thus failing to choose views that cover complex, occluded regions.
- **vs FisherRF**: FisherRF quantifies information gain using the Fisher Information Matrix (FIM), which is computationally intensive (13.5 seconds vs. 0.8 seconds for SBV). It also displays a severe selection bias on ImBView (favoring extreme views), whereas SBV's surface-guided approach yields more balanced viewpoints.
- **vs Entropy Methods**: Entropy methods use the entropy of occupancy probability but overlook surface occlusions, causing blurred fields behind obstructions to be aggregated into the IG. On DTU, this drops their mesh reconstruction Chamfer distance to 3.644, significantly underperforming compared to SBV's 2.002.
- **Inspiration from Voxel-based Methods**: Traditional voxel occupancy probability approaches (e.g., Isler et al.) update stably but lack compatibility with neural representations. SBV bridges this gap by replacing voxel occupancy probabilities with SDF values.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Using SDF sign changes as a replacement for volume density to detect surfaces is simple and effective, though the overall framework consists of incremental combinations of existing methodologies.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated comprehensively over four major benchmarks and a self-curated dataset. Covers five baseline comparisons under various 1/2/4/10-image settings, with exhaustive computational efficiency tests.
- **Writing Quality**: ⭐⭐⭐⭐ The analysis is highly illustrative (especially the volume density visualization in Fig. 2), the formulation is mathematically rigorous, and the appendix is highly informative.
- **Value**: ⭐⭐⭐⭐ Successfully addresses a practical bottleneck in visibility estimation for continuous active learning. Demonstrates significant gains across multiple benchmarks while maintaining a fast NBV selection speed (0.8s).

<!-- Note written on 2026-04-10, based on paper_cache/AAAI2026/2405.02568.txt full cache -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Uncertainty-driven 3D Gaussian Splatting Active Mapping via Anisotropic Visibility Field](../../CVPR2026/3d_vision/uncertainty-driven_3d_gaussian_splatting_active_mapping_via_anisotropic_visibili.md)
- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](../../ICLR2026/3d_vision/peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)

</div>

<!-- RELATED:END -->

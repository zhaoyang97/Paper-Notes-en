---
title: >-
  [Paper Note] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper identifies Gaussian densification in 3DGS as the key factor responsible for transient-object artifacts, and proposes a delayed Gaussian growth strategy along with a scale-cascaded mask bootstrapping method to decouple densification from dynamic region modeling, achieving state-of-the-art transient-free novel view synthesis across multiple benchmark datasets.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Novel View Synthesis"
  - "Transient Object Removal"
  - "Gaussian Densification"
  - "Robust Reconstruction"
date: 2026-05-08
content_hash: c14fced8399f5053
---

# RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS

**Conference**: ICCV 2025
**arXiv**: [2506.02751](https://arxiv.org/abs/2506.02751)  
**Code**: [https://fcyycf.github.io/RobustSplat/](https://fcyycf.github.io/RobustSplat/) (project page available)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Novel View Synthesis, Transient Object Removal, Gaussian Densification, Robust Reconstruction

## TL;DR

This paper identifies Gaussian densification in 3DGS as the key factor responsible for transient-object artifacts, and proposes a delayed Gaussian growth strategy along with a scale-cascaded mask bootstrapping method to decouple densification from dynamic region modeling, achieving state-of-the-art transient-free novel view synthesis across multiple benchmark datasets.

## Background & Motivation

3D Gaussian Splatting (3DGS) has attracted widespread attention in novel view synthesis and 3D reconstruction owing to its real-time rendering capability and photorealistic quality. However, real-world scenes inevitably contain transient objects (e.g., pedestrians, vehicles) that violate the multi-view consistency assumption and introduce severe artifacts into reconstruction results.

**Three existing paradigms and their limitations**:
- *Category-specific semantic masks*: Limited to predefined categories (e.g., people, cars) and fail to generalize to arbitrary transient objects.
- *Uncertainty-based masks*: Predict motion masks from photometric reconstruction uncertainty, but are often unreliable.
- *Learning-based motion masks*: Use MLPs to predict masks (with DINO features as input), supervised by photometric residuals or feature similarity. However, 3DGS representations are under-optimized in the early training stage, producing overly smooth renderings that lead to inaccurate mask estimation.

**Key Challenge**: Mask learning and Gaussian optimization suffer from a chicken-and-egg problem — low rendering quality in early training causes static regions to be misclassified as dynamic (over-masking) or dynamic regions to go unfiltered (under-masking). Once early densification introduces Gaussians that model transient objects, they are difficult to remove in subsequent stages.

**Key Findings**: Through empirical analysis, the authors make the surprising discovery that **vanilla 3DGS with densification entirely disabled already achieves transient removal performance comparable to SpotLessSplats**. Without densification, the image reconstruction loss can only optimize Gaussian shape and color, while initial Gaussian positions remain stable and do not overfit transient regions. The downside, however, is loss of detail — rendering is overly smooth in regions with sparse initial points.

**Key Insight**: Since "no densification" prevents transient overfitting but sacrifices detail, while "early densification" captures detail but overfits transients, the key lies in **delayed densification** — allowing the static scene structure to stabilize first, then progressively adding Gaussians to refine detail, coupled with more robust mask supervision signals.

## Method

### Overall Architecture

RobustSplat builds upon the standard 3DGS + mask MLP framework and introduces two core designs: (1) a **Delayed Gaussian Growth** strategy that defers the onset of densification, and (2) **Scale-Cascaded Mask Bootstrapping** that progressively improves mask supervision signals from low to high resolution.

### Key Designs

1. **Delayed Gaussian Growth**

    - **Function**: Postpones the start of Gaussian densification from the default 500 iterations to 10K iterations.
    - **Mechanism**: During the first 10K iterations, only the shape, color, and opacity of existing Gaussians are optimized; splitting and cloning are disallowed. This focuses optimization on reconstructing the global structure of the static scene. Experiments demonstrate that later densification onset consistently yields better final results, as early densification causes new Gaussians to overfit transient objects.
    - **Design Motivation**: The authors' diagnostic experiments (Fig. 5a) clearly show that as densification proceeds, vanilla 3DGS PSNR gradually declines (new Gaussians overfit transients), and delaying densification effectively mitigates this issue.

2. **Mask Regularization at Early Stage**

    - **Function**: Encourages the mask MLP to classify all regions as static during early training, gradually allowing dynamic region detection as training progresses.
    - **Mechanism**: An exponentially decaying regularization term $\mathcal{L}_{reg} = e^{-i/\beta_{reg}} \|1 - M_t\|$ is introduced, where $i$ denotes the current iteration. At initialization, this term strongly constrains the mask toward 1 (i.e., all regions treated as static); it decays over iterations, progressively allowing the MLP to learn transient detection.
    - **Design Motivation**: Since delayed densification ensures early optimization involves only the static scene, this regularization complements the delayed growth strategy and prevents biased mask learning from being introduced prematurely.

3. **Scale-Cascaded Mask Bootstrapping**

    - **Function**: Transitions the mask MLP training supervision from low-resolution to high-resolution features and residuals progressively.
    - **Mechanism**:
     - *Before densification onset*: 224×224 low-resolution images are used to extract DINOv2 features, and cosine similarity is computed as the mask supervision signal. Low-resolution features have larger receptive fields, effectively suppressing local noise and being more tolerant of under-reconstructed regions.
     - *After densification onset*: Supervision switches to 504×504 high-resolution images, leveraging finer feature similarity and image residuals for more precise transient region detection.
    - **Design Motivation**: In early training, static regions are under-reconstructed due to sparse point initialization; high-resolution features and residuals erroneously label these regions as dynamic. Low-resolution features naturally smooth out local discrepancies and better capture global consistency (clearly illustrated in Fig. 6).

### Loss & Training

The 3DGS rendering loss follows the standard formulation: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$

The mask MLP optimization loss is: $\mathcal{L}_{MLP} = \lambda_{residual}\mathcal{L}_{residual} + \lambda_{cos}\mathcal{L}_{cos} + \lambda_{reg}\mathcal{L}_{reg}$
- $\mathcal{L}_{residual}$: Robust image residual loss (from SpotLessSplats)
- $\mathcal{L}_{cos}$: $\|M_t - M_{cos}\|$, mask supervision based on DINOv2 feature cosine similarity
- $\mathcal{L}_{reg}$: Exponentially decaying regularization guiding early masks toward full static classification

Hyperparameters: $\lambda_{residual}=0.5$, $\lambda_{cos}=0.5$, $\lambda_{reg}=2.0$, $\beta_{reg}=2000$; densification starts at 10K iterations; total training runs for 30K iterations. The MLP uses DINOv2 ViT-S/14 for feature extraction.

## Key Experimental Results

### Main Results

NeRF On-the-go dataset (6 scenes, low/medium/high occlusion):

| Method | PSNR (Mean) | SSIM (Mean) | LPIPS (Mean) |
|--------|-------------|-------------|--------------|
| 3DGS | 19.09 | 0.717 | 0.248 |
| SpotLessSplats | 22.17 | 0.757 | 0.220 |
| WildGaussians | 22.45 | 0.784 | 0.190 |
| T-3DGS | 22.87 | 0.803 | 0.167 |
| **RobustSplat (Ours)** | **23.22** | **0.818** | **0.149** |

RobustNeRF dataset (4 indoor scenes):

| Method | PSNR (Mean) | SSIM (Mean) | LPIPS (Mean) |
|--------|-------------|-------------|--------------|
| 3DGS | 26.21 | 0.864 | 0.168 |
| SpotLessSplats | 28.58 | 0.875 | 0.162 |
| T-3DGS | 28.25 | 0.888 | 0.149 |
| **RobustSplat (Ours)** | **29.36** | **0.895** | **0.135** |

### Ablation Study

Ablation on the NeRF On-the-go dataset (PSNR as metric):

| Configuration | Mountain | Corner | Patio | Spot | Patio-High | Notes |
|---------------|----------|--------|-------|------|------------|-------|
| 3DGS | 19.21 | 22.65 | 17.04 | 18.54 | 17.04 | Baseline |
| + Mask | 19.81 | 25.05 | 21.23 | 24.75 | 22.19 | With mask learning |
| + Mask + DG | 20.85 | 26.01 | 21.49 | 25.61 | 22.74 | With delayed growth |
| + Mask + MB | 20.78 | 25.52 | 20.88 | 25.25 | 22.11 | With mask bootstrapping |
| Full Model | **21.15** | **26.42** | **21.63** | **26.21** | **22.87** | DG + MB combined |

### Key Findings

- Delayed Gaussian Growth (DG) yields consistent improvements across all scenes, with average PSNR approximately 0.8 dB higher than the mask-only baseline.
- Scale-Cascaded Mask Bootstrapping (MB) alone provides modest gains, but achieves the best performance when combined with DG, demonstrating the complementarity of the two components.
- Improvements are most pronounced in high-occlusion scenes (Spot, Patio-High), with PSNR gains of 7–8 dB over vanilla 3DGS.
- RobustSplat achieves the best results across all 3 metrics on all 6 NeRF On-the-go scenes.

## Highlights & Insights

- The discovery that "disabling densification alone achieves transient removal" is the most important analytical contribution of this paper, revealing the mechanism by which 3DGS overfits transient objects.
- The delayed densification idea is extremely simple — it requires only adjusting the densification start iteration as a hyperparameter, with no additional network components.
- The coarse-to-fine cascade supervision strategy from low to high resolution reflects the classical multi-scale philosophy and is highly effective in addressing the under-reconstruction vs. over-detection trade-off.
- The overall approach does not rely on large-scale foundation models (e.g., SAM, Stable Diffusion), using only DINOv2 ViT-S/14, which ensures high computational efficiency.

## Limitations & Future Work

- The delay duration (10K iterations) and the low-to-high resolution switching point are fixed hyperparameters set manually; adaptive adjustment remains unexplored.
- For very dense transient occlusions (>50%), the delayed strategy may be insufficient to stabilize static structure first.
- The mask MLP relies solely on DINOv2 features; incorporating multi-scale or multi-modal features (e.g., depth priors) may further improve accuracy.
- Validation is currently limited to static scenes with transient distractors; truly dynamic scenes (e.g., 4D reconstruction) would require more sophisticated modeling.

## Related Work & Insights

- **SpotLessSplats**: Predicts masks using two clustering strategies on Stable Diffusion features; achieves strong performance but with high computational cost.
- **WildGaussians**: Predicts uncertainty-based masks from DINO features, but early-stage masks are inaccurate.
- **T-3DGS**: Introduces an unsupervised transient detector and a video object segmentation module; suitable for video but with high complexity.
- **Insight**: Analyzing the root cause of the problem (densification → transient overfitting) is more valuable than directly designing complex modules; simpler solutions tend to be more robust.

## Rating

- Novelty: ⭐⭐⭐⭐ (Core observation is novel and the method is concise and effective, though the overall framework design is relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two standard benchmarks, detailed ablations, comprehensive baseline comparisons)
- Writing Quality: ⭐⭐⭐⭐⭐ (Motivation analysis is clear; the empirical analysis in Figure 2 is highly convincing)
- Value: ⭐⭐⭐⭐ (Provides a simple and efficient solution for 3DGS in in-the-wild scenes)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EDGS: Eliminating Densification for Efficient Convergence of 3DGS](../../CVPR2026/3d_vision/edgs_eliminating_densification_for_efficient_convergence_of_3dgs.md)
- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgs_lm_faster_gaussian_splatting_optimization_with_levenberg_marquardt.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[ICCV 2025\] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)
- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)

</div>

<!-- RELATED:END -->

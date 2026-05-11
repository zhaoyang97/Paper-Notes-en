---
title: >-
  [Paper Note] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes
description: >-
  [CVPR 2026][3D Vision][4D Gaussian Splatting] GP-4DGS integrates variational Gaussian Processes (GP) into 4D Gaussian Splatting, enabling probabilistic motion modeling via spatiotemporal composite kernels and variational inference, while endowing 4DGS with three new capabilities: uncertainty quantification, motion extrapolation, and adaptive motion priors.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D Gaussian Splatting
  - Gaussian Processes
  - Uncertainty Quantification
  - Motion Extrapolation
  - Dynamic Scene Reconstruction
date: 2026-05-08
content_hash: 871387e1d7e8e060
---

# GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes

**Conference**: CVPR 2026  
**arXiv**: [2604.02915](https://arxiv.org/abs/2604.02915)  
**Code**: Unavailable  
**Area**: 3D Vision  
**Keywords**: 4D Gaussian Splatting, Gaussian Processes, Uncertainty Quantification, Motion Extrapolation, Dynamic Scene Reconstruction

## TL;DR
GP-4DGS integrates variational Gaussian Processes (GP) into 4D Gaussian Splatting, enabling probabilistic motion modeling via spatiotemporal composite kernels and variational inference, while endowing 4DGS with three new capabilities: uncertainty quantification, motion extrapolation, and adaptive motion priors.

## Background & Motivation

**Background**: 4D Gaussian Splatting (4DGS) is the dominant paradigm for dynamic novel view synthesis, modeling dynamic scenes by deforming 3D Gaussian primitives over time. Existing methods such as D-3DGS (MLP-based deformation), 4DGS (HexPlane), and STG (polynomial deformation) have achieved strong visual quality.

**Limitations of Prior Work**: (1) Existing methods treat motion as a deterministic optimization problem and impose hand-crafted motion priors (polynomial deformation, rigidity constraints, etc.) that are applied uniformly across all primitives, making them ill-suited for under-observed or occluded regions. (2) No mechanism exists for uncertainty estimation in motion prediction. (3) Temporal extrapolation beyond training frames is not supported.

**Key Challenge**: Fixed deterministic motion priors cannot distinguish between well-observed and sparsely observed regions—over-constraining the former and under-constraining the latter. A mechanism is needed that automatically adapts regularization strength based on observation confidence.

**Goal**: (1) How to provide principled uncertainty quantification for motion prediction in 4DGS? (2) How to learn motion priors from well-observed regions and propagate them to sparse or unobserved regions? (3) How to enable temporal extrapolation beyond training frames?

**Key Insight**: Gaussian Processes are naturally distributions over function spaces, with kernel functions defining the correlation structure between data points. Applying GPs to model deformation fields simultaneously enables adaptive priors, uncertainty quantification, and extrapolation—all as direct consequences of the probabilistic GP formulation, requiring no additional modeling effort.

**Core Idea**: Replace deterministic deformation functions with variational Gaussian Processes, capture geometric and motion correlations via spatiotemporal composite kernels, and naturally obtain uncertainty estimates and extrapolation capability while using the GP posterior mean to guide 4DGS optimization.

## Method

### Overall Architecture
The input is a monocular video. Each Gaussian primitive's 4D coordinate $\mathbf{x}=(\bm{p}, t)$ (canonical 3D position + time) is used as GP input to predict a 9-dimensional deformation vector (3D translation + 6D continuous rotation representation). Computational scalability is achieved via variational inference with inducing points. An alternating GP-GS optimization strategy is adopted: (1) the GP training stage learns motion priors from high-confidence data; (2) the GS optimization stage uses GP predictions as regularization guidance.

### Key Designs

1. **Spatiotemporal Composite Kernel Design**:

    - Function: Captures two fundamentally different correlation structures—spatial geometric smoothness and temporal motion periodicity.
    - Mechanism: The kernel is decomposed into spatial and temporal components: $k_i(\mathbf{x},\mathbf{x}') = k_i^{\text{spatial}}(\bm{p},\bm{p}') + k_i^{\text{temporal}}(\mathbf{x},\mathbf{x}')$. The spatial component uses an anisotropic Matérn kernel (rather than RBF), as Matérn handles discontinuities and is suitable for modeling spatially disconnected objects. The temporal component multiplies a per-axis Matérn kernel with a periodic kernel $k^{\text{periodic}}(t,t') = \sigma^2 \exp(-2\sin^2(\pi|t-t'|/\tau)/\ell^2)$, capturing motion periodicity while preserving spatial locality.
    - Design Motivation: Standard GP kernels assume isotropic correlations, which are fundamentally mismatched to spatiotemporal data where spatial and temporal dimensions exhibit entirely different correlation structures. The periodic kernel provides a strong inductive bias for temporal extrapolation.

2. **Variational Gaussian Process with Inducing Points**:

    - Function: Reduces GP inference complexity from $\mathcal{O}(N^3)$ to $\mathcal{O}(NM^2+M^3)$, making inference tractable for tens of thousands of Gaussian primitives.
    - Mechanism: $M$ inducing points $\mathbf{Z}=\{\mathbf{z}_m\}_{m=1}^M$ ($M \ll N$) are introduced to parameterize the variational posterior $q(\mathbf{u}_i)=\mathcal{N}(\mathbf{m}_i, \mathbf{S}_i)$. Kernel hyperparameters, inducing point locations, and variational parameters are jointly optimized by maximizing the ELBO. At inference, the per-query complexity is only $\mathcal{O}(M)$. Inducing points are initialized by extracting time-series features via Chronos followed by k-means clustering to select representative canonical positions, with uniform temporal sampling along the time axis.
    - Design Motivation: Exact GP inference is entirely infeasible for the typical tens of thousands of Gaussian primitives. Initialization based on time-series features consistently achieves higher ELBO than random or velocity-based baselines.

3. **Alternating GP-GS Optimization Strategy**:

    - Function: Establishes a synergistic feedback loop between the GP and 4DGS.
    - Mechanism: Stage 1 (GP Training): The confidence of each primitive is measured by its accumulated rendering contribution $C_k = \sum_{\mathbf{I}}\sum_{\mathbf{r}} \omega_{k,t}^{\pi}(\mathbf{r})$; the high-confidence subset with $C_k > \tau_C$ is used to train the GP, with spatial coordinate noise injection as regularization. Stage 2 (GS Optimization): The GP posterior mean $\bar{\bm{\mu}}$ serves as a pseudo-guidance signal, regularizing primitives that deviate from GP predictions via $\mathcal{L}_{\text{GP}} = \frac{1}{NT}\sum_{k,t} \delta_{(k,t)} \|\mathbf{y}_{(k,t)} - \bar{\bm{\mu}}_{(k,t)}\|^2$, with the threshold $\tau_\delta$ gradually tightened during training. GP predictions are cached every 2000 steps.
    - Design Motivation: Reliable observations progressively refine the motion prior, which in turn stabilizes reconstruction in under-observed regions—forming a self-reinforcing cycle.

### Loss & Training
The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{GP}}\mathcal{L}_{\text{GP}}$ with $\lambda_{\text{GP}}=0.1$. The reconstruction loss follows SoM, including photometric loss, D-SSIM, flow loss, and smoothness loss. Uncertainty is obtained from the GP posterior via Monte Carlo sampling, as variance cannot be directly propagated through the nonlinear transformation applied to rotations.

## Key Experimental Results

### Main Results (DyCheck)

| Method | mPSNR ↑ | mSSIM ↑ | mLPIPS ↓ |
|--------|---------|---------|----------|
| D-3DGS | 11.92 | 0.49 | 0.66 |
| 4DGS | 13.42 | 0.49 | 0.56 |
| HyperNeRF | 15.99 | 0.59 | 0.51 |
| Gaussian Marbles | 15.84 | 0.54 | 0.57 |
| SoM | 17.09 | 0.65 | 0.39 |
| **GP-4DGS** | **17.38** | **0.65** | **0.37** |

### Motion Extrapolation (PSNR ↑)

| Method | Periodic (5 frames) | Periodic (15 frames) | Non-periodic (5 frames) | Non-periodic (15 frames) |
|--------|--------------------|--------------------|----------------------|----------------------|
| Linear extrapolation | 11.55 | 8.11 | 15.02 | 11.92 |
| **GP-4DGS** | **17.62** | **16.65** | **15.27** | **13.22** |

### Uncertainty Quantification (AUSE-MSE ↓ ×10⁻²)

| Method | Top 20 Frames | Top 40 Frames | All Frames |
|--------|--------------|--------------|-----------|
| Random | 9.76 | 9.30 | 10.98 |
| UA-4DGS | 7.60 | 8.11 | 8.62 |
| **GP-4DGS** | **7.22** | **8.00** | **8.49** |

### Key Findings
- Performance gains are larger on the DyCheck challenging subset (mPSNR 14.56→15.02, mLPIPS 0.53→0.51), confirming that GP priors are most beneficial in sparsely observed regions.
- The advantage in periodic motion extrapolation is substantial (17.62 vs. 11.55), as the periodic kernel accurately captures cyclic dynamics.
- GP guidance effectively regularizes motion trajectories, eliminates noise and oscillations, and produces physically plausible motion patterns.
- Time-series-based inducing point initialization consistently achieves higher ELBO than random and velocity-KNN baselines (average 1.53 vs. 1.10 vs. 1.37).
- Better geometric structural integrity is maintained under extreme viewpoint shifts in DAVIS.

## Highlights & Insights
- The integration of probabilistic modeling into 4DGS is particularly elegant: uncertainty quantification and motion extrapolation arise as natural byproducts of the GP probabilistic formulation, requiring no additional design effort. This "free lunch" approach to acquiring extra capabilities is cleaner than designing dedicated modules.
- The GP-GS alternating optimization is ingeniously designed: confidence-weighted data selection ensures the GP learns from reliable data, while GP predictions in turn guide unreliable regions—resembling a form of bootstrapped learning.
- The spatiotemporal composite kernel has clear physical intuition: spatial Matérn permits discontinuities (across different objects), the temporal periodic kernel encodes motion regularity, and their orthogonal combination is well-motivated.

## Limitations & Future Work
- GP inference still introduces additional computational overhead (GP cache updated every 2000 steps), which may be a bottleneck for real-time applications.
- The periodic kernel assumes periodic motion; extrapolation for complex non-periodic motions remains limited (gains are smaller on non-periodic scenes).
- Evaluation is conducted only on DyCheck (7 scenes) and DAVIS, offering limited scene diversity.
- The approximation quality of the variational GP is bounded by the number of inducing points; extremely high-resolution scenes may require substantially more inducing points.

## Related Work & Insights
- **vs. SoM (STG)**: SoM employs polynomial deformation as a fixed prior, whereas GP-4DGS learns a data-adaptive prior, with a clear advantage on the challenging subset (14.56→15.02).
- **vs. Stochastic GS**: Stochastic GS models Gaussian attributes as random variables but is limited to static scenes; this work is the first to extend probabilistic modeling to 4D dynamic scenes.
- **vs. UA-4DGS**: UA-4DGS attempts uncertainty estimation for dynamic scenes, but GP-4DGS provides better-calibrated estimates through kernel correlation structure (lower AUSE).
- **vs. D-3DGS/4DGS**: These deterministic methods lack constraints in sparsely observed regions; the spatial correlation prior of GPs effectively fills this gap.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Combining GPs with 4DGS is a genuinely novel direction; the spatiotemporal kernel design and alternating optimization strategy demonstrate significant depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage of reconstruction quality, extrapolation, uncertainty, and ablation, though the number of evaluated scenes is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous, method motivation is clear, and figures are well-crafted.
- Value: ⭐⭐⭐⭐ — Opens a new direction for probabilistic modeling in 4DGS; uncertainty quantification has practical value for safety-critical applications such as autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 4C4D: 4 Camera 4D Gaussian Splatting](4c4d_4_camera_4d_gaussian_splatting.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[ICLR 2026\] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](../../ICLR2026/3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)
- [\[CVPR 2026\] VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection](virpro_visual-referred_probabilistic_prompt_learning_for_weakly-supervised_monoc.md)

</div>

<!-- RELATED:END -->

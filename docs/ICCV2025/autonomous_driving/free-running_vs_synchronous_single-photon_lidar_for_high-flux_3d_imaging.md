---
title: >-
  [Paper Note] Free-running vs. Synchronous: Single-Photon Lidar for High-flux 3D Imaging
description: >-
  [ICCV 2025][Autonomous Driving][single-photon lidar] This paper systematically compares free-running and synchronous single-photon lidar (SPL) operating modes for depth imaging under high-flux conditions…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "single-photon lidar"
  - "free-running mode"
  - "maximum likelihood estimation"
  - "depth regularization"
  - "high-flux 3D imaging"
date: 2026-05-08
content_hash: affbc319160121e8
---

# Free-running vs. Synchronous: Single-Photon Lidar for High-flux 3D Imaging

**Conference**: ICCV 2025
**arXiv**: [2507.09386](https://arxiv.org/abs/2507.09386)
**Code**: None
**Area**: Autonomous Driving / 3D Vision
**Keywords**: single-photon lidar, free-running mode, maximum likelihood estimation, depth regularization, high-flux 3D imaging

## TL;DR

This paper systematically compares free-running and synchronous single-photon lidar (SPL) operating modes for depth imaging under high-flux conditions, proposes an efficient joint maximum likelihood estimator and a score-based depth regularization algorithm SSDR, and demonstrates that the free-running mode consistently outperforms the synchronous mode across diverse flux levels and signal-to-background ratios.

## Background & Motivation

**Background**: Single-photon lidar (SPL) leverages picosecond-pulsed lasers and single-photon detectors for high-resolution ranging, with broad applications in topographic mapping, underwater sensing, and autonomous driving. Single-photon avalanche diodes (SPADs) serve as the core detection element; after registering a photon, a SPAD requires a dead time for quenching and reactivation.

**Limitations of Prior Work**: Conventional wisdom holds that SPL should operate under low-flux conditions to minimize dead-time effects (the 5% rule), severely limiting data acquisition rates and signal-to-noise ratios. The synchronous mode (reactivation deferred until the next pulse) has been widely studied due to statistical independence of detections, yet its pile-up effect drastically distorts histograms at high flux, obscuring depth information. The free-running mode (immediate reactivation after dead time) produces less histogram distortion, but the statistical dependence of detection times renders the likelihood function complex and available estimation methods scarce.

**Key Challenge**: SPL performance degrades sharply under high flux — the synchronous mode suffers from pile-up, while the free-running mode lacks efficient estimation algorithms. Existing methods either assume low flux (ignoring dead time) or are designed exclusively for synchronous-mode correction.

**Goal**: (1) Derive joint ML estimators for signal flux, background flux, and depth under both free-running and synchronous modes; (2) demonstrate the superiority of the free-running mode theoretically and experimentally; (3) propose a depth regularization framework incorporating learned priors.

**Key Insight**: Starting from the likelihood function of the free-running mode, the authors show that depth estimation can be reformulated as an efficient matched-filtering operation, while a point cloud score model serves as a 3D structural prior to regularize depth reconstruction.

**Core Idea**: Derive an efficient joint ML estimator for free-running SPL by reformulating depth estimation as a matched-filtering problem, and propose the SSDR algorithm combining point cloud score priors for high-flux 3D reconstruction, ultimately demonstrating that the free-running mode comprehensively outperforms the synchronous mode.

## Method

### Overall Architecture

The system operates at two levels: (1) **pixel-wise ML estimation** — for the detection-time histogram at each scan position, jointly estimate signal flux $S$, background flux $B$, and depth $z$; (2) **multi-pixel regularization** — the SSDR algorithm applies a point cloud score model as a spatial prior to regularize the pixel-wise ML depth estimates, improving 3D reconstruction quality. The input is a set of SPAD detection timestamps; the output is a high-accuracy 3D point cloud.

### Key Designs

1. **Joint Maximum Likelihood Estimator**:

    - **Function**: Simultaneously estimate signal flux, background flux, and depth from SPL histograms.
    - **Mechanism**: Log-likelihood functions are derived for three detector modes (ideal, synchronous, free-running). For the free-running mode, detection times follow a self-exciting point process, and the approximate log-likelihood is $\mathcal{L}_{\text{free}} \approx -n_r \Lambda + \sum_{i=1}^{N} \log \tilde{\lambda}(X_i) + \Phi(X_i + t_d) - \Phi(X_i)$. An alternating maximization strategy is adopted: fixing $z$, L-BFGS-B optimizes $(S, B)$; fixing $(S, B)$, matched filtering optimizes $z$.
    - **Design Motivation**: Prior methods either require pre-calibrated background flux or cannot perform joint estimation. This is the first approach to achieve three-parameter joint estimation, with depth accuracy independent of histogram bin size.

2. **Matched-Filter-Based Depth Estimation**:

    - **Function**: Reformulate ML depth estimation as an efficient cross-correlation operation.
    - **Mechanism**: For the free-running mode, depth estimation is expressed as $\hat{\tau}_{\text{free}} = \arg\max_\tau h(\tau) \otimes u(\tau) + g(\tau) \otimes v(\tau)$, where $h(\tau)$ is the detection-time histogram, $g(\tau)$ is a cyclic shift of $h$, and $u$, $v$ are filters. This can be computed efficiently via FFT. Compared to the Rapp et al. steady-state distribution method, speed is improved by approximately 5000× at 10,000 bins (14.3 s vs. 2.84 ms).
    - **Design Motivation**: Prior free-running estimation methods involve expensive eigenvector computation (solving for the Markov chain stationary distribution), with complexity scaling steeply with bin count, making them unsuitable for real-time systems.

3. **Score-based SPL Depth Regularization (SSDR)**:

    - **Function**: Use a learned point cloud score model as a spatial prior to regularize pixel-wise ML depth estimates.
    - **Mechanism**: A pretrained point cloud denoising model estimates the Stein score $\mathbf{s}_p$ of the 3D point cloud, which is projected onto the detector line-of-sight direction to obtain the depth score $\sigma_p = \langle \mathbf{s}_p, \hat{\mathbf{r}}_p \rangle$. The algorithm is built on a Plug-and-Play Monte Carlo framework: outlier depth values are initialized via median smoothing, followed by iterative updates. A key innovation is hard-thresholding the depth score (retaining only large scores), since prior guidance in low-score regions is unreliable.
    - **Design Motivation**: ML estimation is performed independently per pixel without exploiting spatial structure. Unlike total variation regularization, SSDR directly models priors in 3D point cloud space and is applicable to arbitrary scan geometries, not limited to uniform grids.

### Loss & Training

ML estimation employs alternating maximization of the log-likelihood. The iterative SSDR update comprises three terms: the log-likelihood gradient (data fidelity), the depth score (prior), and a noise term (to escape local optima). Hyperparameters (threshold $\epsilon$, step size $\gamma$, regularization weight $\alpha$) are tuned via Optuna on a held-out dataset.

## Key Experimental Results

### Main Results

| Scene / Mode | RMSE($\hat{S}$) | MAE($\hat{z}$) | Notes |
|---|---|---|---|
| Duck / Sync ML | 2.5×10⁻² | 0.084 m | Pile-up causes depth error in synchronous mode |
| Duck / Free ML | 2.4×10⁻² | 0.022 m | Free-running error significantly lower |
| Duck / Free+SSDR | 2.4×10⁻² | 0.007 m | SSDR further reduces error by 67% |
| Man / Sync ML | 1.8×10⁻² | 0.307 m | Synchronous mode fails at far depth |
| Man / Free ML | 1.8×10⁻² | 0.294 m | Free-running more robust |
| Man / Free+SSDR | 1.8×10⁻² | 0.020 m | SSDR improves by 93% |
| Exp. Dog / Sync | — | 2.137 m | Synchronous mode completely fails |
| Exp. Dog / Free ML | — | 0.136 m | Free-running usable |
| Exp. Dog / Free+SSDR | — | 0.098 m | SSDR further improves by 28% |

### Ablation Study

| SSDR Configuration | Effect on RMSE | Notes |
|---|---|---|
| Full SSDR | Lowest RMSE | All three components cooperate |
| w/o depth score threshold | RMSE increases with iterations | Most impactful; removal prevents convergence |
| w/o median smoothing init. | RMSE decreases then stagnates | Second most important; removal causes local optima |
| w/o iterative noise | RMSE close to full model | Least impactful; helps escape local optima in some scenes |

### Key Findings

- **Free-running mode comprehensively outperforms synchronous mode**: across varying flux levels, signal-to-background ratios (SBR), depths, and dead times, the free-running mode consistently yields lower estimation error. Even with dead time as long as 90 ns (90% of the repetition period), the free-running mode remains superior.
- **Optimal flux is far above the 5% rule**: the optimal flux $\Lambda^*$ for the free-running mode is substantially higher than the conventionally recommended value ($\Lambda < 0.05$), indicating that better performance is achievable at higher photon flux.
- **Synchronous mode depth error depends on target depth** (worse at greater range due to more severe pile-up at long distances), whereas the free-running mode error is depth-independent.
- **The depth score threshold in SSDR is the most critical component** — its removal causes RMSE to increase rather than decrease, demonstrating that indiscriminate use of the prior introduces misleading guidance.

## Highlights & Insights

- **Reformulating depth estimation as matched filtering** is the most elegant technical contribution — the analytical correlation filters are derived by exploiting the structure of the log-likelihood, reducing the algorithmic complexity of the free-running mode from $O(M^3)$ to $O(M \log M)$, which is key to making the free-running mode practical.
- **The conclusion that "free-running comprehensively outperforms synchronous"** overturns conventional wisdom — the synchronous mode was previously considered superior due to statistical independence of detections; the theoretical and experimental evidence in this paper revises that view.
- **Using a 3D point cloud score model as a prior in SSDR** is a forward-looking idea — unlike depth map priors, it is viewpoint-agnostic and generalizable to arbitrary scan geometries. This approach can be transferred to other inverse problems requiring 3D regularization.

## Limitations & Future Work

- The **point cloud score model used in SSDR is trained only on isolated 3D model datasets**, limiting applicability to complex multi-object scenes — results on scenes with objects at different depths are suboptimal.
- **Multi-target / multi-depth scenarios are not addressed** — each pixel is assumed to contain a single target depth, precluding handling of semi-transparent objects or fog scattering.
- The current method **regularizes depth only**; regularization of signal flux and background flux is left as future work.
- Experimental validation uses a **small-scale scan grid (64×64)**, and scalability to large-scale outdoor scenes (e.g., autonomous driving) remains unknown.
- Future work could train larger-scale, more diverse 3D score models, or explore integrating the derived likelihood functions into existing regularization frameworks (TV, ManiPoP, RT3D).

## Related Work & Insights

- **vs. Rapp et al. (2019)**: Also studies free-running SPL, but relies on steady-state distribution computation, requires absolute detection times, and is computationally expensive (5000× slower than this work). The proposed method requires only relative detection-time histograms.
- **vs. Coates correction**: Coates correction is a classical pile-up mitigation method for the synchronous mode, but it estimates photon arrival intensity rather than directly estimating depth. The proposed ML estimator achieves higher accuracy at high flux.
- **vs. ManiPoP**: ManiPoP is a Bayesian SPL regularization method that produces smoother reconstructions but may introduce geometric distortion (e.g., deformation of a duck's beak). SSDR better preserves geometric structure.
- The likelihood function derivations in this paper open a path toward **integrating diffusion model priors into SPL reconstruction**, representing an interesting direction for combining 3D perception with generative models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The joint ML estimator and matched-filtering derivation are solid original contributions; SSDR is the first to apply learned 3D priors to SPL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Simulations comprehensively cover varying flux, SBR, depth, and dead time; validated on real hardware; ablation study is thorough.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and clear; figures are abundant and information-dense; 20 pages with rich supplementary material.
- **Value**: ⭐⭐⭐⭐ Provides important guidance for the SPL community and revises conventional understanding of synchronous vs. free-running modes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs](robust_3d_object_detection_using_probabilistic_point_clouds_from_single-photon_l.md)
- [\[CVPR 2026\] Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals](../../CVPR2026/autonomous_driving/rascene_high-fidelity_3d_scene_imaging_with_mmwave_communication_signals.md)
- [\[ICCV 2025\] DAMap: Distance-aware MapNet for High Quality HD Map Construction](damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[ICCV 2025\] DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic](duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)

</div>

<!-- RELATED:END -->

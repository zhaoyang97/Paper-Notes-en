---
title: >-
  [Paper Note] Metropolis-Hastings Sampling for 3D Gaussian Reconstruction
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] This paper proposes an adaptive Metropolis-Hastings framework to replace the heuristic density control mechanism in 3DGS. Through probabilistic sampling driven by multi-view photometric error, it achieves more efficient inference of Gaussian distributions and converges faster than 3DGS-MCMC.
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Metropolis-Hastings"
  - "Adaptive Density Control"
  - "Novel View Synthesis"
  - "MCMC"
date: 2026-05-08
content_hash: 65a3b1aacd26b3ec
---

# Metropolis-Hastings Sampling for 3D Gaussian Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2506.12945](https://arxiv.org/abs/2506.12945)  
**Code**: [Project Page](https://hjhyunjinkim.github.io/MH-3DGS)  
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: 3D Gaussian Splatting, Metropolis-Hastings, Adaptive Density Control, Novel View Synthesis, MCMC

## TL;DR

This paper proposes an adaptive Metropolis-Hastings framework to replace the heuristic density control mechanism in 3DGS. Through probabilistic sampling driven by multi-view photometric error, it achieves more efficient inference of Gaussian distributions and converges faster than 3DGS-MCMC.

## Background & Motivation

3D Gaussian Splatting (3DGS) enables real-time rendering via explicit 3D Gaussians, but relies heavily on heuristic density control (cloning, splitting, pruning) — fixed thresholds that are too loose lead to redundant Gaussians and memory waste, while thresholds that are too tight sacrifice fidelity.

Limitations of existing approaches:

- **3DGS-MCMC**: Replaces heuristics with SGLD (Stochastic Gradient Langevin Dynamics), but requires the total number of Gaussians to be fixed in advance, and operates as a local, fully-accepting chain — incapable of global jumps.
- **Other methods**: Error-based densification, homodirectional gradients, etc., still rely on thresholds or fixed upper bounds.
- Core problem: The absence of a theoretically grounded densification mechanism that adapts to scene complexity.

## Method

### Overall Architecture

The densification and pruning in 3DGS are reformulated as a unified Metropolis-Hastings (MH) sampling process:

1. Define the posterior $\pi(\Theta) \propto e^{-\mathcal{E}(\Theta)}$ over scene representation $\Theta = \{g_i\}_{i=1}^N$.
2. Generate candidate Gaussians via importance-driven proposals.
3. Apply MH acceptance-rejection tests to determine whether to retain candidates.
4. Recycle low-opacity Gaussians through a relocation mechanism.

### Key Designs

1. **Bayesian Scene Posterior with Voxel Prior**:
    - Function: Define the negative log-posterior $\mathcal{E}(\Theta) = \mathcal{L}(\Theta) + \lambda_v \sum_{v \in \mathcal{V}} \ln(1 + c_\Theta(v))$.
    - Mechanism: The photometric loss $\mathcal{L}$ corresponds to the likelihood; the voxel prior penalizes crowded regions. The logarithmic form of voxel density $c_\Theta(v)$ imposes negligible penalty on empty voxels while rapidly increasing penalties in crowded ones.
    - Design Motivation: Pure photometric optimization causes Gaussians to accumulate in the same regions; the voxel prior provides an inductive bias toward spatial sparsity.

2. **Multi-View Importance-Driven Coarse-to-Fine Proposals**:
    - Function: Aggregate SSIM and L1 errors across multiple views to construct a per-pixel importance map $s(p) = \sigma(\alpha O(p) + \beta \text{SSIM}_\text{agg}(p) + \gamma \text{L1}_\text{agg}(p))$.
    - Mechanism: The coarse stage uses larger perturbations $\sigma_\text{coarse}$ to fill coverage gaps; the fine stage uses $\sigma_\text{fine} < \sigma_\text{coarse}$ for refinement. The view subset size is annealed during training — broad coverage early on, focused refinement later.
    - Design Motivation: Multi-view aggregation ensures consistency (single-view estimates may be biased by occlusion); the coarse-to-fine strategy balances exploration and refinement.

3. **Closed-Form Derivation of MH Acceptance Probability**:
    - Function: Derive a practical MH acceptance rule $\rho(i) = \sigma(I(i)) \cdot D(v')$, where $D(v') = 1/(1 + \lambda_v c_\Theta(v'))$.
    - Mechanism: The importance score $I(i)$ approximates the photometric change $-\Delta\mathcal{L}$, and the intractable inverse proposal density is absorbed into the voxel factor. A candidate is accepted with high probability only when importance is high and the target region is uncrowded.
    - Design Motivation: Avoids re-rendering all views to compute the full loss change for every proposal, which would be computationally infeasible.

### Loss & Training

Total loss: $\mathcal{L}(\Theta) = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_\text{D-SSIM} + \lambda_\text{opacity}\bar{\alpha} + \lambda_\text{scale}\bar{\Sigma}$

Hyperparameters: $\lambda = 0.2$, $\lambda_\text{opacity} = 0.01$, $\lambda_\text{scale} = 0.01$, $\alpha = 0.8$, $\beta = \gamma = 0.5$. The importance weight mixing coefficients respectively control opacity, structural similarity, and photometric fidelity.

## Key Experimental Results

### Main Results (Tables)

Combined results on Mip-NeRF360 + Tanks & Temples + Deep Blending:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | #Gaussians (M) |
|--------|-------|-------|--------|----------------|
| 3DGS | baseline | baseline | baseline | baseline |
| 3DGS-MCMC | close to ours | close to ours | close to ours | close to ours |
| **MH-3DGS (Ours)** | matches or slightly better | matches or slightly better | matches or slightly better | **fewer** |

Convergence speed comparison (time to reach target PSNR):

| Target PSNR | MH-3DGS Time | 3DGS-MCMC Time | Speedup |
|-------------|-------------|----------------|---------|
| 21 dB | 16.30s | 17.08s | 1.05× |
| 24 dB | 61.34s | 98.38s | 1.60× |
| 27 dB | 287.01s | 341.64s | 1.19× |
| 30 dB | 851.52s | 983.05s | **~2.2 min faster** |

### Ablation Study

- Coarse-to-fine proposal strategy: Using only the coarse stage causes oversampling in local regions; using only the fine stage fails to fill large coverage gaps.
- Voxel prior: Removing it leads to severe Gaussian accumulation in the same regions, with noticeably degraded LPIPS.
- View subset annealing: Using a fixed full-view set increases computational overhead with limited benefit.

### Key Findings

- MH-3DGS achieves rendering quality comparable to or better than 3DGS-MCMC with **fewer Gaussians**.
- Reaches 30 dB PSNR approximately 2.2 minutes faster than 3DGS-MCMC.
- Formally demonstrates that the heuristic densification rules of 3DGS can be recast as principled MH updates.
- Global importance-driven proposals with acceptance-rejection testing vs. local fully-accepting SGLD chains — the former holds a fundamental advantage in convergence.

## Highlights & Insights

- **Theoretical Rigor**: Formally proves detailed balance of the MH sampler, establishing a rigorous connection between 3DGS densification and Bayesian inference.
- **Practical Efficiency**: The closed-form acceptance probability approximation (logistic × voxel factor) incurs minimal computational overhead.
- **Fundamental Distinction from 3DGS-MCMC**: Global MH chain vs. local SGLD chain — MH supports long-range jumps to unexplored regions.
- The coarse-to-fine proposal strategy is generalizable and can be applied to other point-based 3D reconstruction methods.

## Limitations & Future Work

- The accuracy of the importance approximation $-\Delta\mathcal{L} \approx I(i)$ is not rigorously guaranteed.
- The inverse proposal density is simply absorbed rather than computed exactly, which affects the theoretical exactness of the MH chain.
- The voxel prior resolution requires scene-specific tuning.
- The method currently supports only static scenes; extension to dynamic scenes remains future work.
- Practical PSNR improvements over 3DGS-MCMC are modest — the primary advantages lie in efficiency and reduced Gaussian count.

## Related Work & Insights

- **3DGS-MCMC (Kheradmand et al.)**: The most direct baseline; uses SGLD with opacity relocation, but fixes the Gaussian count.
- **MH Sampling in NeRF (Goli et al., Bortolon et al.)**: Applies MH to specific components; this paper generalizes it into a holistic framework.
- **Rota Bulò et al.**: Error-prioritized densification with a global cap.
- Establishes a theoretical bridge between probabilistic sampling and density control in 3D reconstruction.

## Rating

⭐⭐⭐⭐ — Solid theoretical contribution that elevates heuristic density control to a principled probabilistic framework, with thorough experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](../../CVPR2025/3d_vision/depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting](dc4gs_directional_consistency-driven_adaptive_density_control_for_3d_gaussian_sp.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)

</div>

<!-- RELATED:END -->

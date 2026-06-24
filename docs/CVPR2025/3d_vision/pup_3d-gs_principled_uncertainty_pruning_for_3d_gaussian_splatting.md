---
title: >-
  [Paper Note] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Proposes PUP 3D-GS, a principled 3D Gaussian Splatting pruning method based on the Fisher Information Matrix, which achieves a 90% Gaussian pruning rate through second-order sensitivity scoring of spatial parameters (position + scale) while maintaining better visual quality and foreground details than existing heuristic methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "model pruning"
  - "Fisher Information Matrix"
  - "uncertainty estimation"
  - "scene compression"
date: 2026-05-08
content_hash: 44a2f46ebe97ff30
---

# PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2406.10219](https://arxiv.org/abs/2406.10219)  
**Code**: [Project Page](https://pup3dgs.github.io)  
**Area**: 3D Vision / Novel View Synthesis  
**Keywords**: 3D Gaussian Splatting, model pruning, Fisher Information Matrix, uncertainty estimation, scene compression

## TL;DR

Proposes PUP 3D-GS, a principled 3D Gaussian Splatting pruning method based on the Fisher Information Matrix, which achieves a 90% Gaussian pruning rate through second-order sensitivity scoring of spatial parameters (position + scale) while maintaining better visual quality and foreground details than existing heuristic methods.

## Background & Motivation

3D Gaussian Splatting (3D-GS) has achieved great success in novel view synthesis, enabling real-time rendering and high reconstruction quality. However, complex scenes usually contain millions of Gaussians, leading to high storage and memory requirements, which limits deployment on resource-constrained devices.

Existing 3DGS pruning methods (such as LightGaussian, EAGLES, Compact-3DGS) rely on **heuristic** criteria to decide which Gaussians to prune—such as hand-designed rules based on opacity, size, transmittance, etc. These heuristics severely degrade visual fidelity, especially foreground details, at high compression rates (>80%).

The core insight of this paper is that 3D scene reconstruction is inherently an **under-constrained problem**—images taken from limited views cannot uniquely determine the 3D position and scale of each Gaussian (a distant large Gaussian and a near small Gaussian can be equivalent in projection space). This **spatial uncertainty** can be accurately quantified using the Fisher Information Matrix. Gaussians with high uncertainty contribute little to scene reconstruction and are prioritized for pruning; Gaussians with low uncertainty (high sensitivity) are critical to reconstruction and should be retained.

## Method

### Overall Architecture

PUP 3D-GS is a post-hoc pruning pipeline that can be applied to **any** pre-trained 3D-GS model: (1) compute the spatial sensitivity score for each Gaussian on the transitioned model; (2) remove the Gaussians with the lowest scores; (3) fine-tune the remaining Gaussians; (4) optionally repeat steps 1–3 for multi-round pruning.

### Key Designs

1. **Sensitivity Scoring based on Fisher Information Matrix**:
    - **Function**: Computes a principled importance score for each Gaussian.
    - **Mechanism**: Starting from the $L_2$ reconstruction error, the second-order Hessian with respect to Gaussian parameters is calculated. On a converged model, the residual term approaches zero, and the Hessian is approximated as the Fisher Information Matrix $\nabla^2_\mathcal{G} L_2 \approx \sum_\phi \nabla_\mathcal{G} I_\mathcal{G}(\phi) \nabla_\mathcal{G} I_\mathcal{G}(\phi)^T$. Taking a block-diagonal approximation yields the independent Hessian $\mathbf{H}_i$ for each Gaussian, and the final sensitivity score is its log-determinant $U_i = \log|\nabla_{x_i, s_i} I_\mathcal{G} \nabla_{x_i, s_i} I_\mathcal{G}^T|$. This score measures the impact of perturbing the Gaussian's spatial parameters on the reconstruction error—higher values indicate more important Gaussians.
    - **Design Motivation**: FisherRF uses only a diagonal approximation and color parameters, which is insufficiently accurate; the block-diagonal approximation coupled with spatial parameters (position + scale) captures the geometric invariance in 3D projection.

2. **Utilizing Spatial Parameters Only (Position + Scale)**:
    - **Function**: Reduces the Hessian computation from all parameters to 6 dimensions.
    - **Mechanism**: Through ablation studies, it is found that effective sensitivity scores can be obtained using only $x_i \in \mathbb{R}^3$ (position) and $s_i \in \mathbb{R}^3$ (scale). Rotation parameters are unnecessary because rotation does not cause 3D geometric changes under projection invariance. Color and spherical harmonics parameters can also be omitted. Consequently, only a $6 \times 6$ block Hessian needs to be computed for each Gaussian.
    - **Design Motivation**: Reduces computational cost while retaining the most informative parameters based on projective geometry principles.

3. **Patch-wise Computation + Multi-round Pruning Pipeline**:
    - **Function**: Lowers computational overhead and enhances final quality.
    - **Mechanism**: (i) Images are downsampled to a $4\times4$ patch resolution to compute the Fisher approximation, which is then summed over all views to obtain the scene-level Hessian—highly correlated with pixel-wise computation but much faster. (ii) Multi-round pruning: Phase 1 prunes 88.5% and fine-tunes for 15K iterations; Phase 2 prunes ~13% of the remaining and fine-tunes for 15K iterations, reaching a total 90% pruning rate. This yields better quality than an equivalent single-round 90% pruning.
    - **Design Motivation**: Large-scale pruning in a single round is too aggressive. Multi-round progressive pruning allows the model to readapt between rounds.

### Loss & Training

- **Post-Pruning Fine-Tuning**: Uses the original 3DGS loss $L = \|I_\mathcal{G}(\phi) - I_{gt}\|_1 + L_{SSIM}(I_\mathcal{G}(\phi), I_{gt})$.
- **No Densification**: Gaussian densification is disabled during the fine-tuning phase to prevent the count from bouncing back.
- **Two-Round Pipeline**: Round 1 retains 11.5% (15K iteration fine-tuning) $\rightarrow$ Round 2 retains 10% (15K iteration fine-tuning).

## Key Experimental Results

### Main Results

Mip-NeRF 360 Dataset (90% Pruning Rate):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Rendering FPS↑ |
|------|-------|-------|--------|----------|
| 3D-GS (Original) | 27.49 | 0.815 | 0.214 | 134 |
| LightGaussian | 25.30 | 0.756 | 0.280 | — |
| **PUP 3D-GS** | **26.56** | **0.792** | **0.230** | **477** |

Deep Blending Dataset Example (90% Pruning):

| Scene | Original GS Count | Pruned GS Count | Original FPS | Pruned FPS | Gain (Speedup) |
|------|-----------|------------|---------|-----------|-------|
| Playroom | 2.65M | 0.265M | 76.65 | 318.06 | 4.15× |

### Ablation Study

| Parameter Selection | PSNR | Description |
|---------|------|------|
| Position + Scale (Ours) | **26.56** | Best, captures projection invariance |
| All Parameters | 26.41 | Adding rotation/color degrades score quality |
| Position Only | 26.32 | Lacks scale information |
| Color Only (e.g., FisherRF) | 25.89 | Color does not reflect spatial uncertainty |

### Key Findings

- Under a 90% pruning rate, PUP 3D-GS outperforms LightGaussian by approximately 1.26 dB in PSNR.
- Average rendering speed increases by 3.56×, and foreground details are significantly better preserved.
- Multi-round pruning improves performance by about 0.3–0.5 dB compared to equivalent single-round pruning.
- It can be seamlessly combined with orthogonal compression technologies such as Vectree quantization.

## Highlights & Insights

- **From Heuristics to Principled Pruning**: Utilizing the Fisher Information Matrix (second-order optimization theory) to replace hand-designed rules provides a mathematical guarantee of optimality—removing the Gaussians that have the least impact on reconstruction error. This methodological advancement is more valuable than simple performance gains.
- **Key Insight on Spatial Parameters**: The discovery that position + scale are the most effective pruning parameters (rather than color) perfectly aligns with the intuition of projective geometric invariance in 3D reconstruction—where uncertainty primarily arises from ambiguity along the depth direction.

## Limitations & Future Work

- Patch-wise computation still requires traversing all training views, which incurs considerable computational overhead on large-scale scenes.
- Fine-tuning after pruning requires an additional 15K–30K training iterations.
- Joint optimization with adaptive densification strategies remains unexplored.
- The optimal number of rounds and pruning ratios for multi-round pruning still need to be selected manually.

## Related Work & Insights

- **vs LightGaussian**: LightGaussian uses global significance scores (heuristically combining opacity, coverage, etc.), heavily losing foreground details at high compression rates; the Fisher score in PUP 3D-GS is more principled and yields better performance.
- **vs FisherRF**: FisherRF also computes Fisher information but uses only a diagonal approximation and color parameters for active view selection; PUP 3D-GS uses a block-diagonal approximation and spatial parameters for pruning, differing in both method and objective.
- **vs BayesRays**: BayesRays quantifies uncertainty on NeRF using Fisher information, requiring a hypothetical perturbation field; PUP 3D-GS calculates it directly on 3DGS and is much more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces second-order optimization theory to 3DGS pruning, ensuring methodological principledness.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple compression rates across three datasets, accompanied by detailed parameter selection ablations.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, motivating the pruning logic naturally from uncertainty.
- Value: ⭐⭐⭐⭐ 90% pruning rate with superior quality, highly practical, and orthogonal to other compression techniques.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2025\] POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality](pop-gs_next_best_view_in_3d-gaussian_splatting_with_p-optimality.md)
- [\[CVPR 2025\] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)
- [\[CVPR 2025\] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)

</div>

<!-- RELATED:END -->

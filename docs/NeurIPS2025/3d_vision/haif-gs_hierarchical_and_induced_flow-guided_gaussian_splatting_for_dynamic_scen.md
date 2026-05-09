---
title: >-
  [Paper Note] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene
description: >-
  [NeurIPS 2025][3D Vision][Dynamic scene reconstruction] HAIF-GS proposes a dynamic 3DGS framework built upon sparse motion anchors, achieving state-of-the-art rendering quality on the NeRF-DS and D-NeRF benchmarks via three key mechanisms: an anchor filter that separates dynamic and static regions, a self-supervised induced scene flow that guides temporally consistent deformation, and hierarchical anchor densification that captures fine-grained non-rigid motion.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Dynamic scene reconstruction
  - 3D Gaussian splatting
  - motion anchors
  - scene flow
  - hierarchical deformation
date: 2026-05-08
content_hash: 83991dfd28729815
---

# HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene

**Conference**: NeurIPS 2025
**arXiv**: [2506.09518](https://arxiv.org/abs/2506.09518)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: Dynamic scene reconstruction, 3D Gaussian splatting, motion anchors, scene flow, hierarchical deformation

## TL;DR
HAIF-GS proposes a dynamic 3DGS framework built upon sparse motion anchors, achieving state-of-the-art rendering quality on the NeRF-DS and D-NeRF benchmarks via three key mechanisms: an anchor filter that separates dynamic and static regions, a self-supervised induced scene flow that guides temporally consistent deformation, and hierarchical anchor densification that captures fine-grained non-rigid motion.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) enables real-time, high-quality rendering of static scenes. The dominant paradigm for extending it to dynamic scenes is to learn a deformation field that predicts time-varying Gaussian attributes (e.g., position, rotation).

**Limitations of Prior Work**:
   - **Redundant Gaussian updates**: Methods that predict per-Gaussian deformations (e.g., Deformable 3DGS) must query and update a large number of Gaussians at every timestep, incurring severe computational redundancy.
   - **Insufficient motion supervision**: Training relies solely on image reconstruction loss, lacking explicit motion guidance or structural constraints, which leads to temporal inconsistency and artifacts.
   - **Weak non-rigid modeling capacity**: Sparse control-point methods (e.g., SC-GS) improve efficiency by driving Gaussians via interpolation, but simple MLP deformation fields struggle to capture articulated motion and spatially varying complex deformations.

**Key Challenge**: A fundamental trade-off exists between efficiency (sparse representation) and expressiveness (fine-grained deformation). Sparse control points are efficient but insufficiently expressive, whereas dense per-Gaussian deformation is expressive but redundant.

**Goal**: (1) How to efficiently focus deformation modeling only on regions that truly require it? (2) How to improve temporal consistency without external optical flow supervision? (3) How to capture fine-grained non-rigid deformation while maintaining sparsity?

**Key Insight**: Sparse motion anchors serve as the core deformation unit. A dynamic–static decomposition filters out unnecessary updates; self-supervised scene flow provides implicit motion guidance; and hierarchical densification increases anchor resolution in regions of complex motion.

**Core Idea**: Sparse anchors + dynamic filtering + induced flow guidance + hierarchical densification = efficient and fine-grained dynamic scene deformation modeling.

## Method

### Overall Architecture
The input is a monocular video sequence with known camera poses. A sparse set of motion anchors $\mathcal{A} = \{(x_i, \rho_i)\}_{i=1}^{M}$ is initialized in canonical space via farthest-point sampling. The pipeline proceeds as follows: (1) an anchor filter predicts dynamic confidence scores to select motion-relevant anchors; (2) an induced flow-guided deformation module aggregates multi-frame features to predict spatiotemporal anchor transformations; (3) hierarchical anchor densification adds anchors in regions of complex motion and propagates transformations layer by layer; (4) anchor transformations are transferred to individual Gaussians via spatial interpolation to render the final image.

### Key Designs

1. **Sparse Motion Anchors and Dynamic–Static Decomposition**:

    - **Function**: Replace per-Gaussian deformation with a small number of anchors, and distinguish dynamic from static regions to avoid redundant computation.
    - **Mechanism**: Anchors are initialized via farthest-point sampling; each anchor has a position $x_i$ and an influence radius $\rho_i$. A Gaussian $g_j$ aggregates transformations from its $K$ nearest anchors with normalized Gaussian kernel weights: $\omega_{ij} = \frac{\exp(-\|\mu_j - x_i\|^2 / \rho_i^2)}{\sum_{x_k \in \mathcal{A}_j} \exp(-\|\mu_j - x_k\|^2 / \rho_k^2)}$. A lightweight anchor filter MLP predicts a dynamic confidence score $\alpha_i \in [0,1]$ from positional and temporal encodings. Training proceeds in two stages: soft-weight modulation to allow gradient flow, followed by hard thresholding to retain only dynamic anchors.
    - **Design Motivation**: Static backgrounds require no deformation modeling. Explicit decomposition eliminates futile computation on static regions and concentrates deformation learning on areas with genuine motion.

2. **Induced Flow-Guided Deformation (IFGD)**:

    - **Function**: Implicitly induce scene flow through multi-frame feature aggregation—without external optical flow labels—to enhance temporal consistency.
    - **Mechanism**:
        - An **induced flow MLP** predicts forward and backward scene flows $(\bm{F}^{t-1}, \bm{F}^{t+1}) = \text{MLP}_{\text{flow}}(x, t)$.
        - Three temporally offset queries are constructed: $q_{t-1} = (x + \bm{F}^{t-1}, t-1)$, $q_t = (x, t)$, $q_{t+1} = (x + \bm{F}^{t+1}, t+1)$.
        - A **deformation MLP** maps each query to a feature embedding; these are fused with temporal weighting to obtain a temporally consistent feature: $\tilde{\bm{f}^t} = \lambda \bm{f}^{t-1} + (1-2\lambda) \bm{f}^t + \lambda \bm{f}^{t+1}$ (with $\lambda=0.25$).
        - Multiple independent transformation heads predict rotation $\Delta R_i^t$ and translation $\Delta T_i^t$ from the fused feature.
    - **Design Motivation**: Training with reconstruction loss alone lacks structural motion constraints. Through joint optimization, the induced flow naturally converges to coherent motion patterns, providing implicit motion supervision without requiring external flow labels.

3. **Hierarchical Anchor Densification (HAD)**:

    - **Function**: Adaptively increase anchor density in regions of complex motion, forming a multi-scale hierarchical structure that captures fine-grained deformation.
    - **Mechanism**: The translation variance of each anchor is computed over $N_t=16$ randomly sampled timesteps: $var(a_i) = \frac{1}{N_t} \sum_t \|\Delta T_i^t - \overline{\Delta T_i}\|_2^2$. Anchors whose variance exceeds threshold $\tau$ are marked for refinement and duplicated with small positional offsets to produce child anchors. Child anchors encode their own position, timestep, and parent translation to enable cross-level motion propagation. All levels share the deformation MLP but use level-specific feature extractors.
    - **Design Motivation**: Globally sparse anchors cannot capture fine-grained non-rigid motion such as finger articulation. The hierarchical structure increases resolution only where needed, keeping the computational overhead manageable.

### Loss & Training
The total loss is:
$$\mathcal{L} = \lambda \mathcal{L}_1 + (1-\lambda) \mathcal{L}_{\text{D-SSIM}} + \lambda_1 \mathcal{L}_{\text{cycle}} + \lambda_2 \mathcal{L}_{\text{entropy}} + \lambda_3 \mathcal{L}_{\text{sparsity}}$$

- **Cycle-consistency loss** $\mathcal{L}_{\text{cycle}}$: the forward flow followed by the backward flow should return to the original position, encouraging bidirectionally consistent motion patterns.
- **Sparsity loss** $\mathcal{L}_{\text{sparsity}} = \mathbb{E}_i[\alpha_i]$: encourages the use of as few dynamic anchors as possible (Occam's razor principle).
- **Entropy loss** $\mathcal{L}_{\text{entropy}} = \mathbb{E}_i[\alpha_i(1-\alpha_i)]$: penalizes confidence scores near 0.5 to promote binary decisions.

Hyperparameters: $\lambda=0.8$, $\lambda_1=0.01$, $\lambda_2=0.2$, $\lambda_3=0.5$.

## Key Experimental Results

### Main Results — NeRF-DS Dataset (Real-world, 480×270)

| Scene | Method | PSNR↑ | MS-SSIM↑ | LPIPS↓ |
|-------|--------|-------|----------|--------|
| Mean | 3DGS | 20.29 | 0.7816 | 0.2920 |
| Mean | HyperNeRF | 23.45 | 0.8488 | 0.1990 |
| Mean | 4DGS | 24.18 | 0.8845 | 0.1405 |
| Mean | SC-GS | 24.05 | 0.8848 | 0.1439 |
| Mean | **HAIF-GS** | **24.63** | **0.9014** | **0.1342** |

Averaged across 7 scenes, HAIF-GS surpasses 4DGS by +0.45 dB in PSNR, improves MS-SSIM from 0.8845 to 0.9014, and reduces LPIPS from 0.1405 to 0.1342.

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Full model (HAIF-GS) | 24.63 PSNR / 0.9014 MS-SSIM | Complete model |
| w/o anchor filter | PSNR drops | Redundant deformation perturbation in static regions |
| w/o induced flow guidance | Temporal consistency drops | Loss of implicit motion supervision |
| w/o hierarchical densification | Quality drops in fine-motion regions | Detail loss in high-dynamic regions (e.g., fingers, tongue) |
| w/o cycle-consistency loss | Poor flow field consistency | Inconsistent forward/backward predictions |

### Key Findings
- HAIF-GS outperforms SC-GS and 4DGS on all NeRF-DS scenes, with especially pronounced improvements on scenes containing specular reflection and fine motion (e.g., *As*, *Bell*).
- Hierarchical anchor densification contributes most in regions of non-rigid deformation (e.g., hand motions in D-NeRF).
- Although the induced flow module requires no external flow labels, it learns coherent motion representations through cycle-consistency self-supervision.

## Highlights & Insights
- **Self-supervised design of induced flow**: Scene flow is "induced" through multi-frame feature aggregation without pre-extracted optical flow or additional annotations, yet converges naturally to consistent motion patterns under joint optimization. This design is transferable to any deformation modeling task that requires motion regularization.
- **Elegant implementation of dynamic–static decomposition**: A lightweight MLP for confidence prediction combined with a two-stage (soft → hard) training strategy effectively focuses computation on dynamic regions with minimal complexity.
- **On-demand hierarchical densification**: Motion-variance-triggered densification adds resolution only in locally complex regions, keeping computational overhead under control.

## Limitations & Future Work
- The method assumes known camera poses; robustness to pose estimation errors has not been verified.
- The number of hierarchical anchor levels and the densification threshold require manual tuning; adaptive strategies may be preferable.
- Performance under extreme topological changes (e.g., object appearance/disappearance) remains unknown.
- The induced flow exploits only three frames ($t-1, t, t+1$); longer-range temporal information may further improve consistency.
- Evaluation is limited to D-NeRF (synthetic) and NeRF-DS (real but small-scale); validation on large-scale real dynamic scenes is lacking.

## Related Work & Insights
- **vs. Deformable 3DGS**: Per-Gaussian MLP deformation fields are computationally redundant; HAIF-GS substantially improves efficiency through sparse anchor interpolation.
- **vs. SC-GS**: Both methods employ sparse control points, but SC-GS lacks motion supervision and a hierarchical structure, leading to inferior performance in regions of fine-grained deformation. HAIF-GS's induced flow and hierarchical densification address these shortcomings.
- **vs. 4DGS**: 4DGS uses plane-based encoding to model time-varying scenes; HAIF-GS achieves superior quantitative results through its explicit anchor structure and hierarchical design.

## Rating
- Novelty: ⭐⭐⭐⭐ The induced flow self-supervision and on-demand hierarchical densification are both novel contributions, though the overall framework is a natural extension of sparse control-point methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons and ablations on standard benchmarks, but large-scale scene experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly with rigorous formulations, though certain modules (e.g., hierarchical propagation details) could be elaborated further.
- Value: ⭐⭐⭐⭐ Advances both efficiency and accuracy for dynamic 3DGS; individual components exhibit strong reusability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[NeurIPS 2025\] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment](va-gs_enhancing_the_geometric_representation_of_gaussian_splatting_via_view_alig.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](../../ICCV2025/3d_vision/beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](../../CVPR2026/3d_vision/ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[NeurIPS 2025\] DGH: Dynamic Gaussian Hair](dgh_dynamic_gaussian_hair.md)

</div>

<!-- RELATED:END -->

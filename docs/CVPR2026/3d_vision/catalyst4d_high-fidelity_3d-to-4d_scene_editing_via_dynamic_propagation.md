---
title: >-
  [Paper Note] Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation
description: >-
  [CVPR 2026][3D Vision][4D scene editing] This paper proposes Catalyst4D, a framework that propagates high-quality 3D static editing results into 4D dynamic Gaussian scenes through two modules — Anchor-based Motion Guidance (AMG) and Color Uncertainty-guided Appearance Refinement (CUAR) — achieving spatiotemporally consistent, high-fidelity dynamic scene editing.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D scene editing
  - 3DGS
  - dynamic propagation
  - anchor-based motion guidance
  - optimal transport
date: 2026-05-08
content_hash: 448399eb6cbb44e6
---

# Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation

**Conference**: CVPR 2026
**arXiv**: [2603.12766](https://arxiv.org/abs/2603.12766)
**Code**: None
**Area**: 3D Vision
**Keywords**: 4D scene editing, 3DGS, dynamic propagation, anchor-based motion guidance, optimal transport

## TL;DR
This paper proposes Catalyst4D, a framework that propagates high-quality 3D static editing results into 4D dynamic Gaussian scenes through two modules — Anchor-based Motion Guidance (AMG) and Color Uncertainty-guided Appearance Refinement (CUAR) — achieving spatiotemporally consistent, high-fidelity dynamic scene editing.

## Background & Motivation
Static scene editing with 3DGS has reached a relatively mature stage, with methods such as DGE, DreamCatalyst, and SGSST supporting fine-grained object manipulation and global style transfer. Extending editing capabilities to dynamic 4D scenes, however, remains highly challenging. **Limitations of Prior Work**: Methods such as Instruct 4D-to-4D, CTRL-D, and Instruct-4DGS primarily adapt 2D diffusion models to spatiotemporal settings and fit 4D representations from edited 2D frames. Because 2D editing lacks explicit geometric reasoning, these approaches suffer from spatial distortion, temporal flickering, and unintended modifications to non-edited regions.

**Key Challenge**: 4D Gaussian scenes typically consist of canonical 3D Gaussians combined with a learned deformation network. After editing, the Gaussians (following clone/split/prune operations) deviate from the original geometric distribution, while the deformation network is trained only on the original geometry — leaving it without motion priors for newly added Gaussians and unable to generalize to edited configurations. **Key Insight**: Rather than editing directly in the 4D domain, this paper adopts a "edit in 3D first, then propagate to 4D" strategy — leveraging mature 3D editors to modify the first frame and designing dedicated propagation mechanisms to ensure temporal consistency of the edits.

## Method

### Overall Architecture
The Catalyst4D framework operates in two stages: (1) an existing 3D editor (DGE / DreamCatalyst / SGSST) is used to edit the 3D Gaussians of the first frame, $\mathcal{G}^1_{\text{edit}}$; (2) the AMG module establishes region-level correspondences between edited and original Gaussians and propagates motion, after which the CUAR module corrects appearance artifacts introduced by motion propagation.

### Key Designs
1. **Anchor-based Motion Guidance (AMG)**:

    - Function: Establishes reliable motion supervision for edited Gaussians by transferring temporal deformations from original Gaussians to edited ones.
    - Mechanism (three steps):
        - **Anchor construction**: kNN local neighborhoods $\{\mathcal{N}_{ei}\}$ are built for both original and edited Gaussian point clouds. Candidate lines are generated from uniformly sampled point pairs on bounding sphere surfaces, parameterized as $S_r(u,\varphi) = (r\sqrt{1-u^2}\cos\varphi, r\sqrt{1-u^2}\sin\varphi, ru)$. Intersections between lines and neighborhoods (entire neighborhood within a cylinder of radius $\delta$) are detected, and for each intersecting neighborhood a distance-weighted centroid is computed as the anchor: $\mathbf{p} = \frac{\sum d_x \mathbf{x}}{\sum d_x}$.
        - **Correspondence establishment**: Unbalanced Optimal Transport (UOT + Sinkhorn algorithm) is applied to compute a soft correspondence matrix $P \in \mathbb{R}^{n\times m}$; reliable correspondences are determined by taking the column-wise maximum.
        - **Deformation aggregation**: For each edited Gaussian $\mathbf{g}$, the corresponding source Gaussian set $\mathcal{G}^{\text{sub}}_{\text{src}}$ is identified via the correspondence matrix, and their temporal deformations are aggregated with weights: $\Delta\boldsymbol{\mu}^t_\mathbf{g} = \frac{\sum w_{\mathbf{g}'}\Delta\boldsymbol{\mu}^t_{\mathbf{g}'}}{\sum w_{\mathbf{g}'}}$, where weights combine opacity and Mahalanobis distance: $w_{\mathbf{g}'} = \sigma_{\mathbf{g}'}\exp(-\frac{1}{2}(\boldsymbol{\mu}_{\mathbf{g}'}-\boldsymbol{\mu}_{\mathbf{g}})^T\boldsymbol{\Sigma}^{-1}_{\mathbf{g}'}(\boldsymbol{\mu}_{\mathbf{g}'}-\boldsymbol{\mu}_{\mathbf{g}}))$.
    - Design Motivation: Direct point-wise KNN matching is susceptible to noise and causes cross-semantic motion confusion (e.g., hand motion erroneously influencing the torso). Anchors provide stable region-level references, and optimal transport ensures semantically consistent soft correspondences.

2. **Color Uncertainty-guided Appearance Refinement (CUAR)**:

    - Function: Detects and corrects color artifacts arising from motion propagation or occlusion.
    - Mechanism:
        - **Optical flow rendering**: An optical flow map $F^v_{1\to t}$ from frame 1 to frame $t$ is rendered from the motion deformation $\Delta\boldsymbol{\mu}^t$, and the edited image from frame 1 is warped to frame $t$ to serve as pseudo ground truth.
        - **Color uncertainty estimation**: Per-Gaussian inter-frame SH color differences are computed as $C^{v,t}_{\text{diff}} = \|\text{SH}(\mathbf{sh},\mathbf{v})_t - \text{SH}(\mathbf{sh},\mathbf{v})_1\|_1$; uncertainty is defined as $\xi^v_t = 1 - \exp(-C^{v,t}_{\text{diff}})$ and composited into pixel-level uncertainty maps via $\alpha$-blending, then binarized into an artifact mask: $M^v_t = (U^v_t > \epsilon \cdot \text{mean}(U^v_t))$.
        - **Selective refinement**: A foreground refinement loss (L1 + SSIM) using the warped image is applied only in artifact regions; a background regularization loss preserves non-artifact regions unchanged.
    - Design Motivation: Editing operations inevitably affect interior Gaussians that only expose color artifacts upon motion. Using the first-frame editing result — already guaranteed to be multi-view consistent by the 3D editor — as warp supervision is more reliable than diffusion-based post-processing.

3. **Compatibility with General 4D Representations**:

    - Function: Adapts to different 4D Gaussian representations.
    - Mechanism: Swift4D is used for multi-camera settings and 4DGS for monocular settings; opacity and color attributes are shared consistently across frames.
    - Design Motivation: The propagation mechanism of Catalyst4D is decoupled from the underlying 4D representation and is applicable to any system with canonical Gaussians and a deformation field.

### Loss & Training
- AMG contains no learnable parameters; it is a purely geometric computation.
- CUAR refinement loss: $L_{\text{refine}} = (1-\zeta)L_{\text{fore}} + \zeta L_{\text{back}}$
    - Foreground: $L_{\text{fore}} = (1-\eta)\|M \odot (\text{render} - \text{warp})\|_1 + \eta L_{\text{ssim}}$
    - Background: $L_{\text{back}} = \|(1-M) \odot (\text{render} - \text{render}_{\text{org}})\|_1$

## Key Experimental Results

### Main Results

| Scene | Method | CLIP sim.↑ | Consistency↑ | Time↓ |
|------|------|------------|-------------|-------|
| Sear-steak | **Catalyst4D** | **0.252** | 0.983 | 50min |
| | CTRL-D | 0.249 | **0.985** | 55min |
| | I4DGS | 0.220 | 0.980 | 40min |
| Coffee-martini | **Catalyst4D** | **0.249** | **0.986** | 50min |
| | CTRL-D | 0.246 | 0.983 | 55min |
| Trimming | **Catalyst4D** | **0.251** | **0.967** | 40min |
| | IN4D | 0.243 | 0.945 | 2h* |

### Ablation Study

| Configuration | CLIP Sim.↑ | Consistency↑ | Note |
|------|------------|-------------|------|
| w/o AMG | 0.245 | 0.966 | Erroneous motion propagation due to missing anchor guidance |
| w/o CUAR | 0.248 | 0.969 | Color artifacts due to missing appearance refinement |
| Full model | **0.252** | **0.971** | Two modules are complementary |

### Key Findings
- Catalyst4D achieves the highest CLIP similarity across all scenes, indicating superior semantic alignment.
- Compared to KNN-Guide: KNN causes cross-region motion confusion (hand motion affecting the torso); AMG resolves this effectively via anchors and OT.
- Compared to DeformNet-Guide: The deformation network fails to generalize to edited Gaussians, producing large deformation artifacts.
- Although CTRL-D appears visually plausible, it introduces unintended modifications to non-edited regions (e.g., objects on the table, a dog on a stool).
- Catalyst4D is more efficient than IN4D (which requires 2 GPUs and 2 hours) and comparable to CTRL-D and I4DGS.

## Highlights & Insights
- **Decoupled "edit in 3D, propagate to 4D" paradigm**: Fully leverages mature 3D editing capabilities, avoiding the pitfalls of directly supervising 4D representations with 2D diffusion outputs.
- **Region-level correspondences via anchors and optimal transport**: More robust than point-wise matching; the core idea is to identify stable structural reference points.
- **Uncertainty-driven selective correction**: Only repairs problematic regions, avoiding the introduction of global inconsistencies.
- Supports both local editing (clothing/color) and global style transfer, applicable to both monocular and multi-camera settings.

## Limitations & Future Work
- The deformation network is not retrained and the density of edited Gaussians is not modified, leading to slight consistency degradation in certain scenes.
- Quality is dependent on the underlying 3D editor — poor first-frame editing cannot be corrected during propagation.
- Evaluation is conducted only on the DyNeRF, MeetRoom, and HyperNeRF datasets.
- Future directions include adaptive anchor density, joint optimization with 3D editors, and support for topology-changing edits.

## Related Work & Insights
- The "3D→4D propagation" paradigm is generalizable to other 3D operations, such as temporal extension following 3D completion or 3D generation.
- The anchor + OT correspondence establishment approach is applicable to any task requiring geometric correspondences between pre- and post-editing configurations.
- The color uncertainty idea — using SH color differences and physics-based propagation for selective correction — is broadly applicable to inter-frame inconsistency problems.

## Rating
- Novelty: ⭐⭐⭐⭐ The "3D editing → 4D propagation" paradigm is novel, and the AMG and CUAR designs are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, quantitative and qualitative evaluation, ablation studies, and comparison with multiple baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly derived, module logic is compact, and figures are well-crafted.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for 3D→4D editing propagation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization](customtex_high-fidelity_indoor_scene_texturing_via_multi-reference_customization.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[CVPR 2026\] TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification](topomesh_high-fidelity_mesh_autoencoding_via_topological_unification.md)
- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)

</div>

<!-- RELATED:END -->

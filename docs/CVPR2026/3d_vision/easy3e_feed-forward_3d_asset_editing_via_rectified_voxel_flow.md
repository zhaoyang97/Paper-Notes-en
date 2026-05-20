---
title: >-
  [Paper Note] Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow
description: >-
  [CVPR 2026][3D Vision][3D editing] This paper proposes a feed-forward 3D asset editing framework built upon the TRELLIS 3D generation backbone. It achieves globally consistent geometric deformation in a sparse voxel late…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D editing"
  - "feed-forward generation"
  - "voxel flow"
  - "Flow Matching"
  - "texture refinement"
date: 2026-05-08
content_hash: 73bcd226ec0736ea
---

# Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow

**Conference**: CVPR 2026
**arXiv**: [2602.21499](https://arxiv.org/abs/2602.21499)  
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: 3D editing, feed-forward generation, voxel flow, Flow Matching, texture refinement

## TL;DR

This paper proposes a feed-forward 3D asset editing framework built upon the TRELLIS 3D generation backbone. It achieves globally consistent geometric deformation in a sparse voxel latent space via Voxel FlowEdit, and recovers high-frequency details through normal-guided multi-view texture refinement.

## Background & Motivation

Existing 3D editing methods fall into two categories: (1) **2D-lifting pipelines** (e.g., Instruct-NeRF2NeRF), which supervise per-scene iterative optimization of 3D representations with 2D edited images — computationally expensive, reliant on multi-view coverage, and prone to collapse under large geometric edits; (2) **multi-view diffusion models**, which improve cross-view consistency but still reason about 3D structure implicitly in 2D feature space, making topology and volumetric changes difficult to handle.

Recently emerged **3D-native generative models** (e.g., TRELLIS, LRM) directly learn structured 3D latent spaces, offering a new paradigm for feed-forward editing. However, two key challenges remain:
- Paired 3D editing data is unavailable, requiring adaptation of training-free 2D editing methods to 3D latent spaces, where many 2D components (e.g., cross-attention manipulation) are not directly transferable.
- Compressed 3D features cause loss of high-frequency texture details, limiting appearance fidelity.

## Method

### Overall Architecture

Easy3E builds on the TRELLIS generation backbone and operates in two stages: **geometric editing** and **texture refinement**.

**Input**: source 3D asset $\mathcal{A}_{\text{src}}$, 3D region mask $\mathcal{M}$, target-view image $I^{\text{tgt}}$ obtained via 2D editing.
**Output**: edited 3D asset.

**Pipeline**: **Voxel FlowEdit** performs global geometric deformation in the sparse voxel latent space → **SLAT Repainting** refines local latent features → mesh is decoded → **normal-guided multi-view generation** recovers high-fidelity texture.

### Key Designs

1. **Structured Latent (SLAT) Representation**: A 3D asset is represented as $\mathbf{Z}=(\mathcal{V}, \{\mathbf{z}_{\mathbf{p}}\}_{\mathbf{p}\in\mathcal{V}})$, where $\mathcal{V}$ denotes the set of active voxels intersecting the mesh surface and $\mathbf{z}_{\mathbf{p}}$ are local latent features obtained by projecting and aggregating DINOv2 multi-view features. TRELLIS employs two rectified flow transformers to predict the voxel structure and the latent feature field, respectively. This representation explicitly encodes geometry and provides the foundation for editing directly in the 3D latent space.

2. **Voxel FlowEdit (Sparse Voxel Editing)**: The core contribution. A continuous editing trajectory from source to target is constructed in the 3D VAE latent space of the voxel structure. Inspired by FlowEdit, the editing velocity field is defined as the difference between target- and source-conditioned velocities:

    $\mathbf{v}_{\text{edit}}(\mathbf{x}_t, t) = \mathbf{v}_{\theta}(\mathbf{x}^{\text{tgt}}_t, t \mid I^{\text{tgt}}) - \mathbf{v}_{\theta}(\mathbf{x}^{\text{src}}_t, t \mid I^{\text{src}})$

   However, directly integrating this ODE causes trajectory drift and structural collapse due to discretization error. **Guided Flow Regularization** is therefore introduced:
    - **Silhouette guidance** $\mathbf{G}_{\text{sil}}$: gradients of a BCE loss w.r.t. the target silhouette, aligning the evolving structure to the target outline.
    - **Trajectory consistency correction** $\boldsymbol{\xi}_{\text{traj}}$: projects deviated latent states back onto the manifold.

   The final update rule is:

   $\mathrm{d}\mathbf{x}_t = \mathcal{M}_\ell \odot \big[\mathbf{v}_{\text{edit}} + \Gamma\boldsymbol{\xi}_{\text{traj}} - \eta\mathbf{G}_{\text{sil}}\big]\mathrm{d}t$

   where $\mathcal{M}_\ell$ restricts updates to the editable region, and $\Gamma=0.1$, $\eta=0.2$ control the weighting of each term.

3. **SLAT Repainting (Latent Space Inpainting)**: Local latent features are refined on the edited voxel set $\mathcal{V}_{\text{tgt}}$. In the editable region, features are updated using the target-conditioned velocity field; in the non-editable region, features follow the forward diffusion trajectory of the source distribution to maintain consistency:

    $\mathbf{z}_{k-1} = \mathcal{M}_z \odot [\mathbf{z}_k + \Delta t \cdot \mathbf{v}_\theta(\mathbf{z}_k, t_k \mid I^{\text{tgt}})] + (1-\mathcal{M}_z) \odot [(1-t_k)\mathbf{z}^{\text{src}} + t_k\boldsymbol{\epsilon}_k]$

   A softened mask $\widetilde{\mathcal{M}_z}=\text{blur}(\mathcal{M}_z; \sigma_b)$ is applied to avoid seam artifacts.

4. **Normal-Guided Texture Refinement**: An optional module that addresses the loss of high-frequency texture in compressed 3D representations.

    - **Control Branch**: A frozen ControlNet with a trainable Ctrl-Adapter takes per-view normal maps of the edited mesh as input and extracts multi-scale geometric control features.
    - **Generation Branch**: Based on the ERA3D multi-view diffusion architecture, conditioned on the edited image $I^{\text{tgt}}$, it generates 6 geometrically consistent auxiliary views under the guidance of control features.
    - **Texture Fusion**: Visibility-aware, mask-weighted blending into the UV texture map.

### Loss & Training

- Voxel FlowEdit is **training-free**, leveraging the velocity field of the pretrained TRELLIS model at inference time.
- The ODE is discretized into 25 sampling steps; CFG scale is set to 5–15 on the target side and fixed at 5 on the source side.
- The editing velocity $\mathbf{v}_{\text{edit}}$ is averaged over $n_{\text{avg}} \in \{2,4\}$ noise samples to improve stability.
- The Ctrl-Adapter is trained on an Objaverse subset (6 views at $512 \times 512$ with normal maps); only the Adapter parameters are updated, while the ControlNet and ERA3D backbones remain frozen.

## Key Experimental Results

### Main Results

**Evaluation set**: 100 3D assets from Sketchfab, NPHM, THuman2.1, and Objaverse, covering human heads, full bodies, and general objects.

| Method | CLIP-T ↑ | DINO-I ↑ | LPIPS ↓ | FID ↓ |
|------|----------|----------|---------|-------|
| TRELLIS | 0.323 | 0.895 | 0.243 | 45.8 |
| MVEdit | 0.267 | 0.851 | 0.282 | 67.6 |
| Vox-E | 0.266 | 0.734 | 0.673 | 90.3 |
| Instant3DiT | 0.285 | 0.874 | 0.286 | 49.7 |
| **Easy3E** | **0.326** | **0.952** | **0.138** | **25.8** |

Easy3E achieves top performance across all metrics: FID is reduced by 43.7% and LPIPS by 43.2% relative to the second-best method TRELLIS.

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Without Flow Guidance ($\mathbf{G}_{\text{sil}}$ + $\boldsymbol{\xi}_{\text{traj}}$) | Structural collapse and excessive retention of source structure | Both components must be used jointly; enabling either alone leads to imbalanced updates |
| Without Texture Refinement | Blurring and color shift in edited regions | The normal-guided module substantially improves surface detail and view-consistent appearance |
| Full model | Clean geometry + high-fidelity texture | All modules are complementary |

### Key Findings

- **User study** (46 participants × 10 groups): Easy3E achieves substantial leads across five dimensions — prompt fidelity (88.98%), identity preservation (94.63%), editing quality (94.92%), 3D consistency (97.51%), and overall preference (97.00%).
- Among baselines, MVEdit produces only texture-level changes with minimal geometric modification; Vox-E and Instant3DiT struggle to maintain structural integrity.
- Feed-forward inference requires no per-scene optimization.

## Highlights & Insights

- **Core insight**: Editing directly in a 3D-native structured latent space is better suited for large geometric deformations than 2D-lifting or multi-view diffusion approaches.
- Voxel FlowEdit elegantly adapts the 2D flow-matching editing paradigm to 3D sparse voxels, resolving discretization drift via silhouette guidance and trajectory correction.
- The decomposition of editing into "geometry via latent space + appearance via multi-view diffusion" is an elegant design that exploits the geometric strengths of 3D generative models and the texture strengths of 2D diffusion models.

## Limitations & Future Work

- Performance is upper-bounded by TRELLIS's generative capacity; extreme geometric modifications remain challenging.
- The normal-guided texture refinement currently synthesizes auxiliary views at relatively low resolution, limiting recovery of very fine-grained textures.
- Users are required to provide a 3D editing mask and a 2D edited view, leaving room to further reduce interaction cost.
- Inference time is not quantitatively reported; the paper claims "fast" inference without concrete comparison.

## Related Work & Insights

- **TRELLIS**: The 3D generation backbone adopted in this work, which learns a structured 3D latent space.
- **FlowEdit**: A 2D flow-matching editing method extended to 3D voxel space in this paper.
- **ERA3D**: A multi-view diffusion generation architecture used in the texture refinement branch.
- **Takeaway**: The emergence of 3D-native generative models opens an entirely new paradigm for 3D editing; "editing in latent space" may become the dominant approach.

## Rating

- Novelty: ⭐⭐⭐⭐ Adapting flow-matching editing to 3D sparse voxel space is a novel attempt; the overall framework is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative metrics, user study, and ablation experiments are all provided, though inference speed comparison is absent.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with detailed mathematical derivations and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Feed-forward 3D editing addresses a practical need; this work presents the strongest solution to date.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Particulate: Feed-Forward 3D Object Articulation](particulate_feed-forward_3d_object_articulation.md)
- [\[CVPR 2026\] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)
- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)

</div>

<!-- RELATED:END -->

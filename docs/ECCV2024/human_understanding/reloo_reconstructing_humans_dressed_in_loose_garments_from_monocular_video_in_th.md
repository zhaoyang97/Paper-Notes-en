---
title: >-
  [Paper Note] ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild
description: >-
  [ECCV 2024][Human Understanding][Human Reconstruction] Proposed ReLoo, which reconstructs high-quality 3D human models dressed in loose garments from monocular in-the-wild videos through a layered neural human representation and a non-hierarchical virtual bone deformation module.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Human Reconstruction"
  - "Loose Garments"
  - "Virtual Bones"
  - "Layered Neural Representation"
  - "Monocular Video"
date: 2026-05-08
content_hash: 36a242422b23eab2
---

# ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild

**Conference**: ECCV 2024  
**arXiv**: [2409.15269](https://arxiv.org/abs/2409.15269)  
**Code**: [Available](https://moygcc.github.io/ReLoo/)  
**Area**: Human Understanding  
**Keywords**: Human Reconstruction, Loose Garments, Virtual Bones, Layered Neural Representation, Monocular Video

## TL;DR

Proposed ReLoo, which reconstructs high-quality 3D human models dressed in loose garments from monocular in-the-wild videos through a layered neural human representation and a non-hierarchical virtual bone deformation module.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Existing monocular human reconstruction methods mainly focus on tight-fitting clothing and perform poorly on loose garments (e.g., skirts, loose tops). The core reason is that these methods model the human body and clothing as a single entity, relying solely on skeletal deformation driving, which cannot handle the large non-rigid deformations of loose garments that are weakly correlated with body poses. Template-based methods rely on pre-scanned templates, limiting deployment to new subjects; NeRF-based methods lack the ability to model independent clothing movement.

## Method

### Overall Architecture

1. Establish a layered neural implicit representation, decomposing the human body into an inner body layer and an outer garment layer.
2. Introduce a non-hierarchical virtual bone deformation module for the garment layer.
3. Jointly optimize shape, appearance, and deformation through multi-layer differentiable volume rendering.

### Key Designs

**Layered Neural Representation**: The body layer $f^B$ and garment layer $f^G$ model SDF and radiance values using neural networks respectively, and are combined by taking the minimum value to obtain the complete dressed human.

**Virtual Bone Deformation**: Defines $n_v$ non-hierarchical virtual bones, and predicts their rigid transformations from bone positions, poses, and time embeddings using an MLP. Virtual bones can move freely without hierarchical constraints, making them suitable for capturing the deformations of highly dynamic loose garments. Skinning weights are computed based on distance.

**Multi-layer Volume Rendering**: Points are sampled along rays in the body layer and garment layer respectively, and volume integration is performed after depth sorting. Occlusion is processed independently in each layer to correctly handle the occlusion relationship between the garment and the body.

**Two-stage Training**: The first stage drives two layers with skeletal deformation and warms up the virtual bone deformation field; the second stage activates the virtual bone deformation module and extracts virtual bone positions from the garment mesh.

### Loss & Training

Includes reconstruction loss $L_{\text{rgb}}$, segmentation loss $L_{\text{seg}}$ (using SAM to obtain masks), adaptive Eikonal loss $L_{\text{eikonal}}$, and virtual bone regularization loss $L_{\text{reg}}$.

## Key Experimental Results

### Main Results

**Surface Reconstruction on MonoLoose Dataset**:

| Method | Chamfer-L2↓ | NC↑ | V-IoU↑ |
|------|-------------|-----|--------|
| SelfRecon | 2.22 | 0.788 | 0.844 |
| Vid2Avatar | 2.34 | 0.794 | 0.776 |
| SCARF | 3.13 | 0.711 | 0.691 |
| **ReLoo** | **1.93** | **0.831** | **0.881** |

**Novel View Synthesis (MonoLoose / DynaCap)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|-------|-------|--------|
| SelfRecon | 22.5 | 0.953 | 6.08 | 26.8 | 0.982 | 1.56 |
| Vid2Avatar | 25.9 | 0.968 | 4.66 | 27.1 | 0.983 | 1.82 |
| SCARF | 23.3 | 0.953 | 6.59 | 25.5 | 0.979 | 2.55 |
| **ReLoo** | **29.2** | **0.970** | **3.15** | **27.9** | **0.985** | **1.27** |

### Ablation Study

**Impact of the Number of Virtual Bones on Performance**:

| Number of Virtual Bones $n_v$ | LPIPS↓ | Time per Iteration |
|----------------|--------|----------|
| 20 | ~4.2 | Fast |
| 40 | ~3.8 | Medium |
| **80** | **~3.2** | **Medium** |
| 160 | ~3.1 | Slow |
| 320 | ~3.0 | Very slow |

- Removing virtual bones: MonoLoose PSNR drops from 29.2 to 28.7, and DynaCap drops from 27.9 to 27.3.
- Removing multi-stage sampling: V-IoU drops from 0.881 to 0.879, but leads to garment-body interpenetration issues.
- Experiment on the number of virtual bones: $n_v=80$ is the optimal performance-efficiency trade-off point, where LPIPS shows the largest drop.

### Key Findings

- SMPL skeletal deformation cannot handle loose garments that are far from the body, making the virtual bone module crucial.
- Layered representation can capture changing garment topologies (e.g., the gap between skirt legs).
- Robust on in-the-wild videos.

## Highlights & Insights

1. Virtual bones are non-hierarchical, free from anatomical constraints, and can capture garment dynamics freely.
2. Multi-layer volume rendering correctly handles the occlusion relationship between the body and the garment.
3. No 3D supervision or garment template priors are required.
4. Using SAM for segmentation provides weak supervision.

## Limitations & Future Work

- Relies on reasonable pose estimation and segmentation masks, sometimes requiring manual adjustment of SAM results.
- Mainly supports the reconstruction of up to two garments.
- Complexity increases linearly with the number of garments.

## Related Work & Insights

- SCARF reconstructs loose clothing with NeRF based on SMPL-X, but is limited to rotational motion.
- Pan et al. use virtual bones for garment animation, but require known templates and 3D simulation data.
- Insight: Layered representation + free deformation module is an effective paradigm for handling complex garment dynamics.

## Rating

- Novelty: ★★★★☆ The design of the virtual bone deformation module and layered implicit representation is ingenious.
- Value: ★★★★☆ Reconstructing humans in loose garments from monocular video offers broad application prospects.
- Experimental Thoroughness: ★★★★★ Proposed a new dataset MonoLoose, with comprehensive experimental comparisons.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](../../ICCV2025/human_understanding/monocular_facial_appearance_capture_in_the_wild.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](../../CVPR2025/human_understanding/d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[ECCV 2024\] How Video Meetings Change Your Expression](how_video_meetings_change_your_expression.md)
- [\[CVPR 2025\] FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](../../CVPR2025/human_understanding/fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)
- [\[CVPR 2025\] Homogeneous Dynamics Space for Heterogeneous Humans](../../CVPR2025/human_understanding/homogeneous_dynamics_space_for_heterogeneous_humans.md)

</div>

<!-- RELATED:END -->

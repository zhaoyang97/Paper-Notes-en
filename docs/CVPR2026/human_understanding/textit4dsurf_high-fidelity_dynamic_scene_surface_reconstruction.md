---
title: >-
  [Paper Note] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction
description: >-
  [CVPR 2026][Human Understanding][dynamic surface reconstruction] This paper proposes 4DSurf, a general-purpose dynamic scene surface reconstruction framework based on 2D Gaussian splatting. By introducing Gaussian motion-induced SDF flow regularization to constrain the temporally consistent evolution of surfaces, and adopting an overlapping segment partitioning strategy to handle large deformations, 4DSurf surpasses existing SOTA methods by 49% and 19% in Chamfer distance on the Hi4D and CMU Panoptic datasets, respectively.
tags:
  - CVPR 2026
  - Human Understanding
  - dynamic surface reconstruction
  - Gaussian splatting
  - SDF flow regularization
  - temporal consistency
  - large deformation handling
date: 2026-05-08
content_hash: 8b24f15f6521183e
---

# 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.28064](https://arxiv.org/abs/2603.28064)
**Code**: N/A
**Area**: Human Understanding
**Keywords**: dynamic surface reconstruction, Gaussian splatting, SDF flow regularization, temporal consistency, large deformation handling

## TL;DR

This paper proposes 4DSurf, a general-purpose dynamic scene surface reconstruction framework based on 2D Gaussian splatting. By introducing Gaussian motion-induced SDF flow regularization to constrain the temporally consistent evolution of surfaces, and adopting an overlapping segment partitioning strategy to handle large deformations, 4DSurf surpasses existing SOTA methods by 49% and 19% in Chamfer distance on the Hi4D and CMU Panoptic datasets, respectively.

## Background & Motivation

**State of the Field**: Dynamic surface reconstruction aims to recover temporally consistent 3D geometry from video sequences, serving as a foundation for applications such as digital humans and virtual reality. Gaussian splatting (GS)-based methods have become mainstream due to their real-time rendering capabilities and efficient optimization.

**Limitations of Prior Work**: Existing GS-based dynamic surface reconstruction methods (e.g., D-2DGS, DG-Mesh, DGNS) typically perform well only on single-object or small-deformation scenarios. In the presence of large deformations, they suffer from surface jitter and temporally inconsistent geometric distortions. Many methods also rely on human body priors such as SMPL-X or pretrained depth/normal estimation models, limiting their generalizability.

**Root Cause**: The central challenge is how to simultaneously achieve, without relying on any object-specific priors: (1) general surface reconstruction for arbitrary dynamic scenes (multi-object, non-rigid); (2) temporal consistency under large deformations; and (3) high-fidelity geometry from sparse viewpoints.

**Paper Goals**: (1) Align Gaussian motion with surface evolution to eliminate temporal inconsistency; (2) handle large deformations in long sequences without error accumulation; (3) build a general-purpose framework free of prior dependencies.

**Starting Point**: The approach departs from the concept of SDF flow (the temporal derivative of the SDF field), establishing a connection between Gaussian motion and SDF variation — if the motion of Gaussians correctly reflects the temporal evolution of the surface, the SDF flows derived from both perspectives should be consistent. This constraint enables temporally consistent surface reconstruction.

**Core Idea**: Temporal consistency in dynamic surface reconstruction is achieved without object-specific priors by enforcing consistency between the SDF flow defined by the Gaussian velocity field and the SDF flow estimated from depth map variations.

## Method

### Overall Architecture

The 4DSurf pipeline divides the video sequence into overlapping segments via Overlapping Segment Partitioning, where each segment contains $K+1$ timesteps (including one virtual timestep shared with the next segment). Each segment maintains its own canonical space and Gaussian Velocity Field. The first segment is initialized from a visual convex hull, while subsequent segments are initialized from the virtual-timestep Gaussians of the preceding segment. SDF flow regularization is applied during training to enforce temporally consistent surface evolution.

### Key Designs

1. **Gaussian Velocity Field**:

    - **Function**: Explicitly models the motion of Gaussians from canonical space to arbitrary timesteps, providing the basis for SDF flow derivation.
    - **Mechanism**: Given the canonical center $\mu_i$ of the $i$-th Gaussian and timestep $t$, an MLP $\mathcal{F}_\theta$ predicts three motion parameters: linear velocity $\mathbf{v}(\mu_i, t)$, angular velocity $\omega(\mu_i, t)$, and dilation velocity $\mathbf{e}(\mu_i, t)$. Position, rotation, and scale are obtained by integration: $\mu_i^t = \mu_i + \mathbf{v} \cdot t$, $q_i^t = \phi(\omega \cdot t) \otimes q_i$, $\xi_i^t = \xi_i + \mathbf{e} \cdot t$. Unlike methods that directly predict deformations, predicting velocities naturally enables the derivation of SDF flow.
    - **Design Motivation**: Parameterizing the velocity field rather than the displacement field makes the mathematical derivation of SDF flow tractable, seamlessly bridging motion modeling and geometric constraints.

2. **SDF Flow Regularization**:

    - **Function**: Constrains Gaussian motion to be consistent with surface evolution, eliminating temporal jitter and inconsistency.
    - **Mechanism**: SDF flow is derived from two complementary perspectives and required to be mutually consistent. (1) From the Gaussian motion perspective, based on the theorem $\mathbf{f} = -(\omega \times R^t \mathbf{x} + \mathbf{v})^\top \mathbf{n}(R^t \mathbf{x})$, i.e., the SDF change equals the negative normal-direction projection of the scene flow. (2) From the geometric variation perspective, the rendered depth map approximates the SDF value as $\tilde{s}(\mu_i^t, t) = \hat{D}(\mathbf{p}^*, t) - d(\mu_i^t, t)$, whose temporal derivative yields the SDF flow. The regularization loss is the L1 norm of the discrepancy: $\mathcal{L}_{flow} = \sum_i |\mathbf{f}_i^t - \tilde{\mathbf{f}}_i^t|$.
    - **Design Motivation**: SDF flow directly links the motion field to geometric evolution, constituting a strong and elegant physical constraint. The dual-perspective consistency provides complementary supervisory signals.

3. **Overlapping Segment Partitioning + Incremental Motion Tuning (OSP + IMT)**:

    - **Function**: Handles large deformations in long sequences while reducing error accumulation and memory overhead.
    - **Mechanism**: The sequence is divided into overlapping segments, with each segment sharing a virtual timestep to propagate geometric information across segments. Incremental Motion Tuning (IMT) avoids training the velocity field from scratch for the $N$-th segment ($N \geq 2$); instead, it fine-tunes the preceding segment's velocity field via LoRA: $\theta^N = \theta^{N-1} + \Delta\theta^N$, $\Delta\theta^N = A^N B^N$ (where $r \ll d$), significantly reducing storage costs.
    - **Design Motivation**: A single deformation field with a canonical space struggles to model large deformations. The segment-based strategy decomposes large deformations into smaller intra-segment deformations, while overlap ensures geometric continuity. LoRA exploits the high correlation between motions in adjacent segments for parameter-efficient incremental training.

### Loss & Training

The total loss is a weighted combination of five terms: $\mathcal{L}_{total} = \mathcal{L}_{img} + \lambda_1 \mathcal{L}_n + \lambda_2 \mathcal{L}_d + \lambda_3 \mathcal{L}_{flow} + \lambda_4 \mathcal{L}_m$, where $\mathcal{L}_{img}$ is the L1+D-SSIM photometric loss, $\mathcal{L}_n$ is the normal alignment loss (from 2DGS), $\mathcal{L}_d$ is the depth distillation loss, $\mathcal{L}_{flow}$ is the SDF flow regularization, and $\mathcal{L}_m$ is the alpha mask loss.

## Key Experimental Results

### Main Results

Chamfer distance (mm) on the CMU Panoptic dataset:

| Method | Band1 | Ian3 | Haggling_b2 | Pizza1 |
|--------|-------|------|------------|--------|
| Neural SDF-Flow | 17.2 | 15.8 | 13.5 | 16.1 |
| Dynamic-2DGS | 16.0 | 12.5 | 13.7 | 16.2 |
| Space-Time-2DGS | 16.4 | 12.6 | 13.7 | 15.8 |
| GauSTAR | 17.6 | 13.7 | 14.8 | 14.7 |
| **Ours w IMT-64** | **12.8** | **10.4** | **11.0** | **12.1** |
| **Ours wo IMT** | **12.7** | **10.5** | **10.8** | **12.2** |

### Ablation Study

| Configuration | Effect (Overall Chamfer Distance) |
|---------------|----------------------------------|
| Full 4DSurf | Best |
| w/o SDF flow regularization | Significant drop in temporal consistency; surface jitter |
| w/o overlapping segments | Severe error accumulation in large-deformation scenes |
| IMT-64 vs. full velocity field | Negligible performance loss; substantially reduced storage |

### Key Findings

- **Large margin over existing SOTA**: Overall Chamfer distance improves by approximately 19% on CMU Panoptic and 49% on Hi4D.
- **Strong performance without priors**: Without relying on priors such as SMPL-X, the method generalizes far better than specialized approaches in general scenarios involving multi-person interaction.
- **SDF flow regularization is critical**: Ablation results show significant temporal consistency degradation upon removal of this regularization.
- **IMT reduces storage with negligible quality loss**: At LoRA rank 64, performance is nearly identical to that of the full velocity field while storage is substantially reduced.
- **Robust under sparse viewpoints**: Superior performance is maintained in sparse settings with fewer than 10 views.

## Highlights & Insights

- **Elegant theoretical derivation**: The theorem-based derivation of SDF flow from Gaussian motion is the most notable contribution, mathematically unifying motion constraints and geometric constraints in an elegant manner.
- **Strong generalizability**: A truly prior-free method with no restrictions on object count, type, or degree of deformation.
- **Novel application of LoRA in 3D reconstruction**: The incremental motion tuning paradigm is transferable to other dynamic scene modeling tasks.
- **Simple yet effective segmentation strategy**: The intuition of decomposing long-sequence large deformations into short-sequence small deformations is both principled and effective.

## Limitations & Future Work

- The hyperparameters of the segment strategy (segment length $K$, number of overlapping frames) affect results and require manual tuning per scene.
- Merging canonical spaces across segments remains a non-trivial problem, causing storage to grow linearly with the number of segments.
- Topological changes (e.g., object appearance/disappearance) are not handled; cross-segment initialization transfer may fail in extreme scenarios.
- Future work may explore combining SDF flow regularization with other 3DGS variants (e.g., 3DGS, Mip-Splatting).

## Related Work & Insights

- Neural SDF-Flow first proposed the concept of SDF flow but relied on NeRF, resulting in low efficiency; this work elegantly migrates the concept to the Gaussian splatting framework.
- 2DGS provides a stronger geometric modeling foundation compared to 3DGS; 4DSurf builds its dynamic extension upon it.
- The application of LoRA in dynamic 3D reconstruction is a direction that warrants further exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The combination of SDF flow regularization and the Gaussian velocity field constitutes a highly original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparisons across two datasets and multiple baselines, though ablation study details could be further enriched.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear; the method is presented in a well-organized manner.
- **Value**: ⭐⭐⭐⭐ — Addresses core challenges in dynamic surface reconstruction (temporal consistency and large deformations) with significant impact on the field.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)
- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](../../ICCV2025/human_understanding/avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [\[CVPR 2026\] HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation](humanorbit_3d_human_reconstruction_as_360_orbit_generation.md)
- [\[CVPR 2026\] A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)

<!-- RELATED:END -->

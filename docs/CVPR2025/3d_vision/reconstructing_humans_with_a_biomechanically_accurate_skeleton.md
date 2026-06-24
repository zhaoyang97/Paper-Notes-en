---
title: >-
  [Paper Note] Reconstructing Humans with a Biomechanically Accurate Skeleton
description: >-
  [CVPR 2025][3D Vision][Human pose estimation] HSMR represents the first method to estimate biomechanically accurate skeleton (SKEL) parameters from a single image. It overcomes the lack of ground-truth training data via an iterative pseudo-label refinement strategy. HSMR matches HMR2.0's performance on standard human pose estimation benchmarks while outperforming it significantly on extreme pose scenarios (MOYO yoga dataset) by over 18mm MPJPE…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human pose estimation"
  - "biomechanical skeleton"
  - "SKEL model"
  - "parametric human model"
  - "pseudo-label refinement"
date: 2026-05-08
content_hash: 807fd7651d90b598
---

# Reconstructing Humans with a Biomechanically Accurate Skeleton

**Conference**: CVPR 2025  
**arXiv**: [2503.21751](https://arxiv.org/abs/2503.21751)  
**Code**: [https://isshikihugh.github.io/HSMR/](https://isshikihugh.github.io/HSMR/)  
**Area**: 3D Vision / Human Reconstruction  
**Keywords**: Human pose estimation, biomechanical skeleton, SKEL model, parametric human model, pseudo-label refinement

## TL;DR

HSMR represents the first method to estimate biomechanically accurate skeleton (SKEL) parameters from a single image. It overcomes the lack of ground-truth training data via an iterative pseudo-label refinement strategy. HSMR matches HMR2.0's performance on standard human pose estimation benchmarks while outperforming it significantly on extreme pose scenarios (MOYO yoga dataset) by over 18mm MPJPE, all while effectively avoiding unnatural joint rotations.

## Background & Motivation

**Background**: 3D human pose estimation has achieved significant progress recently, with parametric human models represented by SMPL being widely adopted. From HMR to HMR2.0, Transformer-based regression methods have continuously set new SOTA performance on standard benchmarks. However, the outputs of these methods primarily serve visual applications (animation, AR/VR) and find very limited adoption in the field of biomechanics.

**Limitations of Prior Work**: SMPL and its successors utilize simplified skeletal designs where each joint is modeled as a ball-and-socket joint with three degrees of freedom (DoFs). This introduces two major issues: (1) the kinematic tree does not conform to real human anatomy; (2) the extra DoFs allow models to predict unnatural joint angles (such as hyperextension of the knee), making the output incompatible with biomechanical simulations.

**Key Challenge**: Visually plausible poses can be biomechanically invalid. The over-parameterization of SMPL allows networks to minimize 2D/3D joint errors through unnatural joint rotations at the expense of physical plausibility.

**Goal**: Replace SMPL with the biomechanically accurate SKEL model, directly regressing SKEL parameters from a single image. The core challenge is the complete absence of paired image-SKEL parameter training data.

**Key Insight**: The SKEL model shares the same surface mesh topology as SMPL, enabling the conversion of existing SMPL pseudo-labels into SKEL parameters. However, direct conversion yields limited quality and necessitates iterative refinement during training.

**Core Idea**: Borrowing the "optimization-in-the-loop" concept from SPIN, a SKELify optimization procedure is designed to periodically refine pseudo-label quality during training, initialized by network predictions and aligned with 2D keypoints.

## Method

### Overall Architecture

The input is a cropped image of a person, which is processed by a ViT backbone to extract features, followed by a Transformer head that regresses the SKEL model's pose parameters $q \in \mathbb{R}^{46}$, shape parameters $\beta \in \mathbb{R}^{10}$, and camera parameters $\pi$. The SKEL model outputs a skin mesh (6890 vertices) and a skeletal mesh based on these parameters. The training data incorporates initial pseudo-labels obtained via SMPL-to-SKEL conversion, which are iteratively refined using SKELify during training.

### Key Designs

1. **Continuous Rotation Representation instead of Euler Angles**:

    - Function: Addresses the issue that SKEL's Euler angle parameters are unsuitable for direct regression.
    - Mechanism: The network outputs continuous rotation representations $q_{\text{cont}}$, which are first converted to rotation matrices $q_{\text{mat}}$ via Gram-Schmidt (parameter losses are computed on this representation), and then converted to Euler angles $q_{\text{Euler}}$ to be input into the SKEL model. This avoids gimbal lock and discontinuity of Euler angles while remaining compatible with SKEL.
    - Design Motivation: Directly regressing Euler angles leads to unstable training, and continuous rotation representations have proven to be more friendly for regression tasks.

2. **SKELify Pseudo-label Iterative Refinement**:

    - Function: Progressively improves the label quality of the training data in the absence of ground truth.
    - Mechanism: For each training image, the current HSMR network prediction $(q^{\text{reg}}, \beta^{\text{reg}})$ is used as initialization. The SKEL parameters are optimized to align 3D joint projections with 2D keypoint ground truth. The optimization objective consists of three terms: 2D reprojection error $E_{\text{kp2D}}$ (with a robust kernel), shape prior $E_{\text{shape}} = \|\beta\|^2$, and pose prior based on biomechanical joint limits $E_{\text{pose}} = \sum_i \exp(l_i - q_i) + \exp(q_i - u_i)$. The optimization results replace the original pseudo-labels for subsequent training.
    - Design Motivation: Initial SMPL-to-SKEL conversion produces numerous failure cases (e.g., arm-body interpenetration, spinal twisting). Using network predictions as initialization is closer to a good solution than random initialization or direct SMPL conversion, forming a virtuous cycle.

3. **Biomechanical Joint Limit Constraints**:

    - Function: Guarantees that the predicted poses conform to the natural range of motion of human joints.
    - Mechanism: The SKEL model defines explicit upper and lower bounds for each joint DoF (e.g., knee joint: 0° extension to 135° flexion). The pose space of SKEL is only 46-dimensional (compared to SMPL's 72 dimensions), where each parameter corresponds to the Euler angle of a single DoF. This implicitly constrains the valid range of joint rotations, and limit constraints are further softly enforced via the exponential penalty term $E_{\text{pose}}$ in SKELify.
    - Design Motivation: SMPL's ball-and-socket joint design permits unnatural movements like lateral or reverse bending of the knees. Experiments confirm that methods like HMR2.0 frequently violate joint limits.

### Loss & Training

The total loss consists of four terms: parameter losses $\mathcal{L}_q = \|q_{\text{mat}} - q_{\text{mat}}^*\|_2^2$ and $\mathcal{L}_\beta = \|\beta - \beta^*\|_2^2$ (only used when pseudo-labels are available), together with 3D and 2D keypoint losses $\mathcal{L}_{\text{kp3D}} = \|X - X^*\|_1$ and $\mathcal{L}_{\text{kp2D}} = \|\pi(X) - x^*\|_1$. Large-scale training datasets from HMR2.0 (Human3.6M, COCO, MPII, etc.) are utilized. SKELify refinement is executed periodically in a batched manner.

## Key Experimental Results

### Main Results

| Dataset | Metric | HSMR | HMR2.0 | Difference |
|--------|------|------|--------|------|
| COCO | PCK@0.05 ↑ | 0.85 | 0.86 | -0.01 |
| 3DPW | MPJPE ↓ | 81.5 | 81.3 | +0.2 |
| 3DPW | PA-MPJPE ↓ | 54.8 | 54.3 | +0.5 |
| Human3.6M | MPJPE ↓ | 50.4 | 50.0 | +0.4 |
| MOYO | MPJPE ↓ | 104.5 | 123.3 | **-18.8** |
| MOYO | PA-MPJPE ↓ | 79.6 | 90.4 | **-10.8** |
| MOYO | MPVPE ↓ | 120.1 | 142.2 | **-22.1** |

### Ablation Study

| Configuration | COCO@0.05 | 3DPW MPJPE | MOYO MPJPE |
|------|-----------|------------|------------|
| HMR2.0 + SKEL fit (Two-stage) | 0.78 | 81.0 | 130.5 |
| HSMR (End-to-end) | 0.85 | 81.5 | 104.5 |

Joint violation frequency (MOYO, proportion of knee joints exceeding threshold):

| Method | 10° | 20° | 30° |
|------|-----|-----|-----|
| SMPL methods | High-frequency violations | High-frequency violations | High-frequency violations |
| HSMR | Extremely low | Extremely low | Almost no violations |

### Key Findings

- The gap between HSMR and HMR2.0 on standard benchmarks (3DPW, H36M) is within 0.5mm, proving that the constraints of the SKEL model do not sacrifice performance in conventional scenarios.
- On the extreme-pose MOYO dataset, MPJPE is improved by 18.8mm, showing that biomechanical constraints exert a strong regularization effect on difficult poses.
- The two-stage approach (HMR2.0 first, then SKEL fitting) is poorly performing and slow (3 min/frame), far inferior to end-to-end HSMR.
- All methods regressing SMPL parameters exhibit significant joint rotation violations, which HSMR almost completely avoids.

## Highlights & Insights

- **Trading constraints for generalization**: Intuitively, a more restricted model (46 DoFs vs 72 DoFs) should perform worse, but experiments show that appropriate constraints actually enhance generalization ability, especially on out-of-distribution extreme poses. This is an important insight—over-parameterization might be an implicit weakness in human reconstruction methods.
- **Loop refinement of pseudo-labels**: Incrementally upgrading data quality in the absence of ground truth through a loop of "network prediction $\to$ optimization $\to$ update labels $\to$ retrain". This strategy can be generalized to other label-scarce tasks.
- **Exposing systematic issues in SMPL methods**: The quantitative analysis of joint rotation violations is highly convincing, pointing out an overlooked direction of problems for the entire community.

## Limitations & Future Work

- Currently handles only single-person single-frame scenarios, without extension to multi-person or video settings.
- The SKEL model lacks fine-grained modeling of hands and the face.
- Pseudo-label refinement still cannot guarantee 100% correctness, and some failure cases from the SMPL-to-SKEL conversion might persist.
- True SKEL ground truth could be obtained in the future by incorporating biomechanical datasets like AddBiomechanics.
- When integrated with temporal models, the constraints of a biomechanically accurate skeleton will hold even greater value in motion analysis.

## Related Work & Insights

- **vs HMR2.0**: The architecture and training data are almost identical, with the only difference being SKEL vs SMPL. Drawing even on conventional datasets while leading significantly on extreme poses shows that skeleton design of itself is a crucial dimension of design.
- **vs HybrIK**: HybrIK introduces inverse kinematics constraints but remains based on the DoF design of SMPL, rendering it unable to fundamentally prevent unnatural rotations.
- **vs SKEL fitting**: Fitting SKEL directly to SMPL outputs is not only slow (3 min/frame) but also yields poor results, highlighting the clear advantage of end-to-end learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Although the SKEL model was not proposed in this paper, this work is the first to integrate it into an end-to-end regression framework and resolve the challenges of label-free training.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The evaluation across multiple datasets, the joint violation analysis, and the comparison with the two-stage baseline are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, fair experimental comparison, and precise pinpointing of issues.
- Value: ⭐⭐⭐⭐ Introduces a biomechanical dimension to human reconstruction, with the potential to impact applications such as motion analysis and rehabilitation medicine.

## Related Papers

- [\[CVPR 2025\] Reconstructing People, Places, and Cameras](reconstructing_people_places_and_cameras.md)
- [\[CVPR 2025\] Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [\[CVPR 2025\] PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [\[CVPR 2025\] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning](reconstructing_close_human_interaction_with_appearance_and_proxemics_reasoning.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reconstructing People, Places, and Cameras](reconstructing_people_places_and_cameras.md)
- [\[CVPR 2025\] Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [\[CVPR 2025\] PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[CVPR 2025\] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning](reconstructing_close_human_interaction_with_appearance_and_proxemics_reasoning.md)

</div>

<!-- RELATED:END -->

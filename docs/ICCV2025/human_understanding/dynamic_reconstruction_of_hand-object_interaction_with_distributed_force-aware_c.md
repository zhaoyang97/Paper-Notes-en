---
title: >-
  [Paper Note] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation
description: >-
  [ICCV 2025][Human Understanding][hand-object interaction] This paper proposes ViTaM-D, a vision-tactile fusion framework that achieves dynamic reconstruction of hand-object interaction for both rigid and deformable objec…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "hand-object interaction"
  - "tactile perception"
  - "contact modeling"
  - "deformable objects"
  - "SDF reconstruction"
date: 2026-05-08
content_hash: 183f6db0bb941cf0
---

# Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation

**Conference**: ICCV 2025
**arXiv**: [2411.09572](https://arxiv.org/abs/2411.09572)  
**Code**: [sites.google.com/view/vitam-d](https://sites.google.com/view/vitam-d/)  
**Area**: Human Understanding
**Keywords**: hand-object interaction, tactile perception, contact modeling, deformable objects, SDF reconstruction

## TL;DR

This paper proposes ViTaM-D, a vision-tactile fusion framework that achieves dynamic reconstruction of hand-object interaction for both rigid and deformable objects. The framework introduces a novel Distributed Force-aware Contact Representation (DF-Field) and a two-stage pipeline consisting of visual dynamic tracking followed by force-aware optimization. The HOT dataset is also introduced to address the evaluation gap in deformable object hand-object interaction.

## Background & Motivation

### Problem Definition

Hand-object interaction reconstruction aims to recover the complete geometry and pose of both hands and objects from visual and/or tactile inputs, which is critical for downstream tasks such as VR/AR and robot imitation learning. The core challenges are contact region occlusion and object deformation during interaction, which render visual information alone insufficient.

### Limitations of Prior Work

**Vision-only methods** (gSDF, HOTrack): rely on cross-frame feature fusion to compensate for occlusion, but remain information-deficient in contact regions.

**Tactile fusion methods** (ViTaM): directly fuse tactile and visual data via Transformers, ignoring the information asymmetry between force signals and point cloud features.

**Contact modeling methods** (CPF, TOCH): employ empirical spring-mass systems or optimization approaches, lacking physical plausibility.

**Dataset limitations**: existing hand-object interaction datasets (DexYCB, OakInk) cover only rigid or articulated objects and lack precise tactile data for deformable objects.

### Core Idea

Rather than directly concatenating tactile and visual features, tactile information is integrated into the optimization pipeline through an energy-based contact representation (DF-Field). A coarse reconstruction is first obtained visually, and then tactile information drives force-aware refinement of the hand pose.

## Method

### Overall Architecture

ViTaM-D consists of two stages:
1. **Visual Dynamic Tracking**: online tracking and reconstruction of hand-object interaction via point cloud flow prediction and feature fusion.
2. **Force-aware Optimization**: refinement of hand pose and contact state using distributed tactile array readings and the DF-Field representation.

### Key Designs

#### 1. DF-Field (Distributed Force-aware Contact Representation)

- **Function**: models the hand-object contact state via an energy function.
- **Mechanism**: defines two energy terms:
    - **Relative potential energy**: $E_{ij} = \kappa l_{ij}^2$, where $\kappa$ is correlated with the tactile reading and $l_{ij}$ is the distance between hand and object vertices. When $\kappa > 0$, contact is indicated and the energy tends toward 0.
    - **Barrier energy**: $B_{ij} = -e^{-\kappa}(l_{ij}-\hat{l})^2\log(l_{ij}/\hat{l})$, which generates a repulsive force when the distance falls below threshold $\hat{l}$, preventing interpenetration.
    - Total energy: $E = \sum_i \sum_j (E_{ij} + B_{ij})$
- **Force parameter estimation**: $\kappa_{ij} \sim \overline{\mathcal{M}^j} / l_{ij}$, where $\overline{\mathcal{M}^j}$ is the region-averaged tactile reading.
- **Design Motivation**: the hand is partitioned into 22 regions, each supplied with force readings from its corresponding tactile sensor, balancing physical plausibility and computational efficiency.

#### 2. Visual Dynamic Tracking Network

- **Function**: real-time reconstruction of hand-object interaction from monocular depth image sequences.
- **Mechanism**:
    - **Flow prediction module**: extracts per-point features from the current and previous frames using PointNet++, predicts point cloud flow $f_{t-1 \to t}$, and fuses static features with correspondence features via a Transformer.
    - **Object decoder**: scatters fused features into a volume, processes them with a 3D-UNet, predicts SDF values via MLP, and extracts meshes using Marching Cubes.
    - **Hand decoder**: estimates joint positions via a voting mechanism based on the MANO parametric model, then recovers pose parameters through inverse kinematics.
    - **Contact constraint**: for sampled points in contact regions ($c_x = 1$), SDF values are forced toward 0: $\mathcal{L}_C = \sum_{x \in \mathcal{X}} s_x \cdot \mathbb{1}_{c_x=1}$

#### 3. Force-aware Hand Pose Optimization

- **Function**: refines hand pose using DF-Field.
- **Mechanism**: minimizes the total DF-Field energy as the objective, jointly with a pose plausibility constraint $\mathcal{L}_r$ and a deviation penalty $\mathcal{L}_o$:
  $$\theta^* = \arg\min_\theta (E + \mathcal{L}_r + \mathcal{L}_o)$$
  Optimization is performed with the Adam optimizer for 100 iterations, taking approximately 3.5 seconds per frame.
- **Design Motivation**: hand poses obtained from visual tracking lack accuracy in contact regions, necessitating supplemental tactile information.

### Loss & Training

Total loss: $\mathcal{L} = \lambda_f \mathcal{L}_{flow} + \lambda_S \mathcal{L}_{SDF} + \lambda_H \mathcal{L}_{Hand} + \lambda_C \mathcal{L}_C$

- $\lambda_f = 0.01$, $\lambda_S = 0.5$, $\lambda_H = 1$, $\lambda_C = 0.05$
- Flow loss uses Chamfer Distance; SDF loss uses L1; hand joint loss uses L2.
- Trained for 100 epochs with a learning rate of 1e-4; approximately 15 hours on an Nvidia A40.

## Key Experimental Results

### Main Results

**Quantitative results on DexYCB and HOT datasets**

| Method | Dataset | IoU↑ | CD(mm)↓ | MPJPE(mm)↓ | PD(mm)↓ | CIoU↑ |
|--------|---------|------|---------|------------|---------|-------|
| gSDF (RGB) | DexYCB | 86.8 | 13.4 | 14.4 | 8.9 | 31.3 |
| HOTrack | DexYCB | 88.2 | 10.2 | 25.7 | 12.3 | 28.5 |
| **Ours (w/o Force)** | DexYCB | **90.1** | **9.6** | **13.2** | 9.9 | 35.4 |
| ViTaM | HOT | 80.5 | 11.5 | 15.1 | 10.6 | 28.5 |
| **Ours (w/o Force)** | HOT | 81.0 | 10.9 | 13.6 | 10.7 | 29.8 |
| **Ours (w. Force Opt.)** | HOT | - | - | **11.3** | **7.3** | **40.3** |

### Ablation Study

**Ablation on contact constraint strategies (HOT dataset)**

| Configuration | IoU↑ | CD↓ | CIoU↑ | MPJPE↓ |
|---------------|------|-----|-------|--------|
| No contact constraint | 75.9 | 12.8 | 25.3 | 12.0 |
| GT contact | 81.2 | 11.2 | 29.2 | 11.9 |
| Tactile readings | 81.0 | 10.9 | 29.8 | 11.9 |
| PointNet prediction | 78.3 | 12.1 | 27.6 | 12.1 |

**Ablation on force representation (HOT dataset)**

| Configuration | MPJPE↓ | PD↓ | CIoU↑ |
|---------------|--------|-----|-------|
| No force optimization | 13.6 | 10.7 | 29.8 |
| Fixed force value | 12.9 | 8.5 | 36.8 |
| **Tactile force optimization** | **11.3** | **7.3** | **40.3** |

### Key Findings

- Force-aware optimization reduces penetration depth from 10.7 mm to 7.3 mm and improves contact IoU from 29.8% to 40.3%, demonstrating substantial gains.
- Even without tactile data, DF-Field with fixed force values improves results (CIoU on DexYCB: 35.4→39.7), validating the effectiveness of the energy-based representation itself.
- Contact constraints provide notable benefits for object reconstruction (IoU: 75.9→81.0), and tactile readings achieve performance close to GT contact labels.
- Models trained on simulation transfer directly to real-scene reconstruction (plush toy experiment).

## Highlights & Insights

1. **Correct integration of tactile information**: rather than naively concatenating force data with visual features, the framework uses physics-based energy functions during the optimization stage, avoiding the information asymmetry problem.
2. **Flexibility of DF-Field**: when tactile data is available, it is used directly; otherwise, empirical force values can be set, making the representation compatible with vision-only pipelines.
3. **Value of the HOT dataset**: FEM-based contact modeling in the ZeMa simulator provides accurate, interpenetration-free ground truth, filling the evaluation gap for deformable object hand-object interaction.
4. **Practicality of the two-stage design**: the visual tracking stage runs in real time, while force optimization can be applied offline on demand.

## Limitations & Future Work

1. **Force optimization takes 3.5 seconds per frame**, far from real time, limiting online applications.
2. **Accessibility of tactile sensors**: high-density distributed tactile gloves (e.g., ViTaM) remain uncommon in practice.
3. **HOT dataset is simulation-based**: the domain gap with real tactile data warrants further investigation.
4. **Single-hand, single-object only**: the framework has not been extended to two-hand or multi-object scenarios.
5. **Material parameters of deformable objects must be predefined**: Young's modulus, Poisson's ratio, and similar parameters are difficult to obtain in real-world settings.

## Related Work & Insights

- gSDF uses Transformer + SDF to model complex hand-object interaction and serves as a strong baseline on DexYCB.
- CPF employs an empirical spring-mass system for contact optimization; DF-Field offers more rigorous physical modeling.
- TOCH contributes spatiotemporal consistency optimization, but force-aware optimization yields superior results (CIoU: 34.1 vs. 40.3).
- The ZeMa simulator provides high-fidelity FEM contact modeling and serves as the foundation of the HOT dataset.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The energy-based modeling in DF-Field is novel; integrating tactile information through optimization rather than feature concatenation is the right direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated on both DexYCB and HOT datasets with well-designed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, the pipeline is thoroughly described, and figures are of high quality.
- **Value**: ⭐⭐⭐⭐ — The HOT dataset and DF-Field representation offer long-term value to the hand-object interaction community, though the requirement for tactile devices limits practical accessibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Contact-Aware Refinement of Human Pose Pseudo-Ground Truth via Bioimpedance Sensing](contact-aware_refinement_of_human_pose_pseudo-ground_truth_via_bioimpedance_sens.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](../../CVPR2026/human_understanding/textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)
- [\[NeurIPS 2025\] HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion](../../NeurIPS2025/human_understanding/hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)
- [\[ICCV 2025\] NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction](ngd_neural_gradient_based_deformation_for_monocular_garment_reconstruction.md)
- [\[NeurIPS 2025\] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](../../NeurIPS2025/human_understanding/part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning
description: >-
  [AAAI 2026][Autonomous Driving][End-to-End Planning] This paper proposes DriveSuprim, which addresses three key bottlenecks in selection-based end-to-end planning — difficulty distinguishing similar trajectories…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "End-to-End Planning"
  - "Trajectory Selection"
  - "Coarse-to-Fine"
  - "Data Augmentation"
  - "Self-Distillation"
date: 2026-05-08
content_hash: f611ac5767416847
---

# DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning

**Conference**: AAAI 2026
**arXiv**: [2506.06659](https://arxiv.org/abs/2506.06659)
**Code**: [Available](https://github.com/William-Yao-2000/DriveSuprim)
**Area**: Autonomous Driving
**Keywords**: End-to-End Planning, Trajectory Selection, Coarse-to-Fine, Data Augmentation, Self-Distillation

## TL;DR

This paper proposes DriveSuprim, which addresses three key bottlenecks in selection-based end-to-end planning — difficulty distinguishing similar trajectories, directional bias, and hard-label instability — through a coarse-to-fine trajectory selection paradigm, rotation-based data augmentation, and a self-distillation soft-label framework, achieving state-of-the-art performance on NAVSIM v1/v2 and Bench2Drive.

## Background & Motivation

End-to-end autonomous driving planning methods fall into two main categories:

- **Regression-based methods**: Directly predict a single expert trajectory without explicit safety evaluation of candidates.
- **Selection-based methods**: Generate and score multiple trajectory candidates, then select the optimal one. Oracle experiments show that selection-based methods can theoretically surpass human demonstrations (Top-256 PDMS = 98.7 vs. human 94.8).

However, selection-based methods face three critical bottlenecks:

**Difficulty distinguishing hard negatives**: During training, the vast majority of candidates are obviously unsafe "easy negatives," providing insufficient fine-grained discriminative supervision. The model struggles to make optimal choices among trajectories that appear plausible but differ subtly.

**Directional bias**: In NAVSIM, only 18% of GT trajectories involve turns exceeding 30°. The straight-driving-dominated data distribution leads to poor model performance in turning scenarios.

**Hard-label instability**: Threshold-based binary safety labels make the model overly sensitive to minor score changes, where small differences can flip a safe/unsafe judgment.

## Method

### Overall Architecture

DriveSuprim adopts a selection-based planning paradigm with three core innovative components:

1. **Coarse-to-Fine Trajectory Selection**: Coarse filtering followed by fine re-ranking.
2. **Rotation-based Data Augmentation**: Synthesizing turning scenarios to mitigate directional bias.
3. **Self-Distillation Framework**: Using a teacher model to generate soft labels for stable training.

Architecture pipeline: Image encoder extracts BEV features → Trajectory encoder encodes candidate vocabulary → Trajectory Decoder (coarse filtering) → Refinement Decoder (fine re-ranking) → Output optimal trajectory.

### Key Designs

**1. Coarse-to-Fine Trajectory Selection**

**Coarse filtering stage**: Similar to Hydra-MDP, trajectory features and image features interact via Transformer Decoder cross-attention. Multiple prediction heads regress L2 distance and rule-based metric scores to select Top-K candidates:

$$g_j = \text{TransDec}(\mathcal{E}_{img}, f_j), \quad s_j^{(m)} = \text{Sigmoid}(\text{head}^{(m)}(g_j))$$

**Fine re-ranking stage**: For the filtered candidates (which contain a high proportion of hard negatives), a Refinement Decoder performs multi-layer fine-grained scoring:

$$\{h_{j,l}\}_{l=1}^{n_{ref}} = \text{RefineDec}(\mathcal{E}_{img}, g_j)$$

Each layer output is supervised by a loss, and the highest-scoring candidate at the final layer is taken as the prediction. Since only a small number of candidates are re-ranked, computational overhead remains manageable.

**2. Rotation-based Augmentation**

To address the straight-driving data bias, an end-to-end rotation augmentation pipeline is designed:

- Randomly sample a rotation angle $\theta \sim U[-\Theta, \Theta]$
- Concatenate the original FOV and laterally extended view images into a "pseudo-panoramic" image
- Crop the input image via a sliding window according to $\theta$, simulating ego-vehicle rotation
- Apply a corresponding 2D rotation transformation (angle $-\theta$, rotated about the initial position) to the GT trajectory, preserving world coordinates

This approach synthesizes more challenging turning scenarios, enabling the model to accurately select trajectories regardless of vehicle heading.

**3. Self-Distillation Soft-Label Framework**

An EMA-updated teacher model generates soft labels to replace hard binary labels:

$$\hat{y}_i^{(m)} = y_i^{(m)} + \text{clip}(s_{i,\text{teacher}}^{(m)} - y_i^{(m)}, -\delta_m, \delta_m)$$

- The teacher receives only original (unaugmented) data to generate scores as soft labels.
- The student receives augmented data with noise.
- The clipping threshold $\delta_m$ controls the allowed deviation of teacher output from GT.
- At inference, the teacher model is used to output the planned trajectory.

### Loss & Training

The total loss consists of three components:

$$L = L_{ori} + L_{aug} + L_{soft}$$

- $L_{ori} = L_{coarse} + L_{refine}$: Coarse filtering and fine re-ranking losses on original data.
- $L_{aug}$: Loss on augmented data (isomorphic to $L_{ori}$).
- $L_{soft}$: Distillation loss based on teacher soft labels.

The coarse filtering loss includes an imitation loss and a BCE classification loss.

## Key Experimental Results

### Main Results

**Table 1: NAVSIM v1 Evaluation**

| Method | Backbone | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
|--------|----------|-----|------|-----|------|----|----|
| Hydra-MDP | ResNet34 | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 |
| DiffusionDrive | ResNet34 | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| **DriveSuprim** | ResNet34 | 97.8 | 97.3 | 86.7 | 93.6 | 100 | **89.9 (+1.8)** |
| Hydra-MDP | ViT-L | 98.4 | 97.7 | 85.0 | 94.5 | 100 | 89.9 |
| **DriveSuprim** | ViT-L | 98.6 | 98.6 | 91.3 | 95.5 | 100 | **93.5 (+3.6)** |

**Table 2: Bench2Drive Evaluation**

| Method | DS↑ | SR↑ | Eff.↑ | Comf.↑ |
|--------|-----|-----|-------|--------|
| DriveAdapter | 64.22 | 33.08 | 70.22 | 16.01 |
| AutoVLA | 78.84 | 57.73 | 146.93 | 39.33 |
| **DriveSuprim** | **83.02** | **60.00** | **238.78** | 20.89 |

### Ablation Study

- Removing the Refinement Decoder leads to a ~1–2% PDMS drop, confirming its importance for discriminating hard negatives.
- Removing rotation augmentation causes significant performance degradation in turning scenarios, particularly on the EP (Ego Progress) metric.
- Removing self-distillation results in unstable training; soft labels contribute substantially to mitigating hard-boundary sensitivity.
- Increasing Top-K (number of re-ranking candidates) yields diminishing and eventually negative returns, as more candidates introduce additional noise.

### Key Findings

1. Oracle experiments demonstrate that the theoretical upper bound of selection-based methods far exceeds human performance (Top-256 PDMS 98.7 vs. human 94.8); the key challenge is how to approach this upper bound.
2. With a ViT-L backbone on NAVSIM v1, DriveSuprim achieves a PDMS of 93.5, leaving approximately 5% room for improvement relative to the Oracle upper bound of 98.7.
3. Rotation augmentation not only improves turning-scenario performance but also enhances overall robustness, as the model must learn to handle more diverse visual inputs.
4. The clipping mechanism $\delta_m$ in self-distillation is critical — it prevents excessive propagation of teacher errors.

## Highlights & Insights

- The **coarse-to-fine paradigm** is a classic concept in computer vision, but its application to trajectory selection is refreshing: by narrowing the search space, the model can focus on hard negatives.
- The rotation augmentation design is elegant: simulating ego-vehicle rotation directly at the image level is simpler and more efficient than performing transformations in 3D space.
- The Oracle analysis clearly demonstrates the substantial potential of selection-based methods and provides a well-defined direction for future research.
- State-of-the-art performance (93.5 PDMS) is achieved without additional training data, demonstrating the value of methodological innovation.

## Limitations & Future Work

- The trajectory vocabulary is a predefined fixed set, limiting the diversity and granularity of selectable trajectories.
- Rotation augmentation only simulates yaw rotation, without accounting for more complex pose changes such as pitch and roll.
- The self-distillation hyperparameter $\delta_m$ requires separate tuning for different metrics.
- The method lacks targeted designs for extreme scenarios (e.g., emergency braking, anomalous traffic participants).

## Related Work & Insights

- **Hydra-MDP** is a representative selection-based method; DriveSuprim builds upon it by introducing the coarse-to-fine mechanism and augmentation strategies.
- The coarse-to-fine idea draws inspiration from iterative refinement methods in optical flow and detection (e.g., Deformable DETR, RAFT).
- Self-distillation adapts the well-established EMA teacher approach from semi-supervised learning, following the same lineage as Mean Teacher.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 5 |
| Overall | 4.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [\[AAAI 2026\] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)

</div>

<!-- RELATED:END -->

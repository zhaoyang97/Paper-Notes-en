---
title: >-
  [Paper Note] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning
description: >-
  [AAAI 2026][Autonomous Driving][End-to-End Planning] DriveSuprim is proposed to address key challenges in selection-based end-to-end planning—difficulty in distinguishing similar trajectories, orientation bias, and instability of hard labels—via a coarse-to-fine trajectory filtering paradigm, rotation-based data augmentation, and a self-distillation soft label framework, achieving state-of-the-art (SOTA) performance on NAVSIM v1/v2 and Bench2Drive.
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "End-to-End Planning"
  - "Trajectory Selection"
  - "Coarse-to-Fine"
  - "Data Augmentation"
  - "Self-Distillation"
date: 2026-05-08
content_hash: 0c0319c09fd2298b
---

# DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning

**Conference**: AAAI 2026  
**arXiv**: [2506.06659](https://arxiv.org/abs/2506.06659)  
**Code**: [Available](https://github.com/William-Yao-2000/DriveSuprim)  
**Area**: Autonomous Driving  
**Keywords**: End-to-End Planning, Trajectory Selection, Coarse-to-Fine, Data Augmentation, Self-Distillation

## TL;DR

DriveSuprim is proposed to address key challenges in selection-based end-to-end planning—difficulty in distinguishing similar trajectories, orientation bias, and instability of hard labels—via a coarse-to-fine trajectory filtering paradigm, rotation-based data augmentation, and a self-distillation soft label framework, achieving state-of-the-art (SOTA) performance on NAVSIM v1/v2 and Bench2Drive.

## Background & Motivation

End-to-end autonomous driving planning methods are mainly categorized into two groups:

- **Regression-based methods**: Directly predict a single expert trajectory, offering no explicit mechanism to evaluate trajectory safety.
- **Selection-based methods**: Generate and score multiple candidate trajectories to select the optimal one. Oracle experiments demonstrate that the upper bound of selection-based methods even surpasses human demonstrations (Top-256 PDMS=98.7 vs. Human 94.8).

However, selection-based methods face three key bottlenecks:

**Difficulty in distinguishing hard negatives**: The vast majority of candidates during training are obviously unsafe "easy negatives." Consequently, the model receives insufficient fine-grained supervisory signals, making it difficult to make optimal choices among plausible trajectories with subtle differences.

**Orientation bias**: In NAVSIM, only 18% of ground-truth (GT) trajectories involve turns exceeding 30°. This straight-line-dominated data distribution leads to poor model performance in cornering scenarios.

**Instability of hard labels**: Threshold-based binary safety labels make the model overly sensitive to minor score fluctuations, where slight differences can flip the safe/unsafe decision.

## Method

### Overall Architecture

DriveSuprim adopts a selection-based planning paradigm and consists of three core innovative components:

1. **Coarse-to-Fine Trajectory Selection**: Coarse screening followed by fine ranking.
2. **Rotation-based Data Augmentation**: Synthesizes turning scenarios to mitigate orientation bias.
3. **Self-Distillation Framework**: Utilizes a teacher model to generate soft labels, stabilizing training.

Architectural workflow: Image encoder extracts BEV features $\rightarrow$ Trajectory encoder encodes candidate vocabulary $\rightarrow$ Trajectory Decoder (Coarse Filtering) $\rightarrow$ Refinement Decoder (Fine Ranking) $\rightarrow$ Output optimal trajectory.

### Key Designs

**1. Coarse-to-Fine Trajectory Selection**

**Coarse Filtering Stage**: Similar to Hydra-MDP, trajectory features and image features interact via cross-attention in a Transformer Decoder. Multiple prediction heads regress the $L_2$ distance and rule-metric scores to select the Top-$K$ candidates:

$$g_j = \text{TransDec}(\mathcal{E}_{img}, f_j), \quad s_j^{(m)} = \text{Sigmoid}(\text{head}^{(m)}(g_j))$$

**Fine Ranking Stage**: For the filtered candidates (which contain a substantial amount of hard negatives), a Refinement Decoder is used to perform multi-layer fine-grained scoring:

$$\{h_{j,l}\}_{l=1}^{n_{ref}} = \text{RefineDec}(\mathcal{E}_{img}, g_j)$$

The output of each layer is supervised by a loss, and the candidate with the highest score in the final layer is selected as the prediction. Since the fine ranking stage only processes a small number of candidates, the computational overhead remains manageable.

**2. Rotation-based Augmentation**

To address the data bias dominated by straight-line driving, an end-to-end rotation augmentation pipeline is designed:

- Randomly sample a rotation angle $\theta \sim U[-\Theta, \Theta]$
- Concatenate the original FOV and adjacent side-view images into a "pseudo-panoramic image"
- Crop the input image via a sliding window based on $\theta$ to simulate ego-vehicle rotation
- Apply a corresponding 2D rotation transform to the GT trajectory (by angle $-\theta$, rotating around the starting position) to maintain world coordinate consistency

This method synthesizes more challenging turning scenarios, enabling the model to precisely select trajectories regardless of the vehicle's heading.

**3. Self-Distillation Framework**

An EMA-updated teacher model is utilized to generate soft labels, replacing the hard binary labels:

$$\hat{y}_i^{(m)} = y_i^{(m)} + \text{clip}(s_{i,\text{teacher}}^{(m)} - y_i^{(m)}, -\delta_m, \delta_m)$$

- The teacher only receives raw data to generate scores as soft labels.
- The student receives noise-augmented data.
- The clipping threshold $\delta_m$ controls the deviation range of the teacher's output from the GT.
- During inference, the teacher model is used to output the planned trajectory.

### Loss & Training

The total loss consists of three components:

$$L = L_{ori} + L_{aug} + L_{soft}$$

- $L_{ori} = L_{coarse} + L_{refine}$: Coarse filtering and fine ranking losses on the original data.
- $L_{aug}$: Loss on augmented data (isomorphic to $L_{ori}$).
- $L_{soft}$: Distillation loss based on teacher soft labels.

The coarse filtering loss includes imitation loss and BCE classification loss.

## Key Experimental Results

### Main Results

**Table 1: Evaluation on NAVSIM v1**

| Method | Backbone | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
|------|----------|-----|------|-----|------|----|----|
| Hydra-MDP | ResNet34 | 98.3 | 96.0 | 78.7 | 94.6 | 100 | 86.5 |
| DiffusionDrive | ResNet34 | 98.2 | 96.2 | 82.2 | 94.7 | 100 | 88.1 |
| **DriveSuprim** | ResNet34 | 97.8 | 97.3 | 86.7 | 93.6 | 100 | **89.9 (+1.8)** |
| Hydra-MDP | ViT-L | 98.4 | 97.7 | 85.0 | 94.5 | 100 | 89.9 |
| **DriveSuprim** | ViT-L | 98.6 | 98.6 | 91.3 | 95.5 | 100 | **93.5 (+3.6)** |

**Table 2: Evaluation on Bench2Drive**

| Method | DS↑ | SR↑ | Eff.↑ | Comf.↑ |
|------|-----|-----|-------|--------|
| DriveAdapter | 64.22 | 33.08 | 70.22 | 16.01 |
| AutoVLA | 78.84 | 57.73 | 146.93 | 39.33 |
| **DriveSuprim** | **83.02** | **60.00** | **238.78** | 20.89 |

### Ablation Study

- Removing the Refinement Decoder $\rightarrow$ PDMS drops by approximately 1-2%, confirming the importance of fine ranking for hard negative discrimination.
- Removing rotation-based augmentation $\rightarrow$ Performance in turning scenarios drops significantly, with a particularly pronounced degradation in the EP (Ego Progress) metric.
- Removing self-distillation $\rightarrow$ Instability in training occurs, highlighting the crucial contribution of soft labels in mitigating sensitivity to hard boundaries.
- Increasing Top-$K$ (number of refined candidates) $\rightarrow$ Performance first rises and then falls, as excessive candidates introduce more noise.

### Key Findings

1. Oracle experiments show that the theoretical upper bound of selection-based methods far exceeds human performance (Top-256 PDMS of 98.7 vs. 94.8 for humans); the key lies in how to approach this upper limit.
2. On NAVSIM v1, the PDMS of the ViT-L backbone reaches 93.5, leaving a gap of approximately 5% to the Oracle upper bound of 98.7.
3. Rotation augmentation not only improves performance in turning scenarios but also enhances overall robustness, as the model must learn to handle more diverse visual inputs.
4. The clipping mechanism $\delta_m$ in self-distillation is critical—it prevents the propagation of errors from the teacher.

## Highlights & Insights

- The **coarse-to-fine paradigm** is a classic concept in computer vision, but its application to trajectory selection is refreshing, narrowing the search space to focus on hard negatives.
- The design of rotation augmentation is ingenious: simulating ego-vehicle rotation directly at the image level is simpler and more efficient than performing transformations in 3D space.
- The Oracle analysis clearly demonstrates the immense potential of selection-based methods, providing a clear direction for future research.
- Achieving SOTA (93.5 PDMS) without using external training data demonstrates the value of methodological innovation.

## Limitations & Future Work

- The trajectory vocabulary is a predefined fixed set, which limits the diversity and fine-grained resolution of candidate trajectories.
- Rotation-based augmentation only simulates yaw rotation, neglecting more complex pose changes such as pitch and roll.
- The hyperparameter $\delta_m$ in self-distillation requires separate tuning for different metrics.
- There is a lack of targeted designs for safety-critical edge cases (e.g., emergency braking, anomalous road users).

## Related Work & Insights

- **Hydra-MDP** represents typical selection-based methods, and DriveSuprim builds upon it by adding the coarse-to-fine mechanism and augmentation strategies.
- The coarse-to-fine concept is inspired by iterative optimization in fields such as optical flow and object detection (e.g., Deformable DETR, RAFT).
- The self-distillation scheme leverages mature practices utilizing an EMA-updated teacher in semi-supervised learning, sharing the same lineage as Mean Teacher.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
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
- [\[ICLR 2026\] RAP: 3D Rasterization Augmented End-to-End Planning](../../ICLR2026/autonomous_driving/rap_3d_rasterization_augmented_end-to-end_planning.md)
- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [\[ICLR 2026\] VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning](../../ICLR2026/autonomous_driving/vadv2_end-to-end_vectorized_autonomous_driving_via_probabilistic_planning.md)

</div>

<!-- RELATED:END -->

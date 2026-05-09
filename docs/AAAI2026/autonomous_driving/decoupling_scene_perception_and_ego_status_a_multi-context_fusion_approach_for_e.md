---
title: >-
  [Paper Note] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving
description: >-
  [AAAI 2026][Autonomous Driving][end-to-end autonomous driving] This paper identifies the architectural root cause of ego-status over-reliance in end-to-end autonomous driving—namely, the premature fusion of ego status within the BEV encoder—and proposes AdaptiveAD, a dual-branch architecture consisting of a scene-driven branch (with ego status removed) and a self-driven branch that independently generate planning decisions. A scene-aware fusion module then adaptively integrates the two branches. Complemented by path attention, BEV unidirectional distillation, and an autoregressive online mapping auxiliary task, AdaptiveAD achieves state-of-the-art planning performance on nuScenes.
tags:
  - AAAI 2026
  - Autonomous Driving
  - end-to-end autonomous driving
  - causal confusion
  - ego-status over-reliance
  - dual-branch architecture
  - multi-context fusion
date: 2026-05-08
content_hash: 2f97ec58f1ddacbc
---

# AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving

**Conference**: AAAI 2026
**arXiv**: [2511.13079](https://arxiv.org/abs/2511.13079)
**Code**: N/A
**Area**: Autonomous Driving
**Keywords**: end-to-end autonomous driving, causal confusion, ego-status over-reliance, dual-branch architecture, multi-context fusion

## TL;DR

This paper identifies the architectural root cause of ego-status over-reliance in end-to-end autonomous driving—namely, the premature fusion of ego status within the BEV encoder—and proposes AdaptiveAD, a dual-branch architecture consisting of a scene-driven branch (with ego status removed) and a self-driven branch that independently generate planning decisions. A scene-aware fusion module then adaptively integrates the two branches. Complemented by path attention, BEV unidirectional distillation, and an autoregressive online mapping auxiliary task, AdaptiveAD achieves state-of-the-art planning performance on nuScenes.

## Background & Motivation

End-to-end autonomous driving models commonly suffer from "inertia-based driving" rather than "vision-based driving" (causal confusion), leading to poor performance in novel or long-tail scenarios.

**Root Cause Analysis**:
- Existing architectures (e.g., UniAD, VAD) inject ego status (vehicle kinematic state) into perception features upstream of the BEV encoder.
- This premature fusion creates an information shortcut: the planning module can bypass complex scene understanding by directly relying on ego status.
- When encountering sudden obstacles at high speed, inertia-driven models produce fatal trajectories.

**Limitations of Existing Mitigation Strategies**:
- Data-level approaches (e.g., balanced sampling): alleviate dataset bias without modifying the internal information flow.
- Regularization (e.g., dropout, contrastive imitation learning): improve feature quality but may exacerbate difficulties in multi-task learning.
- These methods "dress up the inputs" rather than restructuring the decision-making process itself.

## Method

### Overall Architecture

AdaptiveAD adopts a **dual-branch + adaptive fusion** strategy that explicitly decouples scene perception and ego-state inference at the architectural level:

1. **Scene-driven Branch**: Environment-aware reasoning via multi-task learning, with ego status augmentation **deliberately removed** from the BEV encoder.
2. **Self-driven Branch (Planning-only Branch)**: Ego-state reasoning based solely on the planning task, retaining ego status augmentation.
3. **Multi-context Decision Fusion Module**: Adaptively integrates complementary decisions from both branches.

### Key Designs

**(1) Scene-driven Branch**

Built upon the VAD architecture, this branch comprises a BEV encoder, a vectorized scene decoder, and a decision generator. **Core modification**: the BEV query augmentation step (which typically injects ego status) is removed from the BEV encoder, yielding pure environmental BEV features $B_{woes} \in \mathbb{R}^{C \times H_{bev} \times W_{bev}}$.

The vectorized scene decoder converts the dense BEV into sparse agent queries $A$ and map queries $M$. The decision generator initializes multi-modal ego queries $E_{woes}$ and sequentially interacts with $A$, $M$, and $B_{woes}$.

**(2) Self-driven Branch**

The BEV query augmentation operation is retained, producing motion-compensated BEV features $B_{wes}$. This branch omits the explicit scene decoder; the ego query $E_{wes}$ interacts directly with $B_{wes}$. Initial reference points are predicted directly from ego status, reflecting a strong prior for kinematic extrapolation.

**(3) Path Attention**

Replacing standard deformable attention, path attention introduces trajectory-guided semantic sampling:
- A preliminary trajectory is first decoded, and $T$ reference points are uniformly sampled along it.
- Each reference point is assigned an independent attention head, which learns to sample $K$ local features within its neighborhood.
- This process mimics a human driver scanning along the planned route.

$$\text{PathAttn}(E^i, P^i, B) = \sum_{t=1}^T W_t \left[\sum_{k=1}^K a^{i,t,k} W_t' B_{samp}^{i,t,k}\right]$$

Weights are normalized within each head ($\sum_k a^{i,t,k} = 1$), and inter-head feature separation enables simultaneous modeling of long-range context and local detail.

**(4) Multi-context Decision Fusion**

The fusion ego query $E_{fusion}$ is initialized from scene-aware features via global average pooling over the scene BEV:

$$E_{fusion}^{com} = \text{GAP}(B_{woes})$$

A transformer fusion layer then performs context alignment: decisions from both branches are concatenated and passed through self-attention, enabling rich cross-context interaction and intra-context refinement. Subsequently, $E_{fusion}$ adaptively weights the aligned multi-context representation via cross-attention.

**(5) BEV Unidirectional Distillation**

The scene-driven branch may suffer from motion blur due to the absence of ego motion compensation. $B_{wes}$ (with motion compensation) serves as the teacher and $B_{woes}$ as the student:

$$L_{distill} = \alpha L_{distill}^{DF} + \beta L_{distill}^{IK} + \gamma L_{distill}^{IC}$$

This comprises dense feature distillation (agent-guided weighting), inter-keypoint correlation distillation, and inter-channel correlation distillation. Gradients are not back-propagated through the teacher.

**(6) Autoregressive Online Mapping**

A planning-to-perception feedback loop is established: within the overlapping perception region between the predicted trajectory and the ground-truth trajectory, a masked L1 loss is imposed on map instances:

$$L_{autoreg}^{MAP} = \frac{1}{T} \sum_{\tau=1}^T \frac{1}{\|\mathcal{M}\|_1 + \epsilon} \|(\hat{P}_M - P_M) \odot \mathcal{M}\|_1$$

This ensures that the perceived map remains consistent whether the vehicle follows the predicted trajectory or the ground-truth trajectory, thereby alleviating optimization conflicts between the mapping and planning heads.

### Loss & Training

- Distillation and autoregressive loss weights $(\alpha, \beta, \gamma, \delta, \lambda) = (0.01, 0.1, 0.01, 0.01, 0.01)$
- 60 epochs, 32× A100 GPUs, AdamW + CosineAnnealing, batch size 2 per GPU
- Based on the VAD architecture; predicts 3-second trajectories using 2-second history; 60m×30m perception range
- 6 layers of ego-BEV interaction + 6 layers of multi-context fusion

## Key Experimental Results

### Main Results

**Table 1: Open-loop Planning Performance on nuScenes**

| Method | L2 Avg↓ | CR Avg↓ | FPS |
|--------|---------|---------|-----|
| UniAD | 0.73 | 0.61 | 1.8 |
| VAD | 0.61 | 0.28 | 3.4 |
| PPAD | 0.58 | 0.19 | 2.6 |
| SparseDrive | 0.61 | 0.10 | 5.2 |
| BridgeAD | 0.58 | 0.08 | 3.1 |
| **AdaptiveAD** | **0.47** | **0.12** | 3.0 |

Compared with VAD: L2 error reduced by 22%, collision rate reduced by 57%.

**Table 2: Scene Generalization (Straight ST vs. Turning LR)**

| Method | ST L2↓ | LR L2↓ | ST CR↓ | LR CR↓ |
|--------|--------|--------|--------|--------|
| VAD | 0.62 | 0.91 | 0.33 | 0.18 |
| VAD (Turning-nuScenes) | - | 0.92 | - | 0.38 |
| **Ours** | **0.47** | **0.63** | **0.11** | **0.16** |
| **Ours** (Turning-nuScenes) | - | **0.63** | - | **0.28** |

Advantages are particularly pronounced in turning scenarios.

**Table 3: Degree of Ego-Status Dependency**

| Method | Normal L2 | velocity×0.0 L2 | velocity×0.5 L2 | 100m/s L2 |
|--------|-----------|-----------------|-----------------|-----------|
| VAD | 0.61 | 5.54 (+808%) | 3.05 | 14.93 |
| **Ours** | **0.47** | **4.08** (+768%) | **2.41** | **5.06** |

VAD's L2 explodes by 810% when ego velocity is zeroed out; AdaptiveAD under extreme noise (100 m/s) still outperforms normal VAD by 17%.

### Ablation Study

**Table 5: Incremental Component Ablation**

| ID | Dual-branch | BEV Distill | Scene Init | Autoreg. Map | L2 Avg↓ | CR Avg↓ |
|----|-------------|-------------|------------|--------------|---------|---------|
| 1 | - | - | - | - | 0.57 | 0.22 |
| 2 | ✓ | - | - | - | 0.62 | 0.15 |
| 3 | ✓ | ✓ | - | - | 0.58 | 0.08 |
| 4 | ✓ | ✓ | ✓ | - | 0.52 | 0.12 |
| 5 | ✓ | ✓ | ✓ | ✓ | **0.47** | **0.12** |

- Introducing the dual-branch temporarily degrades L2 (motion blur), but significantly reduces CR (improved scene understanding).
- BEV distillation restores L2 and reduces CR by 60% (a critical component).
- Scene-aware initialization further improves L2 by approximately 10%.

### Key Findings

1. **Path attention outperforms deformable attention** (Table 6): CR Avg decreases from 0.15 to 0.12 with identical computational cost.
2. **Components exhibit generalizability** (Table 7): Path Attention and autoregressive online mapping both yield improvements when plugged into UniAD and SparseDrive.
3. **NAVSIM/Bench2Drive closed-loop validation**: Under ego-status noise, AdaptiveAD's PDMS/DS/SR metrics substantially outperform VAD.

## Highlights & Insights

1. **Architectural-level solution**: Unlike data- or regularization-based mitigation, the paper directly severs the ego-status shortcut at the information-flow level.
2. **Clear complementary logic of dual branches**: The scene branch provides environment-aware decisions (critical for complex scenarios); the ego branch provides inertial motion priors (efficient for simple scenarios); the fusion module adaptively weights them according to scene complexity.
3. **Elegant BEV distillation design**: Addresses the motion-blur side effect introduced by decoupling; the teacher–student paradigm is natural and effective.
4. **Causal consistency of autoregressive mapping**: Establishes a planning-to-perception feedback loop inspired by world model thinking.

## Limitations & Future Work

1. Evaluation is primarily conducted in the nuScenes open-loop setting; the depth of closed-loop validation is insufficient.
2. The dual-branch architecture increases model complexity, reducing FPS from 3.4 (VAD) to 3.0.
3. Completely removing ego status from the scene-driven branch may be overly aggressive; in certain scenarios (e.g., highway cruising at constant speed), ego status provides useful information.
4. Fusion with LiDAR is not discussed.
5. Extending the decoupling idea to generative world models is a promising direction for future exploration.

## Related Work & Insights

- **Planning-oriented end-to-end**: UniAD (CVPR 2023) → VAD (ICLR 2024) → SparseDrive → BridgeAD
- **Causal confusion mitigation**: EgoStatus analysis (Li 2024), PLUTO contrastive imitation learning (Cheng 2024)
- **Multi-sensor fusion inspiration**: The conceptual shift from sensor fusion to decision context fusion represents a notable innovation.
- **World model feedback**: Autoregressive ideas from Think2Drive and Vista
- **Inspiration**: The decoupling + adaptive fusion paradigm is generalizable to multimodal decision-making and human–machine cooperative driving.

## Rating

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | ★★★★★ | Architectural-level resolution of causal confusion; dual-branch decoupling is pioneering |
| Technical Depth | ★★★★☆ | Complete combination of path attention + distillation + autoregressive mapping |
| Experimental Thoroughness | ★★★★★ | Open-loop + closed-loop + ego-noise stress test + scene-category evaluation |
| Writing Quality | ★★★★★ | Precise problem formulation, compelling motivation, clear structure |
| Value | ★★★★☆ | Plug-and-play components (validated), though dual-branch increases complexity |

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [\[AAAI 2026\] Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception](rethinking_the_spatio-temporal_alignment_of_end-to-end_3d_perception.md)

<!-- RELATED:END -->

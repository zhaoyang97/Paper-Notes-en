---
title: >-
  [Paper Note] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving
description: >-
  [AAAI 2026 Oral][Autonomous Driving][End-to-End Autonomous Driving] This work identifies the architectural root cause of over-reliance on ego status in end-to-end autonomous driving (premature fusion of ego status within the BEV encoder). It proposes AdaptiveAD, a dual-branch architecture consisting of a scene-driven branch (with ego status omitted) and a planning-only branch to generate decisions independently. These decisions are subsequently integrated adaptively via a sce…
tags:
  - "AAAI 2026 Oral"
  - "Autonomous Driving"
  - "End-to-End Autonomous Driving"
  - "Causal Confusion"
  - "Ego-status Over-reliance"
  - "Dual-branch Architecture"
  - "Multi-context Fusion"
date: 2026-05-08
content_hash: 4cb33df7a314d104
---

# AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.13079](https://arxiv.org/abs/2511.13079)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: End-to-End Autonomous Driving, Causal Confusion, Ego-status Over-reliance, Dual-branch Architecture, Multi-context Fusion

## TL;DR

This work identifies the architectural root cause of over-reliance on ego status in end-to-end autonomous driving (premature fusion of ego status within the BEV encoder). It proposes AdaptiveAD, a dual-branch architecture consisting of a scene-driven branch (with ego status omitted) and a planning-only branch to generate decisions independently. These decisions are subsequently integrated adaptively via a scene-aware fusion module, working in tandem with path attention, unidirectional BEV distillation, and an autoregressive online mapping auxiliary task to achieve state-of-the-art open-loop planning performance on nuScenes.

## Background & Motivation

End-to-end autonomous driving models commonly suffer from "driving by inertia" rather than "driving by vision" (causal confusion), leading to poor performance in unseen or long-tail scenarios.

**Root Cause Analysis**:
- Existing architectures (such as UniAD and VAD) fuse ego status (the vehicle's kinematic state) into perception features upstream of the BEV encoder.
- This premature fusion creates an information shortcut: the planning module can directly rely on the ego status and bypass complex environmental understanding.
- When encountering sudden obstacles at high speeds, these "inertia-driven" models generate highly dangerous trajectories.

**Limitations of Prior Work**:
- Data-level mitigation (such as balanced sampling) addresses dataset bias but fails to alter the internal information flow of the model.
- Regularization (such as dropout and contrastive imitation learning) improves feature representation but can exacerbate the optimization difficulty in multi-task learning.
- These methods mostly decorate inputs rather than reconstructing the decision-making process itself.

## Method

### Overall Architecture

AdaptiveAD employs a **dual-branch + adaptive fusion** strategy to explicitly decouple scene perception and ego status at the architectural level:

1. **Scene-driven Branch**: Environmental perception reasoning based on multi-task learning, which **deliberately removes** the ego status enhancement from the BEV encoder.
2. **Planning-only Branch**: Ego-status reasoning based solely on the planning task, retaining the ego status enhancement.
3. **Multi-context Decision Fusion Module**: Adaptively integrates the complementary decisions from both branches.

### Key Designs

**(1) Scene-driven Branch**

Based on the VAD architecture, this branch includes a BEV encoder, a vectorized scene decoder, and a decision generator. **Core Modification**: The BEV query enhancement step (which typically injects ego status) is removed from the BEV encoder to yield pure environment BEV features $B_{woes} \in \mathbb{R}^{C \times H_{bev} \times W_{bev}}$.

The vectorized scene decoder converts the dense BEV representation into sparse agent queries $A$ and map queries $M$. The decision generator initializes a multi-modal ego query $E_{woes}$ and sequentially interacts with $A$, $M$, and $B_{woes}$.

**(2) Planning-only Branch**

This branch retains the BEV query enhancement operation to generate motion-compensated BEV features $B_{wes}$. Bypassing the explicit scene decoder, the ego query $E_{wes}$ directly interacts with $B_{wes}$. The initial reference points are directly predicted from the ego status, representing a strong prior of motion extrapolation.

**(3) Path Attention**

Replacing standard deformable attention, this introduces trajectory-guided semantic sampling:
- First decode the initial trajectory, then sample $T$ reference points uniformly along the path.
- Each reference point is assigned an independent attention head to learn and sample $K$ local features in its neighborhood.
- This mimics a human driver scanning along the planned route.

$$\text{PathAttn}(E^i, P^i, B) = \sum_{t=1}^T W_t [\sum_{k=1}^K a^{i,t,k} W_t' B_{samp}^{i,t,k}]$$

The weights are normalized within each head ($\sum_k a^{i,t,k} = 1$), utilizing feature separation across heads to model both long-range context and local details simultaneously.

**(4) Multi-context Decision Fusion**

The fused ego query $E_{fusion}$ is first initialized with scene awareness by applying Global Average Pooling (GAP) to the scene BEV representation:

$$E_{fusion}^{com} = \text{GAP}(B_{woes})$$

Then, context alignment is performed via a transformer fusion layer: the decisions from both branches are concatenated and subjected to self-attention, facilitating rich cross-context interaction and intra-context refinement. Subsequently, $E_{fusion}$ adaptively aggregates weights from the aligned multi-context representation via cross-attention.

**(5) Unidirectional BEV Distillation**

The scene-driven branch may suffer from motion blur due to the lack of ego motion compensation. Using $B_{wes}$ (with motion compensation) as the teacher and $B_{woes}$ as the student:

$$L_{distill} = \alpha L_{distill}^{DF} + \beta L_{distill}^{IK} + \gamma L_{distill}^{IC}$$

This includes dense feature distillation (weighted by agent guidance), inter-keypoint correlation, and inter-channel correlation distillation. Gradients are not backpropagated to the teacher.

**(6) Autoregressive Online Mapping**

Establishes a planning-to-perception feedback loop: within the overlapping perception region of the predicted and ground-truth (GT) trajectories, a masked L1 loss is imposed on map instances:

$$L_{autoreg}^{MAP} = \frac{1}{T} \sum_{\tau=1}^T \frac{1}{\|\mathcal{M}\|_1 + \epsilon} \|(\hat{P}_M - P_M) \odot \mathcal{M}\|_1$$

This ensures that the perceived map remains consistent when following either the predicted trajectory or the GT trajectory, mitigating optimization conflicts between the mapping and planning heads.

### Loss & Training

- Distillation and autoregressive loss weights: $(\alpha, \beta, \gamma, \delta, \lambda) = (0.01, 0.1, 0.01, 0.01, 0.01)$
- 60 epochs, 32×A100 GPUs, AdamW + CosineAnnealing, batch size 2/GPU
- Based on the VAD architecture, predicting a 3-second trajectory using 2 seconds of historical data, with a 60m×30m perception range
- 6 layers of ego-BEV interaction + 6 layers of multi-context fusion

## Key Experimental Results

### Main Results

**Table 1: nuScenes Open-loop Planning Performance**

| Method | L2 Avg↓ | CR Avg↓ | FPS |
|------|---------|---------|-----|
| UniAD | 0.73 | 0.61 | 1.8 |
| VAD | 0.61 | 0.28 | 3.4 |
| PPAD | 0.58 | 0.19 | 2.6 |
| SparseDrive | 0.61 | 0.10 | 5.2 |
| BridgeAD | 0.58 | 0.08 | 3.1 |
| **AdaptiveAD** | **0.47** | **0.12** | 3.0 |

Compared to VAD: L2 error is reduced by 22%, and the collision rate is reduced by 57%.

**Table 2: Scene Generalization Capability (Straight ST vs. Turning LR)**

| Method | ST L2↓ | LR L2↓ | ST CR↓ | LR CR↓ |
|------|--------|--------|--------|--------|
| VAD | 0.62 | 0.91 | 0.33 | 0.18 |
| VAD (Turning-nuScenes) | - | 0.92 | - | 0.38 |
| **Ours** | **0.47** | **0.63** | **0.11** | **0.16** |
| **Ours** (Turning-nuScenes) | - | **0.63** | - | **0.28** |

The advantage is particularly significant in turning scenarios.

**Table 3: Degree of Ego-Status Dependence**

| Method | Normal L2 | velocity×0.0 L2 | velocity×0.5 L2 | 100m/s L2 |
|------|---------|-----------------|-----------------|-----------|
| VAD | 0.61 | 5.54 (+808%) | 3.05 | 14.93 |
| **Ours**| **0.47** | **4.08** (+768%) | **2.41** | **5.06** |

VAD's L2 explodes by over 800% when the ego velocity is set to zero. Under extreme noise (100 m/s), AdaptiveAD still outperforms VAD's normal baseline by 17%.

### Ablation Study

**Table 5: Step-by-Step Component Ablation Study**

| ID | Dual-branch | BEV Distillation | Scene Init | Autoregressive Mapping | L2 Avg↓ | CR Avg↓ |
|----|--------|---------|-----------|-----------|---------|---------|
| 1 | - | - | - | - | 0.57 | 0.22 |
| 2 | ✓ | - | - | - | 0.62 | 0.15 |
| 3 | ✓ | ✓ | - | - | 0.58 | 0.08 |
| 4 | ✓ | ✓ | ✓ | - | 0.52 | 0.12 |
| 5 | ✓ | ✓ | ✓ | ✓ | **0.47** | **0.12** |

- Introducing the dual-branch temporarily degrades L2 (due to motion blur) but significantly reduces CR (improved scene understanding).
- BEV distillation recovers L2 and reduces CR by 60% (acting as a key component).
- Scene-aware initialization further improves L2 by approximately 10%.

### Key Findings

1. **Path Attention outperforms Deformable Attention** (Table 6): CR Avg decreases from 0.15 to 0.12 with identical computational cost.
2. **Components are highly versatile** (Table 7): Integrating Path Attention and Autoregressive Online Mapping as plug-and-play modules into UniAD and SparseDrive yields consistent improvements.
3. **NAVSIM/Bench2Drive Closed-loop Validation**: Under ego status noise, AdaptiveAD's PDMS, DS, and SR metrics significantly outperform VAD.

## Highlights & Insights

1. **Architecture-level Solution**: Unlike data- or regularization-level mitigation strategies, this approach directly cuts off the ego-status shortcut at the information flow level.
2. **Clear Complementary Logic of Dual Branches**: The scene-driven branch provides environmental perception decisions (crucial in complex scenes), while the ego-driven branch provides inertial motion priors (efficient in simple scenes). The fusion module adaptively weights them based on scene complexity.
3. **Sophisticated BEV Distillation**: Mitigates the motion blur side effects introduced by decoupling, using a natural and highly effective teacher-student paradigm.
4. **Causal Consistency in Autoregressive Mapping**: Establishes a planning-to-perception feedback loop drawing inspiration from world models.

## Limitations & Future Work

1. Evaluation is predominantly conducted on open-loop nuScenes; the depth of closed-loop performance validation remains insufficient.
2. The dual-branch architecture increases model complexity, with FPS dropping from VAD's 3.4 to 3.0.
3. Completely removing ego status from the scene-driven branch might be overly aggressive, as ego status remains valuable in certain scenarios (e.g., cruising at a constant speed on highways).
4. Merging with LiDAR-based modalities is not discussed.
5. Expanding the decoupling philosophy to generative world models represents an interesting direction for exploration.

## Related Work & Insights

- **Planning-oriented End-to-End**: UniAD (CVPR 2023) $\rightarrow$ VAD (ICLR 2024) $\rightarrow$ SparseDrive $\rightarrow$ BridgeAD
- **Mitigating Causal Confusion**: EgoStatus Analysis (Li 2024), PLUTO Contrastive Imitation Learning (Cheng 2024)
- **Inspirations from Multi-sensor Fusion**: Shifting from sensor fusion to decision context fusion offers a conceptual innovation.
- **World Model Feedback**: Autoregressive concepts inspired by Think2Drive and Vista.
- **Insights**: The decoupling + adaptive fusion paradigm can be generalized to multimodal decision-making and human-machine co-driving.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ★★★★★ | Architecture-level mitigation of causal confusion; pioneering dual-branch decoupling formulation. |
| Technical Depth | ★★★★☆ | Methodologically sound pipeline integrating path attention, distillation, and autoregressive mapping. |
| Empirical Quality | ★★★★★ | Comprehensive evaluation spanning open-loop, closed-loop, ego-noise stress testing, and scene classification. |
| Writing Quality | ★★★★★ | Precise problem formulation, highly compelling motivational arguments, and crystal-clear structure. |
| Practical Value | ★★★★☆ | Verified plug-and-play potential of individual modules, though the dual-branch framework introduces structural overhead. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/percept-wam_perception-enhanced_world-awareness-action_model_for_robust_end-to-e.md)
- [\[ICML 2026\] RoCA: Robust Cross-Domain End-to-End Autonomous Driving](../../ICML2026/autonomous_driving/roca_robust_cross-domain_end-to-end_autonomous_driving.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[AAAI 2026\] Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception](rethinking_the_spatio-temporal_alignment_of_end-to-end_3d_perception.md)
- [\[CVPR 2026\] ActiveAD: Planning-Oriented Active Learning for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/activead_planning-oriented_active_learning_for_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->

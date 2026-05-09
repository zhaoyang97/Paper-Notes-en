---
title: >-
  [Paper Note] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution
description: >-
  [NEURIPS2025][Autonomous Driving][end-to-end driving] Proposes SeerDrive, which achieves SOTA on NAVSIM and nuScenes through bidirectional modeling of scene evolution and trajectory planning (future-aware planning + iterative interaction).
tags:
  - NEURIPS2025
  - Autonomous Driving
  - end-to-end driving
  - world model
  - BEV
  - trajectory planning
  - iterative refinement
date: 2026-05-08
content_hash: be65058794f8fc20
---

# Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution

**Conference**: NEURIPS2025
**arXiv**: [2510.11092](https://arxiv.org/abs/2510.11092)
**Code**: [LogosRoboticsGroup/SeerDrive](https://github.com/LogosRoboticsGroup/SeerDrive)
**Area**: Autonomous Driving
**Keywords**: end-to-end driving, world model, BEV, trajectory planning, iterative refinement

## TL;DR
Proposes SeerDrive, which achieves SOTA on NAVSIM and nuScenes through bidirectional modeling of scene evolution and trajectory planning (future-aware planning + iterative interaction).

## Background & Motivation
Most existing end-to-end autonomous driving methods adopt a "one-shot" paradigm: directly predicting trajectories for the next few seconds based solely on sensor observations from the current frame. This approach has two core limitations:

1. **Ignoring dynamic scene evolution**: A snapshot of the current frame cannot adequately capture future changes in the traffic scene (e.g., a vehicle ahead decelerating, a pedestrian crossing), leading to a lack of foresight in planning.
2. **Ignoring bidirectional coupling**: The ego vehicle's future behavior in turn affects the evolution of the surrounding scene (e.g., the reaction of following vehicles after a lane change), yet this bidirectional dependency is rarely explicitly modeled in existing methods.

The authors draw inspiration from the trend of world model research — if future scenes can be predicted and deeply coupled with the planning process, more adaptive decision-making becomes achievable.

## Core Problem
How to explicitly model the **bidirectional relationship** between **future scene evolution** and **trajectory planning** in an end-to-end driving framework, such that the planner can both leverage foresight into future scenes and feed the ego vehicle's planning intent back into the scene prediction model.

## Method

### Overall Architecture
SeerDrive consists of two core modules that collaborate iteratively in a closed loop:
- **BEV World Modeling Network**: Predicts future BEV semantic maps.
- **End-to-End Planning Network**: Generates trajectories based on current and future BEV features.

### 1. Feature Encoding
- Multi-view images and LiDAR are fused into current BEV features $F_{\rm bev}^{\rm curr} \in \mathbb{R}^{H \times W \times C}$ via TransFuser.
- Anchored multimodal trajectories and ego-vehicle state are encoded via MLP into current ego features $F_{\rm ego}^{\rm curr} \in \mathbb{R}^{M \times C}$ (where $M$ is the number of trajectory modes).
- A lightweight BEV decoder generates the current BEV semantic map $\mathcal{B}_{\rm curr}$ for supervision.

### 2. Future BEV World Modeling
- $F_{\rm bev}^{\rm curr}$ is flattened and repeated along the modal dimension, then concatenated with $F_{\rm ego}^{\rm curr}$ to form scene features $F_{\rm scene}^{\rm curr}$.
- A Transformer Encoder (i.e., the BEV World Model) predicts future scene features $F_{\rm scene}^{\rm fut}$.
- Future BEV features $F_{\rm bev}^{\rm fut}$ are extracted and decoded via the BEV decoder into future semantic maps $\mathcal{B}_{\rm fut}$ for supervision.
- **Only the BEV at the final planning step is predicted** (e.g., 4 seconds ahead), rather than intermediate frame sequences — ablation studies show this is sufficient and more efficient.

### 3. Future-Aware Planning
The core challenge is how to allow the planner to leverage both current and future BEV features without causing representational entanglement. The solution is a **decoupled strategy**:

- **Current branch**: $F_{\rm ego}^{\rm curr}$ interacts with $F_{\rm bev}^{\rm curr}$ via a Transformer Decoder → decoded by MLP to produce trajectory $\mathcal{T}_a$.
- **Future branch**: Future ego features $F_{\rm ego}^{\rm fut}$ are initialized using the endpoints of anchored trajectories, interact with $F_{\rm bev}^{\rm fut}$ via a Transformer Decoder → decoded by MLP to produce trajectory $\mathcal{T}_b$.
- **Fusion**: Motion-aware Layer Normalization (MLN) is used to inject $F_{\rm ego}^{\rm fut}$ into $F_{\rm ego}^{\rm curr}$, producing a future-aware ego representation → decoded into the final trajectory $\mathcal{T}_{\rm final}$.

### 4. Iterative Scene Modeling and Vehicle Planning
- The updated ego features $F_{\rm ego}^{\rm curr}$ output by the planning network are fed back into the BEV World Model to generate updated future BEV representations.
- This process is iterated $N$ times (with $N=2$ found optimal in experiments); each iteration produces a set of semantic maps and trajectories, all of which participate in training supervision.
- This design embodies the core idea of bidirectional coupling: scene prediction guides planning, and planning results in turn refine scene prediction.

### 5. End-to-End Training
Total loss = BEV semantic map loss + trajectory planning loss, covering outputs from all iterative rounds. On NAVSIM, the loss weights are set to $\lambda_1=10, \lambda_2=0.1, \lambda_3=1$.

## Key Experimental Results

### NAVSIM (navtest, closed-loop evaluation)

| Method | PDMS ↑ | NC ↑ | DAC ↑ | EP ↑ |
|--------|--------|------|-------|------|
| TransFuser | 84.0 | 97.7 | 92.8 | 79.2 |
| DiffusionDrive | 88.1 | 98.2 | 96.2 | 82.2 |
| WoTE | 88.3 | 98.5 | 96.8 | 81.9 |
| Hydra-NeXt | 88.6 | 98.1 | 97.7 | 81.8 |
| **SeerDrive** | **88.9** | 98.4 | 97.0 | **83.2** |
| SeerDrive (V2-99) | **90.7** | 98.8 | 98.6 | 84.2 |

### nuScenes (open-loop, L2 displacement error / collision rate)

| Method | Avg L2 ↓ | Avg Col. ↓ |
|--------|----------|------------|
| SparseDrive | 0.61 | 0.08 |
| BridgeAD | 0.59 | 0.09 |
| MomAD | 0.60 | 0.09 |
| **SeerDrive** | **0.43** | **0.06** |

### Ablation Study (NAVSIM PDMS)
- Remove Future-Aware Planning and Iterative: 87.1 (−1.8)
- Remove only Future BEV injection: 87.9 (−1.0)
- Remove only iteration: 88.1 (−0.8)
- Full SeerDrive: **88.9**
- Number of iterations: 1 → 88.1, 2 → **88.9**, 3 → 88.7
- MLN outperforms Concat (88.3) and Add (88.5)

## Highlights & Insights
1. **Paradigm innovation**: For the first time, SeerDrive explicitly models the bidirectional closed-loop interaction between scene evolution and trajectory planning in end-to-end driving, surpassing the conventional one-shot paradigm.
2. **Elegant decoupled design**: Current and future BEV features interact independently with ego features before being fused via MLN, avoiding representational entanglement — ablations confirm that naive joint learning leads to performance degradation.
3. **Fast iterative convergence**: Only 2 iterations are needed to reach optimal performance; 3 iterations slightly decrease performance, demonstrating a compact and efficient design.
4. **Low training cost**: Only ~5 hours on 8 RTX 3090 GPUs (NAVSIM), offering good reproducibility.
5. **Only predicting the terminal-frame BEV**: Ablations show that predicting intermediate frame sequences (1s–2s–3s–4s) does not outperform predicting only the final frame, keeping the design simple and efficient.

## Limitations & Future Work
1. **Evaluated only in non-reactive/open-loop settings**: NAVSIM is a non-reactive simulator and nuScenes is open-loop replay; full closed-loop validation in environments such as CARLA is lacking.
2. **Limited expressiveness of BEV semantic maps**: Only BEV semantic maps are predicted, without modeling 3D height information or occlusion relationships.
3. **Upper bound on iteration count**: Performance begins to degrade at 3 iterations, suggesting potential information degradation in the current iterative mechanism.
4. **Future BEV predicts only the terminal frame**: Although ablations show limited benefit from intermediate frames, this may be due to a coarse fusion strategy (simple concatenation) rather than an inherent limitation.
5. **No dedicated analysis of long-tail scenarios**: Performance in scenarios such as extreme weather or complex intersections is not separately discussed.

## Related Work & Insights
- **vs. DiffusionDrive / GoalFlow**: These methods improve trajectory generation via diffusion/flow matching but do not model future scenes; SeerDrive approaches the problem from a scene prediction perspective, offering a complementary viewpoint.
- **vs. WoTE**: WoTE uses a world model to evaluate candidate trajectories online and selects the best one; SeerDrive instead enables the world model to directly participate in planning optimization through iterative interaction.
- **vs. OccWorld / Drive-OccWorld**: These methods use occupancy for scene prediction and jointly predict actions, but generate frames autoregressively; SeerDrive directly predicts the terminal-frame BEV and refines it iteratively.
- **vs. LAW / SSR**: These methods use the world model only as an auxiliary supervision signal during training and do not involve it at inference time; SeerDrive employs the world model for iterative interaction at inference as well.

## Key Findings
1. The bidirectional modeling idea is extensible to other decision-making tasks (e.g., coupling environment prediction with action planning in robot manipulation).
2. The iterative interaction paradigm resembles the denoising process in diffusion models — could diffusion/flow matching replace the current deterministic iteration?
3. Replacing the BEV world model with a stronger generative model (e.g., diffusion-based BEV generation) may further improve future scene prediction quality.
4. Combined with the scaling law research in DriveTransformer, the synergistic effects of iteration count and model scale are worth exploring.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The bidirectional closed-loop interaction paradigm offers a meaningful new perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets with extensive ablations, though closed-loop simulation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and well-organized figures and tables.
- **Value**: ⭐⭐⭐⭐ — Provides new insights into how world models can be integrated into end-to-end driving.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[NeurIPS 2025\] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)
- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](../../AAAI2026/autonomous_driving/diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](../../AAAI2026/autonomous_driving/drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)

<!-- RELATED:END -->

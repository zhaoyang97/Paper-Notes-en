---
title: >-
  [Paper Note] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving
description: >-
  [AAAI 2026][Autonomous Driving][Latent World Model] WorldRFT is a planning-oriented latent world model framework that integrates VGGT-based spatial encoding, hierarchical planning decomposition with local-aware iterative refinement, and GRPO-based collision-aware reinforcement fine-tuning. It reduces collision rate by 83% on nuScenes (0.30% → 0.05%) and achieves near-LiDAR SOTA performance using camera only on NavSim (87.8 vs. 88.1 PDMS).
tags:
  - AAAI 2026
  - Autonomous Driving
  - Latent World Model
  - Reinforcement Fine-Tuning
  - GRPO
  - Hierarchical Planning
  - VGGT
  - Collision Awareness
date: 2026-05-08
content_hash: 79ee16799a529701
---

# WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving

**Conference**: AAAI 2026  
**arXiv**: [2512.19133](https://arxiv.org/abs/2512.19133)  
**Code**: [pengxuanyang/WorldRFT](https://github.com/pengxuanyang/WorldRFT)  
**Area**: Autonomous Driving / End-to-End Planning / World Models  
**Keywords**: Latent World Model, Reinforcement Fine-Tuning, GRPO, Hierarchical Planning, VGGT, Collision Awareness

## TL;DR
WorldRFT is a planning-oriented latent world model framework that integrates VGGT-based spatial encoding, hierarchical planning decomposition with local-aware iterative refinement, and GRPO-based collision-aware reinforcement fine-tuning. It reduces collision rate by 83% on nuScenes (0.30% → 0.05%) and achieves near-LiDAR SOTA performance using camera only on NavSim (87.8 vs. 88.1 PDMS).

## Background & Motivation

**State of the Field**: End-to-end autonomous driving is shifting from multi-task perception architectures (UniAD, VAD, SparseDrive, etc., requiring 3D annotations) toward self-supervised latent world model paradigms (LAW, SSR, etc., requiring no perception labels).

**Limitation 1 — Weak Spatial Awareness**: Existing reconstruction-oriented latent representations lack 3D spatial understanding. World4Drive's monocular depth estimation suffers from cross-view inconsistency.

**Limitation 2 — Inefficient Planning Interaction**: A single global planning query generates trajectories from the entire feature map, dispersing attention and failing to capture locally critical structures.

**Limitation 3 — Lack of Safety Awareness**: Pure imitation learning minimizes deviation from expert trajectories without distinguishing safe from unsafe deviations, yielding no active collision avoidance capability.

**Core Idea**: Three modules are aligned along the full pipeline of "scene understanding → planning decision → safety optimization" — VGGT geometric priors enhance spatial perception, hierarchical planning with local-aware refinement extracts planning-relevant features, and GRPO reinforcement fine-tuning enables active collision avoidance.

## Method

### Overall Architecture
Surround-view RGB → **SWE** (ResNet + frozen VGGT fusion, outputs spatially-aware latent representation $W^t_{\text{latent}}$) → **HPR** (three parallel sub-task query interactions + local-aware iterative refinement with $K=3$ rounds) → **RFT** (trajectory Gaussianization + collision reward + GRPO policy optimization)

### Key Design 1: Spatial-aware World Encoder (SWE)
- ResNet-50 extracts multi-view features $F_t \in \mathbb{R}^{M \times h \times w \times D}$
- Frozen VGGT extracts multi-view consistent 3D tokens $t_{3D}$ from surround-view images
- Lightweight cross-attention fusion: $W^t_{\text{latent}} = \text{Cross-Attn}(F_t, t_{3D})$
- Grounded-SAM generates pseudo semantic labels; cross-entropy loss $\mathcal{L}_{sem}$ provides auxiliary semantic supervision

### Key Design 2: Hierarchical Planning Refinement (HPR)
**Three parallel sub-tasks**:
- **Goal Region Localization**: Modeled with a Laplace distribution (center $\mu$ + scale $b$), trained with NLL loss. The scale $b$ reflects scene complexity and serves as a conditioning signal for subsequent refinement.
- **Spatial Path Planning**: Samples $N=30/50$ spatially uniform waypoints every 2 m (not temporally uniform), decoupling spatial and temporal learning.
- **Temporal Trajectory Prediction**: $T=6/8$ trajectory points at 0.5 s intervals, supervised with L1 loss against expert trajectories.

**Local-aware Iterative Refinement ($K=3$ rounds)**:
1. Encode outputs of the three sub-tasks into a unified state $F_s$
2. Project trajectory points onto the feature map → deformable convolution to sample local features $F_{\text{local}}$
3. Condition on the goal region scale $b$; fuse local and global features
4. Residual update: $T^{(k+1)} = T^{(k)} + 0.1 \cdot \Delta T^{(k)}$

### Key Design 3: Safety-aware RFT (GRPO Collision-aware Fine-Tuning)
1. **Trajectory Gaussianization**: Converts deterministic trajectories into Gaussian distributions (mean $\mu_\theta$ + auxiliary variance network $\Sigma_\theta$), enabling RL exploration.
2. **Collision-aware Reward**: $r=-1$ (collision) / $0$ (safe), based on the distance between the ego vehicle and surrounding agent bounding boxes.
3. **GRPO Optimization**: Sample $G=10$ trajectories → point-wise reward normalization → cumulative advantage $Adv_j=\sum_{t\geq j}\tilde{r}_t$ → PPO-clip objective + KL divergence constraint.
4. Auxiliary reference loss ($l_2$ regression to pre-trained outputs) + maximum entropy loss (to prevent premature convergence).

### Loss & Training
- **Pre-training**: $\mathcal{L} = 0.2\mathcal{L}_{sem} + 0.2\mathcal{L}_{rec} + 0.001\mathcal{L}_{target} + 1.0\mathcal{L}_{traj}$
- **RFT**: $\mathcal{L}_{RL} = -J(\theta) + 0.1 D_{KL} + 0.12\mathcal{L}_{ref} + 0.1\mathcal{L}_{entropy}$

## Key Experimental Results

### Main Results 1: nuScenes Open-Loop Planning

| Method | Training | L2 Avg↓ | CR 1s↓ | CR 2s↓ | CR 3s↓ | CR Avg↓ |
|--------|----------|---------|--------|--------|--------|---------|
| UniAD | P-IL | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| VAD | P-IL | 0.72 | 0.07 | 0.18 | 0.43 | 0.23 |
| PARA-Drive | P-IL | 0.48 | 0.14 | 0.23 | 0.39 | 0.25 |
| DiffusionDrive | P-IL | 0.57 | 0.03 | 0.05 | 0.16 | 0.08 |
| LAW (no labels) | SS-L | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| World4Drive | SS-L | 0.50 | 0.02 | 0.12 | 0.33 | 0.16 |
| **WorldRFT (w/o RFT)** | SS-L | **0.47** | 0.10 | 0.11 | 0.23 | 0.15 |
| **WorldRFT (w/ RFT)** | SS-L & RL | 0.48 | **0.00** | **0.00** | **0.16** | **0.05** |

### Main Results 2: NavSim Closed-Loop Planning

| Method | Input | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|--------|-------|-----|------|------|--------|-----|-------|
| UniAD | C | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| DiffusionDrive | C&L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| LAW | C | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| World4Drive | C | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 |
| **WorldRFT (w/ RFT)** | **C** | **97.8** | **96.8** | **94.0** | **100.0** | **81.7** | **87.8** |

### Ablation Study (nuScenes)

| ID | VGGT | Target | Path | Refine | L2↓ | CR↓ |
|----|------|--------|------|--------|-----|-----|
| 1 | — | — | — | — | 0.59 | 0.16 |
| 4 | — | ✓ | ✓ | ✓ | 0.52 | 0.08 |
| 7 | ✓ | ✓ | ✓ | — | 0.50 | 0.06 |
| 8 | ✓ | ✓ | ✓ | ✓ | **0.48** | **0.05** |

### Key Findings
1. **RFT improves safety by 83%**: Collision rate drops from 0.30% to 0.05%; 1 s and 2 s collision rates reach 0.00%; L2 increases by only 0.01 m — RFT learns to yield proactively.
2. **VGGT is a critical component**: Its inclusion reduces L2 by 7.7% and CR by 37.5%; DAC of 96.8 is the highest across all methods including LiDAR-based ones.
3. **Hierarchical planning and refinement progressively improve performance**: Adding Target + Path reduces CR from 0.16 to 0.08 (−50%); refinement further reduces it to 0.05.
4. **Probabilistic goal modeling outperforms deterministic**: Laplace distribution modeling reduces CR by 50% (0.10 → 0.05).
5. **Camera-only performance approaches LiDAR SOTA**: NavSim score of 87.8 trails LiDAR-based DiffusionDrive by only 0.3 PDMS points.

## Highlights & Insights
- First work to introduce GRPO (originally from DeepSeek-Math) into autonomous driving trajectory planning, marking a paradigm shift from behavior cloning to active collision avoidance.
- In the annotation-free paradigm, WorldRFT achieves a lower collision rate than all perception-annotated methods (0.05% vs. DiffusionDrive's 0.08%).
- Frozen VGGT combined with lightweight cross-attention fusion elegantly introduces 3D geometric priors.
- The hierarchical planning decomposition is well-designed: goal region (directional intent) + spatial path (geometric shape) + temporal trajectory (dynamics), each supervised by the most appropriate signal.

## Limitations & Future Work
1. **Safety–accuracy trade-off**: L2 increases slightly from 0.47 to 0.48 after RFT.
2. **Closed-loop evaluation limited to NavSim**: Validation in more complex simulation environments such as CARLA has not been conducted.
3. **Overly simplistic collision reward**: Binary reward (−1/0) does not account for continuous distance-based rewards or comfort penalties.
4. **Navigation commands still require annotation**: Left/right/straight commands depend on human-annotated labels.
5. **ResNet-50 backbone**: The performance ceiling of stronger backbones such as ViT has not been explored.

## Related Work & Insights
- **End-to-End Driving**: UniAD/VAD (cascaded BEV multi-task), PARA-Drive (parallel architecture), SparseDrive (sparse perception), DiffusionDrive (diffusion-based planning)
- **World Models**: GAIA-1 (generative scene modeling), DriveWorld (occupancy prediction), LAW/SSR (self-supervised latent world models)
- **RL for Driving**: RAD (3DGS virtual environment + RL), AlphaDrive (VLM + RL)

## Rating
⭐⭐⭐⭐⭐ — The work is highly complete, with each of the three modules addressing a distinct core problem and supported by thorough ablations. Introducing GRPO into driving planning is a notable contribution, and the 83% reduction in collision rate is convincing. Camera-only performance approaching LiDAR SOTA on NavSim demonstrates the substantial potential of the annotation-free paradigm. Ablations cover backbone choice, number of refinement iterations, path configuration, goal modeling, and data volume, among other dimensions.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/autonomous_driving/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[ICLR 2026\] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](../../ICLR2026/autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[AAAI 2026\] ReflexDiffusion: Reflexion-Enhanced Trajectory Planning for High Lateral Acceleration in Autonomous Driving](reflexdiffusion_reflection-enhanced_trajectory_planning_for_.md)

<!-- RELATED:END -->

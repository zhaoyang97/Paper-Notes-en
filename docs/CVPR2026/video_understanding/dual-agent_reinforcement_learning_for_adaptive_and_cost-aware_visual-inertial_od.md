---
title: >-
  [Paper Note] Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry
description: >-
  [CVPR2026][Video Understanding][Visual-Inertial Odometry] This paper proposes a dual-agent reinforcement learning framework comprising a Select Agent (which decides whether to activate the visual front-end based on IMU s…
tags:
  - "CVPR2026"
  - "Video Understanding"
  - "Visual-Inertial Odometry"
  - "Reinforcement Learning"
  - "Adaptive Fusion"
  - "Computation Scheduling"
  - "IMU Bias Estimation"
  - "PPO"
date: 2026-05-08
content_hash: d660c4704c63c99c
---

# Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry

**Conference**: CVPR2026
**arXiv**: [2511.21083](https://arxiv.org/abs/2511.21083)
**Code**: To be confirmed
**Area**: Video Understanding / Visual Odometry
**Keywords**: Visual-Inertial Odometry, Reinforcement Learning, Adaptive Fusion, Computation Scheduling, IMU Bias Estimation, PPO

## TL;DR

This paper proposes a dual-agent reinforcement learning framework comprising a Select Agent (which decides whether to activate the visual front-end based on IMU signals) and a Fusion Agent (which adaptively fuses visual-inertial states). Without completely removing VIBA, the framework substantially reduces its invocation frequency and computational overhead, achieving a superior accuracy–efficiency–memory trade-off.

## Background & Motivation

**Root Cause of VIO**: Filter-based methods (MSCKF, ROVIO) are efficient but suffer from significant drift; optimization-based methods (VINS-Mono, ORB-SLAM3) achieve high accuracy but incur heavy VIBA computation, making deployment on resource-constrained edge devices difficult.

**Limitations of Prior Work — End-to-End Deep Learning**: Methods such as VINet and SelfVIO directly regress poses but remain inferior to mature optimization frameworks in accuracy and generalizability.

**Hybrid Methods Do Not Resolve the Core Bottleneck**: Works such as iSLAM and DPVO introduce learning modules to enhance feature matching or optimization, yet are still bottlenecked by VIBA computation.

**Inefficient Keyframe Selection**: Conventional strategies must process images before determining frame utility, precluding skip decisions prior to visual computation.

**Inflexible Fixed Fusion Weights**: Static EKF or fixed-gain fusion cannot dynamically adjust the trust in visual versus inertial measurements based on motion intensity and sensor reliability.

**Shallow Application of RL in Odometry**: Prior work (Messikommer et al.) applies RL only to optimize internal heuristics within VO, without considering high-level scheduling of whether to activate the entire VO pipeline.

## Method

### Overall Architecture

The system consists of four decoupled modules:
- **IMU Preprocess**: IMU bias encoder + pre-integration, outputting inter-frame inertial states $(\Delta\mathbf{p}, \Delta\mathbf{q}, \Delta\mathbf{v}, \Delta t)$
- **Select Agent**: An RL policy that decides whether to activate the VO module based solely on IMU states
- **Visual Odometry**: Patch-based recurrent optimization built on DPVO, activated only when the Select Agent outputs 1
- **Fusion Agent**: A composite module consisting of MLP1 (supervised velocity estimation) and MLP2 (RL policy for adaptive full-state fusion)

### Key Designs

**1. IMU Bias Estimator**
- Two lightweight encoder networks are trained: $f_{bias}^g$ (gyroscope) and $f_{bias}^a$ (accelerometer), taking raw sensor sequences and noise parameters as input and outputting three-axis bias estimates.
- Fixed bias is estimated rather than stochastic noise, as the bias model is better suited to capturing slowly varying dominant patterns.

**2. Select Agent (RL Scheduling)**
- **State space**: A compact IMU-only state $s_t^{sel} = \{\Delta\mathbf{p}_t, \Delta\mathbf{q}_t, \Delta\mathbf{v}_t, \Delta t_t^{vo}\}$, requiring no visual features.
- **Action space**: Binary decision $a_t^{sel} \in \{0, 1\}$, where 0 = skip VO and 1 = run VO.
- **Reward function**: Terminal ATE reward + dense per-step penalty + VO invocation cost: $R_{episode} = \frac{A}{ATE + \epsilon} - B \cdot N_f$
- Trained with PPO using an MLP-parameterized policy.

**3. Fusion Agent (Adaptive Fusion)**
- **MLP1** (supervised): Estimates metric velocity from scaled VO pose and IMU pre-integration.
- **MLP2** (RL policy): Outputs a 7-dimensional per-axis fusion weight $\mathbf{w} \in [0,1]^7$, performing convex combination over position and velocity, and SLERP interpolation over orientation.
- When VO is skipped, $\mathbf{w}$ defaults to zero, yielding pure IMU propagation.
- Reward: $r_k = -\|\mathbf{p}_k - \mathbf{p}_{gt}\|_2^2 - \lambda \text{Tr}(\mathbf{\Sigma}_k)$, jointly driving low error and well-calibrated uncertainty.

**4. Scale Initialization**
- The world coordinate frame is aligned with the initial IMU body frame. An overdetermined linear system is constructed using IMU pre-integration and VO relative translations within a sliding window, and the global metric scale $s$ is solved via least squares.

### Loss & Training

- Two-stage training: supervised pre-training of MLP1, followed by supervised initialization then PPO fine-tuning of MLP2.
- PPO is trained in a Gym-style replay environment using real IMU–VO pairs, naturally incorporating sensor drift and noise.

## Key Experimental Results

### Main Results

**EuRoC MAV Dataset (Table 1)**: Comparison with classical CPU-based VIO methods

| Method | MH2 | MH3 | MH4 | MH5 | V11 | V12 | V13 | V21 | V22 | V23 | Avg |
|--------|------|------|------|------|------|------|------|------|------|------|------|
| MSCKF | 0.45 | 0.23 | 0.37 | 0.48 | 0.34 | 0.20 | 0.67 | 0.10 | 0.16 | 1.13 | 0.413 |
| VINS-MONO | 0.15 | 0.22 | 0.32 | 0.30 | 0.079 | 0.11 | 0.18 | 0.080 | 0.16 | 0.27 | 0.187 |
| DM-VIO | 0.044 | 0.097 | 0.102 | 0.096 | 0.048 | 0.045 | 0.069 | 0.029 | 0.050 | 0.114 | 0.069 |
| ORB-SLAM3 | 0.037 | 0.046 | 0.075 | 0.057 | 0.049 | 0.015 | 0.037 | 0.042 | 0.021 | 0.027 | **0.041** |
| **Ours** | 0.064 | 0.119 | 0.112 | 0.112 | 0.047 | 0.125 | 0.073 | 0.055 | 0.036 | 0.179 | 0.092 |

**GPU-Based Method Comparison (Table 4)**: Joint evaluation of accuracy, efficiency, and memory

| Method | Avg ATE | FPS | VRAM (GB) |
|--------|---------|-----|-----------|
| DPVO | 0.106 | 22 | 4.92 |
| iSLAM | 0.529 | 31 | 6.47 |
| DROID-VO | 0.188 | 14 | 8.63 |
| **Ours** | **0.092** | **39** | **4.37** |

### Ablation Study

1. **IMU Bias Estimator**: Jointly estimating gyroscope and accelerometer biases (Omega+Accel) yields the best performance; the fixed bias model outperforms the stochastic noise model.
2. **Select Agent**: Under aggressive frame-skipping (75%–87.5%), the IMU-only prior scheduling exhibits a more gradual degradation curve compared to heuristic and RL-gating (KF) baselines; at a 50% skip target, the IMU-only policy achieves significantly higher FPS with less than 3% increase in ATE.
3. **Fusion Agent**: RL fusion (ATE 0.112 m) outperforms EKF fusion (0.127 m) and fixed-weight fusion (0.143 m); the fusion policy remains effective when switching to a DROID-VO front-end (0.399→0.237 m), demonstrating cross-backend generalization.
4. **Robustness**: Under 5%/10% image blur degradation, the proposed method consistently achieves lower ATE (0.138/0.153 m) than DPVO (0.174/0.192 m).

### Key Findings

- The proposed method achieves the best average ATE (0.092 m) among GPU-based methods, while reaching 39 FPS (1.77× that of DPVO) and consuming only 4.37 GB VRAM (49.4% less than DROID-SLAM).
- CPU-side BA/VIBA time is reduced from 121 ms (ORB-SLAM3) to 12.77 ms, structurally lowering optimization overhead.
- Accuracy remains acceptable relative to classical optimization-based VIO (Avg 0.092 vs. DM-VIO 0.069), with a substantial computational efficiency advantage.
- On the TUM-VI dataset, the average ATE is 0.80 m, marginally higher than DM-VIO's 0.77 m, while remaining competitive.

## Highlights & Insights

- **IMU-only prior scheduling** is the core innovation: skip decisions are made before visual computation, saving substantially more computation than optimizing only internal VO heuristics.
- The dual-agent design decouples scheduling and fusion, with each agent optimizing a distinct objective, yielding a clean architecture.
- The Fusion Agent generalizes across VO backends (DPVO→DROID-VO), demonstrating that the learned policy depends on physical quantities rather than architecture-specific features.
- Training PPO via Gym-style replay on real data avoids the sim-to-real gap.

## Limitations & Future Work

- Accuracy remains noticeably below top optimization-based methods such as ORB-SLAM3 (Avg 0.092 vs. 0.041).
- Scale initialization relies on a sliding window with sufficient motion; stationary start-up scenarios may fail.
- Evaluation is limited to two indoor datasets (EuRoC and TUM-VI); outdoor and driving scenarios are not assessed.
- The robustness of the Select Agent's frame-skipping strategy under extreme conditions (e.g., rapid consecutive rotations) is not thoroughly discussed.
- Training requires ground-truth sequences, and zero-shot generalization to new environments is not validated.

## Related Work & Insights

- **Classical VIO**: MSCKF, ROVIO (filter-based); VINS-Mono, ORB-SLAM3, DM-VIO (optimization-based).
- **Deep Learning VO/VIO**: VINet, SelfVIO (end-to-end); DPVO, DROID-SLAM (hybrid); iSLAM (learning-enhanced classical pipeline).
- **Adaptive Visual Selection**: VS-VIO and similar methods dynamically reweight features but still run the visual encoder on every frame.
- **RL in VO**: Messikommer et al. apply RL to replace internal keyframe selection heuristics within VO; this paper elevates RL to high-level pipeline scheduling.
- **Most Relevant**: This work is the first to model "whether to run the entire VO pipeline" as a prior RL decision.

## Rating

- Novelty: ⭐⭐⭐⭐ — The dual-agent RL architecture and IMU-only prior scheduling are conceptually novel, though RL in robotics is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations are comprehensive; cross-backend generalization and robustness are both evaluated, though only on two indoor datasets.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear, motivation is well articulated, and derivations are complete.
- Value: ⭐⭐⭐⭐ — The accuracy–efficiency trade-off is practically meaningful for deployment, though the accuracy gap relative to SOTA optimization methods limits applicability in high-precision scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)
- [\[CVPR 2026\] LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding](longvideo-r1_smart_navigation_for_low-cost_long_video_understanding.md)
- [\[AAAI 2026\] Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy](../../AAAI2026/video_understanding/coordinated_humanoid_robot_locomotion_with_symmetry_equivariant_reinforcement_le.md)
- [\[CVPR 2026\] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)

</div>

<!-- RELATED:END -->

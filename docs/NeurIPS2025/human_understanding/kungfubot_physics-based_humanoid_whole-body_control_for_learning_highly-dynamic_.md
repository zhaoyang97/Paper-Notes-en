---
title: >-
  [Paper Note] KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills
description: >-
  [NeurIPS 2025][Human Understanding][humanoid control] This paper proposes the PBHC framework, which enables a humanoid robot (Unitree G1) to learn highly dynamic whole-body skills such as kung fu and dance through a physics-aware motion processing pipeline and a bi-level optimization scheme for adaptive tracking factors. The approach achieves substantially lower tracking errors than existing methods and is successfully deployed on real hardware.
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "humanoid control"
  - "motion imitation"
  - "reinforcement-learning"
  - "adaptive tracking"
  - "sim-to-real"
date: 2026-05-08
content_hash: 28beb4f696dab1d7
---

# KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills

**Conference**: NeurIPS 2025
**arXiv**: [2506.12851](https://arxiv.org/abs/2506.12851)  
**Code**: [Project Page](https://kungfu-bot.github.io)  
**Area**: Video Understanding
**Keywords**: humanoid control, motion imitation, reinforcement-learning, adaptive tracking, sim-to-real

## TL;DR

This paper proposes the PBHC framework, which enables a humanoid robot (Unitree G1) to learn highly dynamic whole-body skills such as kung fu and dance through a physics-aware motion processing pipeline and a bi-level optimization scheme for adaptive tracking factors. The approach achieves substantially lower tracking errors than existing methods and is successfully deployed on real hardware.

## Background & Motivation

- **High-dynamic motion imitation is extremely challenging**: Existing humanoid motion imitation methods (ExBody, OmniH2O, etc.) are limited to low-speed, smooth motions and cannot handle highly dynamic behaviors such as kung fu kicks or spinning jumps.
- **Physical feasibility of reference motions**: Human motion sequences extracted from video may violate the robot's physical constraints (joint limits, dynamics), making it difficult for RL to converge when directly maximizing a tracking reward.
- **Limitations of fixed tracking tolerances**: Existing methods use a fixed tracking reward parameter (tracking factor $\sigma$), which cannot adapt to motions of varying difficulty — a large $\sigma$ makes the reward insensitive to errors, while a small $\sigma$ drives the reward toward zero.
- **High cost of dataset preprocessing**: H2O requires training a privileged policy to filter infeasible motions, and ExBody2 requires training an initial policy to assess motion difficulty — both are expensive prerequisite steps.

## Method

### Stage 1: Motion Processing Pipeline

**1. Video Motion Extraction**: GVHMR is used to estimate SMPL-format motion from monocular video; its gravity-aligned coordinate frame eliminates body tilt artifacts.

**2. Physics-Aware Motion Filtering**: Dynamic stability is assessed using the CoM–CoP distance. Let $\bar{\mathbf{p}}_t^{\text{CoM}}$ and $\bar{\mathbf{p}}_t^{\text{CoP}}$ denote the ground projections of the center of mass and center of pressure, respectively:

$$\Delta d_t = \|\bar{\mathbf{p}}_t^{\text{CoM}} - \bar{\mathbf{p}}_t^{\text{CoP}}\|_2 < \epsilon_{\text{stab}}$$

An $N$-frame sequence is considered stable if: ① both the first and last frames are stable; and ② the maximum number of consecutive unstable frames does not exceed threshold $\epsilon_N$.

**3. Contact Mask Estimation and Motion Correction**: Contact masks are estimated from ankle displacement under a zero-velocity assumption:

$$c_t^{\text{left}} = \mathbb{I}[\|\mathbf{p}_{t+1}^{\text{l-ankle}} - \mathbf{p}_t^{\text{l-ankle}}\|_2^2 < \epsilon_{\text{vel}}] \cdot \mathbb{I}[p_{t,z}^{\text{l-ankle}} < \epsilon_{\text{height}}]$$

A vertical offset correction is applied to floating artifacts: $\psi_{t,z}^{\text{corr}} = \psi_{t,z} - \Delta h_t$, where $\Delta h_t = \min_{v \in \mathcal{V}_t} p_{t,z}^v$, followed by EMA smoothing to remove jitter.

**4. Motion Retargeting**: A differentiable inverse-kinematics optimization aligns end-effector trajectories while satisfying joint limits.

### Stage 2: Adaptive Motion Tracking

**Exponential Tracking Reward**:

$$r(x) = \exp(-x / \sigma)$$

where $x$ is the tracking error (e.g., joint angle MSE) and $\sigma$ is the tracking factor. A large $\sigma$ renders the reward insensitive to errors; a small $\sigma$ drives the reward toward zero.

**Bi-Level Optimization for the Optimal Tracking Factor**: The selection of $\sigma$ is formalized as a bi-level optimization problem:

$$\max_{\sigma \in \mathbb{R}_+} J^{\text{ex}}(\mathbf{x}^*), \quad \text{s.t.} \quad \mathbf{x}^* \in \arg\max_{\mathbf{x} \in \mathbb{R}_+^N} J^{\text{in}}(\mathbf{x}, \sigma) + R(\mathbf{x})$$

where the inner objective $J^{\text{in}} = \sum_{i=1}^N \exp(-x_i/\sigma)$ represents the simplified cumulative reward, and the outer objective $J^{\text{ex}} = \sum_{i=1}^N -x_i^*$ minimizes total tracking error. Solving this yields the optimal tracking factor equal to the mean optimal tracking error:

$$\sigma^* = \left(\sum_{i=1}^N x_i^*\right) / N$$

**Adaptive Update Mechanism**: An EMA estimate $\hat{x}$ of the tracking error is maintained, and $\sigma$ is dynamically tightened during training:

$$\sigma \leftarrow \min(\sigma, \hat{x})$$

$\sigma$ is monotonically non-increasing: starting from a large initial value, it tightens progressively as the policy improves, forming a closed-loop feedback mechanism.

### RL Training Framework

- **Asymmetric Actor-Critic**: The actor observes only proprioception and temporal phase; the critic additionally receives reference motion positions, root linear velocity, and randomized physical parameters.
- **Vectorized Rewards**: Each reward component $r_i$ is paired with an independent value head $V_i(\mathbf{s})$, avoiding inaccurate value estimation caused by scalar aggregation.
- **Reference State Initialization (RSI)**: Time steps are randomly sampled from the reference motion for initialization, enabling parallel learning across different motion phases.
- **Domain Randomization + Zero-Shot Transfer**: Simulation physics parameters are varied; the policy is deployed directly to the real G1 robot without fine-tuning.

## Key Experimental Results

### Table 1: Main Tracking Performance Comparison (Mean ± Std, Bold = Best)

| Method | $E_{\text{g-mpbpe}}$↓(mm) | $E_{\text{mpbpe}}$↓(mm) | $E_{\text{mpjpe}}$↓($10^{-3}$rad) |
|:---|:---:|:---:|:---:|
| **Easy** | | | |
| OmniH2O | 233.54±4.0 | 103.67±1.9 | 1805.1±12.3 |
| ExBody2 | 588.22±11.4 | 332.50±3.6 | 4014.4±21.5 |
| **PBHC** | **53.25±17.6** | **28.16±6.1** | **725.6±16.2** |
| **Medium** | | | |
| OmniH2O | 433.64±16.2 | 151.42±7.3 | 2333.9±49.5 |
| ExBody2 | 619.84±26.2 | 261.01±1.6 | 3738.7±26.9 |
| **PBHC** | **126.48±27.0** | **48.87±7.6** | **1043.3±104** |
| **Hard** | | | |
| OmniH2O | 446.17±12.8 | 147.88±4.1 | 1939.5±23.9 |
| ExBody2 | 689.68±11.8 | 246.40±1.3 | 4037.4±16.7 |
| **PBHC** | **290.36±139** | **124.61±54** | **1326.6±379** |

PBHC consistently outperforms the deployable baselines (OmniH2O, ExBody2) across all difficulty levels and all metrics. On the Easy level, the global position error is reduced from 234 mm to 53 mm (a 77% reduction).

### Table 2: Real-World Tai Chi Tracking Performance

| Platform | $E_{\text{mpbpe}}$↓ | $E_{\text{mpjpe}}$↓ | $E_{\text{mpbve}}$↓ | $E_{\text{mpjve}}$↓ |
|:---|:---:|:---:|:---:|:---:|
| MuJoCo | 33.18±2.7 | 1061±83.3 | 2.96±0.34 | 67.71±6.7 |
| Real | 36.64±2.6 | 1130±9.5 | 3.01±0.13 | 65.68±2.0 |

Real-world metrics closely match simulation results, validating the effectiveness of zero-shot sim-to-real transfer.

### Ablation Study

Fixed-$\sigma$ configurations (Coarse/Medium/UpperBound/LowerBound) yield inconsistent performance across different motions — certain configurations perform well on specific motions but poorly on others. The adaptive mechanism consistently achieves near-optimal performance across all motion types.

### Physical Filtering Effectiveness

Among 10 motion sequences subjected to filtering, 4 are rejected and 6 are accepted. Accepted motions all exhibit high Episode Length Ratios (ELR), while the highest ELR among rejected motions is only 54%, validating the effectiveness of the physical feasibility metric.

## Highlights & Insights

- **Theoretically elegant adaptive tracking factor**: Formalizing $\sigma$ selection as a bi-level optimization and deriving the closed-form solution $\sigma^* = \text{mean error}$ eliminates the need for manual tuning.
- **Demonstrated high-dynamic capability**: Kung fu punches, 360° spinning kicks, and tai chi are successfully demonstrated on the real G1 robot.
- **Complete motion processing pipeline**: An end-to-end pipeline is established from video extraction → physics filtering → contact correction → retargeting.
- **Zero-shot sim-to-real transfer**: Successful deployment requires no real-world fine-tuning.

## Limitations & Future Work

- **One policy per motion**: Each motion requires training an independent policy, which does not scale efficiently to large motion libraries.
- **Lack of environment awareness**: The system has no terrain perception or obstacle avoidance capability, limiting deployment in unstructured environments.
- **Dependence on MoCap video quality**: Motions extracted by GVHMR may be inaccurate; physical filtering mitigates this but is not comprehensive.
- **High variance on Hard-level motions**: The large standard deviations in tracking error for Hard motions indicate that stability remains an open challenge.

## Related Work & Insights

- **Humanoid motion imitation** (DeepMimic [Peng+ 2018], ExBody2 [Ji+ 2024], OmniH2O [He+ 2024]): Prior methods are limited to low-speed motions; PBHC extends the capability frontier to highly dynamic behaviors.
- **Humanoid whole-body control** (HugWBC, ASAP [He+ 2025]): ASAP addresses the sim-to-real gap via multi-stage and residual policies, whereas PBHC resolves it entirely within simulation.
- **Motion processing** (IPMAN [Tripathi+ 2023], GVHMR [Shen+ 2024]): PBHC augments GVHMR with physics-aware filtering and contact correction.

## Rating

- Novelty: ⭐⭐⭐⭐ — Solid theoretical contribution via bi-level optimization derivation of adaptive $\sigma$; complete physics filtering pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Simulation comparisons + ablations + real-world deployment + quantitative sim-to-real validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations and systematic experimental organization.
- Value: ⭐⭐⭐⭐ — Substantially advances the frontier of highly dynamic skill learning for humanoid robots.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AudioAvatar: Personalized Audio-driven Whole-body Talking Avatars](../../CVPR2026/human_understanding/audioavatar_personalized_audio-driven_whole-body_talking_avatars.md)
- [\[CVPR 2026\] InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions](../../CVPR2026/human_understanding/interprior_scaling_generative_control_for_physics-based_human-object_interaction.md)
- [\[CVPR 2026\] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](../../CVPR2026/human_understanding/cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)
- [\[CVPR 2026\] InterPhys: Physics-aware Human Motion Synthesis in a Dynamic Scene](../../CVPR2026/human_understanding/interphys_physics-aware_human_motion_synthesis_in_a_dynamic_scene.md)
- [\[CVPR 2026\] PAMotion: Physics-Aware Motion Generation for Full-Body Interaction with Multiple Objects](../../CVPR2026/human_understanding/pamotion_physics-aware_motion_generation_for_full-body_interaction_with_multiple.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving
description: >-
  [ICCV 2025][Autonomous Driving][Synthetic Data] This paper proposes SceneCrafter, a unified 3DGS-based simulation framework that simultaneously supports synthetic data generation and closed-loop evaluation via an adaptiv…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Synthetic Data"
  - "End-to-End Autonomous Driving"
  - "3D Gaussian Splatting"
  - "Closed-Loop Evaluation"
  - "Interactive Simulation"
date: 2026-05-08
content_hash: dccb4c4bcc7bf575
---

# Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: [2503.18108](https://arxiv.org/abs/2503.18108)  
**Code**: [GitHub](https://github.com/cancaries/SceneCrafter)  
**Area**: Autonomous Driving
**Keywords**: Synthetic Data, End-to-End Autonomous Driving, 3D Gaussian Splatting, Closed-Loop Evaluation, Interactive Simulation

## TL;DR
This paper proposes SceneCrafter, a unified 3DGS-based simulation framework that simultaneously supports synthetic data generation and closed-loop evaluation via an adaptive kinematic model and bidirectional interactive agent control. Experiments demonstrate that synthetic data significantly improves the generalization of end-to-end autonomous driving models, yielding up to 18% improvement in Route Completion.

## Background & Motivation
End-to-end (E2E) autonomous driving models are data-driven, and scene diversity along with trajectory distribution are critical to model performance. However, large-scale real-world data collection is expensive and time-consuming. Existing simulators exhibit notable limitations:

**Game engine simulators** (CARLA, SUMO): Sensor data realism is insufficient, resulting in a sim-to-real gap.

**NeRF/diffusion-based methods**: High computational and temporal cost, with spatial-temporal inconsistency.

**Existing 3DGS simulators** (HugSim): Lack of reasonable traffic flow interaction; only agent-to-ego unidirectional interaction is supported.

**Limitations of open-loop evaluation**: Existing open-loop metrics (L2 distance, collision rate) fail to genuinely assess planning quality, and models may learn shortcuts that merely fit the training distribution.

An ideal simulator should simultaneously possess: photorealistic rendering, real-time efficiency, spatial-temporal consistency, bidirectional interactive traffic flow, and the ability to generate training data for E2E models. No existing method satisfies all these requirements.

**Core Idea**: 3DGS reconstruction of real scenes + editable model library + adaptive kinematic model + bidirectional interaction control = an efficient and photorealistic unified simulation framework.

## Method

### Overall Architecture
SceneCrafter consists of two core components:
- **Scene Controller**: Manages the states and behaviors of all agents in the environment, dynamically updating traffic flow.
- **Scene Renderer**: Generates high-fidelity driving scene images based on updates from the Controller.

Two modes are supported: synthetic data generation (ego controlled by an expert planner) and closed-loop evaluation (ego controlled by a learned policy).

### Key Designs
1. **Heuristic Agent Generation and Bidirectional Interaction Control**:

    - Function: Generates diverse traffic participants on real HD maps and enables ego-agent bidirectional interaction.
    - Mechanism: Two generation strategies are provided — path-based (preferring spawn points likely to interact with ego) and trigger-based (agents activated when ego reaches a trigger point). At each timestep, an agent updates its behavior $b_j^{t+1}$ based on its own state, ego state, route, and map topology, establishing bidirectional interaction. Adversarial behavior modes (aggressive lane changes, emergency braking, etc.) are also introduced for long-tail scenario evaluation.
    - Design Motivation: Existing simulators (e.g., HugSim) only support agent-to-ego interaction and cannot model realistic complex traffic dynamics.

2. **Adaptive Kinematic Model (AKM)**:

    - Function: Corrects discrepancies between simulated and real ego dynamics.
    - Mechanism: Two learnable parameters $u_1, u_2$ are introduced into the standard bicycle model and estimated from real IMU data:
    $v_u = (1 - \boldsymbol{u_1})v_t + \boldsymbol{u_1}v_{t+1}$
    $x_{t+1} = x_t + v_u \cos(\varphi_t + \boldsymbol{u_2} \cdot \beta_t) \Delta t$
      $u_1$ regulates velocity interpolation (preventing unnatural displacements), and $u_2$ regulates directional dynamics.
    - Design Motivation: The standard bicycle model produces unnatural vehicle orientations during turns, and sim-to-real dynamic discrepancies significantly affect planning quality.

3. **Gaussian Splatting Model Library + Ground Height Estimation + Directional Shadows**:

    - Function: Generates high-fidelity driving scene images.
    - Mechanism:
        - **Background model**: Trained with Street Gaussians, retaining only static elements.
        - **Foreground model library**: Virtual vehicle assets constructed using BlenderNeRF and 3DRealCar.
        - **Ground height estimation**: Ground LiDAR points are extracted via RANSAC; a three-layer MLP fits a global ground model $\hat{z} = f(x, y)$ to correct z-axis offsets. The loss is asymmetric: MSE is applied when $\hat{z} > z$ (preventing placement below ground), and Huber loss when $\hat{z} \leq z$ (tolerating points slightly above ground).
        - **Directional shadows**: The global illumination direction is estimated, and multi-directional shadow 3DGS models are generated for each foreground object.
    - Design Motivation: Floating or sunken vehicles are common rendering artifacts; the absence of shadows reduces the realism of foreground objects.

### Loss & Training
AKM parameters are learned from real IMU data. Background and foreground 3DGS models are trained independently. During synthetic data generation, an expert planner controls the ego (following the original trajectory while adjusting speed according to traffic flow); during closed-loop evaluation, a learned policy replaces the expert planner.

## Key Experimental Results

### Main Results (Data Augmentation Effect)

| Training Data | Real volume | Sim volume | Planning L2 Avg(m)↓ | CR Avg(%)↓ |
|----------|-------------|------------|---------------------|------------|
| $\mathcal{D}_{200}$ | 8k | - | 1.39 | 0.52 |
| $\mathcal{D}_{200}$ + Sim | 8k | 2k | **1.18** | **0.45** |
| $\mathcal{D}_{recon}^c$ | 28k | - | 1.09 | 0.32 |
| $\mathcal{D}_{recon}^c$ + Sim | 28k | 2k | **1.04** | **0.25** |

### Closed-Loop Evaluation Results

| Model | Sim Data | RC↑(%) | VC↓(%) | LCR↓(%) | Turn RC↑(%) |
|------|---------|--------|--------|---------|------------|
| VAD | None | 42.01 | 7.55 | 7.48 | 25.71 |
| VAD | +6k | **53.03** | **3.43** | **6.08** | **34.72** |
| GenAD | None | 44.87 | 7.62 | 8.50 | 28.47 |
| GenAD | +6k | **49.65** | **5.45** | **5.76** | **34.02** |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Virtual vehicles (no shadow) vs. 3DRealCar (with shadow) | Car AP: 38.4 vs. 39.6 | Real vehicles + shadows improve detection scores |
| Bicycle Model (BM) vs. AKM | Planning L2 1s: 0.47 vs. **0.38** | AKM produces trajectories closer to ground truth |
| Interaction rate comparison | Sim > Real | Synthetic data has higher interaction rates; ego speed alteration distribution is more heavy-tailed |

### Key Findings
- Synthetic data yields significant gains on small-scale real datasets (Planning L2 reduced by 15%), with diminishing returns on large-scale datasets.
- In closed-loop evaluation, Route Completion improves by up to 18% (VAD) and Vehicle Collision Rate decreases by over 50%.
- AKM generates smoother turning trajectories that more closely match real IMU records.
- The interaction diversity of synthetic data (heavy-tailed distribution of ego speed alterations) effectively prevents models from learning "always accelerate/decelerate" shortcuts.
- This work presents the first systematic study of the effects of synthetic data on imitation-learning-based E2E models trained on real data and evaluated in photorealistic closed-loop environments.

## Highlights & Insights
- **Unified framework**: A single platform simultaneously supports data generation and closed-loop evaluation, which is unprecedented.
- **E2E model as surrogate evaluator**: A trained VAD model is used as a zero-shot evaluator to measure simulation realism, which is more meaningful than pixel-level metrics such as FID/PSNR.
- **Interaction quantification**: Interaction Rate and Ego Speed Alteration metrics are defined to measure traffic flow interaction quality.
- The asymmetric loss design for ground height estimation is concise and effective.

## Limitations & Future Work
- The current range of traffic participant types is limited (vehicles only); pedestrians and cyclists should be incorporated in future work.
- The rendering range is constrained by the original training scene; unseen regions require supplementation with lightweight diffusion models.
- In open-loop evaluation, marginal improvement from synthetic data is limited when added on top of large-scale real data, as the Waymo validation set is dominated by simple cruising scenarios.
- Foreground object insertion may introduce false-positive detection boxes.
- No quantitative rendering quality comparison with DriveArena, UniSim, or similar methods is provided, as these methods cannot generate complete driving logs.

## Related Work & Insights
- This is the first work to systematically study the effects of synthetic data on E2E models trained on real data.
- The learnable-parameter concept in AKM can be generalized to kinematic model calibration in other simulators.
- The bidirectional interaction control design can be applied to safety-critical scenario generation.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified framework and systematic study of synthetic data effects are novel, though individual component contributions are largely incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive experimental design covering open-loop, closed-loop, interaction analysis, ablation studies, and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, but placing some technical details in the appendix slightly disrupts coherence.
- Value: ⭐⭐⭐⭐⭐ Addresses the critical pain points of lacking interactivity and data generation capability in simulators for E2E autonomous driving research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](../../CVPR2026/autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)

</div>

<!-- RELATED:END -->

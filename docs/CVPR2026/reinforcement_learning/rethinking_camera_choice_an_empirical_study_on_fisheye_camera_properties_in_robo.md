---
title: >-
  [Paper Note] Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation
description: >-
  [CVPR 2026][Reinforcement Learning][fisheye camera] This paper presents the first systematic empirical study on the properties of wrist-mounted fisheye cameras in imitation learning for robotic manipulation. Centered on…
tags:
  - "CVPR 2026"
  - "Reinforcement Learning"
  - "fisheye camera"
  - "robotic manipulation"
  - "imitation learning"
  - "field of view"
  - "generalization"
date: 2026-05-08
content_hash: 19c42f34db8dfc00
---

# Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation

**Conference**: CVPR 2026
**arXiv**: [2603.02139](https://arxiv.org/abs/2603.02139)
**Authors**: Han Xue, Min Nan, Xiaotong Liu, Wendi Chen, Yuan Fang, Jun Lv, Cewu Lu, Chuan Wen (Shanghai Jiao Tong University, Southeast University, USTC, et al.)
**Project Page**: [robo-fisheye.github.io](https://robo-fisheye.github.io/)
**Area**: Reinforcement Learning
**Keywords**: fisheye camera, robotic manipulation, imitation learning, field of view, generalization

## TL;DR

This paper presents the first systematic empirical study on the properties of wrist-mounted fisheye cameras in imitation learning for robotic manipulation. Centered on three core research questions—spatial localization, scene generalization, and hardware generalization—it reveals both the advantages and limitations of wide field-of-view (FoV) imaging, and proposes Random Scale Augmentation (RSA) to address scale overfitting in cross-camera transfer.

## Background & Motivation

Fisheye cameras, with their ultra-wide FoV (> 180°), are increasingly adopted in robotic manipulation, yet the academic understanding of how they affect policy learning lags far behind their practical deployment.

**Existing Problems**:
- The effect of strong radial distortion introduced by fisheye cameras on visual encoders remains unclear
- The practical benefit of wide FoV across scenes of varying complexity lacks quantitative analysis
- Policy transfer between different fisheye lenses (hardware generalization) exhibits systematic failures whose root causes are unknown
- No systematic benchmark spanning simulation and the real world exists to guide large-scale fisheye dataset collection

**Core Motivation**: Establish the first systematic empirical research framework to answer three key research questions:

**RQ1 – Spatial Localization**: Can wide FoV enhance the spatial localization capability of policies?

**RQ2 – Scene Generalization**: How does a fisheye camera affect generalization to novel backgrounds?

**RQ3 – Hardware Generalization**: Can policies transfer across different fisheye lenses?

## Method

### Experimental Platform
- **Real World**: Flexiv Rizon 4 seven-axis robotic arm + DH AG-160-95 adaptive gripper; demonstration data collected via teleoperation using a Meta Quest 3 headset
- **Simulation**: A two-stage projection pipeline implemented in MuJoCo (first rendering a panoramic image, then projecting it to a fisheye view) with precise control over lens parameters
- **Task Design**: 3 real-world tasks (Pick Cup, Fold Towel, Hang Chinese Knot) + 6 simulation tasks adapted from Robomimic/MimicGen
- **Evaluation Metric**: Multi-stage normalized score, which provides finer granularity than binary success rate

### RQ1: Spatial Localization Analysis
**Hypothesis**: The wide FoV of fisheye cameras enhances policy localization by capturing more static environmental features as visual anchors; thus performance should strongly depend on scene visual complexity.

**Experimental Design**: Compare policy performance in feature-poor (plain background) vs. feature-rich (textured cloth/clutter) environments using state-free policies to isolate visual localization capability.

**Key Finding**: The fisheye + rich scene combination enables policies to complete high-precision manipulation relying solely on visual input, implicitly encoding the spatial relationship between the robot and its environment, rendering explicit proprioception redundant.

### RQ2: Scaling Law for Scene Generalization
**Hypothesis**: Fisheye policies exploit scene diversity more effectively, exhibiting a steeper performance improvement curve as the number of training scenes $N$ increases.

**Experimental Design**: Fix total data volume (e.g., 200 trajectories) and vary only the number of independent training scenes $N$ (from 1 to 8); evaluate zero-shot on completely unseen test backgrounds. 32 distinct background textures are used.

**Key Finding**: The wide FoV of the fisheye camera acts as implicit data augmentation, enabling policies to better leverage scene diversity. In the real world, fisheye policies achieve a near-perfect score (0.988) with only 8 diverse training scenes.

### RQ3: Hardware Generalization and RSA
**Key Challenge**: Policies overfit to the absolute pixel scale of a specific lens (Scale Overfitting). When deployed on a new lens, the change in object scale within the image causes the policy to misestimate depth—magnification leads to undershoot (object perceived as closer), and minification leads to overshoot (object perceived as farther).

**Random Scale Augmentation (RSA)**:
- During training, a scale factor $s$ (e.g., 0.7–1.3) is sampled uniformly at random
- When $s > 1$, the image is shrunk and padded with black borders (zoom-out effect)
- This forces the network to learn relative spatial relationships (e.g., target scale relative to the gripper) rather than absolute pixel size
- Simple and plug-and-play; no architectural modifications required

## Key Experimental Results

### Table 1: RQ1 – Real-World Spatial Localization (Normalized Score, State-Free Policy)

| Task | Camera Type | Feature-Poor Scene | Feature-Rich Scene | Gain |
|---|---|---|---|---|
| Pick Cup | Fisheye (State-free) | 0.525 | 0.800 | **+0.275** |
| Fold Towel | Fisheye (State-free) | 0.100 | 0.700 | **+0.600** |
| Hang Chinese Knot | Fisheye (State-free) | 0.200 | 0.500 | **+0.300** |

Fisheye + rich scene substantially outperforms the feature-poor condition across all tasks, with Fold Towel showing the largest gain of **+0.600**. In simulation, fisheye achieves a success rate of 0.66 in feature-rich scenes, compared to 0.34 for pinhole (**+0.32**).

### Table 2: RQ3 – RSA Scale Sensitivity Analysis (Simulation, Normalized Score)

| Scale Factor $S$ | Effect | Baseline | RSA |
|---|---|---|---|
| 0.70 | Strong zoom-in | 0.000 | **0.900** |
| 0.85 | Moderate zoom-in | 0.950 | **1.000** |
| 1.00 | Training scale | 1.000 | 1.000 |
| 1.15 | Moderate zoom-out | 0.750 | **0.975** |
| 1.30 | Strong zoom-out | 0.650 | **1.000** |

The baseline degrades sharply in an inverted-V pattern under scale shift (dropping to 0 at $S=0.70$), while RSA maintains robust performance above 0.9 across the full scale range.

### Supplementary Results

**RQ2 Scene Generalization (Real-World Pick Cup)**:

| Training Scenes $N$ | Pinhole | Fisheye |
|---|---|---|
| 1 | 0.081 | 0.556 |
| 4 | 0.238 | 0.869 |
| 8 | 0.181 | **0.988** |

The fisheye scaling curve is far steeper than that of pinhole; fisheye reaches near-perfect performance at $N=8$, while pinhole degrades.

**RQ3 Real-World Cross-Camera Transfer**:

| Lens | FoV | Scale Change | Baseline | RSA |
|---|---|---|---|---|
| Training lens | 180° | 1.0× | 1.000 | 1.000 |
| Narrow lens | 150° | ~1.2× (zoom-in) | 0.500 | **0.950** |
| Wide-angle lens | 220° | ~0.8× (zoom-out) | 0.003 | **0.600** |

The baseline nearly completely fails on the wide-angle lens (0.003); RSA raises it to 0.600.

## Highlights & Insights

- **First Systematic Empirical Study**: Fills the gap in systematic analysis of fisheye cameras in robot manipulation policy learning; the three research questions build progressively, and the conclusions offer actionable guidance
- **Critical Role of Scene Complexity**: Reveals the prerequisite for "fisheye being useful"—the wide FoV localization advantage is fully realized only in visually feature-rich environments; improvement is limited on plain backgrounds
- **Implicit Data Augmentation Effect**: The wide FoV of the wrist-mounted fisheye camera naturally introduces larger viewpoint variation during arm movement, equivalent to scene-level data augmentation, which is the fundamental source of its generalization advantage
- **Diagnosis and Remedy of Scale Overfitting**: Precisely identifies scale overfitting as the root cause of cross-camera transfer failure; the proposed RSA strategy is extremely simple (random scaling + black-border padding) yet highly effective
- **Practical Guidelines**: Provides three concrete recommendations for large-scale fisheye dataset collection: collect in feature-rich environments, maximize scene diversity, and train with RSA

## Limitations & Future Work

- **Wrist-Mounted Perspective Only**: All experiments are based on wrist-mounted fisheye cameras; third-person or multi-view fusion scenarios are not explored
- **Limited Task Scope**: 3 real-world tasks + 6 simulation tasks; more complex scenarios such as dexterous manipulation, long-horizon tasks, or high-precision assembly are not covered
- **Limitations of RSA**: Cross-camera transfer to the 220° wide-angle lens reaches only 0.600, still far from perfect; under extreme focal length changes, simulation performance is only 0.06, indicating that RSA cannot fully resolve all hardware discrepancies
- **Distortion Correction Not Considered**: The approach of first applying geometric correction before training policies is not explored, which may be a more direct path for cross-camera transfer
- **Imitation Learning Only**: Reinforcement learning and online adaptation are not addressed; the effectiveness of RSA under an RL paradigm remains unknown

## Related Work & Insights

- **Fisheye Cameras in Robotics**: FisheyeStereoNet (fisheye depth estimation), BiFuse/OmniFusion (omnidirectional depth) → focus on the perception layer, lacking systematic analysis of policy learning
- **Camera Selection in Robotic Manipulation**: UMI/ALOHA (wrist-camera setups), RoVi-Aug/MimicGen (visual augmentation) → all use pinhole cameras without considering the effect of FoV
- **Domain Adaptation and Generalization**: Domain Randomization, Random Crop Augmentation → RSA can be viewed as domain randomization along the scale dimension, but with more targeted design
- **Paper Positioning**: First systematic study of the impact of camera model selection from a policy learning perspective, filling the gap at the camera-selection node in the "camera → perception → policy" pipeline

## Rating

- Novelty: ⭐⭐⭐⭐ — As an empirical study, methodological innovation is limited, but the research questions raised and the RSA finding carry genuine practical value
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Dual-track validation in simulation and the real world, 6+3 tasks, rigorous ablation design with clear variable control
- Writing Quality: ⭐⭐⭐⭐ — The three-RQ structure is clear; the hypothesis–validation–conclusion organization is easy to follow
- Value: ⭐⭐⭐⭐ — Provides directly actionable guidelines for large-scale fisheye dataset collection; RSA is concise and practical

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](../../ACL2026/reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)
- [\[CVPR 2026\] RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](radar_closedloop_robotic_data_generation_via_seman.md)
- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](../../AAAI2026/reinforcement_learning/actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[ICLR 2026\] Routing, Cascades, and User Choice for LLMs](../../ICLR2026/reinforcement_learning/routing_cascades_and_user_choice_for_llms.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SimScale: Learning to Drive via Real-World Simulation at Scale
description: >-
  [CVPR 2026 (Oral)][Autonomous Driving][simulation data] This paper proposes the SimScale framework, which generates large-scale, high-fidelity simulation data by applying trajectory perturbations to existing driving logs, simulating reactive environment responses, and synthesizing sensor observations via neural rendering. Combined with pseudo-expert trajectory supervision and a sim-real co-training strategy, SimScale achieves substantial gains on NAVSIM v2 (navhard +8.6 EPDMS), with performance scaling smoothly with the volume of simulation data.
tags:
  - CVPR 2026 (Oral)
  - Autonomous Driving
  - simulation data
  - end-to-end planning
  - sim-to-real
  - data scaling
  - pseudo-expert trajectory
  - neural rendering
  - co-training
date: 2026-05-08
content_hash: 263ad85f9d2f17da
---

# SimScale: Learning to Drive via Real-World Simulation at Scale

**Conference**: CVPR 2026 (Oral)  
**arXiv**: [2511.23369](https://arxiv.org/abs/2511.23369)  
**Code**: [OpenDriveLab/SimScale](https://github.com/OpenDriveLab/SimScale)  
**Authors**: Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang et al. (CASIA, OpenDriveLab@HKU, Xiaomi EV)  
**Area**: Autonomous Driving  
**Keywords**: simulation data, end-to-end planning, sim-to-real, data scaling, pseudo-expert trajectory, neural rendering, co-training

## TL;DR
This paper proposes the SimScale framework, which generates large-scale, high-fidelity simulation data by applying trajectory perturbations to existing driving logs, simulating reactive environment responses, and synthesizing sensor observations via neural rendering. Combined with pseudo-expert trajectory supervision and a sim-real co-training strategy, SimScale achieves substantial gains on NAVSIM v2 (navhard +8.6 EPDMS), with performance scaling smoothly with the volume of simulation data.

## Background & Motivation

Fully autonomous driving requires learning sound decisions across a broad spectrum of scenarios, including safety-critical and out-of-distribution (OOD) cases. However:

- **Data distribution bias**: Real-world data collected by human experts is dominated by routine driving; safety-critical scenarios (hard braking, near-misses) and OOD scenes are severely underrepresented.
- **Demonstration bias**: Imitation learning policies are only exposed to states within the expert distribution and cannot learn to recover from deviations.
- **Limitations of existing simulation approaches**:
    - Traditional simulators (CARLA/MetaDrive): insufficient rendering realism, large sim-to-real gap.
    - NeRF/3DGS-based neural rendering: high visual quality but lacks scene interactivity (non-reactive environments).
    - Pure trajectory perturbation: generates new states but lacks corresponding high-quality sensor observations.

**Mechanism**: Ego trajectories are perturbed on existing real driving logs to produce new states; a reactive environment model simulates responses from other traffic participants; neural rendering generates high-fidelity multi-view images for each new state; and pseudo-expert supervision trajectories are generated for the perturbed states — enabling scalable synthesis of large training datasets.

## Method

### Overall Architecture: Perturb → React → Render → Annotate

The SimScale simulation data generation pipeline comprises three core modules:

### 1. Trajectory Perturbation

Perturbations are applied to the ego vehicle's original trajectory over the interval $[T, T+H]$, producing new state sequences that deviate from normal driving. Perturbation types include lateral offsets and speed variations, driving the ego into states absent from the original data.

### 2. Reactive Environment Rollout

Following ego perturbation, other traffic participants (vehicles, pedestrians, etc.) must respond accordingly. A reactive simulation engine (based on MTGS and related work) is employed to ensure physical plausibility and interaction consistency, preventing artifacts such as clipping or unrealistic collisions.

### 3. Neural Rendering

High-fidelity multi-view camera observations are synthesized using 3D Gaussian Splatting (3DGS), conditioned on the perturbed ego pose and the reactive environment state, providing visual inputs for the end-to-end model.

### 4. Pseudo-Expert Trajectory Generation

Action supervision labels are provided for simulated states. Two strategies are compared:

- **Recovery-based**: At the perturbation endpoint $T+H$, a trajectory is planned directly from the current deviated state back to a reasonable driving state. Analogous to DAgger, this teaches the model "how to correct after an error."
- **Planner-based**: A rule-based planner (PDM) replans the optimal trajectory within the simulated environment, providing higher-quality action supervision.

### 5. Sim-Real Co-Training Strategy

Simulation and real data are mixed for joint training using a straightforward co-training strategy without complex domain adaptation. The approach is compatible with multiple end-to-end planner types:

- **Regression-based** (LTF / Transfuser): directly regresses trajectory waypoints.
- **Diffusion-based** (DiffusionDrive): generates trajectory distributions via diffusion models.
- **Scoring-based** (GTRS-Dense): ranks candidate trajectories by score. This paradigm additionally supports a **rewards-only** mode, where simulation data provides only reward signals rather than imitation learning supervision.

## Key Experimental Results

Evaluation is conducted on the NAVSIM v2 benchmark, comprising two splits: **navhard** (high-difficulty safety-critical scenarios) and **navtest** (standard test set).

### Table 1: Model Zoo Main Results (EPDMS Metric)

| Model | Backbone | Co-Train Mode | navhard EPDMS | navhard Gain | navtest EPDMS | navtest Gain |
|---|---|---|---|---|---|---|
| LTF | ResNet34 | w/ pseudo-expert | 30.3 | **+6.9** | 84.4 | **+2.9** |
| DiffusionDrive | ResNet34 | w/ pseudo-expert | 32.6 | **+5.1** | 85.9 | **+1.7** |
| GTRS-Dense | ResNet34 | w/ pseudo-expert | 46.1 | **+7.8** | 84.0 | **+1.7** |
| GTRS-Dense | ResNet34 | rewards only | 46.9 | **+8.6** | 84.6 | **+2.3** |
| GTRS-Dense | V2-99 | w/ pseudo-expert | 47.7 | **+5.8** | 84.5 | **+0.5** |
| GTRS-Dense | V2-99 | rewards only | 48.0 | **+6.1** | 84.8 | **+0.8** |

**Key Findings**:
- All planner types benefit from simulation data, with particularly pronounced gains on navhard (+5.1 ~ +8.6).
- GTRS-Dense + rewards only achieves the largest navhard gain (+8.6), indicating that scoring-based planners can fully leverage simulation data through reward signals alone, without pseudo-expert trajectory labels.
- Consistent improvements on navtest (+0.5 ~ +2.9) confirm that simulation data also enhances generalization.

### Table 2: Scalability Analysis — Simulation Data Volume vs. Performance

| Simulation Rounds | Sim Tokens | GTRS navhard (pseudo-expert) | GTRS navhard (rewards only) | LTF navhard |
|---|---|---|---|---|
| 0 (real data only) | 0 | 38.3 | 38.3 | 23.4 |
| 1 round (round 0) | ~65K | 42.5 | 43.1 | 27.8 |
| 3 rounds (round 0–2) | ~166K | 44.8 | 45.6 | 29.5 |
| 5 rounds (round 0–4) | ~236K | 46.1 | 46.9 | 30.3 |

**Scalability Insights**:
- Performance improves smoothly with increasing simulation data volume, with no clear saturation.
- Expanding simulation data alone — without adding real data — consistently yields further gains.
- Different planner architectures exhibit distinct scaling characteristics: scoring-based planners scale best, followed by diffusion-based planners.

## Highlights & Insights

- **CVPR 2026 Oral**: Selected for oral presentation, reflecting high recognition from the community.
- **Complete simulation-training closed loop**: The pipeline from perturbation to reaction to rendering to annotation to training forms a fully scalable data augmentation framework.
- **Pseudo-experts should be exploratory**: Recovery-based pseudo-experts teach the model to recover from mistakes, outperforming planner-based supervision in certain scenarios — suggesting that data diversity matters more than trajectory optimality.
- **Multimodal modeling enables scaling**: Diffusion-based and scoring-based planners leverage expanded simulation data more effectively than regression-based planners, as they model trajectory distributions rather than point estimates.
- **Reward is All You Need**: GTRS-Dense achieves its best performance in rewards-only mode, demonstrating that for scoring-based planners, imitation learning supervision on simulation data is unnecessary — reward signals suffice.
- **Sim-to-real gap is manageable**: A straightforward co-training strategy is sufficient; complex domain adaptation or domain randomization techniques are unnecessary, attributable to the high fidelity of neural rendering.
- **Fully open-sourced**: TB-scale simulation data, training code, and model weights are all publicly released, enabling strong reproducibility.

## Limitations & Future Work

- **Infrastructure dependency**: High-quality 3DGS neural rendering (MTGS) and a reactive simulation engine are prerequisites, incurring significant upfront cost.
- **Large simulation data footprint**: Five rounds of simulation produce several TB of sensor data, resulting in substantial storage and I/O overhead.
- **Scene diversity bounded by original logs**: Perturbations can only generate variants in the neighborhood of existing scenes; entirely new scenario types (e.g., snowy conditions absent from the original data) cannot be synthesized.
- **Evaluation scope**: Validation is primarily conducted on NAVSIM v2 closed-loop evaluation; generalization to other benchmarks (e.g., nuPlan, CARLA closed-loop) remains unverified.
- **Pseudo-expert quality ceiling**: The quality of pseudo-expert trajectories is fundamentally limited by the performance ceiling of the PDM planner.
- **Longer simulation horizons unexplored**: The current simulation window is fixed at 6 seconds; longer rollouts and accumulated error handling have not been addressed.

## Related Work & Insights

- **End-to-end autonomous driving**: Methods such as UniAD, VAD, and Transfuser enable direct sensor-to-trajectory planning but are constrained by insufficient safety-critical scenarios in training data.
- **Driving scene simulation**: The field has progressed from traditional rendering (CARLA/MetaDrive) to high-fidelity neural rendering (NeRF/3DGS, which remains static) and further to reactive simulation (e.g., DriveArena, MTGS). SimScale builds on reactive simulation and adds scalable pseudo-expert generation.
- **Data scaling and co-training**: DAgger-style methods (online interaction) and large-scale collection efforts (DROID/Scaling-up) pursue data scaling through real-world collection; SimScale instead scales via simulation, avoiding the cost of additional data acquisition.
- **Scoring-based planning**: Reward-based trajectory selection paradigms such as GTRS demonstrate a unique advantage in sim-real settings (rewards-only mode), as validated in this work.

## Rating

- **Novelty**: 4/5 — Formalizes a complete closed-loop framework integrating trajectory perturbation, reactive simulation, neural rendering, and pseudo-expert generation; provides the first systematic study of simulation data scaling laws for end-to-end planners.
- **Experimental Thoroughness**: 5/5 — Covers 3 planner architectures × 2 backbones × 2 pseudo-expert strategies × 5 scaling rounds with comprehensive ablations; data and code are fully open-sourced.
- **Writing Quality**: 4/5 — Well-structured with clearly articulated core insights (three scaling findings); meets CVPR Oral standards.
- **Value**: 5/5 — Establishes a scalable simulation data augmentation paradigm for end-to-end autonomous driving with a mature open-source ecosystem and strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)
- [\[ICCV 2025\] LookOut: Real-World Humanoid Egocentric Navigation](../../ICCV2025/autonomous_driving/lookout_real-world_humanoid_egocentric_navigation.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[ICLR 2026\] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](../../ICLR2026/autonomous_driving/egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)
- [\[NeurIPS 2025\] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](../../NeurIPS2025/autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework
description: >-
  [ICML 2026][World Models] iWorld-Bench is the first unified evaluation benchmark specifically designed for "Interactive World Models." It proposes an Action Generation Framework that maps text, one-hot…
tags:
  - "ICML 2026"
  - "World Models"
  - "Action-Controllable Video Generation"
  - "Camera Control"
  - "Memory Alignment"
  - "Cross-modal Evaluation"
date: 2026-05-08
content_hash: 4a689679c528028f
---

# iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework

**Conference**: ICML 2026  
**arXiv**: [2605.03941](https://arxiv.org/abs/2605.03941)  
**Code**: iWorld-Bench.com (Project Page)  
**Area**: Interactive World Models / Benchmark / Video Generation Evaluation  
**Keywords**: World Models, Action-Controllable Video Generation, Camera Control, Memory Alignment, Cross-modal Evaluation

## TL;DR
iWorld-Bench is the first unified evaluation benchmark specifically designed for "Interactive World Models." It proposes an Action Generation Framework that maps text, one-hot, and camera extrinsic/intrinsic inputs into a shared instruction space. Based on 4.9K tasks selected from 330K videos and 9 metrics, the study provides a comprehensive comparison of 14 mainstream models.

## Background & Motivation
**Background**: World models (e.g., Genie, HunyuanVideo, Wan, Matrix-Game, CameraCtrl) are evolving toward interactivity, where they must predict the future while responding to external action commands. Potential applications include game engines, autonomous driving, and embodied AI.

**Limitations of Prior Work**: (1) Existing benchmarks are mostly single-perspective and single-scene, often restricted to pedestrian street views. (2) "Action" modalities vary wildly across models—textual instructions, one-hot keys, or camera parameters—making direct comparison impossible (e.g., "move forward" corresponds to dozens of different low-level signals). (3) Most benchmarks are designed for general T2V or embodied manipulation, lacking evaluation of interaction response, camera following, and memory capabilities.

**Key Challenge**: The lack of cross-modal action semantic alignment and a task system that fails to cover the core dimension of "interactivity" results in "apples-to-oranges" comparisons for world model evaluation.

**Goal**: (1) Construct a multi-scene, multi-perspective, all-weather dataset for world models; (2) Provide a cross-modal unified action encoding framework; (3) Design task sets that distinguish "action-following ability" from "memory ability" with 9 corresponding metrics.

**Key Insight**: First-person camera movement can be decomposed into orthogonal translation and rotation subspaces. By discretizing each subspace into 27 actions, a total of 729 combinations are formed. Filtering for 81 core actions compatible with most models creates a unified "Action Dictionary," to which all input modalities are mapped.

**Core Idea**: Use a "Unified Action Dictionary" to bring heterogeneous world models onto a level playing field. Overlay this with multi-scene/weather data and closed-loop tasks to evaluate action following, visual quality, and spatial memory.

## Method

### Overall Architecture
iWorld-Bench consists of three components: (1) Data Pipeline — 330,000 videos collected from 12 public datasets (KITTI, NCLT, TartanAir, SpatialVid, etc.) and 4 simulators (AerialVLN, UAV_ON, Openfly, EmbodiedCity), with unified coordinate systems/camera formats and 2,100 evaluation videos selected via VLM labeling and manual audit; (2) Action Generation Framework — defines a 729-dimensional action space mapped to 81 cross-modal compatible actions, each represented as a triplet (text, one-hot, camera parameters); (3) 6 Task Categories — Difficulty 1-4 action following (4,000 tasks), Memory (200 tasks requiring closed-loop paths to return to the origin), and Camera Following (700 tasks specifically for camera-parameter models), totaling 4,900 tasks with 9 metrics.

### Key Designs

1.  **Unified Action Dictionary (Action Generation Framework)**:
    - **Function**: Unifies heterogeneous instructions like "move forward 1m," "press W," or "translation (0,0,1)" into a discrete code for cross-modal comparison.
    - **Mechanism**: Decomposes first-person motion into translation $T$ and rotation $R$ subspaces, each with 27 atomic actions, yielding a $|T|\times|R|=729$ space. Given that many models do not support vertical translation or roll, it filters for $9\times9=81$ core actions. Each action is mapped to a (text, one-hot, camera intrinsics+extrinsics) triplet with difficulty $D\in\{1,...,6\}$ and validity $V\in\{0,1\}$ labels.
    - **Design Motivation**: Comparing camera-parameter models using only text prompts would disadvantage low-degree-of-freedom models. The unified dictionary standardizes evaluation in a shared 81-dimensional subspace.

2.  **Closed-loop Memory Ability Task**:
    - **Function**: Detects whether a world model possesses spatial memory during long-range reasoning (i.e., whether the starting scene and geometry can be reproduced when returning to the origin).
    - **Mechanism**: Constructs closed camera trajectories (Start = End) and requires the model to reason the full sequence. Performance is measured by "Memory Symmetry" (pixel consistency of symmetric frames) and "Trajectory Alignment" (mirror similarity of instantaneous displacement vectors).
    - **Design Motivation**: Traditional open-loop benchmarks allow "goldfish-memory" models to score high; closed-loop tests expose issues like KV-cache truncation and attention decay. Results show over half of the models have Symmetry < 0.5.

3.  **9-Dimensional Metric System**:
    - **Function**: Categorizes "usability" into Visual Quality (4), Action Following (3), and Memory (2).
    - **Mechanism**: Visual quality uses MUSIQ, temporal consistency, HSV drift, and Tenengrad+BRISQUE for sharpness. Action following uses ViPE to extract trajectories from generated videos and compares them with GT to separate "intrinsic estimation error" from "execution error."
    - **Design Motivation**: Single scores mask trade-offs. For example, CogVideoX ranks first in visual quality but achieves only 0.595 in camera following, highlighting a typical trade-off.

### Loss & Training
This is an evaluation benchmark; no training is conducted. Evaluation protocol: All 14 models run original inference settings (default sampling) on NVIDIA A800, driven by the same initial frame and action sequence per task. All 9 metrics were calibrated via human preference experiments.

## Key Experimental Results

### Main Results
Total score comparison for 14 models across 4 action-following and memory task types (abridged):

| Control Type | Model | Avg | Trajectory Acc | Memory Sym | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| One-hot | HY-World 1.5 | **0.787** | 0.747 | **0.848** | 1 |
| Camera | videox-fun-Wan | 0.747 | 0.717 | **0.901** | 2 |
| Text | HunyuanVideo-1.5 | 0.719 | 0.684 | 0.634 | 3 |
| Camera | AC3D | 0.715 | 0.579 | 0.907 | 4 |
| Text | CogVideoX-I2V | 0.696 | 0.595 | 0.601 | 5 |
| Camera | MotionCtrl | 0.549 | 0.673 | 0.310 | 14 |

For the 700 camera-parameter tasks, AC3D leads significantly with 0.909 Trajectory Tolerance and 0.992 Motion Smoothness, while the early method ASTRA only reaches 0.428.

### Ablation Study
Cross-modal comparison (differences in execution across input modalities for the same action semantics):

| Dimension | Text Control (Avg of 5) | One-hot (Avg of 2) | Camera (Avg of 7) | Insight |
| :--- | :--- | :--- | :--- | :--- |
| Image Quality | 0.64 (High) | 0.58 | 0.51 | Text models inherit strong T2V visual priors. |
| Trajectory Acc | 0.62 | **0.72** | 0.62 | One-hot discrete signals provide strongest control. |
| Memory Sym | 0.51 | 0.59 | 0.59 | Text models are most prone to "forgetting." |

### Key Findings
- **Universal Visual Quality ↔ Controllability Trade-off**: CogVideoX-I2V has the highest visual consistency (0.899) but low trajectory accuracy (0.595). Conversely, AC3D (fine-tuned from CogVideoX) shows sharp gains in control but degraded visual metrics, indicating a cost for camera-control fine-tuning.
- **One-hot Discrete Actions are Overall Optimal**: Models using keyboard-style encoding (HY-World 1.5, Matrix-Game 2.0) outperform text models in action following, suggesting that discrete, strongly aligned action signals are better for learning physics.
- **Memory Ability is a Global Weakness**: None of the 14 models achieved Memory Symmetry > 0.85; almost all models deviated from the initial scene when returning to the origin.
- **Strong Encoders Enable Early Feature Formation**: Early methods (MotionCtrl, CameraCtrl) lag significantly in Trajectory Tolerance, reflecting the progress of newer camera-control injection mechanisms.

## Highlights & Insights
- The "729→81 Action Dictionary" is the most clever design: it forces uncomparable modalities into alignment and is extensible for future modalities.
- Framing memory as a closed-loop path task is simple yet powerful, exposing the "goldfish memory" of existing world models.
- Separating Trajectory Accuracy from Trajectory Tolerance (the latter comparing GT video trajectories via the same estimator) effectively cancels out simulator noise from the ViPE estimator.
- Data diversity across UGV/UAV/human/robot perspectives and various weather/lighting conditions surpasses earlier benchmarks like WorldScore (3,000 cases).

## Limitations & Future Work
- Evaluation primarily uses single inference results rather than averaged multi-run results; sampling-sensitive models might be undervalued.
- The 81-action dictionary remains discrete, with limited coverage for truly continuous fine-grained operations (e.g., tiny steering angles).
- Camera trajectory metrics depend on the ViPE extractor; errors in ViPE could affect multiple scores.
- Real-time performance and long-term consistency (>10s) were not evaluated and are listed as future work.
- High cost of data collection (330K videos + 4 simulators).

## Related Work & Insights
- **vs WorldScore (Duan 2025)**: WorldScore focuses on general generation without interactive/memory tasks; iWorld-Bench treats interactivity as a first-class citizen.
- **vs EWMBench / WorldEval**: Those focus on robotic arm manipulation; iWorld-Bench focuses on first-person mobile world models.
- **vs VBench (Huang 2024)**: VBench evaluates general T2V quality; iWorld-Bench is complementary by focusing on action controllability.
- **vs MovBench / WorldBench**: Previous benchmarks are single-modal; this work achieves cross-modal comparability via the Action Generation Framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The cross-modal dictionary and closed-loop memory tasks are genuine conceptual innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 14 models, 4,900 tasks, 9 metrics, and 3 input modalities offer rare depth.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear narrative and high information density, though sub-task hierarchies are slightly redundant.
- **Value**: ⭐⭐⭐⭐ — Establishes the first standard metric for the emerging field of "Interactive World Models."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](../../ICLR2026/others/distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](../../AAAI2026/others/beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[ICML 2026\] CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities](cybergym-e2e_scalable_real-world_benchmark_for_ai_agents_end-to-end_cybersecurit.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](../../ICLR2026/others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](../../CVPR2026/others/next-scale_autoregressive_models_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework
description: >-
  [ICML 2026][LLM Evaluation][World Model] iWorld-Bench is the first unified evaluation benchmark specifically designed for "interactive world models." It proposes an Action Generation Framework that maps three types of ac…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "World Model"
  - "Action-Controllable Video Generation"
  - "Camera Control"
  - "Memory Ability"
  - "Cross-Modal Evaluation"
date: 2026-05-08
content_hash: b43c92f32e4d4fc1
---

# iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework

**Conference**: ICML 2026  
**arXiv**: [2605.03941](https://arxiv.org/abs/2605.03941)  
**Code**: iWorld-Bench.com (project homepage)  
**Area**: Interactive World Models / Benchmark / Video Generation Evaluation  
**Keywords**: World Model, Action-Controllable Video Generation, Camera Control, Memory Ability, Cross-Modal Evaluation

## TL;DR
iWorld-Bench is the first unified evaluation benchmark specifically designed for "interactive world models." It proposes an Action Generation Framework that maps three types of action inputs—text, one-hot, and camera intrinsics/extrinsics—into a unified command space. Based on 330K videos, it carefully selects 4.9K tasks and 9 metrics to comprehensively compare 14 mainstream models.

## Background & Motivation
**Background**: World models (e.g., Genie, HunyuanVideo, Wan, Matrix-Game, CameraCtrl) are evolving toward an interactive paradigm that can both predict the future and respond to external action commands. Potential applications span game engines, autonomous driving, and embodied intelligence.

**Limitations of Prior Work**: (1) Existing benchmarks are mostly single-view, single-scene, and often use pedestrian street scenes; (2) Different models use entirely different "action" modalities—text commands, one-hot keys, camera intrinsics/extrinsics—so the same "move forward" command corresponds to dozens of different low-level signals across models, making direct comparison impossible; (3) Most benchmarks are designed for general T2V or embodied manipulation, lacking evaluation of interaction response, camera following, and memory ability.

**Key Challenge**: The lack of cross-modal action semantic alignment and the absence of an evaluation system covering "interactivity" as a core dimension result in world model evaluation being a comparison of "apples to oranges."

**Goal**: (1) Build a multi-scene, multi-view, all-weather world model dataset; (2) Provide a cross-modal unified action encoding framework; (3) Design a task set that distinguishes "action following ability" and "memory ability," with 9 corresponding metrics.

**Key Insight**: The authors observe that first-person camera motion can be decomposed into orthogonal translation and rotation subspaces, each discretized into 27 actions, yielding 729 combinations. Filtering for compatibility with most models results in 81 actions, forming a unified "action dictionary." All modalities are mapped to this dictionary.

**Core Idea**: Use a "unified action dictionary" to bring heterogeneous world models onto the same evaluation plane, then layer on multi-scene, multi-weather data and closed-loop task designs to comprehensively assess action following, visual quality, and spatial memory.

## Method

### Overall Architecture
iWorld-Bench consists of three components: (1) Data pipeline—collects 330K videos from 12 public datasets (KITTI, NCLT, TartanAir, SpatialVid, etc.) and 4 simulators (AerialVLN, UAV_ON, Openfly, EmbodiedCity), unifies coordinate systems/camera parameter formats, and selects 2,100 evaluation videos after VLM auto-labeling and manual verification; (2) Action Generation Framework—defines a 729-dimensional action space, mapped to 81 cross-modal compatible actions, each with a triplet representation (text, one-hot, camera parameters); (3) Six types of evaluation tasks—Difficulty 1-4 action following (4,000 tasks), Memory (200 tasks, closed-loop paths requiring the model to return to the origin), and Camera Following (700 tasks, specifically for camera parameter models), totaling 4,900 tasks with 9 metrics.

### Key Designs

1. **Unified Action Dictionary (Action Generation Framework)**:

    - **Function**: Unifies heterogeneous commands such as "move forward 1m," "press W," and "translate (0,0,1)" into the same discrete encoding, enabling cross-modal comparison.
    - **Mechanism**: First-person camera motion is decomposed into translation $T$ and rotation $R$ subspaces, each with 27 atomic actions, yielding a combination space of $|T|\times|R|=729$. Considering most models do not support vertical translation or roll/pitch, 81 core actions ($9\times9$) are selected. Each action is mapped to a (text, one-hot, camera intrinsics+extrinsics) triplet, and labeled with difficulty $D\in\{1,...,6\}$ and validity $V\in\{0,1\}$.
    - **Design Motivation**: Directly comparing text prompts with camera parameter models disadvantages low-DoF models. The unified dictionary projects all models into an 81-dimensional common subspace, ensuring fairness and extensibility (new modalities only require an additional mapping).

2. **Closed-Loop Memory Ability Tasks**:

    - **Function**: Tests whether world models possess spatial memory in long-range reasoning—can they reproduce the initial scene and geometry when returning to the starting point.
    - **Mechanism**: Constructs closed camera trajectories (start = end), requiring the model to infer the entire sequence at once. Evaluated using "Memory Symmetry" (pixel consistency of symmetric frames along the trajectory) and "Trajectory Alignment" (mirror similarity of instantaneous displacement vectors).
    - **Design Motivation**: Previous benchmarks only tested open-loop following, allowing even "goldfish memory" models to score high. Closed-loop testing exposes real issues such as KV-cache truncation and attention decay. Results show over half the models have Symmetry < 0.5.

3. **9-Dimensional Evaluation Metric System**:

    - **Function**: Decomposes "usability" into three non-redundant categories: visual quality (4), action following (3), and memory (2), totaling 9 sub-scores.
    - **Mechanism**: Visual quality is measured by MUSIQ, brightness-temperature-time consistency, HSV color drift, and Tenengrad+BRISQUE for sharpness; action following uses ViPE to extract generated video camera trajectories and compares them to GT, separating "intrinsic estimation error" and "model execution error"; memory metrics as above. All metrics are validated against human preference alignment.
    - **Design Motivation**: A single score can mask real trade-offs such as "good visual quality but poor controllability" or vice versa. Table 3 shows that the text-controlled model CogVideoX ranks first in visual quality but only 0.595 in camera following, exemplifying this trade-off.

### Loss & Training
This work is an evaluation benchmark and does not involve training. Evaluation protocol: all 14 models are run under their original inference settings (default sampling, no ensemble) on NVIDIA A800; each task uses the same initial frames and action sequences. All 9 metrics are calibrated via human preference experiments (significant correlation with human judgment).

## Key Experimental Results

### Main Results
Comparison of total scores for 14 models on 4 types of action following and memory tasks (excerpt):

| Control Mode | Model | Avg | Trajectory Acc | Memory Sym | Rank |
|--------------|-------|-----|----------------|------------|------|
| One-hot | HY-World 1.5 | **0.787** | 0.747 | **0.848** | 1 |
| Camera | videox-fun-Wan | 0.747 | 0.717 | **0.901** | 2 |
| Text | HunyuanVideo-1.5 | 0.719 | 0.684 | 0.634 | 3 |
| Camera | AC3D | 0.715 | 0.579 | 0.907 | 4 |
| Text | CogVideoX-I2V | 0.696 | 0.595 | 0.601 | 5 |
| Camera | MotionCtrl | 0.549 | 0.673 | 0.310 | 14 |

Camera Following ranking on 700 camera parameter tasks: AC3D leads with Trajectory Tolerance 0.909 and Motion Smoothness 0.992, while the early method ASTRA scores only 0.428.

### Ablation Study
Cross-modal comparison table (execution differences for the same action semantics across input modalities):

| Dimension | Text Control (mean of 5 models) | One-hot (mean of 2 models) | Camera (mean of 7 models) | Interpretation |
|-----------|---------------------------------|----------------------------|---------------------------|---------------|
| Image Quality | 0.64 (high) | 0.58 | 0.51 | Text models inherit T2V visual priors |
| Trajectory Accuracy | 0.62 | **0.72** | 0.62 | One-hot discrete signals offer strongest controllability |
| Memory Symmetry | 0.51 | 0.59 | 0.59 | Text models most prone to "forgetting the path" |

### Key Findings
- **There is a universal trade-off between visual quality and controllability**: CogVideoX-I2V achieves the highest visual consistency (0.899) but only 0.595 in trajectory accuracy; AC3D, fine-tuned from CogVideoX, shows the opposite—improved control at the cost of visual metrics, indicating that fine-tuning for camera control comes at a price.
- **One-hot discrete actions offer overall optimal controllability**: Models using keyboard-style encoding (HY-World 1.5, Matrix-Game 2.0) outperform text models in action following, suggesting that "action signals must be discrete and strongly aligned" to learn physical dynamics well.
- **Memory ability is a common weakness**: None of the 14 models achieves Memory Symmetry > 0.85; almost all deviate from the initial scene when returning to the origin in closed-loop tasks.
- **Strong encoders yield clear features at early stages**: Early methods (MotionCtrl, CameraCtrl) lag significantly in Trajectory Tolerance, reflecting real progress brought by new camera control injection mechanisms.

## Highlights & Insights
- The "729→81 action dictionary" is the most ingenious aspect: aligning dimensions to force comparability across modalities, with extensibility—new modalities only require an additional mapping table.
- Framing memory ability as a closed-loop path task is a simple yet powerful design—directly exposing the "goldfish memory" problem in all current world models.
- Separating Trajectory Accuracy and Trajectory Tolerance (the former compares camera commands, the latter compares GT video trajectories estimated by the same estimator) effectively offsets noise from the ViPE estimator, a valuable evaluation design trick.
- The dataset covers UGV/UAV/human/robot perspectives, 9 outdoor weather types, and 5 indoor lighting conditions, making it more comprehensive than WorldScore (3,000 cases) and EWMBench (2,100 robotic arm cases).

## Limitations & Future Work
- Evaluation is based mainly on single inference runs, without multiple averages or ensemble; some sampling-sensitive models may be underestimated.
- The 81 actions remain a discrete dictionary, limiting coverage of truly continuous fine-grained operations (e.g., small steering angles).
- Evaluation metrics for camera trajectories rely on the ViPE extractor; if ViPE itself has large errors in certain scenarios, multiple scores may be affected.
- Real-time performance and long-range consistency (>10s videos) are not evaluated; the authors list these as future work.
- The entire dataset (330K videos + 4 simulator renderings) is costly to collect, posing a high barrier to reproduction.

## Related Work & Insights
- **vs WorldScore (Duan 2025)**: WorldScore focuses on general world generation, without interactive/memory task design; iWorld-Bench explicitly treats "interaction" as a first-class citizen.
- **vs EWMBench / WorldEval**: Both focus on robotic arm embodied policy evaluation, a different track; iWorld-Bench targets first-person motion world models.
- **vs VBench (Huang 2024)**: VBench is a general video generation quality benchmark, not involving action controllability; iWorld-Bench is complementary.
- **vs MovBench / WorldBench**: These benchmarks are single-modality; this work is the first to achieve cross-modal comparability via the Action Generation Framework.

## Rating
- Novelty: ⭐⭐⭐⭐ — The cross-modal action dictionary and closed-loop memory tasks are genuine conceptual innovations, filling a key gap in world model evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 14 models × 4,900 tasks × 9 metrics × three input modalities, with rare depth and breadth.
- Writing Quality: ⭐⭐⭐⭐ — Clear narrative and information-dense tables, though the subtask design hierarchy is somewhat redundant and may confuse first-time readers.
- Value: ⭐⭐⭐⭐ — Provides the first recognized benchmark for the emerging "interactive world model" direction, serving as an anchor for subsequent evaluation and model iteration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](../../CVPR2026/llm_evaluation/vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)
- [\[ACL 2026\] CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems](../../ACL2026/llm_evaluation/clarity_a_framework_and_benchmark_for_conversational_language_ambiguity_and_unan.md)
- [\[NeurIPS 2025\] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally](../../NeurIPS2025/llm_evaluation/words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)
- [\[ICML 2026\] Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees](decompose_structure_and_repair_a_neuro-symbolic_framework_for_autoformalization_.md)

</div>

<!-- RELATED:END -->

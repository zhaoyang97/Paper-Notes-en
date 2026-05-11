---
title: >-
  [Paper Note] TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation
description: >-
  [CVPR 2026][LLM Evaluation][Football tactical imitation] This paper presents TacSIm, the first large-scale dataset and benchmark that reconstructs full-team trajectories from real Premier League broadcast footage and per…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Football tactical imitation"
  - "multi-agent learning"
  - "trajectory reconstruction"
  - "tactical evaluation"
  - "virtual simulation"
date: 2026-05-08
content_hash: 8c4434d803192491
---

# TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation

**Conference**: CVPR 2026
**arXiv**: [2603.25199](https://arxiv.org/abs/2603.25199)
**Code**: TacSIm (publicly available)
**Area**: Sports Analytics / Multi-Agent Imitation Learning
**Keywords**: Football tactical imitation, multi-agent learning, trajectory reconstruction, tactical evaluation, virtual simulation

## TL;DR

This paper presents TacSIm, the first large-scale dataset and benchmark that reconstructs full-team trajectories from real Premier League broadcast footage and performs tactical style imitation in a virtual football environment, quantifying imitation fidelity via two metrics: spatial occupancy similarity and motion vector similarity.

## Background & Motivation

**Background**: Imitation learning research in football is predominantly reward-optimization-driven (e.g., proxy metrics such as goal count and win rate), focusing on behavioral cloning of individual actions or reinforcement learning policy optimization, rather than accurately replicating the tactical organization of real teams.

**Limitations of Prior Work**: Three major challenges constrain the development of tactical imitation. First, data acquisition is restricted — fine-grained tracking data from top leagues is locked behind commercial barriers, while broadcast footage suffers from multi-camera switching, occlusion, and inconsistent frame rates, making it difficult to obtain full 11v11 team trajectories. Second, there is an imbalance between individual behavioral cloning and team-level cooperative optimization during imitation, resulting in weak generalization under partial observability. Third, evaluation frameworks focus on individual error or segment-level rewards, lacking systematic assessment of team-level spatiotemporal consistency.

**Key Challenge**: Existing research lacks a unified closed-loop benchmark spanning real matches to virtual simulation, making fair evaluation of tactical imitation quality across different methods impossible.

**Goal**: (1) How to obtain standardized full-team trajectory data from broadcast footage; (2) How to define and quantify the quality of tactical style imitation; (3) How to fairly compare different imitation learning methods within a unified environment.

**Key Insight**: The authors start from Premier League broadcast footage, recover full-team coordinates via camera calibration, trajectory reconstruction, and VAE-based completion, then map these to the Google Research Football (GRF) virtual environment for tactical replay and evaluation.

**Core Idea**: Construct the first football tactical imitation benchmark spanning broadcast footage to virtual simulation, and systematically evaluate team-level tactical style reproduction capability using a dual-metric framework of spatial occupancy and motion vectors.

## Method

### Overall Architecture

The TacSIm pipeline consists of three stages: (1) **Data acquisition** — reconstructing standardized player and ball coordinates from Premier League broadcast footage via object detection, tracking, and camera calibration; (2) **Trajectory completion** — filling in player positions in occluded regions using a conditional VAE; (3) **Virtual simulation and evaluation** — feeding the reconstructed initial states into the GRF virtual football platform, training multi-agent systems to learn and reproduce subsequent tactical behaviors, and evaluating against ground-truth trajectories.

### Key Designs

1. **Trajectory Reconstruction from Broadcast Footage to Standardized Coordinates**:

    - **Function**: Maps player positions in broadcast video to a standardized bird's-eye-view pitch coordinate system.
    - **Mechanism**: YOLOv11 is used to detect players and the ball; DeepSORT maintains temporally consistent identity tracking; TVCalib estimates camera parameters from pitch line markings to compute a homography transformation, converting image coordinates to GRF standard coordinates $x \in [-1,1]$, $y \in [-0.42, 0.42]$.
    - **Design Motivation**: Leverages existing computer vision toolchains to extract data from publicly available broadcast footage, circumventing the lock-in of commercial tracking data.

2. **VAE-Based Off-Camera Trajectory Completion**:

    - **Function**: Produces continuous and physically plausible completions of trajectories for players not visible in the broadcast frame.
    - **Mechanism**: A "demonstrator–learner" architecture is adopted — the demonstrator (a bidirectional RNN) observes complete trajectories to learn spatiotemporal dynamics, while the learner receives masked partial trajectories and reconstructs missing motion via a masked decoder. The training objective is a reconstruction loss plus KL divergence regularization: $\mathcal{L} = \mathbb{E}[\|(1-M) \odot (X - \hat{X})\|_2^2] + \beta \cdot KL$.
    - **Design Motivation**: Broadcast footage contains substantial trajectory fragmentation due to occlusion and camera cuts; the VAE framework generates smooth and diverse motion sequences while effectively capturing trajectory uncertainty.

3. **Adaptive Grid Tactical Evaluation Protocol**:

    - **Function**: Quantifies tactical imitation fidelity through spatial discretization and a dual-metric system.
    - **Mechanism**: The pitch is discretized into a uniform grid, with grid size dynamically adjusted based on average displacement: $\Delta_g = \min(\Delta_{max}, \max(\Delta_{min}, \alpha/s_t))$. Two complementary metrics are computed: spatial occupancy similarity (Jaccard index) $S_t = |O^{gt} \cap O^{pred}| / |O^{gt} \cup O^{pred}|$, and motion vector similarity (cosine similarity) $S_v = (v^{gt} \cdot v^{pred} / \|v^{gt}\| \|v^{pred}\| + 1) / 2$. The final score is the arithmetic mean of the two.
    - **Design Motivation**: The adaptive grid ensures evaluation consistency across varying motion intensities; the dual metrics separately capture static positional alignment and dynamic flow consistency, each being indispensable.

### Loss & Training

The dataset contains 194,565 annotated video clips (approximately 38,913 seconds), split into 70%/15%/15% train/validation/test sets by match identity to prevent team information leakage. Training employs multi-window lengths ($L \in \{1, 10, 25, 50\}$) and includes short-horizon closed-loop rollouts to mitigate exposure bias. At test time, only the first-frame context (player and ball positions) is provided, and the model must infer the subsequent sequence.

## Key Experimental Results

### Main Results

Results under a 150-cell (15×10) grid at 3.0s prediction horizon:

| Method | Score | $S_t$ | $S_v$ |
|--------|-------|-------|-------|
| BC | 37.86 | 28.57 | 47.14 |
| CMIL | 42.98 | 40.22 | 45.73 |
| IRL | 32.53 | 28.34 | 36.72 |
| CoDAIL | **50.89** | **48.56** | **53.22** |
| DRAIL | 41.72 | 39.88 | 43.56 |

### Ablation Study

| Grid Resolution | Best Method | 3s Score | 10s Score |
|----------------|-------------|----------|-----------|
| 60 cells (10×6) | CoDAIL | 46.63 | 33.00 |
| 150 cells (15×10) | CoDAIL | 50.89 | 28.37 |
| 240 cells (20×12) | CMIL | 47.87 | 20.11 |
| 600 cells (30×20) | CoDAIL | 37.12 | 14.12 |
| 1768 cells (105×68) | CoDAIL | 27.10 | 6.45 |

### Key Findings

- **Prediction horizon is the primary driver of performance degradation**: All models exhibit substantial performance drops from 3s to 10s, indicating a fundamental challenge in transitioning from "motion state imitation" to "tactical intent inference."
- **An optimal grid resolution range exists**: Medium-resolution grids (150/240 cells) achieve the best balance between preserving tactical information and model learnability; coarser grids lose information while finer grids suffer from the curse of dimensionality.
- **CoDAIL is overall best**: Its multi-agent coordination mechanism and adversarial learning framework yield the best performance in short-to-medium-term prediction.
- **DRAIL is more robust for long-horizon prediction**: The diffusion model demonstrates a relative advantage on 10s-level spatial occupancy metrics.

## Highlights & Insights

- **Strong originality**: This is the first closed-loop football tactical imitation benchmark spanning broadcast footage to virtual simulation, filling a gap in the field.
- **Elegant evaluation design**: The adaptive grid combined with the dual-metric system provides the first systematic quantification of team-level tactical imitation quality.
- **Clear practical value**: Applicable to real-world scenarios such as coaching tactical analysis, opponent-specific simulation, and player adaptability assessment.
- **Depth in cross-factor analysis**: The time × space cross-analysis reveals optimal configurations for different task settings (short-horizon/fine-grained vs. long-horizon/macro-level).

## Limitations & Future Work

- Only ball trajectories rather than individual player trajectories are evaluated, which sidesteps identity ambiguity but also limits evaluation granularity.
- Data coverage is restricted to the Premier League, limiting tactical diversity to a single league's style.
- Baseline comparisons with Transformer-based architectures and large-scale pretrained models are absent.
- The sim-to-real gap between the virtual environment (GRF) and real-world physics is not sufficiently discussed.

## Related Work & Insights

- The SoccerNet series provides video understanding and object tracking annotations but lacks tactical-level spatiotemporal data.
- Google Research Football provides a fully observable simulation environment but exhibits a gap from real matches.
- The evaluation protocol design in this paper can inspire the construction of tactical analysis benchmarks for other team sports (basketball, ice hockey).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first football tactical imitation benchmark; the topic is highly original with clear application value.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers 5 baselines and multiple grid configurations, but lacks comparisons with more advanced models.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured, with thorough descriptions of problem formulation and the evaluation protocol.
- **Value**: ⭐⭐⭐⭐ — Advances both sports analytics and provides a novel testbed for multi-agent imitation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](../../ICLR2026/llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[CVPR 2026\] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)
- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](../../ICLR2026/llm_evaluation/can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)
- [\[AAAI 2026\] BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction](../../AAAI2026/llm_evaluation/bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model
description: >-
  [AAAI2026][Autonomous Driving] This paper proposes M-Diffusion Planner, a strategy-level motion planning framework built upon a multi-head diffusion model and GRPO post-training, enabling users to switch among driving styles such as aggressive, conservative, and comfortable via natural language, while maintaining state-of-the-art planning performance.
tags:
  - AAAI2026
  - Autonomous Driving
  - diffusion model
  - GRPO
  - Motion Planning
  - Driving Preferences
date: 2026-05-08
content_hash: c8e00849c4c9cc58
---

# Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model

**Conference**: AAAI2026
**arXiv**: [2508.16947](https://arxiv.org/abs/2508.16947)
**Code**: To be confirmed
**Area**: Autonomous Driving
**Keywords**: autonomous driving, diffusion model, GRPO, Motion Planning, Driving Preferences

## TL;DR

This paper proposes M-Diffusion Planner, a strategy-level motion planning framework built upon a multi-head diffusion model and GRPO post-training, enabling users to switch among driving styles such as aggressive, conservative, and comfortable via natural language, while maintaining state-of-the-art planning performance.

## Background & Motivation

- Existing learning-based planners produce smooth but uniform trajectories after supervised training, failing to capture individual driving preferences.
- Some controllable planning methods operate at the action level, requiring step-by-step user instructions, which contradicts the goal of fully autonomous driving.
- Conventional behavior cloning and supervised learning struggle to model the multimodal distribution of human driving behavior.
- While diffusion models possess strong generative diversity, naive post-training tends to degrade planning capability.

## Core Problem

How can a planner support multiple driving strategies (aggressive / conservative / comfortable) with real-time strategy switching via natural language, while preserving high-quality trajectory planning performance?

## Method

### Overall Architecture

M-Diffusion Planner consists of three core components:

1. **Encoder**: MLP-Mixer + Transformer
   - MLP-Mixer alternately mixes heterogeneous inputs (lane boundaries, navigation routes, dynamic agents, static obstacles) along token and channel dimensions to produce compact fixed-length embeddings.
   - Transformer models spatiotemporal dependencies among traffic participants via self-attention.

2. **Multi-Head Diffusion Decoder**: based on the DiT (Diffusion Transformer) architecture
   - Multiple output heads correspond to different driving strategies (base / aggressive / conservative / comfortable).
   - Trajectories are generated conditioned on scene encodings and high-level strategy identifiers.

3. **LLM Semantic Interpreter**: serves as the bridge between the user and the planner
   - Parses natural language instructions (e.g., "please drive faster," "stay safe") into structured strategy identifiers.
   - The selected strategy remains active throughout execution unless explicitly changed by the user.

### Training

**Base Model Training**:

- Adopts a score-based generative framework (VP-SDE formulation), training the model to predict noise added to ground-truth trajectories.
- Loss function: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{neighbor}} + \alpha \cdot \mathcal{L}_{\text{ego}}$
- Key design: all decoder heads share parameters during training, ensuring each head receives sufficient supervision.

**GRPO Post-Training**:

- The encoder and all other output heads are frozen; only the output layer parameters of the target strategy head are fine-tuned.
- $S$ trajectories are sampled → evaluated by a reward function → normalized into relative advantages $A_i = \frac{r_i - \mu}{\sigma + \epsilon}$
- Total loss = policy gradient term + KL divergence regularization: $\mathcal{L} = \sum_{i=1}^{S} A_i \cdot \log \pi_i + \beta \cdot \log \sigma$
- KL divergence constraint prevents over-aggressive updates from degrading planning capability.
- Only 10,000 trajectories from nuPlan-mini and 30 training epochs are required.

### Inference

- DPM-Solver++ second-order multistep method is used for deterministic reverse ODE solving.
- Hard constraints are applied to the current state and remain fixed throughout sampling.
- The pipeline is: natural language → LLM parsing → strategy ID → corresponding head selection → trajectory generation.
- Real-time strategy switching is supported without retraining or reloading the model.

## Key Experimental Results

### Closed-Loop Evaluation (nuPlan Val14)

| Planner | Non-Reactive (NR) | Reactive (R) |
|---|---|---|
| PDM-Closed | 92.84 | 92.12 |
| PLUTO | 92.88 | 76.88 |
| Diffusion Planner | 89.87 | 82.80 |
| **M-Diffusion (base, Ours)** | **93.43** | **85.65** |
| M-Diffusion (Conservative) | 85.51 | 78.69 |
| M-Diffusion (Aggressive) | 82.63 | 75.11 |
| M-Diffusion (Comfortable) | 88.72 | 79.80 |

- The base model achieves state-of-the-art performance under both non-reactive and reactive settings.
- Strategy heads after GRPO fine-tuning exhibit clear behavioral differentiation with acceptable performance degradation.

### Open-Loop Evaluation (2,000 Scenarios)

| Strategy | Speed (m/s) | Acceleration (m/s²) | Jerk (m/s³) | High-Speed Ratio |
|---|---|---|---|---|
| Base | 10.59 | 1.97 | 2.66 | 16.54% |
| Aggressive | 12.50 | 2.31 | 2.43 | 26.56% |
| Conservative | 9.57 | 1.80 | 2.58 | 7.68% |
| Comfortable | 11.03 | 1.72 | **1.85** | 17.7% |

- The Aggressive strategy achieves the highest speed, with high-speed segments accounting for 26.56%.
- The Conservative strategy yields the lowest speed, with high-speed segments at only 7.68%.
- The Comfortable strategy achieves the lowest jerk (1.85), consistent with the comfort-oriented driving objective.

## Highlights & Insights

1. **Strategy-Level Interaction**: This work is the first to propose a motion planning framework that injects human intent at the strategy level; once set, the strategy remains active without requiring step-by-step intervention.
2. **Efficient Post-Training**: GRPO requires only 10,000 trajectories and 30 epochs to learn differentiated behaviors across strategy heads, with minimal computational overhead.
3. **Shared-Parameter Design**: Multi-head weight sharing during base training ensures all strategy heads are grounded in high-quality planning capability.
4. **Zero-Shot Strategy Switching**: At inference time, the LLM instantly translates natural language into strategy IDs, enabling real-time switching without model reloading.

## Limitations & Future Work

- The strategy space is limited (only 3 styles + base); finer-grained preferences (e.g., "slightly faster") cannot be precisely modeled.
- In closed-loop evaluation, strategy heads—especially Aggressive—show notable performance drops compared to the base head (82.63 vs. 93.43), raising safety concerns.
- Open-loop evaluation does not report safety metrics such as collision rate or lane departure rate.
- The reliability and latency of the LLM interpreter are not evaluated, which may become a bottleneck in real-world deployment.
- Validation is limited to the nuPlan dataset, with no cross-dataset generalization or real-vehicle experiments.

## Related Work & Insights

| Method | Interaction Mode | Diversity | Performance Retention |
|---|---|---|---|
| Diffusion Planner | No interaction | Limited | Good |
| SceneControl | Action-level instructions | Moderate | Moderate |
| PLUTO (contrastive learning) | No interaction | Limited | Good |
| **M-Diffusion (Ours)** | **Strategy-level natural language** | **High** | **Good** |

- Compared to action-level controllable methods (e.g., SceneControl), this work's strategy-level interaction better reflects real-world usage scenarios.
- Compared to Diffusion Planner, GRPO post-training significantly enhances trajectory diversity.
- Compared to PLUTO's contrastive learning approach, this work achieves preference alignment via RL-based post-training.

The successful application of GRPO to diffusion model post-training suggests that alignment techniques from the LLM domain can transfer to continuous decision-making tasks. The paradigm of shared-parameter base training followed by single-head fine-tuning is generalizable to other generative tasks requiring multimodal outputs. The strategy-level interaction concept can be extended along additional dimensions, such as lane-change style, following distance preference, and intersection-crossing behavior.

## Rating
- Novelty: 7/10 (the combination of strategy-level interaction and GRPO fine-tuning of a diffusion model is relatively novel)
- Experimental Thoroughness: 6/10 (closed-loop SOTA results are convincing, but safety metrics and real-vehicle validation are absent)
- Writing Quality: 7/10 (well-structured with complete formulations, though some descriptions are redundant)
- Value: 7/10 (provides a practical paradigm for controllable driving planning, but strategy granularity is limited)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](../../ICLR2026/autonomous_driving/multi-head_low-rank_attention.md)
- [\[AAAI 2026\] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)
- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](../../ICCV2025/autonomous_driving/langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[CVPR 2026\] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](../../CVPR2026/autonomous_driving/drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)

</div>

<!-- RELATED:END -->

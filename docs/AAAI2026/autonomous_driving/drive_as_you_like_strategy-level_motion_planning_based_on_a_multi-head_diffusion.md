---
title: >-
  [Paper Note] Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model
description: >-
  [AAAI2026][Autonomous Driving][diffusion model] Proposes M-Diffusion Planner, which achieves strategy-level motion planning based on a multi-head diffusion model and GRPO post-training, allowing users to switch between aggressive, conservative, and comfortable driving styles via natural language while maintaining SOTA planning performance.
tags:
  - "AAAI2026"
  - "Autonomous Driving"
  - "diffusion model"
  - "GRPO"
  - "Motion Planning"
  - "Driving Preferences"
date: 2026-05-08
content_hash: f334d19c25d2812e
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model

**Conference**: AAAI2026  
**arXiv**: [2508.16947](https://arxiv.org/abs/2508.16947)  
**Code**: To be confirmed  
**Area**: Autonomous Driving  
**Keywords**: autonomous driving, diffusion model, GRPO, Motion Planning, Driving Preferences

## TL;DR

Proposes M-Diffusion Planner, which achieves strategy-level motion planning based on a multi-head diffusion model and GRPO post-training, allowing users to switch between aggressive, conservative, and comfortable driving styles via natural language while maintaining SOTA planning performance.

## Background & Motivation

- Existing learning-based planners have fixed strategies after supervised training, producing smooth but highly uniform trajectories that fail to reflect individual driving preferences.
- Some controllable planning methods interact at the action level, requiring step-by-step user commands, which contradicts the original goal of "freeing the driver" in autonomous driving.
- Traditional behavior cloning and supervised learning struggle to model the multimodal distribution of human driving behaviors.
- Diffusion models possess powerful diverse generation capabilities, but direct post-training easily leads to degradation in planning performance.

## Core Problem

How to enable the planner to support multiple driving strategies (aggressive/conservative/comfortable) and achieve real-time strategy switching via natural language, while maintaining high-quality trajectory planning capabilities?

## Method

### Overall Architecture

M-Diffusion Planner comprises three core components:

1. **Encoder**: MLP-Mixer + Transformer
    - The MLP-Mixer alternately mixes heterogeneous inputs (such as lane lines, navigation routes, dynamic objects, and static obstacles) across token and channel dimensions to generate compact, fixed-length embeddings.
    - The Transformer models spatiotemporal dependencies among traffic participants using self-attention.

2. **Multi-Head Diffusion Decoder**: Based on the DiT (Diffusion Transformer) architecture
    - Multiple output heads correspond to different driving strategies (base/aggressive/conservative/comfortable).
    - Generates trajectories conditioned on scene encoding and high-level strategy identifiers.

3. **LLM Semantic Interpreter**: Acts as a bridge between the user and the planner
    - Parses natural language commands (e.g., "please drive faster", "pay attention to safety") into structured strategy identifiers.
    - The strategy remains active during execution unless explicitly modified by the user.

### Training Phase

**Base Model Training**:

- Adopts a score-based generative framework (VP-SDE formulation), adding Gaussian noise to ground-truth trajectories and training the model to predict the noise.
- Loss function: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{neighbor}} + \alpha \cdot \mathcal{L}_{\text{ego}}$
- Key design: All decoder heads share parameters during the training phase, avoiding separate training and ensuring that each head is fully learned.

**GRPO Post-Training**:

- Freezes the encoder and other output heads, only fine-tuning the output layer parameters of the target strategy head.
- Samples $S$ trajectories $\rightarrow$ evaluated by reward functions $\rightarrow$ normalized into relative advantage $A_i = \frac{r_i - \mu}{\sigma + \epsilon}$.
- Total loss = Policy gradient term + KL divergence regularization term: $\mathcal{L} = \sum_{i=1}^{S} A_i \cdot \log \pi_i + \beta \cdot \log \sigma$.
- The KL divergence constraint prevents updates from being too aggressive, which would otherwise degrade planning capability.
- Only requires 10,000 trajectories from nuPlan-mini and 30 training epochs to complete.

### Inference Phase

- Uses the DPM-Solver++ second-order multi-step method to deterministically solve the reverse ODE.
- Imposes hard constraints on the current state, keeping them unchanged during the sampling process.
- User natural language $\rightarrow$ LLM parsing $\rightarrow$ strategy ID $\rightarrow$ select corresponding head $\rightarrow$ generate trajectory.
- Supports real-time strategy switching without retraining or reloading the model.

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

- The base model achieves SOTA in both non-reactive and reactive settings.
- The strategy heads fine-tuned via GRPO exhibit distinct behavioral differences within a controllable range of performance degradation.

### Open-Loop Evaluation (2000 Scenes)

| Strategy | Speed (m/s) | Acceleration (m/s²) | Jerk (m/s³) | High-Speed Ratio |
|---|---|---|---|---|
| Base | 10.59 | 1.97 | 2.66 | 16.54% |
| Aggressive | 12.50 | 2.31 | 2.43 | 26.56% |
| Conservative | 9.57 | 1.80 | 2.58 | 7.68% |
| Comfortable | 11.03 | 1.72 | **1.85** | 17.7% |

- The Aggressive strategy has the highest speed, with the high-speed segment accounting for 26.56%.
- The Conservative strategy has the lowest speed, with the high-speed segment accounting for only 7.68%.
- The Comfortable strategy has the lowest jerk (1.85), aligning with comfortable driving goals.

## Highlights & Insights

1. **Strategy-Level Interaction**: Proposes for the first time a motion planning framework that injects human intent at the strategy level, where strategy settings remain active continuously without step-by-step intervention.
2. **Efficient Post-Training**: GRPO requires only 10,000 trajectories and 30 epochs to learn differentiated behaviors for each strategy head, with minimal training overhead.
3. **Shared Parameter Design**: Shared weights among multiple heads during the base training phase ensure that all strategy heads possess a high-quality planning foundation.
4. **Zero-Shot Strategy Switching**: Translates natural language instantly into strategy IDs via LLM during inference, achieving real-time switching without reloading.

## Limitations & Future Work

- Limited variety of strategies (only 3 types + base); more fine-grained preferences (e.g., "slightly faster") cannot be modeled precisely.
- In closed-loop evaluation, the performance of strategy heads (especially Aggressive) declines significantly compared to base (82.63 vs 93.43), raising safety concerns.
- Safety metrics such as collision rates and lane deviations are not reported in open-loop evaluation.
- The reliability and latency of the LLM interpreter are not evaluated, which may become a bottleneck in real deployment scenarios.
- Validated only on the nuPlan dataset, lacking cross-dataset or real-vehicle experiments.

## Related Work & Insights

| Method | Interaction Method | Diversity | Performance Preservation |
|---|---|---|---|
| Diffusion Planner | No Interaction | Limited | Good |
| SceneControl | Action-Level Commands | Medium | Medium |
| PLUTO (Contrastive Learning) | No Interaction | Limited | Good |
| **M-Diffusion (Ours)** | **Strategy-Level Natural Language** | **High** | **Good** |

- Compared to action-level controllable methods (such as SceneControl), the strategy-level interaction of this work is more aligned with real-world scenarios.
- Compared to Diffusion Planner, trajectory diversity is significantly enhanced through GRPO post-training.
- Compared to the contrastive learning of PLUTO, this work achieves preference alignment via RL post-training.

## Inspirations & Connections

- The successful application of GRPO in diffusion model post-training demonstrates that alignment techniques in the LLM field can be transferred to continuous decision-making tasks.
- The paradigm of multi-head shared training + single-head fine-tuning can be generalized to other generative tasks requiring multimodal outputs.
- The concept of strategy-level interaction can be extended to broader dimensions: lane-changing style, car-following distance preference, intersection passing strategy, etc.

## Rating
- Novelty: 7/10 (The combination of strategy-level interaction and GRPO fine-tuning for diffusion models is relatively novel)
- Experimental Thoroughness: 6/10 (Closed-loop SOTA is convincing, but lacks safety metrics and real-vehicle validation)
- Writing Quality: 7/10 (Clear structure and complete formulations, but some expressions are redundant)
- Value: 7/10 (Provides a practical paradigm for controllable driving planning, but the strategy granularity is limited)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Think Before You Drive: World Model-Inspired Multimodal Grounding](../../CVPR2026/autonomous_driving/think_before_you_drive_world_model-inspired_multimodal_grounding.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](../../ICLR2026/autonomous_driving/multi-head_low-rank_attention.md)
- [\[AAAI 2026\] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)
- [\[CVPR 2026\] WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving](../../CVPR2026/autonomous_driving/wam-flow_parallel_coarse-to-fine_motion_planning_via_discrete_flow_matching_for_.md)

</div>

<!-- RELATED:END -->

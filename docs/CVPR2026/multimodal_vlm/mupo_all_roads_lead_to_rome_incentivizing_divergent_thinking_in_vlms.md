---
title: >-
  [Paper Note] MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Reinforcement Learning] MUPO identifies a reasoning diversity collapse in GRPO training — models prematurely converge to a small number of reasoning strategies while discarding most alternatives. By partitioning responses into groups for localized advantage estimation and introducing a diversity reward, MUPO incentivizes VLMs to maintain divergent thinking, achieving 2–7% improvements across multiple reasoning benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Reinforcement Learning
  - GRPO
  - Divergent Thinking
  - Reasoning Diversity
  - Vision-Language Models
date: 2026-05-08
content_hash: e11774cc0a4e2566
---

# MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models

**Conference**: CVPR 2026  
**arXiv**: [2604.00479](https://arxiv.org/abs/2604.00479)  
**Code**: [https://xytian1008.github.io/MUPO/](https://xytian1008.github.io/MUPO/)  
**Area**: LLM Reasoning / Multimodal VLM  
**Keywords**: Reinforcement Learning, GRPO, Divergent Thinking, Reasoning Diversity, Vision-Language Models

## TL;DR

MUPO identifies a reasoning diversity collapse in GRPO training — models prematurely converge to a small number of reasoning strategies while discarding most alternatives. By partitioning responses into groups for localized advantage estimation and introducing a diversity reward, MUPO incentivizes VLMs to maintain divergent thinking, achieving 2–7% improvements across multiple reasoning benchmarks.

## Background & Motivation

RL (particularly GRPO) has become the dominant approach for enhancing VLM reasoning capabilities. However, the authors identify a critical contradiction:

**RL models are deep but narrow; Base models are shallow but broad**: RL models achieve higher accuracy on individual attempts (deeper reasoning), yet when given multiple attempts, Base models can solve a greater variety of problems (more diverse strategies). For example, on geometry problems, RL models consistently resort to equation solving (prone to logical errors), whereas Base models sometimes adopt a verification-based strategy to arrive at answers more concisely.

**Diversity Collapse**: By tracking the GRPO training process, reasoning diversity is found to drop sharply to negligible levels very early in training. The model rapidly converges to a small number of "dominant" strategies, discarding a large number of potentially viable alternative paths. This leads to: (1) exploitation prioritized over exploration, resulting in local optima; (2) poor scalability, as converged reasoning fails to cover a broad range of problem types.

## Method

### Overall Architecture

MUPO serves as a plug-and-play replacement for GRPO. Multiple responses from the model are divided into several groups; advantage estimation is performed locally within each group, while a diversity reward is introduced across groups to encourage different groups to represent distinct reasoning strategies.

### Key Designs

1. **Multi-Group Policy Optimization**:

    - Function: Maintains diversity of reasoning strategies and prevents all responses from converging to the same strategy.
    - Mechanism: Replaces GRPO's global advantage computation with localized, group-wise advantage estimation. The $K$ responses are divided into $G$ groups, each computing advantage values independently. This allows each group to maintain its own "optimal strategy" without being overwhelmed by the globally dominant strategy. Intuitively, each group constitutes an independent instantiation of a reasoning strategy.
    - Design Motivation: GRPO's global advantage computation causes a small number of high-reward strategies to obtain disproportionately large advantage values, suppressing the update signals for other strategies.

2. **Diversity Reward**:

    - Function: Promotes separation of reasoning strategies across groups.
    - Mechanism: In addition to accuracy and format rewards, a diversity reward is introduced — measuring the embedding distance of reasoning across different groups. Groups are incentivized to maximize inter-group distance, ensuring that distinct groups represent genuinely different reasoning paths.
    - Design Motivation: Grouping alone, without encouraging differentiation, may still lead groups to converge to similar strategies. The diversity reward provides an explicit incentive for separation.

3. **Unification of Depth and Breadth**:

    - Function: Enables the model to simultaneously achieve deep single-path reasoning and broad multi-path coverage.
    - Mechanism: Intra-group optimization ensures each strategy is thoroughly refined (depth), while inter-group diversity ensures multiple strategies are maintained (breadth). This mirrors human problem-solving — when given multiple attempts, one approaches the problem from different angles, reasoning carefully within each angle.
    - Design Motivation: This is the essence of divergent thinking — not merely generating different answers, but thinking about the same problem through different methods.

### Loss & Training

Standard RL training pipeline, with MUPO replacing GRPO as the policy optimization algorithm. Rewards consist of accuracy reward + format reward + diversity reward.

## Key Experimental Results

### Main Results

| Model | MathVerse | LogicVista | WeMath | HallusionBench | Avg. Gain |
|-------|-----------|------------|--------|----------------|-----------|
| GRPO Baseline | baseline | baseline | baseline | baseline | — |
| **MUPO-Thinker-7B** | +gain | +gain | +gain | +gain | **2–7%** |

Consistent improvements of 2–7% across multiple reasoning benchmarks, establishing a new state of the art.

### Ablation Study

| Configuration | acc@1 | acc@4 | Diversity | Notes |
|---------------|-------|-------|-----------|-------|
| GRPO | High | Limited gain | Low (collapsed) | Deep but narrow |
| Base Model | Lower | Large gain | High | Shallow but broad |
| MUPO | **Highest** | **Highest** | **High** | Deep and broad |

### Key Findings

- acc@k analysis reveals a fundamental difference between RL and Base models: RL wins at k=1, Base wins at k>1. This demonstrates that diversity itself is a capability.
- Diversity collapse in GRPO occurs extremely early in training (<10% of training steps), indicating this is an algorithmic issue rather than insufficient training.
- Diversity and accuracy are positively correlated — more diverse reasoning strategies increase the probability of arriving at the correct answer.

## Highlights & Insights

- **Divergent vs. Convergent Thinking**: Introducing the psychological concepts of divergent/convergent thinking into RL training provides a novel perspective for understanding the limitations of GRPO.
- **Diagnosing Diversity Collapse**: Quantifying reasoning diversity via embedding distance and tracking training dynamics constitutes a reusable analytical methodology.
- **acc@k as a Complementary Metric**: Evaluating not only single-attempt accuracy but also the proportion of problems solvable across multiple attempts offers a more comprehensive assessment of reasoning models.
- **Plug-and-Play Replacement for GRPO**: MUPO can directly replace GRPO without modifying any other part of the training pipeline.

## Limitations & Future Work

- The number of groups $G$ is a hyperparameter whose optimal value may vary across tasks.
- The weight of the diversity reward requires tuning; an excessively large value may sacrifice single-path accuracy.
- Current validation is primarily on mathematical and logical reasoning tasks; effectiveness on other tasks (e.g., open-ended generation) remains to be verified.
- Future work may explore adaptive grouping and dynamic diversity reward weighting.

## Related Work & Insights

- **vs. GRPO/DeepSeekMath**: GRPO pursues depth of reasoning at the cost of breadth; MUPO preserves both simultaneously.
- **vs. DAPO/GVPO**: These methods optimize GRPO from a sampling perspective but do not address the diversity collapse problem.
- **vs. Best-of-N/Self-Consistency**: These are inference-time scaling strategies, whereas MUPO operates at training time; the two approaches are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The diagnosis of GRPO diversity collapse and the introduction of divergent thinking are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Behavioral analysis, training dynamics, and multi-benchmark validation provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ — In-depth analysis, clear illustrations, and coherent argumentation.
- Value: ⭐⭐⭐⭐⭐ — Makes an important methodological contribution to RL-based reasoning model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)

</div>

<!-- RELATED:END -->

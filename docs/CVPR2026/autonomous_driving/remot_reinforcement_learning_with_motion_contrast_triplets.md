---
title: >-
  [Paper Note] ReMoT: Reinforcement Learning with Motion Contrast Triplets
description: >-
  [CVPR 2026][Autonomous Driving][Motion Contrast Triplets] This paper proposes ReMoT, a unified training paradigm that automatically constructs a 16.5K motion contrast triplet dataset (ReMoT-16K) via a rule-driven multi-e…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "Motion Contrast Triplets"
  - "Spatiotemporal Reasoning"
  - "GRPO"
  - "VLM"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 79ae269f85e1a3a1
---

# ReMoT: Reinforcement Learning with Motion Contrast Triplets

**Conference**: CVPR 2026
**arXiv**: [2603.00461](https://arxiv.org/abs/2603.00461)  
**Code**: None  
**Area**: Autonomous Driving / Vision-Language Models
**Keywords**: Motion Contrast Triplets, Spatiotemporal Reasoning, GRPO, VLM, Reinforcement Learning

## TL;DR

This paper proposes ReMoT, a unified training paradigm that automatically constructs a 16.5K motion contrast triplet dataset (ReMoT-16K) via a rule-driven multi-expert collaborative pipeline, and combines GRPO reinforcement learning with a composite reward (logical consistency + length regularization) to systematically address the fundamental deficiencies of VLMs in spatiotemporal consistency reasoning, achieving a 25.1% performance improvement.

## Background & Motivation

**Background**: Vision-language models (VLMs) such as GPT-4o, Claude, Gemini, and Qwen have evolved into general-purpose perception systems, demonstrating strong performance in static image understanding and semantic alignment, and have been deployed in critical domains including AIGC, embodied intelligence, and autonomous driving.

**Limitations of Prior Work**: (1) Current mainstream VLMs exhibit fundamental deficiencies in spatiotemporal consistency reasoning—frequently confusing camera rotation with object motion, misjudging gripper states, and incorrectly inferring motion direction; (2) existing training data predominantly consists of static image-text pairs, lacking explicit modeling of fine-grained motion attributes; (3) approaches such as architectural modifications or data augmentation provide only sporadic patches and cannot systematically address the problem.

**Key Challenge**: VLMs excel at visual-semantic alignment but lack deep understanding of spatial-physical regularities, while existing methods address data, training, and evaluation independently without a unified framework.

**Goal**: To systematically address VLM spatiotemporal reasoning deficiencies across three dimensions: data construction, training optimization, and evaluation benchmarks.

**Key Insight**: (1) Automatically constructing motion contrast triplets from video meta-annotations (camera pose matrices, robot action logs); (2) replacing SFT with GRPO for policy learning optimization; (3) designing a composite reward incorporating logical consistency verification.

**Core Idea**: Formalizing motion understanding as structured learning over contrast triplets, and achieving systematic improvement in VLM spatiotemporal reasoning through rule-driven data construction and GRPO optimization.

## Method

### Overall Architecture

ReMoT comprises three core components: (1) **ReMoT-16K Data Construction**: a multi-expert collaborative pipeline that automatically generates 16.5K motion contrast triplets from video meta-annotations; (2) **Training Optimization**: systematic exploration of SFT, GRPO, and hybrid strategies (sequential SFT→GRPO, alternating SFT↔GRPO); (3) **Evaluation Benchmark**: construction of ReMoT-16K-Test, containing 600 evaluation triplets and 1,776 questions covering navigation, robotic manipulation, and simulation game scenarios.

### Key Designs

1. **Multi-Expert Collaborative Data Construction Pipeline**:
    - Function: Automatically generates large-scale, high-quality motion contrast triplets $(I_{anchor}, I_{pos}, I_{neg})$ from video meta-annotations.
    - Motion Estimation Expert: Domain-specific extractors that compute camera rotation from $SE(3)$ pose matrices, extract end-effector trajectories from robot telemetry, etc., and output composite motion attributes $m$.
    - Triplet Construction Expert: (a) Positive sample selection—filters perceptually salient and coherent transitions via attribute thresholds $\mathcal{T}_m$ (e.g., camera rotation in $[10°, 50°]$); (b) Negative sample generation—synthesizes reversed motion via attribute-conditioned generation $\mathcal{T}_{geo}$, or retrieves visually similar but attribute-conflicting frames via retrieval $\mathcal{R}$.
    - VQA Formulation Expert: Designs multi-perspective reasoning chain questions for each triplet, including multiple-choice, true/false, fill-in-the-blank, and comparative reasoning formats.
    - Design Motivation: Directly using VLMs to generate data results in 55% format errors at high cost, yielding only 632 valid triplets, whereas the multi-expert pipeline produces 16.5K high-quality triplets.

2. **GRPO with Composite Reward Design**:
    - Function: Optimizes VLM motion reasoning capability via reinforcement learning, replacing SFT which offers limited effectiveness.
    - Core Algorithm: Adopts GRPO (Group Relative Policy Optimization), sampling $G$ responses for a given query $q$ and computing group-normalized advantages $\hat{A}_i = \frac{R_i - \bar{R}}{\sigma(\{R_j\})}$.
    - CoT Length Regularization: $R_{length}(o_i) = -\max(0, |o_i^{think}| - L_{target})$, suppressing excessively long reasoning chains.
    - Logical Consistency Reward: Detects logical contradictions in responses (e.g., transitivity violations $L_1 < L_2, L_2 < L_3, L_3 < L_1$), assigning $+1/-1/0$ rewards.
    - Composite Reward: $R_i = R_{task} + \lambda_1 \cdot R_{logic} + \lambda_2 \cdot R_{length}$, with weight ratio 3.5:3.5:1.3:1.7.
    - Design Motivation: Analysis reveals that 31.4% of errors stem from logical inconsistency; the explicit logical reward improves logical correctness from 46.6% to 99.3%.

3. **Hybrid Optimization Strategy**:
    - Function: Explores the optimal combination of SFT and GRPO.
    - Sequential Hybrid (SFT→GRPO): SFT provides a stable initialization before switching to GRPO fine-tuning.
    - Alternating Hybrid (SFT↔GRPO): SFT and GRPO steps alternate every few updates, controlled by $(t \bmod (K_{SFT}+K_{GRPO})) < K_{SFT}$.
    - Design Motivation: The alternating strategy allows language alignment and reward alignment to co-evolve, avoiding pattern forgetting.

### Loss & Training

- SFT phase: Cross-entropy loss computed only on tokens within `<answer>` tags.
- GRPO phase: Standard PPO objective with KL regularization (coefficient 0.01), batch size 16, 4 rollouts/sample.
- Base model: Qwen3-VL-4B-Thinking, retaining its built-in CoT reasoning capability.
- Training configuration: 8×A800 GPUs, mixed precision, 2 epochs.

## Key Experimental Results

### Main Results (ReMoT-16K-Test Benchmark)

| Model | Overall Acc. | Partial Acc. | Navigation (Ov.) | Manipulation (Ov.) | Composite Manipulation (Ov.) |
|-------|-------------|-------------|-----------------|-------------------|------------------------------|
| Qwen2.5-VL-7B | 5.1 | 25.4 | 4.8 | 4.0 | 0.0 |
| Qwen3-VL-CoT-4B (Baseline) | 20.7 | 38.9 | 2.4 | 15.3 | 4.8 |
| InternVL3-8B | 12.2 | 28.9 | 2.8 | 1.6 | 0.0 |
| GRPO | 33.6 | 61.6 | 27.0 | 54.5 | 61.3 |
| SFT→GRPO | 35.0 | 63.3 | 26.6 | 57.3 | 62.9 |
| **SFT↔GRPO (Ours)** | **38.0** | **64.0** | 21.4 | **68.6** | **69.4** |

### Ablation Study (Training Strategy and Data Composition)

| Configuration | Overall Acc. | Partial Acc. |
|--------------|-------------|-------------|
| No training (baseline) | 20.7 | 38.9 |
| Manipulation data only | 23.9 | 46.7 |
| + Navigation data | 32.4 | 57.6 |
| + Simulation data (full) | **38.0** | **64.0** |

| Logical Reward Ablation | Overall | Partial | Logical Correctness |
|------------------------|---------|---------|---------------------|
| Base model | 16.2 | 39.6 | 46.6% |
| GRPO w/o logical reward | 68.6 | 77.3 | 98.6% |
| GRPO w/ logical reward | **78.0** | **81.3** | **99.3%** |

### Key Findings

- The alternating SFT↔GRPO strategy achieves the best overall performance (38.0% Overall), representing a 25.1% relative improvement over the base model.
- ReMoT with 4B parameters outperforms Qwen3-VL-30B-CoT (7.5× larger) on spatiotemporal benchmarks (VLM2: 70.0 vs. 68.2, VSI: 58.8 vs. 56.1).
- The multi-expert pipeline data exhibits smooth scaling behavior, whereas VLM-generated data shows instability and a lower performance ceiling (~0.49 vs. 0.66).
- Performance on general multimodal benchmarks is maintained or improved, demonstrating that enhanced spatiotemporal reasoning does not cause catastrophic forgetting.

## Highlights & Insights

- **Systematic approach**: This is the first work to address VLM spatiotemporal reasoning deficiencies from a unified data/training/evaluation perspective, rather than through isolated patches.
- **Efficient data construction**: The rule-driven pipeline outperforms VLM-generated data by two orders of magnitude in scale (16.5K vs. 632) while achieving higher quality.
- **Logical consistency reward**: The paper identifies and resolves the critical finding that 31.4% of errors originate from logical contradictions; the logical reward improves accuracy by 10.6%.
- **Small model, large capability**: The 4B model surpasses the 30B model and GPT-4o through precise data curation and RL training, validating that data quality and training paradigm matter more than model scale.

## Limitations & Future Work

- Navigation task performance degrades under alternating training (Overall 21.4 vs. 27.0 for GRPO), suggesting potential optimization conflicts across tasks.
- Data construction relies on structured meta-annotations (pose matrices, etc.) and is not applicable to videos lacking such annotations.
- Validation is limited to the 4B model; whether there is a performance ceiling for larger models (7B+) remains unexplored.
- The evaluation benchmark is relatively small in scale (600 triplets), and scenario diversity can be further expanded.

## Related Work & Insights

- **GRPO** (Shao et al.): ReMoT validates the superiority of GRPO over SFT for visual reasoning tasks and introduces logical consistency reward as a novel contribution.
- **SimCLR / Contrastive Learning**: The design of motion contrast triplets draws inspiration from the core principles of contrastive learning.
- **Qwen3-VL**: As one of the strongest open-source VLMs, its Thinking mode provides high-quality initialization for RL training.
- **Insights**: The paradigm of rule-driven data construction combined with RL optimization is generalizable to repairing other capability gaps in VLMs.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion](../../ICML2026/autonomous_driving/constrained_multi-objective_reinforcement_learning_with_max-min_criterion.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting](sharp_short-window_streaming_for_accurate_and_robust_prediction_in_motion_foreca.md)
- [\[AAAI 2026\] SAML: A Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Prediction](../../AAAI2026/autonomous_driving/differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)
- [\[CVPR 2026\] FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)

</div>

<!-- RELATED:END -->

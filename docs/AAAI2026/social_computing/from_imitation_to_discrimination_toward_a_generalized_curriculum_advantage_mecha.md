---
title: >-
  [Paper Note] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks
description: >-
  [AAAI 2026][Social Computing][Reinforcement Learning] This paper proposes CAPO (Curriculum Advantage Policy Optimization), an adaptive curriculum mechanism based on advantage signals. Through a two-phase strategy — first imitation (using only positive-advantage samples) then discrimination (introducing negative signals) — CAPO stably and significantly improves LLM performance on mathematical reasoning and multimodal GUI reasoning tasks.
tags:
  - AAAI 2026
  - Social Computing
  - Reinforcement Learning
  - Curriculum Learning
  - Advantage Function
  - LLM Reasoning
  - Multimodal Reasoning
date: 2026-05-08
content_hash: aa3b5d947b5fafe2
---

# From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks

**Conference**: AAAI 2026
**arXiv**: [2512.02580](https://arxiv.org/abs/2512.02580)
**Code**: None
**Area**: Social Computing
**Keywords**: Reinforcement Learning, Curriculum Learning, Advantage Function, LLM Reasoning, Multimodal Reasoning

## TL;DR

This paper proposes CAPO (Curriculum Advantage Policy Optimization), an adaptive curriculum mechanism based on advantage signals. Through a two-phase strategy — first imitation (using only positive-advantage samples) then discrimination (introducing negative signals) — CAPO stably and significantly improves LLM performance on mathematical reasoning and multimodal GUI reasoning tasks.

## Background & Motivation

Reinforcement learning (RL) has become the dominant paradigm for LLM post-training, substantially enhancing model reasoning capabilities. The core of RL algorithms such as PPO and GRPO is the **advantage function**, which quantifies whether a trajectory is better or worse than an expected baseline, thereby providing positive and negative feedback to guide policy updates.

However, existing methods **indiscriminately mix positive and negative advantage signals from the very beginning of training**, which introduces fundamental problems:

**Early mixing leads to ambiguous guidance**: Exposing the model to negative samples before a stable foundation is established results in high gradient noise and unstable learning.

**Limits further improvement**: Premature mixing of positive and negative signals impedes progressive learning from easy to hard examples.

This phenomenon parallels observations in developmental psychology: **children first learn basic behaviors through positive imitation, and only later generalize through corrective feedback and punishment**. This staged learning process naturally positions advantage signals as effective curriculum signals.

Core research question: **Can advantage signals themselves serve as curriculum guidance indicators to achieve structured integration of positive and negative feedback?**

## Method

### Overall Architecture

CAPO is a general-purpose training mechanism compatible with advantage functions, and can be seamlessly integrated into various RL algorithms including GRPO, PPO, RLOO, and Reinforce++. Its core idea is to divide training into two phases:

- **Phase 1 (Imitation Phase)**: Uses only positive-advantage samples ($A_\tau \geq 0$) to establish a stable foundation.
- **Phase 2 (Discrimination Phase)**: Introduces the full-spectrum advantage signal (positive + negative) to cultivate discriminative ability and improve generalization.

### Key Designs

#### 1. **Phase 1: Positive Imitation Phase**

The training objective selectively updates only trajectories with positive advantage:

$$\mathcal{J}_{\text{phase-1}}(\theta) = \mathbb{E}_\tau\left[\mathbb{I}_{A(\tau)\geq 0}\left(\frac{1}{T}\sum_{t=1}^{T}\min(\rho_t A_t, \hat{\rho}_t A_t) - \beta\mathbb{D}_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\right)\right]$$

The indicator function $\mathbb{I}_{A_\tau \geq 0}$ filters out negative-advantage samples, and $\beta$ controls the KL penalty strength. This phase encourages the model to reinforce correct reasoning behaviors while remaining close to the reference distribution.

**Design Motivation**: Excluding negative outliers reduces gradient variance $\text{Var}(\hat{g})$; even though bias is introduced, the overall MSE is reduced, ensuring stable improvement.

#### 2. **Phase 2: Discrimination Phase**

Once a stable foundation is established, CAPO transitions to the discrimination phase, which accepts the full-spectrum advantage signal. By incorporating negative-advantage samples, the model not only learns to reinforce high-quality reasoning trajectories but also learns to suppress suboptimal ones, thereby enhancing generalization.

At this stage, the policy variance has naturally narrowed, restoring unbiased estimation: $\mathbb{E}[\hat{g}_{\text{phase-2}}] = g$, enabling the model to achieve generalization.

#### 3. **Curriculum Scheduling Strategy**

A **hard switch point** is adopted for phase transition, e.g., switching at 10% or 20% of total training steps. Experiments show that a simple hard switch is more effective and robust than any gradual introduction scheme, requiring neither fine-grained hyperparameter tuning nor task-specific monitoring.

### Loss & Training

- Phase 1 employs the standard clipped objective of PPO/GRPO with KL regularization, but filters negative samples using the indicator function.
- Phase 2 restores the standard RL objective using all samples.
- Theoretical guarantee: Under Robbins-Monro conditions, the variance-bias trade-off in CAPO ensures asymptotic convergence to a local optimum.

### Theoretical Support

The authors provide a theoretical justification from the perspective of variance-bias trade-off:

- Phase 1 reduces gradient estimation variance by excluding negative outliers; even with bias, the total MSE is lower.
- In Phase 2, as the policy improves, the variance of advantage estimates naturally shrinks, and using all samples restores unbiasedness.
- Under reasonable step-size conditions, the MSE of both phases asymptotically vanishes and parameters converge to a stationary point.

## Key Experimental Results

### Main Results

Evaluated on Qwen2.5-Math-7B and Qwen2.5-Math-1.5B across four RL algorithms and seven mathematical reasoning benchmarks:

| Method | AIME24 | AMC | MATH500 | GSM8K | Minerva | Olympiad | Avg. |
|--------|--------|-----|---------|-------|---------|----------|------|
| **7B - GRPO** | 16.7 | 52.5 | 75.2 | 86.5 | 29.4 | 36.9 | 48.9 |
| **7B - GRPO+CAPO** | **20.0** | **65.0** | **76.8** | **88.9** | **33.1** | **39.7** | **52.8↑3.9** |
| 7B - PPO | 26.7 | 52.5 | 71.0 | 80.9 | 34.2 | 34.1 | 48.6 |
| 7B - PPO+CAPO | **30.0** | **57.5** | **72.6** | **85.2** | **37.9** | **37.8** | **51.8↑3.2** |
| 7B - RLOO | 30.0 | 55.0 | 73.8 | 82.7 | 35.5 | 36.0 | 50.4 |
| 7B - RLOO+CAPO | **33.3** | **67.5** | **74.8** | **84.6** | **36.0** | **35.6** | **53.3↑2.9** |
| **1.5B - GRPO** | 13.3 | 52.5 | 71.2 | 83.2 | 26.8 | 30.1 | 45.6 |
| **1.5B - GRPO+CAPO** | **23.3** | **62.5** | **71.8** | **83.9** | **32.0** | **32.9** | **49.6↑4.0** |

CAPO consistently delivers average improvements of **+1.7 to +4.0** across all algorithms and model scales.

### Ablation Study

| Configuration | AIME24 | AMC | MATH500 | GSM8K | Avg. | Note |
|---------------|--------|-----|---------|-------|------|------|
| GRPO (baseline) | 16.7 | 52.5 | 75.2 | 86.5 | 49.5 | Standard RL |
| GRPO + Static Curriculum | 16.7 | 65.0 | 75.0 | 86.3 | 51.8 | Difficulty-sorted data |
| GRPO + ADARFT | 15.8 | 55.0 | 74.4 | 91.0 | 47.8 | Adaptive fine-tuning |
| **GRPO + CAPO** | **20.0** | **65.0** | **76.8** | **88.9** | **53.9** | Ours |

**GUI Multimodal Reasoning (QwenVL2.5-3B)**:

| Method | GUI-Act-Web SR | OmniAct-Web SR | AndroidControl-Low SR | Overall |
|--------|---------------|----------------|----------------------|---------|
| GRPO | 70.23 | 70.76 | 63.87 | 70.79 |
| **GRPO+CAPO** | **85.85** | **74.16** | 61.41 | **74.60↑3.81** |

### Key Findings

1. **Broad compatibility**: CAPO functions as a plug-and-play enhancement, delivering stable improvements across GRPO, PPO, RLOO, and Reinforce++.
2. **Cross-modal generalization**: Successfully transfers from mathematical reasoning to multimodal GUI reasoning, achieving an overall gain of +3.81.
3. **Optimal switch point**: Switching from Phase 1 to Phase 2 at 20%–30% of training steps yields the best results.
4. **OOD robustness**: On ARC-C and GPQA-Diamond (math-only training → general reasoning), CAPO outperforms GRPO by +3.8.
5. **Training dynamics**: After Phase 2 introduces negative samples, entropy steadily increases — indicating more diverse exploration — while reward continues to improve, confirming enhanced generalization.

## Highlights & Insights

- **Redefining advantage signals as curriculum signals**: A concise yet profound insight that reframes the conventional understanding of advantage in RL.
- **Developmental psychology inspiration**: The imitation→discrimination two-phase analogy to children's cognitive development is natural and compelling.
- **Unified theory and practice**: The theoretical analysis from a variance-bias trade-off perspective provides rigorous justification for the two-phase design.
- **Extremely simple implementation**: The core modification only requires adding an indicator function to filter negative samples, followed by a hard switch at a specified step — engineering overhead is minimal.
- **Broadly validated plug-and-play compatibility** demonstrates the generality of the approach.

## Limitations & Future Work

- The switch point currently requires a preset value (10%–30%); future work may explore adaptive switching strategies based on training dynamics.
- Validation is limited to mathematical reasoning and GUI reasoning; effectiveness on other reasoning types such as code generation and natural language inference remains unknown.
- Theoretical analysis assumes advantage estimation noise is independent and bounded, which may not fully hold in practice.
- Finer-grained curriculum scheduling (e.g., gradually increasing the proportion of negative samples) remains unexplored, although experiments indicate a hard switch is optimal.
- Completely discarding negative samples during Phase 1 may forfeit some valuable feedback signals.

## Related Work & Insights

- **Relation to DeepSeek-R1**: CAPO can be viewed as an enhancement to R1-style RL training pipelines, modifying sample utilization strategy rather than the underlying algorithm.
- **Distinction from traditional curriculum learning**: Conventional methods rely on external static difficulty metrics (e.g., pass@k), whereas CAPO employs the model's intrinsic advantage signal.
- **Implications for RLHF**: In preference optimization, one could similarly consider first learning from "good" preferences before learning to discriminate between "good" and "bad."
- **Insights for multimodal reasoning**: Advantage curriculum signals transfer across modalities, offering a new perspective for unified reasoning training.

## Rating

- Novelty: ⭐⭐⭐⭐ — The insight of redefining advantage as a curriculum signal is concise and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four RL algorithms × two model scales × seven math benchmarks + multimodal GUI + OOD + ablation.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, theoretical derivations are rigorous, and experimental organization is systematic.
- Value: ⭐⭐⭐⭐⭐ — Extremely high practical value; plug-and-play and engineering-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)
- [\[ACL 2026\] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](../../ACL2026/social_computing/toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)
- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](../../ACL2026/social_computing/on_the_step_length_confounding_in_llm_reasoning_data_selection.md)

</div>

<!-- RELATED:END -->

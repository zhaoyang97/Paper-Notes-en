---
title: >-
  [Paper Note] Nudging the Boundaries of LLM Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][Reinforcement Learning Reasoning] This paper identifies a fundamental limitation of GRPO: it cannot learn from problems that the model completely fails to solve (pass rate = 0%), producing zero gradients. The proposed method, NuRL, addresses this by injecting self-generated abstract hints (without revealing answers) into hard problems during training, converting them into learnable samples. NuRL consistently outperforms GRPO across 3 models and 6 benchmarks, and genuinely improves the pass@k capability upper bound.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Reinforcement Learning Reasoning
  - GRPO Improvement
  - Self-Generated Hints
  - Capability Upper Bound
  - Zone of Proximal Development
date: 2026-05-08
content_hash: cf64c1314251d600
---

# Nudging the Boundaries of LLM Reasoning

**Conference**: ICLR 2026
**arXiv**: [2509.25666](https://arxiv.org/abs/2509.25666)
**Code**: [GitHub](https://github.com/SalesforceAIResearch/NuRL)
**Area**: LLM Reasoning
**Keywords**: Reinforcement Learning Reasoning, GRPO Improvement, Self-Generated Hints, Capability Upper Bound, Zone of Proximal Development

## TL;DR
This paper identifies a fundamental limitation of GRPO: it cannot learn from problems that the model completely fails to solve (pass rate = 0%), producing zero gradients. The proposed method, NuRL, addresses this by injecting self-generated abstract hints (without revealing answers) into hard problems during training, converting them into learnable samples. NuRL consistently outperforms GRPO across 3 models and 6 benchmarks, and genuinely improves the pass@k capability upper bound.

## Background & Motivation
- **Core Limitation**: Online RL (GRPO) produces zero gradients for problems that are completely unsolvable (all rollouts incorrect), preventing the model from learning from them.
- **Distribution Sharpening Hypothesis**: Growing evidence suggests that RL post-training primarily performs "distribution sharpening"—increasing the probability of already-known solutions—rather than discovering new reasoning capabilities.
- **Unchanged pass@k**: pass@k at large $k$ often remains unchanged after RL training, indicating that the capability upper bound has not been surpassed.
- **Zone of Proximal Development**: Drawing an analogy to Vygotsky's Zone of Proximal Development—hard problems lie in the "learning zone" but cannot be entered without guidance.
- **Value of Hard Problems**: These "unsolvable" problems contain rich learning signals, exposing the model's weaknesses.
- **Self-Sufficiency Requirement**: A method is needed to push beyond capability boundaries without relying on external, stronger models.

## Method

### Overall Architecture
NuRL = Offline Hint Collection + Online Rollout Augmentation (two-stage training)

### Key Designs
1. **Offline Hint Collection**:

    - Input: (question $q$, correct answer $a$)
    - Step 1: The model generates a CoT explaining "why the answer is correct": $y = \pi_{old}(q, a; p_y)$
    - Step 2: An abstract high-level hint (core knowledge cue) is extracted from the CoT: $h = \pi_\theta(q, a, y; p_h)$
    - Key Constraint: Hints must be abstract and high-level, containing neither the specific answer nor explicit solution steps.

2. **Online Rollout Augmentation**:

    - During GRPO training, $\mathcal{G}$ rollouts are generated per problem.
    - If all rollouts fail (pass rate = 0%): the hint is appended to the problem.
    - $\mathcal{G}-1$ hint-augmented rollouts are regenerated, plus 1 rollout without a hint (to avoid zero variance when all are correct).
    - Hints are not used at inference time—hints during training help the model internalize reasoning patterns.

3. **Hint Type Exploration**:

    - Abstract cues (best) > Partial steps > Explanations > Direct answers (worst)
    - Core finding: exposing more answer information leads to worse performance—consistent with human learning principles.

### Loss & Training
- Stage 1: Standard GRPO training until training reward and validation accuracy converge.
- Stage 2: NuRL continues training—easy problems where all rollouts are correct in Stage 1 are filtered out.

## Key Experimental Results

### Main Results

| Model | Method | MATH500 | MATH Hard | AIME | GPQA | Date | Avg. |
|------|------|:-------:|:---------:|:----:|:----:|:----:|:----:|
| Llama-3B | GRPO | 56.92 | 30.11 | 8.33 | 27.98 | 57.10 | 35.87 |
| Llama-3B | **NuRL(Self)** | **58.04** | **31.62** | **9.17** | **28.28** | **61.65** | **37.49** |
| OctoThinker-3B | GRPO | 68.81 | 41.29 | 8.33 | 23.26 | 69.85 | 42.63 |
| OctoThinker-3B | **NuRL(Self)** | **70.13** | **42.07** | **9.66** | **27.15** | **71.75** | **44.38** |

### Ablation Study

| Configuration | MATH500 | GPQA | Note |
|------|:-------:|:----:|------|
| Hint from scratch + no trigger | 53.41 | 24.84 | Worst |
| Hint from scratch + full-failure trigger only | 56.06 | 27.63 | Trigger helps |
| Two-stage + no trigger | 53.09 | 26.62 | Two-stage also helps |
| Two-stage + full-failure trigger only (NuRL) | **58.04** | **28.28** | Best |

### Key Findings
- NuRL improves pass@1024: GPQA 63.6%→69.7%, Date 86.4%→94.0%, while GRPO remains unchanged—demonstrating a genuine breakthrough in the capability upper bound.
- Teacher hints (GPT-o4-mini) yield a further +3.44% gain; self-generated hints are already effective.
- NuRL + Self-Consistency improves by 9.4% vs. GRPO + SC at only 7.8%—indicating stronger complementarity.
- The proportion of learnable problems increases from 66% to 70% during training, whereas standard GRPO remains at 66%.

## Highlights & Insights
- Clearly and insightfully reveals the fundamental limitation of GRPO in learning from unsolvable problems.
- The analogy to Vygotsky's Zone of Proximal Development is accurate and apt, lending strong motivational grounding.
- "More abstract hints are better" is counterintuitive yet compelling—providing direct answers performs worst due to reward hacking.
- Self-generated hints require no external model, avoiding distribution shift and achieving full self-sufficiency.
- The two-stage strategy (GRPO convergence first, then NuRL) is concise and practically deployable.

## Limitations & Future Work
- Performance gains are relatively modest (+1–2% on average), with limited improvement on stronger models (Qwen3-4B, +0.79%).
- The quality of self-generated hints is bounded by the model's own capability—extremely hard problems may not yield useful hints.
- The binary trigger (complete failure vs. partial success) for hint injection lacks a more fine-grained strategy.
- Offline hint collection requires gold answers, limiting applicability to answer-free settings.
- Hint quality evaluation and dynamic update mechanisms remain unexplored.

## Related Work & Insights
- **vs. GRPO/DAPO/Dr.GRPO**: These methods improve advantage estimation, KL divergence, or sampling; NuRL is orthogonal, addressing the "unsolvable sample" problem directly.
- **vs. STaR**: STaR uses answer-conditioned reasoning; NuRL further abstracts this into hints that do not reveal the answer.
- **vs. TBA**: TBA generates diverse trajectories via multi-node search; NuRL reduces problem difficulty through hints.

## Rating
- Novelty: ⭐⭐⭐⭐ — Insight into GRPO upper-bound limitations + self-generated hint framework
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 models, 6 benchmarks, multiple hint types, pass@k analysis
- Writing Quality: ⭐⭐⭐⭐⭐ — ZPD analogy is elegant; motivation → method → experiments flows smoothly
- Value: ⭐⭐⭐⭐ — Addresses a practical bottleneck in RL reasoning training with a concise, deployable approach

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](../../ACL2026/llm_reasoning/challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ICLR 2026\] The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_with_prefix_cons.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)

<!-- RELATED:END -->

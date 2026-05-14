---
title: >-
  [Paper Note] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning
description: >-
  [NeurIPS 2025][LLM Reasoning][Self-Explanation] This paper proposes Self-Explanation Policy Optimization (ExPO), a modular framework that addresses the fundamental challenge of distribution sharpening in RL post-training…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Self-Explanation"
  - "GRPO"
  - "DPO"
  - "Positive Sample Generation"
  - "Hard Reasoning"
  - "Distribution Sharpening"
date: 2026-05-08
content_hash: 573cedf634182284
---

# ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2507.02834](https://arxiv.org/abs/2507.02834)
**Code**: [GitHub](https://github.com/HumainLab/ExPO_rl_reasoning_by_explanation)
**Area**: LLM Reasoning / Reinforcement Learning Post-Training
**Keywords**: Self-Explanation, GRPO, DPO, Positive Sample Generation, Hard Reasoning, Distribution Sharpening

## TL;DR
This paper proposes Self-Explanation Policy Optimization (ExPO), a modular framework that addresses the fundamental challenge of distribution sharpening in RL post-training methods such as GRPO. When the model's initial success rate on hard reasoning tasks is near zero, effective positive samples are unavailable for learning. ExPO resolves this by prompting the model to generate reasoning chains (self-explanations) conditioned on the ground-truth answer. The resulting self-explanation samples are both in-distribution with respect to the current policy and provide positive learning signals. ExPO integrates seamlessly into both DPO and GRPO frameworks.

## Background & Motivation

1. **Distribution Sharpening in GRPO**: Current RL post-training methods such as GRPO rely on the model's own rollouts to obtain positive samples. On hard reasoning tasks (e.g., MATH Level 5), where the model's initial accuracy is extremely low, all sampled responses may be incorrect, causing the advantage term to vanish and the KL term to dominate, leading to policy degradation rather than improvement.

2. **Discarding Hard Problems Is Not a Solution**: Existing work sidesteps this issue by discarding training samples for which all rollouts are incorrect. This avoidance, however, does not solve the underlying problem—the model never learns to solve problems it cannot currently handle.

3. **Limitations of Expert CoT**: Although using human-written expert reasoning chains (expert CoT) may seem intuitive, experiments show they consistently underperform self-generated samples. Expert CoT has very low probability under the current policy (out-of-distribution), so the resulting gradient signal contributes little to policy improvement.

4. **Reinforcing Only Existing Capabilities**: GRPO-style methods are essentially distribution sharpening operations—they increase the probability of already-correct high-probability responses but cannot guide the model to explore novel reasoning paths for problems it has never solved.

5. **Scarcity of Positive Samples Is the Core Bottleneck**: Negative samples (incorrect responses) are always abundant in RL post-training, but effective positive samples are extremely scarce for hard tasks, constituting the fundamental bottleneck limiting reasoning capability improvement.

6. **Lack of Theoretical Guidance**: Prior to this work, there was no systematic theoretical analysis of what properties make positive samples effective for RL post-training. Methods such as STaR were empirically useful but their success was not fully understood.

## Core Problem
When the model's initial success rate is near zero during RL post-training, how can one obtain positive samples that effectively guide learning? What properties should such samples possess?

## Method

### Two Properties of Ideal Positive Samples
Through gradient analysis of policy improvement, the paper rigorously demonstrates that ideal positive samples must satisfy:
1. **In-distribution**: Samples should have high probability under the current policy $\pi_\theta$—when probability is too low, the gradient contribution to policy improvement (the $T_1$ term) approaches zero.
2. **Positive learning signal**: The CoT $c_1$ of the sample should yield a higher conditional probability of the correct answer than alternative CoTs, i.e., $\pi_\theta(a^*|q, c_1) > \pi_\theta(a^*|q, c_2)$.

### Self-Explanation Generation
The core idea is elegantly simple: given a question $q$ and the correct answer $a^*$, the model generates a reasoning chain conditioned on both:
$$\tilde{c} \sim \pi_\theta(\text{cot} | q, a^*)$$
This conditional generation reduces task difficulty (from open-ended problem solving to conditional explanation), enabling the model to produce reasoning chains of higher quality than standard CoT while remaining within the policy's distribution.

### Why Self-Explanation Outperforms Expert CoT
- **In-distribution**: The self-explanation prompt differs from the standard CoT prompt by only a few tokens (the correct answer), so $\pi(\cdot|q, a^*)$ is close to $\pi(\cdot|q)$, yielding substantially lower NLL than expert CoT.
- **Positive learning signal**: Lemma 2 proves that self-explanations are on average more likely to lead to the correct answer than standard CoT.
- **Natural process supervision**: The deviation between self-explanations and incorrect CoTs is small, helping the model precisely identify which reasoning steps require adjustment.

### ExP-DPO Instantiation
- **Offline variant**: The initial policy generates all self-explanations once as preferred responses, with self-generated standard CoT as dispreferred responses.
- **Online iterative variant**: Self-explanations are periodically regenerated using the updated policy to prevent distributional drift from invalidating the positive samples.

### ExP-GRPO Instantiation
An ExP-SFT term $\beta \log \pi_\theta(\tilde{c}, a^* | q)$ is added to the GRPO objective. When all sampled responses are incorrect, ExP-SFT provides a learning signal to restart the otherwise stalled trial-and-error loop. An annealing schedule for $\beta$ can be applied to gradually reduce reliance on potentially imperfect CoTs.

## Key Experimental Results

### Table 1: ExP-DPO Results (Pass@4)

| Setting | Model | Positive Sample Type | MATH | GSM8K |
|---------|-------|----------------------|------|-------|
| Offline | Qwen2.5-3B | ExPO $\tilde{c}$ | **54.3** | **80.1** |
| Offline | Qwen2.5-3B | Expert CoT $c_E$ | 43.7 | 69.6 |
| Online | Qwen2.5-3B | ExPO $\tilde{c}$ | **60.4** | **85.4** |
| Online | Qwen2.5-3B | Expert CoT $c_E$ | 49.3 | 76.3 |

ExPO self-explanations substantially outperform expert CoT across all settings (MATH: +10.6–11.1%).

### Table 2: MATH Accuracy by Difficulty Level (Qwen2.5-3B-Instruct)

| Method | Level-1 | Level-2 | Level-3 | Level-4 | Level-5 |
|--------|---------|---------|---------|---------|---------|
| Base (pass@64) | 97% | 88% | 75% | 32% | 4% |
| GRPO | 91% | 84% | 77% | 39% | 2% |
| GRPO SFT-GT-CoT | 95% | 89% | 83% | 65% | 12% |
| **ExP-GRPO** | **96%** | **91%** | **86%** | **76%** | **23%** |

On the hardest Level-5 problems, ExP-GRPO raises accuracy from GRPO's 2% to **23%**, an 11.5× improvement.

## Highlights & Insights

1. **Theoretical rigor combined with practical utility**: Gradient analysis rigorously establishes the two necessary properties of ideal positive samples, providing a theoretical explanation for the empirical success of prior methods such as STaR.
2. **Minimalist and elegant design**: The core idea requires only conditioning generation on the correct answer—no external models, no additional annotations, and no architectural changes.
3. **Modular design**: ExPO integrates seamlessly into both DPO and GRPO, the two dominant post-training frameworks.
4. **Addresses a genuine pain point**: ExPO not only improves sample efficiency but enables the model to learn problems it previously could not solve at all—a qualitative shift from reinforcing existing capabilities to acquiring new ones.
5. **Counterintuitive yet compelling finding**: The result that expert CoT underperforms self-generated explanations challenges the common assumption that higher data quality is always better.

## Limitations & Future Work

1. **Experiments limited to mathematical reasoning**: Although code generation and other settings are discussed, empirical validation is confined to MATH and GSM8K; generalization to other reasoning tasks remains to be demonstrated.
2. **Dependence on verifiable rewards**: The method requires an outcome verifier to judge answer correctness, limiting direct applicability to open-ended generation tasks.
3. **Limited model scale**: Experiments use 3B-scale models (Qwen2.5-3B, LLaMA-3.2-3B); whether the gains persist for 7B+ models is unknown.
4. **Upper bound on self-explanation quality**: If the model cannot generate meaningful reasoning chains even when conditioned on the correct answer, the method fails—the authors acknowledge this implies the problem exceeds the scope of RL post-training.
5. **Scheduling of the SFT weight $\beta$**: While an annealing strategy is proposed, the optimal schedule does not appear to be thoroughly explored in the experiments.

## Related Work & Insights

- **vs. STaR**: STaR first proposed regenerating reasoning chains conditioned on the correct answer, but only within SFT training via direct prompting. ExPO generalizes this idea to the RL framework and provides theoretical justification.
- **vs. GRPO / DeepSeek-R1**: GRPO suffers from distribution sharpening on hard tasks; ExPO directly addresses this by injecting self-explanation learning signals.
- **vs. Expert CoT-based methods**: Experiments demonstrate that expert CoT (out-of-distribution) is less effective than self-explanations (in-distribution) in DPO/GRPO training—a counterintuitive result that is nonetheless theoretically interpretable.
- **vs. Self-correction / self-refinement methods**: Such methods iteratively refine the model's own outputs but lack a reliable starting point when the initial response is entirely incorrect. ExPO provides a dependable "cold-start" mechanism through conditional generation.

## Rating
- Novelty: ⭐⭐⭐⭐ — Formalizes conditional generation as the ideal positive sample mechanism with rigorous theoretical support, unifying scattered intuitions into a coherent framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual-framework validation (DPO/GRPO) and fine-grained difficulty-level analysis are thorough, though model scale and task diversity are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative from problem formulation through theoretical analysis, algorithm design, and empirical validation is exceptionally fluent and internally consistent.
- Value: ⭐⭐⭐⭐⭐ — Targets a core bottleneck in RL post-training for reasoning; the method is simple and general, offering important guidance for both research and engineering practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](../../AAAI2026/llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)

</div>

<!-- RELATED:END -->

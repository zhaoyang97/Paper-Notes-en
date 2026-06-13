---
title: >-
  [Paper Note] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback
description: >-
  [AAAI 2026][LLM Reasoning][Chain-of-Thought reasoning] This paper proposes the InTRO framework, which aligns the model's generation policy with its answer-conditioned posterior via KL divergence minimization. By enabling…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought reasoning"
  - "token-level exploration"
  - "self-feedback"
  - "KL divergence alignment"
  - "mathematical reasoning"
date: 2026-05-08
content_hash: b2f71fbd2b5bbf0c
---

# In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback

**Conference**: AAAI 2026
**arXiv**: [2511.09865](https://arxiv.org/abs/2511.09865)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Chain-of-Thought reasoning, token-level exploration, self-feedback, KL divergence alignment, mathematical reasoning

## TL;DR

This paper proposes the InTRO framework, which aligns the model's generation policy with its answer-conditioned posterior via KL divergence minimization. By enabling token-level exploration and self-generated feedback within a single forward pass, InTRO improves both accuracy and conciseness of LLM reasoning without relying on any external supervision.

## Background & Motivation

Training LLMs for Chain-of-Thought (CoT) reasoning faces a fundamental dilemma:

**Limitations of SFT**: Supervised fine-tuning relies on a single "golden" reasoning path, penalizing equally valid alternative reasoning chains and leading to poor generalization. Obtaining high-quality step-by-step human annotations is also prohibitively expensive.

**Challenges of RL methods**: Reinforcement learning approaches such as GRPO use sequence-level sparse rewards, suffering from the "curse of dimensionality"—the space of valid reasoning sequences grows exponentially with length, making credit assignment extremely difficult and computational cost prohibitive.

**Cost of process supervision**: Introducing external verifier models or human annotations to evaluate individual reasoning steps alleviates reward sparsity but introduces new problems, including high annotation cost, large computational overhead, and verifier noise.

The paper poses a core question: **Can a model autonomously explore at the token level while generating its own feedback signals, completely eliminating dependence on external supervision?**

InTRO answers affirmatively. The core insight is that the generation distribution $\pi_\theta(z|x)$ of the optimal reasoning policy should naturally favor correct and logically consistent reasoning paths—which are precisely the paths the model would generate given the correct answer. The model's own answer-conditioned posterior $\pi_\theta(z|x,y)$ thus serves as an ideal "teacher" distribution.

## Method

### Overall Architecture

The InTRO training pipeline consists of three stages:
1. The policy $\pi_\theta$ generates multiple reasoning paths for query $x$; only paths with correct answers are retained.
2. For each retained path, $n$ candidate tokens are sampled at every position; the forward policy probability and the answer-conditioned posterior probability are computed for each candidate, yielding a token-level correction factor.
3. The policy is updated via weighted gradients that reinforce tokens most relevant to the correct answer.

### Key Designs

#### 1. **From Intractable Optimization to KL Divergence Alignment**

The ideal objective for optimal reasoning is to maximize the marginal log-likelihood:

$$\mathcal{L}_{\text{marg.}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[-\log\sum_{z\in\mathcal{Z}_y}\pi_\theta(z,y|x)\right]$$

Directly optimizing this objective is computationally intractable, as it requires enumerating all valid reasoning paths. InTRO instead minimizes the forward KL divergence:

$$\min_\theta D_{\text{KL}}(\pi_\theta(z|x,y) \| \pi_\theta(z|x))$$

Forward KL is chosen because it encourages the policy to expand its support, thereby better capturing diverse valid solutions.

**Key theoretical guarantee (Proposition 2.1)**: Under the assumption that $y=f(z)$ is a deterministic mapping, the gradient of the marginal log-likelihood is exactly equivalent to the gradient of KL divergence minimization:

$$\nabla_\theta \log \pi_\theta(y|x) = \mathbb{E}_{z\sim\pi_\theta(z|x,y)}[\nabla_\theta \log \pi_\theta(z|x)]$$

This means that through KL alignment, InTRO effectively performs gradient ascent on the otherwise intractable marginal likelihood objective.

#### 2. **Estimated Posterior and Correction Factor**

Direct sampling from $\pi_\theta(z|x,y)$ remains infeasible, so InTRO constructs an estimated posterior $\pi_\theta(\cdot|x \oplus y)$ by concatenating the question and answer and prompting the model to generate the reasoning chain. This leverages the powerful in-context learning ability of LLMs—given the known answer, the model's generated reasoning tends to follow correct paths.

Via importance sampling, the core training objective becomes:

$$\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\frac{1}{|z|\cdot n}\sum_{t=1}^{|z|}\sum_{i=1}^{n} w_t^i \cdot \log\pi_\theta(z_t^i|x, z_{<t})\right]$$

where the correction factor is defined as:

$$w_{t,i} = \frac{\pi_\theta(z_t^i|x\oplus y, z_{<t})}{\pi_\theta(z_t^i|x, z_{<t})}$$

- $w_t^i > 1$: the posterior probability exceeds the forward policy probability, indicating this token positively contributes to reaching the correct answer and should be reinforced.
- $w_t^i < 1$: the token contributes little to the correct answer and should be suppressed.
- In practice, $w_t^i$ is clipped to $[0, 200]$ to ensure training stability.

#### 3. **Token-Level Exploration Mechanism**

At each time step, $n$ candidate tokens (including the original token) are sampled, directly encouraging token-level exploration. This is fundamentally different from traditional RL methods that explore at the sequence level:

- **Traditional RL** (e.g., GRPO): samples multiple complete reasoning sequences at the sequence level and relies on sparse outcome rewards.
- **InTRO**: performs fine-grained exploration at each token position and obtains dense self-feedback signals via the correction factor.

### Loss & Training

- Implemented on the OpenRLHF framework using 80GB A100 GPUs.
- Training data: MATH dataset difficulty levels 3–5, approximately 9.2k samples.
- Batch size 128, learning rate 5e-7.
- 4 candidate reasoning paths per query ($G=4$), 5 candidate tokens sampled per step ($n=5$).
- Binary reward: correct answer = 1.0, incorrect = 0.0.
- No prompt templates used (following best practices of the Qwen series).

## Key Experimental Results

### Main Results

| Model | MATH500 | Minerva | Olympiad | College | AMC23 | AIME25 | Avg | Gain (%) |
|-------|---------|---------|----------|---------|-------|--------|-----|----------|
| Qwen2.5-1.5B base | 50.4 | 11.4 | 14.1 | 36.7 | 23.2 | 0.6 | 22.7 | - |
| GRPO | 50.6 | 9.6 | 17.9 | 37.0 | 24.5 | 1.1 | 23.5 | +3.5 |
| **InTRO** | **54.2** | **9.6** | **19.6** | **38.4** | **25.5** | **1.8** | **24.9** | **+9.7** |
| Qwen2.5-7B base | 66.4 | 15.1 | 30.4 | 42.4 | 41.9 | 5.0 | 33.5 | - |
| GRPO | 71.8 | 17.6 | 33.9 | 44.7 | 46.2 | 5.1 | 36.6 | +9.3 |
| **InTRO** | **72.6** | **19.9** | **35.3** | **45.0** | **47.0** | **5.6** | **37.6** | **+12.2** |
| Qwen3-4B base | 69.0 | 12.5 | 31.3 | 30.3 | 45.5 | 8.9 | 32.9 | - |
| GRPO | 73.8 | 15.8 | 34.4 | 34.2 | 50.9 | 7.9 | 36.2 | +10.0 |
| **InTRO** | **74.8** | **17.6** | **39.4** | **35.1** | **58.3** | **12.6** | **39.6** | **+20.4** |
| Qwen3-8B base | 65.8 | 11.8 | 34.7 | 29.8 | 53.4 | 10.0 | 34.3 | - |
| GRPO | 74.4 | 14.3 | 36.3 | 33.1 | 55.8 | 10.9 | 37.5 | +9.3 |
| **InTRO** | **75.2** | **18.8** | **38.7** | **35.2** | **56.7** | **12.4** | **39.5** | **+15.2** |

### Ablation Study

**Effect of token exploration count $n$ (Qwen2.5-1.5B)**:

| Sampled tokens $n$ | 1 | 2 | 5 | 10 | 20 | 40 |
|--------------------|---|---|---|----|----|----|
| Average accuracy | 20.1 | 24.4 | 24.9 | 25.6 | 25.0 | 23.8 |

**OOD generalization (Qwen3-4B)**:

| Task | LiveCodeBench | BigCodeBench | GPQA | HumanEval | IFEval | Avg |
|------|--------------|-------------|------|-----------|--------|-----|
| Base | 4.1 | 27.7 | 22.1 | 81.7 | 45.6 | 36.2 |
| GRPO | 6.2 | 27.8 | 21.9 | 83.5 | 40.6 | 36.0 |
| **InTRO** | **22.6** | **35.4** | **38.6** | **89.2** | **50.4** | **47.2** |

### Key Findings

1. **Scaling effect**: Stronger base models (e.g., the Qwen3 series) yield larger gains; Qwen3-4B achieves up to 20.4% improvement on mathematical reasoning.
2. **Reasoning conciseness**: InTRO generates substantially shorter reasoning chains, especially on difficult problems, while maintaining higher accuracy.
3. **Cross-domain generalization**: Despite training solely on mathematical data, InTRO achieves significant improvements on code, knowledge, and instruction-following tasks.
4. **Answer-conditioned reasoning enhancement**: Conditioning on the answer substantially boosts performance on difficult benchmarks (AIME25, Olympiad), but yields limited gains on simpler tasks.
5. **Computational efficiency**: InTRO requires only two forward passes (policy + posterior), whereas GRPO requires $2G$ passes.

## Highlights & Insights

- **Theoretical elegance**: The intractable marginal likelihood optimization is equivalently converted to KL divergence minimization, which is then made practically feasible via the estimated posterior—each step is grounded in clear theoretical justification.
- **Self-feedback mechanism**: The method requires no external reward model or verifier; the model evaluates each token's contribution by comparing generation probabilities conditioned with and without the answer.
- **Emergent conciseness**: InTRO does not explicitly optimize for reasoning length, yet the correction factor naturally suppresses redundant tokens, leading to more concise reasoning.
- **Explanation for OOD generalization**: Token-level minimization of information discrepancy strengthens causal linkages, enabling logic-driven generalization across domains.

## Limitations & Future Work

- The estimated posterior $\pi_\theta(\cdot|x\oplus y)$ deviates from the true posterior (measured KL ≈ 2.3 in experiments), limiting effectiveness on weaker models.
- The clipping range $[0, 200]$ for the correction factor is empirically determined and lacks theoretical justification.
- Training is conducted only on mathematical reasoning; while OOD capability is demonstrated, direct multi-task training may yield further improvements.
- The approach has a strong dependency on the base model's reasoning capability; gains for weaker models such as Llama are less pronounced than for Qwen.

## Related Work & Insights

- Similar to LaTRO in leveraging the posterior distribution, but LaTRO uses $\log p_\theta(y|x\oplus z)$ as the reward, whereas InTRO directly uses the posterior probability ratio as a correction factor.
- The fundamental distinction from GRPO: GRPO performs sequence-level exploration with sequence-level rewards, while InTRO performs token-level exploration with token-level feedback.
- The design of the correction factor may inspire the construction of other token-level training signals.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The theoretical framework for token-level self-feedback is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multiple models and benchmarks, with ablation, OOD, and efficiency analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear, though notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ — Introduces a fundamentally new paradigm for LLM reasoning training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger](sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)
- [\[AAAI 2026\] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)
- [\[NeurIPS 2025\] Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning](../../NeurIPS2025/llm_reasoning/martingale_score_an_unsupervised_metric_for_bayesian_rationality_in_llm_reasonin.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](../../ICLR2026/llm_reasoning/slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](../../ICML2026/llm_reasoning/reward_modeling_from_natural_language_human_feedback.md)

</div>

<!-- RELATED:END -->

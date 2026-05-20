---
title: >-
  [Paper Note] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning
description: >-
  [AAAI 2026][Reinforcement Learning][LLM Reasoning] This paper identifies the **Beginning Lock-in Effect (BLE)** in LLM reasoning — the initial reasoning steps significantly determine subsequent trajectories and final out…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "LLM Reasoning"
  - "RLVR"
  - "Prefix Optimization"
  - "Path Dependence"
  - "GRPO"
date: 2026-05-08
content_hash: bcb330e004a31f99
---

# Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning

**Conference**: AAAI 2026
**arXiv**: [2512.15274](https://arxiv.org/abs/2512.15274)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: LLM Reasoning, RLVR, Prefix Optimization, Path Dependence, GRPO

## TL;DR

This paper identifies the **Beginning Lock-in Effect (BLE)** in LLM reasoning — the initial reasoning steps significantly determine subsequent trajectories and final outcomes. Based on this finding, the paper proposes PPPO, a method that optimizes only prefix tokens (approximately 26% of all tokens), achieving accuracy improvements of up to 18.02% while reducing output token counts by up to 18.35%.

## Background & Motivation

**Training Efficiency in RLVR**: Reinforcement learning from verifiable rewards (RLVR, e.g., GRPO, DAPO) is a key technique for improving LLM reasoning capabilities (driving the success of DeepSeek R1 and OpenAI o1). However, existing methods suffer from a fundamental efficiency problem:

**Uniform Token-Level Training**: All generated tokens are trained indiscriminately, ignoring the heterogeneous contributions of different tokens to final performance.

**Drag from Low-Impact Tokens**: Substantial computational resources are spent optimizing low-return tokens, which in turn impedes potential improvements on critical tokens.

**Insufficiency of Existing Solutions**: Methods such as DAPO-FT optimize only high-entropy tokens (e.g., *wait*, *however*, *rethink*), which encourages exploration but cannot guarantee the quality of the triggered reasoning paths.

**Core Insight — Path Dependence Theory**:

The paper draws inspiration from **Path Dependence** (David, 1975) in human cognition: initial thought patterns substantially constrain subsequent reasoning trajectories. High-quality initial thinking guides toward desirable outcomes (benign path dependence), while low-quality initial thinking leads to suboptimal results (vicious path dependence).

**Empirical Validation — BLE Phenomenon**: The paper verifies a similar phenomenon in LLM reasoning through controlled experiments:
- Reasoning initiated from prefix tokens of correct answers: accuracy improves by up to 20.2%
- Reasoning initiated from prefix tokens of incorrect answers: accuracy drops by up to 27.5%
- Even after inserting reflective tokens (e.g., "wait", "however") following an incorrect prefix, the maximum recovery is only 9.2%

These results demonstrate that LLMs struggle greatly to recover from low-quality initial reasoning, confirming the **Beginning Lock-in Effect (BLE)** in LLM reasoning.

## Method

### Overall Architecture

Progressive Prefix-token Policy Optimization (PPPO) is a novel RLVR method whose core idea is:

**Apply gradient updates only to prefix tokens while masking gradient updates for all subsequent tokens.**

By teaching the LLM "how to begin reasoning with high quality," the method positively influences the entire reasoning trajectory:

$$P(\hat{y} = a | q, I_{high-quality}) > P(\hat{y} = a | q) > P(\hat{y} = a | q, I_{low-quality})$$

### Key Designs

#### 1. Prefix Gradient Masking

For each output $\mathbf{o}_i$, PPPO retains gradients only for the first $\eta$ proportion of tokens and masks the rest:

$$H(j, \mathbf{o}_i) = \mathbb{1}(j \leq \lfloor \eta \cdot |\mathbf{o}_i| \rfloor)$$

where $\eta$ is the retention ratio. PPPO's optimization objective introduces the $H(\cdot, \cdot)$ mask on top of the standard GRPO/DAPO objective:

$$\mathcal{J}_{PPPO}(\theta) = \mathbb{E}\left[\frac{1}{\sum_k |\mathbf{o}_k|} \sum_{i=1}^{N} \sum_{j=1}^{|\mathbf{o}_i|} H(j, \mathbf{o}_i) \cdot \min\left(r_{i,j}(\theta) \hat{A}_{i,j}, \text{clip}(\ldots)\hat{A}_{i,j}\right)\right]$$

**Design Motivation**: Directly concentrates training resources on the most influential tokens rather than dispersing them across the entire sequence. The empirical findings on BLE confirm that prefix quality has a decisive impact on the entire reasoning trajectory.

#### 2. Progressive Prefix Retention

A key innovation: $\eta$ is not fixed but gradually increases during training:

$$\eta = \begin{cases} \eta, & \text{if } \Delta acc > 0 \\ \eta + \Delta\eta, & \text{if } \Delta acc \leq 0 \end{cases}$$

When validation accuracy stops improving (no gains over consecutive steps), $\eta$ is increased. The initial value is $\eta = 15\%$, incremented in steps of 5% up to 35%.

**Design Motivation**:
- Starting with short prefixes reduces learning complexity, allowing the LLM to quickly establish core competencies for high-quality reasoning initiation.
- Gradual expansion builds on a solid foundation, maintaining learning quality and stability as the scope extends to longer sequences.
- Adaptive triggering automatically adjusts based on performance plateaus, avoiding premature or delayed transitions.

**Why 15%–35%?** Experimental findings (Figure 2b) show:
- At 15%, model performance exhibits a significant inflection point — BLE manifests within the first 15% of tokens.
- At 35%, performance stabilizes — BLE is largely established by 35% of tokens.

#### 3. Continuation Accumulated Reward

Addresses high stochasticity in evaluating prefix token quality:

For each output $\mathbf{o}_i$, the first $\eta$ tokens form a prefix $\mathbf{b}_i$. Starting from $\mathbf{b}_i$, $G$ continuation sequences $\{\mathbf{c}_j\}_{j=1}^G$ are generated, and their accumulated scores serve as the reward for the prefix:

$$R_i = \sum_{j=1}^{G} \mathbb{1}(\hat{y}_{\mathbf{c}_j} = a) + \mathbb{1}(\hat{y}_{\mathbf{o}_i} = a)$$

**Design Motivation**:
- Single-sample evaluation of prefix quality suffers from high randomness (the same prefix may lead to either correct or incorrect continuations).
- Accumulating scores over multiple continuation samples reduces randomness and provides a more reliable signal of prefix quality.
- Analogous to Monte Carlo estimation: approximating the value of a state through multiple simulations.

### Loss & Training

- **Backbone**: Qwen3 series (1.7B, 4B, 8B), thinking mode
- **Training Data**: DAPO-Math-17K
- **Sampling**: $N=8$ outputs per question; $G=8$ continuations per prefix (64 samples total per question)
- **Learning Rate**: $1 \times 10^{-6}$
- **Clip Parameters**: $\varepsilon_{low} = 0.2$, $\varepsilon_{high} = 0.28$
- **Max Output Length**: 10,240 tokens
- **All baselines sample 64 outputs** to ensure fair comparison

## Key Experimental Results

### Main Results

**Accuracy comparison across 5 reasoning benchmarks (avg@32)**:

| Method | AIME'24 | AIME'25 | MATH500 | AMC'23 | GPQA Diamond | Avg |
|--------|---------|---------|---------|--------|-------------|-----|
| Qwen3-4B (baseline) | 48.75 | 35.42 | 84.46 | 72.67 | 43.59 | 56.98 |
| + GRPO | 52.08 | 37.71 | 88.40 | 76.77 | 46.78 | 60.35 |
| + DAPO | 56.46 | 42.08 | 92.33 | 81.63 | 49.37 | 64.37 |
| + DAPO-FT | 56.25 | 42.08 | 92.38 | 82.00 | 49.21 | 64.38 |
| + **PPPO (Ours)** | **63.54** | **53.44** | **94.60** | **83.06** | **52.07** | **69.34** |

| Method | AIME'24 | AIME'25 | MATH500 | AMC'23 | GPQA Diamond | Avg |
|--------|---------|---------|---------|--------|-------------|-----|
| Qwen3-8B (baseline) | 52.29 | 38.75 | 86.06 | 75.08 | 46.12 | 59.66 |
| + DAPO | 63.13 | 48.75 | 93.21 | 83.96 | 55.18 | 68.85 |
| + DAPO-FT | 63.75 | 49.38 | 93.65 | 84.11 | 54.77 | 69.13 |
| + **PPPO (Ours)** | **72.19** | **59.69** | **94.73** | **86.75** | **58.13** | **74.30** |

### Ablation Study

**Training efficiency comparison**:

| Model | Method | Avg Accuracy Improvement (AAI) ↑ | Proportion of Optimized Tokens (POT) ↓ | Learning Efficiency (LE=AAI/POT) ↑ |
|-------|--------|----------------------------------|----------------------------------------|------------------------------------|
| Qwen3-4B | GRPO | 3.37 | 100% | 3.37 |
| Qwen3-4B | DAPO | 7.39 | 100% | 7.39 |
| Qwen3-4B | DAPO-FT | 7.39 | 20% | 37.02 |
| Qwen3-4B | **PPPO** | **12.36** | **26.17%** | **47.24** |
| Qwen3-8B | DAPO | 9.19 | 100% | 9.19 |
| Qwen3-8B | DAPO-FT | 9.47 | 20% | 47.36 |
| Qwen3-8B | **PPPO** | **14.64** | **24.83%** | **58.95** |

**Ablation on number of continuation samples** (Qwen3-4B):

| Sampling Strategy | G | avg@4 ↑ | var@4 ↓ |
|------------------|---|---------|---------|
| Single | 1 | 60.46 | 3.30 |
| Multiple | 4 | 66.11 | 1.47 |
| Multiple | 8 | 69.36 | 0.63 |
| Multiple | 16 | 69.53 | 0.56 |

### Key Findings

1. **PPPO achieves the best performance in 14 out of 15 settings** (3 models × 5 benchmarks); the sole exception trails by only 0.14%.
2. **Substantial training efficiency gains**: optimizing only ~25% of tokens yields accuracy improvements of up to 14.64%, with learning efficiency (LE) up to 17.5× that of GRPO.
3. **Simultaneous inference efficiency improvement**: PPPO generates the fewest tokens, reducing output length by up to 18.35%, indicating that high-quality reasoning initiation leads to more concise reasoning paths.
4. **Transferability of prefix quality**: prefix tokens generated by PPPO-trained Qwen3-4B can guide Qwen2.5-7B-Instruct to improve accuracy by 9.04%.
5. **Accumulated reward substantially reduces variance**: at $G=8$, variance decreases from 3.30 to 0.63, with accuracy improving by 8.9 percentage points.
6. **Progressive strategy outperforms fixed strategy**: improves accuracy by 2.19% and reduces training tokens by 8.83% compared to fixed $\eta=35\%$.

## Highlights & Insights

1. **BLE has broad significance**: validating path dependence in LLM reasoning provides a cognitive science perspective for understanding autoregressive generation properties.
2. **Counter-intuitive efficiency**: optimizing fewer tokens yields better results, challenging the intuition that "more optimization = better performance."
3. **Curriculum learning through "simple-to-complex" progression**: the progressive prefix retention strategy naturally induces a curriculum learning effect.
4. **Cross-model transferability of prefix quality**: suggests that high-quality reasoning initiation has a degree of generality not confined to specific models.
5. **Analogy to human cognition**: successfully maps path dependence theory from cognitive science to LLM reasoning, bridging two research communities.

## Limitations & Future Work

1. **Selection of $\eta$ thresholds**: the 15%–35% range is based on empirical observation; different models or tasks may require different ranges.
2. **Computational overhead**: each prefix requires additional sampling of $G=8$ continuations, bringing total samples to $8 \times 8 = 64$.
3. **Validation limited to math/science reasoning**: whether BLE holds equally for code generation, creative writing, and other tasks remains to be verified.
4. **Lack of experiments on larger models**: the largest model used is Qwen3-8B; the effectiveness at the 32B/70B scale is unknown.
5. **Hard truncation at the prefix boundary**: token-level truncation may cut through the middle of logical reasoning units.

## Related Work & Insights

- **GRPO** (Shao et al., 2024): RL optimization framework with the critic model removed.
- **DAPO** (Yu et al., 2025): addresses four issues in GRPO, including entropy collapse.
- **DAPO-FT** (Wang et al., 2025c): optimizes high-entropy forking tokens; the closest baseline method.
- **Path Dependence** (David, 1975): path dependence theory in human cognition.
- **DeepSeek R1**: a successful case study of the RLVR technical paradigm.
- Insight: **trajectory-specific optimization** outperforms uniform full-token optimization; cognitive science theories can effectively guide LLM training strategy design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Both the discovery of BLE and the PPPO method are highly original, offering a cross-disciplinary insight from cognitive science to LLM training.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three model scales, five benchmarks, four baseline methods, and multi-dimensional ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain of motivation → discovery → method → validation is complete and coherent.
- Value: ⭐⭐⭐⭐⭐ — Provides practical guidance for RLVR training efficiency; the BLE finding has the potential to influence a broad range of subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback](in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)
- [\[ICLR 2026\] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](../../ICLR2026/reinforcement_learning/textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)
- [\[AAAI 2026\] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination](reasoning_or_memorization_unreliable_results_of_reinforcement_learning_due_to_da.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](reasoning_with_exploration_an_entropy_perspective.md)

</div>

<!-- RELATED:END -->

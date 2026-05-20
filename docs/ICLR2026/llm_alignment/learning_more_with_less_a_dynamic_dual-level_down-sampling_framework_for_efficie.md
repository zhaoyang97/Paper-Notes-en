---
title: >-
  [Paper Note] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization
description: >-
  [ICLR 2026][LLM Alignment][GRPO] This paper proposes **D3S** (Dynamic Dual-Level Down-Sampling), a framework that maximizes advantage variance at the sample level and prioritizes high-entropy…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "GRPO"
  - "policy optimization"
  - "down-sampling"
  - "advantage variance"
  - "token selection"
  - "curriculum learning"
date: 2026-05-08
content_hash: bb8a8d2ccd6f9fc4
---

# Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization

**Conference**: ICLR 2026
**arXiv**: [2509.22115](https://arxiv.org/abs/2509.22115)

**Authors**: Chao Wang, Tao Yang, Hongtao Tian et al. (Tsinghua University & Tencent WeChat)
**Code**: Available (supplementary material)  
**Area**: LLM Alignment / Reinforcement Learning

**Keywords**: GRPO, policy optimization, down-sampling, advantage variance, token selection, curriculum learning

## TL;DR

This paper proposes **D3S** (Dynamic Dual-Level Down-Sampling), a framework that maximizes advantage variance at the sample level and prioritizes high-entropy, high-advantage tokens at the token level, combined with a dynamic scheduling strategy. D3S achieves faster convergence and superior performance using fewer than 20% of tokens.

## Background & Motivation

1. Critic-free methods (GRPO/GSPO) estimate advantages via group-relative rewards, eliminating the memory overhead of critic networks, yet **efficiency bottlenecks** persist.

2. Large groups contain numerous uninformative samples (e.g., all-correct or all-incorrect groups) that dilute key learning signals; valuable gradients are overwhelmed by the averaging effect of undifferentiated samples.

3. Small groups suffer from insufficient sampling diversity, limiting optimization precision—constituting an inherent **group size trade-off**.

4. Razin et al. find that increasing reward variance $\text{Var}(R)$ accelerates convergence, but since GRPO normalizes advantages to unit variance, **maximizing $\text{Var}(R)$ cannot change the gradient norm upper bound**.

5. At the token level, a large number of low-information tokens (simple or neutral tokens) similarly dilute gradient signals; Wang et al. find that the top 20% high-entropy tokens dominate policy gradients.

6. **Motivation**: Can one simultaneously select the most valuable data at both the sample and token levels to obtain stronger gradient signals with less computation?

## Method

### 3.1 Sample-Level: Cross-Group Advantage Variance Maximization Down-Sampling

**Core Idea**: Estimate group-relative advantages using all rollouts, then select a subset that maximizes advantage variance.

**Within-group selection**: Given query $x$ and its $G$ rollouts, select subset $\hat{\mathcal{S}}_{\text{query}}$ such that:

$$\hat{\mathcal{S}}_{\text{query}} = \arg\max_{\hat{S} \subset \mathcal{S}_{\text{query}},\,|\hat{S}|=N_{\hat{s}}} \text{Var}(A_{\hat{S}})$$

In practice, the $N_{\hat{S},\text{pos}}$ samples with the largest positive advantages and the $N_{\hat{S},\text{neg}}$ samples with the smallest negative advantages are selected.

**Cross-group operation**: Accounting for distributional differences across groups within a batch (some groups yield zero advantage due to all-correct or all-incorrect outcomes), a further cross-group selection is performed over the entire batch:

$$\hat{\mathcal{S}}_{\text{batch}} = \arg\max_{\hat{S} \subset S_{\text{batch}},|\hat{S}|=N} \text{Var}(A_{\hat{S}})$$

**Key design**: Advantages are normalized within each group but not re-normalized during cross-group selection, preserving the original distributional characteristics.

### 3.2 Token-Level: Entropy-Advantage Weighted Selection

A unified token importance metric is proposed, combining generation entropy and advantage:

$$H_{i,t} = -\sum_{j=1}^{V} \pi_\theta(\text{token}_j \mid x_i, y_{i,<t}) \log \pi_\theta(\text{token}_j \mid x_i, y_{i,<t})$$

$$\mathcal{T} = \text{top}_{K\%}(y_{i,t},\; y_{i,t} \in \hat{\mathcal{S}},\; \text{key} = |A_{i,t}| \times H_{i,t})$$

**Intuition**: Large $|A_{i,t}|$ indicates the token has high impact on reward; large $H_{i,t}$ indicates model uncertainty at that position. Tokens satisfying both criteria represent **the most worthwhile decision points for optimization**. Only the top-$K\%$ tokens contribute to gradient updates.

### 3.3 Dynamic Down-Sampling Schedule

To prevent overfitting under aggressive down-sampling, a linear schedule is introduced:

$$[N_s^{(p)}, K^{(p)}] = (1-p) \cdot [N_{\text{init}}, K_{\text{init}}] + p \cdot [N_{\text{final}}, K_{\text{final}}]$$

where $p \in [0,1]$ denotes training progress. At $p=0$, aggressive down-sampling (fewer samples and tokens) accelerates early learning; as $p \to 1$, constraints are progressively relaxed to incorporate more samples and tokens, preventing overfitting.

### Theoretical Guarantees

- **Proposition 1**: The GRPO gradient norm upper bound is fixed at $4\gamma(x;\theta)$, independent of $\text{Var}(R)$.
- **Proposition 2**: After down-sampling, the gradient norm upper bound $\propto (\text{Var}(A'))^{1/3}$, positively correlated with advantage variance.
- **Lemma 1**: A subset with variance $\geq 1$ can always be drawn from a normalized collection, guaranteeing the post-down-sampling gradient upper bound is no lower than the original.

## Key Experimental Results

### Table 1: Main Results (Pass@1 / Pass@8, 32 parallel generations)

| Model / Method | AIME24 | AIME25 | AMC23 | GSM8k | MATH | Minerva | Olympiad | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B Base | 8.9/33.2 | 2.3/13.4 | 22.8/70.4 | 30.1/83.2 | 27.9/64.6 | 8.4/33.7 | 4.1/14.6 | 14.9/44.7 |
| + GRPO | 13.2/37.6 | 5.5/21.6 | 47.0/83.5 | 64.9/94.3 | 48.5/70.2 | 19.8/45.0 | 9.7/19.8 | 29.8/53.1 |
| + GRPO+PODS | 16.1/40.5 | 7.8/24.5 | 52.8/81.5 | 73.3/95.0 | 53.0/71.1 | 24.6/47.5 | 11.0/20.7 | 34.1/54.4 |
| + **GRPO+D3S** | **20.3/48.2** | 7.9/25.8 | **54.4/87.1** | 73.4/95.7 | 52.2/71.5 | 25.0/48.2 | 10.7/20.8 | **34.3/56.8** |
| + **GSPO+D3S** | 18.3/43.3 | **8.3/26.9** | 53.2/83.8 | **76.0/96.1** | **54.9/71.4** | **28.4/51.1** | **11.5/21.1** | **35.8/56.2** |
| Llama3.1-8B + GRPO | 2.0/5.0 | 0.0/0.0 | 13.7/33.4 | 78.6/93.5 | 31.5/52.0 | 15.9/35.6 | 2.1/7.2 | 20.5/32.4 |
| + **GRPO+D3S** | **5.3/20.7** | 0.1/0.8 | **20.3/50.8** | **79.0/95.0** | **35.9/59.2** | **22.5/44.3** | **3.3/10.7** | **23.8/40.2** |

### Table 2: Ablation Study (Qwen2.5-Math-7B, Pass@1/Pass@8)

| Method | AIME24 | AIME25 | AMC23 | MATH | Avg |
|---|---|---|---|---|---|
| GRPO | 13.2/37.6 | 5.5/21.6 | 47.0/83.5 | 48.5/70.2 | 29.8/53.1 |
| +D1S (sample down-sampling only) | 13.2/42.9 | 5.9/20.2 | 50.6/84.4 | 50.1/70.5 | 31.3/54.2 |
| +D1S-Cross (+cross-group) | 17.3/40.0 | 7.7/25.6 | 51.9/83.3 | 52.8/70.9 | 34.1/54.7 |
| +D2S (+token level, no schedule) | 16.9/42.2 | 6.0/21.2 | 49.6/82.8 | 49.5/70.7 | 31.3/54.1 |
| +**D3S** (full framework) | **20.3/48.2** | **7.9/25.8** | **54.4/87.1** | 52.2/71.5 | **34.3/56.8** |

### Table 3: Training Efficiency Comparison

| Comparison | Avg@32 Gain | Training Speedup |
|---|---|---|
| D3S vs. GRPO (Qwen-7B) | +6% | 2.04× |
| D3S vs. GSPO (Qwen-7B) | +17% | 5.51× |
| D3S vs. GRPO (Qwen-1.5B) | +4% | 1.57× |

## Key Findings

1. **Sample-level and token-level down-sampling** effectively eliminate undifferentiated signals in early training, accelerating policy convergence.
2. Down-sampling methods without dynamic scheduling (D1S/D2S) accelerate early convergence but exhibit **overfitting** in later stages, where Avg@32 is eventually surpassed by vanilla GRPO.
3. **Dynamic scheduling** plays a critical role: relaxing down-sampling intensity in later stages sustains continuous improvement without overfitting.
4. D3S raises the Sample Usefulness Rate from ~70% to nearly 100%; cross-group operations effectively filter ambiguous data within the batch.
5. D3S better manages **entropy fluctuations**: it reduces entropy (greater certainty) on well-aligned models while promoting exploration on under-aligned models (e.g., Llama3.1).
6. KL divergence analysis shows D3S deviates less from the reference model, indicating lower overfitting risk.

## Highlights & Insights

- **Theoretical clarity**: Starting from gradient norm upper bounds, the paper rigorously proves that maximizing advantage variance outperforms maximizing reward variance, and derives the positive correlation of the form $(\text{Var}(A'))^{1/3}$.

- **Elegant token-level metric**: The $|A| \times H$ measure—"impact" × "uncertainty"—is intuitively well-motivated and empirically validated.

- **Plug-and-play design**: D3S is compatible with both GRPO and GSPO, and demonstrates consistent effectiveness across model architectures (Qwen/Llama) and scales (1.5B/7B/8B).

- **Curriculum-inspired scheduling**: The dynamic schedule elegantly balances training efficiency and generalization, drawing on principles from curriculum learning.

- **Comprehensive experimental design**: Seven benchmarks, four backbone models, two baseline algorithms, stepwise ablation, and training dynamics analysis.

## Limitations & Future Work

- Validation is limited to **mathematical reasoning** tasks; effectiveness on code generation, open-domain dialogue, and multimodal tasks remains unknown.

- The cross-group sample-level operation may introduce additional communication overhead in **distributed training** settings.

- The dynamic schedule employs simple linear interpolation; non-linear schedules (e.g., cosine, exponential) may be superior but are not explored.

- The behavior of D3S under different reward distributions (sparse vs. dense reward) is not analyzed in depth.

- Computing token entropy requires the full vocabulary probability distribution, introducing non-trivial additional computational cost.

## Related Work & Insights

| Method | Core Strategy | Advantage of D3S |
|---|---|---|
| PODS (Xu, 2025) | Maximizes $\text{Var}(R)$ for sample selection | D3S proves that maximizing $\text{Var}(A)$ yields a tighter gradient upper bound; PODS cannot alter the fixed upper bound after normalization |
| Razin (2024, 2025) | Reward variance accelerates convergence | D3S extends this insight from the reward to the advantage level, and adds fine-grained token-level selection |
| ETPO (Wen, 2024) | Entropy-regularized token-level optimization | D3S uses the product of entropy and advantage magnitude as a joint metric, focusing on tokens with high impact and high uncertainty |
| LPPO (Chen, 2025) | Dynamic reweighting based on learning progress | D3S theoretically guarantees gradient upper bound improvement while operating at both sample and token levels simultaneously |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-level down-sampling combined with dynamic scheduling is novel; the theoretical perspective of advantage variance maximization is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments across multiple models, benchmarks, and algorithms, with stepwise ablation and training dynamics analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The theory–method–experiment logical chain is clear; figures and tables are intuitive; proofs are detailed and complete.
- **Value**: ⭐⭐⭐⭐ — Combines theoretical rigor with practical applicability; strong plug-and-play utility; offers important reference value for RLHF efficiency research.

## Summary

The D3S framework selects the most valuable training data at two levels—sample-level advantage variance maximization and token-level entropy-advantage weighted selection—combined with dynamic scheduling to balance efficiency and generalization. Theoretical analysis rigorously establishes the positive correlation between advantage variance and the gradient norm upper bound. Experiments across seven mathematical reasoning benchmarks validate consistent performance gains and significant training acceleration (up to 5.51×). The method provides a systematic solution to data utilization efficiency in RLHF and can be extended in future work to broader task types and larger-scale models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](reinforcement_unlearning_via_group_relative_policy_optimization.md)
- [\[ICLR 2026\] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models
description: >-
  [NeurIPS 2025][LLM Reasoning][credit assignment] This paper proposes the SPO framework, which adopts segment-level (rather than token-level or trajectory-level) advantage estimation. Through a novel Monte Carlo method and tree-based sampling, SPO outperforms PPO and GRPO by 6–12 and 7–11 percentage points in short-CoT and long-CoT settings, respectively.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - credit assignment
  - segment-level advantages
  - Monte Carlo estimation
  - tree-based sampling
  - PPO improvement
date: 2026-05-08
content_hash: 48c082e47f9bf778
---

# Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.23564](https://arxiv.org/abs/2505.23564)
**Code**: [GitHub](https://github.com/AIFrameResearch/SPO)
**Area**: LLM Reasoning
**Keywords**: credit assignment, segment-level advantages, Monte Carlo estimation, tree-based sampling, PPO improvement

## TL;DR
This paper proposes the SPO framework, which adopts segment-level (rather than token-level or trajectory-level) advantage estimation. Through a novel Monte Carlo method and tree-based sampling, SPO outperforms PPO and GRPO by 6–12 and 7–11 percentage points in short-CoT and long-CoT settings, respectively.

## Background & Motivation
**Limitations of PPO**: Token-level advantage estimation requires an unstable critic model and yields inaccurate estimates.

**Limitations of GRPO**: Trajectory-level advantage is too coarse — all tokens share a single reward, lacking fine-grained feedback.

**Credit Assignment Challenge**: Sparse, delayed rewards in long sequences are difficult to assign precisely.

**Research Opportunity**: Finding a balance between token and trajectory granularity, leveraging MC estimation to eliminate the need for a critic.

## Method

### Overall Architecture
SPO adopts a three-layer architecture, each supporting multiple implementations:

1. **Flexible segment partitioning**
2. **Segment-level advantage estimation**
3. **Policy optimization based on segment advantages**

### Key Designs

**SPO-Chain (Short-CoT Setting)**:

1. **Adaptive cutpoint partitioning**:

    - Identifies low-probability tokens ($p_\theta(y_t|s_t) < \rho$) as cutpoints
    - Optimization problem: $\min \sum_k |U_\theta \cap [t_k, t_{k+1})|^2$
    - Solution: each segment contains an equal number of cutpoints

2. **Chain-based MC advantage estimation**:

    - Samples $N$ trajectories at each segment boundary
    - $\hat{V}(s_{t_k}) = \frac{1}{N}\sum_j R(x, [y_{<t_k}, \tau_{t_k}^{(j)}])$
    - $\hat{A}_k^{\text{seg}} = \hat{V}(s_{t_{k+1}}) - \hat{V}(s_{t_k})$

3. **Probability-masked optimization**:

    - Advantages are assigned only at low-probability tokens ($M_t = 1$ if $\pi(y_t) < \rho$)
    - $Z = \sum M_t$ serves as the normalization factor

**SPO-Tree (Long-CoT Setting)**:

1. **Fixed token-count partitioning**: One segment per $K$ tokens

2. **Tree-based MC advantage estimation**:

    - Constructs a tree structure with shared prefixes
    - Leaf nodes: $v = R(x, \text{hist})$
    - Internal nodes: $v = \frac{1}{|\text{Ch}|}\sum v(\text{child})$
    - Advantage: normalized difference relative to siblings

3. **Sample efficiency**:

    - Tree structure enables multi-level sampling reuse
    - Each node can serve as a training sample
    - Significantly reduces sampling cost

## Key Experimental Results

### SPO-Chain Performance on GSM8K

| Method | Accuracy | Relative Gain | Notes |
|--------|----------|---------------|-------|
| RestEM | 45.2% | — | baseline |
| DPO | 47.8% | +2.6% | vanilla DPO |
| PPO | 45.7% | baseline | token-level |
| GRPO | 44.6% | −1.1% | trajectory-level |
| VinePPO | 50.2% | +4.5% | prior work |
| **SPO-Chain** | **56.7%** | **+11.0%** | **best** |

### SPO-Tree Performance on MATH500 (Table 1)

| Context | Base | GRPO | SPO-Tree | Gain |
|---------|------|------|----------|------|
| **2K** | 56.6% | 62.0% | **73.6%** | **+11.6%** |
| **4K** | 74.0% | 75.2% | **82.8%** | **+7.6%** |
| **32K** | 83.8% | 84.0% | **84.8%** | +0.8% |

### Comparison with Competitive Methods (Table 1, lower)

| Method | MATH500-2K | MATH500-4K | AIME24-2K | AIME24-4K |
|--------|-----------|-----------|----------|----------|
| Base (DeepSeek-1.5B) | 56.6% | 74.0% | 6.7% | 16.7% |
| GRPO | 62.0% | 75.2% | 3.3% | 20.0% |
| **SPO-Tree** | **73.6%** | **82.8%** | **10.0%** | **20.0%** |
| DeepScaleR* | 53.8% | 74.4% | 0% | 16.7% |
| STILL-3* | 66.2% | 79.4% | 6.7% | 13.3% |

### Effect of Segment Granularity (Figure 4a)

| Interval | Wall-Clock Accuracy | Final Accuracy | Optimality |
|----------|--------------------|--------------|-----------| 
| interval=2 | 54.8% | **55.9%** | Fine-grained but inefficient |
| interval=5 | **55.6%** | 55.7% | **Optimal** |
| interval=100 | 45.2% | 42.1% | Too coarse |

### Tree Structure Comparison (Figure 5b)

| Tree Structure | GSM8K Accuracy | Convergence Speed | Memory |
|---------------|---------------|-----------------|--------|
| 4-4-4 | 55.0% | Fast | Low |
| **6-6-6** | **56.7%** | Medium | **Balanced** |
| 8-8-8 | 56.5% | Slow | High |

## Highlights & Insights
1. **Golden granularity for credit assignment**: Segment-level is finer than trajectory-level yet coarser than token-level, balancing accuracy and stability.
2. **Critic-free MC estimation**: Monte Carlo estimation eliminates the instability of critic models; sparse reward variance is bounded ($\leq 0.25$).
3. **Tree-based sampling innovation**: The hierarchical structure enables sample reuse, substantially improving efficiency.
4. **Substantial performance gains**: +11% on GSM8K, +11.6% on MATH500 (@2K), representing state-of-the-art results.

## Limitations & Future Work
1. **Hyperparameter sensitivity**: The number of segments $K$, MC sample count $N$, and probability threshold $\rho$ all require careful tuning.
2. **Long-sequence limitations**: Current evaluation covers at most 4K context; improvements diminish at 32K.
3. **Task scope**: Evaluation is primarily on mathematical reasoning; domains such as code generation remain untested.
4. **Computational cost**: Tree construction and MC sampling still incur overhead; absolute inference time is not analyzed in detail.

## Related Work & Insights
- **Credit assignment**: GAE, TD-$\lambda$, critic models
- **RL for LLM**: PPO, GRPO, VinePPO
- **Monte Carlo methods**: MCTS, AlphaGo
- **Reasoning models**: DeepSeek-R1, o1, OpenAI-o1

## Rating
⭐⭐⭐⭐⭐

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](stop_summation_minform_credit_assignment_is_all_process_rewa.md)
- [\[ICLR 2026\] Segment-Level Attribution for Selective Learning of Long Reasoning Traces](../../ICLR2026/llm_reasoning/segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[NeurIPS 2025\] DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization](disco_reinforcing_large_reasoning_models_with_discriminative_constrained_optimiz.md)
- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)

</div>

<!-- RELATED:END -->

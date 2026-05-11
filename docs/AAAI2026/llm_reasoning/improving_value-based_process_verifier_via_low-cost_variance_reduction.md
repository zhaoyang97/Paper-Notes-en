---
title: >-
  [Paper Note] Improving Value-based Process Verifier via Low-Cost Variance Reduction
description: >-
  [AAAI 2026][LLM Reasoning][Process verifier] To address the high-variance issue in value-based process reward model (PRM) training caused by limited Monte Carlo (MC) samples…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Process verifier"
  - "variance reduction"
  - "Monte Carlo sampling"
  - "mathematical reasoning"
  - "test-time scaling"
date: 2026-05-08
content_hash: 27faa95b0aa3e9a1
---

# Improving Value-based Process Verifier via Low-Cost Variance Reduction

**Conference**: AAAI 2026
**arXiv**: [2508.10539](https://arxiv.org/abs/2508.10539)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Process verifier, variance reduction, Monte Carlo sampling, mathematical reasoning, test-time scaling

## TL;DR
To address the high-variance issue in value-based process reward model (PRM) training caused by limited Monte Carlo (MC) samples, this paper proposes Compound Monte Carlo Sampling (ComMCS), which constructs an unbiased low-variance estimator by linearly combining MC estimates from the current step and subsequent steps. The method introduces no additional LLM inference overhead and achieves a 2.2-point improvement on MATH-500 under Best-of-32 evaluation.

## Background & Motivation

**Background**: Value-based process verifiers are trained by estimating the state-action value at each reasoning step — i.e., the probability that the step ultimately leads to a correct answer — and represent an effective approach for improving LLM mathematical reasoning. Training annotations rely on MC sampling: multiple reasoning trajectories are independently generated and their success rate is computed.

**Limitations of Prior Work**: The number of MC samples is constrained by LLM inference costs (typically only 8–16 samples), resulting in high variance in value estimates. High-variance training labels cause the verifier to learn noise rather than true values, degrading performance.

**Key Challenge**: Increasing the number of samples reduces variance but incurs linearly growing costs. Under a fixed sample budget, the MC estimator is already the minimum variance unbiased estimator (MVUE), and variance cannot be further reduced without introducing additional information.

**Goal**: Exploit the temporal structure of existing multi-step MC samples to reduce estimation variance without incurring additional LLM inference overhead.

**Key Insight**: Drawing inspiration from temporal difference (TD) methods in reinforcement learning — value estimates from future steps can be used to update estimates at the current step. MC estimates from subsequent steps within the same trajectory constitute freely available auxiliary information.

**Core Idea**: Construct a new unbiased estimator by linearly combining MC estimates from the current and subsequent steps, leveraging the recursive structure of the Bellman equation to reduce variance without additional sampling.

## Method

### Overall Architecture
Mathematical reasoning is modeled as an MDP → the statistical properties of MC estimates are analyzed (equivalence to binomial sampling, MVUE property) → ComMCS is proposed: an unbiased low-variance estimator is constructed by compounding multi-step MC results → a one-step value distribution is modeled to compute variance comparisons → a classification-based process verifier is trained with cross-entropy loss.

### Key Designs

1. **Statistical Analysis of MC Estimates**:

   - **Function**: Establish the equivalence between MC estimates and the binomial distribution, and prove the MVUE property.
   - **Mechanism**: In an MDP with binary rewards ($\{0,1\}$), the total number of successes from $N$ MC samples follows $B(N, V^\pi(s))$. The MC estimate $\hat{V}^\pi(s) = \frac{1}{N}\sum_i G^{(i)}$ is unbiased with variance $\frac{V^\pi(s)(1-V^\pi(s))}{N}$. As the MVUE, its variance cannot be further reduced given fixed information.
   - **Design Motivation**: This analysis clarifies that the performance bottleneck stems from variance rather than bias, and that "additional information" is necessary to break through the theoretical limit.

2. **Compound Monte Carlo Sampling (ComMCS)**:

   - **Function**: Construct a new unbiased low-variance estimator by linearly combining MC estimates from the current and subsequent steps.
   - **Mechanism**: By the Bellman equation $V^\pi(s_n) = \mathbb{E}_\pi[V^\pi(s_m | s_n)]$, the MC estimate at a subsequent step is also an unbiased estimator of the current step's value. Setting $\hat{V}_{n \to m} = \sum_i c_i \cdot \hat{V}^\pi(s_{n+i})$ (with $\sum c_i = 1$) preserves unbiasedness while potentially reducing variance. The compound variance formula is: $\mathbb{V}[\hat{V}_{n\to m}|s_n] = \sum_i c_i^2 (\frac{1}{N}\mathbb{E}[\sigma_i^2|s_n] + \mathbb{V}[V_i|s_n])$
   - **Design Motivation**: MC results from subsequent steps are already collected within the same trajectory; ComMCS leverages this free auxiliary information to reduce variance.

3. **One-Step Value Distribution Modeling**:

   - **Function**: Approximate the next-step value distribution with a categorical distribution to estimate variance and determine optimal combination coefficients.
   - **Mechanism**: In practice, $m = n+1$ (only the next step is used), simplifying the variance formula to two terms. The value distribution is assumed to belong to a Gaussian family; the verifier's softmax outputs are used to model a categorical distribution approximating the value distribution. MC estimate values serve as proxies for the ground truth to estimate variance, and optimal coefficients $c_n, c_{n+1}$ are found heuristically.
   - **Design Motivation**: Exact variance computation is infeasible in practice, but the categorical distribution combined with the Gaussian assumption provides a sufficiently practical approximation.

### Loss & Training
- The verifier is trained with cross-entropy loss; MC estimate values are binned to obtain categorical labels.
- Iterative optimization: train the verifier for one round with standard MC → use the verifier to estimate the value distribution → compute compound coefficients → retrain using ComMCS estimates.
- The choice of value distribution support set and binning strategy affects estimation accuracy.

## Key Experimental Results

### Main Results

Best-of-N sampling on MATH-500 (DeepSeek-Math-7B-Instruct):

| Method | Best-of-8 | Best-of-16 | Best-of-32 |
|--------|-----------|------------|------------|
| BCE (baseline) | 71.4 | 74.2 | 76.8 |
| MSE (regression) | 70.8 | 73.6 | 76.2 |
| ComMCS (ours) | **73.2** | **76.0** | **79.0** |

Consistent improvements are also observed on GSM8K.

### Ablation Study

| Configuration | MATH-500 BoN-32 | Note |
|---------------|-----------------|------|
| BCE baseline | 76.8 | Standard MC estimate training |
| + ComMCS | **79.0** | +2.2 pts, variance reduction effective |
| MSE regression | 76.2 | Regression-based optimization |
| + ComMCS | 79.0 | +2.8 pts, larger gain over regression baseline |

### Key Findings
- **Consistent improvement from ComMCS**: Effective under both Best-of-N and Beam Search strategies, across both DeepSeek and Qwen model families.
- **Equivalent to a 25% increase in sampling**: The variance of 8 samples + ComMCS is approximately equal to that of 10 samples (Figure 1).
- **Classification-based modeling outperforms regression**: Even without ComMCS, BCE (classification) outperforms MSE (regression), indicating that value distribution modeling itself is beneficial.
- **Variance reduction is most effective in the medium difficulty range**: MC variance is largest when the true value is near 0.5, where ComMCS also yields the greatest benefit.

## Highlights & Insights
- **Theory-driven practical method**: ComMCS is derived from the MVUE property, providing a rigorous theoretical foundation. The key insight is that the temporal structure of MC estimates contains freely available information.
- **Zero additional inference cost**: Unlike brute-force approaches that increase the number of samples, ComMCS exploits already-collected MC results from subsequent steps without any additional LLM calls.
- **Analogy to TD learning**: The idea of TD methods from RL is elegantly transferred to the PRM training setting, while maintaining unbiasedness — a property that TD methods do not possess.

## Limitations & Future Work
- **Gaussian distribution assumption may not hold**: The value distribution may in practice be multimodal or skewed.
- **Restricted to binary reward MDPs**: The theoretical derivation relies on the assumption that rewards $\in \{0, 1\}$, and does not directly extend to partial or graded reward settings.
- **Iterative training increases engineering complexity**: A base verifier must first be trained, followed by value distribution estimation and retraining.
- **Modest performance gains**: Improvements of 2–3 points are relatively limited in the context of highly competitive mathematical reasoning benchmarks.

## Related Work & Insights
- **vs. ORM (Outcome RM)**: ORM evaluates only the final outcome, while PRM performs step-wise evaluation; ComMCS is an internal optimization for PRM training and does not affect the framework choice.
- **vs. Math-Shepherd and similar PRMs**: These methods focus on better trajectory sampling, whereas ComMCS focuses on better utilization of existing samples — the two approaches are complementary.
- **vs. variance reduction in RL**: TD is biased but low-variance; ComMCS is unbiased and low-variance, representing a superior trade-off.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Analyzing the variance of PRM training annotations from the MVUE perspective is an original contribution; the theoretical basis of ComMCS is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on two benchmarks, two search strategies, and two model families, though the dataset scale and model size are relatively modest.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are rigorous and mathematical notation is clear, though the discussion of approximations in the practical implementation section lacks sufficient detail.
- **Value**: ⭐⭐⭐⭐ Offers a practical improvement to PRM training; the zero-additional-cost characteristic is particularly valuable from an engineering perspective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[ICLR 2026\] CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling](../../ICLR2026/llm_reasoning/cyclicreflex_improving_reasoning_models_via_cyclical_reflection_token_scheduling.md)
- [\[AAAI 2026\] SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger](sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)
- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](../../ICLR2026/llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)
- [\[ICCV 2025\] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Better Estimation of the Kullback-Leibler Divergence Between Language Models
description: >-
  [NeurIPS 2025][KL divergence estimation] This paper proposes a Rao-Blackwellized Monte Carlo estimator for KL divergence—computing the exact KL over the next-token distribution at each position (rather than relying solely on the sampled token). The estimator is theoretically proven to be unbiased with variance strictly no greater than the standard MC estimator, incurs zero additional computational overhead, and yields more stable training in an RLHF sentiment-control task, with models appearing on the Pareto frontier 78% of the time.
tags:
  - NeurIPS 2025
  - KL divergence estimation
  - Rao-Blackwellization
  - RLHF
  - variance reduction
  - language models
date: 2026-05-08
content_hash: ba2bedbc9e5a5029
---

# Better Estimation of the Kullback-Leibler Divergence Between Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2504.10637](https://arxiv.org/abs/2504.10637)
**Code**: [https://github.com/rycolab/kl-rb](https://github.com/rycolab/kl-rb)
**Area**: Interpretability
**Keywords**: KL divergence estimation, Rao-Blackwellization, RLHF, variance reduction, language models

## TL;DR
This paper proposes a Rao-Blackwellized Monte Carlo estimator for KL divergence—computing the exact KL over the next-token distribution at each position (rather than relying solely on the sampled token). The estimator is theoretically proven to be unbiased with variance strictly no greater than the standard MC estimator, incurs zero additional computational overhead, and yields more stable training in an RLHF sentiment-control task, with models appearing on the Pareto frontier 78% of the time.

## Background & Motivation

**State of the Field**: KL divergence is widely used in RLHF (as a regularization term), interpretability (as a measure of distributional shift), and knowledge distillation. Exact computation of KL divergence between language models is intractable due to the countably infinite string space $\Sigma^*$.

**Limitations of Prior Work**: (a) The standard MC estimator $\mu_{mc} = \frac{1}{M}\sum_m \log\frac{p(Y^{(m)})}{q(Y^{(m)})}$ suffers from high variance and can produce negative values; (b) the control variate method proposed by Schulman ($\alpha=1$) guarantees non-negativity but can exhibit exploding variance (confirmed experimentally); (c) unstable KL estimation during training leads to instability in RLHF.

**Root Cause**: The MC estimator computes log-ratios only over sampled complete strings, thereby discarding the full next-token distribution already produced by the forward pass at each position.

**Paper Goals**: Substantially reduce KL estimation variance at zero additional computational cost.

**Starting Point**: Rao-Blackwellization—computing the exact KL over the next-token distribution at each position $n$ (a summation over $|\bar{\Sigma}|$ tokens) rather than using only the sampled token.

**Core Idea**: $\mu_{rb} = \frac{1}{M}\sum_m \sum_{n=1}^{|Y^{(m)}|} KL(\vec{p}(\cdot|Y^{(m)}_{<n}) \| \vec{q}(\cdot|Y^{(m)}_{<n}))$.

## Method

### Overall Architecture
The standard MC estimator computes log-ratios at the string level. The RB estimator instead computes the exact KL over the full next-token distribution at each position, then sums over positions and averages over samples. Since the forward pass already produces the complete distribution at each position, the additional computation is only $O(MN|\bar{\Sigma}|)$ and is negligible in practice.

### Key Designs

1. **Rao-Blackwellized KL Estimator**:

    - **Function**: Computes the distribution-level KL exactly at each token position rather than relying on the sampled token alone.
    - **Mechanism**: Replaces each term $\log\frac{\vec{p}(Y_n|Y_{<n})}{\vec{q}(Y_n|Y_{<n})}$ in the MC estimator with $KL(\vec{p}(\cdot|Y_{<n}) \| \vec{q}(\cdot|Y_{<n}))$—summing over the entire vocabulary rather than conditioning on the sampled token.
    - **Design Motivation**: The Rao-Blackwell theorem guarantees that the variance of the conditional expectation does not exceed the original variance. Variance reduction arises because the estimator no longer depends on the particular token sampled; instead, it exploits the complete distributional information.

2. **Theoretical Guarantees (Theorem 2)**:

    - **Unbiasedness**: $\mathbb{E}[\mu_{rb}] = KL(p \| q)$
    - **Variance Reduction**: $Var[\mu_{rb}] \leq Var[\mu_{mc}]$
    - **Non-negativity**: Each term is an exact KL (hence non-negative), so the sum is non-negative—unlike MC, which can produce negative estimates.

3. **RB Gradient Estimator**:

    - **Function**: Extends Rao-Blackwellization to gradient estimation of the KL divergence for use in the RLHF training loop.
    - **Mechanism**: Theorem 4 derives a local decomposition of the KL gradient, which is then Rao-Blackwellized to obtain $\delta_{rb}$. Theorem 5 proves $\mathbb{E}[\|\delta_{rb} - G\|^2] \leq \mathbb{E}[\|\delta_{mc} - G\|^2]$.
    - **Design Motivation**: Gradient variance in the RLHF loop directly affects training stability.

### Loss & Training
- RLHF objective: expected reward $- \beta \cdot KL(p_\theta \| q)$
- The RB estimator replaces the MC estimator for computing both the KL term and its gradient.

## Key Experimental Results

### KL Estimation Quality (GPT-2 Sentiment Control)

| Estimator | M=1 Mean±Std | M=5 Mean±Std | M=10 Mean±Std |
|-----------|-------------|-------------|--------------|
| $\mu_{mc}$ | 6.76±0.16 | 6.76±0.07 | 6.76±0.05 |
| $\mu_{cv}$ ($\alpha=1$) | 6.28±2.54 | 6.28±1.13 | 6.28±0.79 |
| $\mu_{cv}$ (optimal $\alpha$) | 6.76±0.16 | 6.76±0.07 | 6.76±0.05 |
| **$\mu_{rb}$** | **6.76±0.11** | **6.76±0.05** | **6.76±0.03** |

The RB estimator achieves the lowest standard deviation across all sample sizes. Schulman's CV ($\alpha=1$) is biased and exhibits exploding variance.

### RLHF Training Stability

| Estimator | Reward Stability | KL Stability | Pareto Frontier Fraction |
|-----------|----------------|-------------|------------------------|
| MC | High variance (large variation across 5 runs) | High variance | 22% |
| **RB** | **Stable** | **Stable** | **78%** |

In the region KL < 5, RB models occupy **95%** of the Pareto frontier.

### Gradient Variance Experiment

| Estimator | Gradient Norm Variance |
|-----------|----------------------|
| MC | 59.90 |
| **RB** | **45.44 (−24.6%)** |

### Key Findings
- **RB is the only method achieving meaningful variance reduction**: The CV approach fails to reduce variance—and often increases it—on the majority of prompts.
- **Schulman's $\alpha=1$ estimator exhibits serious pathologies**: $Var[g(Y)]$ is unbounded for certain prompts, resulting in bias and extremely large variance.
- **RB substantially stabilizes RLHF training**: Reward and KL curves across 5 repeated runs nearly coincide (MC exhibits large variation).
- **Zero additional computational overhead**: The full next-token distribution is already produced by the forward pass.

## Highlights & Insights
- A profound insight of **"exploiting information already at hand"**: the forward pass produces the complete next-token distribution at every position, yet the MC estimator uses only a single sampled token, whereas RB utilizes the full distribution.
- **Non-negativity as a free bonus**: MC estimates can be negative (destabilizing RLHF), and Schulman's method trades variance for non-negativity. RB achieves both non-negativity and lower variance simultaneously.
- The improvement is directly applicable to open-source RLHF libraries (trl, OpenRLHF, etc.)—code snippets are provided.

## Limitations & Future Work
- RLHF experiments are validated only on GPT-2 (computational constraints necessitated training 36 models for significance testing).
- Exact KL computation still incurs overhead when the vocabulary size $|\bar{\Sigma}|$ is large, though this is far smaller than the cost of the forward pass.
- The analysis assumes $KL(p \| q) < \infty$, which may not hold in practice.

## Related Work & Insights
- **vs. Schulman (2020) control variate**: $\alpha=1$ guarantees non-negativity but variance can be unbounded. RB achieves both non-negativity and low variance simultaneously.
- **vs. Horvitz-Thompson**: Another unbiased estimator, but without variance improvement.
- **vs. Analytic computation**: Tractable only for special model classes (e.g., PFSAs). RB applies to arbitrary neural language models.

## Rating
- Novelty: ⭐⭐⭐⭐ Rao-Blackwellization is a classical technique, but its application to KL estimation for language models is novel and consequential.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers estimation quality, RLHF training, Pareto analysis, and gradient variance.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are exceptionally clear, with an elegant theorem-proof structure.
- Value: ⭐⭐⭐⭐⭐ Offers a directly deployable improvement for RLHF practice.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)
- [\[NeurIPS 2025\] ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation](arecho_autoregressive_evaluation_via_chain-based_hypothesis_optimization_for_spe.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)

<!-- RELATED:END -->

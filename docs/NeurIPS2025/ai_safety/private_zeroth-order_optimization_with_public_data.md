---
title: >-
  [Paper Note] Private Zeroth-Order Optimization with Public Data
description: >-
  [NeurIPS 2025][AI Safety][Differential Privacy] This paper proposes the PAZO framework, which leverages public data to guide gradient approximation in private zeroth-order optimization. PAZO achieves a superior privacy-utility tradeoff compared to DP-SGD on both vision and text tasks, while delivering up to a 16× speedup.
tags:
  - NeurIPS 2025
  - AI Safety
  - Differential Privacy
  - Zeroth-Order Optimization
  - Public Data
  - DP-SGD
  - Privacy-Utility Tradeoff
date: 2026-05-08
content_hash: f342de936f84b4ed
---

# Private Zeroth-Order Optimization with Public Data

**Conference**: NeurIPS 2025

**arXiv**: [2511.10859](https://arxiv.org/abs/2511.10859)

**Code**: None

**Area**: AI Security / Differential Privacy

**Keywords**: Differential Privacy, Zeroth-Order Optimization, Public Data, DP-SGD, Privacy-Utility Tradeoff

## TL;DR

This paper proposes the PAZO framework, which leverages public data to guide gradient approximation in private zeroth-order optimization. PAZO achieves a superior privacy-utility tradeoff compared to DP-SGD on both vision and text tasks, while delivering up to a 16× speedup.

## Background & Motivation

### Limitations of Prior Work

**State of the Field**: Differentially private (DP) machine learning algorithms such as DP-SGD are the standard approach for protecting training data privacy, but suffer from severe computational bottlenecks:

**High computational and memory overhead**: DP-SGD requires computing per-sample gradients, followed by clipping and noise addition.

**Potential of zeroth-order methods**: Zeroth-order (ZO) methods approximate gradients via function value queries, making them naturally amenable to privatization, yet they typically yield lower utility.

**Underutilization of public data**: Public data (e.g., pre-training corpora) is often available in practice, but existing ZO methods do not exploit this resource.

**Core Problem**: Can public data be leveraged to improve the utility of private zeroth-order optimization while preserving its computational efficiency advantages?

## Method

### Overall Architecture

The PAZO (Public-data-Assisted Zeroth-Order) framework incorporates public-data guidance into the standard zeroth-order optimization pipeline to improve the quality of gradient approximation.

### Key Designs

**1. Public-Data-Guided Direction Selection**

- Standard ZO methods probe along random directions, resulting in high variance.
- PAZO constructs an **importance sampling distribution** using gradient information derived from public data.
- Probing is concentrated along directions indicated by public-data gradients.
- Core formulation: $\hat{g} = \sum_{i=1}^{q} \frac{f(x + \mu u_i) - f(x)}{\mu} u_i$, where the sampling of $u_i$ is guided by public-data gradients.

**2. PAZO Variants**

- **PAZO-Subspace**: Performs zeroth-order optimization within the subspace spanned by public-data gradients.
- **PAZO-Projection**: Projects zeroth-order estimates onto the signal directions identified by public data.
- **PAZO-Precondition**: Preconditions zeroth-order gradients using the Fisher information matrix estimated from public data.

**3. Privacy Analysis**

- Only function value queries involve private data; all public-data operations consume no privacy budget.
- Standard privacy accounting tools for zeroth-order DP methods are directly applicable.

### Loss & Training

- **Privacy budget**: Rényi DP or zero-concentrated DP is used for tight privacy accounting.
- **Noise calibration**: Gaussian noise standard deviation is determined based on sensitivity and the target $\varepsilon$.
- **Learning rate schedule**: Cosine annealing, consistent with non-private training.

## Key Experimental Results

### Main Results

Vision task (CIFAR-10 fine-tuning, $\varepsilon=3$):

| Method | Accuracy | Training Time | Peak Memory |
|--------|----------|--------------|-------------|
| DP-SGD | 82.1% | 48 min | 12.3 GB |
| DP-SGD + Public | 84.5% | 52 min | 13.1 GB |
| Zero-order DP | 76.3% | 8 min | 3.2 GB |
| PAZO (Ours) | **85.2%** | **3 min** | **2.8 GB** |

Text task (SST-2 fine-tuning, $\varepsilon=8$):

| Method | Accuracy | Training Time | Speedup |
|--------|----------|--------------|---------|
| DP-SGD | 90.1% | 120 min | 1× |
| DP-SGD + Public | 91.3% | 135 min | 0.9× |
| Zero-order DP | 85.7% | 12 min | 10× |
| PAZO (Ours) | **91.8%** | **7.5 min** | **16×** |

### Ablation Study

Performance under varying privacy budgets (CIFAR-10):

| $\varepsilon$ | DP-SGD | DP-SGD+Public | ZO-DP | PAZO |
|---------------|--------|--------------|-------|------|
| 1 | 71.2% | 75.8% | 62.3% | **77.5%** |
| 3 | 82.1% | 84.5% | 76.3% | **85.2%** |
| 8 | 88.5% | 89.2% | 83.1% | **89.8%** |
| ∞ (no privacy) | 94.2% | 94.2% | 91.5% | 93.8% |

### Key Findings

1. PAZO achieves the most pronounced advantage under high privacy protection (low $\varepsilon$), outperforming even first-order methods augmented with public data.
2. Runtime speedup reaches 16×, primarily due to the elimination of per-sample gradient computation.
3. Greater similarity between public and private data yields larger improvements; however, significant gains are observed even with limited similarity.
4. PAZO-Subspace is optimal in low-dimensional settings, while PAZO-Precondition is optimal in high-dimensional settings.

## Highlights & Insights

- **Surpassing the first-order ceiling**: This is the first demonstration in private learning where a zeroth-order method outperforms a first-order method that also uses public data.
- **High practical value**: A 16× speedup combined with low memory usage makes privacy-preserving training on edge devices feasible.
- **Theoretical grounding**: Convergence analysis is provided under assumptions on the similarity between public and private data distributions.

## Limitations & Future Work

1. The method requires a degree of similarity between public and private data; performance degrades when cross-domain discrepancy is large.
2. Theoretical analysis relies on convexity or the PL condition, offering limited guarantees for non-convex deep learning models.
3. The framework is currently validated in fine-tuning settings; its effectiveness for pre-training from scratch remains unverified.
4. A systematic investigation of how public data quality and quantity affect final performance is lacking.

## Related Work & Insights

- **DP-SGD** (Abadi et al.): The standard approach for differentially private stochastic gradient descent.
- **MeZO** (Malladi et al.): A zeroth-order optimization framework for large language models.
- **Public-data-assisted DP**: First-order methods proposed by Yu et al. and De et al.

## Rating

- ⭐ Novelty: 8/10 — The idea of guiding zeroth-order methods with public data is both elegant and effective.
- ⭐ Value: 9/10 — Dual advantages in speed and privacy-utility tradeoff yield high practical deployment value.
- ⭐ Writing Quality: 8/10 — Comprehensive comparison across multiple variants with well-designed experiments.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization](stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[NeurIPS 2025\] Private Continual Counting of Unbounded Streams](private_continual_counting_of_unbounded_streams.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)

<!-- RELATED:END -->

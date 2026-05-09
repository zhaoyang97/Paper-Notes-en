---
title: >-
  [Paper Note] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery
description: >-
  [ICLR 2026][Causal Inference][conditional independence test] This paper proposes E-CIT (Ensemble Conditional Independence Test), a framework that partitions data into subsets, performs independent tests on each subset, and aggregates the resulting p-values via a **stable distribution**-based combination method. E-CIT reduces the computational complexity of any base CIT to linear in sample size, while maintaining or improving test power in challenging settings such as heavy-tailed noise and real-world data.
tags:
  - ICLR 2026
  - Causal Inference
  - conditional independence test
  - causal discovery
  - ensemble method
  - stable distribution
  - p-value combination
date: 2026-05-08
content_hash: 85003f1de1e73555
---

# Efficient Ensemble Conditional Independence Test Framework for Causal Discovery

**Conference**: ICLR 2026
**arXiv**: [2509.21021](https://arxiv.org/abs/2509.21021)
**Code**: None
**Area**: Causal Inference
**Keywords**: conditional independence test, causal discovery, ensemble method, stable distribution, p-value combination

## TL;DR

This paper proposes E-CIT (Ensemble Conditional Independence Test), a framework that partitions data into subsets, performs independent tests on each subset, and aggregates the resulting p-values via a **stable distribution**-based combination method. E-CIT reduces the computational complexity of any base CIT to linear in sample size, while maintaining or improving test power in challenging settings such as heavy-tailed noise and real-world data.

## Background & Motivation

**State of the Field**: Constraint-based causal discovery methods (e.g., the PC algorithm) rely on numerous conditional independence tests (CITs) to determine causal graph structure. KCIT (kernel-based CIT) is among the most popular approaches, but incurs $O(n^3)$ time complexity with respect to sample size.

**Limitations of Prior Work**:
- The high computational cost of individual CITs is the primary bottleneck in causal discovery, not the number of tests performed.
- Existing acceleration methods (RCIT, FastKCIT) are tailored specifically to KCIT and do not constitute general-purpose frameworks.
- Shah & Peters (2018) proved that no single CIT is uniformly powerful across all conditional dependence structures — making a general acceleration framework more valuable than improving any single method.

**Root Cause**: Large samples are necessary to ensure test power, yet the high complexity of CITs renders large-sample computation infeasible.

**Paper Goals**: Design a general, plug-and-play framework applicable to any CIT method to reduce computational overhead while preserving statistical power.

**Starting Point**: Drawing inspiration from ensemble learning — data are partitioned into fixed-size subsets, tests are performed independently, and the resulting p-values are aggregated. The key innovation lies in the aggregation step: the closure property of stable distributions is exploited to design a p-value combination method with guaranteed consistency.

**Core Idea**: Divide-and-conquer + stable distribution p-value aggregation = a linear-complexity acceleration framework for arbitrary CITs.

## Method

### Overall Architecture

E-CIT follows a three-step pipeline (Figure 1):
1. **Divide**: Partition $n$ samples uniformly into $K$ subsets of fixed size $n_k$, where $K = n / n_k$.
2. **Test**: Apply the base CIT independently to each subset, yielding $K$ p-values $\{p_1, \ldots, p_K\}$.
3. **Aggregate**: Combine the $K$ p-values into a final p-value using a stable distribution-based method.

With $n_k$ fixed, the total complexity of the base CIT becomes $K \times O(f(n_k)) = O(n)$, achieving linearization compared to the original $O(f(n))$.

### Key Designs

1. **Stable Distribution-Based P-value Aggregation (Definition 2)**:

    - **Function**: Combines the p-values from $K$ sub-tests into a single final p-value with guaranteed statistical properties.
    - **Mechanism**: Exploits the closure property of stable distributions — if $X_j \sim \mathbf{S}(\alpha, \beta, \gamma, \delta)$ are i.i.d., then $\frac{1}{K}\sum X_j \sim \mathbf{S}(\alpha, \beta, K^{1/\alpha - 1}\gamma, \delta)$. The test statistic is defined as:
    $$T_e = \frac{1}{K} \sum_{k=1}^K F_S^{-1}(p_k)$$
    The final p-value is $p_e = F_{S'}(T_e)$, where $S' = \mathbf{S}(\alpha, \beta, K^{1/\alpha-1}\gamma, \delta)$.
    - **Design Motivation**: The parameter $\alpha$ controls tail heaviness; $\alpha = 2$ recovers the Stouffer method (Gaussian), and $\alpha = 1$ corresponds to the Cauchy combination. Tuning $\alpha$ allows adaptation to different CITs and data characteristics.

2. **Theoretical Guarantees (Theorem 1 & 2)**:

    - **Function**: Establishes validity, admissibility, unbiasedness, and consistency of the ensemble test.
    - **Mechanism**:
        - **Validity**: Under the null hypothesis, $p_e$ is uniformly distributed on $[0,1]$ (for exact p-values).
        - **Consistency** (Theorem 2): Power approaches 1 as $K \to \infty$, requiring only: ① the expected p-value of sub-tests is $\le \alpha_e$; ② the p-value density on $[0, 1/2]$ is no less than its mirror value; ③ stable distribution parameters satisfy $\alpha \ge 1, \beta = \delta = 0$.
    - **Design Motivation**: The consistency conditions impose **no assumptions on the data-generating process**, requiring only that sub-tests be reasonably valid. This enables E-CIT to provide consistency guarantees even in complex settings where the base CIT itself lacks them.

3. **Flexibility via the $\alpha$ Parameter**:

    - **Function**: Controls the degree of flexibility in p-value aggregation.
    - **Mechanism**: By the Neyman–Pearson lemma, the optimal combination statistic is a monotone transformation of $-\sum \log f_1(p_k)$. Since different CITs yield different alternative p-value distributions under different dependence structures, $\alpha$ enables adaptive tuning.
    - **Design Motivation**: Classical methods (Fisher, Stouffer) correspond to fixed values of $\alpha$ and lack flexibility; E-CIT provides a parsimonious one-dimensional control via $\alpha$.

### Loss & Training

- E-CIT is an unsupervised method requiring no training.
- Practical recommendations: $n_k = 400$ (empirically determined), $\alpha \in \{1.75, 2.0\}$, $\beta = \delta = 0$, $\gamma = 1$.

## Key Experimental Results

### Main Results

Data generation follows a post-nonlinear model, with $Z$ drawn from normal or Laplace distributions, and noise from Student-t, Cauchy, or Laplace distributions.

**Computational Efficiency (Figure 2, KCIT acceleration)**:

| Method | Time Complexity | Runtime at n=2000 | Type I Error | Power |
|--------|----------------|-------------------|-------------|-------|
| KCIT (original) | $O(n^3)$ | ~100s | ~0.05 | baseline |
| RCIT | $O(n)$ | ~0.1s | ~0.05 | slightly below KCIT |
| FastKCIT | $O(n \log n)$ | ~1s | ~0.05 | close to KCIT |
| **E-KCIT** | $O(n)$ | ~0.1s | ~0.05 | **matches or exceeds KCIT (better under heavy tails)** |

**Cross-Method Generality (Table 2, n=1200, Normal Z, t-noise df=2)**:

| Method | Orig. Power | Ensemble Power (α=1.75) |
|--------|------------|------------------------|
| RCIT | 0.548 | **0.623** |
| LPCIT | 0.422 | **0.447** |
| CMIknn | 0.982 | 0.988 |
| FisherZ | 0.510 | **0.561** |
| CCIT | 0.904 (Type I=0.454!) | 0.816 (Type I=0.286↓) |

### Ablation Study

**Real Data: Flow-Cytometry (Table 3)**:

| Method | Orig. F1 | Ensemble F1 |
|--------|---------|------------|
| KCIT | 0.624 | **0.695** |
| RCIT | 0.665 | **0.687** |
| LPCIT | 0.691 | **0.741** |
| CMIknn | 0.779 | 0.756 |
| FisherZ | 0.737 | **0.767** |

### Key Findings

1. **Significant speedup**: E-KCIT reduces KCIT's $O(n^3)$ complexity to $O(n)$, achieving runtime comparable to RCIT.
2. **Power gains, not losses**: Under heavy-tailed noise (Student-t df=2, Cauchy), E-KCIT surpasses both KCIT and RCIT in power — subset-level estimation is more stable.
3. **Generality**: Effective across 6 distinct CIT methods (KCIT, RCIT, LPCIT, CMIknn, CCIT, FisherZ).
4. **Real-data advantage**: On the Flow-Cytometry dataset, E-CIT improves F1-score by 2–5 percentage points for most methods.
5. **Unexpected finding for CCIT**: E-CIT substantially reduces CCIT's inflated Type I error (from 0.45+ to 0.28–0.34), at the cost of a modest reduction in power, yielding better-calibrated tests.
6. **Causal discovery application (Figure 3)**: On nonlinear additive noise causal graphs, E-KCIT outperforms both KCIT and RCIT in F1 and SHD.

## Highlights & Insights

1. **A general framework, not a specific method**: E-CIT functions as an **accelerator** rather than a new CIT — it can be plugged into any existing method.
2. **Extremely mild theoretical consistency conditions**: No assumptions are placed on the data or model; only reasonable validity of sub-tests is required.
3. **Elegant application of stable distributions**: The closure property of stable distributions enables exact p-value aggregation, with $\alpha$ providing flexible control.
4. **Practical insight**: In complex settings (heavy tails, real data), ensemble aggregation can improve power — small-sample estimates are more stable, and aggregation compensates for individual weaknesses.

## Limitations & Future Work

1. The theoretical analysis assumes sub-test p-values are i.i.d. — correlation may arise in practice (e.g., time-series data or distributional shift).
2. Optimal selection of $\alpha$ is context-dependent; only empirical recommendations ($\{1.75, 2.0\}$) are currently provided.
3. Subset size $n_k$ must be sufficiently large for valid sub-tests — the curse of dimensionality may persist for very high-dimensional conditioning sets $Z$.
4. Methods already exhibiting strong performance (e.g., CMIknn) benefit less, suggesting the framework is most effective for accelerating moderately powerful tests.
5. Future directions include handling correlated p-values, optimizing $\alpha$ for specific CITs, and developing adaptive subset size selection.

## Related Work & Insights

- **RCIT (Strobl et al., 2019)**: Accelerates KCIT via random Fourier features — applicable only to KCIT, whereas E-CIT is a general framework.
- **FastKCIT (Schacht & Huang, 2025)**: Partitions data using GMMs — conceptually similar but designed exclusively for KCIT.
- **Cauchy combination method (Liu & Xie, 2020)**: Combines p-values via the Cauchy distribution for whole-genome sequencing tests — E-CIT generalizes this to arbitrary stable distributions in the CIT context.
- **Insight**: Divide-and-conquer with aggregation is a general paradigm for large-scale statistical testing, with potential applicability to other computationally demanding testing problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The closure property of stable distributions is creatively applied to p-value aggregation for CIT, yielding a clear and practical framework; however, the divide-and-aggregate paradigm itself is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Synthetic data + real data + causal discovery application; 6 CIT methods × multiple noise distributions × multiple sample sizes; comprehensive ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with good integration of theory and experiments; Figure 1 provides an intuitive overview; proofs are deferred to the appendix, keeping the main text accessible.
- **Value**: ⭐⭐⭐⭐ Highly practical — directly integrable into existing causal discovery pipelines; meaningful for large-scale causal discovery applications.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](../../NeurIPS2025/causal_inference/differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](../../NeurIPS2025/causal_inference/gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[CVPR 2026\] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](../../CVPR2026/causal_inference/maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed
description: >-
  [ICML 2025][Optimization][Gradient Clipping] This paper proves that the high-probability convergence of AdaGrad/Adam under heavy-tailed noise can be poor (with polynomial dependence on the confidence level) and demonstrates that gradient clipping resolves this issue—specifically, Clip-AdaGrad-Norm and Clip-Adam-Norm achieve high-probability convergence bounds with polylogarithmic dependence on the confidence level under heavy-tailed noise, which are then extended to delayed s…
tags:
  - "ICML 2025"
  - "Optimization"
  - "Gradient Clipping"
  - "Adam"
  - "AdaGrad"
  - "Heavy-Tailed Noise"
  - "High-Probability Convergence"
date: 2026-05-08
content_hash: 02bb16e04720dda4
---

# Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed

**Conference**: ICML 2025  
**arXiv**: [2406.04443](https://arxiv.org/abs/2406.04443)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Gradient Clipping, Adam, AdaGrad, Heavy-Tailed Noise, High-Probability Convergence

## TL;DR
This paper proves that the high-probability convergence of AdaGrad/Adam under heavy-tailed noise can be poor (with polynomial dependence on the confidence level) and demonstrates that gradient clipping resolves this issue—specifically, Clip-AdaGrad-Norm and Clip-Adam-Norm achieve high-probability convergence bounds with polylogarithmic dependence on the confidence level under heavy-tailed noise, which are then extended to delayed stepsizes versions.

## Background & Motivation

**Background**: Adaptive stepsize methods (AdaGrad, Adam) are core optimizers in deep learning, particularly for LLM training. Gradient clipping is also widely used, especially in BERT/GPT training.

**Limitations of Prior Work**:
   - Zhang et al. (2020) found that gradient noise in BERT pre-training displays a heavy-tailed distribution ($\alpha$-th moment is bounded, $\alpha \in (1,2]$), under which SGD may diverge.
   - Clip-SGD has provable convergence under heavy-tailed noise, and Adam behaves similarly in practice—yet theoretical guarantees for the high-probability convergence of Adam are missing.
   - Some authors claim that "Adam inherently incorporates clipping effects" (since dividing the adaptive stepsize by the gradient norm resembles clipping)—but this remains a conjecture rather than a proof.
   - In practice, Adam and clipping are often used together (e.g., in BERT fine-tuning), but there is a lack of theoretical guidance on whether clipping is truly necessary.

**Key Challenge**: While Adam and Clip-SGD appear similar, does Adam really not need extra clipping?

**Goal**: To rigorously answer whether AdaGrad/Adam requires gradient clipping under heavy-tailed noise.

**Key Insight**: Construct counterexamples to show that the high-probability convergence of AdaGrad/Adam under heavy-tailed noise is indeed poor (negative result), and then prove that incorporating clipping significantly improves convergence (positive result).

**Core Idea**: The "inherent clipping" in Adam is insufficient—while adaptive stepsizes scale the gradient, they do not truncate extreme values. Explicitly truncating extreme values via gradient clipping enables the tail probability of convergence to decay logarithmically rather than polynomially.

## Method

### Overall Architecture
Two-fold results:
1. **Negative Result**: Constructing problem instances to prove that the high-probability convergence bounds of AdaGrad-Norm and Adam-Norm under heavy-tailed noise have at least a polynomial dependence $O(1/\delta^p)$ on the confidence level $\delta$.
2. **Positive Result**: Proving that Clip-AdaGrad-Norm and Clip-Adam-Norm achieve a poly-logarithmic dependence $O(\log^q(1/\delta))$ on the confidence level.

### Key Designs

1. **Negative Result: AdaGrad/Adam Can Be Poor**:

    - Function: Prove that unclipped AdaGrad/Adam has suboptimal high-probability convergence under heavy-tailed noise.
    - Mechanism:
        - Construct a 1D convex optimization problem where the noise distribution is an $\alpha$-stable distribution (with $\mathbb{E}[|\xi|^\alpha] < \infty$ but $\mathbb{E}[|\xi|^2] = \infty$).
        - On this problem, the high-probability error bound of AdaGrad-Norm is $O(1/\delta^{2/\alpha - 1})$—which degrades as $\alpha \to 1$ (heavier-tailed noise).
        - Key Insight: Although the denominator of AdaGrad $\sqrt{\sum g_i^2}$ scales large gradients, it does not truncate them. When a single extreme noise makes $g_t$ very large, the update step size from $g_t / \sqrt{\sum_{i \leq t} g_i^2}$ may still remain excessively large.
    - Design Motivation: Refute the popular belief that "Adam ≈ implicit clipping"—scaling is not equivalent to truncation.

2. **Positive Result: Clipping Fixes the Problem**:

    - Function: Prove that Clip-AdaGrad-Norm and Clip-Adam-Norm achieve polylogarithmic high-probability convergence.
    - Mechanism:
        - The clipping operation is formulated as $\text{clip}(g_t, c_t) = g_t \cdot \min(1, c_t/\|g_t\|)$, where $c_t$ is a time-varying clipping threshold.
        - With bounded clipped gradients, the analysis of adaptive stepsizes becomes highly tractable.
        - For convex problems: $\mathbb{E}[f(\bar{x}_T) - f(x^*)] \leq O(T^{-1/2} \cdot \log^q(T/\delta))$
        - For non-convex problems: $\min_t \|\nabla f(x_t)\|^2 \leq O(T^{-1/4} \cdot \log^q(T/\delta))$
    - Novelty: Address momentum and bias correction specific to Adam—not simply transferring results directly from Clip-SGD.
    - Extension to delayed stepsize versions: Prove that delayed versions, where stepsizes have delays in practical distributed training, enjoy similar guarantees.

3. **Delayed Stepsize Analysis**:

    - Function: Handle the misalignment between gradient computation and stepsize updates in distributed training.
    - Mechanism: A delay of $d$ steps implies that the stepsize used is based on the gradient history from $d$ steps ago—requiring additional handling for this lag.
    - Results: The delay introduces an extra error term of $O(d^2)$, but does not affect the polylogarithmic dependence on $\delta$.

### Loss & Training
- Mainly theoretical analysis.
- Convex + non-convex settings.
- Heavy-tailed noise assumption: $\mathbb{E}[\|g_t - \nabla f(x_t)\|^\alpha] \leq \sigma^\alpha$, $\alpha \in (1, 2]$.
- Selection of clipping threshold: $c_t = O(\sigma \cdot t^{1/(2\alpha)})$ (theoretically guided time-varying threshold).

## Key Experimental Results

### Synthetic Problem Verification
1D convex problem, $\alpha$-stable noise:

| Method | High-probability Convergence ($\delta = 0.01$) | Heavy-tailedness $\alpha=1.5$ |
|------|----------------------|---------------------|
| SGD | Diverges | Confirms theory |
| AdaGrad-Norm | Slow convergence | Confirms poly dependence |
| Clip-SGD | Converges | polylog dependence |
| **Clip-AdaGrad-Norm** | **Fast convergence** | **polylog dependence** |

### BERT Fine-Tuning Experiments

| Optimizer | Final Performance (Val Acc) | Training Stability |
|--------|----------------|----------|
| AdamW (No clipping) | 89.2% | Unstable (occasional spikes) |
| AdamW + gradient norm clip | 89.8% | Stable |
| **Clip-Adam-Norm** | **90.1%** | **Most stable** |

### Ablation Study

| Configuration | Type of High-probability Error Bound |
|------|-------------|
| AdaGrad (No clipping) | $O(1/\delta^{p})$ polynomial—Poor |
| Clipping threshold too large | Close to no clipping—Poor |
| Clipping threshold too small | Large bias—Poor |
| **Theoretically guided clipping threshold** | **$O(\log^q(1/\delta))$ polylog—Good** |

### Key Findings
- Adam does not "inherently clip" gradients—explicit gradient clipping is indeed necessary under heavy-tailed noise.
- Clipping improves high-probability convergence from polynomial dependency to polylogarithmic dependency—a qualitative leap.
- Empirical evidence in BERT fine-tuning further validates the theory; clipping is not only theoretically required but also practically beneficial.
- The analysis on delayed stepsizes makes the theory applicable to actual distributed training scenarios.
- The choice of clipping threshold is crucial—it should be set according to the order of the noise moment $\alpha$.

## Highlights & Insights
- **"Scaling ≠ Truncation"**—a simple yet profound distinction that clarifies the fundamental difference between Adam and Clip-SGD.
- The proof of the negative result (AdaGrad/Adam can perform poorly) is a major contribution itself—dispelling speculative risks of using Adam without clipping.
- Theoretical results directly explain why BERT/GPT training always employs gradient clipping in practice—it is a mathematical necessity rather than a heuristic.
- The extension to delayed stepsizes provides practical guidance for distributed LLM training.
- Directly impacts optimizer designers and LLM training engineers.

## Limitations & Future Work
- The negative result is based on specifically constructed problem instances—the empirical performance of Adam on "typical" DL problems may not be as poor.
- The theoretically optimal choice of clipping threshold requires knowledge of $\alpha$—which is unknown in practice.
- Only Adam-Norm (one-dimensional stepsize) is analyzed; coordinate-wise Adam remains to be analyzed.
- Whether the heavy-tail noise assumption holds across all DL problems is uncertain.

## Related Work & Insights
- **vs Zhang et al. (2020)**: Identified the heavy-tailed noise phenomenon but only analyzed Clip-SGD; this work extends the coverage to Adam/AdaGrad.
- **vs Faw et al. (2023)**: Analyzed Adam's convergence under generalized smoothness, without addressing heavy-tailed noise.
- **vs Gorbunov et al. (2020)**: Analyzed high-probability convergence for Clip-SGD; this work combines clipping with adaptive stepsizes.
- **Insight**: Heavy-tailed noise may be the theoretical root cause of why "Adam + clipping is standard practice in LLM training".

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to rigorously prove Adam needs clipping + provide tight bounds after clipping.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic + BERT validation, though the experimental scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Well-defined problems, parallel presentation of positive and negative results.
- Value: ⭐⭐⭐⭐⭐ Direct theoretical guidance for LLM training practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](../../ICML2026/optimization/can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICLR 2026\] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis](../../ICLR2026/optimization/clipped_gradient_methods_for_nonsmooth_convex_optimization_under_heavy-tailed_no.md)
- [\[ICLR 2026\] Solving the 2-norm k-hyperplane clustering problem via multi-norm formulations](../../ICLR2026/optimization/solving_the_2-norm_k-hyperplane_clustering_problem_via_multi-norm_formulations.md)
- [\[ICLR 2026\] Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence](../../ICLR2026/optimization/decentralized_nonconvex_optimization_under_heavy-tailed_noise_normalization_and_.md)

</div>

<!-- RELATED:END -->

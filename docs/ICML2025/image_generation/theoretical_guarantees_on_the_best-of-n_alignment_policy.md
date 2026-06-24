---
title: >-
  [Paper Note] Theoretical Guarantees on the Best-of-n Alignment Policy
description: >-
  [ICML2025][Image Generation][best-of-n sampling] This paper refutes the claim of exactness for the widely used best-of-n policy KL divergence formula $\log(n) - (n-1)/n$ in the literature, proving it to be only an upper bound, and proposes tighter KL divergence estimators and theoretical bounds for the win rate.
tags:
  - "ICML2025"
  - "Image Generation"
  - "best-of-n sampling"
  - "KL divergence"
  - "alignment policy"
  - "inference-time compute"
  - "win rate"
  - "theoretical guarantee"
date: 2026-05-08
content_hash: 5105e99b1663b8fa
---

# Theoretical Guarantees on the Best-of-n Alignment Policy

**Conference**: ICML2025  
**arXiv**: [2401.01879](https://arxiv.org/abs/2401.01879)  
**Code**: None (Pure theoretical work)  
**Area**: Image Generation  
**Keywords**: best-of-n sampling, KL divergence, alignment policy, inference-time compute, win rate, theoretical guarantee

## TL;DR

This paper refutes the claim of exactness for the widely used best-of-n policy KL divergence formula $\log(n) - (n-1)/n$ in the literature, proving it to be only an upper bound, and proposes tighter KL divergence estimators and theoretical bounds for the win rate.

## Background & Motivation

Best-of-n sampling is a simple and efficient inference-time alignment method: it draws $n$ samples from a reference policy $\pi_{\text{ref}}$, ranks them with a reward function, and selects the best one. This method is widely applied in scenarios such as Llama 2 and inference-time compute scaling, often outperforming more complex methods like RLHF and DPO on the win rate vs KL trade-off curve.

A widely used "formula" in the literature claims that the KL divergence between the best-of-n policy and the reference policy **equals**:

$$D_{\text{KL}}(\pi^{(n)} \| \pi_{\text{ref}}) = \log(n) - \frac{n-1}{n}$$

This formula has been cited by multiple key works, including Stiennon et al. (2020), Gao et al. (2023), and Coste et al. (2024). However, the authors construct a simple binary-output counterexample to prove that **this formula is incorrect**—under a binary uniform distribution, the true KL is bounded by $\log 2$, whereas this formula grows unboundedly with $n$.

## Method

### Probability Mass Function of the Best-of-n Policy

The authors first rigorously derive the PMF of the best-of-n policy. Sorting the outputs from lowest to highest rewards yields:

$$\pi^{(n)}(\mathbf{y}|\mathbf{x}) = \mathcal{F}_{\pi_{\text{ref}}}(\mathbf{y}|\mathbf{x})^n - \mathcal{F}_{\pi_{\text{ref}}}^{-}(\mathbf{y}|\mathbf{x})^n$$

where $\mathcal{F}$ and $\mathcal{F}^{-}$ represent the reward-based CDF and strict CDF, respectively. The core concept is to equivalent best-of-n to a quantile transformation of the maximum of $n$ uniform random variables.

### Upper and Lower Bounds of KL Divergence

**Theorem 3.1 (Upper Bound)**: For any $n$ and context $\mathbf{x}$,

$$D_{\text{KL}}(\pi^{(n)}(\cdot|\mathbf{x}) \| \pi_{\text{ref}}(\cdot|\mathbf{x})) \leq \log(n) - \frac{n-1}{n}$$

That is, the classical formula is actually an **upper bound** rather than an equality.

**Theorem 3.4 (Gap Upper Bound)**: The gap $G_{\text{KL}}^{(n)}$ is controlled by the second-order Rényi entropy of the model:

$$G_{\text{KL}}^{(n)}(\mathbf{x}) \leq 2n(n-1) e^{-H_2(\pi_{\text{ref}}|\mathbf{x})}$$

For $\delta$-bound models (where all output probabilities $\leq \delta$), $\text{Gap} \leq 2n(n-1)\delta$. When $n^2\delta \ll 1$, the classical formula holds approximately.

**Theorem 3.6 (Gap Lower Bound)**: When $n \cdot \varepsilon_\infty \gg 1$ (where $\varepsilon_\infty$ is the probability of the output with the highest reward), the gap grows unboundedly, and the classical formula severely overestimates.

### New KL Estimator

The authors propose an estimator based on the observed value $\varepsilon_n = \pi_{\text{ref}}(\mathbf{y}|\mathbf{x})$ (the reference probability of the sample selected by best-of-n):

$$\hat{D}_{\text{KL}}(\varepsilon_n) = (1-\varepsilon_n)^n \left(\log n + (n-1)\log(1-\varepsilon_n) - \frac{n-1}{n}\right) + (1-(1-\varepsilon_n)^n)\log\frac{1-(1-\varepsilon_n)^n}{\varepsilon_n}$$

This estimator satisfies $0 \leq \hat{D}_{\text{KL}}(\varepsilon_n) \leq \log(n) - (n-1)/n$ and closely tracks the true KL in numerical experiments.

### Win Rate Theory

**Theorem 5.3**: The upper bound for the win rate is $\mathcal{W}_r \leq \frac{n}{n+1}$.

**Theorem 5.4**: The win rate gap is controlled by Rényi entropy: $G_\mathcal{W}^{(n)} \leq \frac{n-1}{2} e^{-H_2(\pi_{\text{ref}}|\mathbf{x})}$.

When $n\delta \ll 1$, the win rate $\approx n/(n+1)$.

## Overview of Theoretical Results

| Metric | Classical Claim | Ours | Tightness Conditions |
|------|---------|---------|---------|
| KL Divergence | $= \log n - (n-1)/n$ | $\leq \log n - (n-1)/n$ (Upper bound) | Approx. holds when $n^2\delta \ll 1$ |
| KL Gap Upper Bound | — | $\leq 2n(n-1)\delta$ | $\delta$-bound model |
| KL Gap Lower Bound | — | $\geq \log(n\varepsilon_\infty) + o(\log n)$ | Gap is unbounded when $n\varepsilon_\infty \gg 1$ |
| Win Rate | No theory | $\leq n/(n+1)$ | Holds for any model |
| Win Rate Gap | — | $\leq (n-1)\delta/2$ | Approx. holds when $n\delta \ll 1$ |
| New KL Estimator | — | $\hat{D}_{\text{KL}}(\varepsilon_n) \in [0, \log n - (n-1)/n]$ | Numerical experiments show tightness |

## Experimental Validation

- **Synthetic Experiments**: A uniform distribution (alphabet sizes $L=10, 10^2, 10^3, 10^4$) validates that KL saturates when $n \approx L$, and the new estimator closely tracks the true KL throughout.
- **Cherry-picked Distributions**: Demonstrates that when the probability of the high-reward output is very small (such as $10^{-4}$), KL exhibits a step-like growth, and the classical formula fails completely.
- **Alpaca + Gemma 9B**: Using log-likelihood and negative length as reward functions, validates that the new estimator significantly outperforms the classical formula in low-entropy prompts (such as "What is the capital of France?").
- **Win Rate vs KL Trade-off**: The trade-off curve of best-of-n is close to the optimal solution of KL-regularized RL, outperforming the rewind-and-repeat strategy.

## Highlights & Insights

1. **Correcting Widespread Misuse**: Overturns the 6-year misuse of the best-of-n KL divergence "formula" in multiple top-tier conference papers, which impacts key comparison curves in works like Gao et al. (2023) and Llama 2.
2. **Good News**: That the classical formula is an upper bound means the actual reward-KL trade-off of best-of-n is **better than reported in the literature**, further strengthening its position as a strong baseline.
3. **Practical Estimator**: The new estimator only requires observing the reference probability $\varepsilon_n$ of the selected sample, making it simple to compute with controlled variance (requiring only $M = O(\log n \cdot \log(1/\delta))$ samples).
4. **Unified Theoretical Framework**: Simultaneously analyzes best-of-n and rewind-and-repeat, proving that best-of-n is approximately optimal in terms of the win rate vs KL trade-off.
5. **Key Insight**: Rényi entropy $H_2$ is the core quantity controlling all gaps—under high Rényi entropy (many low-probability outputs), the classical formula works well; under low Rényi entropy, it fails.

## Limitations & Future Work

1. **Unproven Conjectures**: Conjecture 4.4 (the expected value of the estimator is an upper bound of KL) and Conjecture 7.2 (the upper bound of the win rate-KL trade-off) are only validated numerically and lack rigorous proofs.
2. **Discrete Output Assumption**: The analysis assumes a finite output space; extending this to continuous outputs (such as diffusion models) is left to future work (e.g., Mroueh 2024).
3. **Unique Reward Assumption**: Assumption 2.1 requires unique reward values, whereas in practice, there may be many tied samples.
4. **No Practical LLM Alignment Experiments**: All experiments are synthetic or small-scale, without verifying the estimator's utility in actual RLHF pipelines.
5. **Estimator Requires $\pi_{\text{ref}}(\mathbf{y}|\mathbf{x})$**: Calculating the complete sequence probability in autoregressive models is feasible but requires an extra forward pass.

## Rating

- Novelty: ⭐⭐⭐⭐ — Overturns a widely used "formula" and proposes tighter theoretical bounds and a practical estimator.
- Experimental Thoroughness: ⭐⭐⭐ — Synthetic experiments are sufficient, but lacks validation in large-scale practical alignment scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous and clear reasoning, progressing step-by-step from counterexamples to theorems and the estimator.
- Value: ⭐⭐⭐⭐ — Holds significant corrective value for the LLM alignment community, already cited and extended by multiple subsequent works.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)
- [\[ICML 2026\] OMP: One-step Meanflow Policy with Directional Alignment](../../ICML2026/image_generation/omp_one-step_meanflow_policy_with_directional_alignment.md)
- [\[ICML 2025\] Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations](video_prediction_policy_a_generalist_robot_policy_with_predictive_visual_represe.md)
- [\[CVPR 2026\] VA-π: Variational Policy Alignment for Pixel-Aware Autoregressive Generation](../../CVPR2026/image_generation/va-p_variational_policy_alignment_for_pixel-aware_autoregressive_generation.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)

</div>

<!-- RELATED:END -->

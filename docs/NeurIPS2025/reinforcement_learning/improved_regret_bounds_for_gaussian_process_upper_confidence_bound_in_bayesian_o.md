---
title: >-
  [Paper Note] Improved Regret Bounds for GP-UCB in Bayesian Optimization
description: >-
  [NeurIPS 2025][Reinforcement Learning][Bayesian optimization] This paper proves that GP-UCB achieves $\widetilde{O}(\sqrt{T})$ high-probability regret under the Bayesian setting (when the Matérn kernel satisfies a smoothness condition) and $O(\sqrt{T \ln^2 T})$ for the SE kernel, closing the gap between existing upper bounds for GP-UCB and the optimal upper bounds.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Bayesian optimization
  - GP-UCB
  - regret bound
  - information gain
  - Gaussian process
date: 2026-05-08
content_hash: c5ca965b4b51f52c
---

# Improved Regret Bounds for GP-UCB in Bayesian Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2506.01393](https://arxiv.org/abs/2506.01393)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Bayesian optimization, GP-UCB, regret bound, information gain, Gaussian process

## TL;DR
This paper proves that GP-UCB achieves $\widetilde{O}(\sqrt{T})$ high-probability regret under the Bayesian setting (when the Matérn kernel satisfies a smoothness condition) and $O(\sqrt{T \ln^2 T})$ for the SE kernel, closing the gap between existing upper bounds for GP-UCB and the optimal upper bounds.

## Background & Motivation

**Background**: GP-UCB is widely used in Bayesian optimization (BO). The bound of Srinivas et al. (2010) is $O(\sqrt{\beta_T T \gamma_T(\mathcal{X})})$, where $\gamma_T$ denotes the maximum information gain.

**Limitations of Prior Work**: Under the Matérn kernel, GP-UCB yields $\widetilde{O}(T^{(\nu+d)/(2\nu+d)})$, which is worse than the $O(\sqrt{T\ln T})$ bound of Scarlett (2018).

**Key Challenge**: The constraint $I(\mathbf{X}_T) \leq \gamma_T(\mathcal{X})$ is a worst-case bound. Since GP-UCB concentrates its query points near the optimum, the actual information gain is far smaller than the maximum.

**Goal**: Exploit the concentration of GP-UCB query points to derive a tighter upper bound on information gain.

**Key Insight**: Because GP-UCB achieves sublinear regret, query points concentrate near $\mathbf{x}^*$, so the information gain of the concentrated query set satisfies $I(\mathbf{X}_T) \ll \gamma_T(\mathcal{X})$.

**Core Idea**: Leverage the input concentration induced by the algorithm's own behavior to analyze information gain over a shrinking local region.

## Method

### Overall Architecture

GP-UCB selects $\mathbf{x}_t = \arg\max \mu(\mathbf{x}) + \beta_t^{1/2} \sigma(\mathbf{x})$. The bottleneck lies in $\gamma_T(\mathcal{X})$; this paper replaces it with a local maximum information gain (MIG) evaluated on a shrinking ball.

### Key Designs

1. **Decomposition of Regret into Two Parts**:

    - $R_T^{(1)}(\varepsilon)$: rounds with large regret (lenient regret), $= \widetilde{O}(1)$
    - $R_T^{(2)}(\varepsilon)$: rounds with small regret, where query points lie within a quadratically growing neighborhood of $\mathbf{x}^*$

2. **Dyadic Decomposition of $R_T^{(2)}$**:

    - Partition $[T]$ into segments of length $T, T/2, T/4, \ldots$
    - For each segment: worst-case bound $\to$ number of queries $\leq T/2^i$ $\to$ sub-optimality $\leq \eta_i$ + quadratic growth $\to$ queries lie within ball $\mathcal{B}_2(\sqrt{c_{\text{quad}}^{-1}\eta_i}; \mathbf{x}^*)$
    - Replace global $\gamma_T(\mathcal{X})$ with local MIG $\gamma_{T/2^{i-1}}(\mathcal{B}_2(\cdot))$
    - **Key insight**: $\eta_i$ decreases as $i$ increases, the ball shrinks, and the effects of "more time" and "smaller region" cancel each other out

3. **Core Formula (Lemma 4)**:

    - $R_T^{(2)} \leq 2c_{\text{sup}}\bar{T} + O(\log T) + \frac{2\sqrt{2C\beta_T T}}{\sqrt{2}-1} \max_i \sqrt{\gamma_{T/2^{i-1}}(\mathcal{B}_2(\sqrt{c_{\text{quad}}^{-1}\eta_i}))}$

4. **Kernel-Specific Results**:

    - Matérn ($2\nu+d \leq \nu^2$): polynomial growth of MIG is offset by ball shrinkage, giving $\max_i \gamma = \widetilde{O}(1)$
    - SE: $R_T^{(2)} = O(\sqrt{T \ln^2 T})$

### Main Theorem

**Theorem 3**: $R_T = \widetilde{O}(\sqrt{T})$ (Matérn, $2\nu+d \leq \nu^2$); $O(\sqrt{T\ln^2 T})$ (SE).

## Key Experimental Results

### Regret Upper Bound Comparison

| Kernel | Prior GP-UCB | Ours | Scarlett 2018 | Lower Bound |
|--------|-------------|------|--------------|-------------|
| Matérn ($2\nu+d \leq \nu^2$) | $\widetilde{O}(T^{(\nu+d)/(2\nu+d)})$ | $\widetilde{O}(\sqrt{T})$ | $O(\sqrt{T\ln T})$ | $\Omega(\sqrt{T})$ |
| SE | $O(\sqrt{T\ln^{d+2}T})$ | $O(\sqrt{T\ln^2 T})$ | $O(\sqrt{T\ln T})$ | $\Omega(\sqrt{T})$ |

### Information Gain Experiment (Figure 1, 1D Matérn-5/2)

| Algorithm | Information Gain at $t=200$ | Query Distribution |
|-----------|----------------------------|-------------------|
| GP-UCB | ~2.5 | Concentrated near $\mathbf{x}^*$ |
| MVR | ~7.5 | Spread uniformly |
| Fully concentrated | ~1.8 | All at identical location |

### Key Findings
- The concentration of GP-UCB query points is a natural consequence of sublinear regret — a bootstrapping effect
- The smoothness condition $2\nu+d \leq \nu^2$ holds naturally in low-dimensional, high-smoothness regimes
- The improvement is purely analytical, requiring no modification to the algorithm

## Highlights & Insights
- **Algorithm-behavior-driven analysis**: The input concentration induced by GP-UCB itself is exploited to improve the analysis — a bootstrapping argument that represents a pioneering analytical idea
- **Dyadic decomposition**: Segmentation keeps the query ball radius controlled within each segment
- **No algorithmic change**: Zero deployment cost; moreover, knowledge of sample path constants is not required (unlike Scarlett 2018)
- The approach can be extended to instance-dependent regret analysis

## Limitations & Future Work
- The Matérn kernel result requires the smoothness condition $2\nu+d \leq \nu^2$. For example, $d=1$ requires $\nu \geq 3$; $d=2$ requires $\nu \geq (2+\sqrt{12})/2 \approx 2.73$. The condition becomes more restrictive in higher dimensions.
- The analysis does not yield improvements for the Bayesian expected regret $\mathbb{E}[R_T]$, because the dependence of the sample path constants on $\delta_{\text{GP}}$ in Lemma 2 is unknown.
- The results apply only to GP-UCB, as the necessary conditions (i)(ii) have not been verified for other algorithms such as Thompson Sampling or Information-Directed Sampling.
- A $\ln T$ gap between the SE bound $O(\sqrt{T\ln^2 T})$ and the lower bound $\Omega(\sqrt{T})$ remains.
- Hidden constants may grow exponentially in the dimension $d$; the bounds may not be favorable in the joint limit $d, T \to \infty$.
- The results concern only $T$-dependence and do not claim improvement with respect to other parameters.

### Key Properties of GP Sample Paths

The analysis relies on three sample path conditions (Lemma 2): (1) $f$ has a unique maximum $\mathbf{x}^*$ with a gap $c_{\text{gap}}$ from all other local maxima; (2) $f$ is bounded, $\|f\|_\infty \leq c_{\text{sup}}$; (3) $f$ exhibits quadratic growth near $\mathbf{x}^*$, i.e., $f(\mathbf{x}^*) - f(\mathbf{x}) \geq c_{\text{quad}}\|\mathbf{x} - \mathbf{x}^*\|_2^2$. These properties hold almost surely for the SE and Matérn ($\nu > 2$) kernels.

## Related Work & Insights
- **vs. Srinivas et al. (2010)**: The algorithm is unchanged; the analysis is strictly improved.
- **vs. Scarlett (2018)**: Scarlett achieves $O(\sqrt{T\ln T})$ via successive elimination, but the algorithm requires knowledge of sample path constants.
- **vs. Cai & Scarlett (2021)**: The lenient regret framework is borrowed; $R_T^{(2)}$ is handled independently.
- **vs. Janz et al. (2020)**: They use input partitioning in the frequentist setting; this paper achieves an analogous but distinct analysis in the Bayesian setting.
- **vs. Vakili et al. (2021)**: They improve MIG upper bounds but do not improve the regret analysis pipeline for GP-UCB.

### Summary of MIG Upper Bounds

| Kernel | $\gamma_T(\mathcal{X})$ | Source |
|--------|----------------------|--------|
| SE | $O(\ln^{d+1} T)$ | Srinivas et al. 2010 |
| Matérn-$\nu$ | $O(T^{d/(2\nu+d)} \ln^{(4\nu+d)/(2\nu+d)} T)$ | Vakili et al. 2021 |

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of "exploiting algorithmic behavior to improve algorithmic analysis" is remarkably novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐ Illustrative experiments are provided; the primary contribution is theoretical.
- Writing Quality: ⭐⭐⭐⭐⭐ Achieves a strong balance between intuitive explanation and formal rigor.
- Value: ⭐⭐⭐⭐⭐ Closes a 15-year analytical gap in BO theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)
- [\[NeurIPS 2025\] Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning](optimizing_the_unknown_black_box_bayesian_optimization_with_energy-based_model_a.md)
- [\[NeurIPS 2025\] ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](aline_joint_amortization_for_bayesian_inference_and_active_data_acquisition.md)
- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[NeurIPS 2025\] Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits](gaussian_process_upper_confidence_bound_achieves_nearly-optimal_regret_in_noise-.md)

</div>

<!-- RELATED:END -->

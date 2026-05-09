---
title: >-
  [Paper Note] Simultaneous Swap Regret Minimization via KL-Calibration
description: >-
  [NeurIPS 2025][Reinforcement Learning][Swap Regret] This paper introduces KL-Calibration as a stronger calibration measure, establishes its equivalence to the swap regret of log loss, and achieves a simultaneous swap regret bound of $\tilde{\mathcal{O}}(T^{1/3})$ via non-uniform discretization and a novel randomized rounding scheme, covering a broader class of proper losses than prior work.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Swap Regret
  - KL-Calibration
  - Proper Loss
  - Online Forecasting
  - Blum-Mansour Reduction
date: 2026-05-08
content_hash: 6d8e3d4ded867ac8
---

# Simultaneous Swap Regret Minimization via KL-Calibration

**Conference**: NeurIPS 2025
**arXiv**: [2502.16387](https://arxiv.org/abs/2502.16387)
**Code**: None
**Area**: Online Learning / Calibration Theory
**Keywords**: Swap Regret, KL-Calibration, Proper Loss, Online Forecasting, Blum-Mansour Reduction

## TL;DR

This paper introduces KL-Calibration as a stronger calibration measure, establishes its equivalence to the swap regret of log loss, and achieves a simultaneous swap regret bound of $\tilde{\mathcal{O}}(T^{1/3})$ via non-uniform discretization and a novel randomized rounding scheme, covering a broader class of proper losses than prior work.

---

## Background & Motivation

Online calibration is a central problem in sequential probability forecasting: a forecaster issues probability predictions $p_t \in [0,1]$ at each time step while an adversary selects binary labels $y_t \in \{0,1\}$, with the goal of making the empirical conditional distribution of predictions as close as possible to the true conditional distribution. Classical $\ell_1$-Calibration is known to admit a lower bound of $\Omega(T^{0.54})$, precluding the ideal $O(\sqrt{T})$ rate, whereas $\ell_2$-Calibration is achievable at $\tilde{O}(T^{1/3})$.

**Root Cause**: Prior work (Fishelson et al.) establishes $\tilde{O}(T^{1/3})$ pseudo swap regret only for squared loss, and only for proper losses with smooth univariate forms. Corresponding results for broader proper loss classes (e.g., log loss, Tsallis entropy) and for the actual swap regret (as opposed to its pseudo variant) remain absent.

**Starting Point**: The authors introduce KL-Calibration, replacing the $\ell_2$ distance with the KL divergence to measure prediction bias. By Pinsker's inequality, KL-Calibration is strictly stronger than $\ell_2$-Calibration, so any upper bound on it automatically yields richer consequences. The key insight is that KL-Calibration is precisely equivalent to the swap regret of log loss.

## Method

### Overall Architecture

The paper establishes a unified derivation chain from KL-Calibration to swap regret across multiple proper loss classes:

1. Proving the equivalence between KL-Calibration and swap regret of log loss.
2. Using Bregman divergence analysis to show that KL-Calibration controls the swap regret of $\mathcal{L}_2$ (second-order continuously differentiable univariate proper losses) and $\mathcal{L}_G$ ($G$-smooth proper losses).
3. Providing a non-constructive existence proof: $\mathbb{E}[\text{KLCal}] = O(T^{1/3}(\log T)^{5/3})$.
4. Providing an explicit algorithm achieving $\text{PKLCal} = O(T^{1/3}(\log T)^{2/3})$.

### Key Designs

1. **Equivalence between KL-Calibration and swap regret**: Via Proposition 1, the swap regret of any proper loss $\ell$ can be expressed as a weighted sum of Bregman divergences. For log loss, the corresponding Bregman divergence is exactly the KL divergence. Hence $\text{SReg}^{\text{log}} = \text{KLCal}$ and $\text{PSReg}^{\text{log}} = \text{PKLCal}$. Furthermore, for $\ell \in \mathcal{L}_2$, Lemma 3 (a second-derivative growth rate bound from Luo et al.) is invoked to establish $\text{BREG}_{-\ell}(\hat{p}, p) = O(\text{KL}(\hat{p}, p))$.

2. **Non-uniform discretization**: Standard approaches adopt a uniform grid $\mathcal{Z} = \{1/K, \ldots, (K-1)/K\}$, which fails to satisfy the critical condition $\frac{\max^2(z_i - z_{i-1}, z_{i+1} - z_i)}{z_i(1-z_i)} = O(1/K^2)$. The authors instead employ the sine-squared discretization $z_i = \sin^2(\pi i / 2K)$, which is denser near the boundaries of $[0,1]$ and simultaneously satisfies all four required properties. This is the core technical innovation for handling log loss, which is unbounded near the boundaries.

3. **Novel randomized rounding scheme (RROUND^log)**: Standard rounding (nearest-neighbor or the scheme of Fishelson et al.) incurs an $\Omega(1)$ expected loss deviation for log loss. The authors propose a probabilistic rounding that assigns weights proportional to $\frac{z_{i+1} - p}{z_{i+1}(1-z_{i+1})}$ and $\frac{p - z_i}{z_i(1-z_i)}$, and exploit the convexity of log loss to show that the expected loss deviation is only $O(1/K^2)$ (Lemma 5).

### Loss & Training

The explicit algorithm (Algorithm 1) is based on the Blum-Mansour reduction: it maintains $K+1$ external regret minimizers $\mathcal{A}_i$, each using EWOO (Exponential Weights Online Optimization) combined with RROUND^log. Since the scaled log loss is 1-exp-concave, EWOO achieves a regret of only $O(\log T)$. Setting $K = (T/\log T)^{1/3}$ yields $\text{PKLCal} = O(T^{1/3}(\log T)^{2/3})$.

The non-constructive proof (Theorem 1) applies the minimax theorem to swap the roles of the forecaster and the adversary, then analyzes the dual game using the FTL strategy together with Freedman's inequality for concentration.

## Key Experimental Results

This paper is purely theoretical and contains no experimental tables. The core results are stated as theorems.

### Main Results (Corollary 2, Explicit Algorithm)

| Loss Class | Pseudo Swap Regret Upper Bound |
|---|---|
| Log loss / $\mathcal{L}_2$ | $O(T^{1/3}(\log T)^{2/3})$ |
| $\mathcal{L}_G$ (G-smooth) | $O(G \cdot T^{1/3}(\log T)^{2/3})$ |
| $\mathcal{L} \setminus (\mathcal{L}_G \cup \mathcal{L}_2)$ | $O(T^{2/3}(\log T)^{1/3})$ |

### High-Probability Bounds (Corollary 3)

| Measure | High-Probability Bound ($\geq 1-\delta$) |
|---|---|
| $\text{Cal}_2$ | $O(T^{1/3}(\log T)^{-1/3}\log(T/\delta))$ |
| $\text{Msr}_{\mathcal{L}_G}$ | $O(G \cdot T^{1/3}(\log T)^{-1/3}\log(T/\delta))$ |

### Key Findings

- This is the first sub-$\sqrt{T}$ high-probability bound for $\ell_2$-Calibration achieved via an efficient algorithm.
- Compared to the $O(\sqrt{T}\log T)$ bound of Hu & Predict (2024) for all bounded proper losses, this paper achieves a superior $\tilde{O}(T^{1/3})$ rate for important subclasses.
- The unified framework handles both bounded and unbounded (log loss) proper losses simultaneously.

## Highlights & Insights

- **KL-Calibration as a unifying measure**: A single natural measure unifies $\ell_2$-Calibration, swap regret minimization, and proper loss generalization within one framework.
- **Non-uniform discretization**: This breaks the paradigm of uniform grids used in all prior work; the sine-squared grid excels at handling boundary singularities.
- **Refined design of randomized rounding**: The probability weights in RROUND^log precisely match the curvature structure of log loss, making the rounding error proportional to the square of the discretization step size.

## Limitations & Future Work

- The non-constructive proof (Theorem 1) does not yield an explicit algorithm; the explicit algorithm provides only the weaker pseudo variant.
- The bound for $\mathcal{L} \setminus (\mathcal{L}_G \cup \mathcal{L}_2)$ remains $O(T^{2/3})$; it is open whether this can be improved to $\sqrt{T}$ (as achieved by Hu & Predict).
- Only binary labels are considered; extension to the multiclass setting is an important open problem.
- The algorithmic complexity is $\tilde{O}(T^{5/3})$, posing challenges for large-scale practical deployment.
- It remains open whether a matching lower bound for KL-Calibration can be established, confirming the optimality of $\tilde{O}(T^{1/3})$.

## Related Work & Insights

- This work unifies the U-Calibration line of research from Hu & Predict (2024), Fishelson et al., and Kleinberg et al.
- It complements the $\ell_2$-Cal results of Foster (2023) on calibeating, while covering a broader framework.
- The non-uniform discretization idea may inspire other online learning problems involving boundary singularities, such as online regression under log loss.
- A recent breakthrough on distance to calibration (CalDist₁) can also be derived from the results of this paper: $\text{CalDist}_1 = O(T^{2/3}(\log T)^{-1/6}\sqrt{\log(T/\delta)})$.
- The impossibility result for swap omniprediction (Garg et al.) reveals the fundamental difficulty of swap regret minimization; this paper circumvents the obstruction by restricting the loss class.
- In the multiclass direction, Luo & Optimal (2024) establishes $\Theta(\sqrt{KT})$ bounds for $K$ classes, but the definition and optimality of KL-Calibration in the multiclass setting remain to be studied.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The concept of KL-Calibration is natural yet powerful; the non-uniform discretization and novel rounding scheme are original contributions.
- Experimental Thoroughness: ⭐⭐⭐ — A purely theoretical paper; results are presented as theorems, and numerical experiments are unnecessary though absent.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, proof intuitions are explained in detail, and technical contributions are explicitly contrasted with prior work.
- Value: ⭐⭐⭐⭐⭐ — Substantially advances the theoretical frontier of online calibration and simultaneous regret minimization.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization](comparing_uniform_price_and_discriminatory_multi-unit_auctions_through_regret_mi.md)
- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[AAAI 2026\] Deep (Predictive) Discounted Counterfactual Regret Minimization](../../AAAI2026/reinforcement_learning/deep_predictive_discounted_counterfactual_regret_minimization.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)

<!-- RELATED:END -->

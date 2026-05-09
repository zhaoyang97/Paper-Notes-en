---
title: >-
  [Paper Note] Isotropic Noise in Stochastic and Quantum Convex Optimization
description: >-
  [NeurIPS 2025][Optimization][stochastic convex optimization] This paper introduces the concept of an Isotropic Stochastic Gradient Oracle (ISGO)—where noise is bounded in every direction with high probability—and designs a stochastic cutting-plane algorithm achieving a query complexity of $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$, improving over SGD by a factor of $d$ in certain parameter regimes. As corollaries, the paper establishes new state-of-the-art complexities under sub-exponential noise and improves the dimension dependence of quantum stochastic convex optimization via a quantum isotropization subroutine.
tags:
  - NeurIPS 2025
  - Optimization
  - stochastic convex optimization
  - isotropic noise
  - cutting plane
  - sub-exponential distribution
  - quantum optimization
date: 2026-05-08
content_hash: 2fc71486cff977be
---

# Isotropic Noise in Stochastic and Quantum Convex Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.20745](https://arxiv.org/abs/2510.20745)
**Code**: None
**Area**: Optimization
**Keywords**: stochastic convex optimization, isotropic noise, cutting plane, sub-exponential distribution, quantum optimization

## TL;DR
This paper introduces the concept of an Isotropic Stochastic Gradient Oracle (ISGO)—where noise is bounded in every direction with high probability—and designs a stochastic cutting-plane algorithm achieving a query complexity of $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$, improving over SGD by a factor of $d$ in certain parameter regimes. As corollaries, the paper establishes new state-of-the-art complexities under sub-exponential noise and improves the dimension dependence of quantum stochastic convex optimization via a quantum isotropization subroutine.

## Background & Motivation
**Background**: SGD achieves $O(R^2\sigma_B^2/\epsilon^2)$ under a $\sigma_B$-bounded stochastic gradient oracle (BSGO), which is optimal. Under a variance-bounded oracle (VSGO) with variance $\sigma_V^2$, SGD achieves $O(R^2(\sigma_V^2 + L^2)/\epsilon^2)$. The $R^2\sigma_V^2/\epsilon^2$ term is unimprovable, but the $R^2L^2/\epsilon^2$ term arises from the deterministic component and can be reduced to $\tilde{O}(d)$ via cutting-plane methods at sufficiently high accuracy.

**Limitations of Prior Work**: Can one achieve $\tilde{O}(R^2\sigma_V^2/\epsilon^2 + d)$ under VSGO—i.e., decouple the stochastic and deterministic contributions?

**Goal**: The paper defines the ISGO noise model, affirmatively answers the above question under this model, achieves tight complexity for sub-exponential noise, and attains $\tilde{O}(dR^2\sigma_V^2/\epsilon^2 + d)$ for general VSGO.

## Method

### Noise Model Hierarchy

**BSGO (Bounded Stochastic Gradient)**: $\mathbb{E}\|\mathcal{O}_B(x)\|^2 \leq \sigma_B^2$

**VSGO (Variance-Bounded)**: $\mathbb{E}\|\mathcal{O}_V(x) - \nabla f(x)\|^2 \leq \sigma_V^2$

**ISGO (Isotropic Noise, newly defined):**

$$\Pr[|\langle \mathcal{O}_I(x) - \nabla f(x), u \rangle| \geq \sigma_I/\sqrt{d}] \leq \delta, \quad \forall \|u\|=1$$

The $1/\sqrt{d}$ normalization ensures $\sigma_I$ is comparable in magnitude to $\sigma_V$ (a $(\sigma, 0)$-ISGO is a $\sigma$-VSGO).

**ESGO (Sub-Exponential Noise)**: $\Pr[|\langle \mathcal{O}_E(x) - \nabla f(x), u \rangle| \geq t] \leq 2\exp(-t\sqrt{d}/\sigma_E)$

**Hierarchy**: ESGO → ISGO → VSGO (each conversion incurs a logarithmic or $\sqrt{d}$ factor).

### Core Algorithm: Stochastic Cutting-Plane Method

#### Step 1: Marginal Approximate Gradient Oracle (MAGO)
A MAGO is defined as a gradient estimator with error bounded by $\eta$ in a specified direction $u$ and by $\Gamma$ in $\ell_2$ norm:

$$\|\tilde{g}(x,u) - \nabla f(x)\| \leq \Gamma, \quad |\langle \tilde{g}(x,u) - \nabla f(x), u/\|u\| \rangle| \leq \eta$$

**Key Lemma (Lemma 1)**: An $(\eta, \eta\sqrt{d})$-MAGO can be implemented using $K = O(\frac{\sigma_I^2}{d\eta^2}\log(2d/\xi) + 1)$ ISGO queries, **without knowledge of the direction $u$**.

*Proof sketch*: Average the ISGO outputs; apply the Hoeffding inequality along each standard basis direction; conclude via a union bound.

#### Step 2: Cutting-Plane Method + MAGO
Run a cutting-plane method (e.g., the center-of-gravity method), using $-\tilde{g}(x_t, x_t - x^\star)$ as the halfspace oracle. The key decomposition is:

$$f(x_t) \leq f(z) + \underbrace{\langle g_t, x_t - z \rangle}_{< 0} + \underbrace{\langle \nabla f(x_t) - g_t, x_t - x^\star \rangle}_{\leq 4R\eta} + \underbrace{\langle \nabla f(x_t) - g_t, x^\star - z \rangle}_{\leq \Gamma r \leq \epsilon}$$

The second term requires controlling the error only in the direction $x_t - x^\star$, avoiding the $\sqrt{d}$ factor; the third term is absorbed by choosing $r = \min\{\epsilon/(2L), \epsilon/(4\Gamma)\}$ to handle the large $\ell_2$ error.

#### Step 3: Post-Processing to Select the Optimal Candidate
The cutting-plane method returns $T = O(d\log(d + RL/\epsilon))$ candidate points, guaranteeing at least one is $\epsilon$-optimal. Binary search combined with ISGO queries identifies the optimal point, yielding a total complexity of $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$.

### Main Theorems

**Theorem 1 (ISGO Upper Bound)**: $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$ queries suffice to solve SCO.

**Corollary 2 (ISGO Lower Bound)**: $\tilde{\Omega}(R^2\sigma_I^2/(\epsilon^2\log^2(1/\delta)) + \min\{R^2L^2/\epsilon^2, d\})$.

**Corollary 3 (ESGO)**: $\tilde{O}(R^2\sigma_E^2/\epsilon^2 + d)$, matching the lower bound $\tilde{\Omega}(R^2\sigma_E^2/\epsilon^2 + d)$ (Theorem 5).

**Corollary 4 (VSGO)**: $\tilde{O}(dR^2\sigma_V^2/\epsilon^2 + d)$, with an extra factor of $d$.

### Quantum Extension

**Theorem 6 (Quantum Isotropization)**: An ISGO can be constructed using $\tilde{O}(\sigma_V\sqrt{d}\log^7(1/\delta)/\sigma_I)$ QVSGO queries.

**Theorem 7 (Quantum SCO)**: $\tilde{O}(dR\sigma_V/\epsilon)$ queries, improving upon the prior $\tilde{O}(d^{3/2}\sigma_V R/\epsilon)$ by a factor of $\sqrt{d}$.

The core technique for quantum isotropization is the use of a boosted unbiased phase estimation algorithm in place of standard phase estimation, which avoids bias accumulation along certain directions (standard methods incur an extra $\sqrt{d}$ overhead). Multi-level Monte Carlo (MLMC) is then applied to debias the estimator while preserving the isotropic noise property.

## Key Experimental Results

This is a purely theoretical work with no experiments. The core results are summarized as complexity bounds:

| Noise Model | SGD Complexity | Ours | Condition for Improvement |
|-------------|---------------|------|--------------------------|
| ISGO | $O(R^2\sigma_I^2/\epsilon^2 + R^2L^2/\epsilon^2)$ | $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$ | $L \gg \max\{\sigma_I, \epsilon\sqrt{d}/R\}$ |
| ESGO | Same as VSGO | $\tilde{O}(R^2\sigma_E^2/\epsilon^2 + d)$ | Same as above |
| VSGO | $O(R^2\sigma_V^2/\epsilon^2 + R^2L^2/\epsilon^2)$ | $\tilde{O}(dR^2\sigma_V^2/\epsilon^2 + d)$ | $L \gg \sqrt{d}\max\{\sigma_V, \epsilon/R\}$ |

| Quantum SCO | Prev. SOTA | Ours |
|-------------|-----------|------|
| QVSGO | $\tilde{O}(d^{3/2}\sigma_V R/\epsilon)$ | $\tilde{O}(dR\sigma_V/\epsilon)$ |

## Highlights & Insights
- **Value of the ISGO noise model**: It captures information about the *shape* of the noise (beyond moments alone), enabling error control in a single direction within the cutting-plane framework and naturally yielding a factor-of-$d$ improvement.
- **Direction-agnostic MAGO**: Although the analysis requires controlling error in the direction of $x^\star$ (which is unknown), the isotropic structure of ISGO ensures that averaging automatically satisfies directional error bounds in all directions.
- **Complete resolution for sub-exponential noise**: Upper and lower bounds match up to polylogarithmic factors, effectively resolving Open Problem 1 in the sub-exponential special case.
- **Independent value of quantum isotropization**: As a general-purpose subroutine, it converts variance-bounded quantum oracles into isotropic ones, with potential applicability to other quantum optimization problems.

## Limitations & Future Work
- **Open Problem 1 remains open for general VSGO**: The VSGO result carries an extra factor of $d$; whether $\tilde{O}(R^2\sigma_V^2/\epsilon^2 + d)$ is achievable remains open.
- **Practical cost of cutting-plane methods**: Methods such as the center-of-gravity algorithm are computationally expensive (maintaining a representation of a convex body) and are less practical than SGD.
- **Hidden polylogarithmic factors**: The $\tilde{O}$ notation conceals factors of $\log(1/\epsilon)$, $\log d$, and $\log(1/\delta)$; the actual constants may be non-trivial.
- **Practical applicability of ISGO**: Whether stochastic gradient noise in real-world problems satisfies the isotropic condition requires case-by-case verification.
- **Quantum implementation barrier**: Practical realization of quantum stochastic convex optimization on quantum hardware remains a distant prospect.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The ISGO noise model and the MAGO concept are entirely new, introducing noise *shape* into the complexity analysis of SCO.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ Matching upper and lower bounds, quantum isotropization, and MLMC-based debiasing with isotropic preservation demonstrate exceptional technical sophistication.
- **Experimental Thoroughness**: ⭐ Purely theoretical work.
- **Writing Quality**: ⭐⭐⭐⭐ The open-problem-driven narrative is clear and the technical overview is effective; notation and definitions are dense.
- **Value**: ⭐⭐⭐⭐ Represents a significant advance in the foundations of SCO theory; the quantum component achieves a new state of the art.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise](unveiling_m-sharpness_through_the_structure_of_stochastic_gradient_noise.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)

</div>

<!-- RELATED:END -->

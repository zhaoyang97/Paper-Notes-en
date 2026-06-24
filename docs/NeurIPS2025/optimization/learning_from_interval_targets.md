---
title: >-
  [Paper Note] Learning from Interval Targets
description: >-
  [NeurIPS 2025][Optimization][interval regression] This paper studies regression under interval-only supervision (lower and upper bounds), establishes non-asymptotic generalization bounds based on hypothesis class smoothness without requiring a small ambiguity degree assumption, and proposes a minmax learning framework that leverages smoothness constraints to limit worst-case labels, achieving significant improvements over unconstrained methods across 18 real-world datasets.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "interval regression"
  - "weak supervision"
  - "generalization bound"
  - "Lipschitz constraint"
  - "minmax learning"
date: 2026-05-08
content_hash: 0abdd34b46b1afc1
---

# Learning from Interval Targets

**Conference**: NeurIPS 2025
**arXiv**: [2510.20925](https://arxiv.org/abs/2510.20925)  
**Code**: [bloomberg/interval_targets](https://github.com/bloomberg/interval_targets)  
**Area**: Optimization
**Keywords**: interval regression, weak supervision, generalization bound, Lipschitz constraint, minmax learning

## TL;DR
This paper studies regression under interval-only supervision (lower and upper bounds), establishes non-asymptotic generalization bounds based on hypothesis class smoothness without requiring a small ambiguity degree assumption, and proposes a minmax learning framework that leverages smoothness constraints to limit worst-case labels, achieving significant improvements over unconstrained methods across 18 real-world datasets.

## Background & Motivation
**Problem Setting**: In many practical tasks, precise labels are expensive or unavailable — medical measurements are costly, sensors record only at discrete time steps, and bond pricing yields only bid-ask spreads — yet interval bounds are often readily accessible.

**Limitations of Prior Work**: Cheng et al. (2023a) analyzed projection loss methods but relied on two strong assumptions: (a) realizability ($f^* \in \mathcal{F}$); and (b) ambiguity degree $< 1$ (i.e., the intersection of infinitely many intervals recovers the true label). However, for regression, even simple intervals of the form $[y-\epsilon, y+\epsilon]$ yield an ambiguity degree of exactly 1.

**Key Challenge**: (a) Prior results are asymptotic only, lacking finite-sample guarantees; (b) the role of smooth hypothesis classes in interval learning has not been sufficiently exploited.

**Key Insight**: Smooth (Lipschitz) hypothesis classes prevent function values from varying too much at nearby inputs, enabling "denoising" of the original intervals — tightening wide intervals into narrower effective intervals (Fig. 2–3).

## Method

### Problem Definition

Training data $\{(x_i, l_i, u_i)\}_{i=1}^n$, where $l_i \le f^*(x_i) \le u_i$. Goal: learn $f \in \mathcal{F}$ to minimize $\text{err}(f) = \mathbb{E}[\ell(f(X), Y)]$.

### Approach 1: Projection Loss

The projection loss is defined as:

$$\pi_\ell(f(x), l, u) = \min_{\tilde{y} \in [l,u]} \ell(f(x), \tilde{y})$$

By Proposition 2.1, this simplifies to evaluation at the interval boundaries:

$$\pi_\ell(f(x), l, u) = \mathbf{1}[f(x)<l]\cdot\ell(f(x),l) + \mathbf{1}[f(x)>u]\cdot\ell(f(x),u)$$

That is, the loss is zero when the prediction lies within the interval, and penalizes the distance to the nearest boundary otherwise.

### Key Mechanism: Interval Tightening (Proposition 3.4 + Theorem 3.6)

**Core Insight**: For an $m$-Lipschitz hypothesis class $\mathcal{F}$ (i.e., $|f(x) - f(x')| \le m\|x - x'\|$), any function $f$ with zero projection loss is constrained at point $x$ to lie within a tightened interval:

$$f(x) \in [l_{\mathcal{D}\to x}^{(m)},\ u_{\mathcal{D}\to x}^{(m)}]$$

where $l_{\mathcal{D}\to x}^{(m)} = \sup_{x'} (l_{x'} - m\|x-x'\|)$ and $u_{\mathcal{D}\to x}^{(m)} = \inf_{x'} (u_{x'} + m\|x-x'\|)$.

**Intuition**: If $f(x)$ were large, then $f(x')$ at all nearby points $x'$ would also have to be large (by the Lipschitz constraint), yet those values must lie within their respective intervals — a contradiction. Thus, interval information from neighboring points can be used to tighten the effective interval at the current point.

For general $f \in \widetilde{\mathcal{F}}_\eta$ (projection loss $\le \eta$), Theorem 3.6 provides a buffered extended bound:

$$f(x) \in [l_{\mathcal{D}\to x}^{(m)} - r_\eta(x),\ u_{\mathcal{D}\to x}^{(m)} + s_\eta(x)]$$

The buffers $r_\eta, s_\eta$ are implicitly defined by $\mathbb{E}_X[(r - lg_{X\to x}^{(m)})_+^p] = \eta$, and $r, s \to 0$ as $\eta \to 0$.

### Main Generalization Bounds

**Theorem 4.1 (Realizable Case)**: For an $m$-Lipschitz hypothesis class with Rademacher complexity $O(1/\sqrt{n})$, with high probability:

$$\text{err}(f) \le \underbrace{\mathbb{E}_X[|u_{\mathcal{D}\to X}^{(m)} - l_{\mathcal{D}\to X}^{(m)}|]}_{(a)\ \text{irreducible error}} + \underbrace{\tau + O\left(\frac{1}{\sqrt{n}}\right)\Gamma(\tau)}_{(b)\ \text{decays with}\ n}$$

- (a) depends on hypothesis class smoothness and interval quality — smaller $m$ yields narrower intervals, but too small an $m$ limits expressivity.
- (b) decays to $\tau$ as $n$ grows (for arbitrarily small $\tau$).

**Theorem 4.2 (Agnostic Case)**: An additional optimal hypothesis error OPT term is introduced; the upper bound converges to $\text{OPT} + \mathbb{E}[|u^{(m)} - l^{(m)}|] + 2\tau + 2\text{OPT}\cdot\Gamma(\tau)$.

### Approach 2: Minmax Learning

**Basic Minmax**: Optimize against the worst-case label:

$$\min_f \sum_i \max_{\tilde{y}_i \in [l_i, u_i]} \ell(f(x_i), \tilde{y}_i)$$

By Proposition 5.1, for the $\ell_1$ loss this is equivalent to regression on interval midpoints: $f' = \arg\min_f \sum_i |f(x_i) - (l_i+u_i)/2|$.

**Constrained Minmax** (exploiting smoothness, Proposition 5.3): Restricts worst-case labels to those consistent with the hypothesis class:

$$\min_{f \in \mathcal{F}} \max_{f' \in \widetilde{\mathcal{F}}_0} \mathbb{E}[\ell(f(X), f'(X))]$$

**Proposition 5.4**: There exist settings where constrained Minmax achieves zero error while unconstrained Minmax error can be arbitrarily large — demonstrating the critical importance of smoothness constraints.

**Two Practical Approximations**:
1. **Minmax (reg)**: Adds a projection loss regularizer on $f'$ and optimizes via alternating GDA.
2. **PL (Mean/Max)**: First trains $k$ pseudo-label functions $f_j \in \widetilde{\mathcal{F}}_\eta$, then optimizes $f$ via $\min_f \max/\text{mean}_{j} \sum_i \ell(f(x_i), f_j(x_i))$.

## Key Experimental Results

### Lipschitz MLP vs. Standard MLP (18 tabular regression datasets, projection loss)

| Dataset | LipMLP MAE | MLP MAE |
|--------|:---:|:---:|
| Ailerons | **3.278±0.034** | 4.323±0.098 |
| CPU Activity | **10.271±0.026** | 10.560±0.087 |
| Mercedes | **8.791±0.187** | 11.207±0.218 |
| Miami House | **1.013±0.028** | 1.671±0.055 |
| Sulfur | **10.681±0.082** | 14.421±0.279 |
| Superconduct | **0.540±0.021** | 1.459±0.099 |
| Topo 21 | **1.305±0.013** | 2.192±0.177 |
| YProp 4 | **2.360±0.050** | 3.828±0.435 |
| Allstate Claims | 86.547 | **86.542** |
| GPU | 29.817 | **25.123** |

**LipMLP significantly outperforms standard MLP on 14 out of 18 datasets**, validating the critical role of smoothness in interval learning.

### Effect of the Lipschitz Constant (Fig. 7)

- Too small a Lipschitz constant: hypothesis class is overly constrained, OPT term increases, error rises.
- Too large a Lipschitz constant: degenerates to standard MLP, losing the interval-tightening advantage, error rises.
- The optimal Lipschitz constant lies between the two extremes and is close to the value $m$ estimated from the training set.
- PL (Mean) error is shown as a horizontal baseline; LipMLP at the optimal $m$ is generally superior.

### Comparison of Learning Methods

| Method | Best on Uniform Intervals | Best on Centered Intervals | General Recommendation |
|------|:---:|:---:|:---:|
| Projection | ✓ | — | When interval quality is high |
| Midpoint | — | ✓ | When labels are near interval centers |
| Minmax (naive) | — | ✓ | = Midpoint |
| PL (Mean) | ✓ | ✓ | Best overall |
| PL (Max) | ✓ | — | Conservative estimate |

## Highlights & Insights
1. **Elimination of the ambiguity degree assumption** — previously a central limitation in interval learning theory, replaced here by Lipschitz smoothness.
2. The interval-tightening mechanism (Theorem 3.6) is intuitively clear and practically useful, translating smoothness into smaller effective intervals.
3. Constrained Minmax (Proposition 5.4) can arbitrarily outperform its unconstrained counterpart — a strong theoretical guarantee.
4. Non-asymptotic generalization bounds directly inform finite-sample practice.
5. Spectral normalization for Lipschitz MLPs is straightforward to implement, serving as a "free lunch" hyperparameter to tune.

## Limitations & Future Work
1. Assumes intervals always contain the true label — "noisy intervals" (where labels may fall outside the interval) are not addressed.
2. $\Gamma(\tau)$ in Theorem 4.1 is distribution-dependent and may degrade significantly for certain distributions.
3. Minmax (reg) requires alternating GDA optimization, which is less stable than the projection loss approach.
4. Only the i.i.d. setting is considered; temporal scenarios (e.g., sensor data) require new theory.
5. Experiments focus primarily on medium-scale tabular data; validation on high-dimensional settings such as image or NLP tasks is lacking.

## Related Work & Insights
- **vs. Cheng et al. (2023a)**: The latter requires ambiguity degree $<1$, realizability, and yields only asymptotic conclusions; this paper requires none of these assumptions and provides $O(1/\sqrt{n})$ non-asymptotic bounds.
- **vs. Partial Label Learning (Lv et al. 2020)**: The latter targets classification (finite label sets); the projection loss is a natural extension to regression.
- **vs. Semi-supervised Learning**: Interval learning is a form of weak supervision with a distinct theoretical framework.
- **vs. Interval Regression (classical statistics)**: Traditional methods rely on likelihood/EM; this paper provides generalization guarantees from a learning-theoretic perspective.

The smoothness ↔ interval-tightening relationship inspires the use of structural priors in other weakly supervised settings (e.g., label noise, anchor-box regression). The Lipschitz constant tuning strategy can be applied to other constrained learning problems, and the constrained Minmax framework extends naturally to other uncertainty sets (e.g., confidence intervals, distributionally robust optimization).

## Rating
- ⭐ Novelty: 4/5 — The theoretical insight of smoothness-driven interval tightening is original; the constrained Minmax framework is practically meaningful.
- ⭐ Experimental Thoroughness: 4/5 — 18 datasets, multi-method comparisons, and Lipschitz constant ablation provide comprehensive coverage.
- ⭐ Writing Quality: 4/5 — Theoretical derivations are clearly structured; intuitive figures are effective.
- ⭐ Value: 4/5 — Fills a theoretical gap in interval regression and offers practical methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Wasserstein Transfer Learning](wasserstein_transfer_learning.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->

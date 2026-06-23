---
title: >-
  [Paper Note] Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias
description: >-
  [ICLR 2026][learning_theory][Paper Note] For overparameterized linear regression (isotropic Gaussian design, minimal $\ell_p$ interpolation, $p\in(1,2]$), this paper utilizes a simple "dual ray" analysis to provide a **closed-form high-probability characterization** of the scaling of the entire family of parameter norms $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ with s
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 99a48675d8714330
---
# Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=qPKTDOJ5Xs](https://openreview.net/forum?id=qPKTDOJ5Xs)  
**Code**: TBD  
**Area**: Learning Theory / Overparameterized Linear Regression  
**Keywords**: Overparameterization, Minimal $\ell_p$ Interpolation, Norm Scaling Laws, Dual Ray Analysis, Diagonal Linear Networks, Implicit Bias

## TL;DR
For overparameterized linear regression (isotropic Gaussian design, minimal $\ell_p$ interpolation, $p\in(1,2]$), this paper utilizes a simple "dual ray" analysis to provide a **closed-form high-probability characterization** of the scaling of the entire family of parameter norms $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ with sample size $n$. It identifies a data-dependent transition point $n^\star$ ("elbow point") and a universal threshold $r^\star=2(p-1)$ that separates norms that saturate with $n$ from those that continue to grow. These laws are further transferred to Diagonal Linear Networks (DLN) trained by gradient descent.

## Background & Motivation
**Background**: Generalization metrics in modern machine learning are increasingly anchored by **parameter norms** rather than parameter counts (e.g., various norm-based generalization bounds and diagnostic indicators). However, most analyses of overparameterized regression treat "norm" generically—defaulting to $\ell_2$—and rarely address which $\ell_r$ norm should be used.

**Limitations of Prior Work**: When using parameter norms as generalization proxies, two long-neglected issues arise. First, the choice of $r$ is coupled with the "inductive bias of the interpolator" (e.g., minimal $\ell_p$), and their interaction is unclear. Second, existing precise characterizations of norms mostly focus on $\ell_2$ or extreme sparse cases where $w^\star=e_1$ (e.g., Donhauser et al. 2022), lacking a unified conclusion for the **entire family** $\{\|\hat w_p\|_r\}_{r\in[1,p]}$.

**Key Challenge**: Experiments sweeping $(r,p)$ reveal "non-trivial" phenomena—for the same interpolating predictor, some $\ell_r$ norms **plateau** rapidly as $n$ increases, while others **grow continuously** with different slopes. Changing $p$ shifts the elbow point and redistributes which $r$ values plateau. Consequently, conclusions from norm-based proxies can be extremely sensitive to the choice of $r$, which in turn depends on the underlying $\ell_p$ bias. If the effective $p$ of a training pipeline is unknown (as is almost always the case for deep networks), blindly choosing an $r$ for generalization diagnosis is risky.

**Goal**: To provide a closed-form, high-probability characterization of the scaling of $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ versus $n$ in the cleanest setting—overparameterized linear regression—and to bridge this framework to DLNs dominated by implicit bias, connecting "explicit $\ell_p$ penalties" with "optimization-induced implicit geometry."

**Key Insight**: Instead of directly tracking the optimizer $\hat w_p$, the authors examine its **dual certificate**—a vector that must fit the labels while satisfying the norm budget under over-design. By constraining the dual along the direction of the labels into a one-dimensional "ray," a decisive diagnostic scale $t_\star$ is exposed. This approach reduces a high-dimensional problem to one dimension and naturally isolates the competing forces of "signal spikes" and "noise bulk" within $X^\top Y$.

**Core Idea**: Use a 1D dual ray to balance "signal spikes vs. high-dimensional noise bulk" to directly derive the transition sample size $n^\star$ and the universal threshold $r^\star=2(p-1)$. This provides a unified picture of which $\ell_r$ norms saturate, which grow, and at what exponential rates.

## Method

### Overall Architecture
The object of study is an overparameterized linear model: design matrix $X\in\mathbb{R}^{n\times d}$ ($d>n$, rows i.i.d. $\mathcal{N}(0,I_d)$), $Y=Xw^\star+\xi$, $\xi\sim\mathcal{N}(0,\sigma^2 I_n)$. The estimator is the minimal $\ell_p$ interpolator:

$$\hat w_p\in\arg\min_{w\in\mathbb{R}^d}\|w\|_p\quad\text{s.t.}\quad Xw=Y,\qquad p\in(1,2].$$

Let $s=\|w^\star\|_0$ be the true signal support size and $\tau_s^2:=\|w^\star\|_2^2+\sigma^2$. The goal is to characterize the scaling of $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ with $n$.

The logic follows a chain from "dual to criterion to two phases": first, formulate the dual of the constrained problem; restrict the dual to a ray along labels $Y$ to obtain a scalar $t_\star$; decompose $X^\top Y$ into "signal spikes + noise bulk" to see which dominates $t_\star$; define the transition size $n^\star$ at the point of balance; and finally, use KKT conditions to translate the dual scale back to primal norms. This yields closed-form expressions for the spike-dominated ($n\gg n^\star$) and bulk-dominated ($n\ll n^\star$) regimes, identifying the threshold $r^\star=2(p-1)$. This analysis is applied to explicit $\ell_p$ and transferred to DLN via an $\alpha\mapsto p_{\text{eff}}$ calibration.

### Key Designs

**1. Dual Ray Reduction: Crushing High-Dimensional Interpolation into 1D Diagnostic Scale**

The difficulty of minimal $\ell_p$ interpolation lies in tracking a high-dimensional optimizer. The authors use the dual: the unconstrained dual of $\min_w \tfrac1p\|w\|_p^p$ s.t. $Xw=Y$ is $\max_\lambda\ \lambda^\top Y-\tfrac1q\|X^\top\lambda\|_q^q$ ($q=\tfrac{p}{p-1}$). KKT conditions imply $Xw=Y$ and $X^\top\lambda=\nabla f(w)$. By restricting the dual variable to the ray $\lambda=tY$, the optimal $t_\star$ satisfies:

$$t_\star^{\,q-1}=\frac{\|Y\|_2^2}{\|X^\top Y\|_q^q}.$$

This collapses a $d$-dimensional optimization into a scalar balance. Compared to using Gaussian Minimax Theorems (GMT/CGMT), this dual ray argument is more "first-principles" and directly produces closed-form $n$-scaling laws.

**2. Signal Spikes vs. Noise Bulk: Blockwise Competition**

The denominator $\|X^\top Y\|_q^q$ of $t_\star$ is determined by the competition of two forces. $X^\top Y$ is split into "**spikes**" (coordinates on the signal support $S$) which accumulate coherently with $n$, and the "**bulk**" (a large number of zero coordinates) where each contribution is small and noisy but the count $\sim d-s$ is vast. With high probability:

$$t_\star^{\,q-1}\ \asymp\ \underbrace{\frac{\tau_s^2 n}{n^q W_q}}_{\text{spike}}\ +\ \underbrace{(d-s)\,m_q\,\tau_s^q\, n^{q/2}}_{\text{bulk}}\ +\ O(\cdot),$$

where $W_q:=\|w^\star\|_q^q$ and $m_t:=\mathbb{E}|Z|^t$ ($Z\sim\mathcal N(0,1)$). When $n$ is small, the bulk term wins by sheer numbers; when $n$ is large, coherent spike growth dominates.

**3. Universal Threshold $r^\star=2(p-1)$ and Transition Point $n^\star$**

Equating the spike and bulk terms yields the data-dependent transition sample size:

$$n^\star\ \asymp\ \Big(\kappa_{\text{bulk}}\frac{\tau_s^q}{W_q}\Big)^{\frac{2}{q-2}},$$

representing the visible "elbow point." The main theorem provides unified bounds for $\|\hat w_p\|_r$. In the **spike-dominated regime ($n\gg n^\star$)**:

$$\|\hat w_p\|_r\ \asymp\ \begin{cases}\dfrac{\tau_s^{q+1}}{W_q}\,n^{\frac1r-\frac{1}{2(p-1)}},& r\le 2(p-1),\\[2mm]\dfrac{\tau_s^2}{W_q}\,\|w^\star\|_{(q-1)r}^{q-1},& r>2(p-1).\end{cases}$$

This identifies the **universal threshold** $r^\star=2(p-1)$. If $r>r^\star$, the exponent is negative, and the norm plateaus; if $r\le r^\star$, the norm grows with an explicit slope. In the **bulk-dominated regime ($n\ll n^\star$)**, all $\ell_r$ norms grow roughly as $n^{1/r-1/2}$.

**4. DLN Bridge: Calibrating Initialization Scale $\alpha$ to Effective $p_{\text{eff}}(\alpha)$**

DLNs (diagonal weights, effective predictor as coordinate-wise products) possess an "implicit" geometry. Prior theory suggests the initialization scale $\alpha$ tunes this bias: $\alpha\to 0$ approaches $\ell_1$ sparsity, while $\alpha\to\infty$ approaches $\ell_2^2$ density. The authors perform a **data-independent** calibration: estimating the log-log slope of the potential $Q_\alpha$ on $k$-sparse probes to match the $k^{1-p/2}$ law of $\|\cdot\|_p^p$. This yields a mapping $\alpha\mapsto p_{\text{eff}}(\alpha)$. Post-calibration, DLN $\ell_r$-vs-$n$ curves **inherit the same elbow/threshold structure**.

### Loss & Training
The explicit part requires no training—it solves the convex program for minimal $\ell_p$. The DLN part uses gradient descent. Experiments fix $d=50,000$, $\sigma=0.1$, sweeping $\alpha$ to correspond to $p\in\{1.1, 1.5, 1.9\}$.

## Key Experimental Results

### Main Results
Experiments verify theoretical curves rather than seeking SOTA. 

| Setting | Observed Phenomena | Theoretical Consistency |
|------|----------|----------------|
| Single spike $w^\star=e_1$, $p=1.5$ | Clear elbow point near $n^\star\approx1.4\times10^3$; $r>2(p-1)$ curves plateau after this point. | Matches $n^\star$ formula and spike regime expressions. |
| Single spike, $p=1.9$ (Dense) | Bulk-dominated throughout; $\ell_r$ grows as $n^{1/2}$ with fixed ordering. | Matches bulk-dominated formulas. |
| Single spike, $p=1.1$ (Sparse) | $r>2(p-1)$ curves plateau early; smaller $r$ continue to grow. | Matches threshold rules. |
| Flat $w^\star$ ($s=50$) | Similar rules, but transition scales are larger ($n^\star$ shifts right). | Matches $n^\star$ linear shift with $s$. |

### Ablation Study

| Config | Key Phenomena | Explanation |
|------|----------|------|
| Increase sparsity $s\in\{500,5000\}$ | Qualitative picture remains but shifts right; **double descent** appears for $p=1.1, s=5000$. | Confirms $n^\star$ grows with $s$. |
| DLN (calibrated $\alpha$), $w^\star=e_1$ | Smaller $\alpha$ enters spike-dominance and plateau earlier. | DLN inherits explicit $\ell_p$ structure. |
| Finite LR + Noise ($\sigma\in\{0.1,0.5\}$) | Increasing LR raises $\ell_{1.1}$ and shifts elbow right. | LR $\times$ noise acts like raising effective $p_{\text{eff}}$. |

### Key Findings
- **Threshold $r^\star=2(p-1)$ is the watershed**: It is determined solely by $p$ and remains invariant across different signals; it dictates whether a norm saturates or grows.
- **Elbow point $n^\star$ shifts right with support size $s$**: Stronger sparsity results in a longer bulk-dominance window, making double descent more likely.
- **Implicit and explicit biases share laws**: A data-independent calibration allows DLNs to replicate the same phase transitions.

## Highlights & Insights
- **Dimensionality reduction via 1D dual ray**: Mapping high-dimensional interpolation to a scalar balance $t_\star$ bypasses complex GMT machinery and yields closed-form laws from first principles.
- **"Spike vs. bulk" as a unified physical image**: This competition explains both the origin of $n^\star$ and why double descent is prominent in sparse regimes.
- **Practical Caution**: Since many norm-based proxies rely on $\|\hat w\|_r$, and different $(r,p)$ can yield **opposite** scaling behaviors, using norm bounds for diagnosis requires extreme care.

## Limitations & Future Work
- **Idealized Setup**: Isotropic Gaussian design and $p\in(1,2]$ limit the scope. Establishment under heavy-tailed or correlated designs is unverified.
- **Empirical DLN Bridge**: The $\alpha\mapsto p_{\text{eff}}$ mapping is an observation rather than a proved theorem; the "effective temperature" explanation for LR is heuristic.
- **No Direct Generalization Bounds**: The paper characterization ends at norm scaling; connecting these laws to usable new generalization upper bounds remains an open problem.

## Related Work & Insights
- **vs. Donhauser et al. (2022)**: They provided $\ell_p$ norm bounds but were largely limited to the $w^\star=e_1$ limit. Ours covers the entire family $\{\|\hat w_p\|_r\}$ and provides exact scaling instead of just upper bounds.
- **vs. Koehler et al. (2021)**: They used GMT as a bridge to generalization; our method is more direct, deriving transitions from dual balance.
- **vs. Implicit Regularization (Soudry et al.)**: Rather than reinventing implicit bias theory, this paper maps DLN geometry back to explicit $\ell_p$ geometry to verify the phase transition laws.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
<!-- RELATED:END --></div>

## Related Papers

- [\[ICLR 2026\] Covariate-Guided Clusterwise Linear Regression for Generalization to Unseen Data](covariate-guided_clusterwise_linear_regression_for_generalization_to_unseen_data.md)
- [\[ICLR 2026\] Learning under Quantization for High-Dimensional Linear Regression](learning_under_quantization_for_high-dimensional_linear_regression.md)
- [\[ICLR 2026\] Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression](larger_datasets_can_be_repeated_more_a_theoretical_analysis_of_multi-epoch_scali.md)
- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)
- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](a_new_approach_to_controlling_linear_dynamical_systems.md)

</div>

<!-- RELATED:END -->

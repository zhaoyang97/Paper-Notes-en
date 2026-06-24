---
title: >-
  [Paper Note] On Regret Bounds of Thompson Sampling for Bayesian Optimization
description: >-
  [ICML 2026][Learning Theory][Gaussian Process] This paper systematically completes the regret analysis of Gaussian Process Thompson Sampling (GP-TS) in the Bayesian setting: it first constructs a counter-example proving that GP-TS can only achieve polynomial dependence on the failure probability $\delta$ (cannot reach $\log(1/\delta)$), then provides a second-moment upper bound for cumulative regret that tightens the $\delta$ dependence by $1/\sqrt{\delta}$ times…
tags:
  - "ICML 2026"
  - "Learning Theory"
  - "Bayesian Optimization"
  - "Multi-Armed Bandits"
  - "Gaussian Process"
  - "Thompson Sampling"
  - "Cumulative Regret"
  - "Lenient Regret"
  - "Information Gain"
date: 2026-05-08
content_hash: 9ec2f322bbd674ac
---

# On Regret Bounds of Thompson Sampling for Bayesian Optimization

**Conference**: ICML 2026  
**arXiv**: [2603.09276](https://arxiv.org/abs/2603.09276)  
**Code**: None (Pure Theory)  
**Area**: Learning Theory / Bayesian Optimization / Multi-Armed Bandits  
**Keywords**: Gaussian Process, Thompson Sampling, Cumulative Regret, Lenient Regret, Information Gain

## TL;DR
This paper systematically completes the regret analysis of Gaussian Process Thompson Sampling (GP-TS) in the Bayesian setting: it first constructs a counter-example proving that GP-TS can only achieve polynomial dependence on the failure probability $\delta$ (cannot reach $\log(1/\delta)$), then provides a second-moment upper bound for cumulative regret that tightens the $\delta$ dependence by $1/\sqrt{\delta}$ times, the first polylogarithmic upper bound for expected lenient regret, and a high-probability regret bound of $\tilde O(\sqrt T)$ under relaxed Matérn conditions, bringing the theoretical guarantees of GP-TS essentially on par with the well-studied GP-UCB.

## Background & Motivation
**Background**: Bayesian Optimization (BO) is a mainstream framework for black-box, expensive function optimization. Its core involves using a Gaussian Process (GP) as a surrogate model and sequentially selecting points based on an acquisition function. Theoretically, the focus is on **cumulative regret** $R_T=\sum_{t=1}^T f(\boldsymbol x^*)-f(\boldsymbol x_t)$; as long as it is $o(T)$ sublinear, the algorithm is guaranteed to converge to the optimum. The two most prominent algorithms are GP-UCB (deterministic confidence upper bound) and GP-TS (sampling a sample path from the posterior each round and taking its maximizer).

**Limitations of Prior Work**: GP-UCB has been analyzed extensively, with high-probability bounds, expectation bounds, lenient regret bounds, and tight cumulative regret bounds. Although GP-TS often performs better in practice (as it does not rely on the GP-UCB confidence width parameter, which requires fine-tuning and lacks practical guarantees), its theoretical analysis lags significantly: its high-probability regret bound could only be directly derived from "expectation bound + Markov's inequality," leading to a $1/\delta$ dependence on $\delta$ (whereas GP-UCB is $\log(1/\delta)$); furthermore, lenient regret and tighter cumulative regret bounds for GP-TS were non-existent, and Iwazaki (2025b) explicitly listed them as open problems.

**Key Challenge**: The stochasticity of GP-TS (sampling a posterior path each round) ensures that $\boldsymbol x_t$ and $\boldsymbol x^*$ are identically distributed and conditionally independent given the history. This property is the cornerstone of GP-TS expectation regret analysis, but it complicates "high-probability" analysis—applying Markov directly yields the poor $1/\delta$ dependence. The question is: is the poor dependence of GP-TS on $\delta$ due to **suboptimal proof techniques** or an **inherent limitation of the algorithm**?

**Goal**: To close the analytical gap between GP-UCB and GP-TS, specifically addressing four sub-problems: (i) Can GP-TS achieve $\log(1/\delta)$ dependence? (ii) Can the $1/\delta$ dependence be tightened? (iii) Can a lenient regret bound be established? (iv) Can the dependence on time step $T$ be tightened to $\tilde O(\sqrt T)$?

**Core Idea**: For (i), a **negative answer** is provided via a two-armed counter-example (indicating an inherent limitation, not a technical issue). For (ii), (iii), and (iv), **positive answers** are provided by developing new proof tools (second-moment decomposition, lenient regret proof using elliptic potential counting, and refined analysis for relaxed Matérn conditions).

## Method

### Overall Architecture
The paper does not propose a new algorithm but provides four sets of regret results centered on the **same GP-TS algorithm** (Algorithm 1: update GP posterior → sample path $g_t\sim p(f\mid\mathcal D_{t-1})$ → take $\boldsymbol x_t=\arg\max_{\boldsymbol x} g_t(\boldsymbol x)$ → observe noisy $y_t$). All analyses are conducted in a Bayesian setting: the target function $f\sim\mathcal{GP}(0,k)$ is a sample path of a GP (distinct from the frequentist setting where $f$ is assumed to be in an RKHS), and observations are corrupted by Gaussian noise $y_i=f(\boldsymbol x_i)+\epsilon_i$. Complexity is characterized by the **maximum information gain** $\gamma_T:=\max_A I(\boldsymbol y_A;\boldsymbol f_A)$, which is sublinear in $T$ for common kernels (SE kernel $O((\log T)^{d+1})$, Matérn-$\nu$ kernel $O(T^{d/(2\nu+d)}\cdots)$).

There is a clear logical progression among the four results: the **counter-example (Thm 3.1)** sets a ceiling for GP-TS's capability regarding $\delta$; the **second-moment bound (Thm 3.2)** tightens the $\delta$ dependence beneath this ceiling; **lenient regret (Thm 3.3)** is an independent contribution and a key lemma for tightening the $T$ dependence; and the **improved $T$ dependence (Lemma 3.4 + Thm 3.5)** leverages the lenient regret results and refined analysis of relaxed Matérn conditions to achieve a $\tilde O(\sqrt T)$ high-probability bound.

### Key Designs

**1. Two-armed Counter-example: Proving Polynomial Dependence on $\delta$ is an Inherent Limitation**

To address whether the poor $\delta$ dependence of GP-TS is merely due to loose proofs, the authors construct a counter-example. In a simple two-armed problem $\mathcal X=\{\boldsymbol x^{(1)},\boldsymbol x^{(2)}\}$, arm values follow a correlated Gaussian prior $\big(f(\boldsymbol x^{(1)}),f(\boldsymbol x^{(2)})\big)\sim\mathcal N\big(\boldsymbol 0,\big[\begin{smallmatrix}1&1/2\\1/2&1\end{smallmatrix}\big]\big)$, with noise $\epsilon_t\sim\mathcal N(0,1)$. Theorem 3.1 proves that for $T>e$:

$$\Pr\big(R_T\geq T/2\big)\geq \frac{c_1}{T^{c_2}},$$

where $c_1,c_2>0$ are constants. This implies that with a probability of at least $c_1/T^{c_2}$, GP-TS will consistently choose the wrong arm, incurring linear regret $T/2$. Substituting $\delta=c_1/T^{c_2}$ shows that GP-TS **cannot** have an $O(\log(1/\delta))$ cumulative regret bound in general—this is a theoretical limitation of the algorithm itself. Notably, $c_2=17$ is quite large, so this counter-example highlights an asymptotic ceiling rather than suggesting GP-TS is unstable in practice (where it often excels). This result guides the extent to which $1/\delta$ can be tightened.

**2. Second-moment Bound for Cumulative Regret: Tightening $\delta$ Dependence by $1/\sqrt\delta$**

Since $\log(1/\delta)$ is unachievable, the goal is to tighten the polynomial dependence. The best existing high-probability bound is $O(\sqrt{T\gamma_T\log T}/\delta)$, which is $1/\delta$ in $\delta$. Instead of direct high-probability analysis, this paper bounds the **second moment of cumulative regret**. Theorem 3.2 gives:

$$\mathbb E[R_T^2]=O(T\gamma_T\log T),$$

which, by Markov’s inequality, immediately implies that for any $\delta\in(0,1)$, $R_T=O\big(\sqrt{T\gamma_T\log T/\delta}\big)$ with probability at least $1-\delta$, tightening the dependence from $1/\delta$ to $1/\sqrt\delta$. The proof utilizes the Russo & Van Roy decomposition and the core property of GP-TS—where $\boldsymbol x^*$ and $\boldsymbol x_t$ are identically distributed and conditionally independent given $\mathcal D_{t-1}$—to split $\mathbb E[R_T^2]$ into two components involving $T\sum_t \mathbb E\big[(f(\boldsymbol x^*)-U_t(\boldsymbol x^*))_+^2+\beta_t\sigma_{t-1}^2(\boldsymbol x_t)\big]$. The sum of posterior variances is bounded by $\gamma_T$, while the expectation of the **squared** term $(f(\boldsymbol x^*)-U_t(\boldsymbol x^*))_+$ is bounded using new lemmas (Lemma E.5/E.6), a critical step in moving from first to second moments.

**3. Polylogarithmic Bound for Expected Lenient Regret: First Such Bound for Any Algorithm**

Lenient regret $LR_T=\sum_{t\in\mathcal T}f(\boldsymbol x^*)-f(\boldsymbol x_t)$ only counts regret for rounds where suboptimality exceeds a tolerance $\Delta$, i.e., $\mathcal T=\{t\mid f(\boldsymbol x^*)-f(\boldsymbol x_t)\geq\Delta\}$. Theorem 3.3 proves that the expected lenient regret of GP-TS is **polylogarithmic** in $T$: $\mathbb E[LR_T]=O\big(\sqrt{\beta_T T_{\max}\tilde\gamma_{T_{\max}}}\big)$, where $\mathbb E[|\mathcal T|]\leq T_{\max}$ varies by kernel (e.g., SE kernel $O(\beta_T\log^{d+1}T/\Delta^2)$). This is the first **expected** lenient regret bound for any algorithm. The proof uses the elliptic potential counting lemma to convert the count $|\mathcal T|$ into a sum of squared regrets and employs Lemma E.7 to bound $\mathbb E[\sum_{t\in\mathcal T}\sigma_{t-1}^2(\boldsymbol x^*)]$ and $\mathbb E[\sum_{t\in\mathcal T}\sigma_{t-1}^2(\boldsymbol x_t)]$ via $\tilde\gamma_{\mathbb E[|\mathcal T|]}$.

**4. Refined Analysis for Relaxed Matérn Conditions: Achieving $\tilde O(\sqrt T)$ High-Probability Regret**

Finally, the dependence on $T$ is tightened. Lemma 3.4 provides a general result: under specific global maxima conditions and an event $E$ regarding the regret bound per step, $R_T$ is $O(\sqrt T\log T)$ for SE kernels and $\tilde O(\sqrt T)$ for Matérn kernels ($\nu>2$). A key breakthrough is relaxing the Matérn condition from $2\nu+d\leq\nu^2$ (Iwazaki 2025b) to simply **$\nu>2$**, resolving an open problem without stronger regularity conditions. This is achieved by dynamically scheduling the tolerance $\Delta$ relative to $T$ and iteratively applying "lenient regret bounds + information gain scaling" to contract $\boldsymbol x_t$ toward $\boldsymbol x^*$. Theorem 3.5 then extends this to GP-TS by layering cumulative regret and applying union bounds to achieve $R_T=\tilde O(\sqrt T)$ with probability at least $1-\delta-\delta_{\rm GP}$.

### Loss & Training
Not applicable—the paper is purely theoretical and does not involve training objectives or hyperparameters; the only "tunable" quantities are the tolerance sequences $\{\Delta_i\}$ and $\Delta$ used in the analysis.

## Key Experimental Results

As this is a pure theory paper, there are no numerical experiments. The following tables summarize the improvements over existing work.

### Main Results: GP-TS Regret Bounds vs. Prior Work

| Result | Ours | Prev. SOTA (GP-TS) | Gain |
|------|------|------------------|--------|
| Regret w.r.t. $\delta$ (Lower Bound) | Thm 3.1: $\Omega(1/\delta^{c})$ prob. of linear regret | None | First proof that $\log(1/\delta)$ is unachievable |
| Regret w.r.t. $\delta$ (High-prob Upper) | Thm 3.2: $O(\sqrt{T\gamma_T\log T/\delta})$ | $O(\sqrt{T\gamma_T}/\delta)$ | Tightened $\delta$ dependence by $1/\sqrt\delta$ |
| Expected Lenient Regret | Thm 3.3: Polylogarithmic in $T$ | None (only GP-UCB high-prob) | First expected lenient regret bound |
| Regret w.r.t. $T$ (High-prob Upper) | Thm 3.5: $\tilde O(\sqrt T)$ (SE / Matérn $\nu>2$) | $O(\sqrt{\gamma_T T\log T}/\delta)$ | $T$ dependence tightened to $\tilde O(\sqrt T)$ |

### Position Comparison with GP-UCB

| Dimension | GP-UCB | GP-TS (Post-this-paper) | Remarks |
|------|--------|------------------|------|
| $\delta$ Dependence | $\log(1/\delta)$ | Polynomial $1/\sqrt\delta$ (Thm 3.2) | Thm 3.1 proves log is impossible for TS |
| Lenient Regret | High-prob bound (Existing) | Expected bound (Thm 3.3) | Ours can derive GP-UCB expected bound |
| $T$ Dependence | $\tilde O(\sqrt T)$ | $\tilde O(\sqrt T)$ (Thm 3.5) | Now on par |
| Matérn Condition | $\nu>2$ | $\nu>2$ (Relaxed from $2\nu+d\leq\nu^2$) | Lemma 3.4 is universal |
| Param. Dependence | Needs tuned $\beta_t$ | No tuning needed | GP-TS practical advantage |

## Highlights & Insights
- **Inherent Limits Identified**: The two-armed counter-example proves the difference in $\delta$ dependence is fundamental. The large constant $c_2=17$ suggests this is a formal "ceiling" rather than a sign of practical instability.
- **Second Moments as a Key**: Shifting focus to $\mathbb E[R_T^2]$ provides a natural path to $1/\sqrt\delta$ dependence via Markov, bypassing the difficulties of direct high-probability analysis for randomized algorithms.
- **Lenient Regret as a Dual-purpose Result**: Thm 3.3 is both a novel result and a critical machinery for tightening $T$ dependence in Lemma 3.4/Thm 3.5.
- **Unified Matérn Analysis**: Lemma 3.4 applies to both GP-UCB and GP-TS, resolving multiple open problems from prior literature simultaneously.

## Limitations & Future Work
- **Optimality of $\delta$**: There remains a gap between the lower bound in Thm 3.1 and the $1/\sqrt\delta$ upper bound in Thm 3.2.
- **Practicality of Matérn $\nu>2$**: This condition excludes the commonly used $\nu=1/2, 3/2$. Further relaxation is an open question.
- **Variance-inflated GP-TS**: The authors conjecture that GP-TS with variance inflation might achieve $O(\sqrt{T\gamma_T\log(T/\delta)})$, but transplantation of current lenient regret analysis is non-trivial.
- **Sampling Error**: The analysis assumes exact posterior sampling, ignoring discrete approximation errors (e.g., RFF) used in practice.

## Related Work & Insights
- **vs GP-UCB (Srinivas 2010 / Iwazaki 2025b)**: This work brings GP-TS theoretical guarantees to the same level as GP-UCB regarding $T$ and Matérn conditions, while proving why a gap must remain for $\delta$.
- **vs Russo & Van Roy (2014)**: Building on their expectation framework, this paper elevates the analysis to second moments and high-probability bounds.
- **vs Scarlett (2018) Lower Bounds**: While Scarlett provides algorithm-independent bounds, Thm 3.1 provides the first algorithm-specific lower bound for GP-TS regarding the failure probability $\delta$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ (High score for comprehensive theoretical coverage)
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](../../ICLR2026/learning_theory/variance-dependent_regret_lower_bounds_for_contextual_bandits.md)
- [\[ICLR 2026\] Queue Length Regret Bounds for Contextual Queueing Bandits](../../ICLR2026/learning_theory/queue_length_regret_bounds_for_contextual_queueing_bandits.md)
- [\[ICLR 2026\] Discounted Online Convex Optimization: Uniform Regret Across a Continuous Interval](../../ICLR2026/learning_theory/discounted_online_convex_optimization_uniform_regret_across_a_continuous_interva.md)
- [\[ICLR 2026\] Poisson Midpoint Method for Log-Concave Sampling: Beyond the Strong Error Lower Bounds](../../ICLR2026/learning_theory/poisson_midpoint_method_for_log_concave_sampling_beyond_the_strong_error_lower_b.md)
- [\[ICML 2026\] Learning Credal Ensembles via Distributionally Robust Optimization](learning_credal_ensembles_via_distributionally_robust_optimization.md)

</div>

<!-- RELATED:END -->

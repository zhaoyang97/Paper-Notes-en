---
title: >-
  [Paper Note] Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions
description: >-
  [ICML 2026][learning_theory][Paper Note] This paper proves that in discrete first-price auctions, the revenue of any $\epsilon$-approximate correlated equilibrium is at least $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$ (where $v_2$ is the second-highest valuation). This provides the first **polynomial convergence rate** for the revenue of no-swap-regret bidders in
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: 570f55329285320a
---
# Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions

**Conference**: ICML2026  
**arXiv**: [2606.06085](https://arxiv.org/abs/2606.06085)  
**Code**: None (Theoretical paper)  
**Area**: Learning Theory / Online Learning / Algorithmic Game Theory  
**Keywords**: First Price Auctions, No-Swap-Regret, Correlated Equilibrium, Convergence Rate, Dual Fitting

## TL;DR
This paper proves that in discrete first-price auctions, the revenue of any $\epsilon$-approximate correlated equilibrium is at least $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$ (where $v_2$ is the second-highest valuation). This provides the first **polynomial convergence rate** for the revenue of no-swap-regret bidders in first-price auctions—by using an optimal $O(\sqrt{kT})$ swap-regret algorithm, only $O(k^5/\epsilon^2)$ rounds are required for the time-average revenue to approach the second-highest valuation, significantly improving upon the previous quasi-polynomial bound of $k^{O(\log k)}$.

## Background & Motivation

**Background**: Advertising auctions are a multi-billion dollar industry. In recent years, there has been a large-scale shift from second-price auctions (SPA) to first-price auctions (FPA) (e.g., Google Ad Manager switched in March 2019). In SPA, truthful bidding is a dominant strategy and easy to analyze; FPA, however, forces bidders to deviate from straightforward strategies, making them complex and unpredictable. In practice, bidders increasingly use online learning or auto-bidding algorithms to adjust bids based on competitors' behavior. These algorithms carry "no-regret" guarantees—after $T$ rounds, the bidder's total utility approaches the optimal fixed bid regardless of opponents' actions. Consequently, a line of research asks: how much revenue does an FPA generate when all participants use online learning algorithms?

**Limitations of Prior Work**: Existing works ([17, 19]) characterize the set of correlated equilibria in FPAs and prove that the revenue of an **exact** correlated equilibrium is at least $v_2-\Theta(1/k)$. Combining this with the fact that "the time-average behavior of no-swap-regret algorithms converges to a correlated equilibrium," one can infer that revenue approaches $v_2$ as $T\to\infty$. However, this is only an asymptotic conclusion and **fails to provide the number of rounds required** to achieve such revenue.

**Key Challenge**: In reality, although the number of rounds is large (ad auctions occur every millisecond), it is finite. Under finite $T$, no-swap-regret dynamics converge to an **approximate** ($\epsilon$-approximate) correlated equilibrium rather than an exact one. The proofs in [17, 19] hold only for exact equilibria. Attempting to extend their arguments to $\epsilon$-approximate correlated equilibria results in a revenue guarantee with a **quasi-polynomial** dependence on $k$: at least $v_2-\Theta(1/k)-\epsilon\cdot k^{\text{poly}(\log k)}$. This implies that even with an optimal algorithm (swap-regret $O(\sqrt{kT})$), the required number of rounds is $k^{O(\log k)}$—a value so large that auctioneers would never practically reach that high-revenue state.

**Goal**: To answer Question 1.5—can the revenue of any $\epsilon$-approximate correlated equilibrium be bounded by $v_2-\Theta(1/k)-\epsilon\cdot\text{poly}(k)$ (i.e., paying a polynomial rather than quasi-polynomial cost for $\epsilon$)?

**Key Insight**: The authors abandon the proof strategies of [17, 19] and instead introduce the **dual-fitting** technique from approximation algorithms. To the authors' knowledge, this is the first application of dual-fitting to establish convergence rates.

**Core Idea**: Construct a linear program (LP) whose optimal value $Z^\star_{\text{LP}}$ is a lower bound on the revenue of any $\epsilon$-approximate correlated equilibrium. Then, provide a set of explicit feasible assignments for its dual and use weak duality to push the revenue lower bound to $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$.

## Method

### Overall Architecture
The "method" of this paper is a proof framework aimed at providing the tightest possible polynomial lower bound for the revenue of an $\epsilon$-approximate correlated equilibrium in a discrete FPA. Setting: $n$ bidders, discrete bid set $B=\{0,1/k,\dots,1-1/k,1\}$, bidder $i$ has private valuation $v_i$, bids do not exceed valuations, the highest bidder wins, and ties are broken randomly. Revenue is defined as $\sum_{s}\mu(s)\cdot r(s)$, where $r(s)=\max(s_1,\dots,s_n)$ is the winning bid. The argument consist of three steps: **(1)** Relax the problem of "finding the minimum revenue across all $\epsilon$-approximate correlated equilibria" into an LP; **(2)** Use dual-fitting to give explicit assignments to dual variables, proving dual feasibility and achieving a value of $v_2-3/k-\Theta(\epsilon k^2)$; **(3)** Translate this equilibrium-level revenue bound into a specific round-based convergence rate via the conversion from "no-swap-regret time-average behavior to approximate correlated equilibrium." Finally, a matching upper bound is provided to show that the dependence on $\epsilon$ cannot be removed.

### Key Designs

**1. LP Relaxation: Formulating Revenue Lower Bounds as a Linear Program**

The authors first express the minimum revenue among all $\epsilon$-approximate correlated equilibria as an LP with $\mu(\cdot)$ as the variables (Definition 3.1): Objective $\min\sum_s \mu(s)r(s)$, subject to the deviation conditions of approximate correlated equilibrium $\sum_{s_{-i}}\mu(s_i,s_{-i})(U_i(s_i',s_{-i})-U_i(s_i,s_{-i}))\le \epsilon$ (for each bidder $i$ and every bid pair $s_i,s_i'$), and the constraint that $\mu$ is a probability distribution. The Key Insight (Lemma 3.2) is that any $\epsilon$-approximate correlated equilibrium satisfies these constraints and is a feasible solution to the LP, making the LP optimal value $Z^\star_{\text{LP}}$ a lower bound for the revenue. This transforms a difficult infimum problem over equilibria into a clean optimization target.

**2. Dual Fitting: Explicit Feasible Assignments for Dual Variables**

Since solving the primal LP directly is difficult, the authors turn to its dual (Lemma 3.3). The dual variables consist of a scalar $\mu\in\mathbb{R}$ and a family $\lambda^i_{s_is_i'}\ge 0$. Dual feasibility requires $\mu+\sum_i\sum_{s_i'}\lambda^i_{s_is_i'}(U_i(s_i,s_{-i})-U_i(s_i',s_{-i}))\le r(s)$ for all bid profiles. The essence of dual-fitting is **guessing a good assignment**: the authors set $\hat\lambda^i=0$ for all bidders $i\ge 3$, providing non-zero values only for the top two bidders (those with the highest valuations):

$$\hat\lambda^i_{s_is_i'} := \begin{cases} 1, & s_i+1/k \le s_i' \le v_2-2/k \\ 0, & \text{otherwise} \end{cases}\quad (i\in\{1,2\})$$

and setting $\hat\mu := v_2-3/k$. They then prove that this assignment is dual feasible (Lemma 3.7: verified across four categories of bid spaces) and that the dual objective satisfies $\hat\mu-\epsilon\sum_{i,s_i,s_i'}\hat\lambda^i_{s_is_i'}\ge v_2-3/k-2\epsilon k^2$. By **weak duality**, this immediately implies $\sum_s\mu(s)r(s)\ge Z^\star_{\text{LP}}\ge v_2-3/k-\Theta(\epsilon k^2)$, which is the Main Results (Theorem 2.6). This approach is effective because dual-fitting does not require the optimal dual solution; constructing **any** feasible solution provides a lower bound for the primal, reducing the previous quasi-polynomial bound to $k^2$.

**3. From Equilibrium Bounds to Convergence Rates: Answering "How Many Rounds"**

The main result is a static bound for equilibria. The authors then connect it to dynamics. It is known (Theorem 2.3) that if each bidder uses a no-swap-regret algorithm with regret $R_i(T)$, then after $T=O(\max_i R_i(T)/\epsilon+\log(n/\delta)/\epsilon)$ rounds, the time-average distribution is an $\epsilon$-approximate correlated equilibrium with high probability. Combining this with Theorem 2.6 (Corollary 2.7) yields the time-average revenue $\frac{1}{T}\sum_t r(s^t)\ge v_2-3/k-O(\epsilon)$, with required rounds:

$$T=O\!\left(k^2\max_i R_i(T)/\epsilon + k^2\log(n/\delta)/\epsilon\right).$$

If all participants use the optimal algorithm from [10] (swap-regret $O(\sqrt{kT})$), then $T=O(k^5\log(n/\delta)/\epsilon^2)$ rounds suffice—a qualitative leap from the previous $k^{O(\log k)}/\epsilon^2$. This step finally answers Question 1.1 regarding the number of rounds.

**4. Matching Upper Bound: Proving the Essential Nature of $\epsilon$ Dependence**

To demonstrate that the form of the bound is correct, the authors provide a revenue upper bound (Theorem 2.8): There exists a discrete FPA with two bidders ($v_1=v_2=1$) and an $\epsilon$-approximate correlated equilibrium whose revenue is $\le 1-\Theta(1/k)-\Theta(\epsilon)$. This indicates that the $O(\epsilon)$ term in the revenue is intrinsic and cannot be removed. Note that while the lower bound (Theorem 2.6) is $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$ and the upper bound (Theorem 2.8) is $v_2-\Theta(1/k)-\Theta(\epsilon)$, there remains a gap in the coefficient of $\epsilon$ ($k^2$ vs. constant). Based on experiments, the authors conjecture the upper bound is closer to the truth.

### Loss & Training
This is a purely theoretical paper with no training objective. The only "algorithm" analyzed is the optimal no-swap-regret online learning algorithm proposed by [10], which has a swap-regret of $R^{\text{swap}}_A(T)=O(\sqrt{|S_i|T})=O(\sqrt{kT})$.

## Key Experimental Results

### Main Results
Theoretical comparison (improvement of convergence rounds from quasi-polynomial to polynomial):

| Result | Revenue Lower Bound | Rounds $T$ required to reach $v_2-\Theta(1/k)$ |
|------|---------|-----------------------------------|
| Prev. extension of [17, 19] (Theorem 1.4) | $v_2-\Theta(1/k)-\epsilon\cdot k^{\text{poly}(\log k)}$ | $k^{O(\log k)}/\epsilon^2$ (Quasi-polynomial) |
| **Main Results (Theorem 2.6)** | $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$ | $O(k^5/\epsilon^2)$ (Polynomial) |
| **Upper Bound (Theorem 2.8)** | $v_2-\Theta(1/k)-\Theta(\epsilon)$ (Existence, reverse) | — |

Numerical verification (repeated FPA using $O(\sqrt{kT})$ algorithm from [10], time-average revenue constraint $\frac{1}{T}\sum_t r(t)\ge v_2-3/k-O(k^{2.5}/\sqrt{T})$):

| Setup | Parameters | Observations |
|---------|------|------|
| Deterministic Valuations | $k=10$, 3 agents, $v_1=1\ge v_2\ge v_3=0$, $v_2\in\{0,0.2,0.5,0.8,1\}$ | Time-average revenue exceeds $v_2-0.1$ in all cases, faster than $O(k^7)$ |
| Discretization $k$ Trade-off | 3 agents, $v_1=v_2=1,v_3=0$, $k\in\{10,30,50\}$ | Larger $k$ leads to higher long-term revenue but slower convergence |
| Bayesian i.i.d. Setup | $v_i\in\{0,1\}$ w.p. 1/2, $n\in\{2,3,4,5\}$, $k=10$ | Time-average revenue converges to $\mathbb{E}[v_2]=1-(n+1)/2^n$ |
| Bayesian Mixed Nash | $n=2$, unique CDF in continuous space $F_P(x)=1/(1-x)-1$ | Time-average bidding behavior very close to Bayesian Mixed Nash Equilibrium |

### Key Findings
- **Experiments converge faster than theory**: In all numerical experiments, time-average revenue approached $v_2$ much faster than predicted by Theorem 2.6, supporting the conjecture that the correct bound follows the upper bound in Theorem 2.8 ($\Theta(\epsilon)$ instead of $\Theta(\epsilon k^2)$).
- **The Double-edged Sword of Discretization**: Increasing $k$ allows for higher long-term achievable revenue ($v_2-1/k$ is closer to $v_2$), but the convergence rate slows down—reflecting the high power of $k$ in $T=O(k^5/\epsilon^2)$.
- **Extrapolation to Bayesian Settings**: Even with random valuations, no-swap-regret dynamics generate revenue near the expected second-highest valuation $\mathbb{E}[v_2]$, and bidding behavior approaches the Bayesian Mixed Nash Equilibrium, suggesting dual-fitting may generalize to Bayesian settings.

## Highlights & Insights
- **First application of dual-fitting to convergence rate analysis**: Dual-fitting, originally a tool for approximation algorithms, is used here for revenue analysis of game dynamics. Constructing a single dual-feasible solution provides a tight lower bound, bypassing the quasi-polynomial difficulties of direct reasoning over approximate equilibria. This approach may have independent value for other convergence rate problems.
- **Assigning non-zero dual variables only to top two bidders**: The dual assignment (Definition 3.5/3.6) exploits the structure of FPA revenue being dominated by the two highest valuations. Setting variables for $i\ge 3$ to zero reduces combinatorial complexity—a key trick to compressing the bound from $k^{O(\log k)}$ to $k^2$.
- **Complementary Upper and Lower Bounds**: By providing both a lower bound and a matching upper bound to prove that $\epsilon$-dependence is intrinsic, the paper clearly defines the boundaries of the problem, leaving a specific gap ($k^2$ vs. constant) for future research.

## Limitations & Future Work
- **Gap between bounds**: There is a gap between the lower bound $\Theta(\epsilon k^2)$ and the upper bound $\Theta(\epsilon)$. The authors conjecture the upper bound is correct but have not proved it; closing this gap to achieve a faster convergence rate remains an open problem.
- **Limited to no-swap-regret**: The results focus on no-swap-regret dynamics (converging to correlated equilibria). Whether this method applies to weaker no-(external)-regret dynamics (converging to coarse correlated equilibria) is not addressed.
- **Bayesian theory is missing**: Observations in i.i.d./Bayesian settings are purely experimental (revenue approaches $\mathbb{E}[v_2]$). Theoretical analysis of dual-fitting in Bayesian settings is listed as a future direction.

## Related Work & Insights
- **vs. [17, 19] (Correlated Equilibrium Characterization)**: They characterize revenue for exact correlated equilibria (continuous/limit). Extending them to approximate equilibria yields only quasi-polynomial bounds; this paper uses dual-fitting for polynomial bounds and provides explicit finite-round rates.
- **vs. [20, 31, 15] (No-Regret Convergence)**: [20] requires an initial exploration phase; [31] proves convergence to the second-highest price only if convergence occurs; [15] is limited to the top two bidders. This paper provides unconditional convergence results with explicit rates for **all** no-swap-regret dynamics.
- **vs. [2] (Strong Duality for Nash)**: [2] uses strong duality to prove projected gradient ascent converges to Nash equilibrium. This paper uses weak duality and dual-fitting for revenue lower bounds across the broader class of no-swap-regret dynamics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First introduction of dual-fitting to convergence rate analysis, reducing quasi-polynomial bounds to polynomial.
- Experimental Thoroughness: ⭐⭐⭐⭐ Numerical experiments cover deterministic and Bayesian settings and support the conjectures.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined motivations (finite rounds vs. asymptotic) and a clear main argument.
- Value: ⭐⭐⭐⭐ Provides the first guarantee for revenue convergence rounds in auto-bidding/online learning auctions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Robustness of Langevin Dynamics to Score Function Error](on_the_robustness_of_langevin_dynamics_to_score_function_error.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](../../ICLR2026/learning_theory/the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICML 2026\] Formalizing Learning from Language Feedback with Provable Guarantees](formalizing_learning_from_language_feedback_with_provable_guarantees.md)
- [\[ICML 2026\] On Regret Bounds of Thompson Sampling for Bayesian Optimization](on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)
- [\[ICLR 2026\] A Faster Parameter-Free Regret Matching Algorithm](../../ICLR2026/learning_theory/a_faster_parameter-free_regret_matching_algorithm.md)

</div>

<!-- RELATED:END -->

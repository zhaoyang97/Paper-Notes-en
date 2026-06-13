---
title: >-
  [Paper Note] Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions
description: >-
  [ICML 2026][Optimization][Procurement Auctions] Ours introduces the first truthful mechanism, BFM-SWM, with provable approximation guarantees for submodular social welfare maximization in procurement auctions under budge…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Procurement Auctions"
  - "Budget-Feasible"
  - "Submodular Social Welfare"
  - "Descending Clock Auction"
  - "Approximation Ratio"
date: 2026-05-08
content_hash: 4ad8a4c1335951e6
---

# Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions

**Conference**: ICML 2026  
**arXiv**: [2605.00411](https://arxiv.org/abs/2605.00411)  
**Code**: https://anonymous.4open.science/r/BFM-SWM (Available, anonymous repository)  
**Area**: Mechanism Design / Algorithmic Game Theory / Submodular Optimization  
**Keywords**: Procurement Auctions, Budget-Feasible, Submodular Social Welfare, Descending Clock Auction, Approximation Ratio

## TL;DR
Ours introduces the first truthful mechanism, BFM-SWM, with provable approximation guarantees for submodular social welfare maximization in procurement auctions under budget constraints and private costs. By employing a descending clock auction with geometrically increasing thresholds, single-point protection, and a price/payment ratio parameter $\beta$ to ensure non-negative surplus and budget feasibility, it achieves a 0.0328-approximation for general submodular functions and a 0.0877-approximation for monotone submodular functions. As a byproduct, BFM-VM improves the best deterministic approximation ratio for valuation maximization from 1/64 to $1/(12+4\sqrt{3})\approx 0.0528$, while reducing running time from $\mathcal{O}(n^2\log n)$ to $\mathcal{O}(n\log n)$.

## Background & Motivation

**Background**: Procurement auctions—where a single buyer with a budget $B$ procures items from $n$ sellers with private costs $c(u)$—are widely applied in AI markets such as crowdsourcing, influence maximization, industrial procurement, and data acquisition. Following the pioneering work on budget-feasible mechanisms by Singer (2010), the mainstream goal has been submodular valuation maximization $\max_S v(S)$ s.t. $p(S)\le B$. Currently, the best randomized approximation for general submodular functions is 0.0856 (Han 2025), and the best deterministic is 1/64 (Balkanski SODA 2022).

**Limitations of Prior Work**: (1) Deng et al. 2025 (ICML) were the first to shift the objective to the more economically meaningful "social welfare" $v(S)-c(S)$ (gains-from-trade), but their mechanism **directly abandons budget feasibility**—sellers may receive total payments exceeding the buyer's budget, which is impractical for real-world procurement. (2) The strongest deterministic 1/64 mechanism for valuation maximization (Balkanski 2022) requires $\mathcal{O}(n^2\log n)$ time and relies on complex greedy and unconstrained submodular maximization subroutines.

**Key Challenge**: The welfare objective $v(\cdot)-c(\cdot)$ can be negative, and because $c(\cdot)$ is private information, **it cannot be directly queried by the mechanism**. These two factors render all existing budget-feasible mechanisms (which assume non-negative and observable objective functions) ineffective. Nikolakaki (2021) proved that even if costs are public, a constant multiplicative approximation for welfare is impossible in polynomial time; thus, a weak approximation $v(S)-c(S)\ge\gamma_w\cdot v(O)-c(O)$ must be adopted.

**Goal**: To fill the gap in "Welfare Maximization + Budget Feasibility" by designing a mechanism that simultaneously satisfies truthfulness, individual rationality, non-negative seller surplus, and budget feasibility, while providing a provable lower bound for social welfare.

**Key Insight**: Instead of directly calculating the true welfare of a candidate set (impossible due to private costs), the authors use a **geometrically increasing threshold $\rho_t$ as a benchmark proxy for welfare**. In each round, the threshold is multiplied by $\alpha$, and prices descend according to $v(u\mid S)/(\beta+\rho_t/B)$. They also introduce a "single-point candidate $u^*$" to protect high-value individual sellers and utilize a price/payment ratio parameter $\beta$ to enforce $v(S)\ge\beta p(S)$ for ensuring non-negative surplus.

**Core Idea**: By replacing "welfare evaluation" with "welfare proxy via thresholds" + "parallel single-point and multi-candidate sets" + "enforced price/payment ratios," the mechanism bypasses the unobservability of private costs within a descending clock auction framework while strictly maintaining both budget and surplus constraints.

## Method

### Overall Architecture
The BFM-SWM mechanism (Algorithm 1) is a **descending clock auction**. It first collects all sellers who accept an initial price $B$ into an active set $R$. It initializes a threshold $\rho_0=\epsilon/\alpha$ and a single-point candidate $u^*=\varnothing$. The mechanism enters a multi-round loop: in each round $t$, the threshold is increased by a factor $\alpha$ ($\alpha>1$). It maintains $\ell\in\{1,2\}$ parallel candidate sets $\{S_{i,t}\}$. It iterates through sellers in $R$: finding the candidate set $S_{j,t}$ with the maximum marginal gain and lowering the seller's price to $\min\{p(u), v(u\mid S_{j,t})/(\beta+\rho_t/B)\}$. If adding $u$ causes the surplus of $S_{j,t}$ to exceed $\rho_t$, $u$ is moved to the single-point candidate $u^*$ and the process interrupts; otherwise, $u$ is added to $S_{j,t}$. The auction terminates when all active sellers are included in candidates from the last two rounds, finally outputting the $S^*$ with the maximum welfare from $\{S_{i,M-1}, S_{i,M}\}\cup\{u^*\}$.

### Key Designs

1. **Geometrically Increasing Threshold $\rho_t=\alpha^t\cdot\rho_0$ as Welfare Proxy**:
    - **Function**: Provides a monotonically tightening hurdle to determine prices and prune candidates without querying actual welfare.
    - **Mechanism**: The price rule explicitly incorporates $\rho_t$ in the denominator: $p(u)\leftarrow\min\{p(u), v(u\mid S_{j,t})/(\beta+\rho_t/B)\}$. Thus, higher thresholds lead to lower prices, and the threshold acts as a break condition ($v(S_{j,t}\cup\{u\})-p(\ldots)>\rho_t$) to bound the welfare. Theoretical analysis (Lemma 4.5) proves $\rho_M\le 2\alpha(v(S^*)-p(S^*))$, anchoring the final threshold to the output welfare.
    - **Design Motivation**: Private costs make direct welfare calculations impossible, but marginal values are observable. Using a threshold that tightens at a controlled rate ($\alpha$) approximates welfare while ensuring high-value candidates are not lost.

2. **"Temporary Protection" for Single-Point Candidate $u^*$**:
    - **Function**: Captures individual sellers with exceptionally high value and exempts them from multiple rounds of price drops.
    - **Mechanism**: When a seller $u$ joining a candidate set would cause the surplus to cross the threshold, they are moved to $u^*$ instead of $S_{j,t}$. $u^*$ is not subject to subsequent price-drop rules, preventing high-value items from being forced out by threshold inflation. $u^*$ is included in the final selection pool for $S^*$.
    - **Design Motivation**: Submodularity combined with increasing thresholds can "price out" high-value individual elements. Reserving a slot as a "high-value protection chamber" prevents the mechanism from discarding critical single points.

3. **Price/Payment Ratio Parameter $\beta>1$ to Enforce Non-Negative Surplus**:
    - **Function**: Ensures any selected subset satisfies $v(S)\ge\beta\cdot p(S)$, which implies $v(S)-p(S)\ge(1-1/\beta)v(S)\ge 0$.
    - **Mechanism**: By placing $\beta$ in the price denominator $v(u\mid S)/(\beta+\rho_t/B)$, every element added to $S_{i,t}$ satisfies $v(u\mid S_{i,t}^u)\ge\beta p(u)$. Summing these yields $v(S_{i,t})\ge\beta p(S_{i,t})$ (Lemma 4.6). The relationship between welfare, valuation, and payment is thus bound by $v(A)\le(v(A)-p(A))/(1-1/\beta)\le(v(A)-c(A))/(1-1/\beta)$.
    - **Design Motivation**: Standard budget-feasible mechanisms only monitor $p(S)\le B$. The welfare goal requires $v(S)\ge p(S)$ for positive buyer surplus. $\beta$ makes this a hard constraint within the mechanism and couples it with the threshold for the approximation ratio.

### Loss & Training
Parameters are determined via theoretical analysis: For general (non-monotone) submodular functions, setting $\ell=2, \alpha=1+\tfrac{2\sqrt{6}}{3}, \beta=4$ results in a 0.0328-approximation (Thm 4.8). For monotone submodular functions, $\ell=1, \alpha=1+\tfrac{\sqrt{6}}{2}, \beta=3$ gives a 0.0877-approximation (Thm 4.10). The running time is $\mathcal{O}(n\log(\text{OPT}/\epsilon))$. The BFM-VM byproduct uses $\ell=2, \alpha=1+\sqrt{3}, \beta=0$, achieving a $1/(12+4\sqrt{3})$-approximation for valuation maximization in $\mathcal{O}(n\log n)$ time.

## Key Experimental Results

### Main Results
Experiments were conducted on SNAP datasets for influence maximization (Slashdot: 77K nodes, 905K edges; Email: 265K nodes, 420K edges; Epinions: 131K nodes, 841K edges), measuring social welfare $v(S)-c(S)$ and oracle query counts.

| Application / Baseline | Welfare relative to BFM-SWM | Query Counts |
|---|---|---|
| Deng-Distorted / Deng-ROI / Deng-CostScaled (Baselines) | 0.04× – 0.82× (Varies significantly by budget/dataset) | Usually more |
| BFM-SWM (Ours) | **1.00× (Benchmark)** | Usually fewer |
| Avg. Improvement | **4.49×** | – |
| Max Improvement | **26.41×** | – |
| Min Improvement | **1.22×** | – |

Appendix C also showcases BFM-VM's advantage over SOTAs like Balkanski 2022 in crowdsourcing valuation maximization.

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Full BFM-SWM (General submodular, $\ell=2$) | 0.0328-approx | Primary result |
| Monotone special case ($\ell=1$) | **0.0877-approx** | Exploiting monotonicity to remove the second candidate sequence |
| BFM-VM (Byproduct, Valuation maximization) | $1/(12+4\sqrt{3})\approx 0.0528$ | **3.38× improvement** over Balkanski 2022 (1/64) |
| BFM-VM Time Complexity | $\mathcal{O}(n\log n)$ | Reduced by a factor of $n$ compared to Balkanski 2022's $\mathcal{O}(n^2\log n)$ |
| Without $u^*$ | Loss of $\rho_M$ upper bound in Lemma 4.5 | Misses high-value single sellers; approximation ratio fails |
| Without $\beta$ (i.e., $\beta=0$, for val. max only) | Loss of non-negative surplus guarantee | Welfare may become negative |

### Key Findings
- Baselines, even with prefix-truncation to meet budgets, lack theoretical guarantees, leading to near-zero or negative welfare at many budget points, while BFM-SWM consistently maintains positive welfare with a 4.49× average advantage.
- Coupling the threshold $\rho_M$ to the output welfare $S^*$ (Lemma 4.5) is the critical step in the analysis—it transforms a geometrically increasing threshold into an upper bound on the final welfare.
- BFM-VM's speedup primarily comes from using threshold-based filtering to construct disjoint candidate sets, bypassing heavy subroutines like unconstrained submodular maximization.

## Highlights & Insights
- Successfully addressed a longstanding "difficult" open problem: a submodular mechanism that satisfies budget feasibility, welfare goals, truthfulness, individual rationality, and non-negative surplus. This provides the first engineering-viable truthful mechanism for AI market scenarios (data procurement, crowdsourcing).
- The "triad" of geometrically increasing thresholds, single-point protection, and price/payment ratio parameters is a reusable mechanism-design template for any setting involving budget-feasible goals and private costs.
- The byproduct BFM-VM improves the strongest deterministic approximation for valuation maximization (unchanged for 12 years) from 1/64 to $1/(12+4\sqrt{3})$, while simultaneously reducing the complexity by a factor of $n$.
- Choosing a descending clock auction (rather than sealed-bid) provides obvious strategyproofness, making it more robust against manipulation than Deng 2025.

## Limitations & Future Work
- The 0.0328 approximation ratio is still far from the optimal results under public costs (1−1/e≈0.632), partly due to the impossibility result of Nikolakaki 2021, but room for improvement exists.
- Experiments were limited to influence maximization and crowdsourcing; they did not cover AI application scenarios like data markets or API pricing, and performance on complex valuation oracles beyond coverage functions is unknown.
- The rates $\alpha$ and $\beta$ depend on general/monotone constants (e.g., $1+2\sqrt{6}/3$, $1+\sqrt{6}/2$), requiring careful selection during implementation.
- $\ell$ was restricted to 1 or 2; whether additional parallel candidate sequences could further improve the approximation ratio remains an open question.

## Related Work & Insights
- **vs Deng et al. 2025**: They pioneered welfare maximization but abandoned budget feasibility using a sealed-bid approach. Ours provides the first budget-feasible welfare mechanism using a descending clock auction to achieve obvious strategyproofness and better empirical performance.
- **vs Balkanski et al. 2022 (SODA)**: They achieved a 1/64 deterministic valuation approximation. Our byproduct BFM-VM uses simpler threshold filtering to reach $1/(12+4\sqrt{3})$ in $\mathcal{O}(n\log n)$ time.
- **vs Distorted/Cost-Scaled/ROI Greedy**: These are regularized submodular algorithms for public costs. This work "mechanism-izes" them for private costs and provides corresponding approximation ratios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First truthful mechanism for welfare + budget; breaks a 12-year-old record for deterministic valuation approximation.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive on SNAP datasets; includes side product comparisons, though coverage of AI market types remains limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly explains the mapping between challenges (private/negative) and techniques (thresholds/protection/$\beta$).
- Value: ⭐⭐⭐⭐⭐ A significant advancement for the AGT community; provides a practical truthful mechanism for industrial data and crowdsourcing platforms.

<div class="related-papers" markdown="1">
<!-- RELATED:END --></div>

## Related Papers

- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICML 2026\] URS：统一的神经路由求解器](urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICML 2026\] URS: A Unified Neural Routing Solver](urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)

</div>

<!-- RELATED:END -->

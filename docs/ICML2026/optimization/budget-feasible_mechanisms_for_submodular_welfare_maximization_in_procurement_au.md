---
title: >-
  [Paper Note] Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions
description: >-
  [ICML 2026][Optimization][Procurement Auction] This work is the first to provide a truthful mechanism with approximation guarantees for submodular social welfare maximization in procurement auctions with "budget constrai…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Procurement Auction"
  - "Budget Feasibility"
  - "Submodular Social Welfare"
  - "Descending Clock Auction"
  - "Approximation Ratio"
date: 2026-05-08
content_hash: 86e34ee2e9b96da5
---

# Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions

**Conference**: ICML 2026  
**arXiv**: [2605.00411](https://arxiv.org/abs/2605.00411)  
**Code**: https://anonymous.4open.science/r/BFM-SWM (available, anonymous repository)  
**Area**: Mechanism Design / Algorithmic Game Theory / Submodular Optimization  
**Keywords**: Procurement Auction, Budget Feasibility, Submodular Social Welfare, Descending Clock Auction, Approximation Ratio

## TL;DR
This work is the first to provide a truthful mechanism with approximation guarantees for submodular social welfare maximization in procurement auctions with "budget constraint + private costs": BFM-SWM—a descending clock auction with geometrically increasing thresholds, single-point protection, and a price/payment rate parameter $\beta$—achieves non-negative surplus and budget feasibility, with a 0.0328-approximation for general submodular functions and 0.0877-approximation for monotone submodular functions. As a byproduct, BFM-VM improves the deterministic best approximation for value maximization from 1/64 to $1/(12+4\sqrt{3})\approx 0.0528$, and reduces runtime from $\mathcal{O}(n^2\log n)$ to $\mathcal{O}(n\log n)$.

## Background & Motivation

**Background**: Procurement auctions (a buyer with budget $B$ purchases items from $n$ sellers with private costs $c(u)$) are widely used in crowdsourcing, influence maximization, industrial procurement, data procurement, and other AI markets. Since Singer (2010) pioneered budget-feasible mechanisms, the mainstream objective has been submodular value maximization $\max_S v(S)$ s.t. $p(S)\le B$, with the current best approximation ratios being 0.0856 (randomized, Han 2025) and 1/64 (deterministic, Balkanski SODA 2022) for general submodular functions.

**Limitations of Prior Work**: (1) Deng et al. 2025 (ICML) were the first to shift the objective to the more economically meaningful "social welfare" $v(S)-c(S)$ (net social value, akin to gains-from-trade in bilateral trade), but their mechanism **directly abandoned budget feasibility**—sellers could receive total payments exceeding the buyer's budget, which is infeasible in real procurement. (2) Balkanski 2022's strongest deterministic 1/64 value maximization mechanism requires $\mathcal{O}(n^2\log n)$ time, relying on greedy and unconstrained submodular maximization subroutines, making it complex in practice.

**Key Challenge**: The welfare objective $v(\cdot)-c(\cdot)$ can be negative, and since $c(\cdot)$ is private information, it **cannot be directly queried by the mechanism**—these two facts render all existing budget-feasible mechanisms (which assume non-negative, observable objectives) inapplicable. Nikolakaki 2021 has shown that even with public costs, no polynomial-time algorithm can achieve a constant-factor multiplicative approximation for welfare, so a weaker approximation $v(S)-c(S)\ge\gamma_w\cdot v(O)-c(O)$ must be used.

**Goal**: To fill the gap of "welfare maximization + budget feasibility"—design a mechanism that simultaneously satisfies truthfulness, individual rationality, non-negative seller surplus, and budget feasibility, and provide a provable lower bound on welfare approximation.

**Key Insight**: The authors do not directly compute the true welfare of candidate sets (since costs are private and unobservable), but instead use a **geometrically increasing threshold $\rho_t$ as a welfare proxy benchmark**—multiplying by $\alpha$ each round, with prices decreasing according to $v(u\mid S)/(\beta+\rho_t/B)$; a "single-point candidate $u^*$" is introduced to protect high-value individual sellers, and the price/payment rate parameter $\beta$ enforces $v(S)\ge\beta p(S)$ to ensure non-negative surplus.

**Core Idea**: Replace "welfare evaluation" with "threshold proxy for welfare" + "parallel single-point and multi-candidate selection" + "enforce price/payment rate via $\beta$" within a descending clock auction framework, thus circumventing unobservable private costs while simultaneously enforcing budget and surplus constraints.

## Method

### Overall Architecture
The BFM-SWM mechanism (Algorithm 1) is a **descending clock auction**: all sellers willing to accept offer $B$ are added to the active set $R$; initialize threshold $\rho_0=\epsilon/\alpha$ and single-point candidate $u^*=\varnothing$; enter a multi-round loop—each round $t$ multiplies the threshold by $\alpha$ ($\alpha>1$), maintains $\ell\in\{1,2\}$ parallel candidate sets $\{S_{i,t}\}$, and iterates over sellers in $R$: greedily find the candidate set $S_{j,t}$ with the largest marginal gain and decrease the price as $p(u)\leftarrow\min\{p(u), v(u\mid S_{j,t})/(\beta+\rho_t/B)\}$; if adding $u$ causes the surplus of $S_{j,t}$ to exceed the threshold $\rho_t$, $u$ is moved to the single-point candidate $u^*$ and the process breaks; otherwise, $u$ is added to $S_{j,t}$. The process terminates when all active sellers have entered the candidate sets in the last two rounds, and the final output $S^*$ is the candidate with the highest welfare among $\{S_{i,M-1}, S_{i,M}\}\cup\{u^*\}$.

### Key Designs

1. **Geometrically Increasing Threshold $\rho_t=\alpha^t\cdot\rho_0$ as Welfare Proxy**:

    - Function: Provides a monotonically tightening "threshold" for decision-making and candidate pruning when true welfare cannot be queried.
    - Mechanism: The pricing rule explicitly incorporates $\rho_t$ in the denominator: $p(u)\leftarrow\min\{p(u), v(u\mid S_{j,t})/(\beta+\rho_t/B)\}$, so higher thresholds mean lower prices and easier seller exit; the threshold also serves as a break condition ($v(S_{j,t}\cup\{u\})-p(\ldots)>\rho_t$ triggers a stop), providing an upper bound on candidate welfare. Theoretical analysis (Lemma 4.5) proves $\rho_M\le 2\alpha(v(S^*)-p(S^*))$, thus anchoring the "final threshold" to the mechanism's output welfare.
    - Design Motivation: Private costs make "computing welfare" impossible, but "computing marginal value" is feasible; a threshold with controllable rate ($\alpha$) can approximate welfare and ensure high-value candidates are not missed.

2. **Temporary Protection for Single-Point Candidate $u^*$**:

    - Function: Captures "individual sellers with extremely high value" and provides exemption from multi-round price reductions.
    - Mechanism: When adding a seller $u$ to a candidate set would cause surplus to exceed the threshold, $u$ is moved to $u^*$ instead of being absorbed into $S_{j,t}$; $u^*$ is not subject to subsequent price reductions, preventing high-value items from being forced out as the threshold increases. The final candidate set $S^*$ also includes $\{u^*\}$.
    - Design Motivation: Submodularity + increasing thresholds + multiple candidates can cause "single, particularly valuable elements" to be priced out in later rounds; reserving a slot as a "high-value protection chamber" prevents the mechanism from losing critical singletons.

3. **Price/Payment Rate Parameter $\beta>1$ Enforces Non-Negative Surplus**:

    - Function: Ensures any selected subset satisfies $v(S)\ge\beta\cdot p(S)$, so $v(S)-p(S)\ge(1-1/\beta)v(S)\ge 0$.
    - Mechanism: $\beta$ is directly included in the price denominator $v(u\mid S)/(\beta+\rho_t/B)$; each element added to $S_{i,t}$ satisfies $v(u\mid S_{i,t}^u)\ge\beta p(u)$, and summing gives $v(S_{i,t})\ge\beta p(S_{i,t})$ (Lemma 4.6). The relationship among welfare, value, and payment is thus $v(A)\le(v(A)-p(A))/(1-1/\beta)\le(v(A)-c(A))/(1-1/\beta)$.
    - Design Motivation: Classic budget-feasible mechanisms only ensure $p(S)\le B$, but the welfare objective requires "$v(S)\ge p(S)$" for positive buyer surplus; $\beta$ enforces this as a hard constraint at the mechanism level and couples with the threshold in the approximation analysis.

### Loss & Training
Theoretical analysis selects parameters: for general (non-monotone) submodular functions, set $\ell=2, \alpha=1+\tfrac{2\sqrt{6}}{3}, \beta=4$ to achieve 0.0328-approximation (Thm 4.8); for monotone submodular, $\ell=1, \alpha=1+\tfrac{\sqrt{6}}{2}, \beta=3$ yields 0.0877-approximation (Thm 4.10). Runtime is $\mathcal{O}(n\log(\text{OPT}/\epsilon))$; the BFM-VM byproduct with $\ell=2, \alpha=1+\sqrt{3}, \beta=0$ achieves $1/(12+4\sqrt{3})$-approximation for value maximization in $\mathcal{O}(n\log n)$ time.

## Key Experimental Results

### Main Results
Experiments on SNAP for influence maximization (Slashdot 77K nodes 905K edges, Email 265K nodes 420K edges, Epinions 131K nodes 841K edges), measuring social welfare $v(S)-c(S)$ and oracle query counts.

| Application / Baseline | Welfare Relative to BFM-SWM | Query Count |
|---|---|---|
| Deng-Distorted / Deng-ROI / Deng-CostScaled (baseline) | 0.04× – 0.82× (varies by budget/dataset) | Usually higher |
| BFM-SWM (Ours) | **1.00× (reference)** | Usually lower |
| Average Improvement | **4.49×** | – |
| Maximum Improvement | **26.41×** | – |
| Minimum Improvement | **1.22×** | – |

Appendix C also shows BFM-VM's advantage over Balkanski 2022 and other SOTA in crowdsourcing value maximization.

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Full BFM-SWM (general submodular, $\ell=2$) | 0.0328-approximation | Main result |
| Monotone submodular specialization ($\ell=1$) | **0.0877-approximation** | Monotonicity allows dropping the second candidate sequence |
| BFM-VM (byproduct, value maximization) | $1/(12+4\sqrt{3})\approx 0.0528$ | **3.38× improvement** over Balkanski 2022 (1/64) |
| BFM-VM runtime | $\mathcal{O}(n\log n)$ | Improves over Balkanski 2022's $\mathcal{O}(n^2\log n)$ by a factor of $n$ |
| Remove $u^*$ | Loses Lemma 4.5's upper bound on $\rho_M$ | High-value singleton sellers are lost, approximation fails |
| Remove $\beta$ (i.e., $\beta=0$, only for value maximization) | Loses non-negative surplus guarantee | Welfare may be negative |

### Key Findings
- Even with prefix-truncation to satisfy budget, baselines lack theoretical guarantees, leading to welfare near zero or negative at many budget points, while BFM-SWM consistently maintains positive welfare with a 4.49× average advantage.
- Tying the threshold $\rho_M$ to the output welfare of $S^*$ (Lemma 4.5) is a key analytical step—it converts the "geometrically increasing threshold" into an upper bound on final welfare.
- BFM-VM's speedup mainly comes from using threshold filtering to construct disjoint candidate sets, avoiding Balkanski's heavy unconstrained submodular maximization subroutines.

## Highlights & Insights
- Solves a long-standing open problem—simultaneously achieving budget feasibility, welfare objective, truthfulness, individual rationality, and non-negative surplus for submodular mechanisms—providing the first practically deployable truthful mechanism for future AI market scenarios (data procurement, crowdsourcing, etc.).
- The "geometric threshold + single-point protection + price/payment rate parameter" toolkit is reusable in mechanism design—any scenario with "objectives involving private information and budget feasibility" can adopt this template.
- The BFM-VM byproduct raises the best deterministic approximation for value maximization from the 12-year-old 1/64 to $1/(12+4\sqrt{3})$, while also reducing runtime by a factor of $n$—a clean and significant improvement.
- Choosing descending clock auction (rather than sealed-bid) brings obvious strategyproofness, making it harder to manipulate than Deng 2025's sealed-bid approach.

## Limitations & Future Work
- The 0.0328 approximation is still far from the best possible (1−1/e≈0.632) without budget or private cost constraints, partly due to Nikolakaki 2021's impossibility result, but there is room for improvement.
- Experiments only cover influence maximization and crowdsourcing, not data markets or pricing APIs; all value functions are coverage functions, and performance on more complex valuation oracles is unknown.
- The geometric growth rate $\alpha$ of threshold $\rho_t$ and price/payment rate $\beta$ require careful tuning for general/monotone cases (e.g., $1+2\sqrt{6}/3$, $1+\sqrt{6}/2$), which needs attention in practice.
- $\ell$ is only set to 1 or 2; the theory does not extend to general $\ell$. Whether more candidate sequences can further improve the approximation ratio remains open.

## Related Work & Insights
- **vs Deng et al. 2025**: They first addressed welfare maximization but abandoned budget feasibility and used sealed-bid; this work provides the first mechanism for budget feasibility + welfare objective, switching to descending clock auction for obvious strategyproofness; BFM-SWM significantly outperforms in welfare and efficiency.
- **vs Balkanski et al. 2022 (SODA)**: They achieved deterministic value maximization at 1/64; the BFM-VM byproduct here uses simpler threshold filtering instead of greedy + unconstrained submodular subroutines, achieving $1/(12+4\sqrt{3})$ and $\mathcal{O}(n\log n)$ runtime.
- **vs Distorted Greedy / Cost-Scaled Greedy / ROI Greedy**: These are regularized submodular maximization algorithms (with public costs), unable to handle private costs directly; this work "mechanizes" them via threshold mechanisms and provides corresponding approximation ratios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide a truthful mechanism for welfare objective + budget feasibility, and byproduct breaks a 12-year-old deterministic value maximization approximation record.
- Experimental Thoroughness: ⭐⭐⭐ Three SNAP datasets + complete influence maximization main experiments; appendix adds crowdsourcing value maximization comparison; but only covers one or two application types, not a full survey of AI markets.
- Writing Quality: ⭐⭐⭐⭐⭐ The correspondence between challenges (private + non-negativity) and techniques (threshold proxy + single-point protection + $\beta$) is clearly explained, with stepwise theorem analysis.
- Value: ⭐⭐⭐⭐⭐ A clear advance for the algorithmic game theory community; provides a deployable truthful mechanism for industry (data procurement, crowdsourcing platforms).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](transformed_latent_variable_multi-output_gaussian_processes.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[ICML 2026\] Probing Neural TSP Representations for Prescriptive Decision Support](probing_neural_tsp_representations_for_prescriptive_decision_support.md)

</div>

<!-- RELATED:END -->

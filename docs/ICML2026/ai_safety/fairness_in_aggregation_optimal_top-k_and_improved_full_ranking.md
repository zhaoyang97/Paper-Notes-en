---
title: >-
  [Paper Note] Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking
description: >-
  [ICML 2026][AI Safety][Fair rank aggregation] Under the Spearman footrule distance, the authors prove that the ILP constraint matrix is totally unimodular (TU)…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Fair rank aggregation"
  - "Spearman footrule"
  - "Top-$k$"
  - "Totally Unimodular Matrix"
  - "LP Relaxation"
date: 2026-05-08
content_hash: a212f74209c055bb
---

# Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking

**Conference**: ICML 2026  
**arXiv**: [2605.23265](https://arxiv.org/abs/2605.23265)  
**Code**: https://github.com/Aussiroth/Spearman-FRA  
**Area**: AI Safety / Algorithmic Fairness  
**Keywords**: Fair rank aggregation, Spearman footrule, Top-$k$, Totally Unimodular Matrix, LP Relaxation

## TL;DR
Under the Spearman footrule distance, the authors prove that the ILP constraint matrix is totally unimodular (TU), providing the first polynomial-time optimal algorithm for fair top-$k$ rank aggregation. They further propose a two-step strategy—solving fair top-$k$ first and then completing it into a full permutation via minimum-cost perfect matching—improving the approximation ratio for fair (full) rank aggregation from 3 to 2.

## Background & Motivation

**Background**: Rank aggregation combines multiple preference orderings into a single consensus ranking, serving as a core primitive in scenarios such as recruitment, recommendation, web search, and meta-search. One classic metric is the Spearman footrule, defined as the rank-wise $L_1$ distance $F(\pi_1, \pi_2) = \sum_i |\pi_1(i) - \pi_2(i)|$. It differs from the Kendall-tau distance by at most a factor of 2 (Diaconis & Graham, 1977) but is more computationally efficient, as the unconstrained optimal can be found in polynomial time (Dwork et al., WWW'01).

**Limitations of Prior Work**: Directly applying unconstrained algorithms to scenarios with fairness constraints exacerbates the under-representation of disadvantaged groups. Existing fair versions only achieve a 3-approximation (Wei et al., SIGMOD'22; Chakraborty et al., NeurIPS'22). This involves a meta-algorithm derived from the triangle inequality—finding the nearest fair ranking for each input and then selecting the one with the smallest total distance—which has remained the state-of-the-art for a long time. Furthermore, this 3-approximation only covers full rankings, with no specific guarantees for the top-$k$ case, which is often more relevant in practice (e.g., recruitment and recommendation only consider the top $k$ items).

**Key Challenge**: A massive complexity gap exists between unconstrained and fair versions—while the former is polynomially solvable, the latter remains stuck at a 3-approximation. This gap was explicitly listed as an open problem in Wei et al. The fundamental technical difficulty is that after formulating the problem as an ILP, standard LP relaxation and rounding techniques often violate minority protection or restricted dominance constraints.

**Goal**: To decompose the problem into two sub-problems: (i) find an optimal polynomial-time algorithm for fair top-$k$ rank aggregation; (ii) achieve a better approximation ratio for fair full rank aggregation.

**Key Insight**: Instead of rounding the LP solution (which inevitably violates fairness constraints), the authors prove that the ILP constraint matrix is **totally unimodular (TU)**. This ensures that the LP will always have an integer optimal solution, allowing the ILP optimal to be obtained directly via the ellipsoid method.

**Core Idea**: For fair top-$k$, the exact solution is found using the "TU structure + Ellipsoid method." For fair full ranking, a two-step approach is used: solve for the fair top-$k$ first and then extend it via minimum-cost perfect matching. Combined with a lemma that decomposes the Spearman footrule into leftward and rightward displacements, the approximation ratio is reduced to 2.

## Method

### Overall Architecture

The algorithm takes $d$ candidates partitioned into $g$ groups $G_1, \ldots, G_g$, with given lower bounds $\alpha_a$ and upper bounds $\beta_a$ for each group, along with $n$ input rankings $S \subseteq \mathcal{S}_d$ and an integer $k$.

Fairness Definition $(\bar\alpha, \bar\beta)$-$k$-fair: In the output top-$k$ list, each group $G_a$ must have at least $\lfloor \alpha_a \cdot k \rfloor$ and at most $\lceil \beta_a \cdot k \rceil$ candidates.

Two main research lines:
- **Fair Top-$k$**: Output a top-$k$ list $\tau$ containing $k$ candidates that minimizes $\sum_{\pi \in S} F(\pi, \tau)$ (using the generalization by Fagin et al., where candidates outside $\tau$ are treated as being at rank $k+1$).
- **Fair Full Ranking**: Output a full ranking $\sigma$ of all $d$ candidates to minimize the same objective.

### Key Designs

1. **ILP for Fair Top-$k$ and Proof of Total Unimodularity**:

    - **Function**: Precisely models fair top-$k$ rank aggregation as an ILP and proves the TU property of its LP relaxation matrix to solve the ILP optimally via the ellipsoid method.
    - **Mechanism**: Variables $x_{ij} \in \{0,1\}$ indicate whether candidate $i$ is placed at rank $j$ ($j \le k$). The weight $w_{ij} = \sum_{\pi \in S} |\pi(i) - j|$ represents the cost of this placement; candidates not in the top $k$ incur a cost as if placed at $k+1$. Constraints ensure: each candidate occupies at most one position, each position has exactly one candidate, and group bounds $\lfloor \alpha_a k \rfloor$ / $\lceil \beta_a k \rceil$ are met. The TU property is proven using the characterization by Wolsey-Nemhauser: for any row set $R$, there exists a partition $R = R_1 \cup R_2$ such that the difference in each column is $\le 1$.
    - **Design Motivation**: Standard LP rounding often fails fair constraints (e.g., in fair correlation clustering or fair $k$-clustering). The TU property bypasses rounding, allowing fairness constraints to be **strictly satisfied** rather than "approximately satisfied," which is the key technical lever for moving from a 3-approximation to an exact solution.

2. **Two-step Fair Full Rank Aggregation: Top-$k$ followed by Matching Completion**:

    - **Function**: Extends the optimal fair top-$k$ solution $\tau$ into a 2-approximate fair full ranking $\sigma$.
    - **Mechanism**: First, use algorithm $\mathcal{A}$ to solve a "modified objective"—minimizing $2 \sum_{\pi} \sum_{i \in D_\tau} (\pi(i) - \tau(i)) \cdot \mathbb{1}_{\tau(i) < \pi(i)}$ (counting only leftward displacement multiplied by 2). Since the constraint matrix is identical, TU still holds, allowing for a polynomial solution for the optimal fair top-$k$ list $\tau$. Then, define the "Minimum Cost Top-$k$ List Completion" sub-problem: keep $\tau$ fixed and fill the remaining $d-k$ candidates into positions $k+1, \ldots, d$ to minimize the global objective. This is solvable as a minimum-cost perfect matching on a complete bipartite graph with $2d$ vertices in $O(nd^2 + d^3)$.
    - **Design Motivation**: Since fair full ranking with fair perfect matching is NP-hard (Appendix D), a single-step approach is unfeasible. By concentrating lahat "fairness constraints" in the top-$k$ portion for exact TU solving and treating the extension as an unconstrained minimum-cost matching, both parts become polynomially solvable.

3. **2-approximation Analysis based on Displacement Decomposition**:

    - **Function**: Proves that the output $\sigma$ of the two-step algorithm satisfies $\mathrm{Obj}(\sigma) \le 2 \cdot \mathrm{OPT}$, improving the previous 3-approximation to 2.
    - **Mechanism**: Utilizing the observation by Mathieu-Mauras—that the Spearman footrule equals twice the total "leftward displacement" (since leftward and rightward displacements must be equal), i.e., $F(\pi,\sigma) = 2 \sum_i (\pi(i)-\sigma(i)) \mathbb{1}_{\sigma(i)<\pi(i)}$. Let $L, R$ be the sets of elements in the first $k$ and last $d-k$ positions of $\sigma$, respectively, and $L^*, R^*$ be the corresponding sets for the optimal solution $\sigma^*$. For the first $k$ segment, the optimality of algorithm $\mathcal{A}$ yields $\overleftarrow{\mathrm{Obj}}(\sigma_L) \le \overleftarrow{\mathrm{Obj}}(\sigma^*_{L^*})$. For the last $d-k$ segment, a reference ranking $\tilde\sigma$ is constructed—elements in $R \cap R^*$ follow $\sigma^*$, while elements in $R \setminus R^*$ (which must be in $L^*$) are placed arbitrarily. Since these elements only shift rightward in $\tilde\sigma$ compared to their original positions, it can be proven that $\overleftarrow{\mathrm{Obj}}(\sigma_R) \le \overleftarrow{\mathrm{Obj}}(\tilde\sigma_R) \le \mathrm{OPT}$. Summing both yields $\mathrm{Obj}(\sigma) \le \overleftarrow{\mathrm{Obj}}(\sigma^*_{L^*}) + \mathrm{OPT} \le 2 \mathrm{OPT}$.
    - **Design Motivation**: Proving a 2-factor upper bound directly on $F(\pi,\sigma)$ is difficult. Decomposing it into leftward displacement aligns the analysis with the objective of sub-algorithm $\mathcal{A}$. A side effect of this decomposition is the algorithm's robustness to the choice of metric, yielding a 4-approximation for Kendall-tau.

### Loss & Training
This work presents a combinatorial optimization algorithm rather than a machine learning model; thus, no training is involved. The final execution relies on the ellipsoid method for LP and Edmonds-Karp for minimum-cost perfect matching, resulting in polynomial total complexity (the ILP was solved using Gurobi 12.0.3 in experiments).

## Key Experimental Results

### Main Results

Datasets: (i) Movielens subset—7 users' preference rankings of 268 movies across 8 genres; (ii) Fantasy Football—25 experts' 16-week rankings of 57 players across two conferences. $\bar\alpha = \bar\beta$ is set based on the actual proportions of groups in the input to enforce "proportional fairness."

| Dataset | Algorithm | Gap to Optimal | vs KT (Kendall-tau SOTA) | vs BFI (3-approx SOTA) |
|--------|------|-----------|--------------------------|------------------------|
| Movielens | Ours (Algorithm 3) | 2%–3% | 5%–11% lower | 13%–15% lower |
| Football (week 4) | Ours (Algorithm 3) | ≤ 1% | 1%–2% lower | 5%–10% lower |

### Ablation Study

| Configuration | Dataset | Result |
|------|--------|------|
| Algorithm 1 (Basic 2-approx) | Movielens | Better than BFI but weaker than Algorithm 3 |
| Algorithm 3 (Best of two 2-approx) | Movielens / Football | Consistently the best performer |
| Metric changed to Kendall-tau (Theoretical 4-approx) | Movielens | 8%–10% lower than BFI (3-approx), within 2% of KT SOTA, occasionally better |
| Metric changed to Kendall-tau | Football | At most 2% worse than KT SOTA, still 4%–10% better than BFI |

### Key Findings
- While the theoretical bound is a 2-approximation, empirical results consistently show proximity to the optimal (≤ 3% or even ≤ 1%), suggesting the analysis might be loose and the constant could be tighter.
- Taking the better of two 2-approximation algorithms leads to stable improvements, indicating the "worst cases" for different algorithms do not overlap.
- The algorithm is robust to different metrics: even though designed for Spearman, it outperforms BFI and remains comparable to KT on Kendall-tau, proving the two-step "fair top-$k$ + matching completion" framework is generally applicable to $L_1$-style metrics.

## Highlights & Insights
- **Bypassing Fair LP Rounding via TU**: In fair clustering and matching literature, fairness constraints are historically problematic for rounding. This work is the first to directly "embed" fairness constraints into an ILP with integer optimal solutions by proving TU, offering a highly transferable approach.
- **Objective Function Hacking for TU**: By switching from the exact objective to one counting only leftward displacement, the TU property of the constraint matrix is preserved while the new objective aligns perfectly with the inequalities required for the 2-approximation analysis.
- **Displacement Decomposition + Reference Ranking**: The technique of splitting the optimal ranking $\sigma^*$ into $R \cap R^*$ and $R \setminus R^*$ components to localize the loss between fair and unconstrained versions is a valuable trick for analysis under various metrics.

## Limitations & Future Work
- **Non-tight 2-approximation**: There is a 10×–100× gap between the theoretical bound and empirical performance. Lower bound analysis remains an open problem.
- **Narrow Fairness Definition**: The study only considers top-$k$ proportional fairness, ignoring more complex real-world settings like block fairness, overlapping groups, or latent attributes.
- **Metric Limitations**: The analysis depends on the symmetry of leftward and rightward displacements in the Spearman footrule; it does not yet cover Ulam or weighted Kendall-tau metrics. The 4-approximation for Kendall-tau is theoretically inferior to the 2-approximation by (Chakraborty et al., 2025b).
- **NP-hard Fair Perfect Matching**: This suggests an "end-to-end fair full ranking" approach is blocked; future breakthroughs may require relaxing fairness constraints or changing the problem structure.

## Related Work & Insights
- **vs BFI (Chakraborty et al., NeurIPS'22 / Wei et al., SIGMOD'22)**: BFI achieves a 3-approximation by finding the nearest fair ranking for each input list. This work provides the first theoretical breakthrough in this direction by redesigning the framework to reach a 2-approximation.
- **vs KT (Chakraborty et al., 2025b)**: KT is the SOTA for Kendall-tau (18/7-approx, implying a 36/7-approx for Spearman). This algorithm is strictly superior on Spearman and competitive on Kendall-tau.
- **vs (Celis et al., NeurIPS'18) closest fair ranking**: Their work finds the "closest fair ranking to a single input ranking," whereas this work finds the "fair consensus ranking for a group"—a harder problem. This paper shows that using the former as a building block is insufficient for good approximation ratios.
- **Heuristic**: Combining the TU property with the ellipsoid method can be applied to other "hard-to-round" fair combinatorial optimizations (e.g., fair matroid intersection, fair flow).

## Rating
- Novelty: ⭐⭐⭐⭐ Using total unimodularity to bypass rounding is a fresh perspective and solves a known open problem.
- Experimental Thoroughness: ⭐⭐⭐ Two standard datasets and cross-metric tests are sufficient; however, datasets are small ($d \le 268$), lacking larger-scale stress tests.
- Writing Quality: ⭐⭐⭐⭐ Definitions and lemmas are clear, and the TU proof construction is detailed; however, it has a high entry barrier for readers not familiar with combinatorial optimization.
- Value: ⭐⭐⭐⭐ The first optimal fair top-$k$ algorithm and the improvement of the approximation ratio from 3 to 2 represent significant progress in fair ranking literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency](fair_decisions_from_calibrated_scores_achieving_optimal_classification_while_sat.md)
- [\[ICML 2026\] COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs](copf_an_online_framework_for_deployment-stable_counterfactual_fairness_in_evolvi.md)
- [\[AAAI 2026\] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses](../../AAAI2026/ai_safety/an_improved_privacy_and_utility_analysis_of_differentially_p.md)

</div>

<!-- RELATED:END -->

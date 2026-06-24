---
title: >-
  [Paper Note] Bi-Criteria Metric Distortion
description: >-
  [ICLR2026][Learning Theory][Metric Distortion] This paper generalizes the "metric distortion" framework from "selecting a single winner" to "committees of $k$ candidates compared against the optimal single candidate." It proves that on a 1D line, only 2 candidates (for the sum objective) or 4 candidates (for the maximum objective) are sufficient to eliminate distortion (1-distortion = 1). In contrast, this is unachievable in 2D Euclidean or tree metrics even with $m-1$ candid…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Computational Social Choice"
  - "Metric Distortion"
  - "Social Choice Theory"
  - "Voting Mechanisms"
  - "Committee Selection"
  - "Approximation Bounds"
date: 2026-05-08
content_hash: e0d693091a5db330
---

# Bi-Criteria Metric Distortion

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=QBgHVmvN5S](https://openreview.net/forum?id=QBgHVmvN5S)  
**Code**: TBD  
**Area**: Learning Theory / Computational Social Choice  
**Keywords**: Metric Distortion, Social Choice Theory, Voting Mechanisms, Committee Selection, Approximation Bounds

## TL;DR
This paper generalizes the "metric distortion" framework from "selecting a single winner" to "committees of $k$ candidates compared against the optimal single candidate." It proves that on a 1D line, only 2 candidates (for the sum objective) or 4 candidates (for the maximum objective) are sufficient to eliminate distortion (1-distortion = 1). In contrast, this is unachievable in 2D Euclidean or tree metrics even with $m-1$ candidates, revealing a sharp divide between line metrics and higher-dimensional metrics.

## Background & Motivation

**Background**: A fundamental problem in social choice theory is "selecting representatives based on voter preferences." Ideally, voters provide cardinal utility (specific numerical values for each candidate), but in practice, humans struggle to quantify these precisely. Consequently, modern voting systems typically collect only ordinal rankings. By modeling voters and candidates as points in a metric space where "the distance from a voter to a candidate" is the cost of selection, the **metric distortion** framework (Anshelevich et al. 2018) measures the efficiency loss from having only rankings without true distances. For a voting mechanism $f$, its distortion is the worst-case ratio of the cost of $f$'s solution to the optimal cost.

**Limitations of Prior Work**: In the classic single-winner setting, this loss constitutes a hard limit. For deterministic mechanisms, the optimal distortion is exactly $3$ (Gkatzelis-Halpern-Shah, FOCS'20) for any metric, and this cannot be reduced even on a simple 1D line. Introducing randomization only lowers the bound to $2.112$ (Charikar-Ramakrishnan, SODA'22) with an upper bound of $2.753$ (SODA'24 Best Paper). Essentially, as long as one insists on "selecting only one winner," distortion is trapped at the magnitude of $3$ (deterministic).

**Key Challenge**: The inability to reduce distortion stems from the extreme restriction of outputting a single candidate—ordinal information is insufficient to pinpoint that unique optimal point in the worst case. A natural relaxation is to allow selecting more candidates. In clustering or facility location, "permitting multiple centers to approximate the optimal cost" (bi-criteria approximation) is a standard tool; however, those works assume the **metric is known**, whereas the difficulty in metric distortion lies in the metric being unknown, with only rankings available.

**Goal**: To introduce bi-criteria approximation into metric distortion. The mechanism is allowed to select a committee $S$ of size $k$, where a voter's cost is the "distance to the nearest candidate in $S$," $d(v,S)=\min_{c\in S} d(v,c)$. Crucially, the **baseline remains the cost of the optimal single candidate**. This ratio is termed **1-distortion**:

$$\text{1-distortion}(f) := \sup_{E}\ \sup_{d \triangleright E}\ \frac{\text{cost}(f(E))}{\mathrm{OPT}},\qquad \mathrm{OPT}=\min_{c\in C}\text{cost}(c)$$

where $\text{cost}$ is either the sum cost $\text{cost}_s(S)=\sum_{v} d(v,S)$ (utilitarian objective) or the maximum cost $\text{cost}_m(S)=\max_v d(v,S)$ (egalitarian objective), and $d\triangleright E$ denotes that the metric $d$ is consistent with the rankings $E$.

**Key Insight**: Compared to existing "$k$-committee election" works (e.g., Caragiannis et al. 2022), the critical difference in this paper lies in the **choice of the baseline**. Previous work set the baseline as the "optimal $k$-committee," which resulted in a distortion that is unbounded in the worst case for $k\ge 3$ under $q=1$ costs—making all selection rules appear equally poor and providing no theoretical guidance. By switching to the "optimal single candidate" as the baseline, different selection rules can be distinguished, making the theory meaningful again.

**Core Idea**: Use a bi-criteria perspective—selecting multiple candidates but comparing against a single optimal candidate—to tackle metric distortion. The research asks "how many candidates are needed to reduce distortion to 1" and draws a clear boundary of possibility between line and high-dimensional metrics.

## Method

### Overall Architecture

The paper consists of a paired set of "upper bound constructions + lower bound impossibility" proofs. The goal is to provide tight 1-distortion bounds for each combination of (objective function × metric space × committee size). The input is the ordinal preferences of voters (including point order recoverable from rankings in the 1D case), and the output is a deterministic, ranking-dependent committee selection rule, along with its guaranteed 1-distortion value, accompanied by a matching lower bound.

The results follow two main tracks:

- **Upper Bounds (Possibility)**: First, they use the classic result of Elkind-Faliszewski (2014) to recover the left-to-right order of voters and candidates from ordinal preferences in a line metric. Then, they employ "greedy positioning" along this order—placing candidates at key positions near the median voter or the leftmost/rightmost voters—to bound the cost using the triangle inequality.
- **Lower Bounds (Impossibility)**: They construct families of instances that are completely indistinguishable via rankings but have vastly different true distances. This forces any deterministic rule—which must output the same committee for all such instances—to miss the unique optimal candidate in at least one instance, thereby incurring a high cost.

### Key Designs

**1. Sum Objective on a Line: Using Median Voters to Bracket the Optimal Candidate in a 2-Candidate Committee**

The pain point for the sum objective is that the optimal candidate is unidentifiable under unknown distances. The key observation is that for sum costs, the **median voter is a "balance point"**: moving a candidate toward the median voter reduces the distance to the majority more than it increases the distance to the minority, meaning the total cost does not increase. This leads to Lemma 6: among the candidates immediately to the left and right of the median voter, one must be the optimal single candidate.

The construction (Theorem 5) proceeds in two steps. First, it uses rankings to sort voters and candidates, discards candidates who are never "closest" to any voter, and takes the candidate $c$ closest to the median along with its immediate neighbors in the candidate sequence to get a 3-candidate set. Lemma 6 ensures the optimal candidate is within this triplet (Lemma 7). Second, by dividing voters into segments based on their closest candidate among the three, it is shown that one of the outer candidates can be safely removed without losing the optimal candidate's efficiency. Thus, **2 candidates on a line ensure 1-distortion = 1** (Theorem 1/5), which is stronger than "distortion of 1"—it means the committee actually contains the optimal candidate.

For general metrics (Theorem 8), a rule selects all but the "least popular first-choice" candidate among $m-1$ candidates. This yields a bound of $1+\frac{2}{m-1}$.

**2. Max Objective on a Line: Using Left/Right Voters as Anchors to Iteratively Compress Distortion to 1**

The max objective focuses on the "worst-off voter." The paper treats the leftmost and rightmost voters $v_l, v_r$ as anchors. Lemma 9 shows that the optimal single candidate's cost $\mathrm{OPT} \ge D/2$ where $D=d(v_l, v_r)$. Picking the closest candidates to these anchors $\{c_l, c_r\}$ yields a 2-approximation (Theorem 12). However, this doesn't reach distortion 1 if the anchors are outside the voter range. By **adding internal candidates** and subdividing the distance, the bound improves: 3 candidates $\to$ 1.5 (Theorem 13); 4 candidates $\to$ distortion 1 (Theorem 14). Overall, the relation is $\text{1-distortion} = 3-\frac{k}{2}$ for $k \in \{2,3,4\}$.

**3. Lower Bounds for High-Dimensional/Tree Metrics: Indistinguishable Instances**

To prove high-dimensional impossibility, the paper constructs $m+1$ metric instances $I_0, I_1, \dots, I_m$ with identical rankings but different distances. In $I_j$, voter $v_j$ and candidate $c_j$ are pushed far away. Since any deterministic rule outputs the same subset $S$ of size $m-1$, it must miss some $c_j$. In that instance $I_j$, the cost is dominated by $v_j$'s distance to $S$, leading to a bound of $1+\frac{2}{m-1}$ for the sum objective (Theorem 16/17) and 3 for the max objective (Theorem 18), even with $m-1$ candidates.

**4. Impossibility for General $k$ \& Tight Max Lower Bounds on a Line**

To show the difficulty of the sum objective for general $k$, Theorem 15 uses binary string encoding to show that distortion remains unbounded if the committee size $k$ is compared against an optimal committee of size $O(\log k)$. Finally, for the max objective on a line, cyclic preferences among 3 voters/candidates are used to prove a tight lower bound of $2-\varepsilon$ for $k=2$ and $3/2-\varepsilon$ for $k=3$ (Theorem 19/20).

### Loss & Training
N/A. This is a purely theoretical paper. There is no training or loss function. The "optimization target" refers to the social costs (sum and max), focusing on proving upper and lower bounds for 1-distortion.

## Key Experimental Results

### Main Results: Tight 1-Distortion Bounds

| Objective | Metric Space | Committee Size (Total $m$) | Lower Bound | Upper Bound |
| :--- | :--- | :--- | :--- | :--- |
| Sum (Utilitarian) | 1D Line | $\ge 2$ | 1 | **1** (Ours, 2 candidates contain OPT) |
| Sum | Single Winner $k=1$ | Any | 3 | 3 (Anshelevich 2018 / Gkatzelis 2020) |
| Sum | 2D / Tree | $m-1$ | $1+\frac{2}{m-1}$ | $1+\frac{2}{m-1}$ (Ours, tight) |
| Max (Egalitarian) | 1D Line | $k \in \{2,3,4\}$ | $3-\frac{k}{2}$ | $3-\frac{k}{2}$ (Ours: $2, 1.5, 1$) |
| Max | Single Winner $k=1$ | Any | 3 | 3 |
| Max | 2D | $m-1$ | 3 | 3 (Ours, more candidates don't help) |

### Key Findings
- **The Line vs. High-Dimension Divide**: On a line, a constant number of candidates (2 for sum, 4 for max) can eliminate distortion. In 2D Euclidean or tree metrics, distortion remains high even with $m-1$ candidates. Dimensionality, not the number of candidates, is the primary obstacle.
- **Sum is Easier to Compress than Max**: On a line, the sum objective reaches distortion 1 with just 2 candidates, whereas the max objective requires 4. In 2D, the max objective is stuck at 3, while the sum objective at least approaches 1 as $m$ increases ($1+\frac{2}{m-1}$).
- **Correct Baselines Make Theory Meaningful**: Using the "optimal single candidate" as a denominator allows for the differentiation of voting rules that were previously deemed equally "unbounded" under the "optimal $k$-committee" baseline.

## Highlights & Insights
- **The "Bi-criteria + Single Baseline" shift is a subtle but powerful perspective change**: By simply changing the definition of the ratio, a problem stuck at distortion 3 is transformed into a manageable trade-off between committee size and approximation.
- **Clean Lower Bound Templates**: The construction of $m+1$ instances that are indistinguishable by rankings provides a unified engine for proving impossibility results for committees of size $m-1$.
- **Leveraging the Line Order**: The ability to recover the 1D point order from rankings is the "lever" that allows geometric intuition (like median points and anchors) to be implemented as ranking-based algorithms.

## Limitations & Future Work
- **Deterministic Mechanisms Only**: All lower bounds apply to deterministic algorithms. Whether randomization can break the 3 or $1+\frac{2}{m-1}$ barriers in high dimensions remains an open question.
- **Characterization of Applicable Metric Spaces**: The paper identifies lines (yes) and 2D/Trees (no), but intermediate spaces (low treewidth, low-dimensional manifolds, bounded doubling dimension) are unexplored.
- **Theoretical Existence vs. Empirical Runtime**: There is no discussion of computational efficiency or robustness in practical voting scenarios.
- **Philosophical Baseline Question**: While the single-candidate baseline is theoretically useful, in real committee elections, users might naturally want to compare against the optimal committee.

## Related Work & Insights
- **vs. Single-Winner Distortion (Gkatzelis 2020 / Charikar 2022/2024)**: While they struggle to lower distortion beyond 2.7–3.0 with one winner, this paper shows that "expanding output" is a viable way to bypass the single-winner wall.
- **vs. k-Committee Elections (Caragiannis et al. 2022)**: Previous work found unbounded distortion for $k \ge 3$ because they used the optimal committee as a baseline. This paper’s finite bounds are a direct result of changing that denominator.
- **vs. Bi-criteria in Facility Location**: This work essentially answers "what remains of bi-criteria approximation's power when distances are unknown and only rankings are visible."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ (Pure theory, but bounds are tight)
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐

## Related Papers

- [\[ICLR 2026\] Bi-Lipschitz Autoencoder With Injectivity Guarantee](bi-lipschitz_autoencoder_with_injectivity_guarantee.md)
- [\[ICLR 2026\] Persistence Spheres: Bi-Continuous Representations of Persistence Diagrams](persistence_spheres_bi-continuous_representations_of_persistence_diagrams.md)
- [\[ICLR 2026\] An Optimal Diffusion Approach to Quadratic Rate-Distortion Problems: New Solution and Approximation Methods](an_optimal_diffusion_approach_to_quadratic_rate-distortion_problems_new_solution.md)
- [\[ICLR 2026\] Metric $k$-Clustering using only Weak Comparison Oracles](metric_k-clustering_using_only_weak_comparison_oracles.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bi-Lipschitz Autoencoder With Injectivity Guarantee](bi-lipschitz_autoencoder_with_injectivity_guarantee.md)
- [\[ICLR 2026\] Persistence Spheres: Bi-Continuous Representations of Persistence Diagrams](persistence_spheres_bi-continuous_representations_of_persistence_diagrams.md)
- [\[ICLR 2026\] An Optimal Diffusion Approach to Quadratic Rate-Distortion Problems: New Solution and Approximation Methods](an_optimal_diffusion_approach_to_quadratic_rate-distortion_problems_new_solution.md)
- [\[ICLR 2026\] Metric $k$-Clustering using only Weak Comparison Oracles](metric_k-clustering_using_only_weak_comparison_oracles.md)
- [\[ICML 2026\] Realizable Bayes-Consistency for General Metric Losses](../../ICML2026/learning_theory/realizable_bayes-consistency_for_general_metric_losses.md)

</div>

<!-- RELATED:END -->

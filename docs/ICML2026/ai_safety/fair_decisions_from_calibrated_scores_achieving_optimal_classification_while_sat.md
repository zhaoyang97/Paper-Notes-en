---
title: >-
  [Paper Note] Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency
description: >-
  [ICML 2026][AI Safety][Algorithmic Fairness] This paper addresses a long-neglected issue: "even if scores are fully group-calibrated…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Algorithmic Fairness"
  - "Sufficiency"
  - "Predictive Parity"
  - "Calibrated Scores"
  - "Post-processing"
date: 2026-05-08
content_hash: a911c375bb104420
---

# Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency

**Conference**: ICML 2026  
**arXiv**: [2602.07285](https://arxiv.org/abs/2602.07285)  
**Code**: https://github.com/etambenger/fair-decisions-from-calibrated-scores (Paper commits to open source)  
**Area**: AI Safety / Algorithmic Fairness  
**Keywords**: Algorithmic Fairness, Sufficiency, Predictive Parity, Calibrated Scores, Post-processing

## TL;DR
This paper addresses a long-neglected issue: "even if scores are fully group-calibrated, applying a single threshold violates sufficiency (predictive parity)." The authors provide an **exact solution** for the optimal binary classifier under sufficiency constraints with finite discrete scores. By geometrically characterizing the $(\mathrm{PPV}, \mathrm{FOR})$ feasible region, they derive a post-processing algorithm that depends only on scores and group labels, and prove it can simultaneously solve both "loss minimization" and "minimizing deviation from separation under sufficiency."

## Background & Motivation

**Background**: Algorithmic fairness research typically revolves around three incompatible statistical criteria: independence (demographic parity), separation (equalized odds), and sufficiency (predictive parity). Standard practices for the first two involve post-processing: training a score $S$, then thresholding or randomizing per group (Hardt et al., 2016; Corbett-Davies et al., 2017). A frequently cited "benign conclusion" suggests that if the score $S$ itself satisfies a criterion, any post-processing (e.g., a single threshold) will preserve it.

**Limitations of Prior Work**: This benign conclusion **does not hold** for sufficiency. Even if $S$ is the true conditional probability $P(Y=1\mid X,A)$ (which naturally satisfies sufficiency), applying a global threshold to obtain a binary decision $R$ will almost always violate predictive parity. This is the root cause of the "calibrated scores, biased decisions" phenomenon in the COMPAS recidivism risk score case (Chouldechova, 2017; Canetti et al., 2019).

**Key Challenge**: The sufficiency equality $P(Y=1\mid R,A)=P(Y=1\mid R)$ is a constraint on the group conditional distribution of **discrete binary outcomes** $R$. Using a single threshold induces different $\Pr(R=1\mid A=a)$ across groups, which breaks the consistency of PPV/FOR between them. Existing works either allow abstention (Canetti et al., 2019), assume continuous scores with full support (Baumann et al., 2022), or only approximate sufficiency (Celis et al., 2019; Delaney et al., 2024). **No existing method** provides an "exact optimal sufficient classifier under finite discrete group-calibrated scores."

**Goal**: To answer three questions in the most practical deployment setting (finite discrete, group-calibrated scores): (i) Which $(\mathrm{PPV}, \mathrm{FOR})$ pairs are achievable for a sufficient classifier? (ii) How to precisely find the one that minimizes expected loss among these pairs? (iii) Given that sufficiency and separation cannot be strictly satisfied simultaneously, can the deviation from separation be minimized while strictly satisfying sufficiency?

**Key Insight**: The authors discovered that for a given score distribution within a group, all binary classifiers (possibly randomized) that can be constructed from that score form a **star-convex region** $\mathcal{C}$ in a 2D plane with a closed-form description. Its boundary is attained by "soft-thresholding rules." Once the feasible regions $\mathcal{C}^a$ for each group are geometricized, sufficiency reduces to "selecting points within the intersection $\mathcal{C}^0\cap\mathcal{C}^1$."

**Core Idea**: Reformulate "finding the optimal sufficient classifier" as "one-dimensional optimization along the boundary curve of the 2D feasible region intersection," and design a boundary-tracing algorithm with $O(m^0+m^1)$ segments to solve both loss minimization and separation deviation minimization within the same framework.

## Method

### Overall Architecture
Input: Finite score values $\{s_i^a\}$ and within-group distributions $P(s_i^a\mid A=a)$ for each group $a\in\{0,1\}$, group proportions $P(A)$, and an objective function $\mathcal{F}$ (loss or $\Delta_{\mathrm{sep}}$). Output: The optimal (possibly randomized) binary decision rule $P(R=1\mid S,A)$ satisfying sufficiency. The process has three layers: (1) Characterize the geometry of the feasible region $\mathcal{C}^a$ for a single group; (2) Intersect the two groups $\mathcal{C}^0\cap\mathcal{C}^1$ and characterize the non-decreasing boundary; (3) Trace the boundary and solve the objective function closed-form within each segment.

### Key Designs

1.  **Geometric Characterization of $(\mathrm{PPV}, \mathrm{FOR})$ and Soft-Thresholding Rules**:
    - **Function**: For any finite calibrated score $S$, provides an exact closed-form description of all achievable $(p,q)=(\mathrm{PPV}(R),\mathrm{FOR}(R))$ pairs and constructs the simplest rule for boundary points.
    - **Mechanism**: Scores are sorted in descending order $s_1>\dots>s_m$, and selection rate $\mu=P(R=1)$ is introduced. Calibration and the Markov chain $Y \leftrightarrow S \leftrightarrow R$ yield the fundamental equation $\pi=\mu p+(1-\mu)q$. Geometrically, this means for a fixed $\mu$, all $(p,q)$ lie on a line passing through $(\pi,\pi)$ with slope $-\mu/(1-\mu)$. Maximizing $p$ for a fixed $\mu$ is a linear program—reducing to the **fractional knapsack greedy solution** (Dantzig, 1957): select the largest score bins and use randomization for the last bin. This results in the soft-thresholding rule $P(R^*=1\mid s_i)$, which only takes fractional values at one $k^*(\mu)$.
    - **Design Motivation**: Compresses the infinite-dimensional search over all possible randomized decision rules into a one-dimensional parameterization along a 2D curve. The star-convexity of $\mathcal{C}$ (centered at $(\pi,\pi)$) ensures internal points can be reproduced by convex combinations of boundary points and $(\pi,\pi)$, hence **only the boundary needs to be traced**.

2.  **Multi-Group Sufficiency = Non-decreasing Boundary of Subgroup Intersections**:
    - **Function**: Translates sufficiency constraints $p^0=p^1=p, q^0=q^1=q$ directly into the geometric condition $(p,q)\in\mathcal{C}^0\cap\mathcal{C}^1$ and provides the closed-form boundary of the intersection.
    - **Mechanism**: Each non-trivial boundary of $\mathcal{C}^a$ is described by a non-decreasing function $q^a(p)$. The intersection boundary is given by the **pointwise maximum** $q(p)=\max\{q^0(p),q^1(p)\}$, which remains non-decreasing. Dividing the $p$-axis into sub-intervals $J_{k,l}$ based on breakpoints $\{p_k^0\}$ and $\{p_l^1\}$ of both groups, $q^0$ and $q^1$ are rational functions. Their difference $q^0(p)-q^1(p)$ depends on the sign of a quadratic function $\Phi_{k,l}(p)$, yielding at most two intersections per segment. Thus, the active boundary switches a finite number of times.
    - **Design Motivation**: This reveals a key phenomenon—points on the intersection boundary **usually do not correspond to hard threshold rules** for any group; at least one group must use randomized decisions. This explains why optimal sufficient classifiers cannot be achieved using group-specific hard thresholds alone.

3.  **Algorithm 1: Universal Boundary Trace + Closed-form Optimization**:
    - **Function**: Sequentially traverses all $J_{k,l,i}$ sub-segments of $\partial(\mathcal{C}^0\cap\mathcal{C}^1)$ and finds the segment-wise optimal for objective $\mathcal{F}$ to determine the global optimum.
    - **Mechanism**: Maintains the current active group $a\in\{0,1\}$ and index $(k,l)$, with $O(1)$ cost per segment and total segments on the order of $\le m^0+m^1$. Both objectives are closed-form per segment: (i) **Loss Minimization**: Substituted expected loss $L$ into the first-order condition for $p$ yields a quadratic equation. (ii) **$\Delta_{\mathrm{sep}}$ Minimization**: Expansion of TV distance shows the derivative with respect to $p$ is non-positive, meaning the minimum also lies on the boundary and is closed-form.
    - **Design Motivation**: Uses the **same geometric skeleton** to unify two objective types often discussed in opposition (accuracy vs. equalized odds), allowing practitioners to **explicitly compare** optimal solutions.

### Loss & Training
The method is strictly post-processing: it requires no retraining of the scoring model and no raw features, necessitating only group-calibrated scores and group labels to output decision rules $P(R=1\mid S,A)$. For cases where calibration is only approximate, Appendix J provides robustness bounds and validates small predictive parity violations on ACS Income data (PPV gap $4.8\times 10^{-3}$).

## Key Experimental Results

### Main Results
| Dataset | Setting | Unconstrained Bayes Accuracy | Ours: Optimal Sufficient Accuracy | Fairness Result |
| :--- | :--- | :--- | :--- | :--- |
| FICO (White/Black, 200 bins) | Group-level calibrated | 0.8819 (PPV: 0.91/0.79; FOR: 0.20/0.13) | 0.8676 | Shared PPV $=0.91$, shared FOR $=0.23$, strict sufficiency |
| COMPAS (White/Black, 10 deciles) | Pooled $P(Y=1\mid d)$ as group-calibrated | —— | —— | Optimum falls on breakpoint of $\partial\mathcal{C}^1$ (Black threshold $\ge 5$); White group must randomize |
| ACS Income (CA, Gender) | Logistic + out-of-fold 100-bin calibration | 0.8190 | 0.8167 | Mean PPV gap $4.8\times 10^{-3}$, mean FOR gap $3.7\times 10^{-3}$ |

### Ablation Study
| Observation | Conclusion | Explanation |
| :--- | :--- | :--- |
| Does the optimum use hard thresholds? | At least one group must randomize in FICO/COMPAS | Validates the theory that sufficiency optima generally cannot be achieved via hard thresholds alone. |
| Loss-optimal vs. $\Delta_{\mathrm{sep}}$-optimal | Figure 2 Left: Merged; Figure 2 Right: Separated | The two may coincide or diverge on the boundary depending on the problem instance; the algorithm identifies both. |
| PPV/FOR drift under approximate calibration | $\sim 4\times 10^{-3}$ on ACS Income | Acceptable for engineering practice and much smaller than the worst-case bound in Appendix J. |

### Key Findings
- The "geometric perspective" directly yields **structural insights**: the intersection boundary rarely coincides with any group's hard threshold breakpoint, implying that any framework prohibiting randomization or group-specific rules is **theoretically incapable** of reaching sufficiency optimality.
- The accuracy cost is **moderate**: FICO decreased from 0.8819 to 0.8676, and ACS Income from 0.8190 to 0.8167, a loss of only $\sim 1\%$ to $1.6\%$. This supports the notion that the incompatibility between sufficiency and accuracy is weaker than that between separation and accuracy.
- A single algorithm handles "accuracy optimality" and "closest to separation": in many instances, they coincide or are close, allowing engineers to decide based on both curves without retraining.

## Highlights & Insights
- **Translating fairness into 2D geometry** is the most elegant contribution of this paper. Once $\mathcal{C}^a$ and its intersections are drawn on the $(\mathrm{PPV}, \mathrm{FOR})$ plane, the theory becomes intuitive for communication with domain experts.
- The **optimality of soft-thresholding** (Theorem 3.3) is equivalent to a fractional knapsack greedy algorithm, suggesting that many post-processing fairness tasks are essentially sorting plus boundary searching with low enough complexity for online deployment.
- **Acknowledging the inevitability of randomization** is crucial. While many real-world scenarios (lending, bail) resist randomized decisions for identical scores, this paper provides geometric evidence that **refusing randomization is equivalent to refusing optimality** under strict multi-group sufficiency.

## Limitations & Future Work
- As the number of groups increases, the condition for multiple $\mathcal{C}^a$ to have a non-empty intersection becomes strict; **strict sufficiency may have no feasible solution**. The authors suggest relaxing this to PPV/FOR gap minimization for multiple groups.
- The assumption of **group-calibrated** scores is central. While multicalibration (Hébert-Johnson et al., 2018) can approximate this, calibration errors propagate. Systematic assessment under distribution shift or label noise is missing.
- Experiments are "case studies" on FICO/COMPAS/ACS Income, lacking unified benchmark comparisons against recent tools like OxonFair (Delaney et al., 2024).
- The current formulation covers binary $A$; multivariate generalizations are claimed to be "straightforward" but are not explicitly provided.

## Related Work & Insights
- **vs. Canetti et al. (2019)**: They satisfy sufficiency by introducing "abstention." This paper does not require abstention and finds the optimum directly on the $(\mathrm{PPV}, \mathrm{FOR})$ plane, which is closer to practical binary decision-making.
- **vs. Baumann et al. (2022)**: They also pursue sufficiency without abstention but assume continuous scores with full support. This work specifically targets **finite discrete scores**, matching real-world scoring systems like COMPAS.
- **vs. Hardt et al. (2016)**: While that paper provided the "textbook answer" for separation post-processing, this work provides the counterpart for sufficiency and an additional trade-off by minimizing the distance to separation.

## Rating
- Novelty: ⭐⭐⭐⭐ Provides the first exact solution for sufficiency optimality in the practical "finite discrete group-calibrated" setting.
- Experimental Thoroughness: ⭐⭐⭐ Three classic cases demonstrate utility, but lacks quantitative baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Definitions, theorems, geometric plots, and pseudocode are logically progressive. Highly readable.
- Value: ⭐⭐⭐⭐ Provides a clear proof of why calibration is insufficient and offers a solution for high-stakes decision-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)
- [\[ICML 2026\] Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](fairness_in_aggregation_optimal_top-k_and_improved_full_ranking.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)

</div>

<!-- RELATED:END -->

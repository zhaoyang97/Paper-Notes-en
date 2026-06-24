---
title: >-
  [Paper Note] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely $F_1$
description: >-
  [CVPR 2026][F-score] This paper systematically investigates the properties of the $F_\beta$ score family as a ranking tradeoff between Precision and Recall from a ranking theory perspective. It proves that the rankings induced by $F_\beta$ constitute a geodesic (shortest path) between Precision and Recall rankings. Consequently, it proposes a closed-form formula to find the optimal $\beta$ value and demonstrates that the commonly used $F_1$ and skew-insensitive $F_1$ are not…
tags:
  - "CVPR 2026"
  - "F-score"
  - "ranking optimization"
  - "Kendall distance"
  - "Precision-Recall tradeoff"
  - "performance evaluation theory"
date: 2026-05-08
content_hash: e9674b3bafe4e726
---

# What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely $F_1$

**Conference**: CVPR 2026  
**arXiv**: [2511.22442](https://arxiv.org/abs/2511.22442)  
**Code**: [https://github.com/pierard/cvpr-2026-optimal-tradeoff-precision-recall](https://github.com/pierard/cvpr-2026-optimal-tradeoff-precision-recall)  
**Area**: Others  
**Keywords**: F-score, ranking optimization, Kendall distance, Precision-Recall tradeoff, performance evaluation theory

## TL;DR

This paper systematically investigates the properties of the $F_\beta$ score family as a ranking tradeoff between Precision and Recall from a ranking theory perspective. It proves that the rankings induced by $F_\beta$ constitute a geodesic (shortest path) between Precision and Recall rankings. Consequently, it proposes a closed-form formula to find the optimal $\beta$ value and demonstrates that the commonly used $F_1$ and skew-insensitive $F_1$ are not the optimal ranking tradeoffs in most cases.

## Background & Motivation

1. **Background**: Precision and Recall are the most fundamental evaluation metrics for classification tasks. Since neither is comprehensive on its own, their weighted harmonic mean, the $F_\beta$ score, is frequently used. $F_1$ (equal weighting) is the most common in practice; Google Scholar shows approximately 315,000 papers using F-measure, and about 10% of CVPR 2025 papers utilize $F_1$.
2. **Limitations of Prior Work**: Although $F_\beta$ serves as a numerical tradeoff between Precision and Recall, whether the ranking induced by $F_1$ is truly the optimal tradeoff between Precision and Recall rankings has never been rigorously studied. Notably, as the class prior $\pi_+ \to 0$, all $F_\beta$ scores degenerate into rankings that mimic Precision while ignoring Recall.
3. **Key Challenge**: $F_1$ is adopted by default as a "balanced" tradeoff, but the choice of $\beta=1$ lacks a foundation in ranking theory—the same $\beta$ cannot provide an optimal ranking across all class priors.
4. **Goal**: (a) Do the rankings induced by $F_\beta$ form a meaningful path? (b) Is $F_1$ the optimal ranking tradeoff? (c) How can the optimal $\beta$ be found?
5. **Key Insight**: Utilize the theoretical framework of Kendall rank correlation/distance to measure the distance between rankings, formalizing the search for the optimal tradeoff as a Fréchet variance minimization problem.
6. **Core Idea**: The manifold of rankings induced by $F_\beta$ is a geodesic between Precision and Recall rankings, and the optimal $\beta^2$ equals the median of $\vartheta$ values across all performance pairs.

## Method

### Overall Architecture

Ours does not train any models but instead addresses the question "Is $F_1$ the best tradeoff between Precision and Recall?" within a **ranking space**. Given a set of classifier performances $\Pi = \{P_1, \ldots, P_n\}$, where each $P_i$ corresponds to a confusion matrix, any scoring function $X$ (Precision, Recall, $F_\beta$, etc.) will order these $n$ classifiers into a ranking $\mathbf{x}$. The difference between two rankings is measured by the Kendall distance $d_\tau$, which counts the number of classifier pairs with disagreeing relative orders. Finding the optimal tradeoff thus becomes: within the $F_\beta$ family of rankings, find a $\beta$ that is "equidistant" from the Precision ranking and the Recall ranking. The reasoning follows three steps: proving the $F_\beta$ family lies on the shortest path (worth searching), providing a closed-form solution for the optimal $\beta$ (how to search), and introducing a reportable metric to quantify how far any $\beta$ is from the optimal (how to evaluate).

### Key Designs

**1. Geodesic Property of $F_\beta$ Rankings: Proving the family is worth searching**

Before selecting $\beta$, it must be confirmed that the family of scores does not "take a detour"; otherwise, finding an equidistant point is meaningless. A key observation is that $F_\beta$ is essentially a weighted $f$-mean of Precision and Recall (where $f(x)=x^{-1}$ is the harmonic mean). This mean possesses a desirable property: for any two performances $P_A, P_B$, if both Precision and Recall agree that $A$ should be ranked above $B$, then all $F_\beta$ scores will also agree. Aggregating this property over the entire performance set yields the core equation:

$$d_\tau(Pr; Re) = d_\tau(Pr; F_\beta) + d_\tau(F_\beta; Re)$$

This implies that the "distance" from the Precision ranking to the Recall ranking via any $F_\beta$ ranking is additive—the $F_\beta$ ranking lies strictly on the **geodesic** (shortest path) between the two endpoints. This ensures that searching for a tradeoff point within the $F_\beta$ family is a meaningful geometric problem. It is worth emphasizing that this is non-trivial: arithmetic and geometric means do **not** satisfy performance ranking axioms and can induce counter-intuitive rankings; thus, only the $F_\beta$ family serves as a valid tradeoff.

**2. Closed-form Solution for Optimal Tradeoff: Calculating the "equidistant point" directly**

Since $F_\beta$ rankings move along the geodesic as $\beta$ varies, the optimal tradeoff $F_*$ is defined as the point equidistant to both ends, minimizing the Fréchet variance $\sigma^2(\beta) = d_\tau^2(Pr; F_\beta) + d_\tau^2(F_\beta; Re)$. This is equivalent to finding the Karcher mean satisfying $d_\tau(Pr; F_*) = d_\tau(F_*; Re)$. The difficulty lies in the fact that $\beta$ is continuous while rankings are discrete: as $\beta$ increases, the $F_\beta$ ranking does not change smoothly but undergoes **discrete jumps** at certain critical points where two performances $P_1, P_2$ are tied and then swap positions. Each such jump corresponds to a critical value:

$$\beta^2 = \vartheta(P_1, P_2) = -\frac{PTP(P_1)\cdot PFP(P_2) - PTP(P_2)\cdot PFP(P_1)}{PTP(P_1)\cdot PFN(P_2) - PTP(P_2)\cdot PFN(P_1)}$$

where $PTP/PFP/PFN$ are normalized probabilities of true positives, false positives, and false negatives. Since each jump moves the ranking one step toward Recall, the "equidistant" condition is equivalent to splitting the jump points—**the optimal $\beta^2$ is exactly the median of all positive $\vartheta$ values**. The solution is entirely analytical, requiring only the enumeration of performance pairs, calculation of $\vartheta$, and finding the median.

**3. Optimality $\mathcal{O}$: Scoring how far any given $\beta$ is from the optimal**

With $F_*$ identified, users naturally want to know how well their chosen $\beta$ (e.g., the default $F_1$) performs. The paper categorizes classifier pairs into three types: those where Precision and Recall already agree, those where a tradeoff choice is made correctly, and those where it is made incorrectly. Optimality is defined as:

$$\mathcal{O} = 1 - \frac{d_\tau(F_\beta; F_*)}{d_\tau(Pr; Re)}$$

The numerator is the Kendall distance between the current $\beta$ ranking and the optimal ranking $F_*$, while the denominator normalizes by the total distance. $\mathcal{O}=1$ if and only if $\beta$ is optimal. This is a single number that can be reported in papers, allowing anyone using $F_\beta$ to quantify the optimality of their $\beta$ choice.

### Loss & Training

This is a theoretical work with no training process. Implementation requires a single line: $\beta^2 = \text{median}(\{\vartheta(P_i, P_j) \mid i \neq j \wedge \vartheta(P_i, P_j) \geq 0\})$. Taking the median of all positive pairwise critical values yields the optimal $\beta$.

## Key Experimental Results

### Main Results

**Optimal $\beta^2$ values for six case studies:**

| Case | $\tau(Pr; Re)$ | $F_1$ Optimality | SIVF Optimality | Optimal $\beta^2$ |
|------|---------------|-------------|------------|---------------|
| Uniform distribution over all performances | 1/3 | 100% (Optimal) | Meaningless ranking | 1.0 (Exactly $F_1$) |
| Uniform distribution with fixed TN prob | 1/3 | 100% (Optimal) | Meaningless ranking | 1.0 |
| Beta distribution | Varies | Far from optimal | Far from optimal | Requires calculation |
| CADA-RRE Medical Challenge (16 models) | - | Non-optimal | Non-optimal | Fixed by formula |
| VOC 2012 Segmentation (47 classes) | - | Non-optimal | Non-optimal | Varies by class |
| Custom Scenarios | - | Non-optimal | Non-optimal | Data-dependent |

### Ablation Study

| Configuration | Description |
|------|------|
| $F_1$ ($\beta^2=1$) | Only optimal under specific highly symmetric conditions like uniform distributions. |
| SIVF ($\beta^2=\pi_-/\pi_+$) | Satisfies axioms (at fixed priors) but usually non-optimal in real scenarios. |
| Heuristic $\beta^2 = E[PFP]/E[PFN]$ | Exactly optimal or near-optimal under several distributions. |
| Closed-form median | Always optimal, $\mathcal{O}=100\%$. |

### Key Findings
- **$F_1$ is rarely optimal**: $\beta=1$ is the optimal tradeoff only when the performance distribution possesses specific symmetry. In most practical scenarios, $F_1$ produces rankings biased toward either Precision or Recall.
- **$F_\beta$ family is the unique correct search space**: Arithmetic and geometric means do not satisfy performance ranking axioms (they can induce meaningless rankings), whereas all $F_\beta$ scores do.
- The heuristic rule $\beta^2 = E[PFP]/E[PFN]$ provides optimal results across multiple distributions and cases, serving as a simple approximation.
- The optimal $\beta$ is data-dependent—different challenges, semantic categories, and performance distributions require different $\beta$ values.

## Highlights & Insights
- **Solid Theoretical Contribution**: Starting from ranking axioms, through geodesic proofs and Karcher mean derivations, to a final closed-form solution, the derivation is complete and elegant. It rigorously addresses a fundamental problem regarding a metric used for over 50 years.
- **Deep Practical Significance**: Approximately 10% of CVPR papers use $F_1$; this paper points out that $F_1$ is not optimal for ranking in most cases. While it may not overturn existing conclusions (as deviations are often small), it provides a theoretical weapon for rigorous future evaluations.
- **Simple Closed-form Formula**: Calculating the median of $\vartheta$ values for all performance pairs is straightforward and requires no complex optimization.
- The authors suggest reporting both $\tau(Pr; F_\beta)$ and $\tau(F_\beta; Re)$ to evaluate the optimality of a selected $\beta$, which is an excellent practical recommendation.

## Limitations & Future Work
- Ours only investigates the Precision-Recall tradeoff for binary classification; multi-class classification requires extension to macro/micro averaging scenarios.
- Calculating $\vartheta$ requires full confusion matrices for all classifiers, which is not directly applicable in scenarios where only leaderboard scores are available.
- The paper focuses on the $F_\beta$ family, but whether more optimal non-harmonic mean forms of ranking tradeoffs exist has not been discussed.
- Case studies focus on classification; the AP metric common in detection/segmentation (which integrates multiple thresholds) is not within the scope.

## Related Work & Insights
- **vs. Traditional $F_1$ Usage**: While many papers default to $\beta=1$, this work proves it is almost never ranking-optimal—the specific set of performances determines the optimal $\beta$.
- **vs. Skew-Insensitive $F_1$ (SIVF)**: The SIVF proposed by Flach & Kull is equivalent to $F_\beta$ with $\beta^2 = \pi_-/\pi_+$ under fixed priors, guaranteeing meaningful rankings but not optimality.
- **vs. Ferri et al. / Liu et al.**: Previous studies on metric correlation used Pearson/Spearman correlation; this work is the first to use Kendall rank correlation to analyze tradeoff rankings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Provides the first theoretically complete answer to a 50-year-old fundamental problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six case studies cover theoretical distributions and real challenges, though more large-scale competition data could be included.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous and clear, though the notation density may be challenging for non-theoretical readers.
- Value: ⭐⭐⭐⭐ Provides a rigorous theoretical foundation for evaluation practices, though the practical impact on existing conclusions may be modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] What Does It Take to Build a Performant Selective Classifier?](../../NeurIPS2025/others/what_does_it_take_to_build_a_performant_selective_classifier.md)
- [\[AAAI 2026\] How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?](../../AAAI2026/others/how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](../../AAAI2026/others/how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[CVPR 2026\] Region-Wise Correspondence Prediction between Manga Line Art Images](region-wise_correspondence_prediction_between_manga_line_art_images.md)
- [\[ICLR 2026\] It's All Just Vectorization: einx, a Universal Notation for Tensor Operations](../../ICLR2026/others/its_all_just_vectorization_einx_a_universal_notation_for_tensor_operations.md)

</div>

<!-- RELATED:END -->

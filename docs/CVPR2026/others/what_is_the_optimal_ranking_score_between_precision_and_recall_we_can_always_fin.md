---
title: >-
  [Paper Note] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁
description: >-
  [CVPR 2026][F-score] This paper systematically studies the $F_\beta$ score family as a ranking tradeoff between Precision and Recall from a ranking-theoretic perspective. It proves that the rankings induced by $F_\beta$ form a geodesic (shortest path) between the Precision and Recall rankings, derives a closed-form formula for finding the optimal $\beta$, and demonstrates that the commonly used $F_1$ and skew-insensitive $F_1$ are rarely optimal ranking tradeoffs in practice.
tags:
  - CVPR 2026
  - F-score
  - ranking optimization
  - Kendall distance
  - precision-recall tradeoff
  - performance evaluation theory
date: 2026-05-08
content_hash: ca8f2ed7a1b25ed0
---

# What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁

**Conference**: CVPR 2026
**arXiv**: [2511.22442](https://arxiv.org/abs/2511.22442)
**Code**: [https://github.com/pierard/cvpr-2026-optimal-tradeoff-precision-recall](https://github.com/pierard/cvpr-2026-optimal-tradeoff-precision-recall)
**Area**: Other
**Keywords**: F-score, ranking optimization, Kendall distance, precision-recall tradeoff, performance evaluation theory

## TL;DR

This paper systematically studies the $F_\beta$ score family as a ranking tradeoff between Precision and Recall from a ranking-theoretic perspective. It proves that the rankings induced by $F_\beta$ form a geodesic (shortest path) between the Precision and Recall rankings, derives a closed-form formula for finding the optimal $\beta$, and demonstrates that the commonly used $F_1$ and skew-insensitive $F_1$ are rarely optimal ranking tradeoffs in practice.

## Background & Motivation

1. **Background**: Precision and Recall are the most fundamental evaluation metrics for classification tasks, yet neither is sufficient on its own. Their weighted harmonic mean, the $F_\beta$ score, is widely adopted. In practice, $F_1$ (equal weighting) is the most common choice; Google Scholar records approximately 315,000 papers using the F-measure, and roughly 10% of CVPR 2025 papers employ $F_1$.
2. **Limitations of Prior Work**: Although $F_\beta$ numerically interpolates between Precision and Recall, it has never been rigorously examined whether the ranking induced by $F_1$ constitutes an optimal tradeoff between the Precision ranking and the Recall ranking. In particular, as the class prior $\pi_+ \to 0$, all $F_\beta$ scores degenerate into rankings that mimic Precision while ignoring Recall.
3. **Key Challenge**: $F_1$ is assumed by default to be a "balanced" tradeoff, yet the choice $\beta = 1$ lacks any ranking-theoretic justification — a single fixed $\beta$ cannot provide an optimal ranking across all class priors.
4. **Goal**: (a) Do the rankings induced by $F_\beta$ form a meaningful path? (b) Is $F_1$ an optimal ranking tradeoff? (c) How can the optimal $\beta$ be found?
5. **Key Insight**: The Kendall rank correlation/distance framework is adopted to measure distances between rankings, and the search for an optimal tradeoff is formalized as a Fréchet variance minimization problem.
6. **Core Idea**: The manifold of rankings induced by $F_\beta$ constitutes a geodesic between the Precision and Recall rankings, and the optimal $\beta^2$ equals the median of the $\vartheta$ values computed over all pairs of performances.

## Method

### Overall Architecture

Given a set of classifier performances $\Pi = \{P_1, \ldots, P_n\}$, where each $P_i$ corresponds to a confusion matrix, every scoring function $X$ (e.g., Precision, Recall, $F_\beta$) induces a ranking $\mathbf{x}$. The paper analyzes the geometric properties of the $F_\beta$ family in ranking space and identifies the $\beta$ value that places the induced ranking equidistant from the Precision and Recall rankings.

### Key Designs

1. **Geodesic Property of $F_\beta$ Rankings**

   - **Function**: Proves that $F_\beta$ is the correct score family for finding a Precision-Recall ranking tradeoff.
   - **Mechanism**: Because $F_\beta$ is a weighted $f$-mean of Precision and Recall with $f(x) = x^{-1}$ (i.e., harmonic mean), for any two performances $P_A, P_B$, if both Precision and Recall agree on an ordering, then every $F_\beta$ agrees as well. This yields the key identity $d_\tau(Pr; Re) = d_\tau(Pr; F_\beta) + d_\tau(F_\beta; Re)$, establishing that $F_\beta$ rankings lie on the geodesic (shortest path) between the Precision and Recall rankings. This guarantees that restricting the search for an optimal tradeoff to the $F_\beta$ family is well-founded.
   - **Design Motivation**: Not every score that numerically lies between Precision and Recall satisfies this geodesic property — the arithmetic mean and the geometric mean both fail to satisfy the performance ranking axioms.

2. **Closed-Form Solution for the Optimal Tradeoff**

   - **Function**: Provides an analytic formula for computing the optimal $\beta$.
   - **Mechanism**: The optimal tradeoff $F_*$ is defined as the Karcher mean minimizing the Fréchet variance $\sigma^2(\beta) = d_\tau^2(Pr; F_\beta) + d_\tau^2(F_\beta; Re)$, equivalently characterized by the equidistance condition $d_\tau(Pr; F_*) = d_\tau(F_*; Re)$. For a finite set of performances, the ranking induced by $F_\beta$ changes through a sequence of discrete jumps as $\beta$ increases (occurring whenever two performances are tied under $F_\beta$), with each jump at $\beta^2 = \vartheta(P_1, P_2) = -\frac{PTP(P_1) \cdot PFP(P_2) - PTP(P_2) \cdot PFP(P_1)}{PTP(P_1) \cdot PFN(P_2) - PTP(P_2) \cdot PFN(P_1)}$. The optimal $\beta^2$ is then the median of all positive $\vartheta$ values.
   - **Design Motivation**: The result is entirely analytic and requires no numerical optimization.

3. **Optimality Metric $\mathcal{O}$**

   - **Function**: Quantifies how close any given $\beta$ choice is to the optimal ranking.
   - **Mechanism**: All classifier pairs are partitioned into three categories: those on which Precision and Recall agree (no choice needed), those on which a choice is needed and the correct one is made, and those on which a choice is needed but the wrong one is made. The optimality score is $\mathcal{O} = 1 - \frac{d_\tau(F_\beta; F_*)}{d_\tau(Pr; Re) - d_\tau(F_\beta; F_*) + d_\tau(F_\beta; F_*)}$, which equals 1 if and only if $\beta$ is optimal.
   - **Design Motivation**: Provides a simple, reportable metric that allows practitioners to assess how far their choice of $\beta$ deviates from optimal.

### Loss & Training

This paper is purely theoretical and involves no training procedure. The central formula is $\beta^2 = \text{median}(\{\vartheta(P_i, P_j) \mid i \neq j \wedge \vartheta(P_i, P_j) \geq 0\})$.

## Key Experimental Results

### Main Results

**Optimal $\beta^2$ values across six case studies:**

| Case | $\tau(Pr; Re)$ | $F_1$ Optimality | SIVF Optimality | Optimal $\beta^2$ |
|------|---------------|-----------------|-----------------|-------------------|
| Uniform distribution over all performances | 1/3 | 100% (optimal) | Meaningless ranking | 1.0 (exactly $F_1$) |
| Uniform distribution with fixed TN probability | 1/3 | 100% (optimal) | Meaningless ranking | 1.0 |
| Beta distribution | Varies | Far from optimal | Far from optimal | Requires computation |
| CADA-RRE medical challenge (16 models) | — | Non-optimal | Non-optimal | Determined by closed-form formula |
| VOC 2012 segmentation (47 classes) | — | Non-optimal | Non-optimal | Varies per class |
| Custom scenario | — | Non-optimal | Non-optimal | Data-dependent |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| $F_1$ ($\beta^2=1$) | Optimal only under highly symmetric conditions such as the uniform distribution |
| SIVF ($\beta^2=\pi_-/\pi_+$) | Satisfies ranking axioms (under fixed priors) but is typically non-optimal in real scenarios |
| Heuristic $\beta^2 = E[PFP]/E[PFN]$ | Exactly optimal or near-optimal across multiple distributions |
| Closed-form median formula | Always optimal, $\mathcal{O}=100\%$ |

### Key Findings

- **$F_1$ is rarely optimal**: $\beta = 1$ is the optimal tradeoff only when the performance distribution exhibits specific symmetry (e.g., the uniform distribution). In the vast majority of practical settings, $F_1$ yields rankings biased toward either Precision or Recall.
- **The $F_\beta$ family is the uniquely correct search space**: The arithmetic mean and geometric mean fail to satisfy the performance ranking axioms and can induce meaningless rankings, whereas every $F_\beta$ satisfies them.
- The heuristic $\beta^2 = E[PFP]/E[PFN]$ yields the optimal result across multiple distributions and several case studies, and can serve as a simple approximation.
- The optimal $\beta$ is data-dependent — different classification challenges, different semantic categories, and different performance distributions require different $\beta$ values.

## Highlights & Insights

- **Exceptionally strong theoretical contribution**: Starting from ranking axioms, proceeding through the geodesic proof and Karcher mean derivation, and concluding with a closed-form solution, the entire chain of reasoning is complete and elegant. The paper provides the first rigorous answer to a fundamental question about a metric that has been in use for over 50 years.
- **Broad practical significance**: Approximately 10% of CVPR papers use $F_1$; this work shows that $F_1$ is non-optimal in most cases. While existing conclusions are unlikely to be overturned (since deviations are typically small), the paper provides a theoretical foundation for more rigorous future evaluation.
- **Closed-form formula is minimal and practical**: Computing the median of all pairwise $\vartheta$ values requires no complex optimization.
- The authors propose reporting both $\tau(Pr; F_\beta)$ and $\tau(F_\beta; Re)$ alongside results to assess the optimality of the chosen $\beta$, which is a valuable practical recommendation.

## Limitations & Future Work

- The paper studies only the binary classification Precision-Recall tradeoff; extension to multi-class settings involving macro/micro averaging remains open.
- Computing $\vartheta$ requires complete confusion matrices for all classifiers, making direct application infeasible in settings where only leaderboard scores are available.
- The analysis is confined to the $F_\beta$ family; whether superior ranking tradeoffs based on non-harmonic means exist is not discussed.
- The case studies focus on classification tasks; the widely used AP metric in detection and segmentation, which integrates over multiple thresholds, is outside the scope of this work.

## Related Work & Insights

- **vs. conventional $F_1$ usage**: Numerous papers default to $\beta = 1$; this work proves that such a choice is almost never ranking-optimal — the specific performance set determines the optimal $\beta$.
- **vs. skew-insensitive $F_1$ (SIVF)**: The SIVF proposed by Flach & Kull is equivalent to $F_\beta$ with $\beta^2 = \pi_-/\pi_+$ under a fixed prior, guaranteeing meaningful rankings but not optimality.
- **vs. Ferri et al. / Liu et al.**: Prior studies of inter-score correlation use Pearson or Spearman correlation; this paper is the first to employ Kendall rank correlation for analyzing ranking tradeoffs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Provides the first theoretically complete answer to a fundamental question about a metric used for over 50 years.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six case studies covering theoretical distributions and real challenges are provided, though additional large-scale competition data would strengthen the empirical support.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear, but the heavy notation poses a non-trivial barrier for readers without a theoretical background.
- Value: ⭐⭐⭐⭐ — Establishes a rigorous theoretical foundation for evaluation practice, though the practical impact on existing conclusions may be limited.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?](../../AAAI2026/others/how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](../../AAAI2026/others/how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[CVPR 2026\] Bounds on Agreement between Subjective and Objective Measurements](bounds_on_agreement_between_subjective_and_objecti.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the_gaps_in_classification_robustn.md)
- [\[CVPR 2026\] What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)

<!-- RELATED:END -->

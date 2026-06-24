---
title: >-
  [Paper Note] Foundations of the Theory of Performance-Based Ranking
description: >-
  [CVPR 2025][Performance-Based Ranking Theory] This paper establishes a rigorous mathematical foundation for performance-based ranking based on probability theory and order theory. It proposes a general framework consisting of 6 pillars and 3 axioms, defines a parameterized family of "ranking scores," and demonstrates in binary classification tasks that metrics such as accuracy, TPR, TNR, PPV, and F-score satisfy these axioms, whereas commonly used metrics like MCC and geometr…
tags:
  - "CVPR 2025"
  - "Performance-Based Ranking Theory"
  - "Axiomatic Definition"
  - "Ranking Scores"
  - "Binary Classification"
  - "Order Theory"
date: 2026-05-08
content_hash: 279e716ad3675983
---

# Foundations of the Theory of Performance-Based Ranking

**Conference**: CVPR 2025  
**arXiv**: [2412.04227](https://arxiv.org/abs/2412.04227)  
**Code**: None  
**Area**: Robotics  
**Keywords**: Performance-Based Ranking Theory, Axiomatic Definition, Ranking Scores, Binary Classification, Order Theory

## TL;DR
This paper establishes a rigorous mathematical foundation for performance-based ranking based on probability theory and order theory. It proposes a general framework consisting of 6 pillars and 3 axioms, defines a parameterized family of "ranking scores," and demonstrates in binary classification tasks that metrics such as accuracy, TPR, TNR, PPV, and F-score satisfy these axioms, whereas commonly used metrics like MCC and geometric mean are unsuitable for ranking.

## Background & Motivation

1. **Background**: Ranking performance-based entities (algorithms, models, devices, etc.) is extremely common in fields such as computer vision, medical imaging, and sports analytics. Various challenges (e.g., SoccerNet, VOT, CDnet) evaluate the quality of participating algorithms through ranking. In practice, researchers typically choose one or more numerical metrics (e.g., accuracy, F1-score, MCC) to quantify performance and rank entities accordingly.

2. **Limitations of Prior Work**: (a) A critical analysis of 150 biomedical image analysis challenges revealed that only 23% justified the choice of their metrics, and only 36% reported how the ranking was computed; (b) there exist at least 10 different multi-metric ranking methods whose mathematical properties remain largely unknown; (c) the confusion between "performance" and "numerical score" is ubiquitous—researchers often select metrics based on intuition or imitation, leading to a lack of diversity in rankings.

3. **Key Challenge**: There is a lack of a rigorous mathematical foundation to define "what performance is," "what the mathematical properties of performance are," and "how to legally compare performances." In current practice, performance is simply equated with numerical scores, but none have explicitly defined the measurable space where performance resides and its permitted mathematical operations.

4. **Goal**: 
    - Establish a general mathematical framework to manipulate performance objects
    - Propose an axiomatic definition of performance-based ranking
    - Construct a general family of ranking scores that satisfy the axioms
    - Verify which commonly used metrics satisfy/violate the axioms in binary classification tasks

5. **Key Insight**: Define performance as a probability measure, utilize the preorder relation in order theory to define comparison relations such as "better than," "equivalent to," and "incomparable," and model task objectives and application preferences using two random variables: "satisfaction" and "importance," respectively.

6. **Core Idea**: Performance is not a single numerical value, but a probability measure. Based on this, performance-based ranking can be defined axiomatically, and a general family of ranking scores $R_I$ parameterized by "importance" can be constructed. This not only covers known valid metrics but also exposes the flaw of some commonly used metrics.

## Method

### Overall Architecture

The framework of this paper is designed like a "temple" (Figure 1), consisting of 6 pillars and 3 epistyles:

**6 Pillars (Mathematical Framework)**:
- Pillar 1: Performance $P$ as a probability measure
- Pillar 2: Preorder relation $\lesssim$ for comparing performance
- Pillar 3: Satisfaction $S$ to model the task
- Pillar 4: Evaluation function $\Phi$ to model entity evaluation
- Pillar 5: Score $X$ mapping performance to a real value
- Pillar 6: Importance $I$ encoding application preferences

**3 Epistyles (Theoretical Construction)**:
- Epistyle 1: 3 axioms to define performance-based ranking
- Epistyle 2: 3 theorems providing sufficient conditions for scores to satisfy the axioms
- Epistyle 3: The family of ranking scores $R_I$ providing a general solution satisfying all axioms

### Key Designs

1. **Performance as a Probability Measure**:
    - **Function**: Define the mathematical essence of "performance"
    - **Mechanism**: Performance $P$ is a probability measure defined on a measurable space $(\Omega, \Sigma)$, rather than a simple real value. For example, in binary classification, $\Omega = \{tn, fp, fn, tp\}$, and performance is a four-dimensional probability distribution (i.e., the normalized confusion matrix).
    - **Design Motivation**: Probability theory is an ideal framework for handling uncertainty and randomness, and performance naturally involves uncertainty. Defining performance as a probability measure allows for a rigorous definition of all permitted operations.

2. **Preorder Relation and Axiomatic Ranking**:
    - **Function**: Rigorously define "better," "worse," "equivalent," and "incomparable"
    - **Mechanism**: Derive four relations via a preorder $\lesssim$ (reflexive + transitive): $P_1 \sim P_2$ (equivalent), $P_1 > P_2$ (better), $P_1 < P_2$ (worse), and $P_1 \not\lesseqqgtr P_2$ (incomparable).
    - **Three Axioms**:
        - **Axiom 1** (using preorders): Ranking functions must be based on the preorder $\lesssim$, ensuring that adding or removing entities does not affect the relative ranking of the remaining entities.
        - **Axiom 2** (using satisfaction $S$): If the satisfaction of using entity $\epsilon_1$ is deterministically $\leq$ that of using $\epsilon_2$, then $P_1$ cannot be strictly preferred to $P_2$.
        - **Axiom 3** (using compound $\Phi$): A performance reachable through compounding cannot be strictly better than the best component performance or strictly worse than the worst component performance.
    - **Design Motivation**: An axiomatic definition is necessary to objectively judge whether a metric is mathematically "valid" for ranking purposes.

3. **The Family of Ranking Scores $R_I$**:
    - **Function**: Provide general parameterized ranking scores satisfying all axioms
    - **Mechanism**: Define $R_I(P) = \frac{\mathbf{E}_P[IS]}{\mathbf{E}_P[I]}$, where $I$ is the importance random variable and $S$ is the satisfaction random variable. This score is a generalization of the expected satisfaction $\mathbf{E}_P[S]$ filtered through the importance-weighted filter $\text{filter}_I$.
    - **Key Properties**:
        - $R_I$ can be decomposed into $X_S^E \circ \text{filter}_I$ (importance filtering + expected satisfaction).
        - Linear transformations of satisfaction do not affect the ranking.
        - Scaling of importance does not affect the ranking ($R_{kI} = R_I$).
        - $R_I$ is a pseudolinear function, guaranteeing that all contour sets are convex.
    - **Design Motivation**: Researchers face a lack of criteria when choosing among numerous metrics. The family of ranking scores provides an infinite family of metrics parameterized by application preferences, which both covers known valid metrics and can generate new ones.

### Loss & Training

This work is purely theoretical and does not involve training strategies. The core "validation strategy" consists of three tests:
- **Test 1**: Verify if the ranking induced by the score $\lesssim_X$ satisfies Axiom 2 (satisfaction consistency).
- **Test 2**: Verify the upper-bound property of convex combinations (related to Axiom 3).
- **Test 3**: Verify the lower-bound property of convex combinations (related to Axiom 3).

Additionally, Kendall's $\tau$ rank correlation coefficient is used to analyze the correlation of each metric with the family of ranking scores.

## Key Experimental Results

### Main Results

In binary classification tasks, approximately 30 commonly used metrics were subjected to three types of tests (unconstrained / fixed prior of 0.2 / fixed prior of 0.5):

| Metric | Unconstrained Three Tests | Fixed Prior Three Tests | Max Correlation with Ranking Score |
|------|-------------|---------------|-------------------|
| Accuracy | V/V/V | V/V/V | $\tau_{max} = 1$ |
| F1-score | V/V/V | V/V/V | $\tau_{max} = 1$ |
| TPR (Recall) | V/V/V | V/V/V | $\tau_{max} = 1$ |
| TNR (Specificity) | V/V/V | V/V/V | $\tau_{max} = 1$ |
| PPV (Precision) | V/V/V | V/V/V | $\tau_{max} = 1$ |
| NPV | V/V/V | V/V/V | $\tau_{max} = 1$ |
| Balanced Accuracy | V/X/X | V/V/V | $\tau_{max} = 0.713$ (unconstrained) |
| MCC | V/X/X | V/X/X | $\tau_{max} = 0.963$ (fixed 0.5) |
| Geometric Mean | V/X/X | V/X/V | $\tau_{max} = 0.831$ (fixed 0.2) |

### Ablation Study

| Metric Category | Usable Unconstrained | Usable with Fixed Prior | Always Unusable |
|---------|-----------|-------------|-----------|
| Green (Always Valid) | Accuracy, TPR, TNR, PPV, NPV, F-scores | Same as left | — |
| Orange (Requires Fixed Prior) | — | Balanced Accuracy, Cohen's κ, Informedness | — |
| Black (Unusable) | — | — | MCC, Geometric Mean, Markedness, Odds Ratio |

### Key Findings
- **Finding 1**: Multiple classic binary classification metrics (green group) satisfy all three axioms under all conditions and exhibit perfect correlation with the ranking scores ($\tau_{max} = 1$).
- **Finding 2**: Orange group metrics such as Balanced Accuracy and Cohen's κ can only be used for ranking under a fixed prior and violate the axioms in unconstrained settings.
- **Finding 3**: Widely used metrics such as MCC and Geometric Mean fail to satisfy the axioms even with a fixed prior and should not be used for ranking.
- **Finding 4**: In all cases where the axioms are satisfied, the corresponding metric exhibits perfect rank correlation with some ranking score, demonstrating the comprehensive coverage of the ranking score family.

## Highlights & Insights
- **Theoretical Breakthrough**: Introduces an axiomatic foundation for performance-based ranking for the first time, elevating "performance" from a vague numerical concept to a rigorous probability measure object.
- **Practical Value**: Challenge organizers can verify whether their chosen metrics are mathematically "valid", avoiding counter-intuitive phenomena such as "introducing a new method changes the relative ranking of existing methods."
- **Surprising Discovery**: While Matthews Correlation Coefficient (MCC) is widely recommended for evaluating unbalanced classifications, this paper proves that it does not satisfy the ranking axioms—serving as an important warning to research communities that rely heavily on MCC.
- **Generality**: The ranking score family is applicable not only to binary classification but can also be extended to multi-class classification, regression, detection, clustering, information retrieval, and other tasks (detailed examples are provided in the appendix).

## Limitations & Future Work
- **Limitation 1**: The most complete instantiation of the theoretical framework is currently limited to binary classification tasks; concrete ranking scores for more complex tasks (such as object detection) require further investigation.
- **Limitation 2**: The family of ranking scores requires $\Phi = \text{conv}$ (closure under convex combinations), and other forms of $\Phi$ have not yet been fully covered.
- **Limitation 3**: Experimental validation primarily relies on theoretical proofs and numerical simulations (approximately 6,550 uniformly distributed performance points), lacking large-scale validation in real challenge scenarios.
- **Future Work**: The authors' subsequent work [Halin et al., 2024] introduced the "Tile" visualization tool to organize ranking scores in a single plot; [Piérard et al., 2024] provided a practical guide for using Tiles, analyzing them on 74 segmentation classifiers.

## Related Work & Insights
- **vs Nguyen et al. [2023]**: This work proposed that rankings should possess three properties: reliability, meaningfulness, and mathematical consistency, but lacked a formal framework. The present work addresses these requirements at a deeper level through axiomatic definitions and the family of ranking scores.
- **vs Maier-Hein et al. [2018]**: This work critiqued the chaos in ranking practices within biomedical challenges but did not propose an alternative theory. The present work provides the theoretical foundation to guide metric selection for such challenges.
- **vs Classic Evaluation Reviews [Sokolova & Lapalme, 2009; Canbek et al., 2017]**: These works systematically compiled and analyzed various metrics but did not examine them from the perspective of ranking validity. This paper demonstrates that some commonly used metrics are mathematically invalid for ranking.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes an axiomatic mathematical foundation for performance-based ranking for the first time, filling a long-standing theoretical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorous theoretical proofs and comprehensive numerical validation, though lacking validation in real challenge scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, with the "temple" metaphor maintained throughout, rendering complex theory easy to understand.
- Value: ⭐⭐⭐⭐⭐ Profound impact on various challenges and benchmark evaluations; the discovery that MCC is unsuitable for ranking is particularly critical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation](../../ICML2025/others/bipartite_ranking_from_multiple_labels_on_loss_versus_label_aggregation.md)
- [\[ICML 2025\] Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability](../../ICML2025/others/constrained_hamiltonian_systems_on_observation-induced_fiber_bundles_theory_of_s.md)
- [\[AAAI 2026\] Measuring Model Performance in the Presence of an Intervention](../../AAAI2026/others/measuring_model_performance_in_the_presence_of_an_intervention.md)
- [\[ICML 2025\] LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection](../../ICML2025/others/lapsum_--_one_method_to_differentiate_them_all_ranking_sorting_and_top-k_selecti.md)
- [\[CVPR 2026\] Bridging Domain Expertise and Generalization for Performance Estimation](../../CVPR2026/others/bridging_domain_expertise_and_generalization_for_performance_estimation.md)

</div>

<!-- RELATED:END -->

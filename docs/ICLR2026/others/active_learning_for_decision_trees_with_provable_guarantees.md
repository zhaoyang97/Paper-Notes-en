---
title: >-
  [Paper Note] Active Learning for Decision Trees with Provable Guarantees
description: >-
  [ICLR 2026][Active Learning] This paper establishes the first theoretical guarantees for active learning with decision trees: (1) the first explicit analysis of the disagreement coefficient for decision trees…
tags:
  - "ICLR 2026"
  - "Active Learning"
  - "Decision Trees"
  - "Label Complexity"
  - "Disagreement Coefficient"
  - "Multiplicative Error"
date: 2026-05-08
content_hash: 82bc81c99fb55863
---

# Active Learning for Decision Trees with Provable Guarantees

**Conference**: ICLR 2026
**arXiv**: [2601.20775](https://arxiv.org/abs/2601.20775)  
**Code**: None  
**Area**: Active Learning / Theory
**Keywords**: Active Learning, Decision Trees, Label Complexity, Disagreement Coefficient, Multiplicative Error

## TL;DR
This paper establishes the first theoretical guarantees for active learning with decision trees: (1) the first explicit analysis of the disagreement coefficient for decision trees, yielding an $O(\ln^{OPT}(n))$ upper bound; (2) the first active learning algorithm for binary classification achieving a multiplicative error guarantee of $(1+\epsilon)$. Combining these two results yields polylogarithmic label complexity in the dataset size.

## Background & Motivation

### State of the Field

**Background**: Active learning reduces labeling costs by strategically selecting the most informative data points for annotation. Decision trees are widely used due to their interpretability and feature selection capabilities, forming the basis of Random Forests and XGBoost.

**Limitations of Prior Work**: Active learning with decision trees lacks rigorous theoretical foundations — no explicit computation of the disagreement coefficient existed, and no multiplicative-error active learning algorithm had been proposed for classification. Additive-error algorithms cannot be directly adapted to the multiplicative-error setting, since adaptively estimating the optimal error $\eta$ requires $O(1/\eta)$ label queries, negating any efficiency advantage.

**Key Challenge**: Bounding the disagreement coefficient is theoretically necessary to guarantee label efficiency, yet for the decision tree hypothesis class this bound was previously known only to be finite, without a quantitative characterization. The multiplicative-error model is strictly stronger than the additive-error model (guaranteeing perfect classification in the realizable setting), but requires fundamentally new algorithmic design.

**Goal**: Establish a theory of label complexity for active learning with decision trees.

**Key Insight**: Analyze the disagreement regions of decision trees via a LineTree decomposition, and design a novel algorithm that uses "version-space shrinkage stagnation" as a signal to lower-bound the optimal error.

**Core Idea**: Under two natural assumptions (each root-to-leaf path queries distinct feature dimensions, and grid-structured data), the disagreement coefficient of decision trees is polylogarithmic. Combined with the new multiplicative-error algorithm, this yields polylogarithmic label complexity.

## Method

### Overall Architecture
The paper makes two independent contributions: (1) an analysis of the disagreement coefficient that provides a theoretical foundation for the decision tree hypothesis class; (2) a general multiplicative-error active learning algorithm applicable to arbitrary binary classification. Combining these two results yields the polylogarithmic label complexity of Corollary 1.3.

### Key Designs

1. **Disagreement Coefficient Analysis for Decision Trees (Theorem 1.1)**:

    - **Function**: The first explicit upper and lower bounds on the disagreement coefficient for the decision tree hypothesis class.
    - **Mechanism**: Decision trees are decomposed into LineTrees (substructures corresponding to a single root-to-leaf path); the disagreement region of each LineTree is analyzed and the results are combined. Under two assumptions (each root-to-leaf path queries distinct feature dimensions, and grid-structured data distributions), this yields $\theta = O(\ln^{OPT}(n))$.
    - **Design Motivation**: The coefficient $\theta$ directly controls label complexity — the classical result of Hanneke (2014) gives label complexity $\theta \cdot \text{poly}(\text{VC-dim}, \ln n, 1/\epsilon)$. A polylogarithmic $\theta$ thus ensures polylogarithmic total label complexity.

2. **Multiplicative-Error Active Learning Algorithm (Algorithm 2, Theorem 1.2)**:

    - **Function**: The first active learning algorithm for binary classification with a multiplicative error guarantee of $(1+\epsilon)$.
    - **Mechanism**: The algorithm maintains a version space (the set of hypotheses that do not exclude the optimal classifier) and iteratively refines it. The key innovation is using "version-space shrinkage stagnation" as a signal to lower-bound the optimal error — if the version space ceases to shrink significantly within a given round, this indicates that the optimal error is already non-negligible, and refinement can be halted.
    - **Design Motivation**: Additive-error algorithms cannot be straightforwardly adapted to the multiplicative-error setting (estimating $\eta$ itself requires $O(1/\eta)$ labels, creating a catch-22). The proposed algorithm circumvents this problem through an intrinsic mechanism.

3. **Lower Bound on Label Complexity**:

    - **Function**: Demonstrates that the dependence of label complexity on $\epsilon$ is nearly optimal.
    - **Mechanism**: A lower bound is established for the simplest case of decision stumps.
    - **Design Motivation**: Validates the near-optimality of the proposed algorithm.

### Loss & Training
This paper is a purely theoretical contribution; no training procedure is involved. The algorithmic framework is based on disagreement-based active learning.

## Key Experimental Results

### Main Results
This paper is a purely theoretical contribution; no empirical results are reported.

### Key Theoretical Results

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| Theorem 1.1 | $\theta = O(\ln^{OPT}(n))$ | First disagreement coefficient bound for decision trees |
| Theorem 1.2 | Multiplicative-error algorithm | First $(1+\epsilon)$ active learning algorithm for classification |
| Corollary 1.3 | Polylogarithmic label complexity | Core result combining both contributions |
| Theorem 4.3 | Lower bound | Near-optimal dependence on $\epsilon$ |

### Key Findings
- Both assumptions are necessary: relaxing either (allowing repeated queries of the same dimension along a path, or non-grid data) leads to polynomial label complexity.
- The multiplicative-error framework guarantees perfect classification in the realizable setting — a guarantee that additive-error algorithms cannot provide.
- The version-space stagnation signal is the key technical innovation distinguishing multiplicative-error from additive-error algorithms.

## Highlights & Insights
- **Filling a theoretical gap**: Balcan et al. (2010) asserted that the disagreement coefficient of decision trees is finite but provided no quantitative bound; this paper gives the first explicit bound 16 years later — an important advance from existence to construction in theoretical computer science.
- **Non-equivalence of multiplicative and additive error**: The paper formally demonstrates that additive-error algorithms cannot be directly adapted to the multiplicative-error setting, necessitating a new algorithm — a conceptual contribution to active learning theory.
- **Necessity of assumptions**: The paper not only establishes upper bounds under sufficient conditions but also proves that the assumptions are necessary, yielding a theoretically complete picture.

## Limitations & Future Work
- The two key assumptions (distinct-dimension queries and grid-structured data) may not hold in practical applications.
- As a purely theoretical contribution, empirical validation on real-world datasets is absent.
- Although label complexity is polylogarithmic, the constants involve terms such as $2^{OPT}$, which may be large in practice.
- Only binary classification is considered; extension to multiclass settings remains an open problem.

## Related Work & Insights
- **vs. Balcan et al. (2010)**: They proved the disagreement coefficient is finite but did not quantify it; this paper establishes the explicit bound $O(\ln^{OPT}(n))$.
- **vs. Hanneke (2014)**: Hanneke's framework provides additive-error guarantees; this paper is the first to achieve multiplicative-error guarantees.
- **vs. Hopkins et al. (2021)**: Hopkins et al. use a stronger query model (comparison queries) and assume realizability; this paper uses standard label queries without assuming realizability.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Two independent theoretical contributions: the first analysis of the disagreement coefficient for decision trees and the first multiplicative-error classification algorithm.
- **Experimental Thoroughness**: ⭐⭐ — Purely theoretical paper; no experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theoretical derivations, though notation density is high.
- **Value**: ⭐⭐⭐⭐ — Fills an important gap in active learning theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](../../NeurIPS2025/others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](../../AAAI2026/others/from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](../../NeurIPS2025/others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](../../AAAI2026/others/from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)

</div>

<!-- RELATED:END -->

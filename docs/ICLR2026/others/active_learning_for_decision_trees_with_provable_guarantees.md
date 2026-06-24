---
title: >-
  [Paper Note] Active Learning for Decision Trees with Provable Guarantees
description: >-
  [ICLR 2026][Active Learning] Provides the first theoretical guarantees for active learning of decision trees: (1) Conducts the first analysis of the disagreement coefficient for decision trees and derives an $O(\ln^{OPT}(n))$ upper bound; (2) Proposes the first binary active learning algorithm achieving a $(1+\epsilon)$ multiplicative error guarantee; combining these results achieves polylogarithmic label complexity relative to the dataset size.
tags:
  - "ICLR 2026"
  - "Active Learning"
  - "Decision Trees"
  - "Label Complexity"
  - "Disagreement Coefficient"
  - "Multiplicative Error"
date: 2026-05-08
content_hash: 8d78e1f66eee0b69
---

# Active Learning for Decision Trees with Provable Guarantees

**Conference**: ICLR 2026  
**arXiv**: [2601.20775](https://arxiv.org/abs/2601.20775)  
**Code**: None  
**Area**: Active Learning / Theory  
**Keywords**: Active Learning, Decision Trees, Label Complexity, Disagreement Coefficient, Multiplicative Error  

## TL;DR
Provides the first theoretical guarantees for active learning of decision trees: (1) Conducts the first analysis of the disagreement coefficient for decision trees and derives an $O(\ln^{OPT}(n))$ upper bound; (2) Proposes the first binary active learning algorithm achieving a $(1+\epsilon)$ multiplicative error guarantee; combining these results achieves polylogarithmic label complexity relative to the dataset size.

## Background & Motivation

### Background

**Background**: Active learning reduces labeling requirements by strategically selecting the most informative data points for annotation. Decision trees are widely utilized (serving as the foundation for Random Forest and XGBoost) due to their interpretability and feature selection capabilities.

**Limitations of Prior Work**: Active learning for decision trees lacks a rigorous theoretical foundation—previously, there was no explicit calculation of their disagreement coefficient, nor were there active learning algorithms for classification tasks specifically designed for multiplicative error. Additive error algorithms cannot be adapted to the multiplicative error setting (since adaptively estimating the optimal error $\eta$ requires $O(1/\eta)$ labels, negating the efficiency advantage).

**Key Challenge**: Theoretically, a bound on the disagreement coefficient is required to guarantee label efficiency, but for the class of decision trees, this bound was previously only known to be finite and could not be quantified. The multiplicative error model is stronger than the additive error model (it guarantees perfect classification in the realizable setting), but it necessitates an entirely new algorithmic design.

**Goal**: Establish a label complexity theory for the active learning of decision trees.

**Key Insight**: Analyze the disagreement region of decision trees by decomposing them into LineTrees and design a new algorithm that utilizes a "version space contraction stagnation signal" to lower bound the optimal error.

**Core Idea**: Under two natural assumptions (querying different dimensions along the root-to-leaf path and grid-distributed data), the disagreement coefficient for decision trees is polylogarithmic. Combined with a new multiplicative error algorithm, this enables polylogarithmic label complexity.

## Method

### Overall Architecture
The paper addresses a single question: how many labels can active learning actually save for decision trees, and can this be provably guaranteed? The authors decompose this into two independent parts that are eventually combined. The first part is a purely combinatorial analysis that calculates an explicit bound for the disagreement coefficient $\theta$ of the decision tree hypothesis class, which is a prerequisite for applying existing label complexity frameworks. The second part is a general algorithm, independent of decision trees, which for the first time achieves a $(1+\epsilon)$ multiplicative error guarantee for active learning in classification tasks. Both parts are independently valid; when combined and applied to decision trees, they yield Corollary 1.3: the label complexity is polylogarithmic relative to the dataset size. Finally, a lower bound is provided to prove that the algorithm's dependence on $\epsilon$ is nearly optimal.

### Key Designs

**1. Disagreement Coefficient Analysis (Theorem 1.1): Linking "label expenditure" to a computable geometric quantity**

The label savings in active learning are theoretically determined by the disagreement coefficient $\theta$—classic results (Hanneke 2014) express label complexity as $\theta \cdot \text{poly}(\text{VC-dim}, \ln n, 1/\epsilon)$. Thus, a small $\theta$ ensures a small total label count. The challenge was that $\theta$ for decision trees was only proven to be "finite" by Balcan et al. 2010, and without an explicit calculation, the framework could not be applied. This paper decomposes a decision tree into several LineTrees (each corresponding to a sub-structure induced by a root-to-leaf path), analyzes the disagreement region of individual LineTrees, and aggregates them back into the full tree. Under two natural assumptions—querying different feature dimensions at each step along a path and data following a grid distribution—the result is $\theta = O(\ln^{OPT}(n))$, a polylogarithmic magnitude relative to the dataset size.

**2. Multiplicative Error Active Learning Algorithm (Algorithm 2, Theorem 1.2): Using "contraction stagnation" as a signal to bypass the dead-loop of estimating optimal error**

Multiplicative error models are stronger than additive ones: they guarantee perfect classification in realizable settings, but legacy algorithms cannot be directly applied. Additive error algorithms must first estimate the optimal error $\eta$ to set the precision, yet estimating $\eta$ accurately requires $O(1/\eta)$ labels—as one seeks a multiplicative guarantee (where $\eta$ is small), the estimation becomes increasingly expensive, erasing the efficiency of active learning (a catch-22). This algorithm does not explicitly estimate $\eta$. Instead, it maintains a version space (the set of classifiers not yet excluded) and refines it each round using new labels. The key innovation is reading the "contraction stagnation" signal: if the version space stops shrinking significantly in a given round, it indicates that the remaining disagreement is not noise but that the optimal error itself is already large, allowing the algorithm to terminate. This utilizes the intrinsic dynamics of the algorithm to indirectly lower-bound $\eta$, achieving the $(1+\epsilon)$ multiplicative guarantee without extra label costs for estimation.

**3. Label Complexity Lower Bound (Theorem 4.3): Proving the algorithm's $\epsilon$ dependence is nearly optimal**

An upper bound alone is insufficient; it must be shown that the algorithm is not wasting labels. The paper selects the simplest hypothesis class—decision stumps (single splits)—and constructs a lower bound demonstrating that any multiplicative error active learning algorithm must incur a certain magnitude of label consumption regarding $\epsilon$. Since decision stumps are a special case of decision trees, this lower bound implies that the proposed algorithm's dependence on $\epsilon$ is nearly optimal, with no significant theoretical room for improvement left on the table.

### Loss & Training
This work is a purely theoretical contribution and does not involve specific training procedures. The algorithmic framework is built upon disagreement-based active learning, with all guarantees measured by the number of label queries required for version space refinement.

## Key Experimental Results

### Main Results
This work is a purely theoretical contribution; no experimental results are provided.

### Key Findings

| Theorem | Content | Significance |
|------|------|------|
| Theorem 1.1 | $\theta = O(\ln^{OPT}(n))$ | First explicit disagreement coefficient bound for decision trees |
| Theorem 1.2 | Multiplicative error algorithm | First $(1+\epsilon)$ active learning for classification |
| Corollary 1.3 | Polylogarithmic label complexity | Core result combining the two main contributions |
| Theorem 4.3 | Lower bound | Proves $\epsilon$ dependence is nearly optimal |

### Key Findings
- Both assumptions are necessary: relaxing either (allowing repeated dimension queries on a path or non-grid data) leads to polynomial label complexity.
- The multiplicative error framework guarantees perfect classification in realizable settings—a feat additive error cannot achieve.
- The version space stagnation signal is the key technical innovation distinguishing multiplicative from additive error algorithms.

## Highlights & Insights
- **Filling a Theoretical Void**: Balcan et al. 2010 only asserted that the disagreement coefficient for decision trees is finite without providing a calculation; this paper provides the first explicit bound after 16 years—a significant "existence to construction" advancement in theoretical computer science.
- **In-equivalence of Multiplicative vs. Additive Error**: Formally demonstrates that additive error algorithms cannot be directly adapted to multiplicative settings, necessitating a brand-new algorithm—a conceptual contribution to active learning theory.
- **Proof of Necessity**: Not only provides an upper bound under sufficient conditions but also proves the necessity of those assumptions, ensuring theoretical completeness.

## Limitations & Future Work
- The two key assumptions (querying different dimensions and grid data) may not hold in practical applications.
- Purely theoretical contribution, lacking empirical validation of performance on real-world datasets.
- While the label complexity is polylogarithmic, the constant factors contain terms like $2^{OPT}$, which may be large in practice.
- Limited to binary classification; extensions to multi-class settings remain for future research.

## Related Work & Insights
- **vs. Balcan et al. 2010**: They proved the coefficient was finite but unquantified; this paper gives the explicit bound $O(\ln^{OPT}(n))$.
- **vs. Hanneke 2014**: Hanneke's framework uses additive error guarantees; this paper achieves multiplicative error guarantees for the first time.
- **vs. Hopkins et al. 2021**: Hopkins utilized a stronger query model (comparison queries) and assumed realizability; this work uses standard label queries and does not assume realizability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First analysis of decision tree disagreement coefficient + first multiplicative error classification algorithm.
- Experimental Thoroughness: ⭐⭐ Purely theoretical paper, no experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivation, though high notation density.
- Value: ⭐⭐⭐⭐ Fills a significant gap in active learning theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Direct Feedback Learning with Jacobian Alignment Guarantees](scaling_direct_feedback_learning_with_jacobian_alignment_guarantees.md)
- [\[ICLR 2026\] From Fields to Random Trees](from_fields_to_random_trees.md)
- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](../../NeurIPS2025/others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](../../AAAI2026/others/from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[ICLR 2026\] Exposing Mixture and Annotating Confusion for Active Universal Test-Time Adaptation](exposing_mixture_and_annotating_confusion_for_active_universal_test-time_adaptat.md)

</div>

<!-- RELATED:END -->

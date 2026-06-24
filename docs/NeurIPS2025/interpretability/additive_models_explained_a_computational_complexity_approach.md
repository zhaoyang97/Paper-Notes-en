---
title: >-
  [Paper Note] Additive Models Explained: A Computational Complexity Approach
description: >-
  [NeurIPS 2025][Interpretability][GAM] This paper presents a systematic computational complexity analysis of multiple explanation types for Generalized Additive Models (GAMs), covering 54 combinations of "component model × input domain × explanation method." It reveals that the explanation complexity of GAMs is highly sensitive to the type of input domain — a phenomenon never observed in other ML models such as decision trees or neural networks — thereby challenging the intuit…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "GAM"
  - "Explainability"
  - "Computational Complexity"
  - "Shapley Values"
  - "Sufficient Reasons"
  - "Contrastive Explanations"
date: 2026-05-08
content_hash: 2ef1c7e37776cd2a
---

# Additive Models Explained: A Computational Complexity Approach

**Conference**: NeurIPS 2025
**arXiv**: [2510.21292](https://arxiv.org/abs/2510.21292)  
**Code**: None  
**Area**: Explainable AI / Computational Complexity
**Keywords**: GAM, Explainability, Computational Complexity, Shapley Values, Sufficient Reasons, Contrastive Explanations

## TL;DR

This paper presents a systematic computational complexity analysis of multiple explanation types for Generalized Additive Models (GAMs), covering 54 combinations of "component model × input domain × explanation method." It reveals that the explanation complexity of GAMs is highly sensitive to the type of input domain — a phenomenon never observed in other ML models such as decision trees or neural networks — thereby challenging the intuitive assumption that "additive implies interpretable."

## Background & Motivation

**Background**: Generalized Additive Models (GAMs) $f(\mathbf{x}) = \beta_0 + \sum_i \beta_i f_i(\mathbf{x}_i)$ are widely regarded by the ML community as an "interpretable" model class, whose additive structure makes the relationship between inputs and outputs relatively transparent. This perception has motivated many explanation techniques (e.g., LIME) to approximate model behavior as additive, and metrics such as infidelity evaluate explanation quality via the fidelity of additive approximations.

**Limitations of Prior Work**: Although GAMs are considered interpretable, "interpretable" does not imply "efficiently computable explanations." Prior complexity-theoretic studies have focused primarily on neural networks (NP-Hard) and decision trees (polynomial time), leaving this important model class without systematic analysis. GAMs admit numerous variants — components may be splines (Smooth GAMs), neural networks (NAMs), or boosted tree ensembles (EBMs), and input domains may be discrete or continuous — and explanation complexity may vary drastically across these combinations.

**Key Challenge**: The additive decomposition structure of GAMs intuitively suggests that explanations should be easy to compute, as each feature's contribution can be analyzed independently. However, it remains unclear whether this intuition holds across all combinations. If counterexamples exist, the foundational assumption that "additive = interpretable" requires revision.

**Goal**: To comprehensively answer the following question: for different GAM types (Smooth/NAM/EBM), different input domains (enumerable discrete / general discrete / continuous), and different explanation methods (sufficient reasons, contrastive explanations, Shapley values, feature redundancy), what is the precise complexity class of computing each explanation type?

**Key Insight**: Each of the 54 "component × domain × explanation" combinations is formalized as a precise computational problem, and a complete complexity landscape is established through reductions and algorithmic constructions.

**Core Idea**: The explanation complexity of GAMs is far more varied than commonly assumed — the type of input domain is the decisive factor, a phenomenon that has never been observed in any other ML model class studied previously.

## Method

### Overall Architecture

This is a purely theoretical complexity analysis. The analysis spans the full Cartesian product of three dimensions:

- **Component Models** (3 types): Smooth GAM (splines), NAM (ReLU neural networks), EBM (boosted tree ensembles)
- **Input Domains** (3 types): Enumerable discrete (each variable takes values from a constant-size set), general discrete (bounded discrete values in binary encoding), continuous (real-valued)
- **Explanation Types** (6+): CSR (checking sufficient reasons), MSR (minimum sufficient reasons), MCR (minimum contrastive explanations), FR (feature redundancy), CC (completion counting), SHAP-R (regression Shapley values), SHAP-C (classification Shapley values)

### Key Designs

1. **Input Domain Is the Decisive Factor (Theorems 1 & 2)**:

    - Finding: For most explanation types, transitioning from enumerable discrete to general discrete or continuous domains induces an exponential complexity jump (PTIME → NP-Hard or #P-Hard).
    - Intuition: Under enumerable discrete domains, the maximum/minimum or expectation of each component $f_i$ can be computed by explicitly enumerating all possible values; such enumeration is infeasible under general discrete or continuous domains.
    - Sole exception: Feature redundancy (FR) queries are paradoxically easier in continuous domains — in the continuous case, a feature is redundant if and only if $\beta_i \cdot f_i(\mathbf{x}_i) \equiv 0$, which is efficiently checkable; in discrete domains, verifying whether the value falls below the decision boundary is generally coNP-Complete.
    - Uniqueness: This domain sensitivity is unique to GAMs — decision trees, neural networks, and tree ensembles do not exhibit this behavior.

2. **The Effect of Component Models Depends on the Input Domain (Theorem 3)**:

    - Finding: Smooth GAMs (splines) yield PTIME complexity for CSR/MSR/MCR across all input domains; however, NAMs and EBMs escalate to coNP-Hard/NP-Complete under general discrete and continuous domains.
    - Key Insight: Under enumerable discrete domains, the component model type does not affect complexity — the enumerability of the input space "shields" the internal complexity of the components — whereas under non-enumerable domains, the internal complexity of neural network and tree ensemble components becomes "exposed."
    - Proof Strategy: For NAMs and EBMs, hardness is established by reducing from the known hardness of evaluating individual ReLU network or tree ensemble components.

3. **The Regression vs. Classification Distinction Applies Only to SHAP (Theorem 4)**:

    - Finding: For Smooth GAM + enumerable discrete domain, SHAP-R (regression) is PTIME, while SHAP-C (classification) is #P-Complete.
    - Intuition: Classification introduces an additional step function $\text{step}(z) = \mathbf{1}[z \geq 0]$, whose nonlinearity breaks the linearity axiom underlying Shapley value computation. In the regression setting, Shapley values can be decomposed into independent expected value computations per component by exploiting the additive structure; in classification, the step function couples all component values together.
    - Uniqueness: Other explanation types (sufficient reasons, contrastive explanations) exhibit no complexity difference between regression and classification.

### Summary of Core Complexity Results

| Input Domain | Component Type | CSR/MSR | MCR | FR | SHAP-C | SHAP-R |
|---|---|---|---|---|---|---|
| Enumerable discrete | Any | PTIME | PTIME | coNP-C | #P-C (pseudo-P) | PTIME |
| General discrete | Smooth | PTIME | PTIME | coNP-C | #P-C | PTIME |
| General discrete | NAM/EBM | coNP-H | NP-C | coNP-C | #P-C | #P-C |
| Continuous | Smooth | PTIME | PTIME | **PTIME** | #P-C | PTIME |
| Continuous | NAM/EBM | coNP-H | NP-C | coNP-C | #P-C | #P-C |

### Interpretability Comparison: Additive vs. Non-Additive Models

| Setting | EBM/NAM | Tree Ensembles / Neural Networks |
|---|---|---|
| Enumerable discrete CSR | **PTIME** | coNP-C |
| Enumerable discrete MSR | **PTIME** | $\Sigma_2^P$-C |
| Enumerable discrete MCR | **PTIME** | NP-C |
| Enumerable discrete SHAP-R | **PTIME** | #P-C |
| General discrete / continuous CSR | coNP-C | coNP-C (no difference) |

## Key Experimental Results

This is a purely theoretical work with no empirical experiments. The core contribution is a complete complexity landscape covering 54 settings, with each combination accompanied by a precise complexity-class proof.

### Key Findings

- **GAMs Are Not Always "Easy to Explain"**: MSR for NAM/EBM under continuous domains is coNP-Hard, refuting the assumption that additive structure guarantees interpretability.
- **Smooth GAMs Are the Most "Safe"**: Sufficient reasons and contrastive explanations are PTIME across all input domains, making Smooth GAMs provably interpretable.
- **SHAP for Classification Is the Most Expensive**: It is #P-Complete in nearly all settings, even for Smooth GAMs.
- **Enumerable Discrete Domains Are a "Safe Harbor"**: All GAMs achieve PTIME complexity for CSR/MSR/MCR in this domain; the practical implication is that data discretization can render explanation computation efficient.
- **NAMs Are More Interpretable Than Standalone NNs**: Under enumerable discrete domains, explanations for NAMs are PTIME, whereas general neural networks are at least NP-Hard — the additive structure does help, but only within specific input domains.

## Highlights & Insights

- **Domain Sensitivity Is a Genuinely Novel Finding**: Prior complexity studies of other models (decision trees, NNs, tree ensembles) have never revealed this phenomenon; the unique additive structure of GAMs makes the input domain a decisive "lever variable."
- **Warning That "Interpretable Model ≠ Efficiently Explainable"**: This conclusion carries significant implications for the broader XAI community — one cannot assume explanation computation is simple merely because the model structure appears simple.
- **Clear Practical Recommendations**: (1) Discretizing inputs can render explanation computation tractable; (2) quantizing model coefficients can reduce #P problems to pseudo-polynomial time.
- **Transferable Proof Techniques**: The approach of reducing the hardness of NN verification to the hardness of explaining NAM components generalizes to other additively decomposed model families.

## Limitations & Future Work

- **Only Univariate Components Are Covered**: GAMs in practice frequently employ higher-order interaction terms $f_{ij}(\mathbf{x}_i, \mathbf{x}_j)$ (as in GA2M and related variants); how higher-order interactions affect complexity remains an important open question.
- **Gap Between Theory and Practice**: Although polynomial-time algorithms are provided, their empirical runtime on real-world GAMs has not been implemented or evaluated.
- **Approximation Algorithms Unexplored**: For intractable settings, whether approximation algorithms with provable guarantees exist is an open and practically important question.
- **Concurvity Not Considered**: When components are highly correlated, the effective complexity of explaining a GAM may decrease, but this scenario is not addressed.

## Related Work & Insights

- **vs. Barceló et al. (NeurIPS 2020)**: That work established the foundational framework for analyzing the explanation complexity of NNs and decision trees; this paper extends the analysis to the GAM dimension and uncovers novel complexity phenomena attributable to the additive structure.
- **vs. Complexity Analysis of Linear Models**: Linear models are a special case of GAMs (with identity component functions); this paper generalizes the analysis to nonlinear components and finds that nonlinearity substantially reshapes the complexity landscape.
- **Practical Implications**: For high-stakes applications requiring interpretability guarantees (e.g., healthcare, finance), the combination of Smooth GAM with enumerable discretized inputs is recommended as the most reliable choice.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Domain sensitivity is a genuinely new finding; the complete landscape over 54 combinations constitutes a significant contribution to the field.
- Experimental Thoroughness: ⭐⭐⭐ No experiments, as this is purely theoretical work, but theoretical coverage is exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Result tables are clear, and the verbal summaries of key insights are accessible.
- Value: ⭐⭐⭐⭐ Significant impact on XAI theory research and practical guidance for GAM model selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](spex_a_spectral_approach_to_explainable_clustering.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[ICML 2025\] Validating Mechanistic Interpretations: An Axiomatic Approach](../../ICML2025/interpretability/validating_mechanistic_interpretations_an_axiomatic_approach.md)
- [\[ICML 2025\] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models](../../ICML2025/interpretability/inference-time_decomposition_of_activations_itda_a_scalable_approach_to_interpre.md)
- [\[ICML 2025\] A Reasoning-Based Approach to Cryptic Crossword Clue Solving](../../ICML2025/interpretability/a_reasoning-based_approach_to_cryptic_crossword_clue_solving.md)

</div>

<!-- RELATED:END -->

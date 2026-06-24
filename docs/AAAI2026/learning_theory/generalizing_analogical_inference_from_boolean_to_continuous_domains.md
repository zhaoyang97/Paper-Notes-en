---
title: >-
  [Paper Note] Generalizing Analogical Inference from Boolean to Continuous Domains
description: >-
  [AAAI 2026][AI Foundations][Boolean domain generalization] This paper revisits the theoretical foundations of analogical inference: it first constructs a counterexample demonstrating the failure of classical generalization bounds in the Boolean domain, then proposes a unified analogical inference framework based on parameterized generalized means, extending discrete classification to continuous regression domains.
tags:
  - "AAAI 2026"
  - "AI Foundations"
  - "Analogical Reasoning"
  - "Boolean domain generalization"
  - "continuous domain regression"
  - "generalized mean"
  - "error bounds"
date: 2026-05-08
content_hash: bcd82b2bb07f7db9
---

# Generalizing Analogical Inference from Boolean to Continuous Domains

**Conference**: AAAI 2026
**arXiv**: [2511.10416](https://arxiv.org/abs/2511.10416)  
**Code**: None  
**Area**: AI Foundations / Analogical Reasoning
**Keywords**: analogical reasoning, Boolean domain generalization, continuous domain regression, generalized mean, error bounds

## TL;DR

This paper revisits the theoretical foundations of analogical inference: it first constructs a counterexample demonstrating the failure of classical generalization bounds in the Boolean domain, then proposes a unified analogical inference framework based on parameterized generalized means, extending discrete classification to continuous regression domains.

## Background & Motivation

**Background**: Analogical reasoning (four-term relations of the form $a:b::c:d$) is a fundamental mechanism of human cognition, with broad applications in few-shot learning, transfer learning, morphological analysis, and word vector evaluation. Theoretical foundations in the Boolean domain were established by Couceiro et al. (2017, 2018): analogical inference is exact for affine functions and admits a $4\epsilon$ probabilistic error bound for near-affine functions.

**Limitations of Prior Work**:
- Existing theory is restricted to discrete attribute spaces and binary classification, and cannot handle regression tasks or continuous domains.
- More critically, even within the Boolean domain, the classical generalization bound itself is flawed.

**Key Challenge**: Analogical reasoning is widely applied in practice over continuous domains (e.g., word vector analogies), yet theoretical guarantees are entirely absent; moreover, the core theorem ($4\epsilon$ bound, Theorem 3) underpinning discrete domain theory is incorrect.

**Goal**: To correct the erroneous theoretical bound in the Boolean domain and extend analogical inference to continuous domains, providing theoretical guarantees for regression-based analogical reasoning.

**Key Insight**: A parameterized analogical proportion is defined via Hölder generalized means, constructing a unified framework that encompasses both Boolean classification and continuous regression.

**Core Idea**: The generalized mean parameter $p$ defines the analogical proportion $a:b::^p c:d$ over continuous domains, characterizes the class of functions preserving analogical structure, and derives worst-case and average-case error bounds under smoothness assumptions.

## Method

### Overall Architecture

The theoretical framework proceeds in three progressive layers:

1. **Counterexample Construction**: Refuting the classical Boolean domain generalization bound.
2. **Continuous Domain Analogy Framework**: Parameterized analogy definition based on generalized means.
3. **Error Analysis**: Characterizing analogy-preserving functions and deriving error bounds.

### Key Designs

1. **Counterexample to the Classical Bound (Section 3)**:

    - A function $f: \mathbb{B}^4 \to \mathbb{B}$ is constructed that outputs 1 only at $\mathbf{x} = \mathbf{1}$ and 0 elsewhere.
    - The distance from $f$ to the affine class $\mathcal{L}$ is $d(f, \mathcal{L}) = 1/16$.
    - An exhaustive algorithm (enumerating $2^{15}$ subsets) yields $P(\text{err}_{S,f} > 0) \geq 0.42$.
    - However, the classical Theorem 3 predicts an upper bound of $4 \times 1/16 = 0.25$, yielding a contradiction.
    - This constitutes an important theoretical correction, refuting a core theorem that has been widely cited in the field for a decade.
    - Intuition behind the counterexample: when a training set of all-zero instances meets an all-one test point, the analogical inference system systematically predicts the unique label 1 as 0.

2. **Parameterized Analogy via Generalized Means (Section 4)**:

    - Generalized mean definition: $m_p(x_1, ..., x_n) = \lim_{r \to p} (\frac{1}{n}\sum x_i^r)^{1/r}$
    - $p = 1$ corresponds to the arithmetic mean, $p = 0$ to the geometric mean, and $p = -1$ to the harmonic mean.
    - Analogy definition: $(a,b,c,d) \in \mathbb{R}_+^4$ satisfies $a:b::^p c:d$ if and only if $m_p(a,d) = m_p(b,c)$.
    - Key properties: for any four strictly increasing positive reals, there exists a unique analogical exponent $p$; any such analogy can be reduced to an equivalent arithmetic analogy; solutions always exist for increasing sequences.
    - The notions of analogical root and analogical extension are generalized from the Boolean domain to the continuous domain, governed by the parameter pair $(\mathbf{p}; q)$ controlling the analogical exponents for attribute and label domains respectively.

3. **Characterization of Analogy-Preserving Functions (Sections 4.3 and 5)**:

    - Core theorem (Proposition 9): A continuous function $f$ belongs to $AP_{(\mathbf{p};q)}$ if and only if $f$ maps analogies of exponent $\mathbf{p}$ to analogies of exponent $q$.
    - When $p = q = 1$ (arithmetic analogy), the analogy-preserving functions are exactly affine functions, recovering the classical Boolean domain result.
    - In the general case, analogy-preserving functions form a family of generalized power functions with rich structural properties.
    - This provides a function-theoretic foundation for analogical inference in regression settings.

4. **Error Bounds in Continuous Domains (Section 5)**:

    - An appropriate function distance metric suited to generalized analogies is introduced.
    - Under smoothness assumptions, the following are derived:
     - **Worst-case bound (uniform bound)**: For functions that are $\epsilon$-close to the analogy-preserving function class, the maximum error of analogical inference is bounded.
     - **Average-case bound (probabilistic bound)**: Under random selection of training sets, the expected inference error is bounded.
    - These bounds provide PAC-learning-style theoretical guarantees for analogical inference in continuous domains.

### Loss & Training

This paper is a purely theoretical work and does not involve training. The core contributions are theorems and their proofs.

## Key Experimental Results

### Main Results

The paper is a theoretical contribution; the central "experiment" is counterexample verification:

| Theoretical Quantity | Value |
|---|---|
| $d(f, \mathcal{L})$ | 1/16 = 0.0625 |
| Classical theorem predicted upper bound $4\epsilon$ | 0.25 |
| Exhaustively computed actual lower bound | ≥ 0.42 |
| Violation gap | 0.42 > 0.25 (contradiction) |

The exhaustive algorithm completes in approximately 30 seconds for $n=4$ ($2^{15}$ subsets).

### Ablation Study

Not applicable.

### Key Findings

- Theorem 3 of Couceiro et al. (2018) ($P(\text{err}_{S,f} > \delta) \leq 4\epsilon(1-\delta)$) fails even at $\delta = 0$.
- The generalized mean parameter $p$ provides a natural "knob" that unifies different types of analogies (arithmetic, geometric, harmonic, etc.).
- Analogy-preserving functions in the continuous domain are generalized power functions, structurally richer than the affine functions of the Boolean domain.

## Highlights & Insights

- Refuting a theoretical result that has been widely cited for a decade demands both intellectual courage and rigor. The counterexample, while elementary (a 4-dimensional Boolean function), points to a fundamental flaw in the theoretical framework.
- The parameterization scheme based on generalized means is particularly elegant: a single parameter $p$ subsumes the classical analogical concepts of arithmetic, geometric, and harmonic means.
- Connecting analogical reasoning to regression tasks opens new territory for a theoretical field long confined to classification.

## Limitations & Future Work

- The framework considers only the positive real domain $\mathbb{R}_+$ and does not directly extend to the general real line including negative values (though many applications such as image processing are naturally non-negative).
- The new error bounds have not yet been connected to practical analogical classification or regression algorithms, and empirical validation is lacking.
- While the counterexample identifies the failure of the old bound, it does not provide a corrected and improved bound for the Boolean domain.
- The selection of the analogical exponent $p$ may pose challenges in practice — how should the optimal $p$ be determined for a given dataset?

## Related Work & Insights

- The Boolean domain analogical inference theory of Couceiro et al. (2017, 2018) serves as the direct point of departure for this work.
- The parameterized analogy based on generalized means proposed by Lepage (2024) is the inspirational source for the continuous domain framework.
- The word2vec analogy task of Mikolov et al. (2013) (king:queen::man:woman) represents a canonical application scenario for continuous domain analogy.
- Insight: The theoretical foundations of analogical reasoning may require comprehensive reconstruction, particularly in high-dimensional continuous spaces.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Refutation of a classical theorem combined with an entirely new continuous domain framework constitutes a strong theoretical breakthrough.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work; the counterexample verification is rigorous, but empirical data are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, the logical flow is clear, and the narrative arc from counterexample to generalization is well-paced.
- Value: ⭐⭐⭐⭐ Provides important corrections and advances to the theory of analogical reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understanding the Parameter Space Geometry of Transformers Encoding Boolean Functions](../../ICML2026/learning_theory/understanding_the_parameter_space_geometry_of_transformers_encoding_boolean_func.md)
- [\[ICLR 2026\] Persistence Spheres: Bi-Continuous Representations of Persistence Diagrams](../../ICLR2026/learning_theory/persistence_spheres_bi-continuous_representations_of_persistence_diagrams.md)
- [\[ICLR 2026\] Multiple-Prediction-Powered Inference](../../ICLR2026/learning_theory/multiple-prediction-powered_inference.md)
- [\[ICLR 2026\] Variational Inference for Cyclic Learning](../../ICLR2026/learning_theory/variational_inference_for_cyclic_learning.md)
- [\[ICLR 2026\] Discounted Online Convex Optimization: Uniform Regret Across a Continuous Interval](../../ICLR2026/learning_theory/discounted_online_convex_optimization_uniform_regret_across_a_continuous_interva.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] The Counting Power of Transformers
description: >-
  [ICLR 2026][Transformer expressivity] This paper proves that Transformers can express not only (semi-)linear counting properties but all **semi-algebraic counting properties** (i.e., Boolean combinations of multivariate polynomial inequalities), generalizing all prior results on the counting power of Transformers and deriving novel undecidability conclusions.
tags:
  - ICLR 2026
  - Transformer expressivity
  - counting properties
  - semi-algebraic properties
  - undecidability
  - formal languages
date: 2026-05-08
content_hash: 23566fb652774979
---

# The Counting Power of Transformers

**Conference**: ICLR 2026
**arXiv**: [2505.11199](https://arxiv.org/abs/2505.11199)
**Code**: None
**Area**: Transformer Theory / Formal Languages
**Keywords**: Transformer expressivity, counting properties, semi-algebraic properties, undecidability, formal languages

## TL;DR

This paper proves that Transformers can express not only (semi-)linear counting properties but all **semi-algebraic counting properties** (i.e., Boolean combinations of multivariate polynomial inequalities), generalizing all prior results on the counting power of Transformers and deriving novel undecidability conclusions.

## Background & Motivation

- **Central role of counting properties in Transformer research**: Determining whether certain tokens appear more frequently than others (e.g., MAJORITY: $|w|_a > |w|_b$) is a standard benchmark for studying Transformer expressivity.
- **Prior results cover only linear properties**: Existing frameworks (C-RASP, logical languages, etc.) handle only linear expressions such as $|w|_a + |w|_b > 2 \cdot |w|_c$.
- **Practical need for non-linear counting**: In information retrieval, co-occurrence features such as $\#(\text{nvidia}) \cdot \#(\text{intel}) \cdot \#(\text{deal})$ require polynomial expressions.
- **Core research question**: What counting properties can Transformers express? Can they go beyond linearity?

## Method

### Formalization of Counting Properties

For an alphabet $\Sigma = \{a_1, \ldots, a_m\}$, the Parikh mapping $\Psi(w) = (|w|_{a_1}, \ldots, |w|_{a_m}) \in \mathbb{N}^m$ records the occurrence count of each letter.

- **Semi-linear counting properties**: Defined by Boolean combinations of linear inequalities, e.g., $|w|_a > |w|_b$.
- **Semi-algebraic counting properties**: Defined by Boolean combinations of arbitrary multivariate polynomial inequalities, e.g., $7|w|_a \cdot |w|_b \cdot |w|_c + 2|w|_d - 8|w|_e > 10$.

### Core Theorem 1: Transformers Capture All Semi-Algebraic Counting Properties

**Theorem 1.1**: (Softmax) Transformers can express all semi-algebraic counting properties—that is, Boolean combinations of multivariate polynomial inequalities of arbitrary degree.

Key construction ideas:
1. Uniform attention layers in NoPE (no positional encoding) Transformers can compute the average of letter counts in the input sequence.
2. Through multi-layer composition and feedforward nonlinearities, count ratios can be recovered from the averages.
3. Feedforward networks are used to approximate polynomial functions.
4. No positional encoding or masking is required.

### Core Theorem 2: Complete Characterization of Natural Subclasses

**Theorem 1.2**: NoPE-AHAT (average hard-attention Transformers without positional encoding) and NoPE-AHAT[U] (the uniform-layers-only variant) precisely characterize semi-algebraic counting properties.

$$\text{NoPE-AHAT} = \text{NoPE-AHAT[U]} = \text{semi-algebraic counting properties}$$

This is surprising because whether AHAT is subsumed by SMAT remains an open problem.

### Core Theorem 3: Universality (Corollary on Turing Completeness)

**Theorem 1.3**: Every recursively enumerable counting property is a projection of a language recognized by some NoPE-AHAT[U] (and hence by SMAT). Only **two attention layers** are required.

This follows from Matiyasevich's resolution of Hilbert's tenth problem (every r.e. set is a projection of the set of integer zeros of a polynomial) combined with Theorem 1.1.

### Core Theorem 4: Undecidability

**Theorem 1.4**: Given a NoPE-AHAT[U] or SMAT with only two attention layers, deciding whether its language is empty is **undecidable**.

This result is significantly stronger than prior work, which required complex positional encodings and architectures to obtain undecidability.

### Comparison with C-RASP

The paper proves that C-RASP can capture only **linear** counting properties. Experiments show that non-linear properties such as $L_k: |w|_a^k \geq |w|_b$ for $k \geq 2$ are also trainable, indicating that C-RASP provides only an incomplete characterization of efficiently learnable properties.

## Key Experimental Results

### Trainability of Non-Linear Counting Properties

| Language $L_k$ | Definition | Accuracy | Length Generalization |
|----------------|------------|----------|-----------------------|
| $L_1$ (linear) | $\|w\|_a \geq \|w\|_b$ | 100% | ✓ |
| $L_2$ (quadratic) | $\|w\|_a^2 \geq \|w\|_b$ | ~100% | ✓ |
| $L_3$ (cubic) | $\|w\|_a^3 \geq \|w\|_b$ | ~100% | ✓ |

### Comparison with PARITY

| Property | Trainability | Length Generalization |
|----------|--------------|-----------------------|
| MAJORITY | ✓ efficient | ✓ generalizes |
| Semi-algebraic ($L_k$, $k \geq 2$) | ✓ efficient | ✓ generalizes |
| PARITY | ✗ difficult | ✗ does not generalize |

- Semi-algebraic properties (including non-linear ones) exhibit trainability comparable to linear MAJORITY.
- The sharp contrast with PARITY further supports the view that the difficulty of PARITY does not stem from non-linearity.

## Highlights & Insights

1. **Breaking the linear barrier**: The counting power of Transformers is extended from linear to polynomial expressions of arbitrary degree.
2. **Surprising expressivity of NoPE**: Semi-algebraic counting is achievable without positional encoding or masking.
3. **Complete characterization**: NoPE-AHAT corresponds precisely to semi-algebraic counting properties.
4. **Deep mathematical connection**: Hilbert's tenth problem bridges algebraic geometry and Transformer theory.
5. **Limitations of C-RASP**: It is formally proved that C-RASP captures only the linear fragment.

## Limitations & Future Work

- Theoretical constructions may require a large number of parameters and layers, creating a gap with practical training settings.
- Results focus on counting properties and do not address properties that depend on token order.
- Experiments are small-scale and primarily verify trainability.
- Constructive proofs may yield unnatural network weights.

## Related Work & Insights

- **Transformer expressivity**: Hahn 2020 (communication complexity lower bounds); Pérez et al. 2021 (Turing completeness).
- **C-RASP**: Huang et al. 2025 (formalizing the RASP-L conjecture).
- **Formal languages and neural networks**: RASP, limit transformers.
- **Hilbert's tenth problem**: Matiyasevich 1993.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneering generalization of Transformer counting power to the semi-algebraic setting.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rigorous theoretical proofs connecting algebraic geometry and computational theory.
- **Experimental Thoroughness**: ⭐⭐⭐ — Experiments primarily validate trainability at limited scale.
- **Value**: ⭐⭐⭐ — Primarily a theoretical contribution, yet carries far-reaching implications for understanding the expressive boundaries of Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Stronger Normalization-Free Transformers](../../CVPR2026/others/stronger_normalization-free_transformers.md)
- [\[AAAI 2026\] The Limitations and Power of NP-Oracle-Based Functional Synthesis Techniques](../../AAAI2026/others/the_limitations_and_power_of_np-oracle-based_functional_synthesis_techniques.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](../../AAAI2026/others/model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[AAAI 2026\] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](../../AAAI2026/others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](../../AAAI2026/others/variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)

</div>

<!-- RELATED:END -->

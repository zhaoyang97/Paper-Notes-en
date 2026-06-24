---
title: >-
  [Paper Note] Validating Mechanistic Interpretations: An Axiomatic Approach
description: >-
  [ICML2025][Interpretability][mechanistic interpretability] Drawing inspiration from the concept of abstract interpretation in program analysis, this paper proposes an axiomatic framework to formally define and validate the mechanistic interpretations of neural networks, and verifies the effectiveness of this framework through two case studies: a 2-SAT solver and modular addition.
tags:
  - "ICML2025"
  - "Interpretability"
  - "mechanistic interpretability"
  - "axiomatic validation"
  - "abstract interpretation"
  - "2-SAT"
  - "compositional analysis"
date: 2026-05-08
content_hash: ce732ece2dc456b3
---

# Validating Mechanistic Interpretations: An Axiomatic Approach

**Conference**: ICML2025  
**arXiv**: [2407.13594](https://arxiv.org/abs/2407.13594)  
**Code**: [GitHub](https://github.com/nilspalumbo/axiomatic-validation)  
**Area**: Interpretability / Mechanistic Interpretability  
**Keywords**: mechanistic interpretability, axiomatic validation, abstract interpretation, 2-SAT, compositional analysis

## TL;DR

Drawing inspiration from the concept of abstract interpretation in program analysis, this paper proposes an axiomatic framework to formally define and validate the mechanistic interpretations of neural networks, and verifies the effectiveness of this framework through two case studies: a 2-SAT solver and modular addition.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Mechanistic Interpretability aims to reverse-engineer the internal computational processes of neural networks, representing them as human-understandable circuits and features. However, two core challenges currently exist:

**Vague Definitions**: There lacks a unified standard for what constitutes a "valid" mechanistic interpretation, and existing research often relies on ad-hoc evaluation methods.

**Lack of Compositional Validation**: Prior work (such as the modular addition analysis in Nanda et al. 2023) only validates the equivalence of individual components, ignoring the cascading accumulation effects of errors across multiple components.

The authors observe a deep analogy between mechanistic interpretability and **abstract interpretation** (Cousot & Cousot 1977) in program analysis—both attempt to describe the behavior of a complex system with approximate semantics. Based on this, this paper proposes a system of axioms to formally characterize what constitutes a "valid mechanistic interpretation."

## Method

### Core Formalization Framework

Neural networks are viewed as programs in a pure functional language $\lambda_T$, where basic operations include Embed, Unembed, Lin, ReLU, Self-Attention, etc. Mechanistic interpretations are programs in another human-readable language $\lambda_H$.

**Key Concepts**:
- **Decomposition**: Decomposing a model $t$ into a list of components $d_t = [d_t[1], d_t[2], \ldots]$ such that the composition of the components equals the original model.
- **Abstraction Function $\alpha_i$**: Mapping the real-valued activations of the model to human-interpretable discrete symbols (akin to a probe).
- **Concretization Function $\gamma_i$**: Mapping abstract symbols back to the representation space of the model.

### Four Core Axioms

Given a model $t$, decomposition $d_t$, interpretation $h$ with its decomposition $d_h$, and input distribution $\mathcal{D}$, an $\epsilon$-accurate mechanistic interpretation must satisfy:

**Axiom 1 ($\epsilon$-Prefix Equivalence)**: The abstract output of each prefix must match the output of the interpreted prefix.

$$\forall i: \Pr_{x \sim \mathcal{D}}[\alpha_i \circ d_t[:i{+}1](x) = d_h[:i{+}1] \circ \alpha_0(x)] \geq 1 - \epsilon$$

**Axiom 2 ($\epsilon$-Component Equivalence)**: Each independent component does not introduce excessive error.

$$\forall i: \Pr_{x \sim \mathcal{D}}[\alpha_i \circ d_t[:i{+}1](x) = d_h[i] \circ \alpha_{i-1} \circ d_t[:i](x)] \geq 1 - \epsilon$$

**Axiom 3 ($\epsilon$-Prefix Substitutability)**: Replacing the model prefix with the interpreted prefix leaves the final output essentially unchanged.

$$\forall i: \Pr_{x \sim \mathcal{D}}[t(x) = d_t[i{+}1:] \circ \gamma_i \circ d_h[:i{+}1] \circ \alpha_0(x)] \geq 1 - \epsilon$$

**Axiom 4 ($\epsilon$-Component Substitutability)**: Replacing an individual component of the model with the corresponding interpreted component leaves the output essentially unchanged.

$$\forall i: \Pr_{x \sim \mathcal{D}}[t(x) = d_t[i{+}1:] \circ \gamma_i \circ d_h[i] \circ \alpha_{i-1} \circ d_t[:i](x)] \geq 1 - \epsilon$$

> **Key Difference**: Axioms 1/3 test compositionality (error cascading), and Axioms 2/4 test individual components. Axiom 2 does not imply Axiom 1, as errors can accumulate across components.

### Validation Method

The axioms can be efficiently validated via statistical testing: computing violation rates on a test set and constructing confidence intervals using the Clopper-Pearson method.

## Key Experimental Results

### Case Study 1: 2-SAT Solver

A 2-layer ReLU decoder-only Transformer ($d=128$, 1 head for the first layer, 4 heads for the second layer, and a 512-dimensional MLP hidden layer) is trained on a 2-SAT problem with 10 clauses and 5 variables, achieving a test accuracy of **99.76%**.

**Algorithms found via reverse engineering**:
- **Layer 1 = Parser**: Parses token sequences into clause lists via attention patterns.
- **Layer 2 = Evaluator**: Only 34 neurons in the MLP hidden layer are active, determining satisfiability via brute-force assignment.

### Main Results

| Axiom | Component | Decision Tree Interpretation $\epsilon$ | Disjunction Interpretation $\epsilon$ |
|------|------|----------------------|---------------------|
| Axiom 1 (Prefix Equivalence) | Layer 1 | ≈0.0000374 | ≈0.0000374 |
| Axiom 1 (Prefix Equivalence) | Hidden Layer | ≈0.182 | ≈0.309 |
| Axiom 3 (Prefix Substitutability) | Layer 1 | ≈0.0418 | ≈0.0418 |
| Axiom 3 (Prefix Substitutability) | Hidden Layer | ≈0.0128 | ≈0.00290 |
| Axiom 2 (Component Equivalence) | Output Layer | ≈0.00433 | ≈0.00433 |

### Case Study 2: Modular Addition (Nanda et al. 2023)

The authors validate that Nanda et al.'s mechanistic interpretation (trigonometric identity-based algorithm) of the modular addition model indeed satisfies all axioms. Notably, Nanda et al.'s original paper only provided evidence akin to Axioms 2 and 4, **lacking the compositional validation of Axioms 1 and 3**.

### Key Findings

- The decision tree interpretation outperforms the disjunction interpretation on intermediate representation equivalence ($\epsilon$: 0.182 vs 0.309), but their gap on substitutability is minimal.
- The **amplification step** in the $\gamma$ function is crucial: removing amplification worsens the prefix substitutability $\epsilon$ from 0.0128 to 0.249.
- Compositional axioms (1 and 3) indeed cannot be derived from component axioms (2 and 4), and experiments confirm the problem of error accumulation.

## Highlights & Insights

1. **Clear Theoretical Contribution**: Elevates mechanistic interpretation from ad-hoc evaluations to an axiomatic framework, presenting an elegant analogy to abstract interpretation.
2. **Necessity of Compositional Axioms**: Explicitly points out for the first time that validating only single-component equivalence is insufficient, as error cascading is a real issue.
3. **Bidirectional Validation**: The design of the $\alpha$ (abstraction) and $\gamma$ (concretization) function pair checks both the semantic consistency of intermediate representations and the end-to-end behavior after replacement.
4. **High Practicality**: Axiom validation only requires forward passes and statistical testing, minimizing computational overhead.
5. **Intriguing 2-SAT Case Study**: Reveals that the model learns a brute-force evaluation algorithm, with MLP neurons demonstrating sparsity (only 34 active out of 512).

## Limitations & Future Work

1. **Evaluated Only on Small-Scale Models**: Confined to 2-SAT (5 variables, 10 clauses) and modular addition (mod 113), making its scalability to actual LLMs unclear.
2. **$\alpha$ and $\gamma$ Functions Require Manual Design**: There is currently no automated method to select appropriate abstraction/concretization functions.
3. **Discretization Challenge of Continuous Intermediate Representations**: In the modular addition case study, a simple discretization (rounding to 1 decimal place) causes $\epsilon=1$ for Axioms 1/2, showing that discretization choices heavily dictate the results.
4. **Axioms 5/6 Unimplemented**: The paper acknowledges that the current four axioms might be insufficient and plans to address this in future work.
5. **Only Sequential Composition Covered**: Although parallel composition is discussed in the appendix, the actual case studies only involve sequential decomposition.

## Related Work & Insights

- **Causal Abstraction** (Geiger et al. 2021, 2024): Validates interpretations through causal interventions, which is complementary to this work but lacks the $\gamma$ function.
- **Causal Scrubbing** (Chan et al. 2022): Features a similar concretization operation but lacks the $\alpha$ function.
- **Circuit Analysis** (Wang et al. 2023; Conmy et al. 2023): Discovers functional subgraphs but lacks validation standards for compositionality.
- **Abstract Interpretation** (Cousot & Cousot 1977): A classic framework in program analysis, serving as the core theoretical source for this study.

## Rating

- Novelty: ⭐⭐⭐⭐ — The axiomatic framework is novel, and the analogy to abstract interpretation is profound.
- Experimental Thoroughness: ⭐⭐⭐ — The two case studies are thoroughly analyzed, but limited in scale.
- Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous and clear, with detailed case studies.
- Value: ⭐⭐⭐⭐ — Establishes an evaluation benchmark for mechanistic interpretability, providing directional significance to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracking Equivalent Mechanistic Interpretations Across Neural Networks](../../ICLR2026/interpretability/tracking_equivalent_mechanistic_interpretations_across_neural_networks.md)
- [\[ICML 2025\] MIB: A Mechanistic Interpretability Benchmark](mib_a_mechanistic_interpretability_benchmark.md)
- [\[ICML 2025\] A Reasoning-Based Approach to Cryptic Crossword Clue Solving](a_reasoning-based_approach_to_cryptic_crossword_clue_solving.md)
- [\[ICML 2025\] To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models](to_steer_or_not_to_steer_mechanistic_error_reduction_with_abstention_for_languag.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](../../NeurIPS2025/interpretability/spex_a_spectral_approach_to_explainable_clustering.md)

</div>

<!-- RELATED:END -->

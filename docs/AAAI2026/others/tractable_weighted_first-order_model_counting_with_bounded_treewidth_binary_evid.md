---
title: >-
  [Paper Note] Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence
description: >-
  [AAAI 2026][Weighted first-order model counting] A polynomial-time (in domain size) algorithm is proposed for computing weighted first-order model counting (WFOMC) of the $\text{FO}^2$ and $\text{C}^2$ fragments with bounded-treewidth binary evidence, resolving an open problem on counting stable seating arrangements on bounded-treewidth bounded-degree graphs.
tags:
  - AAAI 2026
  - Weighted first-order model counting
  - binary evidence
  - treewidth
  - lifted inference
  - combinatorial counting
date: 2026-05-08
content_hash: 0276c0b75c47b57b
---

# Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence

**Conference**: AAAI 2026
**arXiv**: [2511.09174](https://arxiv.org/abs/2511.09174)
**Code**: None
**Area**: Artificial Intelligence / Logic & Reasoning
**Keywords**: Weighted first-order model counting, binary evidence, treewidth, lifted inference, combinatorial counting

## TL;DR

A polynomial-time (in domain size) algorithm is proposed for computing weighted first-order model counting (WFOMC) of the $\text{FO}^2$ and $\text{C}^2$ fragments with bounded-treewidth binary evidence, resolving an open problem on counting stable seating arrangements on bounded-treewidth bounded-degree graphs.

## Background & Motivation

1. **State of the Field**: Weighted first-order model counting (WFOMC) requires computing the weighted sum over all models of a given first-order logic sentence on a specified domain. It is a foundational problem in statistical relational learning — Markov logic networks, probabilistic logic programs, and probabilistic databases all reduce to WFOMC. The two-variable fragment $\text{FO}^2$ (and its extension with counting quantifiers $\text{C}^2$) is known to be domain-liftable (polynomial-time) in the absence of evidence.
2. **Limitations of Prior Work**: Imposing binary evidence on WFOMC (fixing the truth values of certain binary predicates) renders the problem #P-hard even for the originally domain-liftable $\text{FO}^2$ fragment. Intuitively, binary evidence breaks the symmetry among domain elements — without evidence, elements can be grouped by their 1-type and treated interchangeably; binary evidence makes each pair of elements behave differently.
3. **Root Cause**: Binary evidence is indispensable in practical applications (graph structures, social networks, etc.), yet its introduction prevents efficient lifted inference.
4. **Paper Goals**: To provide a polynomial-time WFOMC algorithm for $\text{FO}^2$ and $\text{C}^2$ under the restriction that the Gaifman graph of the binary evidence has bounded treewidth.
5. **Starting Point**: Dynamic programming is performed over a nice tree decomposition of the Gaifman graph, leveraging existing 1-type configuration techniques to handle symmetric elements while treating elements within each bag individually.
6. **Core Idea**: Exploit the bounded-treewidth structure of the Gaifman graph to localize the influence of binary evidence through dynamic programming over the tree decomposition.

## Method

### Overall Architecture

Given a $\text{UFO}^2$ sentence $\Psi$, domain $\Delta$, weight function $(w, \bar{w})$, unary evidence $\mathcal{U}$, and bounded-treewidth binary evidence $\mathcal{E}$, a nice tree decomposition $T(V_T, E_T)$ of the Gaifman graph of $\mathcal{E}$ is first constructed. The algorithm then recursively computes the weighted sum $f(u, \tau, \boldsymbol{\zeta})$ of partial models in a bottom-up traversal over the tree.

### Key Designs

1. **Recursive Framework and Partial Models**:
    - **Function**: Define the recursive computation objective at each tree decomposition node.
    - **Mechanism**: For each node $u$ in the tree, $f(u, \tau, \boldsymbol{\zeta})$ is defined as the weighted sum of partial models satisfying the relevant constraints, where $\tau$ is the 1-type assignment for elements in bag $B_u$ and $\boldsymbol{\zeta}$ is the 1-type configuration for elements $S_u$ outside the bag. The final $\text{WFOMC} = \sum_{\boldsymbol{\zeta}} f(\text{root}, \top, \boldsymbol{\zeta})$.
    - **Design Motivation**: The tree decomposition guarantees that interactions between elements outside the bag and the bag are mediated solely through elements within the bag — elements outside can be handled symmetrically via 1-type configurations (preserving polynomial-time complexity), while the finite number of elements inside each bag are processed individually.

2. **Recursion for Four Node Types**:
    - **Function**: Define separate recursions for leaf, introduce, forget, and merge nodes.
    - **Mechanism**:
      - Leaf node: $f(u, \top, \mathbf{0}) = 1$
      - Introduce node: The newly introduced element $a$ has no edges to $S_u$ in the Gaifman graph; 2-table weights can be computed in batch using $r_{\tau_a, C_i}$.
      - Forget node: Enumerate the 1-type of the departing element and its 2-table with each element remaining in the bag.
      - Merge node: $S_{v_1}$ and $S_{v_2}$ are not connected; their 2-table weights can be computed in batch by 1-type configuration.
    - **Design Motivation**: The nice tree decomposition structure ensures that each step involves only a single element change, keeping computation tractable.

3. **Extension to $\text{FO}^2$, $\text{C}^2$, and Asymmetric Weights**:
    - **Function**: Generalize the core algorithm from $\text{UFO}^2$ to richer logical fragments.
    - **Mechanism**: Existing modular transformations reduce $\text{FO}^2$ and $\text{C}^2$ (with counting quantifiers and cardinality constraints) to $\text{UFO}^2$. For asymmetric weights, the relevant element pairs are added to the Gaifman graph.
    - **Design Motivation**: Modular reductions preserve domain size, so polynomial-time complexity is retained.

### Loss & Training

This is a theoretical and algorithmic work with no training process. The time complexity is $O(kn \cdot p^{k+3} \cdot q^{k+1} \cdot k \cdot n^{2p})$, where $n$ is the domain size, $k$ is the treewidth, and $p$, $q$ are the numbers of 1-types and 2-tables (constants for a fixed sentence). The algorithm is therefore polynomial in $n$.

## Key Experimental Results

### Main Results

| Problem | Domain Size | TD-WFOMC | d4 | Forclift | Crane2 |
|---------|-------------|----------|----|----------|--------|
| Friends & Smokers | 60 | ~0.01s | ~100s | ~100s | ~10s |
| Friends & Smokers | 120 | ~0.1s | timeout | timeout | timeout |
| WS (ring + cliques) | 60 | ~1s | timeout | — | — |

### Ablation Study

| Configuration | Key Observation | Remarks |
|---------------|-----------------|---------|
| Fixed domain=60, varying clique size | Runtime grows with clique size | Clique size affects treewidth |
| Simplified vs. full WS model | Simplified model is faster | Simplified model has no cardinality constraints |
| Ring vs. clique base graph | Both efficient | Efficiency holds as long as treewidth is bounded |

### Key Findings

- TD-WFOMC remains efficient as domain size scales, while other solvers time out at larger domains.
- The algorithm is sensitive to clique size (which affects treewidth) but scales extremely well with respect to domain size.
- An open problem is resolved: counting stable seating arrangements with a fixed number of classes on bounded-treewidth bounded-degree graphs (including $2 \times n$ grid tables) is polynomial-time solvable.

## Highlights & Insights

- The #P-hardness barrier imposed by binary evidence is overcome — bounded treewidth constitutes a natural and practically motivated restriction.
- The result is incomparable to Courcelle's theorem: the latter requires all non-unary predicates to be fully interpreted, whereas this work allows binary predicates to remain uninterpreted; however, the present work is restricted to the two-variable fragment.
- The application to stable seating arrangements demonstrates the generality of the WFOMC framework for combinatorial counting.
- The algorithmic framework is clearly structured, with solid theoretical contributions.

## Limitations & Future Work

- Applicability is limited to binary evidence whose Gaifman graph has bounded treewidth; dense graphs with large treewidth are not covered.
- Although $p$ and $q$ are constants for a fixed sentence, the number of 1-types and 2-tables may be large in practice.
- The experiments cover only two problem instances and do not evaluate broader real-world application benchmarks.
- The polynomial degree $2p$ in $n$ may be high, and practical scalability requires further empirical validation.
- Extensions beyond two-variable fragments (e.g., $\text{FO}^3$) are known to be non-liftable, which constitutes a fundamental theoretical barrier.

## Related Work & Insights

- **vs. BMF method (WFOMC-binary-evidence-bmf)**: The BMF approach converts binary evidence into unary evidence but requires the Boolean matrix to have low rank, a condition frequently violated in practice. The treewidth restriction proposed here is more natural.
- **vs. Courcelle's Theorem**: Courcelle's theorem handles monadic second-order logic but requires the graph to be fully interpreted; this work handles $\text{FO}^2$ while permitting uninterpreted predicates.
- **vs. d4 / Forclift**: The propositional solver d4 suffers exponential blowup as the domain grows; Forclift supports lifted inference but cannot efficiently handle binary evidence.
- **vs. Crane2**: Crane2 achieves better performance than classical solvers but does not support the full $\text{FO}^2$ language.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Fills a theoretical gap in WFOMC with bounded-treewidth binary evidence and resolves an open problem.
- Experimental Thoroughness: ⭐⭐⭐ Limited experimental scenarios; primarily validates scalability; lacks real-world application benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous, though the entry barrier for readers outside the field is high.
- Value: ⭐⭐⭐⭐ Significant theoretical contribution with important implications for statistical relational learning and combinatorial counting.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[AAAI 2026\] Higher-Order Responsibility](higher-order_responsibility.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)
- [\[AAAI 2026\] Measuring Model Performance in the Presence of an Intervention](measuring_model_performance_in_the_presence_of_an_intervention.md)

<!-- RELATED:END -->

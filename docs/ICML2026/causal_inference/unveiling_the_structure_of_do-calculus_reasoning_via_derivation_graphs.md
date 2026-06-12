---
title: >-
  [Paper Note] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs
description: >-
  [ICML 2026][Causal Inference][do-calculus] By introducing derivation graphs to explicitly represent all equivalent transformations of do-calculus rules…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "do-calculus"
  - "equivalence"
  - "derivation graphs"
  - "identifiability"
date: 2026-05-08
content_hash: 630b572a986330a1
---

# Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs

**Conference**: ICML 2026  
**arXiv**: [2606.03719](https://arxiv.org/abs/2606.03719)  
**Code**: https://gricad-gitlab.univ-grenoble-alpes.fr/yvernesc/do-calculus-derivation-graphs  
**Area**: Causal Inference  
**Keywords**: Causal Inference, do-calculus, equivalence, derivation graphs, identifiability

## TL;DR
By introducing derivation graphs to explicitly represent all equivalent transformations of do-calculus rules, this work reveals the structure of the causal expression space and proves that any equivalent expression can be reached in at most 4 rule applications.

## Background & Motivation

**Background**: Pearl's do-calculus enables recursive transformations of observational and interventional probabilities through three graphical rewriting rules (R1, R2, R3), and is widely applied to identifiability problems in causal inference.

**Limitations of Prior Work**: Although do-calculus is theoretically complete, the path space from a starting expression to another equivalent expression is extremely complex. The same causal effect may have an exponential number of equivalent expressions with significantly different statistical properties, yet theoretical guidance on how to systematically explore and select the optimal expression is lacking.

**Key Challenge**: While do-calculus is complete, the compositional relationships between rules, the full characterization of equivalent expressions, and the systematic derivation of multiple valid estimators have not been sufficiently characterized.

**Goal**: (1) Explicitly characterize all equivalent transformations in do-calculus; (2) reveal the commutativity between rules; (3) provide principled methods for experimental design and statistical efficiency optimization.

**Key Insight**: Treat the step-by-step application of do-calculus as node transitions in graph theory. Constructing derivation graphs makes all equivalent expressions and their relationships transparent.

**Core Idea**: Structure the sequences of do-calculus rule applications using directed graphs to reveal rule commutativity and the exponential growth patterns of equivalent expressions through topological properties.

## Method

### Overall Architecture
A derivation graph is an undirected graph $D[G] = (V_D, E_D)$, where vertices represent all causal expressions of the form $P(y | \text{do}(x), w)$, and edges denote a single valid application of a do-calculus rule. The workflow includes: (1) defining derivation graph vertices and edges; (2) analyzing rule commutativity; (3) proving the minimum number of rule applications; (4) deriving multiple valid estimators from equivalent expressions.

### Key Designs

1. **Graph-theoretic Representation of Derivation Graphs**:

    - **Function**: Systematize the causal expression space and do-calculus rule applications into graph-theoretic objects to enable explicit enumeration and querying of all equivalent transformations.
    - **Mechanism**: Vertices are defined as all expressions of the form $P(y | \text{do}(x), w)$, where $(Y, X, W)$ are disjoint subsets of $V_G$ and $Y \neq \emptyset$. Edges are defined as single-step valid do-calculus rules between two expressions. Connected components in the graph correspond to equivalence classes in the causal graph.
    - **Design Motivation**: The three do-calculus rules are essentially local transformations; using a graph makes the global equivalence structure explicit, avoiding manual step-by-step derivations.

2. **Rule Commutativity and Minimal Operational Calculus**:

    - **Function**: Prove that any two equivalent expressions can be reached via at most 2 applications of R2 and 2 applications of R3 (total $\le 4$ steps), without requiring arbitrarily long sequences.
    - **Mechanism**: Analyze commutativity relationships between R1 (observation insertion/deletion), R2 (action/observation exchange), and R3 (action insertion/deletion) through graph theory. Key discovery: R1 commutes with other rules, while specific order dependencies exist between R2 and R3. Graphical criteria are introduced to determine equivalence: for $Q_1$ and $Q_2$, check the corresponding d-separation conditions.
    - **Design Motivation**: The complexity of rule ordering hinders practical application. Proving a minimal 4-step bound significantly simplifies equivalence verification and path searching.

3. **Multi-estimator Derivation and Experimental Design Optimization**:

    - **Function**: Start from a single identification formula output by the ID algorithm and find multiple distinct valid estimators with different statistical properties by traversing the set of equivalent expressions.
    - **Mechanism**: Perform graph traversal (BFS/DFS) on the derivation graph to expand from the ID algorithm's output to other expressions in the equivalence class. Each expression corresponds to a different estimator whose variance and computational complexity vary based on the conditioning variables and summation dimensions. For example, a back-door formula $P(y|z)$ might involve summing over one variable, while a front-door formula could involve exponential variables, leading to vast differences in variance and complexity.
    - **Design Motivation**: In scenarios with limited data or infeasible partial interventions (e.g., biological or medical experiments), multiple valid estimators provide different cost-precision trade-offs.

## Key Experimental Results

### Main Results

| Causal Graph Scale | Variable Count | Derivation Graph Vertices | Equivalent Expressions | Max Rule Applications |
|-----------|--------|-------------|------------|-------------|
| Empty Graph (No edges) | 3 | 27 | 27 | 2 |
| Chain Graph A→B→C | 3 | 9 | 9 | ≤4 |
| Fork Graph A←B→C | 3 | 6 | 6 | ≤4 |
| Collider Graph A→B←C | 3 | 18 | 18 | ≤4 |

### Number of Equivalent Expressions

| Expression Type | Initial Expression | Equivalence Count | Max Rules Required | Count (Excl. R1) |
|-----------|-----------|-------------|---------------|--------------|
| Interventional Query | $P(a\|\text{do}(b,c))$ | 12 | 4 | 4 |
| Mixed Expression | $P(b\|\text{do}(a),c)$ | 8 | 3 | 3 |
| Observational Query | $P(a\|b,c)$ | 3 | 2 | 2 |

## Highlights & Insights
- **Commutativity Discovery**: Proved that $\text{R1} \circ \text{R2} = \text{R2} \circ \text{R1}$ and that $\text{R2} \circ \text{R3}$ is commutative under specific graphical conditions, providing a new algebraic structure for do-calculus.
- **Minimal 4-step Bound**: Rule combinations originally thought to require arbitrary steps are strictly bounded within 4 steps, reducing the computational complexity of equivalence testing.
- **New Perspective on Experimental Design**: Different equivalent expressions correspond to estimators with different variances, offering principled selection methods for cost-constrained scenarios like biological experiments or medical interventions.

## Limitations & Future Work
- Computational complexity: The number of vertices in a derivation graph grows exponentially with the number of variables, making full enumeration difficult for large-scale causal graphs.
- Incomplete variance analysis: The paper demonstrates differences in estimator forms but does not provide closed-form analytical solutions for variance.
- Expansion of identifiability: Current methods are restricted to identified causal effects; characterization of derivation graphs for partial identification is missing.

## Related Work & Insights
- **vs ID Algorithm**: The ID algorithm returns a single identification formula; this work reveals all expressions within an equivalence class via derivation graphs, offering a structured alternative.
- **vs Adjustment Formula Theory**: Back-door adjustment and front-door formulas are special cases of the equivalence classes explored here; derivation graphs unify these separate methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic characterization of commutativity between do-calculus rules and the complete graph-theoretic structure of equivalent expressions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Provides graph-theoretic characterization and Jupyter notebook replication; examples are clear but sample size is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic is clear, definitions are rigorous, and notation is consistent.
- Value: ⭐⭐⭐⭐⭐ Deepens the theory of causal inference and provides practical guidance for experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Skill Path: Unveiling Language Skills from Circuit Graphs](../../AAAI2026/causal_inference/skill_path_unveiling_language_skills_from_circuit_graphs.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)

</div>

<!-- RELATED:END -->

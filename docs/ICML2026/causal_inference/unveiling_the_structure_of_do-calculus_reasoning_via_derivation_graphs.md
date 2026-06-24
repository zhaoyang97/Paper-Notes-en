---
title: >-
  [Paper Note] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs
description: >-
  [ICML 2026][Causal Inference][Do-calculus] Explicitly representing all equivalent transformations of do-calculus rules through derivation graphs—revealing the structure of the causal expression space and proving that any equivalent expression is reachable within at most 4 rule applications.
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Do-calculus"
  - "Equivalence"
  - "Derivation Graphs"
  - "Identifiability"
date: 2026-05-08
content_hash: a542744d206bfdc2
---

# Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs

**Conference**: ICML 2026  
**arXiv**: [2606.03719](https://arxiv.org/abs/2606.03719)  
**Code**: https://gricad-gitlab.univ-grenoble-alpes.fr/yvernesc/do-calculus-derivation-graphs  
**Area**: Causal Inference  
**Keywords**: Causal Inference, Do-calculus, Equivalence, Derivation Graphs, Identifiability

## TL;DR
Explicitly representing all equivalent transformations of do-calculus rules through derivation graphs—revealing the structure of the causal expression space and proving that any equivalent expression is reachable within at most 4 rule applications.

## Background & Motivation

**Background**: Pearl's do-calculus enables the recursive transformation of observational and interventional probabilities through three graphical rewriting rules (R1, R2, R3), and is widely applied to identifiability problems in causal inference.

**Limitations of Prior Work**: Although do-calculus is theoretically complete, the path space from a starting expression to another equivalent expression is extremely complex. A single causal effect may have exponentially many equivalent expressions with significantly different statistical properties, yet theoretical guidance for systematically exploring and selecting the optimal expression is lacking.

**Key Challenge**: While do-calculus is complete, the compositional relationships between rules, the full characterization of equivalent expressions, and the systematic derivation of multiple valid estimators have not been fully characterized.

**Goal**: (1) Explicitly characterize all equivalent transformations of do-calculus; (2) reveal the commutativity between rules; (3) provide principled methods for experimental design and statistical efficiency optimization.

**Key Insight**: Treat the step-by-step application of do-calculus as node transitions in graph theory, constructing derivation graphs to make all equivalent expressions and their relationships clear at a glance.

**Core Idea**: Structurally organize the sequences of do-calculus rule applications using a directed graph structure, revealing rule commutativity and the exponential growth of equivalent expressions through the topological properties of the graph.

## Method

### Overall Architecture
The three rewriting rules of do-calculus (R1/R2/R3) were originally dispersed local transformations; manually deriving an equivalent expression from a causal expression provides neither a global view nor termination guarantees. This paper moves the entire derivation process into a single graph: each causal expression is treated as a vertex, and each valid rule application is an edge. Consequently, finding equivalent expressions, the minimum steps to reach them, and available estimators becomes an enumerable problem on the graph.

### Key Designs

**1. Derivation Graph Representation: Turning Rule Applications into Edges**

The pain point of do-calculus is that the rules are local and the expression space is implicit, making it impossible to see the global equivalence structure at once. This paper constructs an undirected derivation graph $D[G] = (V_D, E_D)$ to make it explicit: vertices are all expressions of the form $P(y \mid \text{do}(x), w)$, where $(Y, X, W)$ are pairwise disjoint subsets of the causal graph vertex set $V_G$ and $Y \neq \emptyset$; an edge is connected between two expressions if a single valid do-calculus rule application exists between them. Thus, expressions equivalent to each other in the causal graph fall into the same connected component of the derivation graph. Equivalence relations that previously required manual verification are transformed into graph queries regarding "membership in the same connected component," allowing all equivalent transformations to be explicitly enumerated.

**2. Rule Commutativity and 4-Step Reachability: Constraining Path Length**

If rules could be combined arbitrarily, the cost of equivalence determination and path searching would be uncontrollable. This paper provides a hard upper bound by analyzing the commutativity of the three rules: R1 (insertion/deletion of observations) commutes with other rules, while specific sequential dependencies exist between R2 (exchange of action and observation) and R3 (insertion/deletion of action). Based on these commutativity properties, the authors prove that any two equivalent expressions can always reach each other through at most 2 applications of R2 and 2 of R3 (totaling $\le 4$ steps), rendering longer sequences unnecessary. Correspondingly, a graphical criterion is introduced to directly determine equivalence—by checking d-separation conditions for the queries $Q_1$ and $Q_2$—eliminating the need for actual path searching. This compresses equivalence verification from an open-ended search problem into one decidable within constant steps.

**3. Multi-Estimator Derivation: Expanding One Identification Formula into a Family**

The ID algorithm returns only a single identification formula, but different equivalent forms of the same causal effect vary greatly in statistical properties. Starting from the expression output by the ID algorithm, this paper performs BFS/DFS traversal on the derivation graph to expand all other expressions in its equivalence class. Each equivalent expression corresponds to a valid but distinct estimator. The variance and computational complexity of these estimators depend on the specific conditional variables and summation dimensions: the backdoor formula $P(y \mid z)$ requires summation over only one variable, while the front-door formula may require summation over an exponential number of variables, leading to vast differences in variance and overhead. Thus, in scenarios where data is limited or certain interventions are impractical (e.g., biological or medical experiments), this family of estimators provides different cost-precision trade-offs that can be selected as needed.

## Key Experimental Results

### Main Results

| Causal Graph Scale | Variable Count | Derivation Graph Vertices | Equivalent Expressions | Max Rule Applications |
|-----------|--------|-------------|------------|-------------|
| Empty Graph (No Edges) | 3 | 27 | 27 | 2 |
| Chain A→B→C | 3 | 9 | 9 | ≤4 |
| Fork A←B→C | 3 | 6 | 6 | ≤4 |
| Collider A→B←C | 3 | 18 | 18 | ≤4 |

### Equivalent Expression Counts

| Expression Type | Starting Expression | Equivalent Count | Max Rule Apps Required | Count (Excluding R1) |
|-----------|-----------|-------------|---------------|--------------|
| Interventional Query | $P(a\|\text{do}(b,c))$ | 12 | 4 | 4 |
| Mixed Expression | $P(b\|\text{do}(a),c)$ | 8 | 3 | 3 |
| Observational Query | $P(a\|b,c)$ | 3 | 2 | 2 |

## Highlights & Insights
- **Commutativity Discovery**: Proved that $\text{R1} \circ \text{R2} = \text{R2} \circ \text{R1}$ and that $\text{R2} \circ \text{R3}$ is exchangeable under specific graphical conditions, providing a new algebraic structure for do-calculus.
- **Minimal 4-Step Bound**: Rule combinations originally thought to require an arbitrary number of steps are strictly limited to 4, simplifying the computational complexity of equivalence determination.
- **New Perspective on Experimental Design**: Different equivalent expressions correspond to estimators with different variances; provides a principled selection method for cost-constrained scenarios like biological or medical interventions.

## Limitations & Future Work
- Computational Complexity: The number of vertices in the derivation graph grows exponentially with the number of variables, making full enumeration difficult for large-scale causal graphs.
- Incomplete Variance Analysis: The paper demonstrates differences in estimator forms but does not provide a closed-form analytical solution for variance.
- Expansion of Identifiability: Current methods are limited to identified causal effects, lacking characterization of derivation graphs for partial identification.

## Related Work & Insights
- **vs. ID Algorithm**: The ID algorithm returns a single identification formula; this paper reveals all expressions within an equivalence class via derivation graphs, providing a structured alternative.
- **vs. Adjustment Formula Theory**: Backdoor adjustment theory and the front-door formula are special cases within the equivalence classes of this paper; derivation graphs unify these methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic characterization of commutativity between do-calculus rules and the complete graph-theoretic structure of equivalent expressions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Provides graph-theoretic characterization and Jupyter notebook reproduction; experimental examples are clear but sample size is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, rigorous definitions, and consistent notation.
- Value: ⭐⭐⭐⭐⭐ Deepens causal inference theory and provides practical guidance for experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICLR 2026\] CARL: Preserving Causal Structure in Representation Learning](../../ICLR2026/causal_inference/carl_preserving_causal_structure_in_representation_learning.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[ICLR 2026\] On the Identifiability of Causal Graphs with the Invariance Principle](../../ICLR2026/causal_inference/on_the_identifiability_of_causal_graphs_with_the_invariance_principle.md)
- [\[ICLR 2026\] Causal Structure Learning in Hawkes Processes with Complex Latent Confounder Networks](../../ICLR2026/causal_inference/causal_structure_learning_in_hawkes_processes_with_complex_latent_confounder_net.md)

</div>

<!-- RELATED:END -->

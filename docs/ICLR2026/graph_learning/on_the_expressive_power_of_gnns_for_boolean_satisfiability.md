---
title: >-
  [Paper Note] On the Expressive Power of GNNs for Boolean Satisfiability
description: >-
  [ICLR 2026][Graph Learning][GNN expressiveness] This paper rigorously proves, from the perspective of the Weisfeiler-Leman (WL) test, that the complete WL hierarchy cannot distinguish satisfiable from unsatisfiable 3-SAT…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "GNN expressiveness"
  - "Boolean satisfiability"
  - "Weisfeiler-Leman test"
  - "SAT solving"
  - "theoretical analysis"
date: 2026-05-08
content_hash: e2df6e6281597651
---

# On the Expressive Power of GNNs for Boolean Satisfiability

**Conference**: ICLR 2026
**arXiv**: [2602.08745](https://arxiv.org/abs/2602.08745)  
**Code**: [GitHub](https://github.com/sakupeltonen/sat-expressivity)  
**Area**: Graph Learning
**Keywords**: GNN expressiveness, Boolean satisfiability, Weisfeiler-Leman test, SAT solving, theoretical analysis

## TL;DR

This paper rigorously proves, from the perspective of the Weisfeiler-Leman (WL) test, that the complete WL hierarchy cannot distinguish satisfiable from unsatisfiable 3-SAT instances, revealing the theoretical expressiveness limits of GNNs for SAT solving. It also identifies positive instance families—such as planar SAT and random SAT—where GNNs can successfully distinguish satisfiability.

## Background & Motivation

Boolean satisfiability (SAT) is a classic NP-complete problem. In recent years, GNNs have become the dominant architecture for learning-based SAT solvers (e.g., NeuroSAT, QuerySAT), since CNF formulas are naturally represented as literal-clause bipartite graphs (LCG). However, GNN expressiveness is bounded by the WL test—graphs that are WL-equivalent must receive identical outputs.

**Core Problem**: Do GNNs possess sufficient expressive power to reason about satisfiability?

Prior work lacks a systematic analysis of GNN expressiveness on SAT tasks. Although one might intuit that "SAT is NP-hard, therefore GNNs must be insufficient," expressive power and computational complexity are orthogonal concepts—for instance, planar SAT is also NP-complete, yet 4-WL can identify all planar graphs.

## Method

### Overall Architecture

This is a theory-driven work. The core contributions are a series of theorems and constructions concerning the distinguishability of SAT formulas by the WL hierarchy, supplemented by experimental validation. The research pipeline proceeds as follows:

1. Define a graph representation for SAT formulas (LCN: Literal-Clause graph with Negation connections)
2. Construct families of WL-indistinguishable SAT instances
3. Analyze the distinguishability of special SAT families (regular, planar, random)
4. Validate WL expressiveness on random and competition instances

### Key Designs

#### Graph Representation: LCN (Literal-Clause Graph with Negation Connections)

The standard LCG is a bipartite graph between literals and clauses. This paper emphasizes that edges between each literal and its negation must be added (yielding LCN), because:
- Removing node labels (to preserve permutation invariance) causes information loss in LCG alone
- The LCN representation is sufficient: an unlabeled LCN uniquely determines a SAT formula up to isomorphism

#### Theorem 5.3 (Main Theorem): $k$-WL-Indistinguishable SAT Instances

**Construction**: Based on the classical Cai-Fürer-Immerman (1992) construction.

Given a graph $G$, a formula $f_G$ is constructed to encode "whether $G$ admits an even orientation of edges" (an orientation where every vertex has even out-degree). The answer depends on the parity of the number of edges. A "twisted" formula $\tilde{f}_G$ is then constructed by reversing one edge to bidirectional. Consequently, exactly one of $f_G$ and $\tilde{f}_G$ is satisfiable and the other is not, yet their LCNs are indistinguishable under $n$-WL.

$$\exists\, f, \tilde{f} \text{ (3-SAT)}: |f| = O(n) \text{ variables}, f \text{ satisfiable}, \tilde{f} \text{ unsatisfiable}, \text{LCN}(f) \equiv_{n\text{-WL}} \text{LCN}(\tilde{f})$$

This construction is closely related to Tseitin formulas (hard instances for resolution)—both involve global inconsistency that cannot be detected by local reasoning.

#### Implications for Sequential Solvers (Lemma 5.4)

Many GNN-based SAT solvers (e.g., QuerySAT) operate by setting variables one at a time. Lemma 5.4 shows that if two formulas are $k$-WL-indistinguishable, the residual formulas remain indistinguishable even after $\lfloor k/2 \rfloor - 1$ variables have been assigned.

**Corollary 5.5**: A WL-powerful GNN remains unable to distinguish satisfiable from unsatisfiable residual formulas even after setting $\Theta(n)$ variables—with significant implications for learning restart strategies.

#### 3-Regular SAT: An NP-Complete Problem Where WL Is Entirely Useless

Define $k$-regular SAT: each literal appears in exactly $k$ clauses, and each clause contains exactly $k$ literals.

- **Theorem 5.1**: 3-regular SAT is NP-complete
- **Observation 5.2**: The WL test cannot distinguish any two 3-regular SAT formulas with the same number of variables

#### Positive Results: Distinguishable SAT Families

**Planar SAT (Theorem 6.1)**: 4-WL can distinguish all planar SAT instances, since planar graphs are completely identified by 4-WL.

**Random SAT (Theorem 6.3)**: CNF formulas sampled from a uniformly random literal-incidence graph are identified by WL with probability at least $1 - n^{-1/7}$.

### Loss & Training

This is a purely theoretical work with no model training. The experimental section validates WL expressiveness by:
- Running $r$ rounds of WL on a satisfiable formula and imposing equality constraints on literals within the same equivalence class
- Checking whether the constrained formula $f_r$ remains satisfiable
- Defining $r_{\text{crit}}$ as the minimum number of rounds for which $f_r$ is satisfiable

## Key Experimental Results

### Main Results: Random Instances (G4SAT Benchmark)

| Family | Difficulty | $r_{\text{crit}}$ | $r_{\text{converge}}$ | Variables | Count |
|--------|------------|---------|-------------|-----------|-------|
| 3-SAT | easy | 2.97±0.18 | 3.68±0.47 | 26±9 | 1000 |
| 3-SAT | medium | 3.00±0.04 | 3.92±0.28 | 119±47 | 1000 |
| 3-SAT | hard++ | 3.08±0.28 | 4.00±0.00 | 5001±62 | 25 |
| k-clique | easy | 4.12±0.73 | 6.26±0.83 | 33±13 | 960 |

Random instances typically require only 3–4 WL rounds to distinguish literals; approximately 40% of formulas have all literals uniquely identified upon convergence.

### SAT Competition Instances

| Family | $r_{\text{crit}}$ | $r_{\text{converge}}$ | Variables | Count |
|--------|---------|-------------|-----------|-------|
| argumentation | 2.94±0.44 | 4.31±0.87 | 1266±625 | 16 |
| circuit-multiplier | **unsat** | 7.18±0.40 | 1075±50 | 11 |
| cryptography | 15.74±14.67 | 17.63±14.34 | 41510±29705 | 19 |
| heule-nol | **unsat** | 8.60±1.26 | 1419±0 | 10 |
| maxsat-optimum | 26.64±3.64 | 29.27±2.49 | 22157±5623 | 11 |

Of 448 competition instances, only 234 are solvable by WL; among 69 families, 38 contain instances that WL cannot distinguish.

### Ablation Study

**WL Convergence Pattern Analysis for 3-SAT**:
- Round 1: Each literal observes its own degree
- Round 2: No further refinement (all neighboring clauses have the same degree = 3)
- Round 3: Literals observe the degrees of other literals in shared clauses, enabling effective distinction
- Round 4: WL typically converges

### Key Findings

1. **Large gap between random and industrial instances**: Random instances pose virtually no challenge to WL, whereas industrial instances frequently exceed WL expressiveness
2. **Regular structure is fatal**: The heule-nol family encodes grid coloring problems; its regular structure renders WL entirely ineffective
3. **Connection to resolution complexity**: The indistinguishable instances constructed in this paper share structural similarities with Tseitin formulas (hard instances for resolution)—local reasoning cannot detect global inconsistency
4. The k-clique and k-vertex-cover families contain small fractions of WL-unsolvable instances (2.0% and 0.3%, respectively), arising from symmetries in the underlying graphs

## Highlights & Insights

- **Landmark theoretical result**: The first proof of the expressiveness limits of the complete WL hierarchy for SAT, clarifying the common misconception that "NP-hardness implies WL-indistinguishability" (counterexample: Theorem 6.1)
- **Tseitin–CFI connection**: The paper uncovers a deep structural link between the Cai-Fürer-Immerman construction and Tseitin formulas, establishing a new bridge between the WL test and proof complexity
- **Clear practical guidance**: GNNs are well-suited for random SAT, but solving industrial SAT requires architectures that go beyond WL (e.g., symmetry breaking, higher-order GNNs)

## Limitations & Future Work

1. Theoretical results are based on worst-case constructions; sparse or structured instances may exhibit better behavior in practice
2. Experiments measure only whether WL expressiveness is sufficient, without assessing the ability to generalize from data
3. The paper does not investigate the concrete effectiveness of architectures beyond WL (e.g., $k$-GNNs, random feature augmentation) on SAT tasks
4. Competition instance experiments are constrained by a 10 MB size limit; behavior on larger-scale instances remains unknown

## Related Work & Insights

- **NeuroSAT / QuerySAT / SATformer**: End-to-end learning-based SAT solvers, all subject to the theoretical limitations established in this paper
- **Cai-Fürer-Immerman (1992)**: This paper generalizes their classical indistinguishable graph construction to the SAT domain
- **GNN expressiveness literature** (GIN, $k$-GNN, higher-order WL): This paper provides concrete positive and negative results for these frameworks in the context of SAT
- The findings carry important implications for future learning-based SAT solver design: expressiveness-enhancement strategies tailored to the structural properties of industrial instances are needed

## Rating

- Novelty: ★★★★★ — First systematic theoretical analysis of GNN expressiveness for SAT
- Technical Depth: ★★★★★ — Rigorous proofs, elegant constructions, tight connections to algebra and complexity theory
- Experimental Thoroughness: ★★★★☆ — Comprehensive coverage of random and competition instances, validating theoretical predictions
- Writing Quality: ★★★★☆ — Clear structure with effective integration of theory and practice

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](../../ICML2026/graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](../../NeurIPS2025/graph_learning/graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [\[AAAI 2026\] Logical Characterizations of GNNs with Mean Aggregation](../../AAAI2026/graph_learning/logical_characterizations_of_gnns_with_mean_aggregation.md)
- [\[NeurIPS 2025\] The Underappreciated Power of Vision Models for Graph Structural Understanding](../../NeurIPS2025/graph_learning/the_underappreciated_power_of_vision_models_for_graph_structural_understanding.md)

</div>

<!-- RELATED:END -->

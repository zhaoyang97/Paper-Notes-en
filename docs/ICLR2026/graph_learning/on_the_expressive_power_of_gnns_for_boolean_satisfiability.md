---
title: >-
  [Paper Note] On the Expressive Power of GNNs for Boolean Satisfiability
description: >-
  [ICLR 2026][Graph Learning][GNN Expressivity] This work strictly proves from the perspective of the Weisfeiler-Leman (WL) test that the complete WL hierarchy cannot distinguish between satisfiable and unsatisfiable 3-SAT instances. It reveals the theoretical limits of GNN expressivity for SAT solving while identifying families of positive instances, such as planar SAT and random SAT, that GNNs can successfully distinguish.
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "GNN Expressivity"
  - "Boolean Satisfiability"
  - "Weisfeiler-Leman Test"
  - "SAT Solving"
  - "Theoretical Analysis"
date: 2026-05-08
content_hash: 9b03bb2256499347
---

# On the Expressive Power of GNNs for Boolean Satisfiability

**Conference**: ICLR 2026  
**arXiv**: [2602.08745](https://arxiv.org/abs/2602.08745)  
**Code**: [GitHub](https://github.com/sakupeltonen/sat-expressivity)  
**Area**: Graph Learning  
**Keywords**: GNN Expressivity, Boolean Satisfiability, Weisfeiler-Leman Test, SAT Solving, Theoretical Analysis

## TL;DR

This work strictly proves from the perspective of the Weisfeiler-Leman (WL) test that the complete WL hierarchy cannot distinguish between satisfiable and unsatisfiable 3-SAT instances. It reveals the theoretical limits of GNN expressivity for SAT solving while identifying families of positive instances, such as planar SAT and random SAT, that GNNs can successfully distinguish.

## Background & Motivation

The Boolean Satisfiability problem (SAT) is a classic NP-complete problem. Recently, GNNs have become mainstream architectures for learned SAT solvers (e.g., NeuroSAT, QuerySAT), as CNF formulas are naturally represented as Literal-Clause Bipartite Graphs (LCG). However, GNN expressivity is constrained by the WL test—WL-equivalent graph inputs necessarily yield the same output.

**Core Problem**: Do GNNs have sufficient expressivity to reason about satisfiability?

Existing work lacks a systematic analysis of GNNs on SAT tasks from an expressivity standpoint. While one might intuitively assume "SAT is NP-hard, so GNNs must be insufficient," expressivity and computational complexity are orthogonal concepts—for instance, planar SAT is also NP-complete, yet 4-WL can identify all planar graphs.

## Method

### Overall Architecture

This is a purely theoretical work that does not train any models. Instead, it addresses a specific question: if a SAT formula is fed into a GNN, does it possess sufficient expressivity to infer satisfiability? Since GNN discriminative power is bounded by the Weisfeiler-Leman (WL) test—where two WL-equivalent graph inputs must yield identical outputs—this question is equivalent to asking whether the WL test can distinguish between satisfiable and unsatisfiable formulas.

The argumentation follows a logical chain. First, a **lossless graph representation** for SAT formulas must be selected; otherwise, discussing WL's ability to distinguish formulas is meaningless. This work uses the LCN representation, which incorporates negation connections. Next is the negative core: using the classic Cai-Fürer-Immerman construction to create a pair of formulas—one satisfiable and one unsatisfiable—that cannot be distinguished by any order of WL. It further demonstrates that even sequential solvers that assign variables incrementally cannot bypass this ceiling. Finally, the negative conclusion is generalized to an entire family—3-regular SAT, which is invisible to WL yet remains NP-complete. Conversely, the work delineates regions where WL remains effective (planar SAT, random SAT) and quantifies "how many WL iterations are needed to distinguish an instance" using random G4SAT instances and industrial instances from SAT competitions.

### Key Designs

**1. LCN Graph Representation: Adding negation connections to prevent information loss**

Learned solvers commonly represent CNF formulas as LCGs—bipartite graphs between literals and clauses. This paper points out a flaw: once node labels are removed to maintain permutation invariance (a requirement for GNNs), an LCG alone cannot distinguish a literal from its negation. To fix this, edges between each variable and its negation are explicitly added (marked with a different color from literal-clause edges), resulting in the LCN (Literal-Clause Graph with Negation connections). This representation is sufficient—unlabeled LCNs uniquely determine the original SAT formula up to isomorphism. This makes subsequent discussions regarding WL's ability to distinguish formulas rigorous. By contrast, Variable-Clause Graphs (VCGs) can produce non-isomorphic representations for the same formula, and Literal Association Graphs are lossy.

**2. Main Negative Result: WL-indistinguishable "Sat-Unsat" pairs and the failure of sequential solving**

This is the core of the paper. To prove insufficient expressivity, one must construct two formulas: one satisfiable and one unsatisfiable, whose LCNs are equivalent under any $k$-WL order. The construction adapts the classic Cai-Fürer-Immerman (1992) method: given a graph $G$, a formula $f_G$ is written to encode whether $G$ has an orientation where every node has even out-degree. Such an orientation exists if and only if the number of edges $m$ is even—the answer depends on a **global property** (parity). By creating a "twisted" version $\tilde{f}_G$ where one edge is modified, $f_G$ and $\tilde{f}_G$ become exactly one satisfiable and one unsatisfiable. Crucially, this twist is locally invisible, making their LCNs indistinguishable under $n$-WL (Theorem 5.3):

$$\exists\, f, \tilde{f} \text{ (3-SAT)}: |f| = O(n) \text{ variables}, \ f \text{ satisfiable}, \ \tilde{f} \text{ unsatisfiable}, \ \text{LCN}(f) \equiv_{n\text{-WL}} \text{LCN}(\tilde{f})$$

This shares roots with Tseitin formulas—classic hard instances for resolution—where global inconsistency cannot be detected via local message passing. The paper also notes that this link between the Cai-Fürer-Immerman construction and Tseitin formulas was previously overlooked.

This negative result extends beyond one-shot prediction. Many GNN-SAT solvers (like QuerySAT) assign variables step-by-step. One might hope that after fixing some variables, the remaining formula becomes WL-distinguishable. Lemma 5.4 refutes this: if two formulas are $k$-WL indistinguishable, their residual formulas remain indistinguishable even after partial assignments. Corollary 5.5 pushes this to the limit: even after fixing $\Theta(n)$ variables, a WL-powerful GNN still cannot distinguish between the satisfiable and unsatisfiable residual formulas.

**3. 3-Regular SAT: A family where WL fails despite NP-completeness**

$k$-regular SAT is defined where each literal appears in exactly $k$ clauses and each clause contains exactly $k$ literals. This highly symmetric structure is devastating for WL. Theorem 5.1 proves that 3-regular SAT remains NP-complete, while Observation 5.2 states that the WL test cannot distinguish any two 3-regular SAT formulas with the same number of variables (local neighborhoods are identical, providing no discriminative signal). This family is more compelling than single constructed pairs as it shows WL failure occurs systematically in structured instances.

**4. Positive Results: SAT families distinguishable by WL**

The paper identifies regions where GNNs remain viable, countering the intuition that "NP-hard implies WL-indistinguishable." Planar SAT (Theorem 6.1): since planar graphs can be identified by 4-WL, all planar SAT instances are distinguishable by 4-WL—despite planar SAT being NP-complete. This proves that **expressivity and computational complexity are orthogonal**. Random SAT (Lemma 6.3): for CNF formulas generated from random literal graphs, the paper proves they are identifiable by WL with high probability. This explains why random instances are trivial for WL in experiments.

## Key Experimental Results

The paper measures whether WL expressivity is sufficient by running $r$ rounds of WL. Literals in the same WL equivalence class are constrained with equality, yielding a residual formula $f_r$. The minimum number of rounds to make $f_r$ satisfiable is $r_{\text{crit}}$, while $r_{\text{converge}}$ is the round where colors stop refining. Instances that cannot be distinguished this way are labeled **unsat** (insufficient WL expressivity).

### Main Results: Random Instances (G4SAT Benchmark)

| Family | Difficulty | $r_{\text{crit}}$ | $r_{\text{converge}}$ | Variables | Count |
|------|------|---------|-------------|--------|------|
| 3-SAT | easy | 2.97±0.18 | 3.68±0.47 | 26±9 | 1000 |
| 3-SAT | medium | 3.00±0.04 | 3.92±0.28 | 119±47 | 1000 |
| 3-SAT | hard++ | 3.08±0.28 | 4.00±0.00 | 5001±62 | 25 |
| k-clique | easy | 4.12±0.73 | 6.26±0.83 | 33±13 | 960 |

Random instances typically require only 3-4 WL rounds to distinguish literals.

### SAT Competition Instances

| Family | $r_{\text{crit}}$ | $r_{\text{converge}}$ | Variables | Count |
|------|------|---------|-------------|--------|------|
| argumentation | 2.94±0.44 | 4.31±0.87 | 1266±625 | 16 |
| circuit-multiplier | **unsat** | 7.18±0.40 | 1075±50 | 11 |
| cryptography | 15.74±14.67 | 17.63±14.34 | 41510±29705 | 19 |
| heule-nol | **unsat** | 8.60±1.26 | 1419±0 | 10 |
| maxsat-optimum | 26.64±3.64 | 29.27±2.49 | 22157±5623 | 11 |

Of 448 competition instances, only 234 were solvable by WL; 38 out of 69 families contained instances indistinguishable by WL.

### Ablation Study

**WL Convergence Patterns in 3-SAT**:
- Round 1: Each literal observes its own degree.
- Round 2: No further refinement (as neighbor clause degrees are identical = 3).
- Round 3: Literals observe degrees of other literals in shared clauses, enabling effective differentiation.
- Round 4: WL typically converges.

### Key Findings

1. **Massive Gap Between Random and Industrial Instances**: Random instances are trivial for WL, but industrial instances frequently exceed WL expressivity.
2. **Regular Structure is Fatal**: The `heule-nol` family encodes grid coloring; its regular structure causes WL to fail completely.
3. **Connection to Resolution Complexity**: The constructed indistinguishable instances resemble Tseitin formulas; local reasoning cannot detect global inconsistency.
4. Small percentages of k-clique (2.0%) and k-vercov (0.3%) instances are unsolvable by WL due to underlying graph symmetries.

## Highlights & Insights

- **Landmark Theoretical Result**: First proof of the complete WL hierarchy's limits on SAT, clarifying the misconception that "NP-hard = WL-indistinguishable" (Theorem 6.1 counterexample).
- **Tseitin-CFI Link**: Discovered a deep connection between the Cai-Fürer-Immerman construction and Tseitin formulas, bridging WL tests and proof complexity.
- **Clear Practical Guidance**: GNNs are suitable for random SAT, but industrial SAT requires architectures beyond WL (e.g., symmetry breaking, higher-order GNNs).

## Limitations & Future Work

1. Theoretical results are based on worst-case constructions; sparse/structured instances might behave better in practice.
2. Experiments only test if WL rank expressivity is sufficient, without considering the ability to learn from data (generalization).
3. Architectures beyond WL (e.g., k-GNNs, random feature augmentation) were not explored empirically.
4. Industrial experiments were limited by a 10MB instance size; behavior on larger instances remains unknown.

## Related Work & Insights

- **NeuroSAT / QuerySAT / SATformer**: End-to-end learned solvers, all subject to the theoretical limits established here.
- **Cai-Fürer-Immerman (1992)**: This work generalizes their classic indistinguishable graph construction to the SAT domain.
- **GNN Expressivity Literature** (GIN, k-GNN, high-order WL): This work provides concrete positive/negative results for these theories in SAT applications.
- Provides crucial insights for future learned solvers: architectures must be designed to enhance expressivity specifically for the structural characteristics of industrial instances.

## Rating

- Novelty: ★★★★★ — First systematic theoretical analysis of GNN-SAT expressivity.
- Technical Depth: ★★★★★ — Rigorous proofs with clever constructions linked to algebra and complexity theory.
- Experimental Thoroughness: ★★★★☆ — Comprehensive coverage of random and competition instances.
- Writing Quality: ★★★★☆ — Clear structure with a good balance of theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On The Expressive Power of GNN Derivatives](on_the_expressive_power_of_gnn_derivatives.md)
- [\[ICLR 2026\] Canonical Tree Cover Neural Networks for Expressive and Invariant Graph Learning](canonical_tree_cover_neural_networks_for_expressive_and_invariant_graph_learning.md)
- [\[ICLR 2026\] GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback](gnn-as-judge_unleashing_the_power_of_llms_for_graph_learning_with_gnn_feedback.md)
- [\[ICLR 2026\] Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs](self-consistency_improves_the_trustworthiness_of_self-interpretable_gnns.md)
- [\[ICLR 2026\] Multi-Scale Diffusion-Guided Graph Learning with Power-Smoothing Random Walk Contrast for Multi-View Clustering](multi-scale_diffusion-guided_graph_learning_with_power-smoothing_random_walk_con.md)

</div>

<!-- RELATED:END -->

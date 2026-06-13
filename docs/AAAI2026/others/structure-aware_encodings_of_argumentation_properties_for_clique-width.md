---
title: >-
  [Paper Note] Structure-Aware Encodings of Argumentation Properties for Clique-width
description: >-
  [AAAI 2026][Abstract Argumentation] This paper designs directed decomposition-guided (DDG) reductions from abstract argumentation problems to (Q)SAT that linearly preserve clique-width…
tags:
  - "AAAI 2026"
  - "Abstract Argumentation"
  - "Clique-width"
  - "Parameterized Complexity"
  - "SAT Encoding"
  - "QBF"
  - "$k$-expressions"
  - "Decomposition-Guided Reductions"
date: 2026-05-08
content_hash: b2fe7ad88776d237
---

# Structure-Aware Encodings of Argumentation Properties for Clique-width

**Conference**: AAAI 2026
**arXiv**: [2511.10767](https://arxiv.org/abs/2511.10767)  
**Authors**: Yasir Mahmood (Paderborn University), Markus Hecher (CNRS/University of Artois), Johanna Groven (Linköping University), Johannes K. Fichte (Linköping University)
**Code**: Not available  
**Area**: Other
**Keywords**: Abstract Argumentation, Clique-width, Parameterized Complexity, SAT Encoding, QBF, $k$-expressions, Decomposition-Guided Reductions

## TL;DR

This paper designs directed decomposition-guided (DDG) reductions from abstract argumentation problems to (Q)SAT that linearly preserve clique-width, establishing tractability upper bounds parameterized by clique-width for all standard argumentation semantics (stable, admissible, complete, preferred, semi-stable, stage) across extension existence, argument acceptance, and counting problems. Under the ETH, it further proves that the overhead of these reductions cannot be significantly improved.

## Background & Motivation

### State of the Field
Abstract Argumentation is a core framework in knowledge representation and reasoning, modeling conflicts among arguments via directed graphs (argumentation frameworks, AFs) and determining acceptable sets of arguments (extensions) through semantic conditions. The computational complexity of argumentation problems spans from P to the second level of the polynomial hierarchy, making the use of structural parameters for efficient solving an important research direction.

Treewidth is the most commonly used graph structural parameter, with numerous treewidth-based argumentation algorithms and decomposition-guided reductions already established. However, clique-width is a strictly more general parameter: bounded treewidth implies bounded clique-width, but not vice versa—clique-width can remain small on dense graphs where treewidth is necessarily large. This gives clique-width a unique advantage when handling attack graphs containing large cliques or complete bipartite structures.

### Limitations of Prior Work
- **Treewidth direction is mature**: Decomposition-guided reductions and dynamic programming algorithms for argumentation based on treewidth have been thoroughly studied.
- **Clique-width direction is nearly blank**: Although Dvořák et al. (2010) established a dynamic programming algorithm for preferred semantics parameterized by clique-width, virtually no theoretical results exist regarding structure-guided reductions to (Q)SAT.
- **Practical demand**: Modern SAT solvers run efficiently on instances of small clique-width (Fischer et al. 2008); encoding argumentation problems to (Q)SAT while preserving clique-width would allow direct use of these solvers.
- **Absence of lower bounds**: No systematic lower bound results for argumentation problems parameterized by clique-width existed prior to this work.

### Core Problem
To fill the theoretical gap in abstract argumentation under the clique-width parameter: design reductions to (Q)SAT that linearly preserve clique-width, enabling direct application of known SAT/QSAT clique-width tractability results to argumentation problem solving, while establishing matching ETH lower bounds to prove the structural optimality of the reductions.

## Method

### Core Concept: Directed Decomposition-Guided (DDG) Reductions

**$k$-expressions and clique-width**: The directed clique-width $\mathsf{dcw}(G)$ of a directed graph $G$ is the minimum number of colors $k$ required to construct $G$ using three operations: (i) disjoint union $\oplus$, (ii) relabeling $\rho_{c\to c'}$, and (iii) directed edge introduction $\eta_{c,c'}$ (adding directed edges from all vertices of color $c$ to all vertices of color $c'$). A $k$-expression can be represented by a parse tree $T$.

**Core Idea of DDG reductions**: Along the parse tree of a $k$-expression, auxiliary Boolean variables are introduced for each operation node $b$ and each color $c$, propagating key semantic information (e.g., "whether color $c$ contains an extension member," "whether non-extension arguments in color $c$ are defeated") through formulas.

### First-Order Semantics Encoding (Stable/Admissible/Complete)

**Stable extension encoding**: Two families of variables are propagated along the parse tree:
- **Extension variables $e_c^b$**: at operation $b$, whether color $c$ contains an extension member (disjunctive propagation).
- **Defeat variables $d_c^b$**: at operation $b$, whether every non-extension argument in color $c$ is attacked by the extension (conjunctive propagation).

At initial nodes for argument $a$: $e_c^b \leftrightarrow e_a$, $d_c^b \leftrightarrow e_c^b$; variables are propagated at union/relabeling nodes; at edge introduction nodes, conflict-freeness ($\neg e_c^b \vee \neg e_{c'}^b$) and defeat updates ($d_c^b \leftrightarrow d_c^{b'} \vee e_{c'}^b$) are handled; at the root, all colors are required to be defeated.

**Admissible extension encoding**: An attack variable $a_c^b$ is added alongside the extension variables, recording whether any non-extension argument attacks an extension argument without being counter-attacked. The edge introduction operation distinguishes cases by attack direction.

**Complete extension encoding**: In addition to admissibility, "out" variables $o_c^b$ and "defeated-from-$b$-upward" variables $d_c^{\geq b}$ (propagated backwards from root to leaves) are introduced to ensure no argument is improperly excluded from the extension.

### Second-Order Semantics Encoding (Preferred/Semi-stable/Stage)

**Key auxiliary lemma (Lemma 14)**: When the innermost quantifier of a QBF is $\forall$ and the matrix is of the mixed form $\varphi_{\text{CNF}} \wedge \psi_{\text{DNF}}$, it can be converted to a pure DNF matrix while linearly preserving directed incidence clique-width. This lemma substantially simplifies the encoding of second-order semantics.

**Preferred extensions**: A QBF of the form $\varphi_{\#\mathsf{adm}} \wedge \neg(\exists E^*, A^*, S.\, \varphi_{\mathsf{pref}})$ is constructed, where starred variables encode a candidate strictly larger extension. Lemma 14 is applied to convert the matrix to standard form, yielding a reduction to $\#2\text{-QBF}$.

**Semi-stable extensions**: The range of the extension is maximized by constructing $\varphi_{\#\mathsf{adm}} \wedge (\forall E^*, D^*, A^*, S.\, \neg\varphi_{\mathsf{semiSt}})$.

**Stage extensions**: Based on maximizing the range of conflict-free sets, the formula $\forall E^*, D^*, S.\, (\varphi_{\#\mathsf{conf}} \wedge \neg\varphi_{\mathsf{stage}})$ is constructed.

### ETH Lower Bound Proofs

By constructing clique-width-aware reductions from 3SAT to the credulous acceptance problem for admissible extensions, it is proved that a running time of $2^{o(w)} \cdot \text{poly}(n)$ is not achievable under ETH. These lower bounds naturally extend to stable/complete (first-order) and preferred/semi-stable/stage (second-order, yielding $2^{2^{o(w)}}$ lower bounds).

## Key Experimental Results

### Table 1: Summary of DDG Reduction Results Across Semantics

| Semantics $\sigma$ | CW-Aw. | ETH CW Lower Bound | Running Time Upper Bound | Running Time ETH Lower Bound |
|---|---|---|---|---|
| $\mathsf{stab}$ | $\mathcal{O}(k)$ | $\Omega(k)$ | $2^{\Theta(k)} \cdot \text{poly}(n)$ | $2^{\Theta(k)} \cdot \text{poly}(n)$ |
| $\mathsf{adm}$ | $\mathcal{O}(k)$ | $\Omega(k)$ | $2^{\Theta(k)} \cdot \text{poly}(n)$ | $2^{\Theta(k)} \cdot \text{poly}(n)$ |
| $\mathsf{comp}$ | $\mathcal{O}(k)$ | $\Omega(k)$ | $2^{\Theta(k)} \cdot \text{poly}(n)$ | $2^{\Theta(k)} \cdot \text{poly}(n)$ |
| $\mathsf{pref}$ | $\mathcal{O}(k)$ | $\Omega(k)$ | $2^{2^{\Theta(k)}} \cdot \text{poly}(n)$ | $2^{2^{\Theta(k)}} \cdot \text{poly}(n)$ |
| $\mathsf{semiSt}$ | $\mathcal{O}(k)$ | $\Omega(k)$ | $2^{2^{\Theta(k)}} \cdot \text{poly}(n)$ | $2^{2^{\Theta(k)}} \cdot \text{poly}(n)$ |
| $\mathsf{stage}$ | $\mathcal{O}(k)$ | $\Omega(k)$ | $2^{2^{\Theta(k)}} \cdot \text{poly}(n)$ | $2^{2^{\Theta(k)}} \cdot \text{poly}(n)$ |

Where $k = \mathsf{dcw}(\mathcal{G}_i^d(F))$ and $n = |A|$. All DDG reductions linearly preserve clique-width, and upper and lower bounds match at the exponential level in $k$, establishing the structural optimality of the reductions. Results uniformly cover extension existence $\mathsf{exist}_\sigma$, skeptical acceptance $\mathsf{s}_\sigma$, credulous acceptance $\mathsf{c}_\sigma$, and counting $\#_\sigma$.

### Table 2: Positioning Comparison with Treewidth Results

| Structural Parameter | Applicable Graph Types | First-Order Semantics Running Time | Second-Order Semantics Running Time | Encoding Preservation |
|---|---|---|---|---|
| Treewidth tw | Sparse graphs | $2^{\mathcal{O}(\text{tw})} \cdot \text{poly}(n)$ | $2^{2^{\mathcal{O}(\text{tw})}} \cdot \text{poly}(n)$ | Linearly preserves treewidth |
| Directed clique-width dcw (Ours) | Sparse + dense graphs | $2^{\mathcal{O}(\text{dcw})} \cdot \text{poly}(n)$ | $2^{2^{\mathcal{O}(\text{dcw})}} \cdot \text{poly}(n)$ | Linearly preserves clique-width |
| Undirected clique-width cw | — | SAT-intractable | — | Not applicable (hardness known) |

Clique-width strictly generalizes treewidth, enabling the results of this paper to handle attack graphs with large cliques or complete bipartite structures—cases where treewidth-based methods completely fail. Notably, SAT is known to be intractable under undirected clique-width, making directed incidence clique-width the correct parameter choice.

## Highlights & Insights

- **Comprehensiveness**: This is the first work to systematically establish encodings and complexity results parameterized by clique-width for all six standard argumentation semantics (stable/admissible/complete/preferred/semi-stable/stage), covering extension existence, argument acceptance, and counting.
- **Structural optimality**: Beyond upper bounds, ETH lower bounds demonstrate that the overhead of DDG reductions (linear growth from $k$ to $\mathcal{O}(k)$ in clique-width) cannot be significantly improved under reasonable assumptions, yielding tight matching bounds.
- **Technical innovation**: Lemma 14 introduces a method for converting mixed CNF/DNF matrices to pure DNF/CNF while preserving clique-width, serving as a key tool for handling second-order (maximization) semantics with independent technical value.
- **Practical significance**: Modern SAT/QBF solvers can be directly applied to argumentation instances of small clique-width; since clique-width applies to dense graphs, this addresses the limitations of treewidth-based methods on dense attack graphs.
- **Clear directed vs. undirected argument**: Example 9 elegantly demonstrates why directed clique-width must be used rather than undirected clique-width—two undirected-isomorphic but directed-distinct AFs can have entirely different numbers of stable extensions.

## Limitations & Future Work

- **Purely theoretical, no experimental validation**: The DDG reductions are not implemented or tested on actual SAT/QBF solvers, leaving their practical performance on real argumentation instances unevaluated.
- **Computing clique-width itself is difficult**: Although a $f(k)$-expression can theoretically be computed in polynomial time, the $f(k)$ factor in practice is extremely large, limiting real-world applicability.
- **Limited to abstract argumentation**: The results are not extended to logic-based argumentation, abductive reasoning, answer set programming (ASP), or other related KRR formalisms.
- **Encoding size**: DDG reductions produce formulas of size $\mathcal{O}(|T| \cdot k)$ (where $|T|$ is the parse tree size), which may be large for substantial argumentation frameworks.
- **Enumeration complexity not considered**: While the reductions preserve bijective solution correspondences and are naturally suitable for enumeration, this direction is not explored.
- **Orthogonal parameters such as modular treewidth and symmetric incidence clique-width are not addressed.**

## Related Work & Insights

- **Dvořák, Szeider, Woltran (2010)**: First established a dynamic programming algorithm and tractability result for preferred semantics parameterized by clique-width, but without addressing encodings to (Q)SAT or systematic treatment of other semantics. The present paper provides full coverage of all semantics with matching lower bounds.
- **Hecher (2020), Fichte, Hecher, Mahmood (2021)**: Established decomposition-guided reductions based on **treewidth** that linearly preserve treewidth. The present paper generalizes this paradigm to the strictly more general clique-width parameter.
- **Fischer, Makowsky, Ravve (2008)**: Proved that #SAT on CNF formulas with directed incidence clique-width $k$ is solvable in $2^{\mathcal{O}(k)} \cdot \text{poly}(n)$ time, while SAT is intractable under undirected clique-width. The DDG reductions in this paper directly exploit the positive result.
- **Capelli, Mengel (2019)**: Extended clique-width tractability of SAT to QSAT. The second-order semantics reductions in this paper use this result to obtain running times of $2^{2^{\mathcal{O}(k)}}$.
- **Niskanen, Järvisalo (2020)**: Standard SAT encodings for argumentation problems, without consideration of structural parameter preservation. The key distinction of the present encodings is their linear clique-width preservation.
- **Lampis, Mengel, Mitsou (2018), Fichte, Hecher, Pfandler (2020)**: QBF encodings for tight complexity bounds in argumentation problems under the treewidth framework. The present paper establishes analogous results under the clique-width framework.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic treatment of encoding argumentation problems parameterized by clique-width, pioneering the DDG reduction paradigm.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical contribution; no experiments or empirical evaluation of any kind.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous structure, complete theorem system, and illustrative $k$-expression examples aid understanding, though the dense notation is unfriendly to non-specialists.
- Value: ⭐⭐⭐⭐ — Fills an important gap in argumentation theory in the clique-width direction with solid theoretical contributions, though practical validation is absent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](how_to_marginalize_in_causal_structure_learning.md)
- [\[NeurIPS 2025\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling](../../NeurIPS2025/others/structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](../../CVPR2026/others/crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[AAAI 2026\] DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes](detonation_decoupled_torch_network-aware_training_on_interlinked_online_nodes.md)

</div>

<!-- RELATED:END -->

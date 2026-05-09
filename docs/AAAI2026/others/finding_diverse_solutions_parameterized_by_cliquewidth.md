---
title: >-
  [Paper Note] Finding Diverse Solutions Parameterized by Cliquewidth
description: >-
  [AAAI 2026][Parameterized complexity] This paper extends the parameterized framework for finding diverse solutions from treewidth to the strictly more powerful graph parameter cliquewidth, proving that any monotone dynamic programming algorithm parameterized by a cliquewidth decomposition can be converted into an algorithm for the diverse version with minimal overhead. A new family of Venn diversity measures is also proposed.
tags:
  - AAAI 2026
  - Parameterized complexity
  - diverse solutions
  - cliquewidth
  - dynamic programming
  - MSO1 logic
  - Venn diversity measure
date: 2026-05-08
content_hash: f35d055d7db37e03
---

# Finding Diverse Solutions Parameterized by Cliquewidth

**Conference**: AAAI 2026
**arXiv**: [2405.20931](https://arxiv.org/abs/2405.20931)
**Authors**: Karolina Drabik, Tomáš Masařík (University of Warsaw)
**Code**: Not released (purely theoretical work)
**Area**: Other
**Keywords**: Parameterized complexity, diverse solutions, cliquewidth, dynamic programming, MSO1 logic, Venn diversity measure

## TL;DR

This paper extends the parameterized framework for finding diverse solutions from treewidth to the strictly more powerful graph parameter cliquewidth, proving that any monotone dynamic programming algorithm parameterized by a cliquewidth decomposition can be converted into an algorithm for the diverse version with minimal overhead. A new family of Venn diversity measures is also proposed.

## Background & Motivation

### State of the Field
In combinatorial optimization, one typically seeks a single optimal solution. In practice, however, mathematical models are abstractions of real problems that may omit constraints difficult to formalize. Rather than finding a single optimal solution, it is often more useful to find a set of **maximally diverse solutions**, allowing decision-makers to select the one that best fits their actual needs. This paradigm has found broad application in constraint programming, mixed-integer optimization, evolutionary computation, social choice, and related fields.

### Limitations of Prior Work
- Baste et al. (2022) showed that under **treewidth** parameterization, a DP for a single problem can be converted into one for its diverse version with small overhead.
- However, treewidth is limited to sparse graph classes; for dense graphs (e.g., cliques, complete bipartite graphs), treewidth is large.
- **Cliquewidth** is a strictly more powerful graph parameter that additionally captures certain dense graph classes, yet a systematic framework for diverse problems under cliquewidth had been absent.
- Prior work focused solely on the DivSum and DivMin diversity measures, both of which exhibit degenerate behavior in certain settings.

### Root Cause
The paper addresses a theoretical gap in the study of diverse problems under cliquewidth parameterization, while simultaneously proposing a richer family of diversity measures to provide more flexible tools for practical applications.

## Method

### Problem Formulation
- **Vertex problem**: Given a graph $G$, find a vertex subset $S \subseteq V(G)$ satisfying some property (e.g., vertex cover, independent set, dominating set).
- **Diverse problem**: Given $r$ vertex problems $\mathcal{P}_1, \ldots, \mathcal{P}_r$, a diversity measure Div, and a threshold $d$, find $r$ solutions $S_1, \ldots, S_r$ such that $\mathrm{Div}(S_1, \ldots, S_r) \geq d$.
- **Hamming distance**: $\mathrm{HamDist}(S, S') = |(S \setminus S') \cup (S' \setminus S)|$
- **Two classical measures**: DivSum (sum of pairwise Hamming distances) and DivMin (minimum pairwise Hamming distance).

### Contribution 1: Venn Diversity Measure Family
The paper introduces a new family of diversity measures called **Venn $f$-diversity**. For each vertex $v$, its membership vector $m(S_1,\ldots,S_r,v) \in \{0,1\}^r$ across the $r$ solutions is mapped through a function $f:\{0,1\}^r \to \mathbb{N}$ that quantifies its contribution; the measure sums over all vertices:

$$\mathrm{Div}_f(S_1, \ldots, S_r) = \sum_{v \in V} f(m(S_1, \ldots, S_r, v))$$

DivSum is a special case with $f(m) = |\{i: m[i]=1\}| \cdot |\{j: m[j]=0\}|$. The paper also proposes the Div* measure to penalize the degenerate behavior in DivSum of repeatedly copying two disjoint sets:

$$\mathrm{Div}^*(S_1,\ldots,S_r) = \sum_{v \in V}(r^2 - |\{i \in [r]: v \in S_i\}|^2)$$

### Contribution 2: Monotone Dynamic Programming Framework
**Monotonicity**: A DP core $\mathfrak{C}$ is monotone if there exists a vertex-membership function $\rho$ such that each solution $S$ is fully determined by the values of its witness $\alpha$ at the leaf nodes via $\rho$. Intuitively, each partial solution is the disjoint union of partial solutions over its subtree.

**Main Theorem (Theorem 1.2)**: Given a monotone DP for $r$ vertex problems with worst-case running time $t_\mathcal{D}$ per decomposition node, the diverse version under any Venn diversity function can be solved in $\mathcal{O}(|V(\mathcal{D})| \cdot t_\mathcal{D}^r)$ time. Crucially, the running time **does not depend on the diversity threshold** $d$.

**Key Lemma 3.8**: Venn diversity functions satisfy additivity — for partial solutions over disjoint sets $S$ and $P$:

$$\mathrm{Div}(S_1 \cup P_1, \ldots, S_r \cup P_r) = \mathrm{Div}(S_1, \ldots, S_r) + \mathrm{Div}(P_1, \ldots, P_r) - |V(G)| \cdot f(0^r)$$

This enables the DP's merge operation to correctly combine diversity values.

### Contribution 3: Linear FPT Algorithm for MSO1-Expressible Problems
**Corollary 1.5**: The diverse version of any MSO1-expressible vertex problem can be solved in linear FPT time, parameterized by cliquewidth, number of solutions, and number of quantifiers in the formula.

The proof rests on a reconstruction of the Courcelle–Makowsky–Rotics theorem, using **reduced evaluation trees** and a **tree product** operation to design a new monotone DP. Key steps are:
1. Constructing a partial evaluation tree for each node of the cliquewidth decomposition.
2. Bounding the tree size via isomorphism-class reduction (the size is a tower function in the number of quantifiers and cliquewidth).
3. Proving that the reduced tree product is isomorphic to the unreduced version (Lemma 5.12, Combining Lemma).
4. Verifying that the constructed DP core satisfies monotonicity.

### Handling DivMin
For DivMin (Theorem 1.3), storing the Hamming distance vector across all $\binom{r}{2}$ pairs (rather than a single scalar) is necessary, yielding a running time of $\mathcal{O}(|V(\mathcal{D})| \cdot t_\mathcal{D}^r \cdot d^{r^2})$, since distance vectors are incomparable and all feasible distance combinations must be retained.

### Example: Diverse Vertex Cover
The paper provides a complete illustration using $k$-Vertex Cover:
1. A standard DP on cliquewidth is designed with running time $\mathcal{O}(|V(\mathcal{D})| \cdot (k+1)^{2\omega} \cdot \omega)$.
2. Monotonicity of the DP is established (Lemma 4.3).
3. The main theorem yields the diverse version in $\mathcal{O}(|V(\mathcal{D})| \cdot (k+1)^{2r\omega} \cdot r\omega)$.
4. A non-monotone DP example ($k$-Min Vertex Cover) demonstrates how such DPs can readily be made monotone.

## Summary of Theoretical Results

### Table 1: Comparison of Main Theorems

| Theorem | Graph Parameter | Diversity Measure | Running Time | Source |
|---------|----------------|-------------------|--------------|--------|
| Theorem 1.4 | treewidth | DivSum | $\mathcal{O}(\lvert V(\mathcal{T})\rvert \cdot t_\mathcal{T}^r)$ | Baste et al. 2022 (improved) |
| **Theorem 1.2** | **cliquewidth** | **Venn $f$-diversity** | $\mathcal{O}(\lvert V(\mathcal{D})\rvert \cdot t_\mathcal{D}^r)$ | **Ours** |
| **Theorem 1.3** | **cliquewidth** | **DivMin** | $\mathcal{O}(\lvert V(\mathcal{D})\rvert \cdot t_\mathcal{D}^r \cdot d^{r^2})$ | **Ours** |
| **Corollary 1.5** | **cliquewidth** | **Venn / DivMin** | **Linear FPT** | **Ours** |

A key finding is that the overhead for diversity under cliquewidth — going from one solution to $r$ diverse solutions — is identical to that under treewidth: an $r$-th power of the original running time. This shows that a stronger graph parameter does not incur additional cost for diversification.

### Table 2: Complexity Landscape for Diverse Problems by Logic and Graph Parameter

| Logic | Graph Parameter | Diversity Measure | Complexity | Source |
|-------|----------------|-------------------|------------|--------|
| MSO2 | treewidth | DivSum | Linear FPT | Baste et al. 2022 |
| FO | nowhere dense | DivSum/DivMin | FPT (non-linear) | Hanaka et al. 2021 |
| A&C DN | cliquewidth | DivSum/DivMin | Cubic FPT | Bergougnoux et al. 2023 |
| A&C DN | mimwidth | DivSum/DivMin | XP | Bergougnoux et al. 2023 |
| **MSO1** | **cliquewidth** | **Venn $f$-diversity** | **Linear FPT** | **Ours** |

The results are mutually incomparable with respect to logical expressiveness and graph class coverage. This paper fills the natural gap of MSO1 + cliquewidth. Note that MSO2 model checking on cliquewidth is likely not in XP (unless E = NE), making MSO1 the strongest logic one can reasonably expect to use in this setting.

## Highlights & Insights

- **Natural and important theoretical extension**: Generalizing from treewidth to cliquewidth is a natural direction, but requires overcoming new technical challenges (the node types of cliquewidth decompositions differ from those of tree decompositions). The extension to cliquewidth substantially broadens applicability by covering dense graph classes.
- **Venn diversity measure family**: Rather than a single measure, an entire family is proposed, allowing practitioners to choose a measure dynamically based on application needs and observed degeneracies. The Div* measure, for instance, penalizes the degenerate DivSum behavior of copying a small number of distinct solutions.
- **Near-optimal overhead**: The diverse version's running time is merely the $r$-th power of the original DP, consistent with the treewidth result, and is independent of the threshold $d$ for Venn measures.
- **Formalization of monotonicity**: The notion of a monotone DP is formally defined, with the claim that the vast majority of cliquewidth DPs in the literature are monotone or easily made so — supported by a non-monotone counterexample (Section 4.4).
- **Complete constructive proof for MSO1**: The cliquewidth version of Courcelle's theorem is reproved via reduced evaluation trees, with the resulting DP verified to be monotone — a technically non-trivial contribution.

## Limitations & Future Work

- **Vertex problems only**: The framework applies solely to vertex subset problems; extension to edge problems (e.g., Hamiltonian cycle) is left open. The paper asks whether the diverse Hamiltonian cycle problem admits an XP algorithm under cliquewidth parameterization.
- **DivMin overhead**: The $d^{r^2}$ factor for DivMin raises the question of whether this penalty can be eliminated.
- **Purely theoretical**: No experimental validation is provided; there is no empirical comparison of different Venn diversity measures on practical instances. The authors suggest systematic experimental evaluation as future work.
- **DivMin outside the Venn family**: Since DivMin is not a special case of Venn diversity, the main theorem does not apply directly, and a separate treatment yields worse running time.
- **No general conversion for non-monotone DPs**: Although most DPs are claimed to be monotone, no systematic method for converting non-monotone DPs is provided.
- **Tower-type parameter dependence**: The FPT parameter function in the MSO1 result is a multi-level exponential tower in cliquewidth and quantifier count, limiting practical applicability despite the FPT guarantee.

## Related Work & Insights

- **Baste et al. (AI 2022)**: Established the diverse-solution framework under treewidth, supporting MSO2 logic and DivSum. The present work generalizes this to cliquewidth, with the logic reduced from MSO2 to MSO1 (theoretically necessary), while adding support for the Venn measure family.
- **Hanaka et al. (2021)**: Gave FPT diverse algorithms for FO-definable problems on nowhere dense graph classes — broader graph coverage but weaker logic, significantly worse running time, and heavy reliance on the Grohe–Kreutzer–Siebertz model-checking machinery.
- **Bergougnoux, Dreier, Jaffke (2023)**: Obtained cubic FPT algorithms under cliquewidth and XP under mimwidth for A&C DN logic. A&C DN is only the existential fragment of MSO1 and is incomparable with the present work.
- **Problem-specific diversity studies**: Diverse hitting sets [BJM+19], diverse minimum $s$-$t$ cuts [dLS23], diversity in matroids [FGP+24], diverse pairs of matchings [FGJ+24], diverse SAT [MMR24], etc. — all target specific problems rather than providing a general framework.
- **Cliquewidth DP literature**: Feedback vertex set [BSTV13], triangle detection [CDP19], Hamiltonian cycle [BKK19], Steiner tree [BK24], etc. — the present framework is directly applicable to the diverse versions of all such monotone DPs.

## Rating

- Novelty: ⭐⭐⭐⭐ — The generalization from treewidth to cliquewidth is a natural direction; the introduction of the Venn measure family and the constructive MSO1 proof carry genuine technical novelty.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical; no experimental validation or empirical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-organized, rigorously defined, richly illustrated (complete four-section treatment of vertex cover; Venn diagram and path graph examples for degenerate behavior), with thorough discussion of open problems.
- Value: ⭐⭐⭐⭐ — Fills a natural theoretical gap in the cliquewidth + MSO1 setting; the Venn measure family offers a new perspective for diversity research.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)
- [\[AAAI 2026\] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)
- [\[NeurIPS 2025\] The Parameterized Complexity of Computing the VC-Dimension](../../NeurIPS2025/others/the_parameterized_complexity_of_computing_the_vc-dimension.md)
- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](../../NeurIPS2025/others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[ICLR 2026\] DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](../../ICLR2026/others/distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)

<!-- RELATED:END -->

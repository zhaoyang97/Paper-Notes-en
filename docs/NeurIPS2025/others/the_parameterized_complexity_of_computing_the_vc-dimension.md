---
title: >-
  [Paper Note] The Parameterized Complexity of Computing the VC-Dimension
description: >-
  [NeurIPS 2025][VC dimension] This paper systematically studies the parameterized complexity of computing the VC dimension, proving that the naive exhaustive algorithm is asymptotically optimal under ETH…
tags:
  - "NeurIPS 2025"
  - "VC dimension"
  - "parameterized complexity"
  - "treewidth"
  - "fixed-parameter tractability"
  - "ETH lower bounds"
date: 2026-05-08
content_hash: c9a801b07c2e67bf
---

# The Parameterized Complexity of Computing the VC-Dimension

**Conference**: NeurIPS 2025
**arXiv**: [2510.17451](https://arxiv.org/abs/2510.17451)  
**Code**: None (purely theoretical work)  
**Area**: Computational Complexity Theory, Machine Learning Theory
**Keywords**: VC dimension, parameterized complexity, treewidth, fixed-parameter tractability, ETH lower bounds, hypergraphs

## TL;DR

This paper systematically investigates the parameterized complexity of computing the VC dimension, establishing that the naive exhaustive algorithm is asymptotically optimal under ETH, presenting an FPT 1-additive approximation algorithm parameterized by maximum degree, an exact $2^{O(\text{tw} \cdot \log \text{tw})} \cdot |V|$ algorithm parameterized by treewidth, and a complete characterization of the tractability landscape across all standard structural parameters.

## Background & Motivation

**Background**: The VC dimension is a fundamental complexity measure for set systems (hypergraphs), playing a central role in machine learning ($\epsilon$-nets, sample compression schemes, PAC learning, machine teaching), combinatorial geometry, and combinatorics. The decision problem of computing the VC dimension was shown by Papadimitriou and Yannakakis to be LogNP-complete (lying between $\mathsf{P}$ and $\mathsf{NP}$), and the naive exhaustive algorithm requires $2^{O(|V|)}$ time.

**Limitations of Prior Work**: Despite the importance of the VC dimension, its computational complexity is poorly understood. It is known that parameterizing by solution size $k$ yields $\mathsf{W}[1]$-completeness (Downey–Evans–Fellows), and Manurangsi further rules out FPT approximation with factor $o(k)$ under Gap-ETH. Parameterization by degeneracy is also $\mathsf{W}[1]$-hard. However, systematic analysis of other structural parameters is almost entirely absent.

**Key Challenge**: The VC dimension is ubiquitous and critically important in machine learning, yet its computation is theoretically hard — unlikely to lie in $\mathsf{P}$ but also unlikely to be $\mathsf{NP}$-hard — occupying an "awkward" complexity tier. Can this barrier be overcome by exploiting structural properties of the input?

**Goal**: (1) Is the naive exhaustive $2^{O(|V|)}$ algorithm truly unavoidable? (2) Which structural parameters of hypergraphs or graphs render VC dimension computation fixed-parameter tractable? (3) Provide a complete complexity classification across all major structural parameters.

**Key Insight**: Adopting the lens of parameterized complexity, the paper systematically examines parameters including solution size, maximum degree, dimension, degeneracy, hypertreewidth, transversal number, treewidth, and vertex cover number, providing either algorithms or hardness results to draw a complete tractability landscape. In particular, Gen-VC-Dimension is introduced to unify the VC dimension of set systems and graph VC dimension.

**Core Idea**: Exploit the structural properties of shattered sets within tree decompositions — they are either contained within a single bag, or their size is bounded by $O(\log \text{tw})$ — and reduce VC dimension computation to pattern-graph-matching dynamic programming over tree decompositions.

## Method

### Overall Architecture

The paper contains three major technical contributions:

1. **Algorithmic lower bounds**: Via a reduction from 3-Coloring to Graph-VC-Dimension, establishing tightness of the naive algorithm under ETH.
2. **Structural parameter analysis**: Systematic examination of all major hypergraph structural parameters, yielding FPT tractability or hardness results.
3. **Treewidth algorithm**: Introduction of the Gen-VC-Dimension unified framework and a two-phase dynamic programming algorithm over tree decompositions.

### Key Designs

1. **ETH-tight lower bound (Theorem 9)**: Via a reduction from 3-Coloring to Graph-VC-Dimension.

    - **Function**: Proves that the naive exhaustive algorithm $2^{O(|V|)}$ is asymptotically optimal under ETH; no $2^{o(|V|)}$-time algorithm exists.
    - **Mechanism**: The vertices of graph $G'$ are partitioned into $k = \lceil \epsilon_1 |V(G')| \rceil$ parts; all 3-colorings of each part are enumerated to form the vertex set $X$ (an independent set) of a new graph $G$. Three auxiliary vertex sets $I_1, I_2, I_{\geq 3}$ encode coloring consistency constraints, so that $G$ has a shattered set of size $k$ if and only if $G'$ has a valid 3-coloring. The reduction requires superpolynomial time — which is unavoidable, since VC dimension is solvable in quasipolynomial time while 3-Coloring is NP-hard.
    - **Design Motivation**: Complements the LogNP-completeness result of Papadimitriou–Yannakakis by proving that even exponential brute-force search in $|V|$ is inherently unavoidable, providing strong motivation for the structural parameterization approach.

2. **FPT 1-additive approximation parameterized by maximum degree $\Delta$ (Theorem 12)**: Based on the structural relationship between shattered sets and witness sets.

    - **Function**: Yields an algorithm with running time $2^{O(\Delta \log \Delta)} \cdot |\mathcal{H}|^{O(1)}$ that, whenever a shattered set of size $k$ exists, is guaranteed to find one of size $k-1$.
    - **Mechanism**: Uses the key observation (Lemma 11) — if $S$ is a shattered set of size $k$, then for any $v \in S$, a subset of $\text{inc}(v)$ shatters $S \setminus \{v\}$. The algorithm enumerates all subsets $W \subseteq \text{inc}(v)$ of size $2^{k-1}$ for each vertex $v$, and applies Lemma 10 to determine whether $W$ witnesses a shattered set of size $k-1$ (by checking all permutations of $W$ for a good ordering).
    - **Design Motivation**: While solution size and degeneracy are both intractable, the maximum degree $\Delta$ naturally bounds both the size of shattered sets ($\leq \log\Delta + 1$) and the number of incident hyperedges per vertex. This is one of only two core hypergraph structural parameters that can be exploited (the other being dimension $D$).

3. **FPT exact algorithm parameterized by treewidth tw (Theorem 19)**: Two-phase dynamic programming.

    - **Function**: Solves Gen-VC-Dimension exactly in $2^{O(\text{tw} \cdot \log \text{tw})} \cdot |V|$ time.
    - **Mechanism**: **Phase 1** — Assuming the largest shattered set is contained within some bag; uses existing techniques to detect this in $2^{O(\text{tw})} \cdot |V|$ time. **Phase 2** — If the shattered set spans multiple bags, Lemma 15 guarantees $|S| \leq \log\text{tw} + 2$ (since the size of a shattered set crossing a separator is bounded by $O(\log|Z|)$), making $k$ small; the problem is then reduced to finding an embedding of a pattern graph $\mathcal{P}$ (containing $k$ shattered-set vertices and $2^k$ witness vertices) in the tree decomposition. DP states $\Gamma(t, f)$ record mappings from pattern graph vertices to $\text{bag} \cup \{\uparrow, \downarrow\}$.
    - **Design Motivation**: Treewidth is the most successful graph parameter in parameterized algorithms. While Courcelle's theorem applies directly, it yields a tower-of-exponentials dependency. The single-exponential $2^{O(\text{tw} \cdot \log \text{tw})}$ dependence achieved here is far superior, and contrasts sharply with the (tight) double-exponential dependence required for closely related problems.

### Loss & Training

Not applicable (purely theoretical work). Core technical tools include: parameterized reductions, dynamic programming over tree decompositions, pattern graph matching, and ETH-conditional lower bound proofs.

## Key Experimental Results

### Main Results

This is a theoretical work; the main results are algorithmic and complexity-theoretic theorems:

| Theorem | Parameter | Result | Type |
|---------|-----------|--------|------|
| Theorem 9(i) | vcn + k | No $2^{\epsilon(\text{vcn}+k)} |V|^{O(1)}$ algorithm | ETH lower bound |
| Theorem 9(ii) | $|V|$ | No $2^{\epsilon|V|} |\mathcal{H}|^{O(1)}$ algorithm | ETH lower bound |
| Theorem 12 | Max degree $\Delta$ | $2^{O(\Delta\log\Delta)} |\mathcal{H}|^{O(1)}$ | FPT 1-additive approx. |
| Natural observation | Dimension $D$ | $2^D |\mathcal{H}|^{O(1)}$ | FPT exact |
| Proposition 13 | Hypertreewidth / transversal number | LogNP-hard (even when transversal number = 1) | Hardness |
| Theorem 19 | Treewidth tw | $2^{O(\text{tw}\cdot\log\text{tw})} |V|$ | FPT exact |

### Ablation Study

Complete tractability classification across structural parameters ("parameterized landscape"):

| Structural Parameter | Tractability | Key Reason |
|----------------------|-------------|------------|
| Solution size $k$ | $\mathsf{W}[1]$-complete | No $o(k)$-approximation FPT even under Gap-ETH |
| Max degree $\Delta$ | FPT (1-additive approx.) | Shattered set size $\leq \log\Delta + 1$ |
| Dimension $D$ | FPT (exact) | Shattered set must be contained in some hyperedge |
| Degeneracy | $\mathsf{W}[1]$-hard | Prior result [Drange et al.] |
| Hypertreewidth | LogNP-hard | Even hypertree structure cannot be exploited |
| Transversal number | LogNP-hard | Even transversal number 1 cannot be exploited |
| Treewidth tw | FPT (exact) | Core algorithmic contribution of this paper |
| Vertex cover number vcn | $2^{o(\text{vcn})}$ lower bound | Stronger parameter than treewidth, yet lower bound holds |

### Key Findings

- **Naive algorithm is optimal**: The $2^{O(|V|)}$ exhaustive search cannot be improved under ETH — a clean tightness result.
- **Few exploitable parameters**: Among all major hypergraph structural parameters, only maximum degree $\Delta$ and dimension $D$ yield FPT algorithms (the former only approximately).
- **Single-exponential treewidth dependence**: $2^{O(\text{tw}\cdot\log\text{tw})}$ contrasts with the double-exponential dependence required for closely related problems, suggesting a unique algorithmic advantage for VC dimension under treewidth parameterization.
- **Unifying value of Gen-VC-Dimension**: The binary incidence graph representation unifies algorithmic study of set-system VC dimension and graph VC dimension.

## Highlights & Insights

- **Elegant argument of Lemmas 14/15**: If a shattered set $S$ crosses both sides of a separator $Z$, then $2^{|S|-2}$ witnesses must lie in $Z$, yielding $|S| \leq \log|Z| + 2$. This concise combinatorial argument is the key foundation enabling the treewidth algorithm.
- **Delicate balance in reduction design**: The reduction from 3-Coloring to VC dimension permits superpolynomial time (since the target problem is weaker than NP), a "non-standard" reduction type that is instructive within parameterized complexity.
- **Complete landscape characterization**: The paper not only provides positive algorithms but also systematically rules out other parameters, drawing a complete tractability boundary.
- **Pattern graph matching DP framework**: The combinatorial relationship between shattered sets and witnesses is encoded as a graph embedding problem; DP states are compact ($2^{O(\text{tw}\cdot\log\text{tw})}$ states per node), constituting a refined technical contribution in theoretical computer science.

## Limitations & Future Work

- **Upper–lower bound gap**: A gap remains between the treewidth algorithm $2^{O(\text{tw}\cdot\log\text{tw})}$ and the lower bound $2^{o(\text{vcn}+k)}$ (where vcn $\geq$ tw); closing or narrowing this gap is the main open problem.
- **Approximation vs. exactness**: Under maximum degree $\Delta$ parameterization, only a 1-additive approximation is obtained; whether an exact FPT algorithm exists remains open.
- **Circuit-defined set systems**: The setting where set systems are described implicitly by Boolean circuits is not considered; inputs may be more succinct there, though lower bounds still apply.
- **Experimental evaluation**: As a purely theoretical work, no empirical evaluation is provided; since treewidth tends to be large in practical graphs and hypergraphs, the practical utility of the algorithms remains to be assessed.

## Related Work & Insights

- **Papadimitriou–Yannakakis (1996)**: The foundational LogNP-completeness result; this paper establishes fine-grained tightness on top of it.
- **Manurangsi (2023)**: Approximation hardness under Gap-ETH (ruling out $o(\log|\mathcal{H}|)$ approximation); this paper provides a complementary structural-parameter perspective.
- **Drange–Greaves–Muzi–Reidl (2023)**: Pioneer work on $\mathsf{W}[1]$-hardness under degeneracy and the pattern graph idea; this paper extends and deepens that direction.
- **Implications for ML**: Computing the VC dimension on graph-structured data has direct applications (e.g., expressiveness analysis of graph neural networks); the treewidth algorithm provides a feasible computational pathway for structured data.

## Rating

- ⭐⭐⭐⭐⭐ **Theoretical Depth**: Rationale: A complete parameterized landscape characterization, combining sophisticated algorithm design with matching hardness proofs; technically first-rate.
- ⭐⭐⭐⭐ **Novelty**: Rationale: The unified perspective of Gen-VC-Dimension, the non-standard exponential-time reduction, and the single-exponential treewidth dependence are all original contributions.
- ⭐⭐⭐ **Practicality**: Rationale: Purely theoretical; treewidth may be large in practical settings and no experimental validation is provided, though the work lays the theoretical foundation for future practical development.
- ⭐⭐⭐⭐ **Clarity of Presentation**: Rationale: The paper is well-structured with a clear hierarchy of lemmas and theorems; the reductions and DP are presented with coherent logic, though the density of mathematical notation is high.

---
title: >-
  [Paper Notes] The Parameterized Complexity of Computing the VC-Dimension
description: >-
  [NeurIPS 2025][VC Dimension] This paper systematically studies the parameterized complexity of computing the VC dimension, proving that the naive exhaustive algorithm is asymptotically optimal under ETH, and proposing an FPT 1-additive approximation parameterized by maximum degree and an exact $2^{O(\text{tw}\cdot\log \text{tw})}\cdot|V|$-time algorithm parameterized by treewidth.
tags:
  - NeurIPS 2025
  - VC Dimension
  - Parameterized Complexity
  - Treewidth
  - Fixed-Parameter Tractability
  - ETH Lower Bounds
---

# The Parameterized Complexity of Computing the VC-Dimension

**Conference**: NeurIPS 2025
**arXiv**: [2510.17451](https://arxiv.org/abs/2510.17451)  
**Code**: None  
**Area**: Computational Complexity Theory, Machine Learning Theory
**Keywords**: VC dimension, parameterized complexity, treewidth, fixed-parameter tractability, ETH lower bounds

## TL;DR

This paper systematically studies the parameterized complexity of computing the VC dimension, proving that the naive exhaustive algorithm is asymptotically optimal under ETH, proposing an FPT 1-additive approximation parameterized by maximum degree, and an exact $2^{O(\text{tw}\cdot\log \text{tw})}\cdot|V|$-time algorithm parameterized by treewidth.

## Background & Motivation

- The VC dimension is a fundamental complexity measure for set systems (hypergraphs), playing a central role in machine learning ($\epsilon$-nets, sample compression schemes, PAC learning, machine teaching), and related fields.
- The decision problem of computing the VC dimension is LogNP-complete (lying between $\mathsf{P}$ and $\mathsf{NP}$); the naive algorithm runs in $2^{O(|V|)}$ time.
- Prior results: parameterization by solution size $k$ is $\mathsf{W}[1]$-complete; parameterization by degeneracy is $\mathsf{W}[1]$-hard.
- **Core Problem**: Do algorithms exploiting structural parameters of the input exist?

## Method

### Overall Architecture

Three main directions of contribution:
1. **Algorithmic lower bounds**: Proving tightness of the naive algorithm.
2. **FPT approximation parameterized by maximum degree**.
3. **FPT exact algorithm parameterized by treewidth**.

### Key Designs

**Contribution 1 — ETH-tight lower bound (Theorem 9)**

Via a reduction from 3-Coloring to Graph-VC-Dimension, proving:
- Graph-VC-Dimension admits no $2^{\epsilon(\text{vcn}+k)}\cdot|V|^{O(1)}$-time algorithm (where vcn is the vertex cover number).
- VC-Dimension admits no $2^{\epsilon|V|}\cdot|\mathcal{H}|^{O(1)}$-time algorithm.

Reduction core: Partition the vertices of $G'$ into $k$ parts, enumerate all 3-colorings of each part to form the vertex set $X$ (a bipartite graph) of a new graph $G$, and use consistency connections to define $Y$, so that $G$ has a shattered set of size $k$ if and only if $G'$ has a valid 3-coloring.

**Contribution 2 — FPT 1-additive approximation (Theorem 12)**

- Observation: If $S$ is a shattered set of size $k$, then for any $v \in S$, a subset of $\text{inc}(v)$ shatters $S \setminus \{v\}$.
- Algorithm: Enumerate all subsets $W \subseteq \text{inc}(v)$ of size $2^{k-1}$ for each vertex $v$, and check whether $W$ witnesses a shattered set of size $k-1$.
- Running time: $2^{O(\Delta \log \Delta)}\cdot|\mathcal{H}|^{O(1)}$.

**Contribution 3 — FPT algorithm parameterized by treewidth (Theorem 19)**

Introduces the unified problem Gen-VC-Dimension (encompassing both VC-Dimension and Graph-VC-Dimension).

The algorithm proceeds in two phases:
- **Phase 1**: Assuming the largest shattered set is contained in some bag of the tree decomposition; detectable in $2^{O(\text{tw})}\cdot|V|$ time using existing techniques.
- **Phase 2**: If $k \leq \log \text{tw} + 2$ (guaranteed by Lemma 15), find an embedding of a pattern graph $\mathcal{P}$ in the tree decomposition via **dynamic programming**; DP states are $\Gamma(t, f)$, where $f$ maps pattern graph vertices to $\text{bag}\cup\{\uparrow,\downarrow\}$.
- Total running time: $2^{O(\text{tw}\cdot\log \text{tw})}\cdot|V|$.

### Loss & Training

Not applicable (purely theoretical work).

## Key Experimental Results

### Main Results

| Result | Parameter | Time Complexity | Type |
|--------|-----------|----------------|------|
| Theorem 9(i) | vcn + k | $2^{\epsilon(\text{vcn}+k)}\cdot|V|^{O(1)}$ lower bound | ETH lower bound |
| Theorem 9(ii) | $|V|$ | $2^{\epsilon|V|}\cdot|\mathcal{H}|^{O(1)}$ lower bound | ETH lower bound |
| Theorem 12 | Max degree $\Delta$ | $2^{O(\Delta\log\Delta)}\cdot|\mathcal{H}|^{O(1)}$ | FPT 1-additive approx. |
| Natural observation | Dimension $D$ | $2^D\cdot|\mathcal{H}|^{O(1)}$ | FPT exact |
| Proposition 13 | Hypertreewidth / transversal number | LogNP-hard (transversal number = 1) | Hardness |
| Theorem 19 | Treewidth tw | $2^{O(\text{tw}\cdot\log \text{tw})}\cdot|V|$ | FPT exact |

### Parameterized Landscape Summary

| Parameter | Tractability |
|-----------|-------------|
| Solution size $k$ | $\mathsf{W}[1]$-complete + no $o(k)$-approx. FPT |
| Max degree $\Delta$ | FPT (1-additive approx.) |
| Dimension $D$ | FPT (exact) |
| Degeneracy | $\mathsf{W}[1]$-hard |
| Hypertreewidth | LogNP-hard |
| Transversal number | LogNP-hard |
| Treewidth tw | FPT (exact, $2^{O(\text{tw}\cdot\log \text{tw})}$) |
| Vertex cover number vcn | $2^{o(\text{vcn})}$ lower bound |

### Key Findings

- The treewidth parameterization achieves a **single-exponential** dependence $2^{O(\text{tw}\cdot\log \text{tw})}$, contrasting with the double-exponential dependence required for closely related problems.
- Maximum degree and dimension are the only two core hypergraph structural parameters that can be exploited.
- The introduction of Gen-VC-Dimension unifies the algorithmic study of graph VC dimension and set-system VC dimension.

## Highlights & Insights

- The paper provides a complete parameterized complexity landscape for VC dimension computation from a theoretical perspective.
- Lemmas 14/15 are elegant: if a shattered set crosses both sides of a separator, its size is bounded by $O(\log|Z| + 2)$.
- The DP over pattern graph matching is carefully designed — encoding the combinatorial relationship between shattered sets and witnesses as a graph embedding problem.

## Limitations & Future Work

- A **gap** remains between the treewidth algorithm $2^{O(\text{tw}\cdot\log \text{tw})}$ and the lower bound $2^{o(\text{vcn}+k)}$; closing this gap is the main open problem.
- Whether the 1-additive FPT approximation (parameterized by $\Delta$) can be improved to an exact FPT algorithm remains open.
- The setting where set systems are defined by circuits (potentially allowing more succinct input representations) is not considered.

## Related Work & Insights

- A natural continuation of the LogNP-completeness result of Papadimitriou–Yannakakis.
- A structural-parameter complement to Manurangsi's approximation hardness result (no $o(\log|\mathcal{H}|)$ approximation under Gap-ETH).
- Provides direct theoretical support for applications of VC dimension to graph-structured data in machine learning.

## Rating

⭐⭐⭐⭐ — Solid theoretical contributions, a complete parameterized complexity landscape, elegant proof techniques; a high-quality work at the intersection of computational learning theory and parameterized complexity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[NeurIPS 2025\] The Structural Complexity of Matrix-Vector Multiplication](the_structural_complexity_of_matrix-vector_multiplication.md)
- [\[NeurIPS 2025\] How Many Domains Suffice for Domain Generalization? A Tight Characterization via the Domain Shattering Dimension](how_many_domains_suffice_for_domain_generalization_a_tight_characterization_via_.md)
- [\[NeurIPS 2025\] UniFormer: Unified and Efficient Transformer for Reasoning Across General and Custom Computing](uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)
- [\[AAAI 2026\] Finding Diverse Solutions Parameterized by Cliquewidth](../../AAAI2026/others/finding_diverse_solutions_parameterized_by_cliquewidth.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds
description: >-
  [ICLR 2026][Non-Clashing Teaching] This paper investigates the non-clashing teaching of closed neighborhood concept classes in graphs. It provides matching algorithmic upper and lower bounds ($2^{\mathcal{O}(|E|)}$ tight bound for N-NCTD⁺), FPT algorithms parameterized by treedepth and vertex cover (including the first FPT result for negative labels), and combinatorial upper bounds for planar and unit square graphs, comprehensively advancing the computational and combinatoria…
tags:
  - "ICLR 2026"
  - "Non-Clashing Teaching"
  - "Graph Concept Classes"
  - "FPT Algorithms"
  - "Teaching Dimension"
  - "Combinatorial Complexity"
  - "Parameterized Complexity"
date: 2026-05-08
content_hash: 2c4cc95f3ac03dbe
---

# Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds

**Conference**: ICLR 2026  
**arXiv**: [2602.00657](https://arxiv.org/abs/2602.00657)

**Area**: Learning Theory/Machine Teaching  
**Keywords**: Non-Clashing Teaching, Graph Concept Classes, FPT Algorithms, Teaching Dimension, Combinatorial Complexity, Parameterized Complexity

## TL;DR

This paper investigates the non-clashing teaching of closed neighborhood concept classes in graphs. It provides matching algorithmic upper and lower bounds ($2^{\mathcal{O}(|E|)}$ tight bound for N-NCTD⁺), FPT algorithms parameterized by treedepth and vertex cover (including the first FPT result for negative labels), and combinatorial upper bounds for planar and unit square graphs, comprehensively advancing the computational and combinatorial understanding of non-clashing teaching.

## Background & Motivation

**Background**: Non-Clashing Teaching is currently the most efficient batch machine teaching model known to satisfy the Goldman-Mathias anti-collusion benchmark. Given a concept class $\mathcal{C}$, a teacher assigns a teaching set $T(C)$ to each concept $C \in \mathcal{C}$ such that for any two distinct concepts $C, C'$, at least one sample in $T(C) \cup T(C')$ is consistent with only one of them (i.e., it "distinguishes" them). The non-clashing teaching dimension $\text{NCTD}(\mathcal{C})$ is the minimum size of teaching sets satisfying this condition.

**Limitations of Prior Work**:
   - Previous FPT algorithms for "balls in graphs" concept classes were only established for the vertex integrity parameter, while solvability under more general parameters like treedepth remained unknown.
   - When negative labels (negative examples) are allowed in teaching, **no FPT results existed**, representing a significant theoretical gap.
   - Previous algorithmic lower bounds for NCTD were only $2^{o(\sqrt{|V|})}$ (derived from a reduction to open neighborhoods in [KSZ19]), leaving a massive gap with the known upper bound of $2^{\mathcal{O}(|V| \cdot k \cdot \log|V|)}$.

**Key Insight**: By selecting closed neighborhoods (radius-1 balls) as the concept class, any finite binary concept class $\mathcal{C} \subseteq 2^V$ can be equivalently represented as a set of closed neighborhoods $\{N[x_C] \mid C \in \mathcal{C}\}$ in some graph $G$ (by constructing $V(G) = V \cup \{x_C\}$, where concept vertices form a clique and $x_C$ is adjacent to $v \in C$). Thus, the results achieve **maximum generality**.

**Core Problem**:
   - **N-NCTD**: Given a graph $G$, a set of closed neighborhoods $\mathcal{B}$, and an integer $k$, determine if $\text{NCTD}(\mathcal{B}) \leq k$.
   - **N-NCTD⁺**: Same as above, but restricting teaching sets to only positive labels; determine if $\text{NCTD}^+(\mathcal{B}) \leq k$.

**Value**: The relationship between VC dimension and NCTD is a core open problem (does $\text{NCTD}(\mathcal{C}) > \text{VCD}(\mathcal{C})$ exist?). Researching closed neighborhoods may provide key clues to answering this.

**Applications**: Machine teaching applies to sample compression, inverse reinforcement learning, training data security, and human-computer interaction, with theoretical advancements potentially impacting all these directions.

## Method

### Overall Architecture

The paper deconstructs non-clashing teaching for closed neighborhood concept classes $\mathcal{B} = \{N[v]\}$ into three parallel theoretical lines: first, using a 3-SAT reduction combined with exhaustive search to provide precise exponential time bounds for N-NCTD and N-NCTD⁺; second, using treedepth pruning and kernelization to obtain FPT algorithms for treedepth and vertex cover; and third, using graph-theoretic structures to prove constant combinatorial upper bounds for planar and unit square graphs. All three lines rely on the same set of tools: equivalent representation of closed neighborhoods, false twins analysis, and the layered pigeonhole principle.

### Key Designs

**1. Equivalent Representation of Closed Neighborhoods: Embedding General Concept Classes**

To ensure conclusions apply not just to specific graph types but to all finite concept classes, the first step is finding a lossless encoding. Any finite binary concept class $\mathcal{C} \subseteq 2^V$ can be written as a set of closed neighborhoods in some graph: construct $V(G) = V \cup \{x_C \mid C \in \mathcal{C}\}$, let concept vertices $\{x_C\}$ form a clique, and let $x_C \sim v \iff v \in C$. Thus, $\mathcal{B} = \{N[x_C]\}$ equivalently represents $\mathcal{C}$. This step is lossless, meaning bounds proven for closed neighborhoods (radius-1 balls) apply to all finite concept classes, offering generality far beyond traditional graph theory.

Beyond encoding, the work utilizes false twins (pairs of vertices with the same open neighborhood, $N(u)=N(v)$) and the layered pigeonhole principle. False twins impose strong constraints on teaching mappings—they are difficult to distinguish, forcing teaching sets to use specific vertices. The pigeonhole principle translates "too many twins" into the ability to "safely remove a vertex," a lever used repeatedly in lower bounds and kernelization.

**2. Improved Algorithmic Bounds: Pinning N-NCTD⁺ at $2^{\Theta(|E|)}$**

For the general N-NCTD problem, a lower bound (Theorem 2) is derived from a 3-SAT reduction. Given an instance $\varphi$ with $n$ variables and $m$ clauses, a graph $G$ is constructed with variable vertices $v_i$, literal vertices $t_i, f_i$, and clause vertices $c_j$, along with virtual variable vertices $v_0$, auxiliary sets $\mathcal{V}_i = \{v_i^0, \ldots, v_i^4\}$, and special vertices $v_i^\star$. Clauses and variables form cliques, yielding $|V(G)| = \mathcal{O}(n+m)$. The pigeonhole principle (Lemma 1) shows that 4 pairwise false twins produce 6 pairs of closed neighborhoods to distinguish, while a size-1 teaching set can distinguish at most 1 pair per non-self vertex. This forces the teaching set to be exactly $\{u_i\}$, locking the SAT truth assignment into the teaching mapping. Consequently, unless ETH fails, N-NCTD cannot be solved in $2^{o(f(k)\cdot|V(G)|)}$, significantly improving the previously implied $2^{o(\sqrt{|V|})}$ bound.

For N-NCTD⁺ (positive labels only), the upper and lower bounds match perfectly. A similar reduction provides a $2^{o(f(k)\cdot(|V|+|E|))}$ lower bound (Theorem 3). The upper bound (Theorem 4) is obtained by exhausting all positive teaching mappings—since $T(N[v]) \subseteq N[v]$ has $2^{d(v)+1}$ options, the total is $2^{\sum(d(v)+1)} = 2^{\mathcal{O}(|E|)}$. The exact match at $2^{\Theta(|E|)}$ is the first precisely determined exponential algorithmic bound in the field of non-clashing teaching.

**3. FPT Algorithms: Treedepth Pruning and Vertex Cover Kernelization**

The FPT for N-NCTD⁺ under treedepth (Theorem 5) uses bottom-up pruning on a treedepth decomposition $\mathcal{T}$. Reduction Rule 1 selects $X \subseteq V(G)$ and identifies $A = \{A_1, \ldots, A_\ell\}$ as a subset of connected components in $G-X$. If $\ell > (|X|+t)\cdot 2^{(|X|+t)^2}\cdot 2^{2t+|X|+1}$ (where $t = \max|A_i|$), a specific component is deleted. Soundness (Lemma 6) relies on the pigeonhole principle: if the number of components exceeds this threshold, there must exist 3 isomorphic components $A_P, A_Q, A_R$ whose adjacency and teaching sets are identical. Thus, teaching elements in $A_P$ can be replaced by copies in $A_Q/A_R$. Pruning reduces the graph to size $f(\text{td}(G))$ for brute-force solving. The FPT for N-NCTD under vertex cover (Theorem 7) provides the first result for negative labels using kernelization. Lemma 8 bounds the solution size by $\text{NCTD}(\mathcal{B}) \leq 2^{|X|+1} + |X|$ (where $X$ is the vertex cover). Reduction rules handle pairwise false twins and vertices whose neighborhoods are not in $\mathcal{B}$. The resulting kernel size depends only on $|X|$. Negative labels allow teaching sets to include vertices outside the closed neighborhood, making the problem "non-local" and the proof significantly more complex.

**4. Combinatorial Upper Bounds: Trading Graph Structure for Constant Teaching Sets**

Finally, constant upper bounds for specific graph classes are summarized in the table below.

| Graph Class | NCTD⁺ | NCTD |
|------|-------|------|
| Planar Graphs | $\leq 7$ (Thm 12) | $\leq 5$ (Thm 13) |
| Unit Square Graphs | $\leq 4$ (Thm 14) | — |

The planar graph bound $\text{NCTD}^+ \leq 7$ is handled by degree classification. For vertices with degree $\leq 6$, the teacher simply sets $T(N[v]) := N[v]$. For degree $\geq 7$, 3 neighbors are selected for $T(N[v])$. The $K_{3,3}$-free property of planar graphs ensures at most one other vertex $u$ also neighbors these 3, requiring at most 4 additional vertices to reach $|T| \leq 7$. For unit square graphs, $\text{NCTD}^+ \leq 4$ is a geometric argument: each closed neighborhood is contained in a bounding box $R(v)$. $T(N[v])$ takes the leftmost, rightmost, topmost, and bottommost squares; as long as $R(u) \neq R(v)$, an extremal square will distinguish them. Note that NCTD $\leq 5$ for planar graphs may be higher than VCD $\leq 4$, suggesting counterexamples for $\text{NCTD} > \text{VCD}$ may exist in planar graphs.

## Main Results

### Algorithmic Complexity Overview

| Problem | Parameter | Result (Ours) | Prev. Best |
|------|------|---------|---------|
| N-NCTD | — | Lower bound $2^{o(f(k)\cdot|V|)}$ impossible | $2^{o(\sqrt{|V|})}$ [KSZ19] |
| N-NCTD⁺ | — | **Exact** $2^{\Theta(|E|)}$ | No lower bound |
| N-NCTD⁺ | treedepth | **FPT** | vertex integrity [GKM+25] |
| N-NCTD | vertex cover | **FPT** | None (First negative label FPT) |

### Combinatorial Upper Bounds

| Graph Class | NCTD⁺ Bound | NCTD Bound | VCD |
|------|-----------|----------|-----|
| Planar Graphs | $\leq 7$ | $\leq 5$ | $\leq 4$ |
| Unit Square Graphs | $\leq 4$ | — | $\leq 4$ |
| Trees/Cycles/Cactus | Known optimal [CCM+24] | — | — |

## Key Findings

- The exponential time complexity of N-NCTD⁺ is precisely determined at $2^{\Theta(|E|)}$, marking the first matching algorithmic bound in non-clashing teaching.
- Treedepth strictly generalizes the vertex integrity parameter, meaning the FPT results cover a broader class of graphs.
- Allowing negative labels makes the problem "non-local" (as teaching sets can contain vertices not in the closed neighborhood), significantly complicating proofs, though kernelization remains feasible.
- The possibility that NCTD $\leq 5$ for planar graphs exceeds VCD $\leq 4$ hints that counterexamples to the conjecture $\text{NCTD} \leq \text{VCD}$ might be found in planar graphs.

## Highlights & Insights

- **Matching Exponential Bounds**: $2^{\Theta(|E|)}$ indicates the problem is "closed" under this parameter with no room for further improvement.
- **Generality**: The "Any finite concept class = closed neighborhood" insight makes the results highly universal, extending beyond graph theory.
- **First Negative Label FPT**: Breaks the bottleneck where previous FPT results were restricted to positive teaching.
- **Interdisciplinary Contribution**: A fusion of learning theory, graph theory, and parameterized complexity.
- **Technique**: Sophisticated use of the pigeonhole principle across multiple layers for reduction rule soundness.

## Limitations & Future Work

- Purely theoretical work without experimental validation; all results are mathematical theorems rather than tests in practical teaching scenarios.
- The computable function $f$ in FPT algorithms may grow extremely fast (multi-exponential), limiting practical solvability.
- FPT for treedepth is limited to the positive variant (N-NCTD⁺); the complexity of the negative variant under treedepth remains unknown.
- Tightness of the planar graph bound $\text{NCTD} \leq 5$ is undetermined—does there exist a planar graph where $\text{NCTD} = 5$?
- Complexity under the treewidth parameter (expected to be W[1]-hard) has not yet been proven.
- Kernel sizes are multi-exponential, making them potentially impractical for real-world applications.

## Related Work & Insights

### vs. Chalopin et al. [COLT 2024] (CCM+24)
CCM+24 studied positive non-clashing teaching for all balls in graphs: (1) Strict Non-Clash was FPT for vertex cover but with $2^{2^{\mathcal{O}(\text{vc})}}$ complexity; (2) Optimal NCTM for trees/cycles/cactus graphs. This paper focuses on closed neighborhoods (radius-1) and achieves **stronger results**: FPT for the more general treedepth, the first negative label FPT, and broader combinatorial bounds.

### vs. Ganian et al. [ICLR 2025] (GKM+25)
GKM+25 proved Non-Clash is FPT for vertex integrity and W[1]-hard for the combined parameter of feedback vertex number + pathwidth + $k$. This paper advances these findings by: (1) establishing FPT for treedepth; (2) providing a matching $2^{\Theta(|E|)}$ bound (GKM+25's upper bound contains an extra $\log$ factor); (3) handling negative labels for the first time.

### vs. Kirkpatrick et al. [ALT 2019] (KSZ19)
KSZ19 introduced the non-clashing model and proved that deciding NCTD for open neighborhoods is NP-complete. This paper's $2^{o(|V|)}$ lower bound significantly improves upon the $2^{o(\sqrt{|V|})}$ bound implied by KSZ19.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Exact matching bounds + first negative label FPT are significant.
- Theoretical Depth: ⭐⭐⭐⭐⭐ Sophisticated proof techniques involving graph theory and parameterized complexity.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematics with clear structure and helpful figures.
- Impact: ⭐⭐⭐⭐ Significant advancement for theoretical machine teaching, though specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](distributed_algorithms_for_euclidean_clustering.md)
- [\[ICLR 2026\] Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds](deterministic_bounds_and_random_estimates_of_metric_tensors_on_neuromanifolds.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[ICLR 2026\] The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)

</div>

<!-- RELATED:END -->

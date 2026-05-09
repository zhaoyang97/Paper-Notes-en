---
title: >-
  [Paper Note] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds
description: >-
  [ICLR 2026][LLM Evaluation][non-clashing teaching] This paper studies non-clashing teaching of closed-neighborhood concept classes in graphs, providing tight algorithmic bounds (a matching $2^{\mathcal{O}(|E|)}$ bound for N-NCTD⁺), FPT algorithms parameterized by treedepth and vertex cover (including the first FPT result with negative labels), and combinatorial upper bounds for planar graphs and unit square graphs, substantially advancing both the computational and combinatorial understanding of non-clashing teaching.
tags:
  - ICLR 2026
  - LLM Evaluation
  - non-clashing teaching
  - graph concept classes
  - FPT algorithms
  - teaching dimension
  - combinatorial complexity
  - parameterized complexity
date: 2026-05-08
content_hash: 5873f3b12aab674e
---

# Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds

**Conference**: ICLR 2026
**arXiv**: [2602.00657](https://arxiv.org/abs/2602.00657)

**Area**: Learning Theory / Machine Teaching
**Keywords**: non-clashing teaching, graph concept classes, FPT algorithms, teaching dimension, combinatorial complexity, parameterized complexity

## TL;DR

This paper studies non-clashing teaching of closed-neighborhood concept classes in graphs, providing tight algorithmic bounds (a matching $2^{\mathcal{O}(|E|)}$ bound for N-NCTD⁺), FPT algorithms parameterized by treedepth and vertex cover (including the first FPT result with negative labels), and combinatorial upper bounds for planar graphs and unit square graphs, substantially advancing both the computational and combinatorial understanding of non-clashing teaching.

## Background & Motivation

**Background**: Non-Clashing Teaching is currently the most efficient batch machine teaching model known to satisfy the Goldman–Mathias collusion-avoidance benchmark. Given a concept class $\mathcal{C}$, a teacher assigns a teaching set $T(C)$ to each concept $C \in \mathcal{C}$, subject to the requirement that for any two distinct concepts $C, C'$, at least one example in $T(C) \cup T(C')$ is consistent with exactly one of the two concepts (i.e., it "separates" them). The non-clashing teaching dimension $\text{NCTD}(\mathcal{C})$ is the minimum teaching set size satisfying this condition.

**Limitations of Prior Work**:
   - Prior FPT algorithms for "balls in graphs" concept classes were only established under the vertex integrity parameterization; tractability under more general parameters such as treedepth remained unknown.
   - When negative labels (negative examples) are permitted in teaching, **no FPT result existed**—a significant theoretical gap.
   - The best known algorithmic lower bound for NCTD was only $2^{o(\sqrt{|V|})}$ (derived from [KSZ19]'s reduction for open neighborhoods), far from the known upper bound of $2^{\mathcal{O}(|V| \cdot k \cdot \log|V|)}$.

**Key Insight**: The paper focuses on closed neighborhoods (radius-1 balls) as the concept class. Since any finite binary concept class $\mathcal{C} \subseteq 2^V$ can be equivalently represented as a collection of closed neighborhoods $\{N[x_C] \mid C \in \mathcal{C}\}$ in some graph $G$ (constructed by setting $V(G) = V \cup \{x_C\}$, making concept vertices a clique, and connecting $x_C$ to $v \in C$), the results achieve **maximum generality**.

**Key Problem Definitions**:
   - **N-NCTD**: Given a graph $G$, a closed-neighborhood collection $\mathcal{B}$, and an integer $k$, decide whether $\text{NCTD}(\mathcal{B}) \leq k$.
   - **N-NCTD⁺**: Same as above but restricting teaching sets to positive labels only; decide whether $\text{NCTD}^+(\mathcal{B}) \leq k$.

**Theoretical Significance**: The relationship between VC dimension and NCTD is a central open problem (does $\text{NCTD}(\mathcal{C}) > \text{VCD}(\mathcal{C})$ ever hold?), and the study of closed neighborhoods may provide critical insight toward resolving it.

**Broader Applications**: Machine teaching has applications in sample compression, inverse reinforcement learning, training data security, and human–computer interaction; advances in its theoretical foundations have potential impact across these areas.

## Method

### Core Contribution 1: Improved Algorithmic Upper and Lower Bounds

**N-NCTD Lower Bound (Theorem 2)**:
Via reduction from 3-SAT. Given a 3-SAT instance $\varphi$ with $n$ variables and $m$ clauses, a graph $G$ is constructed as follows:

- Each variable $x_i$ yields a variable vertex $v_i$ and two literal vertices $t_i, f_i$.
- Each clause $C_j$ yields a clause vertex $c_j$.
- Auxiliary variable vertices $v_0$, sets $\mathcal{V}_i = \{v_i^0, \ldots, v_i^4\}$, and special vertices $v_i^\star$ are introduced.
- Clause/variable vertices form a clique.

Key Lemma (Lemma 1): For 4 pairwise false twins, under a size-1 NCTM, at least one must have its teaching set equal to $\{u_i\}$ (by a pigeonhole argument: 4 vertices generate 6 pairs to be separated, and each non-self vertex can separate at most 1 pair).

Conclusion: Unless ETH fails, N-NCTD cannot be solved in $2^{o(f(k) \cdot |V(G)|)}$ time (where $|V(G)| = \mathcal{O}(n+m)$), **substantially improving** the prior $2^{o(\sqrt{|V|})}$ lower bound.

**Tight Bounds for N-NCTD⁺**:
- **Lower bound (Theorem 3)**: An analogous reduction establishes that $2^{o(f(k) \cdot (|V|+|E|))}$ is impossible.
- **Upper bound (Theorem 4)**: Exhaustively enumerate all positive teaching mappings—for each $v$, the positive teaching set $T(N[v]) \subseteq N[v]$ admits $2^{d(v)+1}$ choices, giving $2^{\sum(d(v)+1)} = 2^{\mathcal{O}(|E|)}$ mappings in total.
- **Exact match**: The upper and lower bounds meet at $2^{\Theta(|E|)}$.

### Core Contribution 2: FPT Algorithms

**N-NCTD⁺ Parameterized by Treedepth (FPT, Theorem 5)**:

The key idea is a bottom-up pruning procedure on the treedepth decomposition $\mathcal{T}$.

- **Reduction Rule 1**: Let $X \subseteq V(G)$, and let $A = \{A_1, \ldots, A_\ell\}$ be a subset of connected components of $G - X$ with $\max|A_i| = t$. If $\ell > (|X|+t) \cdot 2^{(|X|+t)^2} \cdot 2^{2t+|X|+1}$, remove a specific component.
- **Safeness (Lemma 6)**: A pigeonhole argument guarantees three automorphic components $A_P, A_Q, A_R$ with identical adjacency structure and teaching sets (including positive constraints); teaching set elements in $A_P$ can be replaced by their counterparts in $A_Q$ or $A_R$, so $A_P$ can be safely removed.
- **Algorithm**: Pruning proceeds layer by layer from the leaves; after pruning each layer, the node count is bounded by $g_j(\text{td}(G))$, yielding a graph of size $f(\text{td}(G))$ that is then solved by brute force.

**N-NCTD Parameterized by Vertex Cover (FPT, Theorem 7)**:

This is the **first** FPT result that admits negative labels. The approach is kernelization.

- **Lemma 8**: $\text{NCTD}(\mathcal{B}) \leq 2^{|X|+1} + |X|$ for vertex cover $X$, providing an upper bound on solution size.
- **Reduction Rule 2**: If an equivalence class in the independent set contains $q + 2k + 1$ pairwise false twins whose closed neighborhoods are all in $\mathcal{B}$, remove one (Lemma 9, via pigeonhole).
- **Reduction Rule 3**: If two false twins $u, v$ in the independent set both have their closed neighborhoods absent from $\mathcal{B}$, remove $v$.
- After exhaustive application, the vertex count is bounded by $2^{|X|}(2^{2^{|X|}+|X|} + 2^{|X|+2} + 2|X|) + |X|$, yielding a kernel whose size depends solely on $|X|$.

### Core Contribution 3: Combinatorial Upper Bounds

| Graph Class | NCTD⁺ | NCTD |
|-------------|--------|------|
| Planar graphs | $\leq 7$ (Thm 12) | $\leq 5$ (Thm 13) |
| Unit square graphs | $\leq 4$ (Thm 14) | — |

**Planar Graphs (Theorem 12, NCTD⁺ ≤ 7)**:
- Vertices of degree $\leq 6$: set $T(N[v]) := N[v]$.
- Vertices of degree $\geq 7$: place 3 neighbors in $T(N[v])$; by the $K_{3,3}$ forbidden subgraph property, at most one other vertex $u$ is adjacent to all 3, so at most 4 additional vertices need to be added, giving $|T| \leq 7$.

**Unit Square Graphs (Theorem 14, NCTD⁺ ≤ 4)**:
A geometric argument shows that the closed neighborhood of each vertex $v$ is contained in a minimum bounding rectangle $R(v)$; the teaching set $T(N[v])$ consists of the leftmost, rightmost, topmost, and bottommost squares. If $R(u) \neq R(v)$, some extreme square separates the two neighborhoods.

### Key Designs

- **Closed neighborhood representation**: Setting $V(G) = V \cup \{x_C \mid C \in \mathcal{C}\}$, making concept vertices a clique, and connecting $x_C \sim v \iff v \in C$ yields $\mathcal{B} = \{N[x_C]\}$ as an equivalent representation of $\mathcal{C}$.
- **False twin analysis**: A central tool throughout all reduction rules—vertices sharing the same open neighborhood impose strong constraints on teaching mappings.
- **Pigeonhole principle**: Applied at multiple levels, from Lemma 1 (4 twins $\Rightarrow$ one must self-select) to Lemma 9 (many twins $\Rightarrow$ two teaching sets are "identical").

## Key Experimental Results

### Algorithmic Complexity Overview

| Problem | Parameter | This Paper | Prior Best |
|---------|-----------|-----------|-----------|
| N-NCTD | — | Lower bound: $2^{o(f(k)\cdot|V|)}$ impossible | $2^{o(\sqrt{|V|})}$ [KSZ19] |
| N-NCTD⁺ | — | **Exact** $2^{\Theta(|E|)}$ | No lower bound |
| N-NCTD⁺ | treedepth | **FPT** | vertex integrity [GKM+25] |
| N-NCTD | vertex cover | **FPT** | None (first negative-label FPT) |

### Combinatorial Upper Bounds

| Graph Class | NCTD⁺ Upper Bound | NCTD Upper Bound | VCD |
|-------------|-------------------|-----------------|-----|
| Planar graphs | ≤ 7 | ≤ 5 | ≤ 4 |
| Unit square graphs | ≤ 4 | — | ≤ 4 |
| Trees/cycles/cacti | Optimal [CCM+24] | — | — |

## Key Findings

- The exponential time complexity of N-NCTD⁺ is precisely characterized at $2^{\Theta(|E|)}$—the first exact matching algorithmic bound in the non-clashing teaching literature.
- Treedepth strictly generalizes vertex integrity, so the FPT result covers a broader class of graphs.
- Allowing negative labels "delocalizes" the problem (teaching sets may contain vertices outside the closed neighborhood), significantly complicating proofs, yet kernelization remains feasible.
- The bound NCTD $\leq 5$ for planar graphs may exceed VCD $\leq 4$, suggesting that counterexamples to $\text{NCTD} > \text{VCD}$ might be found among planar graphs.

## Highlights & Insights

- **Exact matching exponential bound**: $2^{\Theta(|E|)}$ shows the problem is fully "closed" under this parameter, leaving no room for further improvement.
- **Any finite concept class = closed neighborhoods**: The results have maximum generality, extending well beyond the graph-theoretic setting.
- **First FPT with negative labels**: Breaks the barrier where all prior FPT results were restricted to positive teaching.
- **Elegant cross-disciplinary contribution** at the intersection of learning theory, graph theory, and parameterized complexity.
- The multi-level application of the pigeonhole principle in proving reduction rule safeness is technically sophisticated.

## Limitations & Future Work

- This is purely theoretical work with no empirical validation; all results are mathematical theorems with no evaluation in practical teaching scenarios.
- The computable function $f$ in the FPT algorithms may grow super-exponentially, limiting practical scalability.
- The FPT result parameterized by treedepth applies only to the positive variant (N-NCTD⁺); complexity of the negative variant under treedepth remains open.
- The tightness of NCTD $\leq 5$ for planar graphs is unresolved—does there exist a planar graph achieving NCTD $= 5$?
- The complexity of treewidth parameterization (conjectured W[1]-hard) has not been established.
- The kernel size is multiply exponential, likely rendering the approach impractical in real-world settings.

## Related Work & Insights

### vs. Chalopin et al. [COLT 2024] (CCM+24)
CCM+24 studies positive non-clashing teaching of all balls in graphs: (1) Strict Non-Clash is FPT parameterized by vertex cover, but with runtime $2^{2^{\mathcal{O}(\text{vc})}}$; (2) optimal NCTMs are given for special graph classes such as trees, cycles, and cacti. The present paper focuses on closed neighborhoods (radius-1 balls) and obtains **stronger results**: FPT under the more general treedepth parameter, the first FPT with negative labels, and combinatorial upper bounds for broader graph classes.

### vs. Ganian et al. [ICLR 2025] (GKM+25)
GKM+25 establishes that Non-Clash is FPT under vertex integrity and W[1]-hard under the combined parameterization of feedback vertex number + pathwidth + $k$, yielding an almost complete complexity landscape. The present paper advances this in three ways: (1) FPT under treedepth, which is more general than vertex integrity; (2) an exact $2^{\Theta(|E|)}$ algorithmic bound (GKM+25's upper bound contains an additional logarithmic factor); (3) the first treatment of variants with negative labels.

### vs. Kirkpatrick et al. [ALT 2019] (KSZ19)
KSZ19 introduces the non-clashing teaching model and proves NP-completeness of NCTD for open neighborhoods. The present paper's lower bound of $2^{o(|V|)}$ substantially improves the $2^{o(\sqrt{|V|})}$ lower bound implied by KSZ19.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Exact matching algorithmic bounds, first negative-label FPT, and treedepth FPT are all entirely new results.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — A purely theoretical work with elegant proof techniques (multi-level pigeonhole, automorphic component pruning, kernelization).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematically rigorous, clearly structured, with figures (Fig. 1–8) that greatly aid comprehension.
- **Impact**: ⭐⭐⭐⭐ — Makes important contributions to the foundations of machine teaching and parameterized complexity, though the target audience is relatively specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)
- [\[AAAI 2026\] Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](../../AAAI2026/llm_evaluation/towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)
- [\[NeurIPS 2025\] Tight Lower Bounds and Improved Convergence in Performative Prediction](../../NeurIPS2025/llm_evaluation/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)
- [\[NeurIPS 2025\] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](../../NeurIPS2025/llm_evaluation/a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [\[ICLR 2026\] Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](rethinking_benign_relearning_syntax_as_the_hidden_driver_of_the_safety_tax.md)

</div>

<!-- RELATED:END -->

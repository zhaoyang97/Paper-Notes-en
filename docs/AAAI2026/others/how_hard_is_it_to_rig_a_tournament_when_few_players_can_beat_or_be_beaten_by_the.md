---
title: >-
  [Paper Note] How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?
description: >-
  [AAAI 2026][tournament fixing] This paper introduces two novel structural parameters — the in-degree $k$ and out-degree $\ell$ of the favorite player in the tournament digraph — for analyzing the Tournament Fixing Proble…
tags:
  - "AAAI 2026"
  - "tournament fixing"
  - "knockout tournament"
  - "parameterized complexity"
  - "FPT"
  - "color coding"
date: 2026-05-08
content_hash: 5dbde3c86a1f1207
---

# How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?

**Conference**: AAAI 2026
**arXiv**: [2601.08530](https://arxiv.org/abs/2601.08530)  
**Code**: None  
**Area**: Other
**Keywords**: tournament fixing, knockout tournament, parameterized complexity, FPT, color coding

## TL;DR

This paper introduces two novel structural parameters — the in-degree $k$ and out-degree $\ell$ of the favorite player in the tournament digraph — for analyzing the Tournament Fixing Problem (TFP). It proves that TFP is FPT under both parameterizations, where the in-degree algorithm involves sophisticated structural analysis and the color coding technique.

## Background & Motivation

Knockout tournaments are among the most widely used competition formats: players are paired each round, losers are eliminated, and winners advance until a single champion remains. The FIFA World Cup and NCAA basketball tournament both adopt this format. Beyond sports, knockout tournaments are also used in elections and organizational decision-making.

**Core Problem**: Given a tournament digraph $D$ encoding all pairwise outcomes and a favorite player $v^*$, does there exist a seeding under which $v^*$ is guaranteed to win? This is the Tournament Fixing Problem (TFP), which has been shown to be NP-hard.

**Prior Parameterized Results**: TFP is FPT when parameterized by the feedback arc set number (fas) or feedback vertex set number (fvs), both of which are global structural parameters measuring how far the tournament digraph is from being acyclic.

**New Perspective**: If $v^*$ lies in no cycle, TFP can be solved in polynomial time. This motivates two new local parameters — $k$ (the number of players who can beat $v^*$, i.e., in-degree) and $\ell$ (the number of players $v^*$ can beat, i.e., out-degree). These parameters are centered on $v^*$, are intuitively clear, easy to compute, and may admit efficient solutions even when fas/fvs are large.

## Method

### Overall Architecture

The paper is structured in two parts: out-degree parameterization (relatively straightforward) and in-degree parameterization (the core technical contribution). The out-degree case yields an FPT result via a clean observation. The in-degree case follows three steps: structural analysis → intermediate structure (Winning Witness Forest, WWF) → FPT algorithm via color coding.

### Key Designs

1. **Out-Degree Parameterization (Theorem 1, simple case)**:

    - Function: Proves TFP is FPT when parameterized by the out-degree $\ell$ of $v^*$.
    - Mechanism: The champion must win $\log n$ rounds, but $v^*$ can only beat $\ell$ opponents. If $\ell < \log n$, the instance is necessarily a No-instance; otherwise $n \leq 2^\ell$, and a $2^n \cdot n^{O(1)}$-time exact algorithm suffices for FPT runtime.
    - Running time: $2^{2^\ell} \cdot n^{O(1)}$.
    - Lower bound (Theorem 2): Under ETH, no $2^{2^{\ell/c}} \cdot n^{O(1)}$-time algorithm exists for any $c > 1$, indicating the simple algorithm is nearly optimal.

2. **Binomial Arborescence Structure and LBA (Key Equivalence Reduction)**:

    - Function: Reduces TFP to finding a labeled binomial arborescence (LBA) rooted at $v^*$.
    - Core equivalence (Proposition 1, Williams 2010): A seeding making $v^*$ win exists if and only if $D$ admits an LBA rooted at $v^*$.
    - Design Motivation: An LBA precisely corresponds to the winner tree in a knockout tournament — the root is the champion, and each node's subtree root is the opponent it defeated. This equivalence transforms TFP from a combinatorial scheduling problem into a subgraph search problem.

3. **Structural Properties of Nice Seedings (Foundation for In-Degree Parameterization)**:

    - Function: Proves that if TFP is a Yes-instance and $k < \log n$, a "nice" winning seeding necessarily exists.
    - Definition of nice (Definition 3): In each round, either at least one in-neighbor of $v^*$ (a player who can beat $v^*$) is eliminated, or no in-neighbors remain alive.
    - Key corollary (Corollary 1): All players capable of beating $v^*$ must be eliminated within the first $k$ rounds.
    - Proof method: Starting from an arbitrary winning seeding, an iterative "repair" process constructs a nice seeding by reorganizing the losers of non-nice rounds into a sub-tournament that is merged back into the main bracket.
    - Design Motivation: This severely restricts the structure of winning seedings, allowing subsequent algorithm design to focus on small-scale substructures involving $v^*$'s in-neighbors.

4. **Winning Witness Forest (WWF) (Core Intermediate Structure)**:

    - Function: Defines WWF and proves its existence is equivalent to TFP being a Yes-instance (when $k \cdot 2^k < n$).
    - WWF definition (Definition 4): A forest of $k$ vertex-disjoint LBAs each of size $2^k$, containing all in-neighbors of $v^*$, with each tree's root in $N_{\text{out}}(v^*) \cup \{v^*\}$.
    - Equivalence (Lemma 4): The forward direction (constructing a WWF from a winning seeding) uses Lemma 3; the reverse (constructing a winning seeding from a WWF) uses an LBA merging procedure.
    - Design Motivation: WWF decomposes the global LBA existence problem into finding $k$ small trees locally, creating the conditions necessary for applying color coding.

5. **Color Coding Algorithm (Algorithm 1)**:

    - Function: Uses a randomized algorithm to efficiently detect the existence of a WWF.
    - Step 1 (Coloring): Deterministically assigns colors $1, \ldots, k$ to the $k$ vertices of $N_{\text{in}}(v^*)$; remaining vertices are assigned colors sampled uniformly at random from $\{k+1, \ldots, k \cdot 2^k\}$.
    - Step 2 (Colorful WWF detection): Constructs an auxiliary graph $F'$ ($k$ UBAs plus a virtual root $f$) and $D'$ (obtained from $D$ by removing arcs into $v^*$ and adding a virtual vertex $d$), reducing the problem to a colorful subgraph isomorphism problem.
    - Runtime analysis: The probability of a WWF being correctly colored is at least $e^{-t}$ (where $t = k \cdot 2^k - k$); the subgraph isomorphism is solved in $2^t \cdot n^{O(1)}$ time; iterating $e^t$ times yields success probability $1 - 1/e$, giving total time $(2e)^t \cdot n^{O(1)}$.
    - Derandomization: The random coloring can be replaced by a standard splitter construction.

### Runtime Summary

| Parameterization | Running Time | Notes |
|-----------------|-------------|-------|
| Out-degree $\ell$ | $2^{2^\ell} \cdot n^{O(1)}$ | Nearly matches ETH lower bound |
| In-degree $k$ | $(2e)^{k \cdot 2^k - k} \cdot n^{O(1)}$ | Derandomized to $(2e)^{t+o(t)} \cdot n^{O(1)}$ |

## Key Experimental Results

This is a purely theoretical work in parameterized algorithms; no experiments are conducted.

### Theoretical Result Comparison

| Parameterization | Prior Results | This Paper | Relationship |
|-----------------|--------------|------------|-------------|
| fas $p$ | $p^{O(p)} \cdot n^{O(1)}$ FPT | — | $k, \ell$ can be large when $p$ is small |
| fvs $q$ | $q^{O(q)} \cdot n^{O(1)}$ FPT | — | $q \leq p$, but $k, \ell$ have no monotone relation with $q$ |
| Out-degree $\ell$ | New parameter | $2^{2^\ell} \cdot n^{O(1)}$ FPT + ETH lower bound | Complementary perspective |
| In-degree $k$ | New parameter | $(2e)^{k \cdot 2^k} \cdot n^{O(1)}$ FPT | Core technical contribution |
| Subset fas/fvs | Unknown | Remains open | $\leq k$ and $\leq \ell$ |

### Parameter Hierarchy

| Relation | Description |
|---------|-------------|
| subset fas $\leq$ fas | Subset parameter does not exceed global parameter |
| subset fvs $\leq$ fvs | Same as above |
| subset fas $\leq k$ | Subset fas does not exceed in-degree |
| subset fvs $\leq \ell$ | Subset fvs does not exceed out-degree |
| $k, \ell$ vs. fas/fvs | No monotone relationship |

### Key Findings

- The in-degree $k$ can be small even when both fas and fvs are large, so in-degree parameterization opens genuinely new tractable regions.
- The existence of nice seeings is the cornerstone of the entire in-degree parameterization analysis — it constrains the "global tournament structure" to the local condition that all in-neighbors are eliminated within the first $k$ rounds.
- The WWF equivalence compresses the existence problem from a full-size LBA ($n$ nodes) to $k$ small trees ($2^k$ nodes each), making color coding applicable.

## Highlights & Insights

- The problem motivation is highly natural: the new parameterization perspective arises directly from the intuition of "how many players can defeat the favorite."
- The constructive proof of nice seedings (Lemma 2) is elegant: by isolating the losers of non-nice rounds into a separate sub-tournament and merging them back, the seeding structure is iteratively repaired.
- WWF is a beautifully designed intermediate structure — small enough (only $k \cdot 2^k$ nodes) to make color coding feasible, yet preserving sufficient information so that its existence is equivalent to the original problem.
- The simplicity of out-degree parameterization ("too few beatable opponents means winning is impossible") contrasts sharply with the technical depth of in-degree parameterization, revealing two complementary facets of the same problem.

## Limitations & Future Work

- The running time for in-degree parameterization has $k \cdot 2^k$ in the exponent, limiting practical applicability to very small values of $k$.
- The parameterized complexity of subset fas and subset fvs remains open (it is not even known whether TFP is NP-hard when these values equal 1).
- Only deterministic win-loss outcomes (tournament digraph) are considered; probabilistic match results are not addressed.
- The tournament structure is restricted to complete binary trees (player counts being powers of 2); real competitions often involve byes and other mechanisms.
- Although derandomization of color coding is theoretically feasible, the additional $\log n$ factor may affect practical efficiency.

## Related Work & Insights

- Williams's (2010) LBA equivalence theorem is the cornerstone of the entire field; this paper further analyzes the fine-grained structure of LBAs.
- The fas-parameterized FPT algorithms of Ramanujan & Szeider (2017) and Gupta et al. (2018) are direct predecessors; the new parameters provide an incomparable alternative perspective.
- The fvs-parameterized result of Zehavi & Zehavi (2023) represents the strongest known global-parameter result (since $q \leq p$).
- The successful application of color coding (Alon et al., 1995) to subgraph isomorphism is extended here in a novel and non-trivial manner.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introduces entirely new local parameter perspectives; the structural analysis for in-degree parameterization and the WWF concept are highly original.
- Experimental Thoroughness: ⭐⭐⭐ — Purely theoretical work with complete theorems and proofs, but lacking experimental validation of practical algorithm performance.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, progressing systematically from the simpler out-degree case to the more complex in-degree case, with illustrations aiding comprehension.
- Value: ⭐⭐⭐⭐ — Significantly advances the parameterized understanding of TFP, though practical applicability of the algorithms is limited by parameter size.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[AAAI 2026\] Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](align_when_they_want_complement_when_they_need_human-centere.md)
- [\[AAAI 2026\] How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](how_to_marginalize_in_causal_structure_learning.md)
- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](../../CVPR2026/others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)

</div>

<!-- RELATED:END -->

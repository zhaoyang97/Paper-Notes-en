---
title: >-
  [Paper Note] How Hard is it to Explain Preferences Using Few Boolean Attributes?
description: >-
  [AAAI 2026][Boolean Attribute Model] This paper systematically investigates the computational complexity of explaining preference data using the Boolean Attribute Model (BAM). It proves that the problem is NP-complete fo…
tags:
  - "AAAI 2026"
  - "Boolean Attribute Model"
  - "Computational Complexity"
  - "Parameterized Complexity"
  - "Preference Explanation"
  - "NP-Complete"
date: 2026-05-08
content_hash: 6bd8860c75a6ea40
---

# How Hard is it to Explain Preferences Using Few Boolean Attributes?

**Conference**: AAAI 2026
**arXiv**: [2511.13445](https://arxiv.org/abs/2511.13445)
**Code**: None
**Area**: Computational Complexity / Social Choice Theory / Preference Modeling
**Keywords**: Boolean Attribute Model, Computational Complexity, Parameterized Complexity, Preference Explanation, NP-Complete

## TL;DR

This paper systematically investigates the computational complexity of explaining preference data using the Boolean Attribute Model (BAM). It proves that the problem is NP-complete for $k \geq 3$ attributes and solvable in linear time for $k \leq 2$; further, it provides a complete parameterized complexity landscape with respect to the number of voters $n$, candidates $m$, and attributes $k$, and analyzes how problem hardness changes when partial information (cares/has) is known.

## Background & Motivation

**Background**: Preference modeling is a central problem in social choice theory, decision science, and machine learning. A common approach assumes that voters' preferences over candidates are derived from a set of underlying attributes—candidates possess certain attributes, voters care about certain attributes, and preferences are determined by "how many attributes a candidate has that a voter cares about." Such attribute models have been widely adopted in speed-dating experiments (Fisman et al., 2006), stable matching (Künnemann et al., 2019), multi-criteria decision analysis (UTA methods), and related areas.

**Limitations of Prior Work**: Despite the widespread theoretical and practical use of attribute models, there has been almost no algorithmic or complexity-theoretic study of the fundamental problem of computing a Boolean attribute model from revealed preferences. In particular, it remains unknown: (a) how hard is it to find a model with the minimum number of attributes? (b) under which parameter settings is the problem tractable? (c) does partial prior knowledge simplify the problem?

**Key Challenge**: Intuitively, explaining preferences with few Boolean attributes is both concise and interpretable, yet the search space grows exponentially with the number of attributes—one must simultaneously assign attribute subsets to all candidates and to all voters such that every preference pair is correctly explained. Whether this combinatorial explosion leads to inherent hardness is the key question.

**Goal**: Three sub-questions are addressed: (a) Where is the complexity threshold for the BAM problem itself? (b) What is the parameterized complexity with respect to $n$, $m$, $k$, and their combinations? (c) Does knowing the cares or has function make the problem easier or harder?

**Key Insight**: The authors construct a reduction from 3-Coloring, revealing a deep connection between BAM and graph coloring—the attribute subset of each candidate corresponds to a "color," and adjacency is encoded via preference constraints.

**Core Idea**: By establishing reductions between BAM and classical problems such as graph coloring, SAT, and exact set cover, the paper systematically characterizes the computational complexity landscape of Boolean attribute preference explanation under various parameter settings.

## Method

### Overall Architecture

This paper is a theoretical computational complexity study and does not involve deep learning frameworks. The overall research approach is:

- **Input**: A preference profile $\mathcal{P} = (\mathcal{C}, \mathcal{V}, \mathcal{R})$ (set of candidates, set of voters, set of preference orders) + an upper bound $k$ on the number of attributes
- **Output**: Whether a $k$-BAM exists that explains the preference profile
- **Methodology**: For three problem variants (BAM, BAM with Cares, BAM with Has), NP-hardness proofs or FPT/polynomial-time algorithms are developed with respect to $n$, $m$, $k$, and their combinations

### Key Designs

1. **NP-Completeness of BAM (Theorem 1)**:

    - **Function**: Proves that BAM is NP-complete already when $k = 3$ and each preference order has length 2.
    - **Mechanism**: Reduction from 3-Coloring. Given a graph $G = (U, E)$, a candidate $c_u$ is created for each vertex $u$, along with 7 dummy candidates (divided into groups $D_1, D_2, D_3$ possessing 1, 2, and 3 attributes respectively). Key design: (a) three voters $v_u^1, v_u^2, v_u^3$ ensure that $c_u$ possesses at most one attribute; (b) edge voters $v_{(u,w)}, v_{(w,u)}$ ensure that adjacent vertices' candidates have distinct attribute subsets; (c) 24 dummy voters fix the attribute assignments of dummy candidates. A valid 3-coloring corresponds exactly to a valid 3-BAM.
    - **Design Motivation**: This reduction elegantly maps "adjacent vertices receive different colors" to "adjacent candidates have distinct attribute subsets," revealing the combinatorial nature of preference explanation.

2. **Linear-Time Algorithm for $k \leq 2$ (Theorem 2)**:

    - **Function**: Proves that BAM is solvable in $O(n)$ time when $k \leq 2$.
    - **Mechanism**: The BAM problem is reduced to 2-SAT. For $k = 2$, each candidate's has-set has only 4 possibilities ($\emptyset, \{1\}, \{2\}, \{1,2\}$), and similarly for each voter's cares-set. Preference constraints are encoded as 2-SAT clauses, which are solvable in linear time.
    - **Design Motivation**: The complexity jump between $k = 2$ and $k = 3$ constitutes a clean complexity dichotomy.

3. **FPT Algorithm Parameterized by $m$ (Theorem 3)**:

    - **Function**: Proves that BAM is fixed-parameter tractable with respect to the number of candidates $m$, with time complexity $2^{O(m^3)} \cdot |\mathcal{P}|^{O(1)}$.
    - **Mechanism**: All possible attribute subset assignments for candidates (at most $(2^k)^m$ options) are enumerated; for each assignment, each voter is independently checked for consistency. A key observation (Lemma 3) is that $k \geq m(m-1)$ guarantees the existence of a BAM, so $k$ is also bounded by $m$.
    - **Design Motivation**: In practice, the number of candidates $m$ is typically far smaller than the number of voters $n$, making FPT parameterization by $m$ practically meaningful.

4. **Linear-Time Algorithm for Two Voters (Theorem 4)**:

    - **Function**: For $n = 2$, determines the minimum attribute count $k$ in $O(m)$ time and constructs the corresponding BAM in $O(m^2)$ time.
    - **Mechanism**: Attributes are categorized into three types: $\mathsf{AT}_u$ (cared about only by $u$), $\mathsf{AT}_w$ (cared about only by $w$), and $\mathsf{AT}_{u,w}$ (cared about by both). For each candidate $c$, a "tightness score" $\lambda_v(c) = |\succ_v| - r_v(c) - 1$ is computed, and a greedy assignment minimizes the total number of attributes. A key technique introduces $\text{conv}_c$ variables to enable conversion between attribute types.
    - **Design Motivation**: The $n = 2$ case has sufficient structure to derive exact lower bounds and prove they are always achievable.

5. **NP-Hardness of BAM with Cares (Theorems 5–6)**:

    - **Function**: Proves that BAM remains NP-hard even when the cares function is known; hardness persists even for $m = 3$ or $k = 6$.
    - **Mechanism**: Reduction from 3-SAT. SAT variables are encoded as attribute assignments to candidates, and clause constraints are encoded as voter preferences.
    - **Design Motivation**: Demonstrates that knowing what voters care about does not fundamentally simplify the problem—the difficulty lies in determining which attributes candidates possess.

6. **Complexity Analysis of BAM with Has (Theorems 9–11)**:

    - **Function**: Proves that BAM remains NP-hard when the has function is known (even for $n = 1$), but is FPT with respect to both $m$ and $k$.
    - **Mechanism**: NP-hardness is established via a reduction from Restricted Exact 3-Set Cover. FPT results are obtained via ILP or direct enumeration of $2^k$ possible cares assignments.
    - **Design Motivation**: Reveals an interesting asymmetry—BAM with Has is generally easier than BAM with Cares (FPT in $k$), yet harder in parameter $n$ (NP-hard for $n = 1$).

### Theoretical Analysis

The core contributions are theoretical, relying on several lemmas establishing structural properties of BAM:

- **Lemma 1**: Bounds on the number of attributes each voter cares about and each candidate possesses.
- **Lemma 2**: Lower bound on $k$ induced by rank differences of the same candidate across different preference orders.
- **Lemma 3**: A BAM always exists when $k \geq (m-1) \cdot m$, establishing an equivalence between $k$ and $m$.

## Key Experimental Results

This is a purely theoretical work with no conventional experiments. Core results are presented as a complexity classification table.

### Main Results: Complexity Landscape for Three Problem Variants

| Parameter | BAM | BAM with Cares | BAM with Has |
|-----------|-----|----------------|--------------|
| General | NP-complete [T1] | NP-complete [T5] | NP-complete [T9] |
| $n$ | Open | Open | NP-hard ($n \geq 1$) [T9] |
| $m$ | FPT $2^{O(m^3)}$ [T3] | NP-hard ($m \geq 3$) [T5] | FPT [T10] |
| $k$ | NP-hard ($k \geq 3$) [T1] | NP-hard ($k \geq 6$) [T6] | FPT $2^k$ [T11] |
| $n + m$ | FPT (from $m$) | FPT [T7] | FPT (from $m$) |
| $n + k$ | FPT [C1] | FPT [C2] | FPT (from $k$) |
| $m + k$ | FPT (from $m$) | FPT [T8] | FPT (from $m$ or $k$) |

### Complexity Dichotomy Comparison

| Attribute count $k$ | BAM complexity | Preference order length | Key technique |
|--------------------|---------------|------------------------|---------------|
| $k = 1$ | $O(n)$ linear time | Unrestricted | Direct checking |
| $k = 2$ | $O(n)$ linear time | Unrestricted | Reduction to 2-SAT |
| $k = 3$ | NP-complete | Length 2 only | Reduction from 3-Coloring |
| $k \geq 4$ | NP-complete | Length 2 only | Follows from $k = 3$ |
| $\ell = k + 1$ | Polynomial time | Full preference orders | Structural exploitation |

### Key Findings

- **The complexity dichotomy is remarkably sharp**: A jump in computational complexity occurs between $k = 2$ and $k = 3$, and $k = 3$ is already NP-hard even when preference orders have length only 2 (the shortest meaningful case).
- **The asymmetry among the three variants is surprising**: BAM with Cares is harder than BAM in parameter $m$ (NP-hard for $m = 3$ vs. FPT), while BAM with Has is harder than BAM in parameter $n$ (NP-hard for $n = 1$) but easier in $k$ (FPT vs. NP-hard).
- **An elegant linear-time algorithm exists for $n = 2$**: Via attribute-type decomposition and greedy assignment, any two-voter preference profile can be explained with an optimal number of attributes.

## Highlights & Insights

- **Deep connection between BAM and graph coloring**: The NP-hardness proof reveals that preference explanation is equivalent to graph coloring—each candidate's attribute subset is a "color," and preference constraints encode "adjacent vertices receive different colors." This not only establishes hardness but also suggests that approximation and heuristic methods for graph coloring may be transferable to BAM.
- **A clean complexity dichotomy**: The result that $k \leq 2$ is solvable in linear time while $k \geq 3$ is NP-complete represents the most desirable form of a dichotomy theorem in computational complexity. The key insight is that $k = 2$ reduces to 2-SAT (polynomial), while $k = 3$ can encode 3-Coloring (NP-complete).
- **Counterintuitive phenomenon: more information does not necessarily make the problem easier**: Knowing the cares function (BAM with Cares) actually makes the problem harder under certain parameterizations ($m = 3$ is NP-hard), defying the intuition that "more input information = easier problem." The reason is that a fixed cares function constrains the solution space while also removing flexibility.
- **Transferable ILP modeling technique**: The approach in Theorem 7—grouping attributes by "type" (which voters care about them) and solving via ILP—can be transferred to other combinatorial optimization problems in preference learning.

## Limitations & Future Work

- **Complexity with respect to $n$ remains open**: Whether BAM is FPT in the number of voters $n$ alone has not been resolved, and the authors explicitly list this as an open problem. It is likely the most natural direction for future work.
- **Lack of empirical validation**: As a purely theoretical work, the paper does not validate BAM's fitting ability or attribute count distribution on real preference data, nor does it implement or benchmark the proposed algorithms.
- **Restrictive model**: Only Boolean attributes (present/absent) are considered, without support for attribute weights (e.g., "strongly cares" vs. "slightly cares") or non-Boolean attribute values, which are important in practical applications.
- **Simple preference model**: The assumption that preferences are entirely determined by attribute counts (additive utility functions) ignores interaction effects between attributes (complementarity/substitutability), limiting the model's expressive power.
- **The case $\ell = k$ remains unresolved**: The complexity when preference order length equals exactly $k$ is still open ($\ell = k+1$ is polynomial; $\ell = k-1$ is NP-hard).
- **Approximation algorithms not studied**: When exact solutions are NP-hard, the natural follow-up question—whether a near-optimal BAM can be found in polynomial time—is left entirely unexplored.

## Related Work & Insights

- **vs. Künnemann et al. (2019) on stable matching**: Their work uses a known $k$-BAM to compute stable matchings in $O(4^k \cdot n \cdot (k + \log n))$ time. The present paper addresses the upstream problem of how hard it is to compute a BAM in the first place. Together, the two results imply that if $k$ is small and a BAM can be computed efficiently, the entire preference processing pipeline is efficient.
- **vs. UTA methods (Siskos et al., 2016)**: UTA methods learn additive utility functions over continuous attribute domains, typically via linear programming. BAM is their Boolean special case, but the combinatorial nature makes LP methods inapplicable, necessitating SAT/CSP techniques.
- **vs. single-peaked preferences**: Single-peaked preferences are the most classical restricted preference domain, yet empirical studies suggest that real-world preferences are rarely single-peaked. BAM provides a multi-dimensional restriction on preference domains that may better reflect practical settings.
- **vs. combinatorial voting (Lang & Xia, 2016)**: Combinatorial voting assumes a known attribute model and studies voting rule design. BAM requires recovering the attribute model from preference data, posing the inverse problem.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study of the computational complexity of BAM, filling a theoretical gap; however, the underlying model (Boolean attribute preferences) has a long history.
- **Experimental Thoroughness**: ⭐⭐⭐ — No experiments as a purely theoretical work, but the complexity analysis is comprehensive and the open problems are clearly identified.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, Table 1 provides an at-a-glance summary of results, and proof sketches are well-explained.
- **Value**: ⭐⭐⭐⭐ — Significant theoretical contribution to preference modeling; the complexity dichotomy is elegant, though practical impact awaits future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?](how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[AAAI 2026\] How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](how_to_marginalize_in_causal_structure_learning.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)

</div>

<!-- RELATED:END -->

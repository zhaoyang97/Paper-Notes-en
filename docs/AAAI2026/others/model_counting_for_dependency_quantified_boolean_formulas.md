---
title: >-
  [Paper Note] Model Counting for Dependency Quantified Boolean Formulas
description: >-
  [AAAI 2026][DQBF] This paper presents the first study of the model counting problem for Dependency Quantified Boolean Formulas (DQBF). It proves that #2-DQBF — restricted to only two existential variables — is already #E…
tags:
  - "AAAI 2026"
  - "DQBF"
  - "model counting"
  - "#EXP-completeness"
  - "BDD"
  - "symbolic reachability"
date: 2026-05-08
content_hash: aa645ce718e3d72e
---

# Model Counting for Dependency Quantified Boolean Formulas

**Conference**: AAAI 2026
**arXiv**: [2511.07337](https://arxiv.org/abs/2511.07337)
**Code**: [GitHub](https://github.com/Sat-DQBF/sharp2DQR)
**Area**: Theoretical Computer Science / Formal Methods
**Keywords**: DQBF, model counting, #EXP-completeness, BDD, symbolic reachability

## TL;DR

This paper presents the first study of the model counting problem for Dependency Quantified Boolean Formulas (DQBF). It proves that #2-DQBF — restricted to only two existential variables — is already #EXP-complete, and implements a practical 2-DQBF model counter, sharp2DQR, based on BDD symbolic reachability. The proposed approach significantly outperforms unfolding-based baselines on instances with large dependency sets.

## Background & Motivation

Dependency Quantified Boolean Formulas (DQBF) generalize QBF by explicitly specifying the dependency set of each existential variable on universal variables, rather than relying on a linear quantifier ordering. DQBF has broad applications in hardware verification, program synthesis, and related areas. The satisfiability problem for DQBF is NEXP-complete, and several DQBF solvers have been developed in recent years, with a dedicated DQBF track established in QBF competitions.

Beyond satisfiability, many synthesis and verification tasks require knowing the **number of solutions** — for instance, an unexpectedly large number of Skolem functions may indicate that a specification permits unintended behaviors. Although model counting for QBF and Boolean synthesis has been studied, the model counting problem for DQBF (#DQBF) has never been investigated. The problem is particularly challenging: determining satisfiability alone is already NEXP-complete, and the arbitrary, potentially incomparable dependency sets in DQBF prevent direct application of existing #QBF techniques, which rely on a linear quantifier ordering.

The central mechanism of this paper is to exploit the close correspondence between $k$-DQBF and $k$-SAT — recent work has shown that $k$-DQBF is a compact representation of $k$-SAT — in order to lift classical #SAT complexity results to the DQBF setting and to develop practical counting algorithms based on symbolic reachability.

## Method

### Overall Architecture

The paper is organized into two main parts: (1) on the theoretical side, proving the #EXP-completeness of #2-DQBF and its implications for first-order model counting; (2) on the algorithmic side, proposing a 2-DQBF model counting algorithm based on BDD symbolic reachability.

### Key Designs

1. **Poly-monious Reduction**:

    - Function: Defines a novel polynomial-time reduction to establish #EXP-hardness.
    - Mechanism: Given functions $F$ and $G$, a poly-monious reduction is a polynomial-time Turing machine $M$ that, on input $w$, outputs $t$ strings $v_1, \ldots, v_t$ such that $F(w) = p(G(v_1), \ldots, G(v_t))$ for some polynomial $p$. This notion is strictly stronger than parsimonious reductions but strictly weaker than general polynomial-time Turing reductions.
    - Design Motivation: Functions in #EXP may output doubly exponentially large numbers (requiring exponentially many bits to represent), making standard polynomial-time Turing reductions unsuitable for establishing #EXP-hardness. Poly-monious reductions strike the right balance between parsimonious and Turing reductions.

2. **#EXP-Completeness of #2-DQBF**:

    - Function: Proves the central theorem — model counting for DQBF with only two existential variables is already #EXP-complete.
    - Mechanism: Using projection techniques and Lemma 3, an arbitrary DQBF instance $\Psi$ is reduced in polynomial time to two 2-DQBF instances $\Phi_1, \Phi_2$ satisfying $\#\Psi = \#\Phi_1 - \#\Phi_2$. This critically relies on the observation that the reduction from #SAT to #2-SAT (due to Bannach et al.) is in fact a projection reduction, which can be composed with the compact representation of DQBF unfolding. The explicit reduction runs in $O(k^2|\psi|)$ time.
    - Design Motivation: This mirrors Valiant's classical theorem that #2-SAT is #P-complete despite 2-SAT being polynomial-time solvable, establishing an analogous complexity jump at the DQBF level.

3. **Application to First-Order Model Counting**:

    - Function: Proves that First-Order Model Counting (FOMC) is #EXP-complete even when restricted to domain size 2 and a PSPACE-decidable fragment of first-order logic.
    - Mechanism: The reduction from uniform 2-DQBF to FOMC uses a predicate $U$ to represent Boolean values and a predicate $S$ to encode Skolem functions, showing that $\#\Phi$ equals exactly half the number of models of an FO sentence over domain $\{1,2\}$.
    - Design Motivation: This result explains why scalable first-order model counters have remained elusive despite more than a decade of research.

4. **2-DQBF Model Counting Algorithm**:

    - Function: Designs the practical #2-DQBF counter sharp2DQR.
    - Mechanism: (a) The matrix of a 2-DQBF instance is interpreted as a compact encoding of the implication graph of the unfolded formula; (b) BDD symbolic reachability is used to compute the transitive closure $\varphi_{tr}$ of the implication graph; (c) the graph is decomposed into weakly connected components, each counted independently; (d) within each component, the Skolem functions for $y_1$ are enumerated, and for each such function the problem is reduced to 1-DQBF counting, which can be solved efficiently.
    - Design Motivation: Direct enumeration of Skolem functions is infeasible (the number may be doubly exponential), and unfolding to a propositional formula leads to exponential blowup. The component decomposition idea, analogous to component decomposition in propositional model counting, substantially reduces practical computation.

### Loss & Training

This paper presents theoretical and algorithmic work; no training is involved. Key algorithmic strategies include:
- Using ABC's IC3/reach command for BDD reachability computation
- Using the CUDD package for BDD operations
- Preprocessing for non-supporting variables: Lemma 10 is applied to efficiently compute the number of non-supporting variables without enumeration
- Pruning during candidate Skolem function enumeration using transitive closure information (Eq. 3) to eliminate impossible assignment combinations

## Key Experimental Results

### Main Results

Experiments evaluate sharp2DQR against the baseline Exp+ganak (unfolding + propositional counter) on three benchmark families:

| Instance Type | Dependency Set Size | sharp2DQR | Exp+ganak | Notes |
|--------------|--------------------|-----------|-----------|----|
| PEC_opt (370 instances) | 10–50 | **Significantly better** | Bottleneck at unfolding | Large dependency sets make unfolding prohibitively expensive |
| PEC_small (192 instances) | 3–10 | Slightly worse | Better | BDD operation overhead dominates on small instances |
| 2_colorability | 2–127 bits | **Up to 127 bits** | Up to 12 bits | Exp+ganak cannot handle graphs beyond 12 bits |

### Ablation Study

| Method | Domain Size Limit | Notes |
|--------|------------------|-------|
| sharp2DQR | $2^{127}$ | Can handle more than $2^{2^{64}}$ Skolem functions |
| Exp+ganak (cryptominisat) | 12 bits | Unfolding size grows exponentially |
| Exp+ganak (z3) | 12 bits | Performance similar to cryptominisat |
| WFOMC (FOMC tool) | Domain size 4 | Far below sharp2DQR |

### Key Findings

- sharp2DQR substantially outperforms unfolding-based methods on large dependency sets (10–50 variables), since unfolding size is exponential in the dependency set size.
- On small dependency sets, unfolding-based methods perform better, as BDD operations themselves incur non-trivial overhead.
- On independent set counting instances, Exp+ganak outperforms sharp2DQR, indicating complementary strengths between the two approaches.
- On 2-colorability instances, sharp2DQR handles graphs with 127-bit encodings ($2^{127}$ nodes), while WFOMC is limited to domain size 4.

## Highlights & Insights

- **Strong theoretical contribution**: The paper establishes a complete complexity landscape for DQBF model counting — #2-DQBF is already #EXP-complete, forming a perfect analogy to Valiant's result that #2-SAT is #P-complete.
- **Poly-monious reduction** is a promising new tool that fills the gap between parsimonious and Turing reductions.
- **The new lower bound for FOMC** provides a principled explanation for why scalable first-order model counters have been difficult to build in practice.
- **Component decomposition** is naturally and effectively adapted to the DQBF setting.

## Limitations & Future Work

- The current algorithm is restricted to 2-DQBF; extending it to 3-DQBF and general DQBF is an important next step.
- Performance lags behind unfolding-based methods on small dependency set instances; there is room to optimize BDD operation overhead.
- Unfolding-based methods are superior on structurally specific problems such as independent set counting, suggesting that the scope of applicability remains to be broadened.
- The FOMC application is currently limited to binary relations; generalizing to higher-arity relations is worth exploring.

## Related Work & Insights

This work builds upon DQBF satisfiability solving (solvers such as PEDANT and HQS), symbolic model checking (IC3/PDR algorithms), propositional model counting (Ganak, #SAT), and first-order model counting (WFOMC). The component decomposition idea is inspired by successful techniques in propositional model counting. This paper opens model counting for DQBF as a new research direction, providing theoretical foundations for formal verification and statistical relational learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)
- [\[AAAI 2026\] Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence](tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)

</div>

<!-- RELATED:END -->

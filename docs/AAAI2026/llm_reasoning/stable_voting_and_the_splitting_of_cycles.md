---
title: >-
  [Paper Note] Stable Voting and the Splitting of Cycles
description: >-
  [AAAI 2026][LLM Reasoning][Voting Theory] This paper investigates the conjecture that Simple Stable Voting (SSV)—a recursive voting rule already used in hundreds of real-world elections—always refines Split Cycle (SC). Through mathematical proof (≤5 candidates) and SAT solving (6–7 candidates), the paper establishes that the conjecture holds for ≤6 candidates, is refuted for ≥7 candidates, and generalizes the counterexample to arbitrarily many candidates via a constructive proof.
tags:
  - AAAI 2026
  - LLM Reasoning
  - Voting Theory
  - Condorcet Cycles
  - Split Cycle
  - SAT Solving
  - Computational Social Choice
date: 2026-05-08
content_hash: fbfbb6bced132121
---

# Stable Voting and the Splitting of Cycles

**Conference**: AAAI 2026
**arXiv**: [2512.00616](https://arxiv.org/abs/2512.00616)
**Code**: [https://github.com/chasenorman/ssv-sc](https://github.com/chasenorman/ssv-sc)
**Area**: LLM Reasoning / Social Choice Theory
**Keywords**: Voting Theory, Condorcet Cycles, Split Cycle, SAT Solving, Computational Social Choice

## TL;DR
This paper investigates the conjecture that Simple Stable Voting (SSV)—a recursive voting rule already used in hundreds of real-world elections—always refines Split Cycle (SC). Through mathematical proof (≤5 candidates) and SAT solving (6–7 candidates), the paper establishes that the conjecture holds for ≤6 candidates, is refuted for ≥7 candidates, and generalizes the counterexample to arbitrarily many candidates via a constructive proof.

## Background & Motivation

### Limitations of Prior Work

**Background**: When there are more than two candidates, majority voting may produce cycles (A>B>C>A), which has been a central problem in social choice theory since Condorcet (1785). Split Cycle resolves such cycles by discarding the weakest majority victory in each cycle. Simple Stable Voting (SSV) is a concise recursive rule that has been used in hundreds of real elections on stablevoting.org.

**Core Problem**: Are SSV winners always a subset of SC winners? This conjecture (Holliday & Pacuit 2023a) had resisted both extensive computational search and pen-and-paper proof attempts.

**Key Insight**: The conjecture is encoded as a SAT satisfiability problem, and the CaDiCal SAT solver is employed to systematically search for counterexamples.

**Core Idea**: A combination of mathematical proof and SAT solving establishes the precise boundary of the SSV–SC relationship conjecture: it holds for ≤6 candidates and is refuted for ≥7.

## Method

### Overall Architecture
Split Cycle (SC) identifies cycles in the majority graph, removes the weakest edge (smallest margin of victory) in each cycle, and declares undominated candidates as winners. SSV is defined recursively: for $n$ candidates, the SSV winners are those who, in a specific pairing, remain SSV winners in the sub-election obtained by removing their opponent.

### Key Designs

1. **SAT Encoding**:

    - Variables representing the ordinal margin matrix, along with the definitions of SSV and SC, are encoded as a Boolean satisfiability problem.
    - Symmetry-breaking constraints reduce redundant search: candidates are indexed in a canonical order, and equivalent matrices are pruned.
    - The CaDiCal solver is run on a MacBook Air (M4), producing DRAT proofs as correctness certificates.

2. **Layered Verification Strategy**:

    - ≤5 candidates: classical mathematical induction proof, proceeding by contradiction across three cases (cycle returning to $b$ / returning to $a$ / returning to $a_1$).
    - 6 candidates: SAT verification (0.5 seconds), encoding approximately $\binom{6}{2}^2 \approx 225$ ordinal comparison variables.
    - 7 candidates: SAT finds a counterexample (73 seconds), over a search space of approximately $21.2 \times 10^{21}$ possible matrices.
    - ≥7 candidates: constructive proof generalizing the 7-candidate counterexample by adding "dummy" candidates while preserving the counterexample property.

3. **Counterexample Structure Analysis**:

    - In the 7-candidate counterexample, the SSV winner does not belong to the SC undominated set.
    - Counterexamples account for 25.2% of isomorphism classes, indicating that this is not an extreme edge case but a prevalent phenomenon.

## Key Experimental Results

### Main Results

| # Candidates | Result | Method | Time |
|---|---|---|---|
| ≤5 | ✓ Conjecture holds | Mathematical proof | — |
| 6 | ✓ Conjecture holds | SAT solving | 0.5 s |
| 7 | ✗ Conjecture refuted | SAT counterexample | 73 s |
| ≥7 | ✗ Conjecture refuted | Constructive generalization | — |

### Ablation Study

| Analysis | Result | Notes |
|---|---|---|
| Fraction of counterexample isomorphism classes at $n=7$ | 115/456 (25.2%) | Counterexamples are not rare |
| Single-SC-winner counterexample at $n=7$ | Unsatisfiable | 6h45m search; does not exist |
| Single-SC-winner counterexample at $n=8$ | Exists | A unique SC winner ≠ SSV winner exists for 8 candidates |
| Counterexample model enumeration | >9.8 million | New counterexamples continuously found over 19 days |

### Key Findings
- **Precise boundary established**: ≤6 candidates is the exact threshold, which carries significant theoretical implications.
- **Counterexamples are common at $n=7$**: 25.2% of tournament isomorphism classes contain a counterexample.
- **SAT solving is highly efficient**: A counterexample is found in 73 seconds within a search space of approximately $21.2 \times 10^{21}$ possible matrices.

## Highlights & Insights
- **Innovative application of SAT solving in social choice theory**: Encoding voting-theoretic conjectures as SAT problems is orders of magnitude more efficient than exhaustive search.
- **Integration of theory and practice**: SSV is already deployed in real elections, making a precise understanding of its relationship to SC directly relevant to institutional design.

## Limitations & Future Work
- The practical electoral significance of the counterexample remains unclear—are 7-candidate elections common in practice?
- No formal verification in systems such as Lean is provided; only pen-and-paper proofs and DRAT certificates are given.
- The contribution is purely theoretical, with no direct implications for LLM or AI applications.

## Related Work & Insights
- **vs. the Condorcet family**: Split Cycle and SSV are both Condorcet methods; this paper determines their precise relationship.
- **vs. Ranked Pairs / Beat Path / River**: These methods are all refinements of SC. SSV is equivalent to the more complex Stable Voting (SV) for ≤6 candidates but diverges for ≥7.
- **vs. SAT in combinatorics**: The methodology of proving or refuting mathematical conjectures via SAT has analogues in graph theory and related fields (Brandt & Geist 2016).
- Insight: The SAT encoding developed here is general-purpose and can be used to test properties of any voting method based on a descending ordering of winning edges, making it a broadly applicable tool for computational social choice.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolves an open conjecture posed in 2023 and establishes the precise boundary.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Mathematical proof + SAT solving + model enumeration + multiple tie-breaking variants.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, but the presentation has a high barrier for non-specialist readers.
- Value: ⭐⭐⭐ Theoretically significant contribution, but limited in scope and weakly connected to LLM reasoning.

## Additional Notes
- The methodology and experimental design of this work offer reference value for related fields.
- Future work may validate the generalizability and scalability of the approach across broader settings and larger scales.
- Potential research value exists in combining this work with recent related methods (e.g., intersections with RL/MCTS/multimodal approaches).
- The feasibility of deployment and computational efficiency should be assessed against practical application requirements.
- The generalizability of conclusions may be affected by dataset and metric choices; cross-validation on additional benchmarks is recommended.

## Additional Notes
- The methodology and experimental design of this work offer reference value for related fields.
- Future work may validate the generalizability and scalability of the approach across broader settings and larger scales.
- Potential research value exists in combining this work with recent related methods (e.g., intersections with RL/MCTS/multimodal approaches).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Evaluating, Synthesizing, and Enhancing for Customer Support Conversation](evaluating_synthesizing_and_enhancing_for_customer_support_conversation.md)
- [\[AAAI 2026\] ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation](esg-bench_benchmarking_long-context_esg_reports_for_hallucination_mitigation.md)
- [\[AAAI 2026\] Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[AAAI 2026\] Text-to-Scene with Large Reasoning Models](text-to-scene_with_large_reasoning_models.md)

</div>

<!-- RELATED:END -->

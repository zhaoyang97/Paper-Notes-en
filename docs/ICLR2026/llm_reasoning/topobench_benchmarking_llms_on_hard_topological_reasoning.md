---
title: >-
  [Paper Note] TopoBench: Benchmarking LLMs on Hard Topological Reasoning
description: >-
  [ICLR2026][LLM Reasoning][benchmark] TopoBench is a benchmark comprising 6 categories of topological puzzles × 3 difficulty levels for evaluating the global spatial reasoning capabilities of LLMs. Frontier models solve f…
tags:
  - "ICLR2026"
  - "LLM Reasoning"
  - "benchmark"
  - "topological reasoning"
  - "spatial reasoning"
  - "puzzle"
  - "error diagnosis"
  - "causal intervention"
date: 2026-05-08
content_hash: cf690c779cba1112
---

# TopoBench: Benchmarking LLMs on Hard Topological Reasoning

**Conference**: ICLR2026
**arXiv**: [2603.12133](https://arxiv.org/abs/2603.12133)  
**Code**: [GitHub](https://github.com/mayug/topobench-benchmark)  
**Area**: LLM Reasoning
**Keywords**: benchmark, topological reasoning, spatial reasoning, puzzle, error diagnosis, causal intervention

## TL;DR
TopoBench is a benchmark comprising 6 categories of topological puzzles × 3 difficulty levels for evaluating the global spatial reasoning capabilities of LLMs. Frontier models solve fewer than 24% of hard-tier instances. Causal intervention experiments reveal that error frequency does not equal causal impact — low-frequency constraint forgetting is more destructive than high-frequency repetitive reasoning.

## Background & Motivation
1. LLMs perform strongly on algebraic and symbolic reasoning, yet struggle with tasks requiring the maintenance of global spatial invariants (connectivity, closed loops, symmetry).
2. Existing puzzle and reasoning benchmarks largely test local pattern matching or cell-level operations, without requiring the maintenance of global constraints across a grid.
3. Topological constraints are pervasive in practical applications such as circuit layout, path planning, and molecular structure analysis.
4. Existing evaluations report only accuracy, making it impossible to distinguish whether model failures stem from reasoning itself or from limitations in spatial information extraction and representation.
5. A diagnostic methodology that combines observational error categorization with causal validation is needed.

## Method

### Overall Architecture
TopoBench = topological puzzle benchmark construction + observational error categorization + causal intervention validation + mitigation strategy testing

### TopoBench Benchmark
Six puzzle categories covering diverse topological and geometric constraints, each with three difficulty levels (easy/medium/hard), totaling 900 instances:
- **FlowFree**: Path connectivity — connect color-matched endpoints without crossing paths (5×5 → 12×12).
- **Bridges (Hashiwokakero)**: Network connectivity — connect numbered islands with bridges satisfying degree, crossing, and connectivity constraints.
- **Loopy (Slitherlink)**: Closed-loop constraint — draw a single closed loop along grid edges satisfying the edge-count requirement per cell.
- **Galaxies (Tentai Show)**: Rotational symmetry — partition the grid into regions that are rotationally symmetric about marked centers.
- **Undead**: Reflection and visibility — place monsters satisfying line-of-sight counts through mirrors.
- **Pattern (Nonogram)**: Continuity — fill a binary grid matching row/column run-length clues.

Difficulty is controlled along two axes: (1) grid size (5×5 → 10×10/12×12), and (2) an internal difficulty knob in the generator (reasoning depth without backtracking). Each puzzle is equipped with a puzzle-specific verifier and scored binaryly (correct/incorrect, no partial credit).

### Two-Stage Diagnostic Pipeline
**Stage 1 (Observation)**: An LLM-as-Judge protocol (GPT-5-mini) annotates 750 chain-of-thought reasoning traces, classifying them into 11 error types and computing the frequency of each.

**Stage 2 (Causal Intervention)**: Four error patterns are injected into partial gold-standard solution prefixes (300 instances per condition), and the downstream accuracy change after injection is measured. The causal effect of each error type is quantified by the accuracy difference ($\Delta$ accuracy) before and after injection.

### Four Intervention Error Patterns
- **RR (Repetitive Reasoning)**: Repeating previously attempted reasoning paths without substantive change — observed frequency 33%, but causal effect ≈ 0.
- **PC (Premature Commitment)**: Prematurely locking onto an incorrect direction and continuing — causal effect ~11 pp accuracy drop.
- **STF (State Tracking Failure)**: Inconsistency between the internal board state during reasoning and the actual state.
- **CF (Constraint Forgetting)**: Executing actions that violate rules — appears in only 4% of traces, but causal effect ~11 pp.

### Mitigation Strategies
1. **Cell-aligned grid representation**: An input format in which each row is tokenized into an equal number of tokens; improves accuracy for most puzzle families.
2. **Tool-augmented constraint querying**: An external engine maintains board state and provides structured constraint information (Bridges hard: +10%).
3. **Prompt-level planning guidance**: Prompt variants encouraging planning and backtracking — no significant improvement, indicating that such behaviors cannot be reliably elicited through prompting.

## Key Experimental Results

| Model | Easy Avg | Medium Avg | Hard Avg |
|------|:--------:|:----------:|:--------:|
| GPT-5-mini-high | **0.71** | **0.44** | **0.24** |
| Gemini-3-Flash | 0.60 | 0.35 | 0.09 |
| DeepSeek V3.2 | 0.58 | 0.37 | 0.10 |
| Qwen3-235B | 0.31 | 0.12 | — |
| Qwen3-32B | 0.07 | — | — |

### Ablation Study (Causal Intervention)

| Intervention Error | Observed Frequency | Bridges Δacc | Undead Δacc | Causal Effect |
|---------|:--------:|:----------:|:---------:|:-------:|
| RR (Repetitive Reasoning) | 33% | -0.5pp | +0.3pp | **None** |
| PC (Premature Commitment) | 18% | **-11pp** | **-11pp** | **Strong** |
| CF (Constraint Forgetting) | 4% | **-11pp** | **-9pp** | **Strong** |
| STF (State Tracking Failure) | 12% | -5pp | -6pp | Moderate |

### Key Findings
1. On Galaxies and Loopy at medium/hard difficulty, nearly all models achieve near-zero accuracy; global invariants (rotational symmetry/closed loops) represent the most challenging constraint types.
2. **Error frequency ≠ causal impact**: Constraint forgetting (CF) appears in only 4% of failure traces yet has a causal effect of ~11 pp; repetitive reasoning (RR) appears in 33% of traces but has a causal effect of ≈ 0 — it is a benign byproduct of search.
3. Premature commitment (PC) and constraint forgetting (CF) are the truly fatal error patterns: low in frequency but highly destructive.
4. Tool augmentation: providing structured constraint information (e.g., remaining degree, connectivity status) improves Bridges hard by 10%, whereas providing ASCII grid visual state actually degrades accuracy.
5. **Core conclusion**: The bottleneck lies in extracting structured constraint information from spatial representations, rather than in reasoning about constraints per se.
6. Prompt-level interventions (encouraging planning/backtracking) yield no meaningful improvement across all settings.
7. The strongest model, GPT-5-mini-high, achieves only 24% on the hard tier; the strongest open-source model, DeepSeek V3.2, achieves only 10% — far below the human baseline of 100%.

## Highlights & Insights
- The finding that error frequency does not equal causal impact is highly insightful and challenges a common assumption.
- The causal intervention experiment is rigorously designed, injecting controlled variables into gold-standard solution paths.
- The mitigation strategy experiments distinguish between the bottlenecks of "spatial representation parsing" and "constraint reasoning."
- The six puzzle categories provide comprehensive coverage of diverse topological constraint types.

## Related Work & Insights
- Compared with GridPuzzle (Tyagi et al., 2024), which performs only observational error categorization, TopoBench adds causal intervention validation — decoupling frequency from causality.
- Compared with ARC/BIG-Bench Hard, which tests abstract generalization, TopoBench focuses specifically on topological and geometric constraint maintenance.
- Compared with Sudoku-Bench and similar Latin-square variants, TopoBench requires global invariants (connectivity/closed loops/symmetry) rather than local constraints.
- The finding that prompt-level guidance is ineffective suggests that topological reasoning capabilities require breakthroughs at the architectural or training level.

## Limitations & Future Work
- Causal intervention analysis is conducted only on DeepSeek V3.2 (other models do not expose complete CoT or are subject to API restrictions).
- Although the puzzles are well-controlled, a gap remains between them and real-world engineering tasks (circuit layout/path planning).
- ASCII text input limits the potential of multimodal models (though preliminary multimodal exploration is included).
- The human reference baseline is based on experienced solvers; the difficulty perception of novice humans is not reported.
- Most hard-tier scores are near zero, resulting in insufficient discriminability — a finer-grained difficulty gradient may be needed.

## Related Papers
- Reasoning benchmarks: GSM8K/MATH (algebra), ARC (abstraction), SATBench (logic), Sudoku-Bench (Latin square).
- Error diagnosis: GridPuzzle (Tyagi et al., 2024) observational error categorization; LLM-as-judge (Liu et al., 2023).
- Spatial reasoning: Othello-GPT (Li et al., 2023) state tracking; VGRP-Bench, Enigmata visual grid evaluation.
- Tool augmentation: ReAct (Yao et al., 2023), Toolformer (Schick et al., 2023).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Unique combination of causal intervention and topological reasoning diagnosis)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (9 models × 6 puzzles × 3 difficulty levels + causal experiments + mitigation strategies)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, in-depth analysis)
- Value: ⭐⭐⭐⭐ (Reveals the fundamental bottleneck in LLM spatial reasoning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)
- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)

</div>

<!-- RELATED:END -->

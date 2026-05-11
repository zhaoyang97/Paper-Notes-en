---
title: >-
  [Paper Note] Regular Games – an Automata-Based General Game Playing Language
description: >-
  [AAAI 2026][LLM Evaluation][General Game Playing] This paper introduces Regular Games (RG), a general game playing system centered on nondeterministic finite automata (NFA) for encoding game rules. RG employs a multi-lev…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "General Game Playing"
  - "Game Description Language"
  - "Finite Automata"
  - "Forward Model"
  - "Procedural Content Generation"
date: 2026-05-08
content_hash: 251b0af2da8a14c1
---

# Regular Games – an Automata-Based General Game Playing Language

**Conference**: AAAI 2026
**arXiv**: [2511.10593](https://arxiv.org/abs/2511.10593)
**Code**: [GitHub (RG)](https://github.com/radekmie/rg) / [GitHub (Compiler)](https://github.com/WoojtekP/RGcompiler)
**Area**: General Game Playing / Formal Languages / Artificial Intelligence
**Keywords**: General Game Playing, Game Description Language, Finite Automata, Forward Model, Procedural Content Generation

## TL;DR

This paper introduces Regular Games (RG), a general game playing system centered on nondeterministic finite automata (NFA) for encoding game rules. RG employs a multi-level language architecture (low-level RG, high-level HRG, and domain-specific frameworks) that covers all finite turn-based games — including those with imperfect information and stochasticity — while generating forward models that consistently outperform the previously fastest GGP system, RBG, and typically run 10–20× faster than Ludii.

## Background & Motivation

General Game Playing (GGP) seeks to build agents capable of adapting to and successfully playing any given game, making it a concrete instantiation of general artificial intelligence in the game domain. Existing GGP systems face a fundamental trilemma:

**GDL (Game Description Language)**: Based on logic programming, GDL is elegantly designed but computationally inefficient. The overhead of logical inference makes large games impractical; GDL-II extends support to imperfect information but exacerbates the performance problem.

**Ludii**: Offers thousands of high-level keywords (ludemes), enabling concise and human-friendly game descriptions with imperfect information support. However, its excessive complexity confines it to its own Java ecosystem; while faster than GDL, it still suffers from performance bottlenecks.

**RBG (Regular Boardgames)**: Prioritizes minimalism and efficiency, defining rules via regular expressions compiled to C++, making it the previously fastest GGP language. However, it only supports perfect-information deterministic games, and complex game descriptions tend to be verbose with long compilation times.

The root cause is that expressiveness, usability, and computational efficiency are difficult to achieve simultaneously. RG aims to attain all three — as general as GDL, as convenient as Ludii, and as efficient as RBG.

## Method

### Overall Architecture

RG adopts a multi-level architecture:

- **Low-level language RG**: Centers on NFAs (directed graphs with labeled edges), with a minimal set of primitives (types, variables, constants, and automaton edges), facilitating automated processing (agents, optimization, rule analysis). Covers all finite turn-based games (including imperfect information and stochasticity), achieving expressiveness equivalent to GDL-II and Ludii.
- **High-level language HRG**: Designed for human game designers using declarative and structured programming syntax, maintaining full generality while greatly improving readability. HRG compiles down to low-level RG.
- **Domain-specific frameworks**: Such as the LineGames Python library, which allows defining rules for specific game categories in a few lines of code and generates HRG.
- **Cross-language translation**: Supports automatic translation from RBG → RG and GDL → RG (experimental).

Compiled RG descriptions generate highly optimized C++ inference modules, providing a unified forward model interface (terminal detection, legal move generation, move execution, etc.).

### Key Designs

1. **NFA-based rule description**: Game rules are represented as nondeterministic finite automata $(Q, \delta)$, where $Q$ is the set of nodes and $\delta$ is the transition function. The game state is $(\mathcal{S}, q)$ (semi-state + current node). NFAs are more flexible than regular expressions (used by RBG) and theoretically exponentially more concise. **Design Motivation**: Automaton representations are naturally amenable to optimization (dataflow analysis, constant propagation, path pruning, etc.), and by not embedding concepts such as boards or arithmetic, they operate purely on abstract symbols, preserving maximum flexibility.

2. **Five action types**: Empty (no-op), comparison (equality/inequality), assignment (variable modification), reachability check (complex condition verification, e.g., "does a legal path exist"), and labels (markers constituting moves). Reachability checks are a particularly powerful mechanism — they allow verifying the existence of paths satisfying certain conditions within the automaton without modifying the state, supporting complex queries such as "is there an empty cell on the board" or "does a winning path exist."

3. **Imperfect information and stochasticity**: The `visible` variable controls which parts of a move each player can observe (move masking). The `random` special player selects uniformly at random from legal moves; probability distributions can be controlled via move duplication or move sequences. The `keeper` manages game state and always has exactly one legal move, handling non-player operations.

4. **Optimization pipeline**: Includes redundant label pruning, assignment inlining, constant propagation, and other transformations executed in a fixpoint loop. Dataflow analysis computes game-state knowledge at each node. Automata translated from RBG exhibit 72% fewer nodes, 66% fewer edges, and 21% smaller state size after optimization. HRG games see reductions of 47% in nodes and 41% in edges. Full translation and optimization of most HRG games completes within 100 ms.

5. **IDE and toolchain**: Provides an LSP-enabled code editor (syntax highlighting, autocompletion, diagnostics), automaton visualization, benchmarking tools, and a transformation debugger. RG and Ludii are the only two GGP languages with industrial-grade IDEs.

### Theoretical Properties

- **Universality theorem**: RG can encode all finite turn-based games (including imperfect information and stochasticity with rational probabilities). Simultaneous moves can be modeled via imperfect information, achieving expressiveness equivalent to GDL-II and Ludii.
- **Computational complexity**: With fixed type lengths, determining whether a legal move exists in the initial state is PSPACE-complete; in the general case, it is EXPSPACE-complete.

## Key Experimental Results

### Main Results

Efficiency is measured in Monte Carlo random playouts per second across 30+ games:

| Game | RG (HRG) | RBG | Ludii | RG/RBG Speedup |
|------|----------|-----|-------|----------------|
| Connect Four | 1,297,176 | 914,514 | 55,858 | 1.4× |
| Chess | 1,572 | 995 | 113 | 1.6× |
| Breakthrough | 82,135 | 50,977 | 3,365 | 1.6× |
| Reversi | 28,445 | 19,838 | 1,497 | 1.4× |
| Pentago | 43,875 | 22,553 | — | 1.9× |
| Tic-Tac-Die | 2,708,648 | — | 36,702 | — |
| Fox and Hounds | 444,243 | 331,884 | 14,216 | 1.3× |

### Ablation Study (Optimization Effectiveness)

| Configuration | Node Reduction | Edge Reduction | State Size Reduction |
|---------------|---------------|----------------|----------------------|
| RBG → RG (after optimization) | 72% | 66% | 21% |
| HRG (after optimization) | 47% | 41% | — |

### Key Findings

1. **Consistently outperforms RBG**: For all games implemented in HRG, the RG-generated forward model is faster than RBG (the previously fastest GGP language). The largest speedup occurs on Pentago, where RG benefits from not requiring workarounds for rotation operations as RBG does.
2. **10–20× faster than Ludii**: RG is more than an order of magnitude faster than Ludii on typical games (e.g., Connect Four: 1.3M vs. 56K; Chess: 1.6K vs. 113).
3. **Auto-translated RBG games remain competitive**: Games automatically translated from RBG to RG achieve efficiency comparable to native RBG — and in some cases faster — validating RG's viability as a "GGP assembly language."

## Highlights & Insights

- **Multi-level architecture is well-conceived**: The low level targets machine processing through minimalism, while the high level prioritizes human-friendliness; the two are bridged by compilation. This positioning of RG as a "GGP assembly language" allows it to serve as a unified target for multiple high-level languages.
- **Theoretical justification for NFAs over regular expressions**: Automata are exponentially more concise than regular expressions and naturally amenable to graph-based optimization algorithms.
- **Correctness-first engineering practice**: A validator (type checking, reachability, correctness) is run after each transformation, ensuring that optimizations do not alter game semantics.
- **Cross-language interoperability**: Support for RBG → RG and GDL → RG translation allows RG to serve the broader community without requiring a game library to be built from scratch.

## Limitations & Future Work

- The current game library is limited in scale and falls far short of Ludii's large historical game database; community adoption remains to be established.
- GDL → RG translation is still experimental and has limited coverage (e.g., alternating moves are not detected).
- Deep integration with reinforcement learning training pipelines has not been experimentally demonstrated; whether the efficiency advantage translates to accelerated AI training remains to be validated.
- Real-world showcases for imperfect-information games (e.g., poker) are sparse; the focus is predominantly on board games.
- Compilation to C++ achieves high efficiency but limits direct integration with the Python ecosystem prevalent in RL research.

## Related Work & Insights

- Viewed through the generalization trajectory from AlphaGo → AlphaZero → MuZero, GGP serves as an important testbed for AI generalization. RG's efficiency advantage could provide faster environment simulation for large-scale RL experiments.
- Compared to systems with fixed implementations such as OpenSpiel, RG maintains formal analyzability of rules, supporting rule analysis and procedural content generation.
- The multi-level language architecture offers design insights applicable to DSLs in other domains, such as programmable networks and hardware description languages.
- The dataflow analysis strategies in the optimization pipeline may inspire compilation optimization research for other formal description languages (e.g., planning PDDL).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — NFA as the core game description primitive, multi-level language architecture, and cross-language translation are all original contributions
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Formal definitions are rigorous; universality and complexity theorems are complete; the optimization pipeline demonstrates substantial engineering depth
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Efficiency comparisons across 30+ games are comprehensive, but validation on the AI training side is absent
- **Practicality**: ⭐⭐⭐⭐ — A complete toolchain and IDE are provided, though the community ecosystem is still under development
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](../../NeurIPS2025/llm_evaluation/can_large_language_models_master_complex_card_games.md)
- [\[AAAI 2026\] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)
- [\[AAAI 2026\] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)
- [\[AAAI 2026\] Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)

</div>

<!-- RELATED:END -->

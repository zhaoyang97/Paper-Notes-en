---
title: >-
  [Paper Note] Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification
description: >-
  [NeurIPS 2025 (MATH-AI Workshop)][Reinforcement Learning][Lean 4] This paper presents Kimina Lean Server — a high-performance Lean 4 verification server designed for large-scale reinforcement learning training. By leveraging server-side parallelization and an LRU caching mechanism, it achieves 1.5–2× speedups over existing tools and has been used to train the state-of-the-art theorem proving model Kimina-Prover.
tags:
  - NeurIPS 2025 (MATH-AI Workshop)
  - Reinforcement Learning
  - Lean 4
  - theorem proving
  - formal verification
  - high-performance server
date: 2026-05-08
content_hash: bf0ad3ce1420b9ec
---

# Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification

**Conference**: NeurIPS 2025 (MATH-AI Workshop)  
**arXiv**: [2504.21230](https://arxiv.org/abs/2504.21230)  
**Code**: [GitHub](https://github.com/project-numina/kimina-lean-server)  
**Area**: Reinforcement Learning / Theorem Proving  
**Keywords**: Lean 4, theorem proving, formal verification, reinforcement learning, high-performance server

## TL;DR

This paper presents Kimina Lean Server — a high-performance Lean 4 verification server designed for large-scale reinforcement learning training. By leveraging server-side parallelization and an LRU caching mechanism, it achieves 1.5–2× speedups over existing tools and has been used to train the state-of-the-art theorem proving model Kimina-Prover.

## Background & Motivation

**State of the Field**: Neural theorem proving has advanced rapidly in recent years, with large language models trained via reinforcement learning on Lean 4 achieving significant progress (e.g., DeepSeek-Prover, Kimina-Prover). This training paradigm demands fast, scalable proof verification.

**Limitations of Prior Work**: Existing tools for Lean 4–Python interaction (LeanDojo, Pantograph, leanclient, LeanInteract, etc.) suffer from performance bottlenecks and scalability issues. Most do not support parallelization across CPU cores, and each verification incurs high initialization costs (e.g., loading Mathlib).

**Root Cause**: RL training requires verifying a large number of proofs per second (as reward signals), yet the verification throughput of existing tools is insufficient to support large-scale training pipelines.

**Paper Goals**: To build a high-performance Lean server specifically designed for large-scale RL verification pipelines.

**Starting Point**: Constructing a server-side parallelization and caching layer on top of the official Lean REPL.

**Core Idea**: Maximize CPU utilization and eliminate redundant loading overhead through REPL pool parallelization combined with LRU import caching.

## Method

### Overall Architecture

Kimina Lean Server adopts a client–server architecture:
- **Server**: Exposes verification services via a REST API, manages a pool of Lean REPL processes, and enables parallel verification with import caching.
- **Client**: A lightweight Python package (distributed via PyPI) that submits proof batches and retrieves structured feedback through a single `check` function.

### Key Designs

1. **Server-Side Parallelization**:

    - Maintains a pool of Lean REPL processes, each running in an isolated process.
    - Each Lean REPL is single-threaded and occupies at most one CPU core.
    - One persistent REPL process is assigned per available CPU core, enabling efficient parallelism.
    - Incoming requests are routed to idle REPLs, and responses are returned upon completion.
    - Performance scales nearly linearly with the number of CPU cores (8→32 cores yields ~4× speedup).

2. **LRU Import Caching**:

    - Lean REPL initialization is costly, primarily due to loading large libraries such as Mathlib.
    - Each incoming script is split into a **header** (import statements only) and a **body** (remaining code).
    - The header is used as a key to look up pre-warmed workers in an LRU cache.
    - When a matching worker is found, only the body needs to be verified, reusing the already-loaded context.
    - This mechanism reduces average verification time from 0.099s to 0.051s (**1.94× speedup**).

3. **Data Extraction (Infotree Processing)**:

    - Processes Lean's infotree output to decompose proofs into non-overlapping tactic sequences.
    - Each tactic is annotated with the tactic state (goal state) before and after its application.
    - Supports all Lean tactics, including `have`, `let`, `calc` mode, and `conv` mode.
    - Processing pipeline: extract tactic positions → remove overlaps → extract code segments → handle whitespace/comments/special tactics.
    - Output format is particularly suited for tree-search-based models.

4. **Client API Design**:

    - All interactions are conducted through a single `check` function.
    - Input: a list of Lean scripts.
    - Output: per-script messages (warnings/errors), REPL environment identifiers, elapsed time, and optional infotree.
    - Provides a higher-level `run_benchmark` function that automatically handles data loading and batching.

### Loss & Training

- Compatible with Lean 4 v4.15.0.
- Modular architecture: adaptable to other Lean proof checkers by modifying the process spawning logic.
- Built directly on the official Lean REPL, ensuring compatibility with all REPL-supported Lean versions.

## Key Experimental Results

### Main Results

Verification time comparison on the NuminaMath-LEAN dataset (9,419 valid, sorry-free proofs):

| System | 8 cores | 16 cores | 32 cores | 64 cores |
|--------|---------|----------|----------|----------|
| leanclient | 109:55 | 56:58 | 30:16 | 18:01 |
| LeanInteract | 87:35 | 45:51 | 24:11 | 12:56 |
| **Kimina Lean Server** | **42:40** | **21:48** | **11:33** | **7:56** |

A **1.5–2×** speed advantage is achieved across all core configurations.

### Scalability Results

| CPU Cores | Total Verification Time | Avg. Time per Proof (s) |
|-----------|------------------------|--------------------------|
| 8 | 42:40 | 0.272 |
| 16 | 21:48 | 0.139 |
| 32 | 11:33 | 0.074 |
| 64 | 7:56 | 0.051 |

Near 4× speedup from 8 to 32 cores, demonstrating strong near-linear scalability.

### Ablation Study (Cache Effect)

| Mode | Total Verification Time | Avg. Time per Proof (s) |
|------|------------------------|--------------------------|
| With cache | 7:56 | 0.051 |
| Without cache | 15:28 | 0.099 |

Caching yields a **1.94×** speedup, particularly effective for workflows that frequently reuse the same imports.

### Key Findings

1. Kimina Lean Server **consistently outperforms** the strongest baseline (LeanInteract) across all core configurations.
2. The parallelization design enables **near-linear performance scaling** with core count.
3. LRU caching nearly **halves** the verification overhead.
4. The system has been deployed in the training of Kimina-Prover (achieving state-of-the-art twice on the miniF2F benchmark), validating its robustness in practice.

## Highlights & Insights

- **Engineering-focused yet broadly impactful**: Addresses a critical infrastructure gap in large-scale RL-based Lean verification.
- The LRU cache design is elegant: it exploits the fact that the vast majority of proofs in RL training share the same imports (e.g., Mathlib).
- The header/body split is simple yet effective, targeting caching granularity precisely at the import level.
- The tactic extraction functionality via infotree post-processing is highly valuable for tree-search models (e.g., MCTS).
- Building on the official Lean REPL ensures long-term compatibility and maintainability.

## Limitations & Future Work

1. Currently limited to Lean 4, though the architecture is extensible to other proof assistants.
2. The client exposes a synchronous API, which may require async support at very large scales.
3. The caching mechanism may be less effective in workflows with diverse import patterns.
4. No integration with GPU-accelerated pipelines (e.g., co-scheduling LLM inference and verification on the same machine).
5. Memory footprint grows linearly with REPL pool size, requiring sufficient RAM.

## Related Work & Insights

- **LeanDojo**: Provides a gym-like environment but is not optimized for verification throughput.
- **Pantograph**: Extends LeanREPL with additional features (support for `have`/`let`/`calc` modes) but remains single-threaded.
- **leanclient / LeanInteract**: Support parallelization but with limited performance.
- **ProofWala**: A comprehensive gym environment, not designed for RL training throughput.
- **Insight**: In the AI for Math domain, improvements to infrastructure tooling (i.e., efficient verifiers) directly translate into accelerated model training.

## Rating

- Novelty: ⭐⭐⭐ Core techniques (parallelization + caching) are not entirely novel, but the system design targeting RL verification is distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-core configuration comparisons, cache ablation, and validation in real training.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with abundant code examples.
- Value: ⭐⭐⭐⭐⭐ Open-source and validated in state-of-the-art model training; community value is exceptionally high.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining](graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)
- [\[AAAI 2026\] Formal Verification of Diffusion Auctions](../../AAAI2026/reinforcement_learning/formal_verification_of_diffusion_auctions.md)
- [\[AAAI 2026\] QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](../../AAAI2026/reinforcement_learning/qimeng-kernel_macro-thinking_micro-coding_paradigm_for_llm-based_high-performanc.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](../../ICLR2026/reinforcement_learning/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)

<!-- RELATED:END -->

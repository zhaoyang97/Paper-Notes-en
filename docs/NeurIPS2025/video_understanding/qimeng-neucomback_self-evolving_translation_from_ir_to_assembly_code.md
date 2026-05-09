---
title: >-
  [Paper Note] QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code
description: >-
  [NeurIPS 2025][Video Understanding][Neural Compilation] This paper introduces the NeuComBack benchmark for evaluating neural compilation on IR-to-assembly translation tasks, and proposes a self-evolving prompt optimization method that iteratively improves compilation prompts by learning from LLM self-debugging trajectories. The approach raises correctness from 44% to 64%, with 87.5% of correctly generated programs outperforming clang-O3.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Neural Compilation
  - LLM
  - IR-to-Assembly Translation
  - Self-Evolving Prompt Optimization
  - Compiler
date: 2026-05-08
content_hash: eecff76e21e2ef85
---

# QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code

**Conference**: NeurIPS 2025
**arXiv**: [2511.01183](https://arxiv.org/abs/2511.01183)
**Code**: [https://github.com/](https://github.com/) (not provided)
**Area**: Video Understanding
**Keywords**: Neural Compilation, LLM, IR-to-Assembly Translation, Self-Evolving Prompt Optimization, Compiler

## TL;DR

This paper introduces the NeuComBack benchmark for evaluating neural compilation on IR-to-assembly translation tasks, and proposes a self-evolving prompt optimization method that iteratively improves compilation prompts by learning from LLM self-debugging trajectories. The approach raises correctness from 44% to 64%, with 87.5% of correctly generated programs outperforming clang-O3.

## Background & Motivation

Compilers are indispensable yet extraordinarily complex software systems that demand substantial human expertise to develop and maintain. Given the remarkable performance of LLMs on code-related tasks, **neural compilation**—directly translating high-level languages or intermediate representations (IR) into low-level assembly using LLMs—has emerged as an appealing new paradigm.

**Core Limitations of Prior Work**:

**Lack of benchmarks**: No standardized benchmark exists for evaluating IR-to-assembly compilation, making it impossible to systematically measure and track progress.

**Correctness challenge**: Even the most capable LLMs fall far short of traditional compilers in terms of semantic correctness of generated assembly.

**Optimization challenge**: Generating assembly that outperforms mature compilers (e.g., clang-O3) while maintaining correctness is even more difficult.

**Key Challenge**: Although LLMs possess code understanding and generation capabilities, direct application to assembly generation lacks both evaluation standards and effective capability-enhancement methods.

**Key Insight**:
- Construct a dedicated benchmark, NeuComBack, with two levels: basic compilation (L1) and optimization potential (L2).
- Propose a self-evolving prompt optimization method that enables LLMs to learn from their own debugging experience, iteratively refining prompt strategies for assembly generation.

## Method

### Overall Architecture

The method comprises two phases: **offline prompt learning** and **online inference**. In the offline phase, LLM self-debugging trajectories are collected, insights are extracted, and prompts are iteratively evolved. In the online phase, the evolved prompts guide assembly generation and iterative optimization.

### Key Designs

1. **NeuComBack Benchmark**:

    - **Level 1 (Basic Compilation, 200 tasks)**: Sampled from ExeBench, covering diverse real-world C programs with an emphasis on functional correctness. Programs undergo rigorous filtering for C/C++ standard compliance, selecting those with the longest LLVM IR after compilation.
    - **Level 2 (Optimization Potential, 151 tasks)**: Drawn from the TSVC vectorizing compiler test suite; programs have simple execution paths but complex loop structures, making them suitable for evaluating optimization capability.
    - Evaluation metrics: ACC (functional correctness rate) and ACC+Perf (rate of correct programs that also outperform clang-O3).

2. **Neural Compilation Workflow**:

    - **Initial generation**: Given an IR input, the LLM generates an initial assembly candidate.
    - **Self-debugging**: After generation, correctness is verified via testing; failures trigger iterative self-debugging.
    - **Iterative optimization**: Starting from a correct initial assembly, $T$ rounds of performance optimization are performed.
    - Correctness verification and self-debugging may be applied after each generation or optimization step.

3. **Self-Evolving Prompt Optimization (Core Contribution)**:

    - **Trajectory collection**: The LLM executes compilation tasks and full self-debugging trajectories are collected (covering the complete process from initial erroneous generation → debugging corrections → final correct output).
    - **Insight extraction**: For trajectories that successfully transition from error to correctness, the LLM analyzes error patterns and effective repair strategies.
    - **Prompt evolution**: Extracted insights are integrated into the existing prompt, reviewed and confirmed by the LLM, then applied as an update. One prompt update is performed after each mini-batch of compilation tasks.
    - Key distinction: Unlike general APO methods, this approach learns specifically from **complete self-debugging trajectories**—enabling the LLM to internalize lessons from its own past experience resolving assembly errors.

### Loss & Training

There is no training loss in the conventional sense. Prompt optimization is driven by compilation correctness feedback through an iterative process of 3 epochs with a batch size of 5. DeepSeek-R1 is used as the primary base LLM.

## Key Experimental Results

### Main Results

**Frontier LLM Baseline Performance (NeuComBack-L2, x86_64, 151 cases)**

| Model | ACC (%) | ACC+Perf (%) |
|-------|---------|--------------|
| GPT-4o | 1.99 (3/151) | 0.66 (1/151) |
| O3-Mini | 21.19 (32/151) | 5.30 (8/151) |
| O1 | 19.87 (30/151) | 5.30 (8/151) |
| DeepSeek-V3 | 14.57 (22/151) | 3.31 (5/151) |
| DeepSeek-R1 | 45.70 (69/151) | 21.85 (33/151) |

**Self-Evolving Prompt Optimization Results (DeepSeek-R1, x86_64)**

| Method | L1 Test ACC (%) | L2 Initial ACC (%) | L2 ACC+Perf after Optimization (%) |
|--------|-----------------|--------------------|------------------------------------|
| Baseline Prompt | 50.00 (20/40) | 44.00 (11/25) | 28.00 (7/25) |
| Learned Prompt | **80.00** (32/40) | **64.00** (16/25) | **56.00** (14/25) |

### Ablation Study

| Configuration | ACC (%) | ACC+Perf (%) | Notes |
|---------------|---------|--------------|-------|
| x86_64 baseline → learned prompt | 44%→64% | 28%→56% | +100% programs exceeding O3 |
| aarch64 baseline → learned prompt | 36%→72% | 8%→28% | Effective across architectures |
| L2-learned prompt → L1 | 67.5% vs. L1-specific 80% | — | Transferable across data distributions |
| Reduced self-debugging rounds | 0.9 rounds (baseline) → 0.28 rounds (ours) | — | Learned prompts reduce self-debugging demand |

### Key Findings

- DeepSeek-R1 is the strongest baseline, significantly outperforming GPT-4o (ACC 45.7% vs. 1.99%), underscoring the critical role of reasoning capability.
- The learned prompt achieves a 60% relative improvement in correctness on L1 (50%→80%).
- Among 16 correctly generated x86_64 programs on L2, 14 (87.5%) outperform clang-O3.
- Prompt optimization effects transfer across instruction set architectures (x86_64→aarch64).
- The learned prompts encode multi-level knowledge including formatting rules (`.text` section), syntactic rules (`.L` prefix), and semantic rules (zeroing return registers for void functions).

## Highlights & Insights

- **LLMs' assembly optimization capability is surprisingly strong**: 87.5% of correct programs outperform clang-O3, suggesting that LLMs can discover optimization opportunities unexploited by traditional compilers.
- The design of **learning from self-debugging trajectories** is distinctive—knowledge is distilled not from successes but from the "error → correction" process.
- LLMs can leverage vector instructions (e.g., `cmpps`) to perform vectorization optimizations that traditional compilers do not apply.
- The two-level benchmark design (basic compilation vs. optimization potential) is well-targeted.

## Limitations & Future Work

- Overall functional correctness remains insufficient (64% on L2), leaving a gap before practical deployment.
- Evaluation is limited to the function level; compilation of larger-scale programs is not addressed.
- Heavy reliance on DeepSeek-R1's reasoning capability may result in significantly weaker performance on other models.
- The computational cost of prompt learning is non-trivial, requiring multiple rounds of LLM calls.
- Reliability verification of assembly code in safety-critical scenarios is not considered.

## Related Work & Insights

- Complementary to works such as LLM Compiler (Meta) and SLADE, which focus on pre-training, while this paper focuses on inference-time prompt optimization.
- The self-evolving prompt methodology is transferable to other code translation tasks (e.g., decompilation, cross-language translation).
- NeuComBack can serve as a standard test suite for evaluating the compilation capability of next-generation LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of self-evolving prompts learned from debugging trajectories is original, and the benchmark fills an existing gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-architecture, cross-distribution, transferability, and ablation analyses are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, experimental setup is detailed, and case studies are informative.
- Value: ⭐⭐⭐⭐ Provides both a benchmark and methodology for neural compilation, though practical deployment remains distant.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [\[ICCV 2025\] RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning](../../ICCV2025/video_understanding/rainbowprompt_diversity-enhanced_prompt-evolving_for_continual_learning.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Less is More: Local Intrinsic Dimensions of Contextual Language Models](less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)
- [\[NeurIPS 2025\] Tracking and Understanding Object Transformations](tracking_and_understanding_object_transformations.md)

<!-- RELATED:END -->

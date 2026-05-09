---
title: >-
  [Paper Note] Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency
description: >-
  [NeurIPS 2025][AIGC Detection][Performance constraint synthesis] This paper proposes Wedge, a framework that uses LLMs to synthesize performance-characterizing constraints to guide constraint-aware fuzzing, generating stress-test inputs that expose code performance bottlenecks. It further constructs the PerfForge benchmark, enabling LLM-based code optimizers (e.g., Effi-Learner) to achieve up to 24% additional reduction in CPU instructions.
tags:
  - NeurIPS 2025
  - AIGC Detection
  - Performance constraint synthesis
  - code efficiency evaluation
  - fuzzing
  - LLM code optimization
  - PerfForge
date: 2026-05-08
content_hash: 7000fc3f67bdd52e
---

# Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency

**Conference**: NeurIPS 2025  
**arXiv**: [2505.23471](https://arxiv.org/abs/2505.23471)  
**Code**: [https://github.com/UChiSeclab/perfforge](https://github.com/UChiSeclab/perfforge)  
**Area**: AIGC Detection  
**Keywords**: Performance constraint synthesis, code efficiency evaluation, fuzzing, LLM code optimization, PerfForge  

## TL;DR
This paper proposes Wedge, a framework that uses LLMs to synthesize performance-characterizing constraints to guide constraint-aware fuzzing, generating stress-test inputs that expose code performance bottlenecks. It further constructs the PerfForge benchmark, enabling LLM-based code optimizers (e.g., Effi-Learner) to achieve up to 24% additional reduction in CPU instructions.

## Background & Motivation

**State of the Field**: LLMs have demonstrated strong capabilities in code optimization, yet evaluating their effectiveness lacks test inputs that genuinely expose performance bottlenecks. Existing approaches rely on either random inputs or simplistic large-scale inputs, which fail to selectively trigger performance issues along specific code paths.

**Limitations of Prior Work**: LLMs struggle to directly reason about "which inputs trigger which performance bottlenecks," as this requires complex mapping from program inputs to local execution paths. Existing performance test generation methods (EvalPerf, TG-prompt) offer limited effectiveness.

**Root Cause**: Finding inputs that make unoptimized code extremely slow while optimized code runs fast requires an understanding of the internal performance characteristics of the code, making such input construction inherently difficult.

**Starting Point**: The problem is decomposed — LLMs are well-suited for reasoning about local code behavior (e.g., "under what conditions does this loop iterate the most"), while fuzzing excels at searching for inputs satisfying given constraints.

**Core Idea**: LLM synthesizes performance constraints → constraints are instrumented into the program → constraint-aware fuzzing automatically searches for stress inputs that satisfy the constraints.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Contrastive execution profiling** — comparing per-statement execution counts between fast and slow inputs to identify bottlenecks; (2) **Performance constraint synthesis** — a two-phase LLM reasoning process (natural language description → checker code) that generates and instruments constraints; (3) **Constraint-guided fuzzing** — a modified AFL++ using LLM-generated constraint-aware mutators to search for stress inputs.

### Key Designs

1. **Contrastive Execution Profiling**:

    - Function: Mines contrastive input pairs (similar inputs with large execution cost differences) from existing correctness tests.
    - Mechanism: Runs the program to collect CPU instruction counts per input; uses match rate and Jaccard similarity to measure input similarity and execution cost ratio to measure performance difference; selects the top-ranked $(i_{slow}, i_{fast})$ pair; then collects per-line hit count differences across the two executions.
    - Design Motivation: Contrastive execution provides richer diagnostic signals than single-run profiling; high input similarity ensures that observed differences genuinely reflect performance characteristics rather than input content.

2. **Performance Constraint Synthesis (Two-Phase)**:

    - Phase 1 (Natural Language Reasoning): The LLM compares hit counts between fast and slow executions, identifies hotspot code regions, reasons about "what conditions cause these lines to execute more frequently," and outputs a set of NL-described constraints $\mathbb{C}$.
    - Phase 2 (Code Implementation): The LLM translates natural language constraints into executable checker code (e.g., `all(l[i] > l[i+1] for i in range(len(l)-1))`), which is instrumented into the program as new branches.
    - Design Motivation: The two-phase decomposition avoids the difficulty of directly prompting LLMs to generate stress inputs — LLMs excel at local behavior reasoning but struggle with whole-program input-to-behavior mapping.

3. **Constraint-Aware Mutator**:

    - Function: Replaces AFL++'s default bit-flip mutator with a custom mutator that understands input syntax and performance constraints.
    - Mechanism: The LLM generates a Python mutator function based on the problem description (input grammar), performance constraints $\mathbb{C}$, and contrastive input pairs. An iterative generate-and-fix process ensures robustness.
    - AFL++'s coverage-guided mechanism naturally treats instrumented checker branches as coverage targets — covering a new checker branch is equivalent to finding an input that satisfies the performance constraint.
    - Removing the constraint-aware mutator causes a 3.9× drop in instruction count (the most critical ablation component).

## Key Experimental Results

### Main Results

| Method | Avg. Instruction Count (×10⁸) | Win Rate | Avg. Slowdown |
|--------|-------------------------------|----------|---------------|
| **Wedge** | **5.96** | **60%** | **363×** |
| TG-prompt | 3.87 (↓1.5×) | 12% | 275× |
| PerfFuzz | 3.29 (↓1.8×) | 11% | 149× |
| EvalPerf_slow | 3.23 (↓1.8×) | 8% | 146× |

### Code Optimization Impact (Effi-Learner + GPT-4o)

| Input Source | Execution Time Reduction | CPU Instruction Reduction |
|--------------|--------------------------|---------------------------|
| None | 15.66% | 31.39% |
| CC_default | 22.57% | 39.89% |
| **PerfForge** | **26.47%** | **49.31%** |

### Ablation Study

| Configuration | Avg. Instruction Count (×10⁸) | vs. Full |
|---------------|-------------------------------|----------|
| Full Wedge | 5.96 | — |
| WedgeNoInstr (w/o checker instrumentation) | 4.02 | ↓1.5× |
| WedgeDefaultMut (w/ AFL++ default mutator) | 1.54 | ↓3.9× |
| AFL++ (no constraints, no instrumentation) | 1.49 | ↓4.0× |

### Fair Evaluation of Pie Code Optimization
PerfForge-based evaluation reveals that Pie's actual speedup is higher than its claimed performance (based on correctness tests): instruction count speedup improves by 24%–149%, wall-clock time speedup by 5%–27%, and 7%–48% more programs are found to be meaningfully optimized.

### Key Findings
- Constraint-satisfying inputs are 38.6× slower than constraint-violating inputs, and are significantly slower on 92.84% of programs — validating the performance-characterizing ability of the synthesized constraints.
- The constraint-aware mutator contributes the most (removing it causes a 3.9× drop); checker instrumentation is the second most important (removing it causes a 1.5× drop).
- Wedge shows greater advantage on small inputs (<1KB), achieving 3× slowdown over baselines, demonstrating that the performance pressure stems from implementation bottlenecks rather than input length.
- Evaluation dataset: 154 problems from CodeContests, 33,020 C++ programs, with 1 hour of fuzzing per solution.

## Highlights & Insights
- **Elegant division of labor between LLMs and fuzzing**: LLMs handle "understanding" (constraint reasoning), while fuzzing handles "search" (generating inputs satisfying constraints), leveraging the strengths of each.
- **PerfForge as an accelerator for code optimization**: Beyond serving as an evaluation tool, PerfForge can drive feedback to LLM-based optimizers to discover better optimizations — forming a closed evaluation→improvement loop.

## Limitations & Future Work
- Relies on LLMs to generate correct constraint checker code, which may fail for complex programs.
- Validation is limited to competitive programming problems (CodeContests); applicability to industrial codebases and repository-level code remains untested.
- The 1-hour fuzzing cost per solution is substantial; practical deployment requires efficiency considerations.
- Constraint quality is entirely dependent on the LLM's code comprehension ability, which may degrade on unseen algorithmic patterns.

## Related Work & Insights
- **vs. EvalPerf**: EvalPerf controls input size via a scale parameter, essentially performing length-based stress testing; Wedge generates structured constraints targeting specific code implementations, achieving greater precision.
- **vs. PerfFuzz**: PerfFuzz uses edge hit counts to guide fuzzing but lacks explicit performance constraints, ultimately still generating length-stress inputs.
- **vs. TG-prompt**: Directly prompts LLMs to generate stress inputs, requiring end-to-end reasoning over whole-program behavior — a task that proves too difficult for current LLMs.
- This work provides important reference value for evaluating LLM agent code optimization capabilities — high-quality evaluation inputs can drive substantially better optimization outcomes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of performance constraint synthesis and constraint-guided fuzzing is entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-baseline, multi-ablation, and closed-loop validation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and systematic method description.
- Value: ⭐⭐⭐⭐⭐ Direct value for both code efficiency evaluation and LLM-based code optimization.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[NeurIPS 2025\] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)
- [\[AAAI 2026\] Optimized Algorithms for Text Clustering with LLM-Generated Constraints](../../AAAI2026/aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)
- [\[ICLR 2026\] CLARC: C/C++ Benchmark for Robust Code Search](../../ICLR2026/aigc_detection/clarc_cc_benchmark_for_robust_code_search.md)

<!-- RELATED:END -->

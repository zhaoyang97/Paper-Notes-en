---
title: >-
  [Paper Note] Small Language Models as Compiler Experts: Auto-Parallelization for Heterogeneous Systems
description: >-
  [NeurIPS 2025 (ML for Systems Workshop)][LLM Evaluation][small language models] This work systematically evaluates three language models with fewer than 1.5B parameters (gemma3, llama3.2…
tags:
  - "NeurIPS 2025 (ML for Systems Workshop)"
  - "LLM Evaluation"
  - "small language models"
  - "auto-parallelization"
  - "compiler optimization"
  - "heterogeneous systems"
  - "inference strategies"
date: 2026-05-08
content_hash: d5324aee20ce1197
---

# Small Language Models as Compiler Experts: Auto-Parallelization for Heterogeneous Systems

**Conference**: NeurIPS 2025 (ML for Systems Workshop)  
**arXiv**: [2512.19250](https://arxiv.org/abs/2512.19250)  
**Code**: Not released  
**Area**: LLM Evaluation  
**Keywords**: small language models, auto-parallelization, compiler optimization, heterogeneous systems, inference strategies  

## TL;DR

This work systematically evaluates three language models with fewer than 1.5B parameters (gemma3, llama3.2, qwen2.5) on compiler auto-parallelization tasks. Using six inference strategies across 11 real-world kernels, the approach achieves an average speedup of 6.81x and a peak speedup of 43.25x, demonstrating that small models can serve as powerful compiler optimization reasoning engines.

## Background & Motivation

Following the end of Moore's Law, performance gains increasingly rely on heterogeneous computing (mixed CPU/GPU architectures), yet software toolchains have lagged behind:

**Traditional auto-parallelizing compilers** (LLVM Polly, GCC, etc.) depend on rigid heuristic rules and struggle to capture complex dependencies in real-world code.

**Large model solutions are prohibitively costly**: most AI for Systems research focuses on large proprietary models whose latency and cost are incompatible with compiler integration.

**Core Problem**: Can small, efficient LLMs provide the sophisticated reasoning required for complex compiler tasks?

This paper answers affirmatively: through a carefully designed inference framework, models with fewer than 1.5B parameters can match or surpass traditional compiler performance.

## Method

### Overall Architecture

A three-stage pipeline:

1. **Code Analyzer**: statically analyzes input C/C++ code.
2. **LLM Reasoner**: formulates a parallelization plan based on the analysis results.
3. **Parallelization Generator**: implements the plan as parallel code.

### Key Design 1: LLM-Guided Dependency Reasoning

The LLM acts as a semantic reasoning engine, receiving abstract representations of loop nests, memory access patterns, and control flow, and explicitly reasoning about:

- **Loop-carried dependencies**: dependencies that prevent safe parallel execution.
- **Reduction patterns**: operations that can be safely parallelized via reduction clauses.
- **Privatizable variables**: variables whose scope must be restricted to avoid data races.
- **Target-specific execution strategies**: CPU thread-level parallelism vs. GPU kernel decomposition.

The LLM produces a structured parallelization plan that is validated by a static analyzer before code generation. Unsafe transformations are automatically rejected via sanitizers and regression tests.

### Key Design 2: Six Inference Strategies

- **Tree of Thoughts (ToT)**: explores multiple reasoning paths; yields the best results.
- **Chain of Thought (CoT)**: step-by-step chain reasoning.
- **ReAct**: alternates between reasoning and action.
- **Few-shot**: leverages a small number of demonstrations.
- **Step-by-Step**: sequential execution of reasoning steps.
- **Zero-shot**: direct inference without examples.

### Key Design 3: Safety Guarantees

All LLM-generated parallel code must pass multiple validation checks: regression tests, sanitizer-based detection of data races and memory issues, and cross-compiler compatibility testing (GCC 11+, Clang 14+, ICC 2021+, MSVC 2019+). Unsafe transformations are automatically rejected.

### Evaluation Benchmark

Eleven real-world computational kernels spanning three domains: scientific computing (FFT, Jacobi, MatMul), graph algorithms (BFS, PageRank, Dijkstra), and ML kernels (Conv2D, Attention, Pooling).

## Key Experimental Results

### Main Results

A total of 376 evaluations covering a comprehensive comparison across models, strategies, and baselines.

**Model Performance Comparison (averaged over all strategies)**:

| Model | Avg. Speedup | Best Speedup | Analysis Quality | Response Time |
|-------|:-----------:|:------------:|:----------------:|:-------------:|
| gemma3:1b | 6.2x | 38.7x | 0.78 | 12.3s |
| llama3.2:1b | 6.8x | 41.2x | 0.82 | 15.7s |
| **qwen2.5:1.5b** | **7.2x** | **43.25x** | **0.85** | 18.9s |

**Inference Strategy Comparison (averaged over all models)**:

| Strategy | Avg. Speedup | Success Rate | Quality Score |
|----------|:-----------:|:------------:|:-------------:|
| Tree of Thoughts | **7.1x** | **88%** | **0.84** |
| Chain of Thought | 6.9x | 85% | 0.81 |
| ReAct | 6.7x | 83% | 0.79 |
| Zero-shot | 5.8x | 78% | 0.72 |

**Comparison with Advanced Compiler Baselines**:

| Method | Avg. Speedup | Best Performance | GPU Support |
|--------|:-----------:|-----------------|:-----------:|
| LLM (qwen2.5+ToT) | 7.1x | Conv 43.25x | Yes |
| LLVM Polly | 5.8x | MatMul 8.2x | No |
| TVM | 7.4x | Conv 11.2x | Yes |
| Triton | **8.9x** | Attn 13.7x | Yes |

### Ablation Study

**Scalability (matrix multiplication)**: LLM speedup grows from 4.2x at size 1K to 13.1x at size 16K, consistently outperforming LLVM Polly.

**Correctness verification**: LLM-ToT achieves a validation rate of 88%, race-freedom of 91%, and memory safety of 94%—lower than LLVM Polly (95%/97%/98%) but still reliable.

### Key Findings

1. **Inference strategy matters more than model scale**: ToT consistently yields the best results.
2. **LLMs generalize better than domain-specific tools**: more consistent performance across domains.
3. **Small models are sufficient**: average speedup surpasses both LLVM Polly and GCC.
4. **Generated code scales well**: performance improves steadily with input size and core count.

## Highlights & Insights

1. **Small model potential is underestimated**: 1B-scale models perform strongly on structured compiler tasks.
2. **The inference framework is the key lever**: ToT yields a 22% speedup improvement over Zero-shot.
3. **Safety guarantees are well-designed**: incorrect parallelizations are automatically intercepted.
4. **Compiler integration is feasible**: latency of 12–19s is acceptable for offline optimization.
5. **Strong cross-platform compatibility**: compilation success rates of 98% on GCC and 96% on Clang.

## Limitations & Future Work

1. **High compilation latency**: 18.9s far exceeds the 2–4s of traditional compilers, precluding JIT use.
2. **Correctness gap**: 88% falls short of the 95% achieved by traditional compilers.
3. **Dependence on prompt engineering**: effectiveness is highly sensitive to code abstraction and prompt design.
4. **Limited evaluation scope**: only 11 kernels are evaluated.
5. **Workshop paper**: methodological details are insufficiently elaborated.

## Related Work & Insights

- **LLVM Polly**: polyhedral model auto-parallelization baseline.
- **TVM / Triton**: domain-specific compilation optimization.
- **Tree of Thoughts** (Yao et al., 2023): multi-path reasoning strategy.
- Future directions: verifier-in-the-loop feedback, multi-hardware backends, multi-language extension.

## Rating

⭐⭐⭐⭐ (3.5/5)

- **Novelty** ⭐⭐⭐⭐: using small models as compiler experts is an interesting and practical direction.
- **Experimental Thoroughness** ⭐⭐⭐⭐: 376 evaluations provide broad coverage.
- **Methodological Depth** ⭐⭐⭐: limited by workshop paper length constraints.
- **Value** ⭐⭐⭐⭐: provides a clear pathway for integrating compilers with LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](../../ACL2026/llm_evaluation/retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[NeurIPS 2025\] On the Entropy Calibration of Language Models](on_the_entropy_calibration_of_language_models.md)
- [\[ACL 2026\] CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems](../../ACL2026/llm_evaluation/clarity_a_framework_and_benchmark_for_conversational_language_ambiguity_and_unan.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)

</div>

<!-- RELATED:END -->

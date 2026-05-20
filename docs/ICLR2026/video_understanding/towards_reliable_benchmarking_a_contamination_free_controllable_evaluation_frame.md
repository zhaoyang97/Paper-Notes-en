---
title: >-
  [Paper Note] FuncBenchGen: A Contamination-Free Controllable Evaluation Framework for Reliable Benchmarking
description: >-
  [ICLR 2026][Video Understanding][Tool-augmented LLM] This paper proposes FuncBenchGen, a framework that models multi-step function calling as a DAG traversal problem…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "Tool-augmented LLM"
  - "multi-step function calling"
  - "benchmark"
  - "data contamination"
  - "DAG traversal"
date: 2026-05-08
content_hash: 9b48a653cc924ed5
---

# FuncBenchGen: A Contamination-Free Controllable Evaluation Framework for Reliable Benchmarking

**Conference**: ICLR 2026
**arXiv**: [2509.26553](https://arxiv.org/abs/2509.26553)  
**Area**: Video Understanding
**Keywords**: Tool-augmented LLM, multi-step function calling, benchmark, data contamination, DAG traversal

## TL;DR

This paper proposes FuncBenchGen, a framework that models multi-step function calling as a DAG traversal problem, enabling contamination-free and finely controllable evaluation of LLM tool-use capabilities. The framework further reveals critical failure modes of reasoning models under long call chains and connected irrelevant functions.

## Background & Motivation

Existing benchmarks for tool-augmented language models (TaLMs) suffer from two core issues:

**Data contamination risk**: QA pairs in existing benchmarks (e.g., API-Bank, BFCLv4, ToolBench) may be leaked through pretraining data or test-time web search, rendering evaluation results unreliable.

**Uncontrollable task complexity**: Existing benchmarks lack fine-grained control over task difficulty, making it impossible to systematically analyze which factors most significantly affect model performance.

| Benchmark | Contamination-Free | Function Set Size Control | Dependency Depth Control | Distractor Type Control |
|---|---|---|---|---|
| API-Bank | ✗ | ✗ | ✗ | ✗ |
| BFCLv4 | ✗ | ✓ | ✗ | ✗ |
| ToolBench | ✗ | ✓ | ✗ | ✗ |
| **FuncBenchGen** | **✓** | **✓** | **✓** | **✓** |

## Method

### Overall Architecture

FuncBenchGen formalizes multi-step function calling as a **Directed Acyclic Graph (DAG) traversal problem**. Given a function set $\mathcal{F}=\{f_1, f_2, \ldots, f_n\}$, an input variable set $\mathcal{V}_{input}$, and a target variable $v_T$, the LLM must determine the value of $v_T$ by iteratively executing a sequence of function calls.

### Key Designs

**1. Graph structure generation**: Accepts four control parameters:
- $n^{\text{core}}$: number of core nodes (functions required to solve the task)
- $d$: dependency depth
- $n^{\text{conn}}$: number of connected irrelevant nodes (CIN, sharing type-compatible variables with core nodes)
- $n^{\text{dis}}$: number of disconnected irrelevant nodes (DIN, with no connections to core nodes)

**2. Function schema creation**: Each DAG node is converted into a function definition comprising a randomly generated function name, typed input/output parameters, and a natural language description. Functions are linked via semantic type and subtype matching.

**3. Deterministic execution**: Each variable is assigned a three-digit random integer value. A function returns the correct output only when all input values are exactly correct; otherwise it returns a random incorrect value, simulating the silent failure behavior of real-world APIs.

### Mitigation Strategy

To address the most prevalent failure mode (use of unknown/incorrect values), the paper proposes a simple **variable value restatement strategy**: upon each function return, the response includes not only the output value but also a list of all currently known variable values.

## Key Experimental Results

### Main Results: Success Rate Under Varying Core Node Counts

| Model | 5 Core Nodes | 10 Core Nodes | 20 Core Nodes |
|---|---|---|---|
| GPT-5 | 72.5% | 38.2% | 15.0% |
| Gemini-2.5-Pro | 46.5% | 14.4% | 6.0% |
| GPT-5-mini | 16.0% | 7.6% | 4.2% |
| Qwen3 | 11.0% | 8.2% | 3.8% |
| GPT-4.1 | 12.0% | 2.2% | 0.2% |

### Failure Type Analysis

| Failure Type | GPT-5 | Gemini-2.5-Pro | Qwen3 | GPT-4.1 |
|---|---|---|---|---|
| Non-existent function | 0.0% | 2.4% | 0.0% | 0.0% |
| Wrong number of input arguments | 0.0% | 0.2% | 0.1% | 0.0% |
| Use of unknown values | 79.6% | 69.1% | 74.0% | 73.2% |
| Use of incorrect values | 20.4% | 28.3% | 25.8% | 26.8% |

### Effect of Dependency Depth

- GPT-5 achieves close to 90% success rate at depth 1 (star structure), dropping to below 30% at depths 4–8.
- Path structures (depth 8–9) show marginal improvement over moderately branched structures (depth 5–7), suggesting that serialized call chains with fewer branches are easier to handle.
- Larger thinking budgets (medium vs. minimal) substantially improve performance in complex scenarios.

### Key Findings

1. **Reasoning models substantially outperform general-purpose models**: GPT-5 achieves 72.5% at 5 core nodes, while GPT-4.1 reaches only 12.0%.
2. **Performance degrades sharply with sequence length**: GPT-5 drops from 72.5% (5 nodes) to 15.0% (20 nodes).
3. **Connected irrelevant nodes (CIN) are the most harmful**: Shared type-compatible variables make it difficult for models to distinguish relevant from irrelevant functions.
4. **The mitigation strategy is highly effective**: Variable restatement improves GPT-5's success rate from 62.5% to 81.3%.
5. **GPT-5 exhibits low call efficiency**: Even when successful, it makes approximately 10% more redundant function calls.
6. **Sufficient reasoning budget is critical**: Under minimal thinking budget, GPT-5's success rate falls below 20% in the presence of distractor functions.

## Highlights & Insights

1. **Elegant formalization**: Abstracting tool use as DAG traversal enables orthogonal decomposition of evaluation dimensions.
2. **Insightful failure analysis**: The study reveals that the primary bottleneck across all models is **state tracking** rather than syntactic comprehension — 79.6% of GPT-5 errors stem from the use of unknown variable values.
3. **Simple yet effective mitigation**: Merely restating known variable values (without providing new information) substantially improves performance, indicating that working memory is the core bottleneck in multi-step tool use.
4. **Warning for the MCP ecosystem**: Even disconnected distractor functions severely degrade GPT-5 performance (<10%) when the function set grows to 40, suggesting current LLMs are not yet ready to handle large-scale MCP servers.
5. **Failure mode differences reveal model characteristics**: When failing, GPT-5 tends to retry repeatedly (making more function calls), whereas Gemini-2.5-Flash tends to give up (making fewer calls).

## Limitations & Future Work

1. A gap exists between synthetic functions and real-world APIs, where function semantics are considerably more complex.
2. Only DAG structures are considered; more complex control flows such as conditional logic and loops are not covered.
3. Each function is fixed to a single output variable; multi-output functions are not supported.
4. The capabilities of open-source small models on this task are not evaluated.
5. Functions are connected via type matching, leaving the evaluation of natural language semantic reasoning underexplored.
6. Model recovery and retry behavior following failed calls is not examined.

## Rating ⭐⭐⭐⭐

This is a systematic and analytically rigorous evaluation framework. Its core contribution lies in revealing the state-tracking bottleneck in LLM multi-step tool use, offering important guidance for the design of agent systems. The DAG-based abstraction is elegant, and the mitigation strategy, though simple, yields deep insight. Limitations include a remaining gap between synthetic tasks and real-world scenarios, and a mismatch between the stated area of video understanding and the paper's actual focus on LLM agent evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Ouroboros of Benchmarking: Reasoning Evaluation in an Era of Saturation](../../NeurIPS2025/video_understanding/the_ouroboros_of_benchmarking_reasoning_evaluation_in_an_era_of_saturation.md)
- [\[ACL 2026\] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](../../ACL2026/video_understanding/vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)
- [\[ACL 2026\] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](../../ACL2026/video_understanding/rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)
- [\[ACL 2026\] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](../../ACL2026/video_understanding/gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)

</div>

<!-- RELATED:END -->

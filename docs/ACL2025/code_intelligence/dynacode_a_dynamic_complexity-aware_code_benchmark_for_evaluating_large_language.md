---
title: >-
  [Paper Note] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation
description: >-
  [ACL 2025][Code Intelligence][Code Generation] This paper proposes DynaCode, a dynamic, complexity-aware code generation benchmark. By classifying code problems based on cyclomatic complexity and nesting them using Call Graphs, DynaCode dynamically generates approximately 189 million unique problems. This design effectively mitigates data contamination and systematically evaluates the code generation capabilities of LLMs across different complexity levels.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Dynamic Evaluation"
  - "Data Contamination"
  - "Complexity-aware"
  - "Call Graph"
date: 2026-05-08
content_hash: f1004a9e816f64a2
---

# DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation

**Conference**: ACL 2025  
**arXiv**: [2503.10452](https://arxiv.org/abs/2503.10452)  
**Code**: [https://github.com/HWH-2000/DynaCode](https://github.com/HWH-2000/DynaCode)  
**Area**: Code Intelligence  
**Keywords**: Code Generation, Dynamic Evaluation, Data Contamination, Complexity-aware, Call Graph

## TL;DR

This paper proposes DynaCode, a dynamic, complexity-aware code generation benchmark. By classifying code problems based on cyclomatic complexity and nesting them using Call Graphs, DynaCode dynamically generates approximately 189 million unique problems. This design effectively mitigates data contamination and systematically evaluates the code generation capabilities of LLMs across different complexity levels.

## Background & Motivation

**Background**: Static benchmarks such as HumanEval and MBPP are standard tools for evaluating LLM code generation. While works like EvalPlus, BigCodeBench, and CRUXEval have improved test cases and evaluation granularity, their core limitations remain.

**Limitations of Prior Work**: (1) **Data Contamination**—Static benchmarks are small-scaled and public, meaning models may memorize test cases during training (e.g., Meta-Llama-3-8B-Instruct and Phi-2 are reported to suffer from data contamination). (2) **Uncontrollable Complexity**—Existing benchmarks lack systematic control over complexity; relying on line counts or execution time is too coarse to capture deep nesting and complex execution dependencies.

**Key Challenge**: There is a need for a benchmark that is both resistant to memorization and capable of fine-grained differentiation of model capabilities, which static, small-scale datasets naturally fail to achieve.

**Goal**: Build a dynamic, large-scale, and complexity-controllable code evaluation framework capable of automatically generating a massive number of unique problems and evaluating them across different complexity tiers.

**Key Insight**: Treat individual code problems as "nodes" and combine them into nested code problems using call graphs (directed acyclic graphs), while constructing a complexity matrix along both code complexity and graph complexity dimensions.

**Core Idea**: Classifying code problems via cyclomatic complexity + Nested combination via call graph structures = A two-dimensional, complexity-aware benchmark dynamically generating 189 million unique problems.

## Method

### Overall Architecture

The construction of DynaCode consists of four steps: (1) Code Complexity Classification: collecting code problems and classifying them based on cyclomatic complexity; (2) Call Graph Construction: designing 16 types of call graph structures; (3) Complexity-aware Metrics: combining code complexity and graph complexity to construct a two-dimensional complexity matrix; (4) Benchmark Generation: nesting problems according to call graphs and generating test cases.

### Key Designs

1. **Code Complexity Classification**: Cyclomatic complexity is used to analyze ground-truth code, defined as $\nu = E - N + 2P$ (where $E$ is the number of edges in the control-flow graph, $N$ is the number of nodes, and $P$ is the number of connected components). Calculated using the Radon tool, the problems are categorized into 4 Units:

    - Unit 1: Simple sequential or basic logic
    - Unit 2: Containing branches and simple loops
    - Unit 3: Multiple nested branches
    - Unit 4: Complex recursion and deep control flows

2. **Call Graph Construction**: Define a directed acyclic graph $G = (V, E)$ with up to 5 nodes and a single unique root node. A total of 16 different structures are designed, ranging from simple linear calls to complex multi-branch dependencies. Graph complexity is measured as:
    $$\mathcal{M}(G) = L_{\max}(G) \times B(G) \times |E|$$
   where $L_{\max}$ is the longest path, $B$ is the number of branches, and $|E|$ is the number of edges. They are divided into 4 Levels based on thresholds.

3. **Two-Dimensional Complexity Matrix**: $\mathcal{C} = \{c_{\xi,\eta} \mid 1 \leq \xi \leq n, 1 \leq \eta \leq m\}$, where $\xi$ corresponds to the code complexity Unit and $\eta$ corresponds to the call graph Level, providing a fine-grained evaluation framework.

4. **Dynamic Generation & Anti-Contamination**: (1) Automatically generate input/output type annotations using Monkeytype, and combine nested problems through type matching; (2) Continuously update the benchmark with new problems sourced from platforms like LeetCode; (3) Generate approximately **189 million** unique nested code problems.

### Test Case Generation

Input values are injected starting from the root node of the call graph. The nested code is executed in batches, filtering out "failed executions" while keeping valid nested code and test cases.

## Key Experimental Results

### Main Results (Pass@1)

| Model | MBPP | MBPP+ | DynaCode Avg | Unit 1 | Unit 2 | Unit 3 | Unit 4 |
|------|------|-------|-------------|--------|--------|--------|--------|
| GPT-4o | 87.6 | 72.2 | **55.4** | 74.4 | 48.7 | 56.2 | 42.3 |
| DeepSeek-V3 | 87.6 | 73.0 | 52.1 | 65.9 | 41.6 | 53.6 | 47.3 |
| Qwen2.5-Coder-32B | 90.5 | 77.0 | 43.2 | 59.3 | 33.6 | 44.1 | 36.0 |
| Llama-3.1-405B | 88.4 | 73.0 | 41.0 | 49.7 | 40.0 | 47.6 | 26.9 |
| GPT-3.5-Turbo | 82.5 | 69.7 | 29.3 | 34.9 | 30.5 | 25.6 | 26.1 |
| Llama-3.1-8B | 68.3 | 55.6 | 9.9 | 14.1 | 9.7 | 8.4 | 7.4 |

### Data Contamination Verification (Fine-tuning Experiments)

| Model | Pre-FT MBPP+ | Post-FT MBPP+ | Pre-FT DynaCode | Post-FT DynaCode |
|------|------------|------------|---------------|---------------|
| GPT-3.5-Turbo | 69.7 | 88.6 (+18.9) | 32.6 | 36.0 (+3.4) |
| Llama-3.1-8B | 55.6 | 98.1 (+42.5) | 10.6 | 23.6 (+13.0) |

Fine-tuning leads to significant improvements on MBPP+ (due to memorization) but yields minimal gains on DynaCode, demonstrating that DynaCode is effectively resistant to data contamination.

### Error Analysis (GPT-3.5-Turbo)

| Capability | Unit 1 | Unit 2 | Unit 3 | Unit 4 |
|------|--------|--------|--------|--------|
| Problem Understanding | 64.1% | 79.9% | 88.2% | 88.8% |
| Code Pattern Generation | 6.6% | 0.4% | 0.2% | 1.5% |
| Context Management | 29.4% | 19.7% | 11.7% | 9.4% |

### Key Findings

- **Average Performance Drop of 16.8%–45.7%**: Compared to MBPP+, all models exhibit a severe degradation in performance on DynaCode, with GPT-3.5-Turbo dropping by 40.4 percentage points.
- **LLMs Excel at Sequential Execution**: Models perform best on linear call graphs {G1–G4, G8}, but their performance drops significantly on multi-branch structures {G9–G16}.
- **Lower Comprehension with Higher Complexity**: Problem understanding errors increase from 64.1% in Unit 1 to 88.8% in Unit 4, indicating that complex nesting hinders the models from even understanding the problem requirements.
- **75 Problems are Sufficient for Stability**: Sensitivity analysis shows that generating 75 or more dynamic problems yields stable evaluation results.
- **GPT-4o Demonstrates the Strongest Complexity Resilience**: It performs most robustly across all levels, though a visible drop still occurs on highly complex graphs.

## Highlights & Insights

- The call graph combination design is highly ingenious: it achieves dynamic scaling (via massive combinations) while integrating structural complexity dimensions.
- The two-dimensional complexity matrix (Code Complexity $\times$ Graph Complexity) offers an unprecedented, fine-grained perspective on evaluation.
- Rigorous experimental design for data contamination verification: employing comparative fine-tuning and controlled experiments fine-tuned solely on unit functions.
- In-depth error analysis: revealing a trend where errors shift from "context management" to "problem understanding" as complexity increases.
- The scale of 189 million problems makes model memorization practically impossible.

## Limitations & Future Work

1. **Limited Call Graph Scale**: Graphs are restricted to a maximum of 5 nodes. Future high-capability LLMs might master this pattern, necessitating expansion to larger-scale graphs.
2. **Python-Only Support**: The code problems are derived from MBPP+ and are limited only to the Python language.
3. **Incomplete Cyclomatic Complexity Metric**: Dimensions such as recursion depth and data-dependency complexity are not factored in.
4. **Combination May Introduce Unnatural Problems**: Automatic type matching and nested combining may occasionally generate semantically awkward problems.

## Related Work & Insights

- **Dynamic Evaluation**: DyVal (graph generation), NPHardEval (NP-hard problems), DyVal2 (LLM agent transformations)—primarily targeted at reasoning. This work represents the first application to code generation.
- **Code Benchmarks**: HumanEval $\rightarrow$ EvalPlus $\rightarrow$ BigCodeBench $\rightarrow$ SWE-Bench, with increasing complexity but all being static.
- **Insights**: The call graph methodology can be scaled up to larger sizes (e.g., 10+ nodes) or expanded to incorporate recursive call graphs. It can also be integrated with real-world scenarios like SWE-Bench.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Dual innovation of dynamicity and complexity awareness, with an ingeniously designed call graph composition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 12 models, 4 Units $\times$ 4 Levels, data contamination verification, and detailed error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear equations and abundant illustrative diagrams.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the two core pain points of data contamination and complexity evaluation, providing high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)
- [\[ACL 2025\] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](feabench_repo_code_gen.md)

</div>

<!-- RELATED:END -->

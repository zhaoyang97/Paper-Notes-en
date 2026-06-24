---
title: >-
  [Paper Note] MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios
description: >-
  [ACL 2025][Code Intelligence][Code Debugging] This work introduces MLDebugging—the first comprehensive benchmark tailored for **multi-library Python code debugging**. It spans 126 Python libraries and 7 bug categories (incorporating 1,175 samples), systematically evaluating the capabilities of mainstream open-source and closed-source LLMs under multi-library debugging scenarios, finding that current LLMs still have substantial room for improvement on this task.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Debugging"
  - "Multi-Library Interaction"
  - "LLM Evaluation"
  - "Benchmark"
  - "Python"
date: 2026-05-08
content_hash: 19b7f49f8bbbb11f
---

# MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios

**Conference**: ACL 2025  
**arXiv**: [2506.13824](https://arxiv.org/abs/2506.13824)  
**Code**: [Yes (GitHub)](https://github.com/hjyTsuki/MLDebugging)  
**Area**: NLP / Code Intelligence / Software Engineering  
**Keywords**: Code Debugging, Multi-Library Interaction, LLM Evaluation, Benchmark, Python

## TL;DR

This work introduces MLDebugging—the first comprehensive benchmark tailored for **multi-library Python code debugging**. It spans 126 Python libraries and 7 bug categories (incorporating 1,175 samples), systematically evaluating the capabilities of mainstream open-source and closed-source LLMs under multi-library debugging scenarios, finding that current LLMs still have substantial room for improvement on this task.

## Background & Motivation

Code debugging is a critical task in software engineering. Although LLMs have achieved remarkable progress in code debugging, existing research and benchmarks almost exclusively focus on **library-free or single-library** settings:

- **DebugBench**: Sourced from LeetCode algorithm problems, without library interactions.
- **xCodeEval / MdEval**: Multi-lingual but lack multi-library scenarios.
- **QuickBugs**: Derived from competitive programming, also omitting libraries.

In real-world software development, collaborating across multiple libraries is the norm. Debugging across multiple libraries introduces two unique challenges: (1) **comprehending multiple libraries to locate bugs**; (2) **leveraging knowledge of multiple libraries to repair bugs**. For instance, a bug involving data flow between pandas and numpy requires the model to understand the APIs of both libraries and variable type compatibility.

This motivated the construction of the MLDebugging benchmark to fill the blank in multi-library code debugging evaluation.

## Method

### Overall Architecture

Benchmark construction pipeline: source code collection $\rightarrow$ LLM annotation and debugging $\rightarrow$ bug category balancing $\rightarrow$ quality control.

### Key Designs

1. **Source Code Collection**:

    - Retreive Python programming tasks involving 2+ libraries from BigCodeBench.
    - Generate 1,038 multi-library code snippets using GPT-4o.
    - Automatically identify 609 buggy codes by executing test cases.

2. **7 Bug Classification Taxonomy** (from three perspectives):

    - **Variable Transmission Perspective**: Type Mismatch (TM) / Data Transfer Issues (DTI)
    - **Library Function Parameter Perspective**: Function Parameter Errors (FPE) / Parameter Configuration Errors (PCE) / Function Misuse (FM)
    - **Functional Understanding Perspective**: Requirement Misunderstanding (RM) / Import Errors (IE)

3. **LLM-assisted Annotation and Debugging**:

    - Employ three LLMs (GPT-4o, DeepSeek-V3, Claude-3.5-sonnet) to classify bugs and generate repair codes.
    - Retry failed repairs up to 5 times (inspired by test-time scaling).

4. **Bug Category Balancing**:

    - Capture multi-library code structures using AST.
    - Samples are proportionally drawn from minority classes to generate new bugs, targeting around 200 samples per class.
    - Successfully injected 566 additional bugs.

5. **Quality Control**:

    - Cross-checked by 4 experienced programmers.
    - Rectified 119 bug descriptions and 340 classification errors, manually repaired 185 samples, and removed 356 unreasonable samples.

### Data Distribution Validation

By comparing the text embedding distribution with real-world bug data from StackOverflow, MLDebugging is verified to be closer to real-world bug distributions than DebugBench (cosine similarity of 0.731 vs 0.660).

## Key Experimental Results

### Main Results: Debugging Pass Rates of LLMs at Different Scales (%)

| Category | Qwen2.5-7B | Qwen2.5-C-7B | Llama3.1-7B | Qwen2.5-72B | DS-V3 | GPT-4 |
|------|-----------|-------------|------------|------------|-------|-------|
| TM | 47.6 | 40.0 | 39.7 | 52.9 | 60.0 | 55.3 |
| DTI | 36.1 | 33.8 | 30.5 | 47.2 | 52.8 | 49.1 |
| PFE | 48.4 | 48.8 | 43.2 | 62.9 | 67.0 | 67.1 |
| PCE | 57.6 | 58.0 | 49.8 | 70.4 | 76.3 | 70.4 |
| FM | 38.2 | 40.4 | 38.8 | 53.8 | 56.2 | 53.0 |
| RM | 12.6 | 7.0 | 5.6 | 16.1 | 23.8 | 21.0 |
| IE | 26.1 | 8.7 | 13.0 | 26.1 | 34.8 | 30.4 |
| **AVG** | 42.7 | 40.6 | 36.7 | 53.7 | **58.7** | 55.6 |

### Ablation Study: Correlation Analysis

| Variable | Correlation Coefficient | P-value |
|------|---------|------|
| Code Lines | -0.0071 | 0.9654 |
| Number of Libraries | -0.2113 | 0.1906 |
| **Library Popularity** | **0.4094** | **0.0087** |

### Key Findings

1. **All LLMs face challenges on MLDebugging**: The highest pass rate is only 58.7% (DeepSeek-V3), which is far below their performance on simple coding tasks.

2. **Diminishing marginal returns with scale**: Performance improves significantly from 7B to 32B, but flattens or even declines from 32B to 72B. This suggests that multi-library debugging cannot be resolved solely by scaling up the model size.

3. **Vast capability disparities across different bug types**:
    - Function-level bugs (TM, DTI, PFE, PCE, FM) achieve pass rates of 30-76%.
    - Library-level reasoning bugs (RM, IE) only obtain pass rates of 5-34%, showing a gap of nearly 20%.
    - **Requirement Misunderstanding (RM) is the most challenging category**, peaking at only 25.2% (Claude).

4. **Library popularity is the most dominant factor affecting debugging difficulty** (correlation coefficient 0.41, $p = 0.0087$), whereas code length and library count are not statistically significant. LLMs master libraries more commonly found in training data better.

5. **CoT prompting significantly enhances debugging performance**, but **distillation-based reasoning models (DeepSeek-R1-Distill) exhibit a decline**, indicating that pure SFT distillation is insufficient to reinforce this capacity.

6. **Both test cases and runtime error messages are indispensable**: Providing both forms of feedback concurrently yields the best and most robust results.

## Highlights & Insights

- **A benchmark filling the gap**: The first evaluation dataset specializing in multi-library code debugging, offering broad coverage across 126 libraries.
- **7-class bug taxonomy**: Systematically characterizes multi-library bugs across three dimensions: variable transmission, function parameters, and functional understanding.
- **Distribution validation**: Confirms the realism of the benchmark by comparing embedding distributions with real StackOverflow data.
- **In-depth model behavior analysis**: Analyzes LLM capabilities by library usage scenarios (general algorithms, data processing, network communication, etc.) and library popularity.
- **Discovered limitations of distilled reasoning models**: This counter-intuitive finding offers constructive insights for CoT distillation research.

## Limitations & Future Work

- The data is mostly generated automatically by models. Although manually verified, gaps still exist compared to real bugs; more real-world data could be introduced in the future.
- The evaluation workflow requires configuring numerous external dependencies and complex environments, which is time-consuming.
- Covers only Python without extending to multi-library scenes in other programming languages.
- The bug injection mechanism could introduce systematic biases (e.g., models leaning towards generating specific types of errors).
- Agentic debugging methods (such as multi-round interactions, tool calls) have not been explored.

## Related Work & Insights

- **DebugBench** (Tian et al., 2024): The first LLM debugging dataset, based on LeetCode.
- **xCodeEval** (Khan et al., 2024): Multi-lingual multi-task code evaluation.
- **MdEval** (Liu et al., 2024b): Multi-lingual debugging across 18 languages.
- **BigCodeBench** (Zhuo et al., 2024): A multi-library code generation benchmark, which serves as the data source for this work.
- The concept of test-time scaling inspired the retry strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Multi-library debugging scenarios were completely unstudied systematically beforehand; the taxonomy is of high value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covered 20+ models ranging from 7B to 72B and closed-source models. Evaluated multi-dimensionally by category, scenario, and library popularity, alongside thorough CoT and distillation ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured with rich tables/charts, featuring a transparent quality control workflow.
- **Value**: ⭐⭐⭐⭐ — Offers a more realistic benchmark for evaluating code LLMs, unveiling distinct limitations of current models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](../../ACL2026/code_intelligence/precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)
- [\[ACL 2025\] LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](benchmarking_long-context_language_models_on_long_code_understanding.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)

</div>

<!-- RELATED:END -->

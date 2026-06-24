---
title: >-
  [Paper Note] CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation
description: >-
  [ACL 2025][Code Intelligence][Code Benchmarks] This paper introduces CoCo-Bench (Comprehensive Code Benchmark), a comprehensive code benchmark covering four dimensions: code understanding, code generation, code modification, and code review. It supports multiple programming languages and difficulty levels, ensures data quality through rigorous manual review, and reveals the unbalanced performance of existing LLMs in coding capabilities.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Benchmarks"
  - "Multi-task Evaluation"
  - "Code Understanding and Generation"
  - "Code Modification and Review"
  - "LLM Coding Capability"
date: 2026-05-08
content_hash: b2fe2ef3ef0cc585
---

# CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation

**Conference**: ACL 2025  
**arXiv**: [2504.20673](https://arxiv.org/abs/2504.20673)  
**Code**: None  
**Area**: Code Intelligence / LLM Evaluation  
**Keywords**: Code Benchmarks, Multi-task Evaluation, Code Understanding and Generation, Code Modification and Review, LLM Coding Capability

## TL;DR

This paper introduces CoCo-Bench (Comprehensive Code Benchmark), a comprehensive code benchmark covering four dimensions: code understanding, code generation, code modification, and code review. It supports multiple programming languages and difficulty levels, ensures data quality through rigorous manual review, and reveals the unbalanced performance of existing LLMs in coding capabilities.

## Background & Motivation

**Background**: LLMs are playing an increasingly crucial role in software engineering, and code generation (such as Copilot) has become a daily tool for developers. Various benchmarks like HumanEval, MBPP, and CodeContests have emerged to evaluate and compare the coding capabilities of different models.

**Limitations of Prior Work**: (1) Most existing benchmarks focus on a single task—HumanEval and MBPP only evaluate code generation, while CodeXGLUE, though containing multiple tasks, does not offer comprehensive coverage; (2) There is a lack of a comprehensive evaluation framework that reflects the full spectrum of real-world development scenarios—in actual development, programmers not only write code but also read, modify, and review it, which existing benchmarks fail to cover fully; (3) The data quality of some benchmarks is uneven and lacks rigorous human validation.

**Key Challenge**: The "fragmentation" of LLM coding capability evaluation—each benchmark only reveals a single facet of a model, and the varying settings (programming languages, difficulties, evaluation methods) across benchmarks make it difficult to obtain a holistic profile of a model's coding capabilities.

**Goal**: To design a unified and comprehensive code benchmark that simultaneously covers four core dimensions: code understanding, generation, modification, and review, supports multiple languages and difficulty levels, and ensures that data undergoes rigorous human validation.

**Key Insight**: Starting from the core needs of real software developers, code-related tasks are abstracted into four dimensions: understanding existing code, writing new code, modifying/maintaining code, and reviewing/inspecting code quality, with targeted sub-tasks designed for each dimension.

**Core Idea**: Building a four-dimensional (understanding, generation, modification, review) code benchmark, CoCo-Bench, to reveal the strengths and weaknesses of models across different dimensions of coding capability within a unified evaluation framework.

## Method

### Overall Architecture

The construction of CoCo-Bench involves four stages: (1) defining the four core evaluation dimensions and their respective sub-tasks; (2) extensively collecting and constructing evaluation data covering multiple programming languages (e.g., Python, Java, C++, JavaScript) and various difficulty levels; (3) establishing a rigorous human review process to ensure data quality; and (4) designing appropriate evaluation metrics tailored to each sub-task.

### Key Designs

1. **Four-Dimensional Evaluation Framework**:

    - Function: Comprehensively covers the core capabilities of coding work.
    - Mechanism: Coding capability is divided into four orthogonal dimensions: (a) Code Understanding: reading code and answering questions about functions, logic, complexity, etc.; (b) Code Generation: writing code based on requirement descriptions; (c) Code Modification: bug fixing, refactoring, or feature addition on existing code; (d) Code Review: identifying issues in code and proposing improvement suggestions.
    - Design Motivation: Existing benchmarks over-focus on code generation, neglecting other equally important dimensions of capability. In real-world development, programmers spend even more time reading and modifying code than writing new code.

2. **Multi-Language and Multi-Difficulty Design**:

    - Function: Ensures the coverage and discriminative power of the evaluation.
    - Mechanism: Tasks in each dimension contain samples from multiple mainstream programming languages and are classified into three difficulty levels: easy, medium, and hard. Different difficulties are distinguished by code complexity (LOC, cyclomatic complexity, etc.) and task complexity (required domain knowledge, reasoning steps, etc.).
    - Design Motivation: Different models may exhibit widely varying performance across languages and difficulties. Multi-language design avoids bias toward any single language, while multi-difficulty ensures the ability to discriminate among models of different performance levels.

3. **Rigorous Human Review Process**:

    - Function: Guarantees the correctness and high quality of the benchmark data.
    - Mechanism: All evaluation data undergoes multiple rounds of human review: first, professional developers verify the correctness and compilability of code samples; then, the clarity and lack of ambiguity of question/task descriptions are validated; finally, the correctness and uniqueness of reference answers are verified (for tasks with unique solutions).
    - Design Motivation: Low-quality benchmark data can lead to erroneous evaluation conclusions. Early benchmarks such as MBPP were criticized for data noise issues, which CoCo-Bench avoids through strict reviews.

### Loss & Training

CoCo-Bench is an evaluation benchmark and does not involve model training. Evaluation metrics are selected based on the sub-task types: Pass@k for code generation; accuracy/F1 for code understanding; edit distance and functional correctness for code modification; and F1 and coverage for code review.

## Key Experimental Results

### Main Results

| Model | Code Understanding | Code Generation | Code Modification | Code Review | Overall |
|------|---------|---------|---------|---------|------|
| GPT-4o | 82.3 | 78.5 | 71.2 | 74.8 | 76.7 |
| Claude 3.5 | 80.1 | 76.8 | 69.5 | 73.2 | 74.9 |
| DeepSeek-Coder-V2 | 78.6 | 80.2 | 65.3 | 68.5 | 73.2 |
| CodeLlama-34B | 65.2 | 62.8 | 52.1 | 55.3 | 58.9 |
| Qwen2.5-Coder | 76.3 | 75.5 | 66.8 | 70.1 | 72.2 |
| StarCoder2-15B | 60.8 | 65.3 | 48.5 | 50.2 | 56.2 |

### Ablation Study

| Dimension × Difficulty | Easy | Medium | Hard | Difficulty Gap |
|-----------|------|------|------|---------|
| Code Understanding | 88.5 | 78.2 | 62.3 | 26.2 |
| Code Generation | 85.2 | 72.1 | 55.8 | 29.4 |
| Code Modification | 78.3 | 65.0 | 48.2 | 30.1 |
| Code Review | 80.1 | 68.5 | 52.0 | 28.1 |

### Key Findings

- Code modification is the dimension where all models perform the worst—even the strongest, GPT-4o, achieves only 71.2%, indicating that room for improvement remains in the ability of existing LLMs to understand code context and perform precise modifications.
- Proficiency in code generation does not equate to comprehensive coding capability—DeepSeek-Coder-V2 leads in generation (80.2) but falls behind significantly in modification and review, demonstrating that the "capable of writing but not modifying" issue is prevalent.
- The difficulty gradient design effectively differentiates model capabilities—performance on hard tasks drops by 26–30 percentage points, with the gap between large and small models being more pronounced on hard tasks.
- Rankings on CoCo-Bench are highly correlated with single-dimension benchmarks like HumanEval, yet CoCo-Bench reveals richer comparative information regarding diverse capabilities.

## Highlights & Insights

- **Systematicness of Four-Dimensional Coverage**: The four-dimensional design of understanding $\rightarrow$ generation $\rightarrow$ modification $\rightarrow$ review completely corresponds to the core activities of the software development lifecycle. Designing benchmarks based on practical needs offers greater practical value than purely tech-driven alternatives.
- **Discovery of "Capable of Writing but Not Modifying"**: Experiments reveal an important imbalance in LLM coding capabilities—generation is significantly stronger than modification. This provides direct guidance for model improvement: more training resources should be invested in code modification and maintenance capabilities.
- **Multi-Dimensional Comparative Rankings**: CoCo-Bench provides a richer view of model comparison than the HumanEval leaderboard, helping users select the most suitable model for specific needs.

## Limitations & Future Work

- The scale of data construction and human review may be limited, potentially affecting the statistical reliability of the evaluation.
- Evaluating the code review dimension remains challenging—quality review feedback can be phrased in multiple ways, making automated evaluation difficult to capture fully.
- Not all programming languages are covered, especially those with rapid growth in recent years such as Rust and Go.
- Future work can incorporate more complex evaluation scenarios, such as collaborative programming and long-context code understanding (e.g., repository level).

## Related Work & Insights

- **vs HumanEval / MBPP**: Single-dimension benchmarks focusing strictly on code generation; CoCo-Bench provides a four-dimensional comprehensive view, serving as a powerful complement to these benchmarks.
- **vs CodeXGLUE (Lu et al., 2021)**: Even though CodeXGLUE is also a multi-task benchmark, it leans toward academic tasks (such as code clone detection, defect detection, etc.), whereas CoCo-Bench aligns closer with actual developer needs.
- **vs SWE-Bench**: SWE-Bench focuses on the specific task of bug fixing within a complex testing environment; CoCo-Bench has a broader coverage, although the depth of each individual task might not match that of SWE-Bench.
- **vs LiveCodeBench**: LiveCodeBench focuses on continuous updates to prevent data contamination; CoCo-Bench emphasizes coverage and multi-dimensional evaluation.

## Rating

- Novelty: ⭐⭐⭐ The four-dimensional code evaluation approach is logical but does not represent a massive breakthrough, as similar multi-task code benchmarks have been proposed before.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple mainstream models were evaluated, providing multi-dimensional comparisons and difficulty analysis with human-validated data.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, benchmark design principles are fully elaborated, and the comparative analysis is deep.
- Value: ⭐⭐⭐⭐ This provides a more comprehensive tool for evaluating code LLMs, and findings like "capable of writing but not modifying" offer valuable guidance for model development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)
- [\[ACL 2025\] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](feabench_repo_code_gen.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)

</div>

<!-- RELATED:END -->

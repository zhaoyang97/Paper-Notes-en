---
title: >-
  [Paper Note] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms
description: >-
  [ACL 2026][LLM Evaluation][Unit Test Generation] This work proposes MultiFileTest, the first multi-file level LLM unit test generation benchmark. It covers 20 projects each for Python, Java…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Unit Test Generation"
  - "Multi-file Benchmark"
  - "Cross-file Dependency"
  - "Error Fixing"
  - "Code Quality"
date: 2026-05-08
content_hash: b499baf2a5c2abb6
---

# MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms

**Conference**: ACL 2026  
**arXiv**: [2502.06556](https://arxiv.org/abs/2502.06556)  
**Code**: [GitHub](https://github.com/MultiFileTest)  
**Area**: LLM Evaluation  
**Keywords**: Unit Test Generation, Multi-file Benchmark, Cross-file Dependency, Error Fixing, Code Quality

## TL;DR
This work proposes MultiFileTest, the first multi-file level LLM unit test generation benchmark. It covers 20 projects each for Python, Java, and JavaScript. The study evaluates 11 frontier LLMs and analyzes the impact of manual and self-fixing mechanisms on test quality, revealing that even the strongest models exhibit significant basic executability errors.

## Background & Motivation

**Background**: LLM-driven unit test generation has become a vital use case for AI-assisted coding, significantly improving test readability and generation efficiency. Existing benchmarks mainly evaluate test generation capabilities for function-level or class-level (single-file) code.

**Limitations of Prior Work**: (1) Functions in real-world projects interact across files with complex dependencies, yet existing benchmarks ignore the challenges of multi-file level test generation; (2) DevBench, the only benchmark involving multi-file tests, contains only 16 projects and is designed for broad coverage rather than in-depth evaluation, lacking systematic analysis of cross-file dependencies and errors; (3) Numerous basic errors (inexecutability, cascading failures) in LLM-generated tests hinder the evaluation of higher-level capabilities (correctness, coverage).

**Key Challenge**: The core difficulty of multi-file test generation lies not in the test logic itself, but in accurately understanding cross-file dependencies and correctly setting up the test environment—which happens to be a weak point in LLM reasoning.

**Goal**: (1) Construct a high-quality multi-file test benchmark; (2) Systematically evaluate the performance of frontier LLMs on this task; (3) Analyze error types and evaluate the effectiveness of fixing mechanisms.

**Key Insight**: By re-evaluating after manually fixing basic errors, this study distinguishes between "lack of basic capability" and "lack of advanced capability," revealing the true potential differences between models.

**Core Idea**: Evaluate across three scenarios—original generation (Scenario 1), after manual fixing (Scenario 2), and after LLM self-fixing (Scenario 3). Ranking changes before and after error fixing reveal the essential differences between models.

## Method

### Overall Architecture
MultiFileTest contains 60 curated GitHub projects (20 each for Python, Java, and JavaScript), with each project having 2-15 files, $<1600$ lines of code, and cross-file dependencies. The evaluation workflow: LLM receives full project code + test generation prompt → original tests are extracted → evaluation of executability/correctness/coverage → re-evaluation after manual fixing of basic errors → re-evaluation after LLM self-fixing.

### Key Designs

1.  **Benchmark Dataset Construction**:
    *   Function: Provide high-quality test scenarios with guaranteed cross-file dependencies.
    *   Mechanism: Projects are filtered from GitHub based on three criteria: appropriate size (2-15 files, $<1600$ lines), inter-file dependencies, and high star/fork counts. For oversized projects, self-contained sub-projects are extracted and dependency paths are adjusted. All projects undergo syntax validation and multi-line statement merging.
    *   Design Motivation: Restricting project size within the LLM context window ensures fair comparison, while mandatory cross-file dependencies ensure that multi-file reasoning is a required attribute.

2.  **Three-Scenario Evaluation Protocol**:
    *   Function: Distinguish between original capability, potential after fixing, and self-fixing capability of LLMs.
    *   Mechanism: Scenario 1 evaluates original generation quality; Scenario 2 involves manual fixing of executability and cascading errors by CS PhDs (averaging 2-6 lines of changes per project) to reveal true potential; Scenario 3 provides error messages and conversation history to the LLM for self-fixing.
    *   Design Motivation: Executability errors (e.g., missing imports) are essentially simple issues that can cause correctness and coverage to drop to zero, masking the models' actual logic design capabilities.

3.  **Error Taxonomy**:
    *   Function: Systematically analyze error types in LLM test generation.
    *   Mechanism: Distinguishes between executability errors (entire test suite fails to run, e.g., `ModuleNotFoundError`) and cascading errors (a single root cause leads to multiple test failures, e.g., a missing NumPy import failing multiple tests simultaneously).
    *   Design Motivation: Distinguishing between "total inexecutability" and "individual test failure" is crucial for understanding LLM error patterns.

### Loss & Training
This work is an evaluation study and does not involve model training. Zero-shot prompting is used with temperature set to 0.

## Key Experimental Results

### Main Results (Python, Original Generation Scenario 1)

| Model | Correctness (CR) | Executability (ER) | Line Cov (LC) | Branch Cov (BC) |
| :--- | :--- | :--- | :--- | :--- |
| Gemini-3.0-Pro | 77% | 85% | 76% | 73% |
| Claude-3.5-Sonnet | 64% | 70% | 51% | 47% |
| GPT-o1 | 60% | 65% | 56% | 54% |
| GPT-5-mini | 53% | 60% | 51% | 50% |
| GPT-4-Turbo | 47% | 65% | 40% | 36% |

### Cross-language Comparison

| Language | Best Model | Best CR | Description |
| :--- | :--- | :--- | :--- |
| Python | Gemini-3.0-Pro | 77% | Relatively the easiest |
| Java | Gemini-3.0-Pro | 62% | Strict syntax increases difficulty |
| JavaScript | GPT-o1 | Highest | Best model varies by language |

### Key Findings
*   Model rankings changed significantly after manual fixing, indicating that error distribution and improvement potential vary greatly among models.
*   Even Gemini-3.0-Pro (the strongest model) still had 15% inexecutable projects in Python, highlighting the fundamental challenge of multi-file understanding.
*   Java is the most difficult language, primarily due to stricter type systems and syntax requirements.
*   LLM self-fixing capability, while effective, falls far short of human fixing quality.

## Highlights & Insights
*   The three-scenario evaluation design is ingenious—distinguishing "problems solvable by simple fixes" from "inherent capability deficiency" through "re-evaluation after fixing basic errors" provides a fairer model assessment.
*   The concept of cascading errors is important for practical applications: a single missing import can cause 20 tests to fail simultaneously, inflating the error count.
*   The gap between open-source models (CodeQwen, DeepSeek-Coder, etc.) and closed-source models in multi-file test generation is massive, emphasizing the bottleneck in complex reasoning.

## Limitations & Future Work
*   Project scale is limited to $<1600$ lines to fit the context window; test generation for real-world large-scale projects poses even greater challenges.
*   Currently, only zero-shot evaluation is used; few-shot or agentic iterative generation strategies might significantly improve performance.
*   The standardization of manual fixing depends on annotators; although a protocol exists, subjective elements remain.

## Related Work & Insights
*   **vs DevBench**: DevBench has only 16 multi-file projects and does not mandate cross-file dependencies; MultiFileTest has 3.75x more projects and guarantees cross-file reasoning.
*   **vs HumanEval/MBPP**: These benchmarks only evaluate function-level code generation and fail to reflect dependency understanding in real-world projects.
*   **vs SWT-Bench**: SWT-Bench focuses on bug fixing rather than test generation; MultiFileTest focuses on test completeness and coverage.

## Rating
*   Novelty: ⭐⭐⭐⭐ First systematic multi-file unit test benchmark.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models, 3 languages, 3 scenarios, detailed error analysis.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive error classification.
*   Value: ⭐⭐⭐⭐⭐ Fills a significant gap in multi-file test evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[ACL 2026\] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)

</div>

<!-- RELATED:END -->

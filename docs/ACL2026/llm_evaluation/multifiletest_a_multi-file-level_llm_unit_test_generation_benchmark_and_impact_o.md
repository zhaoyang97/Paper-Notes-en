---
title: >-
  [Paper Note] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms
description: >-
  [ACL 2026][LLM Evaluation][unit test generation] This paper introduces MultiFileTest, the first multi-file-level benchmark for LLM-based unit test generation, covering 20 projects each in Python, Java, and JavaScript. It evaluates 11 state-of-the-art LLMs and analyzes the impact of manual and self-repair mechanisms on test quality, revealing that even the strongest models produce substantial basic executability errors.
tags:
  - ACL 2026
  - LLM Evaluation
  - unit test generation
  - multi-file benchmark
  - cross-file dependencies
  - error fixing
  - code quality
date: 2026-05-08
content_hash: a264578135851396
---

# MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms

**Conference**: ACL 2026
**arXiv**: [2502.06556](https://arxiv.org/abs/2502.06556)
**Code**: [GitHub](https://github.com/MultiFileTest)
**Area**: LLM Evaluation
**Keywords**: unit test generation, multi-file benchmark, cross-file dependencies, error fixing, code quality

## TL;DR
This paper introduces MultiFileTest, the first multi-file-level benchmark for LLM-based unit test generation, covering 20 projects each in Python, Java, and JavaScript. It evaluates 11 state-of-the-art LLMs and analyzes the impact of manual and self-repair mechanisms on test quality, revealing that even the strongest models produce substantial basic executability errors.

## Background & Motivation

**State of the Field**: LLM-driven unit test generation has become an important use case for code assistance, significantly improving test readability and generation efficiency. Existing benchmarks primarily evaluate test generation for function-level or class-level (single-file) code.

**Limitations of Prior Work**: (1) Real-world projects involve cross-file function interactions and complex dependencies, yet existing benchmarks overlook the challenges of multi-file test generation. (2) DevBench, the only benchmark touching multi-file testing, contains only 16 projects and is designed for breadth rather than depth, lacking systematic analysis of cross-file dependencies and errors. (3) A large proportion of LLM-generated tests contain fundamental errors (non-executable tests, cascading failures) that impede evaluation of higher-level capabilities such as correctness and coverage.

**Root Cause**: The core difficulty in multi-file test generation lies not in generating test logic per se, but in correctly understanding cross-file dependencies and properly setting up the test environment—precisely where LLM reasoning is weakest.

**Paper Goals**: (1) Construct a high-quality multi-file test benchmark; (2) systematically evaluate state-of-the-art LLMs on this task; (3) analyze error types and assess the effectiveness of repair mechanisms.

**Starting Point**: By re-evaluating after manually fixing basic errors, the paper distinguishes between "insufficient foundational capability" and "insufficient advanced capability," revealing the true potential differences among models.

**Core Idea**: Evaluation is conducted under three scenarios—original generation (Scenario 1), after manual repair (Scenario 2), and after LLM self-repair (Scenario 3)—with ranking shifts before and after error fixing used to expose fundamental differences between models.

## Method

### Overall Architecture
MultiFileTest comprises 60 curated GitHub projects (20 each in Python, Java, and JavaScript), each containing 2–15 files, fewer than 1,600 lines of code, and cross-file dependencies. The evaluation pipeline proceeds as follows: the LLM receives the complete project code along with a test generation prompt → raw tests are extracted → executability rate, correctness rate, and coverage are assessed → basic errors are manually repaired and re-evaluated → LLM self-repair is applied and re-evaluated.

### Key Designs

1. **Benchmark Dataset Construction**:

    - Function: Provides high-quality, guaranteed cross-file dependency test scenarios.
    - Mechanism: Projects are selected from GitHub using three criteria: appropriate size (2–15 files, <1,600 lines), inter-file dependencies, and high star/fork counts. For oversized projects, self-contained subprojects are extracted and dependency paths are adjusted. All projects undergo syntax validation and multi-line statement merging.
    - Design Motivation: Constraining project size to fit within LLM context windows ensures fair comparison; enforcing cross-file dependencies guarantees that multi-file reasoning is a necessary property.

2. **Three-Scenario Evaluation Protocol**:

    - Function: Distinguishes between raw capability, post-repair potential, and self-repair ability of LLMs.
    - Mechanism: Scenario 1 evaluates raw generation quality; Scenario 2 involves manual repair of executability and cascading errors by CS PhD students (averaging 2–6 lines of changes per project), revealing each model's true potential after eliminating basic errors; Scenario 3 provides the LLM with error messages and conversation history for self-repair.
    - Design Motivation: Executability errors (e.g., missing imports) are fundamentally simple issues, yet they reduce correctness and coverage to zero, masking real differences in models' ability to design test logic.

3. **Error Taxonomy**:

    - Function: Systematically categorizes error types in LLM-generated tests.
    - Mechanism: Executability errors (the entire test suite fails to run, e.g., `ModuleNotFoundError`) and cascading errors (a single root cause triggers multiple test failures, e.g., a missing NumPy import causing simultaneous failure across many tests).
    - Design Motivation: Distinguishing between "globally non-executable" and "individually failing tests" is critical for understanding LLM error patterns.

### Loss & Training
This paper presents an evaluation study and does not involve model training. Zero-shot prompting is used with temperature set to 0.

## Key Experimental Results

### Main Results (Python, Original Generation — Scenario 1)

| Model | Correctness Rate (CR) | Executability Rate (ER) | Line Coverage (LC) | Branch Coverage (BC) |
|---|---|---|---|---|
| Gemini-3.0-Pro | 77% | 85% | 76% | 73% |
| Claude-3.5-Sonnet | 64% | 70% | 51% | 47% |
| GPT-o1 | 60% | 65% | 56% | 54% |
| GPT-5-mini | 53% | 60% | 51% | 50% |
| GPT-4-Turbo | 47% | 65% | 40% | 36% |

### Cross-Language Comparison

| Language | Best Model | Best CR | Notes |
|---|---|---|---|
| Python | Gemini-3.0-Pro | 77% | Relatively easiest |
| Java | Gemini-3.0-Pro | 62% | Strict syntax increases difficulty |
| JavaScript | GPT-o1 | Highest | Optimal model varies by language |

### Key Findings
- Model rankings shift substantially after manual repair, indicating significant differences in error distribution and improvement potential across models.
- Even Gemini-3.0-Pro, the strongest model, leaves 15% of Python projects non-executable, underscoring fundamental challenges in multi-file understanding.
- Java is the most difficult language, primarily due to its stricter type system and syntax requirements.
- LLM self-repair is effective but falls considerably short of manual repair quality.

## Highlights & Insights
- The three-scenario evaluation design is particularly elegant—re-evaluating after fixing basic errors separates "problems solvable by simple repair" from "fundamental capability deficits," yielding a fairer model comparison.
- The concept of cascading errors is practically important: a single missing import can cause 20 tests to fail simultaneously, inflating error counts substantially.
- Open-source models (CodeQwen, DeepSeek-Coder, etc.) lag far behind closed-source models on multi-file test generation, highlighting a bottleneck in complex reasoning capability.

## Limitations & Future Work
- Project size is capped at fewer than 1,600 lines to fit within context windows; the challenges of test generation for truly large projects are considerably greater.
- Only zero-shot evaluation is employed; few-shot prompting or agent-based iterative generation strategies may yield significant performance improvements.
- The standardization of manual repair depends on annotators; despite a defined protocol, some subjectivity remains.

## Related Work & Insights
- **vs. DevBench**: DevBench contains only 16 multi-file projects and does not enforce cross-file dependencies; MultiFileTest offers 3.75× more projects and guarantees cross-file reasoning.
- **vs. HumanEval/MBPP**: These benchmarks evaluate only function-level code generation and cannot reflect dependency understanding in real-world projects.
- **vs. SWT-Bench**: SWT-Bench focuses on bug fixing rather than test generation; MultiFileTest targets test completeness and coverage more directly.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic multi-file unit test benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models, 3 languages, 3 scenarios, detailed error analysis
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive error taxonomy
- Value: ⭐⭐⭐⭐⭐ Fills an important gap in multi-file test evaluation

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)

<!-- RELATED:END -->

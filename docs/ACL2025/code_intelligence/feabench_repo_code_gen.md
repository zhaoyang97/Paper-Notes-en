---
title: >-
  [Paper Note] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation
description: >-
  [ACL 2025][Code Intelligence][Code Generation] FEA-Bench is proposed as the first benchmark evaluating LLM capabilities in feature implementation within repository-level codebases. It contains 1,401 task instances from 83 GitHub repositories, with each instance equipped with unit tests. The strongest model, DeepSeek-R1, solves only about 10% of the tasks, revealing the significant challenges repository-level incremental development poses to current LLMs.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Repository-Level Development"
  - "Feature Implementation"
  - "Benchmark"
  - "Incremental Development"
date: 2026-05-08
content_hash: 3932301f9ee59ca3
---

# FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation

**Conference**: ACL 2025  
**arXiv**: [2503.06680](https://arxiv.org/abs/2503.06680)  
**Code**: [https://github.com/microsoft/FEA-Bench](https://github.com/microsoft/FEA-Bench)  
**Area**: Code Intelligence  
**Keywords**: Code Generation, Repository-Level Development, Feature Implementation, Benchmark, Incremental Development

## TL;DR
FEA-Bench is proposed as the first benchmark evaluating LLM capabilities in feature implementation within repository-level codebases. It contains 1,401 task instances from 83 GitHub repositories, with each instance equipped with unit tests. The strongest model, DeepSeek-R1, solves only about 10% of the tasks, revealing the significant challenges repository-level incremental development poses to current LLMs.

## Background & Motivation
**Background**: Code generation evaluation has progressed from single-file tasks (such as HumanEval and MBPP) to the repository level (such as SWE-bench), but SWE-bench focuses primarily on bug fixing. In real-world software engineering, feature implementation (adding new functions, classes, or even files) constitutes a more core development activity.

**Limitations of Prior Work**: (a) There is a lack of benchmarks dedicated to evaluating feature implementation—pull requests in SWE-bench are primarily bug fixes and do not involve adding new components; (b) Feature implementation simultaneously requires code generation (for new components) and code editing (modifying dependent parts of existing code), which is more complex than bug fixing; (c) Code completion only focuses on local generation and fails to evaluate global impact.

**Key Challenge**: Feature implementation is the driving force behind software evolution, yet existing evaluations overlook it, focusing on "fixing existing issues" rather than "creating new functionalities."

**Goal**: To define repository-level incremental code development tasks and construct the first benchmark for them.

**Key Insight**: Filter pull requests from GitHub to select instances focused on new feature development (introducing new functions/classes/files) and pair them with unit tests for execution-based verification.

**Core Idea**: To evaluate the capability of LLMs to simultaneously "create new code" and "modify existing code," which is significantly more challenging than bug fixing.

## Method

### Overall Architecture
(1) Collect pull requests from 83 GitHub repositories; (2) Filter task instances using rule-based filtering (whether new functions/classes are introduced) and intent filtering (whether it is feature development rather than a bug fix); (3) Pair unit test files for each instance; (4) Construct the final FEA-Bench with 1,401 task instances.

### Key Designs

1. **Repository-Level Incremental Code Development Task Definition**:
    - **Function**: Given a repository codebase and a feature description (PR description), LLMs must simultaneously create new code components and modify existing code.
    - Difference from SWE-bench: SWE-bench focuses on resolving issues (primarily code modification), whereas FEA-Bench focuses on implementing new features (code creation + modification).
    - **Design Motivation**: Feature implementation requires a more comprehensive codebase understanding—models must know not only "where to modify" but also "what to add."

2. **Data Construction Pipeline**:
    - **Function**: Automatically filter feature implementation instances from GitHub PRs.
    - **Mechanism**: (a) Utilize AST parsing to detect whether the PR introduces new function/class definitions; (b) Employ an intent classifier to filter out non-feature PRs such as bug fixes, refactorings, and documentation updates; (c) Verify if paired unit tests exist and are executable.
    - **Design Motivation**: Simple usage of all PRs is insufficient; precise filtering of "new feature" types is required.

3. **Evaluation Metrics**:
    - **Function**: Execution-based automatic evaluation.
    - **Mechanism**: Run paired unit tests and calculate pass rates.
    - Metrics are classified into: Full Resolve Rate (all tests pass) and Partial Resolve Rate.

### Loss & Training
- **Pure Evaluation Benchmark**—No training component.
- Evaluates multiple SOTA LLMs: DeepSeek-R1/V3, GPT-4/4o/o1, Claude-3.5-Sonnet, etc.

## Key Experimental Results

### Main Results (Full Resolve Rate %)

| Model | Parameters | FEA-Bench (Full) | FEA-Bench (Curated) |
|------|------|-----------------|-----------------|
| DeepSeek-R1 | 671B | **~10%** | ~14.5% |
| DeepSeek-V3 | 671B | ~8% | ~14.5% |
| GPT-4o | - | ~6% | ~5% |
| Claude-3.5-Sonnet | - | ~7% | - |
| GPT-4 | - | ~5% | ~6% |
| o1-mini | - | ~2% | ~2-3% |

### Statistical Differences: FEA-Bench vs. SWE-Bench

| Feature | SWE-Bench | FEA-Bench |
|------|----------|----------|
| Primary Task Type | Bug fix | Feature implementation |
| Introduction of New Functions | Few | **Many (Core)** |
| Code Change Length | Shorter | **Longer** |
| Required Capability | Code Editing | Code Creation + Editing |
| Pass Rate of Strongest Model | ~50%+ | **~10%** |

### Key Findings
- **The strongest LLMs solve only ~10% of new feature tasks**—far lower than the >50% pass rate for bug fixes on SWE-bench.
- Feature implementation requires longer code generation—the task difficulty and required creativity are significantly higher than for bug fixes.
- DeepSeek-R1 leads due to its strong reasoning capabilities—but still lags far behind human developers.
- "Understanding repository structure" and "locating existing files to modify" represent the primary bottlenecks for the models.
- Smaller models (e.g., DeepSeek-R1-Distill-32B) show a significant drop in performance—repository-level tasks demand high model capability.

## Highlights & Insights
- **The finding that "creating is harder than fixing"** is intuitive but now quantitatively proven—the drop from 50%+ on SWE-bench to ~10% on FEA-Bench indicates that LLMs' code "creation" capabilities are vastly inferior to their "patching" capabilities.
- **Simultaneously requiring both code creation and code editing** is the unique challenge of FEA-Bench—which always coexist in real-world development.
- The diversity of 83 repositories ensures the representativeness of the benchmark—covering Python projects of various scales and domains.
- Execution-based evaluation (unit testing) is more reliable than match-based evaluation.
- This benchmark can be used to continuously track the progress of LLM code generation capabilities—especially in the dimension of "creative development."

## Limitations & Future Work
- Only covers Python repositories—feature implementation in other languages such as Java and TypeScript is not addressed.
- Although 1,401 instances are substantial, they can still be expanded.
- Uneven quality of PR descriptions—some descriptions are highly detailed while others are extremely brief, impacting task fairness.
- Evaluates only "whether it passes the tests"—it does not evaluate code quality (readability, maintainability, etc.).
- The filtering pipeline may miss some feature PRs or incorrectly include non-feature PRs.

## Related Work & Insights
- **vs SWE-bench**: SWE-bench = bug fix evaluation, FEA-Bench = feature implementation evaluation—complementary positioning.
- **vs HumanEval/MBPP**: Single-file programming tasks without repository context—FEA-Bench's repository-level analysis is closer to real-world development.
- **vs TestCase-Eval**: TestCase-Eval evaluates test generation, while FEA-Bench evaluates code generation—distinct perspectives.
- The capability of LLM development assistants (Cursor, Devin) in feature implementation can be evaluated using FEA-Bench.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First repository-level feature implementation evaluation benchmark, filling an important gap in SWE-bench.
- Experimental Thoroughness: ⭐⭐⭐⭐ 83 repositories + 1,401 instances + multiple SOTA models + comparison with SWE-bench.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions, and the distinction from SWE-bench is well articulated.
- Value: ⭐⭐⭐⭐⭐ Substantial contribution to code LLM evaluation and software engineering AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation](coco-bench_a_comprehensive_code_benchmark_for_multi-task_large_language_model_ev.md)
- [\[ACL 2025\] Program Synthesis Benchmark for Visual Programming in XLogoOnline Environment](program_synthesis_benchmark_for_visual_programming_in_xlogoonline_environment.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)

</div>

<!-- RELATED:END -->

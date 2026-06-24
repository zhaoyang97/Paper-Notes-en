---
title: >-
  [Paper Note] Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'
description: >-
  [ACL2025][LLM (Other)][Code Generation] RepoCod is constructed—a benchmark containing 980 complex code generation tasks from 11 large-scale Python projects, featuring real repository-level dependencies and an average of 314 developer test cases. It reveals that even the most advanced LLMs achieve less than 30% Pass@1, falling far short of replacing programmers in real-world coding tasks.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Code Generation"
  - "Repository-Level"
  - "Benchmark"
  - "RAG"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: acfe65afbc0968e2
---

# Can Language Models Replace Programmers for Coding? RepoCod Says 'Not Yet'

**Conference**: ACL2025  
**arXiv**: [2410.21647](https://arxiv.org/abs/2410.21647)  
**Code**: Yes (mentioned in the paper)  
**Area**: LLM NLP / Code Generation Benchmark  
**Keywords**: Code Generation, Repository-Level, Benchmark, RAG, LLM Evaluation  

## TL;DR

RepoCod is constructed—a benchmark containing 980 complex code generation tasks from 11 large-scale Python projects, featuring real repository-level dependencies and an average of 314 developer test cases. It reveals that even the most advanced LLMs achieve less than 30% Pass@1, falling far short of replacing programmers in real-world coding tasks.

## Background & Motivation

### Problem Definition
LLMs have achieved 90%+ Pass@1 on self-contained benchmarks like HumanEval, but a core problem remains: do these high scores represent the capability of LLMs in real-world software development? Existing repository-level code generation benchmarks fail to answer this question due to systematic deficiencies.

### Four Deficiencies of Existing Benchmarks

**1. Non-Real-World Tasks**:
HumanEval and MBPP are artificially constructed programming problems, which fail to represent real-world development tasks driven by project requirements.

**2. Low Task and Repository Complexity**:
- CrossCodeEval, RepoBench, and Long-Code-Arena only perform single-line completion.
- CoderEval and DevEval have short functions (average of 108/86 tokens) and small repository scales (average of 152/164 files).
- High risk of benchmark saturation.

**3. Lack of Repository-Level Dependencies**:
- Only 27% of functions in real-world code are self-contained.
- Only 10-31% of tasks in CoderEval and DevEval require cross-file dependencies.
- Existing benchmarks are dominated by self-contained or file-level tasks.

**4. Inappropriate Evaluation Metrics**:
- Similarity metrics such as CodeBLEU and BLEU cannot determine functional equivalence.
- High mismatch rate with human evaluation.

## Method

### Overall Architecture: Data Construction Pipeline

RepoCod employs a three-stage automated data collection pipeline:

**Step I - Repository Selection**:
- Python as the primary language ($\ge 70\%$)
- Open-source repositories with $\ge 2\text{K}$ stars
- Clone the latest version (October 2024)
- Finally selected 11 popular projects

**Step II - Target Function Selection**:
Combining static and dynamic analysis:
- **Static Analysis**: Uses tree-sitter to parse test functions and collect invoked functions.
- **Dynamic Analysis**: Uses the Python `trace` module to execute tests and capture indirect function calls.
- **Filtering Conditions**: $\ge 10$ lines of docstring + $\ge 2$ lines of function body.

**Step III - Relevant Test Case Collection**:
A two-step collection ensures complete coverage:
1. Execute all tests to establish reference results.
2. For each target function, replace its body with an assertion failure and rerun all tests.
3. If the test result changes from pass to fail, the test is deemed relevant to the target function.

### Benchmark Structure

Each instance contains:
- **Target Function Description**: Developer-provided docstring.
- **Repository Snapshot**: Full source code (with the target function body removed).
- **Relevant Test Cases**: Tests written by developers.
- **Ground Truth**: Developer-written function body.

### Key Statistics

| Metric | RepoCod | DevEval | CoderEval | RepoBench |
|------|---------|---------|-----------|-----------|
| Number of Instances | 980 | 1,825 | 230 | 23,561 |
| Average Token Count | **331.6** | 86.3 | 108.2 | 13.7 |
| Cyclomatic Complexity | **9.0** | 3.5 | 4.7 | 1.0 |
| Test Cases / Instance | **313.5** | 2.1 | - | 0 |
| Repository File Count | **2,610** | 164 | 152 | - |
| Repository Lines of Code (LoC) | **290,110** | 36,640 | 48,821 | - |

### Context Complexity Distribution

| Benchmark | Repository-level | File-level | Self-contained |
|------|--------|--------|--------|
| CoderEval | 10.0% | 53.5% | 36.5% |
| DevEval | 31.3% | 41.2% | 27.5% |
| **RepoCod** | **50.8%** | 18.1% | 31.1% |

More than half of the tasks in RepoCod require repository-level context, which far exceeds other benchmarks.

### Test Execution Optimization

Through fine-grained test collection, the number of test cases for each instance was reduced from an average of 17,974 (full repository) to 313 (relevant tests), and the execution time dropped from 216.9 hours to 22.6 hours (only 10.4% of the original time).

## Experiments

### Experimental Setup

**Retrieval Methods**:
- **RAGBM25**: Sparse retrieval based on BM25.
- **RAGDense**: Dense retrieval based on text-embedding-3-small.
- **Current File**: Uses only the file containing the target function as the context.
- **Baseline**: Provides only the signature and docstring.
- **Callees** (Oracle): Provides the actual invoked functions.
- **RAGDense-oracle** (Oracle): Uses the ground truth as the query.

**Models**: 10 SOTA LLMs (GPT-4o, DeepSeek-V2.5, Claude 3.5 Sonnet, etc.).

### Main Results

| Model | RAGBM25 | RAGDense | Current-File |
|------|---------|----------|-------------|
| CodeLlama-7B | 10.7 | 10.4 | 5.7 |
| CodeLlama-34B | 12.4 | 12.8 | 9.6 |
| DeepSeekCoder-6.7B | 14.0 | 14.1 | 10.9 |
| DeepSeekCoder-33B | 16.7 | 17.1 | 14.9 |
| Claude 3.5 Sonnet | 14.4 | 17.5 | 19.8 |
| DeepSeek-V2.5 | 18.5 | 20.7 | 27.0 |
| GPT-4o-Mini | 15.1 | 15.0 | 18.7 |
| **GPT-4o** | **27.4** | **27.0** | **26.8** |

**Core Finding**: Even the best model, GPT-4o, achieves only 27.4% Pass@1—far below the ~90% on HumanEval.

### Oracle Setup Results (GPT-4o)

| Method | Self | File | Repo | Overall |
|------|------|------|------|---------|
| Baseline | 23.6 | 11.3 | 3.8 | 11.3 |
| RAGBM25 | 39.3 | 31.1 | 18.7 | 27.3 |
| RAGDense | 44.6 | 36.7 | 12.9 | 27.0 |
| Current-File | 39.3 | 35.0 | 16.3 | 26.8 |
| Callees (Oracle) | 35.1 | 31.1 | 12.2 | 22.8 |
| **RAGDense-oracle** | **45.2** | **34.5** | **16.3** | **28.6** |

Even when utilizing the ground truth as the retrieval query (RAGDense-oracle), the Pass@1 score reaches only 28.6%.

### Impact of Complexity on Performance

**Context Complexity**: All models perform the worst on repository-level tasks, with Pass@1 decreasing as context complexity increases.

**Cyclomatic Complexity**: Under the most complex setting ($M \ge 11$), the best LLM achieves only 7.0% Pass@1.

**Token Length**: When the length is $> 232$ tokens, even GPT-4o scored under 10% Pass@1.

### Impact of Retrieval Recall

| Model | Recall = 0 | Recall $\in (0, 0.5]$ | Recall $\in (0.5, 1]$ |
|------|---------|---------------|----------------|
| GPT-4o (BM25) | 20.3 | 16.4 | **41.1** |
| GPT-4o (Dense) | 16.7 | 14.2 | **38.0** |

Retrieval with high recall ($> 0.5$) significantly boosts performance, whereas low-recall retrieval performs even worse than no retrieval at all.

### Key Findings

1. **LLMs are far from competent at real-world repository-level code generation** (maxing out at only 28.6% Pass@1).
2. **Higher dependency recall enhances performance**, whereas low-recall retrieval can be unhelpful.
3. **Directly providing dependency functions is not optimal**—the Callees setting underperforms compared to RAG and Current-File.
4. **Additional context is helpful even for self-contained functions**.
5. **Different models possess unique problem-solving capabilities**—each model has distinct tasks that only it can solve.
6. **Commercial models perform better in the Current-File setting**, whereas open-source models perform better in the RAG setting.

### Failure Cause Analysis

An analysis of 30 failure cases from GPT-4o revealed two main categories of reasons:
1. **Insufficient Input Validation**: Ignoring or mishandling unexpected inputs.
2. **Incomplete or Incorrect Core Logic Implementation**: Selecting suboptimal algorithms, incorrectly managing object states, or misuse of parallel processing strategies.

## Highlights & Insights

1. **A truly realistic benchmark**: An average ground truth size of 331 tokens, repositories with 2,610 files, and 314 test cases per instance—these figures far exceed existing benchmarks, truly reflecting the complexity of software engineering.
2. **Ingenious design for automated test collection**: The "replace with assertion failure" method is simple yet effective in identifying all relevant tests without manual intervention.
3. **The counter-intuitive finding that Callees underperforms RAG**: Direct provision of dependency functions is actually less effective than retrieving similar functions, likely because retrieved results offer richer contextual patterns and coding style information.
4. **50.8% Repository-level dependencies**: This proportion far exceeds other benchmarks, ensuring that the benchmark will not be saturated quickly.
5. **Practical scalability of the benchmark**: The fully automated annotation pipeline makes it easy to scale the benchmark to more projects.

## Limitations & Future Work

1. The dataset is sourced from only 11 repositories, restricted to Python.
2. Only 10 models were evaluated, yielding limited representation.
3. Data may become outdated over time (as repositories continue to evolve).
4. Automated test collection might miss certain indirect call scenarios.
5. Using Pass@1 as the sole evaluation metric may overlook "close-to-correct" solutions.
6. The impact of API version evolution on code generation was not factored in (though the paper discusses this related direction).

## Related Work & Insights

- **Self-contained Benchmarks**: HumanEval, MBPP, APPS
- **Repository-level Benchmarks**: CrossCodeEval (single-line completion), RepoBench (single-line), CoderEval (short functions), DevEval (manually annotated), R2E-Eval1 (LLM-generated tests), RepoEval
- **Version-aware Benchmarks**: LibEvolutionEval, GitChameleon, CodeUpdateArena, VersiCode
- **Code LLMs**: CodeLlama, DeepSeek-Coder, GPT-4o, etc.
- **Test Collection**: CoderEval (static analysis + manual), R2E-Eval1 (LLM-generated)

## Rating ⭐⭐⭐⭐

This work construct the most challenging and realistic code generation benchmark to date, featuring a clear design philosophy with statistical indicators that comprehensively outperform existing benchmarks. It reveals a massive performance gap for LLMs in real-world coding tasks (~90% vs. ~28%), providing significant calibration value to the community. The automated design of the test collection pipeline is highly referable. Minor drawbacks involve its coverage of only 11 Python projects and the limited number of evaluated models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)
- [\[ACL 2025\] Can Language Models Reason about Individualistic Human Values and Preferences?](can_language_models_reason_about_individualistic_human_values_and_preferences.md)
- [\[ACL 2025\] Can Large Language Models Address Open-Target Stance Detection?](can_large_language_models_address_open-target_stance_detection.md)
- [\[ACL 2025\] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?](can_large_language_models_accurately_generate_answer_keys_for_health-related_que.md)

</div>

<!-- RELATED:END -->

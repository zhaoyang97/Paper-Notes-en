---
title: >-
  [Paper Note] CodeMEnv: Benchmarking Large Language Models on Code Migration
description: >-
  [ACL 2025][LLM Evaluation][code migration] This paper proposes CodeMEnv, the first benchmark to systematically evaluate the cross-environment code migration capabilities of LLMs. It contains 922 samples from 19 Python/Java packages, covering 3 hierarchical tasks (locating incompatible functions -> describing changes -> migrating code). The average Pass@1 of 9 evaluated LLMs is only 26.50%, with GPT-4o achieving the highest at 43.84%. The findings reveal that LLMs are more fam…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "code migration"
  - "benchmark"
  - "API versioning"
  - "software engineering"
date: 2026-05-08
content_hash: 79c58dba00d52e20
---

# CodeMEnv: Benchmarking Large Language Models on Code Migration

**Conference**: ACL 2025  
**arXiv**: [2506.00894](https://arxiv.org/abs/2506.00894)  
**Code**: [GitHub](https://github.com/xdshen-ai/Benchmark-of-Code-Migration)  
**Area**: LLM Evaluation  
**Keywords**: code migration, benchmark, LLM evaluation, API versioning, software engineering

## TL;DR

This paper proposes CodeMEnv, the first benchmark to systematically evaluate the cross-environment code migration capabilities of LLMs. It contains 922 samples from 19 Python/Java packages, covering 3 hierarchical tasks (locating incompatible functions -> describing changes -> migrating code). The average Pass@1 of 9 evaluated LLMs is only 26.50%, with GPT-4o achieving the highest at 43.84%. The findings reveal that LLMs are more familiar with newer function versions and exhibit inconsistency in version-reasoning logic.

## Background & Motivation

**Background**: LLMs have achieved remarkable success in software engineering tasks such as code generation and cross-language code translation, with models like GPT-4, DeepSeek-V3, and CodeLlama consistently setting new records on benchmarks like HumanEval.

**Limitations of Prior Work**: Code migration (adapting code to run in different environments) is a highly frequent pain point in actual development. After obtaining code from GitHub, users often need to spend massive manual effort on adaptation due to library version incompatibilities. However, this scenario has rarely been systematically studied.

**Key Challenge**: The continuous evolution of libraries leads to frequent API changes (e.g., in NumPy 1.26 -> 2.0, `compare_chararrays` was moved from the top-level module to the `numpy.char` sub-module). The implementation of the same functionality can be completely different across different versions. Yet, almost all existing benchmarks focus on cross-programming-language translation rather than cross-version migration.

**Limitations of Prior Work**: Google researchers Ziftci et al. (2025) explored automated migration, and Amazon Q Developer provides tools specifically for Java 8/11 -> 17 migration, but a comprehensive evaluation benchmark tailored for LLMs remains lacking.

**Key Insight**: Starting from three types of API function changes (addition, deprecation, replacement), this work designs a hierarchical evaluation framework covering the entire pipeline of "locating -> understanding -> patching".

**Core Idea**: Code migration requires LLMs to possess version-aware API knowledge, cross-version reasoning, and code generation capabilities simultaneously. Since none of these can be omitted, a dedicated benchmark is required to measure them individually.

## Method

### Benchmark Construction Process

**Step 1 — Data Collection**: Function changes are systematically gathered from the official documentation release notes of 19 packages (11 Python + 8 Java). This yields 212 Python function changes and 114 Java function changes, with their compatible version ranges determined by actual execution across multiple versions.

**Step 2 — Code Generation**: GPT-4 is utilized to generate target code based on the function change information and usage instructions. For "addition" changes, New2Old samples are generated (new environment code -> migrate to old environment); for "deprecation" changes, Old2New samples are generated; and for "replacement" changes, samples are generated in both directions.

**Step 3 — Test Case Generation**: GPT-4 generates 3 test cases for each code sample. Ground-truth outputs are obtained by executing them on the original code. If a test case fails, iterative fixing is applied for up to 3 rounds, failing which the sample is discarded.

### Three Hierarchical Tasks

1. **Task-1: Locating Incompatible Functions**: Given a code snippet and the target environment version, the model must accurately pinpoint all incompatible functions. This is divided into easy (only 1 incompatible function) and hard (2-3 incompatible functions) difficulties.
2. **Task-2: Describing Function Changes**: For each incompatible function, the model is required to identify the change type (addition/deprecation/replacement), the target version (permitting an error within $\pm 0.5$), and the substitute function name.
3. **Task-3: Code Migration**: This requires modifying the code to execute correctly in the target environment, evaluated by passing all 3 unit tests. This task is split into two directions: Old2New and New2Old.

### Evaluation Methodology

- Task-1/Task-2: Evaluated using an agent-based approach, comparing model predictions precisely against the ground truth.
- Task-3: Evaluated via unit testing. The migrated code must produce outputs identical to the original code across all 3 test cases. Pass@1 and Pass@5 are reported.

## Key Experimental Results

### Table 1: Task-1 & Task-2 Accuracy (%)

| Model | Task-1 Python (easy) | Task-1 Python (hard) | Task-1 Java | Task-1 Avg | Task-2 Avg |
|------|-----------|-----------|------|------|------|
| GPT-3.5-Turbo | **85.10** | **32.98** | 80.89 | **66.32** | 34.13 |
| GPT-4o | 70.71 | 25.65 | 81.19 | 59.18 | 37.02 |
| DeepSeek-v3 | 78.48 | 26.17 | 82.08 | 62.24 | **42.06** |
| Llama-3.1-70B | 75.51 | 29.84 | 81.19 | 62.18 | 35.44 |
| Llama-3.1-8B | 70.71 | 21.99 | 67.16 | 53.29 | 23.79 |

### Table 2: Task-3 Code Migration Pass@1 (%)

| Model | Old2New easy | Old2New hard | New2Old easy | New2Old hard |
|------|-------------|-------------|-------------|-------------|
| GPT-4o | **43.84** | **26.83** | **31.60** | **22.94** |
| DeepSeek-v3 | 41.20 | 20.73 | 29.60 | 14.68 |
| Llama-3.1-70B | 32.88 | 19.51 | 28.80 | 17.43 |
| Code Llama-34B | 35.62 | 21.95 | 29.60 | 15.76 |
| GPT-3.5-Turbo | 26.03 | 7.32 | 24.80 | 7.34 |
| Qwen2.5-Coder-7B | 32.19 | 14.63 | 29.20 | 8.26 |

### Key Findings

- **Low Overall Performance**: The average Pass@1 of the 9 LLMs on the migration task is only 26.50%, indicating that cross-version code migration is far from being resolved.
- **Newer Version Preference**: All models perform significantly better in the Old2New direction than in New2Old (e.g., GPT-4o easy: 43.84% vs 31.60%), indicating a higher proportion of newer function versions in the training data of LLMs.
- **Localization $\neq$ Migration**: GPT-3.5-Turbo achieves the highest performance in Task-1 (locating incompatible functions) at 66.32%, but its performance on the migration task (Task-3 hard) is only 7.32%. This signifies that "finding a problem" and "solving a problem" are completely distinct capabilities.
- **Inconsistent Version Reasoning**: In case studies, Llama-3.1-8B and GPT-3.5-Turbo mistakenly referenced changes from version 1.17/1.18 when the target environment was NumPy 1.16, exposing a systematic weakness in version sequential reasoning.
- **Sharp Difficulty Increase with Multiple Incompatible Functions**: For the hard set (2-3 incompatible functions), Pass@1 typically drops by 50-70%, as models struggle to handle multiple incompatibility points simultaneously.
- **Distribution of Error Types**: CallError (still invoking the incompatible function) accounts for the largest proportion (e.g., up to 50.8% for Llama-3.1-8B), followed by RunError (infinite loops, reaching 33.0% for DeepSeek-v3) and WrongAnswer (up to 19.4% for GPT-4o).

## Highlights & Insights

- **First Comprehensive Code Migration Benchmark**: This work fills a critical gap in cross-version migration scenarios within LLM evaluations. The three hierarchical tasks are elegantly designed to decouple different dimensions of migration capability.
- **Authentic Data Sources**: Function changes are manually curated from the official documentation of 19 packages rather than synthetically generated, ensuring the practical relevance and credibility of the evaluation.
- **Task-Decoupling Discovery**: This study quantitatively proves for the first time that "locating incompatible APIs" and "completing code migration" represent completely distinct capabilities, providing a clear path forward for future LLM enhancements.
- **Unique Value of the New2Old Direction**: It reveals a counter-intuitive challenge, showing that adapting new code to older environments is harder than the reverse, which carries significant implications for developers maintaining legacy systems.

## Limitations & Future Work

- **Small Data Scale**: The benchmark contains 922 samples in total (587 Python + 335 Java). The Java partition only contains easy difficulty, and documentation coverage for some packages might be insufficient.
- **Limited Language Coverage**: The benchmark currently supports only Python and Java, leaving out widely used languages such as JavaScript/TypeScript, Rust, and Go.
- **GPT-4 Generated Code**: Code blocks and test cases are both generated by GPT-4 rather than harvested from real-world projects, which may introduce distribution bias and might not fully reflect the actual complexity of real-world migration scenarios.
- **Evaluation Limitations**: Correctness in Task-3 is validated using only 3 test cases, which might miss certain edge cases. The accuracy of the agent-based evaluation depends heavily on the capabilities of the evaluation agent itself.

## Related Work & Insights

### vs. CodeUpdateArena (Liu et al. 2024)
CodeUpdateArena focuses on knowledge editing in LLMs following API updates, evaluating whether the model "knows" an API has changed. CodeMEnv is broader: it requires models to not only know the change but also locate incompatible code and execute actual migration, covering the complete pipeline from identification to fixing.

### vs. Amazon Q Developer (Code Migration Tool)
Amazon Q is a production-grade tool designed for Java 8/11 -> 17 upgrades, focusing on specific version upgrades within a single language. CodeMEnv is an evaluation benchmark rather than a tool, covering various version combinations across 19 packages, and supports bi-directional migration (Old2New + New2Old), offering a more systematic and comprehensive evaluation.

### vs. Cross-Language Code Translation Benchmarks (Yuan et al. 2024, Eniser et al. 2024)
Cross-language translation focuses on mapping between different programming languages (e.g., Python -> Rust) without involving library version compatibility. CodeMEnv, on the other hand, targets code adaptation necessitated by library version shifts within the same language, capturing a complementary but fundamentally distinct dimension.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic benchmark for cross-version code migration, offering clear task definitions and a well-thought-out hierarchical design.
- **Experimental Thoroughness**: ⭐⭐⭐ Evaluates 9 main LLMs with a thorough analysis, though limited by smaller data scale and language coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Distinct structure with direct and insightful case and error analyses.
- **Value**: ⭐⭐⭐⭐ Pinpoints key weaknesses of LLMs in actual code migration, offering useful insights for both the evaluation community and tool developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](cov_eval_evaluating_llms_from_code_security_perspective.md)
- [\[ACL 2025\] Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](mis-prompt_benchmarking_large_language_models_for_proactive_error_handling.md)
- [\[ACL 2025\] Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories](benchmarking_llms_and_llm-based_agents_in_practical_vulnerability_detection_for_.md)
- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

</div>

<!-- RELATED:END -->

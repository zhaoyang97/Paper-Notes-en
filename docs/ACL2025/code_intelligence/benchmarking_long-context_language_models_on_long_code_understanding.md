---
title: >-
  [Paper Note] LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding
description: >-
  [ACL 2025][Code Intelligence][long-context] The authors propose the LongCodeU benchmark, which designs 8 tasks across four dimensions—code unit perception, intra-code unit understanding, inter-code unit relation understanding, and long documentation understanding—to evaluate the comprehension capabilities of 9 long-context language models (LCLMs) on real-world, repository-level long code, revealing that 32K tokens is the practical upper limit for current LCLM long code unders…
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "long-context"
  - "code understanding"
  - "benchmark"
  - "code unit"
  - "dependency analysis"
date: 2026-05-08
content_hash: a5dfddf81dde18dd
---

# LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding

**Conference**: ACL 2025  
**arXiv**: [2503.04359](https://arxiv.org/abs/2503.04359)  
**Code**: None  
**Area**: Code Intelligence  
**Keywords**: long-context, code understanding, benchmark, code unit, dependency analysis

## TL;DR

The authors propose the LongCodeU benchmark, which designs 8 tasks across four dimensions—code unit perception, intra-code unit understanding, inter-code unit relation understanding, and long documentation understanding—to evaluate the comprehension capabilities of 9 long-context language models (LCLMs) on real-world, repository-level long code, revealing that 32K tokens is the practical upper limit for current LCLM long code understanding.

## Background & Motivation

**Background**: Long-Context Language Models (LCLMs) such as GPT-4o (128K) and Gemini-1.5 (1M) claim to support ultra-long context windows, enabling potential software engineering applications like repository-level code generation, issue fixing, and long code summarization. However, current frameworks lack rigorous evaluation to measure whether LCLMs truly "understand" long code.

**Limitations of Prior Work**: The first type of benchmarks (e.g., RepoQA, L-Eval) suffers from four issues: ❶ insufficient task diversity, focusing only on single tasks such as needle function search; ❷ artificial concatenation of independent code snippets to construct "long code," which ignores dependencies in real-world code; ❸ lack of restriction on code release dates, posing data contamination risks; ❹ a maximum length of only 36.5K tokens, failing to stress-test 128K-1M context windows. The second type of benchmarks (e.g., LongBench, SWE-bench) evaluates indirectly via downstream tasks, leading to a fifth issue: ❺ code understanding capability is entangled with task-specific challenges such as code generation or bug fixing, making it impossible to measure comprehension independently.

**Key Challenge**: Are LCLMs claiming to support ultra-long contexts truly effective in real-world long code understanding tasks? Existing benchmarks cannot provide reliable answers.

**Key Insight**: Constructing a long code understanding benchmark covering a complete skill spectrum from basic perception to complex reasoning, utilizing real repository code created after 2024-06 to independently evaluate comprehension capabilities.

## Method

### Overall Architecture

LongCodeU designs 8 tasks across four dimensions, with approximately 500 samples per task, covering lengths from 0 to 128K tokens:

1. **Code Unit Perception** (CU_P) — 1 task
2. **Intra-Code Unit Understanding** — 2 tasks: Data Flow Analysis (CU_DFA) + Semantic Analysis (CU_SA)
3. **Inter-Code Unit Relation Understanding** — 4 tasks: Dependency Relation Analysis T1/T2 (DRA) + Semantic Relation Extraction T1/T2 (SRE)
4. **Long Documentation Understanding** (LDU) — 1 task

All tasks share a unified workflow: Given instruction + long code + anchor input $\rightarrow$ LCLM outputs the answer.

### Key Designs

1. **Four-Dimension Eight-Task System**:

    - **Function**: Progressive evaluation from basic perception to complex reasoning
    - **Mechanism**: CU_P tests function identification capability (foundation) $\rightarrow$ CU_DFA/CU_SA test intra-unit variable tracking and semantic understanding $\rightarrow$ DRA tests cross-unit invocation relations (including code-to-code and natural language-to-code modes) $\rightarrow$ SRE tests semantic similarity reasoning $\rightarrow$ LDU tests long document information extraction
    - **Design Motivation**: The four dimensions correspond to the practical requirements of repository-level development: identifying functions, understanding function logic, analyzing relationships between functions, and reading technical documentation

2. **Real-World Repository Code Construction Pipeline**:

    - **Function**: A six-stage pipeline to ensure data quality and authenticity
    - **Mechanism**: Stage ❶ selects 116 repositories created after 2024-06, non-forked, and with 50+ stars from top 50 popular PyPI packages $\rightarrow$ Stage ❷ uses tree-sitter static analysis to extract function definitions and dependency relationships + uses embedding models to compute semantic similarity $\rightarrow$ Stage ❸ extracts requirements descriptions from code signatures $\rightarrow$ Stage ❹ manually annotates 500 document understanding samples $\rightarrow$ Stage ❺ excludes trivial functions and deduplicates $\rightarrow$ Stage ❻ constructs samples according to a length distribution of 0-128K.
    - **Design Motivation**: Utilizing code developed after 2024-06 significantly mitigates data contamination risks; real repository code contains natural dependencies, far superior to concatenated code

3. **Fine-grained Evaluation Metric System**:

    - **Function**: Customizing evaluation metrics according to the output granularity of different tasks
    - **Mechanism**: Utilizing EM-R/EM-P for code line outputs; LCS-R/LCS-P (longest common subsequence) for function name outputs; CodeBLEU-R/CodeBLEU-P for code unit outputs; and BLEU for natural language descriptions
    - **Design Motivation**: The Kendall-Tau $\tau$ correlation between automatic evaluation and human evaluation averages $\ge 0.75$ across all tasks, with the minimum value exceeding 0.7, validating the reliability of the metrics.

### Loss & Training

This paper presents a benchmark and does not involve model training. Evaluation employs greedy search, conducted within the maximum context window supported by each model.

## Key Experimental Results

### Main Results

Evaluation of 9 LCLMs (6 general-purpose + 3 code models), Recall metrics:

| Model | Parameters | CU_P | CU_SA | CU_DFA | DRA_T1 | DRA_T2 | SRE_T1 | SRE_T2 | LDU | Average |
|------|--------|------|-------|--------|--------|--------|--------|--------|-----|------|
| DeepSeek-V2.5 | 236B | 70.58 | 82.11 | 77.47 | 72.25 | 56.80 | 49.08 | 47.42 | 85.85 | **67.70** |
| GPT-4o | — | 56.42 | 86.76 | 87.87 | 71.58 | 48.88 | 44.45 | 43.14 | 87.54 | **65.83** |
| Gemini-1.5-Flash | — | 58.45 | 83.46 | 80.37 | 72.51 | 46.42 | 39.84 | 38.69 | 81.43 | **61.39** |
| CodeLlama | 33.7B | 68.57 | 62.41 | 79.87 | 68.82 | 34.94 | 44.48 | 36.34 | 46.92 | **55.29** |
| Mistral-v0.3 | 7.3B | 57.42 | 63.90 | 58.00 | 46.66 | 18.92 | 33.91 | 32.50 | 58.64 | **46.24** |
| Claude-3.5-Sonnet | — | 43.82 | 40.60 | 45.65 | 29.37 | 28.70 | 26.55 | 27.77 | 41.81 | **35.53** |
| Phi-3.5 | 3.8B | 39.92 | 46.75 | 49.52 | 30.76 | 9.66 | 18.99 | 14.48 | 34.14 | **30.53** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| 0-8K Length | Normal Performance | All models exhibit reasonable performance on short code |
| 8K-32K Length | Slow Decline | Performance begins to degrade but remains acceptable |
| 32K+ Length | **Sharp Decline** | Performance drops off a cliff, far below claimed capability |
| 64K-128K Length | DRA/SRE near 0 | Some tasks fail completely |
| w/o Code Context | Performance far below w/ context | Proves the models perform actual comprehension rather than memory retrieval |
| Code Models vs General-purpose Models | Code models of the same scale perform better | Qwen2.5-Coder outperforms Phi-3.5 by 24.31% on CU_SA |
| Automatic vs Human Evaluation | Kendall-Tau $\tau \ge 0.75$ | Confirms the reliability of automatic metrics |

### Key Findings

- **32K is the Practical Limit**: The performance of all LCLMs drops sharply once the code length exceeds 32K tokens, which is in stark contrast with their claimed 128K-1M context windows.
- **Inter-unit relation understanding is the hardest**: DRA and SRE tasks pose the most significant bottlenecks for all models, particularly DRA_T2, which requires cross-file dependency tracking.
- **No universal champion**: No single LCLM achieves optimal performance across all 8 tasks—code models excel in code perception, while general-purpose models perform better in documentation understanding.
- **Performance degradation rate varies by task**: Documentation understanding exhibits the steepest degradation curve, whereas code unit understanding remains relatively stable.
- **Comprehension vs. Memory**: The w/o context ablation confirms that LCLMs genuinely engage in code comprehension rather than simple memory retrieval.

## Highlights & Insights

- The design of four dimensions and eight tasks covers the complete skill spectrum from basic perception to complex reasoning, representing the most systematic evaluation of long code understanding to date.
- Utilizing real code repositories created after 2024-06 successfully avoids data contamination while preserving natural code dependencies.
- Clearly distinguishing between "code comprehension" and "code memory," with a highly ingenious design of the "w/o context" ablation study.
- Providing practical rules of thumb for model selection: choose smaller models for volumes $\le 16\text{K}$, GPT-4o/Gemini for documentation understanding, and the strongest available model for relation understanding.
- Explaining why GPT-4o achieves only 4% Success@1 on RepoTransBench, which aligns with the bottleneck in code relation understanding discovered in this paper.

## Limitations & Future Work

- Currently only supporting Python, with a need to extend to other languages like Java and C++.
- Next-generation reasoning models such as DeepSeek-R1 and GPT-o3-mini were not evaluated due to API instability.
- Due to API limitations, DeepSeek-V2.5 was tested only up to 64K, failing to be evaluated on the full 128K range.
- Semantic relation extraction relies on a specific embedding model (`stella_en_400M_v5`), which may introduce bias.
- Code units are restricted to function granularity, without considering larger granularities like classes or modules.
- No end-to-end correlation analysis from comprehension to generation is covered.

## Related Work & Insights

- **vs RepoQA**: RepoQA only features a single needle function search task up to 16K, whereas ours contains 8 tasks covering up to 128K.
- **vs L-Eval**: L-Eval uses artificial long code concatenated from independent code snippets and ignores dependencies, while ours uses real repository code.
- **vs SWE-bench**: SWE-bench evaluates end-to-end capabilities integrating both comprehension and generation, while ours focuses on decoupling the understanding capability.
- **vs LongBench**: LongBench has an average length of only 0.4K, whereas ours averages 54.8K, which belongs to completely different magnitudes.

## Rating

- Novelty: ⭐⭐⭐ A benchmark paper with comprehensive and systematic task designs, but limited methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 models, 8 tasks, 5 length intervals, with sufficient ablation and human validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich charts, and practical rules of thumb.
- Value: ⭐⭐⭐⭐ Fills the gap in long code understanding evaluation; the discovery of the 32K cliff has direct instructing significance for model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](../../ACL2026/code_intelligence/sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [\[ACL 2025\] MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios](mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)

</div>

<!-- RELATED:END -->

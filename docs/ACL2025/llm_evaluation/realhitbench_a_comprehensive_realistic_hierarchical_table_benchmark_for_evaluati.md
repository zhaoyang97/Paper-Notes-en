---
title: >-
  [Paper Note] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis
description: >-
  [ACL2025][LLM Evaluation][benchmark] This paper proposes RealHiTBench, the first benchmark to comprehensively evaluate LLMs' capacity to understand complex hierarchical tables. It contains 708 real-world complex tables from 13 platforms across 24 domains and 3,752 questions. It defines 5 complex structural types and 5 major task types, and introduces TreeThinker, a tree-style reasoning pipeline that significantly enhances model understanding of hierarchical headers.
tags:
  - "ACL2025"
  - "LLM Evaluation"
  - "benchmark"
  - "hierarchical table"
  - "table reasoning"
  - "TreeThinker"
date: 2026-05-08
content_hash: 7cd3754fda2a727d
---

# RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis

**Conference**: ACL2025  
**arXiv**: [2506.13405](https://arxiv.org/abs/2506.13405)  
**Code**: [cspzyy/RealHiTBench](https://github.com/cspzyy/RealHiTBench)  
**Area**: LLM Evaluation  
**Keywords**: benchmark, hierarchical table, table reasoning, LLM evaluation, TreeThinker

## TL;DR

This paper proposes RealHiTBench, the first benchmark to comprehensively evaluate LLMs' capacity to understand complex hierarchical tables. It contains 708 real-world complex tables from 13 platforms across 24 domains and 3,752 questions. It defines 5 complex structural types and 5 major task types, and introduces TreeThinker, a tree-style reasoning pipeline that significantly enhances model understanding of hierarchical headers.

## Background & Motivation

**Tabular data is ubiquitous**: Economic, scientific, and employment fields widely organize multi-dimensional relational data using tables. Consequently, table analysis represents a crucial application scenario for LLMs.

**Limitations of Prior Work in existing benchmarks**: Mainstream benchmarks like TAT-QA, TableBench, and InfiAgent-DABench predominantly utilize "flat tables" (one attribute per column, one record per row). These fail to reflect the intricate hierarchical structures found in real-world applications.

**Hierarchical tables are underestimated**: While benchmarks like HiTab and SciTab consider hierarchical tables, they either focus on a single domain (e.g., science or aviation), accept only a single input format (e.g., images), feature insufficiently complex structures (hierarchies rarely exceed two levels), or offer limited task types.

**Inherent flaws of HiTab**: HiTab pre-extracts tree structures in a lossy JSON format, which prevents realistic evaluation of whether an LLM can directly comprehend structural information from raw table inputs. Moreover, it covers only 3 domains, features a single task type, and lacks complete supervision signals.

**Lack of multimodal evaluation**: Existing hierarchical table benchmarks focus exclusively on either text or images, lacking a unified framework to evaluate LLMs and MLLMs across multiple input formats (LaTeX/HTML/PNG).

**Absence of dedicated tests for structural comprehension**: Existing benchmarks do not incorporate tasks specifically designed to examine a model's ability to grasp complex tabular structures (such as nested sub-tables or implicit multi-table joins).

## Method

### Overall Architecture

RealHiTBench comprises two core components: (1) The benchmark dataset, consisting of 708 real-world complex tables harvested from 13 public platforms across 24 domains. It defines 5 complex structure types and 5 major task types (including fine-grained sub-types). GPT-4o auto-annotation combined with three rounds of manual validation yielded 3,752 questions, supporting LaTeX, HTML, and PNG input formats. (2) The TreeThinker reasoning pipeline, which organizes hierarchical headers into tree structures $\rightarrow$ aligns keywords with headers $\rightarrow$ localizes sub-tables $\rightarrow$ performs ReAct-style multi-turn reasoning to boost LLM comprehension of complex hierarchies.

### Key Design 1: Defining 5 Types of Complex Table Structures

- **Function**: It systematically defines and classifies 5 complex table structures from real-world scenarios: (1) hierarchical column headers (multi-level merged cells), (2) hierarchical row headers (indented or multi-column classification), (3) nested sub-tables (split by full-width cells), (4) multi-table joins (explicit and implicit multi-tables, where sub-tables with identical structures appear as a single table but are actually compared as multiple tables), and (5) miscellaneous (unstructured information like extra-tabular footnotes, cell background colors, etc.).
- **Core Idea**: Distill the essential sources of structural complexity from real-world tables to establish a quantifiable classification system, moving beyond the binary simplified judgment of "whether it is hierarchical."
- **Design Motivation**: Prior benchmarks lack granular definitions of table complexity. Simple descriptions of column hierarchies fail to capture real-world structural challenges like nested sub-tables and implicit multi-table joins. An accurate classification system is essential for fine-grained evaluation of structural comprehension.

### Key Design 2: Structure Comprehending Task Type

- **Function**: In addition to standard tasks like Fact Verification (FC), Numerical Reasoning (NR), Data Analysis (DA), and Chart Generation (CG), the benchmark introduces a "Structure Comprehending" (SC) task. This task perturbs (shuffles/transforms) complex parts of the source table to generate a new table, then poses identical questions to both tables to assess whether the LLM detects structural discrepancies and answers accurately.
- **Core Idea**: Isolate and evaluate the model's understanding of the table structure itself rather than its retrieve-and-calculate capacities by controlling variables (altering only the structure while keeping the content equivalent).
- **Design Motivation**: Conventional TableQA tasks can often be bypassed by "guessing" or "pattern matching" strategies. Only specifically designed structural contrast questions can genuinely expose a model’s deficiencies in understanding hierarchical structures.

### Key Design 3: TreeThinker Tree-Structured Reasoning Pipeline

- **Function**: It introduces a three-stage pipeline: (1) Tree Generation: prompts the model to encode headers into tuples $(flag, start, end, content)$ and assemble them into a tree structure; (2) Tree-based Reasoning: decomposes the question into keywords and aligns them with the header tree to localize the relevant sub-tables; (3) ReAct-Style Refinement: refines the answer through a sequence of Thought-Action-Result loops.
- **Core Idea**: Explicitly represent parent-child relationships within headers using tree structures, eliminating the perception blind spots of LLMs caused by flattened text inputs, while reducing noise through keyword alignment.
- **Design Motivation**: Experiments indicate that LLMs perform poorly on complex hierarchical tables primarily because they cannot automatically reconstruct hierarchical structures from linearized table text. TreeThinker overcomes this limitation by explicitly injecting structural information.

## Key Experimental Results

### Main Results: Performance of 25 Models on RealHiTBench

| Model | Input | FC-EM | NR-F1 | SC-F1 | DA-GPT | CG-PASS@1 |
|------|------|-------|-------|-------|--------|-----------|
| DeepSeek-R1 | Text | **70.91** | **72.54** | **84.62** | **42.59** | 7.14 |
| GPT-4o | Text | 60.31 | 50.12 | 71.14 | 36.36 | 20.13 |
| GPT-4o+TreeThinker | Text | 64.50 | 65.08 | 75.67 | 37.63 | **39.47** |
| Gemini-1.5-Pro | Text | 59.08 | 43.74 | 69.71 | 36.17 | 9.74 |
| Llama3.3-70B | Text | 53.08 | 48.99 | 68.93 | 27.98 | 24.03 |
| Qwen2.5-72B | Text | 51.93 | 39.23 | 68.34 | 35.90 | 14.29 |
| TableGPT2-7B | Text | 46.10 | 39.81 | 56.68 | 32.47 | 67.53 |
| GPT-4o | Image | 43.39 | 36.68 | 52.89 | 33.10 | 10.39 |

### Ablation Study: Contribution of Each TreeThinker Component

| Configuration | GPT-4o Avg | Δ | Llama3-70B Avg | Δ |
|------|-----------|---|----------------|---|
| TreeThinker (Full) | 63.29 | — | 62.23 | — |
| w/o Tree Generation | 55.27 | -8.02 | 54.04 | -8.19 |
| w/o Tree-based Reasoning | 60.75 | -2.54 | 54.58 | -7.65 |

### Key Findings

1. **Low overall performance**: Almost all models score under 70 in EM, and Chart Generation PASS@1 is generally below 30. Real-world hierarchical tables remain a major bottleneck for LLMs.
2. **Text outperforms image**: GPT-4o with text input outperforms image inputs by ~15 points on average, and Gemini-Pro by ~10 points. However, images can complement text (Image+Text achieves the highest accuracy).
3. **DeepSeek-R1 leads by a wide margin**: It dominates key metrics in FC, NR, and SC, demonstrating that strong reasoning capabilities significantly assist in hierarchical table understanding.
4. **TreeThinker significantly improves efficiency**: For GPT-4o, the PASS@1 metric in Chart Generation increases from 14.29 to 33.55 (+134.7%), and NR-F1 improves from 36.68 to 49.35. Tree Generation is the most critical component (performance drops by ~8 points without it).
5. **Table length remains a bottleneck**: For tables with >20K tokens, GPT-4o's score drops sharply from 56.45 to 30.77, with the visual modality being affected even more severely.
6. **Overfitting of table-specific models**: Table-specific fine-tuned models like TableLlama underperform compared to general LLMs on complex hierarchical tables.

## Highlights & Insights

1. **Comprehensiveness**: Simultaneously covers 5 complex structures, 5 major task types, dual-modality (text + image) evaluation, 13 data sources, and 24 domains, making it the most comprehensive benchmark in the field of hierarchical table understanding.
2. **Ingenious Design of Structure Comprehending**: By utilizing contrastive tests after structure transformation, the model's structural understanding capability is accurately isolated. This evaluation paradigm can be generalized to other structured data domains.
3. **Novel and Practical TreeThinker Method**: Explicitly encoding header hierarchies as tree structures and aligning them with question keywords is a simple yet effective approach that significantly improves performance without requiring additional training.
4. **Discovery of Implicit Multi-table Joins**: Identifies and defines the implicit join type ("seemingly a single table, actually multiple tables") for the first time, revealing a previously neglected difficulty in understanding.

## Limitations & Future Work

1. **Limited data scale**: The dataset of 708 tables and 3,752 questions is relatively small. The 6 annotators invested 540 hours each, resulting in high expansion costs.
2. **TreeThinker efficiency issues**: The multi-turn prompting strategy significantly increases inference costs, presenting a performance-efficiency trade-off.
3. **Unresolved long table handling**: 127 ultra-long tables cannot be fully imported in a single conversation context. The paper only identifies this problem without providing a solution.
4. **Potential annotation bias**: Although university student annotators were trained, they might still introduce errors when annotating tables from highly specialized fields (e.g., finance and science).

## Related Work & Insights

| Comparison Dimension | RealHiTBench | HiTab | TableBench |
|----------|-------------|-------|------------|
| Table Complexity | High (5 complex structures, median larger than other benchmarks) | Medium (pre-extracted JSON tree structure, only 2-level hierarchy) | Low (dominated by flat tables) |
| Task Type | FC/NR/DA/CG/SC (including structural comprehension) | Basic QA only | FC/NR/DA/CG (no structural comprehension) |
| Input Format | LaTeX/HTML/PNG (Text + Image) | Lossy JSON | Text only |
| Domain Coverage | 13 platforms, 24 domains | 3 domains | 6 platforms, 18 domains |
| Model Evaluation | 25 models (LLM + MLLM + Table-specific) | A few models | Multiple models but text-only |

**vs MultiHierTT**: MultiHierTT points to hierarchical tables but focuses solely on numerical reasoning with a single source (only listed company reports). RealHiTBench comprehensively outperforms it in task diversity, domain coverage, and structural complexity.

**vs TableVQA-Bench**: Although TableVQA supports image inputs, its table structures are simple (low hierarchy) and do not support comparative analysis with text inputs. RealHiTBench seamlessly supports dual-modality and focuses on complex structures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First benchmark to systematically define complex hierarchical table structures and design the Structure Comprehending task; the TreeThinker tree-reasoning approach is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive evaluation across 25 models, multimodal/multi-format inputs, complete ablation studies, and long table analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear definitions of complex structures, rich illustrations, and detailed description of the TreeThinker pipeline.
- **Value**: ⭐⭐⭐⭐ — Provides a much-needed, high-quality, and challenging benchmark for table understanding research, with TreeThinker offering direct practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](../../ACL2026/llm_evaluation/arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Radar: Benchmarking Language Models on Imperfect Tabular Data
description: >-
  [NeurIPS 2025][tabular reasoning] This paper introduces the Radar benchmark, which systematically evaluates language models' data-aware reasoning on imperfect tabular data by injecting five categories of data artifacts (…
tags:
  - "NeurIPS 2025"
  - "tabular reasoning"
  - "data awareness"
  - "data artifacts"
  - "language model benchmarking"
  - "robustness evaluation"
date: 2026-05-08
content_hash: 4730009a9cd0212e
---

# Radar: Benchmarking Language Models on Imperfect Tabular Data

**Conference**: NeurIPS 2025
**arXiv**: [2506.08249](https://arxiv.org/abs/2506.08249)
**Code**: [GitHub](https://github.com/kenqgu/RADAR) / [HuggingFace](https://huggingface.co/datasets/kenqgu/RADAR)
**Area**: Other
**Keywords**: tabular reasoning, data awareness, data artifacts, language model benchmarking, robustness evaluation

## TL;DR

This paper introduces the Radar benchmark, which systematically evaluates language models' data-aware reasoning on imperfect tabular data by injecting five categories of data artifacts (missing values, bad values, outliers, formatting inconsistencies, and logical inconsistencies) into real-world tables. The benchmark reveals that even frontier models suffer substantial performance degradation upon the introduction of data artifacts.

## Background & Motivation

Language models are increasingly deployed as autonomous data analysis agents for summarizing trends, identifying relationships, and manipulating tabular data. However, a critical capability has been overlooked: whether models possess **data awareness**—the ability to detect, reason about, and correctly handle data artifacts such as missing values, outliers, and logical inconsistencies.

Data quality issues are pervasive in real-world tabular data. In high-stakes scenarios (e.g., an erroneously recorded resting heart rate of 220 bpm in a medical record), failure to recognize such artifacts can lead to harmful or misleading conclusions. Existing tabular reasoning benchmarks largely assume clean data; even those that study structural perturbations (e.g., row/column shuffling) only test whether models rely on positional heuristics, rather than evaluating semantic understanding of data quality issues.

Furthermore, real-world tables often contain hundreds or thousands of rows, whereas existing benchmarks typically use small tables without controlling for table size. Radar addresses both gaps: it systematically evaluates artifact-handling capabilities and controls table size to study how reasoning performance varies with input complexity.

## Method

### Overall Architecture

The core idea of Radar is as follows. Given a clean source table $T$, a programmatic perturbation function $g:(T, Q) \mapsto (T_p, T_r)$ generates a perturbed table $T_p$ and a corresponding recovered table $T_r$. During evaluation, the model receives $T_p$ and a query $Q$; the ground-truth answer is defined as $A = f(Q, T_r)$, meaning the model must autonomously identify and handle data artifacts to arrive at the correct answer. Computing directly on $T_p$ yields an incorrect result.

### Key Designs

1. **Five categories of data artifacts**:

    - **Missing Data**: valid cells replaced with null values
    - **Bad Values**: injection of obviously erroneous or placeholder values (e.g., -1, 9999, TEST, #REF!)
    - **Outliers**: insertion of extreme, implausible numerical values (e.g., a resting heart rate of 220 bpm)
    - **Inconsistent Formatting**: multiple representations of the same datum (e.g., "22 lbs" vs. "22 pounds" vs. "weight = 22")
    - **Inconsistent Logic**: cross-field contradictions (e.g., an end time earlier than the start time, or BMI inconsistent with recorded height and weight)

2. **Programmatic perturbation framework**: Each perturbation function is designed for a specific query, ensuring that direct computation on the perturbed table yields an incorrect result. Perturbations affect no more than 10% of rows, preserving the overall distributional integrity of the data. Both the answer function $f$ and the perturbation function $g$ operate across tables of varying sizes, provided that the core fields $\mathcal{C}$ are present.

3. **Multi-dimensional table size control**: Table size is measured in tokens as $\tau \in \{2K, 4K, 8K, 16K\}$, with the number of columns controlled as $c \in \{5, 10, 20\}$. For a given $(\tau, c)$ combination, the number of rows is selected as $R = \arg\min_r |tok(T_s, r, c) - \tau|$. This enables systematic study of how table size affects reasoning performance while holding semantic content and complexity constant.

### Dataset Construction

Fifty-three tasks were crowdsourced by 12 data science experts from 27 source tables spanning 9 application domains. Experts authored 260 perturbation functions, which underwent multiple rounds of code review and cross-validation. The final Radar benchmark contains 2,980 task instances. Two subsets are provided: Radar-T (53 tasks standardized to 10 columns / 8K tokens) and Radar-S (10 tasks with the full size-variation configuration).

### Evaluation Setup

Two baselines are evaluated: (1) **Direct Prompting**, where the model performs text-based reasoning directly; and (2) **Code Agent**, where the model is equipped with a Python shell tool. System prompts explicitly instruct the model to attend to the five artifact categories, without reference to specific table instances. Exact match (EM) is used as the primary metric.

## Key Experimental Results

### Main Results

| Model | Clean (Direct) | Artifact Avg (Direct) | Clean (Code) | Artifact Avg (Code) |
|------|------|------|------|------|
| StructLM (table fine-tuned) | 2.3 | 0.8 | - | - |
| TableGPT2 (table fine-tuned) | 0.8 | 1.1 | 35.0 | 6.9 |
| Gemma3 27B | 1.9 | 2.3 | 75.5 | 13.6 |
| DeepSeek-V3 | 1.9 | 3.5 | 96.2 | 37.6 |
| GPT-4.1 | 17.0 | 14.3 | 98.1 | 48.6 |
| Gemini 2.5 Flash Thinking | 39.6 | 19.9 | 88.7 | 43.3 |
| DeepSeek-R1 | 34.0 | 26.3 | 84.9 | 52.8 |
| o3-mini (high) | 73.6 | 35.6 | 75.5 | 44.9 |
| Gemini 2.5 Pro | 71.7 | 50.9 | 84.9 | 62.3 |
| **o4-mini (high)** | **83.0** | **53.9** | **100** | **61.8** |

Under the Code Agent setting, o4-mini achieves 100% exact match on clean tables, yet performance drops by approximately **59%** upon the introduction of data artifacts.

### Analysis by Artifact Type

| Artifact Type | o4-mini (Direct) | o4-mini (Code) | Gemini 2.5 Pro (D) | Gemini 2.5 Pro (C) |
|------|------|------|------|------|
| Missing | 49.1 | 50.9 | 50.9 | 73.6 |
| Bad Values | 58.5 | 54.7 | 56.6 | 54.7 |
| Outliers | 56.2 | 83.3 | 47.9 | 64.6 |
| Formatting | 73.6 | 79.2 | 56.6 | 73.6 |
| **Logic** | **32.1** | **41.1** | **42.9** | **44.6** |

**Logical inconsistency is the most challenging artifact type**, with all models performing worst in this category.

### Key Findings

- **Code execution is not a panacea**: Although code agents perform near-perfectly on clean tables, they still exhibit significant performance gaps when facing data artifacts, particularly for logical inconsistencies.
- **Effect of table size**: Under direct prompting, performance degrades monotonically as token count increases, approaching zero at 16K tokens; code agent performance is largely unaffected by table size.
- **Wide vs. narrow tables**: At equivalent token counts, models perform better on wide tables (more columns, fewer rows) than on narrow ones, as models tend to inspect data row by row, and increasing row count linearly amplifies computational demand.
- **Value imputation vs. row discarding**: Code agents show marked improvement on tasks requiring row discarding, but offer limited gains on tasks requiring cross-column value derivation.

## Highlights & Insights

- The paper surfaces a neglected but critically important issue: LLM performance on clean data is a poor proxy for real-world capability.
- The programmatic perturbation framework is elegantly designed, supporting automated generation of large-scale, verifiable evaluation instances.
- Logical inconsistency is the hardest artifact type to detect, as it requires cross-column and cross-row reasoning.
- The findings have direct implications for building reliable data science agents: dedicated data quality detection modules are essential.

## Limitations & Future Work

- Only five independent perturbation types are considered; scenarios involving multiple co-occurring artifacts are not studied.
- The benchmark requires a unique correct remediation action, excluding cases where multiple reasonable corrections coexist.
- Perturbation functions are manually authored by experts, requiring additional human effort to extend to new domains.
- The paper does not explore whether explicitly informing models that the data "may contain errors" leads to improved performance.

## Related Work & Insights

- In contrast to works such as ROBUT and BIG-Bench Extra Hard that apply only structural perturbations, Radar demands semantic-level data understanding.
- Radar is complementary to data analysis benchmarks such as BLADE and DABench: the latter evaluate analytical capability, while Radar evaluates data quality awareness.
- Implication: future LLM-based agents should incorporate data validation and cleaning as the first step of the analysis pipeline.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First tabular benchmark to systematically evaluate data-aware reasoning in LLMs
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 11 models, two paradigms, and multi-dimensional analyses
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear; framework is described in detail
- **Value**: ⭐⭐⭐⭐⭐ — Significant contribution to reliability evaluation of LLM-as-data-analyst systems, exposing a critical capability gap

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Product Distribution Learning with Imperfect Advice](product_distribution_learning_with_imperfect_advice.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](../../ACL2026/others/are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](adaptive_data_analysis_for_growing_data.md)
- [\[NeurIPS 2025\] Note 4: WebThinker — Empowering Reasoning Models with Deep Research Capabilities](webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)

</div>

<!-- RELATED:END -->

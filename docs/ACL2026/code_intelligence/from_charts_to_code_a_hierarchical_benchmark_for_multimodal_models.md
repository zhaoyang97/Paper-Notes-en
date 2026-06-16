---
title: >-
  [Paper Note] From Charts to Code: A Hierarchical Benchmark for Multimodal Models
description: >-
  [ACL 2026][Code Intelligence][Paper Note] This paper proposes Chart2Code, a hierarchical benchmark containing 2,186 tasks covering 22 chart types. It is divided into three progressive difficulty levels: Chart Replication (Level 1), Chart Editing (Level 2), and Long Table-to-Chart (Level 3). Evaluations of 29 SOTA multimodal models reveal that even the stronges
tags:
  - ACL 2026
  - Code Intelligence
date: 2026-05-08
content_hash: 26816bcd99558486
---
# From Charts to Code: A Hierarchical Benchmark for Multimodal Models

**Conference**: ACL 2026  
**arXiv**: [2510.17932](https://arxiv.org/abs/2510.17932)  
**Code**: [GitHub](https://github.com/CSU-JPG/Chart2Code)  
**Area**: Code Intelligence / Multimodal Understanding  
**Keywords**: Chart Generation, Code Generation, Multimodal Benchmark, Hierarchical Evaluation, Visual Fidelity

## TL;DR

This paper proposes Chart2Code, a hierarchical benchmark containing 2,186 tasks covering 22 chart types. It is divided into three progressive difficulty levels: Chart Replication (Level 1), Chart Editing (Level 2), and Long Table-to-Chart (Level 3). Evaluations of 29 SOTA multimodal models reveal that even the strongest GPT-5.2 achieves a chart quality score of only 33.41 on editing tasks, uncovering significant deficiencies in current models for practical chart code generation.

## Background & Motivation

**Background**: Charts are essential visualization tools in scientific papers and business reports. With the rapid development of Large Multimodal Models (LMMs), AI systems can now not only understand charts but also generate executable plotting code (chart-to-code), promising significant productivity gains.

**Limitations of Prior Work**: (1) Existing benchmarks (e.g., ChartMimic) are reaching performance saturation—GPT-4o achieves 82.2% on ChartMimic, failing to differentiate current and future model capabilities; (2) Lack of systematic benchmarks covering real-world usage scenarios—users need not only to replicate charts but more frequently to edit them (changing types, adding elements) and generate charts from raw long tables, scenarios that are under-tested; (3) Existing evaluations primarily focus on code correctness, ignoring the visual fidelity of the rendered charts.

**Key Challenge**: There is a massive gap between the high scores reported by existing benchmarks and model performance in practical use—models score high on simple replication but show a sharp decline in common editing and data-to-chart scenarios. Existing benchmarks fail to expose these issues.

**Goal**: (1) Construct a hierarchical chart code generation benchmark from a user perspective; (2) Design a multi-level evaluation protocol (code-level + chart-level); (3) Comprehensively evaluate 29 SOTA multimodal models.

**Key Insight**: Design three progressive difficulty levels based on real-world user workflows—simple replication → complex editing → long table-to-chart—gradually increasing requirements for model understanding, reasoning, and code generation.

**Core Idea**: Extend chart code generation from a single replication task to a hierarchical benchmark covering the complete user workflow, while introducing dual-layer evaluation (code-level and chart-level) to measure generation quality comprehensively.

## Method

### Overall Architecture

Chart2Code formalizes chart code generation as $C = f(R, I, D)$: given a reference chart $R$, a natural language instruction $I$, and an optional data source $D$ (supporting text, table screenshots, or Excel), the multimodal model $f$ generates executable Python plotting code $C$. The benchmark follows the real user workflow across three levels: "Replication → Editing → Long Table-to-Chart." It employs dual-layer scoring: code-level (rule-based Base score + LLM score) and chart-level (LMM visual score + human evaluation) to expose "correct code but wrong chart" discrepancies.

### Key Designs

**1. Three-level Hierarchical Task Design: Deconstructing User Workflow into Progressive Difficulty**

Existing benchmarks only test simple replication, whereas real user needs involve "view-to-draw → edit → data-to-draw." Chart2Code splits tasks accordingly: Level 1 (Chart Replication) includes Direct Reproduction (DR, vision only) and Style Transfer (CRD/CFD, vision + data), totaling 863 tasks; Level 2 (Chart Editing) requires complex modifications like changing chart types, adding trend lines, calculating correlation coefficients, or splitting by category, totaling 1,010 tasks; Level 3 (Long Table-to-Chart) requires extracting, calculating, and then plotting from raw long tables averaging 2,647 rows, totaling 313 tasks. Difficulty steps up across understanding, reasoning, and long-context retrieval, targeting the current capability boundaries of models.

**2. Multi-level Evaluation Protocol: Correct Code Does Not Mean Correct Chart**

The same data can be rendered with different code paths resulting in massive visual differences; thus, relying solely on code similarity overestimates quality. Evaluation follows three paths: Code-level Base scoring parses Figure objects (e.g., Matplotlib) to provide rule-based scoring across 8 dimensions (Color, Grid, Layout, Legend, Visual Elements, Data, Text, Type)—more comprehensive and accurate than prior work; Code-level LLM scoring uses GPT-5-mini to evaluate visual fidelity from the code; Chart-level LMM scoring uses GPT-5-mini to compare multi-dimensional similarity between GT and generated charts, supported by human evaluation to validate LMM consistency.

**3. Rich Data Diversity: Preventing Single-capability Gaming**

The benchmark covers 22 chart types (radar, heatmap, scatter, boxplot, tree map, error bar, pie, violin, etc.) and strengthens granularity at each level: Level 1 emphasizes chart uniqueness (719 unique charts); Level 2 provides at least one editing instruction per chart (1,010 unique edits, with the longest instructions averaging 267 words); Level 3 includes 71 Excel files (up to 30,427 rows). This multi-type, multi-task coverage ensures the benchmark cannot be easily solved by models with specific narrow skills.

### Loss & Training

As this is a benchmarking study, no model training was performed. All models were evaluated using public weights or APIs. Open-source models were run on NVIDIA V100 GPUs, non-thinking model inference length was set to 4,096 tokens, and images were input at original resolutions.

## Key Experimental Results

### Main Results

**Level 1 Chart Replication (Selected Models)**

| Model | DR Exec% | DR Base | DR LMM | CRD Base | CFD Base |
|------|----------|---------|--------|----------|----------|
| GPT-5.2 | 97.08 | 79.91 | 43.73 | 66.31 | 73.02 |
| Gemini-3-Pro | 97.50 | 78.65 | 45.42 | 69.23 | 70.78 |
| Claude-Sonnet-4 | 96.52 | 65.60 | 32.36 | 61.46 | 65.27 |
| Qwen3-VL-32B | - | - | - | - | - |
| InternVL-3-38B | 85.26 | 53.57 | 16.68 | 58.17 | 60.17 |

**Level 2 Chart Editing (8-dimension Code-level Score)**

| Model | Exec% | Color | Data | Text | Type | Base | LMM |
|------|-------|-------|------|------|------|------|-----|
| GPT-5.2 | 96.04 | 58.44 | 64.66 | 83.77 | 94.52 | 70.93 | 33.03 |
| Gemini-3-Pro | 97.23 | 52.32 | 62.75 | 77.16 | 93.86 | 70.78 | 33.41 |
| Claude-Sonnet-4 | 90.20 | 47.17 | 54.88 | 80.52 | 93.29 | 63.65 | 25.40 |

### Ablation Study

**Gap between Code-level and Chart-level Scores**

| Model | Level 2 Code-level Base | Level 2 Chart-level LMM | Gain |
|------|-------------------|-------------------|------|
| GPT-5.2 | 70.93 | 33.03 | -37.90 |
| Gemini-3-Pro | 70.78 | 33.41 | -37.37 |
| Claude-Sonnet-4 | 63.65 | 25.40 | -38.25 |

**Open-source Thinking vs Non-thinking (Level 2, MiMo-VL-7B)**

| Configuration | Base | LMM |
|------|------|-----|
| MiMo-VL-7B-SFT (non-thinking) | 63.99 | 22.61 |
| MiMo-VL-7B-RL (non-thinking) | 64.41 | 21.43 |
| MiMo-VL-7B-SFT (thinking) | 65.49 | 16.95 |
| MiMo-VL-7B-RL (thinking) | 70.45 | 30.04 |

### Key Findings

- Performance drops significantly for all models moving from Level 1 to Level 2/3—even GPT-5.2 scores only 33.03 (LMM) on editing tasks, indicating that editing and data-to-chart are far from solved.
- There is a massive gap (~37-38 points) between code-level and chart-level scores, suggesting code correctness does not guarantee visual fidelity.
- Closed-source models (GPT-5.2, Gemini-3-Pro) significantly outperform open-source models at all levels; among open-source models, Qwen3-VL-32B and InternVL-3.5-38B perform best.
- Thinking mode helps some models (e.g., MiMo-VL-7B-RL thinking LMM 30.04 vs non-thinking 21.43), but not all models benefit.
- Level 3 long-table tasks are extremely challenging, requiring a synthesis of long-context understanding, information retrieval, mathematical calculation, and code generation.

## Highlights & Insights

- The hierarchical design accurately maps to the three stages of user workflows—transitioning from "view-to-draw" to "modify" to "data-to-draw"—revealing true capability boundaries.
- The discovery of the large gap between code-level and chart-level evaluation is crucial—it warns the community not to rely solely on code similarity to evaluate chart generation quality.
- The 8-dimension rule-based scoring is transferable to other chart or code generation tasks.

## Limitations & Future Work

- Evaluation depends on GPT-5-mini as a judge, which may introduce LMM evaluation bias.
- Annotation and GT code construction for Level 3 were extremely time-consuming, limiting the data scale (313 tasks).
- Evaluated only Python/Matplotlib ecosystems; did not cover R, D3.js, or other plotting tools.
- Did not test iterative correction capabilities—real-world users often interact over multiple rounds to fix charts.

## Related Work & Insights

- **vs ChartMimic**: ChartMimic only tests replication (Level 1) and is saturated at 82.2% by GPT-4o; Chart2Code extends to editing and long-table scenarios where SOTA is only ~33 points.
- **vs ChartEdit**: ChartEdit only covers simple local edits (233 samples); Chart2Code’s Level 2 includes 1,010 complex editing tasks (e.g., adding trend lines).
- **vs Plot2Code**: Plot2Code has only 132 samples and lacks rule-based evaluation; Chart2Code is significantly more comprehensive in scale and evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ First hierarchical chart code generation benchmark; task design aligns with real user needs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 29 models, multi-level protocol, validated by human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions and rich statistics, though many tables require careful cross-referencing.
- Value: ⭐⭐⭐⭐⭐ Identifies true bottlenecks in chart code generation and sets a clear direction for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](../../CVPR2026/code_intelligence/geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[NeurIPS 2025\] Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models](../../NeurIPS2025/code_intelligence/table2latex-rl_high-fidelity_latex_code_generation_from_table_images_via_reinfor.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[AAAI 2026\] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings](../../AAAI2026/code_intelligence/mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)

</div>

<!-- RELATED:END -->

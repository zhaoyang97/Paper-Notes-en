---
title: >-
  [Paper Note] From Charts to Code: A Hierarchical Benchmark for Multimodal Models
description: >-
  [ACL 2026][Multimodal VLM][Chart generation] This paper proposes Chart2Code, a hierarchical benchmark featuring 2,186 tasks across 22 chart types. It is organized into three progressive difficulty levels: Chart Replication (Level 1), Chart Editing (Level 2), and Long Table-to-Chart (Level 3). Evaluating 29 SOTA multimodal models reveals that even the strongest GPT-5.2 achieves a chart quality score of only 33.41 on editing tasks, highlighting significant deficiencies in curre…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Chart generation"
  - "Code generation"
  - "Multimodal benchmark"
  - "Hierarchical evaluation"
  - "Visual fidelity"
date: 2026-05-08
content_hash: 57e1f2a160e08e8a
---

# From Charts to Code: A Hierarchical Benchmark for Multimodal Models

**Conference**: ACL 2026  
**arXiv**: [2510.17932](https://arxiv.org/abs/2510.17932)  
**Code**: [GitHub](https://github.com/CSU-JPG/Chart2Code)  
**Area**: Code Intelligence / Multimodal Understanding  
**Keywords**: Chart generation, Code generation, Multimodal benchmark, Hierarchical evaluation, Visual fidelity

## TL;DR

This paper proposes Chart2Code, a hierarchical benchmark featuring 2,186 tasks across 22 chart types. It is organized into three progressive difficulty levels: Chart Replication (Level 1), Chart Editing (Level 2), and Long Table-to-Chart (Level 3). Evaluating 29 SOTA multimodal models reveals that even the strongest GPT-5.2 achieves a chart quality score of only 33.41 on editing tasks, highlighting significant deficiencies in current models for practical chart code generation.

## Background & Motivation

**Background**: Charts are essential visualization tools in scientific papers and business reports. With the rapid development of Large Multimodal Models (LMMs), AI systems can now not only understand charts but also generate executable chart-to-code, which promises to significantly enhance productivity.

**Limitations of Prior Work**: (1) Existing benchmarks (e.g., ChartMimic) are reaching performance saturation—GPT-4o has reached 82.2% on ChartMimic, making it difficult to distinguish between current and future models. (2) There is a lack of systematic benchmarks covering real-world usage scenarios; users require more than mere replication, often needing to edit charts or generate charts from raw long tables. (3) Current evaluations focus primarily on code correctness while neglecting the visual fidelity of the rendered charts.

**Key Challenge**: A massive gap exists between the high scores reported on existing benchmarks and actual model performance in practical scenarios. Models score high on simple replication but show significant performance drops in more common editing and data-to-chart scenarios, which existing benchmarks fail to expose.

**Goal**: (1) Construct a hierarchical chart code generation benchmark from a user perspective; (2) Design a multi-level evaluation protocol (code-level + chart-level); (3) Comprehensively evaluate 29 SOTA multimodal models.

**Key Insight**: Design three progressive difficulty levels based on real-world user workflows—Simple Replication → Complex Editing → Long Table-to-Chart—to incrementally increase requirements for model understanding, reasoning, and code generation.

**Core Idea**: Expand chart code generation from a single replication task to a hierarchical benchmark covering the complete user workflow, while introducing dual-layer evaluation (code-level and chart-level) to measure generation quality comprehensively.

## Method

### Overall Architecture

Chart2Code formalizes chart code generation as $C = f(R, I, D)$: given a reference chart $R$, a natural language instruction $I$, and an optional data source $D$ (supporting text, table screenshots, or Excel), the multimodal model $f$ generates executable Python plotting code $C$. The benchmark follows the user workflow through "Replication → Editing → Long Table-to-Chart" difficulty tiers and evaluates outputs using dual-layer scoring: code-level (rule-based Base score + LLM score) and chart-level (LMM visual score + human evaluation) to expose "correct code but incorrect chart" discrepancies.

### Key Designs

**1. Three-tier Hierarchical Task Design: Decomposing User Workflow into Increasing Difficulty**

Existing benchmarks only test simple replication. Real user needs follow "see chart, draw chart → modify chart → see data, draw chart." Chart2Code accordingly defines three levels: Level 1 Chart Replication includes Direct Replication (DR) and style transfer (CRD/CFD), totaling 863 tasks; Level 2 Chart Editing requires complex modifications like changing chart types, adding trend lines, or splitting by categories, totaling 1,010 tasks; Level 3 Long Table-to-Chart requires extraction, calculation, and plotting from tables averaging 2,647 rows, totaling 313 tasks. Difficulty scales with requirements for understanding, reasoning, and long-context retrieval.

**2. Multi-level Evaluation Protocol: Code Correctness Does Not Equal Chart Correctness**

The same data can be rendered with different code paths resulting in vast visual differences; thus, relying solely on code similarity overestimates quality. Evaluation is three-pronged: Code-level Base scoring parses Figure objects (e.g., Matplotlib) across 8 dimensions (color, grid, layout, legend, visual elements, data, text, type), which is more comprehensive than ChartMimic's 4 dimensions; Code-level LLM scoring uses GPT-5-mini to evaluate visual fidelity from a code perspective; Chart-level LMM scoring use GPT-5-mini to compare multi-dimensional similarity between GT and generated charts, supported by human evaluation to verify consistency.

**3. Rich Data Diversity: Preventing Single-Ability Score Gaming**

The benchmark covers 22 chart types (radar, heatmaps, scatter, box plots, tree maps, error bars, pie charts, violin plots, etc.). Each level emphasizes different strengths: Level 1 focuses on chart uniqueness (719 unique charts); Level 2 provides at least one complex editing instruction per chart (1,010 unique edits, average instruction length of 267 words); Level 3 includes 71 Excel files (reaching up to 30,427 rows). This broad coverage ensures the benchmark cannot be easily bypassed by models with singular capabilities.

### Loss & Training

This study focus on benchmarking and does not involve model training. All models were evaluated using public weights or APIs. Open-source models were run on NVIDIA V100 GPUs, with inference lengths for non-thinking models set to 4,096 tokens, and images were input at their original resolution.

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

| Model | Level 2 Code-level Base | Level 2 Chart-level LMM | Gap |
|------|-------------------|-------------------|------|
| GPT-5.2 | 70.93 | 33.03 | -37.90 |
| Gemini-3-Pro | 70.78 | 33.41 | -37.37 |
| Claude-Sonnet-4 | 63.65 | 25.40 | -38.25 |

**Open-source Models Thinking vs Non-thinking (Level 2, MiMo-VL-7B)**

| Configuration | Base | LMM |
|------|------|-----|
| MiMo-VL-7B-SFT (non-thinking) | 63.99 | 22.61 |
| MiMo-VL-7B-RL (non-thinking) | 64.41 | 21.43 |
| MiMo-VL-7B-SFT (thinking) | 65.49 | 16.95 |
| MiMo-VL-7B-RL (thinking) | 70.45 | 30.04 |

### Key Findings

- Performance for all models drops significantly from Level 1 to Levels 2/3. Even GPT-5.2 scores only 33.03 on chart quality (LMM) for editing tasks, suggesting editing and data-to-chart tasks are far from solved.
- A huge gap (approx. 37-38 points) exists between code-level and chart-level scores, indicating that code correctness does not guarantee visual fidelity.
- Closed-source models (GPT-5.2, Gemini-3-Pro) lead significantly across all levels. Among open-source models, Qwen3-VL-32B and InternVL-3.5-38B perform best.
- Thinking mode helps some models (e.g., MiMo-VL-7B-RL thinking LMM 30.04 vs non-thinking 21.43), but the benefit is not universal.
- Level 3 long table tasks are extremely challenging, requiring a combination of long-context understanding, information retrieval, mathematical calculation, and code generation.

## Highlights & Insights

- The hierarchical design precisely corresponds to the three stages of user workflow—from replication to modification to data-driven generation—revealing the true capability boundaries of models.
- The discovery of the massive gap between code-level and chart-level evaluation is significant, warning the community against relying solely on code similarity to evaluate chart generation quality.
- The 8-dimensional rule-based scoring protocol is transferable to other chart or code generation tasks.

## Limitations & Future Work

- Evaluation relies on GPT-5-mini as a judge, which may introduce LMM evaluation bias.
- Annotation and GT code construction for Level 3 are extremely time-consuming, limiting the data scale to 313 tasks.
- Only the Python/Matplotlib ecosystem was evaluated; other tools like R or D3.js were not covered.
- The models' iterative self-correction capabilities were not tested, which is common in actual multi-turn user interactions.

## Related Work & Insights

- **vs ChartMimic**: ChartMimic only tests replication (Level 1) and is near saturation with GPT-4o at 82.2%. Chart2Code extends to editing and long-table scenarios where current SOTA scores only around 33 points.
- **vs ChartEdit**: ChartEdit only covers simple local edits (233 samples). Chart2Code’s Level 2 includes 1,010 complex editing tasks (e.g., adding trend lines, calculating correlation coefficients).
- **vs Plot2Code**: Plot2Code has only 132 samples and lacks rule-based evaluation. Chart2Code is significantly more comprehensive in both scale and evaluation framework.

## Rating

- Novelty: ⭐⭐⭐⭐ First hierarchical chart-to-code benchmark; task design aligns with real-world user needs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 29 models, multi-level protocols, and human validation.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions and rich statistics, though many tables require careful cross-referencing.
- Value: ⭐⭐⭐⭐⭐ Identifies the true bottlenecks in chart code generation and provides a clear direction for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChartDiff: A Large-Scale Benchmark for Comprehending Pairs of Charts](chartdiff_a_large-scale_benchmark_for_comprehending_pairs_of_charts.md)
- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](../../ACL2025/multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)
- [\[ACL 2026\] Aligned Multi-View Scripts for Universal Chart-to-Code Generation](aligned_multi-view_scripts_for_universal_chart-to-code_generation.md)
- [\[ACL 2026\] MSEarth: A Multimodal Benchmark for Earth Science Phenomenon Discovery with MLLMs](msearth_a_multimodal_benchmark_for_earth_science_phenomenon_discovery_with_mllms.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)

</div>

<!-- RELATED:END -->

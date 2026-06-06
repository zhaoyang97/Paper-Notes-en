---
title: >-
  [Paper Note] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark
description: >-
  [NeurIPS 2025][Robotics][table understanding] This paper introduces MMTU, a large-scale benchmark comprising 28,136 questions spanning 25 real-world table tasks…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "table understanding"
  - "benchmark"
  - "LLM evaluation"
  - "multi-task reasoning"
  - "structured data"
date: 2026-05-08
content_hash: 1cde311b9fa7198e
---

# MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark

**Conference**: NeurIPS 2025
**arXiv**: [2506.05587](https://arxiv.org/abs/2506.05587)  
**Code**: [Available](https://github.com/MMTU-Benchmark/MMTU)  
**Area**: Robotics
**Keywords**: table understanding, benchmark, LLM evaluation, multi-task reasoning, structured data

## TL;DR

This paper introduces MMTU, a large-scale benchmark comprising 28,136 questions spanning 25 real-world table tasks, designed to systematically evaluate LLMs on professional-level table understanding, reasoning, and manipulation. Even frontier reasoning models such as GPT-5 achieve only approximately 69.6% on this benchmark.

## Background & Motivation

Tabular data is central to real-world applications such as spreadsheets, databases, and computational notebooks, and has traditionally required domain experts such as data engineers and DBAs to operate. Although LLMs have demonstrated considerable promise on table tasks, existing evaluations suffer from severe limitations.

**First, task coverage is too narrow.** Existing table benchmarks focus predominantly on NL-to-SQL and Table-QA, neglecting a large body of specialized table tasks accumulated over decades of computer science research—such as table transformation, entity matching, data cleaning, and column relationship discovery.

**Second, existing benchmarks are limited in scale.** Spreadsheet benchmarks, for instance, typically contain only a few hundred test cases and are tied to specific file formats. Compared to NLP benchmarks such as MMLU (15,908 questions) and MMMU (11,550 questions), the table domain lacks a comparably comprehensive evaluation suite.

**Third, existing NLP benchmarks are prone to saturation.** Benchmarks such as GSM8k and HumanEval have been largely solved by frontier models, necessitating more challenging benchmarks to sustain progress.

MMTU is designed to fill this gap by constructing a large-scale, diverse, and challenging benchmark covering professional-level table tasks.

## Method

### Overall Architecture

MMTU is a carefully curated benchmark rather than a new model. Its construction pipeline consists of five stages: literature survey → task selection → data standardization → quality verification → expert validation.

### Key Designs

1. **Comprehensive coverage across 25 tasks and 7 categories.** Tasks are systematically drawn from two decades of research in the data management (SIGMOD/VLDB), programming languages (PLDI/POPL), and Web data (WWW/WSDM) communities. The final selection comprises 25 user-facing, objectively evaluable tasks grounded in real data. The seven categories include: Table Transform, Table Matching, Data Cleaning, Table Join, Column Transform, Column Relationship, Table Understanding, NL-to-Code, Table QA, and KB Mapping. The majority of these tasks have never previously been used to evaluate foundation models.

2. **Standardized triplet format.** All 28,136 questions are unified into the format $\langle\text{Instruction, Input-Table(s), Ground-truth}\rangle$, enabling plug-and-play evaluation across different LLMs. The questions involve 61,763 real tables (74.9% Web tables, 7.4% spreadsheet tables, 17.7% relational tables). A total of 26.1% of questions require code generation (SQL/Python Pandas/formulas).

3. **Multi-layer quality assurance.** (a) o4-mini is used to check for ambiguity and correctness, removing 8% of questions; (b) LLM-based screening for privacy and safety risks; (c) a per-dataset cap of 1,000 questions to ensure diversity; (d) 20 questions per task are randomly sampled for manual validation by domain experts.

4. **Flexible evaluation framework.** Unlike multiple-choice benchmarks such as MMLU, MMTU adopts open-form structured answers, supporting execution-based evaluation (SQL/Python code execution) and structured output evaluation (e.g., unordered JSON list comparison), more closely reflecting real-world scenarios.

### Loss & Training

Not applicable (benchmark paper).

## Key Experimental Results

### Main Results

**Overall model performance (Table 3, selected):**

| Model Type | Model | MMTU Score | Cost per Question ($) |
|------------|-------|-----------|----------------------|
| Reasoning | GPT-5 | 0.696 | 0.01727 |
| Reasoning | o3 | 0.691 | 0.01539 |
| Reasoning | GPT-5-mini | 0.667 | 0.00276 |
| Reasoning | Gemini-2.5-pro | 0.665 | 0.00790 |
| Reasoning | DeepSeek-R1 | 0.579 | 0.00167 |
| Reasoning | Qwen3-32B | 0.506 | 0.00017 |
| Chat | GPT-5-Chat | 0.577 | 0.00534 |
| Chat | DeepSeek-V3 | 0.555 | 0.00095 |
| Chat | GPT-4o | 0.507 | 0.01019 |
| Chat | Llama-3.3-70B | 0.454 | 0.00150 |

### Ablation Study

**Reasoning vs. chat model comparison:**

| Dimension | Reasoning (Best) | Chat (Best) | Gap |
|-----------|-----------------|-------------|-----|
| Best score | 69.6% (GPT-5) | 57.7% (GPT-5-Chat) | +11.9pp |
| Best cost-efficiency | Qwen3-32B (0.506, $0.00017) | — | — |

### Key Findings

1. **Reasoning models substantially outperform chat models.** The best reasoning model surpasses the best chat model by over 10 percentage points, indicating that MMTU tasks inherently require coding and logical reasoning capabilities.
2. **Frontier models still have substantial room for improvement.** Even GPT-5 achieves only 69.6%, remaining well below human expert performance.
3. **Frontier models are relatively insensitive to table serialization format.** Markdown/CSV/JSON/HTML formats have little impact, reflecting improved model ability to handle diverse data formats.
4. **LLMs still struggle with long table contexts.** Complex tasks requiring holistic cross-cell reasoning over tables with many rows or columns—especially in the column direction—remain particularly challenging.
5. **Table-level perturbations (row/column shuffling) degrade performance.** Even semantically invariant transformations expose robustness deficiencies in model table understanding.
6. **Newer and larger models significantly outperform older and smaller ones.** This suggests that foundation models are making rapid progress in table-related capabilities.

## Highlights & Insights

- **Fills a critical gap.** MMTU is the first benchmark to incorporate the specialized table tasks accumulated over decades in the data management and PL communities into the LLM evaluation ecosystem; the majority of its 25 tasks have never been used to evaluate foundation models.
- **Comparable in scale to MMLU.** With 28,136 questions, MMTU is the most comprehensive benchmark in the table domain.
- **Emphasis on rigor and fairness.** Multi-layer quality assurance—from automated checks to expert validation—combined with privacy and safety screening ensures evaluation integrity.
- **Reasoning capability identified as the bottleneck.** The substantial advantage of reasoning models over chat models demonstrates that simple pattern matching is insufficient for these tasks.

## Limitations & Future Work

- Only objectively evaluable tasks are included; subjective or creative tasks such as table summarization and table augmentation are excluded.
- Tasks are sampled solely from existing research literature, potentially omitting practically important but underrepresented scenarios such as multi-turn table manipulation.
- The current benchmark considers only text-based input and does not account for multimodal (e.g., visual table) inputs.
- The difficulty distribution across tasks is uneven, and some tasks may already be largely solved by current models.

## Related Work & Insights

- Analogous to the role MMLU has played in advancing NLP, MMTU has the potential to become the central evaluation benchmark for table AI.
- It complements coding benchmarks such as SWE-bench by covering data-processing-level coding requirements.
- It has direct evaluation value for product directions such as spreadsheet Copilots and database Copilots.
- Future extensions may include multilingual versions or multimodal table inputs.

## Rating

- Novelty: ⭐⭐⭐⭐ (breadth of task coverage is the primary innovation; individual task designs are not novel in isolation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (extensive model comparisons and multi-dimensional analyses)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear organization, rich figures and tables, detailed data)
- Value: ⭐⭐⭐⭐⭐ (fills a critical gap with long-term impact on the field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)
- [\[NeurIPS 2025\] Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras](talk2event_grounded_understanding_of_dynamic_scenes_from_event_cameras.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](../../ICCV2025/robotics/rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)

</div>

<!-- RELATED:END -->

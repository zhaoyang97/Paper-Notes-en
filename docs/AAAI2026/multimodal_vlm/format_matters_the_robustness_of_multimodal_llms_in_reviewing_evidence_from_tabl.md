---
title: >-
  [Paper Note] Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts
description: >-
  [AAAI 2026][Multimodal VLM][Scientific claim verification] This paper systematically investigates the robustness of multimodal LLMs in verifying scientific claims using tables and charts as evidence. By extending SciTabAlign and ChartMimic into a table–chart aligned evaluation benchmark, the authors find that all 12 evaluated multimodal LLMs consistently perform better on table-based evidence than chart-based evidence, while human annotators perform consistently across both formats — revealing a critical weakness in current models' chart comprehension capabilities.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Scientific claim verification
  - tables vs. charts
  - multimodal robustness
  - format sensitivity
  - evidence format
date: 2026-05-08
content_hash: 849806f01148e819
---

# Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts

**Conference**: AAAI 2026
**arXiv**: [2511.10075](https://arxiv.org/abs/2511.10075)
**Code**: [https://github.com/Alab-NII/tables-vs-charts](https://github.com/Alab-NII/tables-vs-charts)
**Area**: Multimodal VLM
**Keywords**: Scientific claim verification, tables vs. charts, multimodal robustness, format sensitivity, evidence format

## TL;DR
This paper systematically investigates the robustness of multimodal LLMs in verifying scientific claims using tables and charts as evidence. By extending SciTabAlign and ChartMimic into a table–chart aligned evaluation benchmark, the authors find that all 12 evaluated multimodal LLMs consistently perform better on table-based evidence than chart-based evidence, while human annotators perform consistently across both formats — revealing a critical weakness in current models' chart comprehension capabilities.

## Background & Motivation

### State of the Field
Scientific claim verification requires a model to determine whether a given claim is supported by provided evidence. With the rapid growth of scientific publications in the generative AI era, the demand for automated paper review assistance is increasingly urgent. Experimental results are typically presented as tables or charts, which constitute core components of academic papers.

### Limitations of Prior Work

**Single-format evidence in existing datasets**: Most claim verification datasets provide evidence in only one format — either all structured text tables (e.g., SciTab) or all charts (e.g., MuSciClaims). Even SciVer, which covers multiple modalities, presents tables as images rather than structured text.

**Lack of aligned evaluation**: No existing dataset provides paired tables and charts conveying **the same information**, making it impossible to fairly evaluate model performance across formats.

**Unknown model bias**: Whether multimodal LLMs behave consistently when processing semantically equivalent but differently formatted evidence remains an open and unaddressed question.

### Root Cause
A reliable paper review assistance system must accurately verify claims regardless of the format in which supporting evidence is presented. If an LLM performs well on one format but poorly on another, it will produce biased or incomplete assessments.

### Starting Point
Construct a semantically aligned table–chart dataset and systematically compare model performance under three settings — table-only, chart-only, and mixed input — while incorporating human evaluation as an upper-bound reference.

## Method

### Overall Architecture
The methodological core of this paper is **dataset construction + systematic evaluation**:
1. Extend two existing datasets into augmented versions (SciTabAlign+ and ChartMimic+)
2. Evaluate 12 multimodal LLMs under three input settings
3. Conduct human evaluation to validate the reasonableness of task difficulty

### Key Designs

#### 1. **SciTabAlign+ Dataset Construction**
- **Source**: SciTabAlign (an extension of SciTab with ambiguous cases removed)
- **Data cleaning**: Remove HTML/bracket tags, normalize numerical values; retain 70 tables (162 claims) from 136 original tables
- **Four chart types generated**:
    - **Basic bar chart**: different colors represent different bars
    - **Symbol bar chart**: colors replaced by "/" or "-" symbols
    - **Line chart**: data points connected by lines
    - **Swapped chart**: x-axis methods and metrics are swapped
- **Design Motivation**: Diverse chart types test model comprehension across different visual encoding schemes
- Final dataset: 372 table claims + 648 chart claims (162 × 4 chart types)

#### 2. **ChartMimic+ Dataset Construction**
- **Source**: Direct Mimic subset of ChartMimic (a chart2code dataset)
- Selected 70 line charts and 80 bar charts
- Automatically extracted underlying tabular data using Python code
- Four NLP researchers validated and edited the tables and wrote supporting/refuting claims
- Final dataset: 152 claims (52 bar charts + 24 line charts)
- **Design Motivation**: ChartMimic originates from real academic papers with high-quality charts; underlying data can be precisely extracted via code

#### 3. **Evaluation Settings**
- **Three input modes**: Table-only, Chart-only, Table+Chart combined
- **Fairness handling**: In chart settings, "Table X" in claims is replaced with "Figure X"
- **Evaluation method**: Zero-shot CoT prompting; macro-F1 as the primary metric
- **12 models**: Spanning 4 families — InternVL3 (1B–38B), Qwen-VL 2.5 (3B–72B), LLaVA-v1.6 (7B–34B), Llama-3.2 (11B)

### Human Evaluation
- Uses the same random samples as model evaluation
- Validates the gap between task difficulty itself and model capability

## Key Experimental Results

### Main Results (SciTabAlign+ Macro-F1)

| Model | Table-only (All) | Table (162) | Chart Avg | Chart+Table |
|-------|-----------------|-------------|-----------|-------------|
| Qwen2.5-VL-72B | **88.5** | 86.3 | 68.5 | 88.0 |
| InternVL3-38B | 80.7 | 82.4 | 62.5 | **88.8** |
| Qwen2.5-VL-32B | 84.6 | 86.2 | 67.6 | 86.2 |
| InternVL3-14B | 81.5 | 81.1 | 62.4 | 84.9 |
| InternVL3-8B | 69.9 | 70.4 | 55.7 | 70.2 |
| Qwen2.5-VL-7B | 75.7 | 80.0 | 58.3 | 75.9 |
| Qwen2.5-VL-3B | 52.7 | 53.6 | 39.9 | 50.4 |
| InternVL3-1B | 31.1 | 32.6 | 23.3 | 34.1 |
| LLaVA-v1.6-34B | 60.2 | 56.7 | 33.4 | 37.1 |

### Ablation Study (Effect of Chart Type, Average over 12 Models)

| Chart Type | Avg Macro-F1 | Note |
|------------|-------------|------|
| Basic bar chart | **53.0** | Easiest |
| Line chart | 51.9 | Second best |
| Swapped chart | 51.3 | Moderate |
| Symbol bar chart | 50.4 | Hardest |

### Table vs. Chart Performance Gap (Top-5 Models)

| Model | Table (162) | Chart Avg | Gap |
|-------|-------------|-----------|-----|
| LLaVA-v1.6-34B | 56.7 | 33.4 | **23.3** |
| Qwen2.5-VL-7B | 80.0 | 58.3 | **21.7** |
| InternVL3-38B | 82.4 | 62.5 | **19.9** |
| InternVL3-14B | 81.1 | 62.4 | **18.7** |
| Qwen2.5-VL-32B | 86.2 | 67.6 | **18.6** |

### Key Findings
1. **Tables consistently outperform charts**: 11 of 12 models perform better under the table setting; the sole exception, LLaVA-v1.6-Mistral-7B, ties (57.6 vs. 57.7)
2. **Large performance gaps**: The maximum gap reaches 23.3% (LLaVA-v1.6-34B), indicating strong model reliance on structured textual input
3. **Humans are format-agnostic**: Human annotators maintain strong performance on both formats, confirming that the difficulty originates from model limitations rather than task design
4. **Small models generalize poorly across modalities**: Models below 8B show very weak correlation between table and chart task performance
5. **Combined input is not always beneficial**: For some models (e.g., Qwen2.5-VL-3B, LLaVA-v1.6-34B), adding charts actually degrades performance
6. **Chart+Table consistently outperforms Chart-only**: All 12 models achieve higher performance under combined input than chart-only input

## Highlights & Insights
- **Precise problem formulation**: The paper clearly distinguishes "format robustness" from general multimodal capability; the semantically equivalent evidence paired experimental design is particularly elegant
- **Symbol bar chart finding**: Replacing colors with symbols further degrades performance, indicating that models struggle even more with non-standard visual encodings
- **Anomalous pattern in LLaVA-v1.6-34B**: A larger model exhibits the worst chart comprehension performance (33.4%), and combined input also fails to help — suggesting that certain architectural choices may be detrimental to visual reasoning
- **Practical implication**: When building automated paper review systems, converting charts into structured tables prior to reasoning is preferable to directly processing chart images

## Limitations & Future Work
- Limited data scale: SciTabAlign+ contains only 162 claims matched with charts; ChartMimic+ contains only 152 claims
- Coverage restricted to bar charts and line charts; heatmaps, scatter plots, box plots, and other chart types are not included
- Multi-chart and subplot scenarios are not explored
- Only zero-shot CoT is used; few-shot and fine-tuning settings are not evaluated
- Charts are programmatically generated and may differ in complexity from manually styled figures in actual papers
- Claims are restricted to binary classification (support/refute); more nuanced labels such as "insufficient information" are not explored
- Closed-source large models (GPT-4V, Gemini, etc.) are not evaluated

## Related Work & Insights
- SciTab (Lu et al., 2023): Scientific claim verification dataset using real paper claims; serves as the data foundation for this work
- ChartMimic (Yang et al., 2025): chart2code dataset providing high-quality chart–code pairs
- SciVer (Wang et al., 2025): Multimodal scientific verification dataset, but tables are presented as images
- ChartQA (Masry et al., 2022), CharXiv (Wang et al., 2024): Chart question answering tasks focused on content extraction rather than claim verification
- Insight: Format robustness is a fundamental requirement for multimodal models. Future models should strengthen training on the comprehension of chart-based visual data

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel problem formulation; the format robustness perspective is distinctive)
- Experimental Thoroughness: ⭐⭐⭐⭐ (12 models × 3 settings × 4 chart types × human evaluation, though dataset scale is small)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, illustrative figures, complete logical chain)
- Value: ⭐⭐⭐⭐ (Reveals a critical weakness of multimodal LLMs; directly informative for automated paper review system development)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](../../ACL2026/multimodal_vlm/tree-of-evidence_efficient_34system_234_search_for_faithful_multimodal_grounding.md)
- [\[AAAI 2026\] VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[AAAI 2026\] Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks](phantom_menace_exploring_and_enhancing_the_robustness_of_vla_models_against_phys.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)

</div>

<!-- RELATED:END -->

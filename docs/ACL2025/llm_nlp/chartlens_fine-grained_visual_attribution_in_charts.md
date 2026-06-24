---
title: >-
  [Paper Note] ChartLens: Fine-Grained Visual Attribution in Charts
description: >-
  [ACL 2025][LLM (Other)][Chart Attribution] Proposes the task of Post-Hoc Fine-grained Visual Attribution in charts, designs the ChartLens algorithm that leverages segmentation techniques to label chart elements and prompt multimodal LLMs with a Set-of-Marks for precise attribution, and constructs the ChartVA-Eval benchmark, achieving an F1 improvement of 26-66% across three chart types.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Chart Attribution"
  - "Visual Grounding"
  - "Multimodal LLM"
  - "Hallucination Detection"
  - "Set-of-Marks"
date: 2026-05-08
content_hash: c62478e80334ba7c
---

# ChartLens: Fine-Grained Visual Attribution in Charts

**Conference**: ACL 2025  
**arXiv**: [2505.19360](https://arxiv.org/abs/2505.19360)  
**Code**: Yes  
**Area**: Others  
**Keywords**: Chart Attribution, Visual Grounding, Multimodal LLM, Hallucination Detection, Set-of-Marks

## TL;DR

Proposes the task of Post-Hoc Fine-grained Visual Attribution in charts, designs the ChartLens algorithm that leverages segmentation techniques to label chart elements and prompt multimodal LLMs with a Set-of-Marks for precise attribution, and constructs the ChartVA-Eval benchmark, achieving an F1 improvement of 26-66% across three chart types.

## Background & Motivation

- Multimodal Large Language Models (MLLMs) have made progress in chart understanding tasks, but suffer from severe **hallucination issues**: generated texts are often inconsistent with the visual data of the charts.
- **Attribution** is a key strategy for mitigating hallucinations, helping model responses trace back to specific visual evidence.
- Existing LLM attribution methods mostly focus on text (e.g., citation retrieval, post-retrieval QA), leaving **visual attribution**, especially **chart visual attribution**, virtually unexplored.
- Particular challenges of charts:
    - They contain complex relationships like precise quantities, trends, and proportions.
    - Visual elements (bars, pie slices, line charts) are highly structured and may overlap.
    - Understanding requires contextual comprehension of chart types, data encodings, axes, legends, and colors.
- Reliable chart attribution is crucial for critical domains such as financial analysis, policy-making, and scientific research.

## Method

### Overall Architecture

ChartLens two-stage pipeline:
1. **Mark Generation**: Detects chart elements (bars/pies/lines) using segmentation techniques and generates marks for each element.
2. **Attribution with MLLMs**: Feeds the marked charts into MLLMs to perform validation and attribution via Set-of-Marks prompting.

### Key Designs

#### 1. Mark Generation — Segmentation and Marking

**Bar Chart Segmentation**:
- Input images are binarized using Otsu thresholding in both RGB and HSV spaces.
- Dark backgrounds are automatically inverted to extract foreground contours.
- Contours are decomposed based on unique pixel values to isolate individual bar elements.
- Noise contours are filtered out using solidity and area thresholds.
- **Key Improvement**: Utilizes SAM (Segment Anything Model) to refine the segmentation—sampling $n$ points from detected elements as SAM prompts to generate precise masks.
- SAM naturally suppresses background features such as gridlines (low IoU masks).

**Pie Chart Segmentation**:
- Identifies the largest contour as the pie chart boundary.
- Calculates the minimum enclosing circle to approximate the chart area.
- Unrolls the chart radially into a linear representation, detecting full edges to identify sector boundaries.
- Maps back to the original circular area to obtain individual sector segmentations.

**Line Chart Segmentation**:
- Utilizes **LineFormer** (a Transformer architecture) to specifically handle the slender structures, overlapping trajectories, and intersecting lines of line charts.
- After detecting candidate lines, divides them into small segments along the horizontal dimension to serve as fine-grained marks.
- The global context of the Transformer helps differentiate closely-spaced or crossing lines.

#### 2. Set-of-Marks Prompting for Attribution

- Overlays alphabetical/numerical labels of the segmented elements onto the chart images.
- Constructs structured prompts containing two objectives:
    - **Validation**: Verifies whether the QA pair is consistent with the chart information.
    - **Attribution**: Identifies and cites the specific marked elements that support the answer.
- Employs few-shot textual examples and Chain-of-Thought (CoT) reasoning to guide the MLLM in step-by-step validation and attribution.

### Problem Formulation

Given a chart image $c$ and an associated response $v$, the objective is to produce an attribution set $A_{c,v} = \{a_1, a_2, ..., a_n\}$, where each $a_i$ corresponds to a visual region (bar/line segment/sector) in the chart that supports the response. The attribution set must satisfy:
1. **Relevance**: Each region is directly relevant to the response.
2. **Completeness**: It covers all necessary visual evidence.
3. **Precision**: It excludes irrelevant parts.

## Key Experimental Results

### ChartVA-Eval Benchmark Statistics

| Dataset | Queries | Charts | Bar Charts | Pie Charts | Line Charts | Source |
|-------|-------|-------|-------|------|-------|------|
| ChartVA-AITQA | 301 | 301 | 203 | 0 | 98 | Synthetic |
| ChartVA-PlotQA | 595 | 581 | 396 | 0 | 199 | Synthetic |
| ChartVA-ChartQA | 348 | 266 | 121 | 109 | 118 | Real-world |

### Main Results: Bar Chart Attribution (P / R / F1)

| Method | AITQA F1 | PlotQA F1 | ChartQA F1 |
|------|---------|----------|-----------|
| Zero-shot GPT-4o | 22.77 | 3.30 | 7.75 |
| KOSMOS-2 | 0.51 | 1.01 | 3.13 |
| LISA | 1.62 | 0.34 | 1.01 |
| **ChartLens** | **69.28** | **34.65** | **64.14** |

### Line Chart Attribution (Detection % / Chart Area %)

| Method | AITQA Det.↑ | AITQA Area↓ | PlotQA Det.↑ | PlotQA Area↓ | ChartQA Det.↑ | ChartQA Area↓ |
|------|-----------|-----------|------------|------------|-------------|-------------|
| Zero-shot GPT-4o | 18.28 | 1.94 | 6.79 | 8.63 | 3.39 | 1.15 |
| KOSMOS-2 | 74.19 | 46.03 | 38.83 | 27.06 | 87.29 | 41.49 |
| LISA | 94.62 | 63.18 | 50.21 | 40.92 | 50.21 | 40.92 |
| **ChartLens** | **59.14** | **1.25** | **51.84** | **9.98** | **77.8** | **5.34** |

The high detection rate of LISA/KOSMOS-2 is due to covering a large percentage of the chart area (40-63%), whereas ChartLens covers only 1-10% of the area, demonstrating much better precision.

### Pie Chart Attribution (ChartQA)

| Method | P | R | F1 |
|------|---|---|-----|
| Zero-shot GPT-4o | 8.94 | 5.99 | 7.17 |
| KOSMOS-2 | 20.18 | 8.24 | 11.70 |
| LISA | 1.32 | 13.86 | 2.41 |
| **ChartLens** | **53.33** | **44.57** | **48.56** |

### Ablation Study

- **Modularity of the Segmentation Module**: SAM can be replaced with more advanced segmentation models to achieve further improvements.
- **Choice of MLLM**: ChartLens is based on ChatGPT-4o, but the underlying multimodal model can be flexibly replaced.
- **Effectiveness of SoM Prompting**: Directly allowing GPT-4o to perform zero-shot bounding box prediction yields extremely poor performance (F1 of only 3-23%), verifying the necessity of SoM prompting combined with segmentation.

### Key Findings

1. **ChartLens significantly outperforms baselines across all chart types**: achieves +46-66% F1 on bar charts and +37% F1 on pie charts.
2. **Existing visual grounding models are unsuitable for chart attribution**: KOSMOS-2 and LISA perform poorly in precisely locating chart elements.
3. **Zero-shot bounding box prediction by GPT-4o is unreliable**: It lacks accurate visual localization capabilities.
4. **Precision vs. Recall Tradeoff**: LISA/KOSMOS-2 achieve high recall by covering large areas, but at the cost of extremely low precision.
5. **SoM prompting significantly improves attribution quality**: Tokenization enables MLLMs to explicitly reference specific visual elements.

## Highlights & Insights

- First to systematically define the task of fine-grained visual attribution in charts, filling an important gap in multimodal attribution.
- Modular design with segmentation + marking + MLLM: Each component can be upgraded independently, showing good extensibility.
- The ChartVA-Eval benchmark covers both synthetic and real-world charts, sourced from authoritative data such as SEC filings, the World Bank, and GTD.
- Provides a plug-and-play post-processing attribution mechanism that can be used with any MLLM without affecting the generation of original answers.

## Limitations & Future Work

- Segmentation precision is a bottleneck: inaccurate segmentation leads to incomplete attribution.
- Only handles visual chart elements (bars/lines/sectors), **excluding text elements** (attribution of titles, labels, and legends).
- Relies on GPT-4o as the attribution reasoning engine, which incurs a relatively high cost.
- Has not been tested on more complex chart types (e.g., scatter plots, heatmaps, combined charts).
- Attribution annotation relies partly on automation and manual verification, presenting challenges for large-scale standardized annotation.

## Related Work & Insights

- **Textual Attribution** (Direct/Post-retrieval/Post-hoc) $\rightarrow$ This work extends it to the visual modality.
- **MATSA** proposed table attribution, which ChartLens further extends to chart visual elements.
- The hybrid strategy of **SAM + LineFormer** serves as a reference for other structured visual understanding tasks.
- Insights: Similar attribution frameworks could be applied to other structured visual data, such as infographics, maps, and flowcharts.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ First to define the chart visual attribution task |
| Experimental Thoroughness | ⭐⭐⭐⭐ Three chart types + four baselines + complete benchmark |
| Value | ⭐⭐⭐⭐ Modular and plug-and-play |
| Writing Quality | ⭐⭐⭐⭐ Clear problem definition and detailed experiments |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models](behaviorbox_automated_discovery_of_fine-grained_performance_differences_between_.md)
- [\[ACL 2025\] RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)
- [\[ACL 2025\] MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)
- [\[ACL 2025\] Attribution Methods in NLP: Navigating a Fragmented Landscape](attribution_methods_in_nlp_navigating_a_fragmented_landscape.md)
- [\[ACL 2025\] JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution](jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)

</div>

<!-- RELATED:END -->

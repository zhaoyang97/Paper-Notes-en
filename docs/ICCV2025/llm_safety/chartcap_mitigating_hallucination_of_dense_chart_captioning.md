---
title: >-
  [Paper Note] ChartCap: Mitigating Hallucination of Dense Chart Captioning
description: >-
  [ICCV 2025][LLM Safety][Chart understanding] This work constructs ChartCap, a large-scale dataset of 565K real chart–caption pairs. By adopting type-specific caption schemas that exclude irrelevant information while emph…
tags:
  - "ICCV 2025"
  - "LLM Safety"
  - "Chart understanding"
  - "vision-language models"
  - "hallucination mitigation"
  - "chart captioning"
  - "dataset construction"
date: 2026-05-08
content_hash: 0e07a7e18c19a7ad
---

# ChartCap: Mitigating Hallucination of Dense Chart Captioning

**Conference**: ICCV 2025
**arXiv**: [2508.03164](https://arxiv.org/abs/2508.03164)
**Code**: [https://junyoung-00.github.io/ChartCap/](https://junyoung-00.github.io/ChartCap/)
**Area**: LLM Safety
**Keywords**: Chart understanding, vision-language models, hallucination mitigation, chart captioning, dataset construction

## TL;DR

This work constructs ChartCap, a large-scale dataset of 565K real chart–caption pairs. By adopting type-specific caption schemas that exclude irrelevant information while emphasizing structure and key insights, and by introducing a reference-free Visual Consistency Score (VCS) evaluation metric, the paper effectively mitigates hallucination in VLM-based chart captioning.

## Background & Motivation

Chart captioning is a core task for evaluating VLMs' ability to understand charts. An ideal caption must satisfy two conditions: (1) it should not contain erroneous information that cannot be inferred from the chart alone; and (2) it should include structural descriptions and key insights (e.g., maximum/minimum values, trend patterns).

However, existing real-chart datasets suffer from two critical problems:

**Irrelevant information contamination**: Charts are typically embedded in papers or documents, and their original captions are written with surrounding context in mind (e.g., referencing parameter names or describing missing error bars). Such information cannot be inferred from the chart image alone. This creates an "impossible task" for the model — expecting it to predict information absent from the chart — ultimately inducing hallucination.

**Insufficient structural and insight information**: Original captions frequently omit critical information (e.g., data trends, extreme values), since human authors assume readers can infer these directly from the chart. The absence of type-specific caption schemas leads to the neglect of key information across different chart types.

## Method

### Overall Architecture

The construction of ChartCap involves a four-stage automated pipeline followed by a cycle-consistency-based human verification step:
1. Filter non-chart images → 2. Classify chart types and extract captions → 3. Extract type-specific structural information and key insights → 4. Convert into coherent sentence-level captions.

### Key Designs

#### 1. Type-Specific Caption Schema
- **Function**: Defines standardized schemas for structural descriptions and key insights across 9 chart types (line chart, bar chart, pie chart, histogram, scatter plot, area chart, bubble chart, contour plot, treemap).
- **Mechanism**: Grounded in data visualization research — Munzner's *Visualization Analysis and Design* provides the design framework, and the VLAT test blueprint identifies the cognitive tasks associated with each chart type. For example, scatter plots emphasize clustering and distribution, while line charts emphasize temporal trends and changes.
- **Design Motivation**: Standardization reduces ambiguity in caption quality, enabling models to learn which key information is relevant for each chart type.

#### 2. Four-Stage Data Generation Pipeline
- **Stage 1 – Filtering**: InternVL2.5-8B is used to filter out non-data-driven charts (e.g., diagrams, illustrations), retaining 1.2M images from 3.1M, with 100% precision.
- **Stage 2 – Classification and Extraction**: GPT-4o is used to obtain chart types and captions, retaining 577K charts belonging to the 9 predefined types, with 99% accuracy.
- **Stage 3 – Information Extraction**: GPT-4o handles coarse-grained tasks (overall trends), while Claude 3.5 Sonnet handles fine-grained tasks (precise numerical values). Accuracy: 94%.
- **Stage 4 – Caption Generation**: GPT-4o-mini converts semi-structured data into sentence-level captions, with 100% accuracy.

#### 3. Cycle-Consistency-Based Human Verification
- **Function**: Leverages the millisecond-level speed of human visual perception to validate caption quality by comparing the original chart against a chart reconstructed from the caption.
- **Mechanism**: Claude 3.5 Sonnet converts captions into Python code to reconstruct charts; annotators then visually compare the original and reconstructed charts. This achieves approximately 24× speedup over direct chart–caption comparison while maintaining 95% F1.
- **Design Motivation**: Humans compare two images far faster than carefully reading lengthy caption text. Visual comparison simultaneously ensures both accuracy and informational completeness.

#### 4. Visual Consistency Score (VCS)
- **Function**: Introduces a reference-free evaluation metric for chart caption quality.
- **Mechanism**: A caption $C_i$ is converted by an LLM into Matplotlib code $G_i$, which is executed to generate a reconstructed chart $\hat{I}_i$; a visual encoder then computes the cosine similarity with the original chart $I_i$:

$$\text{VCS} = \frac{1}{N} \sum_{i=1}^{N} \text{Sim}(I_i, \hat{I}_i)$$

This is supplemented by an OCRScore measuring the retention of textual elements:

$$\text{OCRScore} = 2 \cdot \frac{P \times R}{P + R}$$

where $P = \frac{\sum|\mathcal{T}_i \cap \hat{\mathcal{T}}_i|}{\sum|\hat{\mathcal{T}}_i|}$ and $R = \frac{\sum|\mathcal{T}_i \cap \hat{\mathcal{T}}_i|}{\sum|\mathcal{T}_i|}$.

- **Design Motivation**: Existing metrics (BLEU, BERTScore) fail to capture deep semantic quality and are heavily dependent on reference caption quality. The unique property of charts — that they can be deterministically generated from code — makes "reverse verification" feasible.

### Loss & Training

LoRA fine-tuning is applied to InternVL2.5-8B and Phi3.5-vision-4B on 509K training samples. A standard supervised fine-tuning paradigm is adopted, with the input consisting of a chart image and the instruction "Please provide a detailed caption for the chart."

## Key Experimental Results

### Main Results

Performance of various models on the ChartCap test set:

| Model | sacreBLEU | ROUGE-L | BERTScore | VCS (Large) | OCRScore |
|-------|-----------|---------|-----------|-------------|----------|
| Claude 3.5 Sonnet | 5.35 | 0.2265 | 0.6606 | 0.8834 | 0.4868 |
| InternVL2.5-78B | 8.15 | 0.2510 | 0.6642 | 0.8841 | 0.4677 |
| Phi3.5-4B | 8.41 | 0.2466 | 0.6626 | 0.8433 | 0.4875 |
| **Phi3.5-4B_ChartCap** | **23.82** | **0.3900** | **0.7427** | **0.8933** | **0.5179** |
| **InternVL2.5-8B_ChartCap** | **19.47** | **0.3393** | **0.7238** | **0.8913** | **0.5089** |

### Ablation Study / Dataset Comparison

VCS and informativeness comparison across real-chart datasets:

| Dataset | VCS (Large) | VCS (So400M) | OCRScore | Avg. Word Count |
|---------|-------------|--------------|----------|-----------------|
| ArxivCap | 0.7561 | 0.7421 | 0.1781 | 43.7 |
| ChartSumm | 0.8940 | 0.9008 | 0.2635 | 45.4 |
| Chart-to-Text | 0.6925 | 0.7089 | 0.0951 | 62.2 |
| **ChartCap** | **0.8983** | **0.9089** | **0.5424** | **231.1** |

Zero-shot transfer on the VisText test set:

| Model | VCS (Large) | VCS (So400M) | OCRScore |
|-------|-------------|--------------|----------|
| Human annotation | 0.9172 | 0.9151 | 0.3407 |
| Claude 3.5 Sonnet | 0.8970 | 0.9008 | 0.3286 |
| **Phi3.5-4B_ChartCap** | **0.9443** | **0.9382** | 0.3414 |

### Key Findings

1. The fine-tuned Phi3.5 model with only 4B parameters outperforms InternVL2.5-78B and commercial Claude 3.5 Sonnet on all metrics, demonstrating that **high-quality data matters more than model scale**.
2. Captions generated by ChartCap-fine-tuned models are judged by human evaluators to be superior even to human-written annotations (on the VisText dataset).
3. Models fine-tuned on original captions (containing irrelevant information) or on ChartSumm exhibit degraded performance, confirming that irrelevant information indeed induces hallucination.
4. VCS shows significantly higher agreement with human judgments than traditional metrics such as BERTScore.

## Highlights & Insights

- **Accurate root-cause analysis**: The paper clearly identifies that the source of hallucination is not insufficient model capability, but rather the presence of irrelevant information in training data that cannot be inferred from the chart alone.
- **Cycle-consistency verification**: The pipeline elegantly transforms the time-consuming text review process into rapid visual comparison via a "caption → code → reconstructed chart → visual comparison" cycle.
- **VCS metric**: By exploiting the unique property that charts can be generated from code, the paper achieves the first truly reference-free evaluation of chart captions.
- **Counter-intuitive finding**: Small model + high-quality data outperforms large models; the ChartCap-fine-tuned 4B model surpasses human annotations.

## Limitations & Future Work

- The caption schema covers only 9 chart types (based on VLAT) and excludes types such as heatmaps and Sankey diagrams.
- Dataset construction relies on commercial APIs (GPT-4o, Claude 3.5 Sonnet), incurring high costs and limiting full reproducibility.
- VCS evaluation itself depends on the LLM's code generation capability, requiring a retry mechanism for code generation failures.
- Transfer to chart question-answering tasks such as ChartQA has not been explored.

## Related Work & Insights

- **ArxivCap** and **SciCap**, collected from academic papers, corroborate the problem that original captions contain irrelevant information.
- **CLIPScore** proves unreliable for long captions and precise chart understanding scenarios; VCS offers a superior alternative.
- The methodology of this work — "define schema → automated generation → cycle-consistency verification" — is generalizable to caption generation tasks for other types of structured visual content.

## Rating

- Novelty: ⭐⭐⭐⭐ Addresses hallucination from a data quality perspective; the VCS metric is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across reference metrics, human evaluation, and VCS; cross-dataset generalization testing included.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear; the dataset construction process is described in thorough and transparent detail.
- Value: ⭐⭐⭐⭐ The 565K high-quality dataset and VCS metric represent significant contributions to the chart understanding community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/llm_safety/enhancing_hallucination_detection_via_future_context.md)
- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](../../ICLR2026/llm_safety/enhancing_hallucination_detection_through_noise_injection.md)
- [\[ICLR 2026\] VeriTrail: Closed-Domain Hallucination Detection with Traceability](../../ICLR2026/llm_safety/veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](../../CVPR2026/llm_safety/hulluedit_subspace_editing_hallucination.md)
- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](../../ICLR2026/llm_safety/biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)

</div>

<!-- RELATED:END -->

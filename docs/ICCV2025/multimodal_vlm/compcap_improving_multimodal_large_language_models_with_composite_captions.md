---
title: >-
  [Paper Note] CompCap: Improving Multimodal Large Language Models with Composite Captions
description: >-
  [ICCV 2025][Multimodal VLM][composite images] This paper proposes CompCap, an automated framework for synthesizing six categories of composite images (collages, image-text mixtures, charts, tables, code, and diagrams) along with high-quality captions. The resulting CompCap-118K dataset, when incorporated into the SFT stage, significantly improves MLLM comprehension of composite images.
tags:
  - ICCV 2025
  - Multimodal VLM
  - composite images
  - image captioning
  - multimodal large language models
  - data synthesis
  - vision-language alignment
date: 2026-05-08
content_hash: d765a0522105e966
---

# CompCap: Improving Multimodal Large Language Models with Composite Captions

**Conference**: ICCV 2025
**arXiv**: [2412.05243](https://arxiv.org/abs/2412.05243)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: composite images, image captioning, multimodal large language models, data synthesis, vision-language alignment

## TL;DR

This paper proposes CompCap, an automated framework for synthesizing six categories of composite images (collages, image-text mixtures, charts, tables, code, and diagrams) along with high-quality captions. The resulting CompCap-118K dataset, when incorporated into the SFT stage, significantly improves MLLM comprehension of composite images.

## Background & Motivation

Multimodal large language models (MLLMs) have demonstrated strong performance on natural image understanding, yet exhibit notable deficiencies in **Composite Image (CI)** understanding. Composite images—visual content assembled from multiple heterogeneous elements such as photographs, charts, text, and code—are ubiquitous in practical applications (e.g., posters, infographics, webpage screenshots).

The authors identify three key observations through empirical analysis:

**MLLMs perform substantially worse on CIs than on natural images**: Models such as LLaVA-1.5 and InstructBLIP show significantly lower accuracy on CIs across captioning and VQA tasks compared to natural images.

**Captioning errors and VQA errors are highly correlated**: Errors made during CI description generation closely mirror those made during direct VQA, indicating that the root cause lies in insufficient vision-language alignment.

**High-quality CI captions are absent from training data**: Existing training corpora primarily contain high-quality captions for natural images and instruction-tuning QA data for CIs (e.g., ChartQA), but lack detailed descriptive texts for CIs.

These findings suggest that instruction data alone is insufficient for comprehensive CI understanding in MLLMs—high-quality CI-caption data is essential for establishing better vision-language alignment.

## Method

### Overall Architecture

CompCap is a general-purpose composite image–caption synthesis framework. The core pipeline follows: **metadata → image generation + caption generation → CI-caption pairs**. Dedicated generation pipelines are designed for each of the six CI categories.

### Key Designs

1. **Metadata-driven image synthesis**: Metadata consists of source data (e.g., image-caption pairs, tabular data, code snippets) and configuration information (e.g., layout, chart type, color style). Configurations are generated through randomized processes to ensure diversity. Images are rendered using tools including Plotly, Matplotlib, OpenCV, PIL, Mermaid, and Carbon.

2. **LLM-driven caption generation**: An LLM is employed to produce accurate and detailed captions for the synthesized CIs. Prompt design is critical—instructions are customized for each CI category (e.g., collages require attention to spatial relationships and associations among sub-images; charts require data analysis). Active in-context example selection is adopted to improve caption quality and diversity.

3. **Collage generation pipeline** (representative example):

    - **Source data retrieval**: Three strategies—random retrieval (uncorrelated image combinations for counterfactual debiasing), similarity-based retrieval (visual/textual similarity via DINO-v2 and CLIP), and entity-based retrieval (multiple images of the same entity)
    - **Layout generation**: Grid layout (define grid first, then sample images) and Auto layout (sample images first, then arrange automatically)
    - **Caption design**: Prompts include a coordinate system, the position and description of each sub-image, and in-context examples; relational inferences are additionally generated when images are semantically related

4. **Other CI category pipelines**:

    - Image-Text: rendered using OpenCV/PIL/Augraphy
    - Chart: generated from tabular data using Plotly; 22K samples with an average description length of 1,468 characters
    - Diagram: Mermaid code rendered via Selenium
    - Code: rendered using Carbon; 2K samples
    - Table: rendered using Matplotlib

### Loss & Training

CompCap-118K is incorporated into the SFT stage of MLLM training. To ensure fair comparison, the original SFT dataset is uniformly downsampled to maintain a constant total number of training samples. Models at three scales (4B/7B/13B) are trained based on LLaVA-NeXT and xGen-MM architectures.

## Key Experimental Results

### Main Results

| Model | SEEDBench | TextVQA | MMBench | ChartQA | DocVQA | InfoVQA | Avg. | Gain |
|-------|-----------|---------|---------|---------|--------|---------|------|------|
| xGen-MM-4B (baseline) | 71.3 | 67.7 | 75.5 | 54.8 | 55.2 | 27.6 | 57.2 | - |
| CompCap-4B | 71.6 | 67.9 | 76.2 | 57.4 | 58.1 | 27.9 | 58.9 | +1.7% |
| LLaVA-NeXT-7B (baseline) | 71.2 | 65.2 | 67.6 | 63.5 | 76.5 | 39.2 | 62.5 | - |
| CompCap-7B | 70.5 | 65.6 | 68.9 | 68.9 | 77.6 | 40.8 | 64.5 | +2.0% |
| LLaVA-NeXT-13B (baseline) | 71.9 | 67.6 | 68.9 | 68.5 | 79.9 | 43.8 | 65.6 | - |
| CompCap-13B | 72.2 | 67.8 | 70.8 | 73.9 | 81.1 | 47.0 | 68.5 | +2.9% |

### Ablation Study

| Cumulative CI Types Added | NI Benchmark Avg. | CI Benchmark Avg. | Overall Avg. | Gain |
|--------------------------|-------------------|-------------------|--------------|------|
| Baseline (no CompCap) | 70.9 | 61.3 | 65.6 | - |
| + Collage | 71.5 | 62.4 | 66.4 | +0.8 |
| + Code | 71.3 | 62.8 | 66.6 | +1.0 |
| + Table | 71.7 | 63.0 | 67.0 | +1.4 |
| + Diagram | 71.5 | 63.1 | 67.4 | +1.8 |
| + Chart | 72.2 | 63.9 | 68.0 | +2.4 |
| + Image-Text (full = CompCap-118K) | 73.1 | 64.6 | 68.5 | +2.9 |

### Key Findings

- Each additional CI category yields consistent performance improvements, validating the effectiveness of every pipeline in the framework.
- Caption data facilitates cross-domain transfer more effectively than instruction data (e.g., chart captions benefit DocVQA and InfoVQA).
- Experiments replacing chart instructions with captions show that captions more effectively improve performance; however, a 100% replacement degrades instruction-following ability, with the optimal ratio being approximately 60–80%.
- MathVista performance improves substantially even without math-specific data, suggesting that CI understanding indirectly benefits mathematical reasoning.

## Highlights & Insights

- **Precise diagnosis of a data gap**: This work is the first to systematically demonstrate that MLLM deficiencies in CI understanding stem from the absence of CI-caption data in training corpora.
- **A framework-level approach to data generation**: CompCap is a general framework rather than a single pipeline, making it readily extensible to new CI categories.
- **In-depth comparison of captions vs. instructions**: The paper quantitatively demonstrates the unique value of caption data for vision-language alignment.

## Limitations & Future Work

- Coverage is limited to six CI categories; more complex composite formats such as infographics and posters are not directly addressed.
- LLM-generated captions may contain hallucinations, and quality control relies solely on post-hoc filtering.
- CI-caption data is only introduced at the SFT stage; its effect during pre-training remains unexplored.
- The dataset scale is relatively modest (118K samples), and further scaling may yield additional gains.

## Related Work & Insights

- Similar to ShareGPT4V in emphasizing the importance of high-quality captions for MLLMs, but this work focuses specifically on the CI domain.
- Unlike synthetic datasets such as DVQA and PlotQA, CompCap covers a broader range of CI categories and targets captioning rather than QA.
- This work provides practical guidance for future MLLM training data construction: a balanced mix of natural image and CI caption data is advisable.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The problem formulation is precise and the analysis of missing CI-caption data is valuable, though the data synthesis methodology itself is not particularly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three model scales, eleven benchmarks, and detailed ablations are provided, though comparisons with additional state-of-the-art models are lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, reasoning is rigorous, and figures and tables are abundant.
- **Value**: ⭐⭐⭐⭐ The paper identifies an important gap in the MLLM training data landscape and offers practical reference value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)

<!-- RELATED:END -->

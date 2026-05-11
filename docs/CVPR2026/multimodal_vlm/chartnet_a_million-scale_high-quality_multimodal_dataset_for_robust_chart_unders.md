---
title: >-
  [Paper Note] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding
description: >-
  [CVPR 2026][Multimodal VLM][Chart Understanding] This paper introduces ChartNet—a 1.5-million-scale, high-quality multimodal chart dataset. A code-guided synthesis pipeline generates aligned quintuples comprising image–code–data table–text–reasoning QA. Fine-tuning on ChartNet significantly improves VLM performance on chart understanding and reasoning tasks, enabling small models to surpass GPT-4o.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Chart Understanding
  - Large-Scale Dataset
  - Code-Guided Generation
  - VLM
  - Multimodal Reasoning
date: 2026-05-08
content_hash: 3dc8fa1dbea28bf9
---

# ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.27064](https://arxiv.org/abs/2603.27064)
**Code**: [HuggingFace](https://huggingface.co/datasets/ibm-granite/ChartNet)
**Area**: Signal/Communication (Multimodal Document Understanding)
**Keywords**: Chart Understanding, Large-Scale Dataset, Code-Guided Generation, VLM, Multimodal Reasoning

## TL;DR

This paper introduces ChartNet—a 1.5-million-scale, high-quality multimodal chart dataset. A code-guided synthesis pipeline generates aligned quintuples comprising image–code–data table–text–reasoning QA. Fine-tuning on ChartNet significantly improves VLM performance on chart understanding and reasoning tasks, enabling small models to surpass GPT-4o.

## Background & Motivation

**Background**: Charts are a fundamental medium for conveying quantitative information, and understanding them requires the integration of visual, numerical, and linguistic capabilities. Existing VLMs remain insufficient for complex chart reasoning, with open-source models lagging noticeably behind closed-source systems.

**Limitations of Prior Work**: (a) Existing datasets are small in scale, narrow in chart type coverage, and incomplete in modality—most focus on a single task (QA or summarization) and lack critical modalities such as plotting code, grounding annotations, and reasoning chains. (b) The most widely used benchmark, ChartQA, contains only 14K charts across 3 chart types and is biased toward simple data extraction, on which modern VLMs have already reached saturation.

**Key Challenge**: Training frontier multimodal models requires large-scale, high-quality, multimodally aligned supervision data, yet such data is severely scarce in the chart domain.

**Goal**: Construct a million-scale multimodal chart dataset that comprehensively covers the full spectrum of chart understanding tasks.

**Key Insight**: Charts are inherently **programmatically generatable**—plotting code serves as a structured intermediate representation that can drive large-scale automated synthesis and augmentation.

**Core Idea**: A code-guided chart generation pipeline that reconstructs code from seed charts, iteratively augments it to produce diverse chart variants, and derives data tables, descriptions, and reasoning QA directly from the code.

## Method

### Overall Architecture

A five-stage pipeline: (1) chart-to-code reconstruction (a VLM generates plotting code from seed images); (2) code-guided augmentation (an LLM iteratively rewrites the code); (3) chart rendering (code execution to produce images); (4) quality filtering (a VLM detects rendering defects); (5) attribute generation (data tables, descriptions, etc. are extracted from the code and rendered image).

### Key Designs

1. **Code-Guided Data Generation Pipeline**:

    - Seed set: 150K chart images sourced from TinyChart
    - Reconstruction: pixtral-large converts images into Python plotting code
    - Augmentation: gpt-oss-120b iteratively rewrites code, varying chart type, style, and data; each seed chart can yield an arbitrary number of variants
    - Rendering: code execution with a 77% successful execution rate
    - Filtering: a VLM detects visual defects (text overlap, label clipping, etc.), removing 36.5% of candidates

    Design Motivation: Conventional augmentation in image space cannot guarantee data accuracy or diversity. Augmentation in code space inherently ensures data–image consistency and naturally produces rich variation in chart types and styles.

2. **Reasoning QA Generation (CoT Reasoning)**: Based on the Vision-R1 framework, a four-step pseudo-CoT is generated (Summary → Caption → Reasoning → Conclusion). A modality-bridging strategy allows the text-only model gpt-oss-120b to produce detailed reasoning chains, wrapped in `<think>` and `<answer>` tags.

    Design Motivation: Complex chart questions require multi-step reasoning; explicit reasoning traces provide verifiable intermediate-step supervision.

3. **Specialized Subsets**:

    - **Human-annotated subset**: 96,643 charts verified through rigorous manual inspection
    - **Real-world charts**: 30K charts sourced from authoritative sources such as the World Bank and Pew Research
    - **Grounding QA**: geometry-aware annotations extracted from plotting code to generate QA pairs with bounding boxes
    - **Safety data**: 7K training + 600 test safety-alignment preference pairs

    Design Motivation: Synthetic data provides scale and diversity, but real-world data and human annotations serve as quality anchors.

### Dataset Specifications

- 1.5 million multimodal quintuples (image, code, CSV, text, reasoning QA)
- 24 chart types across 6 plotting libraries
- Approximately 168 GPU-hours on 100+ A100/H100 GPUs

## Key Experimental Results

### Main Results — Pre- and Post-Fine-Tuning Comparison

| Model | Code Exec. Rate | Data Extraction | Summarization | QA Reasoning |
|-------|----------------|-----------------|---------------|--------------|
| Granite-2B | 63.4 → **90.4** | 53.8 → **70.3** | 64.0 → **83.9** | 59.9 → **65.0** |
| LLaVA-7B | 45.3 → **83.9** | 17.0 → **58.8** | 51.2 → **80.3** | 55.1 → **70.3** |
| SmolVLM-256M | N/A → **14.9** | 22.0 → **36.4** | 26.6 → **60.0** | 55.0 → **60.8** |

### Comparison with Large Models and GPT-4o

| Metric | GPT-4o | Granite-2B+ChartNet | LLaVA-7B+ChartNet |
|--------|--------|---------------------|-------------------|
| Code Exec. Rate | 95.9 | 90.4 | 83.9 |
| Data Extraction | 46.7 | **70.3** | **58.8** |
| Summarization | 77.1 | **83.9** | **80.3** |
| QA Reasoning | 61.1 | 65.0 | **70.3** |

### Key Findings

1. **Small models surpass large models**: The fine-tuned 2B model outperforms GPT-4o on data extraction (70.3 vs. 46.7) and summarization (83.9 vs. 77.1); the fine-tuned 7B model surpasses all open- and closed-source models on QA reasoning (70.3 vs. 61.1).
2. **Capability emergence from scratch**: The 256M ultra-compact model originally lacked code generation capability; after fine-tuning it achieves a 14.9% execution rate.
3. **Scale-agnostic gains**: Consistent and significant improvements are observed across all model sizes from 256M to 7B.
4. Generalization is validated on real-world benchmarks ChartCap (summarization) and ChartMimic-v2 (code generation), where substantial gains are also observed.

## Highlights & Insights

- The code-guided synthesis pipeline represents the core methodological contribution—the insight that charts are inherently programmable elegantly reframes data generation as a code augmentation problem.
- The completeness of the dataset is impressive: image–code–CSV–text–reasoning QA quintuples, supplemented with grounding and safety subsets.
- "In the chart domain, scaling model size is far less effective than providing high-quality code-aligned multimodal supervision"—this finding carries important implications for the chart understanding community.
- The quality filtering pipeline (human annotation combined with VLM-based detection) establishes a high quality lower bound.

## Limitations & Future Work

- Synthetic data may still exhibit distributional bias; the style gap between synthetic and real-world charts warrants further investigation.
- The 150K seed charts originate from a single source (TinyChart), limiting seed diversity.
- Only SFT is evaluated; RLHF and DPO remain unexplored, despite the availability of safety preference data.
- Coverage of complex visualization types such as 3D charts and dashboards is limited.

## Related Work & Insights

- Compared to UniChart (611K) and MMC (600K), ChartNet comprehensively leads in scale (1.5M), type diversity (24 types), and modality completeness.
- The code-guided generation paradigm is generalizable to other programmatically generated visual content (e.g., LaTeX formulas, flowcharts, scientific diagrams).
- ChartNet provides critical infrastructure for the multimodal document understanding community.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The code-guided synthesis pipeline is novel and practical, though methodological innovation is inherently limited for a dataset paper
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-model, multi-scale evaluation on public benchmarks with GPT-4o comparison is highly comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ Dataset description is detailed and tables are clear
- **Value**: ⭐⭐⭐⭐⭐ As an open-source 1.5-million-scale dataset, the contribution to the chart understanding field is exceptional

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)
- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](../../ICCV2025/multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[CVPR 2026\] Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)

</div>

<!-- RELATED:END -->

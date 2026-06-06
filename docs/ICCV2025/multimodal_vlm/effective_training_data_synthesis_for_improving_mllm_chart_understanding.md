---
title: >-
  [Paper Note] Effective Training Data Synthesis for Improving MLLM Chart Understanding
description: >-
  [ICCV 2025][Multimodal VLM][chart understanding] This paper proposes a modular five-stage chart data synthesis pipeline that produces a high-quality training set, ECD (Effective Chart Dataset)…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "chart understanding"
  - "data synthesis"
  - "multimodal large language models"
  - "training data"
  - "data quality"
date: 2026-05-08
content_hash: 10ee43df5027929b
---

# Effective Training Data Synthesis for Improving MLLM Chart Understanding

**Conference**: ICCV 2025
**arXiv**: [2508.06492](https://arxiv.org/abs/2508.06492)  
**Code**: [https://github.com/yuweiyang-anu/ECD](https://github.com/yuweiyang-anu/ECD)  
**Area**: Multimodal VLM
**Keywords**: chart understanding, data synthesis, multimodal large language models, training data, data quality

## TL;DR

This paper proposes a modular five-stage chart data synthesis pipeline that produces a high-quality training set, ECD (Effective Chart Dataset), comprising 10k+ chart images and 300k+ QA pairs, consistently improving chart understanding across multiple open-source MLLMs.

## Background & Motivation

Chart understanding is a core capability for building scientific AI agents, yet state-of-the-art open-source MLLMs achieve only 30%–50% accuracy on challenging benchmarks. Although charts can be synthesized programmatically with high precision—an inherent advantage over natural images—existing synthetic chart training sets suffer from significant limitations:

1. Early datasets (PlotQA, OpenCQA) rely on fixed code templates, resulting in limited chart types and low visual diversity.
2. ChartBench separates data generation but lacks diversity in visualization code.
3. ReachQA has LLMs generate both code and data simultaneously, inadvertently constraining data complexity.

These limitations create a substantial gap between synthetic charts and those found in real scientific papers, leading to suboptimal fine-tuning outcomes. The central motivation of this work is to **bridge the gap between synthetic and real charts through modularization and diversification**.

## Method

### Overall Architecture

A five-stage data synthesis pipeline: single-chart generation → composite multi-panel chart generation → visual diversification → quality filtering → QA pair generation and filtering.

### Key Designs

1. **Modular Single-Chart Generation**: Chart functions and data generation are decoupled. Thirty-nine chart functions are manually predefined (each containing parameterized Python plotting code), and GPT-4o receives three inputs—(1) chart theme, (2) chart function, and (3) parameter descriptions with few-shot examples—allowing the model to focus solely on generating data tables and textual elements. This stepwise approach yields richer data distributions and stronger semantic alignment between data values and text. A total of 10,875 single-panel charts are generated.

2. **Conditional Multi-Panel Composition**: Multi-panel charts are generated via iterative conditional generation—when producing the third sub-figure, the data from the preceding two are referenced—ensuring thematic coherence. This mimics the convention in scientific papers of presenting complementary data perspectives across sub-figures. A total of 6,006 multi-panel charts are generated, averaging four sub-figures per chart.

3. **Visual Diversification**: GPT-4o is prompted to modify the Python plotting code, randomly adding annotations, arrows, shaded regions, zoom insets, and subtitles, as well as altering font colors, styles, and sizes, and incorporating additional libraries such as Seaborn. Post-processing further adjusts parameters such as figure size and DPI.

4. **Dual-Metric Quality Filtering**: GPT-4o evaluates each chart along two dimensions—visual clarity $r_{vis}(\mathbf{x}, c_{layout})$ and semantic consistency $r_{sem}(\mathbf{x}, c_{theme})$—retaining only charts that exceed the mean score on both metrics. The dataset is reduced from 16,829 to 10,535 charts (filtering rate: 37.4%).

5. **QA Pair Generation and Filtering**: GPT-4o generates descriptive and reasoning QA pairs conditioned on chart images, code, and data, and assigns a confidence score from 1 to 5; only pairs receiving a perfect score of 5 are retained. The QA set is reduced from 348,862 to 321,544 pairs (filtering rate: 7.8%).

### Loss & Training

- Four open-source MLLMs are fine-tuned: LLaVA-Next-Llama3-8B (LoRA), MiniCPM-V2.6 (LoRA), Phi-3-Vision (full parameters), and Qwen2.5-VL-7B (LoRA).
- The vision tower is frozen; only the remaining parameters are updated.
- Training runs for 1 epoch with a learning rate of 1e-4 (LoRA) or 5e-6 (full fine-tuning).
- Evaluation metric: GPT-Acc (GPT-4o extracts and evaluates answer correctness).

## Key Experimental Results

### Main Results (ECD Fine-tuning Performance)

| Model | CharXiv Avg | ChartQA | ChartX | ECDBench Avg |
|-------|------------|---------|--------|-------------|
| LLaVA-Next-8B | 35.06 | 64.56 | 27.69 | 10.95 |
| + ECD | **51.60** (+16.54) | **68.64** (+4.08) | **46.61** (+18.92) | **31.58** (+20.63) |
| Phi-3-Vision | 54.72 | 81.92 | 67.53 | 31.41 |
| + ECD | **61.08** (+6.36) | **84.88** (+2.96) | **71.44** (+3.91) | **44.40** (+12.99) |
| Qwen2.5-VL-7B | 61.36 | 83.04 | 67.80 | 38.19 |
| + ECD | **67.40** (+6.04) | **85.32** (+2.28) | **70.83** (+3.03) | **50.86** (+12.67) |

ECD yields consistent improvements across all four MLLMs and six evaluation benchmarks.

### Ablation Study (Impact of Design Choices)

**Comparison with Other Training Sets (LLaVA-Next baseline)**:

| Training Set | CharXiv | ChartQA | ReachQA | ChartX | ECDBench |
|-------------|---------|---------|---------|--------|----------|
| No fine-tuning | 35.06 | 64.56 | 15.65 | 27.69 | 10.95 |
| ChartQA | 35.16 | **68.92** | 15.00 | 31.51 | 13.11 |
| ChartBench | 32.86↓ | 61.56↓ | 18.35 | 37.33 | 10.99 |
| ReachQA | 30.68↓ | 64.50 | **24.35** | 39.24 | 13.48 |
| **ECD** | **51.60** | 68.64 | 25.10 | **46.61** | **31.58** |

Other training sets typically improve only benchmarks whose distribution closely matches their own, and may even degrade performance on others. ECD is the only training dataset that consistently improves all six benchmarks.

**Data Scale Effect**: Performance improves monotonically as the number of training images scales from 2k to 40k; ReachQA shows continuous gains (18.25→24.75), while CharXiv saturates beyond 20k.

**Visual Diversification Effect**: FID decreases by 19.64 (80.38→60.74) and mean entropy increases by 0.57 (1.67→2.24), confirming that diversification substantially reduces the distributional gap with real charts.

### Key Findings

- **Modularization is critical**: Decoupling function and data generation is more effective than end-to-end generation, allowing the LLM to focus on data complexity.
- **Chart type diversity yields consistent gains**: Increasing chart types from 5 to 29 produces continuous improvements on CharXiv.
- **Reasoning QA contributes more than descriptive QA**: Reasoning QA alone yields a 6.40% gain; descriptive QA yields 4.16%; combining both is optimal (47.02%).
- **ECD achieves the lowest FID**: The FID against CharXiv (real charts) is 60.74, substantially lower than other synthetic datasets, indicating that ECD is more representative of real scientific charts.
- **QA filtering provides a positive but modest benefit** (+0.38%); image quality filtering has a larger impact.
- Minor performance drops are observed on a few metrics (e.g., ChartBench binary questions) due to distributional mismatch.

## Highlights & Insights

- **Conceptually clean and effective**: Gains are achieved through a better data synthesis pipeline rather than larger models or more data.
- **Thorough data analysis**: FID, mean entropy, and ablation experiments collectively validate the necessity of each pipeline stage.
- **Strong generalizability**: The approach is effective across four MLLMs of different architectures without any modification to model design.
- **Rigorous ECDBench construction**: A two-stage human review process ensures test set quality.

## Limitations & Future Work

- The 10k image scale is relatively modest; scaling up may yield further improvements, particularly at the pre-training stage.
- For models with strong existing chart pre-training (e.g., Qwen2.5-VL), gains on some metrics are limited or slightly negative.
- The optimal ratio of descriptive to reasoning QA pairs has not been thoroughly explored (experiments suggest 1:1 or 2:3 may be preferable).
- The pipeline relies on GPT-4o for data generation and quality evaluation, incurring non-trivial costs.
- The 29 chart functions are manually defined and may not provide sufficient coverage (e.g., 3D charts, network graphs).

## Related Work & Insights

- The CharXiv benchmark highlights the difficulty of real-world chart understanding (open-source models achieve only 30–50%).
- The synthesis logic for programmatic images (charts, SVGs) differs fundamentally from that for natural images, enabling lossless and precise generation.
- Conditional multi-panel generation to ensure cross-figure consistency is a design pattern worth adapting in related settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The five-stage pipeline is systematically designed; the modularization idea is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four models, six benchmarks, detailed ablations and comparative analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with high-quality figures and visualizations.
- **Value**: ⭐⭐⭐⭐ — The systematic methodology for data synthesis is broadly applicable to other programmable image domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)

</div>

<!-- RELATED:END -->

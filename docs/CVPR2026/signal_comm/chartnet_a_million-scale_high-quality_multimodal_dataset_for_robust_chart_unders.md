---
title: >-
  [Paper Note] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding
description: >-
  [CVPR 2026][Signal & Communication][chart understanding] This paper presents ChartNet, a million-scale chart understanding dataset comprising 1.5 million high-quality multimodal aligned samples. Generated through a code-guided synthesis pipeline, the dataset covers 24 chart types and 6 plotting libraries, with each sample organized as a quintuple (code, image, data table, text description, QA with reasoning). A 2B model fine-tuned on ChartNet surpasses GPT-4o and 72B open-source models.
tags:
  - CVPR 2026
  - "Signal & Communication"
  - chart understanding
  - multimodal dataset
  - code-guided synthesis
  - vision-language models
  - data visualization
date: 2026-05-08
content_hash: 85d0f51d7f490662
---

# ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.27064](https://arxiv.org/abs/2603.27064)
**Code**: [HuggingFace](https://huggingface.co/datasets/ibm-granite/ChartNet)
**Area**: Signal & Communication
**Keywords**: chart understanding, multimodal dataset, code-guided synthesis, vision-language models, data visualization

## TL;DR

This paper presents ChartNet, a million-scale chart understanding dataset comprising 1.5 million high-quality multimodal aligned samples. Generated through a code-guided synthesis pipeline, the dataset covers 24 chart types and 6 plotting libraries, with each sample organized as a quintuple (code, image, data table, text description, QA with reasoning). A 2B model fine-tuned on ChartNet surpasses GPT-4o and 72B open-source models.

## Background & Motivation

Chart understanding requires models to simultaneously reason over geometric visual patterns, structured numerical data, and natural language — a known weakness of current vision-language models (VLMs):

1. **Severe data bottleneck**: Existing datasets exhibit clear deficiencies in scale, scope, or multimodal coverage. Many focus on a single task (e.g., QA or captioning) and lack critical modalities such as plotting code, grounding annotations, or reasoning traces.
2. **Limited chart type diversity**: Widely used benchmarks such as ChartQA cover only 3 chart types (bar, line, pie) and are biased toward basic data extraction questions.
3. **Insufficient scale**: Most datasets contain tens of thousands to hundreds of thousands of samples — insufficient for training frontier large models.
4. **Absent multimodal alignment**: Few datasets simultaneously provide complete alignment across chart images, executable code, underlying data tables, text descriptions, and reasoning chains.

The central insight of ChartNet is that **charts are procedurally generated** — executable plotting code serves as a structured intermediate representation of data visualizations, enabling data generation and augmentation to be performed in code space rather than image space.

## Method

### Overall Architecture

The ChartNet data generation pipeline consists of five stages:
1. Chart-to-code reconstruction: a VLM converts seed chart images into executable plotting code.
2. Code-guided chart augmentation: an LLM iteratively rewrites the code to generate diverse variants.
3. Chart rendering: code is executed to produce chart images.
4. Quality filtering: a VLM detects visual defects and removes defective samples.
5. Code-guided attribute generation: data tables and text descriptions are extracted using both visual and code context.

### Key Designs

1. **Code-guided synthesis pipeline**: 150,000 seed chart images are selected from the TinyChart dataset. `pixtral-large-instruct-2411` converts each image into Python plotting code (Chart-to-Code Reconstruction), and `gpt-oss-120b` iteratively rewrites the code to produce diverse variants. Data values and labels are transformed while preserving contextual relevance, allowing an arbitrary number of variants to be generated from a single seed image. Code execution success rate is approximately 77%; after visual quality filtering, approximately 36.5% of images are removed due to visual defects. Human evaluation shows that after filtering only 5.9% of charts contain issues affecting readability (compared to 14.9% before filtering).

2. **QA generation with CoT reasoning**: Building on the Vision-R1 framework, `pixtral-large-instruct-2411` generates complex multi-step reasoning questions for each image. A four-step "Pseudo-CoT" sequence is then constructed (Summary → Caption → Reasoning → Conclusion). Modality bridging enables language models to reason effectively without direct visual input, and `gpt-oss-120b` finally generates detailed reasoning traces (with `<think>` and `<answer>` tags).

3. **Specialized subsets for full-spectrum coverage**:
    - **Human-annotated subset** (96,643 samples): high-quality aligned data with rigorous human verification and annotation.
    - **Real-world charts** (30K samples): sourced from authoritative outlets such as the World Bank and Pew Research, covering economics, technology, environment, and other domains.
    - **Grounding QA pairs**: geometry-aware annotations extracted from plotting code to generate templated grounding QA.
    - **Safety alignment data** (7,600 samples): adversarial questions and safe/unsafe response pairs on sensitive topics, designed for DPO training.

### Loss & Training

Standard supervised fine-tuning (SFT) is applied to train VLMs:
- Training data spans four tasks: Chart-to-Code, Chart-to-Table, Chart-to-Text, and Chart QA with CoT Reasoning.
- Default hyperparameters from the TRL framework are used for all models.
- Evaluation is conducted on an independent held-out set of 2,000 samples.
- Automated evaluation uses GPT-4o as the judge (except for QA tasks, which use RapidFuzz fuzzy matching).

## Key Experimental Results

### Main Results — ChartNet Evaluation Set

| Model | Chart Recon (Exec/Code-D/Code-S/Img) | Data Extract | Summary | QA w/CoT |
|------|---------------------------------------|-------------|---------|----------|
| granite-vision-2B | 63.4/60.7/67.0/77.2 | 53.8 | 64.0 | 59.9 |
| **+ ChartNet** | **90.4/72.8/90.0/92.8** | **70.3** | **83.9** | **65.0** |
| llava-7B | 45.3/27.0/52.9/59.6 | 17.0 | 51.2 | 55.1 |
| **+ ChartNet** | **83.9/69.4/88.6/91.5** | **58.8** | **80.3** | **70.3** |
| GPT-4o | 95.9/48.8/77.2/88.2 | 46.7 | 77.1 | 61.1 |

After fine-tuning, the **2B model surpasses GPT-4o** (data extraction: 70.3 vs. 46.7; summary: 83.9 vs. 77.1).

Comparison with larger off-the-shelf models:

| Model | Data Extract | Summary | QA w/CoT |
|------|-------------|---------|----------|
| Qwen2-VL-72B | 50.3 | 75.9 | 60.3 |
| Mistral-24B | 53.2 | 79.8 | 60.0 |
| **granite-2B + ChartNet** | **70.3** | **83.9** | **65.0** |

### Ablation Study / Public Benchmarks

ChartCap summarization (granite-vision-2B):
- Baseline: BLEU_4=1.6, METEOR=6.4, ROUGE_L=9.6
- +ChartNet: BLEU_4=12.4, METEOR=30.1, ROUGE_L=24.9

ChartMimic-v2 code generation (granite-vision-2B):
- Baseline: v2-direct=30.84
- +ChartNet: v2-direct=58.42 (+27.58)

Ultra-compact models also achieve substantial capability gains: SmolVLM-256M and Granite-Docling-258M transition from near-zero to usable performance.

### Key Findings

1. **Data quality > model scale**: In domains such as chart understanding, where visual, numerical, and linguistic signals are tightly coupled, providing high-quality code-aligned multimodal supervision is far more effective than simply scaling model size.
2. **Consistent gains across scales**: All models from 256M to 7B parameters achieve significant improvements across all tasks, with gain magnitude largely independent of model size.
3. **Value of code as an intermediate representation**: Chart-to-Code alignment training provides models with structured supervision for programmatic chart understanding.
4. **Largest gains on data extraction**: GPT-4o achieves only 46.7%, whereas a ChartNet fine-tuned 2B model reaches 70.3%, demonstrating the value of tight code–data–image alignment.
5. **Synthetic data generalizes to real-world settings**: Gains transfer effectively to real-world benchmarks such as ChartCap and ChartMimic-v2.

## Highlights & Insights

- **Performing data augmentation in code space rather than image space** is an elegant design choice — code naturally provides structured data representations, enabling more precise multimodal alignment.
- The quintuple alignment (code, image, data table, text, reasoning QA) is more complete than any existing dataset.
- The result that a 2B model outperforms GPT-4o and 72B models strongly substantiates the value of domain-specific, high-quality data.
- The safety alignment subset provides infrastructure for AI safety research in the chart domain.

## Limitations & Future Work

- The dataset is predominantly synthetic; although a real-world subset exists, its proportion is small and domain shift may be present.
- Seed charts originate from a single source (TinyChart), which may constrain initial diversity.
- A code execution success rate of 77% implies approximately 23% of generated samples are discarded.
- After visual filtering, 5.9% of charts still exhibit quality issues.
- Evaluation relies on GPT-4o as a judge, which may introduce systematic bias.
- The dataset lacks evaluation of deeper capabilities such as mathematical reasoning and statistical analysis over charts.

## Related Work & Insights

- **UniChart/TinyChart**: Pioneering multi-task chart datasets, but inferior to ChartNet in scale and modality coverage.
- **ChartQA**: Widely used but limited to 3 chart types and 14K samples; performance on this benchmark is approaching saturation.
- **CoSyn**: Also employs code-guided synthesis, but is restricted to 3 plotting libraries and fewer chart types.
- **Insight**: The code-guided data synthesis paradigm is generalizable to other visual understanding tasks involving procedurally generated content (e.g., 3D scene understanding, UI understanding).

## Rating

- **Novelty**: 7/10 — The concept of code-guided synthesis is not entirely new, but the systematic and large-scale execution is commendable.
- **Experimental Thoroughness**: 9/10 — Comprehensive evaluation across multiple models, scales, tasks, and benchmarks.
- **Writing Quality**: 8/10 — Clear structure, detailed comparison tables, and well-articulated contributions.
- **Value**: 9/10 — As the largest open-source chart understanding dataset, its community value is substantial; the "2B > GPT-4o" result is particularly impressive.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Robust Preference Alignment via Directional Neighborhood Consensus](../../ICLR2026/signal_comm/robust_preference_alignment_via_directional_neighborhood_consensus.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[ACL 2026\] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](../../ACL2026/signal_comm/solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](../../NeurIPS2025/signal_comm/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](clay_conditional_visual_similarity.md)

<!-- RELATED:END -->

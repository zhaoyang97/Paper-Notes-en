---
title: >-
  [Paper Note] PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography
description: >-
  [AAAI 2026][Multimodal VLM][PET imaging] This paper presents PET2Rep, the first large-scale benchmark dataset dedicated to positron emission tomography (PET) radiology report generation, comprising 565 whole-body PET/CT image-report pairs. It further introduces PET Clinical Efficacy (CE) evaluation metrics and conducts a systematic assessment of 30 state-of-the-art general-purpose and medical-specialized VLMs, revealing that current SOTA VLMs perform poorly on PET report generation and fail to outperform even simple template baselines.
tags:
  - AAAI 2026
  - Multimodal VLM
  - PET imaging
  - radiology report generation
  - vision-language model
  - benchmark evaluation
  - clinical efficacy metrics
date: 2026-05-08
content_hash: 273bc093e9dc4e78
---

# PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography

**Conference**: AAAI 2026
**arXiv**: [2508.04062](https://arxiv.org/abs/2508.04062)
**Code**: [https://github.com/YichiZhang98/PET2Rep](https://github.com/YichiZhang98/PET2Rep)
**Area**: Multimodal VLM
**Keywords**: PET imaging, radiology report generation, vision-language model, benchmark evaluation, clinical efficacy metrics

## TL;DR

This paper presents PET2Rep, the first large-scale benchmark dataset dedicated to positron emission tomography (PET) radiology report generation, comprising 565 whole-body PET/CT image-report pairs. It further introduces PET Clinical Efficacy (CE) evaluation metrics and conducts a systematic assessment of 30 state-of-the-art general-purpose and medical-specialized VLMs, revealing that current SOTA VLMs perform poorly on PET report generation and fail to outperform even simple template baselines.

## Background & Motivation

PET is a cornerstone of modern oncological and neurological imaging, visualizing metabolic information by tracking radiotracer distribution and enabling early disease detection prior to anatomical changes. Radiology reports are critical to clinical decision-making, yet their manual composition is time-consuming and places a heavy administrative burden on radiologists.

Recent years have witnessed the growing potential of VLMs in medical applications. However, existing medical applications of VLMs have predominantly focused on structural imaging modalities (X-ray, CT), while the unique characteristics of PET imaging—molecular-level imaging, metabolic information, and radiotracer uptake pattern interpretation—have been largely overlooked.

**Core Problem**: *How far are VLMs from effective PET radiology report generation?* No dedicated dataset or evaluation framework currently exists to answer this question.

The unique challenges of PET report generation include: (1) the need to integrate functional and anatomical information; (2) specialized expertise required for interpreting radiotracer uptake patterns; (3) whole-body imaging spanning dozens of organs demanding broad medical knowledge; and (4) the inability of existing NLG metrics to assess diagnostic accuracy.

## Method

### Overall Architecture

PET2Rep is an evaluation benchmark rather than a novel model architecture. Its core contributions lie in dataset construction, evaluation pipeline design, and the systematic assessment of 30 VLMs.

**Pipeline**: PET/CT image preprocessing (CT resampled to PET resolution, z-score normalization, SUV normalization, PET/CT fusion) → key slice selection (coronal plane) → VLM inference (with standardized prompts) → comparison against ground-truth reports (NLG metrics + CE metrics + human evaluation).

### Key Designs

1. **Dataset Construction**:

    - **Function**: Constructing the first dedicated PET/CT report generation dataset.
    - **Mechanism**: 565 whole-body FDG PET/CT scans with paired structured radiology reports are collected from real clinical settings. Reports follow a radiology training template, systematically describing all detected abnormalities from head to toe.
    - **Design Motivation**: (1) Existing medical benchmarks are largely confined to specific anatomical regions (chest X-ray, abdominal CT), whereas PET2Rep covers the whole body from the head and neck to the proximal extremities. (2) Data are sourced from actual clinical practice rather than reworked public image repositories, mitigating data leakage risks and superficial task design.

2. **Key Slice Selection Strategy**:

    - **Function**: Converting 3D PET/CT volumes into 2D slice inputs for existing VLMs.
    - **Mechanism**: The coronal plane is adopted as the slice sampling view (following clinical convention). Two input modes are designed:
        - *Separate input*: One PET and one CT slice at each key position (6 images total).
        - *Fused input*: Pseudo-color PET overlaid onto grayscale CT (3 fused images total).
    - **Design Motivation**: Separate input tests the model's ability to learn functional-structural associations, while fused input simulates the visualization format ultimately used by radiologists.

3. **PET Clinical Efficacy (CE) Metrics**:

    - **Function**: Evaluating the quality of radiotracer uptake pattern descriptions for key organs in generated reports.
    - **Mechanism**: 19 key organs/structures are defined, with four uptake states extracted for each (increased uptake, decreased uptake, absent uptake, normal). Report evaluation is reformulated from a text-matching problem into a multi-label classification evaluation. Macro-averaged precision, recall, and F1 are computed across the three abnormal classes (increased/decreased/absent).
    - **Design Motivation**: NLG metrics (BLEU, ROUGE-L, METEOR) assess only textual similarity and cannot distinguish reports with opposite diagnostic conclusions but similar wording. CE metrics better reflect the core requirements of clinical diagnosis.

4. **Standardized Prompt Design**:

    - **Function**: Providing a unified input format for VLMs.
    - **Mechanism**: Prompts include imaging modality descriptions, clinical task specifications, and a structured report template based on radiology training guidelines.
    - **Design Motivation**: Ensuring that image interpretations are expressed in a format consistent with expert-written reports for fair comparison.

### Evaluation Setup

- Zero-shot evaluation to assess model generalization.
- 30 VLMs evaluated: 19 general-purpose VLMs + 11 medical-specialized VLMs.
- General models: Qwen2.5-VL series, InternVL3 series, Yi-VL series, LLaVA, DeepSeek-VL2, etc.
- Medical models: LLaVA-Med, Med-Flamingo, MedGemma series, Lingshu series, MedVLM-R1, etc.
- Closed-source models: Gemini 2.5 Pro, GPT-4o, Moonshot-v1, Qwen-VL-Max.

## Key Experimental Results

### Main Results (Representative Model Performance, Fused Input Mode)

| Model | BL-4 | MTR | RG-L | CE-Pre | CE-Rec | CE-F1 | Overall (%) |
|-------|------|-----|------|--------|--------|-------|-------------|
| Template Baseline | 0.315 | 0.148 | 0.511 | 0.228 | 0.222 | 0.225 | 27.5 |
| Qwen2.5-VL-7B | 0.306 | 0.139 | 0.509 | 0.228 | 0.202 | 0.214 | 26.6 |
| MedGemma-4B | 0.287 | 0.121 | 0.488 | 0.236 | 0.225 | 0.230 | 26.4 |
| Lingshu-32B | 0.299 | 0.153 | 0.494 | 0.233 | 0.207 | 0.219 | 26.8 |
| GPT-4o | 0.213 | 0.032 | 0.417 | 0.254 | 0.073 | 0.113 | 18.5 |
| Gemini 2.5 Pro | 0.154 | 0.020 | 0.403 | 0.239 | 0.031 | 0.055 | 15.0 |

### Ablation Study (Model Scale vs. Performance)

| Model Family | Smaller Model | Larger Model | Trend |
|--------------|---------------|--------------|-------|
| Qwen2.5-VL | 7B: 26.6% | 72B: 18.7% | Larger performs worse |
| InternVL3 | 8B: 24.4% | 78B: 22.7% | Larger performs worse |
| MedGemma | 4B: 26.4% | 27B: 20.1% | Larger performs worse |

### Key Findings

- **Across-the-board failure**: All VLMs exhibit limited performance on PET report generation, with most unable to surpass a simple template baseline.
- **SOTA barely matches baseline**: The best-performing models (Lingshu-32B, MedGemma-4B) achieve overall scores of only ~26–27%, on par with the template baseline.
- **Bigger is not better**: Larger models within the same family perform worse, likely due to the lack of domain-specific data and task-oriented training.
- **Closed-source models underperform**: GPT-4o (18.5%) and Gemini 2.5 Pro (15.0%) are substantially worse than smaller open-source models.
- **High NLG ≠ diagnostic accuracy**: Some models achieve acceptable NLG scores but extremely low CE scores, indicating the generation of fluent yet clinically uninformative text.
- **Human evaluation confirms**: Two radiologists independently confirm that current model outputs are largely unusable.
- **Diverse failure modes**: Refusal to answer, empty outputs, non-compliance with the report template, and generation of irrelevant information (e.g., fabricated patient names and ages).

## Highlights & Insights

- **Pioneering contribution**: The first PET/CT report generation benchmark, filling a critical gap in VLM evaluation for functional imaging.
- **Whole-body coverage**: Encompasses 19 key organs/structures, far exceeding existing benchmarks limited to the chest or abdomen.
- **CE metric design**: Reformulates report evaluation from text matching into a multi-label classification problem, more closely reflecting the essence of clinical diagnosis.
- **"Larger is dumber" phenomenon**: Reveals the counterintuitive finding that larger models may underperform smaller ones on highly specialized structured tasks.
- **Real clinical data**: Avoids data leakage from public datasets, providing a genuine assessment of VLM generalization capability.

## Limitations & Future Work

- Only 2D slices are used, failing to fully exploit 3D spatial and volumetric information.
- Key quantitative indicators such as SUV values and lesion volume are not incorporated.
- Currently supports only Chinese-language reports, lacking multilingual evaluation.
- Dataset size (565 cases) is relatively limited.
- Only zero-shot evaluation is conducted; fine-tuned performance remains unexplored.
- Coronal plane slices may miss lesions better visualized in axial views.

## Related Work & Insights

- **vs. CT2Rep**: CT report generation focuses on specific anatomical regions, whereas PET2Rep requires comprehensive whole-body assessment.
- **vs. chest X-ray benchmarks (GEMeX)**: X-ray benchmarks focus on chest pathology, while PET requires metabolic information interpretation.
- **vs. GMAI-MMBench**: Existing medical multimodal benchmarks predominantly adopt VQA formats testing surface-level understanding; PET2Rep demands deep clinical reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First PET report generation benchmark with innovative CE metric design, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation of 30 models across NLG, CE, and human assessment dimensions with multiple input modes.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem motivation, in-depth experimental analysis, and valuable failure mode taxonomy.
- **Value**: ⭐⭐⭐⭐⭐ — Exposes the substantial gap of VLMs in functional imaging report generation and charts a direction for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[ACL 2026\] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](../../ACL2026/multimodal_vlm/coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)
- [\[AAAI 2026\] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](astar_boosting_multimodal_reasoning_with_automated_structure.md)
- [\[AAAI 2026\] Towards Long-window Anchoring in Vision-Language Model Distillation](towards_long-window_anchoring_in_vision-language_model_distillation.md)
- [\[AAAI 2026\] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit](llmc_benchmarking_vision-language_model_compression_with_a_plug-and-play_toolkit.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis
description: >-
  [ICCV 2025][Medical Imaging][Medical Visual Question Answering] This paper presents GEMeX, the largest chest X-ray VQA dataset to date (151K images, 1.6M questions), which for the first time simultaneously provides textual reasoning explanations and visual region grounding across four question types, and systematically evaluates 12 representative large vision-language models.
tags:
  - ICCV 2025
  - Medical Imaging
  - Medical Visual Question Answering
  - Chest X-ray
  - Explainability
  - Visual Grounding
  - Large-Scale Benchmark
date: 2026-05-08
content_hash: 74934baf109508e6
---

# GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis

**Conference**: ICCV 2025
**arXiv**: [2411.16778](https://arxiv.org/abs/2411.16778)
**Code**: [www.med-vqa.com/GEMeX](https://www.med-vqa.com/GEMeX)
**Area**: Medical Imaging / Medical VQA
**Keywords**: Medical Visual Question Answering, Chest X-ray, Explainability, Visual Grounding, Large-Scale Benchmark

## TL;DR

This paper presents GEMeX, the largest chest X-ray VQA dataset to date (151K images, 1.6M questions), which for the first time simultaneously provides textual reasoning explanations and visual region grounding across four question types, and systematically evaluates 12 representative large vision-language models.

## Background & Motivation

Existing medical VQA datasets suffer from two critical shortcomings:

**Lack of explainability**: Only answers are provided without visual or textual explanations, making it difficult for patients and junior clinicians to understand the diagnostic rationale.

**Limited question diversity**: Datasets typically include only open-ended and closed-ended questions, excluding multiple-choice formats, which fails to reflect the diverse demands of real clinical scenarios.

Existing datasets such as VQA-RAD (3.5K QA), SLAKE (14K QA), and MIMIC-CXR-VQA (377K QA) are limited in scale and all lack multimodal explainability.

The core contribution of GEMeX is its **simultaneous provision of textual reasoning and visual grounding (bounding boxes) as multimodal explainability**, along with support for four question types.

## Method

### Overall Architecture

A two-stage construction pipeline:
- **Stage I**: Report Re-grounding — refining anatomical region-to-text correspondences from Chest ImaGenome.
- **Stage II**: VQA Generation — generating diverse QA pairs from the refined reports using GPT-4o.

### Key Designs

1. **Report Re-grounding**:

    - **Problem**: In Chest ImaGenome, a single sentence may correspond to multiple anatomical regions (e.g., "blunting of the right costophrenic angle" maps to both "right lung" and "right lower lung zone"), introducing ambiguity.
    - **Solution**: In collaboration with radiologists, 29 regions are consolidated into 30 core pathological regions, ensuring each sentence corresponds to a single precise region.
    - OpenBioLLM-70B is used for automatic re-grounding, with iterative prompt refinement guided by radiologist feedback, achieving a final accuracy of 98.4%.
    - Sentences spanning multiple regions are split and rewritten (e.g., "cardiomediastinal silhouette is normal" → separate statements for cardiac silhouette and mediastinum).

2. **VQA Generation and Quality Control**:

    - GPT-4o (2024-08-06) is used as the generator.
    - 11 questions are generated per image: 3 open-ended + 2 closed-ended + 3 single-choice + 3 multiple-choice.
    - Each QA pair is accompanied by: an answer, textual reasoning, and visual region annotation.
    - Question content spans 7 categories: abnormality, disease, location, cause, size, severity, and implication.
    - Quality control: 30 images are manually annotated as few-shot demonstrations, with a 50-sample pre-review of prompt effectiveness.
    - The test set is reviewed by radiologists on a per-sample basis: only 10 incorrect answers and 3 inaccurate groundings were corrected.

3. **LLaVA-Med-GEMeX Baseline Model**:

    - Based on LLaVA-Med-v1-7B with question-type-aware instruction tuning.
    - Each sample includes a type prompt, answer, textual reasoning, and visual location.
    - Each of the four question types uses a distinct Supplement format (e.g., closed-ended: "yes or no").

### Loss & Training

- Standard LLaVA instruction fine-tuning loss (next token prediction).
- Training samples include complete answer, reasoning, and location information.

## Key Experimental Results

### Main Results (GPTScore AR-score, %)

| Model | Open AR | Closed AR | Single AR | Multi AR | Avg |
|------|---------|-----------|-----------|----------|-----|
| GPT-4o-mini | 97.68 | 71.14 | 77.47 | 82.91 | 82.30 |
| LLaVA-v1 | 76.14 | 38.02 | 50.47 | 66.52 | 57.79 |
| LLaVA-v1.5 | 77.62 | 57.00 | 57.05 | 65.17 | 64.21 |
| DeepSeek-VL | 79.30 | 59.86 | 62.03 | 70.35 | 67.89 |
| LLaVA-Med-v1 | 90.34 | 69.91 | 61.74 | 68.14 | 72.53 |
| LLaVA-Med-v1.5 | 94.43 | 76.54 | 66.04 | 67.28 | 76.07 |
| RadFM | 88.57 | 67.91 | 57.82 | 62.41 | 69.18 |
| **LLaVA-Med-GEMeX** | **97.05** | **80.72** | **81.42** | **84.98** | **86.04** |

V-score (mIoU) — Visual Grounding: LLaVA-Med-GEMeX achieves 51.47 / 53.20 / 54.57 / 47.99 across the four question types, substantially outperforming all other models.

### Ablation Study / Further Analysis

| Metric | LLaVA-Med-v1 (Open) | LLaVA-Med-GEMeX (Open) | Gain |
|---------|---------------------|------------------------|------|
| GPTScore AR | 90.34 | 97.05 | +6.71 |
| BERTScore | 25.14 | 42.69 | +17.55 |
| ROUGE-L | 19.63 | 32.75 | +13.12 |
| BLEU-1 | 15.93 | 25.28 | +9.35 |

Transfer Learning (zero-shot evaluation on SLAKE-CXR):

| Model | Open AR-score | A-score | Closed AR-score |
|------|--------------|---------|-----------------|
| LLaVA-Med-v1 | 73.31 | 56.17 | 62.35 |
| LLaVA-Med-GEMeX | 82.78 | 69.79 | 75.06 |

### Key Findings

- **Existing models perform poorly overall**: With the exception of GPT-4o-mini, most models achieve limited performance on GEMeX, confirming the benchmark's difficulty.
- **Strong models rely on shortcut reasoning**: Despite high AR-scores, GPT-4o-mini achieves only 18–28% mIoU on visual grounding (V-score), suggesting it draws on pretraining memorization rather than genuine multimodal reasoning.
- **Multiple-choice questions expose model weaknesses**: Most medical-domain models fail to produce definitive answers to choice-format questions, despite demonstrating some capacity to analyze the options.
- **Simple fine-tuning yields substantial gains**: LLaVA-Med-GEMeX improves the average AR-score over the LLaVA-Med-v1 baseline by 13.5%.
- **GPTScore vs. NLG metrics**: GPTScore, which captures semantic understanding, is more appropriate for evaluating untuned models; after fine-tuning, NLG metrics better reflect the degree of task learning.

## Highlights & Insights

- **First multimodally explainable medical VQA dataset**: Simultaneously providing textual reasoning and visual grounding fills a critical gap in the field.
- **Scale advantage**: 151K images and 1.6M QA pairs make GEMeX the largest chest X-ray VQA dataset currently available.
- **Four-type question design**: Closely reflects real clinical consultation scenarios; the multiple-choice format facilitates standardized evaluation.
- **Rigorous quality assurance pipeline**: Iterative radiologist-guided refinement combined with GPT-4o generation and manual review achieves over 98% accuracy.
- **High-quality test set**: 300 images individually reviewed by radiologists, supplemented with an additional 600 manually curated questions.

## Limitations & Future Work

- The baseline model LLaVA-Med-GEMeX is task-specific and may suffer performance degradation on other tasks.
- Coverage is limited to chest X-rays; other modalities such as CT and MRI are not included.
- Dataset construction relies on GPT-4o and OpenBioLLM, potentially introducing biases inherent to these models.
- Visual grounding uses bounding boxes rather than segmentation masks, limiting localization precision.
- The test set comprises only 300 images (3,960 QA pairs) and could be expanded further.

## Related Work & Insights

- Chest ImaGenome provides a high-quality structured report foundation that nonetheless requires refinement.
- The LLaVA-Med series demonstrates the effectiveness of instruction tuning in the medical domain.
- The multimodal explainability design is generalizable to other medical imaging diagnostic tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of multimodal explainability and multi-type questions is a significant contribution, though the core contribution lies in dataset construction rather than methodological innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Systematic evaluation of 12 models across multiple metrics, with transfer learning validation and detailed case studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with thorough data statistics and analysis.
- **Value**: ⭐⭐⭐⭐⭐ Provides critical infrastructure for medical VQA research; the explainability design carries practical clinical significance.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](../../NeurIPS2025/medical_imaging/radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[CVPR 2026\] XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening](../../CVPR2026/medical_imaging/xseg_a_large-scale_x-ray_contraband_segmentation_benchmark_for_real-world_securi.md)
- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](../../CVPR2026/medical_imaging/instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](../../NeurIPS2025/medical_imaging/cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/medical_imaging/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)

<!-- RELATED:END -->

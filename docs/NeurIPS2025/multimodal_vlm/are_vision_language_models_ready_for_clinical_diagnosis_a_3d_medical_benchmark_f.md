---
title: >-
  [Paper Note] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track)][Multimodal VLM][VQA] This paper presents DeepTumorVQA, a 3D diagnostic-grade visual question answering benchmark for abdominal CT tumors, comprising 9…
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track)"
  - "Multimodal VLM"
  - "VQA"
  - "3D medical imaging"
  - "CT tumor"
  - "vision-language models"
  - "benchmark"
date: 2026-05-08
content_hash: 56f1ae1aa9118dbb
---

# Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering

**Conference**: NeurIPS 2025 (Datasets & Benchmarks Track)  
**arXiv**: [2505.18915](https://arxiv.org/abs/2505.18915)  
**Code**: [GitHub](https://github.com/Schuture/DeepTumorVQA)  
**Area**: Multimodal VLM  
**Keywords**: VQA, 3D medical imaging, CT tumor, vision-language models, benchmark

## TL;DR

This paper presents DeepTumorVQA, a 3D diagnostic-grade visual question answering benchmark for abdominal CT tumors, comprising 9,262 CT volumes (3.7 million slices) and 395K expert-level questions. It systematically evaluates the clinical diagnostic capability of four state-of-the-art VLMs, finding that current models perform acceptably on measurement tasks but fall far short of clinical requirements in lesion recognition and reasoning.

## Background & Motivation

Vision-language models (VLMs) have achieved remarkable progress on 2D visual tasks, yet their readiness for 3D clinical diagnosis remains unclear. 3D medical diagnosis imposes three stringent demands on models:

**Recognition Precision**: Accurate localization of minute lesions within complex 3D anatomical structures.

**Reasoning Capability**: Cross-slice information integration for spatial reasoning.

**Domain Knowledge**: Clinical medical knowledge to support diagnostic conclusions.

Existing medical VQA benchmarks predominantly focus on 2D images (e.g., X-rays, dermoscopy images) and lack systematic evaluation on 3D CT scans. Furthermore, the question designs in existing benchmarks tend to be overly simplistic and fail to reflect the diagnostic complexity encountered in real clinical workflows.

## Method

### Overall Architecture

The construction pipeline of the DeepTumorVQA benchmark:
1. **Data Collection**: Aggregation of 9,262 abdominal CT volumes from 17 public datasets, covering diverse tumor types and anatomical organs.
2. **Question Generation**: Design of four categories of diagnostic-grade questions, yielding 395K question–answer pairs in total.
3. **Quality Control**: Clinical validity of questions and accuracy of answers verified by medical experts.
4. **Standardized Evaluation**: Establishment of a unified evaluation protocol and metric suite.

### Key Designs

**Four-Category Question Taxonomy**:

| Category | Description | Example | Difficulty |
|----------|-------------|---------|------------|
| Recognition | Detecting lesion presence, counting, classification | "How many liver tumors are present in this CT?" | High |
| Measurement | Measuring lesion/organ size, HU value, etc. | "What is the longest diameter of the largest tumor in mm?" | Medium |
| Visual Reasoning | Visual questions requiring spatial reasoning | "Does the tumor invade the adjacent vasculature?" | High |
| Medical Reasoning | Reasoning questions requiring clinical knowledge | "Based on imaging features, what is the most likely pathological type?" | Very High |

**Dataset Scale and Diversity**:
- 17 public CT datasets covering multiple abdominal organs including the liver, pancreas, and kidneys.
- Multiple tumor types: hepatocellular carcinoma, pancreatic cancer, renal cell carcinoma, cysts, etc.
- Annotations include: voxel spacing, image dimensions, patient sex/age, scanner type, and contrast agent usage.

**Evaluated VLMs**:
- **RadFM**: A 3D medical VLM trained with large-scale multimodal pretraining.
- **M3D**: A multimodal model for 3D medical image understanding.
- **Merlin**: A 3D medical model integrating vision and text.
- **CT-CHAT**: A VLM specifically designed for conversational CT image analysis.

### Loss & Training

As a benchmark paper, this work does not involve training new models. Evaluation is conducted under zero-shot and few-shot settings, using exact match (EM) and F1 score as primary metrics. Multiple-choice questions are assessed by option accuracy; open-ended questions are scored with GPT-assisted evaluation.

## Key Experimental Results

### Main Results

**Overall Performance Comparison Across Four Task Categories**:

| Model | Recognition | Measurement | Visual Reasoning | Medical Reasoning | Overall |
|-------|-------------|-------------|-----------------|-------------------|---------|
| RadFM | **32.4** | **45.8** | **28.6** | **24.1** | **32.7** |
| M3D | 21.5 | 38.2 | 18.9 | 16.3 | 23.7 |
| Merlin | 19.8 | 41.3 | 15.2 | 14.7 | 22.8 |
| CT-CHAT | 24.1 | 36.7 | 20.4 | 18.5 | 24.9 |

**Fine-grained Analysis Across Sub-tasks**:

| Sub-task | RadFM | M3D | Merlin | CT-CHAT |
|----------|-------|-----|--------|---------|
| Lesion Detection | 28.3 | 15.7 | 12.4 | 18.9 |
| Lesion Counting | 18.9 | 10.2 | 8.6 | 12.1 |
| Organ HU Measurement | 52.1 | 44.6 | 48.9 | 42.3 |
| Tumor Size Measurement | 39.5 | 31.8 | 33.7 | 31.1 |
| Spatial Relation Reasoning | 25.4 | 16.3 | 13.8 | 18.7 |
| Pathological Type Classification | 22.8 | 14.8 | 13.1 | 17.2 |

### Ablation Study

- **Effect of Image Preprocessing**: Different windowing settings (e.g., liver window vs. lung window vs. abdominal window) significantly affect recognition performance across organs; the optimal preprocessing strategy varies by task.
- **Visual Module Design**: 3D convolutional encoders vs. 2D slice-by-slice encoders show a 15–20% performance gap on tumor detection tasks.
- **Number of Input Slices**: Increasing from 16 to 64 slices yields approximately 8% improvement on measurement tasks, but provides limited gains on reasoning tasks.

### Key Findings

1. **RadFM stands out**: Large-scale multimodal pretraining on 3D medical images is the primary driver of performance differences; RadFM leads across all categories.
2. **Measurement–Recognition Gap**: All models perform relatively better on measurement tasks (leveraging pixel-level information) but poorly on recognition and reasoning.
3. **Small Tumor Detection is a Bottleneck**: Detection accuracy for tumors with diameters smaller than 2 cm is universally below 15%.
4. **Visual Module is Critical**: Image preprocessing and 3D perception capability are the core factors distinguishing model performance.

## Highlights & Insights

- **Clinically Oriented Evaluation Design**: The four-category question taxonomy mirrors the real workflow of radiologists, progressing from recognition to reasoning in a hierarchical manner.
- **Unprecedented Scale**: 9,262 CT volumes and 395K question–answer pairs constitute the largest 3D medical VQA benchmark to date.
- **Exposing Core Weaknesses**: The benchmark clearly reveals specific failure modes of current VLMs in 3D medical diagnosis, pointing the way for future research.
- **Strong Reproducibility**: Built upon 17 public datasets; the full dataset has been publicly released via HuggingFace.
- **Practical Implications**: Results indicate that simply scaling 2D pretraining data is insufficient to address 3D medical understanding, which necessitates dedicated 3D architectures and training strategies.

## Limitations & Future Work

1. **Coverage Limited to Abdominal CT**: Thoracic, head-and-neck, and other anatomical regions are not included.
2. **Limited Number of Evaluated VLMs**: Only four models are tested; general-purpose VLMs such as GPT-4V are not included.
3. **Degree of Question Generation Automation**: Some questions may exhibit templating tendencies.
4. **Absence of Human Baseline**: No radiologist performance on the same question set is provided as a reference.
5. **Multimodal Fusion Not Considered**: Joint evaluation combining clinical text (e.g., radiology reports) is not explored.

## Related Work & Insights

- **PathVQA, SLAKE**: Pioneering 2D medical VQA benchmarks, but limited to 2D imagery.
- **CT-RATE, RadBench**: Early attempts at 3D medical benchmarks, but of smaller scale.
- **Abdomen Atlas**: One of the data foundations of this work, providing large-scale abdominal CT annotations.
- This paper underscores the importance of the triangular relationship among **data scale × 3D architecture × clinical knowledge** in the development of 3D medical VLMs.

## Rating

- Dataset Value: ⭐⭐⭐⭐⭐
- Evaluation Design: ⭐⭐⭐⭐⭐
- Analytical Depth: ⭐⭐⭐⭐
- Novelty: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](../../ICCV2025/multimodal_vlm/reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ACL 2026\] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](../../ACL2026/multimodal_vlm/wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)

</div>

<!-- RELATED:END -->

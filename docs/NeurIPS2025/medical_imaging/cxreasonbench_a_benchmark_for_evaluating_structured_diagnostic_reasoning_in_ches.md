---
title: >-
  [Paper Note] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays
description: >-
  [NeurIPS 2025][Medical Imaging][chest X-ray] This paper proposes CheXStruct and CXReasonBench — a structured diagnostic reasoning evaluation framework for chest X-rays that employs multi-path…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "chest X-ray"
  - "diagnostic reasoning"
  - "vision-language models"
  - "benchmark"
  - "structured evaluation"
date: 2026-05-08
content_hash: 63260f53fda1bb57
---

# CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays

**Conference**: NeurIPS 2025
**arXiv**: [2505.18087](https://arxiv.org/abs/2505.18087)  
**Code**: [GitHub](https://github.com/ttumyche/CXReasonBench)  
**Area**: Medical Imaging
**Keywords**: chest X-ray, diagnostic reasoning, vision-language models, benchmark, structured evaluation

## TL;DR

This paper proposes CheXStruct and CXReasonBench — a structured diagnostic reasoning evaluation framework for chest X-rays that employs multi-path, multi-stage assessment to reveal critical deficiencies in existing LVLMs at intermediate reasoning steps.

## Background & Motivation

**Background**: Large vision-language models (LVLMs) are increasingly applied in medical imaging, including report generation and visual question answering (VQA). Chest X-rays have become a standard evaluation benchmark due to their clinical relevance and accessibility.

**Limitations of Prior Work**: Existing benchmarks (VQA-RAD, PathVQA, PMC-VQA, etc.) primarily evaluate the correctness of final diagnostic answers, offering little insight into whether models engage in clinically meaningful reasoning processes. Some recent works introduce explanations or visual grounding, but still focus on outputs rather than intermediate reasoning steps.

**Key Challenge**: A model may produce a plausible answer (e.g., identifying an abnormality in the cardiac region) while offering no evidence of having correctly identified the relevant anatomical structures, performed the appropriate measurements, or applied the corresponding clinical rule (e.g., the cardiothoracic ratio). Without evaluation of intermediate steps, it is impossible to distinguish genuine image understanding from shallow pattern matching.

**Goal**: To construct a benchmark capable of evaluating intermediate reasoning steps in diagnostic workflows — assessing not only whether the answer is correct, but whether the reasoning process is clinically sound.

**Key Insight**: Beginning from anatomical segmentation, the framework automatically extracts diagnostic measurements, computes diagnostic indices, applies clinical thresholds, and constructs a complete structured reasoning pipeline as the reference.

**Core Idea**: An automated pipeline extracts structured reasoning steps from chest X-rays, and a multi-path, multi-stage evaluation framework is designed to systematically assess the diagnostic reasoning capabilities of LVLMs.

## Method

### Overall Architecture

The system comprises two complementary components:
- **CheXStruct**: A fully automated pipeline that extracts structured clinical information from chest X-rays (anatomical segmentation → anatomical landmarks → diagnostic measurements → diagnostic indices → clinical threshold determination).
- **CXReasonBench**: A multi-path, multi-stage evaluation framework that assesses model performance at each intermediate stage using CheXStruct-derived reference answers.

### Key Designs

#### CheXStruct Pipeline

**Task Definition**: In collaboration with clinical experts, 12 radiological finding and quality assessment tasks are defined, categorized into two criteria types:
- **Standardized quantifiable criteria**: e.g., cardiomegaly assessed via the cardiothoracic ratio (CTR), defined as the ratio of the maximum horizontal cardiac width to the thoracic width.
- **Expert-defined criteria**: For tasks lacking standardized metrics (e.g., mediastinal widening), proportion-based surrogate indices are designed.

**Anatomical Segmentation**: The CXAS segmentation model is used to obtain the necessary anatomical masks (e.g., cardiac and pulmonary masks).

**Quality Control (QC)**:
- Task-specific QC rules are defined for each task.
- Low-quality samples are automatically filtered out.
- Only samples passing QC are included in benchmark construction.

#### CXReasonBench Evaluation Pipeline

**Initial Diagnostic Decision**: A binary diagnostic question is posed for each case (e.g., "Does this patient have cardiomegaly?"), with the model selecting Yes, No, or "I don't know."

**Path 1: Direct Reasoning Process Evaluation** (when the model provides a definitive answer)
- Stage 1: Diagnostic criteria selection — the model identifies the diagnostic standard to be applied.
- Stage 1.5: Refined criteria adoption — additional criteria are provided for tasks requiring expert-defined standards.
- Stage 2: Anatomical structure identification — the model selects relevant anatomical regions from annotated X-rays.
- Stage 3: Measurement/identification — the model performs computations or interpretations related to the diagnostic standard.
- Stage 4: Final decision — the model renders a judgment based on the Stage 3 results.

**Path 2: Guided Reasoning and Re-evaluation** (when the model responds "I don't know" or rejects expert criteria)
- Stage 1: Anatomical structure identification (with prompting assistance).
- Stage 2: Guided measurement/identification (with detailed visual annotations and computational instructions).
- Stage 3: Final decision.
- **Re-evaluation Path 1**: After guided reasoning, the model is tested on whether it can independently apply the acquired reasoning to new cases.

### Evaluation Metrics

- **Final Stage Completion**: The proportion of cases in which all reasoning stages are successfully completed.
- **Average Reasoning Depth**: The mean number of reasoning stages reached.
- **Decision Alignment**: Consistency between the initial and final diagnostic decisions.
- **Measurement Consistency**: Numerical consistency between Stage 3 and Stage 4.

## Key Experimental Results

### Benchmark Scale

| Metric | Count |
|--------|-------|
| Diagnostic tasks | 12 |
| Evaluation cases | 1,200 |
| Total QA pairs | 18,988 |
| Path 1 QA | 8,044 |
| Path 2 QA | 3,600 |
| Re-eval Path 1 QA | 7,344 |

### Main Results: Path 1 (Greedy Decoding)

| Model | Completion(↑) | Depth(↑) | Consistency(↑) | Alignment(↑) |
|-------|--------------|----------|----------------|-------------|
| Gemini-2.5-Pro | 17.03 (16.24) | 1.96 | 68.4 | 60.88 |
| Gemini-2.5-Flash | 12.83 (8.56) | 1.40 | 43.76 | 50.29 |
| GPT-4.1 | 8.32 | 1.15 | 61.22 | 39.80 |
| Pixtral-Large | 3.73 (2.31) | 1.00 | 28.50 | 36.74 |
| Llama-3.2-90B | 0.38 | 0.53 | 61.27 | 23.32 |
| Qwen2.5-VL-72B | 2.34 (2.12) | 0.67 | 34.67 | 38.45 |
| MedGemma 27B | 3.31 (2.34) | — | — | — |
| HealthGPT-L14 | — | — | — | — |
| RadVLM | — | — | — | — |

### Key Findings

1. **Even the strongest model, Gemini-2.5-Pro, completes all reasoning stages in only 17% of Path 1 cases**, reaching Stage 2 on average.
2. **Visual grounding is the primary bottleneck**: Stage 2 (anatomical structure identification) performance varies markedly by task — single salient structures (e.g., lungs) achieve up to 89%, whereas abstract reference line tasks (e.g., tracheal deviation) fall to 48%.
3. **Closed-source models substantially outperform open-source models**, with the gap concentrated in Stage 2+ visual understanding.
4. **Medical-domain models (HealthGPT, RadVLM) perform relatively well on identification-type tasks** but are considerably weaker on measurement-type tasks requiring arithmetic computation.
5. **Structured guidance (Path 2) improves diagnostic reasoning**, yet most models fail to generalize the acquired reasoning to new cases.

## Highlights & Insights

- **The first chest X-ray benchmark to evaluate intermediate steps of diagnostic reasoning**, bridging the gap between "answer correctness" and "reasoning correctness."
- **The fully automated CheXStruct pipeline** is scalable to large datasets without manual intervention.
- **The multi-path design** (Path 1 + Path 2 + Re-evaluation) provides a comprehensive profile of reasoning capabilities.
- **Exposes a "knowledge–vision" disconnect in LVLMs**: models may correctly identify the appropriate diagnostic standard yet fail to localize the relevant anatomical structures in the image.
- **The two-round evaluation format is methodologically elegant**: the correct answer is withheld in the first round to assess whether models can recognize their own limitations, before correct options are provided.

## Limitations & Future Work

1. **Coverage is limited to structurally derivable findings**: reliance on segmentation models precludes handling of pathology-specific patterns (e.g., opacities, air-fluid levels).
2. **Twelve tasks remain insufficient**: not all clinically relevant chest X-ray diagnoses are covered.
3. **Multiple-choice evaluation format** may underestimate models' open-ended reasoning capabilities.
4. **Errors in the segmentation model itself** may compromise the quality of reference answers.
5. **Longitudinal reasoning is not evaluated** (e.g., follow-up comparisons across time points).

## Related Work & Insights

- Compared to structured frameworks such as Chest ImaGenome, PadChest-GR, and GR-Bench, CheXStruct derives information directly from images rather than reports and operates at a finer granularity.
- CXReasonBench complements the visual grounding evaluation of GEMeX with a stronger emphasis on explicit reasoning processes.
- **Core insight**: Evaluating AI-assisted diagnostic systems cannot rely solely on answer correctness; the clinical soundness of the reasoning process must be scrutinized.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The problem is clearly defined, the pipeline design is systematic and comprehensive, and experiments cover 12 models across 12 tasks. However, task scope is limited to structurally derivable findings, and the multiple-choice evaluation format may not fully reflect real-world clinical reasoning. As a benchmark contribution, the work is highly valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](../../CVPR2026/medical_imaging/instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[NeurIPS 2025\] Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications](mind_the_data_gap_evaluating_vision_systems_in_small_data_applications.md)

</div>

<!-- RELATED:END -->

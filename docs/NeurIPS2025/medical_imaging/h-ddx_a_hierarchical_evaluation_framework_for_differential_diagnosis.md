---
title: >-
  [Paper Note] H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis
description: >-
  [NeurIPS 2025][Medical Imaging][Differential Diagnosis] H-DDx proposes a differential diagnosis evaluation framework grounded in the ICD-10 classification hierarchy. By expanding both predicted and ground-truth diagnoses to their ancestor nodes and computing a Hierarchical Diagnostic F1 (HDF1), the framework rewards "clinically relevant approximate correctness" rather than exact match only. Evaluating 22 LLMs reveals that the domain-specialized model MediPhi rises from 20th to 2nd place under HDF1, an advantage completely obscured by Top-5 metrics.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Differential Diagnosis
  - ICD-10 Hierarchy
  - Hierarchical F1
  - LLM Medical Evaluation
  - Approximate Correctness
date: 2026-05-08
content_hash: 7665ce31fd7f97b0
---

# H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis

**Conference**: NeurIPS 2025
**arXiv**: [2510.03700](https://arxiv.org/abs/2510.03700)
**Code**: To be confirmed
**Area**: Medical AI / LLM Evaluation
**Keywords**: Differential Diagnosis, ICD-10 Hierarchy, Hierarchical F1, LLM Medical Evaluation, Approximate Correctness

## TL;DR
H-DDx proposes a differential diagnosis evaluation framework grounded in the ICD-10 classification hierarchy. By expanding both predicted and ground-truth diagnoses to their ancestor nodes and computing a Hierarchical Diagnostic F1 (HDF1), the framework rewards "clinically relevant approximate correctness" rather than exact match only. Evaluating 22 LLMs reveals that the domain-specialized model MediPhi rises from 20th to 2nd place under HDF1, an advantage completely obscured by Top-5 metrics.

## Background & Motivation

**Background**: LLM differential diagnosis evaluation primarily relies on Top-k accuracy — whether the correct diagnosis appears in the predicted list. Over 22 LLMs have been continuously benchmarked on datasets such as DDXPlus.

**Limitations of Prior Work**: Top-k treats all errors equally — predicting "viral URI" instead of "influenza" (same disease category) receives the same score of 0 as predicting "fracture" (entirely unrelated). This is highly unfair to models that produce clinically approximate predictions.

**Key Challenge**: The ICD-10 coding system naturally encodes hierarchical distances between diseases (chapter/section/category/subcategory), yet existing evaluation metrics entirely ignore this structure.

**Goal**: Design evaluation metrics that respect the ICD-10 hierarchical structure and reward clinically relevant approximate predictions.

**Key Insight**: Expand both the predicted and ground-truth diagnosis sets upward along the ICD-10 tree to all ancestor nodes, then compute precision/recall/F1 on the expanded sets.

**Core Idea**: Expand diagnosis sets along the ICD-10 tree to chapter/section/category ancestor nodes → compute Hierarchical F1 on expanded sets → reward approximate predictions within the same category or section.

## Method

### Overall Architecture
**Mapping**: Natural-language diagnoses → ICD-10 codes (retrieved via text-embedding-3-large + reranked by gpt-4o, Top-1 accuracy 93.1%) → **Expansion**: Each ICD-10 code is expanded upward to all ancestor nodes (chapter → section → category → subcategory) → **HDF1 Computation**: Precision/recall/F1 computed on the expanded sets.

### Key Designs

1. **Diagnosis-to-ICD-10 Mapping Pipeline**:

    - Function: Maps natural-language diagnoses from LLM outputs to standard ICD-10 codes.
    - Mechanism: text-embedding-3-large retrieves top-15 candidates → gpt-4o reranks → Top-1 accuracy of 93.1% (vs. 71.3% for retrieval alone).
    - Design Motivation: LLM outputs vary in phrasing (e.g., "flu" vs. "influenza"), requiring robust normalization to ensure fair comparison.

2. **Hierarchical Expansion + HDF1 Metric**:

    - Function: Computes an F1 score that respects the hierarchical taxonomy of diseases.
    - Mechanism: The predicted set $\hat{D}_i$ and ground-truth set $D_i$ are each expanded to $\hat{C}_i$ and $C_i$ comprising all ancestor nodes. $HDP = \frac{1}{N}\sum_i \frac{|\hat{C}_i \cap C_i|}{|\hat{C}_i|}$, $HDR = \frac{1}{N}\sum_i \frac{|\hat{C}_i \cap C_i|}{|C_i|}$, $HDF1 = 2HDP \cdot HDR / (HDP + HDR)$.
    - Design Motivation: If the prediction is "viral URI" while the ground truth is "influenza," the two share chapter-level (respiratory system) and section-level (upper respiratory infection) ancestors, so HDF1 awards partial credit; Top-k awards 0.

3. **Hierarchical Cascade Analysis**:

    - Function: Computes HDF1 separately at each ICD-10 level (chapter/section/category/subcategory).
    - Mechanism: Truncates expansion depth level by level to analyze how model accuracy changes from coarse to fine granularity.
    - Design Motivation: Reveals whether a model is "directionally correct but imprecise in detail" or "entirely off-track."

### Loss & Training
- This is an evaluation framework; no training is involved.
- 22 LLMs are evaluated on 730 test cases from DDXPlus.

## Key Experimental Results

### Main Results

| Model | Top-5 Rank | HDF1 Rank | HDF1 Score |
|-------|-----------|-----------|------------|
| Claude-Sonnet-4 | — | **1st** | 0.3673 |
| MediPhi | **20th** | **2nd** | 0.35+ |
| GPT-4o | Top-5 | Declined | — |

### Hierarchical Cascade (Average Across All Models)

| ICD-10 Level | HDF1 |
|-------------|------|
| Chapter | ~60% |
| Section | ~40% |
| Category | ~30% |
| Subcategory | 10–20% |

### Key Findings
- MediPhi rises from 20th under Top-5 to 2nd under HDF1 — domain fine-tuning yields far superior performance on the "clinical approximation" dimension compared to general-purpose models.
- Case Study: MediPhi achieves HDF1=0.5714 vs. GPT-4o HDF1=0.2069 on a complex respiratory case, despite GPT-4o achieving Top-5=1.0.
- All models perform reasonably at the chapter level (~60%) but drop sharply at the subcategory level (10–20%), indicating that LLM diagnoses are "directionally correct but insufficiently precise."
- The hierarchical competence of medically fine-tuned models is completely obscured by Top-k metrics.

## Highlights & Insights
- **HDF1 exposes the blind spots of Top-k**: The rank reversal (MediPhi 20→2) demonstrates that existing evaluations severely underestimate the true capability of domain fine-tuned models.
- **Clever utilization of the ICD-10 hierarchical structure**: The framework repurposes an existing medical coding system as a measurement foundation, requiring no additional annotation.
- **Implications for all medical AI evaluation**: Any evaluation task with a hierarchical taxonomy can adopt a similar approach.

## Limitations & Future Work
- DDXPlus is a synthetic dataset; validation on real clinical cases is needed.
- ICD-10 may not fully capture clinical similarity (SNOMED CT may be more appropriate).
- The framework evaluates static lists and does not assess sequential reasoning processes.
- The mapping pipeline depends on gpt-4o, introducing additional bias.

## Related Work & Insights
- **vs. Top-k Accuracy**: Top-k is a flat metric; HDF1 respects the disease hierarchy.
- **vs. BLEU/ROUGE**: These metrics assess textual similarity rather than clinical correctness.

## Rating
- Novelty: ⭐⭐⭐⭐ Hierarchical F1 is a novel and practical metric for medical evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 LLMs + hierarchical cascade analysis + case studies.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; case analyses are persuasive.
- Value: ⭐⭐⭐⭐⭐ Potential to reshape the paradigm of medical AI evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs](shallow_robustness_deep_vulnerabilities_multi-turn_evaluation_of_medical_llms.md)
- [\[NeurIPS 2025\] EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)
- [\[NeurIPS 2025\] UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation](unimrseg_unified_modality-relax_segmentation_via_hierarchical_self-supervised_co.md)
- [\[NeurIPS 2025\] RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis](rad_towards_trustworthy_retrieval-augmented_multi-modal_clinical_diagnosis.md)
- [\[CVPR 2026\] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](../../CVPR2026/medical_imaging/focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)

<!-- RELATED:END -->

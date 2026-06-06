---
title: >-
  [Paper Note] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework
description: >-
  [AAAI 2026][Medical Imaging][Radiology report proofreading] This paper proposes a **staged inference + dual-knowledge infusion** framework that decomposes radiology report error correction into three phases—detection → l…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Radiology report proofreading"
  - "large language models"
  - "knowledge graph distillation"
  - "staged inference"
  - "medical error detection"
date: 2026-05-08
content_hash: 9b181bc04d86394a
---

# Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework

**Conference**: AAAI 2026
**arXiv**: [2406.15045](https://arxiv.org/abs/2406.15045)  
**Code**: [https://github.com/knowlab/MedKIC-Radiology-Proofreading](https://github.com/knowlab/MedKIC-Radiology-Proofreading)  
**Area**: Medical Imaging
**Keywords**: Radiology report proofreading, large language models, knowledge graph distillation, staged inference, medical error detection

## TL;DR

This paper proposes a **staged inference + dual-knowledge infusion** framework that decomposes radiology report error correction into three phases—detection → localization → correction—and integrates **Medical Knowledge Graph Distillation (MKGD)** with **External Knowledge Retrieval (EXKR)** to achieve up to **31.56% improvement in error detection accuracy** and **37.4% reduction in processing time** across 6 LLM architectures.

## Background & Motivation

### Problem Setting
Radiology reports are central documents in precision medicine; however, retrospective studies indicate that approximately **30% of radiological examinations contain documentation errors**, including terminological inconsistencies, negation errors, and contextual contradictions. Error sources include:
- **Speech recognition misidentification**: e.g., "effusion" → "infusion"
- **Template inconsistencies**: formatting errors introduced by standardized templates
- **Terminological confusion**: e.g., conflation of "consolidation" with "congestion"
Such errors lead to delayed diagnoses, inappropriate treatment, and potentially life-threatening consequences.

### Limitations of Prior Work

**Lack of systematic medical knowledge integration**: General-purpose text correction methods optimize for linguistic correctness while ignoring clinical appropriateness—they cannot distinguish "consolidation" from "atelectasis" (both legitimate medical terms representing distinct pathological processes).

**Black-box decision-making**: Existing methods provide no verifiable reasoning process, rendering them incompatible with clinical environments that require validation of AI recommendations.

**Monolithic processing**: Error correction is treated as a single-step task, without modeling the cognitive workflow of domain experts.

**Static training paradigm**: Knowledge is frozen at training time, precluding adaptation to emerging terminology (e.g., COVID-19-related expressions).

### Core Motivation

**Emulating the radiologist's systematic review process**: Experts first (1) determine whether a report contains errors, (2) localize the problematic term (e.g., "congestion" is inappropriate in a pulmonary context), and (3) correct it to "consolidation." This paper explicitly encodes this decomposed process into the framework, while providing real-time medical knowledge support through dual-knowledge infusion.

## Method

### Overall Architecture

The framework consists of two complementary components:
1. **Staged Proofreading Inference**: detection → localization → correction
2. **Dual-Knowledge Infusion**: MKGD + EXKR

### Key Designs

#### 1. **Staged Inference**

The complex proofreading task is decomposed into three focused stages:

- **Stage 1 – Error Detection**: Binary classification—whether the report contains errors. Global patterns and medical consistency indicators are analyzed.
- **Stage 2 – Error Localization**: Fine-grained analysis that precisely identifies the erroneous term or phrase within flagged reports.
- **Stage 3 – Error Correction**: Generates clinically appropriate corrections based on the detected error type and location.
- **Design Motivation**:
    - Each stage focuses on a specific aspect, reducing hallucinations
    - Provides a transparent decision chain that clinicians can validate at each stage
    - The stepwise approach encourages more targeted corrections

#### 2. **Medical Knowledge Graph Distillation (MKGD)**

**RadGraph** is used to convert clinical reports into structured entity graphs:

- **Entity Extraction**:
    - **Anatomical entities (ANAT)**: lungs, heart, ribs, etc.
    - **Observation entities (OBS)**: categorized into three certainty levels—Definitely Present (DP), Uncertain (U), and Definitely Absent (DA)

- **Relation Modeling**:
    - `suggestive_of`: diagnostic inference chains (imaging patterns → pathological conditions)
    - `located_at`: anatomical localization (findings → anatomical sites)
    - `modify`: hierarchical and descriptive relationships (e.g., "mild" modifying "opacity")

- **Graph-to-Text Conversion**: A rule-based system converts structured representations back into natural language sentences via entity classification → semantic integration → logical reasoning (with attention to negation and uncertainty) → sentence construction.
    - Example: `⟨lower, modify, lobe⟩` + `⟨opacity, located_at, lobe⟩` → "lower lobe opacity"
    - Example: `⟨opacity, located_at, lobe⟩` + DA → "no lobe opacity"

- **Design Motivation**: Transforming free text into structured representations enables LLMs to "perceive" inter-entity relationships rather than surface text alone.

#### 3. **External Knowledge Retrieval (EXKR)**

- **Reference database**: 112,251 error-free radiology reports
- **Retrieval method**: Cosine similarity computed using the e5-large-unsupervised embedding model; top-$k$=4 most relevant reports are retrieved
- **Knowledge normalization**: Retrieved reference reports are processed through the same MKGD pipeline as the input report, yielding standardized knowledge statements
- **Contextual integration**: Combines MKGD's structural analysis with the domain knowledge patterns from EXKR
- **Design Motivation**: Compensates for the limitations of analyzing reports in isolation by incorporating broader clinical experience and established medical knowledge patterns

#### 4. **Integration with LLMs**

A carefully designed four-part prompt template architecture:
1. Expert role definition ("You are a radiologist specializing in chest radiology")
2. Stage-specific instructions
3. Structured knowledge integration (MKGD summary + EXKR reference examples)
4. Explicit output format specification

### Loss & Training

- **No fine-tuning required**: The framework augments existing LLMs via prompt engineering and knowledge infusion without costly domain-specific fine-tuning.
- **Evaluation metrics**:
    - Detection and localization: Accuracy (%)
    - Correction: AggNLG = (ROUGE-1 + BERTScore + BLEU) / 3
- **Experimental setup**: 4× NVIDIA RTX 3090, bfloat16 precision, temperature=0.001, top_p=0.8

### Benchmark Dataset Construction

Constructed from the MIMIC-CXR dataset:
- **Reference set**: 112,251 error-free reports (EXKR knowledge base)
- **Evaluation set**: 1,622 reports (512 error-free + 1,110 with introduced errors)
- **Error injection strategies**:
    - **Negation errors**: "no pleural effusion" ↔ "pleural effusion"
    - **Entity-level clinical inconsistencies**: substitutions among 12 key radiological findings (speech recognition confusions, terminological ambiguities, template-related errors)
    - All errors validated by licensed radiologists

## Key Experimental Results

### Main Results (Table 1)

| Model | Detection Baseline → Ours | Localization Baseline → Ours | Correction E2E → Staged → Ours |
|------|-----|-----|-----|
| MMedLM2 | 41.49 → **73.05** (+31.56) | 30.94 → **46.05** (+15.11) | 47.80 → 58.27 → 53.50 |
| Llama3-Aloe | 45.31 → **67.26** (+21.95) | 43.34 → **51.35** (+8.01) | 63.33 → 90.17 → **74.77** |
| Phi3-mini | 67.26 → **73.06** (+5.80) | 47.71 → **52.65** (+4.94) | 74.36 → 74.08 → **78.85** |
| Phi3-small | 79.03 → **80.21** (+1.18) | 63.44 → **65.04** (+1.60) | 80.03 → 86.57 → **86.67** |
| Phi3-medium | 73.67 → **79.04** (+5.37) | 69.73 → 63.44 (−6.29) | 84.47 → 90.25 → **92.25** |
| Llama3-8B | 37.79 → **62.27** (+24.48) | 37.29 → **53.14** (+15.85) | 84.34 → 94.29 → **94.43** |
| **Average** | 57.43 → **72.48** (+15.05) | 48.74 → **55.28** (+6.54) | 72.39 → 82.27 → **80.08** |

### Comparison with Simple RAG (Table 2)

| Metric | Simple RAG Avg. | Ours Avg. | Gain |
|------|----------------|----------|------|
| Error Detection | 62.42% | **72.48%** | +10.06% |
| Error Localization | 48.35% | **55.28%** | +6.93% |
| Error Correction | 76.66 | **80.08** | +3.42 |
| Processing Time | 26.64s | **17.65s** | −34.10% |

Structured knowledge extraction substantially outperforms generic document-level retrieval while being more efficient.

### Ablation Study (Table 3, Llama3-8B)

| Configuration | Detection (%) | Localization (%) | Correction (NLG) | Avg. Improvement |
|------|---------|---------|-----------|----------|
| Baseline | 37.79 | 37.29 | 94.29 | — |
| + MKGD | 37.92 | 36.99 | 94.49 | +0.01% |
| + EXKR | 58.14 | 50.21 | 94.38 | +11.12% |
| + MKGD + EXKR (Ours) | **62.27** | **53.14** | **94.43** | +13.49% |

**Key Findings**:
- MKGD alone yields negligible gains (+0.01%), yet serves as an indispensable structural guide for EXKR
- EXKR drives the primary improvements (detection +20.35%, localization +12.92%)
- MKGD + EXKR produce a synergistic effect: detection +24.48%, localization +15.85%

### Human Evaluation

Two licensed radiologists evaluated 50 representative cases on a 5-point Likert scale:
- **Accuracy**: The proposed method significantly outperforms all baselines
- **Factual consistency**: MMedLM2 in particular achieved the largest factual consistency improvement
- **Clinical relevance**: Phi3-medium and Llama3-8B showed the most notable gains in clinical interpretability

### Key Findings

1. **Value of staged inference**: The End-to-End → Staged baseline transition alone yields substantial improvements (correction: 72.39 → 82.27), validating the effectiveness of cognitive process decomposition.
2. **Complementarity of knowledge infusion**: MKGD provides the structural framework that enables EXKR to function more effectively; neither component is dispensable.
3. **Medical-specialist vs. general-purpose models**: Medical models (MMedLM2, Aloe) underperform at baseline but benefit most from knowledge infusion (detection +31.56%).
4. **Dual gains in efficiency and accuracy**: The structured approach is not only more accurate but also reduces processing time by 27.9%–37.4%.

## Highlights & Insights

1. **Clinical workflow-oriented system design**: The three-stage decomposition emulates expert cognitive processes; each intermediate output is verifiable by clinicians, satisfying transparency requirements for medical AI.
2. **Domain adaptation without fine-tuning**: Medical proofreading capability is enhanced through knowledge infusion rather than fine-tuning, yielding strong scalability at low cost.
3. **Realistic error patterns**: Errors in the benchmark dataset are validated by licensed radiologists and reflect real-world scenarios including speech recognition confusions and terminological ambiguity.
4. **Reduced processing time**: The counterintuitive result of being simultaneously more accurate and faster stems from structured information processing reducing unnecessary context handling.

## Limitations & Future Work

1. **Restricted to English chest X-ray reports**: Multilingual, multimodal (CT/MRI), and other specialties (e.g., neuroradiology) are not covered.
2. **Assumes original reports are error-free**: Original MIMIC-CXR reports are treated as the "error-free" gold standard, though they may contain undetected real errors.
3. **Single-error assumption**: Only one error is introduced per report; real-world scenarios may involve multiple interrelated errors.
4. **Limited standalone effectiveness of MKGD**: Knowledge graph distillation requires integration with retrieval to be effective, suggesting that richer utilization of structured representations may be warranted.
5. **Image information not incorporated**: Proofreading is text-only; original X-ray images are not leveraged to verify the accuracy of report content.

## Related Work & Insights

- **MEDIQA-CORR 2024**: General clinical note correction using synthetic data rather than radiology-specific reports.
- **RadGraph**: A radiology information extraction framework used in this paper to implement the MKGD component.
- **RAG paradigm**: This work demonstrates that simple RAG is inferior to structured knowledge retrieval, providing direction for RAG optimization in the medical domain.
- **Insight**: In specialized domains, a "structure-first, then retrieve" paradigm outperforms direct retrieval of similar documents. Staged inference reduces LLM hallucination risk and represents an important paradigm for medical AI applications.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of staged inference and dual-knowledge infusion is novel, though individual components build on existing methods
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 LLMs, multiple ablations, RAG comparison, human evaluation, and processing time analysis
- Writing Quality: ⭐⭐⭐⭐ — Motivation and methodology are clearly described, though the paper is lengthy
- Value: ⭐⭐⭐⭐⭐ — Directly addresses a clinical pain point; 30% error rate in reports combined with specialist shortages makes the practical significance substantial

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Rethink the Role of Neural Decoders in Quantum Error Correction](../../ICML2026/medical_imaging/rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[AAAI 2026\] Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy](pairing-free_group-level_knowledge_distillation_for_robust_gastrointestinal_lesi.md)
- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](../../ACL2026/medical_imaging/multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)

</div>

<!-- RELATED:END -->

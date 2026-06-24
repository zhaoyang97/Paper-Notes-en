---
title: >-
  [Paper Note] RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification
description: >-
  [ACL 2025][Medical LLM][De-identification] RedactX is proposed as a fully automated, multimodal clinical data de-identification framework. By combining multi-round LLM extraction, rule-based processing, and retrieval-based relexicalization, it achieves an F1 score (0.9646) comparable to specialized commercial systems on the i2b2 dataset, while optimizing token usage efficiency.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "De-identification"
  - "PHI/PII"
  - "LLM"
  - "Clinical Data"
  - "Audio Masking"
date: 2026-05-08
content_hash: b4af94e6757efe75
---

# RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification

**Conference**: ACL 2025  
**arXiv**: [2505.18380](https://arxiv.org/abs/2505.18380)  
**Code**: None (Oracle Internal System)  
**Area**: Medical NLP  
**Keywords**: De-identification, PHI/PII, LLM, Clinical Data, Audio Masking  

## TL;DR

RedactX is proposed as a fully automated, multimodal clinical data de-identification framework. By combining multi-round LLM extraction, rule-based processing, and retrieval-based relexicalization, it achieves an F1 score (0.9646) comparable to specialized commercial systems on the i2b2 dataset, while optimizing token usage efficiency.

## Background & Motivation

- **Background**: AI-driven medical tools are increasingly popular, but Electronic Health Records (EHR) contain a large amount of Protected Health Information (PHI) and Personally Identifiable Information (PII), requiring de-identification in compliance with HIPAA/GDPR regulations.
- **Limitations of Prior Work**: (1) Manual de-identification is infeasible for large-scale clinical data; (2) Rule-based approaches generalize poorly, and BERT-based methods require massive amounts of labeled data and computing resources; (3) LLM-based methods (e.g., zero-shot GPT-4) suffer from insufficient recall and lack consistent substitution capabilities; (4) Most systems only process text, ignoring audio data.
- **Key Challenge**: Even missing a single PHI/PII instance can lead to severe privacy breach consequences, but pursuing high recall easily leads to over-redaction (reducing data utility).
- **Goal**: To build a scalable, highly adaptable, and cost-effective multimodal (text + audio) de-identification system that supports consistent entity substitution (relexicalization).
- **Key Insight**: Utilizing the zero/few-shot capabilities of LLMs combined with multi-round iteration, context-aware entity extraction strategies, and retrieval-based relexicalization.
- **Core Idea**: Multi-chunk, multi-pass LLM entity extraction + schema-driven unified structured/unstructured processing + retrieval-based relexicalization = deployable end-to-end de-identification.

## Method

### Overall Architecture

RedactX consists of three major components: (1) **Auto De-ID** — LLM-driven unstructured text de-identification; (2) **Audio De-ID** — two-step audio de-identification; (3) **Auto Relexicalizer** — multi-agent entity relexicalization. The frontend automatically identifies data types via a Schema Identifier and routes them to the corresponding processing modules.

### Key Designs

#### 1. Schema-Driven Data Processing
- **Function**: Automatically selecting processing strategies based on data types.
- **Mechanism**: Processing flags are defined for each field in the Schema Registry:
    - `passThrough`: Non-PHI fields are passed directly.
    - `shouldMask`: Categorically replaced with general placeholders by rules.
    - `shouldHash`: Hashed to achieve secure cross-document linking.
    - `autoDeID`: Sent to the LLM for de-identification.
- **Design Motivation**: The schema-agnostic design makes the system scalable to different EHR formats, where adding new data types only requires updating schema configurations.

#### 2. Multi-Chunk Multi-Pass LLM Entity Extraction
- **Function**: Multi-pass LLM processing of unstructured text after chunking.
- **Mechanism**: 
    - Split text into chunks of size $\omega$ (256 tokens) to ensure they do not exceed the LLM context window.
    - Pass 1: LLM detects as many entities as possible.
    - Pass 2+: Previously detected entities are masked, forcing the model to focus on missed PHI.
    - Aggregate results from all passes.
- **Context-Awareness**: Each entity includes surrounding words as positional hints (e.g., "76 years old" rather than just "76"), avoiding dependency on unreliable character position indices.
- **Design Motivation**: A single LLM call is prone to missing complex or rare PHI; masking known entities "forces" the LLM to discover new ones.

#### 3. Retrieval-Based Relexicalization (Auto Relexicalizer)
- **Function**: Replacing de-identified placeholders with contextually consistent substitute entities.
- **Mechanism**: A multi-agent pipeline:
  1. **LLM Entity Clustering**: Groups extracted entities based on context (e.g., grouping "Wilson" and "Dr. Adam Wilson" together).
  2. **Hybrid Retrieval**: Vector search + filtered retrieval of existing substitution schemes.
  3. **LLM Verification**: Determines if the retrieved substitution scheme is valid.
  4. **LLM Generation**: Generates new substitutions for invalid schemes.
  5. **OpenSearch Index**: Stores new substitution schemes for future re-use.
- **Design Motivation**: Consistent substitution enhances the "Hiding in Plain Sight" (HIPS) effect—making the substituted entities blend seamlessly with any leaked PHI, significantly increasing the difficulty of re-identification.

#### 4. Two-Step Audio De-Identification
- **Function**: Detection and silencing of PHI in clinical audio.
- **Mechanism**:
    - **Step 1**: ASR transcription → LLM de-identification → Mark timestamps (+ 100-200ms margins).
    - **Step 2**: Utilize an aggressive VAD to detect speech regions not recognized by ASR → Use the LLM to analyze the context of these regions to determine if they likely contain PHI → Silence the most probable regions.
- **Design Motivation**: Step 2 resolves the PHI leakage problem caused by ASR errors (misrecognitions/omissions), which practically improved recall by approximately 10% in testing.

### Loss & Training

This is a system paper and does not have a dedicated training loss function. Auto De-ID directly leverages the zero-shot/few-shot capabilities of GPT-4o.

## Key Experimental Results

### Main Results: Performance Comparison on the i2b2 2014 De-ID Dataset

| System | Precision | Recall | F1 | All-or-Nothing Recall |
|------|-----------|--------|-------|----------------------|
| Y&S_Brief | 0.5634 | 0.6580 | 0.6070 | 0.3700 |
| Y&S_Detail | 0.6178 | 0.8270 | 0.7072 | 0.5600 |
| Altalla | 0.9675 | 0.6715 | 0.7927 | 0.3600 |
| **RedactX** | **0.9769** | 0.9525 | **0.9646** | 0.7900 |
| AWS | 0.9549 | 0.9425 | 0.9487 | 0.7500 |
| JSL | 0.9481 | **0.9865** | 0.9669 | **0.9000** |

- RedactX achieves the highest F1 (0.9646) and highest precision (0.9769) among LLM-based methods.
- Highly comparable to dedicated commercial systems (AWS, JSL).

### Ablation Study: Entity Type Analysis

| Entity Type | RedactX-F1 | AWS-F1 | JSL-F1 |
|---------|-----------|--------|--------|
| CONTACT | **1.0000** | 0.6250 | 0.8814 |
| PERSON | **0.9751** | 0.9461 | 0.9749 |
| DATE | 0.9735 | 0.9561 | **0.9900** |
| LOCATION | 0.8799 | 0.8750 | **0.9636** |
| All | 0.9465 | 0.9270 | **0.9751** |

### LLaMA-3.2-3B Open Source Model Ablation

- The recall improvement is most prominent from Pass 1 to 2, especially for sparse types such as ID, DATE, and LOCATION.
- Most entity types saturate after Pass 3.
- This indicates that the number of passes is a model-dependent hyperparameter: smaller models benefit from 2-3 rounds.

### Key Findings

1. **Multi-pass strategy is effective**: Masking known entities before re-detecting significantly improves recall, performing particularly well for smaller models.
2. **Context-aware extraction outperforms simple extraction**: Carrying positional hints (surrounding words) avoids accuracy issues with position indices.
3. **Balance between precision and recall**: RedactX's multi-pass strategy boosts recall without sacrificing precision, whereas other LLM methods (such as Y&S Detailed) experience a sharp decline in precision when improving recall.
4. **Audio second-step detection yields ~10% recall improvement**: 84% of the additionally silenced content was non-harmful (background noise), imposing minimal impact on clinical utility.
5. **Token optimization**: Extracting only entities and positional hints (instead of fully annotated texts) reduces output tokens by approximately 50%.

## Highlights & Insights

- The **multi-pass masking strategy** is simple but highly effective—once the first-pass detected entities are masked, the LLM's attention naturally shifts to the omitted items.
- **Relexicalization** not only improves data utility but also enhances privacy protection (via the HIPS effect), which is an important aspect overlooked by most de-identification systems.
- **Schema-driven design** allows the same system to process multiple EHR formats, indicating a high degree of industrial capability.
- Practical insights summarized from over 12 months of production deployment (dynamic batching, token optimization, etc.) offer substantial value for industry reference.

## Limitations & Future Work

- Dependence on GPT-4o as the LLM leads to higher costs and data security concerns (handling PHI data via cloud APIs).
- Performance on LOCATION and ID types is relatively weak, requiring refined entity-specific prompting instructions.
- The VAD algorithm is relatively simple, leading to false positives and over-silencing.
- Lack of large-scale generalization evaluation across real-world EHR data from multiple institutions.
- The All-or-Nothing Recall (0.79) still has a gap compared to JSL (0.90).

## Related Work & Insights

- **NeuroNER / BERT-based De-ID**: Traditional deep learning de-identification methods, which require large amounts of labeled data.
- **DeID-GPT**: A pioneer in zero-shot LLM de-identification, but with limited single-call recall.
- **JSL (John Snow Labs)**: An industry-leading commercial De-ID system, fine-tuned specifically for the clinical domain.
- **Vakili et al. (2024)**: Work analyzing pseudonymization, which RedactX extends into an LLM-driven automated solution.
- **Key Insight**: Multimodal (text + audio) privacy protection needs to leverage complementary information across modalities.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐: While the combination of multi-pass + context-aware + relexicalization is novel, individual components are relatively conventional.
- **Value** ⭐⭐⭐⭐⭐: Already deployed in production at Oracle Health for over 12 months, demonstrating high industrial value.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Complete multi-method comparison, entity-level analysis, and ablation experiments.
- **System Design** ⭐⭐⭐⭐⭐: Modular, schema-driven, with multimodal support, showcasing high architecture maturity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives](../../ACL2026/medical_nlp/anonpsy_a_graph-based_framework_for_structure-preserving_de-identification_of_ps.md)
- [\[ACL 2025\] Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training](adaptive-vp_a_framework_for_llm-based_virtual_patients_that_adapts_to_trainees_d.md)
- [\[ACL 2025\] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment](a_modular_approach_for_clinical_slms_driven_by_synthetic_data_with_pre-instructi.md)
- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[ACL 2025\] Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review](clinical_coding_eight_recommendations.md)

</div>

<!-- RELATED:END -->

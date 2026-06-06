---
title: >-
  [Paper Note] EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases
description: >-
  [ICLR 2026][Medical Imaging][Electronic Medical Records] This paper proposes EMR-AGENT, the first LLM agent-based framework for automated EMR preprocessing. By replacing hand-crafted rules with dynamic SQL interaction…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Electronic Medical Records"
  - "LLM Agent"
  - "Cohort Selection"
  - "Feature Extraction"
  - "Code Mapping"
date: 2026-05-08
content_hash: acbda8f8ee4b4238
---

# EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases

**Conference**: ICLR 2026
**arXiv**: [2510.00549](https://arxiv.org/abs/2510.00549)  
**Code**: [Available](https://github.com/AITRICS/EMR-AGENT)  
**Area**: Medical Imaging
**Keywords**: Electronic Medical Records, LLM Agent, Cohort Selection, Feature Extraction, Code Mapping

## TL;DR

This paper proposes EMR-AGENT, the first LLM agent-based framework for automated EMR preprocessing. By replacing hand-crafted rules with dynamic SQL interaction, it achieves cross-database cohort selection, feature extraction, and code mapping, demonstrating strong performance and generalization on MIMIC-III, eICU, and SICdb.

## Background & Motivation

Clinical prediction models rely on structured data extracted from EMRs, yet this process remains dominated by hard-coded, database-specific pipelines involving three key steps: cohort definition, feature selection, and code mapping. Two core challenges arise:

**Challenge 1: Cross-institutional semantic and structural heterogeneity.** EMR systems vary significantly across hospitals. For example, "heart rate" corresponds to `itemid=211` in MIMIC-III, `HeartRateECG` in SICdb, and column name `heartrate` in eICU. This severely limits cross-database comparability and generalization.

**Challenge 2: Intra-database inconsistency.** The same clinical concept may be measured in multiple ways (e.g., heart rate via sensor, auscultation, or palpation), resulting in multiple code mappings. Ambiguity in cohort selection criteria (e.g., different interpretations of "first ICU admission") leads to divergent patient populations across studies.

Existing solutions (YAIB, ACES, BlendedICU, etc.) either rely on hard-coded rules that lack flexibility or depend on predefined input formats that limit generalization. **Core Problem**: Can AI agents replace manual rule authoring to enable automated EMR preprocessing?

## Method

### Overall Architecture

EMR-AGENT consists of two LLM agents sharing a **Schema Linking & Guideline Generation** frontend module:

- **CFSA (Cohort and Feature Selection Agent)**: Automates patient cohort selection and clinical variable extraction.
- **CMA (Code Mapping Agent)**: Standardizes clinical feature codes across different EMR systems.

**Core Idea**: SQL is treated as a **tool for exploration and decision-making**, not merely as a final query output. Agents complete preprocessing tasks by iteratively observing query results and reasoning over schema and documentation.

### Key Designs

**1. Schema Linking & Guideline Generation**

Unlike conventional approaches that rely solely on schema information, this module augments schema understanding using multiple knowledge sources (database manuals, evaluation memos):

- For CFSA: clarifies the role of each schema component, identifies missing information, and plans SQL observations.
- For CMA: defines the role of each table and column to enable accurate candidate list construction.

**2. Three-Phase Interactive Loop in CFSA**

- **SQL Sufficiency Assessment**: Determines whether the current schema and guidelines are sufficient to generate the target SQL. If not, observation SQL is issued to obtain additional data.
- **Data Sufficiency Check**: Evaluates whether the returned data improves schema understanding.
- **Schema Update**: Integrates newly retrieved data into the schema and guidelines.

This is followed by an **SQL Generation** and **Error Feedback** loop:
- Syntax errors → direct regeneration.
- Schema mismatch (syntactically correct but semantically wrong) → return to schema linking.
- Correct result → extraction complete.

**3. Candidate Matching in CMA**

- **Feature Localization**: Directly searches the schema for target feature column names.
- **Candidate List**: If direct search fails, identifies tables and columns likely to contain the feature from the schema and executes DISTINCT queries to retrieve all candidate combinations.
- **Target–Candidate Matching**: Batch comparison of user-requested features against candidates, computing similarity scores (0–100) with a dual-threshold strategy (80 then 90).

### Loss & Training

EMR-AGENT is an inference-based agent framework with no training involved. It relies on:

- **Problem Decomposition**: Decomposes complex EMR preprocessing tasks into manageable sub-problems.
- **Temperature Scheduling**: CFSA allows up to 10 observations; temperature is 0 for the first 5, then incremented by 0.1 per step to increase exploration.
- **Error Feedback**: Up to 5 retries.
- **LLM Backbone**: Claude-3.5-Sonnet as the primary model.

## Key Experimental Results

### Main Results

**Cohort & Feature Selection (F1 / Accuracy)**

| Method | MIMIC-III F1 | eICU F1 | SICdb F1 |
|--------|:---:|:---:|:---:|
| **EMR-AGENT** | **0.940** | **0.929** | **0.814** |
| ICL(PLUQ) | 0.749 | 0.132 | 0.407 |
| DinSQL | 0.726 | 0.000 | 0.071 |
| REACT | 0.308 | 0.524 | 0.503 |
| ICL(SeqSQL) | 0.040 | 0.000 | 0.040 |

EMR-AGENT substantially outperforms all baselines across databases. Baseline methods nearly completely fail on eICU and SICdb (F1 < 0.53), while EMR-AGENT maintains F1 > 0.81.

**Code Mapping (F1 / Balanced Accuracy)**

| Method | MIMIC-III F1 | eICU F1 | SICdb F1 |
|--------|:---:|:---:|:---:|
| **EMR-AGENT** | **0.516** | **0.648** | **0.536** |
| ICL(PLUQ) | 0.022 | 0.125 | 0.119 |
| REACT | 0.214 | 0.067 | 0.218 |

Code mapping is inherently more challenging, but EMR-AGENT still leads by a large margin (gains of 0.3–0.5 F1).

### Ablation Study

**CFSA Component Ablation**

| Component | MIMIC-III F1 | SICdb F1 |
|-----------|:---:|:---:|
| Full System | 0.940 | 0.814 |
| w/o SQL Observation | 0.916 | 0.795 |
| w/o Error Feedback | 0.688 | 0.617 |
| w/o All DB Interaction | 0.677 | 0.570 |
| w/o SchemaGuideline | 0.827 | 0.792 |

**DB interaction** is the most critical component. Removing Documents + Modules causes CFSA to drop to F1=0 on eICU and CMA to collapse entirely.

**Different LLM Backbones (SICdb)**

| LLM | CFSA F1 | CMA F1 |
|-----|:---:|:---:|
| Claude-3.5-Sonnet | **0.81** | 0.54 |
| Claude-3.7-Sonnet | 0.80 | **0.63** |
| Claude-3.5-haiku | 0.74 | 0.44 |
| Qwen2.5-72B | 0.22 | 0.31 |
| Llama-3.1-70B | 0.18 | 0.14 |

Open-source models (Qwen/Llama) perform substantially worse than the Claude family, indicating that agent capability is strongly tied to the reasoning quality of the backbone LLM.

### Key Findings

1. Dynamic database interaction (SQL observation + error feedback) is the largest performance contributor.
2. External knowledge (database manuals + evaluation memos) is particularly critical for CMA.
3. Generalization remains strong on an unseen database (SICdb, whose creation postdates LLM training cutoffs).
4. Code mapping is inherently difficult due to multiple encodings per feature; F1 of ~0.5–0.65 represents significant progress.

## Highlights & Insights

- **Paradigm Shift**: Transitions EMR preprocessing from manually authored rules to dynamic AI agent interaction.
- **SQL as an Exploration Tool**: Unlike single-pass Text-to-SQL, the agent employs SQL iteratively for observation, verification, and decision-making.
- **Schema Guideline Approach**: Context-aware schema understanding integrating multiple knowledge sources, surpassing conventional schema linking.
- **Accompanying Benchmark PreCISE-EMR**: The first standardized evaluation protocol for EMR preprocessing, developed in collaboration with clinical experts.
- **High Practical Value**: Data preprocessing is a genuine bottleneck in medical ML; automation can substantially improve efficiency.

## Limitations & Future Work

- Code mapping F1 still has room for improvement (0.5–0.65), particularly for disambiguating synonymous clinical concepts.
- Strong dependence on Claude-family LLMs; open-source models exhibit a large performance gap.
- Evaluation is limited to ICU databases (MIMIC-III/eICU/SICdb); outpatient or specialty-specific EMRs are not addressed.
- The 56 evaluated features are restricted to vital signs and laboratory results, excluding medications, diagnosis codes, imaging reports, etc.
- Computational cost (multiple LLM calls + SQL interactions) is not analyzed in detail.
- Reproducing results requires PhysioNet credentials, raising the barrier to access.

## Related Work & Insights

- **vs. YAIB/BlendedICU**: These rely on hard-coded rules requiring manual adaptation for new databases; EMR-AGENT adapts automatically.
- **vs. ACES/Clairvoyance**: These depend on fixed input formats; EMR-AGENT interacts directly with raw databases.
- **vs. Text-to-SQL** (PLUQ/EHRSQL): These assume user familiarity with the schema and perform single-pass querying; EMR-AGENT handles multi-round exploration under schema uncertainty.
- **vs. EHRAgent**: EHRAgent performs isolated chart queries; EMR-AGENT executes a structured preprocessing pipeline.
- **Insight**: Agent-driven data acquisition may become a new foundational infrastructure layer for medical AI.

## Rating

| Dimension | Score |
|-----------|:---:|
| Novelty | ★★★★★ |
| Theoretical Depth | ★★★☆☆ |
| Experimental Thoroughness | ★★★★☆ |
| Value | ★★★★★ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](../../ACL2026/medical_imaging/ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](../../CVPR2026/medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)

</div>

<!-- RELATED:END -->

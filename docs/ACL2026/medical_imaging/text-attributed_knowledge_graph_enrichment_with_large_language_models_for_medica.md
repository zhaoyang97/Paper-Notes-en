---
title: >-
  [Paper Note] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation
description: >-
  [ACL 2026][Medical Imaging][Medical Concept Representation] This paper proposes CoMed, an LLM-empowered graph learning framework that constructs a global medical knowledge graph by combining EHR statistical evidence with…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Medical Concept Representation"
  - "Knowledge Graph"
  - "LLM-GNN Joint Learning"
  - "Electronic Health Records"
  - "Text-Attributed Graph"
date: 2026-05-08
content_hash: 7ad6b4d493a4a20d
---

# Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation

**Conference**: ACL 2026
**arXiv**: [2604.13331](https://arxiv.org/abs/2604.13331)
**Code**: None
**Area**: Medical Imaging / Graph Learning
**Keywords**: Medical Concept Representation, Knowledge Graph, LLM-GNN Joint Learning, Electronic Health Records, Text-Attributed Graph

## TL;DR

This paper proposes CoMed, an LLM-empowered graph learning framework that constructs a global medical knowledge graph by combining EHR statistical evidence with type-constrained LLM inference, enriches it into a text-attributed graph via LLM-generated node descriptions and edge rationales, and jointly trains a LoRA-finetuned LLaMA encoder with a heterogeneous GNN to learn unified medical concept embeddings, achieving significant improvements in diagnosis prediction on MIMIC-III/IV.

## Background & Motivation

**Background**: Learning high-quality medical concept representations (embeddings of diagnosis, medication, and procedure codes) from EHR data is fundamental to clinical prediction. Existing methods primarily leverage hierarchical structures in medical ontologies (e.g., parent-child relationships in ICD) or limited cross-type semantics (e.g., UMLS) to construct knowledge graphs for guiding representation learning.

**Limitations of Prior Work**: (1) Cross-type dependencies (e.g., diagnosis–medication treatment relationships, drug–procedure associations) are largely missing or incomplete in existing ontologies; (2) rich clinical semantics encoded in text are difficult to integrate with KG structure; (3) unconstrained LLM prompting may produce plausible but unsupported edges with inconsistent outputs.

**Key Challenge**: LLMs encode broad biomedical knowledge, yet KG inference for clinical modeling must remain evidence-grounded, type-aware, and globally consistent — requiring a balance between the semantic richness of LLMs and the empirical grounding of EHR data.

**Goal**: To construct a clinically interpretable and empirically grounded heterogeneous KG, and to learn unified medical concept embeddings that integrate textual semantics with graph structure.

**Key Insight**: Statistically significant code pairs are first extracted from EHR data as candidate relations, and LLMs then infer semantic relation types conditioned on type constraints and statistical evidence — a dual-safeguard strategy of "statistical filtering + LLM inference."

**Core Idea**: EHR statistical evidence provides an empirical foundation while LLMs supply semantic interpretation and relation typing — the two are complementary for KG construction, followed by LLM-GNN joint learning to fuse textual and structural information.

## Method

### Overall Architecture

CoMed proceeds in four stages: (1) extracting co-occurrence and temporal transition statistics from EHR data, retaining statistically significant code pairs; (2) using type-constrained LLM prompting to infer directed relation types, confidence scores, and rationales for each code pair; (3) enriching the KG with LLM-generated node descriptions and edge metadata; (4) jointly training a LoRA-finetuned LLaMA-1B encoder and a heterogeneous GNN to learn concept embeddings.

### Key Designs

1. **EHR Statistical Evidence Extraction and Filtering**:

    - Function: Identify empirically supported candidate relations from the data.
    - Mechanism: Three statistics are computed for each code pair — smoothed conditional probability, PMI association, and chi-square independence test p-value — under both intra-visit co-occurrence and inter-visit temporal transition settings. Code pairs with low support, low association, or non-significant p-values ($p > 0.05$) are filtered out.
    - Design Motivation: Pure LLM inference is prone to hallucination; statistical filtering ensures that every candidate edge is grounded in actual observations within the target EHR dataset — relations are not only "clinically plausible" but also "empirically attested in this dataset."

2. **Type-Constrained LLM Relation Inference**:

    - Function: Infer semantic relation types for statistically significant code pairs.
    - Mechanism: A predefined candidate relation pool is specified for each combination of code types (dx-dx, rx-dx, px-dx, etc.), containing labels such as *causes*, *treats*, and *diagnostic_of*. The structured prompt includes code identifiers, frequency information, eight statistical indicators with explanations. The LLM returns a relation label, a directed triple, a confidence score, and a 50–60-word clinical rationale.
    - Design Motivation: Type constraints prevent semantically implausible relations (e.g., a diagnosis "treating" another diagnosis); evidence conditioning enables the LLM to integrate clinical knowledge with statistical signals. Clinical expert auditing of 50 edges yielded a mean rating of 4.84/5, validating high quality.

3. **LLM-GNN Joint Learning (CoMed)**:

    - Function: Fuse textual semantics and graph structure to learn unified concept embeddings.
    - Mechanism: A LoRA-finetuned LLaMA-1B encodes node descriptions into text embeddings, which are projected into the GNN space via type-specific linear projections. A heterogeneous GNN performs relation-aware message passing over the KG to produce final concept embeddings. The model is trained end-to-end with a two-stage LoRA update schedule — an "least-updated-first" strategy in early training ensures broad coverage, while a mixture of low- and high-frequency codes is used in later stages.
    - Design Motivation: GNNs excel at aggregating graph structure but do not interpret long-form text; LLMs encode semantics but do not exploit global relational constraints — joint learning enables mutual complementarity. The two-stage schedule addresses the under-updating of rare codes inherent in mini-batch training.

### Loss & Training

A multi-label cross-entropy loss is used for the next-visit diagnosis prediction task. CoMed serves as a plug-and-play concept encoder integrated into standard EHR models for end-to-end training.

## Key Experimental Results

### Main Results

**Diagnosis Prediction Performance on MIMIC-III**

| Method | AUPRC | F1 | Acc@15 |
|--------|-------|-----|--------|
| Base Transformer | 41.00 | 33.16 | 47.20 |
| GRAM | 41.70 | 34.60 | 48.60 |
| LINKO | 44.91 | 38.20 | 52.30 |
| GraphCare | 43.35 | 35.46 | 52.76 |
| **CoMed** | **47.21** | **42.28** | **54.20** |

### Ablation Study

**Plug-and-Play Analysis (CoMed Integrated into Different Backbones)**

| Backbone | w/o CoMed | w/ CoMed | Gain |
|----------|-----------|----------|------|
| Transformer | 41.00 | 47.21 | +6.21 |
| RETAIN | ~40 | ~46 | +6 |
| GRAM | 41.70 | ~47 | +5 |

### Key Findings

- CoMed improves AUPRC from 41.00 to 47.21 (+6.21) on MIMIC-III, ranking first among all baselines.
- Gains are particularly pronounced for rare diagnosis labels (0–25% frequency) — from 40.60 to 47.67 (+7.07) — as KG relations allow rare concepts to borrow information from associated concepts.
- CoMed consistently improves performance across multiple backbones as a plug-and-play concept encoder.
- Clinical experts rated LLM-inferred edges at 4.84 ± 0.29 / 5, validating the clinical validity of the KG.
- Consistent improvements on MIMIC-IV demonstrate cross-dataset generalizability.

## Highlights & Insights

- The dual-safeguard KG construction strategy of "statistical filtering + LLM inference" ensures both empirical grounding and semantic plausibility.
- The two-stage LoRA update schedule elegantly addresses the training imbalance induced by the long-tailed distribution of medical codes.
- The substantial gains on rare diagnoses carry significant clinical implications — rare diseases are often the hardest to predict and the most in need of attention.

## Limitations & Future Work

- LLM-generated node descriptions and relational rationales may contain subtle hallucinations or biases.
- Evaluation is limited to the diagnosis prediction task; effectiveness on medication recommendation, readmission prediction, and other tasks remains unverified.
- KG construction relies on statistics derived from the target dataset, meaning different hospitals' EHR data may yield different KGs.
- The text encoding capacity of LLaMA-1B is limited; larger LLMs may produce higher-quality embeddings.

## Related Work & Insights

- **vs. GRAM**: GRAM relies solely on the ICD hierarchy, whereas CoMed introduces cross-type relations and textual semantics — AUPRC gain of +5.51.
- **vs. GraphCare**: GraphCare employs an external medical KG without alignment to EHR data; CoMed ensures empirical grounding through statistical filtering.
- **vs. LINKO**: LINKO constructs a KG via link prediction but does not incorporate textual semantics; CoMed's LLM-GNN joint learning provides a more comprehensive integration.

## Rating

- Novelty: ⭐⭐⭐⭐ The KG construction paradigm combining EHR statistics with LLM inference, and the LLM-GNN joint learning framework, are both novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ MIMIC-III/IV × multiple baselines + plug-and-play analysis + clinical expert validation.
- Writing Quality: ⭐⭐⭐⭐ The methodological pipeline is clearly presented, with explicit motivation for each design choice.
- Value: ⭐⭐⭐⭐⭐ The plug-and-play concept encoder offers high practical value for the EHR research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](../../AAAI2026/medical_imaging/unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation
description: >-
  [ACL 2026][Medical NLP][Medical Concept Representation] This paper proposes CoMed, an LLM-powered graph learning framework. It constructs a global medical knowledge graph by combining EHR statistical evidence with type-c…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Medical Concept Representation"
  - "Knowledge Graph"
  - "LLM-GNN Joint Learning"
  - "Electronic Health Records"
  - "Text-Attributed Graph"
date: 2026-05-08
content_hash: baeba2edfe6f2571
---

# Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation

**Conference**: ACL 2026  
**arXiv**: [2604.13331](https://arxiv.org/abs/2604.13331)  
**Code**: None  
**Area**: Medical NLP
**Keywords**: Medical Concept Representation, Knowledge Graph, LLM-GNN Joint Learning, Electronic Health Records, Text-Attributed Graph

## TL;DR

This paper proposes CoMed, an LLM-powered graph learning framework. It constructs a global medical knowledge graph by combining EHR statistical evidence with type-constrained LLM reasoning, enriches it into a text-attributed graph using LLM-generated node descriptions and edge rationales, and finally performs joint training of a LoRA-fine-tuned LLaMA encoder and a heterogeneous GNN to learn unified medical concept embeddings. This significantly improves diagnosis prediction performance on MIMIC-III/IV.

## Background & Motivation

**Background**: Learning high-quality medical concept representations (embeddings of diagnosis/medication/procedure codes) is fundamental for clinical prediction in EHR mining. Existing methods primarily utilize the hierarchical structure of medical ontologies (e.g., ICD parent-child relationships) or limited cross-type semantics (e.g., UMLS) to construct knowledge graphs for guiding representation learning.

**Limitations of Prior Work**: (1) Cross-type dependencies (e.g., diagnosis-medication treatment, medication-procedure association) are largely missing or incomplete in existing ontologies; (2) Rich clinical semantics usually exist in text form but are difficult to integrate with KG structures; (3) Unconstrained LLM prompting may generate plausible but unsupported edges with inconsistent outputs.

**Key Challenge**: LLMs encode broad biomedical knowledge, but KG inference for clinical modeling must remain evidence-based, type-aware, and globally consistent—requiring a balance between the semantic richness of LLMs and the empirical support of EHRs.

**Goal**: To build a clinically interpretable and empirically supported heterogeneous KG, and learn unified medical concept embeddings that fuse textual semantics and graph structures.

**Key Insight**: Extract statistically significant code pairs from EHRs as candidate relations first, then use an LLM to infer semantic relation types under type constraints and evidence conditions—a "statistical filtering + LLM inference" double-insurance strategy.

**Core Idea**: EHR statistical evidence provides the empirical foundation, while LLMs provide semantic explanations and relation types. The two complement each other to build the KG, followed by LLM-GNN joint learning to fuse textual and structural information.

## Method

### Overall Architecture

CoMed consists of four steps: (1) Extract co-occurrence and temporal transition statistics from EHRs, retaining statistically significant code pairs; (2) Use type-constrained LLM prompting to infer directed relation types, confidence levels, and rationales for each pair; (3) Use an LLM to generate node descriptions and edge metadata to enrich the KG; (4) Jointly train a LoRA-fine-tuned LLaMA-1B encoder and a heterogeneous GNN to learn concept embeddings.

### Key Designs

1.  **EHR Statistical Evidence Extraction and Filtering**:
    *   **Function**: Discover empirically supported candidate relations from data.
    *   **Mechanism**: For each code pair, three statistics are calculated—smoothed conditional probability, PMI association, and Chi-square independence test p-value. Both in-hospital co-occurrence and cross-visit temporal transition settings are considered. Code pairs with low support, low association, or non-significance ($p > 0.05$) are filtered out.
    *   **Design Motivation**: Pure LLM inference is prone to hallucination. Statistical filtering ensures each candidate edge has actual observational support in the target EHR dataset—the relationship is not only "clinically reasonable" but "actually exists in this dataset."

2.  **Type-Constrained LLM Relation Inference**:
    *   **Function**: Infer semantic relation types for statistically significant code pairs.
    *   **Mechanism**: A candidate relation pool (e.g., causes, treats, diagnostic_of) is predefined for each code type combination (dx-dx, rx-dx, px-dx, etc.). The structured prompt includes code identifiers, frequencies, 8 statistical metrics, and metric descriptions. The LLM returns relation labels, directed triples, confidence scores, and 50-60 word clinical reasoning.
    *   **Design Motivation**: Type constraints prevent the generation of semantically unreasonable relations (e.g., a diagnosis "treating" another diagnosis); evidence conditions allow the LLM to synthesize clinical knowledge with statistical signals. Clinical experts audited 50 edges with an average score of 4.84/5, verifying high quality.

3.  **LLM-GNN Joint Learning (CoMed)**:
    *   **Function**: Fuse textual semantics and graph structure to learn unified concept embeddings.
    *   **Mechanism**: A LoRA-fine-tuned LLaMA-1B encodes node descriptions into text embeddings, which are projected into GNN space via type-specific linear layers. A heterogeneous GNN performs relation-aware message passing on the KG to output final concept embeddings. End-to-end joint training uses a two-stage LoRA update schedule—"least-update-first" in the early stage to ensure coverage, and a mix of low-frequency and high-frequency codes in the later stage.
    *   **Design Motivation**: GNNs excel at aggregating graph structures but do not interpret long text; LLMs encode semantics but do not utilize global relational constraints—joint learning allow them to complement each other. The two-stage schedule solves the issue of insufficient updates for rare codes in mini-batch training.

### Loss & Training

A multi-label cross-entropy loss is used to train the next-visit diagnosis prediction task. CoMed is integrated as a plug-and-play concept encoder into standard EHR models for end-to-end training.

## Key Experimental Results

### Main Results

**Diagnosis Prediction Performance Comparison on MIMIC-III**

| Method | AUPRC | F1 | Acc@15 |
|------|-------|-----|--------|
| Base Transformer | 41.00 | 33.16 | 47.20 |
| GRAM | 41.70 | 34.60 | 48.60 |
| LINKO | 44.91 | 38.20 | 52.30 |
| GraphCare | 43.35 | 35.46 | 52.76 |
| **CoMed** | **47.21** | **42.28** | **54.20** |

### Ablation Study

**Plug-and-play Analysis (CoMed integrated into different backbones)**

| Backbone | Without CoMed | With CoMed | Gain |
|----------|---------|---------|------|
| Transformer | 41.00 | 47.21 | +6.21 |
| RETAIN | ~40 | ~46 | +6 |
| GRAM | 41.70 | ~47 | +5 |

### Key Findings

*   CoMed improves AUPRC on MIMIC-III from 41.00 to 47.21 (+6.21), ranking first among all baselines.
*   The improvement is particularly significant for rare diagnosis labels (0-25% frequency)—from 40.60 to 47.67 (+7.07), as KG relations help rare concepts borrow information from associated concepts.
*   CoMed consistently improves performance as a plug-and-play concept encoder across multiple backbones.
*   Clinical experts gave LLM-inferred edges a rating of 4.84±0.29/5, verifying the clinical validity of the KG.
*   Consistent improvements are observed on MIMIC-IV, proving cross-dataset generalization.

## Highlights & Insights

*   The "statistical filtering + LLM inference" double-insurance KG construction strategy ensures both empirical support and semantic rationality of relations.
*   The two-stage LoRA update schedule cleverly addresses the training imbalance caused by the long-tail distribution of medical codes.
*   The substantial improvement for rare diagnoses has significant clinical importance—rare diseases are often the most difficult to predict and require the most attention.

## Limitations & Future Work

*   Node descriptions and relation reasoning generated by LLMs may contain subtle hallucinations or biases.
*   The evaluation is limited to diagnosis prediction tasks; performance on medication recommendation or readmission prediction has not been verified.
*   KG construction depends on statistics from the target dataset; EHRs from different hospitals may produce different KGs.
*   The text encoding capability of LLaMA-1B is limited; larger LLMs might yield better embeddings.

## Related Work & Insights

*   **vs GRAM**: GRAM only uses ICD hierarchical structures, while CoMed introduces cross-type relations and textual semantics—AUPRC +5.51.
*   **vs GraphCare**: The latter uses external medical KGs but does not align them with EHR data; CoMed ensures empirical support through statistical filtering.
*   **vs LINKO**: The latter uses link prediction to build KGs but does not fuse textual semantics; CoMed's LLM-GNN joint learning is more comprehensive.

## Rating

*   Novelty: ⭐⭐⭐⭐ The KG construction idea of EHR statistics + LLM inference and the LLM-GNN joint learning framework are novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ MIMIC-III/IV × multiple baselines + plug-and-play analysis + clinical expert verification.
*   Writing Quality: ⭐⭐⭐⭐ The methodology flow is clear, with explicit motivations for each design step.
*   Value: ⭐⭐⭐⭐⭐ The plug-and-play concept encoder is of high value to the EHR research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)
- [\[ACL 2026\] MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](medfact_benchmarking_the_fact-checking_capabilities_of_large_language_models_on_.md)

</div>

<!-- RELATED:END -->

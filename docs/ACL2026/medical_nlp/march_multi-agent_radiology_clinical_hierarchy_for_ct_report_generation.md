---
title: >-
  [Paper Note] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation
description: >-
  [ACL 2026][Medical NLP][Multi-Agent] This paper proposes MARCH, a multi-agent framework that simulates the hierarchical collaboration process of resident-fellow-attending radiologists. By generating CT reports through th…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Multi-Agent"
  - "Radiology Report Generation"
  - "Consensus-Driven"
  - "Retrieval-Augmented"
  - "3D CT"
date: 2026-05-08
content_hash: d8b6e98c4d13a5a8
---

# MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation

**Conference**: ACL 2026  
**arXiv**: [2604.16175](https://arxiv.org/abs/2604.16175)  
**Code**: None  
**Area**: Medical Imaging / Report Generation  
**Keywords**: Multi-Agent, Radiology Report Generation, Consensus-Driven, Retrieval-Augmented, 3D CT

## TL;DR

This paper proposes MARCH, a multi-agent framework that simulates the hierarchical collaboration process of resident-fellow-attending radiologists. By generating CT reports through three stages (initial drafting, retrieval-augmented revision, and consensus-driven finalization), it achieves a CE-F1 of 0.399 on the RadGenome-ChestCT dataset, a 57.7% improvement over the best baseline Reg2RG (0.253).

## Background & Motivation

**Background**: Automated radiology report generation is a critical direction for medical AI. Existing vision-language models (VLMs) have achieved progress in 2D chest X-ray reports, but report generation for 3D volumetric data (such as chest CT) is still in its early stages.

**Limitations of Prior Work**: (1) End-to-end "black box" models lack iterative verification and cross-check mechanisms found in clinical workflows, making them prone to clinical hallucinations; (2) Abnormalities in 3D CT data are sparse, making it difficult for a single model to reliably detect all pathologies; (3) Cognitive biases inherent in a single-reader mode cannot be corrected.

**Key Challenge**: In clinical practice, radiology departments reduce misdiagnosis rates through a hierarchical review process involving residents, fellows, and attending physicians. However, existing automated systems are single-agent and lack this multi-tier verification mechanism.

**Goal**: Design a multi-agent framework that simulates the clinical hierarchy of radiology to achieve interpretable and verifiable CT report generation.

**Key Insight**: Drawing inspiration from the "readout session" system in radiology—initial reading by a resident, review by a fellow, and final authorization by an attending—different responsibilities are assigned to different AI agents.

**Core Idea**: Replace a single end-to-end model with a multi-agent hierarchical structure, significantly enhancing clinical accuracy through retrieval augmentation and multi-round consensus discussions.

## Method

### Overall Architecture

MARCH consists of three stages: (1) A Resident Agent generates an initial report draft from 3D CT scans; (2) A Retrieval Agent retrieves relevant cases from a clinical database, which a Fellow Agent uses to revise the report; (3) An Attending Agent presides over multi-round consensus discussions where multiple Fellow Agents iteratively exchange positions until a clinical consensus is reached. The input is chest CT volumetric data, and the output is the final radiology report.

### Key Designs

1.  **Resident Agent + Multi-Region Segmentation**:

    - **Function**: Extracts features from 3D CT and generates an initial draft.
    - **Mechanism**: A SAT (Segment Anything with Text) model segments the CT into 10 anatomical sub-regions (e.g., bones, breasts). A frozen dual-stream ViT3D (from RadFM pretraining) extracts spatial features, and finally, a LoRA-finetuned LLaMA-2-Chat-7B generates the text report $T = A_{res}(I; \theta_{res})$.
    - **Design Motivation**: Abnormal findings in 3D volumetric data are often confined to specific anatomical regions and are highly sparse; global encoding easily overlooks them. Multi-region segmentation forces the model to focus on local anatomy and pathological entities, mitigating the sparsity issue in abnormality detection.

2.  **Retrieval-Augmented Revision**:

    - **Function**: Provides evidence-based grounds for report revision by retrieving similar cases.
    - **Mechanism**: Three retrieval paradigms are designed: (i) Image-to-image/Image-to-text retrieval: Using a 3D vision encoder to retrieve visually similar CTs and corresponding reports; (ii) Logits retrieval: Using a classification head to predict logits vectors for 18 clinical abnormalities and retrieving reports with similar diagnostic spectrums. The top-3 results from each retrieval method are concatenated as structured evidence $R = A_{ret}(I, D)$, which the Fellow Agent $A_{fel}$ integrates to revise the initial draft: $T' = A_{fel}(T, R)$.
    - **Design Motivation**: Generative models alone may omit details or hallucinate. Retrieval augmentation provides a "second opinion" and an evidence-based foundation, analogous to the clinical process of consulting literature and reference cases.

3.  **Consensus-Driven Finalization**:

    - **Function**: Resolves diagnostic disagreements through multi-round position exchanges.
    - **Mechanism**: The Attending Agent $A_{att}$ first aggregates revised reports from multiple Fellow Agents to generate an initial consensus $T^{(0)}$. In subsequent rounds, each Fellow Agent $A_{fel,i}$ reviews the current consensus and provides a position $S_i^{(t)}$ (agree/correct/supplement). The Attending Agent integrates all positions to update the report $T^{(t+1)} = A_{att}(T^{(t)}, \{S_i^{(t)}\})$. Iteration continues until a stable consensus is reached or the maximum number of rounds is met.
    - **Design Motivation**: Simulating the real-world radiology readout session, where disagreements among multiple physicians are resolved through discussion rather than simple voting. This "devil's advocate" mechanism has been proven in clinical settings to significantly reduce misdiagnosis.

### Loss & Training

The Resident Agent is optimized using AdamW (lr=1e-5) for 10 epochs. The ViT3D backbone is frozen, and LLaMA-2-Chat-7B is finetuned via LoRA. Fellow and Attending agents utilize GPT-4.1/GPT-4o as LLM backbones (temperature=0).

## Key Experimental Results

### Main Results

| Method | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CE-Precision | CE-Recall | CE-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R2GenPT | 0.433 | 0.242 | 0.399 | 0.323 | 0.340 | 0.066 | 0.110 |
| MedVInT | 0.443 | 0.246 | 0.404 | 0.326 | 0.377 | 0.148 | 0.212 |
| M3D | 0.436 | 0.245 | 0.400 | 0.326 | 0.407 | 0.090 | 0.148 |
| RadFM | 0.442 | 0.237 | 0.399 | 0.315 | 0.382 | 0.131 | 0.195 |
| Reg2RG | 0.473 | 0.249 | 0.441 | 0.367 | 0.423 | 0.181 | 0.253 |
| **MARCH** | **0.482** | **0.257** | **0.456** | **0.383** | **0.495** | **0.335** | **0.399** |

### Ablation Study

| Configuration | BLEU-1 | BLEU-4 | METEOR | CE-F1 |
| :--- | :--- | :--- | :--- | :--- |
| Resident-only | 0.469 | 0.246 | 0.435 | 0.219 |
| SR-SA (Single-round, Single-agent) | 0.476 | 0.250 | 0.447 | 0.332 |
| SR-MA (Single-round, Multi-agent) | 0.475 | 0.251 | 0.454 | 0.352 |
| MR-MA (Multi-round, Multi-agent) | 0.479 | 0.255 | 0.456 | 0.362 |
| **MARCH (Full)** | **0.482** | **0.257** | **0.456** | **0.399** |

### Key Findings

- CE-F1 increased from 0.219 for Resident-only to 0.399 for full MARCH, an 82% Gain, primarily stemming from retrieval augmentation (+0.113) and the consensus mechanism (+0.037).
- Retrieval augmentation contributes most to clinical efficacy (SR-SA vs. Resident-only: CE-F1 +0.113), indicating that evidence-based revision is key to reducing hallucinations.
- Performance variance across different LLM backbones (GPT-4.1-mini/GPT-4.1/GPT-4o/GPT-5) is minimal (CE-F1 0.391-0.399), suggesting the framework design is more important than the specific LLM capability.
- MARCH shows particularly significant improvements in detecting low-frequency abnormalities (e.g., hiatal hernia, pericardial effusion).

## Highlights & Insights

- Directly mapping the hierarchical collaboration workflow of a radiology department to a multi-agent architecture is an elegant design—it does not assign roles randomly but corresponds to misdiagnosis prevention mechanisms already validated in clinical practice.
- The three complementary retrieval paradigms (visual, textual, logits) cover different types of similarity; this multimodal retrieval combination is transferable to other medical AI tasks requiring evidence.
- The consensus mechanism uses "positions" (agree/correct/supplement) rather than simple voting, preserving the information content of disagreements.

## Limitations & Future Work

- Relies on GPT-4 series as reasoning backbones, which is high-cost and cannot be deployed locally in hospitals; the feasibility of open-source LLMs has not been verified.
- Lacks a long-term memory mechanism, missing the ability to utilize historical patient imaging comparisons or learn from past diagnostic errors.
- Evaluated only on RadGenome-ChestCT; generalization to other anatomical regions (e.g., brain, abdomen) has not been verified.
- The number of consensus rounds requires a preset upper limit; there is no adaptive mechanism for determining the optimal number of rounds.

## Related Work & Insights

- **vs. Reg2RG**: Reg2RG uses region-guided retrieval augmentation but remains single-agent; MARCH adds multi-agent consensus, boosting CE-F1 from 0.253 to 0.399.
- **vs. RadFM**: RadFM is a general 3D medical foundation model using end-to-end generation; it lacks verification and error-correction mechanisms.
- **vs. MedAgent**: General medical multi-agent systems are primarily used for diagnosis and recommendation; MARCH is the first multi-agent framework targeted specifically at 3D report generation.

## Rating

- Novelty: ⭐⭐⭐⭐ Mapping clinical hierarchy to multi-agent structures is natural and meaningful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation, including LLM backbone comparisons and per-abnormality analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description and sufficient clinical context.
- Value: ⭐⭐⭐⭐ Provides an interpretable collaborative paradigm for high-risk medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[ACL 2026\] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)
- [\[ACL 2026\] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)
- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)

</div>

<!-- RELATED:END -->

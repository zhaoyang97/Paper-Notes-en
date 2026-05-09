---
title: >-
  [Paper Note] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation
description: >-
  [ACL 2026][Medical Imaging][Multi-agent] This paper proposes MARCH, a multi-agent framework that simulates the resident–fellow–attending hierarchical collaboration process in radiology. Through three stages—initial report drafting, retrieval-augmented revision, and consensus-driven finalization—MARCH generates CT reports achieving a CE-F1 of 0.399 on the RadGenome-ChestCT dataset, representing a 57.7% improvement over the best baseline Reg2RG (0.253).
tags:
  - ACL 2026
  - Medical Imaging
  - Multi-agent
  - Radiology Report Generation
  - Consensus-Driven
  - Retrieval-Augmented
  - 3D CT
date: 2026-05-08
content_hash: b884553913d3ce6e
---

# MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation

**Conference**: ACL 2026
**arXiv**: [2604.16175](https://arxiv.org/abs/2604.16175)
**Code**: N/A
**Area**: Medical Imaging / Report Generation
**Keywords**: Multi-agent, Radiology Report Generation, Consensus-Driven, Retrieval-Augmented, 3D CT

## TL;DR

This paper proposes MARCH, a multi-agent framework that simulates the resident–fellow–attending hierarchical collaboration process in radiology. Through three stages—initial report drafting, retrieval-augmented revision, and consensus-driven finalization—MARCH generates CT reports achieving a CE-F1 of 0.399 on the RadGenome-ChestCT dataset, representing a 57.7% improvement over the best baseline Reg2RG (0.253).

## Background & Motivation

**Background**: Automated radiology report generation is an important direction in medical AI. Existing vision-language models (VLMs) have made progress on 2D chest X-ray reporting, but report generation for 3D volumetric data (e.g., chest CT) remains in its early stages.

**Limitations of Prior Work**: (1) End-to-end "black-box" models lack the iterative verification and cross-checking mechanisms present in clinical workflows, making them prone to clinical hallucinations; (2) abnormal findings in 3D CT data are sparse, making it difficult for a single model to reliably detect all pathologies; (3) the cognitive biases inherent in the single-reader paradigm cannot be corrected.

**Key Challenge**: In clinical practice, radiology departments reduce misdiagnosis rates through a hierarchical review process involving residents, fellows, and attendings. However, existing automated systems are single-agent and lack this multi-layer verification mechanism.

**Goal**: To design a multi-agent framework that simulates the hierarchical structure of clinical radiology practice, enabling interpretable and verifiable CT report generation.

**Key Insight**: Drawing on the radiology readout session model—resident initial read, fellow review, attending final approval—different responsibilities are assigned to distinct AI agents.

**Core Idea**: Replace a single end-to-end model with a multi-agent hierarchical structure, significantly improving clinical accuracy through retrieval augmentation and multi-round consensus discussion.

## Method

### Overall Architecture

MARCH consists of three stages: (1) a resident agent generates an initial report draft from the 3D CT scan; (2) a retrieval agent retrieves relevant cases from a clinical database, and a fellow agent revises the report accordingly; (3) an attending agent facilitates multi-round consensus discussions in which multiple fellow agents iteratively exchange positions until clinical consensus is reached. The input is chest CT volumetric data, and the output is the final radiology report.

### Key Designs

1. **Resident Agent + Multi-Region Segmentation**:

    - Function: Extract features from 3D CT and generate an initial report draft.
    - Mechanism: The SAT (Segment Anything with Text) model is used to segment the CT into 10 anatomical sub-regions (e.g., skeleton, breast). A frozen dual-stream ViT3D (pre-trained from RadFM) then extracts spatial features, and a LoRA fine-tuned LLaMA-2-Chat-7B generates the text report $T = A_{res}(I; \theta_{res})$.
    - Design Motivation: Abnormal findings in 3D volumetric data are often confined to specific anatomical regions and are highly sparse; global encoding tends to miss them. Multi-region segmentation forces the model to attend to local anatomy and pathological entities, alleviating the sparsity problem in anomaly detection.

2. **Retrieval-Augmented Revision**:

    - Function: Provide evidence-based support for report revision by retrieving similar cases.
    - Mechanism: Three retrieval paradigms are designed—(i) image-to-image / image-to-text retrieval: a 3D visual encoder retrieves visually similar CTs and their corresponding reports; (ii) logits retrieval: a classification head predicts logit vectors for 18 clinical abnormalities and retrieves reports with similar diagnostic profiles. The top-3 results from each paradigm are concatenated into structured evidence $R = A_{ret}(I, D)$, which the fellow agent $A_{fel}$ integrates to revise the initial draft $T' = A_{fel}(T, R)$.
    - Design Motivation: Generative models alone may hallucinate or omit findings; retrieval augmentation provides a "second opinion" and an evidence base, analogous to consulting literature and reference cases in clinical practice.

3. **Consensus-Driven Finalization**:

    - Function: Resolve diagnostic disagreements through multi-round position exchange.
    - Mechanism: The attending agent $A_{att}$ first aggregates the revised reports from multiple fellow agents to produce an initial consensus $T^{(0)}$. In subsequent rounds, each fellow agent $A_{fel,i}$ reviews the current consensus and submits a position $S_i^{(t)}$ (agree / correct / supplement), and the attending agent integrates all positions to update the report $T^{(t+1)} = A_{att}(T^{(t)}, \{S_i^{(t)}\})$. Iteration continues until a stable consensus is reached or the maximum number of rounds is exceeded.
    - Design Motivation: This simulates the real-world radiology readout session, where disagreements among multiple physicians are resolved through discussion rather than simple voting. This "devil's advocate" mechanism has been shown clinically to substantially reduce misdiagnosis rates.

### Loss & Training

The resident agent is trained with the AdamW optimizer (lr=1e-5) for 10 epochs. The ViT3D backbone is frozen, and LLaMA-2-Chat-7B is fine-tuned via LoRA. The fellow and attending agents use GPT-4.1/GPT-4o as the LLM backbone (temperature=0).

## Key Experimental Results

### Main Results

| Method | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CE-Precision | CE-Recall | CE-F1 |
|--------|--------|--------|--------|---------|-------------|-----------|-------|
| R2GenPT | 0.433 | 0.242 | 0.399 | 0.323 | 0.340 | 0.066 | 0.110 |
| MedVInT | 0.443 | 0.246 | 0.404 | 0.326 | 0.377 | 0.148 | 0.212 |
| M3D | 0.436 | 0.245 | 0.400 | 0.326 | 0.407 | 0.090 | 0.148 |
| RadFM | 0.442 | 0.237 | 0.399 | 0.315 | 0.382 | 0.131 | 0.195 |
| Reg2RG | 0.473 | 0.249 | 0.441 | 0.367 | 0.423 | 0.181 | 0.253 |
| **MARCH** | **0.482** | **0.257** | **0.456** | **0.383** | **0.495** | **0.335** | **0.399** |

### Ablation Study

| Configuration | BLEU-1 | BLEU-4 | METEOR | CE-F1 |
|---------------|--------|--------|--------|-------|
| Resident-only | 0.469 | 0.246 | 0.435 | 0.219 |
| SR-SA (single-round, single-agent) | 0.476 | 0.250 | 0.447 | 0.332 |
| SR-MA (single-round, multi-agent) | 0.475 | 0.251 | 0.454 | 0.352 |
| MR-MA (multi-round, multi-agent) | 0.479 | 0.255 | 0.456 | 0.362 |
| **MARCH (full)** | **0.482** | **0.257** | **0.456** | **0.399** |

### Key Findings

- CE-F1 improves from 0.219 (Resident-only) to 0.399 (full MARCH), an 82% gain, driven primarily by retrieval augmentation (+0.113) and the consensus mechanism (+0.037).
- Retrieval augmentation contributes the largest gain in clinical efficacy (SR-SA vs. Resident-only: CE-F1 +0.113), indicating that evidence-based revision is the key to reducing hallucinations.
- Performance differences across LLM backbones (GPT-4.1-mini / GPT-4.1 / GPT-4o / GPT-5) are minimal (CE-F1 0.391–0.399), suggesting that framework design matters more than raw LLM capability.
- MARCH yields particularly notable improvements in detecting low-frequency abnormalities such as hiatal hernia and pericardial effusion.

## Highlights & Insights

- Directly mapping the radiology hierarchical collaboration process onto a multi-agent architecture is an elegant design choice—the role assignments are not arbitrary but correspond to clinically validated mechanisms for preventing misdiagnosis.
- Three complementary retrieval paradigms (visual, textual, logits) cover different notions of similarity; this multimodal retrieval combination is transferable to other evidence-requiring medical AI tasks.
- The consensus mechanism uses typed positions (agree / correct / supplement) rather than simple voting, preserving the information content of disagreements.

## Limitations & Future Work

- Reliance on the GPT-4 series as the reasoning backbone incurs high costs and precludes in-hospital deployment; the feasibility of open-source LLMs has not been validated.
- The framework lacks a long-term memory mechanism and cannot leverage longitudinal imaging comparisons or learn from past diagnostic errors.
- Evaluation is limited to RadGenome-ChestCT; generalizability to other anatomical regions (e.g., brain, abdomen) has not been verified.
- The number of consensus rounds requires a pre-set upper bound, and an adaptive mechanism for determining the optimal number of rounds is absent.

## Related Work & Insights

- **vs. Reg2RG**: Reg2RG employs region-guided retrieval augmentation but remains a single-agent system. MARCH adds multi-agent consensus on top of this, improving CE-F1 from 0.253 to 0.399.
- **vs. RadFM**: RadFM is a general-purpose 3D medical foundation model that performs end-to-end single-model generation and lacks any verification or error-correction mechanism.
- **vs. MedAgent**: General medical multi-agent systems are primarily designed for diagnosis and recommendation. MARCH is the first multi-agent framework specifically targeting 3D report generation.

## Rating

- Novelty: ⭐⭐⭐⭐ The mapping from clinical hierarchical structure to multi-agent architecture is natural and meaningful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are comprehensive, including LLM backbone comparisons and per-abnormality analysis.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly described, and the clinical background is well motivated.
- Value: ⭐⭐⭐⭐ Provides an interpretable collaborative paradigm for high-stakes medical AI.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](../../CVPR2026/medical_imaging/orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[ICLR 2026\] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](../../ICLR2026/medical_imaging/resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)

<!-- RELATED:END -->

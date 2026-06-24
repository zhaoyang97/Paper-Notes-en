---
title: >-
  [Paper Note] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection
description: >-
  [ACL 2025][Medical LLM][radiology report generation] The Radar framework is proposed to systematically fuse internal and external knowledge sources for more accurate radiology report generation, by distinguishing between trusted internal knowledge already mastered by LLM and external knowledge that needs to be supplemented.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "radiology report generation"
  - "knowledge injection"
  - "supplementary knowledge"
  - "LLM"
  - "chest X-ray"
date: 2026-05-08
content_hash: 0efcd7870c46ef51
---

# Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection

**Conference**: ACL 2025  
**arXiv**: [2505.14318](https://arxiv.org/abs/2505.14318)  
**Code**: [https://github.com/wjhou/Radar](https://github.com/wjhou/Radar)  
**Area**: Medical NLP  
**Keywords**: radiology report generation, knowledge injection, supplementary knowledge, LLM, chest X-ray  

## TL;DR

The Radar framework is proposed to systematically fuse internal and external knowledge sources for more accurate radiology report generation, by distinguishing between trusted internal knowledge already mastered by LLM and external knowledge that needs to be supplemented.

## Background & Motivation

**Background:** Large Language Models (LLMs) have demonstrated exceptional text generation capabilities in radiology report generation tasks. Many works attempt to enhance model performance by retrieving domain-specific knowledge, but these approaches often ignore the knowledge already encoded within the LLMs.

**Limitations of Prior Work:** (1) Existing knowledge enhancement methods often retrieve redundant information that the LLMs already possess; (2) The knowledge learned within the LLMs is not always reliable and often hallucinates (e.g., misidentifying diseases); (3) There is a lack of an effective mechanism to distinguish between the trustworthy and untrustworthy knowledge of the model.

**Design Motivation:** Using the example in Figure 1, the LLM correctly identifies Cardiomegaly (no extra knowledge needed), and the generated Pleural Effusion is consistent with the expert model (trustworthy), but there is uncertainty regarding Edema (requiring supplementary knowledge). Thus, there is a need to balance the utilization of the LLM's internal knowledge and external retrieved knowledge.

## Method

### Overall Architecture

Radar consists of two stages: **Stage I: Preliminary Findings Generation** and **Stage II: Supplementary Findings Augmentation**.

### Key Designs

1. **Internal Knowledge Trustworthiness Assessment (Stage I):** The MLLM first generates an initial report, while an independent expert classification model (image encoder + text encoder + MLP) performs observation classification on the image. The intersection $O_\checkmark = O_I \cap O_R$ between the observation results $O_R$ of the initial report and $O_I$ of the expert model is taken as high-confidence internal knowledge (Preliminary Findings).

2. **Supplementary Knowledge Retrieval and Extraction (Stage II):** Utilizing the 14 observation probability distributions from the expert model, sample similarity is calculated via KL divergence to retrieve Top-K similar reports. The key lies in extracting only supplementary knowledge: filtering out observations overlapping with Preliminary Findings, keeping only sentences corresponding to $O_\delta = \mathcal{O} - O_\checkmark$.

3. **Observation Identification for Enhanced Generation:** Integrating PF and SF into the clinical context, the model is trained to output observation labels before generating the report text, helping the model summarize high-level info before generation.

### Loss & Training

- **Expert Model Training:** Bi-classification cross-entropy loss with log-scale re-weighting is used to handle class imbalance: $\alpha_i = \log(1 + |\mathcal{D}_{train}| / w_i)$
- **Report Generation Model:** Standard negative log-likelihood loss: $\mathcal{L} = -\sum_{t=1}^{T} \log p(y_t)$

## Experiments

### Main Results (MIMIC-CXR Dataset)

| Model | B-1 | B-4 | R-L | RG-F1 | 14Ma-F1 | 5Mi-F1 |
|------|-----|-----|-----|-------|---------|--------|
| R2GenGPT | 0.411 | 0.134 | 0.297 | - | 0.389 | - |
| LLaVA-Med | 0.354 | 0.149 | 0.276 | 0.191 | 0.269 | 0.439 |
| Med-PaLM | 0.323 | 0.115 | 0.275 | 0.267 | 0.398 | 0.579 |
| MAIRA-2 | 0.465 | 0.234 | 0.384 | **0.346** | 0.416 | 0.591 |
| Libra | **0.513** | 0.245 | 0.367 | 0.329 | 0.404 | 0.601 |
| **Radar (Ours)** | 0.509 | **0.262** | **0.397** | **0.346** | **0.460** | **0.627** |

Radar achieves SOTA or tied-SOTA performance across B-4, ROUGE-L, 14-class Macro-F1, and 5-class Micro-F1.

### Ablation Study

| Ablation Variant | Description | Effect |
|----------|------|------|
| Remove Preliminary Findings | Failing to distinguish between trusted/untrusted internal knowledge | Significant drop in clinical metrics |
| Remove Supplementary Findings | Not using externally retrieved knowledge | Leads to incomplete observation coverage |
| Remove Observation Identification | Not predicting observation labels | Drop in generation quality |
| Use all retrieved knowledge (no filtering) | Introduces redundant info | Performance is inferior to that after filtering |

### Key Findings

- Radar outperforms SOTA on three benchmark datasets (MIMIC-CXR, CheXpert-Plus, and IU X-ray).
- Supplementary knowledge filtering (retaining only non-overlapping observations) is more effective than using all retrieved knowledge, validating the necessity of redundancy removal.
- Introducing clinical context (such as Indication) to the expert model yields better classification performance than using images alone.

## Highlights & Insights

- Innovatively distinguishes between trusted internal knowledge of LLMs and external knowledge that needs supplementation, preventing redundant retrieval.
- Ingeniously identifies high-confidence knowledge using the intersection of the expert model and LLM outputs.
- The Observation Identification mechanism enables the model to "think" before "writing reports", improving generation quality.
- The method is highly generalizable and can be extended to other knowledge-enhanced medical NLP tasks.

## Limitations & Future Work

- Relies on the CheXpert 14-class labeling system, which cannot cover all radiological findings.
- The classification accuracy of the expert model directly impacts the quality of knowledge filtering.
- Verified only on chest X-ray data without extension to other modalities like CT or MRI.
- The retrieved knowledge base originates from the training set, potentially leading to distribution shift issues.
- Two-stage inference introduces extra computational overhead (requiring preliminary report generation + retrieval + re-generation).

## Related Work & Insights

- **Radiology Report Generation:** Pioneering works by Chen et al. 2020/2021; LLM-based methods such as R2GenGPT, LLaVA-Med, and Med-PaLM.
- **Knowledge-Enhanced Generation:** Retrieval-augmented methods like Yang et al. 2021; domain knowledge injection like Li et al. 2023.
- **Medical Multimodal Models:** Task-specific medical MLLMs like MAIRA-1/2, CheXagent, and LLaVA-Rad.
- **Hallucination Mitigation:** Research on LLM hallucinations such as Huang et al. 2025.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 7 |
| Technical Depth | 7 |
| Experimental Thoroughness | 8 |
| Writing Quality | 7 |
| Value | 8 |
| Overall Score | 7.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)
- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)
- [\[ACL 2025\] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)
- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_nlp/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models
description: >-
  [ACL 2026][Medical Imaging][Electronic Health Records] This paper proposes RePrompT, a temporally aware LLM framework that consistently outperforms both EHR and LLM baselines on readmission and mortality prediction tasks…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Electronic Health Records"
  - "Prompt Tuning"
  - "Recurrent State Propagation"
  - "Structured Encoder"
  - "Clinical Prediction"
date: 2026-05-08
content_hash: 3083c53648b3548a
---

# RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.17725](https://arxiv.org/abs/2604.17725)  
**Code**: [https://github.com/KU-AI4H/RePrompT](https://github.com/KU-AI4H/RePrompT)  
**Area**: Medical Imaging
**Keywords**: Electronic Health Records, Prompt Tuning, Recurrent State Propagation, Structured Encoder, Clinical Prediction

## TL;DR
This paper proposes RePrompT, a temporally aware LLM framework that consistently outperforms both EHR and LLM baselines on readmission and mortality prediction tasks on MIMIC-III/IV through two complementary mechanisms: recurrent prompt tuning (propagating the hidden state of the previous visit as a soft prompt for the current visit) and struct-encoded prompt tuning (injecting embeddings from a population-level EHR encoder).

## Background & Motivation

**Background**: Electronic Health Records (EHRs) contain longitudinal patient information spanning multiple visits, including diagnoses, medications, and procedures, making them a critical data source for clinical decision support. Large language models have shown promise in EHR mining tasks such as mortality and readmission prediction, yet face two core challenges when processing structured EHR signals.

**Limitations of Prior Work**: The first challenge is the loss of temporal structure. When longitudinal EHR data is linearized into plain text (e.g., "Visit 1: ... Visit 2: ..."), temporal dependencies and the discrete identity of clinical codes become obscured. Although visit boundaries can be delimited by separators, LLMs inherently treat the input as a single document and lack an explicit mechanism for modeling inter-visit temporal dependencies. The second challenge is the absence of population-level information. Traditional EHR predictive models (e.g., RETAIN, GRAM) are trained jointly on patient populations, learning a shared, task-aligned representation space that captures population-level patterns such as disease co-occurrence and longitudinal progression. In contrast, LLMs reason over each patient in isolation and lack any mechanism to leverage information from similar patients.

**Key Challenge**: LLMs excel at extracting rich contextual information from text but lack the ability to model longitudinal temporal structure and exploit population-level knowledge; structured EHR encoders are strong at temporal modeling and population-level pattern discovery but have limited representational capacity. The core problem is how to combine the complementary strengths of both paradigms.

**Goal**: To design a lightweight framework that injects temporal and population-level information from structured EHR encoders into an LLM without modifying its architecture.

**Key Insight**: Prompt tuning enables the injection of external information into an LLM without altering its parameters—external information can be encoded as trainable soft prompt vectors.

**Core Idea**: Augment the LLM with two complementary soft-prompt mechanisms: (1) recurrent prompts that propagate the LLM's hidden state from the previous visit to the current one, explicitly modeling longitudinal dependencies; and (2) struct-encoded prompts that inject embeddings from a population-trained EHR encoder (RETAIN), introducing population-level patterns.

## Method

### Overall Architecture
RePrompT comprises three modules: (1) **Clinical Note Synthesis**: DeepSeek-V3 is used to synthesize discharge notes and structured medical codes into concise patient summaries; (2) **State-Recurrent Prompt Tuning**: the LLM processes visits sequentially, transforming the hidden state of the previous visit via a linear projection into a soft prompt for the current visit; (3) **Struct-Encoded Prompt Tuning**: a RETAIN encoder encodes the structured EHR sequence into a dense representation used as an additional set of soft prompts. The final LLM input is the concatenation of three components: the recurrent prompt $G_{i,t}$, the struct-encoded prompt $S_{i,t}$, and the text embeddings.

### Key Designs

1. **State-Recurrent Prompt Tuning**:

    - **Function**: Explicitly models temporal dependencies across longitudinal visits.
    - **Mechanism**: Rather than concatenating all visits into a single document, the LLM processes one visit at a time. After processing, token-level hidden states $H_{i,t}$ from the final layer are extracted, mean-pooled into a visit-level representation $\hat{H}_{i,t}$, and projected via a linear transformation $G_{i,t+1} = w_t \hat{H}_{i,t} + b_t$ to generate the soft prompt for the next visit. This allows the LLM to "remember" all prior visits when processing the current one.
    - **Design Motivation**: Standard LLMs concatenate all visit texts into a single document and cannot distinguish temporal boundaries between visits. The recurrent mechanism enables the LLM to explicitly pass information from one visit to the next, analogous to hidden state propagation in RNNs.

2. **Struct-Encoded Prompt Tuning**:

    - **Function**: Injects knowledge from a population-trained EHR encoder into the LLM.
    - **Mechanism**: A RETAIN model (with dual-level attention and recurrent modeling) is trained on the patient population to encode the sequence of visit-level medical codes $\{V_{i,j}\}_{j=1}^t$ into a dense representation $S_{i,t} \in \mathbb{R}^{P \times D}$, which is injected into the LLM as soft prompts.
    - **Design Motivation**: LLMs reason over each patient independently and cannot leverage information from similar patients. RETAIN, trained on the population, learns shared patterns such as disease co-occurrence and medication patterns; injecting its embeddings into the LLM effectively provides each patient with a "population reference."

3. **Clinical Note Synthesis**:

    - **Function**: Synthesizes lengthy, noisy discharge notes and structured codes into concise patient summaries.
    - **Mechanism**: DeepSeek-V3 is used to summarize each visit's discharge notes along with the corresponding medical codes, removing boilerplate paragraphs and redundant information to produce a unified patient representation $\hat{C}_{i,t}$.
    - **Design Motivation**: Raw discharge notes are verbose and noisy; feeding them directly into an LLM wastes context window capacity and introduces noise. LLM-based preprocessing distills the key clinical information.

### Loss & Training
Binary Cross-Entropy loss is used for binary/multi-label classification. Trainable parameters include the RETAIN encoder, the linear projection layer for recurrent states, and the output classification head. The LLaMA model remains frozen during both training and inference. Llama 3.1 1B under the LLM2Vec framework is used as the embedding extractor.

## Key Experimental Results

### Main Results (vs. EHR Baselines)

| Model | MIMIC-IV Readmission AUROC | MIMIC-IV Mortality AUROC | MIMIC-III Readmission AUROC | MIMIC-III Mortality AUROC |
|------|---------------------|---------------------|---------------------|---------------------|
| RETAIN | 0.670 | 0.601 | 0.660 | 0.608 |
| StageNet | 0.656 | 0.664 | 0.676 | 0.633 |
| ARCI | 0.663 | 0.611 | 0.652 | 0.618 |
| **RePrompT** | **0.706** | **0.673** | **0.688** | **0.646** |

### Ablation Study

| Configuration | MIMIC-IV Readmission AUROC | MIMIC-IV Mortality AUROC | Note |
|------|---------------------|---------------------|------|
| RePrompT (full) | 0.706 | 0.673 | Full model |
| Remove both modules | 0.673 | 0.635 | Baseline level |
| Remove recurrent prompt | 0.693 | 0.642 | Recurrent prompt contributes more |
| Remove struct-encoded prompt | 0.698 | 0.665 | Struct-encoded prompt also contributes |
| Remove DeepSeek summarization | 0.685 | 0.640 | Summarization preprocessing also helps |

### Key Findings
- RePrompT consistently outperforms all EHR and LLM baselines across all datasets and tasks, with AUROC improvements of approximately 3–7 percentage points.
- The recurrent state prompt contributes more than the struct-encoded prompt—removing the recurrent prompt causes a larger AUROC drop, indicating that longitudinal temporal dependency modeling is the most critical component.
- The comparison with LLM baselines is particularly striking: zero-shot LLM (GPT-5) achieves only 0.512 AUROC on readmission, while RePrompT reaches 0.706—a substantial gap.
- Among different EHR encoders, RETAIN performs best, while Transformer-based encoders perform worst, as Transformers are less effective than RNNs at modeling sequential visit-level dependencies.
- DeepSeek summarization preprocessing is beneficial, but removing it still leaves RePrompT superior to RETAIN, confirming that the performance gains stem primarily from the framework design rather than summarization quality.

## Highlights & Insights
- The **recurrent prompt tuning** design is particularly elegant—it transplants the sequential processing paradigm of RNNs into the prompt space of LLMs, enabling LLMs to process temporal data incrementally, as an RNN would. This idea is transferable to any LLM application involving sequential document processing.
- **Injecting EHR encoder outputs as soft prompts into LLMs** offers a general paradigm for "expert knowledge injection"—any domain-specific encoder can be integrated with an LLM in this manner without modifying the LLM itself.
- The finding that Transformers underperform RNNs in EHR temporal modeling is instructive—positional encodings are insufficient to replace the explicit sequential propagation of RNNs.

## Limitations & Future Work
- Validation is limited to Llama 3.1 1B as the backbone; larger-scale LLMs may exhibit different behavior.
- Using DeepSeek-V3 for summarization preprocessing introduces additional dependencies and computational cost.
- The number of soft prompts $P=10$ is a hyperparameter whose sensitivity is not thoroughly analyzed in the paper.
- The recurrent prompt mechanism could be extended to a multi-scale setting—propagating not only the most recent visit's information but also aggregated representations over longer time horizons.
- Future work may explore extending the framework to more complex clinical prediction tasks such as multi-label medication recommendation.

## Related Work & Insights
- **vs. RETAIN**: RETAIN relies solely on structured code sequences and lacks text comprehension; RePrompT introduces deep semantic understanding of discharge notes via the LLM while retaining RETAIN's population-level patterns.
- **vs. COCONUT**: COCONUT also employs soft tokens for reasoning but lacks explicit temporal structure modeling, underperforming RePrompT on EHR sequential tasks.
- **vs. Zero-shot GPT-5**: Even the most capable general-purpose LLMs fall far short of domain-specialized methods on EHR prediction, underscoring the indispensability of structured medical knowledge injection.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combined design of recurrent prompt tuning and struct-encoded prompt tuning is original, though the individual components (prompt tuning, RNN state propagation) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets, two tasks, eight EHR baselines, three LLM baselines, multi-level ablations, and encoder comparisons—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The architectural diagram is clear, motivations are well articulated, and ablation analyses are thorough.
- **Value**: ⭐⭐⭐⭐ Provides a lightweight and effective LLM–EHR integration paradigm with practical guidance for clinical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)

</div>

<!-- RELATED:END -->

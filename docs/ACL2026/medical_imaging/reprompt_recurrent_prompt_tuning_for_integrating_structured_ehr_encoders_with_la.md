---
title: >-
  [Paper Note] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models
description: >-
  [ACL 2026][Medical Imaging][Electronic Health Records] This paper proposes RePrompT, a time-aware LLM framework that consistently outperforms EHR and LLM baselines on readmission and mortality prediction tasks in MIMIC-I…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Electronic Health Records"
  - "Prompt Tuning"
  - "Recurrent State Propagation"
  - "Structured Encoders"
  - "Clinical Prediction"
date: 2026-05-08
content_hash: 76bbb8a41f0e7433
---

# RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.17725](https://arxiv.org/abs/2604.17725)  
**Code**: [https://github.com/KU-AI4H/RePrompT](https://github.com/KU-AI4H/RePrompT)  
**Area**: Medical Imaging  
**Keywords**: Electronic Health Records, Prompt Tuning, Recurrent State Propagation, Structured Encoders, Clinical Prediction

## TL;DR
This paper proposes RePrompT, a time-aware LLM framework that consistently outperforms EHR and LLM baselines on readmission and mortality prediction tasks in MIMIC-III/IV through two complementary mechanisms: recurrent prompt tuning (passing the hidden state of the previous visit as a soft prompt for the next) and structured-encoded prompt tuning (injecting embeddings from population-level EHR encoders).

## Background & Motivation

**Background**: Electronic Health Records (EHRs) contain longitudinal information such as diagnoses, medications, and procedures from multiple patient visits, serving as a vital data source for clinical decision support. Large Language Models (LLMs) have demonstrated potential in EHR mining tasks (e.g., mortality and readmission prediction) but still face two core challenges when processing structured EHR signals.

**Limitations of Prior Work**: The first challenge is the loss of temporal structure. Linearizing longitudinal EHR data into pure text ("Visit 1: ... Visit 2: ...") obscures temporal dependencies and the discrete identity of clinical codes. Although separators can mark visit boundaries, LLMs inherently treat input as a single document, lacking an explicit mechanism for modeling inter-visit temporal dependencies. The second challenge is the lack of population-level information. Traditional EHR prediction models (e.g., RETAIN, GRAM) are jointly trained on patient populations to learn a shared, task-aligned representation space that discovers population-level patterns such as disease co-occurrence and longitudinal progression. In contrast, LLMs perform independent reasoning per patient (case-isolated), lacking a mechanism to leverage information from similar patients.

**Key Challenge**: LLMs excel at extracting rich contextual information from text but lack the ability to model longitudinal temporal structures and utilize population-level knowledge; structured EHR encoders are proficient in temporal modeling and population pattern discovery but have limited representation capabilities. Combining these complementary strengths is the core problem.

**Goal**: Design a lightweight framework to inject temporal and population-level information from structured EHR encoders into LLMs without modifying the LLM architecture.

**Key Insight**: Prompt tuning allows for the injection of external information without changing LLM parameters—simply by encoding external information into trainable soft prompt vectors.

**Core Idea**: Augment LLMs with two complementary soft prompt mechanisms: (1) recurrent prompts that pass the hidden state of the previous visit to the current one to explicitly model longitudinal dependencies; (2) struct-encoded prompts that inject embeddings from a population-trained EHR encoder (RETAIN) into the LLM to introduce population-level patterns.

## Method

### Overall Architecture
RePrompT consists of three modules: (1) Clinical Record Synthesis: Using DeepSeek-V3 to synthesize discharge summaries and structured medical codes into concise patient summaries; (2) State-Recurrent Prompt Tuning: Processing visits sequentially where the previous visit's hidden state is linearly transformed into a soft prompt for the current visit; (3) Struct-Encoded Prompt Tuning: Using a RETAIN encoder to encode structured EHR sequences into dense representations as another set of soft prompts. The final LLM input is a concatenation of three parts: recurrent prompts $G_{i,t}$, structured prompts $S_{i,t}$, and text embeddings.

### Key Designs

1.  **State-Recurrent Prompt Tuning**:

    - **Function**: Explicitly model temporal dependencies between longitudinal visits.
    - **Mechanism**: Instead of concatenating all visits into a single document, the LLM processes one visit at a time. After processing, the token-level hidden state $H_{i,t}$ from the last layer is extracted, mean-pooled to obtain a visit-level hidden state $\hat{H}_{i,t}$, and then linearly transformed via $G_{i,t+1} = w_t \hat{H}_{i,t} + b_t$ to generate the soft prompt for the next visit. This allows the LLM to "remember" all previous visit information while processing each visit.
    - **Design Motivation**: Standard LLMs treat concatenated visit text as a single document, failing to distinguish temporal boundaries. The recurrent mechanism enables the LLM to explicitly pass information from one visit to the next, similar to hidden state propagation in RNNs.

2.  **Struct-Encoded Prompt Tuning**:

    - **Function**: Inject knowledge from population-trained EHR encoders into the LLM.
    - **Mechanism**: Utilize a RETAIN model (with dual-level attention and recurrent modeling) trained on patient populations to encode visit-level medical code sequences $\{V_{i,j}\}_{j=1}^t$ into dense representations $S_{i,t} \in \mathbb{R}^{P \times D}$, which are then injected as soft prompts.
    - **Design Motivation**: LLMs reason independently for each case and cannot utilize information from other similar patients. RETAIN, trained on populations, learns shared patterns like disease co-occurrence and medication patterns. Injecting its embeddings provides a "population reference" for each patient.

3.  **Clinical Record Synthesis**:

    - **Function**: Synthesize long and noisy discharge summaries and structured codes into concise patient summaries.
    - **Mechanism**: Use DeepSeek-V3 to summarize discharge notes and corresponding medical codes for each visit, removing templated paragraphs and redundant information to produce a unified patient representation $\hat{C}_{i,t}$.
    - **Design Motivation**: Original discharge records are lengthy and noisy; direct input wastes context windows and introduces noise. LLM-based preprocessing extracts key clinical information.

### Loss & Training
Binary Cross-Entropy loss is used for binary/multi-label classification. Trainable parameters include: the RETAIN encoder, linear transformation layers for recurrent states, and the output classification head. The LLaMA model remains frozen during training and inference. Llama 3.1 1B from the LLM2Vec framework is used as the embedding extractor.

## Key Experimental Results

### Main Results (Comparison with EHR Baselines)

| Model | MIMIC-IV Readmission AUROC | MIMIC-IV Mortality AUROC | MIMIC-III Readmission AUROC | MIMIC-III Mortality AUROC |
|------|---------------------|---------------------|---------------------|---------------------|
| RETAIN | 0.670 | 0.601 | 0.660 | 0.608 |
| StageNet | 0.656 | 0.664 | 0.676 | 0.633 |
| ARCI | 0.663 | 0.611 | 0.652 | 0.618 |
| **RePrompT** | **0.706** | **0.673** | **0.688** | **0.646** |

### Ablation Study

| Configuration | MIMIC-IV Readmission AUROC | MIMIC-IV Mortality AUROC | Description |
|------|---------------------|---------------------|------|
| RePrompT Complete | 0.706 | 0.673 | Full model |
| w/o both modules | 0.673 | 0.635 | Baseline level |
| w/o recurrent prompts | 0.693 | 0.642 | Recurrent prompts contribute more |
| w/o struct-encoded | 0.698 | 0.665 | Struct-encoded also contributes |
| w/o DeepSeek summary | 0.685 | 0.640 | Summary preprocessing is helpful |

### Key Findings
- RePrompT consistently outperforms all EHR and LLM baselines across all datasets and tasks, with AUROC improvements of approximately 3-7 percentage points.
- The contribution of state-recurrent prompts is greater than that of struct-encoded prompts—removing recurrent prompts results in a larger AUROC drop, indicating that longitudinal temporal modeling is most critical.
- Comparison with LLM baselines is particularly striking: Zero-shot LLM (GPT-5) achieves an AUROC of only 0.512 (readmission), while RePrompT reaches 0.706, a massive gap.
- In different EHR encoder comparisons, RETAIN performs best, whereas Transformer encoders perform the worst—because Transformers are less effective than RNNs at modeling sequential visit dependencies.
- While DeepSeek summarization helps, RePrompT still outperforms RETAIN even without it, proving the gain comes primarily from framework design rather than summary quality.

## Highlights & Insights
- The design of **State-Recurrent Prompt Tuning** is elegant—it introduces sequential processing from RNNs into the LLM prompt space, allowing LLMs to process temporal data step-by-step. This idea can be transferred to any LLM application requiring temporal document processing.
- **Injecting EHR encoders as soft prompts** provides a universal paradigm for "expert knowledge injection"—any domain-specific encoder can be fused with LLMs this way without modifying the LLM itself.
- The discovery that Transformers are inferior to RNNs for EHR temporal modeling is insightful—positional encodings are insufficient to replace explicit temporal propagation in RNNs.

## Limitations & Future Work
- Currently, only Llama 3.1 1B has been validated as a base model; larger LLMs might behave differently.
- Pre-processing with DeepSeek-V3 introduces extra dependencies and costs.
- The number of soft prompts $P=10$ is a hyperparameter; the paper does not fully discuss the impact of different values.
- Consider extending recurrent prompts to multi-scale—passing information not just from the most recent visit but aggregate information across longer time spans.
- Future work could extend this method to more complex clinical tasks like multi-label medication recommendation.

## Related Work & Insights
- **vs RETAIN**: RETAIN uses only structured code sequences and lacks text understanding; RePrompT introduces deep semantic understanding of discharge notes via LLMs while retaining RETAIN's population-level patterns.
- **vs COCONUT**: COCONUT also uses soft tokens for reasoning but lacks explicit temporal structure modeling, underperforming RePrompT on EHR temporal tasks.
- **vs Zero-shot GPT-5**: Even the strongest general LLMs are inferior to domain-specialized methods in EHR prediction, highlighting the necessity of injecting structured medical knowledge.

## Rating
- Novelity: ⭐⭐⭐⭐ The combination of state-recurrent and struct-encoded prompt tuning is innovative, though individual components (prompt tuning, RNN state passing) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very comprehensive, with two datasets, two tasks, 8 EHR baselines + 3 LLM baselines, multi-level ablations, and encoder comparisons.
- Writing Quality: ⭐⭐⭐⭐ The framework diagram is clear, motivation is well-articulated, and ablation analysis is detailed.
- Value: ⭐⭐⭐⭐ Provides a lightweight and effective LLM-EHR fusion paradigm with practical significance for clinical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[ICLR 2026\] Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](../../ICLR2026/medical_imaging/deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)
- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](../../AAAI2026/medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_imaging/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)

</div>

<!-- RELATED:END -->

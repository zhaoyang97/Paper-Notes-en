---
title: >-
  [Paper Note] MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Recommendation] This paper proposes the MIRA framework, which enables context-aware AI service instruction recommendations on smartphones by long-pressing text or images. Through structured reasoning, template-enhanced reasoning, and trie-constrained decoding, MIRA with a 7B model outperforms GPT-4V (F1: 0.9121 vs. 0.879) while utilizing only 1/7 of the tokens.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Recommendation"
  - "Instruction Recommendation"
  - "Smartphone AI Services"
  - "Trie-constrained Decoding"
  - "Structured Reasoning"
date: 2026-05-08
content_hash: 976ede8202881662
---

# MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation

**Conference**: ACL 2025  
**arXiv**: [2509.13773](https://arxiv.org/abs/2509.13773)  
**Code**: None  
**Area**: Multimodal VLMs  
**Keywords**: Multimodal Recommendation, Instruction Recommendation, Smartphone AI Services, Trie-constrained Decoding, Structured Reasoning

## TL;DR

This paper proposes the MIRA framework, which enables context-aware AI service instruction recommendations on smartphones by long-pressing text or images. Through structured reasoning, template-enhanced reasoning, and trie-constrained decoding, MIRA with a 7B model outperforms GPT-4V (F1: 0.9121 vs. 0.879) while utilizing only 1/7 of the tokens.

## Background & Motivation

**Background**: Generative AI technologies (LLMs, diffusion models, AI Agents) are being deeply integrated into smartphones, providing various AI services such as translation, summarization, navigation, schedule management, and image editing.

**Limitations of Prior Work**: 
   - Current interactions primarily handle user requests through conversational AI assistants (e.g., Siri), which are inefficient for repetitive daily tasks.
   - Users must complete AI tasks through multi-step operations (e.g., OCR $\rightarrow$ information extraction $\rightarrow$ adding to calendar $\rightarrow$ setting reminders).
   - Executing these instructions repeatedly wastes time and effort.

**Key Challenge**: Although smartphone AI service capabilities are rich, user access to these services is neither convenient nor intelligent enough. There lacks a mechanism to automatically infer user intent from multimodal trigger objects (images, text) and recommend instructions.

**Goal**: Design a framework capable of understanding the context from multimodal trigger objects (images/text) and recommending appropriate AI task instructions.

**Key Insight**: Model the problem as an MLLM-based instruction recommendation task, which extracts entities, infers intent, and generates instructions from trigger objects.

**Core Idea**: Achieve one-touch trigger AI services by extracting entities and intent via structured CoT reasoning, improving reasoning accuracy with a template library, and ensuring outputs strictly match predefined instructions using trie-constrained decoding.

## Method

### Overall Architecture

MIRA comprises three core modules:
1. **Structured Chain-of-Thought Reasoning**: Trains the MLLM to perform a three-step reasoning process: entity recognition $\rightarrow$ contextual association analysis $\rightarrow$ instruction generation.
2. **Template-Enhanced Reasoning**: Refines and enriches the reasoning process by retrieving high-level reasoning templates.
3. **Trie-Constrained Decoding**: Switches to a trie during decoding to ensure that the generated output strictly belongs to a predefined set of candidate instructions.

### Key Designs

#### 1. Structured Chain-of-Thought Reasoning
- **Function**: Enables the MLLM to reason out recommended instructions from trigger objects (images/text).
- **Mechanism**: Three-step reasoning — (1) Entity Recognition and Summarization: extracts key entities (phone numbers, addresses, dates, etc.); (2) Contextual Association Analysis: associates entities with user intents (e.g., date $\rightarrow$ calendar reminder, address $\rightarrow$ navigation); (3) Instruction Generation: synthesizes reasoning to output context-aware recommendations.
- **Training Process**: Generates reasoning chains using GPT-4V/Qwen2.5VL-Max via teacher forcing: $r_i = MLLM(p_i^e, q_i, a_i)$, and then trains smaller models using Supervised Fine-Tuning (SFT) on the reasoning dataset.
- **Design Motivation**: Although standard MLLMs excel at OCR and object detection, they struggle to infer implicit user intents from trigger objects.

#### 2. Template-Enhanced Reasoning Mechanism
- **Function**: Corrects and enriches the initial reasoning of the MLLM using a high-level reasoning template library.
- **Mechanism**: 
    - Construct the template library: each template contains a template name, label description, application scenarios, and reasoning steps.
    - Retrieval and Matching: computes the cosine similarity between the embedding of the initial reasoning and the template embeddings, selecting the most relevant template: $j = \text{argmax}_i(\text{Sim}(f(\hat{r}), \{f(D_{T_i})\}_{i=1}^N))$.
    - Filters out irrelevant templates using a similarity threshold $\delta$ (typically $0.5\text{-}0.7$).
    - Updates reasoning: $\hat{r}_{updated} \leftarrow MLLM(T_j, q_i)$.
    - Dynamic evolution: reasoning chains from low-similarity scenarios can be distilled into new templates (deduplication condition: maximum similarity $< \delta$).
- **Design Motivation**: Initial reasoning suffers from randomness and hallucinations; templates offer structured reasoning guidance.

#### 3. Trie-Constrained Decoding
- **Function**: Ensures that the model output strictly belongs to a predefined set of candidate instructions.
- **Mechanism**: Constructs a prefix tree (trie) using the MLLM's tokenizer and candidate instructions. After the reasoning segment ends (following the `</REASONING>` token), the decoder switches to trie mode, masking logits of invalid tokens to ensure that only valid tokens within the trie can be selected at each step.
- **Design Motivation**: Eliminates hallucinations where the MLLM generates irrelevant or nonexistent instructions.

### Loss & Training

- **Training**: Employs standard Supervised Fine-Tuning (SFT) on the constructed reasoning dataset using the cross-entropy loss.
- **Special Tokens**: `<REASONING>` and `</REASONING>` mark the boundary of the reasoning process.
- **Template Library**: ~80 templates extracted from the training data utilizing Qwen2.5VL-Max.
- **Embedding Model**: `jina-embeddings-v3` used for template retrieval.

## Key Experimental Results

### Main Results

Real-world dataset from 1000 smartphone users (4,952 training pairs, 956 test pairs, with an inter-annotator agreement of $\kappa = 0.85$):

| Model | Method | F1 | HR@1 | HR@3 |
|------|------|-----|------|------|
| InternVL2.5-2B | Zero-shot | 0.2971 | 0.3829 | 0.4012 |
| InternVL2.5-2B | MIRA | **0.7271** | **0.8051** | **0.8351** |
| Qwen2.5VL-7B | Zero-shot | 0.3358 | 0.4589 | 0.4924 |
| Qwen2.5VL-7B | Vanilla-SFT | 0.5704 | 0.6012 | 0.6841 |
| Qwen2.5VL-7B | MIRA | **0.9121** | **0.9542** | **0.9629** |

Comparison with large model APIs (MIRA uses Qwen2.5VL-7B):

| Model | F1| Token Length | Inference Time | Parameters |
|------|-----|-----------|---------|--------|
| GPT-4V | 0.879 | 817 | 11.3s | >500B |
| Qwen2.5VL-Max | 0.861 | 807 | 10.7s | >500B |
| **MIRA** | **0.9121** | **116** | 11.2s | **7B** |

### Ablation Study

Impact of template-enhanced reasoning (F1 improvement):

| Model | Initial Reasoning Only | +Template Enhancement | Gain |
|------|-----------|---------|------|
| InternVL2.5-2B | 0.6041 | 0.7271 | +20.4% |
| Qwen2.5VL-2B | 0.6428 | 0.7443 | +15.8% |
| InternVL2.5-8B | 0.7451 | 0.9218 | +23.7% |
| Qwen2.5VL-7B | 0.7348 | 0.9121 | +24.1% |

Sensitivity analysis of the similarity threshold $\delta$: $\delta = 0.6$ consistently yields the optimal result; too low (0.4) leads to overly broad matching, while too high (0.8) yields too few matches.

### Key Findings

1. The MIRA 7B model outperforms GPT-4V ($>500\text{B}$) and Qwen2.5VL-Max ($>500\text{B}$), with F1 score improvements of 3.3% and 5.1% respectively.
2. MIRA's token output is only 1/7 of the large models' (116 vs 807-817), demonstrating extreme efficiency.
3. Template-enhanced reasoning yields an F1 score improvement of 15.8% to 24.1%, with greater gains observed in larger models.
4. User studies (100 participants evaluating 500 trigger objects) show a success rate of 93%-95%.
5. Failure cases primarily comprise missed entities (33%), template mismatching, and trigger ambiguity.

## Highlights & Insights

- **Novel Application Scenario**: First to define and tackle the "one-touch AI service" instruction recommendation problem on smartphones.
- **Small Model Outperforming Large Models**: MIRA with 7B parameters comprehensively outperforms the $500\text{B}+$ GPT-4V in both accuracy and efficiency.
- **Ingenious Engineering Design**: The combination of trie-constrained decoding and template retrieval-augmentation guarantees output validity while improving reasoning quality.
- **Sustainable Template Library Evolution**: Automatic generation of new templates is triggered in low-similarity scenarios, making the framework adaptive to dynamic deployment environments.

## Limitations & Future Work

1. Only supports text and image triggers, lacking support for audio, video, and sensor data.
2. The template library building stage depends on closed-source LLMs, leaving cross-model generalizability to be verified.
3. Predefined instruction sets limit applicability in open-domain scenarios.
4. Privacy concerns: processing sensitive user content such as images, documents, and messages necessitates privacy protection mechanisms.
5. Failure cases still persist in complex or ambiguous trigger scenarios.

## Related Work & Insights

- **LLaVA-CoT** (Xu et al., 2024): A pioneer in structured visual reasoning; MIRA's reasoning design shares a similar philosophy.
- **VIP5** (Geng et al., 2023): A multimodal recommendation foundation model focusing on user behavior sequences rather than trigger objects.
- **MLLM-MSR** (Ye et al., 2024): Employs MLLMs for multimodal sequential recommendation, which differs in task setting from MIRA.
- Trie-constrained decoding is a mature technique in NLP; MIRA ingeniously applies it to the instruction recommendation scenario.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Defines a completely new "one-touch AI service" scenario with an ingeniously designed solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes multi-model comparisons, ablations, LLM API comparisons, and user studies, though the dataset is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, intuitive diagrams, and clearly defined problems.
- **Value**: ⭐⭐⭐⭐ — Driven by practical Huawei deployment scenarios, highly practical as the 7B model is deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](../../NeurIPS2025/multimodal_vlm/scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[NeurIPS 2025\] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images](../../NeurIPS2025/multimodal_vlm/gem_empowering_mllm_for_grounded_ecg_understanding_with_time_series_and_images.md)
- [\[ACL 2025\] A Survey on Patent Analysis: From NLP to Multimodal AI](patent_analysis_survey.md)
- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](branchlora_continual_instruction.md)

</div>

<!-- RELATED:END -->

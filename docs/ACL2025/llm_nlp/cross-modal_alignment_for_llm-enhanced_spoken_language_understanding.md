---
title: >-
  [Paper Note] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding
description: >-
  [ACL 2025][LLM (Other)][Cross-modal alignment] This paper proposes a cross-modal alignment framework that achieves LLM-enhanced spoken language understanding (SLU) by explicitly aligning speech representations with the textual semantic space of LLMs, obtaining SOTA performance on intent detection and slot filling tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Cross-modal alignment"
  - "large language models"
  - "spoken language understanding"
  - "speech-to-text alignment"
  - "SLU"
date: 2026-05-08
content_hash: 84b8c528adf817fa
---

# Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding

**Conference**: ACL 2025  
**Area**: LLM/NLP  
**Keywords**: Cross-modal alignment, large language models, spoken language understanding, speech-to-text alignment, SLU

## TL;DR

This paper proposes a cross-modal alignment framework that achieves LLM-enhanced spoken language understanding (SLU) by explicitly aligning speech representations with the textual semantic space of LLMs, obtaining SOTA performance on intent detection and slot filling tasks.

## Background & Motivation

**Background**: Spoken Language Understanding (SLU) is a key component of spoken dialogue systems. Traditional methods employ cascaded ASR+NLU architectures, while end-to-end SLU models have emerged in recent years. Large Language Models (LLMs) have demonstrated strong capabilities in textual NLU, but how to enable LLMs to process speech inputs directly remains an open problem.

**Limitations of Prior Work**: Cascade methods suffer from error propagation from ASR to downstream NLU, leading to performance degradation, which is especially severe in noisy environments and highly colloquial expressions. End-to-end SLU models avoid error propagation but lack the powerful semantic understanding capability of LLMs. Existing speech LLMs (such as AudioPaLM and Qwen-Audio) primarily focus on speech generation and general audio understanding, with insufficient optimization for SLU tasks.

**Key Challenge**: The semantic understanding capability of LLMs resides in the textual modality space, whereas speech representations naturally have a modality gap with it. Simply concatenating speech features and textual prompts fails to effectively exploit the semantic reasoning capability of LLMs.

**Goal**: To design an efficient cross-modal alignment method that maps the outputs of a speech encoder into the semantic space of LLMs, enabling LLMs to "understand" speech content and perform precise intent detection and slot filling.

**Key Insight**: The authors observe that speech and text naturally correspond at the semantic level—speech and text of the same sentence should map to close positions in the LLM's semantic space. This cross-modal alignment can be explicitly established through contrastive learning and attention bottleneck mechanisms.

**Core Idea**: To use a trainable Modality Bridge to compress and align the continuous representations of a speech encoder to the LLM's text embedding space, and then leverage the frozen parameters of the LLM for semantic understanding, thereby achieving end-to-end audio-to-semantic mapping.

## Method

### Overall Architecture

The system consists of three main components: (1) a pre-trained speech encoder (e.g., Whisper encoder) to extract speech feature sequences; (2) a cross-modal bridge to compress variable-length speech features into fixed-length semantic tokens; (3) a frozen LLM that receives semantic tokens and task prompts to output intent labels and slot tags. During training, only the bridge parameters are updated, while both the speech encoder and the LLM are frozen.

### Key Designs

1. **Attention Bottleneck Bridge**:

    - **Function**: Compresses variable-length speech feature sequences into a fixed number of semantic tokens.
    - **Mechanism**: Uses a set of learnable query tokens (e.g., 32 tokens) to extract crucial semantic information from speech feature sequences via a cross-attention mechanism. Specifically, the query tokens act as Q, and the speech features act as K and V. After multiple layers of cross-attention, the compressed semantic representation $Z = \text{CrossAttn}(Q_{learnable}, K_{speech}, V_{speech})$ is obtained. It is then projected linearly to align its dimensionality with the LLM's embedding dimension.
    - **Design Motivation**: Speech feature sequences are generally long (about 50 frames per second), incurring massive computational overhead if directly input into the LLM. The attention bottleneck achieves information compression while retaining the most semantically relevant information.

2. **Semantic Alignment Contrastive Learning**:

    - **Function**: Ensures that the semantic tokens output by the bridge are semantically consistent with the corresponding text in the LLM space.
    - **Mechanism**: During the training phase, for each speech-text pair, the speech representation $z_s$ via the bridge and the text representation $z_t$ obtained after the LLM tokenizer are computed. A pair-wise speech-text representation is brought closer using the InfoNCE loss $\mathcal{L}_{align} = -\log \frac{\exp(sim(z_s, z_t)/\tau)}{\sum_j \exp(sim(z_s, z_t^j)/\tau)}$.
    - **Design Motivation**: Without explicit alignment constraints, the bridge might learn representations incompatible with the LLM's semantic space, preventing the LLM from correctly "understanding" the speech content.

3. **Task-Adaptive Prompting**:

    - **Function**: Guides the LLM to generate output in the correct format for different SLU subtasks (intent detection, slot filling).
    - **Mechanism**: Structured prompt templates are designed separately for intent detection and slot filling, inserting the compressed semantic tokens into specific positions within the prompts. The intent detection task uses a classification prompt, while slot filling uses a sequence labeling prompt. Both tasks can be decoded jointly or independently.
    - **Design Motivation**: The instruction-following capability of LLMs makes them naturally suited to perform specific tasks according to structured prompts. The key is to design appropriate prompt formats and insertion positions for semantic tokens.

### Loss & Training

The total loss is a weighted sum of three parts: $\mathcal{L} = \mathcal{L}_{task} + \alpha \mathcal{L}_{align} + \beta \mathcal{L}_{reg}$. $\mathcal{L}_{task}$ is the standard cross-entropy, $\mathcal{L}_{align}$ is the contrastive alignment loss, and $\mathcal{L}_{reg}$ is a regularization term to prevent the bridge output from deviating too far from the LLM distribution. A two-stage training strategy is adopted: first pre-training with speech-text pairs for alignment, followed by fine-tuning on SLU labeled data.

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | Ours | Whisper+GPT4 | E2E-SLU SOTA | Gain |
|--------|------|------|---------|-------------|-------------|------|
| SLURP | Intent Acc | Acc | 91.2 | 88.5 | 87.3 | +3.9 |
| SLURP | Slot F1 | SLU-F1 | 82.6 | 79.1 | 78.4 | +4.2 |
| FSC | Intent Acc | Acc | 99.7 | 99.2 | 99.1 | +0.6 |
| SNIPS-Audio | Intent Acc | Acc | 97.8 | 95.6 | 94.9 | +2.9 |
| STOP | Semantic Parsing | EM | 85.3 | 82.1 | 80.7 | +4.6 |

### Ablation Study

| Configuration | Intent Acc | Slot F1 | Description |
|------|---------|--------|------|
| Full model | 91.2 | 82.6 | Full model |
| w/o Contrastive Alignment | 88.1 | 79.3 | Removing alignment loss drops 3.1/3.3 |
| w/o Attention Bottleneck | 89.5 | 80.8 | Direct linear projection |
| w/o Two-Stage Training | 89.8 | 80.1 | End-to-end single-stage training |
| 16 query tokens | 90.4 | 81.5 | Slight degradation with fewer tokens |
| 64 query tokens | 91.0 | 82.4 | Performance gain saturates with more tokens |

### Key Findings
- Contrastive alignment loss is the most critical component, contributing approximately 60% of the performance gains, validating the necessity of explicit cross-modal alignment.
- The number of query tokens achieves the best trade-off at around 32; too few cause information loss, while too many introduce redundancy.
- Under noisy speech conditions (SNR=10dB), the proposed method's advantage over the cascaded approach (ASR+NLU) becomes more pronounced (+6.8 vs +3.9), demonstrating the robustness advantage of the end-to-end methodology.

## Highlights & Insights
- **Modality bridging paradigm with frozen LLMs**: Without fine-tuning the LLM, training only a lightweight bridge allows the LLM to "understand" speech. This approach exhibits high parameter efficiency (trainable parameters <5%), and this paradigm can be extended to other modalities (e.g., integrating video or sensor data into LLMs).
- **Compression efficiency of the attention bottleneck**: Compresses hundreds of frames of speech features into 32 semantic tokens, achieving an over 10x compression ratio with minimal performance loss.
- The two-stage strategy of contrastive alignment followed by task fine-tuning can be reused for other cross-modal understanding tasks.

## Limitations & Future Work
- Currently, only English SLU has been validated; multilingual spoken language understanding scenarios remain unexplored.
- The frozen LLM restricts the model's utilization of speech-specific information (such as tone of voice and emotion).
- The capability to handle long speech segments (>30 seconds) remains unverified.
- Future research can explore encoding speech prosody information into semantic tokens to enhance emotional comprehension capabilities.

## Related Work & Insights
- **vs Whisper+GPT-4 pipeline**: Cascade methods are limited by ASR error propagation and cannot exploit non-textual information in speech. Our end-to-end method consistently outperforms the pipeline across all metrics.
- **vs Qwen-Audio**: Qwen-Audio is a general audio LLM, whereas its precision on specific SLU tasks is inferior to the task-specialized method proposed in this paper.
- **vs SALMONN**: SALMONN also utilizes a bridge architecture but lacks explicit semantic alignment constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ The cross-modal alignment ideology is clear and effective, though the bridge architecture is not pioneered here.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple SLU benchmarks with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and visually intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides an effective solution for the application of speech LLMs to SLU tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)
- [\[CVPR 2025\] Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](../../CVPR2025/llm_nlp/chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)
- [\[ACL 2025\] Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning](beyond_output_matching_bidirectional_alignment_for_enhanced_in-context_learning.md)

</div>

<!-- RELATED:END -->

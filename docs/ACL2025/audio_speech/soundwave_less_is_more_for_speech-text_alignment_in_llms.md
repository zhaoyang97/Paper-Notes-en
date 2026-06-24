---
title: >-
  [Paper Note] Soundwave: Less is More for Speech-Text Alignment in LLMs
description: >-
  [ACL 2025][Audio & Speech][Speech LLMs] The Soundwave model is proposed to address the representation space gap and sequence length inconsistency between speech and text using efficient training strategies and a novel architecture. With only one-fiftieth of the training data, it outperforms Qwen2-Audio on speech translation and AIR-Bench speech tasks.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Speech LLMs"
  - "Speech-Text Alignment"
  - "Data-Efficient Training"
  - "Sequence Length Inconsistency"
  - "Representation Space Gap"
date: 2026-05-08
content_hash: 48cc81b8a2d1f1ce
---

# Soundwave: Less is More for Speech-Text Alignment in LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.12900](https://arxiv.org/abs/2502.12900)  
**Code**: [GitHub](https://github.com/FreedomIntelligence/Soundwave)  
**Area**: NLP/Speech  
**Keywords**: Speech LLMs, Speech-Text Alignment, Data-Efficient Training, Sequence Length Inconsistency, Representation Space Gap  

## TL;DR

The Soundwave model is proposed to address the representation space gap and sequence length inconsistency between speech and text using efficient training strategies and a novel architecture. With only one-fiftieth of the training data, it outperforms Qwen2-Audio on speech translation and AIR-Bench speech tasks.

## Background & Motivation

**Background**: End-to-end speech large language models (Speech LLMs) have developed rapidly in recent years. Mainstream methods typically require large-scale annotated speech-text paired data for training to achieve speech understanding, translation, and dialogue capabilities. Representative models like Qwen2-Audio typically rely on hundreds of thousands to millions of hours of speech training data.

**Limitations of Prior Work**: Large-scale data-driven training strategies incur high computational and data collection costs. More importantly, two fundamental issues exist between speech and text: (1) **Representation space gap**—the features output by the speech encoder and the text embeddings of the LLM reside in different semantic spaces; (2) **Sequence length inconsistency**—for the same semantic content, the speech feature sequence is much longer than the corresponding text token sequence, posing a major challenge for alignment.

**Key Challenge**: Existing methods attempt to bridge these two gaps by accumulating more data, which is essentially an empirical brute-force approach that fails to address the issue at the architectural level. Data-efficient training remains underexplored in the field of speech LLMs.

**Goal**: Design a data-efficient speech-text alignment scheme that achieves or exceeds the performance of models trained on massive datasets using only a minimal amount of data (only 10k hours).

**Key Insight**: The authors observe that directly addressing the representation space gap and sequence length inconsistency at the architectural level can substantially reduce dependence on training data. Less is More—a well-designed architecture is more effective than scale.

**Core Idea**: Explicitly resolve representation space alignment and sequence compression through a novel speech-text bridging architecture and an efficient training strategy, achieving SOTA performance with 1/50 of the data.

## Method

### Overall Architecture

The overall pipeline of Soundwave is: Speech Input → Speech Encoder (e.g., Whisper encoder) extracting acoustic features → Bridging Module for spatial alignment and length compression → LLM Backbone (based on Qwen2) for understanding and generation → Text Output. The core innovation lies in the design of the bridging module and the efficient training strategy.

### Key Designs

1. **Speech-Text Representation Space Alignment Module**:

    - **Function**: Maps the output features of the speech encoder to the LLM's text embedding space.
    - **Mechanism**: Uses a specially designed projection network to transform speech features from the acoustic space to the semantic space understood by the LLM. Unlike simple linear mappings, this module uses multi-layer transformations to ensure that speech features and text token embeddings lie on the same manifold.
    - **Design Motivation**: Although previous methods also used projection layers, they often relied on massive amounts of data to forcefully learn the alignment. The proposed design makes alignment more precise, thereby reducing data requirements.

2. **Sequence Length Compression Mechanism**:

    - **Function**: Compresses excessively long speech feature sequences to a length comparable to text token sequences.
    - **Mechanism**: Speech signals have high frame rates (e.g., Whisper encoder outputs 50 frames per second), whereas corresponding text may only consist of a few tokens. This module shortens the speech sequence length multiple times through downsampling and pooling operations, making it close to the text sequence length so the LLM can handle speech input more naturally.
    - **Design Motivation**: Sequence length mismatch is the core cause of attention mechanism inefficiency and alignment difficulties. Explicit compression avoids the computational burden of processing overly long sequences in the LLM.

3. **Data-Efficient Training Strategy**:

    - **Function**: Achieves high-quality speech-text alignment using only approximately 10k hours of speech data.
    - **Mechanism**: Employs a multi-stage training strategy—first freezing the LLM and speech encoder to train only the bridging module for preliminary alignment, and then unfreezing parts of the LLM parameters for end-to-end fine-tuning. Meanwhile, high-quality training data is carefully curated to ensure data diversity and coverage.
    - **Design Motivation**: In contrast to Qwen2-Audio which uses around 500k hours of data, this work demonstrates that a well-designed architecture can reduce the data requirement by 50-fold.

### Loss & Training

A standard autoregressive language modeling loss (next-token prediction) is adopted for joint training on speech understanding and translation tasks. The multi-stage training ensures that the bridging module converges first, preventing the large model parameters from being corrupted by noisy gradients in the early stages.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | Soundwave | Qwen2-Audio | Comparison |
|-------------|------|-----------|-------------|------|
| Speech Translation (CoVoST2 en→de) | BLEU | Lead | Baseline | Soundwave Outperforms |
| Speech Translation (CoVoST2 en→zh) | BLEU | Lead | Baseline | Soundwave Outperforms |
| AIR-Bench Speech | Overall Score | Higher | Baseline | Outperforms with 1/50 of the data |
| Speech Recognition (ASR) | WER | Competitive | Baseline | Close or Comparable |
| Dialogue Intelligence Retention | Subjective Score | Retained | Baseline | No Significant Decline |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full Soundwave | Optimal | Full Model |
| w/o Sequence Compression | Significant Decline | Long sequences lead to decreased attention efficiency and alignment quality |
| w/o Representation Space Alignment | Substantial Decline | Simple linear projection fails to fully bridge the spatial gap |
| w/o Multi-stage Training | Moderate Decline | Direct end-to-end training struggles to converge |
| Using More Data (50k h) | Slight Improvement | Diminishing marginal returns, indicating the architectural design is sufficient |

### Key Findings

- The representation space alignment module and the sequence compression mechanism are the two most critical components; omitting either leads to a significant degradation in performance.
- Increasing the training data from 10k to 50k hours yields minimal improvement, validating the core "Less is More" argument.
- Soundwave retains the general intelligence of LLMs in dialogue scenarios, indicating that the bridging module does not damage the baseline capabilities of the LLM.

## Highlights & Insights

- **Astonishing Data Efficiency**: Outperforming strong baselines with a 50x smaller dataset demonstrates that architectural design is more critical than data scale for speech-text alignment. This finding is of great significance for deploying speech LLMs in resource-constrained environments.
- **Decoupled Design**: Separating representation space alignment and sequence length compression into two distinct problems is intuitive and effective. This "divide-and-conquer" strategy can be transferred to other modality alignment tasks (e.g., video-text).
- **Dialogue Capability Retention**: Unlike some multimodal fine-tuning methods that trigger language degradation (catastrophic forgetting), Soundwave successfully preserves the LLM's general capabilities through multi-stage training.

## Limitations & Future Work

- The paper primarily evaluates performance on speech translation and speech understanding; its efficacy on finer-grained tasks like speech emotion analysis and speaker identification remains unexplored.
- The current implementation is based solely on the Qwen2 backbone; generalization to other LLMs (e.g., LLaMA) requires verification.
- Although 10k hours of data is significantly less than 500k, it may still present a high barrier for low-resource languages.
- The selection rationale for specific architectural details of the bridging module (e.g., number of layers, dimensionality) warrants further discussion.
- Future research can explore applying similar efficient alignment strategies to other multimodal scenarios such as video understanding.

## Related Work & Insights

- **vs Qwen2-Audio**: Qwen2-Audio relies on large-scale data training, which is effective but costly. Soundwave demonstrates that architectural optimization can dramatically reduce data requirements.
- **vs SALMONN**: SALMONN is another speech LLM, but its alignment approach is relatively simple (linear projection). Soundwave's multi-layer bridging design is more sophisticated.
- **vs Whisper**: Whisper is a speech-only model without conversation capabilities. Soundwave unifies Whisper's recognition capabilities with the language understanding of LLMs under a single framework.

## Rating

- Novelty: ⭐⭐⭐⭐ The direction of data-efficient speech-text alignment is valuable, though the specific techniques (projection + compression) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple tasks with comprehensive ablation studies, though it lacks evaluation on some fine-grained tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear arguments and a well-defined "Less is More" stance.
- Value: ⭐⭐⭐⭐ Highly relevant for the deployment of speech LLMs in resource-constrained environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](../../AAAI2026/audio_speech/say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](../../AAAI2026/audio_speech/hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model
description: >-
  [ACL 2025][LLM (Other)][simultaneous translation] This paper proposes InfiniSST, which models unbounded streaming speech simultaneous translation as an LLM multi-turn dialogue task, combining robust segment training data construction, multi-delay augmentation strategies, and Λ-shaped KV cache management to reduce computation-aware latency by 0.5-1 second on MuST-C En-Es/De/Zh directions without losing translation quality.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "simultaneous translation"
  - "streaming speech"
  - "KV cache"
  - "multi-turn dialogue"
  - "unbounded input"
date: 2026-05-08
content_hash: 986f9be420a4e2f8
---

# InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model

**Conference**: ACL 2025  
**Authors**: Siqi Ouyang, Xi Xu, Lei Li (CMU)  
**arXiv**: [2503.02969](https://arxiv.org/abs/2503.02969)  
**Code**: [GitHub](https://github.com/LeiLiLab/InfiniSST)  
**Area**: LLM/NLP, Speech Translation  
**Keywords**: simultaneous translation, streaming speech, KV cache, multi-turn dialogue, unbounded input  

## TL;DR

This paper proposes InfiniSST, which models unbounded streaming speech simultaneous translation as an LLM multi-turn dialogue task, combining robust segment training data construction, multi-delay augmentation strategies, and Λ-shaped KV cache management to reduce computation-aware latency by 0.5-1 second on MuST-C En-Es/De/Zh directions without losing translation quality.

## Background & Motivation

- **Core Problem:** Simultaneous translation (SST) must balance quality and latency, but existing methods mostly assume pre-segmented speech (SST-S) and cannot handle continuous, unbounded streaming speech input in real-world scenarios (SST-U).
- **Limitations of Prior Work:** Conventional LLM-based SST-S methods recalculate historical speech and generated translation features whenever a new speech chunk arrives, incurring massive computational overhead. While some works model SST as multi-turn dialogue to leverage KV cache (e.g., Yu et al., 2025; Wang et al., 2024), these approaches target segmented speech and cannot extend seamlessly to unbounded inputs.
- **Design Motivation:** LLMs possess powerful long-context modeling capabilities (RoPE positional encoding + attention window), making them an ideal foundation for SST-U. The key challenges are: (1) how to construct training data suitable for unbounded speech; (2) how to manage the KV cache during inference to achieve constant memory usage.

## Method

### Overall Architecture

InfiniSST consists of three core components:
1. **Streaming Speech Encoder** (modified wav2vec2): Incrementally computes speech representations to avoid redundant computation.
2. **Speech-to-Token Embedding Adapter**: A two-layer 1-D convolution (kernel=2, stride=2) + linear projection, compressing 48 frames into 12 LLM embedding vectors.
3. **Multi-turn LLM Decoder** (Llama-3.1-8B-Instruct): Alternately reads speech inputs and generates translations, controlling read-write switching via EOT tokens.

Inference workflow: System Instruction → Loop{USER token + 12 speech embeddings + EOT → ASSISTANT generates translation → switches back to reading upon encountering EOT}.

### Key Designs

1. **Streaming Speech Encoder Modification**: Replaces the bidirectional attention of wav2vec2 with chunk-wise causal attention (bidirectional within a chunk, causal across chunks), replaces convolutional positional encoding with RoPE, and adds a sliding window of `$w^s=10$` chunks (approx. 9.6 seconds of context). Each chunk consists of 48 frames, lasting 960ms.
2. **Training Data Construction**: (a) Uses MFA forced alignment + SimAlign to establish a monotonic mapping of speech → transcription → translation; (b) segments talks into 30-chunk "robust segments" to include non-speech sounds for enhanced robustness; (c) multi-delay augmentation—randomly selects a delay multiplier `$m \in [1, M]$` to merge successive chunks and their translations, where `$M=12$`.
3. **KV Cache Management Strategy**: During inference, the LLM maintains the KV cache of the system instruction + the KV cache of the most recent `$w^t=1000$` tokens. No positional information is embedded in the stored KV values (RoPE is removed before storage), and RoPE is reapplied after concatenation. This achieves infinite length extrapolation using a Λ-shaped attention window.

### Loss & Training

Standard cross-entropy loss (applied only to translation tokens and EOT). Two-stage training: (1) Freeze the LLM, train the encoder + adapter for 6 epochs (lr=2e-4); (2) Freeze the encoder + adapter, train the LLM for 1 epoch (lr=7e-6). Single-node training on 8×L40S GPUs.

## Experiments

### Main Results——MuST-C Full TED Talks (27 talks, 3-23 minutes)

| Comparison | Computation-Aware Latency (StreamLAAL_CA) | Non-Computation-Aware Latency (StreamLAAL) | Translation Quality |
|------|---------------------------|--------------------------|---------|
| InfiniSST vs StreamAtt+ | **Reduced by 0.5-1 seconds** | Comparable or slightly better | BLEU comparable or 0.5-1.0 higher |
| InfiniSST RTF vs StreamAtt+ RTF | **Less than half** | — | — |

When StreamLAAL $\le$ 1.5s, InfiniSST achieves slightly higher BLEU (0.5-1.0) in all three language directions, with comparable COMET.

### Ablation Study

| Ablation Component | Results |
|---------|------|
| Robust Segments vs Original Segments | Without robust segments, the model gets stuck in repetitions or stops translating when encountering non-speech sounds like laughter, with COMET degrading from 69.2 to 50.5. |
| Multi-delay Augmentation M=1 vs 12 | A larger $M$ yields a better quality-latency trade-off, but inference $m$ should not exceed $M$. |
| Speech Window `$w^s$`=5/10/20/40 | Quality degrades when the inference window does not match training (66.1 vs 69.2). It is recommended to use the maximum allowable window. |
| LLM Window `$w^t$`=500→4000 | COMET only increases from 69.0 to 69.4, indicating LLM robustness to window size. |
| Instruction KV Cache Removal | The LLM stops translating after the window slides, indicating that the instruction cache is indispensable. |
| Llama-3 (8K ctx) vs Llama-3.1 (128K) | The 8K model still generalizes to >10 min talks (COMET 67.1 vs 68.0). |

### Key Findings

- Robust segment training is crucial for adapting the model to unbounded speech—without it, the model completely fails to handle non-speech sounds.
- KV cache management reduces RTF to less than half of StreamAtt+, which is the core of computational efficiency improvements.
- Even if the base LLM context is only 8K, it can still handle unbounded speech far exceeding 8K tokens through KV cache management.

## Highlights & Insights

- Modeling unbounded streaming SST as a multi-turn dialogue is an elegant abstraction that perfectly leverages the LLM's KV cache mechanism.
- The first successful application of the Λ-shaped attention window in speech translation.
- The meticulously designed training data construction (robust segments + multi-delay augmentation) is essential for model generalization.

## Limitations & Future Work

- Still lags behind AlignAtt and StreamAtt at high theoretical latency levels (chunk-wise causal attention restricts bidirectional information).
- Evaluated only in the En-X direction, without testing X-En and X-X.
- Did not explore other speech encoders and non-Llama LLMs due to computational budget constraints.
- StreamLAAL metrics are not entirely reliable due to mWERSegmenter alignment errors.
- Did not perform human evaluation of translation quality.

## Related Work & Insights

- **Cascaded SST:** Pipeline approach of ASR segmentation + MT translation (Fugen et al., 2006), suffering from error propagation.
- **End-to-End SST-U:** AlignAtt/StreamAtt extended to unbounded speech (Papi et al., 2024a), but requiring full history storage.
- **LLM Length Extrapolation:** RoPE (Su et al., 2021), Λ-shaped attention window (Han et al., 2024; Xiao et al., 2024).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](locateandfocus_enhancing_terminology_translation_in_speech.md)
- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)

</div>

<!-- RELATED:END -->

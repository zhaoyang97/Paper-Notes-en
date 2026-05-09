---
title: >-
  [Paper Note] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation
description: >-
  [ICLR2026][Audio & Speech][Speech Emotion Recognition] This paper proposes Dynamic Parameter Memory (DPM), a mechanism that encodes speech information sentence-by-sentence into the parameter space of a temporary LoRA module during inference, enabling speech large language models (SLLMs) with limited context windows to process arbitrarily long conversational audio. The approach achieves state-of-the-art performance on IEMOCAP and MELD.
tags:
  - ICLR2026
  - "Audio & Speech"
  - Speech Emotion Recognition
  - Large Language Model
  - LoRA
  - Long-Sequence Processing
  - Emotion Recognition in Conversation
date: 2026-05-08
content_hash: de38f63577ef177b
---

# Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation

**Conference**: ICLR2026
**arXiv**: [2507.09076](https://arxiv.org/abs/2507.09076)
**Code**: To be confirmed
**Area**: Audio & Speech
**Keywords**: Speech Emotion Recognition, Large Language Model, LoRA, Long-Sequence Processing, Emotion Recognition in Conversation

## TL;DR

This paper proposes Dynamic Parameter Memory (DPM), a mechanism that encodes speech information sentence-by-sentence into the parameter space of a temporary LoRA module during inference, enabling speech large language models (SLLMs) with limited context windows to process arbitrarily long conversational audio. The approach achieves state-of-the-art performance on IEMOCAP and MELD.

## Background & Motivation

Speech large language models (SLLMs) have demonstrated considerable potential for speech emotion recognition (SER), yet the inherently high frame rate of the speech modality severely constrains their ability to process long audio sequences. At a 50 Hz sampling rate, an SLLM with a 4K context window can process only approximately 80 seconds of audio — far insufficient for real-world conversational or meeting recording scenarios.

Existing solutions fall into two categories: (1) compressing input tokens (e.g., low-frame-rate codecs, multi-scale Transformers), which tend to overlook the continuity and inertia of emotion across conversational turns; and (2) expanding the model's context window (e.g., Kimi, Qwen2.5), where the quadratic complexity of standard attention mechanisms causes computational costs to escalate sharply with sequence length. Both approaches encounter fundamental performance bottlenecks when confronted with sufficiently long audio sequences.

## Core Problem

How can an SLLM with a limited context window efficiently process arbitrarily long speech emotion conversations while preserving cross-utterance emotional context?

## Method

### 1. Emotion SLLM Training

Llama2-7B (32K context window) serves as the base LLM, with audio converted to discrete codes using the CosyVoice2 tokenizer (25 Hz, single codebook). The LLM vocabulary is extended with:

- Speech tokens `<audio_i>`
- End-of-utterance marker `<audio_end>`
- Four emotion markers `<emo_hap>`, `<emo_sad>`, `<emo_ang>`, `<emo_neu>`

Training employs LoRA (rank=64, alpha=64) with two loss functions:

- **Audio autoregressive loss** $\mathcal{L}_a$: at each sentence end, the last $n_o$ tokens are used as prediction targets and the first $n_p$ tokens as a prefix; next-token prediction is trained via teacher forcing.
- **Emotion supervision loss** $\mathcal{L}_e$: the first $n_q$ tokens at the sentence end serve as a prefix for predicting the emotion marker. Output logits are constrained to the four emotion marker positions to improve prediction stability.

The total loss is $\mathcal{L} = \frac{1}{2}(\mathcal{L}_a + \mathcal{L}_e)$.

### 2. DPM Inference Mechanism

During inference, the emotion SLLM and its LoRA module are frozen, and a **temporary LoRA module** is created for each long audio sample. The core procedure is:

1. **Sentence-by-sentence processing**: at the end of sentence $i$, the first $n_r$ tokens serve as a prefix, and the tokens of sentence $i+1$ are predicted autoregressively.
2. **Parameter update**: the autoregressive loss $\mathcal{L}_t$ between predicted and ground-truth tokens is computed, and backpropagation updates the temporary LoRA parameters.
3. **Information encoding**: through loss-driven parameter updates, the semantic content and emotional context of the current sentence are "memorized" into the parameter space of the temporary LoRA.
4. **Emotion prediction**: after the final sentence is processed, the temporary LoRA is updated once more and an emotion token is predicted as the emotion classification result for the entire conversation.
5. **Cleanup**: the temporary LoRA is discarded after inference to avoid interfering with subsequent samples.

The key condition enabling DPM to handle unbounded audio is: $n_{\text{limit}} \geq n_{\text{max}} + n_r$, meaning the context window only needs to accommodate the maximum token count of a single sentence plus the prefix length.

### 3. Design Motivation

The design is analogous to the human memory system: the emotion SLLM acts as **long-term memory** (general emotional knowledge and audio understanding), while the temporary LoRA serves as **short-term working memory** (contextual information for the current conversation). DPM inference resembles "reading sentence by sentence carefully" rather than "skimming," avoiding the omission of critical emotional information.

## Key Experimental Results

### Ablation Study (IEMOCAP / MELD)

| Method | IEMOCAP WA | IEMOCAP UA | IEMOCAP WF1 | MELD WF1 |
|--------|-----------|-----------|------------|---------|
| SLLM-DPM | **79.38%** | **79.62%** | **79.34%** | **51.22%** |
| SLLM (w/o DPM) | 72.82% | 73.34% | 73.58% | 47.90% |
| Classifier | 70.96% | 70.51% | 70.64% | 44.78% |

- DPM improves over direct SLLM inference on full conversations by **6.56% WA** (IEMOCAP) and **3.32% WF1** (MELD).
- Across all sample lengths, DPM still maintains a 2.23% WA advantage.

### Main Results

DPM achieves 79.38% WA / 79.62% UA / 79.34% WF1 on IEMOCAP and 51.22% WF1 on MELD, surpassing all compared methods including GatedxLSTM and MERITS-L.

### Key Hyperparameters

- Prefix length: 1024 for IEMOCAP, 256 for MELD (correlated with average conversation length: ~64.96 utterances/conversation for IEMOCAP, ~9.80 for MELD).
- LoRA rank=64, alpha=64, activating 0.16B parameters.
- Training learning rate: 5e-5; DPM inference learning rate: 5e-5.

## Highlights & Insights

1. **Novel application of inference-time parameter updates**: extends the Temporary LoRA concept from text to speech emotion recognition, addressing key cross-modal transfer challenges including the high temporal density of speech and the unique structure of emotional expression.
2. **Linear complexity**: DPM's computational cost scales linearly with the number of sentences rather than quadratically with total token count, enabling truly scalable long-sequence processing.
3. **Plug-and-play**: DPM is a pure inference-time mechanism requiring no additional training, and can directly augment existing emotion SLLMs for long-sequence scenarios.
4. **Elegant design intuition**: the predict–compare–update cycle naturally encodes semantic and emotional information into the parameter space, avoiding the information loss inherent in fixed-length embeddings.

## Limitations & Future Work

1. **Validated only on discrete speech codes**: uses the CosyVoice2 tokenizer; applicability to continuous representations (e.g., HuBERT/WavLM features) has not been explored.
2. **Dependency on sentence boundaries**: DPM requires prior knowledge of utterance boundaries, necessitating an additional VAD or segmentation module in practical applications.
3. **Inference latency**: each sentence requires one forward and one backward pass to update the temporary LoRA, resulting in considerable inference overhead.
4. **Limited dataset scale**: validation is restricted to IEMOCAP (5,531 utterances) and MELD; evaluation on larger-scale datasets is lacking.
5. **Fixed emotion categories**: the current design requires emotion markers to be predefined in the vocabulary; extending to more emotion classes necessitates retraining.
6. **Unimodal**: only audio is used; text or visual modality information is not incorporated.

## Related Work & Insights

| Dimension | Ours (DPM) | Token Compression | Larger Context Window | Embedding Compression (e.g., Murph) |
|-----------|-----------|-------------------|----------------------|--------------------------------------|
| Theoretical limit | Unlimited length | Bounded by compression ratio | Bounded by window + compute | Unlimited but with information loss |
| Complexity | Linear (# sentences) | Method-dependent | Quadratic (# tokens) | Linear |
| Emotional continuity | Preserved in parameter space | Easily lost | Naturally preserved | May be lost at fixed length |
| Additional training | Not required | Required | Required | Required |

Key difference from Temporary LoRA (Wang et al., 2024): the original method targets the text modality, whereas this paper addresses the speech modality, requiring solutions to high frame-rate and emotion expression structure discrepancies, and leverages a specially trained emotion SLLM as an emotion-aware backbone.

**Inspirations and connections:**

- **Inference-time learning paradigm**: DPM belongs to the test-time training/adaptation paradigm, shifting "memory" from explicit KV caches to implicit parameter spaces — an idea potentially generalizable to other modalities requiring long-sequence processing (video understanding, time-series, etc.).
- **Parameters as memory**: treating LoRA parameters as dynamically updatable memory storage conceptually relates to memory-augmented neural networks, yet with a simpler implementation.
- **LLM-oriented trends in SER**: the autoregressive structure outperforms traditional classifiers in sequential emotion understanding, suggesting the SER field may further migrate toward the LLM paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying test-time LoRA updates to speech emotion recognition is a novel combinatorial contribution.
- Experimental Thoroughness: ⭐⭐⭐ — Validation on two standard datasets is solid, but additional datasets and ablation dimensions are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear, motivation is well-articulated, and mathematical derivations are complete.
- Value: ⭐⭐⭐⭐ — Provides a scalable solution for long-sequence speech emotion recognition with practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](../../AAAI2026/audio_speech/cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](../../AAAI2026/audio_speech/do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)
- [\[ICLR 2026\] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [\[ICLR 2026\] LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision](logicreward_incentivizing_llm_reasoning_via_step-wise_logical_supervision.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Long-Form Speech Generation with Spoken Language Models
description: >-
  [ICML 2025 Oral][Audio & Speech][Spoken Language Models] Proposes SpeechSSM, the first textless spoken language model capable of learning and generating up to 16 minutes of speech in a single decoding session. It leverages the Griffin hybrid SSM architecture to achieve constant-memory decoding and infinite context, and introduces the LibriSpeech-Long evaluation benchmark along with new embedding and LLM-as-a-judge metrics.
tags:
  - "ICML 2025 Oral"
  - "Audio & Speech"
  - "Spoken Language Models"
  - "Long-Form Speech Generation"
  - "State Space Models"
  - "SSM"
  - "Speech Evaluation"
date: 2026-05-08
content_hash: ed9f1e19e7e970a4
---

# Long-Form Speech Generation with Spoken Language Models

**Conference**: ICML 2025 Oral  
**arXiv**: [2412.18603](https://arxiv.org/abs/2412.18603)  
**Code**: [https://google.github.io/tacotron/publications/speechssm/](https://google.github.io/tacotron/publications/speechssm/)  
**Area**: Speech Generation  
**Keywords**: Spoken Language Models, Long-Form Speech Generation, State Space Models, SSM, Speech Evaluation

## TL;DR
Proposes SpeechSSM, the first textless spoken language model capable of learning and generating up to 16 minutes of speech in a single decoding session. It leverages the Griffin hybrid SSM architecture to achieve constant-memory decoding and infinite context, and introduces the LibriSpeech-Long evaluation benchmark along with new embedding and LLM-as-a-judge metrics.

## Background & Motivation
**Background**: Existing spoken language models (such as GSLM, TWIST, and Spirit LM) can only generate speech up to tens of seconds, constrained by the quadratic complexity of self-attention in Transformers and the high temporal resolution of speech tokens (25Hz, where roughly 10 speech tokens correspond to 1-2 text tokens).

**Limitations of Prior Work**: (a) Transformer inference memory grows linearly, preventing infinite continuation; (b) extremely long speech token sequences lead to a loss of semantic consistency; (c) existing evaluation metrics are highly noisy and lack discriminative power in long-form settings.

**Key Challenge**: Real-world applications (voice assistants, podcasts, audiobooks) require minutes of coherent speech, yet current models can only reliably generate speech on a scale of seconds.

**Goal**: (a) Modeling level: constant-memory, infinite-context long-form speech generation; (b) Evaluation level: evaluation methodologies and benchmarks for long-form speech generation.

**Key Insight**: Replace pure Transformers with hybrid SSMs (State Space Models + local attention) and leverage the fixed-size state of SSMs to compress contexts of arbitrary distance.

**Core Idea**: Griffin hybrid SSM + high-quality semantic tokens (USM-v2) + windowed processing = an infinitely generation-capable speech LM.

## Method

### Overall Architecture
SpeechSSM operates in two stages: (1) Semantic stage: A Griffin hybrid SSM is used to autoregressively predict USM-v2 semantic tokens (25Hz, 32k vocabulary); (2) Acoustic stage: SoundStorm is applied non-autoregressively to convert semantic tokens into SoundStream acoustic tokens, which are then decoded into waveforms. Speaker characteristics are injected into the acoustic stage via a 3-second speech prompt.

### Key Designs

1. **Griffin Hybrid SSM Architecture**:

    - **Function**: Serves as the autoregressive decoder for semantic tokens.
    - **Mechanism**: Alternates between gated LRUs (Linear Recurrent Units) and local sliding-window multi-query attention (2:1 ratio). Local attention captures recent context, while the LRUs transmit information across arbitrary distances.
    - **Design Motivation**: To satisfy the three requirements of constant-memory decoding, infinite context, and length extrapolation during generation.

2. **Windowed Tokenization and Decoding**:

    - **Function**: Enables non-SSM components (the semantic tokenizer and acoustic decoder) to handle long-form speech.
    - **Mechanism**: Splits long audio into fixed-length windows (30s) with a 4s overlap. Each window is tokenized and decoded independently, and then concatenated at the overlap boundaries.
    - **Design Motivation**: Both the USM-v2 tokenizer and SoundStorm have context limits, making windowing an essential engineering design.

3. **Avoiding Implicit EOS**:

    - **Function**: Solves the issue where early models fail to generate beyond the training duration.
    - **Mechanism**: Non-causal tokenizers (such as USM-v2) implicitly encode "remaining length" information in the tokens of the final window. The solution is to pad the end of the final window with speech from the beginning of the audio rather than silence.
    - **Design Motivation**: To make the tokens appear as though "more speech is following," thereby supporting length extrapolation.

4. **Text LM Initialization**:

    - **Function**: Initializes SpeechSSM from RecurrentGemma-2B/9B.
    - **Mechanism**: Retains the pre-trained architectural weights while discarding text token embeddings and re-initializing audio token embeddings.
    - **Design Motivation**: Previous work like TWIST has demonstrated that text LM initialization enhances semantic consistency.

### Loss & Training
- Trained on LibriLight unlab-60k, using 4-minute (240s) audio segments by default.
- Utilizes 16 TPUs (v5p), 100k steps, with a batch size of 768k tokens.
- Sampling temperature is set to 1, with checkpoints selected based on the transcript PPL of LibriSpeech-Long dev-clean.

## Key Experimental Results

### Main Results (7s Short-form Continuation)

| Model | PPL ↓ | SBERT ↑ | SpkrSim ↑ | N-MOS ↑ |
|------|-------|---------|-----------|---------|
| TWIST-7B | 6.54 | 0.20 | 0.41 | 3.24 |
| Spirit LM 7B | 6.17 | 0.19 | 0.45 | 3.00 |
| **SpeechSSM-2B** | **5.76** | **0.23** | **0.79** | **3.87** |
| **SpeechSSM-9B** | **5.60** | **0.23** | **0.79** | **3.94** |
| Ground Truth | 5.63 | 1.00 | 0.84 | 4.02 |

### Long-form Evaluation (4-minute Continuation)

| Model | PPL ↓ | SpkrSim ↑ | Description |
|------|-------|-----------|------|
| Spirit LM ⊞ | High | 0.45 | Sliding window extension, semantic degradation |
| SpeechSSM-2B | **Lowest** | **0.79** | Maintains semantic consistency |

### Key Findings
- SpeechSSM achieves a significantly higher SpkrSim than other models (0.79 vs. 0.41-0.45), which is attributed to the large vocabulary of USM-v2 (32k) and the speaker conditioning in the acoustic stage.
- The 9B model outperforms the 2B model in both short-form and long-form settings, indicating that SSMs benefit from scaling in a similar manner.
- The existing sWUGGY metric fails under a large vocabulary setting, showing no positive correlation with generation quality.

## Highlights & Insights
- **First Long-Form Speech LM**: Achieves a qualitative leap by scaling generation capabilities from tens of seconds directly to 16 minutes.
- **LibriSpeech-Long Benchmark**: Fills the gap in long-form speech evaluation, providing an open-source dataset with standardized splits.
- **LLM-as-Judge Evaluation**: Employs LLMs for side-by-side comparative evaluations, resolving the high noise issue inherent in ASR metrics.

## Limitations & Future Work
- Only reading-style and free-form generation are supported, while conversational speech remains unaddressed.
- The semantic tokenizer USM-v2 is not open-source, which limits reproducibility.
- No direct comparison is made with conversational systems like Moshi.
- Windowed concatenation may introduce unnatural transitions at segment boundaries.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic application of SSMs in speech LMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across short-form/long-form contexts, new metrics, and benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Rich in engineering details and thorough in problem analysis.
- Value: ⭐⭐⭐⭐⭐ Pioneering contribution to the field of speech generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios](../../ACL2026/audio_speech/comprehensive_benchmarking_of_long-form_speech_generation_in_diverse_scenarios.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ICLR 2026\] YuE: Scaling Open Foundation Models for Long-Form Music Generation](../../ICLR2026/audio_speech/yue_scaling_open_foundation_models_for_long-form_music_generation.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[CVPR 2026\] AudioStory: Generating Long-Form Narrative Audio with Large Language Models](../../CVPR2026/audio_speech/audiostory_generating_long-form_narrative_audio_with_large_language_models.md)

</div>

<!-- RELATED:END -->

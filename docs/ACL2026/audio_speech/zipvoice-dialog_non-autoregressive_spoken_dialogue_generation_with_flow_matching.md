---
title: >-
  [Paper Note] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching
description: >-
  [ACL 2026][Audio & Speech][Dialogue Speech Generation] Ours proposes ZipVoice-Dialog, the first non-autoregressive (NAR) zero-shot dialogue speech generation model based on flow matching. Through two simple designs—a cur…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Dialogue Speech Generation"
  - "Non-Autoregressive"
  - "Flow Matching"
  - "Speaker Turns"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: 443de84fe879bd18
---

# ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching

**Conference**: ACL 2026  
**arXiv**: [2507.09318](https://arxiv.org/abs/2507.09318)  
**Code**: [https://github.com/k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice)  
**Area**: Speech Generation  
**Keywords**: Dialogue Speech Generation, Non-Autoregressive, Flow Matching, Speaker Turns, Curriculum Learning

## TL;DR

Ours proposes ZipVoice-Dialog, the first non-autoregressive (NAR) zero-shot dialogue speech generation model based on flow matching. Through two simple designs—a curriculum learning strategy and speaker-turn embeddings—it addresses the issues of unintelligible speech and turn confusion when flow matching is directly applied to dialogue scenarios. Ours also releases OpenDialog, the first large-scale open-source dialogue speech dataset (6.8k hours).

## Background & Motivation

**Background**: Text-to-speech (TTS) technology has achieved excellent results in single-speaker monologue scenarios. However, synthesizing natural multi-speaker dialogues remains a significant challenge, as dialogue requires accurate and natural speaker turn-taking and the maintenance of distinct timbres for different speakers.

**Limitations of Prior Work**: Current state-of-the-art dialogue speech generation methods primarily rely on autoregressive (AR) architectures (e.g., MoonCast, Dia). However, AR models suffer from two inherent flaws: (1) high inference latency due to step-by-step sequential generation; (2) severe robustness issues where exposure bias leads to unstable phenomena such as word repetition or skipping.

**Key Challenge**: While flow matching as an NAR method has shown excellent performance in monologue TTS, preliminary experiments by the authors found that directly applying flow matching architectures to dialogue generation results in completely unintelligible speech—the model can mimic the style and timbre of the prompt but fails entirely to reflect the input text content. This is because involving two different speaker timbres in a dialogue makes learning speech-text alignment extremely difficult.

**Goal**: To design an effective method to make flow matching architectures suitable for multi-speaker dialogue generation while addressing the scarcity of training data.

**Key Insight**: The authors observe that the root of the problem lies in the difficulty of alignment learning for multi-speaker timbres. Therefore, the approach starts from a "learn alignment first, then dialogue" curriculum learning perspective and provides clear speaker cues through explicit speaker-turn embeddings.

**Core Idea**: Use curriculum learning (monologue pre-training followed by dialogue fine-tuning) to solve the alignment problem, and use learnable speaker-turn embeddings to solve the turn-switching problem, allowing the NAR flow matching architecture to achieve both high speed and high stability in dialogue generation.

## Method

### Overall Architecture

ZipVoice-Dialog is based on ZipVoice (a monologue flow matching TTS model) and consists of a text encoder and a vector field estimator (both using the Zipformer backbone), along with a pre-trained Vocos vocoder. The input is an interleaved multi-speaker text sequence and prompt audio; the output is the complete dialogue speech. Training utilizes a Conditional Flow Matching (CFM) objective and a speech in-painting task to achieve zero-shot capabilities.

### Key Designs

1.  **Monologue-to-Dialogue Curriculum Learning**:
    *   **Function**: Solves the speech-text alignment collapse issue when flow matching models are trained directly on dialogue data.
    *   **Mechanism**: Training is conducted in two stages. **Stage 1**: Initialize with weights from ZipVoice (pre-trained on 100k hours of monologue data) to establish a robust speech-text alignment foundation. **Stage 2**: Fine-tune on dialogue data to learn dialogue dynamics—alignment adaptation in multi-speaker contexts, correct timbre assignment, and natural turn-switching.
    *   **Design Motivation**: Training from scratch directly on dialogue data leads to alignment collapse (WER surges from ~5% to >100%) because the presence of two speaker timbres greatly increases the difficulty of alignment learning. Learning alignment on simple monologue tasks before transferring to dialogue is a classic curriculum learning approach.

2.  **Speaker-Turn Embeddings**:
    *   **Function**: Enables the model to accurately distinguish between two speakers and assign the correct timbre to each turn.
    *   **Mechanism**: Two randomly initialized learnable embedding vectors are introduced, corresponding to identities [S1] and [S2]. For each token $y_i$ in the text sequence, the corresponding embedding $e_{speaker(i)}$ is retrieved based on its speaker identity and added to the text features: $\widetilde{y_i} = \hat{y_i} + e_{speaker(i)}$, followed by upsampling in the temporal dimension.
    *   **Design Motivation**: Comparative experiments show that using delimiters "|" or text tokens [S1]/[S2] cannot effectively distinguish speakers (cpWER is much higher than WER), whereas speaker-turn embeddings significantly reduce cpWER from 37.82/31.34 to 5.82, achieving near-perfect turn accuracy.

3.  **Interleaved Text & Flexible Prompting**:
    *   **Function**: Handles the complex input formats of multi-speaker dialogues.
    *   **Mechanism**: Multiple turns of speech are arranged chronologically into a single interleaved text sequence, prefixed with speaker identifiers. During training, dialogue audio with random prefix lengths is used as a prompt; during inference, prompt audio of any number of turns is supported. The model implicitly models token and turn duration through the flow matching objective without requiring an external duration predictor.
    *   **Design Motivation**: Compared to schemes requiring predefined timestamps, end-to-end implicit modeling is simpler, and neither training nor inference depends on additional timestamp prediction models.

### Loss & Training

A Conditional Flow Matching (CFM) loss is used, calculated only in masked regions:  
$$L_{CFM} = \mathbb{E}_{t,q(x_1),p_0(x_0)} \| (v_t(x_t, z, (1-m) \odot x_1; \theta) - (x_1 - x_0)) \odot m \|^2$$  
Ours fine-tunes for 60k steps on OpenDialog + internal datasets (7.6k hours total), with a total batch size of 4k seconds. Inference uses an Euler solver with 16-step sampling.

## Key Experimental Results

### Main Results

Comparison with open-source dialogue speech generation models (Chinese and English test sets):

| Model | Parameters | RTF↓ | Zh WER↓ | En WER↓ | cpSIM↑ | UTMOS↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Dia | 1.61B | 1.663 | - | 11.80 | 0.333 | 1.87 |
| MoonCast | 2.67B | 0.953 | 15.85 | 23.62 | 0.356 | 2.37 |
| ZipVoice-Dialog | **123M** | **0.063** | **3.17** | **3.25** | **0.437** | **3.07** |

ZipVoice-Dialog achieves total dominance with only 123M parameters: over 15x faster inference and a 3-7x reduction in WER.

### Ablation Study

| Configuration | En WER↓ | En Short cpWER↓ | Description |
| :--- | :--- | :--- | :--- |
| Full Model (Curriculum + Embeddings) | 3.25 | 3.27 | Optimal |
| Without Curriculum Learning | 116.10 | 116.31 | Alignment collapse, completely unintelligible |
| Delimiter "|" instead of Embeddings | 5.34 | 37.82 | Poor turn accuracy |
| Text tokens instead of Embeddings | 5.57 | 31.34 | Poor turn accuracy |
| OpenDialog data only | 3.34 | 3.53 | Data alone achieves a strong baseline |

### Key Findings

*   Curriculum learning is indispensable: without it, the model fails entirely (WER >100%), indicating that multi-speaker alignment is the core bottleneck for flow matching dialogue generation.
*   Speaker-turn embeddings are extremely simple yet highly effective: two learnable vectors reduce the turn error rate from >30% to <1%.
*   In subjective evaluations (CMOS/SMOS), ZipVoice-Dialog significantly leads MoonCast (CMOS gap of -1.17, SMOS 3.86 vs 2.35).
*   Using the OpenDialog dataset alone can train a model that surpasses baselines, verifying the high quality of the dataset.

## Highlights & Insights

*   The **minimalist yet highly effective design philosophy** is impressive—using only curriculum learning + two learnable embeddings to transform a flow matching architecture from "completely unusable" to "outperforming AR baselines across the board." This approach of gaining maximum effect with minimal changes is worth emulating.
*   The **OpenDialog dataset** is a very valuable open-source contribution—as the first large-scale (6.8k hours) open-source dialogue speech dataset, it fills a gap in the field. The data construction pipeline (VAD → Diarization → ASR → LLM classification → WhisperD refinement → Rule filtering) is reusable.
*   The 123M parameter model outperforms 1.6B-2.7B AR models on all metrics, proving the immense potential of NAR architectures in dialogue scenarios.

## Limitations & Future Work

*   Model and data scales are limited; small models have a ceiling on expressiveness, which larger models and more data might further improve.
*   Subjective evaluation is restricted to Chinese; the subjective quality of English has not been verified.
*   Currently limited to two-person dialogue; although the method is scalable to more people, it has not been validated.
*   Natural dialogue phenomena such as overlapping speech and backchannels have not yet been explored.

## Related Work & Insights

*   **vs MoonCast**: MoonCast adopts an AR+NAR hybrid architecture (LLM → Flow Matching → Vocoder) with 2.67B parameters, but the AR part leads to severe skipping and instability (WER 23.62%). ZipVoice-Dialog, a pure NAR with 123M parameters, achieves 3.25% WER and is 15x faster.
*   **vs Dia**: Dia is a pure AR model that directly predicts audio tokens. Despite 1.61B parameters, it has a WER of 11.80% and the lowest timbre similarity (cpSIM 0.333). This indicates that the pure AR route lacks robustness in dialogue scenarios.

## Rating

*   Novelty: ⭐⭐⭐⭐ The first work to successfully apply flow matching to dialogue speech generation, though the core techniques (curriculum learning, embeddings) are not inherently new.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Complete ablation studies, subjective and objective evaluations, dataset comparisons, and benchmark establishment; very thorough.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, with a very smooth logical chain between problem motivation and solutions; easy to understand.
*   Value: ⭐⭐⭐⭐⭐ Triple contribution of model + dataset + benchmark; the OpenDialog dataset is particularly important to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](../../ICLR2026/audio_speech/flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [\[NeurIPS 2025\] Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis](../../NeurIPS2025/audio_speech/shallow_flow_matching_for_coarse-to-fine_text-to-speech_synthesis.md)
- [\[ACL 2026\] SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness](sdiareward_modeling_and_benchmarking_spoken_dialogue_rewards_with_modality_and_c.md)
- [\[ICML 2026\] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer](../../ICML2026/audio_speech/towards_streaming_synchronized_spatial_audio_generation_via_autoregressive_diffu.md)

</div>

<!-- RELATED:END -->

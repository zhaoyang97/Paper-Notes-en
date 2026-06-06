---
title: >-
  [Paper Note] Latent Speech-Text Transformer
description: >-
  [ICLR 2026 Oral][Audio & Speech][speech-text modeling] This paper proposes the Latent Speech-Text Transformer (LST), which aggregates discrete speech tokens into higher-level "latent speech patches" as autoregressive uni…
tags:
  - "ICLR 2026 Oral"
  - "Audio & Speech"
  - "speech-text modeling"
  - "latent patches"
  - "autoregressive"
  - "ASR"
  - "TTS"
  - "cross-modal alignment"
  - "BLT"
date: 2026-05-08
content_hash: a2f68a8e00fa3c87
---

# Latent Speech-Text Transformer

**Conference**: ICLR 2026 Oral
**arXiv**: [2510.06195](https://arxiv.org/abs/2510.06195)  
**Code**: [GitHub](https://github.com/facebookresearch/lst)  
**Area**: Audio & Speech
**Keywords**: speech-text modeling, latent patches, autoregressive, ASR, TTS, cross-modal alignment, BLT

## TL;DR
This paper proposes the Latent Speech-Text Transformer (LST), which aggregates discrete speech tokens into higher-level "latent speech patches" as autoregressive units (analogous to BLT's treatment of bytes), aligning the sequence modeling granularity of speech and text (reducing the length ratio from 20× to ~1:1). LST achieves +6.5% absolute improvement on Speech HellaSwag, with gains that continue to grow from 420M to 7B parameters, while reducing ASR/TTS inference computation.

## Background & Motivation
**Background**: Discrete speech tokens (e.g., HuBERT at 25 Hz with a 501-entry codebook) have made autoregressive speech language modeling feasible. However, speech token sequences are far longer than their textual counterparts (10–20×), resulting in training and inference efficiency well below that of text LLMs — it is estimated that approximately three orders of magnitude more data are required to achieve comparable capability.

**Limitations of Prior Work**:
   - **Information density mismatch**: The severe asymmetry in sequence length between speech tokens and text tokens impedes cross-modal knowledge transfer.
   - **Unbalanced compute allocation**: Most computation during pretraining and inference is spent on long speech sequences rather than meaningful semantic modeling.
   - **Insufficient alignment attempts**: Warm initialization (from text LLMs) and interleaved training help, but significant performance gaps between speech→speech and text→text tasks remain.
   - BPE fails on speech tokens (Cuervo & Marxer 2024) — simple subword segmentation is not applicable to speech.

**Key Challenge**: Speech modeling requires fine-grained tokens (25 Hz), yet autoregressive modeling is inefficient over long sequences and yields poor cross-modal alignment.

**Core Idea**: Drawing inspiration from the Byte Latent Transformer (BLT), speech tokens are aggregated into "latent patches" (higher-level autoregressive units). A global Transformer performs modeling at the patch level, while a lightweight decoder expands patches back into speech tokens. Patch granularity is aligned with that of text tokens.

## Method

### Overall Architecture
Speech token sequence $\{s_0, \ldots, s_T\}$ → Patch Encoder (sliding-window self-attention + cross-attention, aggregating into patch representations $\{z_0, \ldots, z_{T'}\}$, $T' \ll T$) → Global Transformer (autoregressive modeling over patches + text tokens) → Patch Decoder (lightweight Transformer + cross-attention, reconstructing speech tokens from patches) → standard NTP loss.

### Key Designs

1. **Three Patching Strategies**

    - **Static Patching**: Non-overlapping segmentation of fixed size $p$ (e.g., $p=3$, one patch per 3 speech tokens). Simple and efficient; no auxiliary model required at inference.
    - **Alignment Patching**: Wav2Vec2+CTC forced alignment is used to obtain speech–text timestamps; each text unit (word/BPE) corresponds to one patch, with silence segments forming separate patches. This precisely aligns speech and text granularities.
    - **Curriculum Patching** (final approach): Training gradually transitions from alignment to static patching. The probability $P(u) = 1 \to 0$ decays linearly over training steps $[\tau_1, \tau_2]$. Early training benefits from the semantic correspondence provided by alignment; later training switches to static patching to eliminate dependence on the alignment model at inference.
    - **Design Motivation**: Alignment patching provides the best cross-modal alignment but requires an auxiliary model; curriculum patching retains the benefits while eliminating the inference-time dependency.

2. **Patch Encoder and Patch Decoder**

    - **Encoder**: Sliding-window self-attention + cross-attention layers aggregate token embeddings into patch embeddings.
    - **Decoder**: Lightweight Transformer with cross-attention inserted at each layer to receive patch-level information; self-attention window of 512 tokens.
    - **Compute allocation**: The global Transformer dominates FLOPs; the Encoder and Decoder are lightweight. By performing global modeling at the patch level rather than the token level, computation is substantially reduced.

3. **Cross-Modal Alignment Mechanism**

    - Patch-level modeling causes speech and text to appear at comparable granularities within the same sequence.
    - Interleaved data training: text and speech from the same corpus alternate, with some speech segments replaced by their textual counterparts.
    - Effect: Patches automatically learn correspondences to syllables/words, facilitating S↔T knowledge transfer.

### Loss & Training
- Standard NTP loss (at the token level), applied to the output of the patch decoder.
- End-to-end training (encoder + global Transformer + decoder).
- Speech tokenizer: HuBERT at 25 Hz, 501-entry codebook.
- Text tokenizer: Llama 2 tokenizer.

## Key Experimental Results

### Main Results (Speech HellaSwag, story completion)

| Setting | Condition | LST Gain |
|------|------|---------|
| Compute-controlled (same training steps) | 420M | +6.5% absolute |
| Data-controlled (same data volume) | 420M | +5.3% absolute |
| Compute-optimal scaling | 420M → 1.8B | **Gains increase with scale** |
| Fixed-token budget | 7B, 70B tokens | Gains persist |

Key finding: gains not only do not saturate but **continue to grow as model size increases**, indicating that LST improves compute-optimal scaling.

### Downstream Tasks

| Task | Result | Notes |
|------|------|------|
| ASR adaptation | More stable | Patch-level modeling reduces long-range dependency issues |
| TTS inference | Shorter sequences, lower compute | Direct benefit of compressed sequence length |
| Reconstruction quality | No degradation | Demonstrates lossless patch compression |
| Text→Text | Also improves | Cross-modal training indirectly boosts text capability |

### Ablation Study

| Configuration | Result | Notes |
|------|------|------|
| No patching (baseline) | Baseline | Standard interleaved training |
| BPE on speech tokens | No improvement / degradation | Confirms BPE inapplicability to speech tokens |
| Static patching ($p=3$) | Significant improvement | Even simple segmentation is effective |
| Alignment patching | Best, but requires auxiliary model | Value of semantic alignment |
| **Curriculum patching** | **Best trade-off** | Retains alignment benefits without inference auxiliary model |

### Key Findings
- **Information density alignment is central**: Bringing speech and text to comparable sequence lengths substantially improves cross-modal knowledge transfer, supporting the hypothesis that granularity mismatch is the primary bottleneck.
- Even the simplest static patching is effective — indicating that the issue lies not in precise semantic alignment but in reducing redundancy in speech sequences.
- **Patches automatically learn semantic correspondences**: Curriculum patching begins with alignment but ultimately transitions to static patching; the model retains the learned correspondences, suggesting that alignment signals can be "distilled" into patch representations.
- Gains grow with model scale — an important implication for scaling laws: LST may shift the compute-optimal point for speech LMs.
- Text performance also improves — speech patching not only does not harm text modeling but indirectly enhances it through better cross-modal training.

## Highlights & Insights
- **Successful transfer of the BLT paradigm from text to speech**: The core idea of the Byte Latent Transformer — aggregating fine-grained tokens into patches for global modeling — proves equally effective in the speech domain and may be even more impactful given that speech exhibits greater redundancy than bytes.
- **Efficiency and quality simultaneously improved**: Reducing sequence length while improving quality is not a trade-off but a win-win. The reason: shorter sequences make it easier for the global Transformer to capture long-range dependencies.
- **Elegant curriculum design**: Alignment patching requires an inference-time auxiliary model (impractical); static patching loses semantic alignment (suboptimal); curriculum patching smoothly transitions from the former to the latter — using alignment during training and static patching at inference.

## Limitations & Future Work
- The robustness of patch size selection across languages with diverse syllabic structures and varying speaking rates has not been thoroughly validated.
- Only HuBERT semantic tokens are evaluated; codec-based acoustic tokens (e.g., SoundStorm) are not tested.
- No direct comparison with end-to-end speech LLMs such as Moshi or Spirit-LM.
- The curriculum schedule hyperparameters $\tau_1, \tau_2$ require tuning.
- The 7B experiments are conducted under a suboptimal token budget (70B vs. the estimated optimal ~140B); full compute-optimal experiments are prohibitively expensive.

## Related Work & Insights
- **vs. BLT (Pagnoni 2024)**: LST transfers the byte→patch idea from BLT directly to speech token→speech patch, serving as the primary inspiration.
- **vs. Nguyen 2025 (interleaved training)**: Interleaved training constitutes the baseline; LST adds patching on top — an orthogonal improvement.
- **vs. Moshi (end-to-end speech LLM)**: Moshi employs multi-stream modeling; LST uses patch compression — distinct approaches to resolving information density mismatch.

## Rating
- Novelty: ⭐⭐⭐⭐ The latent patch concept is concise and effective; the transfer from BLT to speech is both natural and non-trivially designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-scale evaluation (420M→7B) × two controlled settings × downstream tasks × thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear and accessible, with a coherent logical flow from motivation to design to experiments.
- Value: ⭐⭐⭐⭐⭐ ICLR Oral is well deserved; the work offers important guidance for joint speech-text modeling and improves scaling behavior.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](../../ACL2026/audio_speech/immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ICLR 2026\] VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation](vowelprompt_hearing_speech_emotions_from_text_via_vowel-level_prosodic_augmentat.md)
- [\[ICML 2026\] Position: Towards Responsible Evaluation for Text-to-Speech](../../ICML2026/audio_speech/position_towards_responsible_evaluation_for_text-to-speech.md)
- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](../../ACL2026/audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)

</div>

<!-- RELATED:END -->

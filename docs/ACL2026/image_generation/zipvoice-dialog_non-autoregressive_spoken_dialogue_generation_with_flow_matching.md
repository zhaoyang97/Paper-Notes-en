---
title: >-
  [Paper Note] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching
description: >-
  [ACL 2026][Image Generation][spoken dialogue generation] This paper proposes ZipVoice-Dialog, the first non-autoregressive zero-shot spoken dialogue generation model based on flow matching. Through two simple designs—curriculum learning and speaker-turn embeddings—it addresses the unintelligible speech and turn confusion problems that arise when flow matching is directly applied to dialogue scenarios. The paper also releases OpenDialog (6.8k hours), the first large-scale open-source spoken dialogue dataset.
tags:
  - ACL 2026
  - Image Generation
  - spoken dialogue generation
  - non-autoregressive
  - flow matching
  - speaker turn
  - curriculum learning
date: 2026-05-08
content_hash: 7807ac61647b912d
---

# ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching

**Conference**: ACL 2026
**arXiv**: [2507.09318](https://arxiv.org/abs/2507.09318)
**Code**: [https://github.com/k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice)
**Area**: Image Generation
**Keywords**: spoken dialogue generation, non-autoregressive, flow matching, speaker turn, curriculum learning

## TL;DR

This paper proposes ZipVoice-Dialog, the first non-autoregressive zero-shot spoken dialogue generation model based on flow matching. Through two simple designs—curriculum learning and speaker-turn embeddings—it addresses the unintelligible speech and turn confusion problems that arise when flow matching is directly applied to dialogue scenarios. The paper also releases OpenDialog (6.8k hours), the first large-scale open-source spoken dialogue dataset.

## Background & Motivation

**Background**: Text-to-speech (TTS) technology has achieved impressive results in single-speaker monologue scenarios. However, synthesizing natural multi-speaker conversations remains a significant challenge, as dialogue requires accurate and natural speaker turn transitions while maintaining distinct timbre for different speakers.

**Limitations of Prior Work**: Current state-of-the-art dialogue speech generation methods primarily rely on autoregressive (AR) architectures (e.g., MoonCast, Dia), which suffer from two inherent drawbacks: (1) high inference latency due to sequential token-by-token generation; and (2) serious robustness issues, where exposure bias leads to instability such as word repetition or word skipping.

**Key Challenge**: Flow matching, as a non-autoregressive approach, has demonstrated strong performance in monologue TTS. However, preliminary experiments reveal that directly applying a flow matching architecture to dialogue generation results in completely unintelligible speech—the model can imitate the style and timbre of the prompt audio but fails entirely to reflect the input text content. This occurs because the presence of two distinct speaker timbres makes speech-text alignment learning extremely difficult.

**Goal**: Design effective methods to adapt the flow matching architecture for multi-speaker dialogue generation while addressing the scarcity of training data.

**Key Insight**: The authors identify that the root cause lies in the difficulty of alignment learning under multiple speaker timbres. They thus approach the problem from a curriculum learning perspective—"learn alignment first, then learn dialogue"—and provide explicit speaker cues via speaker-turn embeddings.

**Core Idea**: Use curriculum learning (monologue pre-training followed by dialogue fine-tuning) to resolve alignment issues, and use learnable speaker-turn embeddings to handle turn switching, enabling the flow matching NAR architecture to achieve both high speed and high robustness in dialogue generation.

## Method

### Overall Architecture

ZipVoice-Dialog is built upon ZipVoice (a flow matching monologue TTS model), comprising a text encoder and a vector field estimator (both using Zipformer backbones), along with a pre-trained Vocos vocoder. The input is an interleaved multi-speaker text sequence and prompt audio; the output is complete dialogue speech. Training uses a conditional flow matching (CFM) objective with a speech infilling task to enable zero-shot capability.

### Key Designs

1. **Monologue-to-Dialogue Curriculum Learning**:

    - Function: Resolves the speech-text alignment collapse that occurs when flow matching models are trained directly on dialogue data.
    - Mechanism: Two-stage training. **Stage 1**: Initialize weights from ZipVoice (pre-trained on 100k hours of monologue data) to establish a robust speech-text alignment foundation. **Stage 2**: Fine-tune on dialogue data to learn dialogue dynamics—alignment adaptation in multi-speaker contexts, correct timbre assignment, and natural turn switching.
    - Design Motivation: Training directly on dialogue data from scratch causes alignment collapse (WER spikes from ~5% to >100%), as the presence of two speaker timbres greatly increases alignment learning difficulty. Learning alignment on the simpler monologue task first and then transferring to dialogue is a classic curriculum learning strategy.

2. **Speaker-Turn Embeddings**:

    - Function: Enables the model to accurately distinguish between two speakers and assign the correct timbre to each turn.
    - Mechanism: Two randomly initialized learnable embedding vectors are introduced, corresponding to speaker identities [S1] and [S2]. For each token $y_i$ in the text sequence, the embedding $e_{speaker(i)}$ corresponding to its speaker identity is retrieved and added to the text feature: $\widetilde{y_i} = \hat{y_i} + e_{speaker(i)}$, followed by upsampling along the time dimension.
    - Design Motivation: Ablation experiments show that using a separator "|" or text tokens [S1]/[S2] both fail to effectively distinguish speakers (cpWER far exceeds WER), whereas speaker-turn embeddings reduce cpWER dramatically from 37.82/31.34 to 5.82, achieving nearly perfect turn accuracy.

3. **Interleaved Text Input & Flexible Prompting**:

    - Function: Handles the complex input format of multi-speaker conversations.
    - Mechanism: Multi-turn utterances are sorted chronologically into a single interleaved text sequence prefixed with speaker identifiers. During training, dialogue audio of random prefix length is used as the prompt; at inference, prompts of arbitrary length are supported. The model implicitly models token and turn durations via the flow matching objective without requiring an external duration predictor.
    - Design Motivation: Compared to approaches requiring predefined timestamps, end-to-end implicit modeling is simpler and does not rely on additional timestamp prediction models during training or inference.

### Loss & Training

The conditional flow matching (CFM) loss is computed only over the masked region: $L_{CFM} = \mathbb{E}_{t,q(x_1),p_0(x_0)} \| (v_t(x_t, z, (1-m) \odot x_1; \theta) - (x_1 - x_0)) \odot m \|^2$. The model is fine-tuned for 60k steps on OpenDialog plus an internal dataset (7.6k hours total), with a total batch size of 4k seconds. Inference uses the Euler solver with 16 sampling steps.

## Key Experimental Results

### Main Results

Comparison with open-source spoken dialogue generation models on Chinese and English test sets:

| Model | Params | RTF↓ | ZH WER↓ | EN WER↓ | cpSIM↑ | UTMOS↑ |
|-------|--------|------|---------|---------|--------|--------|
| Dia | 1.61B | 1.663 | - | 11.80 | 0.333 | 1.87 |
| MoonCast | 2.67B | 0.953 | 15.85 | 23.62 | 0.356 | 2.37 |
| ZipVoice-Dialog | **123M** | **0.063** | **3.17** | **3.25** | **0.437** | **3.07** |

With only 123M parameters, ZipVoice-Dialog achieves comprehensive superiority: inference speed is over 15× faster, and WER is reduced by 3–7×.

### Ablation Study

| Configuration | EN WER↓ | EN Short cpWER↓ | Notes |
|---------------|---------|-----------------|-------|
| Full model (curriculum + embeddings) | 3.25 | 3.27 | Best |
| w/o curriculum learning | 116.10 | 116.31 | Alignment collapse, completely unintelligible |
| Separator "|" instead of embeddings | 5.34 | 37.82 | Very poor turn accuracy |
| Text tokens instead of embeddings | 5.57 | 31.34 | Poor turn accuracy |
| OpenDialog data only | 3.34 | 3.53 | Data alone yields a strong baseline |

### Key Findings

- Curriculum learning is indispensable: without it the model completely fails (WER >100%), indicating that multi-speaker alignment is the core bottleneck for flow matching-based dialogue generation.
- Speaker-turn embeddings are extremely simple yet highly effective: two learnable vectors reduce the turn error rate from >30% to <1%.
- In subjective evaluation (CMOS/SMOS), ZipVoice-Dialog substantially outperforms MoonCast (CMOS gap: −1.17; SMOS: 3.86 vs. 2.35).
- The OpenDialog dataset alone is sufficient to train a model that surpasses all baselines, validating the high quality of the dataset.

## Highlights & Insights

- The **minimalist yet highly effective design philosophy** is impressive—curriculum learning plus two learnable embeddings transforms the flow matching architecture from "completely unusable" to "comprehensively outperforming AR baselines." This approach of achieving maximal gains with minimal modifications is worth emulating.
- The open-source release of the **OpenDialog dataset** is a highly valuable contribution—the first large-scale (6.8k hours) open-source spoken dialogue dataset, filling a gap in the field. The data construction pipeline (VAD → speaker diarization → ASR → LLM classification → WhisperD fine annotation → rule-based filtering) is reusable.
- A 123M-parameter model comprehensively outperforms AR models with 1.6B–2.7B parameters on all metrics, demonstrating the enormous potential of NAR architectures in dialogue scenarios.

## Limitations & Future Work

- The model and data scale are limited; a smaller model has a ceiling on expressiveness, and larger models with more data may yield further improvements.
- Subjective evaluation is limited to Chinese; subjective quality in English is not validated.
- The current system is restricted to two-speaker dialogues; while the method is extensible to multi-party conversation, this has not been verified.
- More naturalistic dialogue phenomena such as overlapping speech and backchannels have not been explored.

## Related Work & Insights

- **vs. MoonCast**: MoonCast adopts a hybrid AR+NAR architecture (LLM → flow matching → vocoder) with 2.67B parameters, but the AR component causes serious word skipping and instability (WER 23.62%). ZipVoice-Dialog is fully NAR with only 123M parameters, achieving WER 3.25% and 15× faster inference.
- **vs. Dia**: Dia is a purely AR model that directly predicts audio tokens, with 1.61B parameters but WER 11.80% and the lowest speaker similarity (cpSIM 0.333), indicating that the purely AR approach lacks sufficient robustness in dialogue scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to successfully apply flow matching to spoken dialogue generation, though the core techniques (curriculum learning, embeddings) are not novel in themselves.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies, subjective and objective evaluations, dataset comparisons, and benchmark establishment.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure; the logical chain from problem motivation to solution is fluid and easy to follow.
- Value: ⭐⭐⭐⭐⭐ Triple contributions of model, dataset, and benchmark; the OpenDialog dataset is particularly important to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](../../CVPR2026/image_generation/freqflow_frequency_aware_flow_matching.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](../../ICLR2026/image_generation/multi-agent_coordination_via_flow_matching.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](../../CVPR2026/image_generation/egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](../../CVPR2026/image_generation/vecor_--_velocity_contrastive_regularization_for_flow_matching.md)

</div>

<!-- RELATED:END -->

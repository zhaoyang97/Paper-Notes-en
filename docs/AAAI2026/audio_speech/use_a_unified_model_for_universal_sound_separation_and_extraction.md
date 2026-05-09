---
title: >-
  [Paper Note] USE: A Unified Model for Universal Sound Separation and Extraction
description: >-
  [AAAI2026][Audio & Speech][Universal sound separation] The paper proposes USE, a unified framework that employs an EDA network to infer the number of sound sources and generate acoustic cues for sound separation (SS), and a multimodal fusion network to interpret user-provided text/video/label cues for target sound extraction (TSE). Joint training with cross-task alignment enables mutual reinforcement between the two tasks, achieving +1.4 dB SDR on SS and 86% matching accuracy on TSE.
tags:
  - AAAI2026
  - "Audio & Speech"
  - Universal sound separation
  - target sound extraction
  - multimodal fusion
  - EDA network
  - cross-task alignment
date: 2026-05-08
content_hash: a6e10246f9ab3e10
---

# USE: A Unified Model for Universal Sound Separation and Extraction

**Conference**: AAAI2026
**arXiv**: [2512.21215](https://arxiv.org/abs/2512.21215)
**Code**: [https://hongyuwang414.github.io/USE-demo/](https://hongyuwang414.github.io/USE-demo/)
**Area**: Speech / Sound Separation
**Keywords**: Universal sound separation, target sound extraction, multimodal fusion, EDA network, cross-task alignment

## TL;DR
The paper proposes USE, a unified framework that employs an EDA network to infer the number of sound sources and generate acoustic cues for sound separation (SS), and a multimodal fusion network to interpret user-provided text/video/label cues for target sound extraction (TSE). Joint training with cross-task alignment enables mutual reinforcement between the two tasks, achieving +1.4 dB SDR on SS and 86% matching accuracy on TSE.

## Background & Motivation

**Background**: Sound separation (SS) decomposes mixed audio into independent sources, while target sound extraction (TSE) extracts a user-specified target from a mixture. The two tasks are typically studied independently.

**Limitations of Prior Work**: (1) SS requires the number of sources to be known in advance, and performance degrades significantly when this number is unknown; (2) TSE is constrained to single-modality cues (text or video only), and fails when cue quality is poor; (3) the absence of a unified framework prevents acoustic separation knowledge from benefiting target extraction.

**Key Challenge**: The attractors learned by SS and the query cues provided by users in TSE semantically correspond to the same targets, yet independent training of the two tasks precludes establishing this bridge.

**Goal**: Unify SS and TSE into a single framework with a shared semantic space, enabling mutual enhancement.

**Key Insight**: Use EDA network attractors as a semantic bridge aligned with user-provided cues.

**Core Idea**: EDA attractors and multimodal cues are aligned in a shared semantic space, unifying separation and extraction.

## Method

### Overall Architecture
An encoder–separator–decoder backbone is augmented with two auxiliary networks: (1) an EDA network (for SS: inferring the number of sources and generating attractors); and (2) a multimodal cue network (for TSE: fusing text/video/label cues). The two networks are bridged via a cross-task alignment loss.

### Key Designs

1. **EDA Network (Encoder-Decoder Attractor)**:

    - **Function**: Autoregressively generates sound source attractors while inferring the number of sources.
    - **Mechanism**: An LSTM encoder processes frame-level embeddings; an LSTM decoder sequentially generates attractors $\mathbf{a}_s$. Each attractor carries an existence probability $p_{\text{exi}} = \sigma(\mathbf{w}^\top \mathbf{a}_s + b)$, with a threshold of 0.5 to determine source presence.
    - **Design Motivation**: Addresses the unknown source count problem — autoregressive generation halts when the existence probability falls below the threshold.

2. **Multimodal Cue Network**:

    - **Function**: Fuses text (DistilBERT), video (Swin Transformer), and sound labels (one-hot embeddings).
    - **Mechanism**: Each modality is encoded and concatenated along the time dimension; multi-head attention (with separator features as Query) is applied for fusion.
    - **Design Motivation**: Multimodal redundancy — missing or low-quality cues in one modality are compensated by others.

3. **Cross-Task Alignment Loss**:

    - **Function**: Maps EDA attractors and user cues into a shared semantic space.
    - **Mechanism**: $\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{InfoNCE}}$, aligning attractors and cues using the optimal permutation determined by PIT.
    - **Design Motivation**: Aligns the acoustic representations learned by SS with the semantic queries of TSE, enabling unification.

### Loss & Training
Two-stage training: Stage 1 trains SS + EDA only (70 epochs, lr=1e-4); Stage 2 jointly trains SS + TSE, randomly selecting the EDA or cue network with a 30:70 ratio (30 epochs, lr=3e-5).

## Key Experimental Results

### Main Results

| Task / Model | 2Mix SI-SNRi↑ | 3Mix SI-SNRi↑ |
|------------|--------------|--------------|
| Libri2Mix: TDANet | 17.5 | - |
| **Libri2/3Mix: USE-B** | **17.8** | **15.0** |
| AudioSet SS: Sepformer | 7.4 | - |
| **AudioSet SS: USE-S (stage2)** | **8.8** | **7.2** |
| FUSS: TDCN++ | 11.2/11.6/7.4 | |
| **FUSS: USE-B** | **12.8/13.1/11.9** | |

### TSE Multimodal Comparison

| Cue Combination | DCCRN SNRi | USE-B SNRi |
|---------|-----------|-----------|
| tag+text+video | 6.9 | **8.9** (+29%) |
| text only | 6.3 | **8.0** (+27%) |
| video only | 5.8 | **6.2** (+7%) |

### Key Findings
- Joint training (Stage 2) further improves SS over single-task training (Stage 1) (AudioSet unseen 3Mix: 5.2→6.3 dB), demonstrating that semantic knowledge from TSE benefits SS.
- Attractor–cue matching accuracy reaches 86% (2Mix), validating the effectiveness of the shared semantic space.
- Under unknown source counts, USE-B* suffers negligible performance loss (17.7 vs. 17.8), with EDA source counting accuracy exceeding 80%.
- Multimodal cue redundancy is evident — tag+text performs comparably to tag+text+video (8.6 vs. 8.9), indicating limited contribution from video.

## Highlights & Insights
- **Attractor–cue alignment is the core innovation** — it establishes a semantic bridge between two seemingly disparate tasks. This paradigm is transferable to any scenario where "automatically discovered" and "user-specified" representations coexist.
- **Bidirectional reinforcement through joint training**: SS improves TSE's separation capability, while TSE endows SS with semantic awareness — forming a positive feedback loop.

## Limitations & Future Work
- EDA source counting accuracy degrades in 3Mix and beyond (65.3%), limiting applicability in complex scenarios.
- The contribution of video cues is limited, potentially requiring stronger video encoders or better video–audio temporal alignment.
- Evaluation is restricted to general sounds in AudioSet-style settings; generalization to music separation remains unverified.

## Related Work & Insights
- **vs. DCCRN**: A conventional TSE method; USE-B outperforms it by 29% across all cue combinations.
- **vs. TDANet**: An SS-only method; USE-B matches its performance on 2Mix (17.8 vs. 17.5) while additionally supporting 3Mix and TSE.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The unified SS+TSE framework with attractor–cue semantic alignment is a novel and effective design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple datasets, SS/TSE/multimodal evaluations, source counting, and matching accuracy.
- **Writing Quality**: ⭐⭐⭐⭐ Clear architecture description with detailed training strategy.
- **Value**: ⭐⭐⭐⭐ A practical unified framework for general-purpose sound processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[AAAI 2026\] DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization](deformtrace_a_deformable_state_space_model_with_relay_tokens_for_temporal_forger.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ICLR 2026\] MAPSS: Manifold-Based Assessment of Perceptual Source Separation](../../ICLR2026/audio_speech/mapss_manifold-based_assessment_of_perceptual_source_separation.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)

</div>

<!-- RELATED:END -->

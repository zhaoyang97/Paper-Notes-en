---
title: >-
  [Paper Note] Improving Language and Modality Transfer in Translation by Character-level Modeling
description: >-
  [ACL 2025][Audio & Speech][character-level] A cross-lingual and cross-modal translation method is proposed based on a character-level encoder, charSONAR. A character-level text encoder is obtained via teacher-student training and is then connected to a 1000+-language CTC ASR model (MMS) using lightweight adapters. It achieves SOTA on text translation in 75 languages and speech translation in 33 languages, with particularly prominent performance in zero-resource and low-resour…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "character-level"
  - "multilingual translation"
  - "speech translation"
  - "SONAR"
  - "cross-modal transfer"
date: 2026-05-08
content_hash: b943d77f645f29b1
---

# Improving Language and Modality Transfer in Translation by Character-level Modeling

**Conference**: ACL 2025  
**arXiv**: [2505.24561](https://arxiv.org/abs/2505.24561)  
**Code**: None  
**Area**: Other  
**Keywords**: character-level, multilingual translation, speech translation, SONAR, cross-modal transfer

## TL;DR
A cross-lingual and cross-modal translation method is proposed based on a character-level encoder, charSONAR. A character-level text encoder is obtained via teacher-student training and is then connected to a 1000+-language CTC ASR model (MMS) using lightweight adapters. It achieves SOTA on text translation in 75 languages and speech translation in 33 languages, with particularly prominent performance in zero-resource and low-resource scenarios.

## Background & Motivation

**Background**: Translation models currently support 200–400 text languages and 100 speech languages, covering only 5% of all languages globally. Scaling to long-tail, low-resource languages faces data scarcity challenges.

**Limitations of Prior Work**: (1) The cross-lingual transfer capability of subword tokenization is limited. (2) In speech translation, the mismatch in length/content between the CTC output (character-level) and the text encoder (subword-level) causes a modality gap. (3) Phonemization methods are ambiguous and cannot scale to 1000+ languages.

**Key Challenge**: How to maximize knowledge transfer between text and speech modalities, as well as between high-resource and low-resource languages?

**Goal**: Unify the representation input space of text and speech via character-level modeling to enhance cross-lingual and cross-modal transfer.

**Key Insight**: Based on SONAR (a multilingual fixed-dimensional embedding space) and MMS (a 1000+-language CTC ASR model), the character-level encoder naturally aligns with the CTC output, eliminating the length mismatch between subwords and characters.

**Core Idea**: Character-level SONAR encoder + pretrained CTC-to-character adapter = data-efficient cross-lingual and cross-modal translation.

## Method

### Overall Architecture
(1) Teacher-Student: SONAR (subword) -> charSONAR (character), using interpolation MSE loss. (2) Adapter: MMS-CTC -> charSONAR, using MSE loss. During inference, translation is generated from embeddings using the SONAR decoder.

### Key Designs

1. **charSONAR Training**:

    - Retain only single-character tokens in the SONAR vocabulary (256K -> 8K).
    - Three types of MSE targets: reconstruction ($\text{MSE}(\mathbf{c}^x, \mathbf{e}^x)$), translation ($\text{MSE}(\mathbf{c}^x, \mathbf{e}^y)$), and **interpolation** ($\text{MSE}(\mathbf{c}^x, \frac{\mathbf{e}^x + \mathbf{e}^y}{2})$).
    - The interpolation target performs best: the average SONAR embeddings of a language pair are more suitable for the cross-lingual space than monolingual embeddings.
    - Incorporate ASR-style augmentations (uncasing, punctuation removal, and character noise) to enhance cross-modal robustness.

2. **Cross-modal Adapter**:

    - **Pretrained Adapter**: Leverage the CTC classification layer of MMS and the embedding layer of charSONAR to perform soft prediction (softmax -> embedding lookup), requiring only ~200K parameters.
    - **Dual Adapter**: A combination of the pretrained adapter and a randomly initialized adapter using gated weighting, totaling 2.5M parameters.
    - CTC Compression: Collapse identical consecutive predictions and remove blank tokens to compress audio representations to character-level lengths.

3. **Zero-Resource Speech Translation**:

    - Freeze both charSONAR and MMS, training only the adapter.
    - Training requires only ASR data (audio-transcript pairs) and does not need parallel speech translation data.

## Key Experimental Results

### Main Results

| Method | Text Translation (FLORES+ 75 languages) | Speech Translation (FLEURS 33 languages) |
|------|:---:|:---:|
| SONAR (subword) | xCOMET: 0.925 | - |
| **charSONAR** | **xCOMET: 0.934** | - |
| SEAMLESS (supervised) | - | SOTA baseline |
| Whisper cascade | - | Strong baseline |
| **charSONAR + MMS** | **Better than SONAR** | **New SOTA** |

### Ablation Study

| Configuration | xCOMET | xSIM++ | Description |
|------|:---:|:---:|------|
| Reconstruction target | 0.929 | 7.4 | Base |
| Translation target | 0.924 | 6.6 | Good retrieval but slightly worse translation |
| **Interpolation target** | **0.931** | **6.6** | Balanced |
| + Pretraining initialization | 0.934 | 6.4 | Faster convergence |
| Gain on low-resource languages | Highest | - | Character-level shows clear advantages in low-resource scenarios |
| Generalization to zero-resource languages | Better than subword | Better than subword | Character sharing enhances transfer |

### Key Findings
- **Character-level outperforms subword-level**: It performs better overall across 75 languages, with particularly prominent advantages in low-resource and zero-resource scenarios.
- **Interpolated embedding space is superior**: The "average" SONAR embedding of a language pair serves as a better cross-lingual anchor than monolingual embeddings, benefiting low-resource languages the most (whose quality improves after averaging with high-resource languages).
- **An extremely lightweight adapter achieves SOTA speech translation**: An adapter with only 2.5M parameters outperforms the fully supervised SEAMLESS system.

## Highlights & Insights
- **Deep insight into unifying the two modalities via character-level modeling**: Since the CTC output is naturally character-level, using character inputs for the text encoder eliminates the modality gap. This is more elegant and simpler than prior phoneme-sharing or subword-compression schemes.
- **Unexpected advantage of the fixed-dimensional embedding bottleneck**: SONAR's mean pooling yields fixed-dimensional embeddings, preventing the increased sequence length of character-level inputs from impacting decoder computation.
- **"Pretrained initialization" design of the adapter**: Initializing with the existing MMS CTC layer and charSONAR embedding layer maximizes the utilization of pretrained knowledge.

## Limitations & Future Work
- Character-level modeling at the encoder increases the sequence length (1.5–3x), leading to higher encoding costs.
- Only X -> Eng speech translation directions were tested; other directions remain unverified.
- Character-level tokenization may behave differently for logographic languages like Chinese or Japanese, where each character carries denser semantic meaning.

## Related Work & Insights
- **vs ZeroSwot**: ZeroSwot aligns speech and text representations using Wasserstein distance; charSONAR achieves superior performance using simple MSE loss combined with character-level unification.
- **vs ByT5 (character-level LM)**: ByT5 demonstrates the robustness benefits of character-level modeling against noise; charSONAR extends this to translation and speech.

## Rating
- Novelty: ⭐⭐⭐⭐ The scheme of unifying text and speech translation at the character level is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 75-language text + 33-language speech along with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology and comprehensive experiments.
- Value: ⭐⭐⭐⭐⭐ High practical impact, with the potential to support speech translation for 1000+ languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)
- [\[ACL 2025\] Different Speech Translation Models Encode and Translate Speaker Gender Differently](different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[ACL 2025\] Towards Reliable Large Audio Language Model](towards_reliable_large_audio_language_model.md)
- [\[ACL 2025\] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)

</div>

<!-- RELATED:END -->

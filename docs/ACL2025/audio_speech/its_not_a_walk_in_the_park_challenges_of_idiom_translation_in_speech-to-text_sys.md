---
title: >-
  [Paper Note] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems
description: >-
  [ACL 2025][Audio & Speech][Idiom translation] This paper presents the first systematic comparison of the performance of speech-to-text translation (SLT), text machine translation (MT), and large language models (LLMs) on idiom translation tasks. It reveals that the performance of SLT systems deteriorates significantly when handling idioms, tending towards literal translation even in the higher encoder layers, whereas MT and LLMs demonstrate clearly superior capabilities in id…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Idiom translation"
  - "speech-to-text translation"
  - "DecoderLens"
  - "cascade systems"
  - "figurative language"
date: 2026-05-08
content_hash: 3f32951cb3b953a2
---

# It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems

**Conference**: ACL 2025  
**arXiv**: [2506.02995](https://arxiv.org/abs/2506.02995)  
**Code**: [Yes](https://github.com/IuliiaZaitova/idiom_s2t)  
**Area**: Others  
**Keywords**: Idiom translation, speech-to-text translation, DecoderLens, cascade systems, figurative language

## TL;DR

This paper presents the first systematic comparison of the performance of speech-to-text translation (SLT), text machine translation (MT), and large language models (LLMs) on idiom translation tasks. It reveals that the performance of SLT systems deteriorates significantly when handling idioms, tending towards literal translation even in the higher encoder layers, whereas MT and LLMs demonstrate clearly superior capabilities in idiom processing.

## Background & Motivation

Idioms are conventional expressions whose meanings cannot be derived from the literal meanings of their constituent words, such as "raining cats and dogs" in English or "Es ist mir wurst" in German (literally "it is sausage to me", meaning "I don't care"). Although modern machine translation systems have achieved substantial progress, idiom translation remains a major challenge.

While extensive work has investigated idiom translation in text MT, research on **idiom translation in speech translation (SLT)** is extremely scarce. SLT systems face additional complexities, as they must simultaneously integrate acoustic, syntactic, and semantic information. Understanding why and how these systems fail on idioms is crucial for further improving SLT systems.

The core research questions of this paper include:
1. How do end-to-end SLT systems perform relative to text MT, LLMs, and cascade systems in idiom translation?
2. How do these systems process idiom data compared to conventional news data?
3. Can a layer-by-layer analysis via DecoderLens locate at which layer the system "loses" the figurative meaning of the idiom?

## Method

### Overall Architecture

This paper does not propose a new model, but rather designs a **systematic evaluation framework** to compare the translation quality of idioms and news data across various types of translation systems on two language pairs: German-to-English and Russian-to-English.

### Key Designs

1. **Comprehensive Coverage of Evaluated Systems**:

    - **MT Systems**: SeamlessM4T (text-to-text), NLLB (200-language translation model)
    - **LLMs**: LLaMA 3 (language-specialized fine-tuned version), DeepSeek-V3
    - **SLT Systems**: SeamlessM4T (speech-to-text), Whisper Large v3
    - **Cascade Systems**: ASR (Whisper or SeamlessM4T) → MT/LLM
    - Design Motivation: Covering both end-to-end and cascade paradigms to comprehensively diagnose performance bottlenecks.

2. **Evaluation Dataset Construction**:

    - **News Data**: 250 sentences per language pair randomly selected from the News Commentary parallel corpus as a baseline.
    - **Idiom Data**: 250 idioms requiring non-literal translation manually filtered from the Idioms-InContext-MT dataset.
    - Idioms whose figurative meanings can be preserved through literal translation (e.g., "break someone's heart") are excluded.
    - Microsoft Edge TTS is used to synthesize speech to ensure consistent acoustic conditions.

3. **DecoderLens Layer-by-Layer Analysis**:

    - Replacing the final encoder outputs with activations from intermediate encoder layers to observe translation results at each layer.
    - Manual annotation of 50 samples is conducted to analyze changes in translation quality across layers.
    - Design Motivation: Revealing mechanistic differences in figurative language processing between SLT and MT systems during the encoding process.

4. **Manual Annotation Scheme**:

    - A scheme of 7 annotation categories is designed, ranging from "Correct (idiomatic expression)" to "Empty output".
    - Specifically distinguishing categories unique to idioms: correct idiomatic translation, paraphrased translation, and literal translation.
    - Two annotators resolved discrepancies through discussion.

### Evaluation Metrics

- **Automatic Metrics**: COMET (highly correlated with human judgments on semantic equivalence and fluency).
- **Manual Annotation**: 50 sentences are randomly sampled from each language-dataset-model combination for annotation.
- Mann-Whitney U test with Bonferroni correction is used to verify statistical significance.

## Key Experimental Results

### Main Results (COMET Score Comparison)

| System | De→En News | De→En Idiom | Ru→En News | Ru→En Idiom |
|------|-----------|-----------|-----------|-----------|
| Whisper (Direct SLT) | 0.8437 | 0.6402 | 0.8318 | 0.6916 |
| Seamless (Direct SLT) | 0.8697 | 0.6483 | 0.8512 | 0.6941 |
| NLLB (Text MT) | 0.8841 | 0.6749 | 0.8664 | 0.7214 |
| Seamless (Text MT) | 0.8870 | 0.6784 | 0.8694 | 0.7262 |
| LLaMA | 0.8724 | 0.6971 | 0.8211 | 0.7354 |
| **DeepSeek** | **0.8940** | **0.7675** | **0.8741** | **0.7939** |
| Whisper(ASR)→DeepSeek | 0.8887 | 0.7584 | 0.8607 | 0.7873 |

### Ablation Study / Comparative Analysis

| Dimension | Finding |
|---------|------|
| SLT vs MT (Idiom) | SLT performance on idioms drops by 24.2% COMET (Whisper De→En), while MT drops by approximately 14% |
| German vs Russian | German shows a larger idiom-news gap (avg. 0.198) compared to Russian (approx. 0.143) |
| Cascade vs End-to-end SLT | Cascade systems mostly outperform end-to-end SLT, indicating that the issue does not solely stem from ASR errors |
| DecoderLens Layer Analysis | SLT outputs empty/meaningless content in layers 0-20, exhibits hallucinations in layers 21-30, and only produces relevant translations in the final few layers |

### Key Findings

1. **DeepSeek leads overall**: Achieving the best performance across all configurations (text-only, cascade), with an idiom translation COMET score of approximately 0.76-0.79.
2. **Precipitous drop in SLT idiom performance**: Whisper's COMET score on De→En idioms is only 0.64, a 24% decline compared to news.
3. **Cascade systems outperform end-to-end SLT**: Suggesting that the issue in SLT is not just ASR transcription errors, but a deeper difficulty with acoustic-semantic integration.
4. **Layer-by-layer analysis reveals structural differences**: The SLT system only starts to generate meaningful translations in the higher encoder layers and still reverts to literal translation in the final layers, whereas the MT system shows a smoother transition.

## Highlights & Insights

- **First systematic study of idiom translation in SLT**, filling a research gap in processing figurative language within the speech translation community.
- **Innovative application of DecoderLens**: Revealing internal representation differences between SLT and MT when processing figurative language through layer-by-layer analysis.
- **Practical Recommendation**: It is recommended to use cascade systems instead of end-to-end SLT when the translated content is likely to contain idioms.
- Well-designed manual annotation scheme, particularly the distinction among idiom translation categories (idiomatic, paraphrase, literal), which offers high reference value.

## Limitations & Future Work

1. Only covers two language pairs (De→En and Ru→En), though idiom usage varies greatly across different languages.
2. Uses synthetic speech instead of natural, real-world speech, although prior studies show that this has minimal impact.
3. DecoderLens is only applicable to encoder-decoder architectures and cannot analyze decoder-only models (e.g., LLaMA).
4. Subjectivity in manual annotation with a small sample size (50 sentences per combination).
5. Does not explore idiom enhancement strategies for SLT, such as idiom-aware fine-tuning or data augmentation.

## Related Work & Insights

- Complements the studies on figurative language in Transformers by Dankers et al. (2022) and Baziotis et al. (2023).
- The DecoderLens (Langedijk et al., 2024) method can be generalized to analyze other semantically complex scenarios.
- The advantages of cascade systems in new scenarios merit further research, especially combining strong ASR with powerful LLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study of idiom translation in SLT; the application of DecoderLens is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-system, multi-language, automatic + manual evaluation, and in-depth layer-by-layer analysis. However, more language pairs and larger sample sizes could be included.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich tables/figures, and well-motivated research questions.
- **Value**: ⭐⭐⭐⭐ — Holds significant practical guidance for the speech translation community, revealing fundamental deficiencies of SLT in semantic interpretation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](../../ICML2025/audio_speech/sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee.md)
- [\[ACL 2025\] Different Speech Translation Models Encode and Translate Speaker Gender Differently](different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)
- [\[ICML 2025\] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](../../ICML2025/audio_speech/do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)
- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)
- [\[ICLR 2026\] TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems](../../ICLR2026/audio_speech/ttsds2_resources_and_benchmark_for_evaluating_human-quality_text_to_speech_syste.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation
description: >-
  [ACL2026][Audio & Speech][ASR evaluation] This paper proposes Script-Normalized WER (SN-WER), a training-free evaluation method that decouples script mismatch errors from actual recognition errors in multi-script ASR eva…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "ASR evaluation"
  - "multi-script systems"
  - "WER"
  - "transliteration normalization"
  - "Indic languages"
date: 2026-05-08
content_hash: ab45d14f35a02147
---

# SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation

**Conference**: ACL2026
**arXiv**: [2606.02548](https://arxiv.org/abs/2606.02548)
**Code**: To be confirmed
**Area**: audio_speech
**Keywords**: ASR evaluation, multi-script systems, WER, transliteration normalization, Indic languages

## TL;DR

This paper proposes Script-Normalized WER (SN-WER), a training-free evaluation method that decouples script mismatch errors from actual recognition errors in multi-script ASR evaluation by transliterating both reference and hypothesis texts into a unified canonical script before calculating WER.

## Background & Motivation

WER is the mainstream metric for ASR evaluation. However, in multilingual scenarios, when reference text uses native scripts (e.g., Devanagari, Bengali, Tamil) while the model outputs romanized text, WER misidentifies script differences as word-level errors, leading to a systematic overestimation of error rates. For instance, Whisper achieves a WER of 1.13 on Odia, which drops to 1.02 after script normalization, indicating a significant script mismatch component. Existing methods like toWER target code-mixing scenarios and lack systematic evaluation for single-language multi-script systems, particularly the five major Indic script systems. The authors argue for a companion metric at the evaluation level to quantify the contribution of script mismatch to WER without modifying model training or decoding.

## Method

### Overall Architecture

The core idea of SN-WER is to map both reference and hypothesis sequences to a language-specific canonical script $C$ (usually the native script of the benchmark dataset) before calculating WER. It is defined as: $SN-WER(R, H) = WER(T(R), T(H))$, where $T(\cdot)$ is a deterministic, word-boundary-preserving transliteration mapping. Under conditions where mapping is deterministic and collision-free, $SN-WER \leq WER$ (conditional conservatism), and $SN-WER \approx WER$ when the reference and hypothesis use the same script (identity).

### Key Designs

1.  **Canonical Script Selection**: The reference script of the benchmark dataset is used as the canonical script $C$, making SN-WER directly comparable to the original baseline. Experiments verify that replacing it with other canonical scripts like Devanagari results in a difference $\Delta < 0.005$.
2.  **Romanization Detection and Transliteration**: Romanized tokens are detected using Unicode block heuristics and then mapped to the native script via transliteration libraries such as IAST, ITRANS, or ICU. The difference between these three transliterators is $\Delta < 0.002$, with a collision rate $< 0.1\%$.
3.  **Comparative Experimental Design**: Seven sets of experiments were designed to verify four hypotheses—H1 (script mismatch reduction), H2 (noise robustness), H3 (stability), and H4 (cross-script validation). These include orthogonal stress tests (manual injection of romanization) and lexical sensitivity controls (injection of lexical substitutions).

### Loss & Training

SN-WER involves no training and is a pure evaluation method. The computational complexity is identical to standard WER ($O(nm)$), with only a transliteration preprocessing step added before scoring.

## Key Experimental Results

### Main Results

Evaluations were conducted on FLEURS and Common Voice across 5 Indic languages using 3 ASR models:

| Dataset | Model | WER | SN-WER | Rel. Gain (%) |
| :--- | :--- | :--- | :--- | :--- |
| FLEURS | MMS | 0.32 | 0.30 | -5.4 |
| FLEURS | Whisper-large | 0.70 | 0.64 | -8.0 |
| FLEURS | Whisper-small | 1.27 | 1.21 | -4.7 |
| CommonVoice | MMS | 0.46 | 0.36 | -23.0 |
| CommonVoice | Whisper-large | 0.86 | 0.82 | -4.3 |
| CommonVoice | Whisper-small | 1.46 | 1.36 | -6.9 |

Cross-script extensions (Arabic and Urdu) also showed improvements of 5-9%.

### Ablation Study

| Experiment | Key Conclusion |
| :--- | :--- |
| Transliterator Invariance (E3) | Differences among IAST/ITRANS/ICU mappings are $\Delta < 0.002$ |
| Collision Rate (E3) | Average collision rate is 0.03%, $< 0.1\%$ |
| Normalization Robustness (E3) | Ablation of digits/punctuation yields $\Delta < 0.05$ |
| Orthogonal Stress Test (E5) | With 0→50% romanization injection, SN-WER attenuates 67% of script-induced inflation |
| Lexical Sensitivity (E6) | With 20-30% lexical substitution, $\Delta_{SN}/\Delta_{WER} \approx 1.09$, proving lexical errors are not attenuated |
| Adversarial Validation (E7) | SN-WER $\approx 1.0$ after word shuffling/lexical substitution, confirming it does not mask semantic errors |

### Key Findings

-   SN-WER narrows model gaps by up to 12% on the clean FLEURS dataset (Gujarati) and by 26% on the noisy Common Voice (Odia), while retaining genuine recognition weaknesses.
-   Ranking stability is perfectly maintained (Kendall $\tau = 1.0$), with only the magnitude of gaps between models changing.
-   The romanization rate is highly correlated with the magnitude of SN-WER correction ($r = 0.81$).

## Highlights & Insights

-   **Evaluation Companion rather than Replacement**: SN-WER is explicitly positioned as a companion metric to WER/CER. It is suitable for scenarios where script choice is irrelevant to downstream tasks (search, indexing, retrieval, multilingual LLM pipelines), rather than user-facing transcription tasks.
-   **Extremely Low Adoption Barrier**: Requires no training, no additional data, and no modification to decoding; it only adds transliteration preprocessing during the scoring phase.
-   **Systematic Validation Methodology**: Seven sets of experiments covering identity, conservatism, lexical sensitivity, robustness, and adversarial analysis form a complete metric validation paradigm.

## Limitations & Future Work

-   Validations were limited to 5 Indic languages plus Arabic/Urdu; applicability to other multi-script languages (e.g., CJK) remains to be verified.
-   While the collision rate of transliteration mapping is extremely low ($< 0.1\%$), it may rise for languages with large vocabularies.
-   For morphologically complex languages (e.g., agglutinative affixes in Tamil), transliteration might introduce boundary ambiguities.
-   Future work could extend this to a script-normalized version of CER (SN-CER).

## Related Work & Insights

-   **toWER** (Emond et al., 2018): Transliterated WER for code-mixed Indic ASR, but requires modifying training corpora and targets bilingual code-mixing rather than monolingual multi-script scenarios.
-   **WERd** (Ali et al., 2017): Targeted at spelling variants in Arabic dialects.
-   **Lenient CER** (Karita et al., 2023): Targeted at character-level inconsistencies in Japanese.
-   Insight: The normalization approach for evaluation metrics can be generalized to other scenarios featuring surface form differences with semantic equivalence.

## Rating

| Dimension | Score (1-10) |
| :--- | :--- |
| Novelty | 5 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Value | 7 |
| **Total Score** | **7.3** |

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](../../AAAI2026/audio_speech/hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](multimodal_in-context_learning_for_asr_of_low-resource_languages.md)

</div>

<!-- RELATED:END -->

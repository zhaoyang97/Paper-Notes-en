---
title: >-
  [Paper Note] SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation
description: >-
  [ACL 2026][Audio & Speech][WER] Proposes Script-Normalized WER (SN-WER), a training-free evaluation method that calculates WER after transliterating both reference and hypothesis texts into a uniform canonical script, decoupling script mismatch errors from actual recognition errors in multi-script ASR evaluation.
tags:
  - ACL 2026
  - Audio & Speech
  - WER
date: 2026-05-08
content_hash: 3ea7ff430fa132b8
---
# SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation

**Conference**: ACL2026
**arXiv**: [2606.02548](https://arxiv.org/abs/2606.02548)  
**Code**: TBD  
**Area**: Audio & Speech
**Keywords**: ASR Evaluation, Multi-Script Systems, WER, Transliteration Normalization, Indic Languages

## TL;DR

Proposes Script-Normalized WER (SN-WER), a training-free evaluation method that calculates WER after transliterating both reference and hypothesis texts into a uniform canonical script, decoupling script mismatch errors from actual recognition errors in multi-script ASR evaluation.

## Background & Motivation

WER is the dominant metric for ASR evaluation. However, in multilingual scenarios, when the reference text uses local scripts (e.g., Devanagari, Bengali, Tamil) while the model outputs Romanized text, WER misinterprets script differences as word-level errors, leading to systemic overestimation of error rates. For example, Whisper achieves WER=1.13 on Odia, which drops to 1.02 after script normalization, indicating significant script mismatch components. Existing methods like toWER focus on code-switching scenarios and lack systemic evaluation for mono-lingual multi-script systems, especially for the 5 major Indic scripts. The authors argue for an evaluation-level companion metric to quantify the contribution of script mismatch to WER without modifying model training or decoding.

## Method

### Overall Architecture

The core idea of SN-WER is to map both reference and hypothesis sequences to a language-specific canonical script $C$ (usually the native script of the benchmark dataset) before calculating WER. It is defined as: $SN-WER(R,H) = WER(T(R), T(H))$, where $T(\cdot)$ is a deterministic, word-boundary-preserving transliteration mapping. Under conditions where the mapping is deterministic and collision-free, SN-WER satisfies conditional conservatism ($SN-WER \le WER$) and identity ($SN-WER \approx WER$ when scripts match).

### Key Designs

**1. Script Normalization Definition and Three Diagnostic Properties**: SN-WER formalizes "transliterate first, then calculate WER" as $SN-WER(R,H) = WER(T(R), T(H))$. Its ability to "separate" errors relies on three provable diagnostic properties: Identity (SN-WER $\approx$ WER when scripts match, ensuring no artificial score inflation), Conditional Conservatism (when $T$ preserves boundaries and avoids collisions, pure script differences are eliminated without increasing edit distance, hence $SN-WER \le WER$), and Lexical Sensitivity (actual recognition errors like substitution, deletion, insertion, or word order swaps are not obscured by transliteration). The first two ensure the metric "reduces script inflation without reducing real errors," while the third is verified via lexical replacement and adversarial experiments.

**2. Canonical Script Selection**: The target script $C$ for $T(\cdot)$ is set to the native reference script of the benchmark dataset (e.g., Devanagari, Bengali, Tamil for Indic languages). This makes SN-WER directly comparable to the original benchmark while allowing Romanized hypothesis tokens to be scored based on lexical content rather than surface script. Changing $C$ to other canonical scripts like uniform Devanagari results in a difference of $\Delta < 0.005$, indicating low sensitivity to the specific choice of canonical script.

**3. Romanization Detection and Deterministic Transliteration**: Unicode block heuristics identify Romanized tokens, which are then mapped back to the native script using transliteration libraries such as IAST, ITRANS, or ICU, combined with standard Unicode, punctuation, and numeric normalization. This step requires transliteration to be deterministic, boundary-preserving, and nearly collision-free. Experiments show differences between libraries are $\Delta < 0.002$ with an average collision rate of $0.03\%$ ($< 0.1\%$), establishing the empirical validity of "conditional conservatism."

### Loss & Training

SN-WER involves no training and is a pure evaluation method. The computational complexity matches standard WER ($O(nm)$), adding only a transliteration preprocessing step before scoring.

## Key Experimental Results

### Main Results

Evaluation of 5 Indic languages and 3 ASR models on FLEURS and Common Voice:

| Dataset | Model | WER | SN-WER | Relative Gain (%) |
|--------|------|-----|--------|----------|
| FLEURS | MMS | 0.32 | 0.30 | -5.4 |
| FLEURS | Whisper-large | 0.70 | 0.64 | -8.0 |
| FLEURS | Whisper-small | 1.27 | 1.21 | -4.7 |
| CommonVoice | MMS | 0.46 | 0.36 | -23.0 |
| CommonVoice | Whisper-large | 0.86 | 0.82 | -4.3 |
| CommonVoice | Whisper-small | 1.46 | 1.36 | -6.9 |

Cross-script expansion (Arabic and Urdu) also showed 5-9% improvements.

### Ablation Study

| Experiment | Key Conclusion |
|------|----------|
| Transliterator Invariance (E3) | Differences between IAST/ITRANS/ICU mappings are $\Delta < 0.002$ |
| Collision Rate (E3) | Average collision rate is 0.03%, $< 0.1\%$ |
| Normalization Robustness (E3) | Numeric/punctuation ablation $\Delta < 0.05$ |
| Orthogonal Stress Test (E5) | With 0→50% Romanization injection, SN-WER attenuates 67% of script inflation |
| Lexical Sensitivity (E6) | With 20-30% lexical substitution, $\Delta_{SN}/\Delta_{WER} \approx 1.09$, proving lexical errors are not attenuated |
| Adversarial Validation (E7) | SN-WER $\approx 1.0$ after word shuffling/substitution, confirming semantic errors are not masked |

### Key Findings

- SN-WER reduces model gaps by up to 12% (Gujarati) on clean FLEURS and 26% (Odia) on noisy Common Voice, while retaining true recognition weaknesses.
- Ranking stability is perfectly maintained (Kendall $\tau=1.0$), only affecting the magnitude of gaps between models.
- Romanization rate is highly correlated with the magnitude of SN-WER correction ($r=0.81$).

## Highlights & Insights

- **Evaluation Companion, Not Replacement**: SN-WER is explicitly positioned as a companion metric to WER/CER, suitable for scenarios where script choice is independent of downstream tasks (search, indexing, retrieval, multilingual LLM pipelines), rather than user-facing transcription tasks.
- **Low Barrier to Adoption**: Requires no training, no additional data, and no modification to decoding; transliteration preprocessing is simply added during the scoring phase.
- **Systematic Verification Methodology**: Seven sets of experiments covering identity, conservatism, lexical sensitivity, robustness, and adversariality form a complete metric validation paradigm.

## Limitations & Future Work

- Validated only on 5 Indic languages plus Arabic/Urdu; applicability to other multi-script languages (e.g., CJK) remains to be verified.
- While the collision rate is extremely low ($< 0.1\%$), it may increase for languages with massive vocabularies.
- For morphologically complex languages (e.g., agglutinative suffixes in Tamil), transliteration might introduce boundary ambiguities.
- Future work could extend this to a script-normalized version of CER (SN-CER).

## Related Work & Insights

- **toWER** (Emond et al., 2018): Transliterated WER for code-mixed Indic ASR, but requires training corpus modification and targets bilingual code-switching rather than mono-lingual multi-script scenarios.
- **WERd** (Ali et al., 2017): WER for orthographic variants in Arabic dialects.
- **Lenient CER** (Karita et al., 2023): Targets character-level inconsistencies in Japanese.
- Insight: The normalization approach for evaluation metrics can be generalized to other scenarios with surface form differences but semantic equivalence.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 5 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Value | 7 |
| Total Score | 7.3 |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](../../AAAI2026/audio_speech/hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](multimodal_in-context_learning_for_asr_of_low-resource_languages.md)

</div>

<!-- RELATED:END -->

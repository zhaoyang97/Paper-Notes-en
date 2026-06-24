---
title: >-
  [Paper Note] SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation
description: >-
  [ACL2026][Audio & Speech][ASR Evaluation] Proposal of Script-Normalized WER (SN-WER), a training-free evaluation method that decouples script mismatch errors from true recognition errors in multi-script ASR evaluation by transliterating both reference and hypothesis texts into a uniform canonical script before calculating WER.
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "ASR Evaluation"
  - "Multi-script systems"
  - "WER"
  - "Transliteration normalization"
  - "Indic languages"
date: 2026-05-08
content_hash: 1e666464bfba86c2
---

# SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation

**Conference**: ACL2026  
**arXiv**: [2606.02548](https://arxiv.org/abs/2606.02548)  
**Code**: To be confirmed  
**Area**: Audio and Speech  
**Keywords**: ASR Evaluation, Multi-script systems, WER, Transliteration normalization, Indic languages

## TL;DR

Proposal of Script-Normalized WER (SN-WER), a training-free evaluation method that decouples script mismatch errors from true recognition errors in multi-script ASR evaluation by transliterating both reference and hypothesis texts into a uniform canonical script before calculating WER.

## Background & Motivation

WER is the mainstream metric for ASR evaluation. However, in multilingual scenarios, when the reference text uses a native script (e.g., Devanagari, Bengali, Tamil) while the model outputs Romanized text, WER misinterprets script differences as word-level errors, leading to systematically overestimated error rates. For instance, Whisper achieves a WER of 1.13 on Odia, which drops to 1.02 after script normalization, indicating a significant script mismatch component. Existing methods like toWER target code-switching scenarios and lack systemic evaluation for mono-lingual multi-script systems, particularly the five Indic script systems. The authors argue for an evaluation-level companion metric to quantify the contribution of script mismatch to WER without modifying model training or decoding.

## Method

### Overall Architecture

The core concept of SN-WER is to map both the reference and hypothesis sequences to a language-specific canonical script $C$ (typically the native script of the benchmark dataset) before calculating WER. It is defined as: $SN-WER(R,H) = WER(T(R), T(H))$, where $T(\cdot)$ is a deterministic, word-boundary-preserving transliteration mapping. Under the conditions of determinism and no collisions, $SN-WER \leq WER$ (conditional conservativeness), and $SN-WER \approx WER$ when the reference and hypothesis share the same script (identity).

### Key Designs

**1. Script Normalization Definition and Three Diagnostic Properties**: SN-WER formalizes "transliterate first, then calculate WER" as $SN-WER(R,H) = WER(T(R), T(H))$. Its ability to "separate" errors relies on three provable diagnostic properties: Identity (SN-WER $\approx$ WER when scripts match, ensuring scores are not artificially lowered), Conditional Conservativeness (pure script differences are eliminated without increasing edit distance if $T$ preserves boundaries and avoids collisions, thus $SN-WER \leq WER$), and Lexical Sensitivity (true recognition errors like substitutions, deletions, insertions, or word order issues are not erased by transliteration). The first two ensure the metric "reduces script inflation but not true errors," while the third is verified via lexical substitution and adversarial experiments.

**2. Canonical Script Selection**: The target script $C$ for $T(\cdot)$ is set to the native reference script of the benchmark (Devanagari, Bengali, Tamil, etc.). This makes SN-WER directly comparable to the original baseline while allowing Romanized hypothesis tokens to be scored based on lexical content rather than surface script. Results are insensitive to the specific choice of canonical script (difference $\Delta < 0.005$ when switching to a uniform script like Devanagari).

**3. Romanization Detection and Deterministic Transliteration**: Unicode block heuristics are used to identify Romanized tokens, which are then mapped back to the native script using libraries such as IAST, ITRANS, or ICU, alongside standard Unicode, punctuation, and number normalization. This requires transliteration to be deterministic, boundary-preserving, and nearly collision-free. Experimental results show differences between libraries $\Delta < 0.002$ and an average collision rate of 0.03% ($<0.1\%$), establishing the empirical validity of "conditional conservativeness."

### Loss & Training

SN-WER involves no training and is a pure evaluation method. The computational complexity matches standard WER ($O(nm)$), with only a transliteration preprocessing step added before scoring.

## Key Experimental Results

### Main Results

Evaluation on 5 Indic languages and 3 ASR models using FLEURS and Common Voice:

| Dataset | Model | WER | SN-WER | Relative $\Delta$ (%) |
|--------|------|-----|--------|----------|
| FLEURS | MMS | 0.32 | 0.30 | -5.4 |
| FLEURS | Whisper-large | 0.70 | 0.64 | -8.0 |
| FLEURS | Whisper-small | 1.27 | 1.21 | -4.7 |
| CommonVoice | MMS | 0.46 | 0.36 | -23.0 |
| CommonVoice | Whisper-large | 0.86 | 0.82 | -4.3 |
| CommonVoice | Whisper-small | 1.46 | 1.36 | -6.9 |

Cross-script extensions (Arabic and Urdu) also show improvements of 5-9%.

### Ablation Study

| Experiment | Key Finding |
|------|----------|
| Transliterator Invariance (E3) | Differences between IAST/ITRANS/ICU mappings $\Delta < 0.002$ |
| Collision Rate (E3) | Average collision rate 0.03%, $<0.1\%$ |
| Normalization Robustness (E3) | Number/punctuation ablation $\Delta < 0.05$ |
| Orthogonal Stress Test (E5) | With 0→50% Romanization injection, SN-WER decays 67% of script inflation |
| Lexical Sensitivity (E6) | With 20-30% lexical substitution, $\Delta_{SN}/\Delta_{WER} \approx 1.09$, proving lexical errors are not decayed |
| Adversarial Validation (E7) | SN-WER $\approx 1.0$ after word shuffling/substitution, confirming semantic errors are not masked |

### Key Findings

- SN-WER reduces the model gap by up to 12% on clean FLEURS (Gujarati) and 26% on noisy Common Voice (Odia), while preserving true recognition weaknesses.
- Ranking stability is perfectly maintained (Kendall $\tau=1.0$), only the magnitude of gaps between models changes.
- Romanization rates are highly correlated with the magnitude of SN-WER correction ($r=0.81$).

## Highlights & Insights

- **Evaluation Companion, Not Replacement**: SN-WER is explicitly positioned as a companion metric to WER/CER, suitable for scenarios where script choice is irrelevant to downstream tasks (search, indexing, retrieval, multilingual LLM pipelines) rather than user-facing transcription tasks.
- **Minimal Adoption Barrier**: No training, no extra data, and no modification to decoding are required; transliteration preprocessing is simply added during the scoring phase.
- **Systematic Validation Methodology**: Seven sets of experiments covering identity, conservativeness, lexical sensitivity, robustness, and adversariality form a complete metric validation paradigm.

## Limitations & Future Work

- Validated only on 5 Indic languages plus Arabic/Urdu; applicability to other multi-script languages (e.g., CJK) remains to be tested.
- While the collision rate of transliteration mapping is extremely low ($<0.1\%$), it may rise for languages with large vocabularies.
- For morphologically complex languages (e.g., agglutinative affixes in Tamil), transliteration may introduce boundary ambiguities.
- Future work could extend this to a script-normalized version of CER (SN-CER).

## Related Work & Insights

- **toWER** (Emond et al., 2018): Transliterated WER for code-switched Indic ASR, but requires modifying training corpora and targets bilingual code-switching rather than monolingual multi-scripting.
- **WERd** (Ali et al., 2017): WER for orthographic variants in Arabic dialects.
- **Lenient CER** (Karita et al., 2023): Targets character-level inconsistencies in Japanese.
- Insight: The normalization approach for evaluation metrics can be generalized to other scenarios featuring surface form differences with semantic equivalence.

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

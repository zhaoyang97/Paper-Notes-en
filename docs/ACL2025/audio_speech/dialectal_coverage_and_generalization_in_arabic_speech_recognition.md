---
title: >-
  [Paper Note] Dialectal Coverage and Generalization in Arabic Speech Recognition
description: >-
  [ACL 2025][Audio & Speech][Arabic ASR] This study systematically investigates the impact of Arabic dialect coverage on ASR performance. By utilizing multi-dialectal pre-training and joint fine-tuning, the ArTST model is extended to cover speech variants from 17 Arabic countries, while multilingual optimization strategies in code-switching scenarios are additionally explored.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Arabic ASR"
  - "Dialectal Speech"
  - "Multi-Dialectal Pre-Training"
  - "Code-Switching"
  - "ArTST"
date: 2026-05-08
content_hash: 6bfae5365b1abf78
---

# Dialectal Coverage and Generalization in Arabic Speech Recognition

**Conference**: ACL 2025  
**arXiv**: [2411.05872](https://arxiv.org/abs/2411.05872)  
**Code**: [mbzuai-nlp/ArTST](https://github.com/mbzuai-nlp/ArTST)  
**Area**: Audio & Speech / Speech Recognition  
**Keywords**: Arabic ASR, Dialectal Speech, Multi-Dialectal Pre-Training, Code-Switching, ArTST  

## TL;DR

This study systematically investigates the impact of Arabic dialect coverage on ASR performance. By utilizing multi-dialectal pre-training and joint fine-tuning, the ArTST model is extended to cover speech variants from 17 Arabic countries, while multilingual optimization strategies in code-switching scenarios are additionally explored.

---

## Background & Motivation

### Background
Arabic is a pluricentric language composed of Modern Standard Arabic (MSA) and numerous regional dialects. Existing ASR systems primarily cover MSA and a few high-resource dialects, displaying insufficient coverage and generalization capabilities across diverse spoken variants. While massive multilingual models (e.g., Whisper, MMS) offer broad coverage, their performance on various Arabic variants remains inconsistent. Monolingual pre-trained models (e.g., ArTST) perform excellently on MSA but struggle in dialectal and code-switching scenarios.

### Core Problem
The paper investigates five key questions:
1. Does pre-training on dialectal data benefit downstream dialect performance? Does it degrade MSA performance?
2. Joint multi-dialectal fine-tuning vs. single-dialectal fine-tuning: which is better?
3. Can reasonable zero-shot performance be achieved on unseen dialects?
4. Can multilingual pre-training optimize performance in code-switching scenarios?
5. What is the impact of multilingual pre-training/fine-tuning on monolingual Arabic performance (language interference)?

### Design Motivation
To build a more inclusive Arabic ASR system by expanding dialectal coverage while maintaining high performance.

---

## Method

### Overall Architecture
Based on the ArTST (Arabic Text and Speech Transformer) model, this work adopts the SpeechT5 architecture, which includes an encoder-decoder module and modality-specific pre- and post-processing networks. During the self-supervised pre-training phase, shared representations of speech and text modalities are learned through quantized tokens.

### Model Versions
- **v1**: Pre-trained solely on MSA data (original ArTST)
- **v2**: Pre-trained using a mix of MSA and dialectal data
- **v3**: Pre-trained using MSA, dialectal, and multilingual data

### Key Designs

#### 1. Dialectal Data Collection and Categorization
- Covers 17 Arabic variants, categorized by region as follows:
    - **Gulf Dialect (GLF)**: Saudi Arabia, Kuwait, UAE, Oman, Qatar, Iraq, Yemen
    - **Levantine Dialect (LEV)**: Syria, Jordan, Lebanon, Palestine
    - **North African Dialect (NOR)**: Egypt, Tunisia, Morocco, Algeria, Mauritania, Sudan
- Data sources: Multiple public datasets including MGB2, QASR, SADA, MASC, and Common Voice.
- Imbalanced resource distribution: High-resource (SAU, SYR, EGY, MSA, $\ge 200$h), medium-resource (UAE, MOR, etc., 10-50h), and low-resource (KUW, PAL, <10h).

#### 2. Pre-training Strategy
- v2 incorporates dialectal speech and text data on top of MSA for self-supervised pre-training.
- v3 further incorporates English, French, and Spanish data.
- Pre-training does not use aligned speech-text data; it only utilizes unaligned speech and text data.

#### 3. Fine-tuning Strategy
- **Single-dialectal Fine-tuning**: Fine-tuning adaptation on MSA (MGB2/QASR) first, followed by fine-tuning on the target dialect.
- **Joint Multi-dialectal Fine-tuning**: Merging 12 dialectal training sets (approx. 1501 hours) to train a single joint model.
- **Dialect ID Strategy**: Prepends a dialect identifier before decoding the string: `<S> DIALECT T1 T2 ... Tn </S>`.
    - Dialect Forcing: Manually specifying the dialect ID.
    - Dialect Inference: Allowing the model to predict the dialect token on its own.

#### 4. Multilingual Fine-tuning (Code-switching)
- Incorporates English (1602h), French (732h), and Spanish (408h) on top of the dialectal data.
- Incorporates code-switching datasets: ArZen (Egyptian-English), Mixat (Emirati-English), and TunSwitch (Tunisian-French).

### Normalization Processing
- Conducts standard orthographic normalization for Arabic NLP before training (unifying Alef, Yaa, and Taa characters).
- Performs post-prediction normalization prior to evaluation.
- Uses Word Error Rate (WER) and Character Error Rate (CER) as evaluation metrics.

---

## Experiments

### Experimental Setup
- **Hardware**: 4× A100 GPUs for pre-training (14-21 days), 1× A100 GPU for fine-tuning (2-7 days).
- **Optimizer**: Adam, with a pre-training learning rate of $2 \times 10^{-4}$ and a fine-tuning learning rate of $6 \times 10^{-5}$.
- **Total Computational Budget**: Approx. 6000 GPU hours.

### Main Results

**MSA Benchmark (MGB2)**:

| System | WER (%) | CER (%) |
|------|--------|--------|
| E2E CTC+Attention+LM | 12.50 | — |
| ArTST v1 + LM | 12.78 | 6.33 |
| **ArTST v2** | **12.49** | 6.44 |
| **ArTST v2 + LM** | **12.39** | 6.51 |

Dialectal pre-training (v2) does not degrade MSA performance, but instead achieves the best WER of 12.39%.

**MGB3 Egyptian Dialect**: v2 yields an absolute WER reduction of approximately 4% over v1, establishing a new SOTA.

**MGB5 Moroccan Dialect**: v2 shows slight but non-significant improvements, likely due to the scarce Moroccan data in pre-training.

**Multi-dialectal Zero-Shot and Fine-Tuning**:

| Dialect | v1 Zero-shot | v2 Zero-shot | v1 Fine-tuned | v2 Fine-tuned |
|------|----------|----------|---------|---------|
| SAU | 61.23 | 58.72 | 27.40 | 27.33 |
| SYR | 21.99 | 18.37 | 18.64 | 17.42 |
| EGY | 50.87 | 47.17 | 38.47 | 36.43 |
| KUW | 64.74 | 52.02 | 50.29 | 46.24 |

v2 outperforms v1 in both zero-shot and fine-tuned settings across most dialects.

### Joint Models and Dialect IDs

| Strategy | Macro Avg. WER (%) |
|------|--------------|
| v2 Zero-shot | 46.37 |
| v2→QASR | 37.58 |
| v2→Single-dialect FT | 33.17 |
| Joint (No Dialect ID) | 32.63 |
| Joint (Dialect Forcing) | 34.09 |
| Joint (Dialect Inference) | **31.45** |

The Joint Model + Dialect Inference achieves the best overall performance. Dialect Forcing performs worse than having No Dialect ID, as the dialect annotations in the data themselves are relatively coarse.

### Zero-Shot (Unseen Dialects)

| Dialect | v1→MGB2 | v2→Joint |
|------|---------|----------|
| ALG | 73.18 | 45.20 |
| SUD | 69.20 | 40.69 |
| YEM | 41.64 | 33.08 |

Joint multi-dialectal fine-tuning considerably outperforms v1 on unseen dialects.

### Code-switching Results

| Test Set | v1 (Direct) | v2 (Dialect Adapted) | v3 (Multilingual Adapted) |
|--------|---------|-------------|---------------|
| ArzEn (EGY-EN) | 43.21 | 33.71 | **27.43** |
| TunSwitch (TUN-FR) | 53.85 | 43.59 | **36.66** |
| Mixat (UAE-EN) | 42.50 | 25.73 | **21.66** |

v3 achieves the best performance on all code-switching test sets, yielding a 4% to 7% absolute WER reduction compared to v2.

### Language Interference
- v3 obtains a WER of 13.0% on MGB2 (MSA), slightly worse than v2's 12.49%.
- On dialectal speech, multilingual pre-training results in an absolute WER increase of 4% to 16%, showing a noticeable negative impact.

---

## Highlights & Insights

1. **Largest-scale dialectal Arabic ASR study**: Covers speech variants across 17 countries/regions and systematically addresses 5 key research questions.
2. **Dialectal pre-training does not degrade MSA**: Instead, it achieves SOTA on MGB2, alleviating concerns for practitioners.
3. **Dialect inference outperforms dialect forcing**: Since dialect annotations in the dataset are coarse-grained country-level approximations, letting the model infer dialect tokens allows for more flexibility.
4. **Joint models greatly benefit low-resource dialects**: However, high-resource dialects still benefit more from single-dialectal fine-tuning.
5. **Code-switching requires multilingual pre-training**: Nonetheless, this inevitably introduces language interference, which negatively impacts dialect performance more severely.
6. **Solely built on open-source data**: The models and training scripts are fully open-sourced, which significantly facilitates community investigation.

## Limitations & Future Work

1. The granularity of dialect categorization is coarse (country-level), whereas actual dialectal variations are much more complex than national borders.
2. Datasets may contain inaccurate labels (e.g., Syrian data in MASC is actually entirely MSA).
3. There is no standard spelling system for Arabic dialects, resulting in substantial transcription variations, which may render downstream WER evaluations pessimistic.
4. Except for MGB3/MGB5 which use multi-reference WER, other datasets only rely on single-reference evaluation.
5. The language interference issues introduced by multilingual pre-training remain unresolved.

## Related Work & Insights

- **Arabic ASR**: Whisper (Radford et al. 2023), MMS (Pratap et al. 2024), ArTST (Toyin et al. 2023)
- **Dialectal ASR Datasets**: QASR (Mubarak et al. 2021), SADA (Alharbi et al. 2024), MASC (Al-Fetyani et al. 2021)
- **Code-switching**: ArZen (Al-Sabbagh 2024), Mixat (Al Ali & Aldarmaki 2024)
- **Self-Supervised Speech Models**: wav2vec (Baevski et al. 2020), HuBERT (Hsu et al. 2021), SpeechT5 (Ao et al. 2022)

---

## Rating ⭐⭐⭐⭐

This work presents a large-scale study and systematic experimental design, addressing critical questions in practical applications. Although the methodology is not highly novel (primarily exploring data and training strategies), the experimental conclusions carry substantial practical value. The open-sourcing of models and data represents a valuable contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder](multimed_multilingual_medical_speech_recognition_via_attention_encoder_decoder.md)
- [\[ACL 2025\] MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](../../CVPR2026/audio_speech/echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)
- [\[NeurIPS 2025\] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition](../../NeurIPS2025/audio_speech/mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5
description: >-
  [Audio & Speech] This paper introduces ChildMandarin, a Mandarin speech dataset for young children aged 3-5, containing 397 speakers, 41.25 hours of speech, and covering 22 provincial-level administrative regions in China, along with comprehensive baseline evaluations on ASR and speaker verification tasks.
tags:
  - "Audio & Speech"
date: 2026-05-08
content_hash: 992c4a4ad609ad68
---

# ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5

## Basic Information

- **Conference**: ACL2025
- **arXiv**: [2409.18584](https://arxiv.org/abs/2409.18584)
- **Code**: [https://github.com/flageval-baai/ChildMandarin](https://github.com/flageval-baai/ChildMandarin)
- **Area**: Others (Speech)
- **Keywords**: Child Speech Recognition, Mandarin Speech Dataset, 3-5 Years Old Children, ASR, Speaker Verification

## TL;DR

This paper introduces ChildMandarin, a Mandarin speech dataset for young children aged 3-5, containing 397 speakers, 41.25 hours of speech, and covering 22 provincial-level administrative regions in China, along with comprehensive baseline evaluations on ASR and speaker verification tasks.

## Background & Motivation

### Problem Definition
Automatic Speech Recognition (ASR) systems are primarily trained on adult speech, showing severe performance degradation on child speech. Speech from young children is characterized by unstable pronunciation, higher pitch, high variation in speech rate, and unclear articulation, which significantly differs from adult speech. Specifically, speech data for young children aged 3-5 is extremely scarce.

### Limitations of Prior Work

**Current Status of Chinese Child Speech Datasets**:

| Dataset | Age | Speakers | Duration | Availability |
|---|---|---|---|---|
| Tong Corpus | 1;7-3;4 | 1 | 22h | Available |
| CASS CHILD | 1-4 | 23 | 631h | Unavailable |
| SLT-CSRC C1 | 7-11 | 927 | 28.6h | Unavailable |
| SLT-CSRC C2 | 4-11 | 54 | 29.5h | Unavailable |
| SingaKids | 7-12 | 255 | 75h | Available |

**Tong Corpus**: Contains longitudinal recordings of only a single child, failing to provide the speaker diversity required for training ASR systems.

**CASS CHILD**: Despite having 631 hours of audio, only 80 hours are transcribed and the dataset is not publicly available.

**SLT-CSRC**: Was only accessible during the SLT 2021 Challenge.

**SingaKids**: Broadly covers older children aged 7-12, rather than young children.

**Rich English Datasets but Scarce in Other Languages**: Resource availability is relatively high for English (e.g., Providence, MyST, CSLU Kids), while resources for non-European languages are severely lacking.

### Motivation
The goal is to construct a large-scale, multi-speaker, and geographically diverse Mandarin speech dataset tailored for young children aged 3-5, providing a foundational resource for ASR and speaker verification research in this age group.

## Method

### Overall Architecture
Dataset Construction + ASR Baseline Evaluation + Speaker Verification Baseline Evaluation

### Dataset Construction

**Collection Setup**:
- Conversational recording (non-read speech), encouraging natural interactions
- Parents present throughout to provide emotional support
- Unrestricted recording content, focusing on age-appropriate daily conversations
- Equipment: Smartphones (216 Android devices, 181 iPhones)
- Quiet indoor environments, allowing for minimal background noise
- Format: WAV PCM, 16kHz sampling rate, 16-bit precision

**Transcription and Annotation**:
- Character-level manual transcription, conducted by professional transcribers
- Faithfully records stutters, disfluencies, and developmental speech patterns
- Faithfully records regional pronunciation variations
- Numbers are transcribed phonetically based on actual pronunciation

**Dataset Statistics**:

| Subset | Speakers | Utterances | Duration (hrs) | Average (s) |
|---|---|---|---|---|
| Train | 317 | 32,658 | 33.35 | 3.68 |
| Dev | 39 | 4,057 | 3.78 | 3.35 |
| Test | 41 | 4,198 | 4.12 | 3.53 |
| **Total** | **397** | **40,913** | **41.25** | **3.52** |

**Demographics**:
- Age distribution: 3, 4, and 5 years old, balanced by age and gender
- Geographical distribution: 22 provincial-level administrative regions (Shanxi has the most at 136, followed by Jiangsu at 40, and Henan at 39)
- Accent classification: Mild (majority), Moderate, Severe (~4%)

### Key Designs

**Speaker-independent split**: No speaker overlap among train/dev/test sets, ensuring generalizability in evaluation.

**Ethical Protections**:
- Informed consent obtained from all participants' parents or legal guardians
- Fair compensation of 150 RMB is provided to each child
- Data anonymization, removing all personally identifiable information (PII)
- Used strictly for academic research purposes

## Experiments

### ASR Task

#### Training Models from Scratch

Trained using the Wenet toolkit, with Character Error Rate (CER, %) as the evaluation metric:

| Model | Parameters | Decoding Method | CER (%) |
|---|---|---|---|
| Transformer (CTC-AED) | 29M | Attention Rescoring | 32.15 |
| **Conformer (CTC-AED)** | **31M** | **Attention Rescoring** | **27.38** |
| Conformer (RNN-T AED) | 45M | Attention | 33.84 |
| Paraformer | 30M | Beam Search | 28.94 |

Conformer + CTC-AED + Attention Rescoring achieves the best performance.

#### Fine-tuning Self-Supervised Pre-trained Models

| Model | CER (%) |
|---|---|
| Wav2vec 2.0 (Base) | 20.29 |
| Wav2vec 2.0 (Large) | 21.12 |
| HuBERT (Base) | 18.74 |
| **HuBERT (Large)** | **14.97** |

HuBERT consistently outperforms Wav2vec 2.0, which aligns with recent findings.

#### Fine-tuning Supervised Pre-trained Models

| Model | Parameters | Zero-shot | Fine-tuning |
|---|---|---|---|
| **CW (Conformer-WenetSpeech)** | **122M** | **18.05** | **13.66** |
| Whisper-tiny | 39M | 67.63 | 28.78 |
| Whisper-base | 74M | 51.49 | 23.33 |
| Whisper-small | 244M | 37.99 | 17.45 |
| Whisper-medium | 769M | 28.55 | 18.97 |
| Whisper-large-v2 | 1,550M | 29.43 | - |

- CW (Conformer-WenetSpeech) achieves the best performance in both zero-shot and fine-tuning scenarios.
- Fine-tuning significantly reduces the CER of all models.
- Whisper-medium performs worse than Whisper-small after fine-tuning (likely due to overfitting on the small dataset).

### Performance Analysis

**Impact of Age and Gender**:
- CER decreases as age increases: 3 years old > 4 years old > 5 years old.
- Boys consistently exhibit higher CER than girls in the same age group.
- 3-year-old boys show the highest CER (34.78% in zero-shot, 26.80% in fine-tuning).

**Error Type Analysis (CW Model)**:

| Age_Gender | Substitution (%) | Deletion (%) | Insertion (%) |
|---|---|---|---|
| 3_F | 9.03 | 2.04 | 0.69 |
| 3_M | 26.80 | 4.35 | 2.11 |
| 4_F | 3.94 | 0.53 | 0.15 |
| 5_M | 14.32 | 3.04 | 1.23 |

Substitution errors dominate, followed by deletion errors.

### Speaker Verification Task

| Model | Parameters | Embedding Dim | PLDA EER (%) | PLDA minDCF |
|---|---|---|---|---|
| x-vector | 4.2M | 512 | 8.91 | 0.7198 |
| **ResNet-TDNN** | **15.5M** | **256** | **9.57** | **0.6597** |
| ECAPA-TDNN | 20.8M | 192 | 13.72 | 0.8697 |

- The dataset is viable for speaker verification tasks.
- ECAPA-TDNN suffers from overfitting on the small dataset due to its large parameter size, underperforming compared to x-vector and ResNet-TDNN.
- The incomplete development of vocal tracts in young children masks gender-related vocal features, increasing the difficulty of verification.

### Key Findings
1. Fine-tuning pre-trained models significantly reduces CER compared to training from scratch (the best result decreases from 27.38% to 13.66%).
2. Speech from 3-year-olds represents the greatest challenge, with CERs reaching 2 to 3 times those of 5-year-olds in the same dataset.
3. For small-scale datasets, larger models can lead to performance degradation due to overfitting.
4. Conformer-WenetSpeech shows the best transfer performance on child speech.

## Highlights & Insights

1. **Filling the Gap in Critical Age Groups**: Focuses on young children aged 3-5, which is the most resource-scarce cohort in existing literature.
2. **High-Quality Natural Interaction Data**: Conversational recording is used instead of read speech, which more realistically reflects child speech characteristics.
3. **Broad Geographical Coverage**: Involves 397 speakers across 22 provincial-level administrative regions, offering good dialectal diversity.
4. **Dual-Task Baselines**: Establishes baselines for both ASR and speaker verification, illustrating the multi-task usability of the dataset.
5. **Fine-Grained Analysis**: Examines ASR performance across multiple dimensions, including age, gender, and error types, guiding future research directions.

## Limitations & Future Work

1. **Relatively Limited Data Volume**: 41.25 hours is still small compared to adult speech datasets.
2. **Imbalanced Geographical Distribution**: Shanxi contributes the most speakers (136), while some provinces have only a few participants.
3. **Overfitting in Fine-Tuning Large Models**: The data volume restricts the effectiveness of using models with massive parameter sizes.
4. **Noise Control in Conversational Recordings**: Background noise is inevitable in recordings of very young children.

## Related Work & Insights

- **Chinese Child Speech**: Tong Corpus (Xiangjun and Yip, 2017), CASS CHILD (Gao et al., 2012), SingaKids (Chen et al., 2016), SLT-CSRC (Yu et al., 2021)
- **English Child Speech**: MyST Corpus (Pradhan et al., 2024), CSLU Kids (Shobaki et al., 2007), TBALL (Kazemzadeh et al., 2005)
- **ASR Models**: Conformer (Gulati et al., 2020), Whisper (Radford et al., 2023), HuBERT (Hsu et al., 2021), Wav2vec 2.0 (Baevski et al., 2020)
- **Speaker Verification**: x-vector (Snyder et al., 2018), ECAPA-TDNN (Desplanques et al., 2020)

## Rating ⭐⭐⭐⭐

- Novelty: ⭐⭐⭐⭐ — Fills the gap in Mandarin speech datasets for young children aged 3-5.
- Practicality: ⭐⭐⭐⭐⭐ — Direct application value for educational technology and child-computer interaction.
- Method Novelty: ⭐⭐⭐ — Primarily focused on dataset construction; standard methods are used for baseline evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual tasks of ASR + SV, with comprehensive coverage of both from-scratch training and pre-training fine-tuning, as well as detailed fine-grained analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)
- [\[CVPR 2025\] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](../../CVPR2025/audio_speech/dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)
- [\[ACL 2026\] Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios](../../ACL2026/audio_speech/comprehensive_benchmarking_of_long-form_speech_generation_in_diverse_scenarios.md)
- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)

</div>

<!-- RELATED:END -->

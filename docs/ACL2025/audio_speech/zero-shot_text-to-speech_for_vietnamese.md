---
title: >-
  [Paper Note] Zero-Shot Text-to-Speech for Vietnamese
description: >-
  [ACL 2025][Audio & Speech][Zero-Shot TTS] To address the lack of high-quality long-audio datasets for Vietnamese zero-shot TTS, the 941-hour PhoAudiobook dataset was constructed. Systematic experiments conducted on three SOTA zero-shot TTS models (VALL-E, VoiceCraft, and XTTS-v2) demonstrate that PhoAudiobook significantly improves model performance. Specifically, XTTS-v2 completely outperforms the baseline viXTTS on long-sentence synthesis…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Zero-Shot TTS"
  - "Vietnamese"
  - "PhoAudiobook"
  - "Speech Synthesis"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: 142e07c0187211b9
---

# Zero-Shot Text-to-Speech for Vietnamese

**Conference**: ACL 2025  
**arXiv**: [2506.01322](https://arxiv.org/abs/2506.01322)  
**Code**: [https://huggingface.co/datasets/thivux/phoaudiobook (dataset)](https://huggingface.co/datasets/thivux/phoaudiobook)  
**Area**: Audio & Speech / TTS  
**Keywords**: Zero-Shot TTS, Vietnamese, PhoAudiobook, Speech Synthesis, Low-Resource Languages

## TL;DR
To address the lack of high-quality long-audio datasets for Vietnamese zero-shot TTS, the 941-hour PhoAudiobook dataset was constructed. Systematic experiments conducted on three SOTA zero-shot TTS models (VALL-E, VoiceCraft, and XTTS-v2) demonstrate that PhoAudiobook significantly improves model performance. Specifically, XTTS-v2 completely outperforms the baseline viXTTS on long-sentence synthesis, while VALL-E and VoiceCraft exhibit higher robustness in short-sentence synthesis.

## Background & Motivation
Zero-shot TTS aims to synthesize speech for unseen speakers using only a few seconds of reference audio, which has been a research hotspot in the TTS field recently. Language modeling-based methods such as VALL-E and VoiceCraft have achieved remarkable results in English.

**Limitations of Low-Resource Languages**:
- Low-resource languages like Vietnamese lack the large-scale, high-quality datasets required for training zero-shot TTS.
- Existing Vietnamese speech datasets (such as VinBigData, BUD500, viVoice, etc.) suffer from several key defects:
    - Audio samples are too short (typically <10 seconds), making them unsuitable for TTS models that require long contexts.
    - Lack of speaker identification (viVoice uses YouTube channel names as approximations, but a channel may contain multiple speakers).
    - Text is not normalized (e.g., numbers to words).
    - Inconsistent audio quality (e.g., from consumer-grade devices, background noise).

**Ours Contributions**: Instead of proposing a new TTS model architecture, this work focuses on **dataset construction**—building a large-scale Vietnamese audio dataset truly suitable for zero-shot TTS training, and validating the performance of existing SOTA models on Vietnamese based on this dataset.

## Method

### Overall Architecture
1. Construction of the PhoAudiobook dataset: Starting from raw audiobook audio, passing through a complete pipeline of background removal $\rightarrow$ transcription $\rightarrow$ quality filtering $\rightarrow$ speaker diarization $\rightarrow$ text normalization.
2. Training three zero-shot TTS models (VALL-E, VoiceCraft, and XTTS-v2) on PhoAudiobook.
3. Comparative evaluation against the baseline viXTTS on multiple test sets using both objective and subjective metrics.

### Key Designs

1. **PhoAudiobook Dataset Construction Pipeline**:

    - **Raw Data Collection**: Collected 23K hours of audio, 2,697 audiobooks, and 735 speakers from public audiobook websites.
    - **Background Music Removal**: Used Demucs to extract vocal tracks.
    - **Transcription Generation**: Whisper-large-v3 was used to generate transcriptions and timestamps.
    - **Long Audio Merging**: Merged continuous short audio segments into 10-20 second long samples (this is a key innovation, as other datasets have audio <10 seconds).
    - **Dual-Model Cross-Validation**: Whisper-large-v3 and PhoWhisper-large independently generated transcriptions, keeping only samples where both transcriptions matched exactly — ensuring transcription quality.
    - **Multi-Speaker Filtering**: Used the wav2vec2-bartpho model to identify and filter out audio containing multiple speakers.
    - **Post-processing**: Removed excessively short transcriptions (<25 words), trimmed leading/trailing silences, and normalized volume.
    - **Text Normalization**: Finetuned an mbart-large-50 model to handle conversions such as numbers to words.
    - **Speaker Balancing**: Capped at 4 hours per speaker, resulting in a final dataset of 941 hours across 735 speakers.

2. **Data Augmentation Strategy**:

    - Reprocessed the 940-hour training set through the pipeline as new raw data, but skipped the long audio merging and short sample filtering steps.
    - Obtained an additional 554 hours of short audio, bringing the total training data to 1,494 hours.
    - Design Motivation: To ensure the models can handle input texts of varying lengths.

3. **Evaluation System Design**:

    - 4 Test Sets: PhoAudiobook-Seen (seen speakers), PhoAudiobook-Unseen (unseen speakers), VIVOS (out-of-distribution short audio), viVoice (out-of-distribution).
    - Objective Metrics: WER (word error rate / intelligibility), MCD (mel-cepstral distortion), RMSE_F0 (prosody matching).
    - Subjective Metrics: MOS (mean opinion score for overall quality, 1-5 scale), SMOS (similarity MOS for speaker similarity, 1-5 scale).
    - 10-20 native evaluators, with model names anonymized and randomized during testing.

### Loss & Training
Each of the three models uses its standard training procedure:
- VALL-E: Conditional codec language modeling.
- VoiceCraft: Token rearrangement + left-to-right language modeling.
- XTTS-v2: Multilingual finetuning based on the Tortoise architecture.
All models were trained on the 1,494 hours of augmented training data.

## Key Experimental Results

### Main Results (Objective Metrics)

| Model | PAB-S WER↓ | PAB-U WER↓ | VIVOS WER↓ | viVoice WER↓ |
|------|-----------|-----------|-----------|-------------|
| VALL-E_PAB | 24.96 | 12.90 | **12.63** | 13.58 |
| VoiceCraft_PAB | 7.53 | 15.14 | **13.53** | 21.70 |
| XTTS-v2_PAB | **4.16** | **4.31** | 37.81 | **8.32** |
| viXTTS (baseline) | 4.23 | 5.17 | 37.81 | 12.54 |

| Model | PAB-S MCD↓ | PAB-U RMSE_F0↓ | viVoice MCD↓ |
|------|-----------|---------------|-------------|
| XTTS-v2_PAB | **6.30** | 242.51 | **8.34** |
| viXTTS | 7.47 | 271.70 | 8.71 |

### Main Results (Subjective Metrics)

| Model | PAB-S MOS↑ | PAB-U MOS↑ | VIVOS MOS↑ | viVoice MOS↑ |
|------|-----------|-----------|-----------|-------------|
| XTTS-v2_PAB | **4.20** | 3.89 | 2.79 | **3.98** |
| VoiceCraft_PAB | 4.16 | 3.75 | **3.85** | **3.98** |
| VALL-E_PAB | 3.96 | **4.04** | 3.44 | 3.75 |
| viXTTS | 4.05 | 3.85 | 2.37 | 3.48 |

| Model | PAB-S SMOS↑ | PAB-U SMOS↑ | VIVOS SMOS↑ | viVoice SMOS↑ |
|------|-----------|-----------|-----------|-------------|
| VALL-E_PAB | **3.77** | **3.46** | **3.35** | 3.20 |
| XTTS-v2_PAB | 3.55 | **3.56** | 3.03 | 3.39 |
| viXTTS | 2.88 | 2.63 | 2.48 | 3.11 |

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| PhoAudiobook vs viVoice | XTTS-v2_PAB achieves a WER of 8.32 on the viVoice test set, significantly outperforming viXTTS (12.54), even though the latter was trained on this data. |
| Long Sentences vs Short Sentences | VALL-E and VoiceCraft perform much better than XTTS-v2 on the VIVOS short sentence set (WER 12-13 vs 37.81). |
| XTTS-v2 Short Sentence Issues | XTTS-v2 generates redundant/verbose trailing speech for short input texts, which is an architectural issue rather than a data issue. |
| Data Quality | PhoAudiobook has the highest mel-cepstral SNR (SI-SNR of 4.91dB) among all datasets, with clear speaker IDs. |

### Key Findings
- PhoAudiobook consistently improves the performance of all models, validating the critical role of high-quality data for low-resource language TTS.
- XTTS-v2 leads comprehensively in long-sentence synthesis, but has architectural defects (generating redundant trailing speech) on short sentences.
- VALL-E and VoiceCraft are more robust in short-sentence scenarios and exhibit complementarity.
- Regarding speaker similarity (SMOS), models trained on PhoAudiobook significantly outperform viXTTS (up to a +0.87 points increase).
- The audio length distribution of the dataset has a direct impact on model behavior (10-20s training samples benefit long-sentence synthesis).

## Highlights & Insights
- **Reusable Dataset Construction Pipeline**: The entire pipeline from audiobooks to high-quality TTS datasets can be directly migrated to other low-resource languages, offering high engineering value.
- **Dual-ASR Cross-Validation**: Validating transcription accuracy with both Whisper and PhoWhisper and retaining only consistent results is a strategy worth adopting in any speech dataset construction.
- **Clever Data Augmentation**: Reprocessing the training set through the pipeline while skipping the merging step to obtain short audio augmentation is a simple yet effective solution to the input length generalization problem.
- **Objective Truth**: In low-resource language TTS, data quality is often more important than model choice—XTTS-v2 trained on PhoAudiobook performs better even on viXTTS's own training data.

## Limitations & Future Work
- Performance in code-mixed (Vietnamese + English) scenarios was not evaluated.
- The audiobook domain results in relatively slow speech rates (201 wpm vs 229-243 wpm in other datasets), which may affect generalization to fast speech.
- Non-commercial use-only limits practical applications.
- The performance of newer TTS models (e.g., CosyVoice, etc.) was not explored.
- The "Hard" task contains only 40 samples, lacking statistical reliability.
- The 16kHz sampling rate is lower than the standard of some high-quality TTS studies (24kHz+).

## Related Work & Insights
- **VALL-E (Wang et al., 2023)** pioneered the codec language modeling paradigm for TTS; this work extends it to Vietnamese.
- The token rearrangement strategy of **VoiceCraft (Peng et al., 2024)** performs well on zero-shot TTS and speech editing.
- **XTTS-v2 (Casanova et al., 2024)** is currently the strongest multilingual zero-shot TTS, but this work reveals its short-text limitations.
- **Insight**: The primary bottleneck in low-resource language TTS research is data rather than models. Audiobooks have unique advantages as speech data sources—professional recording, long content, and identifiable speakers—which are worth extending to more languages.

## Rating
- Novelty: ⭐⭐⭐ High engineering innovation in dataset construction, but limited methodology innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage with 3 models, 4 test sets, and both objective and subjective metrics.
- Writing Quality: ⭐⭐⭐⭐ Detailed dataset description, standard experimental design, and high reproducibility.
- Value: ⭐⭐⭐⭐ Direct boost to the Vietnamese TTS community, with pipeline designs that serve as a valuable reference for other low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [\[ACL 2025\] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](controlspeech_zero_shot.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](../../ACL2026/audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ICML 2025\] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](../../ICML2025/audio_speech/do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)

</div>

<!-- RELATED:END -->

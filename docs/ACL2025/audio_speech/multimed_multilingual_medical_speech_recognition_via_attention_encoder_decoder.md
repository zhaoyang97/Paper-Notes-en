---
title: >-
  [Paper Note] MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder
description: >-
  [ACL 2025][Audio & Speech][medical ASR] MultiMed is introduced as the first multilingual medical ASR dataset (150 hours, 5 languages, 10 recording conditions, 16 accents), along with end-to-end Whisper model baselines spanning small to large scales. This work presents the first systematic study of multilingual medical ASR, comparing monolingual vs. multilingual fine-tuning and AED vs. Hybrid architectures. Key findings reveal that multilingual joint training benefits small mo…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "medical ASR"
  - "multilingual speech recognition"
  - "Whisper"
  - "dataset"
  - "attention encoder decoder"
date: 2026-05-08
content_hash: d290899d0c8621d3
---

# MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder

**Conference**: ACL 2025  
**arXiv**: [2409.14074](https://arxiv.org/abs/2409.14074)  
**Code**: [GitHub](https://github.com/leduckhai/MultiMed/tree/master/MultiMed)  
**Area**: Audio and Speech  
**Keywords**: medical ASR, multilingual speech recognition, Whisper, dataset, attention encoder decoder

## TL;DR
MultiMed is introduced as the first multilingual medical ASR dataset (150 hours, 5 languages, 10 recording conditions, 16 accents), along with end-to-end Whisper model baselines spanning small to large scales. This work presents the first systematic study of multilingual medical ASR, comparing monolingual vs. multilingual fine-tuning and AED vs. Hybrid architectures. Key findings reveal that multilingual joint training benefits small models but can lead to performance degradation in larger models.

## Background & Motivation

**Background**: Medical ASR serves as the foundation for downstream tasks such as clinical documentation automation, electronic health records (EHR), and speech translation. The ASR market is projected to reach $15.87 billion by 2030. However, medical ASR research is severely limited by a scarcity of datasets due to privacy constraints.

**Limitations of Prior Work**: (a) Existing medical ASR datasets are small in scale (e.g., PriMock57 has only 9h), limited to a single language (mostly English), and often consist of simulated rather than real-world medical dialogue; (b) Commercial APIs (such as Google Healthcare and Nuance Dragon) are closed-source; (c) There lacks systematic research on multilingual medical ASR.

**Key Challenge**: Medical scenarios are inherently multilingual (e.g., international hospitals, multinational telemedicine), yet the research community lacks public multilingual medical speech resources and baselines.

**Goal**: To build the first multilingual medical ASR dataset, provide comprehensive baselines, and present the first analysis of multilingual characteristics in this domain.

**Key Insight**: Collecting real-world medical dialogue audio from professional YouTube medical channels, covering 5 languages, 10 recording scenarios, and 6 speaker roles.

**Core Contributions**: Dataset (MultiMed) + model baselines (Whisper tiny to medium) + multilingual analysis + AED vs. Hybrid comparison.

## Method

### Dataset Construction
- **Source**: Real-world conversations (not simulated) from professional YouTube medical channels, with manual annotation and verification by medical experts.
- **5 Languages**: Vietnamese (16h), English (109h), French (7h), Chinese (6h), German (11h), totaling 150h.
- **Diversity Metrics**: 10 recording conditions (interviews, lectures, podcasts, news, documentaries, etc.), 16 accents, 6 speaker roles (doctors, patients, hosts, podcasters, etc.), and 198 speakers.
- **vs. Existing Datasets**: Substantially outperforms datasets like PriMock57, VietMed, and AfriSpeech-200 in total duration, number of recording conditions, accents, and roles.

### Models and Training Strategies
- **Model**: Whisper Tiny (38M) / Base (73M) / Small (242M) / Medium (764M)
- **Two Fine-tuning Strategies**:
  1. Decoder-only (frozen encoder): Utilizes the pre-trained encoder and only tunes the decoder.
  2. Full encoder-decoder: Full fine-tuning.
- **Monolingual vs. Multilingual**: Individual training for each language vs. joint training across 5 languages.

### Architectural Evaluation Comparison
- **AED (Attention Encoder Decoder)**: Whisper architecture, end-to-end seq2seq.
- **Hybrid**: Conventional hybrid DNN-HMM + language model architecture.
- Conducts comparative analysis between the two architectures under a fixed parameter budget.

## Key Experimental Results

### Monolingual Fine-Tuning (Decoder-only, WER% test)

| Language | Tiny | Base | Small | Medium |
|------|------|------|-------|--------|
| Vietnamese | 46.98 | 37.74 | 28.77 | **25.43** |
| English | 29.73 | 25.43 | 20.52 | **19.41** |
| French | 52.89 | 42.57 | 33.02 | **31.05** |
| German | 28.22 | 23.09 | 19.91 | **17.92** |
| Chinese | 95.97 | 89.73 | 88.50 | **86.52** |

### Monolingual vs. Multilingual Fine-Tuning Comparison (Medium, WER% test)

| Language | Monolingual | Multilingual | Difference |
|------|------|------|------|
| Vietnamese | **25.43** | 29.81 | +4.38 (degradation) |
| English | **19.41** | 25.65 | +6.24 (degradation) |
| French | **31.05** | 41.40 | +10.35 (degradation) |
| German | **17.92** | 24.13 | +6.21 (degradation) |
| Chinese | **86.52** | 96.80 | +10.28 (degradation) |

### Ablation: Full ft vs Decoder-only (Medium)

| Language | Decoder-only WER | Full ft WER |
|------|-----------------|-------------|
| English | 19.41 | 18.06 |
| German | 17.92 | 17.17 |
| Vietnamese | 25.43 | 24.15 |

### Key Findings
- **Extremely High Chinese WER (>86%)**: Ineffective even with the Medium model, stemming from the complexity of Chinese medical terminology and tokenization issues (character-level vs. word-level).
- **Multilingual Joint Training Degrades on Large Models**: Contrary to experiences in general-domain ASR, multilingual training in the medical domain on the Medium model underperforms compared to monolingual training. This could be due to substantial differences and mutual interference between medical terminologies across different languages.
- **Full Fine-Tuning Outperforms Decoder-Only**: Adapting the encoder to medical acoustic features yields a 1-2% absolute improvement in WER.
- **French/Chinese present the highest challenge**: Due to sparse data, complex linguistic structures, and dense medical terminology.
- **Larger Models Achieve Lower WER**: Consistent with scaling laws, though diminishing returns are observed from Small to Medium.

## Highlights & Insights
- **First Real-World Multilingual Medical ASR Dataset**: The 150h scale is unprecedented in the medical ASR domain, spanning 5 languages and 10 recording conditions with exceptional diversity.
- **Counter-intuitive Finding of Multilingual Training Limitations**: On large-capacity models, monolingual fine-tuning with language-specific models outperforms a "one-model-fits-all" multilingual scheme. This offers crucial design guidance for practical medical ASR deployment.
- **Practically Friendly Training Paradigm**: Decoder-only fine-tuning offers an efficient alternative under fixed parameter budgets, enabling straightforward application in industrial scenarios.

## Limitations & Future Work
- **Imbalanced Data Distribution**: English dominates with 109h while Chinese and French only contain 6-7h, limiting the baseline reliability for low-resource languages.
- **High Chinese WER**: Demands more Chinese medical speech data and specialized tokenization strategies.
- **Unexplored SOTA Large Models**: Stronger models like Whisper large-v3 or Universal-1 are yet to be integrated.
- **No Downstream Evaluation**: The impact of ASR WER improvements on downstream tasks (e.g., NER, summarization) remains unevaluated.
- **Privacy/Ethics**: Although fair-use is claimed, the long-term legal usage rights of YouTube data remain uncertain.
- **Code-switching Unaddressed**: Code-switching (common in multilingual medical contexts) is not currently considered.

## Related Work & Insights
- **vs. Whisper (Radford et al., 2023)**: While Whisper excels at general-domain multilingual ASR, domain adaptation is critical for the medical field. MultiMed bridges this gap.
- **vs. VietMed (Le-Duc, 2024)**: VietMed is a monolingual (Vietnamese) 16h dataset, whereas MultiMed extends to 5 languages totaling 150h.
- **vs. AfriSpeech-200**: AfriSpeech mixes general and medical domains, while MultiMed is purely medical-focused and features diverse recording scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The first multilingual medical ASR dataset, addressing a critical blank.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 models x 2 strategies x monolingual/multilingual x 5 languages; comprehensive but lacking sufficient samples for Chinese and French.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rich statistical tables.
- Value: ⭐⭐⭐⭐⭐ Completely open-sourced dataset, models, and baselines; possesses high long-term resource value for the medical ASR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learnable Fractional Superlets with a Spectro-Temporal Emotion Encoder for Speech Emotion Recognition](../../ICLR2026/audio_speech/learnable_fractional_superlets_with_a_spectro-temporal_emotion_encoder_for_speec.md)
- [\[ACL 2025\] Dialectal Coverage and Generalization in Arabic Speech Recognition](dialectal_coverage_and_generalization_in_arabic_speech_recognition.md)
- [\[NeurIPS 2025\] EuroSpeech: A Multilingual Speech Corpus](../../NeurIPS2025/audio_speech/eurospeech_a_multilingual_speech_corpus.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2025\] MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)

</div>

<!-- RELATED:END -->

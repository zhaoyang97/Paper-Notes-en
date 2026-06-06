---
title: >-
  [Paper Note] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations
description: >-
  [ICCV 2025][Audio & Speech][zero-shot speech recognition] This paper proposes Zero-AVSR, a framework that transcribes speech into language-agnostic romanized text (Roman text) and then leverages an LLM to convert the Rom…
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "zero-shot speech recognition"
  - "audio-visual speech recognition"
  - "romanization"
  - "large language models"
  - "multilingual"
date: 2026-05-08
content_hash: 6c1e24a681107b60
---

# Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations

**Conference**: ICCV 2025
**arXiv**: [2503.06273](https://arxiv.org/abs/2503.06273)  
**Code**: Available (the paper states that code and models are available online)  
**Area**: Audio & Speech
**Keywords**: zero-shot speech recognition, audio-visual speech recognition, romanization, large language models, multilingual

## TL;DR

This paper proposes Zero-AVSR, a framework that transcribes speech into language-agnostic romanized text (Roman text) and then leverages an LLM to convert the Roman text into target-language graphemes, enabling zero-shot audio-visual speech recognition without any target-language speech data. The authors also construct the MARC dataset, covering 82 languages and 2,916 hours of audio-visual data.

## Background & Motivation

Audio-visual speech recognition (AVSR) combines audio and lip-movement information to enhance speech understanding, particularly under noisy conditions. However, existing AVSR research has focused predominantly on English, and multilingual AVSR datasets cover only 9 languages. Acquiring sufficient annotated audio-visual data for each language is highly challenging, severely limiting the extension of AVSR to broader languages.

**Core Problem**: How can speech recognition be performed without using any speech data from the target language?

Key insight: Different languages share phonetic characteristics at the phoneme level, and romanization can unify all languages into a single language-agnostic representation space. Furthermore, pre-trained LLMs already possess the capability to convert Roman text into the writing systems of various languages.

## Method

### Overall Architecture

Zero-AVSR consists of two core components: (1) **AV-Romanizer**, which predicts language-agnostic romanized text from multilingual audio-visual speech input; and (2) an **LLM decoder**, which converts the romanized text into target-language graphemes. The framework is instantiated in two forms: Cascaded and Unified.

### Key Designs

1. **MARC Dataset**: Integrates four datasets—LRS3, MuAViC, VoxCeleb2, and AVSpeech—to construct a multilingual romanized corpus covering 82 languages and 2,916 hours of audio-visual data. Romanization annotations are generated using GPT-4o-mini, which is empirically shown to achieve the best performance in romanization and de-romanization reconstruction tests. For unlabeled datasets, pre-trained language identification and ASR models are used to obtain language IDs and transcriptions.

2. **AV-Romanizer**: Based on the AV-HuBERT architecture, comprising an audio encoder $\mathcal{F}_a$ (linear layer), a visual encoder $\mathcal{F}_v$ (ResNet-18 + 3D convolution), a Transformer encoder $\mathcal{B}$ (24 layers), and a linear classification head. Audio features $f_a$ and visual features $f_v$ are concatenated along the channel dimension, projected via a linear layer, and fed into the Transformer encoder to obtain fused features $f_{av} = \mathcal{B}((f_a \oplus f_v)W)$. Trained with CTC loss to predict romanized text.

3. **Cascaded Zero-AVSR**: Cascades the AV-Romanizer with a pre-trained LLM (e.g., GPT-4o-mini) without fine-tuning the LLM. The AV-Romanizer converts audio-visual speech into Roman text, which is then passed to the LLM via instruction prompting to produce target-language graphemes. This approach is compatible with any LLM, including API-based models.

4. **Zero-AVSR (Unified Model)**: Embeds audio-visual features encoded by the AV-Romanizer directly into an LLM (Llama3.2-3B) and achieves end-to-end zero-shot recognition through multi-task training.

    - **Task 1 (Alignment)**: Uses a length compressor (1D convolution, kernel=2, stride=2) and an adapter to map audio-visual features into the LLM embedding space, trained on seen languages with a language modeling objective. The AV-Romanizer and LLM original weights are frozen; only LoRA weights, the compressor, and the adapter are trained.
    - **Task 2 (Learning De-romanization)**: A text-only task that trains the LLM to convert Roman text into target-language graphemes, covering both seen and unseen languages to prevent the LLM from forgetting multilingual capabilities. Only LoRA weights are trained.

### Loss & Training

- The AV-Romanizer is trained with CTC loss.
- Three-stage learning rate schedule: 10K warmup, 40K hold, 50K decay, with a peak learning rate of 1e-4.
- MUSAN noise is randomly added to audio during training (0 dB SNR).
- The LLM stage of Zero-AVSR uses a cosine scheduler (0.5K warmup + 29.5K decay) with QLoRA fine-tuning.
- Beam search (width=2, temperature=0.3) is used at inference.

## Key Experimental Results

### Main Results

| Method | Modality | Training Hours | Languages | Ara | Deu | Ell | Spa | Fra | Ita | Por | Rus | Eng | Avg (w/ Eng) |
|--------|----------|---------------|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|--------------|
| AV-HuBERT | AVSR | 1759h | 9 | 89.4 | 52.0 | 46.2 | 17.4 | 20.3 | 20.8 | 22.1 | 44.7 | 1.7 | 35.0 |
| XLAVS-R 2B | AVSR | 437Kh | 9 | 79.3 | 44.4 | 19.0 | 9.1 | 12.3 | 10.6 | 11.2 | 25.0 | 1.7 | 23.6 |
| MMS Zero-shot | ASR | 476Kh | 1078+ | 84.9 | 31.5 | 47.9 | 17.7 | 33.6 | 19.0 | 35.5 | 42.8 | 35.7 | 38.9 |
| Cascaded Zero-AVSR | AVSR | 2916h | 82+ | 82.1 | 29.3 | 47.2 | 16.3 | 28.9 | 21.6 | 20.2 | 42.9 | 2.9 | 30.2 |
| **Zero-AVSR** | AVSR | 2916h | 82+ | **81.4** | **27.8** | **38.4** | **13.1** | **14.3** | **15.9** | **15.4** | **32.6** | **1.5** | **25.2** |

### Ablation Study

**Effectiveness of MARC Dataset (Cascaded Zero-AVSR; target unseen language: Rus)**:

| Training Data | # Training Languages | Training Hours | Zero-shot CER↓ | Avg CER↓ |
|--------------|----------------------|---------------|----------------|---------|
| MuAViC | 8 | 745h | 62.3 | 48.3 |
| +MARC (8 langs) | 8 | 1944h | 61.0 | 28.3 |
| +MARC (40 langs) | 40 | 2418h | 49.5 | 25.1 |
| +MARC (81 langs) | 81 | 2793h | **40.0** | **21.9** |

**Effect of Different LLMs on Cascaded Zero-AVSR**:

| LLM | Avg CER↓ |
|-----|---------|
| Llama3.2-3B | 35.7 |
| Mistral-7B | 29.6 |
| Llama3.1-8B | 27.3 |
| Llama3.1-70B | 21.3 |
| GPT-4o-mini | **19.5** |

### Key Findings

- Zero-AVSR achieves an average WER of 25.2% using only 2,916 hours of audio-visual data, comparable to XLAVS-R 2B (23.6%) trained on 436K hours of audio data.
- Increasing language diversity (8 → 81 languages) substantially improves zero-shot performance, reducing the zero-shot CER on Russian from 62.3% to 40.0%.
- Data from linguistically related languages provides greater benefit for zero-shot performance, validating linguistic priors.
- Larger LLMs yield better decoding performance in the Cascaded setting.

## Highlights & Insights

- **Romanization as a language-agnostic representation**: Compared to phonemes, Roman text is simpler and LLMs inherently possess the ability to perform the conversion.
- **Practicality of the Cascaded approach**: No LLM fine-tuning is required, enabling direct use of closed-source API-based models.
- **Scaling effect of data volume and language diversity**: More languages and more data significantly enhance zero-shot capability.
- **LLM as a universal de-romanizer**: This eliminates the need to train separate language models for each target language.

## Limitations & Future Work

- Zero-shot performance still lags behind supervised methods, particularly for languages with highly divergent writing systems such as Arabic.
- The Unified Zero-AVSR employs the relatively small Llama3.2-3B; larger models may yield further improvements.
- Romanization is not a fully invertible transformation, introducing some information loss.
- Coverage of 82 languages remains incomplete with respect to all world languages, especially low-resource ones.

## Related Work & Insights

- **AV-HuBERT**: Foundational architecture for self-supervised audio-visual pre-training.
- **MMS Zero-shot**: A pioneer in zero-shot speech recognition in the audio domain.
- **XLAVS-R**: Multilingual audio-visual self-supervised learning.
- Inspiration: The romanization + LLM paradigm is potentially generalizable to other multilingual tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First zero-shot AVSR framework; the romanization + LLM design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 ablation experiments, comprehensive evaluation across 8 languages, and comparisons with multiple LLMs.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-motivated problem formulation.
- **Value**: ⭐⭐⭐⭐ Contributions in both methodology and dataset for multilingual audio-visual recognition across 82 languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition](../../NeurIPS2025/audio_speech/mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)
- [\[ICML 2026\] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](../../ICML2026/audio_speech/polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](../../NeurIPS2025/audio_speech/textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)

</div>

<!-- RELATED:END -->

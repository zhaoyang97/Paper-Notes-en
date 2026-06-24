---
title: >-
  [Paper Note] InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training
description: >-
  [ACL 2025][LLM Pretraining][SpeechLLM] This paper proposes InSerter (Interleaved Speech-Text Pre-training), which utilizes Text-to-Speech (TTS) to synthesize large-scale text corpora into interleaved speech-text sequences for pre-training. This significantly boosts the speech instruction-following capability of SpeechLLMs. Additionally, the first comprehensive speech instruction-following benchmark, SpeechInstructBench, is constructed.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "SpeechLLM"
  - "Speech Instruction Following"
  - "Interleaved Pre-training"
  - "Speech-Text Alignment"
  - "Benchmark"
date: 2026-05-08
content_hash: cd59c88a4b7c6740
---

# InSerter: Speech Instruction Following with Unsupervised Interleaved Pre-training

**Conference**: ACL 2025  
**arXiv**: [2503.02769](https://arxiv.org/abs/2503.02769)  
**Code**: [SpeechInstructBench](https://huggingface.co/datasets/ddwang2000/SpeechInstructBench)  
**Area**: LLM Pre-training  
**Keywords**: SpeechLLM, Speech Instruction Following, Interleaved Pre-training, Speech-Text Alignment, Benchmark

## TL;DR

This paper proposes InSerter (Interleaved Speech-Text Pre-training), which utilizes Text-to-Speech (TTS) to synthesize large-scale text corpora into interleaved speech-text sequences for pre-training. This significantly boosts the speech instruction-following capability of SpeechLLMs. Additionally, the first comprehensive speech instruction-following benchmark, SpeechInstructBench, is constructed.

## Background & Motivation

**Semantic Gap in Speech Instruction Following**: Current SpeechLLMs perform significantly worse when handling speech inputs compared to text inputs. The semantic inconsistency between speech and text modalities is the core bottleneck.

**Limitations of Representation Alignment**: Directly aligning continuous speech frames with discrete text token representations causes the loss of key acoustic features such as intonation, energy, and pitch, as their granularities are naturally mismatched.

**Poor Scalability of Behavioral Alignment**: Methods that align behaviors during post-training (forcing the model to generate consistent outputs for both speech and text inputs) rely on high-quality paired data, making data construction complex and hard to scale.

**Lack of Dedicated Evaluation**: Existing benchmarks (e.g., VoiceBench) mainly evaluate general conversation abilities, lacking a systematic evaluation of speech instruction following (including accents, noise, disfluencies, etc.).

**Inspiration from the Pre-training Stage**: The emergence of textual intelligence stems from unsupervised next-token prediction during the pre-training stage. The authors aim to transfer this mechanism to the speech modality.

**Low Training Efficiency**: Existing behavioral alignment methods only optimize text continuation for a single speech sequence per sample. InSerter achieves multi-segment alignment via an interleaved format, significantly improving training efficiency.

## Method

### Overall Architecture

InSerter adopts a two-stage training paradigm: (1) introducing large-scale interleaved speech-text data during the pre-training stage to enable speech representations to inherit the cognitive capabilities of text via next-token prediction; and (2) using dialogue data during the SFT stage to enhance interaction performance. The backbone model is Qwen2-Audio-7B (Whisper-Large-v3 encoder + Q-Former adapter + LLM).

### Module 1: Interleaved Data Construction

Three-stage pipeline:
- **Text Corpus Collection & Preprocessing**: Large-scale long texts and dialogue datasets are aggregated and cleaned using regular expressions, resulting in a text corpus of approximately 610 billion tokens.
- **Segment Sampling**: Random segments are selected from the text to be converted into speech, supporting two granularities: **word-level sampling** (randomly selecting words, with at least 5 words to ensure semantic integrity) and **sentence-level sampling** (randomly selecting sentences bounded by punctuation).
- **TTS Conversion**: CosyVoice 2.0 is used with prompt audios of 10,000 different timbres to synthesize speech, producing a total of 301,540 hours of speech data. The synthesized speech is concatenated with the remaining text to form interleaved sequences.

### Module 2: Interleaved Pre-training (Stage 1)

- Inputs are interleaved speech segments (encoded into continuous representations via a speech encoder + adapter) and text segments.
- The training objective is the standard cross-entropy loss, where the **loss is computed only on text tokens**, while tokens corresponding to speech segments are masked.
- Data mixture: 40% interleaved data + 30% multi-task speech data + 30% pure text data.
- Hyperparameters: Batch size of 1024, sequence length of 8192, trained for 1 epoch.

### Module 3: SFT Fine-tuning (Stage 2)

- Supervised fine-tuning is performed using dialogue datasets mixed with 50% text data, totaling 20K samples.
- Trained for 7000 iterations (the optimal step size determined by ablation studies), with a learning rate of 1e-5 and the Adam optimizer.

### Training Details

- The optimal ratio of speech segments is approximately 30% for word-level interleaving and 40% for sentence-level interleaving.
- The optimal proportion of interleaved data in the pre-training data is 40%.
- InSerter can be stacked with post-training alignment methods (such as continuation writing) to obtain further cumulative gains.

## Experiments

### Table 1: VoiceBench Main Results

| Model | AlpacaEval (S/T) | CommonEval (S/T) | OpenBookQA (S/T) | AdvBench RR (S/T) |
|------|-------------------|-------------------|-------------------|---------------------|
| Qwen2-Audio | 3.74/4.11 | 3.43/3.77 | 49.45/67.91 | 96.73/96.73 |
| DIVA | 3.67/4.68 | 3.54/4.29 | 25.49/76.70 | 98.27/99.23 |
| **InSerter** | **4.23**/4.39 | **3.63**/4.05 | **77.14**/83.52 | 97.69/97.50 |

- InSerter achieves an AlpacaEval score of 4.23 (optimal) and an OpenBookQA accuracy of 77.14% under speech input, significantly Outperforming Qwen2-Audio (49.45%).
- The gap between speech and text input performance is narrowed from 23.3% in the baseline to just 1.3%.

### Table 2: SpeechInstructBench Results (English Closed-Ended)

| Model | Standard (P/I) | Background (P/I) | Accent (P/I) | Disfluency (P/I) |
|------|-----------------|-------------------|--------------|-------------------|
| DIVA | 27.64/37.26 | 26.32/36.69 | 26.49/36.26 | 19.16/27.89 |
| Qwen2-Audio | 19.82/30.18 | 18.17/28.82 | 18.59/28.81 | 15.19/24.67 |
| **InSerter** | **39.75/51.35** | **37.56/49.87** | **37.34/48.24** | **36.38/47.28** |

- InSerter leads by a wide margin under all conditions, achieving a prompt-level accuracy of 39.75% under standard conditions (vs. DIVA's 27.64%), an improvement of approximately 12 percentage points.
- It maintains robustness in difficult scenarios such as noise interference, accent variations, and speech disfluencies.

### Key Findings

1. **Word-level Interleaving Outperforms Sentence-level**: Word-level granularity is finer and aligns better with the text continuation objective (47.38% vs. 42.98% I-Acc).
2. **Stackability**: Applying post-training with InSerter + continuation writing yields additional gains (51.35% vs. 47.38%).
3. **Positive Gains from Data Scaling**: Scaling the interleaved data from 0 to 300K hours consistently brings performance improvements, demonstrating robust scalability.
4. **Bilingual Chinese-English SOTA**: The proposed method also achieves state-of-the-art results on the Chinese subset of SpeechInstructBench.

## Highlights & Insights

- **Conceptually Simple and Scalable**: The method only requires TTS to convert text corpora into interleaved sequences to produce scale-ready training data, avoiding the need for meticulously designed paired data.
- **Intervening at the Pre-training Stage**: Speech-text alignment is introduced during the pre-training stage (rather than post-training), enabling speech to inherit textual intelligence more efficiently.
- **SpeechInstructBench**: The first systematic speech instruction-following benchmark, covering real-world scenarios such as accents, noise, emotions, and disfluencies, filling an evaluation gap.
- **High Training Efficiency**: The interleaved format allows aligning multiple speech-text segments within a single sample, making it far more efficient than behavioral alignment methods.

## Limitations & Future Work

- It only covers English and Chinese; the generalization capability to other languages remains unverified.
- The evaluation of Open-ended and Adjustment subtasks relies on GPT-4o API scoring, which limits objectivity and reproducibility.
- It relies on CosyVoice 2.0 synthesized speech. There exists a distribution gap between synthesized and real speech (though this issue is not discussed in the paper).
- The SFT stage uses only 20K samples. The small data size might limit conversation diversity.

## Related Work & Insights

- **SpeechLLM**: Categorized into discrete token pipelines (Moshi, AudioPaLM) and continuous representation pipelines (Qwen-Audio, DIVA). InSerter falls into the latter.
- **Speech Instruction Alignment**: Approaches include BLSP (behavioral alignment, continuation writing) and DIVA (representation alignment + distillation). InSerter intervenes starting from the pre-training stage.
- **Interleaved Pre-training**: Prior works like Spirit-LM use interleaved data to improve speech generation quality, whereas InSerter focuses on instruction following.
- **Evaluation Benchmarks**: Projects like VoiceBench and ADU-bench evaluate general capabilities, whereas SpeechInstructBench focuses on instruction following.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first systematic application of interleaved pre-training to continuous speech representation models. The data construction is simple and elegant.
- **Effectiveness**: ⭐⭐⭐⭐ — It significantly outperforms baselines across multiple benchmarks, features thorough ablations, and validates robust scalability.
- **Practical Value**: ⭐⭐⭐⭐ — The method is simple and general, and is directly applicable to any continuous-representation-based SpeechLLM.
- **Clarity**: ⭐⭐⭐⭐ — The paper is clearly structured with rich figures and tables, and the ablation studies are well-designed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unsupervised Morphological Tree Tokenizer](unsupervised_morphological_tree_tokenizer.md)
- [\[ACL 2025\] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)

</div>

<!-- RELATED:END -->

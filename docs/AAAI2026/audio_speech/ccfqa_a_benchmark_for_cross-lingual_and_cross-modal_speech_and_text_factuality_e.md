---
title: >-
  [Paper Note] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation
description: >-
  [AAAI2026][Audio & Speech][factuality evaluation] CCFQA is proposed—the first cross-lingual and cross-modal benchmark covering 8 languages and 14,400 fully parallel speech-text factuality QA samples. It supports four task settings (QA/XQA/SQA/XSQA) and systematically reveals the factual inconsistency of existing MLLMs under language and modality switching. Concurrently, LLM-SQA is introduced, achieving cross-lingual spoken QA transfer via English bridging with only five shots…
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "factuality evaluation"
  - "multilingual benchmark"
  - "spoken question answering"
  - "cross-lingual consistency"
  - "multimodal LLM"
date: 2026-05-08
content_hash: 848face12e92fc95
---

# CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation

**Conference**: AAAI2026  
**arXiv**: [2508.07295](https://arxiv.org/abs/2508.07295)  
**Code**: [yxduir/ccfqa](https://github.com/yxduir/ccfqa)  
**Area**: Audio and Speech  
**Keywords**: factuality evaluation, multilingual benchmark, spoken question answering, cross-lingual consistency, multimodal LLM

## TL;DR

CCFQA is proposed—the first cross-lingual and cross-modal benchmark covering 8 languages and 14,400 fully parallel speech-text factuality QA samples. It supports four task settings (QA/XQA/SQA/XSQA) and systematically reveals the factual inconsistency of existing MLLMs under language and modality switching. Concurrently, LLM-SQA is introduced, achieving cross-lingual spoken QA transfer via English bridging with only five shots, obtaining an F1 of 51.4 on XSQA, outperforming GPT-4o-mini-Audio (45.7).

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have been widely deployed in multilingual scenarios, but the evaluation of their factual reliability lags significantly behind. Existing factuality benchmarks (e.g., SimpleQA, TruthfulQA, HaluEval) primarily target English text, while the multilingual benchmark KoLasSimpleQA covers 9 languages but is restricted to text. Similarly, speech-side benchmarks (e.g., SD-QA, VoiceBench, SpeechIQ) only address a single spoken modality and are mostly in English.

**Limitations of Prior Work**: When the same factual question is posed in different languages (cross-lingual inconsistency) or presented in different modalities (cross-modal inconsistency), MLLMs frequently generate contradictory answers. For instance, a question correctly answered by the GPT-4o series in English text might be answered incorrectly when presented in Japanese speech. Currently, there is a lack of benchmarks with fully parallel data to systematically measure such inconsistency.

**Key Challenge**: The intersections of multilingual and multimodal scenarios lead to a combinatorial explosion of the evaluation space ($8 \text{ languages} \times 2 \text{ modalities} \times \text{cross-lingual combinations} = 128,000 \text{ requests}$ required to fully evaluate a single MLLM). Constructing such high-quality parallel datasets is extremely costly, as it requires human recordings by native speakers of each language.

**Goal**: To build the first factuality evaluation benchmark that simultaneously covers cross-lingual and cross-modal scenarios, and to propose a low-cost transfer scheme that leverages the powerful factual reasoning capabilities of English LLMs.

**Key Insight**: Utilizing existing English QA datasets (MKQA + MOOCCubeX) $\rightarrow$ translation with GPT-4.1 $\rightarrow$ human verification $\rightarrow$ native speaker recording $\rightarrow$ ASR quality control to build high-quality parallel data; on the model side, utilizing English bridging + 5-shot transfer to break through language barriers.

**Core Idea**: Fully parallel speech-text multilingual data + English-bridged few-shot transfer = systematically diagnosing and mitigating factual inconsistency in MLLMs across languages and modalities.

## Method

### Overall Architecture

CCFQA comprises two orthogonal dimensions: **Benchmark Construction** (to diagnose the problem) and the **LLM-SQA Model** (to mitigate the problem). The benchmark generates 14,400 parallel samples through a three-stage process: cross-lingual data construction $\rightarrow$ cross-modal data construction $\rightarrow$ quality control. LLM-SQA undergoes pre-training via curriculum learning (ASR $\rightarrow$ SRT $\rightarrow$ SQA) followed by a 5-shot cross-lingual transfer, leveraging English factual knowledge to serve non-English spoken question answering.

### Key Design 1: Cross-Lingual Data Construction—Rigorous Screening + Machine Translation + Human Verification

The data source originates from MKQA (open-domain QA) and MOOCCubeX (educational domain), covering 20 subfields across 4 major categories: humanities, social sciences, natural sciences, and applied sciences. Key screening criteria: excluding ambiguous questions (e.g., "Who is the current prime minister?"), sensitive content (PII/offensive language), factually incorrect QA pairs, and culture-dependent questions (e.g., "What is the legal age of marriage?"). The screened English QA pairs were translated into 7 target languages (Chinese, French, Japanese, Korean, Russian, Spanish, Cantonese) using GPT-4.1. Human verification was conducted for Chinese, English, and Japanese, while back-translation + review was applied to the remaining languages to ensure accuracy.

### Key Design 2: Cross-Modal Data Construction—Human Recording + Iterative ASR Quality Control

Native volunteers from 8 languages (gender-balanced) were recruited to record readings following clean, natural, steady-paced, and neutral-toned specifications. Audio enhancement was applied to low-volume samples, after which Whisper-large-v3 was utilized for ASR transcription to calculate and compare WER/CER against the original text. Audio with WER exceeding the threshold was flagged as anomalous and re-recorded. This iterative process of "recording $\rightarrow$ ASR verification $\rightarrow$ re-recording" ensures precise alignment between speech and text. The final speech quality achieved an English WER of only 3.2% and a Chinese CER of 6.8%.

### Key Design 3: LLM-SQA Model—Curriculum Learning + English-Bridged 5-Shot Transfer

Model Architecture: Frozen Whisper speech encoder (~635M) + trainable Q-Former adapter (80 queries, dim=768) + MLP (totaling ~80.5M) + GemmaX2-9B LLM (~9.2B). The LLM is fine-tuned using LoRA ($r=16, \alpha=32$), training only ~8.9M parameters.

The training strategy employs a three-stage curriculum learning:
1. **ASR Stage**: Speech recognition training on the FLEURS dataset to establish the foundation of speech-text alignment.
2. **SRT Stage**: Training speech recognition + translation to learn cross-lingual mapping.
3. **SQA Stage**: Divided into two steps: (a) supervised fine-tuning using ~3,000 synthetic English speech-text pairs to learn the QA task structure; (b) applying 5-shot cross-lingual transfer for each target language.

Core Mechanism: Through special instruction tokens (e.g., `<|qa|><|fra|>`), the model is guided to internally translate non-English spoken questions into English for factual reasoning, then translate the answer back to the target language. This leverages the LLM's strongest knowledge reservoir (which is in English) while minimizing the requirement for non-English annotated data.

### Loss & Training

Standard autoregressive cross-entropy loss is utilized across all stages. ASR and SRT are trained on FLEURS, while SQA is trained on synthetic English data + 5-shot target language data. Training is conducted on 4$\times$A100 (80GB) for one week. The optimizer used is AdamW, with a peak learning rate of $1 \times 10^{-4}$, 1000 warmup steps, and linear decay.

## Key Experimental Results

### Main Results (F1 / LLM Acc)

| Model | Text QA Avg | Text XQA Avg | Speech SQA Avg | Speech XSQA Avg |
|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini | 63.9 / 64.4 | 59.7 / 62.2 | — | — |
| GPT-4o-mini-Audio | — | — | 47.7 / 40.4 | 45.7 / 38.6 |
| Phi-4-Multimodal | 18.0 / 13.8 | 18.0 / 15.3 | 18.5 / 22.0 | 21.0 / 5.7 |
| Qwen2-Audio | 27.9 / 30.6 | 24.2 / 19.2 | 27.7 / 17.0 | 24.1 / 10.7 |
| Qwen2.5-Omni-3B | 20.6 / 11.8 | — | 34.9 / 20.9 | 29.8 / 17.1 |
| Qwen2.5-Omni-7B | 46.5 / 38.2 | 42.1 / 34.5 | 44.0 / 33.2 | 38.5 / 29.5 |
| **LLM-SQA (Ours)** | — | — | **52.0 / 40.3** | **51.4 / 39.7** |

### Cross-Lingual and Cross-Modal Consistency (%, higher means more consistent)

| Model | Cross-Lingual Consistency | Cross-Modal Consistency |
|------|:---:|:---:|
| GPT-4o-mini | 96.6 / 95.5 | 62.7 / 62.1 |
| Phi-4-Multimodal | 90.8 / 25.9 | 62.7 / 37.5 |
| Qwen2-Audio | 62.4 / 62.9 | 55.6 / 56.0 |
| Qwen2.5-Omni-3B | 75.4 / 81.8 | 56.5 / 52.0 |
| Qwen2.5-Omni-7B | 90.3 / 87.2 | 90.3 / 85.5 |

### Effect of Speech Length on SQA Accuracy (LLM Acc)

| Model | 0-5s | 5-10s | 10-30s | Average |
|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini-Audio | 38.6 | 42.6 | 40.1 | 40.4 |
| LLM-SQA (Ours) | 40.3 | 40.8 | 36.1 | 40.3 |
| Qwen2.5-Omni-7B | 32.1 | 34.9 | 29.0 | 33.2 |
| Qwen2-Audio | 14.7 | 19.5 | 18.6 | 17.0 |
| Phi-4-Multimodal | 17.7 | 26.4 | 27.4 | 22.0 |

### Key Findings

- LLM-SQA surpasses all open-source baselines on both SQA and XSQA, with XSQA F1 (51.4) significantly outperforming GPT-4o-mini-Audio (45.7)—proving the high effectiveness of the 5-shot transfer strategy.
- Cross-lingual Challenge: From QA $\rightarrow$ XQA, most models exhibit significant performance degradation, with GPT-4o-mini being the most stable (96.6% consistency).
- Cross-modal Challenge: There is a widespread "modal gap" from text $\rightarrow$ speech. Qwen2.5-Omni-7B achieves the highest cross-modal consistency (90.3%), benefiting from its Omni architecture design.
- High F1 but low LLM Acc $\rightarrow$ hallucination problem (fluent responses but factually incorrect); low F1 but high Acc $\rightarrow$ instruction-following problem (knowing the answer but failing on standardized formatting).
- Language Dimension: English, French, and Spanish perform the best, while Korean and Cantonese perform the worst, reflecting biases in pre-training data distribution.
- Speech Quality: English WER is 3.2%, whereas some languages (French 13.8%, Russian 18.2%, Cantonese 16.8%) exhibit higher error rates due to proper nouns.

## Highlights & Insights

- **Filling the Evaluation Gap**: CCFQA is the only factuality benchmark that simultaneously supports cross-lingual (8 languages) and cross-modal (text + speech) scenarios with fully parallel data. It requires 128K requests for a comprehensive evaluation, offering unprecedented evaluation dimensionality.
- **Strong Diagnostic Capability**: The two orthogonal dimensions—cross-lingual consistency and cross-modal consistency—precisely pinpoint model weaknesses. For instance, Phi-4-Multimodal shows a cross-lingual LLM Acc consistency of only 25.9%, exposing its flaws in multilingual token design.
- **Extremely Low Annotation Cost Transfer Scheme**: LLM-SQA achieves cross-lingual spoken QA transfer with only 5-shot prompts by utilizing English as a "universal interface" for factual knowledge. The approach is simple yet remarkably effective.
- **Decoupled Analysis of F1 and LLM Acc**: It reveals two distinct failure modes—hallucination (high F1, low Acc) and instruction-following failure (low F1, high Acc)—providing a new diagnostic tool for model analysis.
- **High-Quality Speech Data**: The iterative quality control workflow of human recording + ASR ensures the reliability of the evaluation.

## Limitations & Future Work

- Language coverage (8 languages) remains limited, lacking important low-resource and tonal languages such as Arabic, Hindi, etc. (except Cantonese).
- The speech data consists of read-speech recorded in laboratory environments and does not cover real-world scenarios such as accented speech, noise, or natural conversations, potentially overestimating practical model performance.
- The English-bridging strategy of LLM-SQA introduces an English-centric bias; the transfer efficacy may hit a bottleneck for languages with substantial linguistic distance from English (e.g., Korean).
- The scale of the benchmark (1,000 test pairs $\times$ 8 languages) has a limited sample size in certain subdomains.
- It only evaluates factual QA, leaving more complex tasks such as speech reasoning and speech summarization unexplored.
- End-to-end multilingual speech training (instead of English-bridging) under large-scale data has not been explored to see if it delivers better performance.

## Related Work & Insights

- **vs SimpleQA / Chinese SimpleQA**: Only monolingual text. CCFQA extends this to 8 languages + speech modality, enabling cross-modal factual consistency evaluation for the first time.
- **vs KoLasSimpleQA**: Covers 9 languages but is limited to text, and the QAs are not fully parallel. CCFQA's fully parallel design allows precise quantification of cross-lingual consistency.
- **vs VoiceBench / SpeechIQ**: Involve only English speech and non-open-ended responses. CCFQA supports open-ended spoken QA in 8 languages.
- **vs SD-QA**: Spoken QA across 5 dialects/languages but does not support cross-lingual evaluation. CCFQA adds the evaluation dimensions of XQA/XSQA.

| Benchmark | No. of Languages | Modality | Data Size | Supported Tasks |
|------|:---:|:---:|:---:|:---:|
| SimpleQA | 1 | Text | 4,326 | QA |
| KoLasSimpleQA | 9 | Text | 2,147 | QA |
| VoiceBench | 1 | Speech | 5,783 | SQA |
| SD-QA | 5 | Speech | 11,109 | SQA |
| **CCFQA** | **8** | **Text + Speech** | **14,400** | **QA/XQA/SQA/XSQA** |

**Insights**: The English-bridging strategy is highly effective in transferring factual knowledge and can be generalized to other knowledge-intensive multilingual tasks. The progressive curriculum learning (ASR $\rightarrow$ SRT $\rightarrow$ SQA) can be borrowed for multimodal alignment scenarios. The "modality gap" phenomenon warrants deeper research—why does spoken input systematically degrade factual accuracy?

## Rating

⭐⭐⭐⭐ (4/5)

Overall Evaluation: The first cross-lingual and cross-modal factuality evaluation benchmark, filling an important gap. The benchmark construction process is rigorous (GPT translation + human verification + human recording + ASR quality control), the evaluation dimensions are comprehensive (4 tasks $\times$ 2 metrics $\times$ 8 languages), and the 5-shot transfer scheme of LLM-SQA is simple yet effective. The primary limitations are that the language coverage and diversity of speech scenarios remain limited, and the model innovation (English bridging) is somewhat incremental.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Probing Cross-modal Information Hubs in Audio-Visual LLMs](../../ICML2026/audio_speech/probing_cross-modal_information_hubs_in_audio-visual_llms.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ICML 2026\] Position: Towards Responsible Evaluation for Text-to-Speech](../../ICML2026/audio_speech/position_towards_responsible_evaluation_for_text-to-speech.md)
- [\[ICLR 2026\] TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems](../../ICLR2026/audio_speech/ttsds2_resources_and_benchmark_for_evaluating_human-quality_text_to_speech_syste.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)

</div>

<!-- RELATED:END -->

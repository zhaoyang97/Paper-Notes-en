---
title: >-
  [Paper Note] Multimodal In-Context Learning for ASR of Low-Resource Languages
description: >-
  [ACL 2026][Audio & Speech][Multimodal In-Context Learning] This paper systematically investigates whether Multimodal In-Context Learning (MICL) enables Speech LLMs to learn unseen endangered languages. It proposes an MIC…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Multimodal In-Context Learning"
  - "Low-resource ASR"
  - "Speech Large Language Models"
  - "Cross-lingual Transfer"
  - "Hypothesis Selection"
date: 2026-05-08
content_hash: 1b4d936531bafeb1
---

# Multimodal In-Context Learning for ASR of Low-Resource Languages

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.05707](https://arxiv.org/abs/2601.05707)  
**Code**: [github](https://github.com/ZL-KA/MICL)  
**Area**: Audio & Speech / Low-resource ASR  
**Keywords**: Multimodal In-Context Learning, Low-resource ASR, Speech Large Language Models, Cross-lingual Transfer, Hypothesis Selection

## TL;DR

This paper systematically investigates whether Multimodal In-Context Learning (MICL) enables Speech LLMs to learn unseen endangered languages. It proposes an MICL-based hypothesis selection system that combines the complementary strengths of acoustic models and Speech LLMs, achieving significant ASR performance improvements across three endangered languages.

## Background & Motivation

**Background**: Among 7000+ languages globally, current ASR systems cover only a tiny fraction, primarily due to the scarcity of labeled data. While Speech Large Language Models (e.g., Phi4, Qwen3-Omni) possess powerful multi-task capabilities, their performance remains limited to the high-resource languages covered during training.

**Limitations of Prior Work**: (1) Existing ICL research focuses mainly on the text modality and high-resource languages; (2) The effectiveness of Multimodal ICL (MICL) for Speech LLMs on uncovered languages is under-explored; (3) Direct use of Speech LLMs for prompt-based ASR on unseen languages yields extremely poor results (WER > 100%).

**Key Challenge**: While Speech LLMs have powerful in-context learning capabilities, it remains unclear how to effectively leverage this ability for data-scarce endangered languages.

**Goal**: To verify the effectiveness of MICL for unseen languages, analyze its internal mechanisms, and construct a practical ASR system.

**Key Insight**: Conduct systematic experiments comparing three modality settings—text ICL, audio+text ICL, and multimodal ICL—evaluating two Speech LLMs across three endangered languages from different language families.

**Core Idea**: Although MICL cannot directly enable Speech LLMs to generate high-quality transcriptions, it can be combined with acoustic models through hypothesis selection. This leverages the language understanding capabilities of MICL to rerank candidate transcriptions.

## Method

### Overall Architecture

(1) MICL Analysis: Three prompt modes are designed—T-ICL (text-only), ICL (text + target audio), and MICL (audio-text pairs + target audio)—evaluated via perplexity; (2) Cross-lingual Fine-tuning: Fine-tuning is performed on 143 auxiliary languages (excluding target languages) to test transfer effects; (3) Hypothesis Selection System: An MMS acoustic model generates N-best candidates, and the Speech LLM calculates language model scores via MICL for joint reranking to select the optimal hypothesis.

### Key Designs

1.  **Three Prompt Modality Designs**: T-ICL ($c_i = t_i$, text-only examples) measures the contribution of textual context; ICL ($c_i = t_i$ + target audio $a^*$) isolates the role of target audio; MICL ($c_i = (a_i, t_i)$ + $a^*$) tests the marginal gain of paired audio-text examples. Quantifying marginal contributions is achieved by comparing these three.

2.  **Cross-lingual Instruction Fine-tuning (XFT)**: Performs MICL instruction fine-tuning on 143 languages from ML-SUPERB 2.0 (excluding target languages). LoRA is used to fine-tune only decoder parameters, with 1-10 context samples randomly selected during training. Motivation: To enable the model to follow MICL prompt formats and utilize context information more effectively, rather than learning the target language itself.

3.  **MICL Hypothesis Selection System**: Given a 10-best candidate list from MMS, reranking is performed using a joint score: $\hat{h} = \arg\max_{h^{(k)}} [\text{Acoustic\_score}(h^{(k)}) + \text{LM\_score}_{MICL}(h^{(k)})]$, where the LM score is the log-likelihood of the Speech LLM under MICL conditions. Design Motivation: Acoustic models excel at basic recognition while Speech LLMs excel at contextual understanding; the two are complementary.

### Loss & Training

During fine-tuning, loss is calculated only on target transcription tokens, with context examples serving as conditional inputs. LoRA adapters are used while freezing remaining parameters. Evaluation metrics: Perplexity (PPL, for configuration selection) and Word Error Rate (WER, for final evaluation).

## Key Experimental Results

### Main Results (Qwen3-Omni Perplexity, Pre-trained Model)

| Language | Task | 0-shot | 1-shot | 5-shot | 10-shot | 50-shot | 100-shot |
|---|---|---|---|---|---|---|---|
| Khinalug | T-ICL | 1302 | 289 | 69 | 57 | 44 | 43 |
| Khinalug | ICL | 54 | 28 | 11 | 10 | 11 | 15 |
| Khinalug | MICL | 58 | 30 | 9 | 10 | 8 | 13 |
| Kichwa | ICL | 18 | 10 | 5 | 4 | 3 | 3 |
| Kichwa | MICL | 17 | 7 | 4 | 4 | 3 | 4 |
| Mboshi | ICL | 178 | 51 | 21 | 16 | 10 | 9 |
| Mboshi | MICL | 189 | 34 | 13 | 10 | 7 | 9 |

### Hypothesis Selection WER Results

| Model | Khinalug | Kichwa | Mboshi |
|---|---|---|---|
| Acoustic Model (MMS) | 42.1 | 17.3 | 31.4 |
| Phi4 ASR-FT | 41.5 | 17.4 | 29.9 |
| Phi4 XFT | 41.0 | 17.1 | 29.6 |
| Phi4 TFT | 40.8 | 16.6 | 28.6 |
| Qwen3-Omni | 40.7 | 17.2 | 30.0 |
| N-gram LM | 39.6 | 17.7 | 30.6 |
| Oracle | 36.5 | 12.4 | 22.1 |

### Key Findings

*   MICL enables both Speech LLMs to learn unseen languages, with perplexity consistently decreasing as context samples increase.
*   Qwen3-Omni consistently benefits from audio examples in long-context scenarios (100 samples), while Phi4 primarily benefits in short-context scenarios (≤3 samples).
*   Attention analysis reveals the model allocates more attention to text (65-70%) than audio (30-35%), showing a layer-dependent pattern.
*   Cross-lingual fine-tuning on Kichwa approaches the performance of target-language fine-tuning, indicating that linguistic diversity enhances generalization.

## Highlights & Insights

*   **MICL Effectiveness on Unseen Languages**: This is the first systematic demonstration that Speech LLMs can learn uncovered endangered languages through multimodal in-context learning.
*   **Attention Mechanism Analysis**: Discovery of layer-dependent modality preference patterns—shallow and deep layers prefer audio, while middle layers prefer text.
*   **Practical System Design**: The acoustic model + Speech LLM hypothesis selection system is simple and effective, requiring no end-to-end training.

## Limitations & Future Work

*   Due to computational constraints, cross-lingual fine-tuning was only performed on Phi4.
*   The number of context samples during fine-tuning was limited to 1-10, which may constrain long-context effectiveness.
*   The data volume for the three endangered languages is very small (2-4 hours); the generalizability of the conclusions requires further verification.
*   Future work could explore larger-scale cross-lingual instruction fine-tuning and a broader range of endangered languages.

## Related Work & Insights

*   Multimodal extension of text-based ICL for low-resource languages (Li & Niehues, 2025b).
*   The hypothesis selection approach can be generalized to other low-resource multimodal tasks.
*   Attention analysis findings are consistent with the text-bias phenomenon observed in Vision LLMs.

## Rating

*   **Novelty**: ⭐⭐⭐⭐ First systematic study of MICL effects in ASR for endangered languages, offering a unique perspective.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across 3 languages × 2 models × various ICL settings × attention analysis.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear and systematic experimental design with rigorous comparative logic across settings.
*   **Value**: ⭐⭐⭐⭐ Provides a new technical path for ASR in endangered languages, possessing significant social value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages](hard_to_be_heard_phoneme-level_asr_analysis_of_phonologically_complex_low-resour.md)
- [\[ACL 2026\] DRInQ: Evaluating Conversational Implicature with Controlled Context Variation](drinq_evaluating_conversational_implicature_with_controlled_context_variation.md)
- [\[ICML 2026\] Algorithmic Recourse of In-Context Learning for Tabular Data](../../ICML2026/audio_speech/algorithmic_recourse_of_in-context_learning_for_tabular_data.md)
- [\[ACL 2026\] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)
- [\[ICML 2026\] Multiple Choice Learning of Low-Rank Adapters for Language Modeling](../../ICML2026/audio_speech/multiple_choice_learning_of_low-rank_adapters_for_language_modeling.md)

</div>

<!-- RELATED:END -->

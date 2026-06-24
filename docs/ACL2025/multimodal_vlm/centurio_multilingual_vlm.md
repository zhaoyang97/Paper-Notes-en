---
title: >-
  [Paper Note] Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model
description: >-
  [ACL 2025][Multimodal VLM][multilingual LVLM] This paper systematically investigates three dimensions of multilingual LVLM training strategies: the number of training languages, training data distribution, and multilingual OCR. It discovers that 100 languages can be trained simultaneously using only 25-50% non-English data, based on which Centurio, a state-of-the-art model covering 100 languages, is trained.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "multilingual LVLM"
  - "vision-language"
  - "training data distribution"
  - "OCR"
  - "language fidelity"
date: 2026-05-08
content_hash: 32bf7fea52f4ea2d
---

# Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model

**Conference**: ACL 2025  
**arXiv**: [2501.05122](https://arxiv.org/abs/2501.05122)  
**Code**: [gregor-ge.github.io/Centurio](https://gregor-ge.github.io/Centurio)  
**Area**: Multimodal VLM  
**Keywords**: multilingual LVLM, vision-language, training data distribution, OCR, language fidelity  

## TL;DR

This paper systematically investigates three dimensions of multilingual LVLM training strategies: the number of training languages, training data distribution, and multilingual OCR. It discovers that 100 languages can be trained simultaneously using only 25-50% non-English data, based on which Centurio, a state-of-the-art model covering 100 languages, is trained.

## Background & Motivation

**Problem Definition:** Most current LVLMs are trained primarily on English data, causing difficulties in understanding non-English inputs, producing outputs in the incorrect language, and failing to recognize non-English text in images. How to design the optimal multilingual training data distribution under a limited training budget?

**Limitations of Prior Work:** Existing multilingual LVLM works (e.g., Geigle et al., 2023; Sun et al., 2024; Maaz et al., 2024) adopt ad-hoc strategies when adding multilingual data, lacking systematic insights into how different training ratios affect performance across various language groups.

**Core Motivation:** Training data volume is always limited by time, computing resources, and cost. Under a fixed training budget, four key questions need to be answered: (RQ1) How many training languages can be included without degrading English performance? (RQ2-3) What is the optimal language distribution in pre-training and instruction tuning? (RQ4) How to improve multilingual visual text understanding?

## Method

### Overall Architecture

The LLaVA architecture is adopted, using SigLIP SO400/384 as the image encoder, Phi 3.5 (3.8B) as the LLM backbone, and aligning visual and textual spaces via a two-layer MLP. The training consists of two stages:
1. **Pre-training:** Trained on image description data (ShareGPT4v, 1.3M samples).
2. **Instruction Tuning:** Trained on diverse vision-language task data (adapted from LLaVA-Next, 0.77M samples).

The open-source machine translation model NLLB is used to translate English data into other languages, and evaluation covers 13 downstream tasks and 43 languages.

### Key Designs

1. **Progressive Language Scalability Experiments (RQ1):** Gradually expanding from high-resource language groups (T5: 6 languages) to T5-T4 (24 languages) $\rightarrow$ T5-T3 (52 languages) $\rightarrow$ T5-T2 (69 languages) $\rightarrow$ L100 (99 languages), while keeping the total data volume constant, to observe changes in performance.
2. **Language Distribution Search (RQ2-3):** Fixing the number of languages to 100, adjusting the English ratio E from 1% to 90% to find the optimal balance point. Pre-training and instruction tuning are searched independently.
3. **Multilingual OCR Enhancement (RQ4):** Introducing the SMPQA (Synthetic Multilingual Plot QA) benchmark, which covers 11 languages and 7 script systems, and using Synthdog to generate synthetic OCR training data.

### Loss & Training

The standard autoregressive language modeling loss (next-token prediction) is utilized. The image encoder is frozen in all stages, and only the MLP and LLM parameters are updated (using LoRA). The image encoder is additionally unfrozen during the OCR training phase.

## Key Experimental Results

### RQ1: Number of Training Languages (Instruction Tuning Phase, 50% English)

| Training Language Group | T1 (Lowest Resource) | T2 | T3 | T4 | T5 | en |
|-----------|------|------|------|------|------|------|
| English only | 14.4 | 30.4 | 24.4 | 23.6 | 28.5 | 53.6 |
| T5 (6 languages) | 16.5 | 31.0 | 26.3 | 26.7 | 34.0 | 53.7 |
| T5-T4 (24 languages) | 17.4 | 30.6 | 27.9 | 29.6 | 33.5 | 51.5 |
| L100 (99 languages) | 19.3 | 32.6 | 30.7 | 28.9 | 34.4 | 52.6 |

### RQ2: English Data Ratio in Instruction Tuning

| English Ratio | T1 | T2 | T5 | en |
|---------|------|------|------|------|
| 1% | 19.1 | 30.3 | 31.7 | 48.9 |
| 25% | 19.7 | 35.5 | 33.0 | 50.3 |
| 50% | 19.3 | 32.6 | 34.4 | 52.6 |
| 90% | 15.9 | 31.2 | 34.1 | 54.8 |

### Key Findings

1. **No "Curse of Multilinguality":** Expanding from 7 to 100 training languages barely affects the performance of already included languages, while newly added languages receive significant improvements. Language fidelity increases from $<1\%$ to $>95\%$.
2. **A Small Amount of Multilingual Data is Effective:** Only 25-50% of non-English data is sufficient to substantially boost multilingual capabilities, while more non-English data can sometimes degrade performance.
3. **Multilingual Data is More Critical in Pre-training:** Multilingual pre-training significantly improves low-resource languages (T1/T2). Reducing the English data ratio from 100% to 1% does not significantly harm English performance.
4. **OCR Data Has Limited Benefits for Non-Latin Scripts:** Synthetic OCR data shows significant efficacy for Latin-script languages but leaves a substantial performance gap for non-Latin scripts (Arabic, Chinese, etc.), which may require orders of magnitude more training data.

## Highlights & Insights

- The most systematic study on multilingual LVLM training strategies to date, covering four research questions and 100 languages.
- Discovers that "language exposure is more critical than data volume"—a small amount of multilingual data can activate the multilingual capabilities of the underlying LLM.
- Introduces the SMPQA benchmark to fill the gap in multilingual OCR evaluation.
- The final model, Centurio, achieves SOTA performance across 14 tasks and 56 languages, significantly outperforming Qwen2-VL and InternVL 2.5, especially on low-resource languages.
- Utilizes Llama 3 as an additional backbone to validate the generalizability of key findings.

## Limitations & Future Work

- The multilingual training data is obtained through machine translation. The translation quality for low-to-medium resource languages is limited, potentially leading to an underestimation of actual effectiveness.
- The vision encoder SigLIP has limited capability in representing non-Latin scripts visually, constraining the effectiveness of OCR enhancement.
- Computational budget constraints prevent exhausting all combinations (e.g., joint search of language distributions across pre-training and instruction tuning).
- Experiments are conducted solely on the LLaVA architecture; the generalizability of the conclusions to other architectures remains unverified.

## Related Work & Insights

- **Multilingual LVLMs:** PALO (Maaz et al., 2024) supports 10 languages, and Pangea (Yue et al., 2024) covers 39 languages.
- **Vision-Language Pre-training:** LLaVA (Liu et al., 2023), LLaVA-Next (Liu et al., 2024).
- **Cross-lingual Transfer:** Shaham et al. (2024) and Chen et al. (2024) investigate few-language training + zero-shot transfer.
- **Multilingual Text Understanding:** MTVQA (Tang et al., 2024), Synthdog (Kim et al., 2022) for synthetic OCR data.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ACL 2025\] SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning](singakids_a_multilingual_multimodal_dialogic_tutor_for_language_learning.md)
- [\[ACL 2025\] ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](chartcoder_chart_to_code.md)
- [\[ACL 2025\] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus](cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)

</div>

<!-- RELATED:END -->

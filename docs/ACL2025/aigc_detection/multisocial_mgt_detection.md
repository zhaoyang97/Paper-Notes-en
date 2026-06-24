---
title: >-
  [Paper Note] MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts
description: >-
  [ACL 2025][AIGC Detection][machine-generated text detection] Constructed MultiSocial (472k texts), the first large-scale machine-generated text detection benchmark covering 22 languages, 5 social media platforms, and 7 LLM generators. Experimental results show that fine-tuned detectors (Llama-3-8B/Mistral-7B, AUC ROC 0.977) perform exceptionally well on social media texts, and the choice of training platforms significantly impacts cross-platform generalization.
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "machine-generated text detection"
  - "multilingual"
  - "social media"
  - "benchmark dataset"
  - "cross-platform"
date: 2026-05-08
content_hash: 29a9a12d1cc973d9
---

# MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts

**Conference**: ACL 2025  
**arXiv**: [2406.12549](https://arxiv.org/abs/2406.12549)  
**Code**: [https://github.com/kinit-sk/multisocial](https://github.com/kinit-sk/multisocial)  
**Area**: AIGC Detection  
**Keywords**: machine-generated text detection, multilingual, social media, benchmark dataset, cross-platform

## TL;DR
Constructed MultiSocial (472k texts), the first large-scale machine-generated text detection benchmark covering 22 languages, 5 social media platforms, and 7 LLM generators. Experimental results show that fine-tuned detectors (Llama-3-8B/Mistral-7B, AUC ROC 0.977) perform exceptionally well on social media texts, and the choice of training platforms significantly impacts cross-platform generalization.

## Background & Motivation

**Background**: LLMs can generate high-quality multilingual texts, which is often abused on social media for spreading misinformation, social engineering attacks, etc. Machine-generated text detection (MGTD) acts as a critical line of defense against LLM abuse.

**Limitations of Prior Work**: Existing MGTD research focuses almost exclusively on English and long texts (news, academic papers, student essays), heavily neglecting the unique features of social media texts, such as extremely short length, informal language, emojis/hashtags, and frequent grammatical errors. Existing multilingual datasets either do not cover social media (e.g., MULTITuDE only covers news, M4GT-Bench only covers English Reddit) or are limited to a single platform in English (e.g., TweepFake only covers English tweets).

**Key Challenge**: The informal characteristics of social media texts may cause detectors trained on formal texts to fail. Generated social media texts might be "more formal" (with correct grammar and standard wording) than human-written ones, introducing detector bias. Furthermore, text characteristics vary drastically across different platforms (e.g., Discord/Telegram/Twitter) and languages, lacking a systematic cross-platform and cross-language evaluation.

**Goal**: (1) Construct a multi-language, multi-platform, and multi-generator social media MGTD benchmark dataset; (2) systematically evaluate the performance of three types of detection methods (statistical, pre-trained, and fine-tuned) under social media scenarios; (3) study cross-lingual and cross-platform detection generalization capabilities.

**Key Insight**: Target social media scenarios to generate machine text through paraphrasing rather than direct generation (by applying 3 rounds of paraphrasing with 7 LLMs on each human-written text). This ensures that generated texts are highly aligned with the original texts in style and topic, avoiding detection bias.

**Core Idea**: The first 22 languages × 5 platforms × 7 generators social media MGTD benchmark, systematically proving that fine-tuned detectors are effective on social media texts and that platform selection affects cross-platform generalization.

## Method

### Overall Architecture
Data construction pipeline: Collect human-written texts from 5 platforms (Telegram/Twitter/Gab/Discord/WhatsApp) $\rightarrow$ perform 3 rounds of paraphrase using 7 LLMs (Aya-101, Gemini, GPT-3.5-Turbo, Mistral-7B, OPT-IML-Max-30B, v5-Eagle-7B, Vicuna-13B) to generate machine-generated texts $\rightarrow$ quality assessment + noise labeling $\rightarrow$ partition training/test sets $\rightarrow$ evaluate on 17 detection methods.

### Key Designs

1. **Multilingual Multi-platform Collection**:

    - **Function**: Cover authentic social media texts from 22 languages and 5 social platforms.
    - **Mechanism**: Collect 58K human-written texts from existing multilingual social media datasets, covering 4 language families including Indo-European (18 languages), Uralic (2 languages), Semitic (Arabic), and Sino-Tibetan (Chinese), and 5 writing systems (Latin/Cyrillic/Arabic/Han/Greek). The training set covers 18 languages, while the test set extends to 22 (including test-only languages such as Irish and Scottish Gaelic).
    - **Design Motivation**: Ensure training/test languages do not overlap completely to evaluate cross-lingual generalization; address the uneven availability of the same language across different platforms (e.g., Chinese only on Telegram) through subset combinations to study different dimensions.

2. **Paraphrase-based Text Generation Strategy**:

    - **Function**: Generate machine text that is highly style- and topic-aligned with human-written text to avoid topic bias.
    - **Mechanism**: For each human-written text, perform 3 iterative rounds of paraphrasing using 7 LLMs (instead of generation from scratch). Evaluate generation quality and similarity using 6 metrics: METEOR, BERTScore, n-gram overlap, Levenshtein Distance, MAUVE, and LangCheck. Keep ~1% noise samples (such as "As an AI model...") with annotations for further analysis.
    - **Design Motivation**: Directly generating or completing from scratch introduces topic and length biases. Paraphrasing maintains alignment with the topic and length of the original text, focusing the detection task on text style rather than topical differences.

3. **Systematic Evaluation of Three Types of Detection Methods**:

    - **Function**: Comprehensively compare three main categories of detection methods: statistical zero-shot, pre-trained, and fine-tuned.
    - **Mechanism**: **Statistical Zero-shot** (5 methods): Binoculars, Fast-DetectGPT, LLM-Deviation, DetectLLM-LRR, and S5, which require no training and are based on probability/statistical differences; **Pre-trained** (5 methods): detectors trained on other datasets such as ChatGPT-Detector-RoBERTa, tested directly through transfer; **Fine-tuned** (7 methods): mDeBERTa, XLM-RoBERTa, Mistral-7B, Llama-3-8B, Aya-101, BLOOMZ-3B, Falcon-1B fine-tuned on the MultiSocial training set.
    - **Design Motivation**: Can statistical methods work on short texts? How is the cross-domain transfer capability of pre-trained detectors? What is the upper limit of fine-tuning effectiveness on social media? This three-way comparison addresses these core questions.

## Key Experimental Results

### Main Results (Overall Performance on Entire Test Set)

| Category | Detector | AUC ROC | Macro F1@5%FPR |
|------|--------|---------|----------------|
| Fine-tuned | Llama-3-8B-MultiSocial | **0.9769** | **0.8696** |
| Fine-tuned | Mistral-7B-MultiSocial | 0.9768 | 0.8692 |
| Fine-tuned | Aya-101-MultiSocial | 0.9731 | 0.8462 |
| Fine-tuned | XLM-RoBERTa-large | 0.9553 | 0.7840 |
| Fine-tuned | mDeBERTa-v3-base | 0.9544 | 0.7652 |
| Pre-trained | BLOOMZ-3B-mixed-Detector | 0.7553 | 0.3024 |
| Statistical zero-shot | Fast-DetectGPT | 0.7418 | 0.3605 |
| Statistical zero-shot | Binoculars | 0.7248 | 0.2815 |
| Pre-trained | RoBERTa-large-OpenAI | 0.3450 | 0.1376 |

### Cross-lingual Analysis (Fine-tuned Llama-3-8B, AUC ROC by language)

| Language | AUC ROC | Language | AUC ROC |
|------|---------|------|---------|
| English (en) | 0.985 | Arabic (ar) | 0.978 |
| Spanish (es) | 0.983 | Chinese (zh) | 0.976 |
| German (de) | 0.982 | Scottish Gaelic (gd)* | 0.935* |
| Bulgarian (bg) | 0.980 | Irish (ga)* | 0.952* |

\*Test-set-only languages (no data for these languages in the training set)

### Key Findings
- Fine-tuned detectors still achieve excellent performance on short social media texts (AUC ROC of 0.977), proving that informal social media text does not fundamentally impair detectors.
- Statistical zero-shot methods (Binoculars, Fast-DetectGPT) can still achieve ~0.72-0.74 AUC ROC on social media, though they are far behind fine-tuned ones.
- Pre-trained detectors show mixed performance—RoBERTa-OpenAI even falls below random guessing (0.345), indicating that detectors trained on long texts/English completely fail to transfer across domains.
- Detectors trained on Telegram exhibit the best cross-lingual generalization capability, likely because its content types and lengths are most consistent across different languages.
- High detection performance (>0.93) is maintained even on low-resource languages that appear only in the test set (Irish, Scottish Gaelic), demonstrating the cross-lingual generalization of fine-tuned models.

## Highlights & Insights
- Its data scale of 472k texts, 22 languages, 5 platforms, and 7 generators is the largest in the MGTD field. The dataset and code are open-sourced, which is of great value for future research.
- The paraphrase-based generation strategy (instead of generation from scratch) is an elegant design—it maintains topic/length alignment, eliminates artifact bias, and focuses detection on genuine textual stylistic differences.
- Retaining 1% noise samples (generation-failed "As an AI" texts) and labeling them is a responsible dataset design practice, avoiding inflated performance estimates from overly "clean" data.

## Limitations & Future Work
- Among the 7 generators, only GPT-3.5-Turbo is closed-source; the latest and strongest generators such as GPT-4 and Claude are not tested.
- Imbalance in sample sizes of different languages and platforms in training/testing (e.g., Chinese only has ~8K training samples compared to ~39K for English).
- The paraphrasing strategy may underestimate the detection difficulty of "creatively" generated content directly from scratch.
- Robustness of detectors against adversarial attacks (such as post-process paraphrasing, style transfer) is not studied.

## Related Work & Insights
- **vs M4GT-Bench**: M4GT-Bench covers 9 languages, but its social media portion is limited to English Reddit, and language-domain coverage is highly imbalanced. MultiSocial 22 languages × 5 platforms systematically focus on social media.
- **vs MULTITuDE**: MULTITuDE covers 11 languages but only in the news domain with long, formal texts. MultiSocial fills the gap of short social media texts.
- **vs TweepFake**: The earliest social media MGTD dataset (English tweets), but it only contains 24K samples and uses 6 outdated generators, which cannot evaluate modern LLMs.
- **Insights**: The best detectors maintain a performance of >0.93 even on unseen languages, implying that statistical fingerprints of LLM-generated texts might be cross-lingually universal.

## Rating
- Novelty: ⭐⭐⭐⭐ The first large-scale multilingual multi-platform MGTD benchmark focusing on social media, filling a key gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 17 detection methods, cross-lingual, cross-platform, and cross-generator dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear data construction pipeline and systematic experimental organization.
- Value: ⭐⭐⭐⭐⭐ The dataset and benchmark are highly practical for the MGTD community, available for open download.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Authorship Attribution in Multilingual Machine-Generated Texts](../../ACL2026/aigc_detection/authorship_attribution_in_multilingual_machine-generated_texts.md)
- [\[ACL 2025\] Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media](aigt_social_media_monitoring.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](../../NeurIPS2025/aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[ACL 2025\] Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](greater_adversarial_mgt_detection.md)
- [\[ACL 2025\] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)

</div>

<!-- RELATED:END -->

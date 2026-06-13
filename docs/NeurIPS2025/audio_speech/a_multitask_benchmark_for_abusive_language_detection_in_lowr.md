---
title: >-
  [Paper Note] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings
description: >-
  [NeurIPS 2025][Audio & Speech][abusive language detection] This paper introduces TiALD (Tigrinya Abusive Language Detection), the first large-scale multi-task benchmark dataset for the low-resource Tigrinya language. It…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "abusive language detection"
  - "low-resource languages"
  - "multi-task learning"
  - "Tigrinya"
  - "benchmark dataset"
date: 2026-05-08
content_hash: a56efe9d3a86517f
---

# A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings

**Conference**: NeurIPS 2025
**arXiv**: [2505.12116](https://arxiv.org/abs/2505.12116)  
**Code**: [https://github.com/fgaim/TiALD](https://github.com/fgaim/TiALD)  
**Area**: Audio & Speech
**Keywords**: abusive language detection, low-resource languages, multi-task learning, Tigrinya, benchmark dataset

## TL;DR

This paper introduces TiALD (Tigrinya Abusive Language Detection), the first large-scale multi-task benchmark dataset for the low-resource Tigrinya language. It comprises 13,717 YouTube comments annotated jointly across three tasks—abusive language detection, sentiment analysis, and topic classification—and demonstrates that a compact fine-tuned model (TiRoBERTa, 125M parameters) consistently outperforms frontier LLMs such as GPT-4o and Claude Sonnet 3.7 across all tasks.

## Background & Motivation

**Background**: Content moderation research has achieved remarkable progress for high-resource languages such as English, with automated abusive language detection reaching considerable maturity. However, the vast majority of the world's languages remain in a computational resource desert, lacking annotated data, tools, and models, leaving millions of users exposed to unmoderated online hostility.

**Limitations of Prior Work**: Tigrinya, spoken by approximately 10 million people (primarily in Eritrea and Ethiopia), suffers from an extreme scarcity of computational resources—there are no annotated datasets, no evaluation benchmarks, and no trained models. Without benchmarks, progress cannot be measured and research cannot be incentivized. Compounding this, approximately 64% of Tigrinya social media content is written in **Latin transliteration** rather than the native Ge'ez script, a writing convention that existing tools largely fail to cover.

**Key Challenge**: Abusive language detection for low-resource languages is trapped in a vicious cycle: no data → no models → no protection → no incentive to build data. Keyword-based sampling leads to lexical homogeneity, while random sampling yields an extremely low proportion of abusive instances (only 14.3%), making neither approach suitable for constructing high-quality training sets.

**Goal**: (1) Construct the first multi-task annotated benchmark covering both writing systems of Tigrinya; (2) design a data sampling strategy appropriate for low-resource settings; (3) evaluate the performance gap between fine-tuned models and LLMs in this regime.

**Key Insight**: The work begins from data construction, addressing sampling bias through an iterative seed-word expansion strategy and providing richer contextual signals through joint multi-task annotation.

**Core Idea**: Leverage iterative semantic cluster sampling combined with multi-task joint annotation to build the first high-quality abusive language detection benchmark for a low-resource language.

## Method

### Overall Architecture

The TiALD construction pipeline proceeds as follows: (1) collect 4.1 million Tigrinya comments from 51 YouTube channels; (2) apply an iterative seed-word expansion strategy to select 20K representative comments; (3) have 9 native-speaker annotators label 13,717 comments across three tasks—abusive language detection, sentiment analysis, and topic classification; (4) obtain gold labels for a 900-comment test set via three-way annotation with expert adjudication. Baseline experiments cover three paradigms: single-task fine-tuning, multi-task joint learning, and LLM zero/few-shot prompting.

### Key Designs

1. **Iterative Seed-Word Expansion Sampling Strategy**:

    - **Function**: Efficiently select annotation candidates rich in abusive content and lexically diverse from a large unlabeled corpus.
    - **Mechanism**: A word2vec model (CBOW, 300 dimensions) is trained on 4.1 million comments. Starting from 61 seed words, a three-stage iterative expansion is performed—Stage 1 retrieves the 50 nearest neighbors per seed word and filters morphological variants; Stage 2 retrieves 25 neighbors per new term; Stage 3 retrieves 10 neighbors per term. This yields 8,728 diverse terms, from which 15K matched comments are selected alongside a 5K random control group.
    - **Design Motivation**: Pure keyword search achieves a type-to-token ratio of only 0.18 and random sampling 0.13, whereas iterative expansion reaches 0.28. The proportion of abusive samples increases from 14.3% under random sampling to 65.2%.

2. **Dual-Script Coverage**:

    - **Function**: Provide coverage of both the native Ge'ez script and Latin transliteration.
    - **Mechanism**: Annotated data is distributed at a 70% Ge'ez to 30% Latin/mixed ratio; the GeezSwitch library is used for script identification and filtering.
    - **Design Motivation**: Since 64% of Tigrinya social media content uses non-standard Latin transliteration, models covering only Ge'ez effectively miss the majority of real-world content.

3. **Multi-Task Joint Learning Framework**:

    - **Function**: Simultaneously learn abusive language detection (binary classification), sentiment analysis (four-class), and topic classification (five-class) using a shared encoder.
    - **Mechanism**: A shared Transformer encoder produces $\mathbf{h} = \text{Encoder}(x)$; a single linear head maps to logits over $L = 2 + 4 + 5 = 11$ labels, each predicted via sigmoid with a 0.5 threshold, trained with mean BCE loss.
    - **Design Motivation**: Abusiveness, sentiment, and topic share complementary linguistic features (e.g., political topics frequently co-occur with negative sentiment and abusive language); joint learning captures these associations through parameter sharing.

### Evaluation Metric

The paper introduces the **TiALD Score**, defined as the average macro F1 across all three tasks, serving as a unified benchmark-level evaluation metric.

## Key Experimental Results

### Main Results (Fine-Tuned Models vs. LLMs)

| Model | Type | Abusive F1↑ | Sentiment F1↑ | Topic F1↑ | TiALD Score↑ |
|-------|------|-------------|---------------|-----------|--------------|
| TiRoBERTa (single-task) | Fine-tuned 125M | **86.67** | 52.82 | 54.23 | 64.57 |
| TiRoBERTa (multi-task) | Fine-tuned 125M | 86.11 | **53.41** | **54.91** | **64.81** |
| AfroXLMR-76L | Fine-tuned 560M | 85.20 | 54.94 | 51.42 | 63.86 |
| GPT-4o (few-shot) | LLM | 72.06 | 21.88 | 27.56 | 40.50 |
| Claude 3.7 (few-shot) | LLM | 79.31 | 23.39 | 27.92 | 43.54 |
| Gemma-3 4B (few-shot) | LLM | 58.37 | 30.46 | 39.49 | 42.78 |

### Ablation Study: Effect of Video Context

| Setting | GPT-4o Abusive F1 | Claude 3.7 Abusive F1 | Notes |
|---------|-------------------|----------------------|-------|
| Comment only (zero-shot) | 71.05 | 59.20 | Baseline |
| + Video title | 75.59 | 67.64 | +4.5 / +8.4 |
| + Title + video description | 74.70 | 72.02 | Substantial gain for Claude |

### Key Findings

- **Small models decisively outperform large models**: TiRoBERTa (125M) surpasses GPT-4o by 14.6 percentage points on abusive language detection and by 24 points on the TiALD Score. Fine-tuning remains the dominant approach for low-resource languages.
- **LLMs collapse on multi-class tasks**: LLMs achieve reasonable performance on binary abusive language detection (71–79%) but degrade severely on four-class sentiment and five-class topic classification (peaks of only ~30% and ~39%), revealing a fundamental insufficiency in LLM multi-class classification for low-resource languages.
- **Multi-task learning yields consistent improvements**: Joint training improves performance for nearly all models; TiELECTRA-small shows the largest gain (+1.76 TiALD Score).
- **LLaMA-3.2 3B exhibits reversed classification bias**: In zero-shot mode, the model labels 68% of comments as abusive; in few-shot mode, this reverses to 77% labeled as non-abusive, exposing severe instability in small LLMs on out-of-distribution languages.
- **Video context benefits LLMs**: Adding video titles and descriptions improves Claude's zero-shot abusive language detection F1 by 12.8 percentage points.

## Highlights & Insights

- **The iterative seed-word expansion strategy is elegantly designed**: Three-stage iteration combined with morphological deduplication achieves high coverage and diversity, with a type-to-token ratio twice that of random sampling. This method is directly transferable to pre-annotation data selection for any low-resource language.
- **Practical consideration of dual-script systems**: Most low-resource NLP work overlooks non-standard writing practices on social media; TiALD is among the few works to address this problem explicitly. The lesson applies equally to other languages with multiple writing systems (e.g., Hindi in Devanagari vs. Latin).
- **Quantitative evidence for small model superiority**: The paper provides a clear empirical chain of evidence that LLMs underperform fine-tuned small models in low-resource settings, offering actionable guidance for real-world deployment decisions.

## Limitations & Future Work

- Data is drawn exclusively from YouTube comments; coverage of other platforms (Twitter, Facebook, Telegram) is absent, raising questions about generalizability.
- Only 9 annotators were involved, and inter-annotator agreement for topic classification is relatively low, suggesting that annotation guidelines for this task require greater specificity.
- The multi-task framework currently employs equal-weight loss; per-task or per-class weighting may help alleviate class imbalance.
- Data augmentation techniques (e.g., back-translation, cross-script conversion) and their effects on low-resource performance remain unexplored.
- The quality of VLM-generated video descriptions has not been thoroughly validated, and the noise they introduce may affect the reliability of the corresponding analysis.

## Related Work & Insights

- **vs. existing low-resource NLP benchmarks**: Prior African-language NLP work has focused primarily on NER and machine translation; TiALD is the first multi-task benchmark targeting abusive language detection.
- **vs. high-resource abusive language detection**: English abusive language detection has reached F1 scores above 90%, yet direct transfer to Tigrinya performs poorly (XLM-R achieves only ~81%), underscoring the continued importance of language-specific approaches.
- **vs. the LLM-as-judge paradigm**: The results demonstrate that large models' zero/few-shot capabilities for low-resource languages fall far short of expectations, providing an important counterexample to the narrative of "LLMs replacing everything."

## Rating

- **Novelty**: ⭐⭐⭐ — A dataset-contribution paper; the core value lies in filling a gap rather than methodological innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers single-task, multi-task, and LLM paradigms with detailed ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear organization, well-motivated problem statement, and transparent data construction process.
- **Value**: ⭐⭐⭐⭐ — An important reference for the low-resource NLP community; the iterative sampling strategy has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis](deepasa_an_object-oriented_multi-purpose_network_for_auditory_scene_analysis.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](../../ACL2026/audio_speech/hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation
description: >-
  [AAAI 2026][Multilingual & Machine Translation][Vietnamese dialect] ViDia2Std constructs the first manually annotated Vietnamese dialect-to-standard parallel corpus covering all 63 provinces of Vietnam (13…
tags:
  - "AAAI 2026"
  - "Multilingual & Machine Translation"
  - "Vietnamese dialect"
  - "dialect normalization"
  - "low-resource language"
  - "parallel corpus"
  - "sequence-to-sequence model"
date: 2026-05-08
content_hash: 1667bb0ef6c5299d
---

# ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation

**Conference**: AAAI 2026
**arXiv**: [2603.10211](https://arxiv.org/abs/2603.10211)  
**Code**: [GitHub](https://github.com/biuinvincible/ViDia2Std.git)  
**Area**: Multilingual Translation
**Keywords**: Vietnamese dialect, dialect normalization, low-resource language, parallel corpus, sequence-to-sequence model

## TL;DR

ViDia2Std constructs the first manually annotated Vietnamese dialect-to-standard parallel corpus covering all 63 provinces of Vietnam (13,000+ sentence pairs), evaluates multiple seq2seq models on the dialect normalization task, and demonstrates that dialect normalization as a preprocessing step significantly improves downstream task performance in machine translation and sentiment analysis.

## Background & Motivation

**Background**: Vietnamese encompasses three major dialect groups—Northern, Central, and Southern—with significant differences in phonology, vocabulary, and syntax. Standard Vietnamese is based on the Northern dialect, and nearly all mainstream Vietnamese NLP models (e.g., PhoBERT, BARTpho) are trained predominantly on Northern standard language data. As a result, these models suffer sharp performance degradation when processing Central and Southern dialect text.

**Limitations of Prior Work**: Prior Vietnamese dialect normalization research (Le and Luu, 2023) only constructed a parallel corpus from Central dialect to Northern standard language, with data sourced exclusively from Hà Tĩnh province, resulting in severely insufficient diversity. The Central dialect itself exhibits substantial inter-provincial variation (e.g., Huế, Quảng Trị, and Thanh Hóa each have distinct vocabularies and syntactic patterns), while Southern dialect and non-standard Northern variants are entirely absent. Commercial translation systems (e.g., Google Translate, Claude, Gemini) also frequently produce significant misinterpretations when handling dialect input.

**Key Challenge**: Vietnamese dialects span a wide geographic range with high variability, yet annotated resources are extremely scarce. Existing datasets and models cover only a tiny subset of dialects and cannot represent real-world dialectal diversity.

**Goal**: (1) Construct a nationally representative, high-quality dialect-to-standard parallel corpus; (2) establish strong baseline models for dialect normalization; (3) validate the practical gains of dialect normalization on downstream NLP tasks.

**Key Insight**: The authors leverage social media (Facebook) as a data source—over 70% of Vietnam's social media population uses Facebook, and users extensively employ local dialects in comments on regional news pages, providing a natural, nationally scalable source for dialect data collection.

**Core Idea**: Through large-scale social media collection and rigorous manual annotation, this work constructs the first dialect-to-standard parallel corpus covering all 63 Vietnamese provinces, demonstrating that dialect normalization is a critical preprocessing step for improving the robustness of Vietnamese NLP systems.

## Method

### Overall Architecture

The work comprises two major components: **corpus construction** and **model evaluation**. Corpus construction follows a pipeline of raw data collection from social media → automated preprocessing and denoising → manual dialect filtering → three-stage annotation. Model evaluation formalizes dialect normalization as a conditional sequence generation task, fine-tunes multiple pretrained seq2seq models, and comprehensively evaluates them on both intrinsic metrics and extrinsic downstream tasks.

### Key Designs

1. **Data Collection and Denoising Pipeline**:

    - **Function**: Collect authentic dialect text from Facebook public news pages to construct a raw corpus covering all 63 provinces.
    - **Mechanism**: For each province, local news pages are identified (prioritizing personally operated pages over official accounts, as personal pages exhibit higher dialect usage frequency). Comments are scraped using Selenium, targeting 350–500 posts per province. Raw data undergo two-stage processing: the **automated preprocessing stage** performs lowercasing, removal of emojis/URLs/@mentions, and normalization of internet slang via a manually curated dictionary (e.g., "ko" → "không"); the **manual dialect filtering stage** uses pre-compiled region-specific keyword lists, with annotators selecting comments exhibiting high dialect density.
    - **Design Motivation**: Facebook's exceptionally high penetration rate in Vietnam and users' dialect usage habits on local pages make it the most natural and cost-effective source for large-scale dialect data collection. The two-stage processing ensures data quality while reducing the manual annotation burden.

2. **Three-Stage Annotation Protocol and Quality Control**:

    - **Function**: Ensure accuracy of dialect-to-standard mappings and cross-annotator consistency.
    - **Mechanism**: Nine native-speaker annotators are recruited (2 Northern, 4 Central, 3 Southern), and annotation proceeds in three steps: **dialect cleaning**—correcting spelling/abbreviation issues while preserving dialectal features; **dialect-to-standard mapping**—translating dialectal vocabulary into standard language while maintaining semantics, tone, and sentence structure; **ambiguity annotation**—flagging idiomatic expressions or polysemous cases for discussion. Quality control employs the Strict Semantic Group Agreement (SSGA) metric: a sentence is considered consistent only when all annotators provide semantically equivalent mappings for every dialectal token. The pilot phase requires SSGA $\geq$ 80% before full annotation proceeds; final consistency reaches 86% (Northern), 82% (Central), and 85% (Southern).
    - **Design Motivation**: Dialect normalization is a lexical-semantic mapping task rather than free rewriting. SSGA is more stringent than pairwise agreement or majority voting, enforcing a high standard of full-group semantic convergence. More annotators are assigned to the Central dialect due to its highest internal lexical diversity.

3. **Sequence-to-Sequence Normalization Models**:

    - **Function**: Automatically convert dialect input text into standard Vietnamese.
    - **Mechanism**: Dialect normalization is formalized as conditional sequence generation, maximizing $P(y|x) = \prod_{t=1}^{m} P(y_t | y_{<t}, x; \theta)$, where $x$ is the dialect input and $y$ is the standard language output. Five models are evaluated: BARTpho-word-base (word-level tokenization), BARTpho-syllable-base (syllable-level tokenization, better suited to Vietnamese monosyllabic characteristics), ViT5-base (Vietnamese adaptation of T5), Vietnamese-correction-v2 (pretrained on spelling correction), and mBART-large-50 (multilingual denoising pretraining over 50 languages). All models are trained uniformly with AdamW optimizer, learning rate 2e-5, batch size 32, maximum 10 epochs, and fp16 mixed-precision training.
    - **Design Motivation**: Vietnamese's monosyllabic nature gives syllable-level tokenization models (e.g., BARTpho-syllable) an inherent advantage in capturing dialectal variants. Multilingual models (mBART), having been exposed to large volumes of cross-lingual variation, exhibit greater robustness to dialect input.

### Loss & Training

The standard autoregressive cross-entropy loss is used: $L = -\sum_{t=1}^{m} \log P(y_t | y_{<t}, x; \theta)$. Data splits are 10,870 training / 1,184 validation / 1,603 test. Early stopping is applied based on BLEU improvement $\geq$ 0.01 (patience=1), and a unified random seed of 42 is used across all models to ensure reproducibility.

## Key Experimental Results

### Main Results (Intrinsic Evaluation)

| Model | ROUGE-L | BLEU | METEOR | WER | CER | Parameters |
|-------|---------|------|--------|-----|-----|------------|
| BARTpho-word-base | 0.9167 | 0.7601 | 0.8625 | 0.1527 | 0.1049 | 150M |
| BARTpho-syllable-base | 0.9218 | 0.7597 | 0.8627 | 0.1513 | 0.1045 | 132M |
| Vietnamese-correction-v2 | 0.9257 | 0.7723 | 0.8746 | 0.1416 | 0.0988 | 396M |
| ViT5-base | 0.9300 | 0.7934 | 0.8802 | 0.1340 | 0.0876 | 310M |
| **mBART-large-50** | **0.9384** | **0.8166** | **0.8925** | **0.1226** | **0.0754** | 611M |

### Extrinsic Evaluation: Machine Translation

| Translation System | Direct Translation Acceptance | Post-Normalization Acceptance | Gain |
|-------------------|------------------------------|-------------------------------|------|
| Microsoft Azure | 11.33% | 22.33% | +11.00% |
| Google Cloud | 38.33% | 45.83% | +7.50% |
| DeepSeek-V3 | 54.50% | 63.17% | +8.67% |
| Kimi-K2-Instruct | 51.33% | 64.17% | +12.84% |
| Gemini 1.5 Flash | 47.00% | 57.00% | +10.00% |
| Gemini 2.0 Flash | 61.83% | 67.00% | +5.17% |

### Extrinsic Evaluation: Sentiment Analysis

| Configuration | Accuracy | Weighted F1 | Notes |
|--------------|---------|-------------|-------|
| Before normalization | 50.59% | 0.52 | Direct dialect input |
| **After normalization** | **62.13%** | **0.63** | Normalization preprocessing |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| Cross-system consistency (≥4 systems improved) | 32 improved : 1 degraded | Normalization gains are systematic, not coincidental |
| Attribution of translation degradation | 82.5% non-normalization errors | MT fragility (46.8%) and evaluation noise (35.7%) are primary causes |
| Sentiment analysis correction rate | 76.81% | Improved predictions far outnumber introduced errors |

### Key Findings

- mBART-large-50 consistently leads on all intrinsic metrics, demonstrating the advantage of multilingual pretraining for dialect normalization.
- ViT5-base achieves comparable performance with less than half the parameters, making it suitable for resource-constrained deployment.
- Normalization benefits are consistently positive across all 6 translation systems; when the consistency threshold is raised to ≥4 systems, the improvement-to-degradation ratio reaches 32:1.
- In-depth analysis of translation degradation shows that only 17.5% is attributable to normalization model errors, with the majority caused by downstream MT system fragility and evaluator noise.
- F1 for negative sentiment in sentiment analysis improves from 0.59 to 0.72, indicating that dialectal ambiguity has the greatest impact on negative emotion expression.

## Highlights & Insights

- **Data-first research paradigm**: In low-resource NLP, constructing high-quality datasets is often more valuable than model innovation. The core contribution of ViDia2Std lies in the data rather than the model.
- **The proposed Strict Semantic Group Agreement (SSGA) metric** is worth adopting broadly: it is more stringent than the commonly used Kappa coefficient, requiring all annotators to reach semantic equivalence on every dialectal token.
- **The in-depth analysis of extrinsic evaluation** is highly rigorous: rather than reporting only aggregate gains, it categorizes and attributes degradation cases (MT fragility 46.8%, evaluation noise 35.7%, normalization errors 17.5%), lending greater credibility to the conclusions.
- The methodology of using social media as a dialect data source is generalizable to other low-resource languages.
- The validation of dialect normalization as a preprocessing step offers reference value for Chinese dialect processing, Arabic dialect processing, and related areas.

## Limitations & Future Work

- An **over-normalization** problem exists: the model occasionally erases pragmatic or stylistic information embedded in dialect expressions, diminishing expressive richness.
- Extrinsic evaluation relies on a single LLM-as-a-Judge (Gemini 2.5 Flash), raising concerns about evaluation consistency and reliability.
- The corpus is sourced exclusively from Facebook comments; other forms of dialect text such as spoken transcriptions and forum posts are not covered.
- The dataset scale (13K sentence pairs) remains relatively small by deep learning standards, potentially limiting model generalizability.
- The positive sentiment class shows slight degradation after normalization (12 errors corrected but 28 new errors introduced), suggesting that dialectal expressions of positive sentiment are more complex.

## Related Work & Insights

- **Le and Luu (2023)**: The first Vietnamese Central-to-Northern dialect parallel corpus, but with limited geographic coverage; ViDia2Std extends coverage nationwide and incorporates Southern dialect.
- **Multi-dialect machine translation** (Abe et al., 2018; Kuparinen et al., 2023): Treats dialect normalization as character/byte-level transduction; ViDia2Std demonstrates that pretrained seq2seq models are more effective.
- **Low-resource dialect fine-tuning** (Alam and Anastasopoulos, 2025): Fine-tuning open-source LLMs on small-scale dialect corpora can significantly improve BLEU, consistent with the conclusions of ViDia2Std.
- **Implications for Chinese dialect NLP**: Similar methodology could be applied to constructing normalization corpora for Cantonese, Hokkien, and other dialects.

## Rating
- **Novelty**: ⭐⭐⭐ — The methodological contribution (seq2seq fine-tuning) is not novel, but the data contribution is pioneering—the first nationally representative Vietnamese dialect corpus.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual intrinsic/extrinsic evaluation, cross-validation across 6 translation systems, and in-depth attribution analysis of degradation cases; exceptionally rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, effective visualizations, and detailed description of the annotation protocol.
- **Value**: ⭐⭐⭐⭐ — Both the corpus and baseline models are open-sourced, offering direct practical value to the Vietnamese NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)
- [\[NeurIPS 2025\] Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection](../../NeurIPS2025/multilingual_mt/reflective_translation_improving_low-resource_machine_translation_via_structured.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[AAAI 2026\] How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)

</div>

<!-- RELATED:END -->

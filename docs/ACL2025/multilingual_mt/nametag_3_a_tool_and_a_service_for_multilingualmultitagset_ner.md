---
title: >-
  [Paper Note] NameTag 3: A Tool and a Service for Multilingual/Multitagset NER
description: >-
  [ACL 2025][Multilingual & Machine Translation][Named Entity Recognition] This paper introduces NameTag 3, an open-source multilingual, multi-dataset, and multi-tagset named entity recognition (NER) tool and cloud service. Based on fine-tuned pre-trained language models, a single 355M parameter model achieves SOTA performance on 21 test sets across 15 languages, while being over 10,000 times faster than LLMs such as DeepSeek-R1.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Named Entity Recognition"
  - "Multilingual NER"
  - "Nested NER"
  - "Multitagset"
  - "Open-source Tool"
date: 2026-05-08
content_hash: e52b9f9a76b14682
---

# NameTag 3: A Tool and a Service for Multilingual/Multitagset NER

**Conference**: ACL 2025  
**arXiv**: [2506.05949](https://arxiv.org/abs/2506.05949)  
**Code**: [https://github.com/ufal/nametag3](https://github.com/ufal/nametag3)  
**Area**: Multilingual Translation  
**Keywords**: Named Entity Recognition, Multilingual NER, Nested NER, Multitagset, Open-source Tool

## TL;DR
This paper introduces NameTag 3, an open-source multilingual, multi-dataset, and multi-tagset named entity recognition (NER) tool and cloud service. Based on fine-tuned pre-trained language models, a single 355M parameter model achieves SOTA performance on 21 test sets across 15 languages, while being over 10,000 times faster than LLMs such as DeepSeek-R1.

## Background & Motivation

Named Entity Recognition (NER) is a fundamental preprocessing step in NLP and knowledge extraction systems, designed to identify person names, locations, organizations, and other entities in text. Although NER research (especially for English) is highly mature, practically usable multilingual open-source NER tools remain scarce.

**Limitations of Prior Work**:

**Insufficient Tool Coverage**: Although existing tools such as Stanza and spaCy support multiple languages, they require training separate models for each language, which prevents cross-lingual transfer. Stanza relies on a frozen Flair embeddings + Bi-LSTM + CRF architecture, which has become technologically outdated.

**Lack of Support for Nested NER**: Neither Stanza nor spaCy supports nested entity recognition, yet many languages (such as Czech in the CNEC 2.0 corpus) feature complex nested annotations.

**Lack of Flexible Tagset Support**: Different datasets use different tagsets (CoNLL, UNER, OntoNotes), and existing tools cannot support multiple tagsets within a single model.

**Extremely Inefficient NER with LLMs**: Although large language models like GPT and DeepSeek can perform zero-shot NER, their accuracy is typically far lower than that of fine-tuned models, and they are tens of thousands of times slower.

**Key Challenge**: The need for a lightweight, efficient, and unified tool that supports multilingual/multitagset/nested NER and outperforms zero-shot LLMs.

**Key Insight**: Jointly training on a single fine-tuned multilingual pre-trained language model, utilizing a multi-tagset classification head design for tagset flexibility, and employing a sequence-to-sequence decoding head to support nested NER.

## Method

### Overall Architecture
NameTag 3 is fine-tuned on pre-trained language models (XLM-R Large 355M or RobeCzech Base 126M) and offers two recognition modes: flat NER (softmax classification head) and nested NER (seq2seq decoder head). A single multilingual model is jointly trained on 21 datasets, 17 languages, and 3 tagsets.

### Key Designs

1. **Multitagset Learning**:

    - **Mechanism**: Allocating independent classification heads for each tagset while sharing the underlying encoder.
    - Jointly optimizing the encoder and all classification heads during training.
    - Selecting the corresponding classification head based on the requested tagset during inference, ensuring only valid tags are predicted.
    - Supporting three tagsets: CoNLL (PER/LOC/ORG/MISC), UNER (universal NER v1), and OntoNotes.
    - **Design Motivation**: Avoiding the need to maintain independent models for each tagset, enabling a unified service.

2. **Seq2Seq Decoding for Nested NER**:

    - Replacing the softmax classification head with a Transformer seq2seq decoder.
    - The decoder generates a linearized (flattened) nested tag sequence for each input token.
    - Employing a hard attention mechanism to focus on the current token.
    - Pre-training the decoder for a few epochs with frozen encoder weights to adapt to encoder representations, followed by joint fine-tuning.
    - **Design Motivation**: Nested entities require outputting multiple tags for the same text span, which a standard classification head cannot handle.

3. **Square Root Temperature Sampling Training**:

    - The sampling probability of each corpus in a training batch is proportional to the square root of its sentence count.
    - Effect: Under-sampling large corpora and over-sampling small corpora.
    - Evaluating using macro-average span F1 to ensure balanced performance across datasets.
    - **Design Motivation**: Addressing the data imbalance issue in joint training across multiple corpora.

### Loss & Training
- Flat multilingual model: XLM-R Large, 30 epochs, learning rate 2e-5, cosine decay.
- Nested NER model: RoBERTa Large (English) / RobeCzech Base (Czech), pre-training the decoder for 20 epochs with frozen encoder, followed by 50-60 epochs of joint fine-tuning.
- Batch size 4-16, depending on the specific dataset.

## Key Experimental Results

### Main Results — Flat NER F1 (Multilingual Model 355M)

| Language/Dataset | NameTag 3 (Multi) | NameTag 3 (Mono) | Stanza | Prev. SOTA | SOTA Model Size |
|------------|-------------------|------------------|--------|----------|-------------|
| English CoNLL-2003 | **94.09** | 93.80 | 92.1 | 94.60 | 1853M |
| Chinese OntoNotes v5 | **81.63** | 81.76 | 79.2 | 80.20 | 147M |
| Croatian UNER SET | **95.55** | 94.08 | - | 95.00 | 355M |
| Ukrainian Lang-uk | **92.88** | 90.45 | 86.1 | 88.73 | 110M |
| Czech CNEC 2.0 | **86.24** | 85.31 | - | - | - |

### Ablation Study — Comparison with LLMs (English CoNLL-2003 Full Test Set)

| Method | F1 | Speed (sentences/sec) | Total Time |
|------|-----|-------------|--------|
| NameTag 3 (355M) | **94.09** | 801 | **4.6s** |
| DeepSeek R1 70B 5-shot | 74.00 | 0.04 | 25 hours |
| DeepSeek R1 32B 5-shot | 74.26 | 0.06 | 16 hours |
| ChatGPT 3.5 ICL | 74.99 | - | - |

### Nested NER and Cross-lingual Transfer

| Task | NameTag 3 F1 | Prev. SOTA F1 | Description |
|------|-------------|-------------|------|
| ACE-2004 Nested | 88.39 | 88.72 | Close to SOTA |
| CNEC 2.0 Nested (46 classes) | **86.39** | 83.44 | New SOTA |
| Cebuano Cross-lingual | **96.97** | 82.2 | Unseen language, substantial improvement |
| Tagalog Cross-lingual | **97.78** | 83.7 | Unseen language, +14 points |

### Key Findings
- Multilingual jointly trained models outperform monolingual models on most datasets, demonstrating the effectiveness of cross-lingual transfer.
- The zero-shot cross-lingual transfer performance on unseen languages (Cebuano, Tagalog) is remarkable (+14 F1 points).
- The fine-tuned 355M model outperforms the 70B DeepSeek-R1 by 20 F1 percentage points, while being over 10,000 times faster.
- The multi-tagset design enables a single model to serve all tagset requirements.

## Highlights & Insights
- Proves that "small model fine-tuning remains the optimal choice when training data is available" in the LLM era, with highly convincing comparative data.
- The multi-tagset design elegantly solves the practical deployment issue of inconsistent labels across different datasets.
- As a tool paper, it provides three usage options (command line, web application, and REST API), covering various deployment scenarios.
- The performance of cross-lingual transfer to unseen languages demonstrates the enormous value of joint multilingual training.

## Limitations & Future Work
- The training data is predominantly Latin script, with insufficient coverage of non-Latin languages (only Chinese, Arabic, and Ukrainian).
- The models use the CC BY-NC-SA 4.0 license, which restricts commercial usage.
- As a supervised fine-tuned model, it is highly dependent on the availability of high-quality human-annotated data.
- Flat NER simplifies nested annotations into 4 tag classes, losing the fine-grained information of the original annotations.
- Collaboration scenarios with LLMs, such as using LLMs to assist in silver-standard data generation, remain unexplored.

## Related Work & Insights
- Represents the third generation of the NameTag series (2014 $\rightarrow$ 2019 $\rightarrow$ 2025), with a clear evolutionary path.
- Key differences from Stanza: fine-tuning PLMs (instead of frozen embeddings + Bi-LSTM) and single-model multilingualism (instead of separate per-language models).
- Provides a benchmark paradigm for NER tool development: multilingual joint training + multi-tagset classification heads + seq2seq nested decoding.
- Insight: On traditional NLP tasks with annotated data, carefully fine-tuned small models still far outperform LLMs.

## Rating
- **Novelty**: ⭐⭐⭐ Technically a combination of mature methods (fine-tuned PLM + multi-head + seq2seq), with limited algorithmic novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely thorough, with 21 datasets, 15 languages, and comprehensive comparisons against SOTA, Stanza, spaCy, and LLMs.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is well-written for a tool paper, highly informative, though slightly verbose.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical as an open-source NER tool, especially valuable to the academic community and non-English NLP researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] M-RewardBench: Evaluating Reward Models in Multilingual Settings](m_rewardbench.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] LangMark: A Multilingual Dataset for Automatic Post-Editing](langmark_a_multilingual_dataset_for_automatic_post-editing.md)
- [\[ACL 2025\] LangSAMP: Language-Script Aware Multilingual Pretraining](langsamp_multilingual_pretraining.md)
- [\[ACL 2025\] LexGen: Domain-aware Multilingual Lexicon Generation](lexgen_domain-aware_multilingual_lexicon_generation.md)

</div>

<!-- RELATED:END -->

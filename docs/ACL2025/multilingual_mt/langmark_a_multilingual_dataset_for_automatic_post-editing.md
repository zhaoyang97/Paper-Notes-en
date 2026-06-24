---
title: >-
  [Paper Note] LangMark: A Multilingual Dataset for Automatic Post-Editing
description: >-
  [ACL 2025][Multilingual & Machine Translation][automatic post-editing] This paper releases LangMark—a large-scale multilingual Automatic Post-Editing (APE) dataset comprising 206,983 triplets covering English to seven target languages, and demonstrates that LLMs combined with few-shot prompting can effectively improve the output quality of proprietary NMT engines.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "automatic post-editing"
  - "multilingual dataset"
  - "machine translation"
  - "LLM"
  - "few-shot prompting"
date: 2026-05-08
content_hash: c7d0fa0796d336cc
---

# LangMark: A Multilingual Dataset for Automatic Post-Editing

**Conference**: ACL 2025  
**arXiv**: [2511.17153](https://arxiv.org/abs/2511.17153)  
**Code**: None (Dataset released on [Zenodo](https://zenodo.org/records/15553365))  
**Area**: NLP / Machine Translation  
**Keywords**: automatic post-editing, multilingual dataset, machine translation, LLM, few-shot prompting

## TL;DR

This paper releases LangMark—a large-scale multilingual Automatic Post-Editing (APE) dataset comprising 206,983 triplets covering English to seven target languages, and demonstrates that LLMs combined with few-shot prompting can effectively improve the output quality of proprietary NMT engines.

## Background & Motivation

Automatic Post-Editing (APE) aims to automatically correct errors in machine translation outputs, reducing human intervention while ensuring translation quality. Although Neural Machine Translation (NMT) has made significant progress, current APE research faces several key bottlenecks:

**Insufficiency in Dataset Scale**: The WMT APE shared tasks contain only 15K-18K triplets, and while SubEdits has 161K, it only covers a single language pair (English-German).

**Lack of Language Diversity**: Most existing datasets cover only 1-2 language pairs, making it difficult to support multilingual APE research.

**Limitations of Synthetic Data**: Although synthetic datasets like eSCAPE are of large scale (millions of triplets), they fail to capture the subtle edits required for advanced NMT systems.

**Defects Remaining in NMT Outputs**: Even state-of-the-art NMT systems still produce contextually inappropriate translations in specialized domains such as marketing (e.g., mistranslating "our people" as "our nation/ethnic group", or translating "pitch" using its tar-related meaning).

Collectively, these issues highlight that **there is a need for a large-scale, multilingual, human-annotated NMT post-editing dataset**.

## Method

### Overall Architecture

LangMark is primarily a dataset contribution accompanied by baseline evaluation experiments. Its core pipeline is as follows:

1. Data Collection & Annotation → 2. Dataset Statistical Analysis → 3. Retrieval-Augmented Few-Shot APE Evaluation

### Key Designs

1. **Dataset Construction**

    - Source: Marketing-related documents from the Smartsheet platform, segmented into sentences/phrases by a Translation Management System (TMS).
    - Translation: Translated using a proprietary NMT engine specifically trained on the Smartsheet domain.
    - Post-editing: Completed in the TMS by professional linguists with over 5 years of industry experience.
    - Privacy Protection: Personally identifiable information (PII) was removed using Google DLP tools.
    - Deduplication: Duplicate triplets were removed for each language pair.
    - Design Motivation: To preserve the characteristics of real-world industrial data, thereby making evaluations more closely aligned with practical application scenarios.

2. **Language Coverage**

    - English → German (33.3K), Spanish (32.8K), French (33.0K), Italian (32.5K), Japanese (28.2K), Brazilian Portuguese (32.0K), and Russian (8.6K).
    - A total of 206,983 triplets.
    - Design Motivation: To cover diverse language families (Germanic, Romance, Slavic, and Japonic) to enhance the generalizability of the benchmark.

3. **Evaluation Framework**

    - A 90%/10% train/test split, where the training set is utilized for retrieving examples.
    - The source segments are embedded using OpenAI's text-embedding-3-small model.
    - The 20 most similar source-post-edited pairs are retrieved based on cosine similarity to serve as few-shot examples.
    - A unified 20-shot prompting format is used to evaluate all models.
    - Design Motivation: A zero-shot approach fails to outperform strong NMT baselines; thus, in-domain examples are required for guidance.

### Loss & Training

This work focuses on dataset construction and does not involve model training. The evaluation relies on the inference capabilities of pre-existing LLMs without fine-tuning.

## Key Experimental Results

### Main Results (ChrF Score, 20-shot APE)

| Model | EN-RU | EN-BR | EN-JP | EN-IT | EN-FR | EN-ES | EN-DE |
|------|-------|-------|-------|-------|-------|-------|-------|
| NMT Baseline | 68.90 | 89.44 | 70.22 | 89.58 | 81.96 | 86.07 | 81.29 |
| GPT-4o | **69.68** | 89.21 | **73.94** | **89.79** | **82.75** | **86.62** | **81.41** |
| Qwen2.5-72B | 70.13 | 89.03 | 72.93 | 89.10 | 82.34 | 86.44 | 81.16 |
| Claude 3.5-Haiku | 69.08 | 88.81 | 71.64 | 88.76 | 82.21 | 86.08 | 80.66 |
| Gemini-1.5 Flash | 68.92 | 89.18 | 71.69 | 89.40 | 82.20 | 86.24 | 81.01 |
| Llama 3.1-70B | 69.55 | 86.82 | 68.37 | 86.80 | 80.97 | 83.75 | 79.12 |

GPT-4o is the only closed-source model that consistently outperforms the NMT baseline across most language pairs. Qwen2.5-72B performs the best on Russian (EN-RU).

### Comparison with Commercial MT Engines (ChrF on Full Dataset)

| MT Engine | EN-DE | EN-ES | EN-FR | EN-IT | EN-JP | EN-BR | EN-RU |
|---------|-------|-------|-------|-------|-------|-------|-------|
| Proprietary NMT (Ours) | 81.09 | 86.04 | 81.54 | 89.73 | 69.77 | 89.13 | — |
| Google Translate | 73.95 | 79.79 | 76.57 | 79.80 | 62.11 | 83.70 | 64.34 |
| Microsoft Translator | 75.74 | 80.32 | 76.07 | 82.57 | 62.82 | 84.97 | 64.38 |
| DeepL | 73.03 | 75.01 | 74.74 | 76.96 | 55.26 | 83.93 | 67.74 |

The proprietary NMT significantly outperforms generic commercial MT engines across all languages, indicating that the APE task in LangMark is indeed highly challenging.

### Key Findings

1. **High Baselines are Hard to Outperform**: Except for GPT-4o, most models fail to consistently improve NMT outputs, demonstrating the challenging nature of the dataset as an APE benchmark.
2. **Conservatism in Editing**: All evaluated LLMs edit significantly less frequently than human annotators, indicating that deciding "when to edit" remains a critical challenge.
3. **Potential of Open-Source Models**: Qwen2.5-72B performs comparably to the best closed-source models and even achieves superior performance on Russian.
4. **Pronounced Language Discrepancies**: Japanese and Russian (languages requiring a high volume of edits) represent the directions with the greatest room for improvement for LLMs.
5. **Domain Specificity**: Translations in the marketing domain require a profound understanding of context rather than simple grammatical corrections.

## Highlights & Insights

- **Real-World Industrial Data**: Unlike synthetically or academically constructed datasets, LangMark reflects the actual challenges in professional translation workflows.
- **Proprietary NMT as Baseline**: Utilizing a high-quality, in-domain NMT engine as the starting point ensures that the performance ceiling of the APE task remains sufficiently high.
- **The "To Edit or Not to Edit" Dilemma**: This work systematically discusses the editing decision processes of APE models (via precision/recall analysis) for the first time.
- **Evaluation Methodology**: It unifies the evaluation framework for different models through a retrieval-augmented few-shot approach.

## Limitations & Future Work

1. The data source is restricted to the marketing domain and may not generalize to other translation scenarios (e.g., legal, medical, literary).
2. The Russian corpus contains only 8.6K triplets, which is substantially smaller than other language pairs.
3. Fine-tuning strategies (e.g., fine-tuning open-source LLMs on LangMark specifically for APE) have not been explored.
4. The proprietary NMT engine is not publicly reproducible, which limits the exact reproducibility of the initial translation stage.
5. The retrieval methodology is solely based on source segment similarity, without considering error type matching.

## Related Work & Insights

- Compared to WMT APE (15-18K, single language pair) and SubEdits (161K, English-German only), LangMark offers significant improvements in scale and language coverage.
- **Insight**: One can research adaptive editing strategies (e.g., avoiding edits under high confidence and applying fine-grained modifications under low confidence) using this dataset.
- Quality Estimation (QE) methods can be incorporated to assist in the decision of whether to apply edits.

## Rating

- **Novelty**: ⭐⭐⭐ — Primarily a dataset contribution; innovations in methodology are limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers various closed-source and open-source models with comprehensive multilingual comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, with detailed dataset descriptions and abundant visualizations.
- **Value**: ⭐⭐⭐⭐ — Fills a critical gap in multilingual NMT APE datasets, yielding high industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Context Augmented Token-Level Post-Editing for Human Interpreting](context_augmented_token-level_post-editing_for_human_interpreting.md)
- [\[ACL 2025\] An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)](an_expanded_massive_multilingual_dataset_for_high-performance_language_technolog.md)
- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)

</div>

<!-- RELATED:END -->

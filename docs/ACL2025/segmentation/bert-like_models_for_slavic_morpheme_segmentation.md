---
title: >-
  [Paper Note] BERT-like Models for Slavic Morpheme Segmentation
description: >-
  [ACL 2025][Segmentation][Morpheme Segmentation] This paper explores the use of BERT-like pretrained language models for morpheme segmentation tasks in Slavic languages. By modeling morpheme segmentation as a sequence labeling problem, the approach achieves results superior to traditional methods across multiple Slavic languages.
tags:
  - "ACL 2025"
  - "Segmentation"
  - "Morpheme Segmentation"
  - "Slavic Languages"
  - "BERT"
  - "Subword Models"
  - "Lexical Analysis"
date: 2026-05-08
content_hash: 93224f1c9ec5a356
---

# BERT-like Models for Slavic Morpheme Segmentation

**Conference**: ACL 2025  
**Code**: None  
**Area**: Word Segmentation/Morphological Analysis  
**Keywords**: Morpheme Segmentation, Slavic Languages, BERT, Subword Models, Lexical Analysis

## TL;DR
This paper explores the use of BERT-like pretrained language models for morpheme segmentation tasks in Slavic languages. By modeling morpheme segmentation as a sequence labeling problem, the approach achieves results superior to traditional methods across multiple Slavic languages.

## Background & Motivation

**Background**: Morpheme segmentation is the task of segmenting words into their minimal meaningful units (roots, prefixes, suffixes, endings, etc.), which is particularly crucial for the natural language processing of morphologically rich languages such as those in the Slavic family. Traditional approaches rely on rule-based systems or statistical models (e.g., Morfessor).

**Limitations of Prior Work**: Slavic languages (such as Russian, Czech, and Polish) are morphologically extremely rich, featuring abundant complex inflectional and derivational variations. Traditional methods struggle to handle irregular variations and rare morphological patterns. Existing neural approaches are primarily designed for morphologically simpler languages like English, making them ill-suited for Slavic languages.

**Key Challenge**: The subword tokenization (such as BPE/WordPiece) used in BERT-like models is frequency-based, which fundamentally differs from linguistic morpheme segmentation—subwords split by BPE do not necessarily correspond to meaningful morphemes.

**Goal**: (1) To verify whether BERT-like models can effectively capture morphological knowledge of Slavic languages; (2) To compare the performance of different pretrained models and methods on the morpheme segmentation task.

**Key Insight**: Although BERT's subword tokenization does not fully align with morpheme segmentation, the contextual representations learned by BERT during pretraining may contain rich implicit morphological information, which can be unlocked through appropriate fine-tuning.

**Core Idea**: Model morpheme segmentation as a character-level sequence labeling task (predicting whether a morpheme boundary exists after each character) and leverage the pretrained representations of BERT-like models to improve labeling accuracy.

## Method

### Overall Architecture
The input is a character sequence of a Slavic word, and the output is the boundary label for each character position (indicating whether it is a morpheme split point and the type of the segmented morpheme, such as root, prefix, suffix, etc.). A BERT-like model is employed as the encoder, followed by a sequence labeling head.

### Key Designs

1. **Character-Level Sequence Labeling Modeling**:

    - **Function**: Transform morpheme segmentation into a standard sequence labeling problem.
    - **Mechanism**: Treat each character of a word as a token. The model is required to predict a label for each character, indicating whether that position is a morpheme boundary and its boundary type (e.g., B-ROOT for the start of a root, I-SUFFIX for inside a suffix). A BIO or similar labeling scheme is adopted.
    - **Design Motivation**: Sequence labeling is a well-established paradigm in NLP that can directly leverage the capabilities of BERT with a simple conversion process.

2. **Multilingual Pretrained Model Comparison**:

    - **Function**: Evaluate the performance of different BERT variants on Slavic morpheme segmentation.
    - **Mechanism**: Comparative experiments include Multilingual BERT (mBERT), XLM-RoBERTa, and Slavic-specific pretrained models (such as SlavicBERT). The impact of character-level versus subword-level tokenization on morphological analysis tasks is explored.
    - **Design Motivation**: Slavic-specific models may provide better coverage of morphological knowledge for this language family.

3. **Cross-Lingual Transfer Learning**:

    - **Function**: Utilize training data from high-resource Slavic languages to facilitate morpheme segmentation in low-resource languages.
    - **Mechanism**: Leverage the etymological similarities and common morphological paradigms within the Slavic language family, transferring a model trained on one language to other languages in the same family via zero-shot or few-shot learning. For example, a model trained on Russian data is directly applied to Czech.
    - **Design Motivation**: Strong commonalities in morphological regularities within the Slavic language family provide an inherent advantage for cross-lingual transfer.

### Loss & Training
Standard cross-entropy loss is used for sequence labeling training. A CRF layer may be incorporated to model dependencies between labels.

## Key Experimental Results

### Main Results

| Language | Model | F1-Morpheme Boundary | F1-Type Classification |
|------|------|-------------|------------|
| Russian | SlavicBERT | Best | Best |
| Russian | mBERT | Second Best | Second Best |
| Russian | Morfessor | Traditional Baseline | Lower |
| Czech | XLM-R | Best | Best |
| Polish | SlavicBERT | Best | Best |

### Ablation Study

| Configuration | F1 | Description |
|------|-----|------|
| SlavicBERT + CRF | Best | CRF layer models label dependencies |
| SlavicBERT w/o CRF | Slightly Lower | Localized prediction |
| Character-level BERT | Moderate | Retrained character-level model |
| Cross-lingual Zero-shot | Usable | Transfer within the same family is effective |

### Key Findings
- Slavic-specific pretrained models significantly outperform general multilingual models in morpheme segmentation, indicating the importance of pretraining within the language family.
- BERT-like models substantially outperform traditional Morfessor and rule-based methods, showing a distinct advantage particularly when handling irregular morphological changes.
- Cross-lingual transfer works well within the Slavic language family, with the zero-shot transfer from Russian to Czech retaining approximately 85% of performance.
- The inclusion of a CRF layer yields consistent but modest improvements, suggesting that BERT has already implicitly learned label dependencies to some extent.

## Highlights & Insights
- **Addressing Language Family Gaps**: Systematically evaluates Slavic, a morphologically rich yet understudied language family, providing valuable experimental references for the field.
- **Efficacy of Cross-Lingual Transfer**: Demonstrates that knowledge transfer within the same family (specifically the Slavic language family) is viable at the morphological level, providing insights for other low-resource language families.

## Limitations & Future Work
- Only the Slavic language family was evaluated, and it remains unclear whether the conclusions apply to other morphologically rich families (such as Finno-Ugric or Turkic).
- Character-level modeling increases the sequence length, which may pose computational efficiency challenges for long words.
- Annotated data for morpheme segmentation is scarce, and the dataset sizes limit the potential of the models.
- Future work can combine morphological dictionaries with pretrained models to build more powerful hybrid systems.

## Related Work & Insights
- **vs Morfessor**: A traditional statistical method that performs unsupervised morphological segmentation based on the Minimum Description Length principle. This paper demonstrates that supervised BERT-based approaches substantially outperform it.
- **vs Character-level Models like ByT5**: ByT5 operates natively at the character level and might be better suited for morphological analysis tasks, presenting a promising alternative.
- **vs UDPipe/Stanza**: While these tools provide comprehensive NLP pipelines including morphological analysis, their morpheme segmentation capabilities are limited, and the proposed method significantly leads in segmentation accuracy.

## Rating
- Novelty: ⭐⭐⭐ The methodology is relatively standard (BERT + sequence labeling), but the focus on this specific language family is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Features multilingual, multi-model, and cross-lingual transfer experiments.
- Writing Quality: ⭐⭐⭐⭐ Well-introduced linguistic background.
- Value: ⭐⭐⭐ Provides valuable reference for Slavic NLP research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SAM-Veteran: An MLLM-based Human-like SAM Agent for Reasoning Segmentation](../../ICLR2026/segmentation/sam-veteran_an_mllm-based_human-like_sam_agent_for_reasoning_segmentation.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](../../ICCV2025/segmentation/tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[ECCV 2024\] Diffusion Models for Open-Vocabulary Segmentation](../../ECCV2024/segmentation/diffusion_models_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] F-LMM: Grounding Frozen Large Multimodal Models](../../CVPR2025/segmentation/f-lmm_grounding_frozen_large_multimodal_models.md)

</div>

<!-- RELATED:END -->

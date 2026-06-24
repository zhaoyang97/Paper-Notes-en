---
title: >-
  [Paper Note] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus
description: >-
  [ACL 2025][Multimodal VLM][Indian languages] Builds COSMMIC, the first comment-sensitive multimodal multilingual dataset for Indian languages—covering 9 Indian languages, 4,959 article-image pairs, and 24,484 reader comments; proposes comment filtering (IndicBERT) and image classification (CLIP) enhancement schemes, and establishes summarization and headline generation benchmarks using GPT-4 and LLaMA3.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Indian languages"
  - "Multimodal"
  - "Comment-sensitive"
  - "Summarization"
  - "Headline generation"
date: 2026-05-08
content_hash: 84d81987db77cfd1
---

# COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus

**Conference**: ACL 2025  
**arXiv**: [2506.15372](https://arxiv.org/abs/2506.15372)  
**Code**: None  
**Area**: Multilingual / Multimodal  
**Keywords**: Indian languages, Multimodal, Comment-sensitive, Summarization, Headline generation

## TL;DR

Builds COSMMIC, the first comment-sensitive multimodal multilingual dataset for Indian languages—covering 9 Indian languages, 4,959 article-image pairs, and 24,484 reader comments; proposes comment filtering (IndicBERT) and image classification (CLIP) enhancement schemes, and establishes summarization and headline generation benchmarks using GPT-4 and LLaMA3.

## Background & Motivation

**Background**: Reader comments contain reactions, sentiments, and supplementary information about articles, which can enhance the quality of summarization and headline generation; however, existing datasets are almost exclusively in English.  
**Limitations of Prior Work**: There is a lack of comment-sensitive multimodal datasets for Indian languages; existing multilingual summarization resources do not combine reader comments and associated images.  
**Key Challenge**: India has billions of internet users speaking multiple languages, yet NLP resources are severely lacking.  
**Goal**: To construct the first Indian language dataset that simultaneously covers multilingual, multimodal, and comment-sensitive dimensions.  
**Key Insight**: Crawling articles, images, and comments from mainstream Indian news websites.  
**Core Idea**: Comments are not merely noise; after quality filtering, they can provide valuable contextual signals to enhance generation tasks.

## Method

### Overall Architecture

Data construction: Crawling from news websites in 9 languages $\rightarrow$ Comment quality filtering (IndicBERT classification) $\rightarrow$ Image relevance filtering (CLIP matching) $\rightarrow$ GPT-4/LLaMA3 summarization and headline baselines.

### Key Designs

1. **9-Language Data Crawling and Standardization**:
    - **Function**: Crawls article-image-comment triplets in 9 languages from mainstream Indian news websites
    - **Mechanism**: Covers Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, and Odia, with 400-600 articles per language
    - **Design Motivation**: India is one of the countries with the highest linguistic diversity in the world; covering 9 major languages ensures representation

2. **Comment Quality Filtering (IndicBERT)**:
    - **Function**: Filters low-quality/irrelevant comments using an IndicBERT classifier
    - **Mechanism**: Trains a binary classifier to judge whether comments are relevant to the article and informative, retaining high-quality comments
    - **Design Motivation**: Many raw comments are spam, irrelevant, or of low quality; directly using them without filtering introduces noise and degrades generation quality

3. **Image Relevance Filtering (CLIP)**:
    - **Function**: Uses CLIP to determine if the article's associated image is relevant to the article content
    - **Mechanism**: Computes the CLIP similarity between the image and the article text to filter out irrelevant images
    - **Design Motivation**: News illustrations are sometimes advertisements or irrelevant images; retaining only relevant images is necessary to leverage multimodal enhancement

### Loss & Training

The IndicBERT comment filter is trained using binary cross-entropy. Baseline experiments perform zero-shot and few-shot summarization and headline generation using GPT-4 and LLaMA3.

## Key Experimental Results

### Main Results

Summarization generation ROUGE scores (averaged across 9 languages):

| Model | Input | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|------|:---:|:---:|:---:|
| GPT-4 | Article | 32.5 | 12.1 | 28.3 |
| GPT-4 | Article + Comments | **35.2** | **14.3** | **30.8** |
| LLaMA3 | Article | 28.7 | 9.8 | 24.5 |
| LLaMA3 | Article + Comments | **31.4** | **11.5** | **27.1** |

### Ablation Study

Impact of comment filtering:

| Comment Quality | ROUGE-L Change |
|---------|:---:|
| No Comments | Baseline |
| + All Comments (Unfiltered) | +1.2 |
| + Filtered Comments | **+2.5** |

### Key Findings

1. **Comments indeed improve summarization quality**: Average ROUGE-L increases by 2.5 points after incorporating comments.
2. **Comment filtering is crucial**: Filtered comments yield double the performance gain compared to unfiltered ones.
3. **GPT-4 significantly outperforms LLaMA3**: The gap is even wider for low-resource languages.
4. **Multimodal effects of combining images and comments are limited**: Text + comments is already sufficient.

## Highlights & Insights

- **First Indian language dataset simultaneously covering three dimensions**: Multilingual + Multimodal + Comment-sensitive.
- **Comment filtering methodology**: Proves that "comments do not equal noise—filtered comments are valuable signals".
- **Coverage of 9 languages**: Provides infrastructure for low-resource Indian language NLP research.
- **GPT-4 vs LLaMA3 baseline**: Provides a reference point for future work.

## Limitations & Future Work

- Does not cover all Indian languages (such as Punjabi, Assamese, etc.).
- The distribution of the number of comments is uneven across different languages.
- Baseline experiments are mainly zero/few-shot, without fine-tuning.
- The multimodal enhancement effect of images is limited.

## Related Work & Insights

- **vs XL-Sum (English + multilingual summarization)**: Lacks comments—this work introduces the comment dimension.
- **vs IndicNLPSuite**: Covers various Indian language NLP tasks but lacks summarization/comments.
- **Insight**: NLP research in low-resource languages requires a "trinity" dataset (multimodal + multilingual + contextual signals).

## Rating

- Novelty: ⭐⭐⭐⭐ A dataset construction work, with innovation in three-dimensional fusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Baseline experiments are relatively basic.
- Writing Quality: ⭐⭐⭐⭐ Data construction is clearly described.
- Value: ⭐⭐⭐⭐⭐ Fills the gap for low-resource Indian language NLP research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ACL 2025\] SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning](singakids_a_multilingual_multimodal_dialogic_tutor_for_language_learning.md)
- [\[ACL 2025\] Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model](centurio_multilingual_vlm.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)

</div>

<!-- RELATED:END -->

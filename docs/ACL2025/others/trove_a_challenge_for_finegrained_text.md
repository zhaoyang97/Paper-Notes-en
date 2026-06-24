---
title: >-
  [Paper Note] TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification
description: >-
  [ACL 2025][Text Provenance] Proposes the TROVE text provenance challenge, which traces each sentence in the target text back to specific source sentences in the source documents and classifies their fine-grained relationships (quotation, compression, inference, etc.), covering multi-document and long-document scenarios.
tags:
  - "ACL 2025"
  - "Text Provenance"
  - "Sentence Tracing"
  - "Relation Classification"
  - "Long Document"
  - "Multi-Document"
date: 2026-05-08
content_hash: 1fc4e442f43ce74d
---

# TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification

**Conference**: ACL 2025  
**arXiv**: [2503.15289](https://arxiv.org/abs/2503.15289)  
**Code**: [GitHub](https://github.com/ZNLP/ZNLP-Dataset)  
**Area**: Others  
**Keywords**: Text Provenance, Sentence Tracing, Relation Classification, Long Document, Multi-Document

## TL;DR

Proposes the TROVE text provenance challenge, which traces each sentence in the target text back to specific source sentences in the source documents and classifies their fine-grained relationships (quotation, compression, inference, etc.), covering multi-document and long-document scenarios.

## Background & Motivation

**Background**: The reliability and traceability of LLM-generated texts have gained significant attention. Existing work on citation generation and fact verification mainly focuses on document-level or coarse-grained provenance.

**Limitations of Prior Work**: High-risk domains (e.g., law, healthcare) require understanding the source and generation mechanism of each sentence, yet there is a lack of sentence-level fine-grained provenance datasets and evaluation methods.

**Key Challenge**: Existing works only focus on single-document-level source identification, failing to meet the fine-grained provenance requirements in multi-document and long-document scenarios.

**Goal**: To provide a fine-grained text provenance dataset and evaluation framework covering multiple scenarios, multiple languages, and various source text lengths.

**Key Insight**: Constructing provenance data based on three public datasets (LongBench, LooGLE, and CRUD-RAG), combined with a multi-retriever approach and a three-stage annotation using GPT-4o.

**Core Idea**: Text provenance should not only trace source sentences but also classify the fine-grained target-source relationships (quotation/compression/inference/others).

## Method

### Overall Architecture

Constructs a provenance dataset covering 11 scenarios (QA and summarization), bilingual in Chinese and English, with different source text lengths (0-5k, 5k-10k, 10k+), ensuring quality through a three-stage annotation process.

### Key Designs

1. **Multi-Retriever Sentence Retrieval**: Jointly utilizes three retrievers—BM25, Dense Retrieval, and LCS (Longest Common Subsequence). The union of hits from at least two retrievers is selected as candidate source sentences, with top-$k=10$.
2. **GPT-4o Provenance Annotation**: Conducts fine-grained annotation based on the candidate source sentences, identifying source sentences and classifying their relations into four categories: Quotation, Compression, Inference, and Others (such as negation).
3. **Manual Provenance Verification**: Eight graduate students spent approximately 510 hours reviewing the annotations to verify and supplement the source sentences missed by GPT-4o, costing an average of $0.20 per sentence.

### Evaluation Strategy

Proposes a system of 13 metrics, including macro/micro averaged Track-P/R/F1 (source tracing), Relation-P/R/F1 (relationship classification), and Overall F1. It supports two evaluation paradigms: direct prompting and retrieval-augmented evaluation.

## Key Experimental Results

### Main Results

| Model | Direct Prompting | Retrieval-Augmented |
|------|---------|---------|
| GPT-4o | - | Best Closed-Source |
| DeepSeek-V3 (671B) | - | Best Open-Source |
| LLama3-8B | 4.71 | 30.96 |
| Qwen2.5-14B | - | Outperforms 7B in the same series |
| Vicuna-7B | 7.08 | 22.74 |
| ChatGLM-6B | 0.02 | 3.47 |

### Ablation Study

| Retrieval Method | Track F1 (Macro) | Overall F1 |
|---------|-----------------|------------|
| LCS | 29.41 | 14.67 |
| BM25 | 35.70 | 17.81 |
| Dense | 28.28 | 14.10 |
| Union (≥2) | 46.17 | 22.82 |

### Key Findings

- Retrieval augmentation is crucial for provenance; all models perform significantly better under the retrieval-augmented setting compared to direct prompting.
- Larger models demonstrate better performance in complex relationship classification.
- Closed-source models generally lead, but open-source models exhibit significant potential when combined with retrieval augmentation.
- Relationship classification is more challenging than source sentence tracing.

## Highlights & Insights

- First to define text provenance as a dual task of sentence-level tracing and relationship classification, with a granularity far exceeding existing works.
- The three-stage annotation pipeline (multi-retriever $\rightarrow$ GPT-4o $\rightarrow$ human) serves as a practical annotation methodology for long documents.
- Covers both Chinese and English bilingual data and a variety of source text lengths, providing a comprehensive evaluation.

## Limitations & Future Work

- The dataset size is limited (approx. 5,000 sentences); extending it to more domains and languages is worth exploring.
- A sliding window approach is used when the source text exceeds the model's context length, which may result in a loss of cross-window information.
- The definition of relationship types is relatively coarse and can be further refined (e.g., classifying inference into inductive/deductive).

## Related Work & Insights

- Complements citation generation, fact verification, and grounded generation.
- Provides a new evaluation perspective for the interpretability and traceability of RAG systems.
- The multi-retriever fusion strategy possesses universal value for long-document tasks.
- The GPT-4o + human review workflow for data annotation can serve as a general paradigm for long-document annotation.
- The provenance task can be extended to scenarios such as code generation and academic writing.

## Supplemental Technical Details

- Dataset distribution: Chinese single-document averages 196 sentences per source document; English single-document averages 637 sentences per source document.
- Average number of source sentences per target sentence: 7.04 sentences for Chinese single-document, 1.97 sentences for English multi-document.
- Annotation agreement (Fleiss' Kappa): 0.60-0.74 for tracing, 0.48-0.62 for relationship classification, and 0.44-0.70 for GPT-4o correction.
- Evaluation utilizes a sliding window to handle ultra-long source texts: dividing the input into chunks such as $0\text{-}M$, $M\text{-}2M$, and $2M\text{-}3M$ to process them independently before merging.
- Breakdown of relationship types: Quotation (verbatim/partial copy), Compression (summarization/paraphrase), Inference (expansion/generalization/specialization), Others (negation, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ Novel task definition, but the data construction methodology is relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 11 models with multi-dimensional analysis, though some results are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rigorous task definition.
- Value: ⭐⭐⭐⭐ Provides an important foundation for the traceability of LLM-generated content.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)

</div>

<!-- RELATED:END -->

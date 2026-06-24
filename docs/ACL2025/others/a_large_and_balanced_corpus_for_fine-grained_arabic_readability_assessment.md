---
title: >-
  [Paper Note] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment
description: >-
  [ACL 2025][Arabic readability] This work constructs Barec—the first large-scale, balanced, and fine-grained Arabic readability assessment corpus containing over 69K sentences, 1M words, and 19 grading levels, annotated by 6 professional educators. It benchmarks 4 Arabic BERT models × 4 input variants × 5 loss functions, revealing that the morphological tokenization input D3Tok combined with regression loss achieves a QWK of 84.0%.
tags:
  - "ACL 2025"
  - "Arabic readability"
  - "large-scale corpus"
  - "19-level grading"
  - "sentence-level annotation"
  - "ordinal classification"
date: 2026-05-08
content_hash: bcb02c2493d69faf
---

# Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment

**Conference**: ACL 2025  
**arXiv**: [2502.13520](https://arxiv.org/abs/2502.13520)  
**Code**: None  
**Area**: Others  
**Keywords**: Arabic readability, large-scale corpus, 19-level grading, sentence-level annotation, ordinal classification

## TL;DR

This work constructs Barec—the first large-scale, balanced, and fine-grained Arabic readability assessment corpus containing over 69K sentences, 1M words, and 19 grading levels, annotated by 6 professional educators. It benchmarks 4 Arabic BERT models × 4 input variants × 5 loss functions, revealing that the morphological tokenization input D3Tok combined with regression loss achieves a QWK of 84.0%.

## Background & Motivation

**Text readability directly impacts educational outcomes.** Aligning reading materials with student levels is crucial for promoting independent reading and comprehension. While English has a mature 27-level Fountas & Pinnell grading system, Arabic also possesses Taha-Thomure's 19-level system. However, the latter has long been restricted to manual book-level ratings, lacking the large-scale corpora required to train automatic assessment models.

**Existing Arabic readability resources are severely insufficient.** DARES is limited in scale and sourced solely from Saudi school textbooks; SAMER only offers coarse-grained 5-level grading. More fundamentally, most efforts evaluate at the document level rather than the sentence level, and cover limited domains and sources.

**The morphological richness of Arabic presents unique challenges.** Affixes, compounding, and inflectional changes in Arabic carry substantial linguistic information that affects readability, but standard tokenization often obscures these cues. **The core contribution of Barec is twofold**: (1) constructing the first large-scale, fine-grained corpus, and (2) demonstrating that preserving inflectional information via morphological tokenization input variants is crucial for automatic assessment.

## Method

### Overall Architecture

Corpus construction (annotation guideline design $\rightarrow$ text collection $\rightarrow$ professional annotation $\rightarrow$ quality control) $\rightarrow$ automatic evaluation benchmarking (systematic experiments with multi-model × multi-input-variant × multi-loss-function combinations).

### Key Designs

1. **19-Level Readability Grading System**:
    - Function: Establish a 19-level grading system from kindergarten to graduate school based on the Taha/Arabi21 system.
    - Mechanism: Named after the Arabic alphabetical Abjad order (1-alif to 19-qaf), with finer intervals in lower levels (where the most significant changes occur in early reading stages). Annotation is based on 6 dimensions: spelling (word length/syllables), word count, morphology (affixes/inflections), syntax, lexical complexity, and content/prior knowledge. Each sentence is assigned a level based on its most challenging linguistic phenomenon.
    - Design Motivation: Fine-grained grading is more useful for educational practices, as students need to perceive incremental progress. The 19 levels can be fully aligned and collapsed into 7/5/3 levels, flexibly accommodating various applications.

2. **Corpus Construction and Quality Control**:
    - Function: The complete process of collection, annotation, double-review, and Inter-Annotator Agreement (IAA) testing.
    - Mechanism: 1,922 documents were collected from 30 sources (with 25% being newly digitized content entered manually), covering three major domains (humanities, social sciences, and STEM) and three target audiences (basic, advanced, and expert). Six annotators, all experienced Arabic language educators, performed the formal annotation after undergoing three rounds of pilot annotation. Nineteen shared annotation sets (100 sentences each) were utilized for IAA evaluation, and all sentences in the first phase underwent a double-review process.
    - Design Motivation: Ensure balance (domain × level × audience) and high quality (pilot annotation + IAA + double review).

3. **Morphological Tokenization Input Variants**:
    - Function: Design four input variants to evaluate the impact of morphological information on automatic assessment.
    - Mechanism: Word (standard tokenization with diacritics removed), Lex (words replaced by lexemes), D3Tok (segmented into stem + affixes using CamelTools), and D3Lex (stem replaced by lexeme in D3Tok). D3Tok retains inflectional information (such as dual markers and affix types), which standard Word tokenization tends to obscure.
    - Design Motivation: The morphological complexity of Arabic is a key factor in readability, and standard tokenization loses inflectional cues that human annotators rely on.

### Loss & Training

The task is formulated as ordinal classification. Five losses are compared: Cross-Entropy (CE), Ordinal Log Loss (OLL), Soft Labels (SOFT), Earth Mover's Distance (EMD), and Regression (MSE). The experiments are conducted in two stages: Stage 1 identifies the optimal model and input variant using CE, and Stage 2 compares all loss functions using the best combination. The training uses a learning rate of 5e-5, a batch size of 64, 6 epochs, and an NVIDIA V100 GPU.

## Key Experimental Results

### Annotator Consistency (IAA)

| Metric | Pairwise Agreement | Agreement with Consolidated Label |
|------|---------|----------|
| Acc19 (Exact Match) | 61.1% | 71.7% |
| ±1 Acc19 | 74.4% | 82.3% |
| Mean Distance | 0.94 level | 0.65 level |
| QWK | 81.8% | 88.1% |

### Best Model + Input (Dev Set, CE Loss)

| Input | Model | Acc19 | ±1 Acc19 | Dist | QWK |
|------|------|-------|---------|------|-----|
| Word | AraBERTv02 | 55.8% | 69.2% | 1.17 | 79.2% |
| D3Tok | **AraBERTv2** | **56.6%** | **69.9%** | **1.14** | **80.0%** |
| Lex | MARBERTv2 | 50.1% | 64.9% | 1.31 | 77.0% |
| D3Lex | AraBERTv2 | 53.2% | 67.1% | 1.24 | 78.6% |

### Loss Function Comparison (AraBERTv2 + D3Tok)

| Loss | Acc19 | ±1 Acc19 | Dist | QWK |
|------|-------|---------|------|-----|
| CE | 56.6% | 69.9% | 1.14 | 80.0% |
| EMD | 55.3% | 70.3% | 1.11 | 81.2% |
| OLL(1.5) | 47.3% | 71.1% | 1.13 | **82.8%** |
| Reg | 43.1% | **73.1%** | **1.13** | **84.0%** |
| SOFT(3) | 56.4% | 69.9% | 1.14 | 80.1% |

### Multi-Granularity Collapsing Results (AraBERTv2 + D3Tok + CE)

| Granularity | Accuracy |
|------|--------|
| 19 levels | 56.6% |
| 7 levels | 65.9% |
| 5 levels | 70.3% |
| 3 levels | 76.5% |

### Key Findings

- **D3Tok (Morphological Tokenization) is consistently optimal**: Retaining affix and inflectional information provides the model with linguistic cues that human annotators also rely on.
- **AraBERTv2 + D3Tok achieves outstanding performance**: AraBERTv2's pre-training incorporates Farasa morphological segmentation, matching the D3Tok input well.
- **Regression loss significantly leads in QWK** (84.0% vs. 80.0% for CE): However, it yields the lowest exact match rate (43.1%), indicating that regression optimizes distance rather than classification boundaries.
- **Ordinal loss (OLL) also outperforms CE**: This indicates that leveraging the ordered relationships among levels is valuable.
- **Human annotator consistency sets an upper bound for automatic models**: Pairwise Acc19 is only 61.1%, meaning the model’s 56.6% is already close to human performance.

## Highlights & Insights

- **Resource contribution with both scale and quality**: Over 69K sentences across 19 levels, annotated by 6 professional annotators with 3 rounds of pilot annotation and a double-review phase, far exceeding existing Arabic readability resources.
- **Innovative input of morphological tokenization**: This demonstrates the importance of preserving inflectional information in morphologically rich languages for NLU tasks—a finding that can be generalized to languages like Turkish and Finnish.
- **Multi-granularity alignment design**: The 19-level system can be losslessly collapsed into 7/5/3 levels, allowing a single corpus to serve various research requirements.
- **Systematic experiments on ordinal classification**: A comprehensive comparison of 5 loss functions provides valuable references for ordinal classification tasks.

## Limitations & Future Work

- **Sentence-level evaluation only**: It ignores the impact of contextual coherence on readability; document-level evaluation requires separate modeling.
- **Extremely high annotation cost**: 92.6K sentence annotations (including redundancy and IAA) relying on 5 experts with Taha/Arabi21 experience.
- **Fewer samples in high levels**: Professional/academic texts are harder to collect, leading to an not fully balanced distribution.
- **Only BERT-like models**: The performance of generative models such as GPT/LLaMA as ordinal classifiers has not been tested.
- **Cross-lingual transfer unexplored**: The applicability of Arabic annotation guidelines and resources to other Arabic dialects or Semitic languages remains unknown.

## Related Work & Insights

- **vs. DARES (Saudi school materials)**: DARES is single-sourced and small-scale, whereas Barec relies on 30 sources, spans 3 domains, and is over 50 times larger.
- **vs. SAMER (5 levels)**: SAMER has coarse granularity and low coverage, whereas Barec's 19 levels are better suited for the fine-grained grading needs of educational scenarios.
- **vs. English readability datasets**: While English possesses a vast amount of physical and digital resources, Barec fills the blank for fine-grained Arabic readability assessment.
- **vs. Traditional feature-based methods (OSMAN, Al-Khalifa, etc.)**: These methods rely on surface features like word/sentence length, whereas Barec supports pre-trained models for end-to-end ordinal classification.

## Rating

- Novelty: ⭐⭐⭐ Primarily a resource contribution; while methodological innovation is limited, the morphological tokenization inputs and ordinal loss comparisons are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic and comprehensive, featuring 4 models × 4 inputs × 5 losses + IAA analysis + multi-granularity evaluation.
- Writing Quality: ⭐⭐⭐⭐ Highly detailed corpus descriptions and annotation processes, with transparent guideline design.
- Value: ⭐⭐⭐⭐ Significant resource value for the Arabic NLP community and educational technology; findings regarding morphological tokenization offer cross-lingual insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)
- [\[ACL 2025\] TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)

</div>

<!-- RELATED:END -->

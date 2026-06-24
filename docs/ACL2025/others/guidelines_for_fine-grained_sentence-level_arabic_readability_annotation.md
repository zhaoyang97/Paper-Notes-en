---
title: >-
  [Paper Note] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation
description: >-
  [ACL 2025][Arabic Readability] This paper proposes the BAREC corpus and its annotation guidelines, which represent a large-scale Arabic sentence-level readability evaluation resource containing over 69K sentences across 19 readability levels, and establishes benchmark models for automated readability assessment based on this resource.
tags:
  - "ACL 2025"
  - "Arabic Readability"
  - "Sentence-level Annotation"
  - "Fine-grained Leveling"
  - "Corpus Construction"
  - "Text Complexity"
date: 2026-05-08
content_hash: 3a92367add2fc588
---

# Guidelines for Fine-grained Sentence-level Arabic Readability Annotation

**Conference**: ACL 2025  
**arXiv**: [2410.08674](https://arxiv.org/abs/2410.08674)  
**Code**: Yes (public dataset and annotation guidelines)  
**Area**: Others  
**Keywords**: Arabic Readability, Sentence-level Annotation, Fine-grained Leveling, Corpus Construction, Text Complexity

## TL;DR

This paper proposes the BAREC corpus and its annotation guidelines, which represent a large-scale Arabic sentence-level readability evaluation resource containing over 69K sentences across 19 readability levels, and establishes benchmark models for automated readability assessment based on this resource.

## Background & Motivation

Text readability plays an important role in reading comprehension, knowledge retention, and learning engagement. In educational contexts, matching texts of appropriate difficulty to students' reading abilities is a key requirement. While English has systems like Fountas and Pinnell's 27-level system, for Arabic, Taha/Arabi21 proposed a 19-level system, which, however, was previously mainly used for document-level annotation of children's books.

Existing Arabic readability resources suffer from several core limitations:

**Insufficient Grain Size**: Most works perform annotations at the document level (e.g., DARES only has 12 levels) or focus solely on the lexical level (e.g., SAMER Lexicon's 5 levels), lacking sentence-level fine-grained resources.

**Small Scale**: For instance, ReadMe++ contains only 1,945 sentences, and ZAEBUC contains only 214 documents.

**Incomplete Coverage**: Most resources only cover textbooks or specific domains, lacking comprehensive coverage across genres and educational stages.

**Inconsistent Standards**: Some use CEFR standards, while others use school grade levels, lacking a unified Arabic-specific readability framework.

## Method

### Overall Architecture

BAREC (Balanced Arabic Readability Evaluation Corpus) adopts the 19-level naming system of Taha/Arabi21 (based on the Arabic alphabetical Abjad order: 1-alif to 19-qaf). It extends and adapts the guidelines originally designed for book-level annotation to make them suitable for sentence-level annotation tasks. The entire framework also provides three coarse-grained mapping versions: 7-level, 5-level, and 3-level scales.

### Key Designs

1. **Six-dimensional Text Feature System**: Six dimensions are defined to determine readability levels: word count (used only for level 11 and below), orthography and phonology (focusing on syllable counts and special letters), morphology (inflection and derivation), syntax (from words to complex clauses), vocabulary (from dialect-MSA overlapping words to technical terms), and ideas and content (from familiar concepts to abstract reasoning). Different dimensions operate across different level ranges, forming a "pyramid" structure.

2. **Annotation Process Design**: Annotators first read the sentences to check for defects, then determine the meaning (selecting the simpler interpretation in case of ambiguity), establish the minimum possible level based on word count, and finally look for features to elevate the level to the highest possible value. For example, "سلوكي مسؤوليتي" (My behavior is my responsibility) has two words $\rightarrow$ starts at level 2 $\rightarrow$ has a first-person pronoun $\rightarrow$ level 3 $\rightarrow$ the second word has five syllables $\rightarrow$ finally annotated as level 6.

3. **Quality Control Mechanism**: An annotation team of six native Arabic educators (A0-A5) was formed. Three rounds of shared pilot phases were used for training and refining the guidelines, and 19 rounds of blind tests evaluated Inter-Annotator Agreement (IAA) before final consensus labels were reconciled. 25% of the annotated data was excluded due to issues, double-labeling, or consensus rounds.

4. **Handling of Arabic Diacritics**: Unlike Taha-Thomure (2017), this work does not rely on diacritics when evaluating readability. In ambiguous cases, the simpler meaning is selected (e.g., "هذه سلطة بدون خيار" is read as "this is a salad without cucumber" rather than "this is authority without choice").

### Corpus Composition

Collected from 1,922 documents, the corpus covers:
- **Three domains**: Humanities & Arts (73%), Social Sciences (21%), STEM (7%)
- **Three reader groups**: Basic (40%), Intermediate (33%), Advanced (27%)
- **30 data sources**, including public domain, fair use, and licensed materials, with 25% of the sentences sourced from newly digitized web and physical sources.

| Metric | Total |
|---------|------|
| Number of Documents | 1,922 |
| Number of Sentences | 69,441 |
| Number of Words | 1,039,371 |
| Readability Levels | 19 (mappable to 7/5/3) |

### Loss & Training

The baseline models for automatic readability evaluation are fine-tuned using AraBERTv02:
- Trained on an NVIDIA V100 GPU for 3 epochs
- Learning rate of $5\times10^{-5}$, batch size of 64
- Fine-tuned as a 19-class classification task using cross-entropy loss

## Key Experimental Results

### Inter-Annotator Agreement (IAA)

| Phase | Number of Sets | Distance | Acc19 | $\pm$1 Acc19 | QWK |
|------|--------|------|-------|----------|-----|
| Pilot 3 | 1 | 1.69 | 37.5% | 58.5% | 79.3% |
| Phase 1 | 2 | 1.38 | 48.4% | 64.4% | 80.2% |
| Phase 2A | 6 | 1.21 | 49.4% | 67.4% | 72.4% |
| Phase 2B | 10 | 0.80 | 67.6% | 78.3% | 78.8% |
| Overall Macro Average | 19 | 1.04 | 58.2% | 72.3% | 76.9% |
| Phase 2 Micro Average | 16 | 0.95 | 61.1% | 74.4% | **81.8%** |

### Automatic Classification Results (Different Training Data Sizes)

| Training Ratio | Distance | Acc19 | $\pm$1 Acc19 | QWK | Acc3 |
|---------|------|-------|----------|-----|------|
| 12.5% | 1.35 | 45.0% | 61.3% | 77.2% | 71.3% |
| 25.0% | 1.33 | 46.9% | 63.0% | 77.6% | 72.3% |
| 50.0% | 1.16 | 52.4% | 68.1% | 80.7% | 74.0% |
| 100.0% | 1.09 | 55.8% | 69.4% | 81.0% | 74.7% |

### Key Findings

1. **Continuous Improvement of IAA**: From Pilot 3 to Phase 2B, annotator consistency steadily improved, with the final Phase 2 micro-average QWK reaching 81.8%, indicating "substantial agreement".
2. **Analysis of Disagreement Sources**: 45% of disagreements came from domain-specific jargon (different thresholds for defining "general" vs. "specialized"), 25% from basic linguistic features, 18% from general advanced vocabulary, and 12% from emotional or figurative content.
3. **Manageable Gap between Models and Humans**: The QWK of the best model (81.0%) is only 0.8% lower than the human Phase 2 micro-average (81.8%).
4. **Effective Coarse-grained Mapping**: When mapping from 19 levels down to 3 levels, the $\pm$1 accuracy improved from 74.4% to 97.3%.

## Highlights & Insights

- **Systematic Annotation Guidelines Design**: The six dimensions, the pyramid leveling structure, and the bottom-up hierarchical process are highly systematic, making a complex and subjective judgment task highly operational.
- **Pragmatic Handling of Diacritics**: The deliberate choice not to rely on diacritics for readability evaluation enhances objectivity and practical utility.
- **Multi-granular Mapping**: The hierarchical mapping of 19$\rightarrow$7$\rightarrow$5$\rightarrow$3 retains research utility (fine grain) while addressing practical application demands (coarse grain).
- **Balanced Point of Sentence-level Annotation**: Positioned between the document level (too coarse) and the word level (too fine), it controls annotation variables while successfully capturing synthetic and semantic complexities.

## Limitations & Future Work

1. **Geographical Bias**: The guidelines are based on MSA usage habits in Egypt, the Gulf, and the Levant, lacking coverage of North African (Maghrebi) linguistic variants.
2. **Subjectivity Issue**: The high levels (levels 15-19) primarily rely on vocabulary and content judgment, which are inherently more subjective and lead to lower inter-annotator agreement at these levels.
3. **Domain Imbalance**: Humanities and Arts account for 73% of the sentences, while STEM represents only 7%, which may affect readability modeling in STEM domains.
4. **Lack of Lexical Level Anchoring**: Future work needs to develop a 19-level readability dictionary to anchor guideline judgments.

## Related Work & Insights

- This work aligns with the methodology of English readability assessment (Fountas & Pinnell's 27-level system) but introduces substantial customization for the unique linguistic features of Arabic.
- The word-level readability annotation of the SAMER project and the CEFR annotations of ReadMe++ serve as important complementary resources.
- Insights for Chinese readability evaluation: One could similarly design dimensions of Chinese characters, words, syntax, and semantics to define Chinese readability levels.

## Rating

- **Novelty**: ⭐⭐⭐ — Fine-grained 19-level sentence-level readability annotation for Arabic is novel, but the methodological framework (annotation $\rightarrow$ IAA $\rightarrow$ fine-tuning models) is relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The IAA analysis is highly detailed (including phases, confusion matrices, and disagreement case studies), and the learning curve and multi-granularity evaluations are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly structured and rich in tables, with the pyramid diagram and examples greatly facilitating the understanding of the complex leveling system.
- **Value**: ⭐⭐⭐⭐ — Possesses solid resource value for the Arabic NLP community, and the design methodology of the annotation guidelines can be generalized to other low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)
- [\[ACL 2025\] TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] MEXMA: Token-level Objectives Improve Sentence Representations](mexma_token-level_objectives_improve_sentence_representations.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)

</div>

<!-- RELATED:END -->

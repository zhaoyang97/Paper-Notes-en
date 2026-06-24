---
title: >-
  [Paper Note] Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study
description: >-
  [ACL 2025][Text Generation][Grammatical Error Correction] This paper proposes a language-neutral text editing approach (SWEET) that does not rely on language-specific edit sets. By introducing data-driven automated extraction and compression strategies for edit tags, this work successfully applies the text editing paradigm to Arabic Grammatical Error Correction (GEC) for the first time, achieving state-of-the-art performance across multiple benchmarks while increasing inferen…
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Grammatical Error Correction"
  - "Text Editing"
  - "Arabic"
  - "Edit Tags"
  - "Sequence Labeling"
date: 2026-05-08
content_hash: 5ee755ee8ae37284
---

# Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study

**Conference**: ACL 2025  
**arXiv**: [2503.00985](https://arxiv.org/abs/2503.00985)  
**Area**: Text Generation  
**Keywords**: Grammatical Error Correction, Text Editing, Arabic, Edit Tags, Sequence Labeling  

## TL;DR

This paper proposes a language-neutral text editing approach (SWEET) that does not rely on language-specific edit sets. By introducing data-driven automated extraction and compression strategies for edit tags, this work successfully applies the text editing paradigm to Arabic Grammatical Error Correction (GEC) for the first time, achieving state-of-the-art performance across multiple benchmarks while increasing inference speed by over 6x.

## Background & Motivation

**Dominant Approaches for GEC**: Transformer-based Seq2Seq models have long dominated the GEC field. Although effective, they are computationally inefficient. Since most input tokens are copied unchanged to the output in GEC tasks, using full-sequence autoregressive generation introduces significant computational waste.

**Advantages and Challenges of Text Editing Methods**:
   - Text editing reformulates GEC as a sequence labeling problem, assigning edit tags to each input token, making it more efficient and interpretable than Seq2Seq.
   - However, existing methods (such as GECToR, PIE) rely on **manually designed language-specific edit tagsets**, which limits their adaptability to morphologically rich languages.

**Peculiarities of Arabic GEC**:
   - Arabic is a morphologically rich language, leading to a massive potential edit space.
   - It exhibits diglossia: Modern Standard Arabic (MSA) coexists with Dialectal Arabic (DA).
   - DA lacks standard orthography and requires CODAfication (Conventional Orthography for Dialectal Arabic).
   - Prior text editing methods have underperformed in Arabic GEC.

**Goal**: To develop a generic, data-driven text editing method where edit tags are automatically derived, eliminating reliance on language-specific designs while ensuring high efficiency and performance on morphologically rich languages like Arabic.

## Method

### Overall Architecture

The proposed approach is named SWEET (Subword-level Edit Tagger). The core process includes:
1. Word-level alignment of source-target sentence pairs.
2. Extraction of character-level edit operations.
3. Projection to the subword level.
4. Application of edit compression, splitting, and pruning strategies.
5. Training a sequence labeling model using a pretrained Arabic BERT.

### Edit Extraction

Based on weighted Levenshtein edit distance, word-level alignment is performed first, followed by character-level alignment within each aligned word pair, yielding 5 basic operations:

| Operation | Symbol | Description |
|------|------|------|
| Keep | K / K* | Character remains unchanged |
| Delete | D / D* | Character is deleted |
| Merge-Forward | M | Merge with the preceding word |
| Replace | R_[c] | Replace with character c |
| Insert | I_[c] | Insert character c at the current position |
| Append | A_[c] | Append character c at the end of the edit |

### Key Designs

#### Key Design 1: Edit Compression

**Design Motivation**: Words of different lengths might undergo the same type of edit but yield different tags due to length variations (e.g., two "keep" words might have tags KKK and KKKKK).

**Mechanism**: Consecutive identical operations are condensed into generalized tags—consecutive K $\rightarrow$ K\*, consecutive D $\rightarrow$ D\*, consecutive I $\rightarrow$ I\_[c\*], and consecutive A $\rightarrow$ A\_[c\*].

**Effect**: On QALB-2014, the edit vocabulary is reduced from 9,060 to 6,170 tags (a 32% reduction) without sacrificing coverage.

#### Key Design 2: Input Units

Projecting character-level edits to the subword level:

- Edits are first extracted at the word level and then projected onto subword boundaries.
- Unlike Straka et al. (2021), this work does not restrict the number of character-level operations within each subword edit.
- The number of unique edits drops by 44% (16,221 $\rightarrow$ 9,060) when moving from the word to the subword level.

#### Key Design 3: Edit Splitting

Punctuation errors account for an extremely high proportion of errors in MSA datasets (40% in QALB-2014, 15% in ZAEBUC). Therefore, punctuation edits (Pnx) and non-punctuation edits (NoPnx) are split into two subsystems:
1. The first system corrects non-punctuation errors.
2. The second system addresses punctuation errors on the corrected text.

#### Key Design 4: Edit Pruning

Edits in the training set with a frequency lower than a threshold $T$ are pruned (replaced with the "Keep" tag):
- Setting $T=10$ reduces the 6,170 edits to 683, with the OOV rate increasing only from 0.36% to 0.75%, and $F_{0.5}$ dropping by only 0.6%.
- This indicates that a large number of low-frequency edits contribute minimally to the model's upper-bound performance.

## Key Experimental Results

### MSA GEC Dev Set Results ($F_{0.5}$)

| Method | QALB-2014 | ZAEBUC |
|------|-----------|--------|
| Alhafni'23 Seq2Seq | 78.7 | 83.4 |
| Alhafni'23 Seq2Seq++ | 79.6 | 84.5 |
| GPT-4o | 77.2 | 84.3 |
| SWEET | 78.8 | 82.7 |
| SWEET² | 79.3 | 83.0 |
| SWEET²(NoPnx) + SWEET(Pnx) | **80.3** | 83.8 |
| 3-Ensemble | 81.1 | 85.6 |
| 4-Ensemble | **81.8** | **86.9** |

### MSA GEC Test Set Results ($F_{0.5}$)

| Method | QALB-2014 | QALB-2015 | ZAEBUC |
|------|-----------|-----------|--------|
| Alhafni'23 Seq2Seq++ | 79.6 | 80.3 | 83.1 |
| GPT-4o | 77.7 | 79.6 | 82.5 |
| SWEET² | 79.6 | 78.8 | 83.0 |
| SWEET²(NoPnx)+SWEET(Pnx) | **80.5** | 80.3 | 83.1 |
| 4-Ensemble | **81.7** | **82.9** | **87.2** |

### DA GEC Results (MADAR CODA $F_{0.5}$)

| Method | Dev | Test |
|------|--------|--------|
| Alhafni'24 Seq2Seq++ | 85.8 | 86.3 |
| GPT-4o | 53.8 | 55.9 |
| SWEET | **86.0** | **86.5** |
| 3-Ensemble | 88.4 | **88.9** |

### LLM Performance

| Model | QALB-2014 $F_{0.5}$ | MADAR CODA $F_{0.5}$ |
|------|-----------------|-----------------|
| GPT-3.5-turbo | 66.3 | 34.1 |
| GPT-4o | 77.2 | 53.8 |
| Fanar | 68.4 | 25.2 |
| Jais-13B-Chat | 46.0 | 14.3 |

LLMs perform significantly worse on DA GEC than on MSA GEC, and all underperform compared to SWEET.

### Efficiency Comparison

The SWEET model processes **inference over 6 times faster than existing Arabic GEC systems**—the sequence labeling framework for text editing is inherently more efficient than autoregressive generation.

### Key Findings

1. **Subword-level outperforms word-level**: The number of edits is reduced by 44%, and the OOV rate decreases from 1% to 0.36%.
2. **Edit compression is effective**: It further reduces the edit vocabulary size without sacrificing coverage.
3. **Edit splitting improves MSA**: Separating punctuation and non-punctuation edit training brings a $+0.7$ $F_{0.5}$ improvement on QALB-2014.
4. **Ensembling yields significant gains**: Multi-model ensembling via majority voting improves precision, with the 4-Ensemble achieving an $F_{0.5}$ of 86.9 on ZAEBUC.

## Highlights & Insights

1. **Language-neutral design**: Edit tags are entirely data-driven, eliminating the need for language-specific manual design, making it language-agnostic.
2. **Clever edit compression**: By generalizing symbols such as K* and D*, different length words can share edit tags, drastically reducing the tag space.
3. **Remarkable efficiency advantages**: A 6x speedup makes it highly suitable for practical deployment.
4. **Comprehensive ablation**: Detailed analysis of the impact of word-level vs. subword-level, compression vs. non-compression, split vs. non-split, and various pruning thresholds.

## Limitations & Future Work

1. **Serial inference required for edit splitting**: The decoupling of punctuation and non-punctuation edits requires a two-step inference process.
2. **Pruning discards low-frequency edits**: While this has minimal impact on the overall $F_{0.5}$, it might reduce the ability to correct rare errors.
3. **Limited DA GEC data**: MADAR CODA contains only 10K sentences, limiting the upper bound of DA GEC performance.
4. **Single-edit limitation**: The text editing paradigm may struggle with errors that require substantial rewriting.
5. **Lack of evaluation on other languages**: Although designed to be language-independent, experiments were only conducted on Arabic.

## Related Work & Insights

- **Text Editing for English GEC**: GECToR (Omelianchuk et al., 2020) requires manually designed edit tags; PIE (Awasthi et al., 2019) employs iterative editing.
- **Arabic GEC**: Alhafni et al. (2023) achieved the previous SOTA based on Seq2Seq with morphological preprocessing; the QALB Shared Tasks (Mohit et al., 2014) pioneered Arabic GEC datasets.
- **Dialectal Arabic**: Alhafni et al. (2024) formulated CODAfication as a DA GEC problem.

## Rating

⭐⭐⭐⭐ — The method is simple, practical, and resolves the adaptation bottlenecks of text editing methods on morphologically rich languages. The data-driven edit tag extraction concept holds universal value. With significant efficiency improvements and thorough experiments, this work is a solid contribution to the GEC field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] gec-metrics: A Unified Library for Grammatical Error Correction Evaluation](gec-metrics_a_unified_library_for_grammatical_error_correction_evaluation.md)
- [\[ACL 2025\] Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?](rethinking_evaluation_metrics_for_grammatical_error_correction_why_use_a_differe.md)
- [\[ACL 2025\] IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator](impara-ged_grammatical_error_detection_is_boosting_reference-free_grammatical_er.md)
- [\[ACL 2025\] An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)
- [\[ACL 2025\] A Representation Level Analysis of NMT Model Robustness to Grammatical Errors](a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)

</div>

<!-- RELATED:END -->

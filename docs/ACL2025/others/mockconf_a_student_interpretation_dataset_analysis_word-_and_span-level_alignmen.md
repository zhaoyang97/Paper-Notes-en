---
title: >-
  [Paper Note] MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines
description: >-
  [ACL 2025][Simultaneous interpreting] This paper constructs MockConf—a Czech-centric **student simultaneous interpreting dataset** (7 hours, 5 European languages), featuring manually annotated span-level and word-level alignments. Additionally, the paper releases a dedicated annotation tool, InterAlign, and establishes automatic alignment baselines along with an evaluation metric framework.
tags:
  - "ACL 2025"
  - "Simultaneous interpreting"
  - "parallel corpora"
  - "alignment annotation"
  - "span-level alignment"
  - "word-level alignment"
date: 2026-05-08
content_hash: 6eb92f88a91a038e
---

# MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines

**Conference**: ACL 2025  
**arXiv**: [2506.04848](https://arxiv.org/abs/2506.04848)  
**Code**: [Yes (GitHub)](https://github.com/J4VORSKY/MockConf) + [InterAlign Annotation Tool](https://github.com/J4VORSKY/InterAlign)  
**Area**: Others  
**Keywords**: Simultaneous interpreting, parallel corpora, alignment annotation, span-level alignment, word-level alignment

## TL;DR

This paper constructs MockConf—a Czech-centric **student simultaneous interpreting dataset** (7 hours, 5 European languages), featuring manually annotated span-level and word-level alignments. Additionally, the paper releases a dedicated annotation tool, InterAlign, and establishes automatic alignment baselines along with an evaluation metric framework.

## Background & Motivation

Simultaneous interpreting (SI) is an extremely challenging multilingual task where interpreters must translate speech into another language with minimal delay, often before the speaker has finished their sentence. To understand, analyze, and automate this complex process, specialized **parallel speech corpora** and corresponding alignment annotation tools are required.

However, existing parallel corpora for text translation and alignment algorithms fail to meet the requirements of SI because:
- SI does not follow traditional 1-to-1 sentence alignment.
- Interpreters perform operations such as compression, simplification, and generalization, resulting in specific types of deviations.
- Speech transcriptions contain disfluencies like hesitations and false starts.
- Long-distance speech segment interactions must be addressed.

Existing interpreting corpora (such as EUROPARI from the European Parliament) focus on professional interpreters, whereas data from student interpreters holds unique educational and research value. The MockConf dataset introduced in this paper, sourced from mock conferences in university courses, fills this gap.

## Method

### Overall Architecture

Data construction workflow: Recording collection → Automatic transcription (WhisperX) → Manual revision → Span- and word-level alignment annotation (using InterAlign) → Data analysis → Automatic alignment baselines.

### Key Designs

1. **Data Collection and Transcription**:

    - Source: Mock conferences from university interpreting courses, where students roleplay to deliver speeches, and other students perform simultaneous interpreting.
    - Languages: Czech (cs), English (en), French (fr), German (de), and Spanish (es), always translating from/to Czech.
    - Includes direct interpreting and relay interpreting (via Czech as pivot).
    - Automatic transcription is performed using WhisperX, followed by manual revision by native speakers.
    - Total duration is approximately 7 hours, split into dev (2h) and test (5h).

2. **InterAlign Annotation Tool**:

    - A web-based annotation tool specifically designed for long-text simultaneous interpreting alignment.
    - Supports parallel span-level and word-level annotations.
    - Addresses the limitation of existing alignment tools that fail to handle the unique requirements of simultaneous interpreting (e.g., non-sentence alignment, long-distance interactions).

3. **Span Alignment Annotation Schema** (7 tags):

| Category | Subcategory | Tag |
|------|--------|------|
| Translation | - | TRAN |
| Reformulation | Paraphrase | PARA |
| Reformulation | Summarization | SUM |
| Reformulation | Generalization | GEN |
| Error | Factual addition | ADDF |
| Error | Uninformative addition | ADDU |
| Error | Substitution | REPL |

4. **Word Alignment Annotation**: Annotating word alignments within each span-aligned pair, categorized into "sure" (context-independent translation equivalents) and "possible" (contingent on context or grammatical dependencies).

5. **Automatic Alignment Baseline System** (Three-step workflow):

    - **Coarse Alignment**: Uses BERTAlign to achieve high-precision $n$-to-$m$ sentence alignment.
    - **Sub-segmentation**: Uses SimAlign (itermax + XLM-R) to calculate word alignments and segment spans at punctuation alignments.
    - **Label Classification**: A simple classifier based on LaBSE similarity scores and span lengths.

### Evaluation Metrics

- **Segmentation Quality**: Precision / Recall / F1 + $P_k$ + WindowDiff
- **Span Alignment**: Exact match + Relaxed match (P/R/F1 after word-level expansion)
- **Word Alignment**: AER + F1
- **Label Correctness**: token-level accuracy + F1

## Key Experimental Results

### Dataset Statistics

| Split | Recordings | Duration | Source tokens | Target tokens |
|------|--------|------|-----------|-------------|
| dev | 10 | 1:59:31 | 13,545 | 12,372 |
| test | 29 | 5:01:16 | 37,719 | 30,413 |

### Distribution of Span Labels (Token Ratio)

| Label | Source (%) | Target (%) |
|------|---------|----------|
| TRAN | 42.82 | 52.16 |
| PARA | 17.91 | 22.08 |
| SUM | 11.89 | 9.07 |
| ADDF | 13.28 | 4.02 |
| GEN | 4.68 | 4.57 |
| ADDU | 5.45 | 3.91 |
| REPL | 3.96 | 4.18 |

### Inter-Annotator Agreement (Cohen's Kappa)

| Aspect | Source | Target |
|------|------|--------|
| Segmentation | 0.56 | 0.57 |
| Label | 0.41 | 0.25 |

### Automatic Alignment Baselines (Test Set)

| System | Seg F1↑ | Relaxed F1↑ | Exact w/o label↑ | AER↓ | Label acc↑ |
|------|---------|-------------|------------------|------|-----------|
| Random | ~16.6 | ~0.03 | 0.00 | 0.70 | ~37 |
| BA | ~53 | ~0.58 | ~12.5 | 0.33 | ~62 |
| BA+sub | ~60 | ~0.61 | ~15 | 0.37 | ~62 |
| BA+sub+lab | ~60 | ~0.61 | ~15 | 0.37 | ~60 |

### Key Findings

1. **Agreement reflects task difficulty**: Segmentation Kappa is moderate (0.56-0.57), while label Kappa is even lower (0.25-0.41), revealing that interpreting alignment annotation is highly subjective and ambiguous by nature.

2. **Translation is the most dominant label** (approx. 50%), followed by paraphrase (approx. 20%), which is highly intuitive.

3. **13.3% of source tokens are associated with factual omissions (ADDF)**: This holds significant reference value for interpreter training.

4. **Interpreted outputs are shorter than sources**: The length ratio of direct interpreting is 77.5%, while relay interpreting is 97.43% (due to simplification already performed by the first interpreter).

5. **Summarization and generalization yield significant length compression**: The target-to-source length ratio of SUM is approximately 0.6, and GEN is approximately 0.9, while translation and paraphrase are close to 1.0.

6. **Considerable variation in annotator styles**: Annotator 4 marked span lengths that are nearly double those of other annotators.

7. **Large room for improvement in automatic baselines**: BERTAlign + sub-segmentation achieves a Relaxed F1 of ~0.61, but the Exact match score is only ~15%.

## Highlights & Insights

- **Versatile Dataset**: Applicable for linguistic analysis, interpreter training monitoring, MT hallucination detection, MQM quality assessment, and automatic simultaneous interpreting system evaluation.
- **Interpreting-derived Annotation Schema**: The 7 span labels directly map to Barik's (1994) classic classification of interpreting deviations.
- **Release of InterAlign**: Specially tailored for long-text interpreting alignment, supporting double-layer span- and word-level annotations.
- **Analysis of Relay and Multi-track Interpreting**: Highlights that relay interpreting is easier (due to pre-simplification), and multi-track interpreting exhibits an average length variation of only 2%.
- **Comprehensive Evaluation Metrics**: Defines a full suite of metrics for interpreting alignment ranging from segmentation to labelling.

## Limitations & Future Work

- Data is sourced from student interpreters; quality and strategies may differ from those of professional interpreters.
- Inter-annotator agreement is relatively low; cross-annotator variability affects data reliability.
- The dev set only covers the cs→xx direction and disproportionately represents all annotators.
- The label classifier of the simple baseline is trained solely on the dev set (80/20 split), showing a lack of sufficient training data.
- Currently Czech-centric, featuring limited language diversity.
- Lacks analysis of the impact of ASR errors (manual revisions masked ASR issues).

## Related Work & Insights

- **EUROPARI** (Macháček et al., 2021): Interpreting corpus based on the European Parliament.
- **ACL conference interpreting** (Agarwal et al., 2023): A recent resource.
- **BERTAlign** (Liu and Zhu, 2023): A sentence alignment tool, serving as the basis for the coarse alignment in this paper.
- **SimAlign** (Jalili Sabet et al., 2020): Word alignment based on contextualized embeddings.
- **Barik (1994)**: A classic framework for simultaneous interpreting deviation classification.
- **Zhao et al. (2024)**: Recent work on interpreting alignment, serving as a reference for the baselines.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first Czech-centric student interpreting dataset, accompanied by a dedicated annotation tool and a complete evaluation framework.
- **Experimental Thoroughness**: ⭐⭐⭐ — Detailed data analysis, but the automatic alignment baselines are relatively simple, with a lack of exploration of neural methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Rich data descriptions, appropriate examples, and a transparent annotation process.
- **Value**: ⭐⭐⭐⭐ — Holds direct application value for interpreting research and education; the open release of tools and data enhances its overall impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)
- [\[ACL 2025\] On Support Samples of Next Word Prediction](on_support_samples_of_next_word_prediction.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)
- [\[ACL 2025\] GPT-4 as a Homework Tutor can Improve Student Engagement and Learning Outcomes](gpt-4_as_a_homework_tutor_can_improve_student_engagement_and_learning_outcomes.md)
- [\[ACL 2025\] Subword Models Struggle with Word Learning, but Surprisal Hides It](subword_models_struggle_with_word_learning_but_surprisal_hides_it.md)

</div>

<!-- RELATED:END -->

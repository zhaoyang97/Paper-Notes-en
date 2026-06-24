---
title: >-
  [Paper Note] DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing
description: >-
  [ACL 2025][Automated Essay Scoring] This paper releases DREsS, a large-scale standardized rubric-based essay scoring dataset containing three sub-datasets (DREsS_New with student classroom data of 1.7K + DREsS_Std with standardized historical dataset of 6.5K + DREsS_CASE with augmented data of 40.1K), and proposes a corruption-based essay augmentation strategy (CASE) that improves the BERT baseline QWK score from 0.471 to 0.685 (a Gain of 45.44%).
tags:
  - "ACL 2025"
  - "Automated Essay Scoring"
  - "AES"
  - "EFL writing"
  - "rubric"
  - "data augmentation"
  - "CASE"
date: 2026-05-08
content_hash: c1709ac9f21ed6ab
---

# DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing

**Conference**: ACL 2025  
**arXiv**: [2402.16733](https://arxiv.org/abs/2402.16733)  
**Code**: Unreleased (dataset requires consent form submission)  
**Area**: Others  
**Keywords**: Automated Essay Scoring, AES, EFL writing, rubric, data augmentation, CASE  

## TL;DR
This paper releases DREsS, a large-scale standardized rubric-based essay scoring dataset containing three sub-datasets (DREsS_New with student classroom data of 1.7K + DREsS_Std with standardized historical dataset of 6.5K + DREsS_CASE with augmented data of 40.1K), and proposes a corruption-based essay augmentation strategy (CASE) that improves the BERT baseline QWK score from 0.471 to 0.685 (a Gain of 45.44%).

## Background & Motivation

**Background**: Automated Essay Scoring (AES) is a critical tool in EFL (English as a Foreign Language) writing education, providing real-time feedback and scores for students and instructors.

**Limitations of Prior Work**: (1) **Mismatch between datasets and pedagogical contexts**—widely used AES datasets such as ASAP are annotated by non-experts, do not target EFL learners, and primarily provide holistic scores instead of rubric-based analytical scores; (2) **Inconsistent rubrics**—existing small-scale rubric-based datasets (ASAP P7-8, ASAP++, ICNALE EE) define different rubrics and grade scales, hindering joint training; (3) **Data scarcity**—available rubric-based datasets are small, severely limiting model performance.

**Key Challenge**: EFL writing education requires analytical scoring based on rubrics (content, organization, language), yet there is a lack of (a) large-scale datasets annotated by domain experts, (b) standard unified rubrics, and (c) effective data augmentation methods.

**Goal**: Build a large-scale, standardized rubric-based essay scoring dataset and propose a data augmentation strategy to address data scarcity.

**Core Idea**: Integrate three paths—newly collected, standardized, and augmented datasets—to construct a unified rubric-based dataset with 48.9K samples. The CASE augmentation generates labeled synthetic samples by injecting rubric-specific corruptions into high-scoring essays.

## Method

### Overall Architecture
The DREsS dataset consists of three parts:
- **DREsS_New**: Newly collected EFL essays (1,782 essays) from real classrooms, scored by 11 English language education experts using unified rubrics.
- **DREsS_Std**: Standardized integration of 4 historical datasets (ASAP P7-8, ASAP++ P1-2, ICNALE EE), aligned to the same rubrics and score range (6,516 essays).
- **DREsS_CASE**: Synthetic samples generated using the CASE augmentation strategy (40,101 essays).

Unified rubrics: **Content**, **Organization**, and **Language**, with scores ranging from 1 to 5 (step size 0.5).

### Key Designs

1. **DREsS_New Dataset Collection**:
    - Source: EFL writing courses at a South Korean university from 2020 to 2023.
    - Students: Undergraduates with TOEFL writing scores ranging from 15 to 21.
    - Task: 40-minute timed argumentative essays (predominantly pre-test at the beginning of the semester and post-test at the end).
    - Annotators: 11 expert teachers in English education/linguistics.
    - Quality Assurance: Rater training and standardization meetings were conducted prior to annotation; ANOVA and Tukey HSD tests confirmed no significant differences between annotating raters (p<0.05).

2. **CASE Corruption-based Augmentation Strategy for Essays**:
    - **Mechanism**: Starting with high-scoring essays (4.5–5.0), project-specific corruptions are injected for the three rubrics to generate synthetic samples at different score levels.
    - Corruption Sentence Count Formula: $n(S_c) = \lfloor n(S_E) \times (5.0 - x_i) / 5.0 \rceil$, where $n(S_E)$ is the number of sentences in the original essay, and $x_i$ is the target synthetic score.
    - **Content Corruption**: Randomly replaces sentences with sentences from different prompts (off-topic)—more replacements yield poorer content quality.
    - **Organization Corruption**: Randomly swaps the positions of two sentences within the essay—more swaps yield more disorganized structure.
    - **Language Corruption**: Replaces sentences with ungrammatical sentences from the BEA-2019 GEC dataset (sentences with edit count > 10)—more replacements yield more grammatical errors.

3. **Standardization of Historical Datasets**:
    - ASAP P7 (4 rubrics $\rightarrow$ 3 rubrics): style $\times$ 0.66 + convention $\times$ 0.33 = Language
    - ASAP P8 (6 rubrics $\rightarrow$ 3 rubrics): voice + word choice + sentence fluency + convention (equally weighted average) = Language
    - ICNALE EE: vocabulary $\times$ 0.4 + language use $\times$ 0.5 + mechanics $\times$ 0.1 = Language
    - All scores are rescaled to the 1–5 range.

### Loss & Training
The baseline models (BERT fine-tuning) utilize standard regression loss, evaluated by Quadratic Weighted Kappa (QWK).

## Experimental Results

### Main Results

| Model | Training Data | Content | Organization | Language | Total QWK |
|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-3.5-turbo | N/A (zero-shot) | 0.239 | 0.371 | 0.246 | 0.307 |
| EASE (SVR) | DREsS | - | - | - | 0.360 |
| NPCR | DREsS | - | - | - | 0.507 |
| BERT | DREsS_New | 0.414 | 0.311 | 0.487 | 0.471 |
| BERT | + DREsS_Std | 0.599 | 0.593 | 0.587 | 0.551 |
| BERT | + DREsS_Std + CASE | **0.642** | **0.750** | **0.607** | **0.685** |

**Key Findings**: Overlapping three data sources significantly improves performance—training only on DREsS_New yields a QWK of 0.471, adding standardized data increases it to 0.551, and further adding CASE augmented data achieves 0.685, resulting in a total Gain of **45.44%**.

### Ablation Study

| Augmentation Parameter $n_{aug}$ | Optimal Content | Optimal Organization | Optimal Language |
|:---:|:---:|:---:|:---:|
| Optimal Value | 0.5 | 2 | 0.125 |

**Impact of CASE Augmentation Volume**: The optimal augmentation volume varies across the three rubrics—Organization requires the highest (due to internal sentence swapping which does not rely on external source size), Content is moderate, and Language is the lowest (limited by grammatical error resources—only 605 sentences).

### Comparison of Different Pre-trained Models

| Model | Content | Organization | Language | Total |
|:---:|:---:|:---:|:---:|:---:|
| BERT | 0.414 | 0.311 | 0.487 | 0.471 |
| Longformer | 0.409 | 0.312 | 0.475 | 0.463 |
| BigBird | 0.412 | 0.317 | 0.473 | 0.469 |
| GPT-NeoX | 0.410 | 0.313 | 0.446 | 0.475 |

**Key Findings**: There is no significant difference in AES performance across different pre-trained language models, consistent with the observations in Xie et al. (2022).

### ChatGPT Scoring Experiments

| Prompt Strategy | Content | Organization | Language | Total |
|:---:|:---:|:---:|:---:|:---:|
| (A) Standard zero-shot | 0.320 | 0.248 | 0.359 | 0.336 |
| (B) 2-shot | 0.330 | 0.328 | 0.306 | 0.346 |
| (C) zero-shot + Rubric Explanation | 0.357 | 0.278 | 0.342 | 0.364 |
| (D) zero-shot + Feedback Generation | 0.336 | 0.361 | 0.272 | 0.385 |

**Key Findings**: gpt-3.5-turbo performs worst on the scoring task, exhibiting high variance and inconsistent scoring, which suggests that using LLMs directly for AES remains unreliable.

## Highlights & Insights
- **Significant dataset contribution**: The first large-scale, rubric-based dataset annotated by domain experts designed for EFL writing education.
- **Ingenious CASE augmentation strategy**: Generates synthetic data using rubric-specific corruption patterns, which is intuitive and highly effective (+45.44%).
- **Practical standardization work**: Standardizes 4 historical datasets into a unified set of rubrics, facilitating direct application in subsequent research.
- **Comprehensive evaluation**: Covers a wide range of baselines including traditional ML, fine-tuned LMs, and ChatGPT.

## Limitations & Future Work
- Only focuses on English writing, without covering other L2 languages.
- DREsS_New is primarily sourced from South Korean students, which may introduce cultural or language background bias.
- The CASE strategy can only generate lower-score samples from high-scoring essays and cannot synthesize high-scoring essays (which would require LLMs).
- Defining subjective weights (e.g., style $\times$ 0.66 + convention $\times$ 0.33) during rubric alignment may introduce bias.
- Dataset access requires submitting a consent form and is not completely open.

## Related Work
- **Holistic AES**: ASAP P1-6 (10K samples, non-expert annotations), TOEFL11 (not publicly available), models like EASE/NPCR.
- **Rubric-based AES**: ASAP P7-8 (only 2.3K samples), ASAP++ (non-expert annotations), ICNALE EE (only 639 samples).
- **Positioning**: The first AES dataset meeting five criteria: (1) targeting EFL pedagogical contexts, (2) expert-annotated, (3) unified rubrics, (4) large-scale, and (5) publicly available.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The CASE augmentation is cleverly designed, and standardizing the rubrics is a major contribution.
- **Utility**: ⭐⭐⭐⭐⭐ — Directly applicable to EFL writing education systems, addressing a clear practical need.
- **Technical Depth**: ⭐⭐⭐ — A dataset-driven work; modeling methods are relatively simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive baseline comparisons across various models and strategies, with thorough ablation studies.
- **Overall Recommendation**: ⭐⭐⭐⭐ — Significant dataset contribution; the CASE strategy is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)
- [\[ACL 2025\] Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments](enhancing_marker_scoring_accuracy_through_ordinal_confidence_modelling_in_educat.md)
- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)

</div>

<!-- RELATED:END -->

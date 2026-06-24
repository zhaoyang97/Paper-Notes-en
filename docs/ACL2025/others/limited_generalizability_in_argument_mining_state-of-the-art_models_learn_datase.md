---
title: >-
  [Paper Note] Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments
description: >-
  [ACL 2025][argument mining] This paper presents the first large-scale cross-dataset generalization evaluation of four Transformer models across 17 English sentence-level argument mining datasets. The findings reveal that state-of-the-art models primarily learn dataset-specific lexical patterns rather than the structural signals of arguments, leading to generalization performance far below their in-dataset baselines. However, task-specific pre-training and multi-dataset joint…
tags:
  - "ACL 2025"
  - "argument mining"
  - "generalization"
  - "shortcut learning"
  - "cross-dataset evaluation"
  - "BERT transformers"
date: 2026-05-08
content_hash: d8dde9aeac7a1b54
---

# Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments

**Conference**: ACL 2025  
**arXiv**: [2505.22137](https://arxiv.org/abs/2505.22137)  
**Code**: [Limited-Generalizability](https://github.com/marcfeger/Limited-Generalizability)  
**Area**: Other  
**Keywords**: argument mining, generalization, shortcut learning, cross-dataset evaluation, BERT transformers

## TL;DR

This paper presents the first large-scale cross-dataset generalization evaluation of four Transformer models across 17 English sentence-level argument mining datasets. The findings reveal that state-of-the-art models primarily learn dataset-specific lexical patterns rather than the structural signals of arguments, leading to generalization performance far below their in-dataset baselines. However, task-specific pre-training and multi-dataset joint training can partially alleviate this issue.

## Background & Motivation

Argument mining is a core foundational task in automatic discourse analysis, aiming to identify argumentative structures (e.g., claims and premises) from natural language. The field faces a long-standing but rarely systematically verified concern:

**High baseline performance can be misleading**: BERT-like models perform exceptionally well on individual benchmarks (0.67–0.96 macro F1), fostering an optimistic assumption that these models possess broad applicability.

**Arguments should theoretically transfer across domains**: The core of an argument lies in its logical structure (e.g., "X should Y because Z") rather than its specific content. Consequently, trained models should theoretically generalize across datasets.

**Concerns over shortcut learning**: BERT-based models are known to focus heavily on basic grammar, nouns, and coreference relations. They might capture dataset-specific lexical cues rather than genuine argumentative signals.

**Inconsistent definitions**: Different datasets employ varying definitions of "what constitutes an argument" (e.g., claim-based, evidence-based, reasoning-based), further exacerbating the difficulty of generalization.

Rather than proposing a new model or formalization, the authors adopt a data-driven approach to answer: **Do existing state-of-the-art models learn "arguments" or merely "datasets"?**

## Method

### Overall Architecture

The study is designed around three research questions:
- **Q1**: To what extent are existing benchmark datasets comparable?
- **Q2**: Can state-of-the-art models generalize to other datasets?
- **Q3**: Do these models learn generalizable concepts of arguments?

These questions are addressed through three types of experiments: pairwise transfer experiments, joint training experiments, and controlled input manipulation experiments.

### Key Designs

1. **Dataset Selection and Standardization**

    - From 52 argument mining datasets published between 2008 and 2024, datasets were filtered based on three criteria: sentence-level annotation, binary labels (argument/non-argument), and reproducibility.
    - After two rounds of screening, 17 datasets were retained, covering approximately 345K annotated sentences.
    - A unified 60/20/20 stratified split was applied, ensuring a minimum of 850 samples per label.
    - **Design Motivation**: To ensure experimental scale and statistical reliability.

2. **Pairwise Transfer Experiments (Answering Q2)**

    - Models are trained on one dataset and tested on all 17 datasets.
    - This generates a $17 \times 17$ transfer matrix (one for each model).
    - The diagonal represents baseline (in-dataset) performance, while the off-diagonal represents generalization performance.
    - **Design Motivation**: To systematically quantify the transfer capability between each pair of datasets.

3. **Joint Training Experiments (Supplementing Q2)**

    - Models are trained jointly on 16 datasets and tested on the remaining 1 dataset (Leave-One-Out).
    - Results are compared against individual baseline performances.
    - **Design Motivation**: To test whether heterogeneous data can improve generalization.

4. **Controlled Input Manipulation Experiments (Answering Q3)**

    - Stop words, function words, discourse markers, and punctuation are systematically removed.
    - This eliminates approximately half of the words in a sentence, leaving only topical content words.
    - Model performance is compared before and after this removal.
    - **Design Motivation**: If performance does not drop after removing structural argumentative cues (such as "because" or "therefore"), it indicates that the model does not rely on these signals.

5. **Model Selection**

    - BERT, RoBERTa, DistilBERT: Standard NLP baselines.
    - WRAP: The only Transformer pre-trained via contrastive learning to enhance argumentative generalization.
    - A standard GLUE hyperparameter grid was used (batch = 32, epochs = 3, lr = 2e-5 to 5e-5).

### Loss & Training

- Standard classification cross-entropy loss.
- The optimization objective is macro F1 (ensuring equal importance for both labels).
- Each experiment was repeated 3 times, with significance analyzed using repeated-measures ANOVA and paired t-tests.

## Key Experimental Results

### Main Results (Pairwise Transfer vs. Baseline Performance, macro F1)

| Statistic | WRAP | BERT | RoBERTa | DistilBERT |
|--------|------|------|---------|------------|
| Baseline Mean | 0.79 | 0.79 | 0.79 | 0.79 |
| Transfer Mean | 0.61 | 0.58 | 0.57 | 0.56 |
| Transfer SD | 0.10 | 0.11 | 0.12 | 0.11 |
| Best Performer Share | 46% | 20% | 17% | 17% |

Of the transfer experiments, 97% yielded results below the baseline mean (0.79), with 62% falling below 0.65. WRAP consistently outperformed other models in generalization.

### Joint Training Experiments (Leave-One-Out, compared with SOTA)

| Dataset | WRAP | BERT | RoBERTa | DistilBERT | SOTA | $\Delta_{\max}$ |
|--------|------|------|---------|------------|------|-------------|
| ACQUA | 0.66 | 0.60 | 0.59 | 0.59 | 0.84 | 0.18 |
| ABSTRCT | 0.74 | 0.74 | 0.74 | 0.71 | 0.89 | 0.15 |
| CE | 0.77 | 0.72 | 0.76 | 0.72 | 0.85 | 0.08 |
| UKP | 0.70 | 0.67 | 0.70 | 0.68 | 0.79 | 0.09 |
| TACO | 0.76 | 0.61 | 0.65 | 0.55 | 0.88 | 0.12 |
| AEC | 0.52 | 0.57 | 0.51 | 0.56 | 0.96 | 0.39 |

Joint training improved the overall mean (0.63–0.66), but a substantial gap to individual baselines remained (average $\Delta_{\max} = 0.12$).

### Key Findings

1. **Generalization is the exception rather than the rule**: Only a minority of pairwise transfers achieved good generalization ($\ge 0.75$), primarily occurring between datasets of the same domain or definition type.
2. **WRAP consistently outperforms standard models**: Task-specific pre-training significantly aids generalization, with WRAP performing best in 46% of the experiments.
3. **Strong evidence of shortcut learning**:
    - BERT, RoBERTa, and DistilBERT showed almost unchanged performance ($\Delta \le 0.02$) after the removal of argumentative structure words, indicating they did not learn these signals at all.
    - WRAP exhibited the largest performance drop ($\Delta = 0.05$), indicating that it indeed captured some structural argumentative signals.
4. **The cautionary tale of AEC**: The AEC dataset, which defines arguments based on only 5 keywords, achieved the highest baseline (0.96) but showed the worst generalization ($\le 0.63$), with performance plummeting after the keywords were removed ($\Delta \le 0.45$).
5. **Definitional divergence is an inherent limitation**: Definitions of arguments (claim-based, evidence-based, reasoning-based) across different datasets overlap but are not equivalent, making cross-definition transfer fundamentally challenging.
6. **Statistical significance**: Only the advantages of WRAP and the performance degradation after manipulation passed paired t-tests ($p < 0.05$).

## Highlights & Insights

- **The conclusion of "learning datasets instead of arguments" is powerful**: Systematic evidence is provided through the $17 \times 17$ transfer matrix and controlled manipulations.
- **Rigor in experimental design**: The study serves as a methodological exemplar, employing repeated trials, ANOVA, Greenhouse-Geisser corrections, and effect size reporting.
- **A sober critique of the field**: Rather than proposing a marginally better method, the paper exposes a pervasive but previously unverified issue in the community.
- **Implications of joint training**: Although it does not completely resolve the gap, using heterogeneous data indeed helps improve generalizability.

## Limitations & Future Work

1. The study only considers BERT-family models, omitting larger language models (e.g., GPT-4, LLaMA) or prompt-based approaches.
2. Control experiments only removed function words, without exploring other fine-grained interventions (e.g., replacing structural argumentative words, or preserving structure while altering content).
3. The 17 datasets only cover English, leaving cross-lingual generalization unexplored.
4. No solution is proposed (e.g., designing better pre-training objectives to enhance the learning of argumentative signals).
5. The binary (argument vs. non-argument) granularity is relatively coarse, without targeting the generalization of argument component identification (e.g., claim vs. premise).

## Related Work & Insights

- WRAP (Feger & Dietze 2024) is the only prior work to explore pre-training for argumentative generalization, and this study confirms the validity of its direction.
- The issue of "spurious optimism" driven by benchmarks raised by Saphra et al. (2024) is concretely validated here within the domain of argument mining.
- **Insights**:
    - Do similar generalization illusions exist in other NLP subtasks (e.g., sentiment analysis, stance detection)?
    - Can we design "argument-invariant" pre-training objectives (e.g., adversarial content substitution while keeping the argumentative label unchanged)?

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Presents the first large-scale systematic evaluation of generalization in argument mining with highly precise research questions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Features 17 datasets, 4 models, three types of experiments (pairwise, joint, manipulated), and rigorous statistical testing.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized around Q1–Q3; the dataset overview is highly informative, though some statistical details are quite dense.
- **Value**: ⭐⭐⭐⭐ — Serves as an important wake-up call for the argument mining community, with implications for other NLP domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)
- [\[ACL 2025\] An Analysis of Datasets, Metrics and Models in Keyphrase Generation](an_analysis_of_datasets_metrics_and_models_in_keyphrase_generation.md)
- [\[ACL 2025\] Predicting Implicit Arguments in Procedural Video Instructions](implicit_arguments_video_instructions.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)
- [\[ICML 2026\] AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions](../../ICML2026/others/autonumerics-zero_automated_discovery_of_state-of-the-art_mathematical_functions.md)

</div>

<!-- RELATED:END -->

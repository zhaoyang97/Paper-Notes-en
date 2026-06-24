---
title: >-
  [Paper Note] Towards Benchmarking Foundation Models for Tabular Data With Text
description: >-
  [ICML 2025][Self-Supervised Learning][tabular data] The first systematic study on modeling tabular data containing text features: qualitative counterexamples are designed to expose the failure modes of three types of text embeddings, 13 real-world datasets are manually curated, and text features are found to improve predictive accuracy on 11/13 datasets, although no single optimal embedding method exists, indicating that tabular data with text remains an unsolved problem.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "tabular data"
  - "text features"
  - "foundation model"
  - "benchmark"
  - "TabPFNv2"
  - "embedding"
date: 2026-05-08
content_hash: 7378b3a4e9c5ff6a
---

# Towards Benchmarking Foundation Models for Tabular Data With Text

**Conference**: ICML 2025  
**arXiv**: [2507.07829](https://arxiv.org/abs/2507.07829)  
**Code**: TextTabBench repository (open source)  
**Area**: Self-Supervised Learning  
**Keywords**: tabular data, text features, foundation model, benchmark, TabPFNv2, embedding

## TL;DR

The first systematic study on modeling tabular data containing text features: qualitative counterexamples are designed to expose the failure modes of three types of text embeddings, 13 real-world datasets are manually curated, and text features are found to improve predictive accuracy on 11/13 datasets, although no single optimal embedding method exists, indicating that tabular data with text remains an unsolved problem.

## Background & Motivation

**Background**: Tabular foundation models (e.g., TabPFNv2) are rapidly advancing, and the natural next step is supporting multimodal data—co-existence of structured columns and free-text fields. However, existing tabular benchmarks barely contain any text columns.

**Limitations of Prior Work**: Real-world datasets containing semantically rich text features are extremely difficult to find—even exhaustive searches on OpenML and Kaggle yield only a few. Existing methods diverge significantly in text processing: AutoGluon uses TF-IDF sparse vectors, CARTE utilizes fastText sentence vectors, and the method for the TabPFNv2 API remains undisclosed. Among the 51 datasets in the CARTE benchmark, the authors find that at most 11 are suitable for evaluating tabular-and-text scenarios.

**Key Challenge**: A fundamental question remains unanswered: which embedding strategy is best suited for tabular tasks, and under what conditions? A fair benchmark is lacking to address this.

**Goal**: (1) Expose specific failure modes of existing embedding methods; (2) Curate a high-quality "tabular + text" benchmark; (3) Systematically compare embedding strategies on SOTA models.

**Key Insight**: Approach from both qualitative and quantitative levels: first, construct precise failure conditions for each embedding using synthetic counterexamples, then quantitatively evaluate them on real-world data.

**Core Idea**: Reveal that the text-processing capabilities of tabular foundation models are still highly deficient, providing the community with diagnostic tools (counterexamples) and evaluation infrastructure (benchmarks).

## Method

### Overall Architecture

This work does not propose a new model but advances understanding through three complementary contributions: (1) Qualitative investigation—constructing synthetic experiments to expose embedding failures; (2) Benchmark curation—filtering real-world datasets according to five rules; (3) Quantitative experiments—systematically evaluating on the benchmark.

### Key Designs

1. **Qualitative Counterexamples**:

    - Function: Precisely diagnose the respective failure modes of TF-IDF, fastText, and BERT.
    - Mechanism: Select 5 OpenML binary classification datasets, constructing two baselines ("No Text" using raw features; "Complete Leak" leaking labels $\rightarrow$ 100% accuracy). Three stress tests:
        - **N-Gram Break**: Replaces leaked labels with synonyms (e.g., training on "good" $\rightarrow$ testing on "great"). TF-IDF fails due to Out-of-Vocabulary (OOV) terms, while fastText and BERT maintain 100%.
        - **Simple NLP Break**: Pads leaked labels with random words ("apple mountain positive girl"). fastText word vectors degrade as they are overwhelmed by noise, whereas TF-IDF and BERT remain stable.
        - **LLM Break**: Pads labels with semantically conflicting words ("favourable positive sad charming"). BERT and fastText are confused by the ambiguity, while TF-IDF remains robust due to frequency-driven mechanics.
    - Design Motivation: Synonym changes, random noise, and semantic ambiguity are highly common in real-world long texts; each embedding systematically fails in at least one of these patterns.

2. **Benchmark Dataset Curation Rules**:

    - Function: Ensure the benchmark meaningfully evaluates "text processing in tabular data".
    - Five Rules: (i) Real free-form text (not short codes); (ii) Dual-signal requirement (both text and structured features contain predictive information); (iii) Tabular prediction tasks (excluding recommendation/retrieval); (iv) Accessibility (no special permissions required); (v) Diversity in domains and targets.
    - Final 13 Datasets: Cover binary classification (fraud/kick/osha), multi-class classification (cards/complaints/spotify), and regression (airbnb/beer/houses/laptops/mercari/permits/wine), with row sizes from 984 to 100K.

3. **Embedding Strategies and Evaluation Pipeline**:

    - Function: Fairly compare the effects of different embeddings on SOTA models.
    - Three Embeddings: (1) fastText sentence vectors; (2) Skrub TableVectorizer (GapEncoder); (3) AutoGluon TextNgramFeatureGenerator (TF-IDF pipeline).
    - Models: TabPFNv2 (local), XGBoost (local), TabPFNv2 API, AutoGluon Tabular Predictor.
    - Due to TabPFNv2 memory constraints, the number of features is limited to 300, testing dimensional reduction like SHAP, PCA, Lasso, and t-test.

### Loss & Training

This work is a benchmark study and does not involve new loss functions.

## Key Experimental Results

### Main Results

**Text vs No-Text Comparison (Table 2, SHAP Dimensionality Reduction, Best Embedding for Each Model)**

| Dataset | Task | TabPFNv2 with Text | TabPFNv2 no Text | XGBoost with Text | XGBoost no Text |
|--------|------|---------------|---------------|--------------|--------------|
| beer | Regression | 0.646±0.023 | 0.579±0.020 | 0.594±0.036 | 0.468±0.020 |
| mercari | Regression | 0.237±0.050 | 0.001±0.016 | 0.110±0.062 | 0.001±0.006 |
| spotify | Multi-class | 0.815±0.010 | 0.663±0.016 | 0.807±0.012 | 0.636±0.027 |
| frauds | Binary | 0.962±0.008 | 0.852±0.006 | 0.958±0.004 | 0.849±0.015 |
| kick | Binary | 0.779±0.016 | 0.702±0.010 | 0.769±0.014 | 0.657±0.013 |

**Winning Statistics for Each Embedding Method**

| Embedding Method | Number of Best Datasets across All Models / 13 |
|---------|----------------------|
| fastText | 7 |
| AutoGluon Pipeline | 5 |
| Skrub | 1 |

### Key Findings

1. Text features improve prediction accuracy on **11/13** datasets, boosting performance on *mercari* from nearly 0 to 0.237 (where text is almost the only signal source).
2. **There is no single optimal embedding method**: fastText wins most frequently (7/13) but does not completely dominate.
3. **There is no single optimal dimensionality reduction method**: SHAP is most frequently optimal but does not always win.
4. Local models with custom-selected embeddings sometimes outperform the API, suggesting there is still significant room for optimizing embedding strategies.
5. The TabPFNv2 API shows better consistency when text is present, but its gain margin is not as large as the best local embedding.

## Highlights & Insights

- **High diagnostic value of synthetic counterexamples**: The three stress tests (N-Gram/NLP/LLM Break) precisely locate the blind spots of each embedding, allowing researchers to choose embeddings based on their textual characteristics.
- **Systematic audit of the CARTE benchmark**: The audit reveals that a large portion of the 51 datasets does not meet the "tabular + text" evaluation requirements (e.g., they are not prediction tasks, are biased towards short classification text, showcase preprocessing bias towards CARTE, or contain duplicated sources). This audit itself is valuable to the community.
- **Constructive significance of the "no-winner" conclusion**: It clearly points out that tabular-and-text is currently an unsolved problem and provides concrete evaluation infrastructure for new methods.
- **Call to dataset creators**: Recommends publishing raw pre-aggregated data to preserve textual variation information.

## Limitations & Future Work

- The number of curated datasets is limited (13), and the domain coverage can be extended.
- State-of-the-art instruction-tuned embedding models (such as E5-Mistral, GTE) and LLM-based embeddings have not been tested.
- Systematic comparisons with row-as-text methods (e.g., TabLLM) in large-sample scenarios have not been conducted.
- Joint modeling strategies for multiple text columns are not discussed.
- The hard constraint of reducing dimensionality to 300 features may restrict the representational capacity of certain embeddings.

## Related Work & Insights

- CARTE (Kim et al., 2024): Recent tabular + text benchmark, though this work's audit reveals its many limitations.
- TabPFNv2 (Hollmann et al., 2025): Tabular foundation model used for the core evaluations.
- Grinsztajn et al. (2023): Compares 30+ embedding methods but does not focus on foundation models.
- Insight: The ideal tabular foundation model should natively handle text columns during the pre-training phase, rather than relying on post-processed embedding concatenation.

## Rating

⭐⭐⭐⭐ — Fills an important gap in tabular + text benchmarks. Qualitative counterexamples are precise and powerful, and benchmark curation rules are clear and reproducible. The limitation is the lack of testing with more advanced embedding methods and ultra-large-scale models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](../../NeurIPS2025/self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[ICML 2025\] Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions](beyond_sensor_data_foundation_models_of_behavioral_data_from_wearables_improve_h.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[ICML 2025\] Test-Time Canonicalization by Foundation Models for Robust Perception](test-time_canonicalization_by_foundation_models_for_robust_perception.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)

</div>

<!-- RELATED:END -->

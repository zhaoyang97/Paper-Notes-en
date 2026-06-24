---
title: >-
  [Paper Note] An Analysis of Datasets, Metrics and Models in Keyphrase Generation
description: >-
  [ACL 2025 (GEM²)][Keyphrase Generation] Conducts a systematic analysis of 50+ papers in the keyphrase generation field, revealing critical issues such as high similarity among benchmark datasets and inconsistent evaluation metric computations leading to overestimated performance, and releases a strong PLM-based model to facilitate future research.
tags:
  - "ACL 2025 (GEM²)"
  - "Keyphrase Generation"
  - "Dataset Analysis"
  - "Evaluation Metrics"
  - "Pre-trained Language Models"
  - "Survey and Analysis"
date: 2026-05-08
content_hash: 0bd39fae105c5d4a
---

# An Analysis of Datasets, Metrics and Models in Keyphrase Generation

**Conference**: ACL 2025 (GEM²)  
**arXiv**: [2506.10346](https://arxiv.org/abs/2506.10346)  
**Code**: None (but pre-trained models are released)  
**Area**: Others (NLP / Information Retrieval)  
**Keywords**: Keyphrase Generation, Dataset Analysis, Evaluation Metrics, Pre-trained Language Models, Survey and Analysis

## TL;DR

Conducts a systematic analysis of 50+ papers in the keyphrase generation field, revealing critical issues such as high similarity among benchmark datasets and inconsistent evaluation metric computations leading to overestimated performance, and releases a strong PLM-based model to facilitate future research.

## Background & Motivation

Keyphrase Generation (KG) refers to automatically generating a set of words or phrases that summarize the content of a document. In recent years, this field has continuously developed across multiple directions, including model architectures, data resources, and application scenarios. However, the field faces the following core problems:

**Lack of Systematic Review**: Past KG research has been scattered across multiple technical routes (e.g., Seq2Seq architectures, pre-trained models, generation strategies), but no study has systematically organized and compared these works, leaving the state of the field unclear.

**High Similarity Among Benchmark Datasets**: Frequently used benchmarks in the KG field (such as KP20k, Inspec, Krapivin, NUS, SemEval, etc.) exhibit worrying levels of similarity. Most of these datasets originate from academic papers, with heavily overlapping domain distributions and text styles, which may cause models to repeat learning rather than truly generalize.

**Inconsistent Metric Calculations**: Significant inconsistencies exist across different papers when calculating metrics like $F_1@K$ and $F_1@M$—differing in stemming treatments, truncation strategies, and split standards for present/absent keyphrases. These discrepancies make cross-paper performance comparisons unreliable and lead to overestimated performance in some methods.

**Poor Accessibility of Pre-trained Models**: Despite the excellent performance of PLM-based methods in recent years, most papers do not open-source their pre-trained models, hindering subsequent research progress and fair comparison.

The **Key Insight** of this paper is: as an analysis paper, it systematically examines the issues across the three dimensions of datasets, metrics, and models in the KG field, and fills the reproducibility gap by releasing a strong baseline model.

## Method

### Overall Architecture

This study is not a paper proposing a new method, but rather a systematic analysis paper. Its analysis framework includes:
- **Input**: 50+ papers related to keyphrase generation
- **Analysis Dimensions**: Dataset characteristics, evaluation metric consistency, model architecture evolution
- **Output**: Problem diagnosis + improvement recommendations + strong baseline PLM model

### Key Designs

1. **Dataset Similarity Analysis**:

    - **Function**: Quantitatively analyzes the degree of similarity among commonly used KG benchmark datasets.
    - **Mechanism**: Quantifies high redundancy among datasets by calculating statistical characteristics such as text overlap rates, domain distributions, document length distributions, and keyphrase count distributions across different datasets. For example, datasets like KP20k, Krapivin, and SemEval are predominantly composed of computer science academic papers, with highly consistent text styles and topic distributions.
    - **Design Motivation**: If benchmarks are highly similar, performing well on multiple benchmarks does not prove a model's generalization capabilities, which threatens the credibility of the evaluation system in this field.

2. **Evaluation Metric Consistency Analysis**:

    - **Function**: Outlines the detailed differences in calculating metrics like $F_1@K$ and $F_1@M$ across various papers.
    - **Mechanism**: Compares and analyzes implementation differences in the following aspects:
        - **Stemming**: Porter Stemmer vs. no stemming, which significantly impacts matching results.
        - **Present/Absent Subdivision**: Unstandardized determination criteria for present keyphrases (appearing in the document) and absent keyphrases (not appearing in the document).
        - **Truncation Strategy**: Taking the top 5 predictions for $F_1@5$ vs. taking predictions equal to the number of ground truth labels for $F_1@M$.
        - **Deduplication Method**: Varying deduplication strategies for generated results.
    - **Design Motivation**: Tiny differences in metric calculation can lead to massive differences in performance numbers, rendering cross-paper comparison meaningless. The advantage of some methods might stem from the "advantage" of metric calculation rather than the model itself.

3. **PLM-based Strong Baseline Model**:

    - **Function**: Trains and releases a keyphrase generation model based on pre-trained language models as a reproducible strong baseline.
    - **Mechanism**: Fine-tunes pre-trained Seq2Seq models like T5/BART on standard KG datasets using unified data preprocessing and evaluation pipelines.
    - **Design Motivation**: Resolves the poor accessibility of pre-trained models in the field and provides a fair comparison baseline for future research.

### Analytical Methodology

The paper systematically analyzes 50+ KG papers, classifying them along the following dimensions:
- Generation Paradigm: One2One (generating one keyphrase at a time) vs. One2Seq (generating all keyphrases as a sequence at once)
- Model Architecture: Evolution from RNN-based $\rightarrow$ Transformer-based $\rightarrow$ PLM-based
- Evaluation Settings: Datasets, metrics, and preprocessing methods used in different papers

## Key Experimental Results

### Main Results (Dataset Similarity Analysis)

| Dataset Pair | Similarity Metrics | Findings |
|----------|-----------|------|
| KP20k ↔ Krapivin | High | Similar sources (academic papers), large domain overlap |
| KP20k ↔ Inspec | Medium-High | Despite different document lengths, the topic distributions are similar |
| KP20k ↔ SemEval | High | Both are CS academic papers |
| KP20k ↔ NUS | Medium | NUS has a slightly broader domain but is still dominated by CS |

### Ablation Study (Impact of Metric Inconsistencies)

| Evaluation Setting | F1@5 Variation | Explanation |
|----------|---------|------|
| With Stemming vs. Without Stemming | Significant Difference | Stemming usually improves matching rate by 5-10% |
| Different present/absent split standards | Clear Difference | Strict vs. lenient determination affects absent keyphrase evaluation |
| Unified Evaluation Pipeline | Baseline Value | Standardized evaluation recommended by this paper |
| Literature Reported Values | Generally inflated | Some papers obtain artificially high scores due to differences in metric implementations |

### Key Findings

- There is a worrying high similarity among commonly used benchmark datasets, with a single domain (CS academic papers) dominating the entire evaluation system.
- Inconsistency in evaluation metric calculation is a long-neglected systematic issue, making performance comparisons in the field unreliable.
- PLM-based methods are currently the strongest paradigm, but model accessibility remains a bottleneck.
- The improvements brought by the architectural evolution from RNN to Transformer to PLM might partially stem from the "illusion" of inconsistent evaluations.

## Highlights & Insights

- **Dataset redundancy analysis** is the first systematic examination in the KG field, which also serves as a reference for other NLP subfields—many domain benchmarks may suffer from similar redundancy problems.
- The call for **evaluation standardization** is highly timely: the paper identifies specific calculation details that require unification, which is more actionable than a general call for standardization.
- Releasing a reproducible PLM baseline model lowers the barrier of entry for this field.
- As a GEM²-style analysis paper, its methodology can be transferred to the systematic analysis of other NLP tasks.

## Limitations & Future Work

- The analysis primarily focuses on English KG research, with insufficient coverage of multilingual scenarios.
- Although it reveals the issues, it does not propose specific new datasets to address the domain-singularity problem.
- Details and performance data of the released PLM baseline model require reading the full paper for a comprehensive evaluation.
- Does not provide an in-depth analysis of LLMs (e.g., GPT-4, LLaMA) on zero-shot/few-shot performance in KG tasks.

## Related Work & Insights

- **vs CatSeq (Yuan et al., 2020)**: CatSeq is a representative work of the One2Seq paradigm; this paper reveals that its performance variance under different evaluation settings is much larger than the performance gaps among different methods.
- **vs KG Surveys**: Unlike traditional field surveys that merely organize methods, this paper acts more like a "health check report" for the field, deeply analyzing issues within the evaluation system itself.
- **vs NLG Evaluation Analysis**: Similar to the reflection on the BLEU metric in the machine translation field, this paper's reflection on KG evaluation is expected to promote the improvement of evaluation standards in the field.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic analysis of dataset/metric issues in the KG field, offering a unique perspective
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic analysis of 50+ papers; dataset similarity analysis is robust
- Writing Quality: ⭐⭐⭐⭐⭐ Analysis papers require clear logic, and this paper is exceptionally well-structured
- Value: ⭐⭐⭐⭐ Reflections on evaluation standards in the KG field play an important role in driving progress, and releasing baseline models has practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ERU-KG: Efficient Reference-aligned Unsupervised Keyphrase Generation](eru-kg_efficient_reference-aligned_unsupervised_keyphrase_generation.md)
- [\[ACL 2025\] Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments](limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)
- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[AAAI 2026\] HybriDLA: Hybrid Generation for Document Layout Analysis](../../AAAI2026/others/hybridla_hybrid_generation_for_document_layout_analysis.md)
- [\[ACL 2025\] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)

</div>

<!-- RELATED:END -->

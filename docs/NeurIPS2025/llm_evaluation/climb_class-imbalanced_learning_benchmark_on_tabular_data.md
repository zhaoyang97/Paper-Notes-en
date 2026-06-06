---
title: >-
  [Paper Note] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data
description: >-
  [NeurIPS 2025][LLM Evaluation][class imbalance] This paper presents CLIMB — the most comprehensive benchmark to date for class-imbalanced learning on tabular data — encompassing 73 real-world datasets and 29 CIL algorith…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "class imbalance"
  - "tabular data"
  - "benchmark"
  - "ensemble learning"
  - "resampling"
date: 2026-05-08
content_hash: ea877c634185ced8
---

# CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data

**Conference**: NeurIPS 2025
**arXiv**: [2505.17451](https://arxiv.org/abs/2505.17451)  
**Code**: [ZhiningLiu1998/imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble)  
**Area**: LLM Evaluation
**Keywords**: class imbalance, tabular data, benchmark, ensemble learning, resampling

## TL;DR

This paper presents CLIMB — the most comprehensive benchmark to date for class-imbalanced learning on tabular data — encompassing 73 real-world datasets and 29 CIL algorithms. Large-scale experiments reveal several practical insights: naive rebalancing is often ineffective, ensemble methods are critical, and data quality impacts performance more than the degree of imbalance itself.

## Background & Motivation

- **Class imbalance is a core challenge in tabular data**: In high-stakes domains such as financial fraud detection, network intrusion identification, and medical diagnosis, minority classes represent rare yet important outcomes, and standard classifiers degrade severely in these settings.
- **Existing benchmarks are highly fragmented**: Prior studies are mostly confined to specific domains (business/finance/medicine/education), cover datasets with similar imbalance ratios, include narrow algorithm selections, and lack systematic cross-paradigm comparisons (undersampling/oversampling/cost-sensitive/ensemble).
- **Tabular data presents unique challenges**: Unlike images or text, tabular data features heterogeneous attributes, small sample sizes, and the absence of local correlation structures, making tree-based models the dominant choice; class imbalance further exacerbates generalization difficulties for minority classes.
- **Practitioners lack reliable method selection guidance**: When faced with dozens of CIL methods, practitioners have no principled basis for selection, and different evaluation metrics can lead to contradictory conclusions.
- **Efficiency and robustness remain understudied**: Prior work focuses on accuracy, with little systematic analysis of runtime, noise, or missing value effects on performance.
- **High-quality open-source tooling is absent**: No unified-API, well-documented, rigorously tested open-source CIL benchmark library previously existed.

## Method

### Overall Architecture: The CLIMB Benchmark

CLIMB constructs a comprehensive benchmark platform integrating datasets, algorithms, and evaluation protocols, accompanied by a high-quality Python library (unified scikit-learn-style API, 95% test coverage, detailed documentation) that supports fair, reproducible, and extensible evaluation of CIL methods.

### Key Design 1: 73 Real-World Imbalanced Tabular Datasets

- **Function**: Carefully selects 73 naturally class-imbalanced datasets from OpenML, spanning multiple domains with imbalance ratios ranging from 2.1 to 577.9.
- **Mechanism**: Seven strict filtering criteria are applied — data must be real and naturally imbalanced, sufficiently challenging (excluding datasets with AUPRC > 0.95), IR > 2, no missing values, satisfying the i.i.d. assumption, non-deterministic functional relationships, and properly documented.
- **Design Motivation**: Excluding artificially constructed and trivially easy datasets ensures the benchmark genuinely reflects challenges encountered in real-world applications. Datasets are stratified into four tiers (low/medium/high/extreme IR) to facilitate granular analysis.

### Key Design 2: Unified Implementation of 29 CIL Algorithms

- **Function**: Implements 29 representative algorithms spanning six paradigms — undersampling (RUS/CC/IHT/NearMiss), cleaning (Tomek Links/ENN/RENN/AllKNN/OSS/NCR), oversampling (ROS/SMOTE/BorderlineSMOTE/SVMSMOTE/ADASYN), undersampling ensembles (SPE/BC/BRF/EE/RUSBoost/UnderBagging), oversampling ensembles (OverBoost/SMOTEBoost/OverBagging/SMOTEBagging), and cost-sensitive ensembles (CS/AdaCost/AdaUBoost/AsymBoost).
- **Mechanism**: A unified API design with hierarchical modular abstractions supports convenient extensibility via inheritance and polymorphism.
- **Design Motivation**: Eliminates unfair comparisons arising from disparate implementations and provides the first unified comparison spanning all mainstream CIL paradigms.

### Key Design 3: Rigorous Evaluation Protocol

- **Function**: Standardized preprocessing (numerical feature normalization, categorical feature encoding), 5-fold stratified splitting, Optuna hyperparameter search (100 trials per algorithm–dataset pair), and three evaluation metrics (AUPRC/macro-F1/BAC).
- **Mechanism**: A decision tree serves as the unified base classifier and ensemble size is fixed at 100, ensuring fair comparisons across methods.
- **Design Motivation**: Eliminates the stochasticity of single random splits; multi-metric evaluation reveals different performance aspects (AUPRC emphasizes precision, BAC emphasizes recall balance), preventing misleading conclusions from any single metric.

### Key Design 4: Robustness Control Experiments

- **Function**: Introduces label noise (flipping 10%/20%/30% of minority class labels), missing values (mean imputation at 10%/20%/30%), and additional imbalance (further removing minority samples to double/triple/quintuple the IR).
- **Mechanism**: Each perturbation factor is introduced individually to isolate its effect on CIL performance.
- **Design Motivation**: Real-world data frequently co-occurs with noise and missing values; understanding the relative impact of these factors compared to class imbalance itself is essential for practitioners.

## Loss & Training

A decision tree is used as the base classifier; ensemble methods are uniformly set to 100 base learners; cost-sensitive methods assign misclassification costs inversely proportional to class frequencies. Bayesian hyperparameter search is conducted via Optuna with 100 trials per configuration. The full benchmark involves approximately 800,000 hyperparameter search trials and the training of over 10 million base models.

## Key Experimental Results

### Table 2: Main Benchmark Results (AUPRC/F1/BAC Grouped by Imbalance Level)

| Imbalance Group | Base AUPRC | Best Method | Best AUPRC | Gain |
|---|---|---|---|---|
| Low (IR < 5, 28 datasets) | 51.0 | SPE | 59.3 | +8.3 |
| Medium (IR ∈ [5, 10), 24 datasets) | 50.9 | SPE | 64.6 | +13.7 |
| High (IR ∈ [10, 50), 15 datasets) | 34.9 | SPE | 47.1 | +12.2 |
| Extreme (IR > 50, 6 datasets) | 42.6 | SPE | 57.5 | +14.9 |

Key finding: **Self-paced Ensemble (SPE) ranks first or second in AUPRC across all imbalance levels**; naive undersampling methods (RUS/CC/NearMiss) degrade performance in most settings; ensemble methods as a whole substantially outperform non-ensemble methods.

### Key Findings from Robustness Analysis

| Perturbation Factor | Equivalent Impact |
|---|---|
| 10% label noise | ≈ performance drop caused by a 500% increase in IR |
| 30% missing features | ≈ performance drop caused by a 500% increase in IR |

This finding indicates that **data quality may matter more than the degree of class imbalance itself**.

## Highlights & Insights

- **Broadest coverage to date**: 73 datasets × 29 algorithms × 6 paradigms constitutes the largest systematic evaluation in this field.
- **Five concise practical insights**: (1) naive rebalancing is frequently harmful; (2) ensembles are the key to effective CIL; (3) metric choice affects conclusions; (4) undersampling ensembles achieve the best performance–efficiency trade-off; (5) data quality may matter more than imbalance.
- **High-quality open-source library**: Unified API, 95% test coverage, and detailed documentation provide genuine engineering value.
- **Quantitative comparison of data quality vs. imbalance severity** is a novel and practically significant contribution.

## Limitations & Future Work

- Only decision trees are used as base classifiers; the performance of deep learning models (e.g., TabNet, FT-Transformer) under CIL settings remains unexamined.
- Datasets containing missing values are excluded, limiting coverage of real-world "dirty data" scenarios.
- Robustness experiments introduce perturbation factors independently, leaving the compound effects of simultaneous noise, missingness, and imbalance unstudied.
- Only binary and standard multi-class classification are considered; more complex settings such as multi-label imbalance and open-set recognition are not addressed.
- All datasets are sourced from OpenML, which may introduce selection bias.

## Related Work & Insights

Compared to prior empirical studies such as Zhu et al. (2018), Xiao et al. (2021), Khushi et al. (2021), and Kim & Hwang (2022), CLIMB comprehensively advances the state of the art along three dimensions: (1) the number of algorithms is expanded from 4–21 to 29 with full paradigm coverage (the first benchmark to include cost-sensitive methods); (2) the number of datasets is expanded from 2–31 to 73 across diverse domains and IR levels; (3) CLIMB is the first to provide an accompanying high-quality open-source toolkit. Compared to tabular data benchmarks such as TableShift, CLIMB focuses specifically on class imbalance as an orthogonal challenge.

## Rating

- Novelty: ⭐⭐⭐ — No new algorithms are proposed, but the benchmark construction is comprehensive and the insights are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 800,000 hyperparameter search trials, 10 million base models, and multi-dimensional analysis constitute an exceptionally thorough evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, a well-designed RQ-driven analytical framework, and information-dense tables.
- Value: ⭐⭐⭐⭐ — Directly useful for both CIL researchers and practitioners; the open-source library offers sustained impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](../../ICLR2026/llm_evaluation/tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/llm_evaluation/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[NeurIPS 2025\] LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought](lcdb_11_a_database_illustrating_learning_curves_are_more_ill-behaved_than_previo.md)
- [\[ICLR 2026\] Human-LLM Collaborative Feature Engineering for Tabular Learning](../../ICLR2026/llm_evaluation/human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)

</div>

<!-- RELATED:END -->

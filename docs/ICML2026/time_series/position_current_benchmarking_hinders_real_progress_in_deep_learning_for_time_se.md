---
title: >-
  [Paper Note] Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series
description: >-
  [ICML 2026][Time Series][Time series forecasting] This position paper systematically reveals the core issue of current time series forecasting benchmarks—**discrepancies in design choices** (global/local parameters, preprocessing, exogenous variables, temporal and spatial processing) are often overlooked as "implementation details," leading to unfair comparisons between papers. Through controlled experiments across 44 datasets, 7 SOTAs, and multiple reference architectures…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time series forecasting"
  - "benchmarking"
  - "design dimensions"
  - "model comparison"
  - "evaluation methodology"
date: 2026-05-08
content_hash: abaaa83e87cc02de
---

# Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series

**Conference**: ICML 2026  
**arXiv**: [2512.22702](https://arxiv.org/abs/2512.22702)  
**Code**: TBD  
**Area**: Time Series / Benchmarking Methodology  
**Keywords**: Time series forecasting, benchmarking, design dimensions, model comparison, evaluation methodology

## TL;DR
This position paper systematically reveals the core issue of current time series forecasting benchmarks—**discrepancies in design choices** (global/local parameters, preprocessing, exogenous variables, temporal and spatial processing) are often overlooked as "implementation details," leading to unfair comparisons between papers. Through controlled experiments across 44 datasets, 7 SOTAs, and multiple reference architectures, it demonstrates that the impact of these differences (5-15%) often **exceeds the contribution of specific sequence modeling layers** (1-3%).

## Background & Motivation

**Background**: Time series forecasting is popular in deep learning, with a large number of new architectures emerging annually (PatchTST, iTransformer, TimeMixer, Crossformer). However, the results of these new methods are inconsistent across published benchmarks—the same model on the same dataset often shows significantly different reported results in different papers.

**Limitations of Prior Work**: New SOTA results are quickly overturned, indicating a lack of understanding regarding the fundamental sources of performance improvement. The community is caught in a cycle where new evidence constantly contradicts previous results, yet no one can clearly explain the root of these contradictions.

**Key Challenge**: Paper comparisons typically report only final performance metrics while ignoring critical design choices hidden in implementation details: (1) model configuration (global / local / hybrid), (2) preprocessing + exogenous variables, (3) temporal dependency modeling, and (4) spatial dependency processing. When comparing two complex architectures, it is impossible to determine which part of the performance gain stems from a novel attention mechanism versus other design dimensions.

**Goal**: To quantify the impact of four key design dimensions on model performance through rigorous experimental design, proving that current benchmarking practices are systematically failing.

**Key Insight**: Using "**reference architectures**" (minimal, component-clear baselines such as pure MLP / TCN / Transformer / RNN) for controlled comparisons, the true impact of each dimension is evaluated directly by systematically introducing or removing specific design features.

**Core Idea**: Benchmarking practices require a complete overhaul—not only must hyperparameters be consistent, but key design choices must also be explicitly declared in a **model card** to ensure future comparisons are "apples-to-apples."

## Method

### Overall Architecture
**Diagnostic Phase**: Select 7 SOTA models (PatchTST / DLinear / TimeMixer / iTransformer / Crossformer / ModernTCN / Linear baseline) for controlled experiments on 44 real-world datasets; simultaneously implement a set of "reference architectures" (MLP / TCN / RNN / Transformer / Pyramidal Attention, etc.). Isolate and measure the impact by systematically enabling/disabling specific design features on reference architectures.

**Analysis Phase**: Analyze the roots of performance differences along four design dimensions; quantify the magnitude of impact (MSE / MAE changes) for each dimension through comparison tables.

**Output**: A "Forecasting Model Card" template that defines essential fields that must be explicitly declared for every newly proposed forecasting architecture.

### Key Designs

**1. Design Dimensions D1 + D2 (Model Configuration + Preprocessing / Exogenous Variables): Bringing parameter sharing and data preprocessing choices to the forefront.**

Many papers report only final numbers while glossing over choices such as "how parameters are shared across multiple series" or "whether exogenous variables/calendar features/normalization are used." Consequently, the "same model" in two papers might actually be testing different things. D1 distinguishes between three extremes—Global models (all parameters shared), Local models (trained separately for each series), and Hybrid models (most shared but some local parameters). D2 covers exogenous variables, calendar features, and normalization methods. The authors eliminate ambiguity by recording whether each model uses series-specific parameters (learnable series embeddings, local parameters in normalization layers). The results are striking: Table 1 shows that adding local parameters to the same model can change the MSE by 7%-30% (Transformer on Electricity from 0.151 $\to$ 0.136), and Table 2 shows that adding/removing calendar features alone changes the Transformer MSE on Traffic from 0.479 $\to$ 0.417 (a 13% reduction)—the impact of these "implementation details" rivals any claimed architectural innovation.

**2. Design Dimensions D3 + D4 (Temporal Processing + Spatial Processing): Putting the perceived "model core" into controlled comparisons.**

D3 (single-sequence modeling on the time axis: MLP / TCN / RNN / Transformer / Pyramidal Attention) is the core of traditional "model selection," while D4 (channel independence vs. cross-series attention) handles spatial dependencies. The paper uses reference architecture ablations to quantify the individual contributions of these two dimensions. Surprisingly, Table 3 reveals that when other configurations are aligned, the differences between various temporal processing operators are heavily exaggerated—MLP and Transformer are nearly indistinguishable on Electricity / Weather datasets. In Table 4, the difference with or without spatial attention is also < 5%. This implies that a large portion of the "gains brought by new operators" claimed in many papers may actually stem from hidden D1-D4 differences rather than the operators themselves.

**3. Forecasting Model Card: Mandating declaration of key design choices at the source to eliminate hidden discrepancies.**

After diagnosing the problem, merely calling for "consistent hyperparameters" is insufficient because the design choices themselves remain hidden. The authors propose a model card template requiring every newly proposed forecasting architecture to explicitly declare its values for the four D1-D4 dimensions, alongside metadata fields for hyperparameter tuning, data preprocessing, random seeds, etc. This ensures that future comparisons are "apples-to-apples," transforming reproducibility and comparability from slogans into a mandatory form for submissions—a concept that can be extended directly to NLP and CV benchmarking.

## Key Experimental Results

### Main Results (Quantification of the Four Design Dimensions' Impact)

| Dimension | Model | Configuration A | Configuration B | Impact (%) |
|-----------|-------|-----------------|-----------------|------------|
| D1 | Transformer | Global Parameters | Local Parameters | -10.0 (Electricity) |
| D1 | iTransformer | Hybrid | Global | +9.6 |
| D2 | Transformer | W/ Exogenous | W/O Exogenous | +13.6 (Traffic) |
| D2 | PatchTST | W/O Exogenous | W/ Exogenous | -4.7 (Electricity) |
| D3/D4 | MLP | — | — | 0.129 $\pm$ 0.000 (Electricity) |
| D3/D4 | Transformer | — | — | 0.129 $\pm$ 0.001 |
| D3/D4 | TCN | — | — | 0.130 $\pm$ 0.000 |

**Key Findings**: The impact of D1 / D2 (5-15%) is significantly greater than that of D3 (usually < 5%)—directly refuting the implicit assumptions of current benchmarking.

### Ablation Study

| Configuration | Electricity | Weather | Traffic | Solar |
|---------------|-------------|---------|---------|-------|
| Global + No Exog + MLP | 0.129 | 0.148 | 0.376 | 0.194 |
| Global + With Exog + MLP | 0.127 | 0.146 | 0.342 | 0.191 |
| Hybrid + With Exog + Transformer | 0.136 | 0.153 | 0.362 | 0.196 |
| Hybrid + With Exog + TCN | 0.130 | 0.148 | 0.364 | 0.193 |

Simple reference models (e.g., MLP) achieve performance comparable to most SOTA models once design choices are aligned.

### Key Findings
- **Design dimensions are more important than architectural innovations**: Under the premise of keeping other conditions consistent, changing preprocessing or parameter sharing strategies has an impact (5-15%) that exceeds changing the sequence modeling layer (1-3%), completely overturning the implicit claims of papers over the past decade.
- **Hidden Heterogeneity**: While Paper A and Paper B both report on "PatchTST," they are actually testing different things due to hidden D1-D4 discrepancies—explaining why SOTAs in new papers are quickly overturned.
- **Reference Architecture Insight**: Simple architectures (even univariate linear models) can often compete with complex SOTA models when design choices are aligned—the performance gains of complex models are often disguised as "architectural superiority."

## Highlights & Insights
- **Methodological Breakthrough**: Systematic decomposition through reference architectures and design dimensions makes the contribution of each dimension transparent; this "isolation of variables" is simple yet efficient and can be directly applied to benchmarks in other fields (NLP, CV).
- **The Biggest "Aha" Moment**: The fact that performance can shift by 10% just by changing the parameter sharing method reveals an unsettling truth—much of the enthusiasm for "architectural innovation" over the last decade may stem from a neglect of benchmarking practices.
- **Reusable Idea**: The "Model Card" concept provides a template that mandates the declaration of all key design choices—it can be generalized to other fields beyond time series as a metadata standard for future paper submissions.

## Limitations & Future Work
- This paper only analyzes long-term forecasting (96 / 336 steps) scenarios; short-term forecasting may follow different patterns.
- The experiment covers 44 datasets, which is still limited relative to the vast scope of deep learning applications.
- No clear threshold is given for "how much difference is considered too large."
- It does not demonstrate how to retrospectively evaluate "hidden design dimension scores" for already published papers.
- Improvement: Establish a "Time Series Paper Metadata Database" to automatically extract or require authors to declare D1-D4 values for every paper; collaborate with journal editorial boards to develop submission checklists for time series papers.

## Related Work & Insights
- **vs. Brigato et al. (2026)**: Brigato focuses on the evaluation procedure itself (cross-validation settings); this paper delves into architectural design dimensions, pointing out that the root cause is the heterogeneity of design choices themselves.
- **vs. Zeng et al. (2023)**: Zeng found that linear models sometimes outperform Transformers; this paper confirms and explains this within a more systematic framework—it is not that linear models are truly superior, but that different papers apply different design choices to them.
- **Insight**: The ML community sometimes lacks reflection on fundamental experimental design while chasing new architectures; this paper's methodology draws from classic scientific experimental design principles.

## Rating
- Novelty: ⭐⭐⭐⭐ Not the first paper to point out benchmarking issues, but the first to systematically identify and quantify the independent impact of four key design dimensions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 44 datasets + 7 SOTAs + multiple reference architectures + 3 independent runs + clear ablations; the scale and rigor are top-tier.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, moving seamlessly from problem diagnosis to framework proposal, experimental validation, and solution.
- Value: ⭐⭐⭐⭐⭐ The impact on the entire time series forecasting community could be profound, with the potential to drive the field toward more mature and standardized evaluation standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](../../NeurIPS2025/time_series/selective_learning_for_deep_time_series_forecasting.md)
- [\[ICML 2026\] Interpretability in Deep Time Series Models Demands Semantic Alignment](interpretability_in_deep_time_series_models_demands_semantic_alignment.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](../../NeurIPS2025/time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[ICLR 2026\] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](../../ICLR2026/time_series/benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](../../AAAI2026/time_series/counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)

</div>

<!-- RELATED:END -->

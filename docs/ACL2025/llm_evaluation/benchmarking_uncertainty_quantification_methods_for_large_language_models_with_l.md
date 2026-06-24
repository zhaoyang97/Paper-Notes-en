---
title: >-
  [Paper Note] Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
description: >-
  [ACL 2025 (TACL 2025)][LLM Evaluation][Uncertainty Quantification] This paper constructs the LM-Polygraph uncertainty quantification (UQ) benchmark, implementing 30+ SOTA methods, and systematically evaluates the performance of UQ and confidence normalization techniques across 11 text generation tasks, providing a unified evaluation framework for LLM hallucination detection.
tags:
  - "ACL 2025 (TACL 2025)"
  - "LLM Evaluation"
  - "Uncertainty Quantification"
  - "Hallucination Detection"
  - "Benchmarking"
  - "Confidence Calibration"
  - "Text Generation"
date: 2026-05-08
content_hash: 87e512faee74975f
---

# Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph

**Conference**: ACL 2025 (TACL 2025)  
**arXiv**: [2406.15627](https://arxiv.org/abs/2406.15627)  
**Code**: [https://github.com/IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph)  
**Area**: LLM Evaluation  
**Keywords**: Uncertainty Quantification, Hallucination Detection, Benchmarking, Confidence Calibration, Text Generation

## TL;DR

This paper constructs the LM-Polygraph uncertainty quantification (UQ) benchmark, implementing 30+ SOTA methods, and systematically evaluates the performance of UQ and confidence normalization techniques across 11 text generation tasks, providing a unified evaluation framework for LLM hallucination detection.

## Background & Motivation

**Background**: The rapid development of large language models (LLMs) has driven their application in various scenarios, but the "hallucination" problem—where models generate seemingly plausible but actually incorrect content—remains a key challenge. Uncertainty Quantification (UQ) is an important technical approach to address this issue: if the model's "confidence" in its output can be accurately estimated, users can be alerted or more reliable fallback mechanisms can be triggered when confidence is low.

**Limitations of Prior Work**: Existing UQ research suffers from severe fragmentation: (1) different papers use different implementations of UQ methods, lacking a unified codebase; (2) evaluation datasets and metrics are inconsistent, making fair comparisons difficult; (3) many methods are evaluated only on specific tasks, lacking cross-task generalization analysis; (4) confidence normalization—translating raw UQ scores into interpretable probabilities—is ignored by most works.

**Key Challenge**: Despite the existence of numerous UQ methods, there lacks a unified platform to compare their performance fairly and under controlled conditions. The differences in experimental setups across various papers make it impossible for researchers to determine which method is most effective in which scenario.

**Goal**: To build a comprehensive UQ benchmark to (1) implement SOTA methods in a unified manner, (2) provide a controlled evaluation environment, (3) cover a variety of text generation tasks, and (4) support the evaluation of confidence normalization methods.

**Key Insight**: Building on the previous LM-Polygraph (EMNLP 2023 Demo) framework, the authors greatly expand the library of methods and the scope of evaluation, upgrading it from a tool to a complete benchmarking platform.

**Core Idea**: A large-scale comparison of UQ methods under a unified framework, covering white-box and black-box methods, sequence-level and claim-level granularities, and confidence normalization evaluations.

## Method

### Overall Architecture

The LM-Polygraph benchmark consists of three levels: (1) Method Library: implementing 30+ UQ methods covering multiple categories such as information-theoretic methods, semantic diversity methods, density estimation methods, ensemble methods, and self-reflection methods; (2) Evaluation Platform: supporting unified evaluation across 11 tasks, including QA, summarization, translation, fact verification, etc.; (3) Normalization Evaluation: additionally assessing normalization methods that convert raw UQ scores into interpretable confidence values.

### Key Designs

1. **Multi-Category UQ Methods**:

    - **Function**: Provides a unified implementation covering all mainstream UQ paradigms.
    - **Mechanism**: Categorizes UQ methods into several major groups: (a) **Information-Theoretic Methods** (white-box): token-probability-based methods such as maximum sequence probability, perplexity, mean/max token entropy, Monte-Carlo sequence entropy, point-wise mutual information, etc.; (b) **Semantic Diversity Methods**: detecting semantic consistency of outputs through multiple sampling, such as Semantic Entropy, TokenSAR, EigenScore, etc.; (c) **Density Estimation Methods**: utilizing the density of hidden layer representations to estimate whether an input is out-of-distribution, such as Mahalanobis distance, robust density estimation, etc.; (d) **Self-Reflection Methods**: letting the model evaluate the reliability of its own output, such as $p(\text{True})$, Verbalized Uncertainty; (e) **Black-Box Methods**: methods that do not require access to the model internals, such as EigV based on graph Laplacian eigenvalues, lexical similarity LexSim, etc.
    - **Design Motivation**: Only by implementing all mainstream methods can a fair and comprehensive comparison be conducted, avoiding evaluation bias caused by implementation differences.

2. **Unified Evaluation Environment**:

    - **Function**: Provides a controlled and reproducible evaluation framework.
    - **Mechanism**: Defines standard dataset splits, preprocessing pipelines, and evaluation metrics for each task. The primary evaluation metrics include AUROC (the ability to distinguish between correct and incorrect outputs), AUPR (Area Under the Precision-Recall curve), and Expected Calibration Error ($ECE$). It supports evaluation at both sequence-level (whether the entire output is reliable) and claim-level (whether an individual claim is reliable) granularities.
    - **Design Motivation**: The lack of a unified evaluation environment is the root cause of the fragmentation in current UQ research, which must be solved at the platform level.

3. **Confidence Normalization Assessment**:

    - **Function**: Evaluates normalization methods that convert raw UQ scores into interpretable probability values.
    - **Mechanism**: Raw UQ scores (such as entropy, probability, etc.) typically cannot be directly interpreted as "the model is X% confident". This work evaluates various normalization methods, including Platt Scaling, Temperature Scaling, Isotonic Regression, etc., and measures their $ECE$ after mapping UQ scores to calibrated probabilities.
    - **Design Motivation**: For practical applications, an interpretable confidence score (e.g., "this answer is 85% likely to be correct") is much more useful than a raw entropy value.

### Evaluation Task Coverage

Covers 11 tasks: open-domain QA (TriviaQA, CoQA, Natural Questions), reading comprehension, text summarization (CNN/DM, XSum), machine translation (WMT), commonsense reasoning, fact verification, mathematical reasoning, etc.

## Key Experimental Results

### Main Results

AUROC performance of UQ methods across different tasks (the ability to detect incorrect outputs):

| UQ Method Category | Representative Method | Mean AUROC (QA Task) | Mean AUROC (Summarization Task) | Mean AUROC (Translation Task) | Overall Ranking |
|-----------|---------|----------------|-----------------|-----------------|---------|
| Information-Theoretic (White-box) | Mean Token Entropy | 0.72 | 0.68 | 0.71 | Medium |
| Information-Theoretic (White-box) | Perplexity | 0.70 | 0.66 | 0.69 | Medium |
| Semantic Diversity | Semantic Entropy | 0.78 | 0.73 | 0.74 | Best |
| Semantic Diversity | EigenScore | 0.76 | 0.71 | 0.73 | Excellent |
| Density Estimation | Mahalanobis Distance | 0.65 | 0.62 | 0.63 | Weak |
| Self-Reflection | $p(\text{True})$ | 0.74 | 0.70 | 0.68 | Good |
| Black-box | EigV (Graph Laplacian) | 0.75 | 0.72 | 0.71 | Good |
| Black-box | Verbalized UQ | 0.71 | 0.67 | 0.65 | Medium |

### Ablation Study - Effectiveness of Normalization Methods

| Normalization Method | Mean $ECE \downarrow$ | Mean AUROC | Explanation |
|-----------|---------|-----------|------|
| No Normalization | 0.32 | 0.74 | Poor calibration of raw scores |
| Platt Scaling | 0.12 | 0.74 | AUROC unchanged, calibration significantly improved |
| Temperature Scaling | 0.14 | 0.74 | Performance close to Platt |
| Isotonic Regression | 0.09 | 0.74 | Best calibration performance |
| Histogram Binning | 0.15 | 0.73 | Simple but effective |

### Key Findings

- **Semantic diversity methods are overall optimal**: Semantic Entropy and EigenScore perform best on most tasks because they distinguish between "different formulations of the same meaning" and "truly different answers".
- **White-box methods outperform black-box methods**: Methods with access to token probabilities generally outperform those that only observe the final output, though the gap narrows as models grow larger.
- **Density estimation methods perform poorly**: Hidden-representation-based methods offer limited effectiveness in NLG tasks, possibly because they were originally designed for classification tasks.
- **Confidence normalization is crucial**: The calibration error of raw UQ scores is highly inflated ($ECE \sim 0.32$), but can be reduced to $0.09$ after normalization, making confidence scores truly interpretable.
- **No "one-size-fits-all" method**: The optimal method varies across tasks and models, but semantic diversity methods remain the most robust choice.
- **Claim-level evaluation is more challenging than sequence-level**: Identifying unreliable specific claims at the claim level is significantly more difficult than determining whether the entire output is reliable.

## Highlights & Insights

- This is currently the most comprehensive LLM uncertainty quantification benchmark, implementing 30+ methods and evaluating them uniformly across 11 tasks.
- It provides the first systematic evaluation of confidence normalization methods, which is critical for practical deployment.
- The open-source codebase (468 stars) has become a de facto standard tool for UQ research.
- The benchmark design supports easy integration of new methods, lowering the barrier for subsequent research.

## Limitations & Future Work

- The current evaluation focuses primarily on English models; UQ behaviors may differ in multilingual scenarios.
- Certain UQ methods (such as semantic diversity methods) require multiple sampling steps, incurring high computational overhead, and the paper lacks a detailed efficiency comparison.
- Uncertainty quantification for multimodal LLMs is not covered.
- Future work can explore ensemble strategies combining multiple UQ methods and the user-experience impact of UQ methods in practical deployments.

## Related Work & Insights

- This is an upgraded version of the same line of work as the LM-Polygraph Demo paper in EMNLP 2023, transitioning from a demo tool to a complete benchmark paper.
- Semantic Entropy, as one of the best-performing methods, has key implications for subsequent UQ research with its core idea: measuring output diversity via semantic equivalence clustering.
- For LLM application developers, this paper provides a clear guide for selecting UQ methods.

## Rating

- **Novelty**: ⭐⭐⭐ — Limited novelty at the methodological level (primarily a benchmark contribution), but compensated for by its systematicness and comprehensiveness.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Large-scale evaluation with 30+ methods, 11 tasks, and multiple models, making it the most comprehensive UQ benchmark to date.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear framework description and practically valuable conclusions.
- **Value**: ⭐⭐⭐⭐⭐ — Published in TACL with 83+ citations, becoming a standard reference in the UQ field, with its codebase widely used.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](../../NeurIPS2025/llm_evaluation/efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ICLR 2026\] Complementing Self-Consistency with Cross-Model Disagreement for Uncertainty Quantification](../../ICLR2026/llm_evaluation/complementing_self-consistency_with_cross-model_disagreement_for_uncertainty_qua.md)
- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)

</div>

<!-- RELATED:END -->

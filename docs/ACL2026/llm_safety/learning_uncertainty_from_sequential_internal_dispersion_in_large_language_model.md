---
title: >-
  [Paper Note] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models
description: >-
  [ACL 2026][LLM Safety][Uncertainty Estimation] Ours proposes the SIVR framework, which computes the internal variance of LLM hidden states across layers (Generalised Variance, Circular Variance…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Uncertainty Estimation"
  - "Hallucination Detection"
  - "Hidden State Variance"
  - "Sequence Aggregation"
  - "Internal Representation Dispersion"
date: 2026-05-08
content_hash: e0945a4aef747c0a
---

# Learning Uncertainty from Sequential Internal Dispersion in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.15741](https://arxiv.org/abs/2604.15741)  
**Code**: [GitHub](https://github.com/ponhvoan/internal-variance)  
**Area**: Uncertainty Estimation / Hallucination Detection  
**Keywords**: Uncertainty Estimation, Hallucination Detection, Hidden State Variance, Sequence Aggregation, Internal Representation Dispersion

## TL;DR

Ours proposes the SIVR framework, which computes the internal variance of LLM hidden states across layers (Generalised Variance, Circular Variance, and Token Entropy) as token-level features. These features are then processed by a lightweight Transformer encoder to aggregate full-sequence patterns for uncertainty estimation and hallucination detection, significantly outperforming baselines with stronger generalization.

## Background & Motivation

**Background**: Uncertainty estimation is a critical means for detecting hallucinations in LLMs. Existing approaches include sampling consistency (e.g., Semantic Entropy), output probability methods (e.g., Entropy), and internal state probing methods.

**Limitations of Prior Work**: (1) Sampling-based methods incur high computational overhead; (2) Methods like CoE (Consistency of Evolution) impose overly strict assumptions on layer-wise evolution that do not hold across different models or tasks; (3) Relying solely on the last or average token loses temporal patterns.

**Key Challenge**: Compressing internal signals into a single score (as in CoE) ignores variance patterns at different token positions. For instance, in the phrase "Praia is in Portugal," a variance spike at "Portugal" can signal an error, but simple mean aggregation would mask this signal.

**Goal**: Design internal state features based on more relaxed assumptions while preserving full sequence information.

**Key Insight**: Uncertainty is reflected in the "degree of dispersion" of hidden states across layers—representations are more concentrated when correct and more dispersed when incorrect.

**Core Idea**: Use three dispersion statistics (Generalised Variance, Circular Variance, and Token Entropy) to describe the cross-layer dispersion of each token, and employ a Transformer encoder to learn full-sequence patterns for hallucination prediction.

## Method

### Overall Architecture

For each generated token, the hidden states from all layers are extracted. Three internal variance features $\bm{v}_t = [v_t, c_t, e_t]$ are calculated to form a sequence, which is then fed into a lightweight Transformer encoder for binary classification.

### Key Designs

1.  **Generalised Variance**:
    - **Function**: Measures "volume" dispersion across layers.
    - **Mechanism**: Computes the log-determinant of the regularized covariance matrix $v_t = \log\det(\Sigma') = \sum_i \log \lambda_i$, aggregating the entire feature spectrum.
    - **Design Motivation**: Unlike CoE which only considers differences between adjacent layers, Generalised Variance is directly related to differential entropy and provides a more comprehensive measure of dispersion.

2.  **Circular Variance**:
    - **Function**: Measures "directional" dispersion across layers.
    - **Mechanism**: Calculates the magnitude of the mean vector after normalizing hidden states at each layer: $c_t = 1 - \|\frac{1}{L+1}\sum_l \hat{\bm{h}}_t^l\|$.
    - **Design Motivation**: Complementary to Generalised Variance—one captures magnitude while the other captures direction. It implicitly considers all pairwise relationships between layers.

3.  **Sequence Aggregation Transformer Classifier**:
    - **Function**: Learns hallucination detection from full-sequence dispersion patterns.
    - **Mechanism**: Consists of an embedding layer (128 dimensions), a single-layer Transformer encoder, and a linear classification head.
    - **Design Motivation**: Preserves sequential order to capture temporal patterns like "variance spikes," which is more effective than mean or last-token aggregation.

### Loss & Training

Binary Cross-Entropy with $l_2$ regularization. Only a few hundred to a few thousand labeled samples are required.

## Key Experimental Results

### Main Results

Comparison of AUC on 7 datasets using Llama-3.1-8B:

| Method | TriviaQA | SciQ | MedMCQA | MATH | Avg AUC | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Entropy | 80.46 | 72.85 | 62.76 | 62.77 | 67.63 | 7.96 |
| SE | 84.44 | 79.44 | 66.88 | 67.27 | 68.87 | 7.13 |
| CoE-C | 66.97 | 75.06 | 62.14 | 58.67 | 61.25 | 11.08 |
| **SIVR** | **90.75** | **83.64** | **68.37** | **71.22** | **75.35** | **1.88** |

### Ablation Study

| Configuration | Avg AUC | Description |
| :--- | :--- | :--- |
| Token Entropy Only | 71.2 | Basic effectiveness but insufficient |
| Generalised Variance Only | 72.8 | Complementary signal |
| Triple Combination (SIVR) | **75.35** | Best performance |
| Mean Aggregation instead of Sequence | 72.5 | Loss of temporal patterns |

### Key Findings

- SIVR achieves an average rank of 1.88, significantly outperforming the runner-up, with strong complementarity among the three features.
- Sequence aggregation improves AUC by 2-3 points compared to mean or last-token aggregation, proving the value of temporal patterns.
- OOD (Out-Of-Distribution) generalization is significantly better than CoE, requiring only minimal training data.

## Highlights & Insights

- **"Dispersion" assumption is more robust than "step size"**: CoE's assumption of consistent layer-wise evolution varies across models, whereas SIVR's dispersion assumption is more fundamental and universal.
- **Transferable sequence-structure paradigm**: Any task requiring the inference of sequence-level properties from token-level signals can benefit from this approach.
- **Lightweight yet effective**: Using only 3 statistics and a single-layer Transformer, the inference overhead is nearly negligible.

## Limitations & Future Work

- Requires labeled data; although the amount is small, new domains require additional annotations.
- Only validated on greedy decoding; performance under sampling-based decoding remains to be evaluated.
- Insufficient validation on large-scale models (70B+).
- Did not explore using SIVR for active hallucination mitigation.

## Related Work & Insights

- **vs CoE**: CoE has overly strong assumptions that fail across tasks; SIVR uses a more relaxed dispersion assumption.
- **vs Semantic Entropy**: SE requires multiple samples and is computationally expensive; SIVR only requires a single forward pass.
- **vs Lookback Lens**: Focuses on specific layers or attention patterns, whereas SIVR provides a more global perspective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of internal variance features is clear, and while components are simple, the combination is effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablation studies across 7 datasets and multiple models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and effective visualizations.
- **Value**: ⭐⭐⭐⭐⭐ High practicality with direct value for hallucination detection applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)
- [\[ACL 2026\] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](topic-based_watermarks_for_large_language_models.md)
- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](jailbreaking_large_language_models_with_morality_attacks.md)
- [\[ACL 2026\] Multi-component Causal Tracing in Large Language Models](multi-component_causal_tracing_in_large_language_models.md)

</div>

<!-- RELATED:END -->

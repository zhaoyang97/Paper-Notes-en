---
title: >-
  [Paper Note] OrdShap: Feature Position Importance for Sequential Black-Box Models
description: >-
  [NeurIPS 2025][Shapley Value] This paper proposes OrdShap, a feature attribution method for sequential models that, for the first time, decouples **Value Importance (VI)** from **Position Importance (PI)** for each feature, providing theoretical guarantees grounded in the Sanchez-Bergantiños game-theoretic value.
tags:
  - NeurIPS 2025
  - Shapley Value
  - Feature Position Importance
  - Sequential Models
  - Interpretability
  - Sanchez-Bergantiños Value
date: 2026-05-08
content_hash: df69a5f8023cbbca
---

# OrdShap: Feature Position Importance for Sequential Black-Box Models

**Conference**: NeurIPS 2025
**arXiv**: [2507.11855](https://arxiv.org/abs/2507.11855)
**Code**: Not available
**Area**: Explainable AI / Feature Attribution
**Keywords**: Shapley Value, Feature Position Importance, Sequential Models, Interpretability, Sanchez-Bergantiños Value

## TL;DR

This paper proposes OrdShap, a feature attribution method for sequential models that, for the first time, decouples **Value Importance (VI)** from **Position Importance (PI)** for each feature, providing theoretical guarantees grounded in the Sanchez-Bergantiños game-theoretic value.

## Background & Motivation

Deep learning sequential models (Transformers, RNNs) excel on time-series and natural language data, yet their black-box nature necessitates post-hoc attribution methods for prediction explanation. Existing approaches (e.g., KernelSHAP, LIME, Integrated Gradients) share a fundamental assumption flaw: **they assume fixed feature ordering, conflating the effects of feature values and feature positions**.

Consider a medical scenario: when predicting length of hospital stay, whether a blood glucose measurement appears early or late in the sequence can substantially affect the prediction, even when the measurement value itself remains unchanged. Existing methods cannot distinguish whether a feature is important because of its value or because of its position in the sequence.

**Root Cause**: Sequential model predictions depend jointly on feature values and feature positions, yet existing attribution methods cannot separate these two effects.

**Starting Point**: A new game-theoretic framework is defined that augments Shapley values with a permutation dimension, attributing marginal contributions to each feature at each position independently, then deriving separate Value Importance and Position Importance through aggregation.

## Method

### Overall Architecture

The core of OrdShap is a $d \times d$ attribution matrix $\gamma_{i,\ell}$, where rows correspond to features $i$ and columns to positions $\ell$. Each entry represents the importance of feature $i$ when permuted to position $\ell$. OrdShap-VI (Value Importance) and OrdShap-PI (Position Importance) are then extracted via marginalization and linear regression, respectively.

### Key Designs

1. **Generalized Characteristic Function $\tilde{\omega}$ (Section 4.1)**: The standard Shapley characteristic function $\nu(S)$ conditions only on a feature subset $S$. OrdShap introduces a new characteristic function that depends simultaneously on the subset and a permutation:

    $\tilde{\omega}_{f,x}(S,\sigma) = \mathbb{E}_{x' \sim \mathcal{X}}\left[f(x') \mid x'_{\sigma^{-1}(i)} = \begin{cases} x_i & \forall i \in S \\ x'_i & \forall i \in N \setminus S \end{cases}\right]$

   This function ablates features in the manner of SHAP while additionally permuting retained features (the new dimension). When the permutation is the identity, it reduces to the standard SHAP characteristic function.

2. **OrdShap Definition (Definition 1)**: For each feature $i$ and position $\ell$, the OrdShap value is defined as:

    $\gamma_{i,\ell}(N,\tilde{\omega}) = \sum_{\substack{S \subseteq N \\ i \in S}} \sum_{\substack{\sigma \in \mathfrak{S}_N \\ \sigma^{-1}(i)=\ell}} \frac{(|S|-1)!(|N|-|S|)!}{(|N|-1)!|N|!}\left[\tilde{\omega}(S,\sigma) - \tilde{\omega}(S\setminus\{i\},\sigma)\right]$

    - **OrdShap-VI** (Eq. 9): Average over all positions, $\bar{\gamma}_i = \frac{1}{|N|}\sum_\ell \gamma_{i,\ell}$. Theorem 1 establishes that this is equivalent to the Sanchez-Bergantiños value, satisfying efficiency, symmetry, null player, and additivity axioms.
    - **OrdShap-PI** (Eq. 10): Linear regression of $\gamma_{i,\ell}$ on position $\ell$; the slope $\beta_i$ captures the direction and magnitude of the position effect on importance.

3. **Efficient Approximation Algorithms**: Exact computation requires $\mathcal{O}(d! \cdot 2^d)$; two approximations are proposed:

    - **Sampling Algorithm** (Section 5.1): Randomly samples subsets and permutations to estimate each $\gamma_{i,\ell}$; complexity $\mathcal{O}(dKL\delta_f + d^2KL)$.
    - **Least-Squares Algorithm** (Section 5.2, Definition 2): Exploits Corollary 2.1 (OrdShap-VI equals the Shapley value on the averaged characteristic function $\bar{\nu}$) to solve for $\alpha$ via KernelSHAP, then obtains $\beta$ by regression. Complexity $\mathcal{O}(KL\delta_f + d^2KL + d^3)$, generally faster.

### Loss & Training

OrdShap is a post-hoc explanation method and does not involve model training. The optimization objective for the least-squares approximation (Eq. 11) is a weighted least-squares problem. With weights $\mu(|S|) = \frac{|N|-1}{\binom{|N|}{s}|S|(|N|-s)}$ chosen by subset size, the optimal solution recovers the SB value (Theorem 2).

## Key Experimental Results

### Main Results: Inclusion/Exclusion AUC (Value Importance Evaluation)

| Method | EICU-LOS (Inc↑) | EICU-Mort (Inc↑) | MIMICIII-LOS (Inc↑) | MIMICIII-Mort (Inc↑) | IMDB (Inc↑) |
|------|---------|---------|---------|---------|---------|
| KernelSHAP | 0.904 | 0.801 | 0.898 | 0.855 | 0.863 |
| DeepLIFT | 0.906 | 0.804 | 0.893 | 0.854 | 0.784 |
| LIME | 0.866 | 0.776 | 0.885 | 0.804 | 0.859 |
| Random | 0.812 | 0.769 | 0.851 | 0.705 | 0.797 |
| **OrdShap-VI** | **0.913** | **0.809** | **0.899** | **0.862** | **0.866** |

| Method | EICU-LOS (Exc↓) | EICU-Mort (Exc↓) | MIMICIII-Mort (Exc↓) | IMDB (Exc↓) |
|------|---------|---------|---------|---------|
| KernelSHAP | 0.626 | 0.730 | 0.485 | **0.766** |
| **OrdShap-VI** | **0.573** | **0.724** | **0.472** | 0.779 |

### Ablation Study: Position Importance Evaluation (Model Output Change After Feature Permutation)

| Dataset | OrdShap-PI Effect | Baseline Effect | Notes |
|------|---------|---------|------|
| EICU-LOS | Output unchanged / slightly increases | No effect | OrdShap-PI correctly identifies position dependence |
| EICU-Mort | Output significantly increases | Output decreases | Baselines fail to capture position importance |
| MIMICIII-LOS | Output increases | Output decreases | Permuting by OrdShap-PI genuinely strengthens prediction |
| IMDB | Output largely unchanged | Output unchanged | DistilBERT is insensitive to sentence order |

### Key Findings

- OrdShap-VI achieves the highest Inclusion AUC on all EHR datasets among all compared methods, demonstrating that incorporating position information yields more accurate value attribution.
- OrdShap-PI is the only method capable of correctly quantifying the influence of feature position; permuting features according to OrdShap-PI ranking enhances model predictions, whereas ranking by conventional methods degrades or does not affect them.
- On synthetic data, OrdShap fully separates value and position effects across 7 token types (Figure 5), while conventional methods cannot distinguish between the two.
- Position effects are weak on the IMDB dataset, suggesting that DistilBERT's sentiment analysis is relatively insensitive to sentence order—consistent with intuition.

## Highlights & Insights

- **First Decoupling of Value and Position**: Addresses an important gap in sequential model interpretability, with particular practical relevance for time-sensitive domains such as healthcare.
- **Solid Game-Theoretic Foundation**: The connection to the Sanchez-Bergantiños value (Theorem 1) provides axiomatic guarantees for the method.
- **Elegant Least-Squares Approximation**: Corollary 2.1 enables decoupled estimation of $\alpha$ and $\beta$—value importance is first obtained via KernelSHAP, and position importance is subsequently recovered by regression.
- A clinical case study (Figure 6) vividly illustrates practical significance: the high KernelSHAP score of a bedside blood glucose test is shown to stem primarily from a position effect rather than value importance.

## Limitations & Future Work

- Computational cost remains relatively high, despite the reduced number of model calls in the LS algorithm.
- The linear OrdShap-PI formulation assumes a linear position effect, which may fail to capture nonlinear positional dependencies.
- The current framework assumes each feature can be permuted to any position; scenarios with strict temporal ordering constraints (e.g., causal chains) require additional treatment (discussed in Appendix B.1).
- Integration with model-internal information such as attention weights has not been explored.
- Exclusion AUC on IMDB is lower than KernelSHAP, possibly because position effects are genuinely weaker in NLP settings.

## Related Work & Insights

- **Relation to KernelSHAP**: OrdShap is a natural generalization of KernelSHAP to ordered-coalition games.
- **PoSHAP** is the most closely related prior work, but it is a global method (averaged across samples), whereas OrdShap is a local method that decouples value and position effects for individual samples.
- **TimeSHAP** targets sequential models but does not explicitly model position importance.
- The SB value has been applied in network theory; this paper is the first to introduce it in the context of feature attribution.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Decoupling value and position is an elegant problem that has not been previously addressed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset evaluation with quantitative and qualitative analyses and synthetic validation; evaluation on large-scale LLMs is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Game-theoretic definitions are precise and toy examples are highly intuitive.
- Value: ⭐⭐⭐⭐ — Directly applicable to medical time-series analysis; the methodology generalizes to all sequential models.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](../../ACL2026/interpretability/revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[NeurIPS 2025\] TangledFeatures: Robust Feature Selection in Highly Correlated Spaces](tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)
- [\[NeurIPS 2025\] Base Models Know How to Reason, Thinking Models Learn When](base_models_know_how_to_reason_thinking_models_learn_when.md)

<!-- RELATED:END -->

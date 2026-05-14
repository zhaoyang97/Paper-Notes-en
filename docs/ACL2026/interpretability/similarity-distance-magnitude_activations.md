---
title: >-
  [Paper Note] Similarity-Distance-Magnitude Activations
description: >-
  [ACL 2026][Interpretability][Activation functions] This paper proposes SDM (Similarity-Distance-Magnitude) activations as a more robust replacement for softmax. By decoupling and integrating three epistemic dimensions—Si…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Activation functions"
  - "softmax alternatives"
  - "selective classification"
  - "out-of-distribution detection"
  - "predictive uncertainty"
date: 2026-05-08
content_hash: 5d561a2fa653b9e6
---

# Similarity-Distance-Magnitude Activations

**Conference**: ACL 2026
**arXiv**: [2509.12760](https://arxiv.org/abs/2509.12760)
**Code**: None
**Area**: Interpretability / Uncertainty Estimation
**Keywords**: Activation functions, softmax alternatives, selective classification, out-of-distribution detection, predictive uncertainty

## TL;DR

This paper proposes SDM (Similarity-Distance-Magnitude) activations as a more robust replacement for softmax. By decoupling and integrating three epistemic dimensions—Similarity (deep matching with correct training predictions), Distance (proximity to the training distribution), and Magnitude (distance to the decision boundary)—into a novel activation $\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$, the method constructs an SDM estimator for selective classification that is more robust than existing calibration approaches under covariate shift and out-of-distribution inputs.

## Background & Motivation

**Background**: The parameter non-identifiability of neural language models (multiple parameter configurations can yield the same output distribution) makes direct parameter interpretation extremely difficult. Softmax is the most widely used final-layer activation, converting logits into a probability distribution. Existing uncertainty quantification methods span Bayesian (e.g., variational inference), frequentist (e.g., conformal prediction), and empirical approaches (e.g., temperature scaling), yet the prevalence of high-confidence errors and hallucinations in LLMs indicates fundamental shortcomings in these methods.

**Limitations of Prior Work**: Softmax captures only one dimension—Magnitude (distance to the decision boundary)—through the relative magnitudes of logits. It ignores two critical epistemic signals: (1) whether the model's prediction is consistent with correct prediction patterns in the training set (Similarity); and (2) whether the input lies within the coverage of the training distribution (Distance). This causes models to produce high-confidence predictions even for out-of-distribution inputs.

**Key Challenge**: Effective predictive uncertainty requires decomposing the sources of epistemic uncertainty, but softmax's single temperature parameter $\tau$ cannot achieve instance-level multi-dimensional uncertainty representation—$\tau$ is a global hyperparameter, and instance-level variation is determined solely by logit magnitudes.

**Goal**: Design a new activation function that explicitly decomposes and integrates epistemic uncertainty signals along the Similarity, Distance, and Magnitude dimensions, providing a more reliable foundation for selective classification.

**Key Insight**: The method exploits the capacity of neural networks as implicit instance-based metric learners, constructing a compact representation space via an exemplar adaptor (1-D CNN adapter) on top of frozen pretrained LM hidden states, from which Similarity and Distance signals are extracted.

**Core Idea**: Replace softmax's fixed base $e$ with a data-driven base $(2+q)$ (dependent on Similarity), and replace the fixed temperature $\tau$ with an instance-level Distance $d$—so that the activation output directly encodes epistemic uncertainty along all three dimensions.

## Method

### Overall Architecture

The SDM system comprises three layers: (1) a frozen pretrained LM providing hidden states $\mathbf{h}$; (2) an exemplar adaptor (1-D CNN + linear layer) mapping $\mathbf{h}$ to a compact representation $\mathbf{h}'$ and new logits $\mathbf{z}'$; (3) an SDM activation layer that uses $\mathbf{h}'$ to compute Similarity $q$ and Distance $d$, combining them with $\mathbf{z}'$ to output a calibrated probability distribution. Built on top of this, the SDM estimator constructs high-reliability regions for selective classification via data-driven empirical CDF partitioning.

### Key Designs

1. **Similarity ($q$) Computation**:

    - **Function**: Quantifies the degree of deep matching between a test instance and correct prediction patterns in the training set.
    - **Mechanism**: In the representation space $\mathbf{h}'$ of the exemplar adaptor, training instances are ranked by $L^2$ distance, and the number of consecutive matches from the nearest neighbor onward is counted, where each match must satisfy: (a) the training sample's prediction matches the current instance's prediction ($\hat{y} = \hat{y}^{tr}_{(i)}$); (b) the training sample's prediction is correct ($\hat{y}^{tr}_{(i)} = y^{tr}_{(i)}$); and (c) the matches are contiguous (no gaps). $q \in \{0, \ldots, |D_{tr}|\}$; $q=0$ indicates that even the nearest neighbor fails the condition, effectively signaling out-of-distribution inputs.
    - **Design Motivation**: Unlike conventional KNN rules, SDM's Similarity jointly leverages model predictions and ground-truth labels—if the nearest training instances share the same label and are also correctly predicted by the model, this indicates reliable discriminative capability in that region.

2. **Distance ($d$) Computation**:

    - **Function**: Quantifies the normalized distance from a test instance to the training distribution.
    - **Mechanism**: The $L^2$ distance to the nearest training neighbor $d_{\text{nearest}}$ is first computed. Normalization is then performed using per-class empirical CDFs from a calibration set $D_{ca}$: $d = \min[1 - \text{eCDF}^{y_1}_{ca}(d_{\text{nearest}}), \ldots, 1 - \text{eCDF}^{y_C}_{ca}(d_{\text{nearest}})]$. When $d_{\text{nearest}}$ exceeds the maximum distance observed in the labeled data, $d=0$ and SDM outputs a uniform distribution, indicating maximum uncertainty.
    - **Design Motivation**: Taking the minimum over all class CDFs ensures a conservative estimate—even if the distance appears normal relative to some classes, an anomalously large distance relative to any single class triggers high uncertainty.

3. **SDM Activation and High-Reliability Region Estimation**:

    - **Function**: Integrates all three dimensions into a calibrated probability distribution and automatically identifies high-reliability prediction regions.
    - **Mechanism**: The SDM activation is $\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$. The corresponding loss uses the change-of-base formula $\log_{(2+q)}$. The high-reliability region is determined by: first computing the rescaled value $q' = \min(q, (2+q)^{\text{sdm}(\mathbf{z}')_{\hat{y}}})$, then progressively increasing the threshold $q'_{\min}$ over the subset where $q' > 0$, until the conformal thresholds $\psi_c$ across all classes reach the target confidence level $\alpha$ (e.g., 0.95). Predictions satisfying $q' \geq q'_{\min}$ and $\text{sdm}(\mathbf{z}')_{\hat{y}} \geq \psi_{\hat{y}}$ enter the high-reliability region.
    - **Design Motivation**: Progressively tightening the $q'$ threshold identifies a region satisfying both class-conditional and prediction-conditional accuracy requirements, providing theoretically grounded selective classification. When no finite $q'_{\min}$ exists, this serves as a practical indicator that the model or data is insufficient to support reliable estimation.

### Loss & Training

The exemplar adaptor (1-D CNN + linear layer) is trained using the SDM loss with the underlying LM parameters frozen. The first training round initializes with standard softmax ($q=e-2, d=1$); subsequent rounds recompute $q$ and $d$. Training stops at the lowest class-balanced loss on the calibration set. The procedure is repeated $J=10$ times with random splits and parameter initializations, and the global optimum is selected. The CNN uses $M=1000$ filters, with 200 epochs per round.

## Key Experimental Results

### Main Results

**Sentiment Classification (In-Distribution) Selective Classification ($\alpha=0.95$)**

| Model + Estimator | Class-cond. y=0 | y=1 | Pred-cond. $\hat{y}$=0 | $\hat{y}$=1 | Acceptance Rate |
|---|---|---|---|---|---|
| phi3.5 softmax | 0.98 | 0.86 (<α) | 0.88 (<α) | 0.98 | 0.98 |
| phi3.5 tempScaling | 0.99 | 0.91 (<α) | 0.93 (<α) | 0.99 | 0.90 |
| phi3.5+sdm sdmHR | **1.00** | **0.99** | **0.99** | **1.00** | 0.68 |
| Mixtral8x7B softmax | 0.98 | 0.88 (<α) | 0.89 (<α) | 0.98 | 1.00 |
| Mixtral8x7B+sdm sdmHR | **0.99** | **0.98** | **0.99** | **0.98** | 0.74 |

**Sentiment Classification OOD (SentimentOOD, Out-of-Distribution)**

| Model + Estimator | Class-cond. y=0 | y=1 | Acceptance Rate | Notes |
|---|---|---|---|---|
| phi3.5 softmax | 1.00 | 0.54 (<α) | 0.96 | Overconfident, many errors |
| phi3.5 APS | 1.00 | 0.59 (<α) | 0.77 | Still below target |
| phi3.5+sdm sdmHR | **1.00** | **1.00** | **0.01** | Near-total rejection of OOD |

### Ablation Study

| Component | Effect | Notes |
|---|---|---|
| softmax (no adapter) | Class-conditional accuracy below target | Lacks Similarity and Distance |
| softmax (with adapter) | Meets target in-distribution but fails OOD | Better representations but no distance awareness |
| softmax($d \cdot \mathbf{z}'$) | Overly conservative (low in-distribution acceptance) | Distance used as temperature only, lacks Similarity |
| sdm$_\alpha$ (simple threshold) | Prediction-conditional met but class-conditional not guaranteed | Lacks high-reliability region constraint |
| **sdmHR (full estimator)** | **Both conditions met** | Synergy of Similarity + Distance + Magnitude |

### Key Findings

- On in-distribution data, softmax/tempScaling/APS/RAPS estimators without adapters are systematically overconfident, with class-conditional accuracy falling below the target $\alpha=0.95$.
- Differences are more dramatic on out-of-distribution data—phi3.5+sdm's sdmHR estimator reduces the SentimentOOD acceptance rate to approximately 1% (near-total rejection), while softmax still accepts 96% of OOD data with only 0.54 accuracy on the y=1 class.
- When Alg. 1 returns $q'_{\min} = \infty$, this provides a practical indicator that the model or data is insufficient to support reliable estimation.
- On the Factcheck task, softmax and APS exhibit severely below-target class-conditional accuracy on covariate-shifted test sets, while sdmHR appropriately tightens the acceptance region to maintain reliability.

## Highlights & Insights

- The definition of Similarity is particularly elegant—it requires not only that the nearest neighbors share the same label, but also that the model's predictions on those neighbors are correct, and that the matches are contiguous. This adds a dimension of "whether the model is reliable in this region" beyond traditional KNN.
- The mathematical form of SDM is elegant—generalizing softmax's base and temperature from fixed constants to data-driven instance-level variables; when $q=e-2, d=1$, it reduces exactly to standard softmax.
- The high-reliability region concept has direct value for multi-stage LLM pipelines—automatically routing predictions in high-reliability regions forward, while directing the remainder to more expensive tools or human review.

## Limitations & Future Work

- The exemplar adaptor requires maintaining the full training set for Similarity and Distance computation; storage and retrieval efficiency at scale is a concern.
- Validation is limited to binary classification tasks (sentiment analysis, fact-checking); multi-class and more complex NLP tasks require further evaluation.
- Computing $q$ requires ranking the entire training set by distance; inference latency requires optimization (potentially via approximate nearest neighbor search).
- The approach assumes that the exemplar adaptor can effectively learn discriminative representations on top of a frozen LM, an assumption that may not hold for certain tasks.

## Related Work & Insights

- **vs. Temperature Scaling**: Temperature scaling is a single-parameter global calibration method; SDM provides instance-level multi-dimensional calibration via $q$ and $d$, with dramatic differences in OOD scenarios.
- **vs. Conformal Prediction (APS/RAPS)**: The marginal coverage guarantees of conformal methods do not directly apply to selective classification (where only prediction sets of size 1 are accepted); SDM provides class-conditional coverage through the specific construction of high-reliability regions.
- **vs. VBLL**: Variational Bayesian last layers outperform softmax/tempScaling on OOD, but remain less robust than SDM under extreme OOD conditions.
- **vs. Exemplar-Based Methods**: SDM elevates exemplar matching from a post-hoc interpretive tool to a core component of the activation function itself.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Generalizing softmax's base and temperature from constants to data-driven variables; the three-dimensional epistemic uncertainty decomposition is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic ID/OOD/far-OOD comparisons and multi-estimator ablations, though the task scope is narrow (binary classification only).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations; the generalization path from softmax to SDM is clearly presented with consistent notation.
- **Value**: ⭐⭐⭐⭐ Provides a theoretically stronger framework for uncertainty quantification in LLM deployment; the high-reliability region concept has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](../../ICLR2026/interpretability/lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)
- [\[NeurIPS 2025\] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](../../NeurIPS2025/interpretability/explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Style over Story: Measuring LLM Narrative Preferences via Structured Selection](style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Similarity-Distance-Magnitude Activations
description: >-
  [ACL 2026][Interpretability][Activation functions] This paper proposes the SDM (Similarity-Distance-Magnitude) activation function as a more robust alternative to softmax. It achieves this by decoupling and integrating t…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Activation functions"
  - "softmax alternative"
  - "selective classification"
  - "out-of-distribution detection"
  - "predictive uncertainty"
date: 2026-05-08
content_hash: 4618d3cb0fd07622
---

# Similarity-Distance-Magnitude Activations

**Conference**: ACL 2026 Findings  
**arXiv**: [2509.12760](https://arxiv.org/abs/2509.12760)  
**Code**: None  
**Area**: Interpretability / Uncertainty Estimation  
**Keywords**: Activation functions, softmax alternative, selective classification, out-of-distribution detection, predictive uncertainty

## TL;DR

This paper proposes the SDM (Similarity-Distance-Magnitude) activation function as a more robust alternative to softmax. It achieves this by decoupling and integrating three epistemic dimensions: depth matching of correct predictions (Similarity), distance to the training distribution (Distance), and decision boundary distance (Magnitude) into a new activation $\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$. Based on this, an SDM estimator is constructed for selective classification, which proves more robust than existing calibration methods under covariate shift and out-of-distribution inputs.

## Background & Motivation

**Background**: The parameter non-identifiability of neural language models (where multiple sets of parameters can produce the same output distribution) makes direct parameter interpretation extremely difficult. Softmax is the most commonly used activation for the final output layer, transforming logits into probability distributions. Existing uncertainty quantification methods cover Bayesian (e.g., variational inference), frequentist (e.g., conformal prediction), and empirical approaches (e.g., temperature scaling). However, the prevalence of high-confidence errors and hallucinations in LLMs suggests fundamental deficiencies in these methods.

**Limitations of Prior Work**: Softmax only captures information from one dimension—Magnitude (decision boundary distance)—reflecting classification confidence through the relative scale of logits. It ignores two key epistemic signals: (1) whether the model's prediction aligns with correct prediction patterns in the training set (Similarity); and (2) whether the input is within the coverage of the training distribution (Distance). This leads models to output high-confidence predictions even when facing out-of-distribution inputs.

**Key Challenge**: Effective predictive uncertainty requires decomposing the sources of epistemic uncertainty. However, the single temperature parameter $\tau$ in softmax cannot achieve instance-level multi-dimensional uncertainty representation—$\tau$ is a global hyperparameter, and differences between instances are determined solely by logit magnitude.

**Goal**: To design a new activation function that explicitly decomposes and integrates epistemic uncertainty signals from Similarity, Distance, and Magnitude, providing a more reliable foundation for selective classification.

**Key Insight**: Leveraging the capability of neural networks as implicit instance-based metric learners, a compact representation space is constructed using an exemplar adaptor (a 1-D CNN adaptor) on top of frozen pre-trained LM hidden states to extract Similarity and Distance signals.

**Core Idea**: Replace the fixed base $e$ of softmax with a data-driven base $(2+q)$ (dependent on Similarity) and the fixed temperature $\tau$ with an instance-level Distance $d$—enabling the activation function's output to directly encode epistemic uncertainty across three dimensions.

## Method

### Overall Architecture

The SDM system consists of three layers: (1) a frozen pre-trained LM providing hidden states $\mathbf{h}$; (2) an exemplar adaptor (1-D CNN + linear layer) that maps $\mathbf{h}$ to compact representations $\mathbf{h}'$ and new logits $\mathbf{z}'$; (3) an SDM activation layer that utilizes $\mathbf{h}'$ to calculate Similarity $q$ and Distance $d$, combining them with $\mathbf{z}'$ to output a calibrated probability distribution. On top of this, the SDM estimator identifies high-reliability regions for selective classification using data-driven empirical CDF partitioning.

### Key Designs

1. **Similarity ($q$) Calculation**:

    - **Function**: Quantify the degree of depth matching between the test instance and correct prediction patterns in the training set.
    - **Mechanism**: In the representation space $\mathbf{h}'$ of the exemplar adaptor, the training set is sorted by $L^2$ distance. The calculation counts the number of continuous matches starting from the nearest neighbor that satisfy: (a) the training sample's prediction matches the current instance's prediction ($\hat{y} = \hat{y}^{tr}_{(i)}$), (b) the training sample is correctly predicted ($\hat{y}^{tr}_{(i)} = y^{tr}_{(i)}$), and (c) the matches are continuous (no breaks allowed). $q \in \{0, \ldots, |D_{tr}|\}$, where $q=0$ indicates even the nearest neighbor fails, effectively signaling out-of-distribution.
    - **Design Motivation**: Unlike traditional KNN rules, SDM's Similarity utilizes both model predictions and ground-truth labels—if the nearest neighbors in the training set not only have the same labels but are also correctly predicted, it indicates the model has reliable discriminative power in that region.

2. **Distance ($d$) Calculation**:

    - **Function**: Quantify the normalized distance of a test instance to the training distribution.
    - **Mechanism**: First, the $L^2$ distance to the nearest training neighbor $d_{\text{nearest}}$ is computed. Then, it is normalized using the empirical CDF of each class in the calibration set $D_{ca}$: $d = \min[1 - \text{eCDF}^{y_1}_{ca}(d_{\text{nearest}}), \ldots, 1 - \text{eCDF}^{y_C}_{ca}(d_{\text{nearest}})]$. When $d_{\text{nearest}}$ exceeds the maximum distance observed in labelled data, $d=0$, and SDM outputs a uniform distribution, representing maximum uncertainty.
    - **Design Motivation**: Taking the minimum of the CDF across all classes ensures a conservative estimate—even if the distance seems normal for some classes, an anomalously large distance relative to any single class will trigger high uncertainty.

3. **SDM Activation and High-Reliability Region Estimation**:

    - **Function**: Integrate the three dimensions into a calibrated probability distribution and automatically identify high-reliability prediction regions.
    - **Mechanism**: The SDM activation is $\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$. The corresponding loss uses the change of base formula $\log_{(2+q)}$. High-reliability regions are determined by: first calculating a rescaled value $q' = \min(q, (2+q)^{\text{sdm}(\mathbf{z}')_{\hat{y}}})$, then incrementally increasing the $q'_{\min}$ threshold on the $q' > 0$ subset until the conformal thresholds $\psi_c$ for all classes reach the target confidence level $\alpha$ (e.g., 0.95). Predictions satisfying $q' \geq q'_{\min}$ and $\text{sdm}(\mathbf{z}')_{\hat{y}} \geq \psi_{\hat{y}}$ enter the high-reliability region.
    - **Design Motivation**: By progressively tightening the $q'$ threshold, the method finds regions that satisfy both class-conditional and prediction-conditional accuracy requirements, providing theoretically grounded selective classification. When no finite $q'_{\min}$ can be found, it indicates the model or data is insufficient to support reliable estimation.

### Loss & Training

The exemplar adaptor (1-D CNN + linear layer) is trained using the SDM loss while freezing the underlying LM parameters. The first training round is initialized with standard softmax ($q=e-2, d=1$), and $q$ and $d$ are recalculated in each subsequent round. The stopping criterion is the lowest class-balanced loss on the calibration set. The process is repeated $J=10$ times with random partitions and initializations to select the global optimum. The CNN uses $M=1000$ filters and is trained for 200 epochs per round.

## Key Experimental Results

### Main Results

**Sentiment Classification (ID) Selective Classification Performance ($\alpha=0.95$)**

| Model + Estimator | Class-cond. y=0 | y=1 | Prediction-cond. $\hat{y}$=0 | $\hat{y}$=1 | Acceptance Rate |
|-----------|---------|-----|------------|------------|---------|
| phi3.5 softmax | 0.98 | 0.86 (<α) | 0.88 (<α) | 0.98 | 0.98 |
| phi3.5 tempScaling | 0.99 | 0.91 (<α) | 0.93 (<α) | 0.99 | 0.90 |
| phi3.5+sdm sdmHR | **1.00** | **0.99** | **0.99** | **1.00** | 0.68 |
| Mixtral8x7B softmax | 0.98 | 0.88 (<α) | 0.89 (<α) | 1.00 | 1.00 |
| Mixtral8x7B+sdm sdmHR | **0.99** | **0.98** | **0.99** | **0.98** | 0.74 |

**Sentiment OOD (Out-of-Distribution)**

| Model + Estimator | Class-cond. y=0 | y=1 | Acceptance Rate | Note |
|-----------|---------|-----|---------|------|
| phi3.5 softmax | 1.00 | 0.54 (<α) | 0.96 | Overconfident, high error |
| phi3.5 APS | 1.00 | 0.59 (<α) | 0.77 | Still below target |
| phi3.5+sdm sdmHR | **1.00** | **1.00** | **0.01** | Almost all OOD rejected |

### Ablation Study

| Component | Effect | Note |
|------|------|------|
| softmax (no adaptor) | Class-cond. accuracy below target | Lacks Similarity and Distance |
| softmax (with adaptor) | ID target met but OOD fails | Better representation but no distance awareness |
| softmax($d \cdot \mathbf{z}'$) | Excessive conservatism (low ID acceptance) | Uses Distance as temperature only, lacks Similarity |
| sdm$_\alpha$ (simple threshold) | Prediction-cond. targets met but class-cond. not guaranteed | Lacks high-reliability region constraints |
| **sdmHR (Full Estimator)** | **Both conditional dimensions meet targets** | Synergy of Similarity+Distance+Magnitude |

### Key Findings

- On in-distribution data, softmax/tempScaling/APS/RAPS estimators without adaptors generally exhibit overconfidence, with class-conditional accuracy falling below the target $\alpha=0.95$.
- The difference is even more dramatic on out-of-distribution data—the sdmHR estimator for phi3.5+sdm reduces the acceptance rate of SentimentOOD to approximately 1% (rejecting almost everything), while softmax still accepts 96% of OOD data with only 0.54 accuracy for the y=1 class.
- When Alg. 1 returns $q'_{\min} = \infty$, it provides a practical indicator that the model or data is insufficient for reliable estimation.
- On the Factcheck task, softmax and APS fail significantly in class-conditional accuracy on test sets with covariate shift, while sdmHR maintains reliability by appropriately tightening the acceptance range.

## Highlights & Insights

- The definition of Similarity is highly ingenious—it not only requires nearest neighbors to have the same label but also requires the model's predictions for those neighbors to be correct and continuous. This adds a "reliability of the model in this region" dimension compared to traditional KNN.
- The mathematical form of SDM is elegant—generalizing the base and temperature of softmax from fixed constants to data-driven instance-level variables. It precisely reverts to standard softmax when $q=e-2$ and $d=1$.
- The concept of high-reliability regions is directly valuable for multi-stage LLM pipelines—automating predictions that enter high-reliability zones while diverting others to more expensive tools or human review.

## Limitations & Future Work

- The exemplar adaptor requires maintaining the full training set for Similarity and Distance calculations, posing storage and retrieval efficiency challenges for large-scale datasets.
- Validation was limited to binary classification tasks (sentiment analysis, fact-checking); multi-class and more complex NLP tasks require further testing.
- Calculating $q$ requires traversing the training set for distance sorting; real-time inference latency needs optimization (possibly via approximate nearest neighbor search).
- It assumes that the exemplar adaptor can effectively learn discriminative representations on top of a frozen LM, which might not hold for all tasks.

## Related Work & Insights

- **vs Temperature Scaling**: Temperature scaling is a single-parameter global calibration, whereas SDM provides instance-level multi-dimensional calibration via $q$ and $d$, showing massive differences in OOD scenarios.
- **vs Conformal Prediction (APS/RAPS)**: Marginal coverage guarantees of conformal methods do not directly translate to selective classification (where set size = 1 is required). SDM provides class-conditional coverage through the specific construction of high-reliability regions.
- **vs VBLL**: Variational Bayesian Last Layer outperforms softmax/tempScaling on OOD but still lacks the extreme OOD robustness of SDM.
- **vs Exemplar-based methods**: SDM elevates exemplar matching from a post-hoc explanatory tool to a core component of the activation function itself.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Generalizing softmax base and temperature into data-driven variables for three-dimensional epistemic uncertainty decomposition is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic ID/OOD/Far-OOD comparisons and multi-estimator ablations, though the task range is somewhat narrow (binary classification only).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation, clear generalization path from softmax to SDM, and consistent notation.
- **Value**: ⭐⭐⭐⭐ Provides a theoretically stronger foundation for uncertainty quantification in LLM deployment, with wide application potential for the high-reliability region concept.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[ICLR 2026\] Conjuring Semantic Similarity](../../ICLR2026/interpretability/conjuring_semantic_similarity.md)
- [\[ACL 2026\] Embracing Anisotropy: Turning Massive Activations into Interpretable Control Knobs for Large Language Models](embracing_anisotropy_turning_massive_activations_into_interpretable_control_knob.md)
- [\[ICML 2026\] Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis](../../ICML2026/interpretability/disentangling_direction_and_magnitude_in_transformer_representations_a_double_di.md)
- [\[ICLR 2026\] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](../../ICLR2026/interpretability/lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)

</div>

<!-- RELATED:END -->

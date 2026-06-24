---
title: >-
  [Paper Note] Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation
description: >-
  [ICML2025][Causal Inference][Counterfactual Outcome Estimation] Proposes a Transformer-based spatial-temporal counterfactual outcomes estimation framework that utilizes CNNs to compute high-dimensional propensity scores and Transformers to estimate intensity functions, outperforming traditional causal inference methods on both synthetic and real-world data.
tags:
  - "ICML2025"
  - "Causal Inference"
  - "Counterfactual Outcome Estimation"
  - "Spatial-Temporal Data"
  - "Transformer"
  - "Inverse Probability Weighting"
  - "Spatial Point Process"
date: 2026-05-08
content_hash: 8e06f664efc9dd97
---

# Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation

**Conference**: ICML2025  
**arXiv**: [2506.21154](https://arxiv.org/abs/2506.21154)  
**Code**: Yes (links attached in paper)  
**Area**: Causal Inference  
**Keywords**: Causal Inference, Counterfactual Outcome Estimation, Spatial-Temporal Data, Transformer, Inverse Probability Weighting, Spatial Point Process

## TL;DR

Proposes a Transformer-based spatial-temporal counterfactual outcomes estimation framework that utilizes CNNs to compute high-dimensional propensity scores and Transformers to estimate intensity functions, outperforming traditional causal inference methods on both synthetic and real-world data.

## Background & Motivation

Real-world data naturally exhibit both spatial and temporal dimensions (e.g., the causal impact of conflict on forest loss), necessitating the estimation of counterfactual outcomes under spatial-temporal settings. However, existing causal inference frameworks have notable limitations:

- **Pearl/Rubin Frameworks**: Cannot directly handle counterfactual outcome estimation under spatial-temporal settings.
- **Christiansen et al. (2022)**: Extended structural causal models to spatial-temporal data, but current outcomes are primarily influenced by current treatments, failing to adequately address **temporal carryover effects**.
- **Papadogeorgou et al. (2022)**: Extended the potential outcomes framework to spatial-temporal settings, but lacked an explicit propensity score computation method for spatial-temporal attributes and relied on kernel methods, leading to generalization difficulties under complex data patterns.

Core Motivation: Development of a deep learning framework capable of simultaneously capturing spatial correlations and temporal carryover causal effects.

## Method

### Problem Setting

Spatial-temporal data is modeled as a **time series of spatial point patterns**. At each time step $t$, the treatment $Z_t(s)$ and outcome $Y_t(s)$ are binary variables indicating whether a treatment or outcome event occurs at location $s$. The overall structure can be viewed as a high-dimensional tensor.

### Spatial-Temporal Potential Outcomes Framework

- Historical treatment history: $Z_{\leq t} = \{Z_1, Z_2, \dots, Z_t\}$
- Potential outcome: $Y_t(Z_{\leq t})(s)$ represents the outcome at location $s$ and time $t$ under the historical treatment history $Z_{\leq t}$.
- Estimation target: The expected number of outcome events within region $\omega$ under a counterfactual treatment allocation policy $F_H$.

$$N_t^\omega(F_H) = \int_{Z^M} |S_{Y_t^{ob}(z_{\leq t}(F_H))} \cap \omega| \, dF_H(z_{[t-M+1,t]})$$

### Core Estimator (IPW)

Derivation of the estimator based on Inverse Probability Weighting (IPW):

$$\hat{Y}_t(F_H, s) = \prod_{j=t-M+1}^{t} \frac{p_{h_j}(z_j)}{e_j(z_j)} \cdot \lambda_{Y_t^{ob}(z_{\leq t})}(s)$$

where:

- $e_j(z_j)$: Propensity score, representing the conditional probability of treatment given historical information.
- $p_{h_j}(z_j)$: Treatment probability under the counterfactual intervention distribution.
- $\lambda_{Y_t^{ob}}(s)$: Intensity function of the observed outcomes.

### Propensity Score Computation (CNN)

High-dimensional treatments (with $2^{100}$ possible values) cannot be classified directly. Therefore, a **dimension reduction mapping** is introduced:

$$R(Z_t) = |\{Z_t(s); Z_t(s) = 1, s \in \Omega\}|$$

This maps the treatment into a count of treated locations (a scalar). Under the spatial Poisson point process assumption, $R(Z_t)|h_{\leq t-1} \sim \text{Poisson}(\lambda_1)$. A CNN is used to regress $\lambda_1$ (via MSE loss), and the propensity score is computed using the Poisson PMF:

$$e_t(R(z_t)) = \frac{\lambda_1^{R(z_t)}}{R(z_t)!} e^{-\lambda_1}$$

### Intensity Function Estimation (Transformer)

A Transformer is employed to model the intensity function $\lambda_{Y_t^{ob}}(s)$ of the outcome point process, with the training objective to maximize the likelihood:

$$\mathcal{L} = -\sum_{i=1}^{|S|} \ln(\text{net}(s_i)) + \int_\Omega \text{net}(s)\,ds - \text{KL}(q \| p)$$

- First term: Log-likelihood at the event locations.
- Second term: Intensity integral over the region (part of the Poisson likelihood).
- Third term: KL-divergence regularization (between the Transformer encoder's output distribution $q$ and a standard Gaussian prior $p$).

Rationale for choosing Transformer: Captures long-range, high-order dependencies, and exhibits superior computational efficiency compared to RNNs.

### Theoretical Guarantees

- **Proposition 1**: The propensity score is a balancing score.
- **Proposition 2**: The propensity score exhibits dimension reduction properties.
- **Proposition 3**: The estimator possesses **consistency** and **asymptotic normality**: $\sqrt{T}(\hat{N}_\omega(F_H) - N_\omega(F_H)) \xrightarrow{d} \mathcal{N}(0, v)$

## Key Experimental Results

### Synthetic Experiments

Three sets of synthetic data with varying time lengths (T=32, 48, 64) were generated, with each set run independently 20 times.

| Method | Estimation Capability |
|------|----------|
| **Ours (Transformer)** | Achieves the lowest RER across all T and c settings |
| MSMs | Moderately high RER |
| RMSNs | Moderately high RER |
| Causal Forest | Higher RER |
| LR | Highest RER |

Conclusion: The proposed method consistently outperforms all baselines across different time lengths and treatment intensities.

### Real-World Experiments (Colombian Conflict $\rightarrow$ Forest Loss, 2002-2022)

| M\c | c=3 | c=4 | c=5 | c=6 | c=7 |
|-----|-----|-----|-----|-----|-----|
| M=1 | 20.6±2.3 | 20.5±2.2 | 20.7±2.8 | 20.5±1.9 | 20.8±2.0 |
| M=3 | 21.5±1.4 | 21.6±2.4 | 22.3±1.9 | 23.0±1.3 | 23.3±1.9 |
| M=5 | 22.4±2.3 | 22.9±1.8 | 23.6±1.7 | 24.2±1.2 | 24.7±2.2 |
| M=7 | 24.7±1.3 | 23.6±1.7 | 26.7±1.2 | 27.2±1.5 | 28.0±2.1 |

Conclusion: Increased conflict duration (M) and intensity (c) both lead to heightened forest loss, which aligns with existing literature.

### Ablation Study

- **Transformer vs RNN**: The estimation capability of RNN is significantly weaker than that of the Transformer.
- **Relaxation of Poisson Assumption**: Incorporating a Gaussian kernel to violate the standard Poisson assumption yields no significant performance degradation, validating the robustness of the proposed method.

## Highlights & Insights

1. **Ingenious Dimension Reduction**: Maps high-dimensional point-pattern treatments to scalar counts, rendering the previously intractable propensity score computation feasible.
2. **Rigorous Theoretical Support**: The estimator is proven consistent and asymptotically normal, moving beyond simple empirical performance tuning.
3. **Synergistic CNN + Transformer Design**: CNNs extract local features from high-dimensional spatial-temporal data for propensity score regression, while the Transformer captures long-range dependencies for intensity function estimation.
4. **Strong Robustness**: Performance remains robust even when easing the Poisson assumption, demonstrating practical viability.
5. **Meaningful Real-World Application**: The causal analysis between the Colombian conflict and forest loss holds significant environmental science value.

## Limitations & Future Work

1. **Poisson Point Process Assumption**: Although ablation studies show acceptable performance under relaxation, highly complex real-world data might violate this assumption (e.g., clustered point processes).
2. **Binary Treatment Only**: Treatment at each location is restricted to binary values (0/1), without generalization to continuous or multi-valued treatments.
3. **Unconfoundedness Assumption**: Assumes no unobserved confounding, which is difficult to guarantee in observational spatial-temporal data.
4. **Lack of Ground Truth in Real-World Experiments**: Validation is indirectly performed via consistency with existing literature, lacking quantitative evaluation.
5. **Scalability**: The paper tests up to dimensions of (100, 100, 192), which may present computation bottlenecks for larger-scale remote sensing or urban datasets.
6. **Single Application Scenario**: Validated only on conflict-forest data, lacking cross-domain generalization experiments.

## Rating

- Novelty: ⭐⭐⭐⭐ — For the first time, embeds a Transformer into a spatial-temporal causal inference framework, leveraging an ingenious dimension reduction strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across synthetic, real-world, and ablation studies, though real-world experiments lack quantitative ground truth.
- Writing Quality: ⭐⭐⭐⭐ — Clear symbol definitions and rigorous theoretical derivations, despite high formula density.
- Value: ⭐⭐⭐⭐ — Bridges the gap in deep learning-based spatial-temporal causal inference, offering a highly generalized framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Doubly Protected Estimation for Survival Outcomes Utilizing External Controls for Randomized Clinical Trials](doubly_protected_estimation_for_survival_outcomes_utilizing_external_controls_fo.md)
- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](../../ACL2025/causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [\[NeurIPS 2025\] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](../../NeurIPS2025/causal_inference/causality-induced_positional_encoding_for_transformer-based_representation_learn.md)
- [\[ICLR 2026\] An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](../../ICLR2026/causal_inference/an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)

</div>

<!-- RELATED:END -->

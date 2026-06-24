---
title: >-
  [Paper Note] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains
description: >-
  [ICML2025][Causal Inference][Evolving Domain Generalization] This paper proposes a time-aware structural causal model (time-aware SCM) and the SYNC method. By simultaneously learning static and dynamic causal representations and modeling causal mechanism drift, it effectively eliminates spurious correlations in evolving domain generalization (EDG) tasks, achieving superior temporal generalization performance.
tags:
  - "ICML2025"
  - "Causal Inference"
  - "Evolving Domain Generalization"
  - "Causal Representation Learning"
  - "Time-Aware Structural Causal Model"
  - "Variational Autoencoder"
  - "Mutual Information"
  - "Spurious Correlation"
date: 2026-05-08
content_hash: b0f097a452f1cb92
---

# Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains

**Conference**: ICML2025  
**arXiv**: [2506.17718](https://arxiv.org/abs/2506.17718)  
**Code**: GitHub (labeled with GitHub link in the paper, exact URL to be confirmed)  
**Area**: Causal Inference  
**Keywords**: Evolving Domain Generalization, Causal Representation Learning, Time-Aware Structural Causal Model, Variational Autoencoder, Mutual Information, Spurious Correlation

## TL;DR

This paper proposes a time-aware structural causal model (time-aware SCM) and the SYNC method. By simultaneously learning static and dynamic causal representations and modeling causal mechanism drift, it effectively eliminates spurious correlations in evolving domain generalization (EDG) tasks, achieving superior temporal generalization performance.

## Background & Motivation

**Evolving Domain Generalization (EDG)** aims to learn from sequentially varying source domains over time, enabling the model to generalize to future unseen domains. Unlike traditional Domain Generalization (DG) which assumes no temporal relations between domains, EDG requires capturing the patterns of data distribution evolving over time.

The core issue with existing EDG methods (e.g., LSSAE, SDE-EDG) is: **they only model the statistical correlation between data and labels, making them susceptible to interference from spurious correlations**. The paper illustrates this using the Caltran traffic surveillance dataset: most daytime images contain vehicles while most nighttime images do not, leading the model to learn a spurious shortcut of "lighting $\rightarrow$ presence of vehicle" instead of focusing on the causal features of the vehicles themselves.

Traditional causal methods face two major challenges in dynamic scenarios:

**Causal factors change over time**: Static SCMs cannot model time-varying category-related features.

**Causal mechanism drift**: The mapping from causal factors to labels $P(Y|Z_c)$ is no longer invariant across domains.

## Method

### Overall Architecture: SYNC (Static-DYNamic Causal Representation Learning)

SYNC is based on a newly designed time-aware SCM. It simultaneously learns static and dynamic causal representations within a sequential VAE framework and decouples causal factors through information-theoretical objectives.

### Time-Aware Structural Causal Model (Time-aware SCM)

The SCM of traditional DG regards both the causal factor $Z_c$ and the spurious factor $Z_s$ as time-invariant. This paper proposes a more fine-grained decomposition:

- **Static causal factors** $Z_c^{st}$: Class-related information that is stable across domains (e.g., shape features of vehicles).
- **Dynamic causal factors** $Z_c^{dy}$: Class-related information that varies across domains (e.g., differences in vehicle appearance across different periods).
- **Static spurious factors** $Z_s^{st}$ and **dynamic spurious factors** $Z_s^{dy}$.
- **Drift factors** $Z^d$: Modeling the evolution of the causal mechanism itself.

The data generation process is $X \leftarrow (Z^{st}, Z^{dy})$, and the label generation process is $Y \leftarrow (Z_c^{st}, Z_c^{dy}, Z^d)$. Global variables $G$ and local variables $L$ generate static and dynamic factors respectively, while $L$ also serves as a temporal confounder introducing spurious correlations through backdoor paths.

### Evolution Pattern Learning

A sequential VAE framework is employed to model static and dynamic representations respectively:

- Static encoder $q_\psi(z_t^{st}|x_t) = \mathcal{N}(\mu(z_t^{st}), \sigma^2(z_t^{st}))$ with prior $p(z_t^{st}) = \mathcal{N}(0, I)$.
- Dynamic encoder $q_\theta(z_t^{dy}|z_{<t}^{dy}, x_t)$, with prior $p(z_t^{dy}|z_{<t}^{dy})$ modeled via LSTM to capture temporal dependencies.

The evolution pattern loss $\mathcal{L}_{fp}$ consists of a reconstruction term and two KL divergence terms:

$$\mathcal{L}_{fp} = -\sum_{t=1}^{T} \mathbb{E}[\log p(x_t|z_t^{st}, z_t^{dy})] + \sum_{t=1}^{T} D_{KL}(q_\psi \| p(z_t^{st})) + \sum_{t=1}^{T} D_{KL}(q_\theta \| p(z_t^{dy}|z_{<t}^{dy}))$$

### Static-Dynamic Decoupling

The mutual information between static and dynamic representations is minimized: $\mathcal{L}_{MI} = \sum_{t=1}^{T} I(z_t^{st}; z_t^{dy})$, with entropy terms estimated using mini-batch weighted sampling (MWS).

### Causal Representation Mining

The core theoretical foundation is Proposition 1: Under reasonable entropy inequality conditions, intra-class causal factors have higher conditional mutual information than spurious factors.

**Static Causal Representation**: Minimizes the domain distance by maximizing $I(\Phi_c^{st}(X_t); \Phi_c^{st}(X_{t-1})|Y)$ via cross-domain contrastive learning (Eq. 6), bringing static causal representations of the same class across adjacent domains closer. A Gumbel-Softmax is used to generate a 0-1 mask for selecting causal dimensions (with a mask ratio of $\kappa$).

**Dynamic Causal Representation**: Using the learned static causal factors as anchors, the intra-domain contrastive learning (Eq. 9) maximizes $I(\Phi_c^{dy}(X_t); Z_{c,t}^{st}|Y)$, pulling dynamic and static causal representations of the same class within the same domain closer.

### Causal Mechanism Drift Modeling

A drift factor $Z^d$ is introduced and learned via an RNN encoder $q_\zeta(z_t^d|z_{<t}^d, y_t)$, outputting a categorical distribution. The classification loss $\mathcal{L}_{mp}$ jointly predicts labels based on $(z_{c,t}^{dy}, z_{c,t}^{st}, z_t^d)$.

### Total Loss

$$\mathcal{L}_{SYNC} = \mathcal{L}_{evolve} + \alpha_1 \mathcal{L}_{MI} + \alpha_2 \mathcal{L}_{causal}$$

where $\mathcal{L}_{evolve} = \mathcal{L}_{fp} + \mathcal{L}_{mp}$, and $\mathcal{L}_{causal} = \mathcal{L}_{stc} + \mathcal{L}_{dyc}$.

### Theoretical Guarantees

- **Theorem 1**: Optimizing $\mathcal{L}_{evolve}$ can recover the joint distribution of training domains $p(x_{1:T}, y_{1:T})$.
- **Theorem 2**: Optimizing $\mathcal{L}_{SYNC}$ yields the optimal causal predictor in each temporal domain; namely, the predictor based on $(Z_{c,t}, Z_t^d)$ satisfies the optimality conditions of Definition 2.

## Key Experimental Results

### Datasets

2 synthetic datasets (Circle with 30 domains, Sine with 24 domains) + 5 real-world datasets (RMNIST with 19 domains, Portraits with 34 domains, Caltran with 34 domains, PowerSupply with 30 domains, and ONP with 24 domains), split as 1/2 : 1/6 : 1/3 for source/validation/target domains.

### Main Results (Table 1, Overall Wst/Avg)

| Method | Overall Wst | Overall Avg |
|------|-----------|-----------|
| ERM | 51.9 | 63.9 |
| MMD-LSAE (EDG SOTA-) | 58.1 | 70.9 |
| SDE-EDG (EDG SOTA-) | 55.4 | 71.9 |
| iDAG (Best Causal DG) | 56.1 | 63.7 |
| **SYNC (Ours)** | **63.4** | **73.1** |

- SYNC achieves the best overall performance across all 7 datasets.
- Compared to the best causal DG method (iDAG): Wst +7.3%, Avg +9.4%.
- Compared to the best EDG methods (MMD-LSAE/SDE-EDG): Wst +5.3%, Avg +1.2%.
- The Circle dataset shows the most significant improvement: Wst 67.0% vs 54.0% (MMD-LSAE), Avg 84.7% vs 81.5% (SDE-EDG).

### Ablation Study (Table 2, RMNIST)

| Variant | Components | Wst | Avg |
|------|------|-----|-----|
| A (Baseline) | Only $\mathcal{L}_{evolve}$ | 40.5 | 44.1 |
| B | + $\mathcal{L}_{MI}$ Decoupling | 41.9 | 45.7 |
| C | + Static Causal $Z_c^{st}$ | 44.1 | 48.7 |
| D | + Dynamic Causal $Z_c^{dy}$ | 42.9 | 49.2 |
| **SYNC** | **All** | **45.8** | **50.8** |

**Key Findings**:

1. Decoupling (B vs A) yields consistent improvements (+1.5%), showing the value of segregating static/dynamic information.
2. Static causal representations (C) contribute more to Wst (+2.2%), ensuring long-term stable generalization.
3. Dynamic causal representations (D) contribute more to Avg (+3.5%), adapting to current distribution changes.
4. Mutual complementarity: The joint learning of both in SYNC achieves the optimal performance.

### Further Analysis

- **Decoupling Visualization**: The static-dynamic mutual information of SYNC drops faster and more stably (compared to LSSAE).
- **Temporal Robustness**: SYNC maintains high accuracy in the domains at later stages of the timeline, whereas SDE-EDG drops in performance in the later periods of Circle.
- **Decision Boundary Visualization**: The decision boundaries of SYNC are closest to the ground truth.

## Highlights & Insights

1. **Causal perspective on the EDG problem**: Introduces causal representation learning into evolving domain generalization for the first time, designing a fine-grained time-aware SCM that distinguishes static/dynamic causal factors from spurious factors.
2. **Complementary dual causal representations**: Static causal factors guarantee long-term stability (improving worst-case scenarios), while dynamic causal factors adapt to time-varying distributions (improving the average case), making them mutually complementary.
3. **Theoretical guarantees**: Proves that the proposed method can yield the optimal causal predictor for each temporal domain (Theorem 2).
4. **Elegant causal factor extraction**: Utilizes a combination of Gumbel-Softmax masking and contrastive learning, circumventing the challenges associated with traditional causal interventions.

## Limitations & Future Work

1. **Number of domains assumption**: Requires a sufficient number of sequential source domains to learn evolution patterns; its performance remains unclear in few-domain scenarios.
2. **Linear temporal assumption**: The prior modeled by LSTM implicitly assumes that "recent domains are more relevant," leaving its applicability to non-monotonic or periodic distribution drifts questionable.
3. **Sensitivity of the mask ratio $\kappa$**: The ratio for selecting causal dimensions requires hyperparameter tuning, which is not fully discussed in terms of sensitivity.
4. **Computational overhead**: Joint optimization involving VAE reconstruction, contrastive learning, and mutual information estimation leads to high training costs.
5. **Validated only on classification tasks**: Not yet extended to handling regression, detection, or other task types.
6. **Proposition 1 assumptions**: Although the authors claim that the "conditions are easily satisfied," this assertion lacks empirical support.

## Related Work & Insights

- **EDG baselines**: LSSAE/MMD-LSAE (Qin et al., 2022/2023) are representative methods based on sequential VAEs. SYNC incorporates causal learning components on top of these.
- **SDE-EDG** (Zeng et al., 2023a): Models continuous evolution trajectories using stochastic differential equations, achieving strong average performance but exhibiting instability in the worst-case scenario.
- **Causal DG**: Methods like IRM and IIB focus on learning invariance but ignore dynamic causal information, limiting their effectiveness in non-stationary environments.
- **DRAIN** (Bai et al., 2023): Employs a Bayesian framework + dynamic graph generation for network parameters, offering a complementary strategy to this work.
- **Insights**: The concept of decoupling causal representation learning from evolution pattern learning can be extended to other temporal distribution shift scenarios (e.g., continual learning, online learning).

## Rating

- Novelty: ⭐⭐⭐⭐ — The design of the time-aware SCM is elegant, and the static-dynamic causal decomposition is a meaningful novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 7 datasets + 20 baselines + complete ablation studies + visualization analysis, though it lacks verification on large-scale datasets.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured with rigorous theoretical derivations, supported by effective illustrative figures.
- Value: ⭐⭐⭐⭐ — Introduces a causal perspective to the EDG field, offering solid methodological contributions, though its practicality requires validation across more scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Integrating Markov Blanket Discovery into Causal Representation Learning for Domain Generalization](../../ECCV2024/causal_inference/integrating_markov_blanket_discovery_into_causal_representation_learning_for_dom.md)
- [\[ICLR 2026\] CARL: Preserving Causal Structure in Representation Learning](../../ICLR2026/causal_inference/carl_preserving_causal_structure_in_representation_learning.md)
- [\[ICML 2025\] Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes](classifier_reconstruction_through_counterfactual-aware_wasserstein_prototypes.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [\[ICML 2025\] E-LDA: Toward Interpretable LDA Topic Models with Strong Guarantees in Logarithmic Parallel Time](e-lda_toward_interpretable_lda_topic_models_with_strong_guarantees_in_logarithmi.md)

</div>

<!-- RELATED:END -->

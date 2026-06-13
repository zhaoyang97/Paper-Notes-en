---
title: >-
  [Paper Note] Rao-Blackwellised Reparameterisation Gradients
description: >-
  [NeurIPS 2025][Optimization][Rao-Blackwell] This paper proposes the R2-G2 estimator as a Rao-Blackwellised variant of reparameterisation gradients…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Rao-Blackwell"
  - "reparameterisation"
  - "gradient estimation"
  - "variational inference"
  - "Bayesian MLP"
date: 2026-05-08
content_hash: 010184ec804380bd
---

# Rao-Blackwellised Reparameterisation Gradients

**Conference**: NeurIPS 2025

**arXiv**: [2506.07687](https://arxiv.org/abs/2506.07687)

**Code**: None

**Area**: Optimization / Probabilistic Machine Learning

**Keywords**: Rao-Blackwell, reparameterisation, gradient estimation, variational inference, Bayesian MLP

## TL;DR

This paper proposes the R2-G2 estimator as a Rao-Blackwellised variant of reparameterisation gradients, proves that local reparameterisation in Bayesian MLPs is a special case thereof, and extends the low-variance gradient advantage to a broad class of probabilistic models.

## Background & Motivation

### Limitations of Prior Work

**Background**: Latent Gaussian variables are widely used in probabilistic machine learning, and gradient estimators are central to gradient-based optimisation. The reparameterisation trick has become the default choice for variational inference due to its simplicity and low variance.

Core challenges:

**Variance remains improvable**: Although reparameterisation gradients outperform REINFORCE, their variance remains high in complex models.

**Rao-Blackwellisation opportunity**: Leveraging conditional expectations to reduce variance is a classical statistical idea, yet how to systematically apply it to reparameterisation gradients remains unclear.

**Insufficient theoretical understanding of local reparameterisation**: Local reparameterisation (Kingma et al.) performs well in Bayesian MLPs, but its connection to Rao-Blackwellisation has not been established.

## Method

### Overall Architecture

R2-G2 (Rao-Blackwellised Reparameterisation Gradient) systematically reduces the variance of gradient estimates by taking conditional expectations over a subset of the random variables appearing in the reparameterisation gradient.

### Key Designs

**1. Decomposition of the Reparameterisation Gradient**

For the variational objective $\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z)}[f(z)]$, the reparameterisation gradient is:
$$\nabla_\phi \mathcal{L} = \mathbb{E}_{\epsilon \sim p(\epsilon)}[\nabla_\phi f(T_\phi(\epsilon))]$$

When the model contains multiple latent variables $z = (z_1, z_2, \ldots, z_L)$, conditional expectations can be taken over the noise variables of a subset of them.

**2. The R2-G2 Estimator**

For the gradient with respect to layer $l$, the conditional expectation is taken over noise variables from all other layers:
$$\hat{g}^{\text{R2-G2}}_l = \mathbb{E}_{\epsilon_{\backslash l}}[\nabla_{\phi_l} f(T_\phi(\epsilon)) | \epsilon_l]$$

Key result: The Rao-Blackwell theorem guarantees $\text{Var}(\hat{g}^{\text{R2-G2}}) \leq \text{Var}(\hat{g}^{\text{reparam}})$.

**3. Unification with Local Reparameterisation**

- Proof: For Bayesian MLPs, the local reparameterisation gradient is precisely a special case of R2-G2.
- This connection is established for the first time, unifying two seemingly distinct methods.
- Corollary: The advantages of local reparameterisation can be extended to non-MLP architectures.

### Loss & Training

- ELBO objective: $\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z) || p(z))$
- R2-G2 is used to estimate the gradient of the ELBO.
- Applying R2-G2 in the early training phase yields faster convergence; the standard reparameterisation estimator may be used thereafter.

## Key Experimental Results

### Main Results

Variational inference in Bayesian Neural Networks (test log-likelihood):

| Method | Boston Housing | Concrete | Wine Quality | Protein |
|--------|---------------|----------|-------------|---------|
| Standard Reparam. | -2.72 | -3.18 | -0.98 | -2.85 |
| Local Reparam. | -2.65 | -3.09 | -0.94 | -2.79 |
| R2-G2 (Ours) | **-2.62** | **-3.05** | **-0.92** | **-2.76** |

VAE training ELBO (MNIST, intermediate latent layer):

| Method | ELBO (final) | Gradient Variance (log) | Convergence Epochs |
|--------|-------------|------------------------|--------------------|
| Standard Reparam. | -86.5 | -3.2 | 150 |
| R2-G2 (Ours) | **-85.8** | **-4.5** | **110** |

### Ablation Study

Variance reduction ratio of R2-G2 across different model depths:

| Model Depth (layers) | Gradient Variance Reduction | ELBO Gain |
|---------------------|-----------------------------|-----------|
| 2 | 1.5× | +0.2 |
| 4 | 3.2× | +0.5 |
| 8 | 7.8× | +0.9 |
| 16 | 15.1× | +1.4 |

### Key Findings

1. The variance reduction of R2-G2 **grows linearly** with model depth — deeper models benefit more.
2. The largest gains are observed in the early training phase, when gradient variance has the greatest impact on optimisation.
3. R2-G2's advantage is particularly pronounced in multi-layer probabilistic models such as deep VAEs.
4. Local reparameterisation = R2-G2 specialised to Bayesian MLPs, providing a new perspective for understanding the former.

## Highlights & Insights

- **Theoretical elegance**: Rao-Blackwellisation is the most natural approach to variance reduction and provides guaranteed improvement.
- **Unification of existing methods**: Local reparameterisation is subsumed into a unified framework.
- **Scalability**: The method generalises to any probabilistic model employing the reparameterisation trick.

## Limitations & Future Work

1. Analytic computation of the conditional expectation is only tractable for specific distribution families such as Gaussians.
2. In certain model architectures, the additional computational cost of R2-G2 may offset the benefits of variance reduction.
3. Validation on large-scale deep models (e.g., GPT-scale Bayesian training) is lacking.
4. The combined effect of R2-G2 with other variance reduction techniques (e.g., control variates) remains unexplored.

## Related Work & Insights

- **Reparameterisation trick** (Kingma & Welling, 2014): The foundational technique underlying VAEs.
- **Local reparameterisation** (Kingma et al., 2015): Efficient gradient estimation for Bayesian MLPs.
- **Rao-Blackwell theorem**: A fundamental tool for variance reduction in statistics.

## Rating

- ⭐ Novelty: 7/10 — The theoretical connection is elegant, but the idea of Rao-Blackwellisation itself has precedent.
- ⭐ Practicality: 7/10 — Valuable for practitioners working with probabilistic models, though the scope of applicability is relatively narrow.
- ⭐ Writing Quality: 9/10 — Theoretical derivations are rigorous and the relationship to prior work is clearly articulated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)

</div>

<!-- RELATED:END -->

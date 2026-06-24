---
title: >-
  [Paper Note] H-SPLID: HSIC-based Saliency Preserving Latent Information Decomposition
description: >-
  [NeurIPS 2025][Interpretability][Saliency feature learning] This paper proposes H-SPLID, which explicitly decomposes the latent space into two subspaces: **salient (task-related)** and **non-salient (task-unrelated)**. Combined with HSIC regularization for information compression, the authors prove that the upper bound of prediction deviation is controlled by the dimension of the salient subspace and HSIC. This significantly improves robustness against perturbations in non-sa…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Saliency feature learning"
  - "Latent space decomposition"
  - "HSIC"
  - "Adversarial robustness"
  - "Dimensionality compression"
date: 2026-05-08
content_hash: f5308ecff3e873dc
---

# H-SPLID: HSIC-based Saliency Preserving Latent Information Decomposition

**Conference**: NeurIPS 2025  
**arXiv**: [2510.20627](https://arxiv.org/abs/2510.20627)  
**Code**: [GitHub](https://github.com/neu-spiral/H-SPLID)  
**Area**: Representation Learning / Robustness  
**Keywords**: Saliency feature learning, Latent space decomposition, HSIC, Adversarial robustness, Dimensionality compression

## TL;DR

This paper proposes H-SPLID, which explicitly decomposes the latent space into two subspaces: **salient (task-related)** and **non-salient (task-unrelated)**. Combined with HSIC regularization for information compression, the authors prove that the upper bound of prediction deviation is controlled by the dimension of the salient subspace and HSIC. This significantly improves robustness against perturbations in non-salient regions without requiring adversarial training.

## Background & Motivation

Although neural networks achieve high accuracy on classification tasks, they often rely on **task-unrelated redundant features** (such as image backgrounds), leading to:
- Extreme vulnerability to adversarial attacks in non-salient regions: In the double-digit diagnostic test, a PGD attack ($\epsilon=1.0$) on the right digit against a standard CNN causes accuracy to plummet from 96.86% to 31.76%.
- Redundant dimensions expanding the exploitable space for adversarial attacks.

Limitations of prior work:
- Adversarial training is computationally expensive and tailored to specific attacks.
- Information bottleneck methods (such as HBaR) compress within a single latent space and fail to explicitly separate salient and non-salient components.
- Contrastive analysis methods require additional background datasets.

## Method

### Saliency-Aware Latent Space Decomposition

The encoder $f_\psi: \mathcal{X} \to \mathbb{R}^m$ maps inputs to a latent representation $\mathbf{z}$, which is decomposed using a learnable diagonal mask $\mathbf{M}_s = \text{diag}\{\boldsymbol{\beta}\}$ ($\boldsymbol{\beta} \in \{0,1\}^m$) as:

$$\mathbf{z}_s = \boldsymbol{\beta} \odot \mathbf{z}, \quad \mathbf{z}_n = (\mathbf{1} - \boldsymbol{\beta}) \odot \mathbf{z}$$

Classification only utilizes the salient part: $\hat{\mathbf{y}} = \mathbf{W}^\top \mathbf{M}_s f_\psi(\mathbf{x})$.

### Regularization Design

**Clustering loss** facilitates structured separation:
$$\mathcal{L}_s = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{M}_s(\mathbf{z}_i - \boldsymbol{\mu}_k)\|^2, \quad \mathcal{L}_n = \sum_{i=1}^n \|\mathbf{M}_n(\mathbf{z}_i - \boldsymbol{\mu})\|^2$$

$\mathcal{L}_s$ clusters instances of the same class in the salient subspace, while $\mathcal{L}_n$ aligns all samples together in the non-salient subspace.

**HSIC Regularization**:
$$\rho_s \widehat{\text{HSIC}}(\mathbf{X}, \mathbf{Z}_s) + \rho_n \widehat{\text{HSIC}}(\mathbf{Y}, \mathbf{Z}_n)$$

- First term: Compresses the statistical dependence between the salient subspace and the input (removing redundancy).
- Second term: Eliminates the dependence between the non-salient subspace and labels (preventing label information leakage).

**Total Objective**:
$$\mathcal{L} = \lambda_{ce}\mathcal{L}_{ce} + \lambda_s \mathcal{L}_s + \lambda_n \mathcal{L}_n + \rho_s \text{HSIC}(\mathbf{X}, \mathbf{Z}_s) + \rho_n \text{HSIC}(\mathbf{Y}, \mathbf{Z}_n)$$

### Alternating Optimization

- **Optimize network with fixed mask**: Standard SGD
- **Optimize mask with fixed network**: Closed-form solution

$$\beta_i^* = \frac{\lambda_n \sum_{\mathbf{z}}(\mathbf{z}_i - \mu_i)^2}{\lambda_s \sum_k \sum_{\mathbf{z} \in C_k}(\mathbf{z}_i - (\mu_k)_i)^2 + \lambda_n \sum_{\mathbf{z}}(\mathbf{z}_i - \mu_i)^2}$$

Intuition: Dimensions with small intra-class variance are assigned as salient dimensions ($\beta \to 1$).

### Theoretical Guarantee

**Robustness Bound** (Theorem 3.2): For bounded perturbation $\|\delta(\mathbf{x})\|_2 \leq r$:

$$\mathbb{E}_\mathbf{x}\left[\|h_\theta(\mathbf{x}+\delta(\mathbf{x})) - h_\theta(\mathbf{x})\|_2\right] \leq \frac{rRB\sqrt{ks}(LR+\|f_\psi(0)\|_2)}{\sigma^2 K_\mathcal{F} K_\mathcal{G}} \cdot \text{HSIC}(\mathbf{x}, \mathbf{z}_s) + o(r)$$

where $s = \|\mathbf{M}_s\|_0$ is the number of salient dimensions. **Simultaneously reducing HSIC and decreasing $s$ tightens the upper bound.**

## Key Experimental Results

### COCO Background Attack (ResNet-18)

| Method | No Attack | Block PGD $\frac{25}{255}$ | Background PGD $\frac{2}{255}$ | Global PGD $\frac{2}{255}$ |
|------|--------|---------------------------|-------------------------|------------------------|
| Vanilla | 98.1 | 56.3 | 56.6 | 34.2 |
| WD | 94.3 | 43.9 | 59.9 | 40.7 |
| GLA | 97.1 | 60.4 | 57.4 | 37.3 |
| HBaR | — | — | — | — |
| **H-SPLID** | **97.8** | **82.5** | **70.7** | **47.6** |

H-SPLID outperforms Vanilla by **26 percentage points** under background block attacks.

### C-MNIST Diagnostic Test

| Method | Clean Accuracy | PGD Attack on Right Digit ($\epsilon=1.0$) |
|------|-----------|-------------------------------|
| Vanilla (CE) | 96.86 | 31.76 |
| **H-SPLID** | 97.14 | **87.46** |

### ImageNet-9 Transfer Learning (ResNet-50)

| Method | Original | MixedRand | Only-FG |
|------|----------|-----------|---------|
| Vanilla | 94.92 | 73.93 | 89.70 |
| HBaR | 95.03 | 74.12 | 89.76 |
| **H-SPLID** | **95.24** | **75.63** | **90.39** |

### ISIC-2017 Medical Imaging

Under real-world corruptions such as brightness, defocus blur, and snow/occlusion, H-SPLID consistently outperforms all baselines.

## Highlights & Insights

- **No adversarial training or saliency annotations required**: Robustness is achieved purely through latent space decomposition and statistical regularization.
- **Closed-form mask updates**: Avoids the exponential complexity of binary optimization.
- **Clear theoretical connection**: Establishes a rigorous link between HSIC + dimension compression and the upper bound of robustness.
- **Multi-class extension**: Extends the binary classification theory of Wang et al. to an arbitrary number of classes $k$.

## Limitations & Future Work

1. The computational complexity of the empirical HSIC estimator is $O(n^2)$ (due to the kernel matrix), which may become a bottleneck for large batch sizes.
2. The threshold for salient/non-salient dimensions is fixed (0.5); an adaptive threshold has not been explored.
3. Robustness improvement under global attacks is limited (since the salient region itself is also perturbed), which is an inherent limitation of this method.
4. Experiments are mainly conducted on the ResNet family; modern architectures like ViTs have not been verified.
5. The reconstruction capability of the non-salient subspace is not exploited (autoencoder extensions could be considered).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of explicit latent space decomposition and HSIC information compression is highly novel.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rigorous theoretical upper bounds, making solid contributions to both multi-class generalization and volume bounds.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across 5 datasets covering synthetic, natural, and medical scenarios with diverse attack types.
- **Practicality**: ⭐⭐⭐⭐ — Plug-and-play for classification networks, though the computational overhead of HSIC limits large-scale application.
- **Overall**: ⭐⭐⭐⭐

## Related Work & Insights

## Insights & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions](partial_information_decomposition_via_normalizing_flows_in_latent_gaussian_distr.md)
- [\[NeurIPS 2025\] Deep Modularity Networks with Diversity-Preserving Regularization](deep_modularity_networks_with_diversity-preserving_regularization.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)
- [\[NeurIPS 2025\] Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals](knowing_when_to_stop_efficient_context_processing_via_latent_sufficiency_signals.md)

</div>

<!-- RELATED:END -->

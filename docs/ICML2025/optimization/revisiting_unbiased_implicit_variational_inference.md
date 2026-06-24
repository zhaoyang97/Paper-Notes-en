---
title: >-
  [Paper Note] Revisiting Unbiased Implicit Variational Inference
description: >-
  [ICML2025][Optimization][Semi-Implicit Variational Inference] Revisiting the "impractical" Unbiased Implicit Variational Inference (UIVI) by replacing its inner MCMC loop with importance sampling, and unbiasedly learning the optimal proposal distribution by minimizing the expected forward KL divergence, which achieves or surpasses the SOTA on standard SIVI benchmarks.
tags:
  - "ICML2025"
  - "Optimization"
  - "Semi-Implicit Variational Inference"
  - "Importance Sampling"
  - "Conditional Normalizing Flow"
  - "Path Gradient Estimator"
  - "Score Gradient"
date: 2026-05-08
content_hash: a34680ea0badb7f6
---

# Revisiting Unbiased Implicit Variational Inference

**Conference**: ICML2025  
**arXiv**: [2506.03839](https://arxiv.org/abs/2506.03839)  
**Code**: To be confirmed  
**Area**: Variational Inference / Optimization  
**Keywords**: Semi-Implicit Variational Inference, Importance Sampling, Conditional Normalizing Flow, Path Gradient Estimator, Score Gradient

## TL;DR

Revisiting the "impractical" Unbiased Implicit Variational Inference (UIVI) by replacing its inner MCMC loop with importance sampling, and unbiasedly learning the optimal proposal distribution by minimizing the expected forward KL divergence, which achieves or surpasses the SOTA on standard SIVI benchmarks.

## Background & Motivation

### Variational Inference and Semi-Implicit Distributions

Variational inference (VI) performs inference by finding the closest approximate distribution $q_z^*$ to the target distribution $p_z$ within a family of distributions $\mathcal{Q}_z$. **Semi-implicit variational inference** (SIVI) provides a compromise: by sampling parameters $y$ from an implicit distribution $q_y$ and then sampling from an explicit conditional distribution $q_{z|y}$, it constructs a semi-implicit distribution whose expressiveness is close to implicit distributions but has an estimable density:

$$q_z(z) = \mathbb{E}_{\epsilon \sim p_\epsilon}[q_{z|\epsilon}(z|\epsilon)]$$

where $\epsilon$ is a latent variable mapped to the parameters of the distribution via a neural network $f_\phi$.

### Limitations of UIVI

UIVI proposed by Titsias & Ruiz (2019) proved a key identity:

$$\mathbb{E}_{\epsilon \sim q_{\epsilon|z}}[\nabla_z \log q_{z|\epsilon}(z|\epsilon)] = \nabla_z \log q_z(z)$$

meaning that if one can sample from the intractable conditional distribution $q_{\epsilon|z}$, the score gradient $\nabla_z \log q_z(z)$ can be unbiasedly estimated. However, UIVI uses MCMC to sample from $q_{\epsilon|z}$. Since this distribution can be multimodal, extremely long Markov chains are required to break the initialization dependency, making this approach computationally unfeasible and leading to it being largely abandoned by the community.

### Path Gradient Estimator

The key observation of this paper is that, by leveraging the reparameterization property of semi-implicit distributions, one can use the **path gradient estimator** to reduce the variance of gradient estimation and significantly cut computation overhead:

$$\nabla_\phi D_{\mathrm{KL}}(q_z \| p_z) = \mathbb{E}_{\epsilon,\eta}[\nabla_z(\log q_z(z) - \log p_z(z))\big|_{z=h_\phi(\epsilon,\eta)} \cdot \nabla_\phi h_\phi(\epsilon,\eta)]$$

Although this result has appeared in previous literature, its profound implications have not been fully discussed.

## Method

### Core Idea: Replacing MCMC with Importance Sampling

The authors propose replacing the MCMC loop in UIVI with importance sampling (IS) to estimate the score gradient:

$$\nabla_z \log q_z(z) = \nabla_z \log\left(\mathbb{E}_{\epsilon \sim \tau_{\epsilon|z}}\left[\frac{p_\epsilon(\epsilon) q_{z|\epsilon}(z|\epsilon)}{\tau_{\epsilon|z}(\epsilon|z)}\right]\right)$$

where $\tau_{\epsilon|z}$ is a proposal distribution modeled by a **conditional normalizing flow** (CNF).

### Key Theoretical Results

**Proposition 3.1**: When $\tau_{\epsilon|z} = q_{\epsilon|z}$, the IS estimator $s_{\mathrm{IS},k}$ becomes unbiased:

$$\mathbb{E}_{\epsilon_i \sim q_{\epsilon|z}} \nabla_z \log\left(\frac{1}{k}\sum_{i=1}^k \frac{p_\epsilon(\epsilon_i) q_{z|\epsilon}(z|\epsilon_i)}{q_{\epsilon|z}(\epsilon_i|z)}\right) = \nabla_z \log q_z(z)$$

**Proposition 3.2**: Minimizing the expected forward KL divergence $\mathbb{E}_{z \sim q_z}[D_{\mathrm{KL}}(q_{\epsilon|z} \| \tau_{\epsilon|z})]$ is equivalent to minimizing $D_{\mathrm{KL}}(q_{z,\epsilon} \| \tau_{\epsilon|z} \cdot q_z)$, where the global optimal solution is exactly $\tau_{\epsilon|z}^* = q_{\epsilon|z}$.

### Two Algorithms

1. **BSIVI** (Base SIVI): The baseline method, which uses a naive Monte Carlo estimator $s_{\mathrm{MC},k}$ to approximate the score gradient without using importance sampling. Despite uninformative $\epsilon_i$ contributing negligibly in high dimensions, this method performs surprisingly well.

2. **AISIVI** (Adaptively Informed SIVI): The main method, which alternates optimization:
    - Minimize the expected forward KL to train the CNF proposal distribution $\tau_{\epsilon|z}$.
    - Minimize the reverse KL $D_{\mathrm{KL}}(q_z \| p_z)$ to train the SIVI model.

   The training loss of the CNF can be simplified to the negative mean of the conditional log-likelihood: $\text{loss}_{\text{flow}} = -\frac{1}{m}\sum_{i=1}^m \log \tau_{\epsilon|z}(\epsilon_i | z_i)$

### Memory-Efficient Batch Aggregation

Thanks to path gradients, increasing the number of samples $k$ does not increase the backpropagation computation cost. The authors propose a batch aggregation scheme based on logaddexp, allowing the processing of an arbitrary number of $\epsilon_i$ samples with **constant memory**:

$$\ell_3(z,\tilde{z}) = \mathrm{logaddexp}(\ell_1(z,\tilde{z}) + \log j, \ell_2(z,\tilde{z})) - \log(j+1)$$

The aggregated score is estimated as a weighted combination of the two batch estimates, with weights computed in the log space to ensure numerical stability.

## Key Experimental Results

### Experiment 1: 2D Toy Distributions

| Distribution | AISIVI ($D_{\mathrm{KL}}$↓) | BSIVI ($D_{\mathrm{KL}}$↓) |
|------|------|------|
| Banana | **0.0853** | 0.3022 |
| Multimodal | 0.0044 | **0.0017** |
| X-shape | 0.0072 | **0.0034** |

AISIVI significantly outperforms BSIVI on the Banana distribution, while the performance on the other two is comparable.

### Experiment 2: Bayesian Logistic Regression (22-Dimensional)

On the WAVEFORM dataset, the marginal and pairwise density estimates of the four methods—AISIVI, BSIVI, KSIVI, and PVI—all agree well with the SGLD ground truth, without systematic overestimation or underestimation of variance. The scatter plot of pairwise correlation coefficients shows that all methods are comparable, with PVI and KSIVI being slightly tighter.

### Experiment 3: Conditional Diffusion Process (100-Dimensional)

| Method | Log ML↑ | Training Time (s) | Iterations |
|------|---------|-------------|---------|
| KSIVI | 74521 | 0.6k | 100k |
| **AISIVI** | **74062** | 1.4k | 10k |
| IWHVI | 67676 | 1.5k | 10k |
| BSIVI | 60556 | 1.5k | 10k |
| PVI | 53121 | 1.4k | 10k |
| UIVI | 40207 | 1.5k | 10k |

Under the same computational budget (10k iterations), AISIVI far outperforms IWHVI, BSIVI, PVI, and UIVI, approaching the gold standard performance of KSIVI (which requires 100k iterations).

## Highlights & Insights

- **Turning trash into treasure**: UIVI, once deemed "computationally impractical," is revived by simply replacing MCMC with importance sampling, which is an elegant theoretical correction.
- **Guaranteed unbiasedness**: When the proposal distribution $\tau = q_{\epsilon|z}$, the IS estimator is strictly unbiased; even if not exact, the estimator remains consistent as long as the support condition is met.
- **Forward KL training of CNF**: Training CNF using forward KL divergence is mass-covering, which naturally ensures the support condition and makes alternating optimization theoretically sound.
- **Constant memory training**: The batch aggregation scheme based on logaddexp allows arbitrary scaling of latent variable samples without increasing memory, which is highly practical.
- **Highly effective in high dimensions**: In the 100-dimensional conditional diffusion process, AISIVI matches KSIVI (100k iterations) with only 10k iterations.

## Limitations & Future Work

- CNF uses affine coupling layers, which are scalable but may limit expressiveness. The paper also mentions that replacing them with more flexible NF architectures could yield additional gains.
- In low-dimensional toy examples, AISIVI is not universally better than BSIVI (BSIVI is better on Multimodal and X-shape), indicating that the extra CNF may introduce unnecessary overhead in low dimensions.
- Experiments only cover up to 100 dimensions; performance in higher dimensions (e.g., posterior inference of large-scale deep learning models) remains to be verified.
- The training frequency ratio between CNF and the SIVI model in alternating optimization requires hyperparameter tuning, and the paper does not provide systematic guidance.
- The method is not compared against fully implicit VI methods (such as neural samplers) under the same framework.

## Related Work & Insights

- **SIVI Family**: SIVI (Yin & Zhou 2018) $\rightarrow$ UIVI (Titsias & Ruiz 2019) $\rightarrow$ KSIVI (Cheng et al. 2024) $\rightarrow$ PVI (Lim & Johansen 2024, Wasserstein gradient flow).
- **Path Gradient Estimator**: The low-variance gradient estimation idea of Roeder et al. (2017) blossoms with new value in the SIVI setting.
- **Importance Sampling for Improving VI**: IWAE (Burda et al. 2016), NVI (Zimmermann et al. 2021).
- **Inspiration**: This work shows that combining seemingly outdated methods with new tools can produce SOTA results; conditional normalizing flows as auxiliary inference networks is a pattern worthy of broader utilization.

## Rating
- Novelty: ⭐⭐⭐⭐ — Elegant theoretical insights, simple and powerful core idea (replacing MCMC with IS + training CNF with forward KL).
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 2D toy, 22D logistic regression, and 100D diffusion processes, with systematic comparison against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous and clear mathematical derivations, fluent writing, and self-contained proofs.
- Value: ⭐⭐⭐⭐ — Provides a strong new baseline (BSIVI) and a SOTA method (AISIVI) for the SIVI field, balancing both theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Synonymous Variational Inference for Perceptual Image Compression](synonymous_variational_inference_for_perceptual_image_compression.md)
- [\[NeurIPS 2025\] Least Squares Variational Inference](../../NeurIPS2025/optimization/least_squares_variational_inference.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](../../NeurIPS2025/optimization/viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](../../NeurIPS2025/optimization/neuro-symbolic_entity_alignment_via_variational_inference.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](../../NeurIPS2025/optimization/vera_variational_inference_framework_for_jailbreaking_large_language_models.md)

</div>

<!-- RELATED:END -->

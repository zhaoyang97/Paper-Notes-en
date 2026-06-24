---
title: >-
  [Paper Note] Flexible Tails for Normalizing Flows
description: >-
  [ICML 2025][Optimization][normalizing flows] This paper proposes Tail Transform Flow (TTF), which appends a non-Lipschitz transformation based on the complementary error function to the **final layer** of normalizing flows. This converts Gaussian tails to heavy tails with adjustable weights, avoiding the optimization difficulties in neural networks caused by using heavy-tailed base distributions.
tags:
  - "ICML 2025"
  - "Optimization"
  - "normalizing flows"
  - "heavy tails"
  - "extreme value theory"
  - "density estimation"
  - "variational inference"
date: 2026-05-08
content_hash: eecc22bb1e4024a9
---

# Flexible Tails for Normalizing Flows

**Conference**: ICML 2025  
**arXiv**: [2406.16971](https://arxiv.org/abs/2406.16971)  
**Code**: [GitHub](https://github.com/Tennessee-Wallaceh/tailnflows)  
**Area**: Optimization  
**Keywords**: normalizing flows, heavy tails, extreme value theory, density estimation, variational inference

## TL;DR

This paper proposes Tail Transform Flow (TTF), which appends a non-Lipschitz transformation based on the complementary error function to the **final layer** of normalizing flows. This converts Gaussian tails to heavy tails with adjustable weights, avoiding the optimization difficulties in neural networks caused by using heavy-tailed base distributions.

## Background & Motivation

### Importance of Heavy-Tailed Distribution Modeling

Heavy-tailed distributions are ubiquitous in fields such as climate science, infectious disease modeling, and financial risk. Accurately modeling the tail behavior of distributions is crucial in density estimation and variational inference. Extreme Value Theory (EVT) shows that the tail of a distribution can typically be approximated by the Generalized Pareto Distribution (GPD).

### Limitations of Standard NFs

Jaini et al. (2020) proved a key theorem stating that **a light-tailed random variable remains light-tailed after a Lipschitz transformation** (Theorem 2). Since most NFs employ a combination of a Gaussian base distribution and Lipschitz transformations, they are theoretically incapable of generating heavy-tailed outputs and thus fail to model heavy-tailed target distributions effectively.

### Limitations of Prior Work

The current mainstream approach is to **use heavy-tailed base distributions** (such as Student's t-distribution), including:

- **gTAF**: Jointly learns the degrees of freedom parameter of the Student's t-distribution.
- **mTAF**: Estimates the tail parameters first and then fixes them.
- **ATAF**: A similar method in the context of variational inference.

However, these methods suffer from a crucial **limitation**: the transformation layers of NFs typically consist of neural networks, and a heavy-tailed base distribution implies that the neural network must process extreme value inputs. Heavy-tailed inputs lead to heavy-tailed gradients, which severely degrade optimization convergence (Zhang et al., 2020). The authors empirically validate this issue.

### Key Insight

**The issue lies not in the base distribution, but in the location of the transformation**. Rather than allowing extreme values to flow through the entire neural network, it is better to introduce the heavy-tail transformation only at the very last layer. This way, the neural network only processes light-tailed (Gaussian) inputs, leading to more stable optimization.

## Method

### Overall Architecture

The sampling process of TTF is given by:

$$x = R \circ T_{\text{body}}(z), \quad z \sim \mathcal{N}(0, I)$$

where:
- $z$: standard Gaussian sample
- $T_{\text{body}}$: standard NF transformation (Lipschitz, e.g., RQS + affine layers)
- $R$: **TTF tail transformation layer** (non-Lipschitz, final layer)

The key difference is:
- **Prior Work**: Heavy-tailed base distribution $\to$ Neural network layers (extreme values as NN inputs)
- **Ours**: Gaussian base distribution $\to$ Neural network layers $\to$ Tail transformation (extreme values are generated only at the very end)

### Key Designs

The core equation is the 1D TTF transformation $R: \mathbb{R} \to \mathbb{R}$:

$$R(z; \lambda_+, \lambda_-) = \mu + \sigma \frac{s}{\lambda_s} \left[ \text{erfc}\left(\frac{|z|}{\sqrt{2}}\right)^{-\lambda_s} - 1 \right]$$

where:
- $s = \text{sign}(z)$, $\lambda_s = \lambda_+$ if $s=1$, and $\lambda_s = \lambda_-$ if $s=-1$
- $\text{erfc}$: the complementary error function (a standard special function directly supported by AD libraries)
- $\lambda_+, \lambda_- > 0$: control the **weights of the upper and lower tails** respectively, allowing for asymmetric tails
- $\mu \in \mathbb{R}, \sigma > 0$: location and scale parameters

**Theoretical Guarantee**: If $X$ has Gaussian tails, then $R(X)$ belongs to the Fréchet domain of attraction, with shape parameters controlled by $\lambda_+, \lambda_-$. In particular, a $\mathcal{N}(0,1)$ input yields an output with GPD tails having shape parameters exactly equal to $\lambda_+, \lambda_-$.

### Multivariate Extension

The 1D transformation is applied independently to each marginal dimension, where each dimension possesses its own parameters $(\mu_i, \sigma_i, \lambda_{+,i}, \lambda_{-,i})$. For dimensions known to be light-tailed, $\lambda$ can simply be fixed to a very small value ($1/1000$).

The authors also explored using an autoregressive structure to generate the $\lambda$ parameters dynamically, but found that optimization was more challenging and left this for future work.

### Loss & Training

**Density estimation** utilizes the negative log-likelihood:

$$\mathcal{J}_{\text{DE}}(\theta) = -\sum_{i=1}^N \log q_x(x_i; \theta)$$

**Variational inference** maximizes the ELBO:

$$\mathcal{J}_{\text{VI}}(\theta) = \mathbb{E}_{x \sim q_x}[\log \tilde{p}(x) - \log q_x(x; \theta)]$$

Density evaluation is computed via the change-of-variables formula $q_x(x) = q_z(T^{-1}(x)) |\det J_{T^{-1}}(x)|$, where both the inverse and the derivative of the TTF transformation can be solved analytically.

**Two-Stage Method (TTFfix)**:
1. **Stage 1**: Estimate the tail parameter $\lambda$ for each margin using EVT methods (the Hill double-bootstrap estimator) and then fix them.
2. **Stage 2**: Optimize $T_{\text{body}}$ together with the $\mu, \sigma$ parameters of $R$.

This is equivalent to first stripping the heavy-tailed features of the data using $R$, and then fitting the transformed data with standard NFs.

**Joint Training (TTF)**: Direct simultaneous optimization of all parameters (including $\lambda$), which requires careful initialization of the $\lambda$ parameters.

**Universality Guarantees**: The appendix proves that many universality results of NFs still hold after adding the TTF layer—specifically, they can represent heavy-tailed distributions with finite capacity, and do not lose expressivity with infinite capacity.

### Architecture Details

- Base architecture: isotropic Gaussian base distribution $\to$ autoregressive RQS layers $\to$ autoregressive affine layers.
- TTF appends the tail-transformation layer at the very end.
- In real-world data experiments, a trainable linear layer based on LU decomposition is also added.
- The CLIMDEX dataset uses a deeper architecture (alternating 5 layers of RQS + LU).

## Key Experimental Results

### Main Results

**Synthetic Data Density Estimation** ($d=50$, negative test log-likelihood per dimension, lower is better):

| Method | $\nu=0.5$ | $\nu=1$ | $\nu=2$ | Description |
|------|-----------|---------|---------|------|
| normal | Failed to converge | Failed to converge | 2.02 | Standard Gaussian base distribution |
| gTAF | 7.49 | 2.65 | 1.99 | Jointly learns t-distribution parameters |
| mTAF | 5.22 | 2.62 | 1.98 | Fixed t-distribution parameters |
| **TTF** | **3.68** | **2.54** | **1.98** | **Ours (jointly learned)** |
| **TTFfix** | **3.68** | **2.54** | **1.98** | **Ours (fixed parameters)** |
| COMET | 3.74 | 2.55 | 1.97 | Copula method |

**Real-World Data Density Estimation** (negative test log-likelihood, lower is better):

| Method | Insurance | Fama 5 | S&P 500 | CLIMDEX |
|------|-----------|--------|---------|---------|
| gTAF | 1.41 | 4.68 | 321.81 | -2113.48 |
| mTAF | 1.52 | 4.90 | 322.98 | -2121.38 |
| COMET | 1.41 | 4.63 | 324.38 | -2118.60 |
| **TTF** | **1.37** | **4.61** | **317.56** | **-2214.28** |
| **TTFfix** | **1.38** | **4.63** | **314.84** | -2090.91 |

### Ablation Study

| Configuration | Key Performance | Description |
|------|----------|------|
| normal/m_normal/g_normal | $\nu \leq 1$ failed to converge | Non-heavy-tail methods completely fail |
| TTF vs TTFfix | No significant difference on synthetic data | Joint learning vs. two-stage |
| TTF vs TTFfix (CLIMDEX) | TTF significantly better | Tail asymmetry is crucial in complex datasets |
| TTF_tBase | Inferior to TTF and TTFfix | Using both heavy-tailed base and TTF degrades performance |
| Fixed vs. Learned (t-base) | mTAF > gTAF | Fixed parameters perform better in heavy-tailed base distribution methods |
| Fixed vs. Learned (TTF) | No significant difference | TTF is more robust to parameter learning |

### Key Findings

1. **The higher the dimension and the heavier the tail weight, the more pronounced the advantages of TTF**: At $d=50, \nu=0.5$, TTF significantly outperforms heavy-tailed base distribution methods.
2. **Modeling tails at the final layer is superior**: TTF/TTFfix/COMET (which model tails at the final layer) consistently outperform gTAF/mTAF (which model tails at the base distribution).
3. **No performance loss in light-tailed scenarios**: At $\nu=30$ (nearly Gaussian), all methods perform comparably.
4. **Variational Inference**: TTFfix achieves the best $ESS_e$ in heavy-tailed scenarios ($\nu \leq 2$). At $\nu=0.5, d=50$, the $ESS_e$ of mTAF is close to 0, whereas TTF still maintains 0.39.
5. **Surprising performance of COMET**: Although it theoretically cannot produce Fréchet tails, its log-normal tails exhibit GPD-like behavior in the sub-asymptotic region.

## Highlights & Insights

- **Elegant Problem Reframing**: The issue is reframed from "which base distribution to use" to "where to introduce heavy tails." Placing the heavy-tail transformation at the final layer simultaneously resolves both theoretical limitations and optimization difficulties.
- **Minimalist Design**: The core is simply an analytical transformation based on $\text{erfc}$ without any additional neural network parameters, making it extremely simple to implement.
- **Alignment of Theory and Practice**: The theoretical guarantee for the Fréchet domain of attraction aligns well with the experimental results. The universality proofs guarantee the safety and theoretical integrity of the method.
- **An Important Warning to the NF Community**: Heavy-tailed base distribution methods degrade severely in high-dimensional heavy-tailed scenarios, a phenomenon that has been largely overlooked prior to this work.

## Limitations & Future Work

1. **TTF transformation always generates heavy tails**: It cannot precisely generate Gaussian tails (though no negative effects were observed in practice, it is theoretically imperfect).
2. **Coupling of tail and body**: TTF affects both the body and the tail of the distribution simultaneously. Ideally, these two should be decoupled.
3. **Lack of tail dependence modeling**: The current multivariate extension only transforms each margin independently and cannot capture tail dependence.
4. **Limited improvement in VI scenarios**: In realistic target posteriors, extremely heavy tails ($\nu \leq 1$) are rare, which may diminish the practical utility in some VI applications.
5. **High computational overhead of the two-stage method on high-dimensional data**: The Hill estimator can take as much time as NF training on high-dimensional datasets like CLIMDEX.
6. **No extension to diffusion/flow matching**: Although the paper discusses these possibilities, they are not empirically validated.

## Related Work & Insights

- **Jaini et al. (2020)**: Foundational theoretical result showing that Lipschitz transformations preserve light-tailedness.
- **Laszkiewicz et al. (2022) mTAF/gTAF**: Representative works utilizing heavy-tailed base distributions, serving as the main baselines in this paper.
- **McDonald et al. (2022) COMET**: A hybrid approach combining copulas and NFs, where the two-stage strategy is similar to TTFfix.
- **Liang et al. (2022) ATAF**: Anisotropic tail adaptation in the context of variational inference.
- **Inspiration for Diffusion Models**: Continuous NFs and diffusion models also face issues with extreme value inputs as they approach the terminal time, a scenario where the core idea of TTF could be directly transferred.

## Rating

| Aspect | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Novel perspective (focusing on "where" rather than "what" to model heavy tails) with a simple and elegant transformation design. |
| Theoretical Rigor | ⭐⭐⭐⭐⭐ | Fréchet domain of attraction proof, universality guarantees, and deep connection with EVT. |
| Empirical Evaluation | ⭐⭐⭐⭐ | Sufficient coverage of synthetic and real-world data, thorough ablations, though VI experiments only serve as a PoC. |
| Practical Utility | ⭐⭐⭐⭐ | Easy implementation (using erfc special function) that can be directly plugged into existing NF architectures. |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear motivation, holistic theoretical-empirical logical flow, and Figure 1 intuitively illustrates the central idea. |
| **Overall** | **⭐⭐⭐⭐☆** | Addresses an important and long-standing problem in the NF field with a simple and effective approach. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Neural Hamilton–Jacobi Characteristic Flows for Optimal Transport](../../ICLR2026/optimization/neural_hamilton--jacobi_characteristic_flows_for_optimal_transport.md)
- [\[ICLR 2026\] Toward Principled Flexible Scaling for Self-Gated Neural Activation](../../ICLR2026/optimization/toward_principled_flexible_scaling_for_self-gated_neural_activation.md)
- [\[ICLR 2026\] Deep Latent Variable Model based Vertical Federated Learning with Flexible Alignment and Labeling Scenarios](../../ICLR2026/optimization/deep_latent_variable_model_based_vertical_federated_learning_with_flexible_align.md)
- [\[ICML 2026\] Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization](../../ICML2026/optimization/accelerated_multiple_wasserstein_gradient_flows_for_multi-objective_distribution.md)
- [\[ICLR 2026\] Neural Multi-Objective Combinatorial Optimization for Flexible Job Shop Scheduling Problems](../../ICLR2026/optimization/neural_multi-objective_combinatorial_optimization_for_flexible_job_shop_scheduli.md)

</div>

<!-- RELATED:END -->

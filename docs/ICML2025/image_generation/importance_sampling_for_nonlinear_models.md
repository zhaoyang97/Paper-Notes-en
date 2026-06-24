---
title: >-
  [Paper Note] Importance Sampling for Nonlinear Models
description: >-
  [ICML2025][Image Generation][importance sampling] By introducing the adjoint operator of nonlinear mappings, this work systematically extends classical norm sampling and leverage score sampling from linear models to nonlinear models, providing the first theoretical approximation guarantees for importance sampling in nonlinear models such as neural networks.
tags:
  - "ICML2025"
  - "Image Generation"
  - "importance sampling"
  - "nonlinear adjoint"
  - "leverage scores"
  - "active learning"
  - "subspace embedding"
date: 2026-05-08
content_hash: 6d39fe7e3a6d864c
---

# Importance Sampling for Nonlinear Models

**Conference**: ICML2025  
**arXiv**: [2505.12353](https://arxiv.org/abs/2505.12353)  
**Code**: Not open-sourced  
**Area**: Importance Sampling / Nonlinear Models / Randomized Numerical Linear Algebra  
**Keywords**: importance sampling, nonlinear adjoint, leverage scores, active learning, subspace embedding

## TL;DR

By introducing the adjoint operator of nonlinear mappings, this work systematically extends classical norm sampling and leverage score sampling from linear models to nonlinear models, providing the first theoretical approximation guarantees for importance sampling in nonlinear models such as neural networks.

## Background & Motivation

The core of machine learning training is solving the optimization problem $\min_{\theta} \mathcal{L}(\theta) = \sum_{i=1}^{n} \ell(f_i(\theta))$, where $f_i$ can be a nonlinear mapping. With the explosive growth of data scale, how to efficiently select an "important" subset of data for training has become a key challenge.

In **linear models**, Randomized Numerical Linear Algebra (RandNLA) has developed a mature theory of importance sampling:
- **Row norm sampling**: Assigning weights by $q_i = \|\mathbf{x}_i\|_2^2$
- **Leverage score sampling**: Assigning weights by $q_i = \langle \mathbf{e}_i, \mathbf{X}\mathbf{X}^\dagger \mathbf{e}_i \rangle$
- These methods guarantee linear subspace embedding properties when the sample size $s \in \mathcal{O}(d \ln d / \varepsilon^2)$.

However, **extending these tools to nonlinear models has been a major challenge**. Existing works (such as Gajjar et al., 2023, 2024), though exploring single-neuron models, exhibit obvious limitations: a constant $C$ much larger than 1 (exceeding 1000) appears in their approximation guarantees, taking the form $\mathcal{L}(\theta_\mathcal{S}^\star) \leq C \cdot \mathcal{L}(\theta^\star) + \mathcal{O}(\varepsilon)$, which severely weakens their practical utility.

The core motivation of this paper is: **Can we establish a systematic framework that yields high-quality approximation guarantees with $C=1$ for nonlinear models?**

## Method

### 1. Nonlinear Adjoint Operator

The core innovation of this paper is the introduction of the adjoint operator for a nonlinear mapping $f: \mathbb{R}^p \to \mathbb{R}$. Using the integral form of the mean value theorem:

$$f(\theta) = f(\mathbf{0}) + \int_0^1 \left\langle \frac{\partial}{\partial \theta} f(t\theta), \theta \right\rangle \mathrm{d}t$$

The **adjoint operator** is thus defined as:

$$\mathbf{f}^\star(\theta) \triangleq \int_0^1 \frac{\partial}{\partial \theta} f(t\theta) \, \mathrm{d}t$$

This allows any differentiable nonlinear function to be written in an "inner-product-like" form: $f(\theta) = \langle \hat{\theta}, \hat{\mathbf{f}}^\star(\theta) \rangle$, where $\hat{\theta} = [\theta; 1]$ and $\hat{\mathbf{f}}^\star(\theta) = [\mathbf{f}^\star(\theta); f(\mathbf{0})]$.

**Key Significance**: This representation "linearizes" the structure of the nonlinear function, enabling the direct application of linear subspace embedding theory.

### 2. Closed-Form Adjoints for Special Models

When $f = g \circ h$, and $h$ is positively homogeneous of order $\alpha$ ($h(t\theta) = t^\alpha h(\theta)$), the adjoint operator has a closed-form solution (Proposition 3.1):

$$\mathbf{f}^\star(\theta) = \frac{g(h(\theta)) - g(0)}{\alpha \cdot h(\theta)} \cdot \frac{\partial h(\theta)}{\partial \theta}$$

**Generalized Linear Predictors** ($f(\theta) = \phi(\langle \theta, \mathbf{x} \rangle)$):

$$\mathbf{f}^\star(\theta) = \frac{\phi(\langle \theta, \mathbf{x} \rangle) - \phi(0)}{\langle \theta, \mathbf{x} \rangle} \cdot \mathbf{x}$$

**ReLU Two-Layer Neural Networks** ($f(\theta) = \sum_{j=1}^m \phi(a_j \cdot \max\{\langle \mathbf{b}_j, \mathbf{x} \rangle, 0\})$): The adjoint operator decomposes neuron-by-neuron with homogeneity degree $\alpha=2$.

### 3. Nonlinear Importance Sampling Scores

Define the **nonlinear dual matrix** $\hat{\mathbf{F}}^\star(\theta) \in \mathbb{R}^{n \times (p+1)}$, where each row corresponds to the adjoint operator of the $i$-th data point.

- **Nonlinear Leverage Score**: $\tau_i(\theta) = \frac{\langle \mathbf{e}_i, \hat{\mathbf{F}}^\star(\theta) [\hat{\mathbf{F}}^\star(\theta)]^\dagger \mathbf{e}_i \rangle}{\text{Rank}(\hat{\mathbf{F}}^\star(\theta))}$

- **Nonlinear Norm Score**: $\tau_i(\theta) = \frac{\|\hat{\mathbf{f}}_i^\star(\theta)\|_2^2}{\|\hat{\mathbf{F}}^\star(\theta)\|_F^2}$

### 4. Theoretical Guarantees

**Parameter-Dependent Approximation**: For a fixed $\theta$, after sampling $s \in \mathcal{O}(p \log(p/\delta) / \varepsilon^2)$ points, with probability $\geq 1-\delta$:

$$(1-\varepsilon) \mathcal{L}(\theta) \leq \mathcal{L}_\mathcal{S}(\theta) \leq (1+\varepsilon) \mathcal{L}(\theta)$$

**Parameter-Independent Approximation**: The parameter dependency is resolved via two key techniques:
1. Utilizing bounded properties of the activation function (e.g., $l \leq \phi^2(t)/t^2 \leq u$) to upper-bound the nonlinear scores with linear scores.
2. Using an $\varepsilon$-net construction and a union bound to extend the single-point guarantee to the entire parameter space.

This ultimately yields a guarantee with $C=1$: $\mathcal{L}(\theta_\mathcal{S}^\star) \leq \mathcal{L}(\theta^\star) + \mathcal{O}(\varepsilon)$.

### 5. Computational Pipeline

| Step | Operation | Complexity |
|------|------|--------|
| 1 | Compute linear leverage/norm scores as approximations to nonlinear scores | $\mathcal{O}(nd^2)$ |
| 2 | Perform non-uniform sampling of $s$ data points based on the approximate scores | $\mathcal{O}(s)$ |
| 3 | Train the model on the sampled subset | Depends on the model |
| 4 | (Optional) Posterior diagnostics: identify important samples, anomaly detection | $\mathcal{O}(n)$ |

## Key Experimental Results

The experiments cover multiple supervised learning scenarios, validating the theoretical results:

| Experimental Scenario | Sampling Method | Main Findings |
|----------|----------|----------|
| Generalized Linear Predictor (Swish activation) | Leverage score vs. Uniform sampling | Nonlinear leverage scores significantly outperform uniform sampling, closely matching full-data performance. |
| ReLU Two-Layer Neural Network | Norm score vs. Uniform sampling | Using only 10-20% of the data achieves a training loss close to that of the full dataset. |
| Anomaly Detection | Nonlinear score ranking | High-score data points highly correlate with known anomalies. |
| Model Interpretability | Score visualization | Nonlinear scores locate "critical" samples more accurately than linear scores. |

Core Experimental Conclusions:
- Nonlinear importance scores outperform both linear approximate scores and uniform sampling in terms of sampling efficiency.
- As the sample size increases, the subset training loss rapidly converges to the full-data loss.
- This framework can serve as a posterior diagnostic tool to identify important samples and anomalies.

## Highlights & Insights

1. **Theoretical Breakthrough**: For the first time, the approximation guarantee with $C=1$ ($\mathcal{L}(\theta_\mathcal{S}^\star) \leq \mathcal{L}(\theta^\star) + \mathcal{O}(\varepsilon)$) is extended to nonlinear models like neural networks, whereas prior work suffered from $C > 1000$.
2. **Elegant Mathematical Tool**: The introduction of the adjoint operator is akin to a nonlinear generalization of the Riesz representation theorem, representing nonlinear functions in an "inner-product-like" form, thereby bridging linear subspace embedding theory.
3. **Closed-Form Computation**: For composite functions with positively homogeneous inner layers (covering many common models), the adjoint operator can be computed directly without numerical integration.
4. **Dual Application**: It can be used both for training acceleration (forward sampling) and for posterior diagnostics (model interpretability, anomaly detection).
5. **Unified Framework**: Both norm scores and leverage scores are naturally derived under a unified adjoint operator framework, recovering classical definitions in the linear model case.

## Limitations & Future Work

1. **Under-Parameterized Assumption**: The theory requires $n \geq p$ (number of data points $\geq$ number of parameters), which is not applicable to currently dominant over-parameterized deep learning settings.
2. **Activation Function Restrictions**: The bounded condition $l \leq \phi^2(t)/t^2 \leq u$ is not directly satisfied by ReLU ($l=0$), requiring additional assumptions on bounded parameter domains.
3. **Difficulty in Extending to Multi-Layer Networks**: The current analysis primarily focuses on two-layer networks; the adjoint operator structure and theoretical guarantees for deep networks remain to be explored.
4. **Computational Overhead**: Computing leverage scores themselves requires $\mathcal{O}(nd^2)$, which is still burdensome for ultra-large-scale data; the paper does not provide a fast approximation scheme.
5. **Unknown Lipschitz Constants**: The parameter-independent guarantee contains error terms that depend on the Lipschitz constant $L(f, \mathbf{X}, R)$, which is hard to estimate in practice.
6. **Small Scale of Experiments**: Verification on large-scale deep learning tasks (e.g., ImageNet, LLM training) is lacking.

## Related Work & Insights

- **Linear Importance Sampling**: Drineas et al. (2006), Woodruff et al. (2014) — Theoretical foundation of this work.
- **Pioneering Nonlinear Sampling**: Gajjar et al. (2023, 2024) — Leverage score sampling for single-neuron models ($C \gg 1$).
- **Coreset**: Langberg & Schulman (2010), Feldman (2020) — Sensitivity sampling framework.
- **Data Selection & Curriculum Learning**: This framework may inspire more theoretically motivated data selection strategies.
- **Model Pruning/Distillation**: Nonlinear importance scores can guide the identification of "information-rich" subsets of samples.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The concept of the adjoint operator is a completely new theoretical tool that unifies linear and nonlinear importance sampling.
- Experimental Thoroughness: ⭐⭐⭐ — Theoretically well-validated, but lacks experiments on large-scale real-world tasks.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous and clear mathematical derivations, with a complete notation system.
- Value: ⭐⭐⭐⭐ — Establishes a solid theoretical foundation for data sampling of nonlinear models, offering broad inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Tree-Sliced Wasserstein Distance with Nonlinear Projection](tree-sliced_wasserstein_distance_with_nonlinear_projection.md)
- [\[ICML 2025\] Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models](morse_dual-sampling_for_lossless_acceleration_of_diffusion_models.md)
- [\[ICML 2026\] Efficient Learning of Deep State Space Models via Importance Smoothing](../../ICML2026/image_generation/efficient_learning_of_deep_state_space_models_via_importance_smoothing.md)
- [\[CVPR 2026\] Nonlinear Color Transfer via Learnable Bezier Flows](../../CVPR2026/image_generation/nonlinear_color_transfer_via_learnable_bezier_flows.md)
- [\[ICML 2025\] Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions](annealing_flow_generative_models_towards_sampling_high-dimensional_and_multi-mod.md)

</div>

<!-- RELATED:END -->

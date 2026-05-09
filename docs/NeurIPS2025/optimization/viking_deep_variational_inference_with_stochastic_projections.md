---
title: >-
  [Paper Note] VIKING: Deep Variational Inference with Stochastic Projections
description: >-
  [NeurIPS 2025][Optimization][Variational Inference] VIKING proposes a variational approximate posterior family based on the kernel- and image-space decomposition of the Fisher-Rao metric, and achieves scalable full-covariance Bayesian training via a stochastic alternating projections algorithm, surpassing existing Bayesian deep learning methods on multiple benchmarks.
tags:
  - NeurIPS 2025
  - Optimization
  - Variational Inference
  - Bayesian Deep Learning
  - Overparameterization
  - Fisher-Rao Metric
  - Stochastic Alternating Projections
date: 2026-05-08
content_hash: 1bca2cc96c89d26b
---

# VIKING: Deep Variational Inference with Stochastic Projections

**Conference**: NeurIPS 2025
**arXiv**: [2510.23684](https://arxiv.org/abs/2510.23684)
**Code**: [GitHub](https://github.com/fadel/viking)
**Area**: Optimization
**Keywords**: Variational Inference, Bayesian Deep Learning, Overparameterization, Fisher-Rao Metric, Stochastic Alternating Projections

## TL;DR

VIKING proposes a variational approximate posterior family based on the kernel- and image-space decomposition of the Fisher-Rao metric, and achieves scalable full-covariance Bayesian training via a stochastic alternating projections algorithm, surpassing existing Bayesian deep learning methods on multiple benchmarks.

## Background & Motivation

The central challenge of Bayesian deep learning lies in the **many-to-one mappings** induced by overparameterization — distinct parameter configurations can describe identical functions. For instance, $f(x) = w_1 \mathrm{ReLU}(w_2 x)$ can be reparameterized as $f(x) = w_1/\gamma \cdot \mathrm{ReLU}(\gamma w_2 x)$, yielding a large gap between parameter dimensionality and degrees of freedom that grows with model scale.

Traditional mean-field approximations assume parameter independence, failing to capture the strong correlation structure induced by overparameterization, which frequently leads to training instability, poor predictive quality, and poor calibration in practice. The authors observe that the variance learned by IVON (the current state-of-the-art mean-field method) is nearly identical across all weights, suggesting its approximate posterior is highly inaccurate.

The core motivation is: for Bayesian deep learning to succeed, the approximate posterior **must** reflect the geometric structure of overparameterization. Roy et al. (2024) proved that the set of parameters describing the same function forms a continuously connected set in weight space, which can be characterized via the kernel space of the Fisher-Rao metric.

## Method

### Overall Architecture

VIKING (Variational Inference with Kernel- and Image-spaces of numerical Gauss-Newton matrices) decomposes the parameter space into the kernel space (ker) and image space (im) of the Fisher-Rao metric, each governed by a scalar variance, yielding a simple yet full-covariance approximate posterior.

### Key Designs

1. **Variational Family**: The approximate posterior is $q(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} | \hat{\boldsymbol{\theta}}, \boldsymbol{\Sigma}_{\hat{\boldsymbol{\theta}}})$, with covariance matrix:
$$\boldsymbol{\Sigma}_{\hat{\boldsymbol{\theta}}} = \sigma_{\ker}^2 \mathbf{U}_{\hat{\boldsymbol{\theta}}} \mathbf{U}_{\hat{\boldsymbol{\theta}}}^\top + \sigma_{\mathrm{im}}^2 (\mathbb{I} - \mathbf{U}_{\hat{\boldsymbol{\theta}}} \mathbf{U}_{\hat{\boldsymbol{\theta}}}^\top)$$
   Here $\mathbf{U}_{\hat{\boldsymbol{\theta}}}$ is an orthonormal basis for the kernel space. $\sigma_{\ker}^2$ controls uncertainty outside the training data support, while $\sigma_{\mathrm{im}}^2$ controls uncertainty on the training data. Despite having only two scalar parameters, the inclusion of the projection matrix $\mathbf{U}\mathbf{U}^\top$ implicitly captures the full correlation structure among all parameters.

2. **ELBO Optimization**: The ELBO comprises a reconstruction term and a KL term. The KL term admits a closed-form expression (as both the prior and variational distribution are Gaussian), requiring only the kernel space dimension $R$, which can be estimated via the Hutchinson trace estimator. The reconstruction term is estimated via Monte Carlo sampling from $q(\boldsymbol{\theta})$. The key computational challenge lies in projecting samples onto the kernel space.

3. **Stochastic Alternating Projections Algorithm**: This is the core computational innovation of VIKING. Projecting onto the Fisher-Rao kernel space is equivalent to solving a constrained least-squares problem $\boldsymbol{\epsilon}_{\ker} = \arg\min_{\mathbf{u}} \|\mathbf{u} - \boldsymbol{\epsilon}\|^2 \ \text{s.t.} \ \mathbf{J} \mathbf{u} = \mathbf{0}$, which requires solving an $N \times N$ linear system over the entire dataset. The original alternating projections algorithm requires multiple full passes over the dataset to project a single vector, making it incompatible with mini-batch optimization. The authors propose a stochastic extension:
$$\boldsymbol{\epsilon}^{(t)} = \mathbf{U}^{(t)} \mathbf{U}^{(t)\top} (\sqrt{\gamma} \boldsymbol{\epsilon}^{(t-1)} + \sqrt{1-\gamma} \boldsymbol{\eta}^{(t)})$$
   The hyperparameter $\gamma \in [0,1]$ controls how much historical information is retained. $\gamma=1$ corresponds to a noise-free naive approach, $\gamma=0$ relies solely on the current batch. Intermediate values implement a sliding-window effect; experiments show optimal performance near $\gamma=0.5$.

### Loss & Training

- Conjugate gradients (with full re-orthogonalization) are used to solve linear systems in a matrix-free manner, avoiding explicit construction of $D \times D$ matrices.
- Warm-up from a pretrained model is supported: training begins with maximum likelihood estimation before switching to ELBO optimization, which significantly accelerates convergence.
- An optimal switching point exists: switching after maximum likelihood training has fully converged may cause the ELBO optimization to become trapped in local optima.

## Key Experimental Results

### Main Results

| Dataset | Method | Accuracy↑ | NLL↓ | ECE↓ | MCE↓ |
|--------|------|---------|------|------|------|
| MNIST | MAP | 0.986 | 0.070 | 0.247 | 0.861 |
| MNIST | IVON | 0.989 | **0.043** | **0.077** | **0.651** |
| MNIST | **VIKING** | **0.991** | 0.055 | 0.096 | 0.690 |
| Fashion MNIST | MAP | 0.883 | 0.410 | 0.153 | 0.590 |
| Fashion MNIST | IVON | 0.897 | 0.335 | **0.073** | 0.683 |
| Fashion MNIST | **VIKING** | **0.900** | 0.332 | 0.075 | **0.611** |
| SVHN | MAP | 0.947 | 0.201 | 0.055 | 0.608 |
| SVHN | IVON | 0.943 | 0.302 | 0.082 | 0.492 |
| SVHN | **VIKING** | **0.960** | **0.177** | **0.028** | **0.308** |
| CIFAR-10 | MAP | 0.824 | 0.536 | 0.075 | 0.619 |
| CIFAR-10 | **VIKING** | 0.877 | 0.407 | **0.041** | **0.331** |
| Imagenette | **VIKING** | **0.887** | **0.403** | 0.077 | 0.612 |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| $\gamma=0.0$ (pure noise) | Moderate training accuracy, moderate generalization | Each step relies solely on current-batch projection |
| $\gamma=0.5$ (mixed) | Best generalization performance | Balance between historical projection and new noise |
| $\gamma=1.0$ (no noise) | Highest training accuracy but poor generalization | Projected samples updated only once per epoch |
| Posterior fine-tuning vs. full training | Full training superior | Posterior fine-tuning still yields reasonable results |
| Pretrained warm-up | Significantly accelerates convergence | Switching after full convergence may be problematic |

### Key Findings

- On SVHN and CIFAR-10, VIKING achieves substantially better calibration metrics (ECE, MCE) than all baselines, as overparameterization is more pronounced on these datasets.
- In OOD detection (MNIST→FMNIST/KMNIST/EMNIST), VIKING achieves significantly higher AUROC than baseline methods.
- The method scales to training ResNet34 (21.7M parameters) on Imagenette.

## Highlights & Insights

- **Minimalist yet effective design**: Uncertainty in deep neural networks is calibrated using only two scalar parameters; the key lies in implicitly capturing the full correlation structure via the projection matrix.
- **Bridge between theory and practice**: The differential-geometric theory of overparameterization is translated into a practical variational inference algorithm.
- **Innovation in stochastic alternating projections**: Posterior updates and ELBO optimization are elegantly integrated, enabling compatibility with mini-batch training.
- Toy regression experiments intuitively demonstrate VIKING's superiority — posterior samples from IVON fail to reflect uncertainty at the boundaries.

## Limitations & Future Work

- The computational cost of kernel-space projection is non-trivial, requiring multiple conjugate gradient iterations per training step.
- Parameterizing with only two scalars may be overly simplistic — larger models may require more flexible variance structures for the kernel and image spaces.
- On CIFAR-10, accuracy still lags behind Last Layer LA (0.877 vs. 0.894), indicating that a full-covariance posterior is not always the optimal choice.
- The method is primarily validated on small-to-medium-scale models; scalability to larger architectures (e.g., BERT, GPT) remains to be confirmed.

## Related Work & Insights

- A direct comparison with IVON (mean-field ELBO optimization) demonstrates that accounting for the overparameterization structure is critical.
- The posterior projection method of Miani et al. (2025) is the immediate predecessor of VIKING; VIKING extends it from post-hoc approximation to full training.
- The paper provides positive evidence for the debate on whether Bayesian deep learning is a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Incorporating overparameterization geometry into variational inference is a highly elegant idea
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-baseline comparisons, though large-scale model experiments are absent
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear and motivation is well articulated
- Value: ⭐⭐⭐⭐ Provides a new and effective paradigm for Bayesian deep learning

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Brain-like Variational Inference](brain-like_variational_inference.md)
- [\[NeurIPS 2025\] Least Squares Variational Inference](least_squares_variational_inference.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

<!-- RELATED:END -->

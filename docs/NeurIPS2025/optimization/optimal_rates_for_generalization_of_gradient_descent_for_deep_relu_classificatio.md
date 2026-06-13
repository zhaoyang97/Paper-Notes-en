---
title: >-
  [Paper Note] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification
description: >-
  [NeurIPS 2025][Optimization][Deep ReLU networks] This paper establishes a generalization rate of $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$ for gradient descent on deep ReLU networks…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Deep ReLU networks"
  - "generalization bounds"
  - "Rademacher complexity"
  - "NTK separability"
  - "gradient descent"
date: 2026-05-08
content_hash: f2eca5183d717af4
---

# Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification

**Conference**: NeurIPS 2025
**arXiv**: [2510.02779](https://arxiv.org/abs/2510.02779)  
**Code**: None  
**Area**: Optimization
**Keywords**: Deep ReLU networks, generalization bounds, Rademacher complexity, NTK separability, gradient descent

## TL;DR

This paper establishes a generalization rate of $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$ for gradient descent on deep ReLU networks, achieving for the first time simultaneously: (1) the optimal $1/n$ dependence on sample size $n$, and (2) only polynomial dependence on depth $L$.

## Background & Motivation

Understanding the generalization behavior of deep neural networks is a central problem in theoretical research. Prior work on analyzing the generalization of GD-trained deep networks faces two major difficulties:

**Suboptimal sample complexity**: Methods based on Rademacher complexity or the NTK framework typically yield generalization bounds of $\widetilde{O}(1/\sqrt{n})$, or apply only to shallow networks.

**Exponential dependence on depth**: Algorithmic stability-based methods can achieve the optimal $\widetilde{O}(1/n)$ rate, but require smooth activation functions and incur an $e^{O(L)}$ exponential dependence on depth $L$.

The core question of this paper is: can one simultaneously obtain the optimal $1/n$ rate and a polynomial $\text{poly}(L)$ dependence for deep ReLU (non-smooth) networks?

## Method

### Overall Architecture

The analysis follows the classical decomposition $\mathcal{L}(\mathbf{W}) = (\mathcal{L}(\mathbf{W}) - \mathcal{L}_S(\mathbf{W})) + \mathcal{L}_S(\mathbf{W})$. The optimization error is handled via a reference model argument, and the generalization error is controlled using an improved Rademacher complexity technique. The key innovation lies in controlling the variation of ReLU activation patterns.

### Key Designs

1. **Reference model optimization analysis (Theorem 1)**: For a reference model $\bar{\mathbf{W}}$, the paper defines $F_S(\bar{\mathbf{W}}) = 3\eta T \mathcal{L}_S(\bar{\mathbf{W}}) + \|\mathbf{W}(0) - \bar{\mathbf{W}}\|_F^2$ and proves that all GD iterates satisfy $\|\mathbf{W}(t) - \bar{\mathbf{W}}\|_F^2 \leq F_S(\bar{\mathbf{W}})$ with bounded cumulative training loss. The core inequality is $\|\mathbf{W}(t+1) - \bar{\mathbf{W}}\|_F^2 \leq \|\mathbf{W}(t) - \bar{\mathbf{W}}\|_F^2 - \eta \mathcal{L}_S(\mathbf{W}(t)) + 3\eta \mathcal{L}_S(\bar{\mathbf{W}})$. Compared to prior work, the overparameterization requirement is reduced by a factor of $L^6$.

2. **Improved Rademacher complexity (core contribution)**: The network output difference is expressed as $f_{\mathbf{W}}(\mathbf{x}_i) - f_{\mathbf{W}(0)}(\mathbf{x}_i) = \mathbf{a}^\top \sum_{l=1}^L \hat{\mathbf{G}}_{L,0}^l(\mathbf{x}_i)(\mathbf{W}^l - \mathbf{W}^l(0)) h_0^{l-1}(\mathbf{x}_i)$, where $\hat{\mathbf{G}}_{L,0}^l$ is a product of sparse matrices. The key step is showing $\hat{\mathbf{G}}_{L,0}^l = \widetilde{O}(L/\sqrt{m})$, which yields $\mathfrak{R}_{S_1,n}(\mathcal{F}) = \widetilde{O}(L^2 \sqrt{F(\bar{\mathbf{W}})/n})$. Prior methods obtain $\widetilde{O}(4^L L \sqrt{mF/n})$, which contains an exponential depth factor and a $\sqrt{m}$ width factor.

3. **Covering number strategy (eliminating exponential depth dependence)**: Standard recursive Lipschitz analysis leads to an exponential bound $\|h^L(\mathbf{x}) - h_0^L(\mathbf{x})\|_2 \leq C^L R/\sqrt{m}$. This paper instead constructs a $1/(C^L\sqrt{m})$-cover $D$ of the input space, first establishes $\|h^l(\mathbf{x}^j) - h_0^l(\mathbf{x}^j)\|_2 = \widetilde{O}(L^2 R/\sqrt{m})$ on cover points, then extends to all inputs via nearest-cover-point approximation, ultimately obtaining $\sup_\mathbf{x} \|h^l(\mathbf{x}) - h_0^l(\mathbf{x})\|_2 = \widetilde{O}(L^2 R/\sqrt{m})$. Although the covering number itself may be exponential, only its logarithm appears in the analysis, preserving polynomial depth dependence.

### Loss & Training

The logistic loss $\ell(z) = \log(1+\exp(-z))$ is used for classification. GD updates follow $\mathbf{W}^l(k+1) = \mathbf{W}^l(k) - \eta \partial \mathcal{L}_S / \partial \mathbf{W}^l(k)$. Symmetric initialization ensures $f_{\mathbf{W}(0)}(\mathbf{x}) = 0$. The step size satisfies $\eta \leq 4/(5L)$, and the number of iterations $T$ satisfies $\eta T \asymp n$.

## Key Experimental Results

### Main Results

This paper is a purely theoretical contribution. The main results are summarized below:

| Work | Activation | Width Requirement | Generalization Error | Network Depth |
|------|-----------|-------------------|----------------------|---------------|
| Ji et al. 2020 | ReLU | $\widetilde{\Omega}(1/\gamma^8)$ | $\widetilde{O}(1/(\gamma^2\sqrt{n}))$ | Shallow |
| Lei 2024 | ReLU | $\widetilde{\Omega}(1/\gamma^8)$ | $\widetilde{O}(1/(\gamma^2 n))$ | Shallow |
| Chen et al. 2021 | ReLU | $\widetilde{\Omega}(L^{22}/\gamma^8)$ | $\widetilde{O}(e^{O(L)}\sqrt{m/n}/\gamma)$ | Deep |
| Taheri 2025 | Smooth | $\widetilde{\Omega}(1/\gamma^{6L+4})$ | $\widetilde{O}(e^{O(L)}/(n\gamma^2))$ | Deep |
| **Ours** | **ReLU** | $\widetilde{\Omega}(L^{16}/\gamma^8)$ | $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$ | **Deep** |

### Ablation Study

| Technical Contribution | Improvement | Remarks |
|-----------------------|-------------|---------|
| Sparse-matrix Rademacher bound | $4^L \to L^2$ | Eliminates exponential depth dependence |
| Elimination of width factor | $\sqrt{m} \to \log$ | Generalization bound no longer significantly depends on network width |
| Covering-number Lipschitz analysis | $C^L \to L^2$ | Near-initialization Lipschitz constant reduced from exponential to polynomial in depth |
| Overparameterization requirement | $L^{22} \to L^{16}$ | Reduced by a factor of $L^6$ |

### Key Findings

- For NTK-separable data with margin $\gamma$, the generalization rate $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$ matches the optimal SVM rate $\widetilde{O}(1/(n\gamma^2))$ up to depth-polynomial factors.
- The key breakthrough is encoding the variation of ReLU activation patterns as sparse matrices $\hat{\mathbf{G}}$ and precisely controlling their norms.

## Highlights & Insights

- This is the first work to simultaneously achieve the optimal $1/n$ rate and polynomial depth dependence on deep ReLU networks, resolving a long-standing open problem.
- The covering number strategy is a technically elegant innovation: although the covering number itself may be exponential, only its logarithm enters the analysis.
- The sparse matrix representation $\hat{\mathbf{G}}_{L,0}^l$ captures the variation of ReLU activation patterns and is a more refined tool than naive recursive estimation.
- The reference model approach is more flexible than NTK Gram matrix analysis and does not require studying the kernel or its corresponding Gram matrix.

## Limitations & Future Work

- The overparameterization requirement $m = \widetilde{\Omega}(L^{16}/\gamma^8)$ remains large, with a significant gap from the network widths used in practice.
- Only classification (logistic loss) is analyzed; extending to optimal generalization rates for regression (e.g., MSE loss) is a natural next step.
- The symmetric initialization assumption, while not essential, adds theoretical complexity; whether removing it affects the conclusions requires verification.
- Whether the depth factor $L^4(1+\gamma L^2)$ can be further tightened is unclear; the optimal dependence is conjectured to be $\text{poly}(L)$.
- Only GD is analyzed; the additional variance introduced by the stochasticity of SGD requires new technical tools.
- Whether the NTK-separability assumption holds for real-world data remains debated.

## Related Work & Insights

- The NTK framework connects deep networks to kernel methods; this paper achieves an optimal rate under the NTK-separable data assumption.
- Algorithmic stability methods can yield $1/n$ rates but are restricted to smooth activations; the Rademacher-based approach here breaks this limitation.
- The technical combination of covering numbers and sparse matrix norm control may generalize to other piecewise-linear activations.
- The comparison with Taheri 2025 is most instructive: both target the $1/n$ rate, but stability methods inherently require smoothness.
- Advantage of the reference model approach: it does not require the NTK Gram matrix to be positive definite ($\lambda_0 > 0$), which tends to zero in the large-sample regime.
- He initialization $\mathbf{w}_r^l \sim \mathcal{N}(0, 2/m)$ is theoretically equivalent to the $\sqrt{2/m}$ scaling factor used in this paper.
- Practical relevance of the lazy training regime: this paper proves all iterates remain near initialization, but without relying on kernel or Gram matrix analysis.
- Future directions include extending similar analyses to practical architectures such as CNNs and ResNets, as well as the SGD setting.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Resolves an important open problem in the generalization theory of deep ReLU networks
- Experimental Thoroughness: ⭐⭐ Purely theoretical work
- Writing Quality: ⭐⭐⭐⭐ Comparison table is clear; proof sketches aid understanding of core ideas
- Value: ⭐⭐⭐⭐⭐ A significant advance in the generalization theory of deep learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)

</div>

<!-- RELATED:END -->

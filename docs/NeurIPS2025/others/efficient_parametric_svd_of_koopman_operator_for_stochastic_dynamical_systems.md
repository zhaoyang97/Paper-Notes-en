---
title: >-
  [Paper Note] Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems
description: >-
  [NEURIPS2025][Koopman operator] This paper proposes a low-rank approximation (LoRA)-based objective to learn the top-k singular functions of the Koopman operator for stochastic dynamical systems, entirely avoiding the numerically unstable matrix decomposition operations present in VAMPnet/DPNet, with naturally unbiased gradients.
tags:
  - NEURIPS2025
  - Koopman operator
  - singular value decomposition
  - dynamical systems
  - low-rank approximation
  - deep learning
date: 2026-05-08
content_hash: e8e052a6a5bba4ca
---

# Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems

**Conference**: NEURIPS2025  
**arXiv**: [2507.07222](https://arxiv.org/abs/2507.07222)  
**Code**: [MinchanJeong/NeuralKoopmanSVD](https://github.com/MinchanJeong/NeuralKoopmanSVD)  
**Area**: Others  
**Keywords**: Koopman operator, singular value decomposition, dynamical systems, low-rank approximation, deep learning

## TL;DR

This paper proposes a low-rank approximation (LoRA)-based objective to learn the top-k singular functions of the Koopman operator for stochastic dynamical systems, entirely avoiding the numerically unstable matrix decomposition operations present in VAMPnet/DPNet, with naturally unbiased gradients.

## Background & Motivation

Koopman operator theory lifts nonlinear dynamical systems into an infinite-dimensional function space, enabling spectral analysis techniques for linear operators to be applied directly. Dynamic Mode Decomposition (DMD) has demonstrated the ability to identify dominant system modes in a data-driven manner from trajectory data. Building on this, deep learning methods such as VAMPnet and DPNet have been proposed to learn the dominant singular subspace of the Koopman operator.

However, these methods require numerically unstable operations such as SVD or matrix inversion on empirical second-moment matrices during objective computation. This introduces two core issues:

1. **Gradient bias**: Backpropagation through SVD/matrix inversion introduces biased gradient estimates, particularly when matrices are ill-conditioned (small eigenvalue gaps), causing gradient explosion.
2. **Poor scalability**: Second-moment matrices estimated from mini-batches may be rank-deficient, leading to unstable inversion; VAMPnet requires an additional regularization parameter $\lambda$ to mitigate this.

## Core Problem

How to design a conceptually simple, numerically stable method that integrates naturally with modern deep learning pipelines to learn the top-k singular functions of the Koopman operator for stochastic dynamical systems?

## Method

### Problem Setup

Consider a stochastic discrete-time dynamical system $\mathbf{x}_{t+1} = \xi(F(\mathbf{x}_t), \epsilon_t)$, where the Koopman operator is the conditional expectation operator:

$$(\mathcal{K}g)(\mathbf{x}) = \mathbb{E}_{p(\mathbf{x}'|\mathbf{x})}[g(\mathbf{x}')]$$

The goal is to learn the top-k singular subspace of $\mathcal{K}$ from transition pairs $\{(\mathbf{x}_t, \mathbf{x}_{t+1})\}$ collected from $N$ independent trajectories.

### LoRA Objective

The core idea is to directly minimize the low-rank approximation error $\|\mathcal{K} - \sum_{i=1}^k f_i \otimes g_i\|_{\text{HS}}^2$, which simplifies to a remarkably clean objective:

$$\mathcal{L}_{\text{lora}}(\mathbf{f}, \mathbf{g}) = -2 \operatorname{tr}(\mathsf{T}[\mathbf{f}, \mathbf{g}]) + \operatorname{tr}(\mathsf{M}_{\rho_0}[\mathbf{f}] \cdot \mathsf{M}_{\rho_1}[\mathbf{g}])$$

where $\mathsf{T}[\mathbf{f}, \mathbf{g}]$ is the joint second-moment matrix, and $\mathsf{M}_{\rho_0}[\mathbf{f}]$ and $\mathsf{M}_{\rho_1}[\mathbf{g}]$ are the second-moment matrices of the current and future states, respectively.

**Key advantages**: The objective is entirely a polynomial in second-moment matrices, involving no matrix inversion or SVD operations. Consequently:
- Gradients can be naturally estimated in an unbiased manner via mini-batches
- No regularization parameter tuning is required
- Easy integration into standard deep learning training pipelines

The Eckart–Young–Mirsky theorem guarantees that the global optimum of this objective precisely characterizes the singular subspace of $\mathcal{K}$.

### Nesting: Learning Ordered Singular Functions

To learn singular functions ordered by singular value magnitude, a nesting technique is introduced:

- **Joint nesting**: Simultaneously optimizes all dimensions via $\mathcal{L}_{\text{lora}}^{\text{joint}} = \sum_{i=1}^k \alpha_i \mathcal{L}_{\text{lora}}(\mathbf{f}_{1:i}, \mathbf{g}_{1:i})$, with uniform weights being optimal.
- **Sequential nesting**: Incrementally updates the $i$-th function pair $(f_i, g_i)$, with convergence guaranteed by an inductive argument.

Both nesting variants introduce negligible additional computational overhead, and experiments consistently show that nesting improves downstream task performance.

### Inference

After learning the singular subspace, two inference pathways are provided:

**Method 1 (CCA + LoRA)**: Whitens the learned basis functions and aligns them via SVD to obtain ordered singular functions. Multi-step forward/backward prediction is performed using the finite-rank approximation.

**Method 2 (EDMD)**: Projects the Koopman operator onto the subspace spanned by the learned basis functions, obtains an approximate Koopman matrix via least-squares regression, and performs multi-step prediction.

### Extension to Continuous Time

For reversible continuous-time dynamical systems (e.g., overdamped Langevin dynamics), the objective simplifies to $\mathcal{L}_{\text{lora}}^{\text{sa}}(\mathbf{f}) = -2\operatorname{tr}(\mathsf{M}_{\rho_0}[\mathbf{f}, \mathcal{L}\mathbf{f}]) + \|\mathsf{M}_{\rho_0}[\mathbf{f}]\|_F^2$, eliminating the need for a separate lagged encoder.

## Key Experimental Results

### Ordered MNIST

A synthetic experiment where digits transition cyclically as 0→1→2→3→4→0. LoRA and its variants consistently outperform VAMPnet-1, DPNet, and DPNet-relaxed on multi-step prediction RMSE, with joint nesting achieving the best performance. Prediction horizon spans $t \in \{-15, \ldots, 15\}$.

### 1D Langevin Dynamics

A reversible continuous-time process. LoRA$_{\text{seq}}$ reliably recovers the first 8 eigenfunctions and eigenvalues in close agreement with ground truth. By contrast, the DPNet objective fails to converge under the same configuration.

### Chignolin Molecular Dynamics

High-dimensional real-world data (folding dynamics of an artificial miniprotein). Key results (VAMP-E score, higher is better):

| Method | Low-rank (k=16) | High-rank (k=64) |
|--------|-----------------|-------------------|
| DPNet-relaxed | 7.36±0.40 | 6.97±0.31 |
| VAMPnet-1 | 9.54±0.31 | 19.71±0.59 |
| LoRA | 10.27±0.31 | 37.74±0.95 |
| LoRA$_{\text{jnt}}$ | 10.74±0.35 | 38.50±0.83 |
| LoRA$_{\text{seq}}$ | **12.29±0.07** | **37.33±1.66** |

Note: VAMPnet-2 and DPNet diverge in this experiment; VAMPnet-1 is sensitive to PyTorch version and data splits, also diverging under certain settings. LoRA variants are the only methods that converge consistently across all configurations.

## Highlights & Insights

1. **Minimal objective**: The LoRA objective is purely a polynomial in second-moment matrices, entirely avoiding matrix inversion and SVD — conceptually clear and straightforward to implement.
2. **Unbiased gradients**: Unlike VAMPnet/DPNet, mini-batch gradients are naturally unbiased, making the method well-suited for large-scale systems.
3. **No regularization**: Eliminates the need for VAMPnet's $\lambda$ tuning or DPNet's metric distortion loss.
4. **Consistent convergence**: Reliably converges across all experiments; the only method family that does not diverge in the high-dimensional Chignolin experiment.
5. **Complete theoretical guarantees**: Global optimality is ensured by the Eckart–Young–Mirsky theorem, with $O(N^{-1/2})$ sample complexity.

## Limitations & Future Work

1. The quality of the learned dynamics is limited by the temporal resolution of the data, which may preclude recovery of the slowest physical processes.
2. Robust identification of coherent structures in highly non-normal or chaotic systems remains an open problem.
3. Applicability is restricted to stochastic dynamical systems, as the Koopman operator of deterministic systems is not compact.
4. Theoretical analysis of the optimization landscape of the LoRA objective remains incomplete.
5. Specialized neural network architectures tailored for Koopman analysis have not been explored.

## Related Work & Insights

| Aspect | VAMPnet | DPNet | Ours (LoRA) |
|--------|---------|-------|-------------|
| Objective | Schatten norm + matrix inversion | VAMP-2 + metric distortion | Second-moment polynomial |
| Numerical stability | Requires regularization $\lambda$ | Requires relaxation | Inherently stable |
| Gradient unbiasedness | Biased | Biased | Unbiased |
| Regularization parameter | $\lambda$ (cross-validation) | $\gamma$ | None |
| High-dimensional scalability | Poor (degrades at k=64) | Diverges | Remains stable |

The LoRA idea generalizes low-rank matrix approximation to operator low-rank approximation, a strategy that may be applicable to spectral learning problems for other infinite-dimensional operators. A key design insight is to avoid non-differentiable or numerically unstable operations within training objectives that require backpropagation, replacing them with equivalent polynomial forms. The nesting technique provides a means of directing the model toward the most important signals without additional computational cost, potentially applicable to other multi-objective learning settings. The combination with molecular dynamics demonstrates practical value in scientific computing applications.

## Rating
- Novelty: ⭐⭐⭐⭐ (The LoRA objective is simple yet elegant; the first systematic application of this approach in the Koopman learning literature)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Broad coverage across synthetic, Langevin, and molecular dynamics settings with thorough ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous theoretical derivations, clear notation, well-motivated exposition)
- Value: ⭐⭐⭐⭐ (Provides a practical and reliable new baseline for Koopman operator learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)
- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[AAAI 2026\] SVD-NO: Learning PDE Solution Operators with SVD Integral Kernels](../../AAAI2026/others/svd-no_learning_pde_solution_operators_with_svd_integral_kernels.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Deep Learning for Subspace Regression
description: >-
  [ICLR2026][Scientific Computing][subspace regression] This paper formalizes the subspace prediction problem in Reduced Order Modeling (ROM) as a regression task on the Grassmann manifold. It proposes dedicated loss funct…
tags:
  - "ICLR2026"
  - "Scientific Computing"
  - "subspace regression"
  - "Grassmann manifold"
  - "reduced order modeling"
  - "neural operator"
  - "eigenspace"
date: 2026-05-08
content_hash: 3dd2c8e0fdb3d008
---

# Deep Learning for Subspace Regression

**Conference**: ICLR2026
**arXiv**: [2509.23249](https://arxiv.org/abs/2509.23249)
**Code**: [GitHub](https://github.com/VLSF/subreg)
**Area**: Scientific Computing
**Keywords**: subspace regression, Grassmann manifold, reduced order modeling, neural operator, eigenspace

## TL;DR

This paper formalizes the subspace prediction problem in Reduced Order Modeling (ROM) as a regression task on the Grassmann manifold. It proposes dedicated loss functions and a subspace embedding technique—predicting a higher-dimensional subspace containing the target—to reduce mapping complexity. The approach achieves significant improvements across eigenvalue problems, parametric PDEs, and iterative solver acceleration.

## Background & Motivation

**Background**: Reduced Order Modelling (ROM) simplifies the analysis and simulation of systems by identifying linear subspaces that discard uninformative degrees of freedom. This paradigm is particularly valuable in scenarios requiring repeated solutions of related problems, such as parametric PDEs, optimal control, and uncertainty quantification. Canonical methods such as Proper Orthogonal Decomposition (POD) construct reduced bases by computing optimal low-rank approximations of solutions over representative parameter samples.

**Limitations of Prior Work**: When subspaces depend explicitly on parameters (i.e., local POD), one must interpolate from known subspaces at observed parameters to subspaces at unseen parameters. Classical interpolation methods on the Grassmann manifold—such as Riemannian normal coordinate interpolation—are highly unreliable in high-dimensional parameter spaces, where sparse observations degrade the quality of tangent space approximations and computational cost grows rapidly with dimension.

**Key Challenge**: Subspace data possesses a special algebraic structure—right $\text{GL}(k)$ invariance—whereby matrices $W_1$ and $W_2$ are equivalent if they span the same column space. This renders standard $\ell_2$ regression losses entirely inapplicable. Furthermore, the complexity of parametric eigenproblems (i.e., the number of distinct subspace configurations) grows rapidly with both parameter dimension and eigenvector index, making direct learning of individual eigenvectors subject to combinatorial explosion.

**Goal**: The paper relaxes the interpolation problem to a regression problem, designs loss functions satisfying $\text{GL}(k)$ invariance, and parameterizes the mapping from high-dimensional parameters to subspaces via neural networks. The core innovation is **subspace embedding**—predicting a higher-dimensional subspace that contains the target. It is theoretically shown that this reduces the derivative norm of the mapping (i.e., promotes smoothness), thereby aligning with the F-principle (spectral bias) of neural networks and substantially improving accuracy.

## Method

### Overall Architecture

Given a dataset $\mathcal{D} = \{(r_i, V_i)\}_{i=1}^m$ of parameter vectors $r \in \mathbb{R}^p$ and corresponding $k$-dimensional subspaces $V(r) \in \text{Gr}(k, n)$, a neural network $Y_\theta: \mathbb{R}^p \to \text{Gr}(r, n)$ (allowing $r \geq k$) is trained to approximate the mapping $V(r)$ by minimizing a subspace loss:

$$\theta^\star = \arg\min_\theta \frac{1}{m} \sum_{i=1}^m L(Y_\theta(r_i), V_i)$$

The loss function $L$ must satisfy two key properties: (1) invariance to equivalence classes, $L(A,B) = L(\tilde{A}, \tilde{B})$ for any $\tilde{A} \in \lceil A \rceil, \tilde{B} \in \lceil B \rceil$; and (2) $L(A,B) = 0$ if and only if $\mathcal{S}(B) \subset \mathcal{S}(A)$. The network architecture is based on the Factorized Fourier Neural Operator (FFNO), whose output is mapped onto the Grassmann manifold.

### Key Design 1: Subspace Loss Functions

**Theorem 1** establishes two loss functions satisfying the invariance requirements:

- **Deterministic loss $L_1$**: $L_1(A, B) = p - \|Q_B^\top Q_A\|_F^2$, where $A = Q_A R_A$ and $B = Q_B R_B$ are QR decompositions. This essentially measures the Frobenius norm difference between orthogonal projectors. $L_1 = 0$ if and only if $\mathcal{S}(B) \subset \mathcal{S}(A)$.

- **Randomized loss $L_2$**: $L_2(A, B; z) = \min_u \|Au - Q_B z\|_2^2$, where $z \sim \mathcal{N}(0, I_k)$. This replaces the projector computation with a least-squares problem and eliminates the second QR decomposition via the Hutchinson trace estimator. A key theoretical result is that $\mathbb{E}_z[L_2] = L_1$.

The advantage of $L_2$ lies in solving the least-squares problem via the normal equations, yielding significantly better training efficiency than $L_1$ (which requires QR decomposition) as subspace dimension increases. However, the normal equations may introduce numerical instability; applying Cholesky-QR2 stabilization (yielding $L_2^{\text{stab}}$) recovers accuracy.

### Key Design 2: Subspace Embedding

This is the paper's most central contribution. The strategy is straightforward: allow the model to predict an $r > k$ dimensional subspace as an approximation to the $k$-dimensional target, requiring only that $\mathcal{S}(V) \subset \mathcal{S}(Y_\theta)$. This "redundancy" strategy is theoretically supported from two perspectives:

**Theorem 2 (Smoothness Reduction)**: For a continuously differentiable function $V: \mathbb{R} \to \text{Gr}(k, n)$, one can always construct $W: \mathbb{R} \to \text{Gr}(r, n)$ ($r > k$) such that $\|\dot{W}(t)\|_F^2 \leq \|\dot{V}(t)\|_F^2$, with strict inequality at points where the derivative is nonzero. This means that embedding into a larger subspace renders the mapping smoother, perfectly aligning with the F-principle of neural networks—their tendency to learn low-frequency, smooth functions.

**Theorem 3 (Complexity Reduction)**: For a constant-coefficient elliptic eigenproblem $-\sum a_i \partial^2 \phi / \partial x_i^2 = \lambda \phi$, the mapping from parameters to the $k$-th eigenvector is a piecewise constant function with $\#_{F_k}(k, D) \sim \frac{1}{(D-1)!} k(\log k)^{D-1}$ distinct regions, while the subspace mapping has $\#_{G_k} \geq \frac{1}{(D-1)!} k^{D-1}$ regions. Subspace embedding can exploit the finite number of distinct combinations of eigenvectors to reduce the region count—in the extreme case, if the predicted dimension equals the total number of possible eigenvectors $\#_{F_k}$, the mapping degenerates to a constant function.

### Key Design 3: Unified Coverage Across Application Scenarios

The subspace regression framework provides a unified treatment of several core scientific computing problems:

| Application | Mapping | Subspace Meaning |
|-------------|---------|-----------------|
| Parametric eigenproblems | $U(x) \to \text{span}\{\phi_1, \dots, \phi_K\}$ | First $K$ eigenfunctions |
| Local POD (Burgers' equation) | PDE parameters $\to \{\psi_1, \dots, \psi_k\}$ | POD reduced basis |
| CG deflation | $k(x) \to \mathcal{S}(V)$ | Small-eigenvalue eigenspace |
| Coarse-grid correction (Jacobi) | $k(x) \to \mathcal{S}(V)$ | Dominant eigenspace of error propagation matrix |
| Balanced truncation (optimal control) | $A, B, C \to \mathcal{S}(\bar{\mathcal{T}})$ | Controllability/observability reduction basis |

## Key Experimental Results

### Quantum Mechanics Eigenspace Prediction

| Dataset | Riemannian Interpolation | $\mathbb{Z}_2$-adjusted $\ell_2$ | Subspace Regression $L_1$ |
|---------|:---:|:---:|:---:|
| $D=1$ Schrödinger | 4.69% | 2.33% | **0.09%** |
| $D=2$, dataset a | 31.9% | 19.52% | **0.65%** |
| $D=2$, dataset b | 92.64% | 48.56% | **15.58%** |

Subspace regression substantially outperforms classical interpolation and direct eigenvector prediction across all datasets.

### Subspace Embedding Effect ($D=2$ elliptic problem, first 10 eigenvectors)

| Predicted Subspace Dimension $N_{\text{sub}}$ | Test Error | Notes |
|:---:|:---:|:---|
| 10 (no embedding) | ~30% | Same as target dimension |
| 20 | ~10% | Embedding begins to take effect |
| 30 | ~5% | Significant improvement |
| 40 | **~2%** | Only 0.4% of total degrees of freedom |

The embedding technique reduces test error from 30% to 2%, while the train-test generalization gap also decreases systematically.

### Loss Function Comparison on $D=3$ Elliptic Problem

| $N_{\text{sub}}$ | $L_1(A,B)$ | $L_2(A,B;z)$ | $L_2^{\text{stab}}(A,B;z)$ |
|:---:|:---:|:---:|:---:|
| 6 | 24.77% | 31.46% | 28.28% |
| 12 | 13.69% | 17.12% | 15.88% |
| 24 | 9.71% | Failed | 9.49% |
| 48 | 7.54% | 16.3% | **7.4%** |

$L_2$ becomes unstable at large dimensions due to ill-conditioned normal equations; Cholesky-QR2 stabilization recovers accuracy to match or slightly surpass $L_1$.

## Highlights & Insights

- The **subspace embedding** idea is highly elegant: it exploits the monotonicity of subspace containment to trade redundancy for smoothness, perfectly aligning with the spectral bias (F-principle) of neural networks. The cost is a learned subspace of larger-than-optimal dimension, but the additional computational burden on the reduced model is negligible.
- **Theorem 3's complexity characterization** connects combinatorics with approximation theory, providing a precise measure of difficulty for subspace learning.
- The **randomized loss $L_2$** introduces Hutchinson trace estimation into subspace learning, achieving significantly better training efficiency than QR-based approaches at large dimensions.
- **An unexpected finding in the iterative solver experiments**: the dominant eigenspace of the standard Jacobi method contains a mixture of high- and low-frequency functions, causing learning to fail entirely. Switching to damped Jacobi ($\omega=0.9$) yields a purely low-frequency eigenspace, dramatically reducing learning difficulty—demonstrating that problem formulation has a decisive impact on learnability.

## Limitations & Future Work

- The optimal redundancy dimension $r - k$ for subspace embedding lacks an automatic selection mechanism; manual grid search is currently required.
- Validation is limited to linear subspaces; extension to nonlinear manifolds (e.g., constrained approximation on the Stiefel manifold) has not been addressed.
- Results are reported as single runs without error bars (though variance analysis in the appendix suggests the impact is small).
- All learned representations are substantially less efficient than the optimal subspace; whether this fundamental efficiency gap can be closed remains an open question.

## Related Work & Insights

- **vs. Grassmann manifold interpolation**: Classical methods fail in high-dimensional parameter spaces due to sparse observations; the proposed approach overcomes the curse of dimensionality via regression with neural networks.
- **vs. direct eigenvector prediction**: $\mathbb{Z}_2$-adjusted $\ell_2$ handles sign ambiguity but cannot exploit subspace structure to avoid the combinatorial explosion characterized in Theorem 3.
- **vs. neural operators (FNO/DeepONet)**: Complementary relationship—neural operators directly predict PDE solutions, while this work predicts reduced bases for solving a reduced model.
- **vs. DeepPOD**: Shares a similar spirit but uses a projection loss to extract bases directly from snapshot matrices; subspace regression matches or slightly surpasses it in accuracy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Outstanding theoretical contributions in formalizing the subspace regression problem, introducing the embedding technique, and providing the complexity analysis of Theorem 3.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across diverse application scenarios with comprehensive baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretically rigorous, with well-defined mathematical notation and clear structure.
- Value: ⭐⭐⭐⭐⭐ Provides a powerful new tool for reduced order modeling; the subspace embedding idea offers broad methodological inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[NeurIPS 2025\] Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers](../../NeurIPS2025/scientific_computing/symbolic_regression_is_all_you_need_from_simulations_to_scaling_laws_in_binary_n.md)
- [\[ICLR 2026\] Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)
- [\[AAAI 2026\] Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](../../AAAI2026/scientific_computing/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)

</div>

<!-- RELATED:END -->

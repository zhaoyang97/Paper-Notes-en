---
title: >-
  [Paper Note] Deep Learning for Subspace Regression
description: >-
  [ICLR2026][Physics & Scientific Computing][subspace regression] This work formalizes the subspace prediction problem in reduced order modeling (ROM) as a regression task on the Grassmann manifold. It proposes specialized loss functions and a subspace embedding technique—predicting a subspace of higher dimension than the target to reduce mapping complexity—achieving significant results in eigenvalue problems, parametric PDEs, and iterative method acceleration.
tags:
  - "ICLR2026"
  - "Physics & Scientific Computing"
  - "subspace regression"
  - "Grassmann manifold"
  - "reduced order modeling"
  - "neural operator"
  - "eigenspace"
date: 2026-05-08
content_hash: 3e9930457f1ec470
---

# Deep Learning for Subspace Regression

**Conference**: ICLR2026  
**arXiv**: [2509.23249](https://arxiv.org/abs/2509.23249)  
**Code**: [GitHub](https://github.com/VLSF/subreg)  
**Area**: Scientific Computing  
**Keywords**: subspace regression, Grassmann manifold, reduced order modeling, neural operator, eigenspace

## TL;DR

This work formalizes the subspace prediction problem in reduced order modeling (ROM) as a regression task on the Grassmann manifold. It proposes specialized loss functions and a subspace embedding technique—predicting a subspace of higher dimension than the target to reduce mapping complexity—achieving significant results in eigenvalue problems, parametric PDEs, and iterative method acceleration.

## Background & Motivation

**Background**: Reduced Order Modeling (ROM) simplifies system analysis and simulation by identifying linear subspaces to discard uninformative degrees of freedom. This approach is highly valuable in scenarios involving repeated solutions, such as parametric partial differential equations (PDEs), optimal control, and uncertainty quantification. Typical methods like Proper Orthogonal Decomposition (POD) construct a reduced basis by finding the best low-rank approximation to solutions for representative parameters.

**Limitations of Prior Work**: When the subspace explicitly depends on parameters (i.e., local POD), it is necessary to "interpolate" from subspaces corresponding to known parameters to those of unknown parameters. However, classical interpolation methods on Grassmann manifolds (such as Riemannian normal coordinate interpolation) are highly unreliable in high-dimensional parameter spaces—sparse observations lead to poor tangent space approximations, and computational costs grow rapidly with dimensionality.

**Key Challenge**: Subspace data possesses a specific algebraic structure—right $\text{GL}(k)$ invariance, meaning matrices $W_1$ and $W_2$ are equivalent if they share the same column space. This makes standard $\ell_2$ regression losses completely inapplicable. Simultaneously, the complexity of parametric eigenvalue problems (the number of different subspace configurations) grows sharply with parameter dimension and eigenvalue index; directly learning individual eigenvectors faces a combinatorial explosion.

**Goal**: This paper relaxes the interpolation problem into a regression problem, designing loss functions that satisfy $\text{GL}(k)$ invariance and using neural networks to parameterize the mapping from high-dimensional parameters to subspaces. The core innovation is the introduction of **subspace embedding**—predicting a larger-dimensional subspace that contains the target subspace. This is theoretically proven to reduce the derivative norm of the mapping (making it smoother), thereby aligning with the F-principle (spectral bias) of neural networks and significantly improving accuracy.

## Method

### Overall Architecture

This work reformulates the "parameter-to-subspace" interpolation challenge as a regression problem on the Grassmann manifold. Given a dataset $\mathcal{D} = \{(r_i, V_i)\}_{i=1}^m$, where $r \in \mathbb{R}^p$ is the parameter and $V(r) \in \text{Gr}(k, n)$ is the corresponding $k$-dimensional subspace, a network $Y_\theta: \mathbb{R}^p \to \text{Gr}(r, n)$ (parameterized by a Factorized Fourier Neural Operator, FFNO) is used to approximate this mapping, allowing for a predicted dimension $r \geq k$. The objective is to minimize a loss function that tolerates redundant subspace representations: $\theta^\star = \arg\min_\theta \frac{1}{m} \sum_i L(Y_\theta(r_i), V_i)$. The three components of the methodology are: an invariant loss function for subspace equivalence classes, an embedding trick that trades smoothness for higher prediction dimensions, and a modeling perspective that unifies eigenvalue problems, local POD, and iterative acceleration.

### Key Designs

**1. Subspace Loss Functions: Turning $\text{GL}(k)$ Invariance into an Optimizable Objective**

Subspace data has a complex algebraic structure where two matrices are equivalent if their column spaces are identical (right $\text{GL}(k)$ invariance). Element-wise $\ell_2$ loss would treat different bases of the same subspace as different targets. This paper requires the loss $L$ to satisfy two conditions: invariance to equivalence classes $L(A,B) = L(\tilde{A}, \tilde{B})$, and $L(A,B)=0$ if and only if $\mathcal{S}(B) \subset \mathcal{S}(A)$ (note the inclusion rather than equality, which facilitates the embedding trick). Theorem 1 provides two constructions. The deterministic loss $L_1(A, B) = p - \|Q_B^\top Q_A\|_F^2$ measures the difference between orthogonal projectors via QR decompositions ($A = Q_A R_A$, $B = Q_B R_B$). The randomized loss $L_2(A, B; z) = \min_u \|Au - Q_B z\|_2^2$ ($z \sim \mathcal{N}(0, I_k)$) replaces the projector with a least-squares problem and uses Hutchinson trace estimation to avoid the second QR decomposition, being strictly equal to the former in expectation: $\mathbb{E}_z[L_2] = L_1$. $L_2$ can be solved directly via normal equations, offering superior training efficiency for large subspace dimensions, though it may suffer from ill-conditioning. A stabilized version, $L_2^{\text{stab}}$ using Cholesky-QR2, is introduced to maintain numerical precision.

**2. Subspace Embedding: Trading Redundant Dimensions for Mapping Smoothness**

This is the central innovation of the paper. The idea is simple: since the loss only requires $\mathcal{S}(V) \subset \mathcal{S}(Y_\theta)$, the network can predict a larger $r > k$ dimensional subspace that contains the target. While redundant, this improves both optimization and approximation. Theorem 2 proves that for any continuously differentiable $V: \mathbb{R} \to \text{Gr}(k, n)$, one can construct a $W: \mathbb{R} \to \text{Gr}(r, n)$ such that $\|\dot{W}(t)\|_F^2 \leq \|\dot{V}(t)\|_F^2$, with strict inequality where the derivative is non-zero. Thus, embedding into a larger subspace makes the mapping smoother, satisfying the F-principle where networks favor low-frequency smooth functions. Theorem 3 uses combinatorics to explain the difficulty of direct learning: for elliptic eigenvalue problems $-\sum a_i \partial^2 \phi / \partial x_i^2 = \lambda \phi$, the mapping to the $k$-th eigenvector is piecewise constant, with the number of regions $\#_{F_k}(k, D) \sim \frac{1}{(D-1)!} k(\log k)^{D-1}$ growing with dimension $D$. Embedding allows merging these regions by utilizing different combinations of eigenvectors. In the extreme case where the predicted dimension reaches the total number of possible eigenvectors $\#_{F_k}$, the mapping degenerates into a constant function, which is the easiest to learn. This explains the experimental observation of error dropping from 30% to 2% when increasing prediction dimension from 10 to 40.

**3. Unified Multi-Scenario Coverage: A Single Regression Framework for ROM Tasks**

Abstracting the problem as "parameter $\to$ subspace" regression allows various ROM tasks to be unified under one framework, where only the physical meaning of the subspace differs. In parametric eigenvalue problems, the subspace is $\text{span}\{\phi_1, \dots, \phi_K\}$ formed by the first $K$ eigenfunctions. In local POD for the Burgers equation, it is the reduced basis $\{\psi_1, \dots, \psi_k\}$ varying with PDE parameters. In conjugate gradient deflation, it is the eigenspace $\mathcal{S}(V)$ of small eigenvalues. In Jacobi coarse-grid correction, it is the principal eigenspace of the error propagation matrix. In balanced truncation for optimal control, it is the reduced basis $\mathcal{S}(\bar{\mathcal{T}})$ for controllability and observability.

| Application | Mapping | Subspace Meaning |
| :--- | :--- | :--- |
| Parametric Eigenvalue Problem | $U(x) \to \text{span}\{\phi_1, \dots, \phi_K\}$ | First $K$ eigenfunctions |
| Local POD (Burgers Eq.) | PDE parameters $\to \{\psi_1, \dots, \psi_k\}$ | POD reduced basis |
| Conjugate Gradient Deflation | $k(x) \to \mathcal{S}(V)$ | Small eigenvalue eigenspace |
| Coarse Grid Correction (Jacobi) | $k(x) \to \mathcal{S}(V)$ | Principal eigenspace of error matrix |
| Balanced Truncation | $A, B, C \to \mathcal{S}(\bar{\mathcal{T}})$ | Controllability/observability basis |

## Key Experimental Results

### Quantum Mechanics Eigenspace Prediction

| Dataset | Riemannian Interpolation | $\mathbb{Z}_2$-adjusted $\ell_2$ | Subspace Regression $L_1$ |
| :--- | :---: | :---: | :---: |
| $D=1$ Schrödinger | 4.69% | 2.33% | **0.09%** |
| $D=2$, Dataset a | 31.9% | 19.52% | **0.65%** |
| $D=2$, Dataset b | 92.64% | 48.56% | **15.58%** |

Subspace regression significantly outperforms classical interpolation and direct eigenvector prediction across all datasets.

### Subspace Embedding Effect ($D=2$ Elliptic Problem, First 10 Eigenvectors)

| Predicted Subspace Dim $N_{\text{sub}}$ | Test Error | Description |
| :---: | :---: | :--- |
| 10 (No Embedding) | ~30% | Same as target dimension |
| 20 | ~10% | Embedding begins to take effect |
| 30 | ~5% | Significant improvement |
| 40 | **~2%** | Only 0.4% of total degrees of freedom |

Embedding reduces test error from 30% to 2% while systematically narrowing the generalization gap.

### Loss Function Comparison ($D=3$ Elliptic Problem)

| $N_{\text{sub}}$ | $L_1(A,B)$ | $L_2(A,B;z)$ | $L_2^{\text{stab}}(A,B;z)$ |
| :---: | :---: | :---: | :---: |
| 6 | 24.77% | 31.46% | 28.28% |
| 12 | 13.69% | 17.12% | 15.88% |
| 24 | 9.71% | Failed | 9.49% |
| 48 | 7.54% | 16.3% | **7.4%** |

$L_2$ becomes unstable at higher dimensions due to ill-conditioned normal equations; Cholesky-QR2 stabilization restores performance to be slightly better than $L_1$.

## Highlights & Insights

- The **subspace embedding** idea is elegant: it exploits the monotonicity of subspace inclusion to trade redundancy for smoothness, perfectly matching the neural network's spectral bias (F-principle). The cost is a larger learned dimension than optimal, but the extra computation in the reduced model is negligible.
- The **complexity characterization in Theorem 3** links combinatorics with approximation theory, providing a precise measure of difficulty for subspace learning.
- The **randomized loss $L_2$** introduces Hutchinson trace estimation to subspace learning, with training times significantly outperforming QR-based schemes at large dimensions.
- **Unexpected finding in iterative methods**: The principal eigenspace of the standard Jacobi method contains a mix of high and low frequencies, causing learning to fail. Switching to damped Jacobi ($\omega=0.9$) shifts this to pure low frequencies, drastically reducing learning difficulty—showing that problem formulation is decisive for learning feasibility.

## Limitations & Future Work

- There is currently no automatic selection mechanism for the optimal redundancy dimension $r - k$; it requires manual grid search.
- The method is only validated on linear subspaces; extensions to non-linear manifolds (e.g., constrained approximation on Stiefel manifolds) have not been addressed.
- Results are from single runs without error bars (though appendix variance analysis suggests minor impact).
- Representations learned by neural networks remain significantly less efficient than optimal subspaces; whether this fundamental efficiency gap can be closed remains an open question.

## Related Work & Insights

- **vs. Grassmann Manifold Interpolation**: Classical methods fail in high-dimensional parameter spaces due to sparse observations; this work overcomes the curse of dimensionality via regression and neural networks.
- **vs. Direct Eigenvector Prediction**: While $\mathbb{Z}_2$-adjusted $\ell_2$ handles sign ambiguity, it cannot utilize subspace structures to avoid the combinatorial explosion described in Theorem 3.
- **vs. Neural Operators (FNO/DeepONet)**: Complementary relationship—Neural Operators predict PDE solutions directly, while this work predicts reduced bases to solve reduced models.
- **vs. DeepPOD**: Similar approach but uses projection loss to extract bases from snapshot matrices; subspace regression matches or slightly exceeds its accuracy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Strong theoretical contribution through subspace regression formulation, embedding, and Theorem 3.
- Experimental Thoroughness: ⭐⭐⭐⭐ Wide coverage of application scenarios with comprehensive baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theory, standard mathematical notation, and clear structure.
- Value: ⭐⭐⭐⭐⭐ Provides a powerful new tool for ROM; the subspace embedding concept has broad methodological implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials](advancing_universal_deep_learning_for_electronic-structure_hamiltonian_predictio.md)
- [\[ICLR 2026\] Accelerating Eigenvalue Dataset Generation via Chebyshev Subspace Filter](accelerating_eigenvalue_dataset_generation_via_chebyshev_subspace_filter.md)
- [\[ICLR 2026\] GenSR: Symbolic Regression based on Equation Generative Space](gensr_symbolic_regression_based_on_equation_generative_space.md)
- [\[NeurIPS 2025\] Toward Complete Merger Identification at Cosmic Noon with Deep Learning](../../NeurIPS2025/physics/toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)
- [\[ICLR 2026\] Efficient Regression-based Training of Normalizing Flows for Boltzmann Generators](efficient_regression-based_training_of_normalizing_flows_for_boltzmann_generator.md)

</div>

<!-- RELATED:END -->

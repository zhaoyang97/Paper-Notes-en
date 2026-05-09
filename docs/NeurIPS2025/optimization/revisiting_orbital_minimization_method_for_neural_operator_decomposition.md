---
title: >-
  [Paper Note] Revisiting Orbital Minimization Method for Neural Operator Decomposition
description: >-
  [NeurIPS 2025][Optimization][Spectral Decomposition] This paper revisits the classical Orbital Minimization Method (OMM) originating from computational chemistry, provides a concise linear-algebraic consistency proof, reveals its deep connections to Sanger's rule and streaming PCA, and generalizes it into a unified framework for training neural networks to perform spectral decomposition of positive semidefinite operators.
tags:
  - NeurIPS 2025
  - Optimization
  - Spectral Decomposition
  - Orbital Minimization Method
  - Eigenfunction Learning
  - Neural Operator
  - Self-Supervised Learning
date: 2026-05-08
content_hash: 4dc083fead906146
---

# Revisiting Orbital Minimization Method for Neural Operator Decomposition

**Conference**: NeurIPS 2025
**arXiv**: [2510.21952](https://arxiv.org/abs/2510.21952)
**Code**: [GitHub](https://github.com/jongharyu/operator-omm)
**Area**: Optimization
**Keywords**: Spectral Decomposition, Orbital Minimization Method, Eigenfunction Learning, Neural Operator, Self-Supervised Learning

## TL;DR

This paper revisits the classical Orbital Minimization Method (OMM) originating from computational chemistry, provides a concise linear-algebraic consistency proof, reveals its deep connections to Sanger's rule and streaming PCA, and generalizes it into a unified framework for training neural networks to perform spectral decomposition of positive semidefinite operators.

## Background & Motivation

Spectral decomposition of linear operators is a foundational tool in machine learning and scientific computing. Recent years have seen growing interest in approximating eigenfunctions of operators via neural networks, with breakthroughs in quantum chemistry, reinforcement learning, and PDE solving. However, many existing methods rely on **surrogate losses or architectural constraints**, lacking a clear variational foundation, which leads to fragile optimization or poor scalability.

The classical multi-column Rayleigh quotient maximization problem is complicated by **orthogonality constraints**. Its unconstrained formulation requires matrix inversion $(V^\top V)^{-1}$, which is numerically unstable when $V$ is rank-deficient. Many modern approaches propose complex alternatives (e.g., augmented Lagrangian methods such as ALLO, VICReg), but these are computationally expensive and require extensive hyperparameter tuning.

OMM was originally proposed by Mauri, Ordejon, and others in the 1990s for electronic structure calculations, providing an approach to approximate eigenspaces without explicit orthogonalization. The key objective is:

$$\mathcal{L}_{\text{omm}}(V) = -\text{tr}((2I_k - V^\top V) V^\top A V)$$

However, its theoretical derivation is obscure and has remained confined to the computational chemistry literature. This paper aims to establish a cleaner theoretical foundation for OMM and reveal its broad applicability.

## Method

### Overall Architecture

A central finding is that the OMM objective can be equivalently rewritten as:

$$\mathcal{L}_{\text{omm}}(V) = \text{tr}((I_d - VV^\top)^2 A) - \text{tr}(A)$$

This minimizes the trace of the squared residual projection matrix $(I - VV^\top)$ weighted by $A$. The formulation is intuitive and elegant: when $V$ is an orthonormal basis, $VV^\top$ is the subspace projector, and the residual is naturally minimized.

### Key Designs

1. **OMM-p Higher-Order Generalization**: The above form is naturally extended to $\mathcal{L}_{\text{omm}}^{(p)}(V) = \text{tr}((I_d - VV^\top)^{2p} A) - \text{tr}(A)$. The authors prove (Theorem 1) that for any $p \geq 1$, the global minimum equals the negative sum of the top-$k$ eigenvalues, and the optimal solution recovers the top-$k$ eigensubspace. The key insight is that the objective depends only on $VV^\top$, enabling reparameterization via SVD. OMM has no spurious local minima.

2. **Nesting Techniques**: To learn ordered eigenvectors, two nesting strategies are proposed:

   - **Joint Nesting (OMMjnt)**: Minimizes the weighted objective $\sum_{i=1}^k \alpha_i \mathcal{L}_{\text{omm}}^{(p)}(V_{1:i})$, efficiently implemented via matrix masking.
   - **Sequential Nesting (OMMseq)**: Uses stop-gradient to define a surrogate objective such that $\partial_{v_i} \mathcal{L}_{\text{omm}}^{\text{seq}} = \partial_{v_i} \mathcal{L}_{\text{omm}}^{(1)}(V_{1:i})$.

3. **Connection to Sanger's Rule**: The sequential nesting gradient of OMM is a symmetrized version of the Sanger update. The Sanger update $(I - V_{1:i}V_{1:i}^\top)A v_i$ is not itself the gradient of any function, whereas OMM naturally recovers this form from a well-defined objective. This connection is remarkable — a classical method from computational chemistry shares a deep relationship with the core algorithm of streaming PCA.

4. **Operator Extension and Inverse Operator Trick**: OMM is generalized to the infinite-dimensional setting by replacing matrix products with second-moment operator matrices. For operators with unbounded spectra (e.g., harmonic oscillators), an inverse operator trick is proposed: parameterize $\mathbf{f} = \mathcal{L}\mathbf{g}$ and apply OMM to $\mathcal{L}^{-1}$, avoiding explicit inversion.

### Loss & Training

- OMM-1 objective: $-2\text{tr}(M_\rho[\mathbf{f}, \mathcal{T}\mathbf{f}]) + \text{tr}(M_\rho[\mathbf{f}] M_\rho[\mathbf{f}, \mathcal{T}\mathbf{f}])$
- Optimized using Adam with learning rate $10^{-3}$
- For numerical instability (small eigenvalues), spectral shifting $A + \kappa I$ or the Sanger variant is applied
- Regularization is equivalent to spectral shifting: $\kappa\|V^\top V - I_k\|_F^2$ does not alter the global optimum

## Key Experimental Results

### Laplacian Representation Learning in Reinforcement Learning

| Environment | OMMseq | OMMjnt | ALLO | Notes |
|---|---|---|---|---|
| GridMaze-11 | ~0.95 | ~0.95 | ~0.94 | OMM matches ALLO without hyperparameter tuning |
| GridMaze-26 | ~0.75 | ~0.75 | ~0.70 | Small spectral gap makes this challenging |
| GridRoom-32 | ~0.80 | ~0.82 | ~0.85 | All methods struggle at degenerate eigenvalues |
| GridRoom-64 | ~0.60 | ~0.65 | ~0.65 | Difficult instance with small spectral gap |

### Self-Supervised Contrastive Learning (CIFAR-100)

| Method | w/ Projector Top-1 | w/ Projector Top-5 | DirectCLR Top-1 | DirectCLR Top-5 |
|---|---|---|---|---|
| OMM (p=1) | 60.02 | 87.13 | 59.83 | 86.65 |
| OMMjnt (p=1) | 61.30 | 85.22 | 59.92 | 85.13 |
| OMM (p=2) | 63.92 | 89.08 | 61.27 | 87.07 |
| OMM (p=1)+(p=2) | **64.77** | **89.18** | **63.99** | **88.88** |
| SimCLR | ~66.50 | - | - | - |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| OMM vs. LoRA (Schrödinger equation) | Sanger variant matches LoRAseq | Avoids OMM instability on rapidly decaying spectra |
| Higher-order p=2 vs. p=1 | ~4% Top-1 gain | Higher-order OMM provides additional gradient to escape premature flat minima |
| Joint nesting vs. sequential nesting | Comparable or sequential slightly better | Sequential nesting is more general |

### Key Findings

- OMM matches carefully tuned ALLO in RL representation learning without any hyperparameter tuning
- Higher-order OMM ($p=2$) yields significant gains in self-supervised learning, approaching SimCLR performance
- The Sanger variant is more numerically stable than standard OMM when eigenvalues decay rapidly
- OMM can outperform LoRA on sparse matrices such as graph Laplacians

## Highlights & Insights

- **A textbook example of rediscovery**: A method from 1990s computational chemistry is shown to be highly relevant to modern machine learning
- The proof is concise and elegant — the consistency result follows from elementary linear algebra alone
- The property that "regularization is equivalent to spectral shifting" in OMM is particularly distinctive and useful
- The paper reveals deep connections among seemingly unrelated fields: streaming PCA, operator learning, and RL representation learning

## Limitations & Future Work

- OMM applies only to **positive semidefinite operators**; spectral shifting is required for non-PSD cases
- The inverse operator trick remains numerically unstable for certain PDE problems
- Performance in self-supervised learning has not yet reached SimCLR levels
- Extending OMM to SVD decomposition of non-symmetric operators remains unexplored

## Related Work & Insights

- Complements NeuralSVD/LoRA approaches: OMM approximates the projection operator, while LoRA approximates the operator itself
- Provides a variational foundation for streaming PCA, which Sanger's rule itself lacks
- A broader question raised: how many forgotten classical methods from computational chemistry and physics remain to be rediscovered by modern machine learning?

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Rigorously formalizes a rediscovered classical method and reveals cross-domain connections
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers RL, PDE, and SSL tasks, though SSL performance does not reach SOTA
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and concise; historical context is well-situated
- Value: ⭐⭐⭐⭐⭐ Provides a unified variational framework for spectral decomposition with broad impact

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning Single-Index Models via Harmonic Decomposition](learning_single-index_models_via_harmonic_decomposition.md)
- [\[NeurIPS 2025\] Extragradient Method for $(L_0, L_1)$-Lipschitz Root-finding Problems](extragradient_method_for_l_0_l_1-lipschitz_root-finding_problems.md)
- [\[NeurIPS 2025\] OrthoGrad Improves Neural Calibration](orthograd_improves_neural_calibration.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)
- [\[NeurIPS 2025\] Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations](contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)

<!-- RELATED:END -->

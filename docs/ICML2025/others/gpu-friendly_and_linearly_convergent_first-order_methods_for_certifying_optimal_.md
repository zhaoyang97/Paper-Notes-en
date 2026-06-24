---
title: >-
  [Paper Note] GPU-friendly and Linearly Convergent First-order Methods for Certifying Optimal $k$-sparse GLMs
description: >-
  [ICML2025][Sparse GLM] Proposes GPU-friendly and linearly convergent first-order methods. Through composite reformulation and a dual gap restart strategy, the perspective relaxation solving is accelerated by 1-2 orders of magnitude, enabling optimality certification for large-scale sparse GLMs.
tags:
  - "ICML2025"
  - "Sparse GLM"
  - "perspective relaxation"
  - "branch-and-bound"
  - "linear convergence"
  - "GPU acceleration"
  - "dual gap restart"
  - "proximal methods"
date: 2026-05-08
content_hash: ea819af3aeeae4bd
---

# GPU-friendly and Linearly Convergent First-order Methods for Certifying Optimal $k$-sparse GLMs

**Conference**: ICML2025  
**arXiv**: [2603.01306](https://arxiv.org/abs/2603.01306)  
**Code**: To be confirmed  
**Area**: Others  
**Keywords**: Sparse GLM, perspective relaxation, branch-and-bound, linear convergence, GPU acceleration, dual gap restart, proximal methods

## TL;DR

Proposes GPU-friendly and linearly convergent first-order methods. Through composite reformulation and a dual gap restart strategy, the perspective relaxation solving is accelerated by 1-2 orders of magnitude, enabling optimality certification for large-scale sparse GLMs.

## Background & Motivation

- **Sparse GLMs** are core tools in machine learning and statistics, widely applied in high-stakes scenarios such as healthcare and finance, which demand certifiable global optimal solutions rather than approximations.
- The problem is essentially an NP-hard $\ell_0$-constrained optimization, and the standard method is the branch-and-bound (BnB) framework.
- The core bottleneck of BnB lies in the **lower bound computation** at each node: the standard big-M relaxation is too loose, while the perspective relaxation is tighter but computationally expensive to solve.
- **Interior point methods (IPMs)** for solving the perspective relaxation suffer from cubic complexity, cannot be parallelized on GPUs, and do not support warm-start.
- Existing **first-order methods** (ADMM, coordinate descent) suffer from two issues:
    1. The convergence rate is only sublinear, and the convergence rate of the safe lower bound lacks theoretical guarantees.
    2. ADMM requires solving linear systems, and coordinate descent is inherently serial, making both unsuitable for GPUs.

## Method

### 1. Composite Reformulation

Reformulates the perspective relaxation as an unconstrained composite optimization problem:

$$\inf_{\boldsymbol{\beta} \in \mathbb{R}^p} \left\{ \Phi(\boldsymbol{\beta}) := F(\boldsymbol{X}\boldsymbol{\beta}) + G(\boldsymbol{\beta}) \right\}$$

where $F(\boldsymbol{X}\boldsymbol{\beta}) = f(\boldsymbol{X}\boldsymbol{\beta}, \boldsymbol{y})$ is the loss function, and $G(\boldsymbol{\beta}) = 2\lambda_2 g_{\mathcal{N}}(\boldsymbol{\beta})$ is the implicit perspective regularization term. The key lies in defining the implicit regularization function:

$$g_{\mathcal{N}}(\boldsymbol{\beta}) := \inf_{\boldsymbol{z}} \left\{ \frac{1}{2}\sum_{j} \beta_j^2 / z_j : (\boldsymbol{\beta}, \boldsymbol{z}) \in \mathcal{D} \right\}$$

which encodes both the perspective terms $\beta_j^2/z_j$ and the cardinality/branching constraints into a unified regularizer.

### 2. Geometric Analysis: Primal Quadratic Growth + Dual Quadratic Decay

Proves key geometric properties under regularity conditions (firm convexity, polyhedral subdifferential):

- **Primal Quadratic Growth**: $\Phi(\boldsymbol{\beta}) \geq \Phi^\star + \frac{\alpha}{2}\text{dist}^2(\boldsymbol{\beta}, \mathcal{B}^\star)$
- **Dual Quadratic Decay** (proposed for the first time in this paper): $-\Psi(\boldsymbol{\zeta}) \leq -\Psi^\star + \frac{\kappa}{2}\|\boldsymbol{\zeta} - \boldsymbol{\zeta}^\star\|^2$

These two form a symmetric relationship. It further proves that the primal optimality gap strictly controls the dual error:

$$\frac{\sigma}{2}\|\boldsymbol{\zeta} - \boldsymbol{\zeta}^\star\|^2 \leq \Phi(\boldsymbol{\beta}) - \Phi^\star, \quad \Psi^\star - \Psi(\boldsymbol{\zeta}) \leq \frac{\kappa}{\sigma}(\Phi(\boldsymbol{\beta}) - \Phi^\star)$$

### 3. Dual Gap Restart Scheme

Designs a restart strategy based on the Fenchel dual gap $\Delta(\boldsymbol{\beta}, \boldsymbol{\zeta}) = \Phi(\boldsymbol{\beta}) - \Psi(\boldsymbol{\zeta})$:

- Restarts accelerated methods like FISTA from the current iteration.
- Triggers a restart when the dual gap drops below a threshold.
- Upgrades the sublinear $O(1/k^2)$ accelerated method to **linear convergence**.
- Applicable to various proximal method variants, including fixed step size, line search, and line-search-free schemes.
- This framework is general and not limited to sparse GLMs.

### 4. Efficient GPU Implementation

For the implicit perspective regularizer:

- Proves that both its **function value and proximal operator** can be computed exactly in $O(p \log p)$ time (sorting + thresholding).
- Avoids expensive conic programming solvers.
- The primary computational load in each iteration comes from **matrix-vector multiplications** $\boldsymbol{X}\boldsymbol{\beta}$ and $\boldsymbol{X}^\top\boldsymbol{\zeta}$, which naturally fit GPU parallelization.

## Key Experimental Results

| Experimental Setup | Baselines | Speedup of Ours |
|---------|---------|-------------|
| Synthetic Data CPU Lower Bound Computation | IPM / ADMM / CD | **1-2 orders of magnitude** |
| Real Data CPU Lower Bound Computation | IPM / ADMM / CD | **1-2 orders of magnitude** |
| GPU vs CPU (Ours) | CPU version of Ours | An additional **1 order of magnitude** |
| Large-scale BnB Certification | Standard Solvers | Significantly improved BnB scalability |

- Fast convergence of the dual lower bound is validated on both synthetic and real datasets.
- GPU acceleration makes BnB certification feasible on large-scale instances.
- The linear convergence rate of the safe lower bound in experiments aligns with theoretical predictions.

## Highlights & Insights

1. The concept of **dual quadratic decay** is proposed for the first time, revealing a symmetric structure in primal-dual geometry.
2. The **dual gap restart** is a general framework that can upgrade any sublinear proximal method to linear convergence, with applicability extending far beyond sparse GLMs.
3. The $O(p\log p)$ exact computation of the implicit perspective regularizer avoids general conic solvers, which is key to GPU-friendliness.
4. A complete closed loop from theory to engineering: geometric analysis $\to$ convergence guarantees $\to$ efficient implementation $\to$ GPU acceleration $\to$ BnB integration.
5. Comparison with FDPG: while FDPG starts from the dual side and only achieves sublinear primal convergence, this work starts from the primal side and achieves bilinear convergence.

## Limitations & Future Work

1. **Geometric regularity conditions** (e.g., firm convexity) might not hold under certain non-standard loss functions, limiting its scope of applicability.
2. The constants in linear convergence $\alpha, \kappa, \sigma$ depend heavily on the specific problem structure, and the theoretical bounds might not be tight.
3. Experiments primarily focus on sparse linear/logistic regression; the performance on **more complex GLMs** (such as Poisson regression) has not been fully verified.
4. The GPU implementation relies on the matrix $\boldsymbol{X}$ being fully loaded into GPU memory, which might be restricted for **ultra-large-scale features**.
5. The branching strategy in the BnB framework still employs standard schemes, leaving room for further co-optimization with first-order methods.

## Related Work & Insights

- **MIP for ML**: Applying mixed-integer programming (MIP) to interpretable ML, classification/regression, decision trees, etc.
- **Perspective relaxation**: Pioneered by Ceria & Soares (1999), this paper advances it to highly efficient and solvable forms.
- **GPU-accelerated optimization**: Complementary to GPU-LP methods like cuPDLP (Lu et al., 2024), but this work directly handles non-linear objectives.
- The conference version of Liu et al. (2025) only handled the root node, whereas this work extends to every node of BnB and adds GPU experiments.

## Rating

- Novelty: ⭐⭐⭐⭐ — The dual quadratic decay and gap restart framework possess strong theoretical originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dimensional validation across synthetic, real, GPU, and BnB implementations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematically rigorous, well-structured, and motivated thoroughly.
- Value: ⭐⭐⭐⭐ — Provides a practical and theoretically guaranteed solution for the optimality certification of large-scale sparse optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Transformers for Generalizable First-Order Logical Entailment](../../ACL2025/others/enhancing_fol_entailment.md)
- [\[ICML 2025\] Modern Methods in Associative Memory](modern_methods_in_associative_memory.md)
- [\[CVPR 2025\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](../../CVPR2025/others/rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)
- [\[CVPR 2025\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2025/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[ICML 2025\] Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena](position_solve_layerwise_linear_models_first_to_understand_neural_dynamical_phen.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Joker: Joint Optimization Framework for Lightweight Kernel Machines
description: >-
  [ICML2025][Model Compression][Kernel methods] The Joker framework is proposed, which achieves unified and efficient training of various large-scale kernel models (KRR / KLR / SVM, etc.) with ~2GB of memory through dual block coordinate descent with trust region (DBCD-TR) and random Fourier feature approximation, achieving memory savings of up to 90% without performance loss.
tags:
  - "ICML2025"
  - "Model Compression"
  - "Kernel methods"
  - "Large-scale learning"
  - "Dual optimization"
  - "Block coordinate descent"
  - "Trust region method"
  - "Random Fourier features"
  - "Low memory"
date: 2026-05-08
content_hash: d30c412a68403516
---

# Joker: Joint Optimization Framework for Lightweight Kernel Machines

**Conference**: ICML2025  
**arXiv**: [2505.17765](https://arxiv.org/abs/2505.17765)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: Kernel methods, Large-scale learning, Dual optimization, Block coordinate descent, Trust region method, Random Fourier features, Low memory

## TL;DR

The Joker framework is proposed, which achieves unified and efficient training of various large-scale kernel models (KRR / KLR / SVM, etc.) with ~2GB of memory through dual block coordinate descent with trust region (DBCD-TR) and random Fourier feature approximation, achieving memory savings of up to 90% without performance loss.

## Background & Motivation

Kernel methods are a classic paradigm in non-linear learning with a solid theoretical foundation, and their connection to deep learning (such as Neural Tangent Kernel) has further highlighted their potential recently. However, large-scale kernel methods face two main bottlenecks:

**Excessive Memory Overhead**: Current SOTA methods like Falkon are based on Nyström approximation + Cholesky preconditioning, which requires $O(M^2)$ memory to store the preconditioner ($M$ is the number of Nyström centers). On the HIGGS dataset with $M=1.2 \times 10^5$, it requires **>55GB** of memory, making it unaffordable for average users.

**Lack of Model Diversity**: Existing methods mainly focus on KRR, whereas Kernel Logistic Regression (KLR) and Support Vector Machines (SVM) lack efficient large-scale implementations. LIBSVM/ThunderSVM take more than a week to train, and LogFalkon also suffers from high memory issues.

The goal of Joker is: **to cover various kernel models with a unified optimization framework while drastically reducing memory requirements**.

## Method

### 1. Joint Optimization by Duality

For the primal problem of general kernel machines:

$$\min_{\boldsymbol{\theta}} \frac{\lambda}{2} \|\boldsymbol{\theta}\|_{\mathcal{H}}^2 + \sum_{i=1}^{n} \ell(y_i, \langle \boldsymbol{\theta}, \boldsymbol{\varphi}(\boldsymbol{x}_i) \rangle)$$

Through Fenchel conjugate $\xi_y^*(v)$, its dual form is derived (Theorem 1):

$$\min_{\boldsymbol{\alpha} \in \Omega} \frac{1}{2} \boldsymbol{\alpha}^\top \boldsymbol{K} \boldsymbol{\alpha} + \frac{1}{\lambda} \sum_{i=1}^{n} \xi_{y_i}^*(-\lambda \alpha_i)$$

**Key Properties**:
- The dual problem is always a **convex optimization** with simple **box constraints** $\alpha_i \in [\tau_i^L, \tau_i^U]$;
- Strong duality holds (Slater's condition), meaning the dual optimum is equivalent to the primal optimum;
- The dual Hessian depends linearly on $\boldsymbol{K}$, whereas the primal Hessian depends quadratically on $\boldsymbol{K}$, so the **dual optimization converges faster**;
- Through Proposition 2 (conjugate decomposition of infimal convolution), compound loss functions like Huber loss can be easily handled.

Different loss functions correspond to different kernel models, unified under the same dual framework:

| Model | Loss Function | Dual Conjugate $\xi_y^*(v)$ |
|------|---------|----------------------|
| KRR | Squared loss $(y-u)^2/2$ | $v^2/2 + vy$ |
| KLR | Logistic loss | Binary entropy $\text{bEnt}(-vy)$ |
| L1-SVC | Hinge loss | $vy$, $-1 \le vy \le 0$ |
| L2-SVC | Squared Hinge loss | $v^2/2 + vy$, $vy \le 0$|
| SVR | $\varepsilon$-insensitive loss | $\varepsilon|v| + vy$, $-1 \le v \le 1$ |
| Huber Regression | Huber loss | $v^2/2 + vy$, $-\delta \le v \le \delta$ |

### 2. DBCD-TR Solver

The core solver is **Dual Block Coordinate Descent with Trust Region (DBCD-TR)**:

**Block Coordinate Descent**: At each step, a subset of indices $\mathcal{B} \subset [n]$ is selected, freezing the rest of the variables, to solve the subproblem:

$$\min_{\boldsymbol{\alpha}_{\mathcal{B}}} \frac{1}{2} \boldsymbol{\alpha}_{\mathcal{B}}^\top \boldsymbol{K}_{\mathcal{B},\mathcal{B}} \boldsymbol{\alpha}_{\mathcal{B}} + \boldsymbol{\alpha}_{\mathcal{B}_\complement}^\top \boldsymbol{K}_{\mathcal{B}_\complement,\mathcal{B}} \boldsymbol{\alpha}_{\mathcal{B}} + f(\boldsymbol{\alpha}_{\mathcal{B}}), \quad \text{s.t. } \boldsymbol{\tau}^L \le \boldsymbol{\alpha}_{\mathcal{B}} \le \boldsymbol{\tau}^U$$

**Trust Region Method**: Constructs a quadratic model $\mu_k(\boldsymbol{s}) = J(\boldsymbol{\alpha}_k) + \boldsymbol{g}_k^\top \boldsymbol{s} + \frac{1}{2} \boldsymbol{s}^\top \boldsymbol{Q}_k \boldsymbol{s}$, where $\boldsymbol{Q}_k = \boldsymbol{K}_{\mathcal{B},\mathcal{B}} + \nabla^2 f$. The step quality is evaluated through the ratio $\rho_k$ to adaptively adjust the trust region radius $\Delta_k$.

**Truncated CG-Steihaug**: When solving the trust region subproblem, a truncated conjugate gradient method (Algorithm 2) is used, which terminates early if box constraints are violated or the trust region boundary is exceeded, and finally projects back onto the feasible region. This is more efficient than projected Newton methods.

**Complexity**: Space complexity is $O(|\mathcal{B}|^2)$ (storing the sub-kernel matrix), and time complexity is $O(T_{\text{TR}} \cdot T_{\text{CG}} \cdot |\mathcal{B}|^2)$, where in practice $T_{\text{TR}} \le 50$, $T_{\text{CG}} \le 10$.

### 3. Inexact Joker (Random Fourier Feature Approximation)

Exact kernel evaluation of $\boldsymbol{K}_{\mathcal{B},:}\boldsymbol{\alpha}$ has a time complexity of $O(nd|\mathcal{B}|)$, which is unacceptable when $n \ge 10^6$. This is approximated via Random Fourier Features (RFF):

$$\boldsymbol{\psi}(\boldsymbol{x}) = \sqrt{\frac{2}{M}} \cos(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}), \quad \boldsymbol{w}_i \sim p_{\mathcal{K}}, \quad b_i \sim U_{[0,2\pi]}$$

By maintaining a low-dimensional weight vector $\boldsymbol{\theta} = \sum_i \alpha_i \boldsymbol{\psi}(\boldsymbol{x}_i) \in \mathbb{R}^M$, the kernel gradient computation is simplified to:

$$\boldsymbol{K}_{\mathcal{B},:}\boldsymbol{\alpha} = \boldsymbol{\psi}(\boldsymbol{X}_{\mathcal{B}})^\top \boldsymbol{\theta}$$

The time complexity is reduced from $O(nd|\mathcal{B}|)$ to $O(Md|\mathcal{B}|)$ ($M \ll n$), and the space complexity only requires storing $\boldsymbol{\theta} \in \mathbb{R}^M$.

## Key Experimental Results

Comparing performance on the HIGGS dataset ($n=11M$, $d=28$) using **a single RTX 3080 (10GB) GPU**:

| Method | Type | Memory | Time | Supported Models |
|------|------|------|------|---------|
| Falkon | Approximate kernel | >50GB | <1h | KRR |
| LogFalkon | Approximate kernel | >50GB | <1h | KLR |
| EigenPro3 | Approximate kernel | ~7GB | >15h | KRR |
| LIBSVM | Exact kernel | <2GB | >1 week | SVC/SVR |
| ThunderSVM | Exact kernel | ~8GB | >1 week | SVC/SVR |
| **Joker** | **Hybrid** | **~2GB** | **~1h** | **KRR/KLR/SVM, etc.** |

- **Memory savings up to 90%**: Reduced from >50GB to ~2GB compared with Falkon.
- Training time is comparable to Falkon (~1 hour) and far superior to LIBSVM/ThunderSVM (>1 week).
- Performance is on par with or even better than SOTA.
- The unified framework covers multiple models including KRR, KLR, L1-SVC, L2-SVC, SVR, Huber regression, $L_p$ regression, etc.

## Highlights & Insights

1. **Unified Dual Perspective**: Mapping kernel models with different losses into a structurally identical dual problem via Fenchel conjugate. The box constraints + separable structure are naturally suited for coordinate descent, making the design elegant.
2. **Embedding Trust Region into BCD**: Cleverly solves the difficulty of step-size tuning under box constraints without the need for manual parameter tuning. The CG-Steihaug + truncation strategy is highly efficient and robust.
3. **Ingenious Implementation Accelerated by RFF**: By maintaining the low-dimensional weight $\boldsymbol{\theta}$, it avoids recomputing kernel gradients in each round. The incremental update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \boldsymbol{\psi}(\boldsymbol{X}_{\mathcal{B}})(\boldsymbol{\alpha}^{\text{new}} - \boldsymbol{\alpha})$ has extremely low overhead.
4. **Proposition 2 Handles Compound Losses**: Utilizing the conjugate decomposition property of infimal convolution to naturally handle losses that are difficult to optimize in their primal form, such as Huber.
5. **Condition Number Advantage of Dual Optimization**: The dual Hessian linearly depends on $\boldsymbol{K}$, whereas the primal Hessian quadratically depends on $\boldsymbol{K}$, leading to faster dual convergence.

## Limitations & Future Work

1. **Only Supports Shift-Invariant Kernels**: The current RFF implementation is based on Bochner's theorem, requiring extra extensions for non-shift-invariant kernels (e.g., polynomial kernels).
2. **Approximation Error of RFF**: The approximation quality is limited when $M$ is small, while memory/time increases when $M$ is large. How to adaptively select $M$ is worth studying.
3. **Missing Comparison with Deep Learning**: As a kernel methods paper, there is a lack of comprehensive comparison with neural networks of similar scales on the same tasks.
4. **Choice of Block Size $|\mathcal{B}|$**: Needs to balance subproblem accuracy and parallel efficiency. The paper does not fully discuss adaptive strategies.
5. **Handling Non-Smooth Losses**: Although it mentions that trust region is compatible with non-smooth optimization, detailed experiments in non-smooth scenarios are insufficient.

## Related Work & Insights

- **Falkon / LogFalkon**: SOTA for Nyström + Cholesky preconditioning. Excellent performance but suffers from severe memory bottlenecks; the direct competitor of Joker.
- **EigenPro3**: Projected gradient descent avoids high space complexity but suffers from high time costs.
- **LIBSVM / ThunderSVM**: Exact kernel SVM implementations; the SMO solver is outdated for large-scale scenarios.
- **Random Fourier Features (Rahimi & Recht, 2007)**: A cornerstone method for kernel approximation and a key component of Joker.
- **Dual Coordinate Ascent (Shalev-Shwartz & Zhang, 2013)**: A dual optimization framework for linear models, which Joker generalizes to kernel models.
- **Trust Region Methods (Sorensen, 1982)**: Classic non-linear optimization techniques, combined with BCD for kernel optimization for the first time in Joker.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combined design of duality + BCD + trust region + RFF is novel, and the unified multi-model framework has high practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons with multiple methods on large-scale datasets, with convincing conclusions on the 90% memory savings.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous and clear mathematical derivations, complete algorithm pseudocode, and a good integration of theory and practice.
- Value: ⭐⭐⭐⭐ — Fills the gap of low-resource training for large-scale kernel methods, which is highly meaningful for resource-constrained scenarios (e.g., single GPU).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](../../ICCV2025/model_compression/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)
- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](../../CVPR2025/model_compression/jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [\[ACL 2025\] Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs](../../ACL2025/model_compression/direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)
- [\[ECCV 2024\] SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](../../ECCV2024/model_compression/spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)
- [\[ICML 2025\] LightGTS: A Lightweight General Time Series Forecasting Model](lightgts_a_lightweight_general_time_series_forecasting_model.md)

</div>

<!-- RELATED:END -->

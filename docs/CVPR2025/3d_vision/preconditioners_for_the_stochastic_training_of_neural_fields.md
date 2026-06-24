---
title: >-
  [Paper Note] Preconditioners for the Stochastic Training of Neural Fields
description: >-
  [CVPR 2025][3D Vision][Neural Fields] This paper proposes a theoretical preconditioning framework for the stochastic training of neural fields, proving that curvature-aware diagonal preconditioners (such as ESGD) significantly accelerate the training of neural fields with sine/Gaussian/wavelet activations, while showing no significant benefit for ReLU(PE) activations, thereby providing theoretical guidance for optimizer selection in neural fields.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Neural Fields"
  - "Preconditioner"
  - "Stochastic Optimization"
  - "Curvature-aware"
  - "Activation Function"
date: 2026-05-08
content_hash: 01f3bec65f46b910
---

# Preconditioners for the Stochastic Training of Neural Fields

**Conference**: CVPR 2025  
**arXiv**: [2402.08784](https://arxiv.org/abs/2402.08784)  
**Code**: [https://github.com/sfchng/preconditioner_neural_fields](https://github.com/sfchng/preconditioner_neural_fields)  
**Area**: 3D Vision / Neural Field Optimization  
**Keywords**: Neural Fields, Preconditioner, Stochastic Optimization, Curvature-aware, Activation Function

## TL;DR

This paper proposes a theoretical preconditioning framework for the stochastic training of neural fields, proving that curvature-aware diagonal preconditioners (such as ESGD) significantly accelerate the training of neural fields with sine/Gaussian/wavelet activations, while showing no significant benefit for ReLU(PE) activations, thereby providing theoretical guidance for optimizer selection in neural fields.

## Background & Motivation

1. **Background**: Neural fields (or Implicit Neural Representations) are widely applied in computer vision, robotics, and geometric modeling, including tasks such as image reconstruction, shape modeling, and NeRF. Currently, Adam is the default optimizer for training neural fields.
2. **Limitations of Prior Work**: Although Adam is effective, its training time is long. Traditional second-order methods, such as L-BFGS, are inapplicable in stochastic settings (failing to handle mini-batch training), and the community lacks a rigorous theoretical explanation for "why Adam outperforms SGD."
3. **Key Challenge**: SGD requires a small learning rate in high-curvature directions to avoid overshooting, which leads to slow progress in low-curvature directions. Preconditioners can balance curvature in all directions, but there is a lack of theoretical guidance on which preconditioner suits which neural field architecture.
4. **Goal**: (1) Explain the success of Adam from a preconditioning perspective; (2) Discover better preconditioning strategies than Adam to accelerate training.
5. **Key Insight**: View Adam as a preconditioned SGD using the diagonal of the Gauss-Newton matrix, and then analyze the impact of activation functions on the sparsity of Hessian-vector products.
6. **Core Idea**: Neural fields with sine/Gaussian/wavelet activations possess dense Hessian-vector products, so curvature-aware preconditioners (such as equilibrated preconditioners) can effectively reduce the condition number and accelerate convergence; in contrast, the Hessian-vector products of ReLU(PE) are sparse, limiting the effectiveness of preconditioners.

## Method

### Overall Architecture

Rather than proposing a new network architecture, this paper establishes a theoretical framework to analyze and select the optimal preconditioner for training neural fields. The overall pipeline is: (1) analyze Adam as a preconditioned SGD based on the diagonal of the Gauss-Newton matrix; (2) differentiate different activation functions through sparsity theorems of Hessian-vector products; (3) recommend using the equilibrated preconditioner ESGD to replace Adam for training sine/Gaussian/wavelet neural fields.

### Key Designs

1. **Theoretical Establishment of the Preconditioning Framework**:
    - Function: Unify and describe the working principles of preconditioners in various optimization algorithms (Adam, ESGD, AdaHessian, etc.)
    - Mechanism: Map the original optimization problem to a new parameter space via parameter transformation $\tilde{x} = D^{1/2} x$, such that the new Hessian becomes $(D^{-1/2})^T H (D^{-1/2})$. The goal is to choose $D$ to make the condition number of the new Hessian close to 1, i.e., uniform curvature in all directions. The second moment $v_t$ of Adam is essentially a moving average of the diagonal of the Gauss-Newton matrix $\text{Diag}(J^T J)$, which is a first-order approximation diagonal preconditioner.
    - Design Motivation: Explain the fundamental reason why Adam outperforms SGD, and lay the foundation for searching for better preconditioners.

2. **Hessian-Vector Product Density Theorems (Theorem 4.2 & 4.3)**:
    - Function: Theoretically distinguish whether neural fields with different activation functions are suitable for curvature-aware preconditioners.
    - Mechanism: Theorem 4.2 proves that under the MSE loss, the Hessian-vector product $Hv$ of neural fields with sine/Gaussian/wavelet/sinc activations is a dense vector; Theorem 4.3 proves that the $Hv$ of ReLU/ReLU(PE) is a sparse vector. A dense Hessian-vector product implies that the preconditioner can effectively scale multiple components of the gradient, whereas a sparse Hessian-vector product means only a few gradient components are scaled, leading to limited preconditioning effects.
    - Design Motivation: The piecewise linear nature of ReLU causes its second-order derivative to be zero in most regions, leading to a sparse Hessian; in contrast, smooth activation functions like sine/Gaussian have non-zero second-order derivatives everywhere, producing a dense Hessian.

3. **Practical Application of the ESGD Equilibrated Preconditioner**:
    - Function: Provide a computationally efficient curvature-aware optimization algorithm as an alternative to Adam.
    - Mechanism: ESGD utilizes the equilibrated preconditioner $D^E$, namely the 2-norm of each row of the Hessian, as the diagonal preconditioning matrix. To improve efficiency, the preconditioner is recomputed only once every $N=100$ iterations. The computational cost of each Hessian-vector product is comparable to a single gradient computation, and the storage and inversion complexity is linear $O(n)$.
    - Design Motivation: The equilibrated preconditioner reduces the condition number more effectively than the Jacobi preconditioner (experimentally verified by a larger drop in condition number), while avoiding the $O(n^3)$ storage and computational overhead of the full Hessian.

### Loss & Training

- MSE loss is used for 2D image reconstruction and NeRF
- BCE (binary cross-entropy) loss is used for 3D binary occupancy fields
- Theoretical results hold for both MSE and BCE losses
- ESGD updates the preconditioner every 100 iterations, using two variants: exponential moving average (ESGD) or infinity norm (ESGD-max)

## Key Experimental Results

### Main Results

**2D Image Reconstruction (DIV2K, Gaussian activation):**

| Method | Convergence Speed | Computational Complexity | Notes |
|------|---------|-----------|------|
| ESGD | Fastest | Comparable to Adam | Optimal balance |
| AdaHessian(E) | Faster | Slightly higher than Adam | Uses equilibrated preconditioning |
| AdaHessian(J) | Moderate | Slightly higher than Adam | Uses Jacobi preconditioning |
| Shampoo | Faster | Significantly higher than Adam | High overhead from Kronecker decomposition |
| Adam | Baseline | Baseline | Default optimizer |

**NeRF (LLFF dataset, Gaussian activation, average results):**

| Scene | Metric | Adam | ESGD | Notes |
|------|------|------|------|------|
| fern | Test PSNR | 24.38 | 24.41 | ESGD uses 120K iter, Adam 200K |
| flower | Test PSNR | 25.67 | 25.65 | ESGD uses 120K iter, Adam 200K |
| room | Test PSNR | 31.60 | 30.52 | Adam is slightly better |
| trex | Test PSNR | 22.04 | 22.21 | ESGD uses 120K iter, Adam 80K |

### Ablation Study

| Configuration | Key Performance | Notes |
|------|---------|------|
| Gaussian + ESGD | Fastest convergence | Dense Hessian, preconditioning is effective |
| Sine + ESGD | Faster than Adam | Dense Hessian, preconditioning is effective |
| Wavelet + ESGD | Faster than Adam | Dense Hessian, preconditioning is effective |
| ReLU(PE) + ESGD | Worse than Adam | Sparse Hessian, preconditioning is ineffective |
| Gaussian + Adam | Baseline | Gauss-Newton diagonal approximation |
| No preconditioning (SGD) | Slowest | All activations are far worse than Adam |

### Key Findings

- **Activation function is the decisive factor**: ESGD consistently outperforms Adam for Gaussian/sine/wavelet neural fields, but performs worse than Adam for ReLU(PE), perfectly validating our theoretical predictions.
- **ESGD is computationally efficient**: The computation time of the Hessian-vector product is comparable to a single gradient computation, and updating the preconditioner only every 100 steps results in very minor practical overhead.
- **AdaHessian/Shampoo degrade in 3D tasks**: In high-modal signals (3D binary occupancy fields), the local Hessian becomes noisier, which affects the performance of these methods; ESGD is more robust due to updating the preconditioner at intervals.
- **Condition number is directly verifiable**: Experiments show that the equilibrated preconditioner reduces the condition number of the Hessian more substantially than the Jacobi preconditioner.

## Highlights & Insights

- **Deconstructing Adam into preconditioned SGD** is a very elegant theoretical contribution—the second moment of Adam is essentially a moving average of the diagonal of the Gauss-Newton matrix. This perspective unifies Adam into the general framework of preconditioned optimization.
- **The association between Hessian sparsity and activation functions** is highly insightful: the piecewise linear nature of ReLU leads to Hessian sparsity, which not only explains the differences in preconditioning effectiveness, but also provides a new perspective for choosing activation functions in neural fields.
- **The strategy of updating the preconditioner every $N$ steps in ESGD** turns out to be an advantage in high-noise scenarios (3D tasks). This is an interesting finding that can be transferred to other scenarios requiring robust preconditioning.

## Limitations & Future Work

- **No improvement for ReLU(PE)**: Failed to find a ReLU(PE) preconditioning strategy superior to Adam, while ReLU(PE) remains the most widely used activation combination in practical applications.
- **Only diagonal preconditioners considered**: Full-matrix preconditioners might be effective for ReLU(PE) as well, but their computational cost is prohibitively high.
- **Not extended to large-scale scenarios**: Modern highly efficient representations such as instant-NGP and 3D Gaussian Splatting were not considered.
- Future directions: Explore non-diagonal but sparse preconditioners, or design specialized preconditioning strategies for ReLU(PE).

## Related Work & Insights

- **vs Adam**: Adam uses the diagonal of the Gauss-Newton matrix as a preconditioner, which serves as a first-order approximation; the proposed ESGD uses true Hessian information for preconditioning, making it more effective for non-ReLU activations.
- **vs L-BFGS**: Second-order methods like L-BFGS are unsuitable for stochastic training (mini-batch), whereas the proposed framework is specifically designed for stochastic settings.
- **vs AdaHessian**: AdaHessian combines the moving average of Adam with diagonal Hessian information, but its actual implementation uses an equilibrated preconditioner rather than the Jacobi preconditioner claimed in the paper.

## Rating

- Novelty: ⭐⭐⭐⭐ The theoretical perspective is novel, unifying Adam with the preconditioning framework. The link between Hessian sparsity and activation functions is an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three tasks (2D reconstruction, 3D occupancy fields, and NeRF) with multiple optimizer comparisons, showing great consistency between theory and experiments.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivation is clear, and the experimental presentation is intuitive, though the dense mathematical notations present a certain barrier for readers from non-optimization fields.
- Value: ⭐⭐⭐ Directly practical for neural fields using sine/Gaussian/wavelet activations, but offers limited help for the mainstream ReLU(PE) and modern highly efficient representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[CVPR 2025\] FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields](ffacenerf_few-shot_face_editing_in_neural_radiance_fields.md)
- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)
- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](../../NeurIPS2025/3d_vision/learning_neural_exposure_fields_for_view_synthesis.md)
- [\[CVPR 2026\] Evidential Neural Radiance Fields](../../CVPR2026/3d_vision/evidential_neural_radiance_fields.md)

</div>

<!-- RELATED:END -->

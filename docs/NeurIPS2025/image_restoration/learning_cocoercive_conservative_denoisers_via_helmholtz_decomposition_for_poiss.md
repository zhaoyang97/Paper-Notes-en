---
title: >-
  [Paper Note] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems
description: >-
  [NeurIPS 2025][Image Restoration][Plug-and-Play] This paper introduces the concept of Cocoercive Conservative (CoCo) denoisers and proposes a novel training strategy via generalized Helmholtz decomposition — Hamiltonian…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "Plug-and-Play"
  - "Poisson inverse problems"
  - "cocoercive denoiser"
  - "Helmholtz decomposition"
  - "convergence guarantees"
date: 2026-05-08
content_hash: 406d27f99e3d1669
---

# Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems

**Conference**: NeurIPS 2025
**arXiv**: [2505.08909](https://arxiv.org/abs/2505.08909)  
**Code**: [https://github.com/FizzzFizzz/CoCo-PnP](https://github.com/FizzzFizzz/CoCo-PnP)  
**Area**: Image Restoration
**Keywords**: Plug-and-Play, Poisson inverse problems, cocoercive denoiser, Helmholtz decomposition, convergence guarantees

## TL;DR
This paper introduces the concept of Cocoercive Conservative (CoCo) denoisers and proposes a novel training strategy via generalized Helmholtz decomposition — Hamiltonian regularization to promote conservativeness and spectral regularization to promote cocoerciveness — enabling denoisers to serve as proximal operators of implicit weakly convex priors, thereby achieving convergence-guaranteed and high-performance PnP methods for Poisson inverse problems (photon-limited deconvolution, low-dose CT, etc.).

## Background & Motivation

**Background**: Plug-and-Play (PnP) methods replace the proximal operator in variational models with deep denoisers and have achieved impressive results across various image inverse problems. Standard convergence analyses rely on conditions such as convexity/smoothness of the data fidelity term and non-expansiveness of the denoiser.

**Limitations of Prior Work**: In Poisson inverse problems (low-light imaging, low-dose CT, etc.), the data fidelity term $G(u)=\lambda\langle 1, Ku - f\log Ku\rangle$ is neither strongly convex nor smooth, so traditional PnP convergence conditions are not satisfied. Furthermore, non-expansive or residual non-expansive constraints, while sufficient for convergence, severely limit denoiser performance — the larger the noise level, the more pronounced the performance degradation.

**Key Challenge**: Convergence requires the denoiser to satisfy Lipschitz conditions (e.g., non-expansiveness), yet stronger such constraints lead to worse denoising performance. For the denoiser to act as a proximal operator, conservativeness is also required; existing approaches either explicitly construct potential functions at the cost of reduced performance (e.g., GS-DRUNet), or fail to guarantee the proximal operator property.

**Goal**: (a) Identify a Lipschitz condition weaker than non-expansiveness that still guarantees the proximal operator property; (b) promote conservativeness by directly regularizing the denoiser rather than explicitly constructing a potential function.

**Key Insight**: Two mathematical tools are employed: cocoerciveness (a relaxation of the Lipschitz condition) and Helmholtz decomposition (decomposing a vector field into a conservative field and a Hamiltonian field).

**Core Idea**: $\gamma$-cocoerciveness + conservativeness $\Rightarrow$ proximal operator property; moreover, when $\gamma < 0.5$ the denoiser may be expansive, breaking the non-expansiveness barrier.

## Method

### Overall Architecture
**Input**: Degraded images corrupted by Poisson noise. The method employs CoCo denoisers as proximal operators of the regularization term within the PnP framework (ADMM or PEGD algorithms). The key lies in imposing two regularizers during denoiser training: (1) Hamiltonian regularization — encouraging the Jacobian to be symmetric (conservativeness); (2) spectral regularization — controlling $\|2\gamma J - I\|_*$ to ensure cocoerciveness. **Output**: Restored images with theoretical guarantees of global convergence to a stationary point.

### Key Designs

1. **Cocoercive Denoiser**

    - **Function**: Relaxes the conventional non-expansiveness constraint, allowing both the denoiser and its residual to be expansive.
    - **Mechanism**: Defines the $\gamma$-cocoercive condition $\langle x-y, D(x)-D(y)\rangle \geq \gamma\|D(x)-D(y)\|^2$. Setting $\gamma=1$ recovers firm non-expansiveness; $\gamma=0.5$ recovers residual non-expansiveness. Crucially, when $\gamma < 0.5$ the denoiser may be expansive (Lipschitz constant $1/\gamma > 2$), imposing a weaker constraint. The equivalent condition $\|2\gamma J(x) - I\|_* \leq 1$ for all $x$ can be enforced via spectral regularization during training.
    - **Design Motivation**: Analyzing the spectral distribution of the Jacobian in the complex plane, the admissible region for $\gamma$-cocoerciveness strictly contains those of non-expansive and residual non-expansive conditions (e.g., for $\gamma=0.25$, the spectrum is permitted within a disk centered at $2$ with radius $2$), imposing fewer restrictions on the denoiser and enabling better performance.

2. **Conservativeness and Helmholtz Decomposition**

    - **Function**: Provides a mathematical framework to show that an ideal denoiser should be conservative (free of Hamiltonian components).
    - **Mechanism**: The generalized Helmholtz decomposition yields $D = D_c + D_h$, where $D_c = \nabla\phi$ is the conservative field (denoising along the gradient direction) and $D_h$ is the Hamiltonian field (perpendicular to the gradient, contributing no denoising but introducing rotational interference). At the Jacobian level, this corresponds to $J = S + A$, where $S=(J+J^\top)/2$ is the symmetric part (conservative field) and $A=(J-J^\top)/2$ is the skew-symmetric part (Hamiltonian field). Ideally $A=0$, i.e., $J$ is symmetric.
    - **Design Motivation**: A two-dimensional vector field analysis (Fig. 1) intuitively shows that the Hamiltonian component causes the denoising direction to deviate from the optimal path via rotation; removing it improves efficiency. This avoids the difficulty of explicitly constructing a potential function.

3. **Training Strategy**

    - **Function**: Trains DRUNet with two simultaneous regularization terms.
    - **Mechanism**: Total loss = MSE denoising loss + $\alpha_1 \cdot$ Hamiltonian regularization ($\|J - J^\top\|_*$ encouraging Jacobian symmetry) + $\alpha_2 \cdot$ spectral regularization ($\min\{1-\epsilon, \|2\gamma J - I\|_*\}$ enforcing cocoerciveness). A Hutchinson-type stochastic estimator is used to approximate the spectral norm of the Jacobian, avoiding explicit computation of the full Jacobian matrix.
    - **Design Motivation**: Translating the two theoretical properties (cocoerciveness + conservativeness) into differentiable regularization terms enables end-to-end training.

### Theoretical Guarantees
- Proves that a CoCo denoiser is the proximal operator of an implicit weakly convex function $F$: $D_\sigma = \text{Prox}_{F/\beta}$.
- Derives a restoration model with an implicit weakly convex prior; proves that CoCo-ADMM and CoCo-PEGD globally converge to a stationary point for Poisson inverse problems.
- Convergence condition: $t < 1/(2+1/(2\gamma))$; for $\gamma=0.25$, $t < 0.333$.

## Key Experimental Results

### Main Results (Poisson Deconvolution — CBSD68 + Levin Kernels)

| Method | p=100 PSNR/SSIM | p=50 PSNR/SSIM | Convergence Guarantee |
|--------|-----------------|----------------|-----------------------|
| DPIR | 26.51/0.742 | 25.38/0.691 | No |
| RMMO-DRS | 25.94/0.702 | 25.10/0.655 | Yes |
| Prox-DRS | 25.68/0.676 | 25.21/0.657 | Yes |
| DPS | 23.65/0.606 | 23.10/0.582 | No |
| DiffPIR | 24.82/0.643 | 24.08/0.607 | No |
| PnPI-HQS | 26.42/0.716 | 25.61/0.696 | Yes (no optimization problem solved) |
| **CoCo-ADMM** | **26.89/0.736** | **26.00/0.703** | **Yes** |
| **CoCo-PEGD** | **26.79/0.732** | **25.90/0.696** | **Yes** |

### Low-Dose CT Reconstruction (Mayo Dataset)

| Method | p=500 PSNR/SSIM | p=100 PSNR/SSIM |
|--------|-----------------|-----------------|
| FBP | 28.76/0.521 | 24.10/0.297 |
| PWLS-TGV | 33.16/0.846 | 30.43/0.775 |
| UNet (supervised) | 36.93/0.914 | 35.19/0.888 |
| WNet (supervised) | 37.12/0.927 | 35.98/0.913 |
| **CoCo-ADMM** | **37.63/0.940** | **36.68/0.919** |
| **CoCo-PEGD** | **37.72/0.939** | **36.43/0.916** |

### Ablation Study (Denoising Performance on CBSD68)

| Denoiser | σ=15 PSNR | σ=25 | σ=40 | Constraint Strength |
|----------|-----------|------|------|---------------------|
| DRUNet (unconstrained) | 34.14 | 31.54 | 29.33 | None |
| RMMO (firmly non-expansive) | 32.21 | 29.99 | 27.87 | Strongest |
| Prox-DRUNet (residual non-expansive) | 33.18 | 30.60 | 28.38 | Strong |
| SPC-DRUNet | 33.90 | 31.29 | 29.10 | Medium |
| 0.50-CoCo | 33.38 | 30.65 | 28.25 | Medium |
| **0.25-CoCo** | **34.00** | **31.38** | **29.16** | **Weak** |

### Key Findings
- The $\gamma=0.25$ CoCo denoiser achieves performance close to unconstrained DRUNet (gap of only 0.14–0.17 dB), far outperforming strongly constrained methods (RMMO lags by 1.46–1.93 dB).
- Hamiltonian regularization reduces the Jacobian symmetry error from $O(10^1)$ to $O(10^{-4})$, effectively promoting conservativeness.
- CoCo-PnP achieves the best performance among convergence-guaranteed methods, even surpassing the guarantee-free DPIR (+0.38 dB at p=100).
- In low-dose CT, the unsupervised CoCo-PnP even outperforms supervised learning methods (UNet, WNet).

## Highlights & Insights
- **Cocoerciveness as a replacement for non-expansiveness**: The paper elegantly identifies a weaker Lipschitz condition that still guarantees the proximal operator property, with a clear spectral-geometric interpretation (a larger admissible region in the complex plane). This idea is generalizable to other settings where Lipschitz constraints are required.
- **Geometric intuition of Helmholtz decomposition for denoising**: Decomposing the denoiser into a conservative field and a Hamiltonian field, with two-dimensional vector field diagrams intuitively explaining why the Hamiltonian component is harmful (rotation does not contribute to denoising), provides an elegant geometric understanding.
- **Closed loop between theory and practice**: The paper establishes a complete chain from mathematical properties (cocoerciveness + conservativeness) → training strategy (two regularizers) → convergence proof → experimental validation, with rigorous logic at every step.

## Limitations & Future Work
- Stochastic estimation of the Jacobian spectral norm increases training overhead; the paper does not report detailed training time comparisons.
- The choice of $\gamma$ requires manual tuning (the paper selects $\gamma=0.25$); smaller $\gamma$ theoretically yields better performance but imposes stricter convergence conditions.
- The method is built on the DRUNet architecture; whether it generalizes to more modern denoising architectures (e.g., Restormer, NAFNet) remains unvalidated.
- Although the Poisson inverse problem experiments cover deconvolution and CT, large-scale quantitative comparisons for Poisson denoising are relegated to the appendix.

## Related Work & Insights
- **vs. Prox-DRUNet (Hurault et al.)**: Both enforce conservative + Lipschitz constraints to yield proximal denoisers, but Prox-DRUNet requires residual non-expansiveness ($\gamma=0.5$), whereas CoCo uses the weaker $\gamma=0.25$ constraint, achieving +0.82 dB PSNR at $\sigma=15$.
- **vs. SPC-DRUNet**: The SPC condition is weaker but does not guarantee the proximal operator property (i.e., it does not solve any optimization problem); CoCo achieves comparable denoising performance while additionally providing theoretical guarantees.
- **vs. DPIR**: DPIR offers no convergence guarantee but is strong in practice; CoCo, while provably convergent, still surpasses DPIR by 0.38–0.62 dB in the p=50/100 setting.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Employing cocoerciveness and Helmholtz decomposition for denoiser design is an entirely new perspective with outstanding theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Deconvolution and CT experiments are solid, though the focus is primarily on Poisson settings.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous, geometric intuitions are elegant, and the logical flow is clear.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a new theoretical framework for the convergence of PnP methods in non-Gaussian inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MAP Estimation with Denoisers: Convergence Rates and Guarantees](map_estimation_with_denoisers_convergence_rates_and_guarantees.md)
- [\[ICML 2026\] Learning Normalized Energy Models for Linear Inverse Problems](../../ICML2026/image_restoration/learning_normalized_energy_models_for_linear_inverse_problems.md)
- [\[NeurIPS 2025\] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](../../CVPR2026/image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)
- [\[CVPR 2026\] Variational Garrote for Sparse Inverse Problems](../../CVPR2026/image_restoration/variational_garrote_for_sparse_inverse_problems.md)

</div>

<!-- RELATED:END -->

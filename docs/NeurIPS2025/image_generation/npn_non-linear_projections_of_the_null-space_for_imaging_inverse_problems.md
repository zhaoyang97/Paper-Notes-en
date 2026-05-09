---
title: >-
  [Paper Note] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems
description: >-
  [NeurIPS 2025][Image Generation][Null-space projection] This paper proposes Non-linear Projections of the Null-space (NPN)—a novel regularization strategy that trains a neural network to predict, directly from measurements, the projection coefficients of the ground-truth signal onto a low-dimensional subspace of the sensing matrix's null space. These coefficients serve as prior constraints on "invisible features" and can be flexibly integrated into diverse reconstruction frameworks including PnP, unrolled networks, DIP, and diffusion models. Convergence acceleration within the PnP framework is established theoretically.
tags:
  - NeurIPS 2025
  - Image Generation
  - Null-space projection
  - imaging inverse problems
  - regularization
  - plug-and-play methods
  - compressed sensing
date: 2026-05-08
content_hash: cfa2deb18aafd345
---

# NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems

**Conference**: NeurIPS 2025
**arXiv**: [2510.01608](https://arxiv.org/abs/2510.01608)
**Code**: [GitHub](https://github.com/yromariogh/NPN)
**Area**: Diffusion Models / Image Generation
**Keywords**: Null-space projection, imaging inverse problems, regularization, plug-and-play methods, compressed sensing

## TL;DR
This paper proposes Non-linear Projections of the Null-space (NPN)—a novel regularization strategy that trains a neural network to predict, directly from measurements, the projection coefficients of the ground-truth signal onto a low-dimensional subspace of the sensing matrix's null space. These coefficients serve as prior constraints on "invisible features" and can be flexibly integrated into diverse reconstruction frameworks including PnP, unrolled networks, DIP, and diffusion models. Convergence acceleration within the PnP framework is established theoretically.

## Background & Motivation

Imaging inverse problems seek to recover a high-dimensional signal $\mathbf{x}^*$ from undersampled, noisy measurements $\mathbf{y} = \mathbf{H}\mathbf{x}^* + \boldsymbol{\omega}$, and are central to tasks such as image deblurring, super-resolution, MRI, CT, and compressed sensing. The ill-posedness of these problems stems from the non-trivial null space of the sensing matrix $\mathbf{H}$: signal components along null-space directions are entirely invisible to the measurements, giving rise to infinitely many feasible solutions.

Core limitations of existing learned priors:

**Image-domain priors** (e.g., denoisers in PnP, generative model priors) capture global signal structure but do not explicitly account for null-space geometry. Denoisers may arbitrarily alter null-space components, since the data fidelity term $g(\tilde{\mathbf{x}})$ imposes no constraint along these directions.

**Null-space networks** (NSN, DDN) exploit range-null decompositions but operate with the full null-space projection operator $\mathbf{I} - \mathbf{H}^\dagger\mathbf{H}$, whose dimensionality remains as high as $n-m$, and they are tightly coupled to specific reconstruction algorithms.

The key insight of NPN is that it is unnecessary to learn all information in the complete null space; instead, one needs only to identify a maximally informative low-dimensional subspace within it. Training a lightweight network to predict the corresponding subspace coefficients directly from measurements provides sensing-matrix-specific, measurement-orthogonal complementary information. This is essentially a dimensionality reduction perspective—learning a nonlinear mapping in a $p \leq (n-m)$ dimensional subspace is substantially easier than learning in the full signal space.

## Method

### Overall Architecture
Given a sensing matrix $\mathbf{H} \in \mathbb{R}^{m \times n}$, a projection matrix $\mathbf{S} \in \mathbb{R}^{p \times n}$ is constructed whose rows lie in $\text{Null}(\mathbf{H})$. A network $G^*: \mathbb{R}^m \to \mathbb{R}^p$ is trained such that $G^*(\mathbf{y}) \approx \mathbf{S}\mathbf{x}^*$. The NPN regularization term $\phi(\tilde{\mathbf{x}}) = \|G^*(\mathbf{y}) - \mathbf{S}\tilde{\mathbf{x}}\|_2^2$ is then incorporated into any reconstruction framework.

### Key Designs

1. **Design of the projection matrix $\mathbf{S}$**:

    - **Compressed Sensing (CS)**: $\mathbf{H}$ is a dense random matrix; $\mathbf{S}$ is obtained by orthogonalizing via QR decomposition.
    - **MRI**: $\mathbf{H}$ consists of a subset of rows from the undersampled 2D DFT; $\mathbf{S}$ takes the complementary frequency rows (unsampled k-space lines), which are naturally orthogonal.
    - **CT**: $\mathbf{S}$ corresponds to rows of the Radon matrix associated with unacquired projection angles.
    - **Deblurring/Super-resolution**: $\mathbf{S}[i,i+j] = 1 - \mathbf{h}[j]$, blocking in the frequency domain the low frequencies sampled by $\mathbf{H}$.
    - Design Motivation: For each sensing matrix structure, domain knowledge is exploited to construct the most informative null-space subspace.

2. **Joint optimization of $G$ and $\mathbf{S}$**:

    - The objective contains three terms: $\min_{G,\tilde{\mathbf{S}}} \mathbb{E}\|G(\mathbf{Hx}^*) - \tilde{\mathbf{S}}\mathbf{x}^*\|_2^2 + \lambda_1\|\mathbf{x}^* - \mathbf{A}^\dagger \mathbf{A}\mathbf{x}^*\|_2^2 + \lambda_2\|\mathbf{A}^\top\mathbf{A} - \mathbf{I}\|_2^2$
    - where $\mathbf{A} = [\mathbf{H}^\top, \tilde{\mathbf{S}}^\top]^\top$.
    - The second term enforces approximate orthogonality between the row spaces of $\mathbf{S}$ and $\mathbf{H}$; the third ensures the combined system is full rank.
    - Design Motivation: The initial $\mathbf{S}$ is constructed via domain knowledge, and joint optimization adapts the subspace to the statistical properties of the data.

3. **Convergence acceleration theory (Theorem 1)**:

    - The convergence rate of PnP-NPN is $\rho = (1+\delta)\left(\|\mathbf{I} - \alpha(\mathbf{H}^\top\mathbf{H} + \mathbf{S}^\top\mathbf{S})\|_2^2 + (1+\Delta_{\mathcal{M}_D}^S)\|\mathbf{S}\|_2^2\right)$.
    - Since $\mathbf{S}$ is orthogonal to $\mathbf{H}$, the operator norm $\|\mathbf{I} - \alpha(\mathbf{H}^\top\mathbf{H} + \mathbf{S}^\top\mathbf{S})\|$ is small, guaranteeing $\rho < 1$.
    - Convergence Improvement Zone (CIZ): NPN provides acceleration only when $\|N(\mathbf{Hx})\|^2 \leq \|\mathbf{S}(\tilde{\mathbf{x}}^\ell - \mathbf{x}^*)\|^2$.

4. **Regularization convergence (Theorem 2)**:

    - As iterations proceed, the NPN regularization term is asymptotically bounded: $\lim_{\ell\to\infty} \phi(\mathbf{x}^{\ell+1}) \leq K\|\mathbf{x}^*\|_2^2(1 + \Delta_{\mathcal{M}_D}^H)$.
    - The upper bound depends on the Lipschitz constant $K$ of the network approximation error and the RIP constant of the sensing matrix.

### Multi-Framework Integration
- **PnP-FISTA / PnP-ADMM**: NPN is incorporated as an additional regularization term.
- **Unrolled networks**: The NPN proximal step is embedded in end-to-end training.
- **Deep Image Prior (DIP)**: NPN is added as an auxiliary loss term.
- **Diffusion models (DPS, DiffPIR)**: NPN gradient guidance is injected into the sampling process.

## Key Experimental Results

### Main Results: PSNR Comparison across Inverse Problems (dB)

| Inverse Problem | Sparse Baseline | PnP Baseline | NPN+PnP | NSN (DNSN) | DDN-C |
|----------------|----------------|-------------|---------|-----------|-------|
| CS ($m/n=0.1$) | 15.93 | 20.04 | **21.12** | 20.10 | 20.03 |
| MRI (AF=4) | 36.86 | 35.99 | **38.08** | 35.2 | 33.7 |
| Deblurring ($\sigma=2$) | 29.27 | 30.78 | 31.42 | **33.07** | 33.03 |

### Ablation Study: Projection Ratio $p/n$ and Generalization (CS)

| $p/n$ | PnP (CIFAR10) | PnP (STL10) | Unrolled (CIFAR10) | Unrolled (STL10) |
|-------|--------------|-------------|-------------------|-----------------|
| 0 (baseline) | 20.04 | 20.09 | 24.32 | 18.35 |
| 0.1 | **21.12** | 19.91 | 28.53 | 19.64 |
| 0.3 | 21.07 | **21.14** | 28.75 | **20.23** |
| 0.5 | 20.78 | 20.77 | 27.64 | 18.76 |
| 0.9 | 20.41 | 21.02 | **29.90** | 19.48 |

### Diffusion Model Integration (Deblurring)

| Method | $\gamma=0$ (baseline) | Best $\gamma$ | Gain |
|--------|-----------------------|--------------|------|
| NPN-DPS | 28.22 | 30.07 ($\gamma=0.2$) | +1.85 dB |
| NPN-DiffPIR | 31.30 | 31.91 ($\gamma=10^{-4}$) | +0.61 dB |

### Key Findings
- Within the CIZ, NPN provides up to ~10× per-step error reduction ratio (Figure 2c), empirically validating Theorem 1.
- $p/n \approx 0.3$ yields the best balance between accuracy and robustness; excessively large values introduce redundancy.
- Joint optimization of $\mathbf{S}$ substantially narrows the performance gap outside the CIZ compared to a fixed QR initialization.
- NPN yields improvements of up to 5 dB within DIP, demonstrating that null-space priors remain effective even in single-image, training-free settings.

## Highlights & Insights
- The geometric intuition is exceptionally clear: the toy $\mathbb{R}^3$ example in Figure 1 illustrates vividly why learning a mapping in a null-space subspace is more accurate than direct reconstruction—dimensionality reduction simplifies the learning task and improves robustness to out-of-distribution samples.
- The framework is highly flexible: the same NPN regularization term can be seamlessly integrated into reconstruction pipelines ranging from classical iterative algorithms to state-of-the-art diffusion models.

## Limitations & Future Work
- A separate network $G$ must be trained for each sensing matrix configuration $(\mathbf{H}, \mathbf{S})$; although the network is lightweight, this increases deployment complexity.
- Integration with unrolled networks currently follows a two-stage training procedure; end-to-end joint training may yield further performance gains.
- The design of $\mathbf{S}$ may be ineffective for certain sensing matrix structures where no learnable nonlinear relationship exists between $\mathbf{Hx}$ and $\mathbf{Sx}$.

## Related Work & Insights
- **vs. NSN/DDN**: NSN employs the full null-space projection operator and is tightly coupled to specific algorithms; NPN selects a low-dimensional informative subspace and functions as a general-purpose regularizer.
- **vs. PnP denoisers**: Denoisers implicitly impose image priors without accounting for null-space structure; NPN explicitly constrains null-space components, making the two approaches complementary.
- **vs. DPS/DiffPIR**: Diffusion-based posterior sampling focuses on data fidelity; NPN supplements it with information along null-space directions.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of learning in a low-dimensional null-space subspace is novel, with clear geometric intuition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five inverse problems and four reconstruction frameworks (PnP, unrolled, DIP, diffusion) are comprehensively covered, with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are well organized; the geometric illustration in Figure 1 is particularly effective.
- Value: ⭐⭐⭐⭐ High practical value as a general-purpose regularization tool that can directly improve existing reconstruction pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[NeurIPS 2025\] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting](one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)
- [\[NeurIPS 2025\] On the Emergence of Linear Analogies in Word Embeddings](on_the_emergence_of_linear_analogies_in_word_embeddings.md)

</div>

<!-- RELATED:END -->

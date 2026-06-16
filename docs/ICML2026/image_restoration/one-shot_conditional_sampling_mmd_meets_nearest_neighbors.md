---
title: >-
  [Paper Note] One-shot Conditional Sampling: MMD meets Nearest Neighbors
description: >-
  [ICML 2026][Image Restoration][MMD] CGMMD utilizes $k$-nearest neighbor graphs to estimate the "Expected Conditional MMD (ECMMD)" as a directly minimizable non-adversarial objective. This allows training a conditional generator that samples from $P_{Y\mid X}$ in a single forward pass, while providing non-asymptotic error bounds and proofs of distribution
tags:
  - ICML 2026
  - Image Restoration
  - MMD
date: 2026-05-08
content_hash: c12212130f3b8ad1
---
# One-shot Conditional Sampling: MMD meets Nearest Neighbors

**Conference**: ICML 2026  
**arXiv**: [2509.25507](https://arxiv.org/abs/2509.25507)  
**Code**: https://github.com/anirbanc96/cgmmd (Available)  
**Area**: Scientific Computing / Conditional Generation / Kernel Methods  
**Keywords**: Conditional Sampling, MMD, Nearest Neighbor Estimation, One-shot Generation, Kernel Mean Embedding

## TL;DR
CGMMD utilizes $k$-nearest neighbor graphs to estimate the "Expected Conditional MMD (ECMMD)" as a directly minimizable non-adversarial objective. This allows training a conditional generator that samples from $P_{Y\mid X}$ in a single forward pass, while providing non-asymptotic error bounds and proofs of distributional convergence.

## Background & Motivation
**Background**: Conditional distribution modeling is a fundamental problem in statistics and machine learning. While regression only provides conditional means or quantiles, many downstream tasks (uncertainty quantification, simulation-based inference, graphical models, dimensionality reduction) require the full distribution $P_{Y\mid X}$. Modern mainstream approaches include conditional GANs, CVAEs, and conditional diffusion models, which reformulate "density estimation" as "generating samples using noise $\eta$ and input $x$."

**Limitations of Prior Work**: Each of the three main approaches has drawbacks. Conditional GANs involve min-max optimization and rely on JS/KL divergence; when the generator and target supports lie on low-dimensional manifolds, they may become disjoint, leading to vanishing gradients, training instability, and mode collapse. Wasserstein/IPM losses (e.g., W-GAN, MMD-GAN) mitigate instability in unconditional settings, but the **conditional** scenario lacks finite-sample theory and concise $k$-nearest neighbor estimators. Although conditional diffusion is stable, sampling requires tens to thousands of iterative denoising steps, resulting in high inference complexity.

**Key Challenge**: A trade-off exists between training stability, statistical consistency, and sampling time. Adversarial losses sacrifice stability for flexibility, diffusion sacrifices sampling speed for quality, and IPM-based objectives often lack statistical guarantees.

**Goal**: Construct a conditional sampling framework that simultaneously achieves: (i) **non-adversarial, direct minimization**; (ii) **one-shot sampling** in a single forward pass; (iii) non-asymptotic error bounds with proven convergence to the true distribution.

**Key Insight**: Chatterjee et al. (2024) generalized MMD to Expected Conditional MMD (ECMMD), proving it is a strictly proper scoring rule (ECMMD$^2 = 0$ if and only if conditional distributions are equal). To use ECMMD as a training loss, a consistent estimator from finite samples is needed. $k$-nearest neighbors ($k$-NN), a classic tool for conditional mean estimation, can be integrated with the U-statistic kernel of ECMMD to yield a non-adversarial, non-iterative, and end-to-end differentiable objective.

**Core Idea**: Approximate the expectation "conditioned on $X=X_i$" using a $k$-NN graph. By feeding generator outputs and real samples into the kernel function $\mathsf{H}$, the ECMMD estimator is directly minimized. Once trained, a sample $\hat g(\eta, x) \sim P_{Y\mid X=x}$ is obtained via a single forward pass with noise $\eta$ for any given $x$.

## Method

### Overall Architecture
CGMMD addresses how to draw samples from $P_{Y\mid X=x}$ in one shot. It converts the generation problem into a pure minimization objective: using a $k$-NN graph on $X$, it estimates ECMMD as a backpropagatable empirical loss. A ReLU generator $\hat g(\eta, x)$ is trained directly; during testing, a conditional sample is produced by a single forward pass $\eta\sim P_\eta \to \hat g(\eta, x)$ for a new $x$.

Specifically, given training pairs $\{(Y_i, X_i)\}_{i=1}^n$, reference noise $P_\eta=\mathcal{N}(0, I_m)$, kernel function $\mathsf{K}$, and generator class $\mathcal{G}$, each iteration involves sampling auxiliary noise $\eta_i$ to produce pseudo-samples $g(\eta_i, X_i)$. A directed $k$-NN graph $G(\mathcal{X}_n)$ is constructed on the mini-batch of $X$. The empirical loss $\hat{\mathcal{L}}(g)$ (a consistent estimator of ECMMD$^2$) is computed by summing over neighbor pairs in the graph. The parameters are updated via backpropagation. After training, the sampler $\hat g$ generates one-shot samples.

### Key Designs

**1. k-NN Estimator for ECMMD: Differentiable Neighbor Summation**

The obstacle to minimizing ECMMD$^2$ directly is the inner expectation conditioned on $X$, given only finite samples. The paper uses the kernel trick to express ECMMD$^2$ as $\mathbb{E}[\mathsf{H}(W, W')]$ (where $W=(Y,Z)$ and kernel $\mathsf{H}$ is a combination of four kernel values), then separates the outer expectation over $X$ from the inner conditional expectation over $Y, Z\mid X$ using the tower property. Instead of kernel regression, a $k$-NN directed graph $G(\mathcal{X}_n)$ is built on $X$. Samples in the neighbor set $N_G(i)$ are treated as pseudo-replicates under "approximately the same conditions." The estimator is $\widehat{\mathrm{ECMMD}}^2 = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}(W_i, W_j)$. This $k$-NN approach avoids bandwidth selection, adapts to the intrinsic dimension $\bar d$ of $X$, and allows gradients to flow directly through $g$ without reparameterization tricks.

**2. Non-adversarial Direct Minimization: No Discriminator**

By using the estimator as the loss, training becomes pure minimization over generator parameters $\theta$: $\hat g \in \arg\min_{g\in\mathcal{G}} \hat{\mathcal{L}}(g)$, where $\hat{\mathcal{L}}(g) = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}\big((Y_i, g(\eta_i, X_i)), (Y_j, g(\eta_j, X_j))\big)$. In each mini-batch, the $k_B$-NN graph is reconstructed, $\hat{\mathcal{L}}$ is calculated, and $\theta$ is updated. Unlike GANs that may suffer from vanishing gradients under disjoint supports, this kernel loss remains stable. It bypasses mode collapse and min-max instability common in conditional GANs, while only requiring the maintenance of a single generator network.

**3. One-shot Sampling + ReLU Network Function Class**

Sampling during inference relies on the noise outsourcing lemma: for a joint distribution $(Y, X)$, there exists a Borel measurable $\bar g$ and independent noise $\eta$ such that $(Y, X)\overset{d}{=}(\bar g(\eta, X), X)$. By learning an approximation $\hat g$ in the ReLU network class $\mathcal{G}_{\mathcal{H},\mathcal{W},\mathcal{S},\mathcal{B}}$ (depth $\mathcal{H}$, width $\mathcal{W}$, parameters $\mathcal{S}$, $\ell_\infty$ bound $\mathcal{B}$), sampling is reduced to a single step: $\eta\sim\mathcal{N}(0, I_m)\to\hat g(\eta, x)$. While diffusion models bottleneck sampling through iterative denoising, CGMMD compresses distributional information into the weights of a single network, ensuring consistency via the ECMMD loss. This results in sampling speeds two to three orders of magnitude faster than diffusion.

### Loss & Training
The core loss is $\hat{\mathcal{L}}(g) = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}(W_{i,g}, W_{j,g})$, where $\mathsf{H}(W_i, W_j) = \mathsf{K}(Y_i, Y_j) - \mathsf{K}(Y_i, g_j) - \mathsf{K}(g_i, Y_j) + \mathsf{K}(g_i, g_j)$. Experiments use a Gaussian kernel and batch size 200, with $k_B$-NN graphs built per batch. Theoretically, $k_n = o(\sqrt n)$ and network scale must satisfy $\mathcal{B}^2\mathcal{H}\mathcal{S}\log\mathcal{S}\log n / n \to 0$. Theorem 4.4 provides a non-asymptotic bound: with probability at least $1-\delta$, $\mathcal{L}(\hat g) \lesssim \frac{\mathrm{polylog}\, n}{n^{1/(2d)}} + \sqrt{\frac{\mathcal{B}^2\mathcal{H}\mathcal{S}\log\mathcal{S}\log n}{n}} + \omega_{\bar g}\!\big(\frac{2\sqrt{\log n}}{(\mathcal{H}\mathcal{W})^{1/(d+m)}}\big) + \sqrt{\frac{\log(1/\delta)}{n}}$. These terms represent $k$-NN estimation error, generalization error, and approximation error, respectively. Corollary 4.5 proves the induced conditional distribution converges to the true one in both MMD and characteristic function senses.

## Key Experimental Results

### Main Results

| Task / Dataset | Setting | Key Observation |
|---|---|---|
| Bivariate Helix | $\sigma \in \{0.2, 0.4, 0.6\}$ | All methods recover the helix at low noise ($\sigma=0.2$); at high noise, CGMMD preserves the helix structure while GCDS and WGAN degrade. |
| MNIST 4× Super-res | $7\times 7 \to 28\times 28$ | Clear reconstruction of digits $\{0\dots4\}$. |
| STL-10 4× Super-res | $3\times 24\times 24 \to 3\times 96\times 96$ | Sharp mean reconstruction; pixel-wise std-dev plots show significant diversity. |
| MNIST Denoising | $\sigma=0.5$, digits $\{5\dots9\}$ | CGMMD recovers clean glyphs. |
| CelebHQ Denoising | $3\times 64\times 64$, $\sigma=0.25$ | Face reconstruction preserves facial structures. |

### Key Findings vs. Diffusion Models (MNIST Denoising, $\sigma=0.9$)

| Model | PSNR | SSIM | Time/batch (s) | Time/img (s) |
|---|---|---|---|---|
| Diffusion (CFG) | 13.326 | 0.861 | 6.94 | $5.42\times 10^{-2}$ |
| Distilled Diffusion | 10.658 | 0.508 | $1.18\times 10^{-1}$ | $9.2\times 10^{-4}$ |
| **CGMMD** | 8.922 | 0.718 | $7.21\times 10^{-2}$ | $\mathbf{5.6\times 10^{-4}}$ |

### Key Findings
- CGMMD shows significant stability advantages over GCDS/WGAN in high-noise tasks; WGAN often fails to train without $\ell_1$ regularization.
- The comparison with Diffusion highlights a speed-quality trade-off: CGMMD is ~100x faster than CFG Diffusion per image. While PSNR is lower, SSIM remains competitive.
- The ECMMD + $k$-NN framework adaptively handles the intrinsic dimension of $X$, as observed in synthetic experiments (Appx. C.2).

## Highlights & Insights
- **Integrating k-NN as a conditional expectation approximator** within MMD estimation is a simple yet powerful design. It inherits the stability of unconditional MMD-GAN while naturally introducing conditional dependence.
- The combination of **one-shot sampling and non-adversarial training** makes CGMMD highly attractive as a "lightweight conditional sampler," especially for simulation-based inference where per-sample latency is critical.
- The proof for **uniform concentration of k-NN-type nonlinear functionals** is an independently interesting tool applicable to other statistical learning problems (e.g., conditional independence testing).
- The adaptation to intrinsic dimension $\bar d$ provides a bound suitable for "high-dimensional but on-manifold" data, consistent with real-world data distributions.

## Limitations & Future Work
- Current theory requires the network size to grow with the number of samples, which does not cover fixed-architecture settings. Image quality (PSNR) currently lags behind specialized Diffusion models.
- Building the $k_B$-NN graph per mini-batch incurs overhead that might be significant for large batches or very high dimensions; approximate nearest neighbors or caching strategies were not discussed.
- Experiments are limited to relatively small image datasets (MNIST, CelebHQ, STL-10) and do not address high-resolution natural images or text-to-image generation.
- Future work: Extending the loss to flow-matching/OT-flow objectives, replacing $k$-NN with scalable structures (like differentiable ANN), and extending theory to fixed-architecture networks.

## Related Work & Insights
- **vs GCDS (Zhou et al., 2023)**: GCDS uses a GAN-style approach for conditional sampling, suffering from min-max instability. CGMMD uses direct minimization of ECMMD and provides consistency proofs.
- **vs Conditional Wasserstein-GAN (Song et al., 2025)**: W-GAN uses Wasserstein distance for conditional IPM but is sensitive to $\ell_1$ regularization; CGMMD's kernel loss is smoother and more stable.
- **vs Conditional Diffusion (Ho & Salimans, 2021)**: Diffusion offers higher quality but much slower sampling (~50 ms per image). CGMMD (~0.56 ms) is two orders of magnitude faster, suitable for mass sampling in scientific computing.
- **vs Unconditional MMD-GAN (Li et al., 2015; Bińkowski et al., 2018)**: This work generalizes the MMD-GAN framework to the conditional setting, introducing $k$-NN estimators and non-asymptotic bounds.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining ECMMD with $k$-NN for conditional generation with non-asymptotic theory is a clear and novel path.
- Experimental Thoroughness: ⭐⭐⭐ Covers proof-of-concept across several tasks, but lacks large-scale benchmarks and direct comparison with SOTA high-res diffusion.
- Writing Quality: ⭐⭐⭐⭐ Rigorous derivations, consistent notation, and a well-integrated algorithm. The $k$-NN concentration results in the appendix are a highlight.
- Value: ⭐⭐⭐⭐ Significant for communities needing fast conditional sampling with theoretical guarantees (e.g., scientific computing, posterior approximation).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[CVPR 2026\] More Than Meets the Eye: A Unified Image Fusion Framework via Semantic-Pixel Entropy Trade-off for Zero-Shot Generalization](../../CVPR2026/image_restoration/more_than_meets_the_eye_a_unified_image_fusion_framework_via_semantic-pixel_entr.md)
- [\[CVPR 2026\] AceTone: Bridging Words and Colors for Conditional Image Grading](../../CVPR2026/image_restoration/acetone_bridging_words_and_colors_for_conditional_image_grading.md)
- [\[CVPR 2026\] One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation](../../CVPR2026/image_restoration/one-shot_flow_any-time_frame_a_bidirectional_warping_framework_for_event-based_v.md)
- [\[CVPR 2026\] Zero-Shot Image Denoising via Hybrid Prior-Guided Pseudo Sample Generation](../../CVPR2026/image_restoration/zero-shot_image_denoising_via_hybrid_prior-guided_pseudo_sample_generation.md)

</div>

<!-- RELATED:END -->

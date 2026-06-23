---
title: >-
  [Paper Note] One-shot Conditional Sampling: MMD meets Nearest Neighbors
description: >-
  [ICML 2026][Image Restoration][MMD] CGMMD utilizes $k$-nearest neighbor graphs to estimate the "Expected Conditional MMD (ECMMD)" as a directly minimizable non-adversarial objective. It trains a conditional generator capable of sampling from $P_{Y\mid X}$ in a single forward pass and provides non-asymptotic error bounds alongside proofs of distributional
tags:
  - ICML 2026
  - Image Restoration
  - MMD
date: 2026-05-08
content_hash: a7b393ac81090ae8
---
# One-shot Conditional Sampling: MMD meets Nearest Neighbors

**Conference**: ICML 2026  
**arXiv**: [2509.25507](https://arxiv.org/abs/2509.25507)  
**Code**: https://github.com/anirbanc96/cgmmd (Yes)  
**Area**: Scientific Computing / Conditional Generation / Kernel Methods  
**Keywords**: Conditional Sampling, MMD, Nearest Neighbor Estimation, One-shot Generation, Kernel Mean Embedding

## TL;DR
CGMMD utilizes $k$-nearest neighbor graphs to estimate the "Expected Conditional MMD (ECMMD)" as a directly minimizable non-adversarial objective. It trains a conditional generator capable of sampling from $P_{Y\mid X}$ in a single forward pass and provides non-asymptotic error bounds alongside proofs of distributional convergence.

## Background & Motivation
**Background**: Modeling conditional distributions is a fundamental problem in statistics and machine learning. While regression only provides conditional means or quantiles, many downstream tasks (uncertainty quantification, simulation-based inference, graphical models, dimensionality reduction) require the full $P_{Y\mid X}$. Modern mainstream approaches include conditional GANs, CVAEs, and conditional diffusion models, which reformulate "density estimation" as "generating samples using noise $\eta$ and input $x$."

**Limitations of Prior Work**: These three categories of methods have respective drawbacks. Conditional GANs involve min-max optimization and rely on JS/KL divergence; when the generator and target distributions are supported on low-dimensional manifolds with almost no overlap, gradients vanish, training becomes unstable, and mode collapse occurs. Wasserstein/IPM-based losses (e.g., W-GAN, MMD-GAN) mitigate instability in unconditional settings but lack finite-sample theory and simple $k$-nearest neighbor estimators in **conditional** scenarios. While conditional diffusion is stable, sampling requires dozens to thousands of iterative denoising steps, leading to high inference complexity.

**Key Challenge**: There is a trade-off between the stability of the training objective, statistical consistency, and sampling time—adversarial losses sacrifice stability for flexibility, diffusion sacrifices sampling speed for sample quality, and IPM-based objectives lack statistical guarantees.

**Goal**: Construct a conditional sampling framework that simultaneously satisfies: (i) **non-adversarial and directly minimizable**; (ii) **one-shot sampling** capability; (iii) existence of non-asymptotic error bounds and provable convergence to the true distribution.

**Key Insight**: Chatterjee et al. (2024) generalized MMD to Expected Conditional MMD (ECMMD), proving it is a strictly proper scoring rule (ECMMD$^2 = 0$ if and only if the conditional distributions are equal). However, using ECMMD as a training loss requires a form that can be consistently estimated from finite samples. Given that $k$-nearest neighbors are classic tools in conditional mean estimation, grafting them onto the U-statistic kernel function of ECMMD yields an objective that is non-adversarial, avoids iterative sampling, and allows end-to-end backpropagation.

**Core Idea**: Use a $k$-NN graph to approximate taking expectations "conditioned on $X=X_i$." By feeding generator outputs and real samples into the kernel function $\mathsf{H}$, the ECMMD estimator is directly minimized. After training, a noise $\eta$ is drawn for any $x$, and a single forward pass yields $\hat g(\eta, x) \sim P_{Y\mid X=x}$.

## Method

### Overall Architecture
CGMMD addresses the problem of "how to sample from $P_{Y\mid X=x}$ in one shot given $X=x$." It transforms this generation problem into a pure minimization objective: using a $k$-nearest neighbor graph on $X$ to estimate "Expected Conditional MMD (ECMMD)" as a backpropagatable empirical loss. A ReLU generator $\hat g(\eta, x)$ is trained directly. During inference, a conditional sample is obtained by a single forward pass of noise $\eta$ given a new $x$.

Specifically, given training pairs $\{(Y_i, X_i)\}_{i=1}^n$, reference noise $P_\eta=\mathcal{N}(0, I_m)$, a kernel function $\mathsf{K}$, and a generator class $\mathcal{G}$; each iteration starts by sampling auxiliary noise $\eta_i$ for each sample to generate pseudo-samples $g(\eta_i, X_i)$. Next, a directed $k$-nearest neighbor graph $G(\mathcal{X}_n)$ is constructed on the mini-batch $X$. The empirical loss $\hat{\mathcal{L}}(g)$ (a consistent estimator of ECMMD$^2$) is computed by summing over neighbor pairs in the graph, and parameters are updated via backpropagation. Once trained, the sampling phase $\eta\sim P_\eta \to \hat g(\eta, x)$ produces samples in one step.

### Key Designs

**1. $k$-NN Estimator of ECMMD: Making "Conditional Expectation" a Differentiable Neighbor Sum**

The primary obstacle to directly minimizing ECMMD$^2$ is the inner operator of "taking expectation conditioned on $X$," while only finite samples are available. The paper first uses the kernel trick to write ECMMD$^2$ as $\mathbb{E}[\mathsf{H}(W, W')]$ (where $W=(Y,Z)$ and the kernel $\mathsf{H}$ is a combination of four kernel values), then uses the tower property to separate the outer expectation over $X$ from the inner conditional expectation over $Y, Z\mid X$. A critical step is avoiding kernel regression for the inner conditional expectation; instead, a $k$-NN directed graph $G(\mathcal{X}_n)$ is built on $X$. Samples in the neighborhood $N_G(i)$ are treated as pseudo-replicates under "approximately the same condition." Thus, the estimator is written as $\widehat{\mathrm{ECMMD}}^2 = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}(W_i, W_j)$. This approach avoids bandwidth selection required by kernel regression, adapts to the intrinsic dimension $\bar d$ of $X$, and ensures the graph depends only on $X_i$. Since only $g$ is involved in the summation, gradients flow directly without additional reparameterization tricks.

**2. Non-adversarial Direct Minimization Objective: Removing the Discriminator**

By using the estimator as the loss, training reduces to a pure minimization of the generator parameters $\theta$: $\hat g \in \arg\min_{g\in\mathcal{G}} \hat{\mathcal{L}}(g)$, where $\hat{\mathcal{L}}(g) = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}\big((Y_i, g(\eta_i, X_i)), (Y_j, g(\eta_j, X_j))\big)$. The loop in Algorithm 1 reconstructs a $k_B$-NN graph for each mini-batch, computes $\hat{\mathcal{L}}$ forward, and updates $\theta \leftarrow \theta - \alpha\nabla_\theta \hat{\mathcal{L}}$. MMD-GAN has shown that such kernel losses avoid gradient vanishing caused by JS/KL divergence on disjoint supports. This paper further generalizes this to the conditional setting and completely removes the discriminator, bypassing common mode collapse and min-max instability in conditional GANs while only requiring the maintenance of a single generator network.

**3. One-shot Sampling + ReLU Network Function Class: Compressing Distribution Info into Weights**

Sampling at test time relies on the "noise outsourcing" lemma—for a joint distribution $(Y, X)$, there exists a Borel measurable $\bar g$ and independent noise $\eta$ such that $(Y, X)\overset{d}{=}(\bar g(\eta, X), X)$. Therefore, by learning $\hat g$ to approximate $\bar g$ within a ReLU network class $\mathcal{G}_{\mathcal{H},\mathcal{W},\mathcal{S},\mathcal{B}}$ (depth $\mathcal{H}$, width $\mathcal{W}$, sparsity $\mathcal{S}$, $\ell_\infty$ bound $\mathcal{B}$), sampling is completed in one step: $\eta\sim\mathcal{N}(0, I_m)\to\hat g(\eta, x)$. The bottleneck in diffusion models arises from spreading distribution modeling across hundreds of denoising steps; CGMMD focuses distribution information into the weights of a single network. The ECMMD loss ensures consistency between the generated and true distributions, making a single forward pass sufficient, which is two to three orders of magnitude faster than diffusion.

### Loss & Training
The core loss is $\hat{\mathcal{L}}(g) = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}(W_{i,g}, W_{j,g})$, where $\mathsf{H}(W_i, W_j) = \mathsf{K}(Y_i, Y_j) - \mathsf{K}(Y_i, g_j) - \mathsf{K}(g_i, Y_j) + \mathsf{K}(g_i, g_j)$. Experiments use a Gaussian kernel, block size of 200, and a $k_B$-NN graph reconstructed per batch. Theoretically, it requires $k_n = o(\sqrt n)$ and network scale satisfying $\mathcal{B}^2\mathcal{H}\mathcal{S}\log\mathcal{S}\log n / n \to 0$. The accompanying non-asymptotic theory (Theorem 4.4) is given under Assumption 2.1 (bounded, characteristic kernels), 4.1 (network scale conditions), and 4.2 ($X$ is sub-Gaussian, $\bar g$ is uniformly continuous, Lipschitz sensitivity of conditional mean embeddings). With probability at least $1-\delta$, $\mathcal{L}(\hat g) \lesssim \frac{\mathrm{polylog}\, n}{n^{1/(2d)}} + \sqrt{\frac{\mathcal{B}^2\mathcal{H}\mathcal{S}\log\mathcal{S}\log n}{n}} + \omega_{\bar g}\!\big(\frac{2\sqrt{\log n}}{(\mathcal{H}\mathcal{W})^{1/(d+m)}}\big) + \sqrt{\frac{\log(1/\delta)}{n}}$. These three terms correspond to the stochastic error of the $k$-NN estimator, the network generalization error, and the network approximation error. When $X$ is concentrated on a low-dimensional manifold, the dimension $d$ can be replaced by the intrinsic dimension $\bar d$. Corollary 4.5 further proves that the conditional distribution induced by $\hat g$ converges to the true conditional distribution in terms of MMD and characteristic functions.

## Key Experimental Results

### Main Results

| Task / Dataset | Setting | Key Findings |
|---|---|---|
| Bivariate Helix (Synthetic) | $\sigma \in \{0.2, 0.4, 0.6\}$ | At low noise $\sigma=0.2$, all three methods recover the helix structure. As noise increases, CGMMD preserves the "eye" of the helix, while GCDS and WGAN significantly degrade. |
| MNIST 4× Super-resolution | $7\times 7 \to 28\times 28$ | Reconstruction of digits $\{0\dots4\}$ is clear. |
| STL-10 4× Super-resolution | $3\times 24\times 24 \to 3\times 96\times 96$ | The mean reconstruction map is clear, and pixel-level standard deviation maps show significant diversity in generated outcomes. |
| MNIST Denoising | $\sigma=0.5$, Digits $\{5\dots9\}$ | CGMMD restores clean glyphs. |
| CelebHQ Denoising | $3\times 64\times 64$, $\sigma=0.25$ | Reconstructed faces preserve facial structures. |

### Comparison with Diffusion Models (MNIST Denoising, $\sigma=0.9$)

| Model | PSNR | SSIM | Time/batch (s) | Time/img (s) |
|---|---|---|---|---|
| Diffusion (CFG) | 13.326 | 0.861 | 6.94 | $5.42\times 10^{-2}$ |
| Distilled Diffusion | 10.658 | 0.508 | $1.18\times 10^{-1}$ | $9.2\times 10^{-4}$ |
| **CGMMD** | 8.922 | 0.718 | $7.21\times 10^{-2}$ | $\mathbf{5.6\times 10^{-4}}$ |

### Key Findings
- On high-noise synthetic tasks, CGMMD shows a significant stability advantage over GCDS/WGAN—WGAN often fails to train without $\ell_1$ regularization, as explicitly noted by the authors.
- Comparison with diffusion models demonstrates a clear speed-quality trade-off: CGMMD sampling per image is approximately 100x faster than CFG diffusion. While PSNR lags, SSIM remains respectable; compared to distilled diffusion, it has comparable speed but higher SSIM.
- The ECMMD + $k$-NN framework is adaptive to the intrinsic dimension of $X$ (synthetic experiments in Appendix C.2), validating the theoretical claim of $d \to \bar d$ in practice.

## Highlights & Insights
- **Embedding $k$-NN as a "conditional expectation approximator" into MMD estimation** is a simple yet powerful design. it inherits the stability of unconditional MMD-GAN while naturally introducing conditional dependence. It bypasses bandwidth selection and turns the "neighbor" concept into summation indices within a differentiable objective.
- The combination of one-shot sampling and non-adversarial training makes CGMMD highly attractive for the "lightweight conditional sampler" niche. Many simulation-based inference or posterior sampling tasks are sensitive to per-sample latency, where diffusion models are unsuitable.
- The proof of "uniform concentration of $k$-NN-type nonlinear functionals" is an independently interesting tool that can be transferred to other statistical learning problems relying on conditional mean estimation (e.g., conditional independence testing, conditional expectation regression).
- The adaptation of the main result to the intrinsic dimension $\bar d$ provides a usable bound under mild "high-dimensional but manifold" assumptions, consistent with the intuition of real-world data distributions.

## Limitations & Future Work
- The current theory requires the network scale to grow with the sample size, which does not directly cover fixed-architecture networks; on image tasks, PSNR currently cannot match specialized diffusion/super-resolution models.
- Training requires reconstructing a $k_B$-NN graph for each mini-batch. The overhead of graph construction can be significant with large batches or high dimensions; the paper does not discuss approximate nearest neighbors or caching strategies.
- Experiments are limited to relatively small image datasets like MNIST, FashionMNIST, CelebHQ, and STL-10, without addressing high-resolution natural images or text-to-image conditional generation. The impact of kernel choice (Gaussian bandwidth) on actual performance under high-dimensional $Y$ is not detailed.
- Future Work: Generalizing the loss to flow-matching / OT-flow objectives, replacing $k$-NN with more scalable structures (e.g., differentiable ANN), and extending the theory to finite approximation error settings for fixed-architecture networks.

## Related Work & Insights
- **vs GCDS (Zhou et al., 2023)**: GCDS uses a GAN formulation for conditional sampling, suffering from min-max optimization and mode collapse. CGMMD uses ECMMD for direct minimization, removes the discriminator, and adds consistency proofs.
- **vs Conditional Wasserstein-GAN (Song et al., 2025)**: W-GAN uses Wasserstein distance for conditional IPM, but training is sensitive to $\ell_1$ regularization. CGMMD uses kernel MMD, which has a smooth loss and more stable training.
- **vs Conditional Diffusion (Ho & Salimans, 2021)**: Diffusion has higher iterative sampling quality but takes ~50 ms per image. CGMMD takes ~0.56 ms via a single forward pass, a two-order-of-magnitude difference, making it suitable for scientific computing / posterior approximation requiring massive sampling.
- **vs Unconditional MMD-GAN (Li et al., 2015; Bińkowski et al., 2018)**: This work is a statistical generalization to the conditional setting—providing both a $k$-NN estimator and non-asymptotic bounds.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining ECMMD with $k$-NN as a training objective for conditional generation, supported by non-asymptotic theory, is a clear and previously unseen path.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic and three image tasks cover proof-of-concept, but lack large-scale benchmarks and head-to-head comparisons with SOTA diffusion.
- Writing Quality: ⭐⭐⭐⭐ Derivations are rigorous, notation is consistent, and theorems are well-integrated with algorithms; the independent $k$-NN concentration results in the appendix are a highlight.
- Value: ⭐⭐⭐⭐ Directly valuable for the scientific computing / simulation-based inference community that needs fast conditional sampling with theoretical guarantees. The framework is easily extendable to flow-based methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AceTone: Bridging Words and Colors for Conditional Image Grading](../../CVPR2026/image_restoration/acetone_bridging_words_and_colors_for_conditional_image_grading.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[ICLR 2026\] A Statistical Benchmark for Diffusion-Posterior-Sampling Algorithms](../../ICLR2026/image_restoration/a_statistical_benchmark_for_diffusion-posterior-sampling_algorithms.md)
- [\[CVPR 2026\] Zero-Shot Image Denoising via Hybrid Prior-Guided Pseudo Sample Generation](../../CVPR2026/image_restoration/zero-shot_image_denoising_via_hybrid_prior-guided_pseudo_sample_generation.md)
- [\[ICLR 2026\] Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling](../../ICLR2026/image_restoration/adaptive_moments_are_surprisingly_effective_for_plug-and-play_diffusion_sampling.md)

</div>

<!-- RELATED:END -->

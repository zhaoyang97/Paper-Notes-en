---
title: >-
  [Paper Note] One-shot Conditional Sampling: MMD meets Nearest Neighbors
description: >-
  [ICML 2026][Image Restoration][Conditional Sampling] CGMMD utilizes $k$-nearest neighbor graphs to estimate "Expected Conditional MMD (ECMMD)" as a directly minimizable non-adversarial objective. It trains a conditional…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Conditional Sampling"
  - "MMD"
  - "Nearest Neighbors Estimation"
  - "One-shot Generation"
  - "Kernel Mean Embedding"
date: 2026-05-08
content_hash: 886173ca3d573951
---

# One-shot Conditional Sampling: MMD meets Nearest Neighbors

**Conference**: ICML 2026  
**arXiv**: [2509.25507](https://arxiv.org/abs/2509.25507)  
**Code**: https://github.com/anirbanc96/cgmmd (Available)  
**Area**: Scientific Computing / Conditional Generation / Kernel Methods  
**Keywords**: Conditional Sampling, MMD, Nearest Neighbors Estimation, One-shot Generation, Kernel Mean Embedding

## TL;DR
CGMMD utilizes $k$-nearest neighbor graphs to estimate "Expected Conditional MMD (ECMMD)" as a directly minimizable non-adversarial objective. It trains a conditional generator capable of sampling from $P_{Y\mid X}$ in a single forward pass, providing non-asymptotic error bounds and proofs of distributional convergence.

## Background & Motivation
**Background**: Conditional distribution modeling is a fundamental problem in statistics and machine learning. Regression only provides conditional means or quantiles, whereas many downstream tasks (uncertainty quantification, simulation-based inference, graphical models, dimensionality reduction) require the full $P_{Y\mid X}$. Modern mainstream approaches include Conditional GANs, CVAEs, and Conditional Diffusion Models, which reformulate "density estimation" as "generating samples using noise $\eta$ and input $x$."

**Limitations of Prior Work**: The three main categories of methods have distinct drawbacks. Conditional GANs employ min-max optimization and rely on JS/KL divergence; when the generator and target distributions are supported on low-dimensional manifolds, they often become disjoint, leading to vanishing gradients, training instability, and mode collapse. Wasserstein/IPM-based losses (e.g., W-GAN, MMD-GAN) mitigate instability in unconditional settings but lack finite-sample theory and simple $k$-nearest neighbor estimators in **conditional** scenarios. Conditional Diffusion is stable but requires dozens to thousands of iterative denoising steps, resulting in high inference complexity.

**Key Challenge**: A trade-off exists between training stability, statistical consistency, and sampling time—adversarial losses sacrifice stability for flexibility, diffusion sacrifices sampling speed for sample quality, and IPM-based objectives lack statistical guarantees.

**Goal**: Construct a conditional sampling framework that simultaneously satisfies: (i) **non-adversarial and directly minimizable**; (ii) **single forward pass** sampling; (iii) existence of non-asymptotic error bounds and provable convergence to the true distribution.

**Key Insight**: Chatterjee et al. (2024) generalized MMD to Expected Conditional MMD (ECMMD), proving it is a strictly proper scoring rule (ECMMD$^2 = 0$ if and only if conditional distributions are equal). However, using ECMMD as a training loss requires a consistent estimator from finite samples. $k$-nearest neighbors ($k$-NN) have long been classic tools for conditional mean estimation. By grafting $k$-NN onto the U-statistic kernel function of ECMMD, one can obtain a non-adversarial, non-iterative, and end-to-end differentiable objective.

**Core Idea**: Use a $k$-NN graph to approximate the expectation "conditioned on $X=X_i$." Feed both generator outputs and real samples into the kernel function $\mathsf{H}$ to directly minimize the resulting ECMMD estimator. After training, for any given $x$, a simple forward pass with sampled noise $\eta$ yields $\hat g(\eta, x) \sim P_{Y\mid X=x}$.

## Method

### Overall Architecture
Input: Training pairs $\{(Y_i, X_i)\}_{i=1}^n$, reference noise distribution $P_\eta = \mathcal{N}(0, I_m)$, kernel function $\mathsf{K}$, and a generator function class $\mathcal{G}$ (implemented as a ReLU network).

The process consists of four steps: (1) Sample auxiliary noise $\eta_i$ for each sample and perform a forward pass to obtain pseudo-samples $g(\eta_i, X_i)$; (2) Construct a directed $k$-NN graph $G(\mathcal{X}_n)$ on the mini-batch $X$; (3) Sum over $k$-NN pairs on the graph to obtain the empirical loss $\hat{\mathcal{L}}(g)$ as a consistent estimate of ECMMD$^2$; (4) Backpropagate to update generator parameters.

Output: A trained conditional generator $\hat g$. **One-shot sampling**: Given a new $x$, independently sample $\eta \sim P_\eta$; the output $\hat g(\eta, x)$ represents a sample from $P_{Y\mid X=x}$.

### Key Designs

1.  **$k$-NN Estimator of ECMMD**:
    - **Function**: Approximates the "MMD with expectation taken over conditional $X$" in a purely empirical form, allowing it to serve as a differentiable loss.
    - **Mechanism**: The kernel trick is first used to express ECMMD$^2$ as $\mathbb{E}[\mathsf{H}(W, W')]$ (where $W=(Y,Z)$ and $\mathsf{H}$ is composed of four kernel evaluations). The tower property is then used to decompose the outer expectation over $X$ and the inner expectation over $Y,Z\mid X$. For the inner conditional expectation, the authors construct a $k$-NN directed graph $G(\mathcal{X}_n)$ on $X$ instead of using kernel regression. Samples in the neighbor set $N_{G}(i)$ are treated as pseudo-replicates under "approximately the same" conditions, yielding the estimator $\widehat{\mathrm{ECMMD}}^2 = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}(W_i, W_j)$.
    - **Design Motivation**: Compared to kernel regression, $k$-NN does not require bandwidth selection, is more dimension-friendly (adapting to the intrinsic dimension $\bar d$ of $X$), and is decoupled from generator parameters—the graph depends only on $X_i$, and only $g$ appears in the summation, allowing gradients to flow without reparameterization tricks.

2.  **Non-adversarial Direct Minimization Objective**:
    - **Function**: Formulates training as a pure minimization problem over generator parameters $\theta$, avoiding the min-max optimization of GANs.
    - **Mechanism**: The objective is $\hat g \in \arg\min_{g\in\mathcal{G}} \hat{\mathcal{L}}(g)$, where $\hat{\mathcal{L}}(g) = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}((Y_i, g(\eta_i, X_i)), (Y_j, g(\eta_j, X_j)))$. Algorithm 1 details the training loop: for each mini-batch, reconstruct the $k_B$-NN graph, obtain $g(\eta_i, X_i)$, calculate $\hat{\mathcal{L}}$, and update $\theta \leftarrow \theta - \alpha \nabla_\theta \hat{\mathcal{L}}$.
    - **Design Motivation**: The MMD-GAN approach has proven effective in avoiding gradient issues when supports are disjoint in unconditional settings. By extending this to the conditional setting and removing the discriminator, the authors avoid common GAN issues like mode collapse and min-max instability, requiring only a single generator network for implementation.

3.  **One-shot Sampling + Neural Network Function Class**:
    - **Function**: Outputs conditional samples for any $x$ in a single forward pass during inference, which is two to three orders of magnitude faster than the iterative denoising of diffusion models.
    - **Mechanism**: Based on the noise outsourcing lemma, for a joint distribution $(Y, X)$, there exists a Borel measurable $\bar g$ and independent noise $\eta$ such that $(Y, X) \overset{d}= (\bar g(\eta, X), X)$. CGMMD learns a $\hat g$ within the ReLU network class $\mathcal{G}_{\mathcal{H},\mathcal{W},\mathcal{S},\mathcal{B}}$ (depth $\mathcal{H}$, width $\mathcal{W}$, parameters $\mathcal{S}$, $\ell_\infty$ bound $\mathcal{B}$) to approximate $\bar g$. The sampling stage is simply $\eta \sim \mathcal{N}(0, I_m) \to \hat g(\eta, x)$.
    - **Design Motivation**: The time bottleneck of iterative sampling in diffusion models stems from distributing "distribution modeling" across multiple denoising steps. CGMMD integrates distribution information into the weights of a single network, ensuring consistency between the generated and true distributions via the ECMMD loss, thus requiring only one forward pass. This is its most direct practical advantage over diffusion models.

### Loss & Training
Loss: $\hat{\mathcal{L}}(g) = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}(W_{i,g}, W_{j,g})$, where $\mathsf{H}(W_i, W_j) = \mathsf{K}(Y_i, Y_j) - \mathsf{K}(Y_i, g_j) - \mathsf{K}(g_i, Y_j) + \mathsf{K}(g_i, g_j)$. In experiments, a Gaussian kernel is used with a batch size of 200, and the $k_B$-NN graph is constructed per batch on $X$. Theoretically, it requires $k_n = o(\sqrt n)$ and network scaling such that $\mathcal{B}^2 \mathcal{H}\mathcal{S}\log\mathcal{S}\log n / n \to 0$.

### Theory (Theorem 4.4 and Corollary 4.5)
Under Assumption 2.1 (bounded, characteristic kernel), Assumption 4.1 (network scale conditions), and Assumption 4.2 (sub-Gaussian $X$, uniformly continuous $\bar g$, Lipschitz sensitivity of conditional mean embeddings), the error of $\hat g$ satisfies, with probability at least $1-\delta$:
$\mathcal{L}(\hat g) \lesssim \frac{\mathrm{polylog}\, n}{n^{1/(2d)}} + \sqrt{\frac{\mathcal{B}^2 \mathcal{H}\mathcal{S}\log\mathcal{S}\log n}{n}} + \omega_{\bar g}\!\left(\frac{2\sqrt{\log n}}{(\mathcal{H}\mathcal{W})^{1/(d+m)}}\right) + \sqrt{\frac{\log(1/\delta)}{n}}$.
The three terms correspond respectively to: $k$-NN estimation stochastic error, neural network generalization error, and neural network approximation error. When $X$ is concentrated on a low-dimensional manifold, $d$ can be replaced by the intrinsic dimension $\bar d$. Corollary 4.5 further proves that the conditional distribution induced by $\hat g$ converges to the true conditional distribution in both MMD and characteristic function senses.

## Key Experimental Results

### Main Results

| Task / Dataset | Setup | Key Observation |
|---|---|---|
| Bivariate Helix (Synthetic) | $\sigma \in \{0.2, 0.4, 0.6\}$ | At low noise $\sigma=0.2$, all three methods recover the helix; as noise increases, CGMMD maintains the helix "eye" while GCDS and WGAN significantly degrade. |
| MNIST 4× Super-res | $7\times 7 \to 28\times 28$ | Clear reconstruction of digits $\{0\dots4\}$. |
| STL-10 4× Super-res | $3\times 24\times 24 \to 3\times 96\times 96$ | Sharp mean reconstruction; pixel-level standard deviation maps show significant diversity in generated results. |
| MNIST Denoising | $\sigma=0.5$, digits $\{5\dots9\}$ | CGMMD restores clean glyph shapes. |
| CelebHQ Denoising | $3\times 64\times 64$, $\sigma=0.25$ | Reconstructed faces preserve facial structures. |

### Comparison with Diffusion Models (MNIST Denoising, $\sigma=0.9$)

| Model | PSNR | SSIM | Time/batch (s) | Time/img (s) |
|---|---|---|---|---|
| Diffusion (CFG) | 13.326 | 0.861 | 6.94 | $5.42\times 10^{-2}$ |
| Distilled Diffusion | 10.658 | 0.508 | $1.18\times 10^{-1}$ | $9.2\times 10^{-4}$ |
| **Ours (CGMMD)** | 8.922 | 0.718 | $7.21\times 10^{-2}$ | $\mathbf{5.6\times 10^{-4}}$ |

### Key Findings
- CGMMD shows a significant stability advantage over GCDS/WGAN on high-noise synthetic tasks—the authors explicitly note that WGAN often fails to train without $\ell_1$ regularization.
- The comparison with diffusion reveals a clear speed-quality trade-off: CGMMD sampling is approx. 100 times faster per image than CFG Diffusion; while it lags in PSNR, it maintains reasonable SSIM. It shares the same speed tier as distilled diffusion but achieves better SSIM.
- The ECMMD + $k$-NN framework exhibits adaptability to the intrinsic dimension of $X$ (synthetic experiments in Appendix C.2), confirming that the $d \to \bar d$ theoretical claim is observable in practice.

## Highlights & Insights
- **Embedding $k$-NN as a "conditional expectation approximator" within MMD estimation** is a simple yet powerful design—it inherits the stability of unconditional MMD-GAN while naturally introducing conditional dependence. It bypasses the bandwidth selection of kernel regression and transforms the "neighbor" concept into summation indices in a differentiable objective.
- The combination of one-shot sampling and non-adversarial training makes CGMMD highly attractive as a "lightweight conditional sampler"—tasks like simulation-based inference and posterior sampling are sensitive to single-sample latency, where diffusion models are less suitable.
- The proven "uniform concentration of $k$-NN-type nonlinear functionals" is an independently interesting tool that can be transferred to other statistical learning problems relying on conditional mean estimation (e.g., conditional independence tests).
- The main results provide a usable bound under the "high-dimensional but manifold" assumption by adapting to intrinsic dimension $\bar d$, aligning with the intuition of data manifold distributions in the real world.

## Limitations & Future Work
- The authors acknowledge that current theory requires the network scale to grow with sample size, which does not directly cover fixed-architecture networks; image task PSNR currently lags behind specialized diffusion/super-resolution models.
- Constructing the $k_B$-NN graph for each mini-batch entails overhead that becomes non-negligible at large batch sizes or high dimensions; approximate nearest neighbors or caching strategies were not discussed.
- Experiments are limited to relatively small image datasets (MNIST, FashionMNIST, CelebHQ, STL-10) and do not address high-resolution natural images or text-to-image conditional generation; the impact of kernel choice (Gaussian bandwidth) on performance for high-dimensional $Y$ lacks detailed analysis.
- Future work: Extending the loss to flow-matching/OT-flow objectives, replacing $k$-NN with more scalable neighbor structures (e.g., differentiable ANN), and extending theory to finite approximation error settings for fixed-architecture networks.

## Related Work & Insights
- **vs GCDS (Zhou et al., 2023)**: GCDS uses a GAN formulation for conditional sampling, which involves min-max optimization and is prone to mode collapse; CGMMD uses ECMMD for direct minimization, removes the discriminator, and provides consistency proofs.
- **vs Wasserstein-GAN Conditional version (Song et al., 2025)**: W-GAN uses Wasserstein distance as a conditional IPM but is sensitive to $\ell_1$ regularization; CGMMD uses kernel MMD, which provides smooth losses and more stable training.
- **vs Conditional Diffusion (Ho & Salimans, 2021)**: Diffusion offers higher quality via iterative sampling but at ~50 ms per image; CGMMD samples in a single step at ~0.56 ms (two orders of magnitude faster), suitable for scientific computing / posterior approximation requiring mass sampling.
- **vs MMD-GAN Unconditional (Li et al., 2015; Bińkowski et al., 2018)**: This work is a statistical generalization to the conditional setting—providing both a $k$-NN estimator and non-asymptotic bounds.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining ECMMD with $k$-NN as a training objective for conditional generation, supported by non-asymptotic theory, is a clear and novel path.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic + three image tasks provide proof-of-concept, but lacks large-scale benchmarks and head-to-head comparisons against SOTA diffusion models.
- Writing Quality: ⭐⭐⭐⭐ Rigorous derivations, consistent notation, and well-integrated theorems and algorithms; the independent $k$-NN concentration results in the appendix are a highlight.
- Value: ⭐⭐⭐⭐ Directly useful for the scientific computing / simulation-based inference communities that require fast conditional sampling with theoretical guarantees; the framework is easily extensible to flow-based methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[CVPR 2026\] AceTone: Bridging Words and Colors for Conditional Image Grading](../../CVPR2026/image_restoration/acetone_bridging_words_and_colors_for_conditional_image_grading.md)
- [\[ICLR 2026\] SoFlow: Solution Flow Models for One-Step Generative Modeling](../../ICLR2026/image_restoration/soflow_solution_flow_models_for_one-step_generative_modeling.md)
- [\[AAAI 2026\] Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation](../../AAAI2026/image_restoration/depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat.md)
- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](../../CVPR2026/image_restoration/fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)

</div>

<!-- RELATED:END -->

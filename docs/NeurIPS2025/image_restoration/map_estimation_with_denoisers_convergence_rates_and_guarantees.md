---
title: >-
  [Paper Note] MAP Estimation with Denoisers: Convergence Rates and Guarantees
description: >-
  [NeurIPS 2025][Image Restoration][MAP estimation] This paper proves that a simple iterative averaging algorithm based on MMSE denoisers—closely related to practical methods such as Cold Diffusion—provably converges to th…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "MAP estimation"
  - "proximal operator"
  - "denoiser"
  - "convergence rates"
  - "inverse problems"
  - "plug-and-play"
date: 2026-05-08
content_hash: 825366a858e3ecda
---

# MAP Estimation with Denoisers: Convergence Rates and Guarantees

**Conference**: NeurIPS 2025
**arXiv**: [2507.15397](https://arxiv.org/abs/2507.15397)  
**Code**: To be confirmed  
**Area**: Image Restoration
**Keywords**: MAP estimation, proximal operator, denoiser, convergence rates, inverse problems, plug-and-play

## TL;DR
This paper proves that a simple iterative averaging algorithm based on MMSE denoisers—closely related to practical methods such as Cold Diffusion—provably converges to the proximal operator of the negative log-prior under log-concave prior assumptions, achieving a convergence rate of $\tilde{O}(1/k)$. The work provides rigorous theoretical foundations for a class of denoising methods that have demonstrated empirical success but lacked theoretical guarantees, and embeds the approach within a proximal gradient descent framework for MAP estimation.

## Background & Motivation
- **Background**: Inverse problems (deblurring, super-resolution, denoising, etc.) are ubiquitous in science and engineering. Classical approaches model them as optimization problems consisting of a data fidelity term plus a regularization term. MAP (Maximum a Posteriori) estimation is a principled framework: $\arg\min \lambda f(x) - \ln p(x)$.
- **Denoisers as Priors**: Modern methods leverage pre-trained denoisers and generative models to learn the true image distribution $p$, constructing data-driven priors. Diffusion and flow-matching models provide powerful means of distribution learning.
- **Limitations of Prior Work**: Plug-and-Play (PnP) methods replace the intractable proximal operator $\text{prox}_{-\tau \ln p}$ with a pre-trained denoiser. While experimentally effective, denoisers are not designed to match the proximal operator, and convergence to the true MAP objective cannot be guaranteed.
- **Key Challenge**: Cohen et al. and Hurault et al. parameterize denoisers as $D_\sigma = I - \nabla g_\sigma$, which can be shown to be the proximal operator of some explicit functional—but not the desired $-\ln p$.
- **Limitations of Cold Diffusion**: Bansal et al. (2023) propose Cold Diffusion, which alternates between denoising steps and corruption steps toward the observed data, achieving strong empirical results but lacking convergence guarantees; extending the number of iterations with default parameters can lead to divergence.
- **Core Problem**: Existing methods either fail to guarantee convergence to the MAP solution, converge to an incorrect objective, or lack convergence rates. A rigorous theoretical connection between denoising iteration schemes and MAP optimization is needed.

## Method

### Overall Architecture
The paper proposes the MMSE Averaging recursion: $x_{k+1} = (1 - \alpha_k)\,\text{MMSE}_{\sigma_k}(x_k) + \alpha_k y$, where $\text{MMSE}_\sigma(z) = \mathbb{E}[X \mid X + \sigma\varepsilon = z]$ is the theoretically optimal denoiser. Via Tweedie's identity, this recursion is reinterpreted as gradient descent on a sequence of smoothed proximal objectives $F_{\sigma_k}$, which converge to the true proximal operator as $\sigma_k \to 0$. The proximal operator approximation is then embedded within a proximal gradient descent framework to solve the MAP estimation problem.

### Module 1: Equivalence Between MMSE Averaging and Gradient Descent on Smoothed Proximal Objectives
- **Function**: Establishes an equivalence between MMSE denoising iterations and gradient descent.
- **Mechanism**: Applying Tweedie's identity $\text{MMSE}_\sigma(z) = z + \sigma^2 \nabla \ln p_\sigma(z)$, the recursion is rewritten as $x_{k+1} = x_k - \alpha_k \nabla F_{\sigma_k}(x_k)$, where $F_{\sigma_k}(x) = \frac{1}{2}\|y - x\|^2 - \tau \ln p_{\sigma_k}(x)$. Setting $\alpha_k = 1/(k+2)$ and $\sigma_k^2 = \tau/(k+1)$ realizes this equivalence.
- **Design Motivation**: Directly analyzing denoising iterations makes it difficult to obtain convergence guarantees. The equivalence enables rigorous analysis using classical optimization theory (descent lemma, strong convexity). The smoothed objective $F_\sigma$ converges to the true proximal objective $F$ as $\sigma \to 0$.

### Module 2: Condition Number Analysis of the Smoothed Objective
- **Function**: Proves that the smoothed objective $F_\sigma$ possesses a far better condition number than the original objective $F$.
- **Mechanism**: The smoothness constant of $F_\sigma$ is $L_\sigma = 1 + \tau/\sigma^2$ and the strong convexity constant is $\mu_\sigma = 1$ (under log-concavity), giving condition number $\kappa_\sigma = 1 + \tau/\sigma^2$. For large $\sigma$, $F_\sigma$ approaches isotropy ($\kappa \to 1$); for example, at $\sigma = \sqrt{\tau}$, $\kappa = 2$, far superior to the potentially large condition number of the original problem.
- **Design Motivation**: The condition number of the original proximal objective $F$ may be dominated by the smoothness constant $L$ of $-\ln p$, which can be arbitrarily large, causing extremely slow gradient descent convergence. Gaussian convolution smoothing yields a favorable condition number for large $\sigma$ but shifts the optimum, while small $\sigma$ preserves accuracy at the cost of a poor condition number. The decreasing schedule $\sigma_k$ achieves the optimal balance.

### Module 3: Convergence to the Proximal Operator (Theorem 1)
- **Function**: Proves that the MMSE Averaging iterates $x_k$ converge to the true proximal operator $\text{prox}_{-\tau \ln p}(y)$ at rate $\tilde{O}(1/k)$.
- **Mechanism**: The upper bound is $\|x_k - \text{prox}_{-\tau \ln p}(y)\| \leq \frac{(\ln k) + 7}{k+1} \cdot \left[\|y - \text{prox}_{-\tau \ln p}(y)\| + \tau^2 M \sqrt{d}\right]$, where $M$ is an upper bound on the Frobenius norm of the third-order derivatives of $\ln p$. Crucially, the convergence rate does not depend on the smoothness constant $L$ of $-\ln p$ (which may be arbitrarily large), but only on the third-order derivative bound.
- **Design Motivation**: Compared to the iteration complexity $O(L \cdot \log(1/\varepsilon))$ of naive gradient descent on a Gaussian prior, MMSE Averaging achieves $O(1/\varepsilon)$, offering a fundamental advantage when $L$ is large. The algorithm is parameter-free—the sequences $\alpha_k$ and $\sigma_k$ depend only on the regularization parameter $\tau$, requiring no knowledge of problem-specific properties such as smoothness constants.

### Module 4: Extension to MAP Estimation (Theorem 2, Approx PGD)
- **Function**: Embeds the proximal operator approximation into proximal gradient descent (PGD) to solve the full MAP problem $\arg\min \lambda f(x) - \ln p(x)$.
- **Mechanism**: The outer loop performs PGD: first take a data fidelity gradient step $z_0 \leftarrow \hat{x}^n - \tau\lambda \nabla f(\hat{x}^n)$, then use the MMSE Averaging inner loop to approximately compute $\text{prox}_{-\tau \ln p}(z_0)$. The number of inner iterations $k_n = \lfloor c \cdot n^{1+\eta} \rfloor$ increases progressively to improve approximation accuracy. Convergence guarantee: average MAP error $O(1/n)$, approximation error $\tilde{O}(1/n^{1+\eta})$.

### Loss & Training
- **Assumption 1**: The prior $p$ is log-concave and strictly positive on $\mathbb{R}^d$—ensuring $F$ is strongly convex with a unique minimizer.
- **Assumption 2**: The third-order derivatives of $\ln p$ are bounded (upper bound $M$)—controlling the drift of the smoothed objective's minimizer as a function of $\sigma$.
- **Assumption 3**: The data fidelity term $f$ is convex, bounded below, and $L_f$-smooth—standard optimization assumption.
- When the approximate score (neural network denoiser) satisfies $\|\xi_k\| \leq \xi$, the iterates converge to within $O(\xi)$ of the true proximal point.

## Key Experimental Results

### Table 1: Comparison of Convergence Behavior on a 2D Gaussian Prior

| Method | Iteration Complexity at $\kappa=500$ | Depends on Condition Number | Rate on Gaussian Prior ($M=0$) |
|---|---|---|---|
| Naive GD on $F$ | $O(L \cdot \log(1/\varepsilon))$ | Yes (linear in $L$) | $O(500 \cdot \log(1/\varepsilon))$ |
| **MMSE Averaging** | **$O(1/\varepsilon)$** | **No** (depends only on $M$ and $d$) | **$\tilde{O}(1/k)$, optimal when $M=0$** |

### Table 2: Summary of Theoretical Convergence Guarantees

| Setting | Algorithm | Convergence Target | Rate | Key Dependency |
|---|---|---|---|---|
| Proximal operator computation | MMSE Averaging | $\text{prox}_{-\tau \ln p}(y)$ | $\tilde{O}(1/k)$ | $\tau^2 M\sqrt{d}$, independent of $L$ |
| Approximate score | MMSE Avg + error $\xi$ | $O(\xi)$-neighborhood | $\tilde{O}(1/k)$ | $\xi$: score approximation error |
| MAP estimation (outer loop) | Approx PGD | $x^*_{\text{MAP}}$ | $O(1/n)$ function value | Inner loop $k_n = O(n^{1+\eta})$ |
| Low-dimensional subspace prior | MMSE Averaging | $\text{prox}_{-\tau \ln p}(y)$ | $\tilde{O}(1/k)$ | $d \to r$ (effective dimension) |

### Key Findings
- **Condition-Number-Free Convergence**: Figure 2 shows that on a 2D Gaussian prior with $\kappa=500$, naive GD nearly stalls while MMSE Averaging converges rapidly, exhibiting a clear $O(1/k)$ rate.
- **Visualization of Smoothing Effect**: Figure 1 shows the level curves of $F_\sigma$ transitioning from severely anisotropic ($\sigma=0$) to nearly isotropic (large $\sigma$), directly validating the theory that smoothing improves the condition number.
- **Parameter-Free Advantage**: The sequences $\alpha_k = 1/(k+2)$ and $\sigma_k^2 = \tau/(k+1)$ are fully determined, requiring no tuning and eliminating hyperparameter search costs.
- **Comparison with Cold Diffusion**: Cold Diffusion uses a fixed ratio $\alpha_k = k/N$ but can diverge; this work proves convergence guarantees under a specific scheduling strategy.

## Highlights & Insights
1. **Elegant Theoretical Connection**: Tweedie's identity enables an exact equivalence between seemingly heuristic denoising iterations and classical gradient descent, making rigorous analysis tractable.
2. **Bypassing the Condition Number**: The decreasing noise schedule cleverly achieves an optimal trade-off between "good condition number but shifted objective" and "accurate objective but poor condition number."
3. **Innovative Use of PDE Tools**: Applying the heat equation to control the drift of the minimizer as a function of noise level is a key technical innovation in the analysis.
4. **Practical Guidance**: The work provides a theoretical explanation for the divergence of empirical methods such as Cold Diffusion—specifically, the use of inappropriate weight schedules.

## Limitations & Future Work
- **Log-Concavity Assumption Is Restrictive**: Real image priors are far from log-concave; Gaussian mixtures and natural image distributions severely violate this assumption, limiting the direct applicability of the theoretical guarantees in practice.
- **No Real Image Experiments**: Numerical experiments are limited to 2D Gaussian prior visualization, with no empirical validation on real inverse problems (deblurring, super-resolution, inpainting).
- **Practical Approximation of the MMSE Denoiser**: The theory assumes access to exact MMSE denoisers; in practice, the deviation of neural network denoisers from the MMSE is difficult to quantify and control precisely.
- **Iteration Complexity**: The Approx PGD inner loop count $k_n = O(n^{1+\eta})$ grows progressively, and the total computational cost may be substantial.
- **Accelerated Methods Not Analyzed**: The combination with accelerated proximal methods such as FISTA is not analyzed, which may yield faster rates; the authors list this as future work.
- **Dimension Dependence**: The $\tau^2 M\sqrt{d}$ term in the convergence bound grows with dimension $d$, which may affect practical convergence in high-dimensional problems.

## Related Work & Insights
- **Plug-and-Play (Venkatakrishnan et al., 2013)**: PnP replaces the proximal operator with a denoiser but cannot guarantee convergence to the correct functional; the proposed method resolves this issue at a fundamental level.
- **Gradient Step Denoisers (Hurault et al., 2022)**: Parameterizing $D_\sigma = I - \nabla g_\sigma$ is provably the proximal operator of some functional, but not of $-\ln p$. The present work directly proves that MMSE denoising iterations converge to the correct proximal operator.
- **Cold Diffusion (Bansal et al., 2023)**: MMSE Averaging is structurally equivalent to Cold Diffusion, but this work provides a weight schedule that guarantees convergence.
- **PnP-SGD (Laumont et al., 2023)**: A fixed $\sigma$ only permits convergence to the proximal operator of a smoothed density, with convergence rates depending on the smoothness constant of $F_\sigma$ (which may be arbitrarily large). The present work fundamentally improves upon this via a decreasing $\sigma$ schedule.
- **Insight**: The strategy of "smooth first, then progressively refine" has broad applicability in optimization—for any objective with a poor condition number, making rapid progress on a smoothed version and then gradually reducing the smoothing may serve as a general-purpose strategy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (establishes a rigorous theoretical connection between denoising iterations and the proximal operator, filling a core theoretical gap in PnP methods)
- Experimental Thoroughness: ⭐⭐ (limited to 2D Gaussian prior visualization; real inverse problem experiments are absent)
- Writing Quality: ⭐⭐⭐⭐⭐ (theorem statements are concise and clear, intuitions are well-explained, and proof strategies are clearly articulated)
- Value: ⭐⭐⭐⭐ (significant theoretical contribution that provides a theoretical foundation for an important class of practical methods; extension to non-log-concave priors is needed for real-world impact)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)
- [\[NeurIPS 2025\] RGB-to-Polarization Estimation: A New Task and Benchmark Study](rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)
- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[CVPR 2026\] IA-CLAHE: Image-Adaptive Clip Limit Estimation for CLAHE](../../CVPR2026/image_restoration/ia_clahe_image_adaptive_clip_limit.md)

</div>

<!-- RELATED:END -->

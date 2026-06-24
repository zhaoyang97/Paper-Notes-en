---
title: >-
  [Paper Note] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach
description: >-
  [ICML 2026][Optimization][$(L_0] This paper demonstrates that standard first-order and second-order SDEs in current literature completely fail to capture learning rate stability constraints under $(L_0,L_1)$-smoothness (even predicting convergence in regions where the discrete version diverges). By flipping the sign of the curvature term in the drift, the authors construct a family of "stability-faithful" first-order weak approximation SDEs. This enables the first unified ana…
tags:
  - "ICML 2026"
  - "Optimization"
  - "$(L_0"
  - "L_1)$-smoothness"
  - "Distributed Compressed SGD"
  - "SignSGD"
  - "Heavy-tailed noise"
  - "Stability-faithful SDE"
date: 2026-05-08
content_hash: 8d9b3b1e9d188206
---

# On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach

**Conference**: ICML 2026  
**arXiv**: [2506.00181](https://arxiv.org/abs/2506.00181)  
**Code**: TBD  
**Area**: Optimization / Distributed SGD / SDE Continuous-time Analysis  
**Keywords**: $(L_0,L_1)$-smoothness, Distributed Compressed SGD, SignSGD, Heavy-tailed noise, Stability-faithful SDE  

## TL;DR
This paper demonstrates that standard first-order and second-order SDEs in current literature completely fail to capture learning rate stability constraints under $(L_0,L_1)$-smoothness (even predicting convergence in regions where the discrete version diverges). By flipping the sign of the curvature term in the drift, the authors construct a family of "stability-faithful" first-order weak approximation SDEs. This enables the first unified analysis of DCSGD and DSignSGD under compression, affine variance, and heavy-tailed noise, providing specific prescriptions for selecting normalization intensity.

## Background & Motivation

**Background**: Distributed stochastic gradient methods are fundamental to modern large model training. Their behavior is simultaneously governed by batch noise (including heavy-tailed noise as observed by Simsekli et al. 2019), communication compression (quantization, sparsification, sign), and adaptive normalization (Adam, AdaGrad, SignSGD, etc.). SDEs have been widely adopted as continuous-time proxies for discrete optimizers (starting from Li et al. 2017) to study learning rate scheduling, batch size scaling laws, and implicit regularization.

**Limitations of Prior Work**: (1) Almost all SDE analyses rely on global $L$-smoothness, whereas real deep learning losses are closer to $(L_0,L_1)$-smoothness (i.e., $\|\nabla^2 f(x)\|_2 \leq L_0 + L_1\|\nabla f(x)\|_2$, Zhang et al. 2020b), which does not guarantee a stable constant step size for all initializations. (2) Noise is often assumed to be Gaussian or have bounded variance, while affine variance and heavy tails are common in modern DL. (3) There is a lack of unified characterization for DCSGD (distributed SGD with unbiased compression) and DSignSGD when $(L_0,L_1)$-smoothness, compression, and general noise coexist.

**Key Challenge**: The most critical discovery of this paper is that **classic first-order and second-order SDEs fundamentally fail to predict discrete step-size stability under $(L_0,L_1)$-smoothness**. First-order SDEs provide no constraints on the learning rate, while second-order SDEs are worse—predicting "accelerated convergence" in large step-size regimes where discrete GD actually diverges. For a 1D parabola $f(x)=\lambda x^2/2$, GD requires $\eta < 2/\lambda$ for stability, but the classic first-order ODE solution $f(X_t)=f(X_0)e^{-2\lambda t}$ is independent of $\eta$, and the classic second-order ODE $f(X_t)=f(X_0)e^{-2\lambda(1+\lambda\eta/2)t}$ suggests faster convergence for larger $\eta$. For $f(x)=x^4/4$ (a typical non-$L$-smooth but $(L_0,L_1)$-smooth function), no uniform constant step size exists globally for discrete GD, yet classic SDEs still predict global convergence.

**Goal**: (1) Formally identify the flaws in classic SDEs. (2) Derive a **sign-corrected first-order SDE** as a weak approximation for SGD under $(L_0,L_1)$-smoothness. (3) Use this to provide a unified analysis of DCSGD (affine variance + compression) and DSignSGD (heavy-tailed Student-$t$ noise). (4) Provide practical guidelines for normalization intensity.

**Key Insight**: The authors adopt an "ansatz" approach from physics—directly proposing a family of drift-corrected ODEs $dX_t=-\nabla f(X_t)dt+\alpha\eta\nabla^2 f(X_t)\nabla f(X_t)dt$. By matching the "induced loss drift $df(X_t)/dt$" with the "discrete Taylor expansion $(f(x_{k+1})-f(x_k))/\eta$" up to $O(\eta)$, they **uniquely** determine $\alpha=1/2$ and show that the sign must be **+** (not the − found in classic literature).

**Core Idea**: Traditional second-order SDEs have the wrong sign for the curvature term, which is the root cause of their failure to predict instability under $(L_0,L_1)$-smoothness. By changing the sign to +, the resulting first-order SDE correctly recovers stability thresholds across quadratic, quartic, high-dimensional noisy, and adaptive settings.

## Method

### Overall Architecture

The paper does not propose new optimizers but provides "stability-faithful continuous-time surrogates" for two distributed optimizers:

- **DCSGD** (Unbiased Compressed SGD): $x_{k+1}=x_k-\frac{\eta\eta_k}{N}\sum_{i=1}^N \mathcal{C}_{\xi_i}(\nabla f_{i,\gamma_i}(x_k))$, where the compressor $\mathcal{C}_\xi$ satisfies $\mathbb{E}[\mathcal{C}_\xi(x)]=x$ and $\mathbb{E}\|\mathcal{C}_\xi(x)-x\|_2^2\leq\omega\|x\|_2^2$.
- **DSignSGD**: $x_{k+1}=x_k-\frac{\eta\eta_k}{N}\sum_{i=1}^N \operatorname{sign}(\nabla f_{i,\gamma_i}(x_k))$, equivalent to 1-bit quantization per client.

The loss $f(x)=\frac{1}{N}\sum_j f_j(x)$ is assumed to be $(L_0,L_1)$-smooth. Client gradient noise $\nabla f_{i,\gamma_i}(x)=\nabla f(x)+Z_i(x)$ is assumed to have a coordinate-wise symmetric distribution. DCSGD requires $\|\Sigma_i(x)\|_\infty\leq\sigma_{0,i}^2+\sigma_{1,i}^2\|\nabla f(x)\|_2^2$ (affine variance), while DSignSGD permits Student-$t_\nu$ heavy tails (where even the mean may be undefined if $\nu=1$).

The weak approximation definition follows Milshtein (1986): $(X_t)$ is an $\alpha$-order weak approximation of $(x_k)$ if $\max_k|\mathbb{E}g(x_k)-\mathbb{E}g(X_{k\eta})|\leq C\eta^\alpha$ for all polynomials $g$.

### Key Designs

**1. Identifying the failure of classic SDEs via 1D examples**

The paper demonstrates that classic SDEs are qualitatively incorrect under $(L_0,L_1)$-smoothness. For $f(x)=\lambda x^2/2$, discrete GD requires $\eta < 2/\lambda$, but classic ODEs ignore this or predict the opposite. For $f(x)=x^4/4$, GD requires $\eta < 2/x_k^2$, meaning the step size must depend on the current iteration. Classic ODEs predict global convergence regardless of $\eta$, missing the fact that large gradients require smaller steps.

**2. Sign-corrected first-order SDE (Ansatz + Drift Matching)**

The discrete second-order Taylor expansion for a GD step is:
$$\frac{f(x_{k+1})-f(x_k)}{\eta}=-\|\nabla f(x_k)\|^2+\frac{\eta}{2}\nabla f(x_k)^\top\nabla^2 f(x_k)\nabla f(x_k)+O_{x_k}(\eta^2),$$
where the second-order term has a **+** sign. Classic second-order ODEs induce a loss drift where this term has a **−** sign, leading to predicted "extra damping" rather than divergence. The authors propose the corrected SDE:
$$dX_t=-\nabla f(X_t)\,dt+\frac{\eta}{2}\nabla^2 f(X_t)\nabla f(X_t)\,dt+\sqrt{\eta}\sqrt{\Sigma(X_t)}\,dW_t,$$
Theorem C.5 proves this is a first-order weak approximation for SGD. For the quartic case, this induces a drift $df(X_t)=(-X_t^6+\frac{3\eta}{2}X_t^8)dt$, which correctly becomes repulsive (loss increases) when $\eta X_t^2 \gtrsim 1$.

**3. Unified Convergence Theorems**

Thm 4.2 (DCSGD) shows that under $(L_0,L_1)$-smoothness and affine variance, convergence $\mathbb{E}\|\nabla f(X_{\hat t})\|_2^2\to0$ is guaranteed if $\eta\eta_t < \frac{2\epsilon}{G(1+\frac{\bar\omega+d(\overline{\sigma_1^2\omega}+\overline{\sigma_1^2})}{N})+\frac{L_1 d(\overline{\sigma_0^2}+\overline{\sigma_0^2\omega})}{N}}$ (where $G=L_0+L_1\mathbb{E}\|\nabla f(X_t)\|_2$). The critical **+1** term recovers the classic $\eta < 2/L_0$ limit, which was missing in previous SDE analyses. Thm 4.3 (DSignSGD) shows that even with Student-$t_\nu$ heavy tails, standard Robbins–Monro scheduling ($\eta_k=1/\sqrt{k+1}$) ensures convergence due to the inherent element-wise normalization of the sign operator.

## Key Experimental Results

### Main Results

| Setting | Configuration | Observation |
| :--- | :--- | :--- |
| DCSGD + Unbiased Sparsification | $N=8$, MLP, Affine Variance Noise | At constant $\eta$, loss diverges as compression $\omega$ increases. Convergence is restored using the adaptive normalization prescribed by Thm 4.2. |
| DCSGD vs Plain Normalized SGD | Same as above | Plain Normalized SGD is less stable than the theory-prescribed adaptive normalization, showing normalization strength must scale with compression. |
| DSignSGD + Heavy Tail (Student-$t$) | $\nu=1$ (undefined mean), scale $\sigma$ | Stable but non-convergent at constant $\eta$. Converges for all $\sigma$ using $\eta_k=1/\sqrt{k+1}$ as per Thm 4.3. |
| ResNet-18 / ViT on CIFAR-10 | $N=8$, Distributed | Appendix E.4 reports consistent stability/instability patterns matching those found in MLPs. |

### Ablation Study

Table 2 compares learning rate constraints from classic SDEs vs. the proposed corrected SDE:

| Setting | Classic SDE Constraint | Proposed SDE Constraint | Key Difference |
| :--- | :--- | :--- | :--- |
| DCSGD ($(L_0,L_1)$ + Affine + Comp) | $\frac{2\epsilon}{G\frac{\bar\omega+d(\overline{\sigma_1^2\omega}+\overline{\sigma_1^2})}{N}+\dots}$ | $\frac{2\epsilon}{G(\bm{1}+\frac{\bar\omega+d(\dots)}{N})+\dots}$ | The **+1** restores the $\eta < 2/L_0$ limit. |
| DSignSGD (Heavy Tail) | $\ell_\nu/K, K=\frac{L_1 d\sigma_{\mathcal{H},1}}{2N}$ | $\ell_\nu/K, K=\frac{L_1 d\sigma_{\mathcal{H},1}}{2N}+\sqrt{d}(L_0+L_1)M_\nu$ | The extra term ensures step size constraints even in noise-free limits. |
| Quartic Case $f(x)=x^4/4$ | Global convergence for all $\eta$ | Correctly predicts repulsive drift for large $|X_t|$. | Matches discrete behavior. |

### Key Findings
- **DCSGD normalization intensity** is determined by $(\omega, \sigma_1, L_1, N, d)$. More aggressive compression, higher affine variance, and higher dimensions require stronger normalization.
- **DSignSGD is inherently robust to heavy tails** because the sign operator truncates the influence of coordinate-wise outliers, allowing standard scheduling to work even when the noise mean is undefined.
- **Scaling and scheduling laws** derived from classic SDEs are unreliable under $(L_0,L_1)$-smoothness as they may push optimizers into unstable regions.

## Highlights & Insights
- **Sign Correction**: Flipping the sign of the curvature term is not just a mathematical tweak; it is the only way to align continuous loss drift with discrete Taylor expansions, explaining why previous SDE-based scaling laws often mismatched experiments.
- **Matched-Property Methodology**: the authors prioritize a model's faithfulness to a specific property (loss dynamics) over simple high-order expansion, providing a replicable paradigm for constructing continuous-time proxies.
- **Unified Framework**: This is the first work to simultaneously handle the interaction of $(L_0,L_1)$-smoothness, compression, and complex noise (affine/heavy-tailed).

## Limitations & Future Work
- The analysis assumes IID client data and does not cover heterogeneous federated learning, error feedback, or biased compression.
- The theoretical bridge between SDE convergence and discrete convergence (Definition 3.4) still has formal gaps, although empirically supported.
- Constants like $L_1$ and $\sigma_1$ are hard to measure in practice, meaning the results serve better as qualitative guidelines than as closed-form hyperparameters.

## Related Work & Insights
- **vs Li et al. (2017)**: Extends the weak approximation framework by proving that the choice of SDE must remain faithful to the stability properties of interest.
- **vs Zhang et al. (2020b) / Chen et al. (2023)**: These works analyze $(L_0,L_1)$ in discrete time; this paper provides a unified SDE framework including the compression dimension.
- **vs Compagnoni et al. (2025a)**: This work advances their previous research on $L$-smoothness + compression into the more realistic $(L_0,L_1)$ regime.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The sign correction and ansatz matching are non-trivial conceptual shifts for SDE-based optimization analysis.
- Experimental Thoroughness: ⭐⭐⭐ Focuses on mechanism validation rather than large-scale benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ The clarity of the quadratic/quartic sanity checks makes the motivation compelling.
- Value: ⭐⭐⭐⭐⭐ Vital for future work using SDEs to derive scaling or scheduling laws.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)

</div>

<!-- RELATED:END -->

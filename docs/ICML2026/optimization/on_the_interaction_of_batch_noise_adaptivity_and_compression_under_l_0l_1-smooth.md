---
title: >-
  [Paper Note] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach
description: >-
  [ICML 2026][Optimization][$(L_0] This paper demonstrates that standard first- and second-order SDEs in literature fail to capture learning rate stability constraints under $(L_0…
tags:
  - "ICML 2026"
  - "Optimization"
  - "$(L_0"
  - "L_1)$-smoothness"
  - "Distributed Compressed SGD"
  - "SignSGD"
  - "Heavy-tailed Noise"
  - "Stability-corrected SDE"
date: 2026-05-08
content_hash: 4367cb75a1c28848
---

# On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach

**Conference**: ICML 2026  
**arXiv**: [2506.00181](https://arxiv.org/abs/2506.00181)  
**Code**: To be confirmed  
**Area**: Optimization / Distributed SGD / SDE Continuous-time Analysis  
**Keywords**: $(L_0,L_1)$-smoothness, Distributed Compressed SGD, SignSGD, Heavy-tailed Noise, Stability-corrected SDE

## TL;DR
This paper demonstrates that standard first- and second-order SDEs in literature fail to capture learning rate stability constraints under $(L_0,L_1)$-smoothness (even predicting convergence where the discrete optimizer diverges). By flipping the sign of the curvature term in the drift, the authors construct a family of "stability-faithful" first-order weak approximation SDEs. This enables a unified analysis of DCSGD and DSignSGD under compression, affine variance, and heavy-tailed noise, providing specific prescriptions for selecting normalization strength.

## Background & Motivation

**Background**: Distributed stochastic gradient methods are the backbone of modern large model training. Their behavior is governed by three factors: batch noise (including heavy-tailed noise as observed by Simsekli et al. 2019), communication compression (quantization, sparsification, sign), and adaptive normalization (Adam, AdaGrad, SignSGD, etc.). SDEs have been widely used as continuous-time proxies for discrete optimizers (starting from Li et al. 2017) to study learning rate scheduling, batch size scheduling, scaling laws, and implicit regularization.

**Limitations of Prior Work**: (1) Almost all SDE analyses rely on global $L$-smoothness, whereas real deep learning (DL) losses are closer to $(L_0,L_1)$-smoothness ($\|\nabla^2 f(x)\|_2 \leq L_0 + L_1\|\nabla f(x)\|_2$, Zhang et al. 2020b), which does not guarantee the existence of a constant step size stable for all initializations. (2) Noise is often assumed to be Gaussian or have bounded variance, while affine variance and heavy tails are common in modern DL. (3) There is a lack of unified characterization for DCSGD (distributed SGD with unbiased compression) and DSignSGD when $(L_0,L_1)$-smoothness, compression, and general noise coexist.

**Key Challenge**: The paper's most critical discovery is that **classical first-order and second-order SDEs fundamentally fail to predict discrete step-size stability under $(L_0,L_1)$-smoothness**. First-order SDEs impose no constraints on the learning rate, and second-order SDEs are worse—they predict "accelerated convergence" in large step-size regions where GD actually diverges. For a 1D parabola $f(x) = \lambda x^2/2$, GD requires $\eta < 2/\lambda$ for stability, but the classic first-order ODE solution $f(X_t) = f(X_0)e^{-2\lambda t}$ is independent of $\eta$, and the classic second-order ODE $f(X_t) = f(X_0)e^{-2\lambda(1+\lambda\eta/2)t}$ suggests faster convergence for larger $\eta$. For quartic functions $f(x) = x^4/4$ (typical non-$L$-smooth but $(L_0,L_1)$-smooth), a uniform constant step size does not even exist.

**Goal**: (1) Formally identify where classic SDEs fail; (2) derive a **sign-corrected modified first-order SDE** as a weak approximation for SGD under $(L_0,L_1)$; (3) unify the analysis of DCSGD (affine variance + compression) and DSignSGD (heavy-tailed student-$t$ noise); (4) provide practical guidelines for normalization strength.

**Key Insight**: The authors adopt an "ansatz" approach from physics—specifying a family of drift-corrected ODEs $dX_t = -\nabla f(X_t)dt + \alpha\eta\nabla^2 f(X_t)\nabla f(X_t)dt$ and then matching the "induced loss drift $df(X_t)/dt$" with the "discrete Taylor expansion $(f(x_{k+1}) - f(x_k))/\eta$" up to order $O(\eta)$. This **uniquely** determines $\alpha = 1/2$, and the sign is positive (**+**) rather than the negative (**−**) found in classic literature.

**Core Idea**: Traditional second-order SDEs use the wrong sign for the curvature term, which is the root cause of the "false prediction of accelerated convergence" under $(L_0,L_1)$-smoothness. By correcting the sign to +, the resulting first-order SDE recovers the correct stability thresholds for parabolic, quartic, high-dimensional noisy, compressed, and sign-adaptive cases.

## Method

### Overall Architecture

The paper does not propose a new optimizer but provides a "stability-faithful continuous-time surrogate." It investigates two distributed optimizers:

- **DCSGD** (Unbiased Compressed SGD): $x_{k+1} = x_k - \frac{\eta\eta_k}{N}\sum_{i=1}^N \mathcal{C}_{\xi_i}(\nabla f_{i,\gamma_i}(x_k))$, where the compressor $\mathcal{C}_\xi$ satisfies $\mathbb{E}[\mathcal{C}_\xi(x)] = x$ and $\mathbb{E}\|\mathcal{C}_\xi(x) - x\|_2^2 \leq \omega\|x\|_2^2$.
- **DSignSGD**: $x_{k+1} = x_k - \frac{\eta\eta_k}{N}\sum_{i=1}^N \operatorname{sign}(\nabla f_{i,\gamma_i}(x_k))$, equivalent to 1-bit quantization per client.

The loss $f(x) = \frac{1}{N}\sum_j f_j(x)$ is assumed to be $(L_0,L_1)$-smooth. Client gradient noise $\nabla f_{i,\gamma_i}(x) = \nabla f(x) + Z_i(x)$ is assumed to have a coordinate-symmetric distribution. DCSGD requires $\|\Sigma_i(x)\|_\infty \leq \sigma_{0,i}^2 + \sigma_{1,i}^2\|\nabla f(x)\|_2^2$ (affine variance), while DSignSGD allows student-$t_\nu$ heavy tails (where even the mean may be undefined for $\nu=1$).

The weak approximation follows Milshtein (1986): $(X_t)$ is an $\alpha$-order weak approximation of $(x_k)$ if $\max_k|\mathbb{E}g(x_k) - \mathbb{E}g(X_{k\eta})| \leq C\eta^\alpha$ for all polynomials $g$ of polynomial growth.

### Key Designs

1.  **Diagnosing Classical SDE Failure: Parabolic and Quartic Sanity Checks**:
    - **Function**: Reveal the failure modes of classic 1st/2nd order SDEs under $(L_0,L_1)$-smoothness using simple 1D examples.
    - **Mechanism**: For $f(x) = \lambda x^2/2$, the classic first-order ODE $dX_t = -\lambda X_t dt$ yields $f(X_t) = f(X_0)e^{-2\lambda t}$, independent of $\eta$. The classic second-order ODE $dX_t = -\nabla f(X_t)dt - \frac{\eta}{2}\nabla^2 f(X_t)\nabla f(X_t)dt$ predicts faster convergence for larger $\eta$. For $f(x) = x^4/4$, GD is $x_{k+1} = x_k(1 - \eta x_k^2)$, requiring $\eta < 2/x_k^2$ (iterative dependency), so no uniform constant step size exists; however, classic SDEs still predict global convergence.
    - **Design Motivation**: Prior work (e.g., Li et al. 2017) treated higher-order SDEs as "finer" proxies. This work uses counterexamples to prove that "higher order" does not mean "more faithful," shifting focus to "stability matching."

2.  **Sign-Corrected Modified First-Order SDE (Ansatz + Drift Matching)**:
    - **Function**: Construct a continuous-time model that correctly predicts learning rate constraints under $(L_0,L_1)$-smoothness.
    - **Mechanism**: The discrete Taylor expansion for one GD step is $\frac{f(x_{k+1}) - f(x_k)}{\eta} = -\|\nabla f(x_k)\|^2 + \frac{\eta}{2}\nabla f(x_k)^\top\nabla^2 f(x_k)\nabla f(x_k) + O_{x_k}(\eta^2)$ (note the positive sign). The classic second-order ODE induces $df(X_t) = -\|\nabla f(X_t)\|^2 dt - \frac{\eta}{2}\nabla f(X_t)^\top\nabla^2 f(X_t)\nabla f(X_t)dt$. The signs are opposite. The authors propose $dX_t = -\nabla f(X_t)dt + \alpha\eta\nabla^2 f(X_t)\nabla f(X_t)dt$; matching the $O(\eta)$ drift yields $\alpha = 1/2$. The new SDE is $dX_t = -\nabla f(X_t)dt \bm{+} \frac{\eta}{2}\nabla^2 f(X_t)\nabla f(X_t)dt + \sqrt{\eta}\sqrt{\Sigma(X_t)}dW_t$. This is formally proven as a first-order weak approximation (Theorem C.5).
    - **Design Motivation**: Matching the loss dynamics of the SDE with those of GD up to $O(\eta)$ is more crucial for stability analysis than simply adding higher-order terms.

3.  **Unified Convergence Theorems: DCSGD and DSignSGD**:
    - **Function**: Apply the modified SDE to distribute optimizers to provide convergence guarantees under $(L_0,L_1)$, compression, and general noise.
    - **Mechanism**: **Thm 4.2 (DCSGD)**: Given affine variance and compression rate $\omega_i$, the learning rate must satisfy $\eta\eta_t < \frac{2\epsilon}{G(1 + \frac{\bar\omega + d(\overline{\sigma_1^2\omega} + \overline{\sigma_1^2})}{N}) + \frac{L_1 d(\overline{\sigma_0^2} + \overline{\sigma_0^2\omega})}{N}}$, where $G = L_0 + L_1\mathbb{E}\|\nabla f(X_t)\|_2$. The **+1** in the denominator is key—it recovers the classic $\eta\eta_t < 2/L_0$ in the noise-free limit. **Thm 4.3 (DSignSGD)**: Under heavy-tailed noise, it requires $\eta\eta_t < \ell_\nu/K$, where $K = \frac{L_1 d\sigma_{\mathcal{H},1}}{2N} + \sqrt{d}(L_0 + L_1)M_\nu$. The term $\sqrt{d}(L_0 + L_1)M_\nu$ correctly identifies step-size constraints even with zero noise.
    - **Design Motivation**: Consolidate disparate conclusions from prior work into a single SDE framework, making normalization requirements readable via constants like $(L_0, L_1, \bar\omega, \overline{\sigma_1^2}, N, d)$.

### Loss & Training
The paper does not propose new loss designs; theoretical results are established for the SDE, and the discrete optimizer is linked via weak approximation (Def 3.4).

## Key Experimental Results

### Main Results

| Setup | Configuration | Observation |
| :--- | :--- | :--- |
| DCSGD + Unbiased Sparsification | $N=8$ clients, MLP, affine variance noise $Z_t \sim \mathcal{N}(0, \sigma^2\|g_t\|_2^2 I)$ | At constant $\eta$, increasing $\omega$ leads to divergence; adaptive normalization (Eq. 15) ensures convergence. |
| DCSGD vs Plain Normalized SGD | Same conditions | Adaptive normalization from Thm 4.2 is more stable than vanilla Normalized SGD. |
| DSignSGD + Heavy-tailed student-$t$ | $\nu=1$ (undefined mean), varying scale $\sigma$ | Stable but **non-convergent** under constant $\eta$; converges under $\eta_k = 1/\sqrt{k+1}$ as per Thm 4.3. |
| ResNet-18 / ViT on CIFAR-10 | Distributed $N=8$ clients | Qualitative phenomena (stability/instability parity) hold on larger scale models (Appendix E.4). |

### Ablation Study

Table 2 compares learning rate constraints derived from classic vs. modified SDEs:

| Setup | Classic SDE Constraint | Modified SDE Constraint (Ours) | Key Difference |
| :--- | :--- | :--- | :--- |
| DCSGD ($(L_0,L_1)$ + Affine + Comp) | No limit for $L_1$ | Recovers $\eta < 2/L_0$ | Includes a **+1** logic term |
| DSignSGD (Heavy-tailed) | No step-size constraint if noise scale is low | $\ell_\nu/K$ with geometric term | Captures non-trivial constraints |
| Parabolic Sanity Check | Convergent for all $\eta$ | Only convergent for $\eta < 2/\lambda$ | Matches Discrete GD |
| Quartic Sanity Check | Global convergence | Repulsive drift for large $|X_t|$ | Matches local stability limits |

### Key Findings
- **DCSGD normalization depends on $(\omega, \sigma_1, L_1, N, d)$**: Aggressive compression, high affine variance, high dimensionality, or fewer clients require stronger normalization.
- **DSignSGD is naturally robust to heavy tails**: Elementwise normalization from the sign operator allows convergence under standard Robbins-Monro scheduling even with undefined noise means.
- **Client scaling and dimensionality**: Increasing clients ($N$) helps stability, while higher dimensionality ($d$) amplifies the impact of affine variance.
- **Scaling law prescriptions from classic SDEs are unreliable under $(L_0,L_1)$**: Traditional $\eta/B(t)$ scheduling may push optimizers out of the stable region if stability corrections are ignored.

## Highlights & Insights
- **Sign Correction Insight**: Flipping $-\eta/2$ to $+\eta/2$ aligns the SDE loss drift with the discrete Taylor expansion. This explains why scaling laws derived from classic SDEs sometimes fail in experiments.
- **Ansatz Methodology**: Using a physical "ansatz" rather than deriving from a specific expansion allows for constructing a proxy faithful to a specific property (loss drift).
- **Unified Distributed Analysis**: The first framework to simultaneously cover $(L_0,L_1)$, compression, affine variance, and heavy tails.

## Limitations & Future Work
- Assumes i.i.d. client data; does not cover heterogeneous federated learning, error feedback, or biased compression.
- Theoretical guarantees are on the SDE; the gap between "SDE convergence" and "discrete convergence" remains a theoretical challenge.
- The constraint equations involve non-observable constants ($L_0, L_1$, etc.), serving as qualitative guides rather than black-box tuning rules.

## Related Work & Insights
- **vs. Li et al. 2017**: Uses the same weak approximation definition but argues that fidelity to stability is more important than the order of the approximation.
- **vs. Zhang et al. 2020b / Crawshaw et al. 2022**: Moves these discrete-time analyses into the SDE framework with compression.
- **vs. Khirirat et al. 2024**: Extends compressed $(L_0,L_1)$ analysis by relaxing the bounded variance assumption to affine and heavy-tailed noise.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "sign flip" is a non-trivial conceptual correction for the SDE-for-optimization literature.
- **Experimental Thoroughness**: ⭐⭐⭐ Focused on mechanism verification rather than large-scale benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The sanity checks clearly articulate the motivation.
- **Value**: ⭐⭐⭐⭐⭐ Provides a foundational baseline for future $(L_0,L_1)$-SDE analyses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)

</div>

<!-- RELATED:END -->
</div>

## Related Papers

- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)

</div>

<!-- RELATED:END -->

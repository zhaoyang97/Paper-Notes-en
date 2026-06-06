---
title: >-
  [Paper Note] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad
description: >-
  [ICML 2026][Optimization][Adaptive Gradient Methods] This work provides the first proof that AdaGrad converges under heavy-tailed noise ($p \in (4/3…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Adaptive Gradient Methods"
  - "Heavy-Tailed Noise"
  - "AdaGrad"
  - "Non-convex Optimization"
  - "Convergence Rates"
date: 2026-05-08
content_hash: df57e5bce387492d
---

# Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad

**Conference**: ICML 2026  
**arXiv**: [2605.18694](https://arxiv.org/abs/2605.18694)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Adaptive Gradient Methods, Heavy-Tailed Noise, AdaGrad, Non-convex Optimization, Convergence Rates  

## TL;DR
This work provides the first proof that AdaGrad converges under heavy-tailed noise ($p \in (4/3, 2]$) without any algorithmic modifications. It establishes an algorithm-dependent lower bound showing that AdaGrad cannot achieve minimax optimal rates, and demonstrates that AdaGrad-Norm can achieve a faster $O(1/T^{(p-1)/(2p)})$ rate under the assumption of a bounded objective function.

## Background & Motivation

**Background**: In the optimization of modern machine learning tasks (especially training attention models like Transformers), gradient noise commonly exhibits heavy-tailed distributions—meaning the noise has only limited $p$-th moments ($p \in (1, 2]$) rather than the classical finite variance assumption ($p = 2$). To address this challenge, two types of methods have been proven to guarantee convergence: Clipped SGD based on gradient clipping and NSGD(M) based on gradient normalization. However, both require additional algorithmic modifications (clipping thresholds or normalization operations).

**Limitations of Prior Work**: In practice, adaptive gradient methods (AdaGrad, Adam, AdamW, etc.) perform well in heavy-tailed noise scenarios, but existing theories fail to explain this phenomenon. The only related theoretical work, Chezhegov et al. (2025), has three key flaws: (1) its primary results target "delayed" variants rather than standard algorithms; (2) it requires extra modifications like gradient clipping; (3) it necessitates stronger assumptions such as bounded objective functions and prior knowledge of the tail index $p$.

**Key Challenge**: Adaptive gradient methods implicitly handle gradient scale differences by dynamically adjusting stepsizes. This adaptive mechanism shows no theoretical advantage over SGD under the classic finite variance assumption (both are $\tilde{O}(1/T^{1/4})$). Does this adaptivity naturally handle heavy-tailed noise?

**Goal**: Using AdaGrad as a case study, this paper answers the core question: "Can adaptive gradient methods converge in heavy-tailed non-convex optimization without any modifications?"

**Key Insight**: The authors observe that AdaGrad's coordinate-wise stepsize $\gamma / (\lambda + \sqrt{v_t})$ naturally scales based on historically accumulated gradients—when noise is high in a specific coordinate, $v_t$ increases, and the stepsize automatically decreases. This implicit "noise adaptivity" might suffice to control heavy-tailed noise.

**Core Idea**: By generalizing the proxy stepsize technique (introducing a free parameter $\boldsymbol{c}$ to replace the fixed $\boldsymbol{\sigma}$), the authors prove for the first time that AdaGrad's adaptive stepsize mechanism itself is sufficient to guarantee convergence under heavy-tailed noise without clipping or normalization.

## Method

### Overall Architecture
This is a purely theoretical work that does not propose new algorithms but analyzes the convergence behavior of two existing algorithms, standard AdaGrad and AdaGrad-Norm, under heavy-tailed noise. The analysis framework consists of three parts: (1) upper bound analysis for AdaGrad (proving convergence); (2) algorithm-dependent lower bounds for AdaGrad (identifying fundamental limitations); and (3) improved upper bounds for AdaGrad-Norm under stronger assumptions.

The update rule for standard AdaGrad is: $v_t = v_{t-1} + g_t^2$, $x_{t+1} = x_t - \frac{\gamma}{\lambda + \sqrt{v_t}} g_t$, where all operations are coordinate-wise. AdaGrad-Norm uses a global scalar $v_t = v_{t-1} + \|g_t\|_2^2$ instead of coordinate-wise accumulation.

### Key Designs

1.  **Generalized Proxy Stepsize Technique (Core of Upper Bound Proof)**:
    *   **Function**: Resolves the statistical coupling between the AdaGrad stepsize $\gamma/(\lambda + \sqrt{v_t})$ and the stochastic gradient $g_t$, allowing for conditional expectation operations.
    *   **Mechanism**: Defines a predictable ($\mathcal{F}_{t-1}$-measurable) proxy stepsize $w_t = v_{t-1} + (\nabla f(x_t))^2 + c^2$, where $c$ is a free parameter. The key innovation is selecting $c_i = \sigma_i T^{1/2 - 1/\bar{p}} / D_{T,i}^{1/2 - 1/\bar{p}}$ instead of simply setting $c = \sigma$. When $p = 2$, $c$ naturally reduces to $\sigma$, recovering classical analysis.
    *   **Design Motivation**: If $c = \sigma$ is chosen, the optimal rate is only $\tilde{O}(1/T^{(2p-3)/(2p)})$; a carefully chosen $c$ improves the rate to $\tilde{O}(1/T^{(3p-4)/(4p)})$, a strict improvement.

2.  **Algorithm-Dependent Lower Bound Construction**:
    *   **Function**: Proves an insurmountable lower bound for AdaGrad's convergence rate that varies with the input learning rate $\gamma$.
    *   **Mechanism**: In the 1D case ($d=1$), for a given $\gamma$, a specific objective function $f$ and stochastic gradient oracle are constructed such that $f'(x_t) \geq \Omega(\epsilon)$ holds with constant probability for all $t \in [T]$ (when $T$ is smaller than a threshold related to $\gamma$). The lower bound is $\Omega\big(\frac{\Delta^2/\gamma^2 + \gamma^2 L^2 \ln^2(\gamma L/\epsilon)}{\epsilon^2} + \frac{(\cdots)\sigma^{p/(p-1)}}{\epsilon^{(3p-2)/(p-1)}}\big)$.
    *   **Design Motivation**: Existing AdaGrad lower bounds (Jiang et al., 2025) lack dependence on the learning rate $\gamma$, failing to reflect the impact of algorithm configuration on complexity. the new lower bound explicitly captures the role of $\gamma$, revealing that AdaGrad cannot achieve minimax optimality via hyperparameter tuning.

3.  **Accelerated Analysis for AdaGrad-Norm with Bounded Objectives**:
    *   **Function**: Proves that AdaGrad-Norm can achieve a non-trivial rate of $O(1/T^{(p-1)/(2p)})$ for all $p \in (1, 2]$ under the additional assumption $\sup f < +\infty$.
    *   **Mechanism**: Utilizes the scalar stepsize of AdaGrad-Norm to control the sum of $(f(x_t) - f_\star)/\gamma_t$ using the boundedness condition $\Delta_\star / \gamma_T$. It then decomposes $\mathbb{E}[\sqrt{v_T}]$ into a noise term $\|\sigma\|_p T^{1/p}$ and a gradient term $\mathbb{E}[\sqrt{u_T}]$ via Lemma 4.7, using the AM-GM inequality to absorb recursive terms.
    *   **Design Motivation**: The coordinate-wise stepsize of standard AdaGrad prevents this derivation, making this faster rate a structural advantage unique to AdaGrad-Norm.

## Key Experimental Results

### Main Results (Theoretical Comparison)

| Algorithm | Rate | Applicable $p$ Range | Requires Prior Info | Extra Assumptions |
| :--- | :--- | :--- | :--- | :--- |
| Clipped SGD | $O(1/T^{(p-1)/(3p-2)})$ (minimax) | $(1, 2]$ | Requires $p$ | None |
| NSGD(M) (w/ prior) | $O(1/T^{(p-1)/(3p-2)})$ | $(1, 2]$ | Requires $p$ | None |
| NSGD(M) (w/o prior) | $O(1/T^{(p-1)/(2p)})$ | $(1, 2]$ | No | None |
| **AdaGrad (Ours Thm 3.1)** | $\tilde{O}(1/T^{(3p-4)/(4p)})$ | $(4/3, 2]$ | No | None |
| **AdaGrad-Norm (Ours Thm 4.2)** | $O(1/T^{(p-1)/(2p)})$ | $(1, 2]$ | No | Bounded Objective |
| **AdaGrad-Norm (Ours Thm C.1)** | $\tilde{O}(1/T^{(3p-4)/(4p)})$ | $(4/3, 2]$ | No | None |

### Rate Comparison for Different Tail Indices $p$

| Tail Index $p$ | Minimax Optimal | AdaGrad (Ours) | AdaGrad-Norm (Bounded, Ours) | NSGD(M) (w/o prior) |
| :--- | :--- | :--- | :--- | :--- |
| $2.0$ | $O(1/T^{1/4})$ | $\tilde{O}(1/T^{1/4})$ | $O(1/T^{1/4})$ | $O(1/T^{1/4})$ |
| $1.5$ | $O(1/T^{1/5})$ | $\tilde{O}(1/T^{1/12})$ | $O(1/T^{1/6})$ | $O(1/T^{1/6})$ |
| $4/3$ | — | Trivial (Critical Point) | $O(1/T^{1/8})$ | $O(1/T^{1/8})$ |
| $1.2$ | $O(1/T^{1/8})$ | Trivial | $O(1/T^{1/12})$ | $O(1/T^{1/12})$ |

### Key Findings
*   AdaGrad recovers the classic $\tilde{O}(1/T^{1/4})$ rate at $p = 2$, consistent with existing results; however, the rate becomes trivial when $p \leq 4/3$, indicating limited implicit noise control of adaptive stepsizes.
*   AdaGrad-Norm matches the minimax-optimal "no-prior" rate $O(1/T^{(p-1)/(2p)})$ of NSGD(M) under the bounded objective assumption and remains non-trivial for all $p \in (1, 2]$.
*   Algorithm-dependent lower bounds show that even with knowledge of $\Delta$ and $L$, AdaGrad still requires an additional polylog factor, representing a strict improvement over Jiang et al. (2025).
*   This is the first work to theoretically demonstrate that adaptive gradient methods outperform SGD under heavy-tailed noise.

## Highlights & Insights
*   **Generalized Proxy Stepsize with Free Parameters**: Replacing fixed parameters in classic analysis with free quantities optimized at the end of the proof leads to significant rate improvements. This generalizable technique could potentially extend to Adam/AdamW analysis.
*   **Methodological Contribution to Lower Bounds**: Constructing hard instances parameterized by the learning rate $\gamma$ makes lower bounds explicitly dependent on algorithm configuration, providing finer-grained information than traditional minimax bounds.
*   **"Free Lunch" of Adaptivity**: AdaGrad automatically adapts to the maximum available $p$ value and noise level $\|\sigma\|_1$ without knowing the tail index $p$—achieving simultaneous dual adaptivity for the first time in theory.

## Limitations & Future Work
*   The upper bound degrades to triviality at $p \leq 4/3$; it remains unclear if this is an analytical bottleneck or an inherent limitation of AdaGrad.
*   The algorithm-dependent lower bound might not be tight regarding the exponent of $\epsilon$, leaving a gap between upper and lower bounds.
*   Analyses are limited to AdaGrad and AdaGrad-Norm, excluding Adam/AdamW which are more common in practice; extending the analysis to these methods is a vital next step.
*   The faster rate for AdaGrad-Norm requires a bounded objective function, the practical validity of which in deep learning warrants further discussion.
*   Generalized smoothness conditions are not considered, representing another gap between theory and practice.

## Related Work & Insights
*   Comparison with Clipped SGD and NSGD(M) reveals the unique theoretical status of adaptive methods under heavy noise: they require no explicit mechanisms like clipping but at the cost of not reaching minimax optimality.
*   The AdaGrad-Norm analysis framework by Ward et al. (2019) and heavy-tail analysis in online convex optimization by Liu (2026) provided key technical inspiration.
*   The "no-prior" rate $O(1/T^{(p-1)/(2p)})$ for NSGD(M) by Hübler et al. (2025) and Liu & Zhou (2025) serves as a natural benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[ICLR 2026\] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](../../ICLR2026/optimization/faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)

</div>

<!-- RELATED:END -->

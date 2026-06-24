---
title: >-
  [Paper Note] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad
description: >-
  [ICML 2026][Optimization][Adaptive gradient methods] This paper provides the first proof that AdaGrad converges under heavy-tailed noise ($p \in (4/3, 2]$) without any algorithmic modifications. It also establishes an algorithm-dependent lower bound showing that AdaGrad cannot achieve the minimax optimal rate, while proving that AdaGrad-Norm can achieve a faster rate of $O(1/T^{(p-1)/(2p)})$ under the assumption of a bounded objective function.
tags:
  - "ICML 2026"
  - "Optimization"
  - "Adaptive gradient methods"
  - "Heavy-tailed noise"
  - "AdaGrad"
  - "Non-convex optimization"
  - "Convergence rates"
date: 2026-05-08
content_hash: dfecdd02835f956d
---

# Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad

**Conference**: ICML 2026  
**arXiv**: [2605.18694](https://arxiv.org/abs/2605.18694)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Adaptive gradient methods, Heavy-tailed noise, AdaGrad, Non-convex optimization, Convergence rates  

## TL;DR
This paper provides the first proof that AdaGrad converges under heavy-tailed noise ($p \in (4/3, 2]$) without any algorithmic modifications. It also establishes an algorithm-dependent lower bound showing that AdaGrad cannot achieve the minimax optimal rate, while proving that AdaGrad-Norm can achieve a faster rate of $O(1/T^{(p-1)/(2p)})$ under the assumption of a bounded objective function.

## Background & Motivation

**Background**: In the optimization process of modern machine learning tasks (especially training attention models like Transformers), gradient noise commonly exhibits a heavy-tailed distribution—meaning the noise only has finite $p$-th moments ($p \in (1, 2]$) rather than the classical finite variance assumption ($p = 2$). To address this challenge, two types of methods have been proven to guarantee convergence: Clipped SGD (based on gradient clipping) and NSGD(M) (based on gradient normalization). However, both types require extra algorithmic modifications (clipping thresholds or normalization operations).

**Limitations of Prior Work**: In practice, adaptive gradient methods (AdaGrad, Adam, AdamW, etc.) perform well in heavy-tailed noise scenarios, but existing theories cannot explain this phenomenon. The only related theoretical work, Chezhegov et al. (2025), has three key flaws: (1) the main results focus on "delayed" variants rather than the standard algorithm; (2) it requires additional modifications like gradient clipping; (3) it necessitates the stronger assumption of a bounded objective function and prior knowledge of the tail index $p$.

**Key Challenge**: Adaptive gradient methods implicitly handle gradient scale differences by dynamically adjusting stepsizes. This adaptive mechanism has shown no theoretical advantage over SGD under the classical finite variance assumption (both being $\tilde{O}(1/T^{1/4})$). Thus, can this adaptivity naturally handle heavy-tailed noise?

**Goal**: Taking AdaGrad as a case study, this paper aims to answer the core question: "Can adaptive gradient methods converge in heavy-tailed non-convex optimization without any modifications?"

**Key Insight**: The authors observe that the coordinate-wise stepsize of AdaGrad, $\gamma / (\lambda + \sqrt{v_t})$, naturally scales based on the historical accumulation of gradients. When the noise in a certain coordinate is large, $v_t$ increases, and the stepsize automatically decreases. This implicit "noise adaptivity" might be sufficient to control heavy-tailed noise.

**Core Idea**: By generalizing the proxy stepsize technique (introducing a free parameter $\boldsymbol{c}$ to replace the fixed $\boldsymbol{\sigma}$), the authors prove for the first time that the adaptive stepsize mechanism of AdaGrad itself is sufficient to guarantee convergence under heavy-tailed noise without clipping or normalization.

## Method

### Overall Architecture
This is a purely theoretical work that does not propose new algorithms but analyzes the convergence behavior of two existing algorithms, standard AdaGrad and AdaGrad-Norm, under heavy-tailed noise. The analysis framework consists of three parts: (1) Upper bound analysis for AdaGrad (proving convergence); (2) Algorithm-dependent lower bound for AdaGrad (identifying fundamental limitations); (3) Improved upper bound for AdaGrad-Norm under stronger assumptions.

The update rule for standard AdaGrad is: $v_t = v_{t-1} + g_t^2$, $x_{t+1} = x_t - \frac{\gamma}{\lambda + \sqrt{v_t}} g_t$, where all operations are performed coordinate-wise. AdaGrad-Norm replaces the coordinate-wise accumulation with a global scalar $v_t = v_{t-1} + \|g_t\|_2^2$.

### Key Designs

**1. Generalized Proxy Stepsize Technique (Core of the Upper Bound Proof): Leveraging an optimizable free parameter for a better rate**

The stepsize of AdaGrad, $\gamma/(\lambda+\sqrt{v_t})$, is statistically coupled with the stochastic gradient $g_t$—since $v_t$ contains the current $g_t$, conditional expectation cannot be applied directly. The classical approach introduces a predictable ($\mathcal{F}_{t-1}$-measurable) proxy stepsize $w_t = v_{t-1}+(\nabla f(x_t))^2+c^2$, where $c$ is typically fixed as $\sigma$. The key innovation of this paper is treating $c$ as a free parameter to be optimized at the end of the proof, specifically choosing $c_i = \sigma_i T^{1/2-1/\bar{p}}/D_{T,i}^{1/2-1/\bar{p}}$ (which naturally reduces to $\sigma$ and recovers the classical analysis when $p=2$). This subtle generalization brings substantial improvement: if $c=\sigma$ were chosen, the optimal rate would only be $\tilde{O}(1/T^{(2p-3)/(2p)})$; with a carefully chosen $c$, the rate improves to $\tilde{O}(1/T^{(3p-4)/(4p)})$, a strict enhancement. It also demonstrates that the adaptive stepsize mechanism of AdaGrad alone is sufficient to manage heavy-tailed noise without clipping or normalization.

**2. Construction of Algorithm-Dependent Lower Bounds: Proving AdaGrad cannot reach minimax optimality even with tuning**

Proving convergence is not enough; the authors clarify the fundamental limitations of AdaGrad. Existing AdaGrad lower bounds (Jiang et al. 2025) lack dependence on the learning rate $\gamma$, failing to reflect how algorithm configurations affect complexity. This paper constructs a specific objective function $f$ and a stochastic gradient oracle in a 1D setting for a given $\gamma$. This ensures that when $T$ is smaller than a certain threshold related to $\gamma$, the condition $f'(x_t)\geq\Omega(\epsilon)$ holds with constant probability for all $t\in[T]$. The resulting lower bound:

$$\Omega\Big(\tfrac{\Delta^2/\gamma^2 + \gamma^2 L^2 \ln^2(\gamma L/\epsilon)}{\epsilon^2} + \tfrac{(\cdots)\sigma^{p/(p-1)}}{\epsilon^{(3p-2)/(p-1)}}\Big)$$

explicitly includes the learning rate $\gamma$. This construction of "hard instances parameterized by algorithm configurations" is more informative than traditional minimax lower bounds, revealing that even with knowledge of $\Delta$ and $L$, AdaGrad must pay at least an extra polylog factor and cannot reach minimax optimality through parameter tuning.

**3. Accelerated Analysis of AdaGrad-Norm under Bounded Objectives: Scalar stepsizes yield non-trivial rates for all $p$**

Since the coordinate-wise stepsize of standard AdaGrad leads to a trivial rate when $p\leq 4/3$, the paper examines AdaGrad-Norm (using global scalar $v_t=v_{t-1}+\|g_t\|_2^2$). Under the additional assumption $\sup f<+\infty$, the authors leverage the scalar stepsize properties to sum the single-step progress inequality $(f(x_t)-f_\star)/\gamma_t$ and bound it by $\Delta_\star/\gamma_T$. Using Lemma 4.7, $\mathbb{E}[\sqrt{v_T}]$ is decomposed into a noise term $\|\sigma\|_p T^{1/p}$ and a gradient term $\mathbb{E}[\sqrt{u_T}]$. Applying the AM-GM inequality to absorb recursive terms results in a non-trivial rate of $O(1/T^{(p-1)/(2p)})$ for all $p\in(1,2]$. This faster rate is a structural advantage unique to AdaGrad-Norm—the coordinate-wise stepsizes of standard AdaGrad prevent such a derivation.

## Key Experimental Results

### Comparison of Main Theoretical Results

| Algorithm | Rate | Range of applicable $p$ | Prior info required | Extra Assumptions |
|------|------|--------------|-------------|---------|
| Clipped SGD | $O(1/T^{(p-1)/(3p-2)})$ (Minimax optimal) | $(1, 2]$ | Requires $p$ | None |
| NSGD(M) (with prior) | $O(1/T^{(p-1)/(3p-2)})$ | $(1, 2]$ | Requires $p$ | None |
| NSGD(M) (no prior) | $O(1/T^{(p-1)/(2p)})$ | $(1, 2]$ | No | None |
| **AdaGrad (Thm 3.1, Ours)** | $\tilde{O}(1/T^{(3p-4)/(4p)})$ | $(4/3, 2]$ | No | None |
| **AdaGrad-Norm (Thm 4.2, Ours)** | $O(1/T^{(p-1)/(2p)})$ | $(1, 2]$ | No | Bounded Objective |
| **AdaGrad-Norm (Thm C.1, Ours)** | $\tilde{O}(1/T^{(3p-4)/(4p)})$ | $(4/3, 2]$ | No | None |

### Comparison of Convergence Rates under Different Tail Indices $p$

| Tail Index $p$ | Minimax Optimal | AdaGrad (Ours) | AdaGrad-Norm (Bounded, Ours) | NSGD(M) (No prior) |
|-----------|-------------|---------------|--------------------------|-----------------|
| $2.0$ | $O(1/T^{1/4})$ | $\tilde{O}(1/T^{1/4})$ | $O(1/T^{1/4})$ | $O(1/T^{1/4})$ |
| $1.5$ | $O(1/T^{1/5})$ | $\tilde{O}(1/T^{1/12})$ | $O(1/T^{1/6})$ | $O(1/T^{1/6})$ |
| $4/3$ | — | Trivial (Critical point) | $O(1/T^{1/8})$ | $O(1/T^{1/8})$ |
| $1.2$ | $O(1/T^{1/8})$ | Trivial | $O(1/T^{1/12})$ | $O(1/T^{1/12})$ |

### Key Findings
- AdaGrad recovers the classical $\tilde{O}(1/T^{1/4})$ rate when $p = 2$, consistent with existing results; however, the rate becomes trivial when $p \leq 4/3$, indicating that the implicit noise control capability of adaptive stepsizes is limited.
- AdaGrad-Norm can match the minimax-independent optimal rate $O(1/T^{(p-1)/(2p)})$ of NSGD(M) under the bounded objective assumption, and it is non-trivial for all $p \in (1, 2]$.
- Algorithm-dependent lower bounds show that even with knowledge of $\Delta$ and $L$, AdaGrad still requires at least an extra polylog factor, representing a strict improvement over the lower bounds in Jiang et al. (2025).
- This is the first work to theoretically demonstrate that adaptive gradient methods outperform SGD under heavy-tailed noise.

## Highlights & Insights
- **Generalizing the Proxy Stepsize Free Parameter Technique**: By replacing the parameter fixed at $\sigma$ in classical analysis with a free variable $c$ that can be optimized at the end of the proof, the authors achieve a substantive improvement in the rate. This technique is general and expected to extend to the analysis of more complex adaptive methods like Adam/AdamW.
- **Methodological Contribution of Algorithm-Dependent Lower Bounds**: Constructing hard instances parameterized by the learning rate $\gamma$ allows the lower bound to explicitly depend on the algorithm configuration, providing more refined information than traditional minimax lower bounds. This methodology can be directly applied to the theoretical analysis of other adaptive optimizers.
- **The "Free Lunch" of Adaptivity**: AdaGrad automatically adapts to the maximum available $p$ value and the noise level $\|\sigma\|_1$ without needing to know the tail index $p$—this is the first theoretical guarantee to simultaneously achieve this dual adaptivity.

## Limitations & Future Work
- The upper bound becomes trivial when $p \leq 4/3$. It remains unclear whether this is a bottleneck of the analysis or an inherent limitation of AdaGrad.
- The algorithm-dependent lower bound may not be tight on the exponent of $\epsilon$, leaving a gap between the upper and lower bounds.
- Only AdaGrad and AdaGrad-Norm were analyzed, excluding more commonly used methods like Adam/AdamW; extending the analysis to these methods is an important future direction.
- The faster rate of AdaGrad-Norm requires the extra assumption of a bounded objective function; whether this condition is reasonable in deep learning warrants further discussion.
- Reality-aligned generalized smoothness conditions were not considered, which represents another significant gap between theory and practice.

## Related Work & Insights
- Comparison with Clipped SGD and NSGD(M) reveals the unique theoretical status of adaptive methods under heavy-tailed noise: they do not require explicit mechanisms like clipping/normalization, though the cost is failing to reach minimax optimality.
- The AdaGrad-Norm analysis framework of Ward et al. (2019) and the heavy-tail analysis for online convex optimization by Liu (2026) provided key inspiration for the technical path of this paper.
- The "prior-independent" rate $O(1/T^{(p-1)/(2p)})$ for NSGD(M) from Hübler et al. (2025) and Liu & Zhou (2025) serves as a natural comparison benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis](../../ICLR2026/optimization/clipped_gradient_methods_for_nonsmooth_convex_optimization_under_heavy-tailed_no.md)
- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)
- [\[ICLR 2026\] Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence](../../ICLR2026/optimization/decentralized_nonconvex_optimization_under_heavy-tailed_noise_normalization_and_.md)
- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)

</div>

<!-- RELATED:END -->

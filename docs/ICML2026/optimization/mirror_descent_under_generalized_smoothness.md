---
title: >-
  [Paper Note] Mirror Descent Under Generalized Smoothness
description: >-
  [ICML 2026][Optimization][$\ell*$-smoothness] This paper proposes the concept of $\ell*$-generalized smoothness based on an arbitrary norm and its dual norm. By utilizing a "generalized self-bounding lemma," the gradient dual norm is controlled within the initial sub-optimality gap. This establishes, for the first time, convergence rates for Mirror Descent and its accelerated, optimistic, Mirror Prox, stochastic, and composite variants under non-Euclidean geometry that match…
tags:
  - "ICML 2026"
  - "Optimization"
  - "$\\ell*$-smoothness"
  - "Mirror Descent"
  - "Non-Euclidean geometry"
  - "Self-boundness"
  - "LLM training curvature"
date: 2026-05-08
content_hash: 556a8be8338f23b1
---

# Mirror Descent Under Generalized Smoothness

**Conference**: ICML 2026  
**arXiv**: [2502.00753](https://arxiv.org/abs/2502.00753)  
**Code**: None  
**Area**: Optimization Theory / Mirror Descent / Generalized Smoothness  
**Keywords**: $\ell*$-smoothness, Mirror Descent, Non-Euclidean geometry, Self-boundness, LLM training curvature  

## TL;DR
This paper proposes the concept of $\ell*$-generalized smoothness based on an arbitrary norm and its dual norm. By utilizing a "generalized self-bounding lemma," the gradient dual norm is controlled within the initial sub-optimality gap. This establishes, for the first time, convergence rates for Mirror Descent and its accelerated, optimistic, Mirror Prox, stochastic, and composite variants under non-Euclidean geometry that match those under classic $L$-smoothness.

## Background & Motivation

**Background**: Objective functions in modern machine learning generally do not satisfy the classic $L$-smoothness assumption—even for the simplest $\ell_2$ regression, the global smoothness constant can be unbounded. Zhang et al. (2020) observed in LSTM/ResNet training that the Hessian norm grows approximately linearly along the gradient, proposing $(L_0,L_1)$-smoothness: $\|\nabla^2 f(\mathbf{x})\|_2\le L_0+L_1\|\nabla f(\mathbf{x})\|_2$. Li et al. (2023) further generalized this to $\ell$-smoothness, replacing the affine function with an arbitrary non-decreasing sub-quadratic function $\ell(\cdot)$.

**Limitations of Prior Work**: All existing research on generalized smoothness is restricted to the $\ell_2$ norm, exclusively serving the gradient descent family of algorithms in Euclidean space. However, Mirror Descent (MD)—a core player in non-Euclidean optimization for reinforcement learning, network quantization, diffusion model watermarking, and LLM pre-training/post-training (e.g., Muon, Scion)—lacks any matching generalized smoothness theory.

**Key Challenge**: In non-Euclidean geometry, the norm $\|\cdot\|$ and dual norm $\|\cdot\|_*$ are no longer equivalent. The approach by Li et al. (2023), which measures both the Hessian and gradient using $\|\cdot\|_2$ in non-Euclidean scenarios, violates the fundamental fact that "the Hessian mapping $\nabla^2 f(\mathbf{x})\mathbf{h}$ and the gradient $\nabla f(\mathbf{x})$ both belong to the dual space." Forced application of this approach introduces redundant constants proportional to the dimension $n$ at a scale of $\sqrt{n}$.

**Goal**: (i) Provide a generalized smoothness definition that natively supports non-Euclidean geometry; (ii) Integrate it into all mainstream Mirror Descent variants (standard MD, accelerated MD, optimistic MD, Mirror Prox, stochastic MD, and composite MD) to recover classic convergence rates; (iii) Demonstrate that this definition aligns with practice using real training trajectories of LLMs and CNNs.

**Key Insight**: The authors note that $\nabla^2 f(\mathbf{x})\mathbf{h}$ is an element in $\mathcal{E}^*$; therefore, its magnitude should be measured using the dual norm, while the denominator $\mathbf{h}$ uses the primal norm. By rewriting the $\ell$-smoothness inequality as $\sup_{\mathbf{h}\ne\mathbf{0}}\{\|\nabla^2 f(\mathbf{x})\mathbf{h}\|_*/\|\mathbf{h}\|\}\le\ell(\|\nabla f(\mathbf{x})\|_*)$, the entire theory becomes "geometrically correct."

**Core Idea**: Use a "generalized self-bounding lemma $\|\nabla f(\mathbf{x})\|_*^2\le 2\ell(2\|\nabla f(\mathbf{x})\|_*)(f(\mathbf{x})-f^*)$" to backward-control the hard-to-track gradient dual norm through the sub-optimality gap. As long as the algorithm ensures the sub-optimality gap is monotonic or controlled, the gradient is automatically bounded, allowing local reduction to classic $L$-smooth analysis.

## Method

### Overall Architecture

The backbone of the theory is a three-stage reduction: (1) Define $\ell*$-smoothness, measuring all Hessians and gradients with the appropriate (dual) norms; (2) Establish the equivalence "global $\ell*$-smooth ⇔ local $(\ell,r)*$-smooth," such that "given a gradient upper bound $G$, the function behaves like classic $L$-smooth within a ball of radius $G/L$ centered at the current point"; (3) Use the generalized self-bounding lemma to control the gradient dual norm $\|\nabla f(\mathbf{x}_t)\|_*$ of each MD variant via the sub-optimality gap $f(\mathbf{x}_t)-f^*$. Then, prove by induction that the gradient does not explode, constructing an absolute constant $L:=\ell(2G)$ as the effective smoothness parameter to recover classic rates like $O(1/T)$, $O(1/T^2)$, and $O(\log T/\sqrt{T})$.

### Key Designs

**1. $\ell*$-smoothness Definition + Local Equivalence: Returning the Hessian to its rightful Dual Space**

In non-Euclidean geometry, the norm $\|\cdot\|$ and dual norm $\|\cdot\|_*$ are distinct. Since $\nabla^2 f(\mathbf{x})\mathbf{h}$ and $\nabla f(\mathbf{x})$ both reside in the dual space, the measurement using $\|\cdot\|_2$ by Li et al. is geometrically incorrect and introduces $O(\sqrt{n})$ dimension factors. The fix is defining $f\in\mathcal{F}_\ell(\|\cdot\|)$ if and only if $\|\nabla^2 f(\mathbf{x})\mathbf{h}\|_*\le\ell(\|\nabla f(\mathbf{x})\|_*)\|\mathbf{h}\|$ holds almost everywhere: dual norm for the numerator, primal norm for the denominator. This is paired with a "local" version $(\ell,r)*$-smoothness—gradient Lipschitz within a ball of radius $r(\|\nabla f(\mathbf{x})\|_*)$ centered at $\mathbf{x}$ with constant $\ell(\|\nabla f(\mathbf{x})\|_*)$. Proposition 2.6 proves their near-equivalence under Assumption 2.5, allowing analysis to switch between them: the global definition derives sub-optimality gap bounds, while the local definition treats smoothness as constant along a trajectory. When $\|\cdot\|=\|\cdot\|_2$, it reduces to $\ell$-smoothness. The benefit of correct geometry is quantitative: for $f(\mathbf{x})=(\mathbf{1}_n^\top\mathbf{x})^4/4$, the $\ell_1$ version requires $\widetilde\ell(\alpha)=1+2\alpha$, whereas the $\ell_2$ version requires $\widehat\ell(\alpha)=n+2\sqrt{n}\alpha$, a difference of $\sqrt{n}$.

**2. Generalized Self-Bounding Lemma (Lemma 3.4) + Effective Constant $G,L$ Construction: Controlling Gradient via Sub-optimality Gap**

Mirror Descent performs descent in the dual space, causing the self-correlation $\langle\mathbf{x},\mathbf{x}\rangle=\|\mathbf{x}\|_2^2$ of the primal norm to disappear. Thus, one cannot directly prove monotonic gradient norm decrease as in Euclidean analysis. This paper takes a different path: proving a generalized self-bounding lemma for any $\mathbf{x}$

$$\|\nabla f(\mathbf{x})\|_*^2\le 2\ell(2\|\nabla f(\mathbf{x})\|_*)\,(f(\mathbf{x})-f^*)$$

This translates the difficult-to-track gradient dual norm into a relatively manageable sub-optimality gap. Under the sub-quadratic premise of Assumption 3.3 ($\lim_{\alpha\to\infty}\alpha^2/\ell(\alpha)=\infty$), the equation $\alpha^2=2\ell(2\alpha)(f(\mathbf{x}_0)-f^*)$ has a maximum finite solution $G$, serving as a uniform upper bound for the gradient dual norm along the trajectory; $L:=\ell(2G)$ then acts as the "effective classic smoothness constant." For each MD variant, induction proves: if the learning rate is appropriate ($\eta\le 1/L$, $1/(2L)$, $1/(3L)$, etc.), the sub-optimality gap $f(\mathbf{x}_t)-f^*\le f(\mathbf{x}_0)-f^*$ always holds, so $\|\nabla f(\mathbf{x}_t)\|_*\le G$ automatically follows. This circular induction—"assume bounded gradient to prove non-increasing gap, then use non-increasing gap to prove bounded gradient"—is resolved by Lemmas D.3/E.2/3.5 and serves as the paper's engine.

**3. "Time Partitioning" Analysis for Accelerated MD and Unified Framework: Covering Multi-sequence Algorithms**

Algorithms like Accelerated MD, Optimistic MD, and Mirror Prox maintain multiple sequences. Controlling $f(\mathbf{x}_t)-f^*$ only constrains $\nabla f(\mathbf{x}_t)$, not $\nabla f(\mathbf{y}_t)$. This paper introduces an auxiliary term $e_t:=\|\mathbf{y}_t-\mathbf{x}_{t-1}\|$ to measure the difference between sequences, proving that if $e_t\lesssim G/L$, local smoothness implies $\|\nabla f(\mathbf{y}_t)\|_*\lesssim G$. A "time partitioning" technique (Lemma F.3) splits the trajectory at threshold $\tau$: for $t\le\tau$, contraction mapping limits the growth of $e_t$; for $t>\tau$, $e_t$ decays hyperbolically. Unlike Li et al., who required $\eta\simeq 1/L^2$ and extra stabilization sequences, this work achieves $O(1/T^2)$ using $\eta_t=t\eta/(2L)$ without new components. The "controlled distance → local $L$-smooth → standard estimation" pattern extends to Stochastic MD (using "chain of events" for high probability $O(\sqrt{\log T}/\sqrt{T})$) and composite non-convex optimization (gradient mapping $O(1/T)$).

### Loss & Training

The theoretical work does not involve training losses. Key learning rate configurations: standard MD $\eta\le 1/L$, Mirror Prox $\eta\le 1/(2L)$, Optimistic MD $\eta\le 1/(3L)$, and Accelerated MD $L:=\ell(4G)$ (slightly larger). Assumption 3.3 requires $\ell$ to be sub-quadratic ($\lim_{\alpha\to\infty}\alpha^2/\ell(\alpha)=\infty$) to ensure finite solutions for $G$.

## Key Experimental Results

### Main Results

Experimental validation empirically confirms that $\ell*$-smoothness holds for real models. The authors estimate effective smoothness constants across multiple networks using the hierarchical approximation formula from Riabinin et al. (2025): $\|\nabla_i f(X)-\nabla_i f(Y)\|_{(i)*}/\|X_i-Y_i\|_{(i)}\le L_i^0+L_i^1\|\nabla_i f(X)\|_{(i)*}$.

| Setup | Model / Dataset | Measurement | Phenomenon |
|------|--------------|---------|------|
| LLM Pre-training | GPT-2 small/medium/large + FineWeb | Actual layer-wise $L_i$ vs $L_i^0+L_i^1\|\nabla\|$ approximation | High alignment, confirming $\ell*$-smoothness |
| Translation | 6-layer Transformer + WMT'16 | $\ell_1$ local curvature vs $\ell_\infty$ gradient | Curvature grows with gradient, fitting generalized smoothness |
| CV | CNN + CIFAR-10 | Full-batch and mini-batch estimates | Both estimates consistent, CNNs also satisfy it |

### Comparison of Convergence Rates

| Algorithm | Convexity | Classic $L$-smooth Rate | Ours ( $\ell*$-smooth Rate) | Type |
|------|------|-------------------|----------------------|------|
| Mirror Descent | Convex | $O(1/T)$ | $O(1/T)$ (Thm 3.5) | Avg/Last |
| Accelerated MD | Convex | $O(1/T^2)$ | $O(1/T^2)$ (Thm 3.7) | Last |
| Optimistic MD | Convex | $O(1/T)$ | $O(1/T)$ (Eq 13) | Avg |
| Mirror Prox | Convex | $O(1/T)$ | $O(1/T)$ (Eq 15) | Avg |
| Stochastic MD | Convex | $\widetilde O(1/\sqrt T)$ | $\widetilde O(1/\sqrt T)$ (Thm 4.2) | Last (High Prob) |
| Composite MD | Non-convex | $O(1/T)$ | $O(1/T)$ (Thm 5.1) | Gradient Mapping |

### Key Findings
- **Existence of Dimension Gains**: On $f(\mathbf{x})=(\mathbf{1}_n^\top\mathbf{x})^4/4$, the $\ell*$-smoothness ( $\ell_1$ version) is smaller than $\ell$-smoothness ( $\ell_2$ version) by a factor of $\sqrt{n}$. Appendix C provides aggressive examples like $O(1/n)$ and $O(\sqrt{\log n/n})$, showing that correct geometry saves on dimensionality costs.
- **Self-Bounding Lemma as the Engine**: It translates the "controlled gradient" problem into a "controlled gap" problem, avoiding direct tracking of dual norm sequences.
- **Simplified Accelerated MD**: No longer requires the reduced $\eta\simeq 1/L^2$ or extra stabilization sequences from Li et al. (2023). The analysis is cleaner, proving that non-Euclidean perspectives are not only tighter but also more intuitive.

## Highlights & Insights
- **Compactness through Geometric Awareness**: Measuring the Hessian mapping in its proper dual space is a "subtle yet critical" correction—the source of all dimension-independent bounds.
- **Reusable "Circular Induction" Technique**: Assuming bounded gradients to prove non-increasing gaps, then using the gap to prove bounded gradients. This "assumption-induction-self-consistency" trick is transferable to any scenario where gradient bounds and objective descent are interdependent.
- **Time Partitioning**: Splitting the trajectory with threshold $\tau$ (using contraction for growth and hyperbolic decay for the rest) is much lighter than closed-form estimation and can be used for other accelerated methods with auxiliary sequences.

## Limitations & Future Work
- The assumption $\ell$ is sub-quadratic ($\lim\alpha^2/\ell(\alpha)=\infty$) excludes more explosive gradient growth; relaxing this might require finer control over prox steps.
- Experiments only use hierarchical approximations (5) to verify if $\ell*$-smoothness holds; they do not directly compare MD and GD in end-to-end LLM training.
- The paper focuses on convex / weakly convex settings; modern LLM training is non-convex and non-smooth. Bringing the self-bounding lemma to general non-convex scenarios remains open.
- The $L:=\ell(4G)$ for Accelerated MD is twice that of standard MD; there is room to tighten constant factors.

## Related Work & Insights
- **vs Zhang et al. (2020) $(L_0,L_1)$-smoothness**: They used affine functions for Hessian norms and fixed $\ell_2$ geometry, analyzing only gradient clipping; this work uses arbitrary sub-quadratic $\ell$ and arbitrary norms, extending "correct clipping" to "correct geometry."
- **vs Li et al. (2023) $\ell$-smoothness**: Their $\ell$-smoothness is a special case when $\|\cdot\|=\|\cdot\|_2$. Their accelerated analysis requires $\eta\simeq 1/L^2$ and auxiliary sequences; this work needs no such "patches" and has smaller constants.
- **vs Riabinin et al. (2025) Hierarchical Non-Euclidean Smoothness**: They proposed layer-wise approximations (5); this work proves $\ell*$-smoothness encompasses (5) and builds the convergence theory for the MD family.
- **Inspiration for Muon / Scion / Modern LLM Optimizers**: These methods implicitly select non-Euclidean geometries (e.g., singular value norms, column norms). This paper provides their first "foundational geometric theory package," offering a template for convergence analysis of future hierarchical mixed-norm optimizers.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to generalize smoothness to non-Euclidean geometry for the full MD suite.
- Experimental Thoroughness: ⭐⭐⭐ Theoretical paper; empirical verification of $\ell*$-smoothness, but lacks end-to-end MD vs GD contrast.
- Writing Quality: ⭐⭐⭐⭐ Progressive motivation; clear sequence of key lemmas and analysis.
- Value: ⭐⭐⭐⭐⭐ Provides a "theoretical motherboard" for non-Euclidean LLM optimizers like Muon and Scion; a foundational advancement in this research line.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)
- [\[ICML 2025\] Learning Mixtures of Experts with EM: A Mirror Descent Perspective](../../ICML2025/optimization/learning_mixtures_of_experts_with_em_a_mirror_descent_perspective.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)
- [\[ICLR 2026\] Never Saddle for Reparameterized Steepest Descent as Mirror Flow](../../ICLR2026/optimization/never_saddle_for_reparameterized_steepest_descent_as_mirror_flow.md)

</div>

<!-- RELATED:END -->

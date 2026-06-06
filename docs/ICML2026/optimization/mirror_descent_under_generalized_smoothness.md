---
title: >-
  [Paper Note] Mirror Descent Under Generalized Smoothness
description: >-
  [ICML 2026][Optimization][$\ell*$-smoothness] This paper proposes the concept of $\ell*$-generalized smoothness based on arbitrary norms and their dual norms. By utilizing a "generalized self-bounding lemma" to constrain…
tags:
  - "ICML 2026"
  - "Optimization"
  - "$\\ell*$-smoothness"
  - "Mirror Descent"
  - "Non-Euclidean Geometry"
  - "Self-boundedness"
  - "LLM Training Curvature"
date: 2026-05-08
content_hash: 78bcc9bb2c254c36
---

# Mirror Descent Under Generalized Smoothness

**Conference**: ICML 2026  
**arXiv**: [2502.00753](https://arxiv.org/abs/2502.00753)  
**Code**: None  
**Area**: Optimization Theory / Mirror Descent / Generalized Smoothness  
**Keywords**: $\ell*$-smoothness, Mirror Descent, Non-Euclidean Geometry, Self-boundedness, LLM Training Curvature  

## TL;DR
This paper proposes the concept of $\ell*$-generalized smoothness based on arbitrary norms and their dual norms. By utilizing a "generalized self-bounding lemma" to constrain the gradient dual norm within the initial suboptimality gap, it establishes, for the first time, convergence rates for Mirror Descent and its accelerated, optimistic, Mirror Prox, stochastic, and composite variants in non-Euclidean geometries that match those under classical $L$-smoothness.

## Background & Motivation

**Background**: Objective functions in modern machine learning generally do not satisfy the classical $L$-smoothness assumption—even for the simplest $\ell_2$ regression, the global smoothness constant can be unbounded. Zhang et al. (2020) observed that the Hessian norm grows approximately linearly with the gradient during LSTM/ResNet training and proposed $(L_0,L_1)$-smoothness: $\|\nabla^2 f(\mathbf{x})\|_2\le L_0+L_1\|\nabla f(\mathbf{x})\|_2$. Li et al. (2023) further generalized this to $\ell$-smoothness, replacing the affine function with an arbitrary non-decreasing sub-quadratic function $\ell(\cdot)$.

**Limitations of Prior Work**: All existing research on generalized smoothness is restricted to the $\ell_2$ norm, serving only the Gradient Descent family in Euclidean spaces. However, Mirror Descent (MD)—a cornerstone of non-Euclidean optimization playing a central role in reinforcement learning, network quantization, diffusion model watermarking, and LLM pre-training and post-training (e.g., Muon, Scion)—lacks any matching generalized smoothness theory.

**Key Challenge**: In non-Euclidean geometry, the norm $\|\cdot\|$ and the dual norm $\|\cdot\|_*$ are no longer equivalent. The approach by Li et al. (2023), which measures both the Hessian and the gradient using $\|\cdot\|_2$, violates the fundamental geometric fact that the Hessian mapping $\nabla^2 f(\mathbf{x})\mathbf{h}$ and the gradient $\nabla f(\mathbf{x})$ reside in the dual space. Forcefully applying such theory introduces redundant constants proportional to the dimension $n$ (on the order of $\sqrt{n}$).

**Goal**: (i) Provide a generalized smoothness definition that natively supports non-Euclidean geometry; (ii) Incorporate it into all major Mirror Descent variants (standard MD, accelerated MD, optimistic MD, Mirror Prox, stochastic MD, composite MD) and recover classical convergence rates; (iii) Demonstrate the practical relevance of this definition using real training trajectories from LLMs and CNNs.

**Key Insight**: The authors observe that since $\nabla^2 f(\mathbf{x})\mathbf{h}$ is an element in $\mathcal{E}^*$, its magnitude should be measured by the dual norm, while the denominator $\mathbf{h}$ uses the primal norm. By rewriting the $\ell$-smoothness inequality as $\sup_{\mathbf{h}\ne\mathbf{0}}\{\|\nabla^2 f(\mathbf{x})\mathbf{h}\|_*/\|\mathbf{h}\|\}\le\ell(\|\nabla f(\mathbf{x})\|_*)$, the theory becomes "geometrically correct."

**Core Idea**: A "generalized self-bounding lemma $\|\nabla f(\mathbf{x})\|_*^2\le 2\ell(2\|\nabla f(\mathbf{x})\|_*)(f(\mathbf{x})-f^*)$" is used to backwardly control the difficult-to-track gradient dual norm via the suboptimality gap. As long as the algorithm ensures the suboptimality gap is monotonic or controlled, the gradient remains automatically bounded, allowing the analysis to be locally reduced to classical $L$-smooth analysis.

## Method

### Overall Architecture

The framework consists of a three-step reduction: (1) Define $\ell*$-smoothness, measuring all Hessians and gradients with appropriate (dual) norms; (2) Establish the equivalence "Global $\ell*$-smooth ⇔ Local $(\ell,r)*$-smooth," indicating that if the gradient upper bound $G$ is known, the function behaves like a classical $L$-smooth function within a ball of radius $G/L$ centered at the current point; (3) Use the generalized self-bounding lemma to bound the gradient dual norm $\|\nabla f(\mathbf{x}_t)\|_*$ of each MD variant through the suboptimality gap $f(\mathbf{x}_t)-f^*$. Induction is then applied to prove that the gradient does not explode, constructing an absolute constant $L:=\ell(2G)$ as the effective smoothness parameter to recover classical rates such as $O(1/T)$, $O(1/T^2)$, and $O(\log T/\sqrt{T})$.

### Key Designs

1.  **$\ell*$-smoothness Definition + Local Equivalence**:
    - **Function**: Characterizes the Hessian-gradient coupling using a pair of (norm, dual norm), naturally compatible with any geometry ($\ell_1, \ell_\infty$, hierarchical mixed norms, etc.).
    - **Mechanism**: $f\in\mathcal{F}_\ell(\|\cdot\|)$ if and only if $\|\nabla^2 f(\mathbf{x})\mathbf{h}\|_*\le\ell(\|\nabla f(\mathbf{x})\|_*)\|\mathbf{h}\|$ holds almost everywhere. Simultaneously, a "local version" $(\ell,r)*$-smoothness is defined, where the gradient is Lipschitz with constant $\ell(\|\nabla f(\mathbf{x})\|_*)$ within a ball of radius $r(\|\nabla f(\mathbf{x})\|_*)$. Proposition 2.6 proves these are "nearly equivalent" under Assumption 2.5, allowing free switching during analysis. When $\|\cdot\|=\|\cdot\|_2$, it reduces to the $\ell$-smoothness of Li et al. (2023). For example, $f(\mathbf{x})=(\mathbf{1}_n^\top\mathbf{x})^4/4$ under the $\ell_1$ norm requires only $\widetilde\ell(\alpha)=1+2\alpha$, whereas the $\ell_2$ version requires $\widehat\ell(\alpha)=n+2\sqrt{n}\alpha$, a difference of $\sqrt{n}$.
    - **Design Motivation**: Treating $\nabla^2 f\cdot\mathbf{h}$ as an object in $\mathcal{E}^*$ measured by the dual norm avoids the dimensional factors forced by Euclidean perspectives, leading to the quantitative conclusion that "choosing the right geometry reduces the dimensional cost."

2.  **Generalized Self-bounding Lemma (Lemma 3.4) + Effective Constants $G, L$ Construction**:
    - **Function**: Back-constrains the "gradient dual norm sequence," which is difficult to track in MD, using the "suboptimality gap sequence," which is easier to control.
    - **Mechanism**: For any $\mathbf{x}\in\mathcal{X}$, it is shown that $\|\nabla f(\mathbf{x})\|_*^2\le 2\ell(2\|\nabla f(\mathbf{x})\|_*)(f(\mathbf{x})-f^*)$. Under the "sub-quadratic" premise of Assumption 3.3 ($\lim_{\alpha\to\infty}\alpha^2/\ell(\alpha)=\infty$), the equation $\alpha^2=2\ell(2\alpha)(f(\mathbf{x}_0)-f^*)$ has a maximum finite solution $G$, serving as a uniform upper bound for the gradient dual norm along the trajectory. Setting $L:=\ell(2G)$ allows it to act as the "effective classical smoothness constant." For each MD variant, induction is used: if the learning rate $\eta$ is appropriate (e.g., $\eta\le 1/L$, $1/(2L)$, or $1/(3L)$), the suboptimality gap $f(\mathbf{x}_t)-f^*\le f(\mathbf{x}_0)-f^*$ always holds, thus $\|\nabla f(\mathbf{x}_t)\|_*\le G$ is automatically satisfied.
    - **Design Motivation**: In Euclidean analysis, Li et al. directly prove the monotonicity of the gradient $\ell_2$ norm, which relies on $\langle\mathbf{x},\mathbf{x}\rangle=\|\mathbf{x}\|_2^2$. In MD, updates occur in the dual space, and primal norm autocorrelation vanishes. This barrier is bypassed via the link: "gap non-increasing → gradient controlled → local $L$-smooth → standard analysis."

3.  **"Time Partition" Analysis for Accelerated MD and Unified Framework**:
    - **Function**: Extends the two-step reduction to algorithms maintaining multiple sequences (Accelerated MD, Optimistic MD, Mirror Prox) and handles stochastic and non-convex composite cases, uniformly recovering $O(1/T^2)$, $O(\sqrt{\log T/T})$, and $O(1/T)$ rates.
    - **Mechanism**: Accelerated MD maintains three sequences $\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t$. Controlling $f(\mathbf{x}_t)-f^*$ only yields $\|\nabla f(\mathbf{x}_t)\|_*\lesssim G$, which does not directly bound $\nabla f(\mathbf{y}_t)$. The authors introduce an auxiliary term $e_t:=\|\mathbf{y}_t-\mathbf{x}_{t-1}\|$ to measure the distance between sequences and prove that if $e_t\lesssim G/L$, local smoothness implies $\|\nabla f(\mathbf{y}_t)\|_*\lesssim G$. Using the "time partition" technique (Lemma F.3), growth is restricted via contraction mapping for $t\le\tau$, while for $t>\tau$, $e_t$ is shown to decay hyperbolically. Unlike Li et al., who required $\eta\simeq 1/L^2$ and extra stabilization sequences, this work achieves $O(1/T^2)$ using $\eta_t=t\eta/(2L)$ without new components. This is extended to stochastic MD and composite non-convex optimization.
    - **Design Motivation**: A core difficulty in many MD variants is simultaneously bounding gradients of multiple trajectory sequences. The "distance controlled → local $L$-smooth → standard estimation" pattern provides a unified framework for typical variants.

### Loss & Training

This theoretical paper does not involve training losses. Key learning rate configurations: standard MD $\eta\le 1/L$, Mirror Prox $\eta\le 1/(2L)$, optimistic MD $\eta\le 1/(3L)$, and for accelerated MD $L:=\ell(4G)$. Assumption 3.3 requires $\ell$ to be sub-quadratic ($\lim_{\alpha\to\infty}\alpha^2/\ell(\alpha)=\infty$) to ensure finite solutions for $G$.

## Key Experimental Results

### Main Results

The "experiments" in this theoretical paper primarily verify that $\ell*$-smoothness holds on real models. The authors use the hierarchical approximation formula $\|\nabla_i f(X)-\nabla_i f(Y)\|_{(i)*}/\|X_i-Y_i\|_{(i)}\le L_i^0+L_i^1\|\nabla_i f(X)\|_{(i)*}$ from Riabinin et al. (2025) to estimate effective smoothness constants across various networks.

| Setup | Model / Dataset | Measurement | Observation |
|------|--------------|---------|------|
| LLM Pre-training | GPT-2 small/medium/large + FineWeb | Actual layer-wise $L_i$ vs $L_i^0+L_i^1\|\nabla\|$ | High correlation; validates $\ell*$-smoothness |
| Translation | 6-layer Transformer + WMT'16 | $\ell_1$ local curvature vs $\ell_\infty$ gradient | Curvature grows with gradient; fits generalized smoothness |
| CV | CNN + CIFAR-10 | Full-batch vs mini-batch estimation | Consistency between estimations; CNNs also satisfy this |

### Main Results

| Algorithm | Convexity | Classical $L$-smooth Rate | Ours $\ell*$-smooth Rate | Type |
|------|------|-------------------|----------------------|------|
| Mirror Descent | Convex | $O(1/T)$ | $O(1/T)$ (Thm 3.5) | Avg/Last |
| Accelerated MD | Convex | $O(1/T^2)$ | $O(1/T^2)$ (Thm 3.7) | Last |
| Optimistic MD | Convex | $O(1/T)$ | $O(1/T)$ (Eq 13) | Avg |
| Mirror Prox | Convex | $O(1/T)$ | $O(1/T)$ (Eq 15) | Avg |
| Stochastic MD | Convex | $\widetilde O(1/\sqrt T)$ | $\widetilde O(1/\sqrt T)$ (Thm 4.2) | Last (High Prob) |
| Composite MD | Non-convex | $O(1/T)$ | $O(1/T)$ (Thm 5.1) | Grad Mapping |

### Key Findings
- **Existence of Dimensional Gain**: On $f(\mathbf{x})=(\mathbf{1}_n^\top\mathbf{x})^4/4$, $\ell*$-smoothness ($\ell_1$ version) is smaller than $\ell$-smoothness ($\ell_2$ version) by a factor of $\sqrt{n}$. Appendix C provides more aggressive examples such as $O(1/n)$ and $O(\sqrt{\log n/n})$, showing that choosing the right geometry avoids dimensional penalties.
- **The Generalized Self-bounding Lemma is the Engine**: It translates the "gradient control" problem into a "gap control" problem, avoiding direct tracking of dual norm sequences.
- **Accelerated MD Simplification**: It no longer requires the reduced learning rate $\eta\simeq 1/L^2$ or the extra stabilization sequences of Li et al. (2023), confirming that a non-Euclidean perspective leads to tighter and more intuitive methods.

## Highlights & Insights
- **Tightness via Geometric Awareness**: Placing the Hessian mapping back into its true dual space is a subtle but critical correction—it is the source of all dimension-independent bounds.
- **Reusable "Circular Induction" Technique**: The paper first assumes bounded gradients to derive non-increasing suboptimality, then uses the latter to prove the former. This "assumption-induction-self-consistency" technique can be transferred to any scenario where gradient bounds and objective descent are interdependent.
- **Time Partition Technique**: Splitting the trajectory into two segments based on a threshold $\tau$—using contraction mapping to limit growth in the first half and proving hyperbolic decay in the second—is much lighter than direct closed-form estimation and can be adapted for other accelerated methods with auxiliary sequences.

## Limitations & Future Work
- The assumption of sub-quadratic $\ell$ ($\lim\alpha^2/\ell(\alpha)=\infty$) excludes cases with more severe gradient explosion; relaxing this might require finer prox-step control.
- Experiments only use hierarchical approximation (5) to verify the existence of $\ell*$-smoothness and do not directly compare MD vs. Gradient Descent in end-to-end LLM training; a gap remains between the theory and determining "when to choose MD."
- The work focuses on convex/weakly convex scenarios; modern LLM training is almost entirely non-convex and non-smooth. Fully extending the generalized self-bounding lemma to general non-convex settings remains an open problem.
- For Accelerated MD, $L:=\ell(4G)$ is twice as large as in standard MD, suggesting room for tightening constant factors.

## Related Work & Insights
- **vs Zhang et al. (2020) $(L_0,L_1)$-smoothness**: They used affine functions to characterize the Hessian norm with fixed $\ell_2$ geometry, analyzing only gradient clipping. This work allows arbitrary non-decreasing sub-quadratic $\ell$ and any norm, extending "gradient clipping is correct" to "correct geometry selection."
- **vs Li et al. (2023) $\ell$-smoothness**: Their $\ell$-smoothness is a special case of this work when $\|\cdot\|=\|\cdot\|_2$, and their acceleration analysis required $\eta\simeq 1/L^2$ and auxiliary sequences; this work removes these "patches" and offers smaller constants.
- **vs Riabinin et al. (2025) Hierarchical Non-Euclidean Smoothness**: They proposed the hierarchical approximation (5); this work proves that $\ell*$-smoothness includes (5) as a special case and builds the full convergence theory for the MD family.
- **Insights for Muon / Scion / Modern LLM Optimizers**: These methods implicitly select non-Euclidean geometries (e.g., singular value norms). This paper provides the first "foundational geometric theory kit" for them, serving as a template for analyzing future hierarchical mixed-norm optimizers.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to fully extend generalized smoothness to non-Euclidean geometry and unify the MD variant family.
- Experimental Thoroughness: ⭐⭐⭐ A theoretical paper; provides empirical validation of $\ell*$-smoothness but lacks end-to-end MD vs. GD comparisons.
- Writing Quality: ⭐⭐⭐⭐ Motivations are progressive; key lemmas and analysis order are clear.
- Value: ⭐⭐⭐⭐⭐ Provides the "theoretical motherboard" for non-Euclidean LLM optimizers like Muon and Scion; a foundational advancement for this research line.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration
description: >-
  [ICLR 2026][Optimization & Theory][AdaGrad] This paper establishes a **unified convergence analysis framework** for AdaGrad-type adaptive preconditioned SGD. By constraining the preconditioning operator within an operator subspace satisfying specific structural assumptions, the authors use a **single proof** to simultaneously recover SOTA convergence rates for A
tags:
  - ICLR 2026
  - Optimization & Theory
  - AdaGrad
date: 2026-05-08
content_hash: 71d84024e700493d
---
# SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=XhXMzPJJ7J](https://openreview.net/forum?id=XhXMzPJJ7J)  
**Area**: Optimization Theory / Adaptive Gradient Methods  
**Keywords**: Adaptive Preconditioning, AdaGrad, Nesterov Momentum, Matrix Smoothness, Unified Convergence Analysis

## TL;DR
This paper establishes a **unified convergence analysis framework** for AdaGrad-type adaptive preconditioned SGD. By constraining the preconditioning operator within an operator subspace satisfying specific structural assumptions, the authors use a **single proof** to simultaneously recover SOTA convergence rates for AdaGrad-Norm, AdaGrad, and ASGO/One-sided Shampoo, while providing the first convergence guarantee for DASGO. It further proves that diagonal preconditioning (AdaGrad, DASGO) can be combined with Nesterov momentum for **provable acceleration**, theoretically explaining for the first time why the "diagonal preconditioning + momentum" mechanism in Adam is so efficient.

## Background & Motivation

**Background**: Deep learning training is almost entirely dominated by Adam / AdamW, which originate from gradient descent with AdaGrad-Norm steps. A large class of subsequent algorithms (AdaGrad, RMSProp, Shampoo, One-sided Shampoo, ASGO, Muon, Scion, DASGO) are all "preconditioned gradient methods," sharing a unified update rule $x_{k+1}=\arg\min_{x}\langle g_k,x\rangle+\tfrac12\|x-x_k\|_{H_k^{-1}}^2$, differing only in the structure of the preconditioning operator $H_k$ (scalar / diagonal / matrix).

**Limitations of Prior Work**: Every time a new preconditioning method is proposed, a separate convergence proof must be written, despite highly similar update rules and proof structures. The unified framework by Gupta et al. (2017) partially addressed this (defining $H_k$ as an optimal solution over a subspace), but still required separate proofs for different algorithms, only covered non-smooth cases, and failed to explain the benefits of general preconditioning operators.

**Key Challenge**: The theory of adaptive optimization lags seriously behind practice. On one hand, there is a lack of a unified analysis covering scalar, diagonal, and matrix preconditioning. On the other hand, the two pillars of Adam—"diagonal preconditioning" and "momentum"—have never been proven theoretically to be **simultaneously** effective (existing AdaGrad acceleration results only apply to scalar step sizes).

**Goal**: (Q1) Can a unified convergence analysis be designed to cover most adaptive preconditioning methods like AdaGrad / Shampoo / ASGO? (Q2) Can a method be designed to **simultaneously** benefit from both diagonal AdaGrad preconditioning and momentum?

**Key Insight**: The authors follow Gupta et al. (2017) by restricting $H_k$ to an operator subspace $\mathcal{H}\subset\mathcal{S}$, but supplement this subspace with two structural assumptions (order-preserving projection + closure under operator functions). This allows for an **explicit solution** of the preconditioning operator and the use of stronger analytical tools. Smoothness and noise are measured using "weighted / matrix norms" to unify the scalar, diagonal, and matrix cases.

**Core Idea**: Using an abstraction of an "operator subspace $\mathcal{H}$ + potential function $\phi$," all AdaGrad-type preconditioners are brought under the same update and proof framework. Building on this, by introducing the assumption that "smoothness/noise operators commute with the preconditioner" (which holds automatically for diagonal cases), the authors achieve acceleration rates by adding Nesterov momentum.

## Method

### Overall Architecture
This paper does not propose a new optimizer but rather an **analysis framework**: all AdaGrad-type methods are unified into Algorithm 1 (Adaptive SGD with Preconditioning), where the only variable is the subspace $\mathcal{H}$ to which the preconditioning operator belongs. In each step, a stochastic gradient $g_k=\nabla f(x_k;\xi_k)$ is sampled, $H_k$ is solved via a unified formula, and a mirror-descent-style update is performed: $x_{k+1}=\arg\min_x\langle g_k,x\rangle+\tfrac12\|x-x_k\|_{H_k^{-1}}^2$, outputting the iterate average $\bar x_K=\tfrac{1}{K+1}\sum x_k$.

The essence of the framework is: **choosing different $\mathcal{H}$ yields different known algorithms** (see table below), while the convergence proof is written only once.

| Algorithm | Space $\mathcal{X}$ | Preconditioning Subspace $\mathcal{H}$ | Equivalent Norm $R(\cdot)$ |
|-----------|---------------------|---------------------------------------|---------------------------|
| AdaGrad-Norm | $\mathbb{R}^d$ | $\{g\mapsto\beta g:\beta\in\mathbb{R}\}$ (Scalar) | $\tfrac{1}{\sqrt d}\|\cdot\|$ |
| AdaGrad | $\mathbb{R}^d$ | $\{g\mapsto b\odot g:b\in\mathbb{R}^d\}$ (Diagonal) | $\|\cdot\|_\infty$ |
| ASGO/One-sided Shampoo | $\mathbb{R}^{m\times n}$ | $\{G\mapsto BG:B\in\mathcal{S}^m\}$ (Matrix) | $\tfrac{1}{\sqrt n}\sigma_{\max}(\cdot)$ |
| DASGO | $\mathbb{R}^{m\times n}$ | $\{G\mapsto\mathrm{diag}(b)G:b\in\mathbb{R}^m\}$ (Row Diagonal) | $\tfrac{1}{\sqrt n}\|\cdot\|_{2\to\infty}$ |

On top of the non-accelerated framework (Algorithm 1), the authors stack Nesterov momentum to form the acceleration framework (Algorithm 2), both sharing the same preconditioners and FTL-BTL analysis tools.

### Key Designs

**1. Unified Preconditioning Framework: Turning "Optimizer Design" into "Subspace Selection"**

Addressing the pain point that "every algorithm requires a new proof," the authors restrict the preconditioner $H_k$ to an operator subspace $\mathcal{H}\subset\mathcal{S}$ and impose two conditions (Assumption 1): (A1.1) the projection onto $\mathcal{H}$ is **order-preserving** (positive definite operators remain positive definite after projection); (A1.2) $\mathcal{H}$ is **closed** under any operator function $\psi(H)\in\mathcal{H}$. Based on this, $H_k$ is defined as the solution to:

$$H_k=\arg\min_{H\in\mathcal{H}\cap\mathcal{S}_{++}}\langle H,S_k\rangle+\langle I,\phi(H)\rangle,\quad S_k=\textstyle\sum_{i=0}^{k}g_i\langle g_i,\cdot\rangle$$

where $S_k$ is the cumulative gradient outer product. By choosing the potential function $\phi(h)=\delta h+\eta^2/h$ and applying Assumption 1, this abstract optimization problem yields an **explicit solution** (Lemma 2):

$$H_k=\eta\big(\delta I+\mathrm{proj}_{\mathcal{H}}(S_k)\big)^{-1/2},\qquad H_{k+1}\preceq H_k$$

This is precisely the operator version of the AdaGrad-style "inverse square root of cumulative gradient squares," and the preconditioner is monotonically non-increasing. Compared to Gupta et al. (2017), this provides a closed-form solution for the first time, allowing the use of FTL-BTL lemmas (Lemma 1: $\sum_i\|g_i\|_{H_i}^2\le\langle H_k,S_k\rangle+\langle I,\phi(H_k)\rangle$) to bring all algorithms under a single analysis backbone.

**2. Unified Convergence Theorem under Matrix Hölder Smoothness and Generalized Noise**

To allow one proof to cover scalar, diagonal, and matrix preconditioning, smoothness and noise can no longer be measured by standard Euclidean norms. The authors use Assumption 2 to define the objective as $(\|L\|_{\mathrm{tr}}^{(1-\nu)/2},\nu)$-**Matrix Hölder Smooth** with respect to the norm $\|\cdot\|_L$ ($\nu\in[0,1]$, $L\in\mathcal{H}$). Assumption 3 constrains the noise variance as $\mathbb{E}\|n(x;\xi)\|_{\Sigma^{-1}}^2\le\|\Sigma\|_{\mathrm{tr}}$ (weaker and more general than the sorting-type assumptions used by An/Xie et al.). Both assumptions translate into smoothness/variance bounds for a non-Euclidean norm $R(x)=\|\mathrm{proj}_{\mathcal{H}}(X)\|_{\mathrm{op}}^{1/2}$, where $R(\cdot)$ simplifies to the spectral norm, $\|\cdot\|_\infty$, or $\|\cdot\|_{2\to\infty}$ for different $\mathcal{H}$.

Main result Theorem 1: Setting $\eta=R$ ($R$ almost surely bounds $\max_k R(x_k-x^*)$), Algorithm 1 outputs:

$$\mathbb{E}[f(\bar x_K)-f(x^*)]\le\frac{3\|L\|_{\mathrm{tr}}R^{1+\nu}}{(K+1)^{(1+\nu)/2}}+\frac{3\|\Sigma\|_{\mathrm{tr}}R}{\sqrt{K+1}}+\frac{3\sqrt{\delta}\,R\,\dim(\mathcal{X})}{K+1}$$

The three terms correspond to "smooth/deterministic," "stochastic noise," and "regularization $\delta$." In the smooth case ($\nu=1$), it recovers the SOTA rates for AdaGrad (anisotropic smoothness, Liu et al. 2024b) and ASGO/One-sided Shampoo. In the non-smooth case ($\nu=0$), it is more general than An et al. (2025), and it covers the intermediate Hölder cases ($0<\nu<1$). Most importantly, it provides the **first convergence guarantee for DASGO** and reveals the connections: ASGO/Shampoo↔Muon, DASGO↔Scion (turning off gradient accumulation in $S_k$ reduces DASGO to Scion with $\|\cdot\|_{2\to\infty}$).

**3. Nesterov Momentum Acceleration: First "Momentum + Preconditioning" Dual Benefit for Diagonal Cases**

To answer Q2, the authors add Nesterov momentum (Algorithm 2) to the unified framework, adopting the "time-varying function" interpretation from Kovalev-Borodich (2024). The acceleration analysis hinges on Assumption 4: the smoothness/noise operators $L,\Sigma$ **commute** with the preconditioning subspace $\mathcal{H}$ ($LH=HL$). This assumption **holds automatically for diagonal operators**, requiring no extra cost for AdaGrad and DASGO.

Under the commuting assumption, $H_k^2$ itself becomes the solution to an optimization problem (Lemma 8), allowing the FTL-BTL lemma to be applied again to obtain a logarithmic bound for $\sum_i\|g_i\|_{BH_i^2}^2$ (Lemma 9). Final Theorem 2: taking $\eta=2R$,

$$\mathbb{E}[f(\bar x_{K+1})-f(x^*)]\le\frac{C_K\|L\|_{\mathrm{tr}}R^{1+\nu}}{(K+2)^{(1+3\nu)/2}}+\frac{C_K\|\Sigma\|_{\mathrm{tr}}R}{\sqrt{K+2}}+\frac{4\sqrt{\delta}\,R\,\dim(\mathcal{X})}{(K+2)^2}$$

where $C_K=O(1+\ln K+\ln\tfrac{\|L\|_{\mathrm{tr}}R^\nu}{\sqrt\delta}+\ln\tfrac{\|\Sigma\|_{\mathrm{tr}}}{\sqrt\delta})$ contains only logarithmic factors. Compared to the non-accelerated Theorem 1, the **smooth term exponent improves from $(1+\nu)/2$ to $(1+3\nu)/2$** (for $\nu=1$, from $1$ to $2$, i.e., acceleration from $O(1/K)$ to $\tilde O(1/K^2)$), while the noise term remains $O(1/\sqrt K)$ (due to stochastic lower bounds). This is the **first** proof that AdaGrad/DASGO can benefit from both diagonal preconditioning and Nesterov momentum simultaneously, providing a theoretical explanation for Adam's empirical success.

### Loss & Training
Purely theoretical analysis with no training loss. Algorithm hyperparameters are potential function parameters $\delta, \eta > 0$; the accelerated version uses $\eta=2R, \alpha_k=2/(k+2)$. Theorem 1/2 requires $\max_k R(x_k-x^*) \le R$ almost surely, which can be ensured in stochastic settings by a projection step to a ball of radius $R$ (Appendix D).

## Key Experimental Results

> ⚠️ This is a **purely theoretical paper; there are no numerical experiments.** The data below represents **theoretical convergence rate results** (specifically $\tilde O$ rates for DASGO/AdaGrad under simplify conditions $\delta\ll1$).

### Main Results: Accelerated vs. Non-accelerated (DASGO / AdaGrad)

| Setting | Smooth (Deterministic) Term | Noise Term | Source |
|---------|-----------------------------|------------|--------|
| Non-accelerated (Thm 1) | $\dfrac{\|l\|_1\|X^*\|_{2\to\infty}^{1+\nu}}{K^{(1+\nu)/2}}$ | $\dfrac{\|\sigma\|_1\|X^*\|_{2\to\infty}}{\sqrt{K+1}}$ | Recovers AdaGrad + First for DASGO |
| Accelerated (Thm 2) | $\dfrac{\|l\|_1\|X^*\|_{2\to\infty}^{1+\nu}}{K^{(1+3\nu)/2}}$ | $\dfrac{\|\sigma\|_1\|X^*\|_{2\to\infty}}{\sqrt{K+1}}$ | Smooth term exponent $\uparrow$ |

Acceleration improves the smooth term from $K^{(1+\nu)/2}$ to $K^{(1+3\nu)/2}$ ($\nu=1 \implies K \to K^2$), while the noise term remains unchanged.

### Comparison with Existing Scalar AdaGrad Acceleration

| Method | Smoothness Constant | Noise Constant | When Ours is Better |
|--------|---------------------|----------------|---------------------|
| Our Diagonal Accel | $\|l\|_1, \|X^*\|_{2\to\infty}$ | $\|\sigma\|_1$ | — |
| Scalar AdaGrad Accel | $\|l\|_\infty, \|X^*\|$ | $\sqrt m\,\|\sigma\|_\infty$ | When $\|l\|_1\sim\|l\|_\infty$, $\|\sigma\|_1\sim\|\sigma\|_\infty$ and $\|X^*\|\gg\|X^*\|_{2\to\infty}$ |

### Key Findings
- **Benefits of Diagonal/Matrix Preconditioning derive from "Sparse Smoothness + Dense Solution"**: When the smoothness vector $l$ and noise $\sigma$ are sparse while the optimal solution $X^*$ is dense, $\|l\|_1\sim\|l\|_\infty$ but $\|X^*\|\gg\|X^*\|_{2\to\infty}$, making diagonal preconditioning significantly superior to scalar step sizes, consistent with findings for AdaGrad by Liu et al. (2024b).
- **Momentum only accelerates the "smooth term" and does not improve the "noise term"**: The noise term being stuck at $O(1/\sqrt K)$ is a statistical lower bound for stochastic first-order methods; momentum cannot overcome this. This explains why momentum gains are more pronounced in low-noise/large-batch settings in practice.
- **Unifying Power of a Single Proof**: By switching $\mathcal{H}$, the same Theorem 1 recovers SOTA rates for AdaGrad-Norm, AdaGrad, ASGO, and One-sided Shampoo, each of which originally required independent proofs.

## Highlights & Insights
- **"Selection of Subspace = Selection of Algorithm" abstraction is elegant**: Restricting the preconditioner to an operator subspace $\mathcal{H}$ with two structural assumptions unifies scalar, diagonal, and matrix methods into a closed-form solution $H_k=\eta(\delta I+\mathrm{proj}_{\mathcal{H}}(S_k))^{-1/2}$, allowing for one proof. This idea of parameterizing a whole family of algorithms with constrained subspaces is transferable to other optimizer family analyses.
- **"Commuting Assumption holds automatically for diagonal" is the key for practical acceleration**: Nesterov acceleration requires $L, \Sigma$ to commute with $\mathcal{H}$, and diagonal operators commute naturally. Thus, AdaGrad/DASGO gain acceleration without any extra assumption costs, aligning with the engineering reality of Adam's "diagonal + momentum."
- **Connecting multiple new optimizers**: The authors demonstrate the correspondences DASGO↔Scion and ASGO/Shampoo↔Muon (switching off gradient accumulation converts one to the other), placing these non-Euclidean optimizers under a unified theoretical lens.
- **Full Coverage of Hölder Smoothness $\nu\in[0,1]$**: The same theorem seamlessly handles smooth ($\nu=1$), non-smooth ($\nu=0$), and intermediate cases, and can adapt to unknown smoothness levels (universality).

## Limitations & Future Work
- **Lack of Numerical Experiments**: The paper is entirely convergence analysis. There is no verification on real deep learning tasks to see if accelerated DASGO/AdaGrad is truly faster than Adam.
- **Convexity Assumption**: Analysis is built on convex (including Hölder smooth) objectives, while deep learning is non-convex. Although the author discusses the rationale in Appendix C, this remains a significant gap between theory and practice.
- **Acceleration depends on Commuting Assumption**: Theorem 2 acceleration does not automatically hold for **non-diagonal** cases (like matrix-preconditioned ASGO/Shampoo). Whether general matrix preconditioning can be accelerated remains an open question.
- **Requires known $R$**: The step size $\eta\propto R$ relies on a prior bound for $\max_k R(x_k-x^*)$. While ensured by projection, $R$ is unknown in practice; integration with parameter-free AdaGrad is a natural next step.
- **Outlook**: The authors suggest trying DASGO in scenarios suitable for the non-Euclidean $\|\cdot\|_{2\to\infty}$ norm, as pointed out by Pethick et al. (2025)—it is computationally cheap (no matrix inversion), has built-in adaptive preconditioning, and possesses better theoretical properties than Scion.

## Related Work & Insights
- **vs Gupta et al. (2017)**: Both define $H_k$ as an optimization solution on a subspace, but Gupta required separate proofs, covered only non-smooth cases, and didn't explain general preconditioning benefits. This paper achieves a **closed-form preconditioner** and **single proof** covering the full Hölder spectrum via Assumption 1.
- **vs Liu et al. (2024b) / An et al. (2025) / Xie et al. (2025)**: These works provide SOTA rates for AdaGrad and ASGO separately but use stronger noise assumptions and don't cover Hölder cases. This paper **unifies and reproduces** them using weaker variance bounds and adds the first guarantee for DASGO.
- **vs Trifonov et al. (2025)**: This was the only other work attempting "preconditioning + momentum," but they made unrealistic assumptions about preconditioner dynamics and analyzed only smooth strongly convex, non-stochastic settings. This paper is the **first** to prove the dual benefit in a stochastic, matrix/anisotropic Hölder smooth setting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant unification of AdaGrad-family methods via operator subspaces and the first proof of dual preconditioning + momentum benefits.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; the practical speedup of the accelerated version is unverified.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from assumptions to lemmas to theorems is clear. Discussions connecting Muon/Scion are insightful, though the math density is high.
- Value: ⭐⭐⭐⭐⭐ Provides a unified perspective for various modern optimizers and a theoretical explanation for Adam's effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes](high-dimensional_limit_theorems_for_sgd_momentum_and_adaptive_step-sizes.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](../../ICML2026/optimization/rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICLR 2026\] DNT: a Deeply Normalized Transformer that can be trained by Momentum SGD](dnt_a_deeply_normalized_transformer_that_can_be_trained_by_momentum_sgd.md)
- [\[ICML 2026\] Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration](../../ICML2026/optimization/selecting_samples_on_graphs_a_unified_dataset_pruning_framework_for_lossless_tra.md)

</div>

<!-- RELATED:END -->

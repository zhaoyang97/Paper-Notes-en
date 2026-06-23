---
title: >-
  [Paper Note] A Tale of Two Geometries: Adaptive Optimizers and Non-Euclidean Descent
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper characterizes the relationship between adaptive optimizers like Adam/Shampoo and Normalized Steepest Descent (NSD) methods like SignGD/Muon through the lens of "two geometries / two types of smoothness." Both classes exploit the non-Euclidean geometry of the loss function, but adaptive optimizers rely on a s
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: a8a393db75ef9809
---
# A Tale of Two Geometries: Adaptive Optimizers and Non-Euclidean Descent

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=iaoAKDRAJQ](https://openreview.net/forum?id=iaoAKDRAJQ)  
**Code**: None (Pure theory paper)  
**Area**: Optimization Theory  
**Keywords**: Adaptive optimizers, Non-Euclidean geometry, Adaptive smoothness, Convergence analysis, Nesterov acceleration

## TL;DR
This paper characterizes the relationship between adaptive optimizers like Adam/Shampoo and Normalized Steepest Descent (NSD) methods like SignGD/Muon through the lens of "two geometries / two types of smoothness." Both classes exploit the non-Euclidean geometry of the loss function, but adaptive optimizers rely on a stronger "adaptive smoothness" $\Lambda_{\mathcal H}(f)$, whereas NSD depends on standard smoothness $L_{\|\cdot\|_{\mathcal H}}(f)$. The authors extend the analysis of adaptive smoothness from convex to non-convex settings and prove that this stronger assumption yields benefits unattainable under standard smoothness: a Nesterov acceleration rate of $\tilde O(T^{-2})$ and dimension-independent stochastic convergence rates.

## Background & Motivation
**Background**: Large model training has long been dominated by adaptive optimizers like Adam, but recently, simpler "Normalized Steepest Descent" (NSD) methods such as Muon and Lion have shown surprising competitiveness. A consensus is emerging in the academic community: the strength of these optimizers lies in their ability to exploit the **non-Euclidean geometry** of the loss surface (e.g., $\ell_\infty$ geometry or matrix spectral norm geometry) rather than the default Euclidean $\ell_2$ geometry.

**Limitations of Prior Work**: Bernstein & Newhouse (2024) provided a striking connection—if the exponential moving average (EMA) is disabled, certain adaptive optimizers **exactly degenerate** into their corresponding NSD counterparts: Adam without EMA is NSD under the $\ell_\infty$ norm (SignGD), and Shampoo without EMA is NSD under the spectral norm (Muon). While these families seem separated only by EMA, there has been **no formal result systematically characterizing their relationship** beyond this "algorithmic degeneration"—specifically, whether they exploit non-Euclidean geometry in the same way.

**Key Challenge**: From a theoretical perspective, the authors found that even within the **same geometry**, two fundamentally different smoothness assumptions emerge. One is the **standard smoothness** under a general norm (Definition 2.3), which governs the convergence rate of NSD. The other is **adaptive smoothness** (Definition 2.4), proposed by Xie et al. (2025b), which was previously only proven to govern the convergence of adaptive optimizers in the convex case. The problem lies in understanding the difference between these two types of smoothness and whether the stronger assumption provides tangible benefits.

**Goal**: The paper addresses two questions. Q1: Do Adam/Shampoo and their corresponding NSDs (Lion/Muon) exploit non-Euclidean geometry in the same way? Q2: Does the stronger smoothness assumption required by adaptive methods actually yield optimization advantages?

**Key Insight**: The authors use smoothness assumptions as a "universal ruler" to compare the two algorithm classes. A key observation is that adaptive smoothness $\Lambda_{\mathcal H}(f)$ is never smaller than the standard smoothness $L_{\|\cdot\|_{\mathcal H}}(f)$ in the same geometry (differing at most by a dimension factor $d$). Since it is a stronger condition, the goal is to determine if this "strength" is merely a technical burden or if it can be traded for superior convergence rates.

**Core Idea**: The paper elucidates the difference between adaptive optimizers and NSD using the "adaptive smoothness vs. standard smoothness" dichotomy. It first generalizes the analysis of adaptive smoothness to non-convex settings (answering Q1: they indeed rely on different smoothness types). It then proves that this stronger assumption unlocks acceleration and dimension-independent rates that are impossible under standard smoothness (answering Q2: the strong assumption provides real returns). Finally, it extends this "standard vs. adaptive" dichotomy to stochastic noise.

## Method

### Overall Architecture
This is a **pure optimization theory** work. It does not propose a specific "model/network" but rather builds a **unified analytical framework** that encompasses Adam, AdaGrad, one-sided Shampoo, etc., to compare geometric measures. The logical chain is: define a meta-algorithm (Algorithm 1) representing a broad class of adaptive optimizers → define "adaptive smoothness" and contrast it with the "standard smoothness" of NSD → prove that adaptive smoothness governs adaptive optimizers in non-convex settings (answering Q1) → demonstrate that this stronger assumption yields better rates via "acceleration" and "noise" perspectives (answering Q2).

The meta-algorithm expresses updates in a preconditioner form: each step solves a matrix optimization using the second moment of historical gradients $M_t$ (via accumulation, EMA, or weighted aggregation) to obtain a preconditioner $V_t$, followed by a preconditioned gradient step:
$$V_t \leftarrow \arg\min_{H\in\mathcal H}\ \langle M_t+\epsilon I_d,\, H^{-1}\rangle + \mathrm{Tr}(H),\qquad x_{t+1}\leftarrow x_t-\eta\, V_t^{-1} g_t.$$
Here, $\mathcal H\subseteq \mathbb S^d_+$ is a **convex cone (set of preconditioners)**. Choosing $\mathcal H$ as all diagonal PSD matrices recovers Adam/AdaGrad; $\mathcal H=\{c I_d\}$ recovers AdaGrad-Norm; $\mathcal H=\mathbb S^d_+$ recovers full-matrix AdaGrad; and $\mathcal H=\mathbb S^{d_L}_+\otimes I_{d_R}$ recovers one-sided Shampoo/ASGO. Thus, the "choice of geometry" is encoded as the "choice of preconditioner set $\mathcal H$," allowing the analysis to cover a large family of optimizers.

### Key Designs

**1. Two Geometries, Two Smoothness Types: Standard governs NSD, Adaptive governs Adaptive**

This is the central "prism" of the paper. Under a fixed non-Euclidean geometry $\|\cdot\|_{\mathcal H}$, the authors distinguish two measures. Standard smoothness $L_{\|\cdot\|}(f)$ is the classical definition—the smallest $L$ such that $\|\nabla f(x)-\nabla f(y)\|_*\le L\|x-y\|$. Adaptive smoothness $\Lambda_{\mathcal H}(f)$ is the "minimum smoothness across all norms induced by $\mathcal H$":
$$\Lambda_{\mathcal H}(f):=\min_{H\in\mathcal H,\ \mathrm{Tr}(H)\le 1} L_{\|\cdot\|_H}(f)=\min_{H\in\mathcal H,\ \forall x:\,-H\preceq\nabla^2 f(x)\preceq H}\mathrm{Tr}(H).$$
For Adam ($\mathcal H=$ diagonal PSD), it is $\Lambda_{\mathrm{diag}}(f)=\min_{-H\le \nabla^2 f \le H, H \text{ diag}} \mathrm{Tr}(H)$. Why does diagonal adaptive smoothness "look like" $\ell_\infty$ geometry? Because well-structured preconditioner sets possess a duality property (Lemma 2.2): taking the supremum of the primal norm and the infimum of the dual norm yields:
$$\sup_{H\text{ diagonal},\,\mathrm{Tr}(H)\le 1}\|\cdot\|_H=\|\cdot\|_\infty,\qquad \inf_{H\text{ diagonal},\,\mathrm{Tr}(H)\le 1}\|\cdot\|_{H,*}=\|\cdot\|_1.$$
Optimizing the NSD convergence rate $O(\sqrt{\Delta_0 L_{\|\cdot\|_H}(f)/T})$ over $H$ converges to $O(\sqrt{\Delta_0\Lambda_{\mathrm{diag}}(f)/T})$—the exact rate of Adam. This explains Adam's "adaptive" nature: it automatically finds the best diagonal-matrix-induced norm for the given loss **without prior knowledge of the optimal $H$**. The critical comparison (Proposition 2.5) is $L_{\|\cdot\|_{\mathcal H}}(f)\le \Lambda_{\mathcal H}(f)\le d\cdot L_{\|\cdot\|_{\mathcal H}}(f)$. Adaptive smoothness is never smaller than standard smoothness; they are fundamentally different, answering Q1.

**2. Well-Structured Preconditioner Sets + A New Matrix Inequality: Generalizing to Non-Convex**

To make the conclusions hold, a unified analysis for non-diagonal preconditioners is needed. The paper uses the definition of a "well-structured preconditioner set" (Definition 2.1): $\mathcal H=\mathbb S^d_+\cap K$, where $K$ is a subalgebra containing the identity and closed under scaling, addition, and matrix multiplication. Previous analyses of non-diagonal preconditioners (like Shampoo) were mostly limited to **convex** or **diagonal-only** (commutative) cases, where terms could be decomposed per component. The difficulty in the general $\mathcal H$ case is **non-commutativity**. The core technical contribution is Lemma 3.3: for any well-structured set, the sum of second-order terms is controlled by:
$$\sum_{t=0}^{T-1}\|V_t^{-1}g_t\|_H^2 \le \mathrm{Tr}(H) \|S_T\|_{\mathrm{op}}, \quad S_T = \sum_{t=0}^{T-1} V_t^{-1}(V_t^2 - \beta V_{t-1}^2) V_t^{-1}$$
where $\|S_T\|_{\mathrm{op}}$ in the non-commutative case is only a $\log d$ factor larger than in the diagonal case. This relies on a new matrix inequality (Lemma C.1) relating the difference between two PSD matrices to the difference of their logarithms. Consequently, the authors obtain a non-convex convergence rate (Theorem 3.2):
$$\frac1T\sum_{t=0}^{T-1}\|\nabla f(x_t)\|_{H,*}\le \tilde O\!\Big(\log d\cdot \sqrt{\tfrac{\Delta_0\,\Lambda_{\mathcal H}(f)}{T}}\Big),$$
matching the optimal $\tilde O(T^{-1/4})$ order. This confirms that **non-convex** adaptive optimizers are similarly governed by $\Lambda_{\mathcal H}(f)$.

**3. Nesterov Acceleration: Stronger Smoothness Unlocks $\tilde O(T^{-2})$ Unavailable to Standard Smoothness**

To address Q2, the authors provide the first piece of evidence: by equipping adaptive optimizers with Nesterov momentum (Algorithm 2) in the convex setting, they archive an accelerated rate (Theorem 4.3):
$$\mathbb E[f(\bar x_T)-f(x^*)]=\tilde O\!\Big(\frac{\Lambda_{\mathcal H}(f)D^2\log^2 d}{T^2}+\frac{\sigma_{\mathcal H}D\log d}{\sqrt T}\Big).$$
The deterministic part is $\tilde O(\Lambda_{\mathcal H}(f)D^2/T^2)$. In contrast, Guzmán & Nemirovski (2015) proved that under standard $\ell_\infty$ smoothness, **no** first-order optimizer can exceed $\Omega(L_{\ell_\infty}(f)/(T\log T))$. This creates a clean **separation**: adaptive smoothness enables acceleration in non-Euclidean geometry that standard smoothness cannot.

**4. Adaptive Variance: Unlocking Dimension-Independent Stochastic Convergence**

The second piece of evidence comes from the stochastic setting. The authors found a parallel version of the smoothness dichotomy in **gradient noise**. Standard gradient variance $\sigma_{\|\cdot\|}$ measures noise under a fixed norm, while adaptive gradient variance (Definition 4.1) requires consistent control across all induced geometries in $\mathcal H$:
$$\sigma_{\mathcal H}^2=\min_{H\in\mathcal H,\,\mathrm{Tr}(H)\le 1}\ \sup_{t,x}\ \mathbb E\big[\|\nabla f_t(x)-\mathbb E\nabla f_t(x)\|_{H^{-1}}^2\big].$$
The mechanism is based on the intuition that **averaging in dual space does not effectively reduce the norm**. Under standard variance, NSD rates are hindered by a dimension distortion factor $\rho = \sup_x \|x\|_{\mathcal H,*}/\|x\|_2$ (e.g., $\rho = \Theta(\sqrt d)$ for diagonal $H$). The paper proves (Theorem 4.5) that with the adaptive variance assumption, NSD with momentum achieves a **dimension-independent** rate $O((\Delta_0 L_{\|\cdot\|_{\mathcal H}}(f))^{1/4}\sqrt{\sigma_{\mathcal H}}/T^{1/4})$. Lower bounds (Theorems 4.6/4.7) show that under standard variance, the $\Omega(\sqrt d)$ dependence is inevitable.

## Key Experimental Results

> This is a pure theory paper with **no numerical experiments**. The core theoretical results are summarized below as "key data." $\tilde O/\Omega$ hides logarithmic factors, $d$ is dimension, $T$ is iterations, and $D=\max_t\|x_t-x^*\|_{\mathcal H}$.

### Main Results: Convergence Rates under Two Smoothness Types

| Setting | NSD (Std. Smoothness $L_{\|\cdot\|_{\mathcal H}}$) | Adaptive (Adaptive Smoothness $\Lambda_{\mathcal H}$) |
|------|------|------|
| Non-convex (Dev.) | $O\big(\sqrt{\Delta_0 L_{\|\cdot\|_{\mathcal H}}(f)/T}\big)$ | $\tilde O\big(\log d\,\sqrt{\Delta_0\Lambda_{\mathcal H}(f)/T}\big)$ (Thm 3.2, matches $\tilde O(T^{-1/4})$) |
| Convex + Nesterov | $\Omega\big(L_{\ell_\infty}(f)/(T\log T)\big)$ Lower bound, **No acceleration** | $\tilde O\big(\Lambda_{\mathcal H}(f)D^2/T^2\big)$ Acceleration (Thm 4.3) |
| Assumption Strength | Weaker | $L \le \Lambda \le dL$ (Prop 2.5), Stronger |

### Noise Analysis: Std. Variance vs. Adaptive Variance (NSD with momentum)

| Noise Assumption | NSD Stochastic Rate | Dimension Dependence |
|------|------|------|
| Std. Variance $\sigma_{\|\cdot\|_*}$ | Contains distortion factor $\psi$, grows with $d$ | $\Omega(\sqrt d)$ Lower bound inevitable (Thm 4.6/4.7) |
| Adaptive Variance $\sigma_{\mathcal H}$ | $O\big((\Delta_0 L_{\|\cdot\|_{\mathcal H}}(f))^{1/4}\sqrt{\sigma_{\mathcal H}}/T^{1/4}\big)$ | **Dimension-independent** (Thm 4.5) |

### Key Findings
- **Adaptive smoothness is the "true governing quantity"**: In non-convex settings, the rate of adaptive optimizers is determined by $\Lambda_{\mathcal H}(f)$ rather than $\|\nabla f\|_2$, completing the comparison with convex results.
- **Stronger assumptions have real returns**: Accelerated rates and dimension-independent rates are proven **unreachable** under weaker standard smoothness/variance.
- **The cost of non-commutativity is a $\log d$**: Extending from diagonal to general well-structured sets only adds a $\log d$ factor, quantifying the gap between Adam and matrix-based Shampoo.

## Highlights & Insights
- **"One EMA gap" upgraded to "One smoothness gap"**: The algorithmic equivalence found by Bernstein & Newhouse is refined into an analytical distinction between $\Lambda_{\mathcal H}$ and $L_{\|\cdot\|_{\mathcal H}}$, providing a unified coordinate system for these optimizer families.
- **Duality as the pivot**: The duality of well-structured sets (Lemma 2.2) is the bridge that explains why Adam presents $\ell_\infty$ geometry and how NSD rates can be optimized into adaptive rates.
- **Transferable technical tools**: Lemma 3.3 and its underlying matrix logarithm inequality (Lemma C.1) are powerful tools for handling non-commutative preconditioners.
- **Unified intuition**: Both acceleration and noise results are attributed to the same phenomenon—that averaging does not effectively reduce the norm in non-Euclidean geometries.

## Limitations & Future Work
- **Purely theoretical, zero experiments**: All conclusions are based on rates and bounds; there is no validation using real model training to estimate the actual values of $\Lambda_{\mathcal H}$ or $\sigma_{\mathcal H}$.
- **Acceleration limited to convex**: The $\tilde O(T^{-2})$ acceleration is only proven in the convex setting; whether it holds in the non-convex landscapes typical of deep learning remains open.
- **Structural constraints**: The analysis relies on the subalgebra structure of $\mathcal H$, which may not apply to more general preconditioning schemes.
- **Constants and log factors**: Factors like $\log d$ hidden in $\tilde O$ might not be negligible in high dimensions, and worst-case bounds may not reflect "typical" loss functions.

## Related Work & Insights
- **vs. Bernstein & Newhouse (2024)**: While they showed algorithmic equivalence without EMA, this paper proves that their convergence mechanisms differ due to different smoothness governing their behavior.
- **vs. Xie et al. (2025b)**: This paper generalizes their adaptive smoothness (initially $\mathcal H$-smoothness) from convex to non-convex and non-diagonal preconditioners using new matrix inequalities.
- **vs. Kovalev (2025a)**: This work provides strictly better bounds for dimension-independent rates by using standard smoothness rather than adaptive smoothness and avoiding restrictive assumptions.
- **vs. Guzmán & Nemirovski (2015)**: Their lower bounds for $\ell_\infty$ smoothness serve as the control group to highlight the necessity of adaptive smoothness for acceleration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

1. **Bernstein & Newhouse (2024)**: On the Equivalence of Adaptive and Normalized Gradient Descent.
2. **Xie et al. (2025b)**: Adaptive Smoothness and the Efficiency of Adaptive Optimizers.
3. **Guzmán & Nemirovski (2015)**: On Lower Complexity Bounds for Optimization over Simple Domains.

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICLR 2026\] DES-LOC: Desynced Low Communication Adaptive Optimizers for Foundation Models](des-loc_desynced_low_communication_adaptive_optimizers_for_foundation_models.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)
- [\[ICLR 2026\] Towards Dynamic Interleaving Optimizers](towards_dynamic_interleaving_optimizers.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)

</div>

<!-- RELATED:END -->

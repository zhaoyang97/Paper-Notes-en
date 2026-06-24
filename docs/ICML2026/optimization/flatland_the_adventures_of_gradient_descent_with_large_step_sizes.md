---
title: >-
  [Paper Note] Flatland: The Adventures of Gradient Descent with Large Step Sizes
description: >-
  [ICML 2026][Optimization][Large learning rates] This paper provides a unified definition of "large step size" requiring only "local Lipschitz / Hölder gradient continuity." By constructing a first-order adaptive step size using equality-type non-monotone line search, Gradient Descent (GD) is driven to run on the Edge of Stability (EoS) from the start of training, suppressing sharpness to the global minimum of $2/K$. It further discovers that "premature entry into the global f…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Large learning rates"
  - "Edge of Stability"
  - "sharpness"
  - "non-monotone line search"
  - "self-stabilization"
date: 2026-05-08
content_hash: 33a7b3fbcd8288af
---

# Flatland: The Adventures of Gradient Descent with Large Step Sizes

**Conference**: ICML 2026  
**arXiv**: [2606.06722](https://arxiv.org/abs/2606.06722)  
**Code**: To be confirmed  
**Area**: Optimization Theory / Edge of Stability  
**Keywords**: Large learning rates, Edge of Stability, sharpness, non-monotone line search, self-stabilization

## TL;DR
This paper provides a unified definition of "large step size" requiring only "local Lipschitz / Hölder gradient continuity." By constructing a first-order adaptive step size using equality-type non-monotone line search, Gradient Descent (GD) is driven to run on the Edge of Stability (EoS) from the start of training, suppressing sharpness to the global minimum of $2/K$. It further discovers that "premature entry into the global flat region is harmful" and employs a self-stabilization constraint to recover failed training sessions.

## Background & Motivation
**Background**: In deep learning, sharpness (the maximum eigenvalue of the training loss Hessian $\lambda_1(H)$) is widely believed to be related to generalization; the hypothesis that "flat minima generalize better" is longstanding (Hochreiter & Schmidhuber, 1997), supported by the success of SAM. Another clue originates from Cohen et al. (2021): GD with a fixed step size $\eta$ undergoes two phases—progressive sharpening (monotone loss decrease, sharpness increase), followed by entry into EoS (non-monotone loss oscillation, sharpness stabilizing near $2/\eta$). Step sizes that hit EoS earlier exhibit more intense loss oscillation but faster convergence; excessively large ones lead to divergence.

**Limitations of Prior Work**: Although large step sizes provide speed and flatness, their definitions as "large" or "small" vary across literature (e.g., Mohtashami 2023 vs. Wang 2025). Critically, Cohen et al. (2021) noted that **given a step size $\eta$, it is unknown whether or when GD will hit EoS**. Neural network objective functions are generally not globally $L$-smooth, rendering the classical convex optimization conclusion "$\eta < 2/L$ guarantees convergence" inapplicable.

**Key Challenge**: Seeking a "maximum possible" step size for acceleration and flattening while "guaranteeing convergence"—where convergence depends on a global smoothness constant $L$ that either does not exist for deep networks or is too large to be useful. Goodfellow et al. (2016) listed this as a long-standing open problem: **How to choose the largest learning rate that still guarantees GD convergence?**

**Goal**: (1) Provide a unified definition of "large step size" depending only on local smoothness; (2) Design a first-order algorithm to find such step sizes **without binary second-order information**; (3) Characterize the points to which these step sizes lead and their implications for convergence and generalization.

**Key Insight**: The authors observe that non-monotone oscillatory behavior of GD under EoS is highly similar to **non-monotone line search** (Grippo 1986) which allows temporary loss increases while maintaining convergence. Since non-monotone line search has a mature convergence theory, it can be leveraged to "actively induce" EoS.

**Core Idea**: Utilize "equality-type non-monotone line search" for automatic step size determination. This is proven to be Lipschitz/Hölder-aware (implicitly sharpness-aware) and naturally pushes GD to EoS; a self-stabilizing upper bound is added to prevent premature entry into harmful global flat regions.

## Method

### Overall Architecture
The proposed method is not a data pipeline but "a definition + a step size solver + a set of property proofs + a correction term." The logic is as follows: The unrealistic assumption of "global $L$-smoothness" is replaced by **segment smoothness**, which holds locally along the anti-gradient direction. Under this weak assumption, "large step size" is redefined. An **equality line search** with backtracking and extrapolation is then used to approximate this step size. It is proved to be Lipschitz-aware, Hölder-aware, and consequently sharpness-aware near stationary points, while maintaining global convergence. Finally, experimental observations regarding the harm of early flatness lead to a self-stabilization upper bound correction. As this is a theory-heavy paper without a multi-module serial process, no framework diagram is provided.

Let $w_\eta := w - \eta\nabla f(w)$. Local segment smoothness is defined as:

$$l(w,\eta) := \sup_{\hat\eta\in(0,\eta]}\frac{\|\nabla f(w)-\nabla f(w_{\hat\eta})\|}{\|w-w_{\hat\eta}\|},$$

with the convention $0/0=0$. If for each $w$ there exists some $\eta_w>0$ such that $l(w,\eta_w)$ is bounded, $f$ is segment smooth—a condition weaker than global smoothness that neural networks satisfy. Based on this, a local descent lemma exists: for $\eta<2/l(w_k,\eta_{w_k})$, the function value decreases monotonically; otherwise, divergence can occur (e.g., quadratic functions).

### Key Designs

**1. Segment smoothness and unified "large step size" definition: Replacing global smoothness with local, directional weak assumptions**

Traditional theory is limited by the need for a global $L$. This work uses the directional $l(w,\eta)$ instead. A key property is that $l(w,\eta)$ is monotonic with respect to $\eta$, allowing $\eta_w$ to be increased to satisfy $\eta_w\ge 2/l(w,\eta_w)$. Accordingly, a **large step size** is defined (Definition 2.3): a sequence $\{\eta_k\}$ is "large" at iteration $k$ if and only if (i) $\eta_k\ge 1/l(w_k,\eta_{w_k})$, and (ii) there exists $p\in[0,P]$ such that $\eta_{k+p}>2/l(w_{k+p},\eta_{w_{k+p}})$. Essentially, the step size does not need to exceed the threshold for monotone descent every step, but must **cross it at least once every $P$ steps**, inducing EoS-style non-monotone oscillation without diverging.

**2. Equality-type (non-monotone) line search: Pushing step size to EoS without second-order information**

The authors employ a non-monotone line search condition:

$$f(w_k-\eta_k\nabla f(w_k))\le f(w_k)-c\eta_k\|\nabla f(w_k)\|^2+\epsilon_k,$$

where $\epsilon_k:=R_k-f(w_k)\ge 0$ and $R_k$ is a reference value (monotone line search if $R_k=f(w_k)$; Grippo 1986 if $R_k=\max_i f(w_{k-i})$; this paper primarily uses Zhang & Hager 2004). The algorithm uses **backtracking and extrapolation**: if the inequality fails, the step size is **halved**; if it holds, it is **doubled**. Thus, $\eta_k$ approaches the value $\eta_k^*$ where the inequality becomes an equality, forcing the step size to be as large as possible and naturally residing on the edge of stability.

**3. From Lipschitz/Hölder-awareness to sharpness-awareness: First-order methods implicitly perceiving curvature**

The equality line search step size satisfies Lipschitz-awareness (Prop 2.5): $\eta_k\ge \frac{2\delta(1-c)}{l(w_k,\eta_{w_k})}$, ensuring Definition 2.3 (i) as long as $2\delta(1-c)\ge 1$. The non-monotone version also satisfies Hölder-awareness (Prop 2.8) and achieves global convergence $\lim_k\|\nabla f(w_k)\|=0$ (Thm 2.9) without global Lipschitz requirements. Critically, Corollary 2.10 states: as $w_k\to w^*$, $l(w_k,\eta)$ converges to the directional sharpness at $w^*$. For sufficiently large $k$, $\eta_k\ge \frac{2(1-c)\delta}{|\lambda_1(H(w^*))|+\varepsilon}$—**the step size is lower-bounded by the inverse of the maximum eigenvalue of the Hessian at the stationary point**. This means first-order methods implicitly acquire second-order (sharpness) information through oscillation.

**4. Self-stabilizing upper bound: Blocking the path to "premature global flatness"**

Experiments reveal a counter-intuitive phenomenon: large step sizes often suppress sharpness to an surprisingly small and constant value of $2/K$ ($K$ being the number of classes for MSE loss). This is proved to be the **global minimum** of sharpness—where the sub-Hessian for the last layer's bias is a diagonal matrix with elements equal to $2/K$, while adjacent eigenvalues sum to zero, indicating these points are actually **saddle points**. Navigating to this region too early prevents the network from learning meaningful features (loss stuck at "random guess" levels). Using self-stabilization arguments, the authors **limit the step size to be less than $K$**, forcing GD into slightly steeper valleys, which allows training to reach "successfully-flat" regions with low training loss and higher test accuracy.

### Loss & Training
Main experiments use MSE loss (Cross-Entropy in appendix), Full-batch GD, subsampled 5000 images from datasets, and calculate sharpness every 100 iterations. Line search parameters are $c=10^{-4}, \delta=0.5$. CDAT uses $\sigma=2.06$, and SAM step sizes are grid-searched.

## Key Experimental Results

### Main Results
The authors ran experiments on the Cartesian product of "Datasets $\times$ Networks": CIFAR10/100, SVHN, EMNIST with MLP, CNN, ResNet34, WideResNet50_2, DenseNet121, and VGG11.

| Method | Uses 2nd-order info? | $\eta_k\!\cdot\!\lambda_1(H)$ behavior | Suppresses sharpness to | Convergence guarantee |
| :--- | :--- | :--- | :--- | :--- |
| LS (Monotone Line Search) | No | Sharpness increases, $\eta$ decreases | Increases | Yes |
| NLS (Ours, Non-monotone) | No | Constantly $>1-c$, often $>2$; starts at EoS | $2/K$ (Global Min) | Yes (Thm 2.9) |
| PoNLS (Ours, Polyak init) | No | Same as NLS | $2/K$ | Yes |
| CDAT (Roulet 2024) | Yes | Constantly $>2$ | Weaker than NLS | No |
| SAM | Yes (Perturbation) | Different EoS definition | Weaker than NLS | — |

Key Conclusion: NLS/PoNLS $\eta_k\cdot\lambda_1(H(w_k))$ values are consistently $>1-c$ and often $>2$. Combined with frequent loss oscillations, this confirms they **operate at EoS from the very first step** and outperform SAM and CDAT in reducing sharpness.

### Ablation Study

| Phenomenon | Observation | Meaning |
| :--- | :--- | :--- |
| Sharpness converges to $2/K$ | NLS/PoNLS suppress $\lambda_1(H)$ to a constant $2/K$ | Proved to be the global minimum of sharpness (bias sub-Hessian) |
| Adjacent eigenvalues sum to 0 | Eigenvalues below the first $K$ cancel out pairwise | These global flat points are actually saddle points |
| Premature Flatness | PoNLS on EMNIST $\times$ CNN stalls at initial loss | "Unsuccessfully-flat": loss remains at random guess level |
| Self-stabilization ($<K$) | Failed training becomes successfully-flat | Sharpness slightly $>2/K$, low training loss, high test accuracy |

## Key Findings
- **First-order methods can implicitly perceive curvature**: NLS does not touch the Hessian, yet is lower-bounded by the inverse of sharpness due to oscillations, effectively gaining second-order information.
- **"Flatter is always better" is a misconception**: Pushing sharpness to the global minimum of $2/K$ hinders convergence because those points are saddle points. Slightly steeper valleys are better for training.
- **Self-stabilization is a critical correction**: Simply bounding step sizes to $<K$ recovers stalled training, indicating the issue is not step size magnitude but "when and where" it hits flatness.

## Highlights & Insights
- **Linking classical line search to modern EoS**: Using mature theories from Grippo (1986) provides a provable tool to "actively run on EoS," which is more elegant than manually tuning fixed step sizes.
- **Unified definition resolves terminology disputes**: Definition 2.3 uses segment smoothness to concretely define "large step size," ending inconsistencies across different papers.
- **Characterization of "Global Minimum Sharpness = Saddle Point"**: Using the diagonal bias sub-Hessian to derive $2/K$ and eigenvalue cancellation to identify saddle points elegantly explains why "perfect flatness" is undesirable.

## Limitations & Future Work
- **Experimental scope**: Main experiments are full-batch GD with 5000 samples and MSE. While the appendix covers SGD and ViT, the gap to large-scale mini-batch training remains.
- **Runtime comparison**: Line search introduces overhead per step; while comparative to SAM/CDAT at $\delta=0.5$, it becomes expensive at $\delta=0.9$, and actual acceleration benefits are not fully quantified.
- **Empirical nature of the upper bound**: The $<K$ bound is derived from specific analysis; its optimality and adaptivity for general tasks remain to be studied.

## Related Work & Insights
- **vs. Cohen et al. (2021)**: They **observed** the two-phase EoS phenomenon with fixed $\eta$; this work **actively constructs** EoS-operating step sizes with convergence proofs.
- **vs. SAM (Foret 2020)**: SAM explicitly pursues flat minima via perturbations; this work is implicitly sharpness-aware through oscillation without second-order info, showing stronger sharpness reduction but highlighting the risk of "too much flatness."
- **vs. CDAT (Roulet 2024)**: CDAT uses second-order info and lacks convergence guarantees; this work provides a first-order alternative with theoretical guarantees (Thm 2.9).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Gradient Descent with Large Step Size Restores Symmetry in Deep Linear Networks with Multi-Pathway](gradient_descent_with_large_step_size_restores_symmetry_in_deep_linear_networks_.md)
- [\[ICLR 2026\] Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region](../../ICLR2026/optimization/gradient_descent_with_large_step_sizes_chaos_and_fractal_convergence_region.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](../../ICLR2026/optimization/on_the_convergence_direction_of_gradient_descent.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)

</div>

<!-- RELATED:END -->

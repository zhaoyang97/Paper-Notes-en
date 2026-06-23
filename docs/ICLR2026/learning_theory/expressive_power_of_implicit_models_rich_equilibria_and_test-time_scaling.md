---
title: >-
  [Paper Note] Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper characterizes the expressive power of implicit models (fixed-point/DEQ style) from a non-parametric perspective in function space. It proves that "a simple (globally Lipschitz) update operator $G$ can express any complex locally Lipschitz mapping via fixed-point iteration." This provides a theoretical explan
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: fdf6d5c534142d43
---
# Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=Pwnf1vsucu](https://openreview.net/forum?id=Pwnf1vsucu)  
**Code**: https://github.com/liujl11git/IMP-Power  
**Area**: Learning Theory / Implicit Models / Expressivity  
**Keywords**: Implicit models, fixed-point iteration, test-time scaling, local Lipschitz, expressivity

## TL;DR
This paper characterizes the expressive power of implicit models (fixed-point/DEQ style) from a non-parametric perspective in function space. It proves that "a simple (globally Lipschitz) update operator $G$ can express any complex locally Lipschitz mapping via fixed-point iteration." This provides a theoretical explanation for the empirical phenomenon where implicit models match or exceed larger explicit networks by increasing test-time iterations. The paper validates that "iterations $\uparrow \to$ mapping complexity (empirical Lipschitz constant) $\uparrow$ and accuracy improves synchronously" across imaging, scientific computing, operations research, and LLM reasoning.

## Background & Motivation
**Background**: Implicit models are an emerging class of architectures. Unlike feed-forward networks that compute results in a single pass, they repeatedly iterate a shared parameter block $G$ to a fixed point. During training, an operator $G$ is learned such that the target satisfies $y^* = G(y^*, x)$; during inference, a root-solver (most simply Picard iteration $y_{t}=G(y_{t-1},x)$) finds this fixed point. Representative works include DEQ (Bai et al., 2019) and Implicit Deep Learning (El Ghaoui et al., 2021), recently extending to looped transformers for LLM reasoning (Geiping et al., 2025).

**Limitations of Prior Work**: Implicit models have three recognized advantages: (i) equivalence to an infinitely deep weight-tied network with constant memory training; (ii) the ability to "implicitly" bake in physical/geometric/safety constraints; (iii) most surprisingly, they often match or exceed the accuracy of explicit networks with more parameters simply by iterating more steps. While (i) and (ii) are well-understood, **there is currently no convincing theoretical explanation for (iii)**.

**Key Challenge**: Answering (iii) essentially requires understanding the "expressive power" of implicit models—the size of the set of input-to-output mappings they can represent. Prior studies either address universal approximation only in specific settings (Bai et al., 2019; Marwah et al., 2023), provide separation results relative to explicit models (Wu et al., 2024), or approach from infinite-width/kernel limits (Gao et al., 2022). **A complete characterization of the "representable function class" is missing**.

**Goal**: The problem is decomposed into two questions. (Q1) Can the expressivity of implicit models at least match explicit models? i.e., for any target $F$, does a $G$ always exist such that $y_t(x)\to F(x)$? (Q2) Is there "extra expressivity" in implicit models? i.e., can a relatively simple $G$ express a complex $F$ via iteration? If Q2 holds, it directly explains phenomenon (iii).

**Key Insight**: The authors start from a simple observation: many complex functions with singularities are actually solutions to smooth equations. For example, $F(x)=1/x$ explodes as $x\to 0$, requiring increasing depth and width in explicit networks; however, $1/x$ is simply the solution to $xy-1=0$, and the corresponding iteration operator $G(y,x)=y-\eta(xy-1)$ is smooth everywhere without singularities. **Complexity can be hidden in the "solution," while the update rule itself remains simple**.

**Core Idea**: "Simplicity" is formalized as global Lipschitz continuity, and "complexity" as local Lipschitz continuity. It is proved that any locally Lipschitz $F$ can be written as the fixed point of a "regular" implicit operator $G$. Thus, the expressivity of an implicit model is not derived from stacking parameters, but is **unlocked incrementally via test-time compute (iterations)**.

## Method

### Overall Architecture
This is a **theoretical paper** that provides two theorems to tightly bound the expressivity of implicit models, complemented by numerical verification across four domains. The logic follows: establishing intuition of "simple operator $\to$ complex fixed point" via the $1/x$ example, rigorously defining "simple/complex targets" (global vs. local Lipschitz) and "regular update operators" ($G$), providing a **Sufficiency Theorem** (any locally Lipschitz $F$ can be represented by a regular $G$) and a **Necessity Theorem** (any regular $G$ fixed-point map must be locally Lipschitz). Finally, this theory is translated into a testable prediction—"as iterations increase, the effective Lipschitz constant of the current iteration map $y_t(\cdot)$ grows"—which is validated across four domains.

### Key Designs

**1. Distinguishing complex targets and simple operators via "Local vs. Global Lipschitz"**

To discuss "simple operators expressing complex mappings," one must define simple vs. complex. A mapping $Q$ is **$L$-Lipschitz (global)** if there exists $L$ such that $\|Q(x_1)-Q(x_2)\|\le L\|x_1-x_2\|$ for all points. If it is only Lipschitz in a neighborhood of each point, but the constant can grow unbounded with position, it is **locally Lipschitz**.

The authors categorize **globally Lipschitz as "simple" operators and locally Lipschitz as "complex" targets**. This captures functions like $\log x$ (on $(0,1]$), $\tan x$, $\sqrt{x}$, $\Gamma(x)$, and $1/x$, which are locally but not globally Lipschitz due to exploding slopes. Importantly, Assumption 2.2 is permissive: the domain $X$ **need not be bounded, compact, closed, or connected**, allowing $F$ to explode at boundaries or internal "holes."

**2. Precise definition of "simple update rules" via "regular implicit operators"**

A regular operator $G:\mathbb{R}^n\times X\to\mathbb{R}^n$ must satisfy: (i) for fixed $y$, the map $x\mapsto G(y,x)$ is **globally Lipschitz** in $x$, with a constant growing at most linearly with $\|y\|$; (ii) for fixed $x$, the map $y\mapsto G(\cdot,x)$ is $\mu(x)$-**contractive** ($\mu(x)\in(0,1)$ and continuous). 

Condition (i) ensures $G$ is "simple" (smooth/non-exploding) in the $x$ direction; (ii) ensures a unique fixed point $y^*(x)$ exists per Banach Fixed Point Theorem, with Picard iterations $y_t(x)\to y^*(x)$ converging. Note that $G$ does **not** need to be jointly Lipschitz in $(y,x)$.

**3. Sufficiency + Necessity: Defining the precise boundaries of expressivity**

- **Theorem 2.4 (Sufficiency)**: Under Assumption 2.2, for any (locally Lipschitz) target $F$, **there exists a regular implicit operator $G$** whose fixed-point mapping is $F$. This affirmatively answers (Q1) and (Q2).
- **Theorem 2.5 (Necessity)**: Any regular $G$ (on $X$) has a unique fixed point $y^*$, and the fixed-point mapping $x\mapsto y^*(x)$ **is necessarily locally Lipschitz**.

Together, these results show that the class of functions representable by regular implicit models is exactly the class of locally Lipschitz functions.

**4. Dynamical Interpretation: Expressivity "unlocked" via iteration**

The static existence theorem is linked to the dynamics of iterative solvers. For $y_0=0$, the first step $y_1(x)=G(0,x)$ is globally Lipschitz and "simple." As iterations proceed $y_t\to F$, if $F$ has high slopes, the **effective Lipschitz constant of $y_t(\cdot)$ must grow with $t$** to match this complexity:

$$\lim_{t\to\infty}\frac{\|y_t(x)-y_t(x')\|}{\|x-x'\|}=\frac{\|F(x)-F(x')\|}{\|x-x'\|}.$$

This reveals a fundamental difference: **explicit networks expand model scale to approximate targets, while implicit models expand expressivity via test-time compute**.

### A Complete Example
LLM reasoning (Case 4, looped transformer): Inputting two prompts differing by one word: $x=$ "explain the difference between **charge and voltage**" (physics) vs. $x'=$ "explain the difference between **charge and pay**" (finance).

- $t=2$: Outputs echo the input mechanically without context separation.
- $t=8$: Differentiation begins—$x$ moves toward physics definitions; $x'$ lags.
- $t=32$: Both contexts yield stable, precise definitions.

This illustrates the prediction: **a tiny input perturbation $\to$ vastly different outputs (high slope/complexity)**, and this "contextual resolution" grows with iterations.

## Key Experimental Results

### Main Results: Synchronized evolution of $L_t$ and Accuracy

| Case | Parametrized $G$ | $L_1$ (Init) | $L_t$ After Iteration | Accuracy Change |
|------|------|------|------|------|
| Image Deblurring | PGD / HQS Denoiser $H_{\theta,\sigma}$ | 0.140 / 0.436 | Saturates to ≈5.0 | PSNR rises and stabilizes |
| Navier–Stokes (SciComp) | FNO Kernel + MLP | 23.1 | ≈367 (t=50) | Rel. Error 1.1 → 0.078 |
| Linear Programming (OR) | Implicit GNN | Small | Significant growth | Rel. Error 0.575 → 0.146 |
| LLM Reasoning | Looped Transformer | — | Increases with $t$ | Context separation improves |

In all cases, targets are locally Lipschitz. The "simple" initial iteration ($L_1$) evolves into a higher complexity $L_t$ as accuracy improves, confirming that $L_t$ growth reflects the expression of the target mapping's inherent complexity rather than instability.

### Ours vs. Prev. SOTA: Small Implicit ≥ Large Explicit (LP Task, Table 1)

| Model | Emb=4 | Emb=8 | Emb=16 | Emb=32 |
|------|------|------|------|------|
| Explicit GNN Params | 580 | 2,088 | 7,888 | 30,624 |
| Explicit Test Error | 0.397 | 0.273 | 0.283 | 0.318 |
| Implicit GNN Params | 722 | 2,350 | 8,390 | 31,606 |
| Implicit Test Error | 0.218 | 0.177 | 0.152 | 0.156 |

Implicit models significantly outperform explicit models at equivalent embedding sizes. Notably, **smaller implicit models outperform larger explicit models** (e.g., Implicit-4 (0.218) < Explicit-8 (0.273)), validating that iterations of a simple operator can substitute for parameter scale.

### Key Findings
- **$L_t$ growth and accuracy gain are coupled**: Across PSNR and relative error metrics, complexity increases as error decreases, supporting the "unlocking expressivity" thesis.
- **Improved generalization**: Explicit GNNs overfit as width increases (error 0.273 → 0.318), while implicit GNNs continue improving. This is attributed to the "implicit" nature of LP constraints and the regularization effect of a simple $G$.
- **Robustness Trade-off**: Forcing global Lipschitz constraints on the fixed-point map (as done in some robustness literature) **removes** the unique expressivity of implicit models. Regularization should be task-specific rather than a global slope constraint.

## Highlights & Insights
- **Frames test-time scaling as an expressivity problem**: Provides a mechanism-level explanation for why iterating a small shared block can match a large network.
- **The $1/x$ example is highly intuitive**: It clarifies how complexity can reside in the solution of a smooth equation.
- **Predictive and measurable**: The "effective Lipschitz growth" can be directly measured across domains, providing strong verification for the theory.
- **Inverse advice for robustness**: Points out that strict Lipschitz bounds on implicit mappings may "neuter" their ability to represent complex real-world functions.

## Limitations & Future Work
- **Existence vs. Learnability**: The theorems prove such a $G$ exists but do not specify if standard training (e.g., gradient descent) will always find it or how many samples are needed.
- **Lipschitz assumption scope**: Does not directly apply to non-Lipschitz, discontinuous, or discrete-domain targets.
- **Efficiency trade-offs**: Lacks quantitative analysis of how many iterations equal how many parameters in terms of FLOPs or wall-clock time.
- **Future directions**: Extending the framework to discrete/sequence spaces for more rigorous LLM support and studying the誘induction of regularity during training.

## Related Work & Insights
- **vs. DEQ/Implicit Deep Learning**: Complements these architectures by providing the **complete expressivity characterization** lacking in original works.
- **vs. Universal Approximation**: Moves beyond "approximability" to provide a precise functional boundary (Locally Lipschitz).
- **vs. Infinite Width Limits**: Offers a finite-dimensional function space perspective that is more aligned with practical model sizes.
- **vs. Looped Transformers**: Provides a theoretical lens to explain why repeated layers in transformers allow for emergent reasoning capabilities via context separation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Verification via Optimal Transport: Coverage, ROC, & Sub-Optimality](test-time_verification_via_optimal_transport_coverage_roc_sub-optimality.md)
- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](../../ICML2026/learning_theory/on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[ICLR 2026\] Finite-Time Convergence Analysis of ODE-based Generative Models for Stochastic Interpolants](finite-time_convergence_analysis_of_ode-based_generative_models_for_stochastic_i.md)
- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)

</div>

<!-- RELATED:END -->

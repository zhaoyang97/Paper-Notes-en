---
title: >-
  [Paper Note] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness
description: >-
  [ICML 2026][Reinforcement Learning][Non-uniform smoothness] This paper proposes a broader non-uniform smoothness $(H_0,H_1)$-NS than Zhang et al.'s $(L_0…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Non-uniform smoothness"
  - "Steepest Descent"
  - "Adam"
  - "RMSProp"
  - "Łojasiewicz condition"
date: 2026-05-08
content_hash: 1b1742c98519eb50
---

# Convergence of Steepest Descent and Adam under Non-Uniform Smoothness

**Conference**: ICML 2026  
**arXiv**: [2605.30648](https://arxiv.org/abs/2605.30648)  
**Code**: None  
**Area**: Optimization Theory  
**Keywords**: Non-uniform smoothness, Steepest Descent, Adam, RMSProp, Łojasiewicz condition

## TL;DR
This paper proposes a broader non-uniform smoothness $(H_0,H_1)$-NS than Zhang et al.'s $(L_0,L_1)$-NS. Under this assumption and (non-uniform) Łojasiewicz conditions, it provides the first unified convergence rates for deterministic diagonal RMSProp / Adam and general Normalized Steepest Descent (Sign GD, Norm.GD, Sign CD-GS), proving they are strictly faster than GD / AdaGrad / heavy-ball on logistic regression with separated data and softmax policy gradients.

## Background & Motivation

**Background**: Classical first-order method convergence analysis relies on global uniform smoothness—where the Hessian spectral norm is upper-bounded by a constant $L$, which significantly deviates from the actual loss surfaces of neural network training. Recently, Zhang et al. (2020b) proposed $(L_0,L_1)$-NS: $\|\nabla^2 f(\theta)\|_2\le L_0+L_1\|\nabla f(\theta)\|_2$, empirically showing training losses satisfy this condition, thus explaining the effectiveness of gradient clipping. Vaswani & Harikandeh (2025) and Alimisis et al. (2025) further rewrote the upper bound as an affine function of the function value $f(\theta)$, making the condition closed under finite sums and affine transformations.

**Limitations of Prior Work**: (1) Existing $(L_0,L_1)$-NS analysis is mostly limited to the $\ell_2$ norm and GD/heavy-ball, which does not correspond to Adam/RMSProp and sign-based methods (Sign GD), the latter being core to LLM optimizers in practice; (2) Current theories for Adam/RMSProp (Li et al. 2023b; Wang et al. 2024a/b) assume bounded gradients or additional convexity and fail to provide lower bounds strictly separated from GD; (3) Key application problems like logistic regression on separated data and softmax policy gradients are not covered by uniform smoothness, causing classical GD convergence rates to degenerate to $O(1/\epsilon)$ sublinear rates.

**Key Challenge**: To accommodate within a unified framework (a) Steepest Descent under various dual norms $(p,q)$ (including Sign GD), (b) adaptive methods like Adam/RMSProp, and (c) unbounded loss surfaces such as non-convex two-layer networks and policy gradients, one must find a non-uniform smoothness characterization that both ensures a descent inequality holds and distinguishes Adam from AdaGrad.

**Goal**: Define and study $(H_0,H_1)$-NS—where the $(p,q)$ operator norm of the Hessian is controlled by $H_0+H_1 f(\theta)$; combined with non-uniform Łojasiewicz (NL) conditions, provide unified convergence rates for NSD, RMSProp, and Adam on such problems, and prove the provable faster performance of RMSProp/Adam relative to GD/AdaGrad/AMSGrad/heavy-ball.

**Key Insight**: The authors discovered that "the Hessian of $f$ is affinely upper-bounded by $f(\theta)$" automatically implies a "multiplicative Lipschitz" property for both the function and its gradient. That is, within a small $\ell_p$ neighborhood, the "amplification factor" of $f$ and $\|\nabla f\|_q$ is bounded by an exponential factor—a property that exactly fills the gap in Adam/RMSProp analysis where "gradients cannot be compared across steps."

**Core Idea**: Replace the strong "bounded gradient" assumption in traditional Adam/RMSProp analysis with the "multiplicative Lipschitz" property derived from "$(H_0,H_1)$-NS," and treat Sign GD as a special case of steepest descent with $(p,q)=(\infty,1)$, thereby unifying NSD and the Adam family within the same proof framework.

## Method

### Overall Architecture

The study considers unconstrained minimization $\min_{\theta\in\mathbb{R}^D} f(\theta)$, where $f$ is second-order differentiable and non-negative. Core assumptions include:

- **$(H_0,H_1)$-NS** (Assn. 2): For a pair of dual indices $(p,q)$ satisfying $1/p+1/q=1$, $\|\nabla^2 f(\theta)\|_{p\to q}\le H_0+H_1 f(\theta)$.
- **(Non-uniform) Łojasiewicz NL** (Assn. 3): There exists $\tau\in(0,1]$ such that $\|\nabla f(\theta)\|_q\ge\mu(\theta)[f(\theta)-f^*]^\tau$.

These two assumptions cover Prop. 1–4: exponential loss/logistic regression under separated data (Assn. 2 $H_0=0,H_1=\max_i\|x_i\|_q^2$; Assn. 3 $\tau=1, \mu=\gamma_p$), softmax policy gradient (Prop. 3: $H_0=0,H_1\le 24$; $\tau=1, \mu=\pi_\theta(a^*)$), specific two-layer networks (Prop. 4), and GLMs with logit chains. Algorithms are unified as NSD updates $\theta_{t+1}=\theta_t-\eta_t d_t$, where $d_t=\arg\max_{\|d\|_p\le 1}\langle d,\nabla f(\theta_t)\rangle$, with three special cases being Sign GD ($p=\infty$), Norm.GD ($p=2$), and Sign CD-GS ($p=1$).

### Key Designs

1. **Structural Properties of $(H_0,H_1)$-NS: Multiplicative Lipschitz and Non-uniform Descent Lemma**:
    - **Function**: Translates "Hessian controlled by function value" into local comparison inequalities directly applicable in algorithmic analysis.
    - **Mechanism**: Lemma 3 provides a function-value bound for the gradient: $\|\nabla f(\theta)\|_q\le\sqrt{2H_0 f(\theta)+H_1[f(\theta)]^2}$; Lemma 5 proves multiplicative Lipschitz for the "shifted $f$": when $H_1>0$, $(f(y)+H_0/H_1)\le(f(x)+H_0/H_1)\exp(\sqrt{H_1}\|y-x\|_p)$; Lemma 6/10 upgrades this to bounded cross-step ratios for both function and gradient norms; finally, the Lemma provides a non-uniform descent inequality (Eq. 13): when $\|y-x\|_p\le 1/\sqrt{H_1}$, $f(y)\le f(x)+\langle\nabla f(x),y-x\rangle+(H_0+H_1 f(x))\|y-x\|_p^2$.
    - **Design Motivation**: Traditional descent lemmas require a uniform smoothness constant $L$; here $L$ is replaced by $H_0+H_1 f(x)$, automatically adapting to the early training phase where $f$ is large and the Hessian is also potentially large. This inequality is the unified starting point for all subsequent convergence proofs for NSD/RMSProp/Adam.

2. **NSD Two-stage Convergence (Theorem 1)**:
    - **Function**: Provides NSD convergence rates under $(H_0,H_1)$-NS + NL ($\tau, \mu$), covering Sign GD, Norm.GD, and Sign CD-GS.
    - **Mechanism**: Substituting the non-uniform descent inequality into the NSD update yields $f(\theta_{t+1})\le f(\theta_t)-\eta\mu[f(\theta_t)]^\tau+(H_0+H_1 f(\theta_t))\eta^2$. The proof is split into two parts: Phase 1 where $f(\theta_t)\ge\max\{\epsilon,H_0/H_1\}$, using $H_0+H_1 f\le 2H_1 f$ to turn the recurrence into a linear convergence form, resulting in geometric descent of $f$ to $\max\{\epsilon,H_0/H_1\}$; Phase 2 (only if $\epsilon<H_0/H_1$) uses $H_0+H_1 f\le 2H_0$, requiring the step size to be reduced to $\eta=O(\epsilon^\tau)$, resulting in an $O(1/\epsilon^\tau)$ slow phase. In the extreme case $H_0=0$ (e.g., exponential loss on separated data), the entire process remains in Phase 1, allowing linear convergence with a constant step size.
    - **Design Motivation**: Traditional GD is only $O(1/\epsilon)$ on separated data; this work quantifies the "sublinear GD vs. linear NSD" gap for any $\tau$. Step size strategies are also simplified from "requiring line-search" to "constant step size + $\eta=O(\epsilon^\tau)$," corresponding to warm-up + decay in practice.

3. **Cross-step Ratio Analysis for RMSProp / Adam (Theorem 3, etc.)**:
    - **Function**: Provides convergence rates for deterministic diagonal variants of RMSProp ($v_{t,i}$ as exponential moving average of second moments, $d_t=g_t/\sqrt{v_t}$) and Adam (with first-order momentum) under the same assumptions.
    - **Mechanism**: Taking $(p,q)=(\infty,1)$ and applying Lemma 16 gives $\|d_t\|_\infty\le 1/\sqrt{1-\beta}$; subsequently $f(\theta_{t+1})\le f(\theta_t)-\eta\langle\nabla_t,d_t\rangle+\bar L_t\eta^2/(1-\beta)$, where $\bar L_t=H_0+H_1 f(\theta_t)$. The key challenge is lower-bounding $\langle\nabla_t,d_t\rangle=\sum_i g_{t,i}^2/\sqrt{v_{t,i}}$. Lemma 17 uses Cauchy–Schwarz and the $v_{t,i}$ recurrence to show $\langle\nabla_t,d_t\rangle\ge\|\nabla_t\|_1\cdot\|\nabla_t\|_1/\bigl(\sqrt{1-\beta}\sum_{j=0}^{t-1}\sqrt{\beta}^j\|\nabla_{t-j}\|_1\bigr)$. This (*) term manifests the "adaptive preconditioner" as a "current gradient / weighted average of historical gradients"; multiplicative Lipschitz (Eq. 12) is then used to inverse-control $\|\nabla_{t-j}\|_1$ in the denominator to an exponential multiple of $\|\nabla_t\|_1+c$, isolating a linear descent term of the same order as NSD. The Phase 1 / Phase 2 division follows Theorem 1, yielding rates of $O(1/\epsilon^{2\tau})$ for $\tau\le 1/2$ and $O(1/\epsilon^{4\tau-1})$ for $\tau>1/2$. Adam incorporates first-moment momentum, but the analytical framework remains the same.
    - **Design Motivation**: Traditional Adam analysis depends on $\|\nabla\|\le G$, which fails for exponential loss on separated data (gradients can be exponentially large). Using the multiplicative Lipschitz property implied by $(H_0,H_1)$-NS directly instead of the bounded gradient assumption is key to this proof; these techniques also provide faster deterministic rates than existing non-convex Adam results on ordinary $(L_0,L_1)$-NS non-convex functions (Sec. 5.4).

### Loss & Training
The paper does not involve empirical training but provides convergence rates and step size strategies: NSD/RMSProp/Adam achieve $O(\ln(1/\epsilon))$ linear convergence using $\eta=O(1)$ constant step sizes in the $\epsilon>H_0/H_1$ region; in the $\epsilon<H_0/H_1$ region, $\eta=O(\epsilon^\tau)$ (NSD) or $\eta=O(\epsilon^{2\tau})$ (RMSProp/Adam) is required to enter the slow phase.

## Key Experimental Results

### Main Results (Theoretical Rate Comparison — Phase 1 Critical Rates)

| Problem | Method | Step Size | Convergence Rate | Remarks |
|------|------|------|----------|------|
| Exp. Loss (Separated) | GD | const | $\Theta(1/\epsilon)$ | Soudry et al. 2018 |
| Same as above | Sign GD / Norm.GD (NSD) | const $O(1)$ | $O(\ln(1/\epsilon))$ | Ours Theorem 1, **strictly faster** |
| Logistic Reg. (Separated) | GD with Armijo | line-search | $O(\ln(1/\epsilon))$ | Vaswani & Harikandeh 2025 |
| Same as above | NSD / Sign CD-GS | const | $O(n^2/\gamma_p^2+\ln(1/(n\epsilon)))$ | Ours Theorem 2 |
| Softmax PG (MAB) | GD | const | $\Omega(1/\epsilon)$ | Mei et al. 2020 |
| Same as above | NSD | const $O(1)$ | $O(\ln(1/\epsilon))$ | Ours Corollary 1 |
| Two-layer Net + Sep. Data | RMSProp / Adam (det. diag.) | const + const $\beta$ | $O(\ln(1/\epsilon))$ linear | Ours Theorem 3 |
| 1D Logistic Loss | GD / heavy-ball / AdaGrad / AMSGrad | Any | $\omega(\ln(1/\epsilon))$ sublinear LB | Ours Sec. 6, separated from Adam |

### Phase 1 vs Phase 2 Rate Structure

| Method | $\epsilon>H_0/H_1$ (Phase 1) | $\epsilon<H_0/H_1$ (Phase 2, NL $\tau$) | Dimension Dep. |
|------|------------------------------|------------------------------------------|----------|
| NSD (Theorem 1) | $O(\ln(1/\epsilon))$ | $O(1/\epsilon^\tau)$ | None |
| RMSProp (Theorem 3) | $O(\ln(1/\epsilon))$ | $\tau\le 1/2$: $O(1/\epsilon^{2\tau})$; $\tau>1/2$: $O(1/\epsilon^{4\tau-1})$ | None |
| GD / AdaGrad (1D logistic) | $\Theta(1/\epsilon)$ sublinear | — | — |

### Key Findings
- Once $H_0=0$ (e.g., exponential loss, softmax policy gradient), NSD and RMSProp/Adam remain in Phase 1 throughout, allowing linear convergence with constant step sizes; only when the objective has a finite minimizer ($H_0>0$) does the algorithm enter Phase 2 requiring step size reduction, providing a theoretical justification for "learning-rate decay in late training."
- The first provable rate separation between RMSProp/Adam and GD/heavy-ball/AdaGrad/AMSGrad is established on 1D logistic loss (Sec. 6), theoretically confirming the dominance of RMSProp/Adam over AdaGrad in practice.
- By combining the special case of NSD with Sign CD-GS, it is proven that Sign CD-GS—a minimalist algorithm using "Gauss–Southwell coordinate selection + sign direction updates"—can also achieve linear convergence on logistic regression with separated data, matching the normalized coordinate descent rates of Axiotis & Sviridenko (2023).

## Highlights & Insights
- "Attaching smoothness to function values ($H_0+H_1 f$)" is a seemingly modest but revolutionary change: it allows the descent inequality to adapt to the training phase—permitting a large Hessian when the loss is large and naturally tightening as the loss decreases, thus unifying engineering practices like "gradient clipping/warm-up/decay" under a single mathematical path.
- The "exponential bound for gradient cross-step ratios" (Lemma 6/10) is a versatile tool for analyzing adaptive methods (Adam/RMSProp): it transforms the "historical gradients in the denominator"—which hindered traditional proofs—into controllable multiplicative constants. Any optimizer relying on second-moment buffers (Lion, Tiger, etc.) could potentially reuse this framework.
- The lower bound section in Sec. 6 elevates the question of "why RMSProp/Adam generally outperforms AdaGrad" from empirical observation to provable separation; this "upper bound + matching lower bound" argumentation template serves as a strong model for future LLM optimizer comparative studies.

## Limitations & Future Work
- Only **deterministic diagonal** variants of RMSProp/Adam were analyzed, excluding stochastic gradients and matrix preconditioning (Shampoo, Sophia, etc.); extending the cross-step ratio technique to stochastic/full-matrix cases is a natural next step.
- $(H_0,H_1)$-NS requires second-order differentiability and cannot directly cover non-smooth models like ReLU networks, requiring further weakening to weak convexity or generalized smoothness.
- The NL condition requires $\mu(\theta)$ not to degenerate along the path; for deep over-parameterized networks, $\mu$ actually depends on parameter initialization, and quantifying "how long NL holds" remains an open problem.
- The assumptions for "two-layer networks + smooth leaky ReLU + exponential loss" mentioned in the paper are still relatively strict; extending this to sigmoid/softplus and multi-layer networks requires new Hessian control tools.

## Related Work & Insights
- **vs Vaswani & Harikandeh (2025)**: Also based on function-value-based NS assumptions, but they only analyze GD + Armijo line-search; this work generalizes NS to arbitrary $(p,q)$ dual norms, covers NSD/Adam/RMSProp, and proves that constant step sizes match their line-search rates.
- **vs Alimisis et al. (2025)**: Both focus on the "Hessian affinely bounded by function value," with Alimisis using it to explain LR warm-up; this work further utilizes the condition as a bridge for Adam/RMSProp analysis and adds the lower bound separation in Sec. 6, moving from "explaining training phenomena" to "strict rate comparison."
- **vs Li et al. (2023b), Wang et al. (2024a/b)**: These works analyze Adam convergence on $(L_0,L_1)$-NS non-convex functions, requiring bounded gradients and only giving convergence for the best gradient norm; this work does not require bounded gradients and provides linear convergence of function values under convex + NL settings, even achieving faster deterministic rates in the non-convex extension of Sec. 5.4.
- **vs Mei et al. (2020/2021)**: Mei et al. provide an $\Omega(1/\epsilon)$ lower bound for softmax PG using GD and an $O(\ln(1/\epsilon))$ upper bound for Norm.GD; this work generalizes the latter to any NSD (including Sign GD, Sign CD-GS) within a unified NS+NL framework and proves that RMSProp/Adam achieve linear rates on the same problems.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the $O(1/T)$ Convergence of Alternating Gradient Descent-Ascent in Bilinear Games](../../ICLR2026/reinforcement_learning/on_the_o1t_convergence_of_alternating_gradient_descent-ascent_in_bilinear_games.md)
- [\[ICML 2026\] Learning to Approximate Uniform Facility Location via Graph Neural Networks](learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)
- [\[ICML 2026\] Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning](convergence_of_two-timescale_markovian_stochastic_approximations_with_applicatio.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ICML 2026\] Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning](tracking_drift_variation-aware_entropy_scheduling_for_non-stationary_reinforceme.md)

</div>

<!-- RELATED:END -->

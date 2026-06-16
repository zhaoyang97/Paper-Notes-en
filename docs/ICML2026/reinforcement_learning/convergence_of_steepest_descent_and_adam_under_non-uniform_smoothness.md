---
title: >-
  [Paper Note] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness
description: >-
  [ICML 2026][Reinforcement Learning][Adam] This paper proposes $(H_0,H_1)$-NS, a broader non-uniform smoothness than $(L_0,L_1)$-NS by Zhang et al. Under this assumption and the (non-uniform) Łojasiewicz condition, it provides the first unified convergence rates for deterministic diagonal RMSProp / Adam and general Normalized Steepest Descent (Sign GD, Norm.GD,
tags:
  - ICML 2026
  - Reinforcement Learning
  - Adam
  - RMSProp
date: 2026-05-08
content_hash: 3b2e3f84c2dd32bb
---
# Convergence of Steepest Descent and Adam under Non-Uniform Smoothness

**Conference**: ICML 2026  
**arXiv**: [2605.30648](https://arxiv.org/abs/2605.30648)  
**Code**: None  
**Area**: Optimization Theory  
**Keywords**: Non-uniform smoothness, Steepest Descent, Adam, RMSProp, Łojasiewicz condition

## TL;DR
This paper proposes $(H_0,H_1)$-NS, a broader non-uniform smoothness than $(L_0,L_1)$-NS by Zhang et al. Under this assumption and the (non-uniform) Łojasiewicz condition, it provides the first unified convergence rates for deterministic diagonal RMSProp / Adam and general Normalized Steepest Descent (Sign GD, Norm.GD, Sign CD-GS). It proves they are strictly faster than GD / AdaGrad / heavy-ball for logistic regression on separable data and softmax policy gradients.

## Background & Motivation

**Background**: Classical convergence analysis for first-order methods relies on global uniform smoothness, where the Hessian spectral norm is upper-bounded by a constant $L$. This deviates significantly from the loss surfaces encountered in neural network training. Recently, Zhang et al. (2020b) proposed $(L_0,L_1)$-NS: $\|\nabla^2 f(\theta)\|_2\le L_0+L_1\|\nabla f(\theta)\|_2$, empirically showing training losses satisfy this condition, thereby explaining the effectiveness of gradient clipping. Vaswani & Harikandeh (2025) and Alimisis et al. (2025) further reformulated the upper bound as an affine function of the function value $f(\theta)$, making the condition closed under finite sums and affine transformations.

**Limitations of Prior Work**: (1) Existing $(L_0,L_1)$-NS analyses are mostly limited to the $\ell_2$ norm and GD/heavy-ball, which do not correspond to Adam/RMSProp and sign-based methods (Sign GD), the core optimizers for LLMs. (2) Existing theories for Adam/RMSProp (Li et al. 2023b; Wang et al. 2024a/b) assume bounded gradients or additional convexity and fail to provide lower bounds strictly separated from GD. (3) Key applications such as logistic regression on separable data and softmax policy gradients are not covered by uniform smoothness, causing classical GD convergence rates to degenerate to $O(1/\epsilon)$ sublinear rates.

**Key Challenge**: To accommodate (a) Normalized Steepest Descent (including Sign GD) under various dual norms $(p,q)$, (b) adaptive methods like Adam/RMSProp, and (c) unbounded loss surfaces like non-convex two-layer networks and policy gradients within a unified framework, it is necessary to identify a non-uniform smoothness characterization that guarantees descent inequalities and distinguishes Adam from AdaGrad.

**Goal**: Define and investigate $(H_0,H_1)$-NS, where the $(p,q)$ operator norm of the Hessian is controlled by $H_0+H_1 f(\theta)$. In conjunction with the Non-uniform Łojasiewicz (NL) condition, provide unified convergence rates for NSD, RMSProp, and Adam on such problems, and prove the provably faster convergence of RMSProp/Adam relative to GD/AdaGrad/AMSGrad/heavy-ball.

**Key Insight**: The authors discovered that "f's Hessian being affinely upper-bounded by $f(\theta)$" automatically implies that both the function and its gradient are "multiplicative Lipschitz." This means within a small $\ell_p$ neighborhood, the "amplification factor" of $f$ and $\|\nabla f\|_q$ is bounded by an exponential factor—a property that completes the missing puzzle of "cross-step gradient comparison" in Adam/RMSProp analysis.

**Core Idea**: Replace the strong "bounded gradient" assumption in traditional Adam/RMSProp analysis with the "$(H_0,H_1)$-NS $\Rightarrow$ multiplicative Lipschitz $\Rightarrow$ bounded step-wise gradient ratio" chain. By treating Sign GD as a special case of steepest descent with $(p,q)=(\infty,1)$, the authors unify NSD and the Adam family within a single proof framework.

## Method

### Overall Architecture

The study addresses unconstrained minimization $\min_{\theta\in\mathbb{R}^D} f(\theta)$, where $f$ is twice-differentiable and non-negative. Core assumptions include:

- **$(H_0,H_1)$-NS** (Assn. 2): For a pair of dual norms $(p,q)$ satisfying $1/p+1/q=1$, $\|\nabla^2 f(\theta)\|_{p\to q}\le H_0+H_1 f(\theta)$.
- **Non-uniform Łojasiewicz (NL)** (Assn. 3): There exists $\tau\in(0,1]$ such that $\|\nabla f(\theta)\|_q\ge\mu(\theta)[f(\theta)-f^*]^\tau$.

These assumptions cover Prop. 1–4: exponential loss/logistic regression under separable data (Assn. 2 $H_0=0,H_1=\max_i\|x_i\|_q^2$; Assn. 3 $\tau=1, \mu=\gamma_p$), softmax policy gradients (Prop. 3: $H_0=0,H_1\le 24$; $\tau=1, \mu=\pi_\theta(a^*)$), specific two-layer networks (Prop. 4), and GLMs with logical links. These are unified into the NSD update $\theta_{t+1}=\theta_t-\eta_t d_t$, where $d_t=\arg\max_{\|d\|_p\le 1}\langle d,\nabla f(\theta_t)\rangle$. Three special cases yield Sign GD ($p=\infty$), Norm.GD ($p=2$), and Sign CD-GS ($p=1$).

### Key Designs

**1. Structural Properties of $(H_0,H_1)$-NS: Translating "Hessian controlled by function value" into applicable local comparison inequalities**

Traditional descent lemmas require a uniform smoothness constant $L$, which training surfaces lack. $(H_0,H_1)$-NS replaces $L$ with $H_0+H_1 f(x)$. The authors first prove this yields usable comparison inequalities: Lemma 3 provides a function-value bound for the gradient $\|\nabla f(\theta)\|_q\le\sqrt{2H_0 f(\theta)+H_1[f(\theta)]^2}$; Lemma 5 proves multiplicative Lipschitzness for "shifted $f$": when $H_1>0$, $(f(y)+H_0/H_1)\le(f(x)+H_0/H_1)\exp(\sqrt{H_1}\|y-x\|_p)$; Lemma 6/10 upgrades this to bounded ratios for both function and gradient norms across steps. Finally, a non-uniform descent inequality is derived (Eq. 13): when $\|y-x\|_p\le 1/\sqrt{H_1}$, $f(y)\le f(x)+\langle\nabla f(x),y-x\rangle+(H_0+H_1 f(x))\|y-x\|_p^2$. This inequality automatically adapts to the early training phase where "loss is large and Hessian is large," serving as the unified starting point for all convergence proofs of NSD / RMSProp / Adam.

**2. Two-phase Convergence of NSD (Theorem 1): Covering Sign GD / Norm.GD / Sign CD-GS with non-uniform descent inequalities**

Substituting the non-uniform descent inequality into the NSD update $\theta_{t+1}=\theta_t-\eta_t d_t$ yields $f(\theta_{t+1})\le f(\theta_t)-\eta\mu[f(\theta_t)]^\tau+(H_0+H_1 f(\theta_t))\eta^2$. The proof splits into two phases. Phase 1 ($f(\theta_t)\ge\max\{\epsilon,H_0/H_1\}$) uses $H_0+H_1 f\le 2H_1 f$ to turn the recursion into a linear convergence form, where a constant step size achieves geometric descent to $\max\{\epsilon,H_0/H_1\}$. Phase 2 (only if $\epsilon<H_0/H_1$) uses $H_0+H_1 f\le 2H_0$ and reduces the step size to $\eta=O(\epsilon^\tau)$, resulting in a slower $O(1/\epsilon^\tau)$ phase. In the extreme case $H_0=0$ (e.g., exponential loss on separable data), the entire process remains in Phase 1, achieving linear convergence with a constant step size. This quantifies the separation between traditional GD (sublinear $O(1/\epsilon)$) and NSD (linear) on separable data for any $\tau$, simplifying the step-size strategy from "requiring line-search" to "constant step size + $\eta=O(\epsilon^\tau)$," matching the practice of warm-up + decay.

**3. Step-wise Ratio Analysis of RMSProp / Adam (Theorem 3, etc.): Replacing "bounded gradient" with multiplicative Lipschitz**

The difficulty with RMSProp ($d_t=g_t/\sqrt{v_t}$) and Adam (adding first-order momentum) is the historical gradient in the denominator, which prevents cross-step comparison. Setting $(p,q)=(\infty,1)$, Lemma 16 gives $\|d_t\|_\infty\le 1/\sqrt{1-\beta}$, thus $f(\theta_{t+1})\le f(\theta_t)-\eta\langle\nabla_t,d_t\rangle+\bar L_t\eta^2/(1-\beta)$ where $\bar L_t=H_0+H_1 f(\theta_t)$. The key lower bound $\langle\nabla_t,d_t\rangle=\sum_i g_{t,i}^2/\sqrt{v_{t,i}}$ is handled by Lemma 17 using Cauchy–Schwarz and the $v_{t,i}$ recursion to get $\langle\nabla_t,d_t\rangle\ge\|\nabla_t\|_1^2\big/\big(\sqrt{1-\beta}\sum_{j=0}^{t-1}\sqrt{\beta}^j\|\nabla_{t-j}\|_1\big)$, making the "adaptive preconditioner" an "weighted average of current vs. historical gradients." Multiplicative Lipschitzness (Eq. 12) then bounds $\|\nabla_{t-j}\|_1$ in the denominator relative to $\|\nabla_t\|_1+c$, isolating a linear descent term identical to NSD. The Phase 1/Phase 2 division mirrors Theorem 1, yielding $O(1/\epsilon^{2\tau})$ for $\tau\le 1/2$ and $O(1/\epsilon^{4\tau-1})$ for $\tau>1/2$. Adam is included in this framework by incorporating the first moment. Traditional Adam analysis relies on $\|\nabla\|\le G$, which fails for exponential loss on separable data (where gradients can grow exponentially). Using the multiplicative Lipschitzness implied by $(H_0,H_1)$-NS directly replaces the bounded gradient assumption, which is the key to these proofs and yields faster deterministic rates than existing results on non-convex $(L_0,L_1)$-NS functions in Sec. 5.4.

### Loss & Training
The paper does not involve empirical training but provides convergence rates and step-size strategies: NSD/RMSProp/Adam use a constant step size $\eta=O(1)$ for $O(\ln(1/\epsilon))$ linear convergence when $\epsilon>H_0/H_1$; they require $\eta=O(\epsilon^\tau)$ (NSD) or $\eta=O(\epsilon^{2\tau})$ (RMSProp/Adam) to enter the slow phase when $\epsilon<H_0/H_1$.

## Key Experimental Results

### Main Results (Theoretical Rate Comparison — Phase 1 Rates)

| Problem | Method | Step Size | Convergence Rate | Remarks |
|------|------|------|----------|------|
| Exp. loss (Separable) | GD | const | $\Theta(1/\epsilon)$ | Soudry et al. 2018 |
| Same as above | Sign GD / Norm.GD (NSD) | const $O(1)$ | $O(\ln(1/\epsilon))$ | Ours Theorem 1, **strictly faster** |
| Logistic (Separable) | GD with Armijo | line-search | $O(\ln(1/\epsilon))$ | Vaswani & Harikandeh 2025 |
| Same as above | NSD / Sign CD-GS | const | $O(n^2/\gamma_p^2+\ln(1/(n\epsilon)))$ | Ours Theorem 2 |
| Softmax PG (MAB) | GD | const | $\Omega(1/\epsilon)$ | Mei et al. 2020 |
| Same as above | NSD | const $O(1)$ | $O(\ln(1/\epsilon))$ | Ours Corollary 1 |
| 2-layer net + Separable | RMSProp / Adam (det. diag.) | const + const $\beta$ | $O(\ln(1/\epsilon))$ linear | Ours Theorem 3 |
| 1D Logistic loss | GD / heavy-ball / AdaGrad / AMSGrad | arbitrary | $\omega(\ln(1/\epsilon))$ sublinear lower bound | Ours Sec. 6, separated from RMSProp/Adam |

### Key Findings
- Once $H_0=0$ (e.g., exponential loss, softmax policy gradient), NSD and RMSProp/Adam remain in Phase 1 throughout, achieving linear convergence with a constant step size. Phase 2 (requiring step-size decay) only starts when the objective has a finite minimizer ($H_0>0$), providing theoretical justification for "learning-rate decay in late-stage training."
- A first provable rate separation between RMSProp/Adam and GD/heavy-ball/AdaGrad/AMSGrad is established on 1D logistic loss (Sec. 6), theoretically validating the empirical dominance of RMSProp/Adam over AdaGrad.
- Combining special cases of NSD with Sign CD-GS, the paper proves that even the simple Sign CD-GS algorithm ("select coordinate via Gauss–Southwell + update with sign") achieves linear convergence on logistic regression with separable data, matching the normalized coordinate descent rates of Axiotis & Sviridenko (2023).

## Highlights & Insights
- "Tying smoothness to the function value ($H_0+H_1 f$)" is a subtle but transformative shift: it allows the descent inequality to adapt to the training stage—permitting a large Hessian when the loss is high and tightening as loss decreases—unifying "gradient clipping, warm-up, and decay" under a single mathematical path.
- "Exponential bounds on the step-wise gradient ratio" (Lemma 6/10) is a versatile tool for analyzing adaptive methods (Adam/RMSProp): it transforms the "historical gradient in the denominator," which obstructs traditional proofs, into a controllable multiplicative constant. Any optimizer relying on second-moment buffers (e.g., Lion, Tiger) could potentially reuse this framework.
- The lower bound section (Sec. 6) elevates the question of "why RMSProp/Adam outperforms AdaGrad" from empirical observation to provable separation. This "upper bound + matching lower bound" argumentation template serves as a model for future LLM optimizer comparative studies.

## Limitations & Future Work
- Analysis is restricted to **deterministic diagonal** variants of RMSProp/Adam, omitting stochastic gradients and matrix preconditioners (e.g., Shampoo, Sophia). Extending the step-wise ratio technique to stochastic/full-matrix settings is the natural next step.
- $(H_0,H_1)$-NS requires twice-differentiability, failing to cover non-smooth models like ReLU networks directly; it needs to be weakened to weak convexity or generalized smoothness.
- The NL condition requires $\mu(\theta)$ not to degenerate along the path. For deep over-parameterized networks, $\mu$ practically depends on initialization; quantifying "within what radius NL persists" remains an open problem.
- Assumptions for "2-layer network + smooth leaky ReLU + exponential loss" remain strict; extending to sigmoid/softplus and multilayer networks requires new Hessian control tools.

## Related Work & Insights
- **vs Vaswani & Harikandeh (2025)**: Also based on function-value-based NS, but they only analyze GD + Armijo line-search. This paper generalizes NS to any $(p,q)$ dual norm, covers NSD/Adam/RMSProp, and proves constant step sizes achieve the same rates as their line-search.
- **vs Alimisis et al. (2025)**: Both focus on "Hessian affinely upper-bounded by $f$," which Alimisis uses to explain LR warm-up. This paper uses the condition as a bridge for Adam/RMSProp analysis and adds the lower bound separation in Sec. 6, moving from "explaining phenomena" to "strict rate comparison."
- **vs Li et al. (2023b), Wang et al. (2024a/b)**: These works analyze Adam on $(L_0,L_1)$-NS non-convex functions, requiring bounded gradients and only providing rates for the gradient norm. This paper removes the bounded gradient requirement and provides linear convergence for function values under convex + NL settings, achieving faster deterministic rates in non-convex extensions (Sec. 5.4).
- **vs Mei et al. (2020/2021)**: Mei et al. give an $\Omega(1/\epsilon)$ lower bound for GD on softmax PG and an $O(\ln(1/\epsilon))$ upper bound for Norm.GD. This paper generalizes the latter to any NSD (including Sign GD and Sign CD-GS) within a unified NS+NL framework and proves RMSProp/Adam achieve the same linear rates on these problems.

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
- [\[ICML 2026\] Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](safety_generalization_under_distribution_shift_in_safe_reinforcement_learning_a_.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Two-timescale] This paper establishes the stability and almost sure convergence of general two-timescale stochastic approximation (SA) under Markov noise without relying on any project…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Two-timescale"
  - "stochastic approximation"
  - "ODE method"
  - "Markov noise"
  - "TDC"
date: 2026-05-08
content_hash: 9c1b3e053c77aa6d
---

# Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.31172](https://arxiv.org/abs/2605.31172)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Two-timescale, stochastic approximation, ODE method, Markov noise, TDC

## TL;DR
This paper establishes the stability and almost sure convergence of general two-timescale stochastic approximation (SA) under Markov noise without relying on any projection operators for the first time. Based on this, it provides the first almost sure convergence results for the TDC($\lambda$) algorithm under off-policy linear function approximation.

## Background & Motivation

**Background**: Numerous algorithms in reinforcement learning, such as actor-critic, TDC, and target networks, belong to two-timescale SA: two sets of parameters (fast and slow) are updated with large and small step sizes, respectively. Asymptotically, the fast scale appears to converge under a static slow parameter, forming a stochastic version of "nested loops." Theoretical analysis relies on the ODE method proposed by Borkar (Borkar 1997; Borkar & Meyn 2000), where the core premise is iterate boundedness (stability), which allows discrete iterations to be characterized by ODE trajectories.

**Limitations of Prior Work**: Classical two-timescale convergence conclusions almost exclusively assume (i) i.i.d. noise and (ii) pre-established stability. However, in RL, the noise sequence is a Markov chain (joint state-action-eligibility trace), and eligibility traces can be unbounded in off-policy settings, making i.i.d. and bounded noise assumptions inapplicable. Previous work either resorted to mandatory projection operators (Yu 2017; Panda & Bhatnagar 2025) to artificially guarantee boundedness, or assumed decoupled two-scale parameters and noise falling within compact spaces (Karmakar & Bhatnagar 2021), thus failing to directly cover TDC($\lambda$) with eligibility traces.

**Key Challenge**: In two-scale coupled dynamics, how can one relate the norm of fast-scale iterations to the norm of slow-scale iterations using only the ratio of fast-to-slow step sizes $\beta(n)/\alpha(n)\to 0$ and mild Lipschitz assumptions, thereby proving almost sure convergence in a "fully open" setting with Markov noise, non-compact noise spaces, no projection, and no preset stability? This represents a long-standing gap in the literature (see Table 1).

**Goal**: To remove all the aforementioned restrictions within a unified framework and cover real algorithms such as TDC($\lambda$) and actor-critic.

**Key Insight**: The authors noted that Lakshminarayanan & Bhatnagar (2017) used "rescaled iterations + ODE@$\infty$" to prove two-scale stability under i.i.d. noise, while Liu et al. (2025b) completed a similar argument for single-scale Markov noise. To combine these, a key obstacle arises: existing two-scale proofs require "same-step" control of the fast parameter $\|x_n\|$ by the slow parameter $\|y_n\|$, a synchronous bound that naturally fails during the rescaling process.

**Core Idea**: Relax the requirement to "controlling the current fast parameter $x_n$ with the maximum slow parameter $y_n^{\max}$ observed so far"—specifically using $\|x_n\|\le K(1+\|y_n^{\max}\|)$ as a new bridging inequality (Lemma 3.1). This "running max" formulation aligns perfectly with Lakshminarayanan’s "monotone scaling factor," allowing the two sets of tools to be merged seamlessly.

## Method

This is a SA theory paper rather than an algorithmic one; the "method" refers to a proof system developed around the new Lemma 3.1, which is ultimately applied to the RL algorithm TDC($\lambda$).

### Overall Architecture

The object of study is a general two-timescale recursion (fast scale $x\in\mathbb{R}^{d_1}$, slow scale $y\in\mathbb{R}^{d_2}$):

$$
x_{n+1}=x_n+\alpha(n)\,H(x_n,y_n,W_{n+1}),\quad y_{n+1}=y_n+\beta(n)\,G(x_n,y_n,W_{n+1}),
$$

where the noise $\{W_n\}$ is a Markov chain on space $\mathcal{W}$ (possibly non-compact and uncountable), and step sizes satisfy $\lim_n \beta(n)/\alpha(n)=0$. The proof roadmap is: ① Prove Lemma 3.1 (running-max bridging bound) for the fast scale, ② Prove Theorem 3.2 (global stability) for the slow scale, ③ Prove Theorem 3.3 (almost sure convergence) accordingly, and ④ Verify that TDC($\lambda$) satisfies all assumptions to obtain Theorem 7.2.

### Key Designs

1. **Running-max Bridge (Lemma 3.1)**:
    - **Function**: Links the iteration norms of the two timescales, proving there exists a sample-path-dependent constant $K$ such that $\|x_n\|\le K(1+\|y_n^{\max}\|)$ a.s., where $y_n^{\max}=\arg\max_{i\le n}\|y_i\|$ corresponds to the slow iteration.
    - **Mechanism**: The time axis $[0,\infty)$ is sliced into intervals $[T_n, T_{n+1})$ of length approximately $T$ relative to the fast step size. On each segment, $z=(x,y)$ is rescaled by $r_n\doteq\max\{1,r_{n-1},\|\bar z(T_n)\|\}$ to obtain $\tilde z_n(t)=\bar z(T_n+t)/r_n$. An Arzelà–Ascoli subsequence yields a limit trajectory $z^{\lim}$ satisfying ODE@$\infty$ $\dot x=h_\infty(x,y), \dot y=0$. Lemma 4.7 proves that if $\|\bar x(T_n)\|>C_1(1+\|\bar y(T_n)\|)$, then $\|\bar z(T_{n+1})\|\le\|\bar z(T_n)\|$ (i.e., $r_{n+1}=r_n$). Induction yields $\|\bar z(T_n)\le C_1C_2(\max_{m\le n}\|\bar y(T_m)\|+1)$, and Lemma 4.9 extends this bound to all $n$.
    - **Design Motivation**: Unlike previous works (Kushner & Yin 2003; Mokkadem & Pelletier 2006; Dalal et al. 2018; Yaji & Bhatnagar 2020; Zeng et al. 2024), which attempted to control $\|x_n\|$ using the synchronous slow parameter $\|y_n\|$, such synchronous bounds fail when noise is not i.i.d. The "running max" is "looser" than the current value and is the weakest condition capable of accommodating both Lakshminarayanan's monotone scaling scheme and Liu's Markov noise averaging techniques.

2. **Slow-scale Stability Proof (Theorem 3.2)**:
    - **Function**: Derives global iteration boundedness $\sup_n\|z_n\|<\infty$ a.s. via contradiction based on Lemma 3.1.
    - **Mechanism**: Performs rescaling again at the slow scale, but the scaling factor is changed to $r_n\doteq\max\{1,\|z_{m(T_n)}^{\max}\|\}$ to ensure it is at least of the same order as the historical maximum iteration. Lemma 5.3 is then proved—showing that even at the slow scale, the fast iteration approximately tracks $\lambda_\infty(\tilde y_n(t))$, thereby formalizing the heuristic that "the fast scale acts as if it has converged." Combined with the zero-attractor property of ODE@$\infty$, the assumption $\sup_n r_n=\infty$ contradicts the convergence of the limit ODE to 0; thus $r_n$ is bounded and stability holds.
    - **Design Motivation**: Applying the single-scale argument of Liu et al. (2025b) directly would fail because fast iterations have larger step sizes and grow faster. Defining the scaling factor as the "historical maximum $\|z\|$" ensures that scaling rates for both scales are automatically synchronized, allowing Markov noise averaging estimates (Kushner & Yin 2003 techniques) to remain applicable.

3. **Application to Off-policy TDC($\lambda$) Convergence (Theorem 7.2)**:
    - **Function**: Provides the first proof of almost sure convergence for TDC (without projection, with linear function approximation, off-policy, and with eligibility traces). The algorithm is defined as $e_t=\lambda\gamma\rho_{t-1}e_{t-1}+\phi_t$, $\delta_t=R_{t+1}+\gamma\phi_{t+1}^\top\theta_t-\phi_t^\top\theta_t$, $\nu_{t+1}=\nu_t+\alpha_t(\rho_t\delta_t e_t-\phi_t\phi_t^\top\nu_t)$, $\theta_{t+1}=\theta_t+\beta_t(\rho_t\delta_t e_t-\rho_t(1-\lambda)\gamma\phi_{t+1}e_t^\top\nu_t)$.
    - **Mechanism**: $\nu_t$ is treated as the fast scale and $\theta_t$ as the slow scale, with the state expanded to $(S_t,A_t,e_t)$. Since the product of importance sampling ratios $\rho_t$ in off-policy settings makes $e_t$ potentially unbounded, $\mathcal{W}$ is taken as a non-compact space. Verifying that assumptions B.1–B.7 in Appendix B (Markov chain ergodicity, step-size conditions, homogeneous limits of $H/G$, Lipschitz continuity, averaging conditions) naturally hold for TDC($\lambda$) allows Theorem 3.3 to directly yield almost sure convergence.
    - **Design Motivation**: Previously, Yu (2017) and Panda & Bhatnagar (2025) had to add projections to force boundedness. This non-projection framework is the first tool capable of characterizing practical algorithms "as they are." It also demonstrates the practical power of Lemma 3.1—turning "theoretical non-projection" from a slogan into a verifiable condition.

### Loss & Training
As a theoretical paper, there is no training. Key assumptions to be met include: step sizes $\sum\alpha=\sum\beta=\infty$, $\sum\alpha^2, \sum\beta^2<\infty$, $\lim\beta/\alpha=0$; existence of Lipschitz homogeneous limits $h_\infty, g_\infty$ for $H, G$; existence of a unique globally asymptotically stable equilibrium $\lambda_\infty(y), 0$ for the corresponding ODE; and long-run average regularity conditions (B.7) in the style of Kushner & Yin.

## Key Experimental Results

### Main Results (Comparison of Theoretical Results)

| Work | Scales | Noise | Projection Required | Noise Space | TDC($\lambda$) Coverage |
|------|--------|-------|---------------------|-------------|-------------------------|
| Borkar (2009) | Two | i.i.d. | No | Compact | No |
| Lakshminarayanan & Bhatnagar (2017) | Two | i.i.d. | No | Compact | No |
| Karmakar & Bhatnagar (2021) | Two (Decoupled) | Markov | No | Compact | No |
| Liu et al. (2025b) | Single | Markov | No | Non-compact | — |
| Panda & Bhatnagar (2025) | Two | Markov | **Yes** | Non-compact | No |
| **Ours** | Two (Coupled) | Markov | No | Non-compact | **Yes** |

### Summary of Major Theorems

| Result | Type | Key Statement |
|------|------|----------|
| Lemma 3.1 | Bridging Bound | $\|x_n\|\le K(1+\|y_n^{\max}\|)$ a.s. |
| Theorem 3.2 | Stability | $\sup_n\|z_n\|<\infty$ a.s. |
| Theorem 3.3 | Convergence | $\|z_n-(\lambda(y^*),y^*)\|\to 0$ a.s. |
| Theorem 7.2 | RL Application | TDC($\lambda$) converges a.s. under off-policy linear approximation |

### Key Findings
- Replacing "synchronous control" with "historical maximum slow iteration control" is the weakest condition for merging the methodologies of Lakshminarayanan & Bhatnagar (2017) and Liu et al. (2025b); without it, the two sets of tools are incompatible.
- The paper points out that the argument for almost sure convergence of two-timescale Markov SA in Chandak et al. (2025) is questionable: they use "expectation boundedness $\Rightarrow$ almost sure boundedness," but a counterexample where $x_n=\sqrt n$ with probability $1/n$ and 0 otherwise directly disproves this (Second Borel–Cantelli). This indicates that this work not only fills a gap but also corrects errors.
- Provable convergence results for the "deadly triad" (off-policy + eligibility traces + linear approximation) remain extremely rare; this paper provides the first completely non-projection solution, paving the way for full analysis of actor-critic methods.

## Highlights & Insights
- Using "running max" to bridge two timescales is a minimalist yet sharp technique: rather than requiring $\|x_n\|$ to be suppressed by a synchronous $\|y_n\|$, it allows it to be suppressed by the historical peak $\|y\|$. This is highly consistent with the naturally monotone scale factors in rescaling methods. Transferring this to other two-scale algorithms like actor-critic or target networks only requires re-verifying the homogeneous limits of $H, G$.
- The work pushes the ODE@$\infty$ framework to the limits of "non-compact Markov noise + no projection," providing clear technical interfaces for Lemma 4.7 (growth suppression) and Lemma 5.3 (fast-scale transition to equilibrium), which can be reused in other proofs transitioning from discrete iterations to limit ODEs.
- The literature comparison (Table 1) and the precise rebuttal of the Chandak et al. (2025) counterexample serve as a reminder to researchers in the field: under Markov noise, there is an essential chasm between "expectation boundedness" and "almost sure boundedness" that cannot be conflated.

## Limitations & Future Work
- Only asymptotic a.s. convergence is provided without finite-time rates; existing two-scale $L^2$ rates (Doan 2021a/b/2022; Chandak et al. 2025) have not yet been unified with the a.s. path results of this paper.
- The current theory requires the ODE to have a unique globally asymptotically stable equilibrium; in actor-critic, the policy ODE might have multiple fixed points (e.g., local optima), requiring the conclusions to be weakened to "convergence to an invariant set."
- The discrete Markov chain assumption excludes continuous state space RL; the Markov noise needs to be generalized from general state spaces (Borkar 2009 Chapter 6 framework).
- Regarding formalization, Zhang (2025) has verified single-scale Markov SA in Lean; mechanizing the verification of this two-scale framework is a worthwhile next step.

## Related Work & Insights
- **vs Lakshminarayanan & Bhatnagar (2017)**: Both use two-scale stability + ODE@$\infty$, but they assume i.i.d. noise and rely on synchronous bounds; this paper replaces synchronous bounds with more relaxed historical maximum bounds, extending the noise to Markov and non-compact spaces.
- **vs Karmakar & Bhatnagar (2021)**: They assume decoupled two-scale parameters and compact noise spaces; this paper allows coupling and non-compact Markov noise, thereby covering truly coupled algorithms like TDC and actor-critic.
- **vs Panda & Bhatnagar (2025)**: Both face non-compact spaces with Markov noise, but Panda uses extra projection to force boundedness; this paper replaces projection with Lemma 3.1, avoiding algorithmic modifications and maintaining consistency between theory and practice.
- **vs Chandak et al. (2025)**: Chandak provides $L^2$ rates for two-scale systems with an a.s. convergence claim, but the key step (expectation boundedness ⇒ a.s. boundedness) has a counterexample; this paper provides a more complete a.s. path argument and explicitly points out the error.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning](stochastic_minimum-cost_reach-avoid_reinforcement_learning.md)
- [\[ICML 2026\] Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games](revisiting_regularized_policy_optimization_for_stable_and_efficient_reinforcemen.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICML 2026\] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness](convergence_of_steepest_descent_and_adam_under_non-uniform_smoothness.md)
- [\[NeurIPS 2025\] Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/convergence_theorems_for_entropy-regularized_and_distributional_reinforcement_le.md)

</div>

<!-- RELATED:END -->

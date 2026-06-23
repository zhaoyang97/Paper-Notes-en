---
title: >-
  [Paper Note] Sampling Complexity of TD and PPO in RKHS
description: >-
  [ICLR 2026][learning_theory][RKHS] In the unified function space of Reproducing Kernel Hilbert Spaces (RKHS), this paper decouples and analyzes policy evaluation (kernelized TD critic) and policy improvement (KL-regularized proximal/natural gradient updates). It provides non-asymptotic, instance-adaptive convergence bounds dependent on RKHS entropy and
tags:
  - ICLR 2026
  - learning_theory
  - RKHS
  - PPO/NPG
date: 2026-05-08
content_hash: 90de6b1cf1e7d933
---
# Sampling Complexity of TD and PPO in RKHS

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5gUMhTUDi0](https://openreview.net/forum?id=5gUMhTUDi0)  
**Code**: https://github.com/conniemessi/TD-and-PPO-in-RKHS  
**Area**: Learning Theory / Reinforcement Learning Theory  
**Keywords**: RKHS, Temporal Difference, PPO/NPG, Sample Complexity, Non-asymptotic bounds

## TL;DR
In the unified function space of Reproducing Kernel Hilbert Spaces (RKHS), this paper decouples and analyzes policy evaluation (kernelized TD critic) and policy improvement (KL-regularized proximal/natural gradient updates). It provides non-asymptotic, instance-adaptive convergence bounds dependent on RKHS entropy and derives per-iteration sampling rules required to guarantee $O(k^{-1/2})$ convergence, validating the theoretically predicted step-size scheduling on CartPole, Acrobot, and HalfCheetah.

## Background & Motivation

**Background**: Policy gradient and trust region methods (NPG, TRPO, PPO, actor-critic) combined with TD critics are the primary tools for modern continuous control RL, as they are compatible with high-capacity function approximators and provide stable improvement steps.

**Limitations of Prior Work**: Despite empirical success, global convergence theory for such algorithms under "rich function approximation" remains fragmented. Most existing analyses hold only for tabular/linear cases or rely on heavy realizability / concentrability conditions. When using non-linear critics, guarantees often assume the value/advantage functions are "exactly" estimated. More importantly, many policy improvement bounds treat policy expectation terms as having "no sampling noise," thus **the amount of data required for a single improvement step has never been clearly elucidated**.

**Key Challenge**: A coupling exists between the statistical error of policy evaluation (how accurately the critic is learned) and the step size/sample volume of policy improvement—less accurate evaluation necessitates more conservative steps or more samples. Previous theories separate the two, either assuming exact evaluation or failing to specify per-iteration data requirements. Meanwhile, different function classes (tabular, Sobolev, Gaussian kernels, wide neural networks via NTK) have fragmented analyses, lacking a unified framework.

**Goal**: (i) Design a policy evaluation algorithm under rich function classes with controllable statistical errors; (ii) couple this evaluation step with a provable improvement step toward the optimal policy and **explicitly quantify the required samples per iteration**.

**Key Insight**: The authors adopt a "function space" perspective, directly optimizing policies in RKHS. The advantage of RKHS is its ability to represent tabular, linear, Sobolev, Gaussian kernels, and wide neural networks (via their NTK) using a unified language of entropy/covering numbers. The dynamics of kernel gradient descent mirror the NTK dynamics of wide networks trained via gradient descent; hence, RKHS results can be directly transferred to neural critics/actors.

**Core Idea**: Integrate a "kernelized TD estimator" (using implicit preconditioning for geometric convergence, avoiding matrix inversion) with "KL-regularized proximal policy updates" (obtaining NPG/PPO-style updates by exponentiating the evaluation value) in RKHS. Convergence rates are characterized using RKHS entropy, which in turn determines the required per-iteration sampling volume.

## Method

### Overall Architecture

Consider an MDP $(S,A,P,r,\gamma)$, assuming the target Q-function $Q^\pi$ resides in an RKHS $\mathcal{H}(S\times A)$ induced by a symmetric positive definite kernel $K$. The algorithm utilizes an outer-loop policy iteration (NPG in RKHS, Algorithm 1): in each outer iteration $k$, the current policy $\pi_k$ collects $n^{(k)}$ one-step transition quadruplets $(s_0,a_0,s_1,a_1)$. Then, **kernelized TD** iterates $T^{(k)}$ steps in RKHS to obtain a critic $f^{(k)}\approx Q^{(k)}$. Finally, a single **KL-regularized proximal update** advances the policy to $\pi_{k+1}\propto \pi_k\exp\{\Delta_k f^{(k)}\}$. The contribution is establishing non-asymptotic bounds for these two steps and linking them to prove a global convergence rate of $O(k^{-1/2})$, while providing the required sampling rules to achieve this rate.

As a theoretical analysis paper focusing on error decomposition and policy improvement inequalities, it does not utilize a multi-module engineering pipeline; the core resides in the bounds and their coupling relationships.

### Key Designs

**1. Kernelized TD critic: Implicit preconditioning via RKHS functional gradients to avoid cubic matrix inversion**

Policy evaluation is formulated as a fixed-point kernel ridge regression (KRR) in RKHS:
$$\hat{Q}^\pi=\min_{f\in\mathcal{H}}\frac{1}{n}\sum_{i=1}^n\big(f(\omega_0^{(i)})-r(\omega_0^{(i)})-\gamma\hat{Q}^\pi(\omega_1^{(i)})\big)^2+\lambda\|f\|_{\mathcal{H}}^2,$$
where $\omega=(s,a)$. By the Representer Theorem, it has a closed-form solution $\hat{Q}^\pi=K(\cdot,\omega_0)b^\pi$, where $b^\pi=[K+\lambda nI-\gamma C]^{-1}r$. However, direct inversion is cubic. The authors treat this as functional optimization in RKHS, using kernel gradient descent:
$$f_{t+1}=(1-\alpha_t)f_t-\eta_t\sum_{i=1}^n\big(f_t(\omega_0^{(i)})-r(\omega_0^{(i)})-\gamma f_t(\omega_1^{(i)})\big)K(\omega_0^{(i)},\cdot).$$
Crucially, using the RKHS inner product $\langle\cdot,\cdot\rangle_{\mathcal{H}}$ instead of the $\ell_2$ inner product on $\mathbb{R}^n$ changes the gradient and Hessian, **equating to an implicit preconditioner** that directs iterations toward favorable directions. With proper constant weight decay $\alpha$ and step size $\eta$, the error converges **geometrically (exponentially)** to $\hat{Q}^\pi$ without requiring matrix inversion.

**2. Non-asymptotic TD error bounds dependent on RKHS entropy: A unified bound for tabular / Sobolev / NTK / Gaussian**

The authors characterize the statistical error of the critic using empirical processes, where RKHS complexity is measured by entropy: the entropy of the unit ball satisfies $H(\delta,\|\cdot\|_{L^\infty},\mathcal{B})\le C\delta^{-2\beta}|\log\delta|^{2\kappa}$ for $\beta\in[0,1)$ and $\kappa\ge 0$. Theorem 10 provides the non-asymptotic TD error:
$$\sqrt{\frac{1}{n}\sum_{i=1}^n\big(f_t(\omega_0^{(i)})-Q^\pi(\omega_0^{(i)})\big)^2}\le O_p\Big((1-c\gamma)^{-\frac{2+\beta}{2+2\beta}}\,n^{-\frac{1}{2+2\beta}}\,|\log n|^{\frac{\kappa}{1+\beta}}\Big)\|Q^\pi\|_{\mathcal{H}}.$$
A single entropy assumption links four function space classes, demonstrating the "instance-adaptive" nature of the work: rates collapse to $n^{-1/2}$ for tabular/Gaussian and specific non-parametric rates for Sobolev and NTK.

**3. KL-regularized proximal update: RKHS-version NPG by exponentiating evaluations with explicit sampling rules**

Policy improvement uses the following update:
$$\pi_{k+1}=\arg\max_{\int_A\pi=1,\pi\ge0}\mathbb{E}_n\Big[\Delta_k\int_A f^{(k)}(s,a)\pi(a|s)da-\mathrm{KL}\big(\pi(\cdot|s)\,\|\,\pi_k(\cdot|s)\big)\Big].$$
The solution is $\pi_{k+1}\propto\pi_k\exp\{\Delta_k f^{(k)}\}$, equivalent to natural policy gradient, termed **explicit PPO (NPG)**. This generalizes NPG to general RKHS functions where policies can be continuous functions of $(s,a)$. Its most significant contribution is **explicitly quantifying the sample volume required to guarantee improvement**.

**4. Evaluation-Improvement Coupling: Mirror-descent inequality + sampling table for $O(k^{-1/2})$ global convergence**

Measuring performance via expected total return $R[\pi]$, the authors adapt mirror descent analysis to obtain the core inequality (Theorem 13):
$$\inf_k\big(R[\pi^*]-R[\pi_k]\big)\le\frac{\big(\sum_k 2\Delta_k\|f^{(k)}-Q^{(k)}\|_{L^\infty}\big)+\big(\sum_k\Delta_k^2(1-\gamma)^{-1}\|r\|_{L^\infty}\big)+\mathbb{E}_{S\sim\nu^*}\mathrm{KL}(\pi^*\|\pi_0)}{\sum_k\Delta_k}.$$
Global convergence $O(k^{-1/2})$ is achieved by setting $\Delta_k=1/\sqrt{k}$ and following the sampling rules in Table 1 for different kernels. This reveals that the required samples increase as iterations $k$ and policy complexity $\|\pi_k\|_{\mathcal{H}}$ grow.

### Loss & Training

The theoretical algorithm follows a two-stage "Evaluation for $T$ steps + Improvement for one step" structure. In practice, a $T$ that is too small (e.g., $T=4$) might lead to insufficient critic learning and unstable actor updates. Consequently, the implementation utilizes **joint optimization**: the critic's TD-error loss and the actor's NPG objective are combined into a unified loss, minimized together for $T$ epochs per batch. The model uses a two-layer MLP where actor and critic share parameters, and the policy is $\pi_\theta\propto\mathrm{Softmax}(f_\theta)$.

## Key Experimental Results

### Main Results: Step-size Scheduling and Sampling Efficiency

| Environment | Verification Point | Key Observation |
|------|--------|----------|
| CartPole-v1 | Step-size exponent $\alpha$ | $\Delta_k=k^{-0.5}$ converges; $k^{-0.2}$ diverges; $k^{-1.5}$ plateaus. |
| Acrobot-v1 | Step-size exponent $\alpha$ | Same trend; $k^{-0.5}$ shows slight drops at late stages (policy complexity vs. sample scarcity). |
| HalfCheetah-v5 | Scalability | Maintains reasonable runtime; $k^{-0.5}$ remains optimal. |

Results consistently confirm Corollary 14: only the $\Delta_k=1/\sqrt{k}$ schedule achieves both convergence and stability.

### Sampling/Computational Efficiency (CartPole-v1, vs. clip-PPO)

| Method | Data Requirement | Performance | Throughput (steps/s) |
|------|----------|------|------|
| clip-PPO (GAE) | Requires full trajectories | Slower convergence, higher variance | 5406.0 |
| Ours NPG (one-step TD) | Only $(s_0,a_0,r_0,s_1,a_1)$ | Stable, reaches max reward 500 | 7210.1 |

The critic learns via single-step TD-error, which is more lightweight than GAE's recursive advantage calculations, yielding ~33% higher throughput.

### Key Findings
- **Step-size scheduling is the decisive factor for NPG+TD stability**: The theoretical $k^{-1/2}$ is a "critical exponent" verified across environments.
- **One-step TD critic outperforms GAE**: Removing dependence on full trajectories improves both sampling efficiency and computational throughput.
- **Sample demand growing with policy complexity is a real constraint**: When the network grows complex and samples are insufficient, performance drops, matching theoretical predictions of $n(k)$ growing with $\|\pi_k\|_{\mathcal{H}}$.

## Highlights & Insights
- **Unifying four function spaces via RKHS entropy**: Tabular, Sobolev, NTK, and Gaussian kernels are all unified under one framework of minimax optimal rates.
- **Functional gradient as implicit preconditioning**: Performing gradient descent under the RKHS inner product automatically provides preconditioning and geometric convergence.
- **Explicit sample requirement formulas**: Theorem 13 explicitly decomposes global suboptimality into evaluation error, step-size drift, and initial KL, allowing for the derivation of a sampling schedule.
- **NTK bridging theory and practice**: Since kernel gradient descent mirrors the NTK dynamics of wide networks, RKHS conclusions directly guide neural critic/actor design.

## Limitations & Future Work
- **Gap between $L^2$ and $L^\infty$ metrics**: Convergence analysis is established under $L^2$, but RL theory often requires $L^\infty$; bridging this gap remains future work.
- **Difficulty in estimating RKHS norm $\|\pi_k\|_{\mathcal{H}}$**: Particularly for NTK where policies correspond to deep networks, making the sampling table difficult to execute precisely in practice.
- **Dependency on sampling quality (Assumption 1 & 6)**: Requires bounded initial distributions and transitions that do not deviate far from the initial distribution.
- **Gap between theoretical two-stage and joint implementation**: The implementation uses a combined loss, creating a slight discrepancy with the theoretical analysis of sequential stages.

## Related Work & Insights
- **vs. Neural TD (Cai et al., 2019)**: They provide finite-sample analysis for neural TD with sub-linear rates; this paper achieves geometric convergence and minimax rates while covering continuous actions.
- **vs. Mirror Descent for Neural Policies (Liu et al., 2019)**: They analyze neural policies in the NTK regime for discrete actions; this paper generalizes NPG to general RKHS and continuous actions with explicit sampling counts.
- **vs. Kernel LSTD (Duan et al., 2024)**: They analyze kernel LSTD for Markov chains; this work utilizes **gradient-based** kernel TD for policy evaluation and couples it with policy improvement.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unified four function spaces via RKHS entropy and explicitly coupled evaluation error with sample complexity.
- Experimental Thoroughness: ⭐⭐⭐ Standard control tasks verify trends, but experiments are light for a theoretical paper and do not include large-scale benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from theorems to sampling tables; however, notation is dense.
- Value: ⭐⭐⭐⭐ Provides a firmer theoretical foundation for PPO/NPG under rich function approximation with actionable sampling/step-size rules.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] 为什么高秩神经网络也能泛化？：基于 RKHS 的代数框架](why_high-rank_neural_networks_generalize_an_algebraic_framework_with_rkhss.md)
- [\[ICLR 2026\] Complexity Analysis of Normalizing Constant Estimation: from Jarzynski Equality to Annealed Importance Sampling and Beyond](complexity_analysis_of_normalizing_constant_estimation_from_jarzynski_equality_t.md)
- [\[ICLR 2026\] Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](near-optimal_sample_complexity_bounds_for_constrained_average-reward_mdps.md)
- [\[ICLR 2026\] Mitigating the Curse of Detail: Scaling Arguments for Feature Learning and Sample Complexity](mitigating_the_curse_of_detail_scaling_arguments_for_feature_learning_and_sample.md)
- [\[ICLR 2026\] Stable Coresets: Unleashing the Power of Uniform Sampling](stable_coresets_unleashing_the_power_of_uniform_sampling.md)

</div>

<!-- RELATED:END -->

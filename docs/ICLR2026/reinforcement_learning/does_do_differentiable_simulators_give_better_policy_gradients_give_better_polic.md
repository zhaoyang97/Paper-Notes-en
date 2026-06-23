---
title: >-
  [Paper Note] Does “Do Differentiable Simulators Give Better Policy Gradients?” Give Better Policy Gradients?
description: >-
  [ICLR 2026][Reinforcement Learning][Paper Note] This work is a "revisitation" of the identically titled paper by Suh et al. (2022). The authors replace the original REINFORCE-based discontinuity detection with a lightweight statistical test (DDCG) that depends only on function values and gradient variances, robustly reproducing and improving the original method with
tags:
  - ICLR 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: 9108ebe9e3fdc433
---
# Does “Do Differentiable Simulators Give Better Policy Gradients?” Give Better Policy Gradients?

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=RUzxUqTpzW](https://openreview.net/forum?id=RUzxUqTpzW)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / Policy Gradient / Differentiable Simulation  
**Keywords**: Policy Gradient, Differentiable Simulation, Gradient Estimation, Inverse Variance Weighting, Discontinuity Detection

## TL;DR
This work is a "revisitation" of the identically titled paper by Suh et al. (2022). The authors replace the original REINFORCE-based discontinuity detection with a lightweight statistical test (DDCG) that depends only on function values and gradient variances, robustly reproducing and improving the original method with a single hyperparameter. More importantly, they propose Step-wise Inverse Variance Weighting (IVW-H), which outperforms GIPPO on MuJoCo control tasks without any discontinuity detection. This demonstrates that while "estimator switching" is useful in controlled studies, the **real bottleneck in practical robot control is often variance rather than "empirical bias."**

## Background & Motivation
**Background**: The core of policy gradient reinforcement learning is estimating the gradient of the expected return with respect to policy parameters $\hat g \approx \frac{d}{d\theta}\mathbb{E}_{p(\tau)}[R(\tau)]$. When the environment is a black box, a zeroth-order estimator (REINFORCE / Likelihood Ratio) is used; it is unbiased but suffers from extremely high variance and poor sample efficiency. When a differentiable simulator is available, a first-order estimator (Reparameterization / Path Derivative) can be used, which typically has much lower variance and faster convergence.

**Limitations of Prior Work**: Real systems are full of non-smooth effects like contact and friction, which create **discontinuities** that cause bias in first-order gradients. A natural compromise is a linear mixture $\hat g_\alpha = \alpha\hat g_1 + (1-\alpha)\hat g_0$, using Inverse Variance Weighting (IVW) to select $\alpha$: $\alpha_{\mathrm{opt}} = \frac{V[\hat g_0]}{V[\hat g_0]+V[\hat g_1]}$. However, IVW fails in discontinuous scenarios—Suh et al. (2022) identified an "empirical bias" phenomenon: first-order gradients may **appear low-variance but be severely inaccurate** in finite samples, causing IVW to assign excessively high weights to these corrupted gradients.

**Key Challenge**: The root of empirical bias is "heavy tails / rare large gradients." Taking a Sigmoid $\frac{1}{1+\exp(-x/T)}$ as an example, when the temperature $T$ is very small, the function is approximately discontinuous in a narrow transition zone. Extremely large gradients occur with minimal probability—true variance is massive (effectively "infinite" in the limit), but finite samples almost never capture these rare events, leading the **empirical variance to systematically underestimate the true error**. Suh et al.'s AoBG detects this bias by constructing confidence intervals around the REINFORCE estimator, but REINFORCE itself is extremely noisy, resulting in wide intervals, low sample efficiency, and the need to tune a per-task threshold $\gamma$ across a vast range ($\gamma \in [5\times10^{-3},\, 10^8]$ in experiments).

**Goal**: To answer two specific questions: (1) Is empirical bias the primary obstacle to actual performance? (2) Is a minimal fix sufficient?

**Key Insight**: The authors re-examine all experiments from AoBG to expose its fragile dependence on hyperparameters while questioning "what happens if we don't explicitly detect discontinuities but simply perform robust variance control."

**Core Idea**: Two complementary "minimal fixes"—using a more sample-efficient statistical test for discontinuity gating (DDCG), and pushing inverse variance weighting down to each time step and action dimension (IVW-H). Based on these, the authors conclude whether "switching vs. variance control" dominates in different scenarios.

## Method

### Overall Architecture
The paper revolves around a hybrid estimator $\hat g_\alpha = \alpha\hat g_1 + (1-\alpha)\hat g_0$. The input is a batch of trajectory samples (containing zeroth-order estimates $\hat g_0$ and first-order estimates $\hat g_1$), and the output is a composite gradient gated by trust levels. The authors modify it from two angles, corresponding to the two main lines of the paper:

The first line addresses "whether to trust the first-order gradient." Instead of using the noisy REINFORCE term to construct confidence intervals like the original AoBG, the authors implement a lightweight test (DDCG) using only **function value variance + gradient variance**. If the test passes, the IVW weight $\hat\alpha_{\mathrm{opt}}$ is used to enjoy low first-order variance; if it fails, the estimator falls back to pure zeroth-order ($\alpha=0$).

The second line focuses on "proper variance control." The authors move inverse variance weighting from the "entire trajectory/parameter space" down to "each time step $t$ and each action dimension $a$" (IVW-H). It estimates variance using samples from parallel actors at fixed time steps, requiring no additional simulator calls.

These two lines are tested in two types of experiments: DDCG is used on the explicitly discontinuous differentiable simulation tasks from Suh et al. (2022), while IVW-H is used on MuJoCo-style continuous control benchmarks. The core argument is the conclusion drawn from the comparison of these experiments: switching is useful when discontinuities are controllable, but variance control is more critical in practical control.

### Key Designs

**1. DDCG Discontinuity Test: Replacing REINFORCE Confidence Intervals with Function Value and Gradient Variance**

For IVW to be reliable, two assumptions must hold simultaneously: (A1) **Reliable Variance**—the empirical variance of the first-order gradient is close to its true variance so that IVW weights are meaningful; (A2) **Local Smoothness**—$f$ is locally approximately quadratic so that the first-order gradient is both accurate and low-variance. The pain point of AoBG is its reliance on the REINFORCE $\frac{d\log p(\tau;\theta)}{d\theta}$ term, which is extremely noisy. DDCG replaces the criterion with an inequality depending only on function value variance and gradient variance.

Specifically, an upper bound $\varepsilon_v$ on the empirical gradient variance $\hat v = \frac{1}{N-1}\sum_i \lVert \nabla f(x_i) - \overline{\nabla f}\rVert_2^2$ is set based on (A1) (derived using chi-squared confidence intervals), and a lower floor is set for $\hat v$ to avoid underestimating variance and over-trusting the first-order estimate. Then, assuming gradient changes satisfy a Lipschitz-like condition $\lVert\nabla f(x)-\nabla f(y)\rVert \approx L\lVert x-y\rVert$ based on (A2), the "amount of gradient variance a local quadratic model should induce" is used as a reference to obtain the criterion (derived in Appendix C):

$$\hat v + \varepsilon_v \;\overset{?}{\ge}\; \frac{2(1-c)\,V[f(x)]}{\sigma^2} - 2\lVert\nabla f\rVert^2.$$

The right side is the gradient variance expected from a local quadratic model under Gaussian smoothing (with the $\lVert\nabla f\rVert^2$ term subtracting the mean gradient contribution). The intuition is: if $f$ is smooth and the variance estimate is reliable, the empirical gradient variance will not be significantly smaller than this quadratic proxy—large fluctuations in function values must imply non-trivial fluctuations in gradients. If the left side falls below the right, it indicates heavy-tailed or discontinuous behavior making IVW weights unreliable, and the system falls back to the zeroth-order estimate. All quantities are calculated from the same sample batch (using $\hat V[f(x)]$ for $V[f(x)]$). The advantage is that for toy tasks, its bound estimation is approximately $d$ times more sample-efficient than AoBG (where $d$ is the dimension, Appendix D).

**2. Adaptive Fallback and Single Hyperparameter $c$ for DDCG**

With the test, the composite weight is defined as a binary choice:

$$\hat\alpha := \begin{cases}\hat\alpha_{\mathrm{opt}} & \text{if Eq. (14) holds}\\[2pt] 0 & \text{otherwise}\end{cases}$$

Passing the test means trusting IVW; failing means falling back to pure 0-order to avoid corrupted 1-order gradients. Beyond the confidence level $\delta$, the method relies on only one meaningful hyperparameter $c$—which relaxes the requirement that "$f$ must be strictly quadratic." When $f$ is exactly quadratic, $c=0$ makes the inequality hold exactly. Larger $c$ allows $f$ to deviate more from quadratic, tolerating more nonlinearity or slight discontinuities. This is the core advantage of DDCG over AoBG: while AoBG's $\gamma$ must be tuned per task between $[5\times10^{-3}, 10^8]$, DDCG fixes $c=0.3$ throughout, and Appendix H shows that any $c\in[0.1,0.9]$ yields nearly identical performance across all tasks and remains reliable even with small samples.

**3. IVW-H: Step-wise, Per-action Inverse Variance Weighting**

This branch performs no discontinuity detection and simply focuses on robust variance control. Let $t\in\{0,\dots,H-1\}$ index the time step, $n$ index parallel actors, and $a$ index the action dimension. For each $(t,n)$, zeroth-order and first-order gradient vectors $\hat g_{0,t,n}$ and $\hat g_{1,t,n}$ are taken. **At a fixed $t$ across actors**, empirical variances $\hat v_{0,t,a}=\hat V_n[\hat g_{0,t,n,a}]$ and $\hat v_{1,t,a}=\hat V_n[\hat g_{1,t,n,a}]$ are estimated per element to provide step-wise, dimension-wise weights:

$$\hat\alpha_{t,a} = \frac{\hat v_{0,t,a}}{\hat v_{0,t,a}+\hat v_{1,t,a}} \in [0,1],$$

The gradients are synthesized element-wise $\hat g_{\alpha,t,n,a}=\hat\alpha_{t,a}\,\hat g_{1,t,n,a}+(1-\hat\alpha_{t,a})\,\hat g_{0,t,n,a}$, and this composite gradient in action space is backpropagated through the policy network. This corresponds to the "aggregate in action space, then backpropagate" logic of Total Propagation X (TPX). While full TPX is theoretically stronger, it is difficult to implement efficiently due to simulator details; IVW-H is its pragmatic approximation. Crucially, the batch of parallel actors at each time step **naturally provides the sample dimension needed to estimate variance without any additional simulator calls**, making IVW-H’s wall-clock time comparable to a pure first-order baseline—contrasting with GIPPO’s parameter-space IVW, which requires extra simulator evaluations and is observed to be both slower and less effective.

## Key Experimental Results

### Main Results
The paper is divided into two parts: Part I compares DDCG and AoBG on explicitly discontinuous tasks; Part II compares IVW-H with GIPPO and others on MuJoCo-style continuous control benchmarks.

| Experiment / Task | Setting | Key Findings |
| :--- | :--- | :--- |
| Ball with Wall ($N=1000$) | Discontinuity via collision | IVW is biased near collisions due to over-trusting $\hat g_1$; AoBG ($\gamma=0.005$) and DDCG detect this and reduce $\alpha$. |
| Ball with Wall ($N=10$, small sample) | Same as above | AoBG with fixed $\gamma$ is too conservative, $\alpha\to 0$ wastes info; DDCG still detects robustly with the same parameter. |
| Pushing (Soft/Hard collision) | Diff. spring constants $k$ | Both favor $\hat g_1$ in soft collisions; AoBG fails to converge fast with small samples; in hard collisions, both approach IVW (stiffness causes variance, not bias). |
| Friction ($N=100$) | Coulomb friction jump | $\hat g_1$ and IVW stall after threshold; AoBG ($\gamma=30000$) and DDCG switch to $\hat g_0$; AoBG degrades without $\gamma$ retuning in small samples, DDCG remains robust. |
| Tennis ($d=21, H=200$) | Racket-ball collision | $\hat g_1$ and IVW stall; AoBG ($\gamma=1000$) and DDCG ($c=0.3$) detect non-smoothness and continue improving after fallback; **final performance is identical**. |
| MuJoCo Ant (contact_ke to $4\times10^5$, 10×) | Enhanced stiffness | **IVW-H achieves highest return**; AoBG cannot outperform IVW even with tuning; IVW is comparable to GIPPO and better than pure $\hat g_1/\hat g_0$. |
| MuJoCo Hopper (contact_ke to $10^6$, 50×) | Enhanced stiffness | $\hat g_0$ outperforms pure $\hat g_1$; **GIPPO fails**; AoBG and IVW perform well, **IVW-H further exceeds IVW**. |
| MuJoCo CartPole | GIPPO Hyperparameters | $\hat g_0$ performs worst; $\hat g_1$/AoBG/IVW/IVW-H/GIPPO converge to similar returns. |

### Key Findings
- **The value of DDCG is "Robustness + Fewer Hyperparameters" rather than beating SOTA**: On most explicitly discontinuous tasks, DDCG and AoBG achieve similar or identical final performance, but AoBG requires per-task tuning of $\gamma$ across $[5\times10^{-3}, 10^8]$ and degrades in small samples. DDCG maintains $c=0.3$ and remains robust across $c\in[0.1,0.9]$.
- **"Empirical bias is not the primary bottleneck in practical control"**: On MuJoCo tasks, AoBG cannot outperform simple IVW even with tuning (Fig 16, Appendix I), indicating bias is not dominant; robust variance control via step-wise IVW-H is sufficient to beat complex baselines like GIPPO.
- **GIPPO fails under strong contact**: GIPPO fails optimization in Hopper with 50× contact stiffness, while IVW-H improves steadily, suggesting parameter-space IVW is fragile and computationally expensive.

## Highlights & Insights
- **Reducing the cost of "discontinuity detection" from REINFORCE to function/gradient variance**: DDCG's insight is that since REINFORCE noise causes AoBG's wide intervals, one should use a "quadratic proxy" of function fluctuations and gradient variance. This saves approximately $d$ times the samples and can be generalized to other heavy-tail detection scenarios.
- **Step-wise, action-wise IVW is a cheap and effective trick**: IVW-H reuses parallel actor samples at fixed $t$ for variance estimation, incurring zero extra simulator overhead and matching the wall-clock time of pure first-order gradients while consistently outperforming global IVW and GIPPO. This is a low-cost, reusable implementation tip for anyone using differentiable simulation.
- **The meta-position of the paper is the most striking**: The title itself questions whether differentiable simulators provide better gradients, and the conclusion is "it depends"—estimator switching is useful for controlled experiments, but variance control is more vital for actual deployment. It reminds the community that many failures attributed to "bias" may vanish with a robust variance control implementation.

## Limitations & Future Work
- **Limitations acknowledged by the authors**: The task set remains small (toy differentiable simulation + three MuJoCo tasks), and the diagnostic doesn't fully characterize "when bias mechanisms truly become essential." Expanding tasks and deepening diagnostics is left for future work.
- **Boundary of conclusions**: The finding that "variance is the bottleneck, not bias" is derived from their MuJoCo setup (with artificially increased stiffness). Whether this generalizes to real robot tasks with more violent contact, higher dimensions, or longer horizons is uncertain. DDCG and IVW-H were rarely used together (DDCG for Part I, IVW-H for Part II); synergies were not verified.
- **Assumptions of the test**: DDCG's criterion relies on "locally quadratic + Lipschitz-like" assumptions. While the paper uses $c$ to relax this, the reliability of the test in highly non-quadratic landscapes and the specific calculation of $\varepsilon_v$ still rely on standard statistical approximations (see Appendix B/C).

## Related Work & Insights
- **vs. AoBG (Suh et al. 2022)**: Both construct "statistical estimates of bias" for fallback, but AoBG uses noisy REINFORCE terms and requires wide $\gamma$ tuning; DDCG uses function/gradient variance with a single $c$, is robust in small samples, and saves $d$ times the samples. This work systematically replicates all AoBG tasks.
- **vs. IVW / Total Propagation (Parmas et al. 2018, 2023)**: Classic IVW/TP performs mixing globally or in parameter space; IVW-H is a pragmatic step-wise approximation of TPX, performing estimation per step/action dimension with zero extra overhead.
- **vs. GIPPO (Son et al. 2023)**: GIPPO uses an $\alpha$-policy to downweight unreliable analytic gradients within PPO, but its parameter-space IVW requires extra evaluation and fails under high contact; IVW-H works in action space and is faster and more stable.
- **vs. SHAC / AGPO / APG**: These handle non-smoothness via truncated rollouts, batch gradient variance weighting, or simulator derivatives. Ours contributes by clarifying the dominance of variance control over bias detection in different scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a completely new framework, but a refined modification and "revisitation" of existing hybrid gradient methods. DDCG's criterion and IVW-H's implementation are solid.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete replication of AoBG + three MuJoCo tasks with small-sample and stiffness controls; sufficient to support conclusions, though task scale is a bit small.
- Writing Quality: ⭐⭐⭐⭐ Clear problem decomposition, coherent bias-variance argument, and a clever title; some criterion details require checking the appendix.
- Value: ⭐⭐⭐⭐ Provides direct guidance for researchers using differentiable simulators (IVW-H is almost "free") and corrects the misconception that "empirical bias" is always the main bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models](rest_reshaping_token-level_policy_gradients_for_tool-use_large_language_models.md)
- [\[ICLR 2026\] Beyond Softmax and Entropy: Convergence Rates of Policy Gradients with $f$-SoftArgmax Parameterization & Coupled Regularization](beyond_softmax_and_entropy_convergence_rates_of_policy_gradients_with_boldsymbol.md)
- [\[ICLR 2026\] Distributional value gradients for stochastic environments](distributional_value_gradients_for_stochastic_environments.md)
- [\[AAAI 2026\] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](../../AAAI2026/reinforcement_learning/diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)
- [\[ICML 2026\] Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation](../../ICML2026/reinforcement_learning/randomized_advantage_transformation_rat_computing_natural_policy_gradients_via_d.md)

</div>

<!-- RELATED:END -->

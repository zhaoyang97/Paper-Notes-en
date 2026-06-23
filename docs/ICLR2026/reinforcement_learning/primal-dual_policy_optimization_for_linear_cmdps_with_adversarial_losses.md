---
title: >-
  [Paper Note] Primal-Dual Policy Optimization for Linear CMDPs with Adversarial Losses
description: >-
  [ICLR 2026][Reinforcement Learning][Paper Note] This paper proposes the first primal-dual policy optimization algorithm for finite-horizon linear CMDPs with adversarial losses and stochastic costs. By employing a novel "weighted LogSumExp softmax policy" combined with periodic policy mixing and regularized dual updates, it achieves sublinear regret and constraint vi
tags:
  - ICLR 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: a0c7082ff0519f25
---
# Primal-Dual Policy Optimization for Linear CMDPs with Adversarial Losses

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=baP3Dw7bWO](https://openreview.net/forum?id=baP3Dw7bWO)  
**Code**: Not released  
**Area**: Reinforcement Learning / Safe RL / Constrained MDP Theory  
**Keywords**: Linear CMDP, Adversarial Losses, Primal-Dual, Policy Optimization, Covering Number

## TL;DR
This paper proposes the first primal-dual policy optimization algorithm for finite-horizon linear CMDPs with adversarial losses and stochastic costs. By employing a novel "weighted LogSumExp softmax policy" combined with periodic policy mixing and regularized dual updates, it achieves sublinear regret and constraint violation (both $\tilde{O}(K^{3/4})$) while controlling the covering number of the policy class and the dual variables.

## Background & Motivation

**Background**: Safe Reinforcement Learning often employs Constrained MDPs (CMDP), where an agent minimizes cumulative losses while ensuring cumulative costs stay within a budget. To handle large state spaces, "Linear CMDPs" with linear function approximation have been introduced, assuming transition kernels, losses, and costs are linear with respect to features $\phi(s,a)\in\mathbb{R}^d$, making complexity independent of the state count $|S|$.

**Limitations of Prior Work**: Existing linear CMDP studies (Ghosh et al. 2022/2024, Kitamura et al. 2025) assume a **stochastic environment** where losses and costs are either fixed or sampled from fixed distributions. However, in autonomous driving or service robotics, loss signals can fluctuate due to road conditions or task switches, making "fixed loss" assumptions unrealistic. Conversely, works handling adversarial losses (Qiu et al. 2020, etc.) are restricted to the **tabular setting**, which fails to scale to large state spaces.

**Key Challenge**: Applying standard primal-dual policy optimization to adversarial linear CMDPs faces a technical deadlock. To keep dual variables within a compact range (necessary for analysis), "policy mixing" is typically used. However, this additive mixing **breaks the recursive structure of policy optimization**, meaning the resulting policy is no longer a standard softmax. This makes the **covering number of the value function class** impossible to analyze with current tools, preventing the proof of high-probability good events.

**Goal**: Design a primal-dual policy optimization algorithm for finite-horizon linear CMDPs with "adversarial losses (full information) + stochastic costs (bandit feedback)" that achieves sublinear regret and constraint violation.

**Key Insight**: The authors analyze the new policy class induced by mixing and realize that "mixing frequency" is a lever affecting both covering numbers and dual variables. Frequent mixing helps control dual variables but increases the covering number; sparse mixing does the opposite. By finding an optimal mixing rhythm combined with a dual update capable of drift analysis under sparse mixing, the problem becomes solvable.

**Core Idea**: Use a "weighted LogSumExp softmax policy + periodic mixing + regularized dual update" trinity to suppress both the covering number and dual variables to the magnitudes required for sublinear bounds.

## Method

### Overall Architecture

The algorithm runs online over episodes $k=1,\dots,K$ in a finite-horizon CMDP. At the start of each episode, the adversary chooses a loss $f^k$ (full information feedback provided after interaction), while cost $g^k$ is i.i.d. sampled with only bandit feedback. The algorithm maintains a **primal variable (policy $\hat\pi^k$)** and a **dual variable $Y_k$** (Lagrangian multiplier), updated via alternating primal-dual mirror descent. The workflow consists of four components: epoch initialization (triggered by doubling the determinant of the design matrix, resetting policy/dual variables and performing **feature contraction**) → executing the policy and estimating $Q$-functions via Least Squares → policy optimization with periodic mixing → regularized dual update.

Theoretical contributions focus on the latter two components: mixing operations lead to weighted LogSumExp softmax policies requiring new covering number arguments; periodic mixing balances covering numbers and dual variables; and regularized dual updates allow for drift analysis under sparse mixing to bound $Y_k$ by $\tilde O(H^2/\gamma)$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Episode k starts<br/>Adversary chooses f^k"] --> B["Epoch Initialization<br/>Reset if determinant doubles<br/>+ Feature Contraction"]
    B --> C["Execute policy π̂^k<br/>LSTD for Q_f, Q_g"]
    C -->|Mix every K^{3/4} steps| D["1. Weighted LogSumExp softmax<br/>+ 2. Periodic policy mixing"]
    D --> E["3. Regularized dual update<br/>Y_{k+1}"]
    E -->|Enter k+1| A
```

### Key Designs

**1. Weighted LogSumExp softmax policy: Analyzable covering numbers for "mixed + optimized" policy classes**

Combining policy mixing $\hat\pi^{k-1}\leftarrow(1-\theta)\hat\pi^{k-1}+\theta\pi_{\text{unif}}$ and optimization $\hat\pi^k\propto\hat\pi^{k-1}\exp(-\alpha \hat Q^{k-1})$ breaks the multiplicative recursion. The final policy becomes:

$$\hat\pi^k\propto\sum_{i=1}^{k}\zeta_i\exp\Big(-\alpha\sum_{j=k-i}^{k-1}\hat Q^j\Big),$$

representing a weighted LogSumExp softmax using partial sums of $Q$-estimates with weights $\zeta_i$. This is more expressive than standard softmax but harder to analyze because the weights $\{\zeta_i\}$ depend on $\theta$ and $Q$-estimates, varying across policies and resisting standard Lipschitz analysis. The authors prove a recursion for mixing steps $n$: $\|\hat\pi^n_1-\hat\pi^n_2\|_1\le c\|\hat\pi^{n-1}_{1}-\hat\pi^{n-1}_{2}\|_{1}+\|P^n_1-P^n_2\|_2$ to establish Lipschitz continuity, yielding a log-covering number $\log N_\epsilon=\tilde O(n^2 d^2)$. To simplify policy expression, **feature contraction** $\bar\phi=\phi\cdot\sigma(-\beta_w\|\phi\|_{(\Lambda)^{-1}}+\log K)$ is used instead of $Q$-truncation.

**2. Periodic policy mixing: Balancing covering numbers and dual variables**

Since the covering number grows with the square of mixing steps, mixing every episode ($n=K$) results in $\log N_\epsilon = \tilde O(K^2 d^2)$, preventing sublinear bounds. However, rare mixing fails to bound dual variables. The solution is mixing every $K^B$ episodes, reducing mixing steps to $K^{1-B}$. The algorithm uses a $K^{3/4}$ period: $\tilde\pi^k\leftarrow(1-\theta)\hat\pi^k+\theta\pi_{\text{unif}}$ only when $k-k_e\equiv 0\bmod K^{3/4}$, decoupling the benefits (dual control) from the costs (covering number growth) of mixing.

**3. Regularized dual update: Tight dual bounds under sparse mixing**

Periodic mixing invalidates existing drift analyses. The authors add a regularization term:

$$Y_{k+1}\leftarrow\Big[Y_k+\eta\big(\hat V^k_{g,1}(s_1)-b\big)+\underbrace{(-c_1 Y_k-c_2)}_{\text{Regularization}}\Big]_+,$$

where the term $-c_1 Y_k - c_2$ pulls the dual variable toward 0. This term is derived by lower-bounding an inner product term $\mathbb{E}_P[\langle\hat\pi^{k+1}-\hat\pi^k,\hat Q^k\rangle]$ involving the unknown transition kernel $P$, with coefficients $c_1=4\alpha\eta H^3$ and $c_2=4\alpha\eta H^3+4\theta\eta H^2$. This allows drift analysis on the dual sub-sequence $\{Z_n\}$ at mixing episodes, proving $Y_k=\tilde O(H^2/\gamma)$ without hard truncation.

### Loss & Training

The algorithm is essentially online mirror descent with constraints. The primal side uses KL divergence for mirror descent $\hat\pi^{k+1}\propto\tilde\pi^k\exp(-\alpha(\hat Q^k_f+Y_k\hat Q^k_g))$, while the dual side uses regularized gradient ascent. Key hyperparameters: step size $\alpha=H^{-1}K^{-3/4}$, dual step size $\eta=H^{-2}K^{-3/4}$, mixing parameter $\theta=K^{-1}$, mixing period $K^{3/4}$, and optimistic reward parameter $\beta_b=\tilde O(K^{1/4}dH)$. Computational complexity is $O(d^3HK+d^2|A|HK^2)$, independent of $|S|$.

## Key Experimental Results

As a theoretical study, numerical experiments verify trends rather than provide benchmarks. Tests were conducted on a modified job-scheduling CMDP with adversarial losses (10 seeds, $K=10^5$).

### Main Results

| Metric | Ours (Bound) | Comparison | Significance |
|------|--------|------|------|
| Regret$(K)$ | $\tilde O\big(\sqrt{d^3H^4}K^{3/4}+dH^3K^{3/4}+d^3H^4K^{1/2}+\tfrac{H^6}{\gamma^2}K^{1/4}\big)$ | No prior sublinear results for adversarial linear CMDP | First sublinear regret |
| Violation$(K)$ | $\tilde O\big(\tfrac{dH^5}{\gamma}K^{3/4}+\sqrt{d^3H^4}K^{3/4}+d^3H^4K^{1/2}\big)$ | Same as above | First sublinear violation |
| Dual Variable $Y_k$ | $\tilde O(H^2/\gamma)$ | Ghosh truncates at $2H/\gamma$ | Tight bound via drift analysis |
| $\log N_\epsilon$ | $\tilde O(n^2d^2)$ | Softmax requires only Lipschitz | New LogSumExp argument |

### Key Findings
- The dominant term for both bounds is $K^{3/4}$, which is larger than the $\sqrt K$ typical in tabular settings. This gap stems from the $K^{1/4}$ factor in the covering number argument (reflected in $\beta_b$), which is the cost of handling large state spaces.
- Periodic mixing is the vital switch that pulls the covering number from $\tilde O(K^2d^2)$ down to a controllable range.
- The constraint violation curve rises early and eventually converges to zero, reflecting the mechanism of the regularization term pulling the dual variable back.

## Highlights & Insights
- Identifying "mixing frequency" as the single lever for both covering numbers and dual variables is a clean and elegant solution to two seemingly unrelated problems.
- The Lipschitz recursion for weighted LogSumExp policies is a transferable technique for scenarios where mixing/perturbation breaks standard softmax recursions.
- The regularization term is not ad-hoc; it naturally arises from lower-bounding inner product terms involving unknown transition kernels.

## Limitations & Future Work
- Regret and violation are $\tilde O(K^{3/4})$; achieving $\tilde O(\sqrt K)$ remains an open problem.
- Costs use bandit feedback but **losses require full information feedback**. Designing sample-efficient algorithms for bandit loss feedback is a future direction.
- Assumptions are relatively strong: linear kernels/losses/costs, Slater condition, and high polynomial dependence on $H$ and $1/\gamma$ (e.g., $H^5/\gamma$ in violation).
- Experiments are limited to a single modified environment and verify trends without extensive benchmarks.

## Related Work & Insights
- **vs Ghosh et al. (2022) (Stochastic Linear CMDP)**: Both use primal-dual linear approximation, but Ghosh assumes fixed losses and uses hard truncation for $Y_k$. This work handles adversarial losses via regularization and drift analysis.
- **vs Qiu et al. (2020) (Adversarial Tabular CMDP)**: Qiu uses occupancy measures for small state spaces. This paper uses policies with linear approximation for large states. Since expected costs are non-linear with respect to policies, occupancy measure drift analysis does not apply, making this paper's design necessary.
- **vs Cassel & Rosenberg (2024) (Adversarial Linear Unconstrained MDP)**: Borrowed feature contraction to control $Q$-estimates; the novelty here is integrating it into a constrained primal-dual framework with periodic mixing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First sublinear algorithm for adversarial linear CMDPs with a new covering number argument.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; numerical experiments verify trends but lack horizontal comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of the three technical contributions and their motivations.
- Value: ⭐⭐⭐⭐ Opens a theoretical path for adversarial safe RL in large state spaces; techniques are transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] On the Tension Between Optimality and Adversarial Robustness in Policy Optimization](on_the_tension_between_optimality_and_adversarial_robustness_in_policy_optimizat.md)
- [\[ICLR 2026\] SHAPO: Sharpness-Aware Policy Optimization for Safe Exploration](shapo_sharpness-aware_policy_optimization_for_safe_exploration.md)
- [\[ICLR 2026\] DuPO: Enabling Reliable Self-Verification via Dual Preference Optimization](dupo_enabling_reliable_self-verification_via_dual_preference_optimization.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](../../NeurIPS2025/reinforcement_learning/global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)

</div>

<!-- RELATED:END -->

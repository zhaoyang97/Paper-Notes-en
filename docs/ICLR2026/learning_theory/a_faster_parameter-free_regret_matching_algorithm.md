---
title: >-
  [Paper Note] A Faster Parameter-Free Regret Matching Algorithm
description: >-
  [ICLR2026][Learning Theory][Regret Matching] This paper proposes a parameter-free regret matching variant **MI-SPRM+**. By using a technique called "Adaptive Regret Domain (ARD)" to monotonically raise the lower bound of the cumulative regret's 1-norm, it **preserves the parameter-free property while achieving an $O(1/T)$ theoretical convergence rate** in two-player zero-sum games—making it the first RM-type algorithm known to achieve both simultaneously.
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Game Solving"
  - "Online Learning"
  - "Regret Matching"
  - "Nash Equilibrium"
  - "Convergence Rate"
  - "Parameter-Free Algorithm"
  - "Adaptive Stepsize"
date: 2026-05-08
content_hash: 8279705924d7ffcd
---

# A Faster Parameter-Free Regret Matching Algorithm

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=JLllvi7dsg](https://openreview.net/forum?id=JLllvi7dsg)  
**Code**: To be confirmed  
**Area**: Learning Theory / Game Solving / Online Learning  
**Keywords**: Regret Matching, Nash Equilibrium, Convergence Rate, Parameter-Free Algorithm, Adaptive Stepsize

## TL;DR
This paper proposes a parameter-free regret matching variant **MI-SPRM+**. By using a technique called "Adaptive Regret Domain (ARD)" to monotonically raise the lower bound of the cumulative regret's 1-norm, it **preserves the parameter-free property while achieving an $O(1/T)$ theoretical convergence rate** in two-player zero-sum games—making it the first RM-type algorithm known to achieve both simultaneously.

## Background & Motivation
**Background**: In finding the Nash Equilibrium (NE) for two-player zero-sum normal-form games (NFG), algorithms based on the regret minimization framework are currently the most popular. Among them, Regret Matching (RM) and its variants (RM+, PRM+, DRM, etc.) are widely adopted by superhuman game AIs due to their excellent empirical performance in large-scale games like poker, holding a more prominent position than traditional regret minimization algorithms based on OMD/FTRL.

**Limitations of Prior Work**: Paradoxically, while many OMD/FTRL algorithms can achieve an $O(1/T)$ theoretical convergence rate, most RM variants can only prove $O(1/\sqrt{T})$. The "Smooth RM+ variant" (represented by SPRM+) proposed by Farina et al. in 2023 was the first to improve the theoretical convergence of RM+ to $O(1/T)$ by ensuring the 1-norm of cumulative regret is always greater than a positive constant. **However, SPRM+ sacrificed the most attractive "parameter-free" property of RM+/PRM+**—the ability to function without tuning any hyperparameters.

**Key Challenge**: This parameter-free property is extremely important in practice because regret minimization algorithms are exceptionally sensitive to the stepsize $\eta$. In the experiments of this paper, even for the $3\times3$ game used in the original SPRM+ paper, after $10^5$ iterations, the duality gap (the distance to NE, where smaller is better) achieved by SPRM+ with $\eta=0.1$ is 10 times smaller than with $\eta=0.01$ and 5 times smaller than with $\eta=1$. This indicates that using SPRM+ requires significant effort for parameter tuning; otherwise, convergence is significantly slower. Thus, a core contradiction arises: **$O(1/T)$ convergence and the parameter-free property were previously unattainable simultaneously under the RM framework**.

**Goal**: Design an RM+ variant that satisfies (1) parameter-freedom; (2) $O(1/T)$ theoretical convergence; (3) fast empirical convergence (unlike some theoretical $O(1/T)$ algorithms that are very slow in practice).

**Key Insight**: The authors identify a key dependency in the convergence proof of SPRM+: the stepsize range for SPRM+ to achieve $O(1/T)$ is $0<\eta<R\cdot C_0$, where $R$ is the lower bound of the 1-norm of cumulative regret and $C_0$ is a constant related only to the game. In other words, for **any** given $\eta>0$, $O(1/T)$ holds as long as $R>\eta/C_0$.

**Core Idea**: Since convergence depends on the ratio of $R$ to $\eta$, the algorithm **does not tune $\eta$, but instead adaptively raises $R$**, letting it grow monotonically until it exceeds $\eta/C_0$. This enters the $O(1/T)$ interval without any manual tuning. This mechanism of raising $R$ is the Adaptive Regret Domain (ARD).

## Method

### Overall Architecture
First, the notation and background. In a two-player zero-sum NFG, each player $i$ selects a strategy $x_i$ on the simplex $\mathcal{X}^i$. The duality gap $\mathrm{dg}(x)=\max_{x'\in\mathcal{X}}\langle \ell^x, x-x'\rangle$ measures the distance to NE, where $\mathrm{dg}(x)=0$ if and only if $x$ is an NE. The goal of the regret minimization framework is to ensure the social regret $\sum_{t=1}^{T}\langle \ell^t, x^t-x\rangle$ grows sublinearly, so the time-averaged strategy $\bar{x}^T=\sum_t x^t/T$ converges to an approximate NE.

RM+ updates cumulative regret on the non-negative quadrant: $\theta^{t+1}_i=[\theta^t_i+\eta F^t_i(x^t)]_+$, and the strategy is obtained by normalization: $x^t_i=\theta^t_i/\|\theta^t_i\|_1$, where $F^t_i=\langle\ell^t_i,x^t_i\rangle\mathbf{1}-\ell^t_i$ is the instantaneous regret. PRM+ introduces "using feedback from the previous step as current prediction" to accelerate this. SPRM+ shifts the update space from the non-negative quadrant $\mathbb{R}^{|A_i|}_{\geq0}$ to $\mathbb{R}^{|A_i|}_{\geq R}$ (where each component is lower-bounded by a constant $R>0$), thereby ensuring the stability of instantaneous regret $\|F^{t-1}(\theta^{t-1})-F^t(\theta^t)\|_2^2\leq O(\|\theta^{t-1}-\theta^t\|_2^2)$. This improves PRM+'s rate from $O(1/\sqrt{T})$ to $O(1/T)$, at the cost of needing $0<\eta<R\cdot C_0$, which requires tuning.

MI-SPRM+ (Algorithm 1) makes one key modification to SPRM+: **it replaces the fixed constant $R$ with a monotonically increasing $R_t$ per iteration**, turning the decision space into $\mathbb{R}^{|A_i|}_{\geq R_t}$. Each step proceeds by: using previous feedback $F^{t-1}_i(\theta^{t-1})$ to perform a predictive update on $\mathbb{R}^{|A_i|}_{\geq R_t}$ to obtain $\theta^t_i$ and normalize it to $x^t_i$; then using current feedback $F^t_i(\theta^t)$ to update $\hat\theta^{t+1}_i$; finally, deciding whether $R_{t+1}$ is $R_t+1$ or remains $R_t$ based on a criterion. The entire algorithm only calls the loss gradient once per step (single-call), has $O(1)$ per-step complexity, and contains no hyperparameters requiring manual setting. The final output is an average strategy weighted by $R_t$: $\bar{x}^T=\frac{\sum_{t=1}^T R_t x^t}{\sum_{t=1}^T R_t}$.

### Key Designs

**1. Key Observation: The $O(1/T)$ stepsize interval is supported by the 1-norm lower bound of cumulative regret**

This is the fulcrum of the paper. The authors refine Theorem B.1 from the SPRM+ convergence proof: SPRM+ reaches $O(1/T)$ when $0<\eta<R\cdot C_0$, where the game-dependent constant $C_0=1/\sqrt{8D(2L^2+4DL^2+4DP^2)}$ ($D=\max_i|A_i|$, and $L, P$ are the Lipschitz and norm bounds of the loss gradient). Reinterpreting this: **for any fixed $\eta>0$, convergence holds as long as $R>\eta/C_0$**. This means the speed of convergence is not determined by the absolute value of $\eta$, but by whether $R$ is sufficiently large. This observation transforms the problem of "tuning stepsize" into one of "raising the lower bound"—and the lower bound is a value the algorithm can control. This step bridges the seemingly conflicting goals of being parameter-free and achieving $O(1/T)$.

**2. Adaptive Regret Domain (ARD): Monotonically raising $R_t$ instead of lowering the stepsize**

ARD is the specific mechanism for this core idea: it adjusts the decision space lower bound $R_t$ at each step (constraining updates within $\mathbb{R}^{|A_i|}_{\geq R_t}$), causing the cumulative regret 1-norm lower bound to increase monotonically and eventually cross the threshold $1/C_0$ (since $\eta$ can be fixed to a constant, the key is $R_t$ growing to exceed $\eta/C_0$). This stands in sharp contrast to the existing parameter-free OMD algorithm DS-OptMD, which follows a path of "adaptively **lowering** the stepsize $\eta$." However, DS-OptMD's reduction is too conservative, leading to slow empirical convergence. ARD goes the opposite way, following a path of "**aggressively raising** $R_t$," ensuring both theoretical $O(1/T)$ and fast empirical convergence. This explains why MI-SPRM+ is not "theoretically fast but empirically slow" like DS-OptMD.

**3. Growth Criterion for $R_t$: Only incrementing when "stability conditions are violated"**

$R_t$ does not increase blindly; it is triggered by a condition:
$$
R_{t+1}=
\begin{cases}
R_t+1 & \text{if } \|F^t(\theta^t)-F^{t-1}(\theta^{t-1})\|_2^2-\dfrac{B_\psi(\hat\theta^t,\theta^{t-1})+B_\psi(\hat\theta^t,\theta^t)}{2}>0,\\[4pt]
R_t & \text{else,}
\end{cases}
$$
where initial $R_1=1$, and $B_\psi$ is the Bregman divergence corresponding to the quadratic regularizer $\psi(\cdot)=\|\cdot\|_2^2/2$. The term on the left is exactly the surplus of the stability inequality $\|F^t-F^{t-1}\|_2^2\leq O(\|\theta^t-\theta^{t-1}\|_2^2)$ used by SPRM+ to guarantee $O(1/T)$. Once the fluctuation of instantaneous regret exceeds what the current lower bound can suppress (the difference is positive), it indicates $R_t$ is not large enough and the stability condition is violated, thus incrementing $R_t$ to expand the bound. Otherwise, it remains unchanged. This ensures $R_t$ grows only when "necessary," preventing the decision space from expanding excessively and slowing practical convergence.

**4. Product Strategy output weighted by $R_t$**

Because the decision space lower bound $R_t$ changes, simple time-averaging $\sum_t x^t/T$ no longer corresponds to the correct regret bound. MI-SPRM+ uses an $R_t$-weighted average $\bar{x}^T=\frac{\sum_{t=1}^T R_t x^t}{\sum_{t=1}^T R_t}$ as the final strategy. Iterations with larger $R_t$ (occurring when the algorithm has entered the stable $O(1/T)$ interval) are given more weight, while earlier iterations with smaller $R_t$ are less weighted. Theorem 4.1 proves that when all players in a two-player zero-sum NFG use MI-SPRM+, this weighted average converges to an approximate NE at an $O(1/T)$ rate.

### Loss & Training
This work presents an online learning algorithm for game solving and does not involve a training loss. The convergence proof strategy is: first, use the property that $R_t$ grows monotonically and eventually exceeds $1/C_0$ (guaranteed by the definition of $\mathbb{R}^{|A_i|}_{\geq R_t}$ where $R_t$ is the 1-norm lower bound of cumulative regret) to bring each step into the stability argument of SPRM+; then, combine with $R_t$ weighting to obtain a social regret bound, which results in an $O(1/T)$ duality gap bound. For multi-player general-sum NFGs, the authors provide an $O(1)$ social regret bound $\big(\sum_t R_t\langle\ell^t,x^t-x\rangle\big)/\big(\sum_t R_t\big)\leq O(1)$ (since finding NE is PPAD-complete, there is no polynomial-time convergence guarantee), consistent with the original SPRM+ paper. For Extensive-Form Games (EFG), the algorithm design extends directly, though the $O(1/T)$ theoretical guarantee is lost; however, empirical results show $O(1/T)$ or faster convergence when combined with the CFR framework.

## Key Experimental Results

### Main Results
Evaluation was conducted on two types of games: the $3\times3$ zero-sum NFG from the original SPRM+ paper (payoff matrix $[[3,0,-3],[0,3,-4],[0,0,1]]$), and randomly generated two-player zero-sum NFGs with dimensions $\{10,30,50,100\}$ (60 independent instances per dimension, with payoff matrices sampled from Gaussian distributions with standard deviations of 1/10/100). Baseline algorithms: RM+, PRM+, SPRM+, OGDA, OMWU, DS-OptMD. The metric is the duality gap (smaller is better). MI-SPRM+, RM+, PRM+, and DS-OptMD are parameter-free; for non-parameter-free SPRM+ and OGDA, stepsizes were chosen from $\{0.01,0.1,1\}$.

| Algorithm ($3\times3$ NFG, $10^5$ iterations) | Parameter-Free | Theoretical Rate | Duality Gap |
|------|------|------|------|
| **MI-SPRM+ (Ours)** | Yes | $O(1/T)$ | **2.6e-5** |
| SPRM+ ($\eta=0.1$) | No | $O(1/T)$ | 4.0e-5 |
| SPRM+ ($\eta=1$) | No | $O(1/T)$ | 2.1e-4 |
| SPRM+ ($\eta=0.01$) | No | $O(1/T)$ | 3.9e-4 |

MI-SPRM+, **without any tuning**, achieves a duality gap smaller than even the optimally tuned SPRM+ ($\eta=0.1$), and outperforms poorly tuned versions of SPRM+ ($\eta=0.01$/$\eta=1$) by approximately an order of magnitude. On random NFGs, MI-SPRM+ achieves $O(1/T)$ empirical convergence rates across all dimensions and outperforms all baselines.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| SPRM+ with different $\eta$ | Duality gap at $10^5$ iterations varies by up to 10x | Confirms high sensitivity to stepsize for RM+ variants |
| DS-OptMD (Parameter-free OMD) | Empirical convergence significantly slower than $O(1/T)$ | "Lowing stepsize" is too conservative; theoretical speed does not match practice |
| OGDA / OMWU (Traditional Regret) | More sensitive than RM+; only specific $\eta$ converge at std 1/10 | Tuning traditional OMD/FTRL is even more difficult |
| **MI-SPRM+ (ARD raising $R_t$)** | Parameter-free, $O(1/T)$ in theory/practice, fastest | "Raising lower bound" balances theory and practice |

### Key Findings
- The core gain of MI-SPRM+ comes from ARD: replacing SPRM+'s fixed bound $R$ with an adaptive $R_t$ is the sole source of parameter-freedom.
- The comparison with DS-OptMD—which is also parameter-free and theoretically $O(1/T)$—is particularly illustrative: the latter's slow performance validates that "aggressively raising $R$" is superior to "conservatively lowering $\eta$."
- Combined with CFR, it handles EFGs with empirical $O(1/T)$ or faster convergence; per-step complexity remains $O(1)$ (compared to $O(\log T)$ for Clairvoyant CFR).

## Highlights & Insights
- **Translating "tuning" into "control"**: The central insight is that in SPRM+'s convergence condition $0<\eta<R\cdot C_0$, $R$ is a value the algorithm can control. Since the ratio $\eta/R$ determines convergence, the algorithm leaves $\eta$ alone and raises $R$ instead—a clean shift in perspective transferable to other algorithms depending on controllable bounds.
- **Raising bounds vs. Lowering stepsizes**: DS-OptMD lowers the stepsize and is slow empirically, while this work raises the lower bound and is fast in both regards. This suggests that when multiple variables determine convergence, adaptively adjusting the variable that doesn't hurt empirical performance is more effective.
- **Triggered growth + weighted average**: $R_t$ increments only when stability margin is negative, paired with $R_t$-weighted output. This ensures stability and prioritizes refined iterations, a useful engineering lesson for parameter-free algorithms.

## Limitations & Future Work
- **Theoretical $O(1/T)$ only for two-player zero-sum NFG**: For multi-player general-sum NFGs, only an $O(1)$ social regret bound is provided (constrained by PPAD complexity, an inherent barrier not unique to this paper); for EFGs, $O(1/T)$ is an empirical observation without proof.
- **Not stepsize-invariant (Strong Parameter-Free)**: RM+/PRM+ possess a stronger property where output is entirely independent of the choice of $\eta$. The authors state that to their knowledge, no algorithm yet achieves $O(1/T)$ and stepsize-invariance; MI-SPRM+ focuses on the weaker parameter-free property.
- **Coarse growth granularity for $R_t$**: $R_t\leftarrow R_t+1$ is a fixed step. Whether a finer adaptive step could further accelerate performance while maintaining guarantees is a natural next step.

## Related Work & Insights
- **vs SPRM+ (Farina et al., 2023)**: SPRM+ uses a fixed $R$ to lift RM+ to $O(1/T)$ but requires tuning $\eta<R\cdot C_0$. This paper removes that requirement by making $R$ an adaptive $R_t$ and is faster empirically. Both are single-call $O(1)$ algorithms.
- **vs DS-OptMD (Hsieh et al., 2021)**: Also parameter-free and $O(1/T)$, but DS-OptMD follows an OMD route, lowering stepsizes too conservatively. This work follows an RM+ route, raising bounds, and is fast theoretically and empirically.
- **vs RM+/PRM+ (Tammelin 2014; Farina et al., 2021)**: These are (strongly) parameter-free but only achieve $O(1/\sqrt{T})$. This paper improves the rate to $O(1/T)$ while maintaining parameter-freedom.
- **vs doubling trick (Auer et al., 1995)**: A classic parameter-freeing tool, but even with an $O(1/T)$ base algorithm, it yields $O(\log^2 T/T)$. ARD avoids this $\log^2 T$ loss by raising bounds directly.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First known RM-type algorithm to be both parameter-free and $O(1/T)$; "raising bound" instead of "tuning stepsize" is a clean and non-trivial perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers fixed and multi-dimensional random zero-sum NFGs against 6 baselines; multi-player/EFG is empirically verified.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation flows clearly from the SPRM+ condition; ARD's derivation is well-explained.
- Value: ⭐⭐⭐⭐ Practical for large-scale game AIs relying on RM+, removing sensitive stepsize tuning with stronger theoretical guarantees.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dimension-Free Decision Calibration for Nonlinear Loss Functions](dimension-free_decision_calibration_for_nonlinear_loss_functions.md)
- [\[ICLR 2026\] Bandit Learning in Matching Markets Robust to Adversarial Corruptions](bandit_learning_in_matching_markets_robust_to_adversarial_corruptions.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)
- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)
- [\[ICLR 2026\] Queue Length Regret Bounds for Contextual Queueing Bandits](queue_length_regret_bounds_for_contextual_queueing_bandits.md)

</div>

<!-- RELATED:END -->

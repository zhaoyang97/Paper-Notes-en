---
title: >-
  [Paper Note] Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates
description: >-
  [ICML2026][Reinforcement Learning][linear contextual bandit] In linear contextual bandits, the authors explicitly decouple the two axes of "when rewards are received" and "whether context-dependency is allowed within an…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "linear contextual bandit"
  - "batched bandit"
  - "rare parameter updates"
  - "minimax regret"
  - "G-optimal design"
date: 2026-05-08
content_hash: 4c770249717350d2
---

# Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates

**Conference**: ICML2026  
**arXiv**: [2606.00984](https://arxiv.org/abs/2606.00984)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / Bandit Theory  
**Keywords**: linear contextual bandit, batched bandit, rare parameter updates, minimax regret, G-optimal design

## TL;DR
In linear contextual bandits, the authors explicitly decouple the two axes of "when rewards are received" and "whether context-dependency is allowed within an interval," which were previously conflated under the term "batched." They define "rare parameter updates"—a setting more aligned with real-world deployment that limits reward-driven parameter updates while allowing reward-free context adaptivity. Accordingly, they propose two algorithms, BLCE-G and BLCE, requiring only $\mathcal O(\log\log T)$ parameter updates. The former is the first to achieve minimax-optimal regret $\widetilde{\mathcal O}(\sqrt{dT\log K}\wedge d\sqrt T)$ simultaneously in both small-$K$ and large-$K$ regimes. The latter eliminates the computational bottleneck of G-optimal design, achieving the lowest runtime among all optimal algorithms. This methodology is extended to generalized linear bandits (BGLE), removing the dependency on the worst-case curvature parameter $\kappa$.

## Background & Motivation

**Background**: The stochastic linear contextual bandit is a classic model for sequential decision-making: at each round $t$, the agent observes $K$ arm feature vectors $\mathcal A_t=\{x_{t,1},\ldots,x_{t,K}\}\subseteq\mathbb R^d$, selects an arm to pull, and receives $r_t=\langle x_{t,a_t},\theta^*\rangle+\eta_t$. The goal is to minimize the cumulative expected regret $\mathcal R(T)=\mathbb E[\sum_t(\langle x_t^*,\theta^*\rangle-\langle x_{t,a_t},\theta^*\rangle)]$ over $T$ rounds. Deployment bottlenecks usually stem from "injecting new reward feedback for parameter estimation" rather than "selecting an arm given a context." Model retraining, confidence set recomputation, and posterior updates involve expensive pipelines, log aggregation, privacy audits, or human verification, motivating the study of "limited adaptivity / batched bandits."

**Limitations of Prior Work**: The authors identify that the term "batched" in literature is conflated across two independent axes: (i) reward visibility (feedback delay) and (ii) whether the policy can depend on contexts arriving within an interval (context adaptivity). For instance, Karbasi (2021) formalizes a batch policy as $\pi_t:H_{t_{\ell-1}}\times C_t\to\mathcal A$, allowing within-interval context dependency. Conversely, static-grid algorithms like Ruan (2021), Hanna (2023), and Zhang (2025) achieve near minimax-optimal regret with $\mathcal O(\log\log T)$ updates but implicitly require committing to an action rule that ignores within-interval contexts. This strict batching forces the use of expensive primitives like G-optimal design, $(1/T)$-net discretization, or repeated optimization oracles, leading to impractical runtimes (e.g., Zhang (2025) takes 290s for $(K=1000, d=5, T=10^4)$).

**Key Challenge**: Strict batching is not an inherent requirement of batched feedback. Since contexts are observed before selection, ignoring already seen contexts until the next parameter update is unnecessary and unnatural. Decoupling "minimal reward-based parameter updates" from "within-interval context adaptivity" offers an opportunity to discard the computational burden imposed by G-optimal design.

**Goal**: (1) Provide an algorithm that is minimax-optimal in both small-$K$ ($K\le\mathcal O(e^d)$) and large-$K$ ($K\ge\Omega(e^d)$) regimes with $\mathcal O(\log\log T)$ updates—existing static-grid methods only cover one regime. (2) Reduce runtime by removing the G-optimal design subprocess while maintaining minimax optimality. (3) Extend the approach to generalized linear bandits, eliminating the regret bound's dependency on the worst-case curvature $\kappa$ (which diverges as the link function saturates $\dot\mu\to 0$).

**Key Insight**: Maintain a reward-free state variable—the Gram matrix $H_t$ and its inverse (updated via Sherman-Morrison in $\mathcal O(d^2)$)—within each interval $\ell$. Select arms using "uncertainty-driven" exploration $\arg\max_x\|x\|_{H_{t-1}^{-1}}$. This naturally leverages within-interval contexts without requiring rewards, fitting the "rare parameter updates" setting perfectly.

**Core Idea**: Replace "static G-optimal design rollout" with "reward-free in-interval maximum variance exploration + arm elimination + boundary parameter updates." Theoretically, the cumulative bound of $\|x\|_{H^{-1}}$ yields information gain equivalent to G-optimal design, while the runtime removes the $\mathcal O(Kd^3)$ design calls.

## Method

### Overall Architecture
The horizon $[T]$ is partitioned into $B=\mathcal O(\log\log T)$ static-grid intervals: $\mathcal T_1=\lceil\sqrt T/\log_2\log_2 T\rceil+1$, $\mathcal T_\ell=(\mathcal T_{\ell-1}+\lceil T^{1-2^{-\ell}}/\log_2\log_2 T\rceil+2)\wedge T$. Within each interval, rewards are invisible while contexts arrive online. At boundary $\mathcal T_\ell$, ridge regression is performed: $\hat\theta_\ell=V_\ell^{-1}\sum_{t}r_tx_{t,a_t}$ (where $V_\ell=H_{\mathcal T_\ell}$), and $H$ is reset to $\lambda I$. For the subsequent interval, nested arm elimination using $\hat\theta_1,\ldots,\hat\theta_{\ell-1}$ narrows the feasible set $\mathcal A_t^{(k)}$. The core algorithms, BLCE-G and BLCE, share this framework but differ in the interval-wise exploration-exploitation allocation.

### Key Designs

1.  **BLCE-G: near G-optimal design + uncertainty + greedy allocation**:
    - **Function**: Achieves minimax-optimal regret for both small-$K$ and large-$K$ regimes for the first time with $\mathcal O(\log\log T)$ updates.
    - **Mechanism**: The first interval is split $c:(1-c)$; the first segment pulls arms according to a *near G-optimal design* distribution $\pi_{G'}(\mathcal A_t)$ (relaxed by factor 2 for $\mathcal O(Kd^3)$ complexity), while the second segment selects via $\arg\max_x\|x\|_{H_{t-1}^{-1}}$. Intervals $\ell\ge 2$ use a $c^2:c(1-c):(1-c)$ split: near G-optimal design over $\mathcal A_t^{(\ell-1)}$, maximum uncertainty direction, and greedy selection via $\hat\theta_{\ell-1}$. The elimination threshold $\varepsilon_{t,k}$ is the minimum of two confidence bounds: $\max_{y\in\mathcal A_t^{(k-1)}}\|y\|_{V_k^{-1}}\big(\sqrt{2\log(|\mathcal A_t^{(k-1)}|(B-1)T^2)}+\sqrt\lambda \wedge 2\sqrt{\log(2^{6d-5}\pi d(B-1)^2T^2/15^{d-1})}+2\sqrt\lambda\big)$, naturally covering both regimes.
    - **Design Motivation**: The $\min$ elimination threshold is essential for simultaneous optimality across regimes—the former dominated by $\sqrt{\log K}$ and the latter by $\sqrt d$. Near G-optimal design is a necessary engineering relaxation for NP-hard complexity. The greedy segment exploits the refined feasible set to further reduce the regret constant.

2.  **BLCE: Complete removal of G-optimal design**:
    - **Function**: Maintains $\mathcal O(\log\log T)$ updates and minimax optimality while discarding G-optimal design, reducing runtime to $\mathcal O(Kd^2T\log\log T)$, the lowest among optimal algorithms.
    - **Mechanism**: The entire first interval uses $\arg\max_x\|x\|_{H_{t-1}^{-1}}$ (uncertainty-driven exploration), with per-step $\mathcal O(d^2)$ updates for $H_t^{-1}$. Intervals $\ell\ge 2$ use a $c:(1-c)$ split: uncertainty-driven exploration (merging the G-optimal and uncertainty segments of BLCE-G) followed by greedy selection. Theorem 2 proves it still achieves $\mathcal R(T)=\widetilde{\mathcal O}(\sqrt{dT\log K}\wedge d\sqrt T)$, adding only a $\sqrt{\log T}$ factor to the polylog term.
    - **Design Motivation**: Strict batching "requires" G-optimal design because it cannot rely on within-interval contexts. By allowing reward-free Gram-matrix evolution, the rule $\arg\max_x\|x\|_{H_{t-1}^{-1}}$ achieves equivalent information gain. This not only saves computation but also eliminates approximation errors from unattainable $1$-approximate designs.

3.  **BGLE: Extending BLCE to Generalized Linear Bandits with $\kappa$-independence**:
    - **Function**: For generalized linear contextual bandits ($\mathbb E[r_t]=\mu(\langle x_{t,a_t},\theta^*\rangle)$), it uses $\mathcal O(\log\log T)$ updates and ensures both leading and transient regret terms are independent of the worst-case curvature $\kappa$.
    - **Mechanism**: Follows BLCE's structure. In intervals $\ell\ge 2$, the Gram matrix is weighted by $\alpha_{t,\ell-1}(\lambda)\dot\mu(\langle x_{t,a_t},\hat\theta_{\ell-1}\rangle)$. At boundaries, MLE replaces ridge regression: $\hat\theta_\ell=\arg\min_\theta\sum_t[m(\langle x_{t,a_t},\theta\rangle)-r_t\langle x_{t,a_t},\theta\rangle]$. Elimination starts at $\ell\ge 3$ with threshold $\varepsilon'_{t,k}(\lambda)$, simplified to $\max_y\|y\|_{V_k^{-1}}(50RS\sqrt{d+\log T})$ when $\lambda=R^2(d+\log T)$.
    - **Design Motivation**: Standard GLB regret bounds scale with $1/\kappa$ ($\kappa=\max\dot\mu^{-1}$), becoming vacuous for saturated link functions like logistic. Weighting by local curvature $\dot\mu(\langle x_{t,a_t},\hat\theta_{\ell-1}\rangle)$ makes the analysis depend on *average* curvature $\hat\kappa=1/\mathbb E_{\mathcal A\sim\mathcal D}[\dot\mu(\langle x^*,\theta^*\rangle)]$ instead of the worst case.

### Loss & Training
Neither algorithm trains neural weights; they use least-squares or MLE. BLCE-G and BLCE perform ridge regression $\hat\theta_\ell=V_\ell^{-1}\sum_t r_t x_{t,a_t}$ at boundaries, with $\lambda$ set to $\log(dT)$ for BLCE-G and $1$ for BLCE. BGLE uses MLE at boundaries with $\lambda=R^2(d+\log T)$. Interval lengths follow a doubling-trick style static grid: $\mathcal T_\ell=\mathcal T_{\ell-1}+\lceil T^{1-2^{-\ell}}/\log_2\log_2 T\rceil$.

## Key Experimental Results

### Main Results
$T=10,000$, 10 independent runs, arms sampled uniformly from $d$-dim space, $\theta^*$ sampled from $d$-dim normal. Baselines: RS-OFUL (Abbasi 2011), BatchLinUCB-DG (Ruan 2021), SoftBatch (Hanna 2023), BatchLearning (Zhang 2025).

| Config (K,d) | RS-OFUL | SoftBatch | BatchLinUCB-DG | Hanna2023contexts | BatchLearning | BLCE-G | BLCE |
|--------------|---------|-----------|----------------|--------------------|----------------|--------|------|
| (1000,5)     | 0.85s   | 4.18s     | 290.87s        | Exponential        | 166.17s        | 23.40s | **5.91s** |
| (5000,10)    | 4.15s   | 15.17s    | 1300.01s       | Exponential        | 621.09s        | 40.27s | **12.83s** |
| (50,20)      | 0.42s   | 3.74s     | 1031.66s       | Exponential        | 45.85s         | 2.26s  | **1.06s** |
| (100,30)     | 0.61s   | 5.50s     | 2987.07s       | Exponential        | 77.01s         | 3.70s  | **1.62s** |

BLCE achieves the lowest runtime among all optimal algorithms, comparable to suboptimal ones like RS-OFUL. In terms of regret, BLCE-G and BLCE outperform all baselines in both large-$K$ and small-$K$ settings with lower variance.

### Key Findings
- BLCE's runtime matches the leading-order $\mathcal O(Kd^2T\log\log T)$ term of Zhang (2025) but eliminates the costly $\mathcal O(Kd^{7/2}\sqrt{T\log(dKT)\log T})$ first-batch G-optimal design term.
- Decoupling "rare parameter updates" from "strict batching" is a significant empirical contribution: for the same $\mathcal O(\log\log T)$ updates, allowing context adaptivity reduces runtime from 290s to 5.9s (~50x) while tightening regret.
- BGLE's regret leading term $\widetilde{\mathcal O}(RSd\sqrt T/\sqrt{\hat\kappa})$ depends on *average* curvature $\hat\kappa$, a qualitative improvement for saturated link functions where $\kappa$ can be unbounded.
- Remark 1 relaxes the i.i.d. context assumption for BLCE to batch-wise conditions, making it applicable to scenarios with batch-level drift.

## Highlights & Insights
- The most valuable contribution is the conceptual clarification of "batched" into three distinct operational levels: fully sequential, strictly batched, and rare parameter updates. 
- Removing G-optimal design challenges the consensus that it is a necessity for $\mathcal O(\log\log T)$ optimal algorithms; the authors prove it is merely a constraint of strict batching.
- The "uncertainty-driven Gram" mechanism in BLCE adapts LinUCB's fully-adaptive logic to a reward-free within-interval setting, a design principle applicable to any batched feedback system like recommendation or ad ranking.
- BGLE achieving $\kappa$-free leading and transient terms (at order $T^{1/3}$) is vital for production systems during cold-start phases where saturation is common.

## Limitations & Future Work
- The static-grid schedule depends heavily on a known $T$; whether dual regime optimality holds under adaptive grids remains an open question.
- The analysis assumes finite $K$. Scaling to millions of arms for recommendation systems would require combining BLCE with approximate nearest neighbor (ANN) structures, which is not discussed.
- BGLE’s transient term includes $e^{8RS}$, which might dominate if $RS$ is large.
- Experiments are restricted to synthetic uniform/Gaussian settings; validation on real-world industrial datasets is pending.

## Related Work & Insights
- **vs Abbasi 2011 RS-OFUL**: RS-OFUL uses rare-switching with $\mathcal O(d\log T)$ updates. BLCE reduces updates to $\mathcal O(\log\log T)$, optimizes regret to minimax-optimal, and achieves lower runtime for larger $d$.
- **vs Ruan 2021 / Zhang 2025**: These static-grid methods maintain strict batching via G-optimal design, making them computationally intensive ($\mathcal O(Kd^4T)$). BLCE replaces design with reward-free Gram evolution, reaching $\mathcal O(Kd^2T\log\log T)$.
- **vs Karbasi 2021**: One of the few studies allowing within-interval adaptivity but with $\mathcal O(K\log T)$ updates. BLCE reduces updates to $\mathcal O(\log\log T)$ while achieving minimax-optimal regret.
- **vs Sawarni 2024 (GLB)**: Sawarni's transient term still contains $\kappa$, whereas BGLE is completely $\kappa$-free.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](../../ICLR2026/reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](../../NeurIPS2025/reinforcement_learning/tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](../../NeurIPS2025/reinforcement_learning/generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[ICML 2026\] Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](parameter-free_dynamic_regret_time-varying_movement_costs_delayed_feedback_and_m.md)
- [\[AAAI 2026\] Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback](../../AAAI2026/reinforcement_learning/bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)

</div>

<!-- RELATED:END -->

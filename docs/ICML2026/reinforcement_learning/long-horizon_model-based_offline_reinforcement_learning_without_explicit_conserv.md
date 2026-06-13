---
title: >-
  [Paper Note] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism
description: >-
  [ICML 2026][Reinforcement Learning][offline RL] This paper challenges the prevailing consensus that "offline RL must be explicitly conservative." It proposes Neubay…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "offline RL"
  - "model-based RL"
  - "Bayesian RL"
  - "long-horizon rollout"
  - "epistemic POMDP"
date: 2026-05-08
content_hash: 14a6d27978d8eea2
---

# Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism

**Conference**: ICML 2026  
**arXiv**: [2512.04341](https://arxiv.org/abs/2512.04341)  
**Code**: https://github.com/twni2016/neubay (Available)  
**Area**: Offline Reinforcement Learning / Model-Based RL / Bayesian RL  
**Keywords**: offline RL, model-based RL, Bayesian RL, long-horizon rollout, epistemic POMDP

## TL;DR
This paper challenges the prevailing consensus that "offline RL must be explicitly conservative." It proposes Neubay, which utilizes a Bayesian perspective to view model ensembles as posteriors, employs **long-horizon rollouts (hundreds of steps)** to naturally absorb value overestimation, and controls compounding errors via layer normalization and uncertainty thresholds. Without utilizing pessimistic penalties, it matches SOTA conservative algorithms across 33 datasets from D4RL/NeoRL and sets new records on 7 of them.

## Background & Motivation

**Background**: Mainstream offline RL (e.g., CQL, IQL, EDAC, ReBRAC, MOPO, COMBO, MOBILE) is built on the principle of "explicit conservatism"—imposing penalties on out-of-distribution state-action pairs while performing only short-horizon (1–5 steps) rollouts in model-based methods to simultaneously suppress value overestimation and compounding errors. Theoretically, this corresponds to a robust MDP $\max_\pi\min_{m\in\mathfrak{M}_\mathcal{D}}J(\pi,m)$.

**Limitations of Prior Work**: While the conservatism principle reduces overestimation, it also suppresses average-case performance, particularly on low-quality datasets. When the behavior policy is poor, conservative training restricts the agent to suboptimal actions and fails to explore better, unseen actions during testing. Although the Bayesian perspective (epistemic POMDP) as proposed by Ghosh et al. theoretically allows for test-time adaptation, practical algorithms (APE-V, MAPLE, CBOP, MoDAP) have reintroduced uncertainty penalties and short horizons for stability, diluting the Bayesian spirit back into a conservative approach.

**Key Challenge**: The Bayesian objective $\max_\pi \mathbb{E}_{m\sim \mathbb{P}_\mathcal{D}}[J(\pi, m)]$ theoretically requires full rollouts on the posterior. However, in practice, removing explicit conservatism makes compounding errors and value overestimation unmanageable. This has led to the Bayesian route being "theoretically appealing but practically underperforming."

**Goal**: (1) Empirically demonstrate that a pure Bayesian approach (without any uncertainty penalties) can work on mainstream offline RL tasks; (2) Identify key bottlenecks and design corresponding mechanisms; (3) Provide clear boundaries for when to use Bayesian vs. conservative methods.

**Key Insight**: The authors use an extreme two-armed bandit example to illustrate that conservatism on skewed data is **destined** to stay with suboptimal arms already observed. In contrast, a Bayesian approach can adapt at test time. Based on this, they offer a counter-intuitive observation: **long-horizon rollouts themselves can replace explicit conservatism to suppress value overestimation**. This is because the bootstrap term in an $H$-step TD target, $\sum_{j=0}^{H-1}\gamma^j \hat{r}_{t+j+1} + \gamma^H Q(\hat{h}_{t+H}, \pi(\hat{h}_{t+H}))$, is exponentially decayed by $\gamma^H$.

**Core Idea**: Abandon explicit conservatism and commit fully to the Bayesian spirit—randomly sample one model from the posterior (fixing one model per rollout), use an adaptive uncertainty threshold to determine truncation, apply layer normalization to halt compounding errors, and use a recurrent actor-critic to handle the partial observability of the epistemic POMDP, ultimately generating long rollouts of several hundred steps.

## Method

### Overall Architecture
The training loop of Neubay (Algo. 1) is highly MBPO-like: (a) Train a 100-model ensemble $\mathbf{m}_{\boldsymbol{\theta}}$ on $\mathcal{D}$; (b) In each round, sample a starting point $h_t = s_{0:t}$ from $\mathcal{D}$, draw a fixed model $m_\theta$ from the ensemble (Note: this model is fixed for the entire rollout, not resampled at each step), and execute the Rollout in Algo. 2 until (i) a terminal state is reached, (ii) the uncertainty $U_{\boldsymbol{\theta}}(\hat s_t, \hat a_t) > \mathcal{U}(\zeta)$ triggers truncation, or (iii) the episode limit $T$ (up to 1000) is reached; (c) Mix real and imagined data at a ratio $\kappa$ to train a recurrent actor $\pi_\nu(a_t|h_t)$ and critic $Q_\omega(h_t, a_t)$ with independent LRU encoders via off-policy RL.

### Key Designs

1. **Adaptive Rollout Truncation via Uncertainty Quantile Threshold $\mathcal{U}(\zeta)$**:
    - **Function**: Determines where to stop a rollout, allowing it to run as long as possible within "trustworthy regions" and truncating immediately in untrustworthy ones.
    - **Mechanism**: First, calculate the distribution of ensemble disagreement $U_{\boldsymbol{\theta}}(s, a) = \mathrm{std}(\{\mu_{\theta^n}(s, a)\}_{n=1}^N)$ for all $(s, a)$ in the dataset, and set the $\zeta$-quantile as the threshold $\mathcal{U}(\zeta) := F_Y^{-1}(\zeta)$. During rollouts, truncate if $U_{\boldsymbol{\theta}}(\hat s_t, \hat a_t)$ exceeds this threshold (no penalty is applied; extrapolation simply stops). The paper uses $\zeta = 1.0$, taking the maximum in-dataset uncertainty to encourage the longest rollouts possible.
    - **Design Motivation**: Previous works (Pan et al., Zhan et al., Frauenknecht et al.) also used uncertainty thresholds but paired them with fixed short-horizon caps. This paper finds that **once explicit conservatism is removed, short horizons allow bootstrapping to dominate, leading to severe overestimation**; hence, fixed caps must be removed to let the threshold dictate length. Quantiles allow the threshold to adapt to different datasets (which vary greatly in scale and long-tail structure, see Fig. 4) more robustly than absolute thresholds.

2. **Large Ensemble ($N{=}100$) + LayerNorm in World Model to Control Compounding Error**:
    - **Function**: Makes long rollouts feasible by ensuring posterior fidelity while preventing step-wise error explosion.
    - **Mechanism**: The world model is formulated for delta prediction as $\mathbb{E}[\hat s'] = s + \mathbf{W}^\top \mathrm{ReLU}(\mathrm{LN}(\psi(s, a)))$. Since $\|\mathrm{LN}(x)\| = \sqrt{k}$ is constant when LN lacks an affine transform, the single-step increment $\|\mathbb{E}[\hat s'] - s\| \leq \sqrt{k}\|\mathbf{W}\|$ has a hard upper bound. After $H$ steps, the accumulation $\|\mathbb{E}[\hat s_H] - s_0\| \leq H\sqrt{k}\|\mathbf{W}\|$ grows linearly rather than exponentially. Furthermore, the ensemble size is increased from the MBPO default of 5 to 100 to compensate for the magnification of posterior fidelity issues under compounding errors in long rollouts.
    - **Design Motivation**: MBPO-like methods use small ensembles ($N=5$) and 1–5 step rollouts where the posterior is less critical. When the goal is 64–512 steps, the posterior must be more accurate and step error smaller. The LN approach is adapted from Ball et al., who used it in model-free RL to suppress extrapolation error; here it is transferred from "controlling Q-networks" to "controlling dynamics networks."

3. **Recurrent Actor-Critic + Memoroid (LRU) for Epistemic POMDP**:
    - **Function**: The Bayesian objective naturally transforms the environment into a POMDP (the agent does not know which model was sampled from the ensemble and must infer it from history), necessitating history-dependent policies and critics.
    - **Mechanism**: The actor and critic use independent RNN encoders ($\nu_\phi(h_t)$ and $\omega_\phi(h_t)$) based on Memoroid + LRU (Linear Recurrent Unit) to support efficient parallel memory up to 1000 steps. The RNN encoder learning rate $\eta_\phi$ is set **significantly smaller** than the MLP head's learning rate (swept from $3\text{e-}7$ to $1\text{e-}4$ in the paper) because representations are extremely sensitive to parameters under long histories.
    - **Design Motivation**: Prior works like CBOP/APE-V either used short-context GRUs or degraded to model-free approaches to avoid the POMDP. Neubay instead incorporates Memoroid + LRU, which have been proven in online POMDP tasks to handle thousand-step histories. To align with MBPO-style data mixing, a real data ratio $\kappa \in (0, 1)$ is introduced, with higher $\kappa$ used for higher quality data.

### Loss & Training
The RL loss follows a standard TD3+BC style recurrent off-policy actor-critic (details in Appendix E). The world model ensemble is trained to convergence using MLE and then frozen. Key hyperparameters: $\zeta$ (truncation threshold, default $1.0$), $\kappa$ (real data ratio, swept $[0.05, 0.95]$ by dataset), $\eta_\phi$ (RNN learning rate, swept $[3\text{e-}7, 1\text{e-}4]$ by benchmark), and $N=100$ (ensemble size). Each rollout uses a single fixed model $m_\theta \sim \mathbf{m}_{\boldsymbol{\theta}}$ (**not** randomized per step) to strictly adhere to the Bayesian objective $\mathbb{E}_{m \sim \mathbb{P}_\mathcal{D}}[J(\pi, m)]$.

## Key Experimental Results

### Main Results
D4RL locomotion (representative results, higher is better):

| Dataset | CQL | MOBILE | ReBRAC | CBOP (Bayesian) | **Neubay** |
|---|---|---|---|---|---|
| hp-random | 5.3 | 31.9 | — | 31.4 | 24.5 |
| wk-random | 5.4 | 17.9 | — | — | — |
| hc-random | 31.3 | 39.3 | 45.4 | 32.8 | 37.0 |

Overall, across 33 datasets (D4RL locomotion 12 + Adroit 6 + AntMaze 6 + NeoRL 9):

| Category | Performance |
|---|---|
| vs. Best Conservative Algorithms (MOBILE/RAMBO/ARMOR/ReBRAC) | On par |
| vs. Existing Bayesian Algorithms (APE-V/MAPLE/CBOP/MoDAP) | Significantly outperforms |
| New SOTA | 7 datasets |
| Strength Area | Low/medium quality datasets + medium coverage |

### Ablation Study

| Configuration | Performance | Description |
|---|---|---|
| Full Neubay ($\zeta{=}1.0$, $H$ adapted up to 64–512) | Optimal | Long horizon median of 64–512 steps in practice |
| Short Horizon Variant ($\zeta{=}0.9$) | Massive failure | Bootstrap dominates; Q-values skyrocket (Fig. 1 middle) |
| $\zeta=0.99/0.999$ | Intermediate | Both performance and Q-estimation are in between |
| Remove LayerNorm | Compounding error explosion | Long rollouts become unfeasible |
| Ensemble $N{=}5$ vs $N{=}100$ | Posterior distortion | Amplified under long rollouts |

### Key Findings
- **Long Horizon Actively Suppresses Overestimation**: Fig. 1 (middle) shows that as $\zeta$ increases, allowing for longer rollouts, the estimated Q-values on the offline dataset actually decrease while performance improves—completely reversing the dogma that "model-based RL must have short horizons."
- **When Bayesian Outperforms Conservative**: In low-quality datasets (e.g., random, low-coverage NeoRL Low) and scenarios where optimal actions are scarce, the Bayesian approach can adapt at test time to explore better actions. In high-quality datasets, the gap narrows. The bandit example theoretically confirms: conservatism on skewed data is **destined** to pick the observed suboptimal arm.
- **Fixed Model per Rollout is Critical**: The MBPO style of "sampling a model at every step" destroys posterior semantics; Neubay must use model-consistent rollouts to align with the Bayesian expected objective.
- **Ablation shows** that LN + large ensemble + long horizons are **all indispensable**; removing any one makes the long-horizon path non-viable.

## Highlights & Insights
- **The observation that "long horizons self-absorb overestimation" is counter-intuitive yet profound**: By decomposing the $H$-step TD target into $\sum \gamma^j \hat r$ (low bias) + $\gamma^H Q$ (high bias but exponentially decayed), it reveals that the H-axis is not just for error accumulation but also a leverage for bias decay. This suggests the MBRL community's focus on $H=1-5$ as a "safe zone" might have been a collective cognitive error.
- **Converting uncertainty from a "penalty" to a "switch"**: The same ensemble disagreement used as a reward subtraction in conservative routes is used here as a binary switch for rollout continuation—applying the same information differently yields completely different results, a transferable design philosophy.
- **LayerNorm shrinks the single-step geometric bound from $\|W\|\cdot\|\psi\|$ to constant $\sqrt{k}\|W\|$**: This approach of using normalization layers for Lipschitz control is applicable to all rollout-heavy model-based or world model architectures.
- **Quantile-based uncertainty threshold** $\mathcal{U}(\zeta) := F_Y^{-1}(\zeta)$: This makes thresholds automatically comparable across different dataset scales, making it much more robust than a fixed threshold $u_0$. It can be adapted to any truncation/filtering system dependent on OOD scores.

## Limitations & Future Work
- The algorithm requires sweeping two hyperparameters per dataset ($\eta_\phi$ and $\kappa$), maintaining a similar tuning cost to mainstream conservative MBRL (MOPO, RAMBO, MOBILE) without yet achieving "parameter-free" status.
- Running an $N{=}100$ ensemble with hundred-step rollouts and thousand-step RNN contexts results in a wall-clock time significantly higher than model-free algorithms like IQL/CQL, which is less friendly to labs with limited compute.
- The advantage lies in "low quality + medium coverage"; for high-quality, low-coverage "expert near-optimal" datasets, the Bayesian advantage is less pronounced or slightly inferior to conservative algorithms—this is a trade-off boundary rather than a bug.
- Current Bayesian posteriors are approximated only via deep ensembles; higher fidelity posteriors (e.g., SWAG, HMC) could be integrated in the future to improve performance on small data.

## Related Work & Insights
- **vs. MOBILE / RAMBO / COMBO (Conservative MBRL)**: These rely on the dual suppression of overestimation via uncertainty penalties and short horizons; Neubay discards penalties and uses long horizons for self-absorption, surpassing them on 7 datasets and showing structural dominance on low-quality data.
- **vs. APE-V / MAPLE / CBOP / MoDAP (Existing Bayesian-inspired)**: These papers often reintroduce conservatism for stability; Neubay is the first to maintain the Bayesian spirit through to completion and prove its viability on mainstream benchmarks.
- **vs. MBPO**: MBPO provides the model-based template, but uses $H=1-5$ steps, $N=5$ ensembles, and step-wise model sampling. Neubay reverses these three points, demonstrating that this is the "correct approach" in offline settings.
- **Insight**: This work contributes a new perspective to all "model-based + overestimation" problems—do not rush to add penalties or shorten horizons; first consider if the bias can be absorbed by the structure of the model rollouts themselves. This "solving problems through algorithmic self-consistency" approach is relevant for RLHF, world models, and long-form reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The claim that long horizons actively suppress overestimation overturns mainstream cognition; the bandit insight is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 33 datasets across 4 benchmarks, includes 4 ablation dimensions, and characterizes the boundaries of data quality/coverage.
- Writing Quality: ⭐⭐⭐⭐ Concepts are clear (bandit → three challenges → five designs), though the density of terminology (POMDP / BAMDP / robust MDP) requires prior knowledge.
- Value: ⭐⭐⭐⭐ Provides a viable "non-conservative" path for the offline RL community, potentially fostering a new generation of model-based offline RL algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Reinforcement Learning with Universal Horizon Models](offline_reinforcement_learning_with_universal_horizon_models.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ACL 2026\] A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks (EAGLET)](../../ACL2026/reinforcement_learning/a_goal_without_a_plan_is_just_a_wish_efficient_and_effective_global_planner_trai.md)
- [\[ICML 2026\] InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](inftythink_effective_and_efficient_infinite-horizon_reasoning_via_reinforcement_.md)
- [\[ICLR 2026\] Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning](../../ICLR2026/reinforcement_learning/strict_subgoal_execution_reliable_long-horizon_planning_in_hierarchical_reinforc.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism
description: >-
  [ICML 2026][Reinforcement Learning][offline RL] This work challenges the mainstream consensus that "offline RL must be explicitly conservative…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "offline RL"
  - "model-based RL"
  - "Bayesian RL"
  - "long-horizon rollout"
  - "epistemic POMDP"
date: 2026-05-08
content_hash: 027063fb96af3b53
---

# Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism

**Conference**: ICML 2026  
**arXiv**: [2512.04341](https://arxiv.org/abs/2512.04341)  
**Code**: https://github.com/twni2016/neubay (available)  
**Area**: Offline Reinforcement Learning / Model-Based RL / Bayesian RL  
**Keywords**: offline RL, model-based RL, Bayesian RL, long-horizon rollout, epistemic POMDP

## TL;DR
This work challenges the mainstream consensus that "offline RL must be explicitly conservative," and proposes Neubay: adopting a Bayesian perspective on the posterior model ensemble, using **long-horizon rollouts (hundreds of steps)** to naturally absorb value overestimation, and controlling compounding error via layer norm and uncertainty thresholds. As a result, Neubay matches SOTA conservative algorithms on 33 D4RL/NeoRL datasets without pessimistic penalties, and sets new records on 7 datasets.

## Background & Motivation

**Background**: Mainstream offline RL methods (CQL, IQL, EDAC, ReBRAC, MOPO, COMBO, MOBILE, etc.) are built on the principle of "explicit conservatism"—penalizing out-of-dataset state-actions, and in model-based methods, limiting rollouts to 1–5 short steps to suppress both value overestimation and compounding error. Theoretically, this corresponds to robust MDPs: $\max_\pi\min_{m\in\mathfrak{M}_\mathcal{D}}J(\pi,m)$.

**Limitations of Prior Work**: The conservative principle reduces overestimation but also suppresses average-case performance, especially on low-quality datasets: when the behavior policy is poor, conservative training is stuck around suboptimal actions and cannot explore better unseen actions at test time. Bayesian perspectives (epistemic POMDP, Ghosh et al.) theoretically enable test-time adaptation, but practical algorithms (APE-V, MAPLE, CBOP, MoDAP) reintroduce uncertainty penalties and short horizons for stability, diluting the Bayesian spirit back to conservatism.

**Key Challenge**: The Bayesian objective $\max_\pi \mathbb{E}_{m\sim \mathbb{P}_\mathcal{D}}[J(\pi, m)]$ theoretically requires full posterior rollouts, but in practice, compounding error and value overestimation become unmanageable without explicit conservatism, making Bayesian approaches "theoretically appealing but practically weak."

**Goal**: (1) Empirically demonstrate that a pure Bayesian approach (without any uncertainty penalty) can work on mainstream offline RL tasks; (2) Identify key bottlenecks and design corresponding mechanisms; (3) Provide a clear boundary for when to use Bayesian versus conservative approaches.

**Key Insight**: The authors use an extreme two-armed bandit example to clarify—conservatism is **doomed** to stick to observed suboptimal arms on skewed data, while Bayesian methods can adapt at test time. From this, they make a counterintuitive observation: **long-horizon rollouts themselves can substitute for explicit conservatism in suppressing value overestimation**, because the $H$-step TD target $\sum_{j=0}^{H-1}\gamma^j \hat{r}_{t+j+1} + \gamma^H Q(\hat{h}_{t+H}, \pi(\hat{h}_{t+H}))$ exponentially decays the highly overestimated bootstrap term by $\gamma^H$.

**Core Idea**: Abandon explicit conservatism and fully embrace the Bayesian spirit—randomly sample a model from the posterior (fixing one model per rollout), use adaptive uncertainty thresholds to determine truncation, apply layer norm to control compounding error, and employ recurrent actor-critic to handle the partial observability of epistemic POMDPs, ultimately enabling long rollouts of several hundred steps.

## Method

### Overall Architecture
Neubay's training loop (Algo. 1) is highly MBPO-like: (a) Train a 100-model ensemble $\mathbf{m}_{\boldsymbol{\theta}}$ on $\mathcal{D}$; (b) In each iteration, sample a starting point $h_t = s_{0:t}$ from $\mathcal{D}$, draw a fixed model $m_\theta$ from the ensemble (note: the same model is used for the entire rollout, not resampled at each step), and run the Rollout in Algo. 2 until (i) terminal state, (ii) uncertainty $U_{\boldsymbol{\theta}}(\hat s_t, \hat a_t) > \mathcal{U}(\zeta)$ triggers truncation, or (iii) episode limit $T$ (up to 1000 steps); (c) Mix real and imagined data at ratio $\kappa$, and feed to a pair of recurrent actor $\pi_\nu(a_t|h_t)$ and critic $Q_\omega(h_t, a_t)$ with independent LRU encoders for off-policy RL.

### Key Designs

1. **Uncertainty Quantile Threshold $\mathcal{U}(\zeta)$ for Adaptive Rollout Truncation**:

    - **Function**: Determines where to stop a rollout, allowing it to proceed as long as possible within the "trusted region" and truncating immediately in untrusted regions.
    - **Mechanism**: Compute the ensemble disagreement $U_{\boldsymbol{\theta}}(s, a) = \mathrm{std}(\{\mu_{\theta^n}(s, a)\}_{n=1}^N)$ for all $(s, a)$ in the dataset, and set the $\zeta$-quantile as the threshold $\mathcal{U}(\zeta) := F_Y^{-1}(\zeta)$. During rollout, if $U_{\boldsymbol{\theta}}(\hat s_t, \hat a_t)$ exceeds this threshold, truncate (no penalty, just stop extrapolation). The paper uses $\zeta = 1.0$, i.e., the maximum in-dataset uncertainty, encouraging the longest possible rollouts.
    - **Design Motivation**: Prior works (Pan et al., Zhan et al., Frauenknecht et al.) also used uncertainty thresholds, but always with a fixed short horizon cap; this work finds that **removing explicit conservatism makes short horizons dominated by bootstrap, leading to severe overestimation**, so any fixed cap must be removed and let the threshold adaptively determine the rollout length. The quantile form allows the threshold to automatically adapt across datasets with vastly different scales and tail structures (see Fig. 4), making it more robust than absolute thresholds.

2. **Large Ensemble ($N{=}100$) + LayerNorm in World Model to Control Compounding Error**:

    - **Function**: Makes long rollouts feasible—ensuring both posterior fidelity and bounded step-wise error.
    - **Mechanism**: The world model is formulated as a delta predictor $\mathbb{E}[\hat s'] = s + \mathbf{W}^\top \mathrm{ReLU}(\mathrm{LN}(\psi(s, a)))$. Since LN without affine ensures $\|\mathrm{LN}(x)\| = \sqrt{k}$, the single-step increment $\|\mathbb{E}[\hat s'] - s\| \leq \sqrt{k}\|\mathbf{W}\|$ is strictly bounded; after $H$ steps, $\|\mathbb{E}[\hat s_H] - s_0\| \leq H\sqrt{k}\|\mathbf{W}\|$ grows linearly, not exponentially. The ensemble size is increased from MBPO's default 5 to 100 to compensate for amplified compounding error under long rollouts.
    - **Design Motivation**: MBPO and similar methods use small ensembles ($N{=}5$) and short rollouts (1–5 steps), so posterior accuracy is less critical; but for 64–512 step rollouts, the posterior must be more accurate and step error much smaller. LN is inspired by Ball et al.'s work on controlling extrapolation error in model-free RL, here migrated from "controlling Q networks" to "controlling dynamics networks."

3. **Recurrent Actor-Critic + Memoroid (LRU) for Epistemic POMDP**:

    - **Function**: The Bayesian objective naturally turns the environment into a POMDP (the agent does not know which model from the ensemble is sampled and must infer from history), so both policy and critic must consume history.
    - **Mechanism**: Actor and critic each have independent RNN encoders ($\nu_\phi(h_t)$ and $\omega_\phi(h_t)$), using memoroid + LRU (linear recurrent unit) to efficiently support up to 1000-step histories. The RNN encoder learning rate $\eta_\phi$ is set **much lower** than the MLP head (the paper sweeps $3\text{e-}7$ to $1\text{e-}4$), as representation is extremely sensitive to parameters under long histories and diverges with larger rates.
    - **Design Motivation**: Previous works like CBOP/APE-V either use short-context GRUs or degenerate to model-free to avoid the POMDP; Neubay directly imports memoroid + LRU, proven in online POMDPs to handle thousand-step histories. To match MBPO-style real-vs-imagined data mixing, a real data mixing ratio $\kappa \in (0, 1)$ is introduced, increased for higher-quality data.

### Loss & Training
RL loss uses standard TD3+BC style recurrent off-policy actor-critic (details in Appendix E). The world model ensemble is trained to convergence via MLE and then frozen. Key hyperparameters: $\zeta$ (truncation threshold, default $1.0$), $\kappa$ (real data ratio, swept $[0.05, 0.95]$ per dataset), $\eta_\phi$ (RNN learning rate, swept $[3\text{e-}7, 1\text{e-}4]$ per benchmark), $N=100$ (ensemble size). Each rollout uses a fixed model $m_\theta \sim \mathbf{m}_{\boldsymbol{\theta}}$ (**not** resampled at each step), strictly matching the Bayesian objective $\mathbb{E}_{m \sim \mathbb{P}_\mathcal{D}}[J(\pi, m)]$.

## Key Experimental Results

### Main Results
D4RL locomotion (selected representative results, higher is better):

| Dataset | CQL | MOBILE | ReBRAC | CBOP (Bayesian) | **Neubay** |
|---|---|---|---|---|---|
| hp-random | 5.3 | 31.9 | — | 31.4 | 24.5 |
| wk-random | 5.4 | 17.9 | — | — | — |
| hc-random | 31.3 | 39.3 | 45.4 | 32.8 | 37.0 |

Across 33 datasets (D4RL locomotion 12 + Adroit 6 + AntMaze 6 + NeoRL 9):

| Category | Behavior |
|---|---|
| Compared to best conservative algorithms (MOBILE/RAMBO/ARMOR/ReBRAC) | on par |
| Compared to existing Bayesian algorithms (APE-V/MAPLE/CBOP/MoDAP) | significantly better |
| New SOTA | 7 datasets |
| Advantageous regime | low-quality + medium-quality + medium coverage |

### Ablation Study

| Configuration | Behavior | Notes |
|---|---|---|
| Full Neubay ($\zeta{=}1.0$, $H$ adaptively up to 64–512) | optimal | Median long horizon 64–512 steps |
| Short horizon variant ($\zeta{=}0.9$) | severe collapse | Bootstrap dominates, Q-values explode on dataset (see Fig. 1 middle) |
| $\zeta=0.99/0.999$ | intermediate | Performance and Q estimates are intermediate |
| Remove LayerNorm | compounding error explodes | Long rollouts infeasible |
| ensemble $N{=}5$ vs $N{=}100$ | small ensemble posterior distortion | Amplified under long rollouts |

### Key Findings
- **Long horizon actively suppresses overestimation**: Fig. 1 (middle) shows that larger $\zeta$ allows longer rollouts, resulting in lower Q-values and better performance on offline datasets—completely reversing the "model-based RL must use short horizons" dogma.
- **When Bayesian outperforms conservative**: On low-quality datasets (e.g., random, low-coverage NeoRL Low) and scenarios with scarce optimal actions, Bayesian methods can adaptively explore better actions at test time; the gap narrows on high-quality datasets. The bandit example theoretically confirms: conservatism is **doomed** to stick to observed suboptimal arms on skewed data.
- **Fixing one model per rollout is critical**: MBPO-style "resample model at each step" breaks posterior semantics; Neubay must use model-consistent rollouts to match the Bayesian expectation objective.
- Ablation shows LN + large ensemble + long horizon are **all essential**; removing any one makes long rollouts infeasible.

## Highlights & Insights
- **"Long horizon self-absorbs overestimation" is a counterintuitive but profound observation**: Decomposing the H-step TD target into $\sum \gamma^j \hat r$ (low bias) + $\gamma^H Q$ (high bias but exponentially decayed) reveals that the horizon $H$ is not just about error accumulation, but also a lever for bias decay. This suggests the community's belief that H=1-5 is the "safe zone" may be a collective misconception.
- **Repurposing uncertainty from "penalty" to "switch"**: The same ensemble disagreement, used as a reward penalty in conservative approaches, is here a binary switch for whether to continue rollout—the same information, used differently, yields entirely different effects, a transferable design philosophy.
- **LayerNorm shrinks the single-step geometric bound from $\|W\|\cdot\|\psi\|$ to $\sqrt{k}\|W\|$ by a constant factor**: This approach of using normalization layers for Lipschitz control is applicable to all rollout-heavy model-based/world model works.
- **Quantile-form uncertainty threshold** $\mathcal{U}(\zeta) := F_Y^{-1}(\zeta)$: Enables thresholds to be automatically comparable across datasets of different scales, much more robust than fixed thresholds $u_0$, and can be adopted in any OOD-score-based truncation/filtering system.

## Limitations & Future Work
- The algorithm requires sweeping two hyperparameters, $\eta_\phi$ and $\kappa$, per dataset, matching the tuning cost of mainstream conservative model-based RL methods (MOPO, RAMBO, MOBILE), but still not "tuning-free."
- Running an ensemble of $N{=}100$ with hundreds of rollout steps and thousand-step RNN contexts results in much higher wall-clock time per experiment than model-free methods like IQL/CQL, making it less accessible for small labs.
- The advantageous regime is "low-quality + medium coverage"; for high-quality, low-coverage "expert near-optimal" datasets, Bayesian methods show little or even negative advantage over conservative algorithms—this is a trade-off boundary, not a bug.
- The current Bayesian posterior is only approximated by deep ensembles, lacking more refined posteriors (e.g., SWAG, HMC); posterior fidelity may be insufficient for small data, and future work could incorporate better uncertainty quantification.

## Related Work & Insights
- **vs MOBILE / RAMBO / COMBO (conservative model-based RL)**: These rely on uncertainty penalties + short horizons to doubly suppress overestimation; Neubay completely abandons penalties and uses long-horizon self-absorption, surpassing them on 7 datasets and structurally outperforming on low-quality data.
- **vs APE-V / MAPLE / CBOP / MoDAP (existing Bayesian-inspired algorithms)**: These reintroduce conservatism for stability, becoming semi-conservative; Neubay is the first to fully adhere to the Bayesian spirit and demonstrate feasibility on mainstream benchmarks.
- **vs MBPO**: MBPO is the model-based RL template, but uses H=1-5 short horizons, $N=5$ small ensembles, and resamples models at each step; Neubay reverses all three and shows this is the "right approach" for offline settings.
- **Insights**: This work offers a new perspective for all "model-based + overestimation" problems—before adding penalties or shortening horizons, consider whether model rollout structure itself can absorb bias. This "let the algorithm mechanism solve the problem coherently" philosophy is instructive for RLHF, world models, and long reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The assertion that long horizons actively suppress overestimation overturns community consensus; the bandit example is also insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 benchmarks, 33 datasets, 4 ablation dimensions, and clear boundaries for data quality/coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear conceptual flow (bandit → three challenges → five designs), though occasionally dense in terminology (POMDP / BAMDP / robust MDP, etc. require prior knowledge).
- Value: ⭐⭐⭐⭐ Opens a "non-conservative" feasible path for the offline RL community, potentially inspiring the next generation of model-based offline RL algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning](../../ICLR2026/reinforcement_learning/strict_subgoal_execution_reliable_long-horizon_planning_in_hierarchical_reinforc.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[ICLR 2026\] ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](../../ICLR2026/reinforcement_learning/model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)

</div>

<!-- RELATED:END -->

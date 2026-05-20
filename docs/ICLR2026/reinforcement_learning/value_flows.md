---
title: >-
  [Paper Note] Value Flows
description: >-
  [ICLR 2026][Reinforcement Learning][distributional RL] Value Flows is the first work to introduce flow matching into distributional RL — it learns a vector field such that the induced probability density path automatical…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "distributional RL"
  - "flow matching"
  - "return distribution"
  - "uncertainty quantification"
  - "OGBench"
date: 2026-05-08
content_hash: e6cbc2b35a81fee9
---

# Value Flows

**Conference**: ICLR 2026
**arXiv**: [2510.07650](https://arxiv.org/abs/2510.07650)  
**Code**: [GitHub](https://github.com/chongyi-zheng/value-flows)  
**Area**: Reinforcement Learning / Distributional RL / Generative Models
**Keywords**: distributional RL, flow matching, return distribution, uncertainty quantification, OGBench

## TL;DR
Value Flows is the first work to introduce flow matching into distributional RL — it learns a vector field such that the induced probability density path automatically satisfies the distributional Bellman equation. Variance of the return distribution is efficiently estimated via a flow derivative ODE, enabling confidence-weighted prioritized learning. The method achieves an average 1.3× improvement in success rate across 62 OGBench tasks, and estimates return distributions 3× more accurately than C51/CODAC.

## Background & Motivation
**Background**: Standard RL compresses future returns into a scalar Q-value. Distributional RL (C51, QR-DQN, IQN) models the full return distribution, providing richer learning signals and enabling applications in exploration and safe RL.

**Limitations of Prior Work**:
   - **C51**: Discretizes the return distribution into fixed bins → limited resolution, unable to capture fine-grained distributional structure.
   - **IQN/QR-DQN**: Approximates the distribution with a finite number of quantiles → distributional information between quantiles is lost.
   - **Variance estimation is difficult**: Discretization-based methods struggle to accurately estimate return variance, which is critical for uncertainty quantification.
   - Modern generative models (diffusion / flow matching) have been successfully applied to trajectory and policy modeling, but have not yet been used for return distribution modeling.

**Key Challenge**: How can one learn a complete, continuous return distribution (rather than a discretized approximation), and efficiently extract expectations and variances from it to improve policy learning?

**Core Idea**: Flow matching is used to learn a vector field $v(z^t | t, s, a)$ over return distributions — a distributional conditional flow matching (DCFM) loss is constructed to satisfy the distributional Bellman equation, and the flow derivative ODE enables variance estimation without backpropagation through the ODE solver.

## Method

### Overall Architecture
Standard Gaussian noise $\epsilon$ → vector field $v(z^t | t, s, a)$ generates a flow ODE → probability density path $p(z^t | t, s, a)$ → converges to the return distribution $p_{Z^\pi}(z | s, a)$ at $t=1$. Training objective: DCFM loss (distributional conditional flow matching, analogous to TD learning). Inference: samples from the return distribution are obtained at $t=1$.

### Key Designs

1. **Distributional Conditional Flow Matching (DCFM) Loss**

    - Function: Trains the vector field $v$ such that its induced density path satisfies the distributional Bellman equation.
    - Mechanism: An update rule $v_{k+1}(z^t|t,s,a)$ is constructed so that the corresponding density $p_k$ is acted upon by the distributional Bellman operator $\mathcal{T}^\pi$. The DCFM loss is:
      $\mathcal{L}_{DCFM}(v, v_k) = \mathbb{E}_{(s,a,r,s') \sim D} [(v(z^t|t,s,a) - v_k(\frac{z^t-r}{\gamma}|t,s',a'))^2]$
    - Correspondence to TD learning: $v_k(\frac{z^t-r}{\gamma}|t,s',a')$ serves as the "bootstrap target" (analogous to $r + \gamma Q(s', a')$ in Q-learning).
    - **Proposition 2**: DCFM and the theoretical DFM loss share the same gradient (analogous to the CFM vs. FM relationship).
    - A target network $\bar{v}$ with a bootstrapped target (BCFM loss) is used to prevent collapse.

2. **Q-Value Estimation (Proposition 3)**

    - Function: Directly estimates the expected return from the vector field.
    - Formula: $\hat{\mathbb{E}}[Z^\pi(s,a)] \approx \mathbb{E}_{\epsilon \sim \mathcal{N}} [v(\epsilon | 0, s, a)]$ — the expectation of the vector field at $t=0$.
    - **A single forward pass** yields the Q-value, with no full ODE solve required.
    - Design Motivation: This allows Value Flows to serve directly as a critic in actor-critic frameworks.

3. **Variance Estimation (Flow Derivative ODE)**

    - Function: Estimates the variance of the return distribution for uncertainty quantification.
    - Mechanism: A companion ODE $d(\partial\phi/\partial\epsilon)/dt = (\partial v/\partial z) \cdot (\partial\phi/\partial\epsilon)$ is defined, where $\partial\phi/\partial\epsilon$ is the derivative of the flow with respect to the initial noise. At $t=1$, $|\partial\phi/\partial\epsilon|$ reflects local density changes → variance information.
    - **No backpropagation through the ODE solver is required** — forward-mode automatic differentiation or the companion ODE is used directly.
    - Design Motivation: Variance estimation in methods such as C51/IQN requires additional computation or approximation; here, variance is a natural byproduct of the flow matching framework.

4. **Confidence-Weighted Training**

    - Function: Uses variance estimates to prioritize learning on high-uncertainty transitions.
    - Weight: $w = \sigma(-\tau / |\partial\phi/\partial\epsilon|) + 0.5$
    - Large $|\partial\phi/\partial\epsilon|$ → rapid local density change → high variance → higher learning weight.
    - Realizes principled prioritized experience replay based on aleatoric uncertainty rather than bootstrapped error.

### Loss & Training
- Total loss: BCFM loss (bootstrapped DCFM, analogous to fitted Q-learning) + confidence weights.
- Target network updated via EMA.
- Policy extraction: advantage-weighted regression or SAC.
- Supports both offline and offline-to-online settings.

## Key Experimental Results

### OGBench (62 tasks: 37 state-based + 25 image-based)

| OGBench Domain | BC | IQL | ReBRAC | FQL | **Value Flows** |
|---|---|---|---|---|---|
| cube-double-play | 2 | 6 | 12 | 29 | **69±4** |
| puzzle-3x3-play | 2 | 9 | 22 | 30 | **87±13** |
| scene-play | 5 | 28 | 41 | 56 | **59±4** |
| **Average Success Rate** | — | — | — | — | **1.3× improvement** |

### Return Distribution Estimation Accuracy

| Method | 1-Wasserstein Distance ↓ |
|------|---------------------|
| C51 | ~0.09 |
| CODAC | ~0.06 |
| **Value Flows** | **~0.02** |

Value Flows achieves 4.5× better distribution estimation accuracy than C51 and 3× better than CODAC.

### Ablation Study

| Configuration | Effect | Note |
|------|------|------|
| Without confidence weighting | Performance drop | Demonstrates necessity of prioritizing high-uncertainty transitions |
| Without bootstrapped target | Degradation / collapse | DCFM alone is insufficiently stable |
| Q-value estimation vs. ensemble average | Value Flows more accurate | Single-network estimation is sufficient |
| Offline-to-online fine-tuning | Further improvement | Variance estimation naturally supports online exploration |

### Key Findings
- Flow matching provides **significantly more accurate** return distribution estimation than discretization-based (C51) and quantile-based (IQN) methods.
- Q-value estimation requires only a forward pass at $t=0$ — computational cost is comparable to a standard Q-network (no full ODE solve needed).
- Confidence weighting consistently improves performance, especially on play datasets with uneven data coverage.
- The method is effective on image-based tasks (all 25 image tasks improved), demonstrating compatibility with vision backbones.
- In the offline-to-online setting, variance estimates naturally provide exploration signals without requiring additional exploration strategies.

## Highlights & Insights
- **Elegant correspondence between flow matching and the distributional Bellman operator**: The DCFM loss is the continuous generative model counterpart of distributional TD learning — the vector field plays the role of the "critic" and the flow ODE corresponds to the "rollout." This theoretical connection is remarkably natural and elegant.
- **Variance as a byproduct**: Traditional distributional RL methods require additional mechanisms for variance estimation (e.g., ensembles, second-moment networks). Value Flows obtains variance naturally via the flow derivative ODE — a distinctive advantage of the flow matching framework.
- **Q-value from a single forward pass** (Proposition 3) is a key practical feature — inference is no slower than a standard Q-network, and ODE solving is only invoked when the full distribution is needed.

## Limitations & Future Work
- Epistemic uncertainty (from insufficient data) and aleatoric uncertainty (from environmental stochasticity) cannot be disentangled — the confidence weights reflect only aleatoric uncertainty.
- ODE solving introduces additional computational overhead during training and full distribution sampling (though Q-value estimation is unaffected).
- Evaluation is limited to continuous control (OGBench + D4RL); no discrete action space benchmarks such as Atari are included.
- The 1D return scalar is a relatively simple target for a generative model — the advantages of flow matching may be approaching saturation in this setting.
- Whether the method remains effective when scaled to larger action spaces and longer horizons requires further investigation.

## Related Work & Insights
- **vs. C51**: Discretizes the return distribution into 51 bins and optimizes via KL divergence; Value Flows uses continuous flow matching to model the density directly, achieving 4.5× higher accuracy.
- **vs. IQN**: Approximates the distribution with a finite number of quantiles; Value Flows learns a complete continuous distribution.
- **vs. CODAC**: Models the distribution with an ODE but outside the flow matching framework; Value Flows has a more principled theoretical foundation and achieves 3× higher accuracy.
- **Implications for generative models + RL**: Following trajectory generation (Diffuser) and policy generation (DDPO), Value Flows demonstrates the application of generative models on the critic side (value functions), completing the "generative model" treatment of actor-critic methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A novel combination of flow matching and distributional RL with an elegant theoretical foundation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 62 tasks (state + image) × 8 seeds × multiple baselines × distribution estimation accuracy × ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations; the progressive simplification from DFM → DCFM → BCFM is clearly presented.
- Value: ⭐⭐⭐⭐⭐ Opens a new generative model pathway for distributional RL; the variance-as-byproduct property has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)
- [\[ICLR 2026\] Transitive RL: Value Learning via Divide and Conquer](transitive_rl_value_learning_via_divide_and_conquer.md)
- [\[ICLR 2026\] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->

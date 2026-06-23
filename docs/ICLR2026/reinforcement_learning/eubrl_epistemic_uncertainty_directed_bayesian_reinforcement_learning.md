---
title: >-
  [Paper Note] EUBRL: Epistemic Uncertainty Directed Bayesian Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Paper Note] EUBRL directly integrates "epistemic uncertainty" into the RL objective function via probabilistic inference, utilizing a binary "uncertainty variable" to adaptively switch between exploration and exploitation. Theoretically, it is the first to achieve near-minimax optimal regret bounds and sample complexity simultaneo
tags:
  - ICLR 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: 4e0cafb4e928dfa7
---
# EUBRL: Epistemic Uncertainty Directed Bayesian Reinforcement Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=KASqlcI6Nm](https://openreview.net/forum?id=KASqlcI6Nm)  
**Code**: TBD  
**Area**: Reinforcement Learning / Bayesian Reinforcement Learning / Exploration  
**Keywords**: Epistemic Uncertainty, Bayesian RL, Probabilistic Inference, Minimax Optimal, Regret Bound, Sample Complexity  

## TL;DR
EUBRL directly integrates "epistemic uncertainty" into the RL objective function via probabilistic inference, utilizing a binary "uncertainty variable" to adaptively switch between exploration and exploitation. Theoretically, it is the first to achieve near-minimax optimal regret bounds and sample complexity simultaneously in infinite-horizon undiscounted MDPs.

## Background & Motivation
- **Background**: Exploration remains a core challenge in RL, evolving from $\epsilon$-greedy and Boltzmann exploration to "optimism in the face of uncertainty." Bayesian RL explicitly models uncertainty regarding transitions and rewards as posterior beliefs, where epistemic uncertainty naturally characterizes "unfamiliar parts of the environment," providing a principled foundation for exploration.
- **Limitations of Prior Work**: Mainstream optimistic methods add uncertainty as a bonus directly to the reward: $\tilde r = r_b + \eta r_{\text{bonus}}$. however, when $r_b$ itself is inaccurately estimated, small errors in the reward are amplified through value function propagation, resulting in **unnecessary exploration and slow convergence**. Existing Bayesian methods (e.g., PSRL, BEB, VBRB) make limited use of posterior uncertainty and lack systematic empirical evidence of their exploration capabilities.
- **Key Challenge**: Higher uncertainty suggests a greater need to explore, yet higher uncertainty also implies that reward estimations are less reliable. Coupling both into a single reward scalar leads to mutual contamination between the "intent to explore" and "distrust in estimation."
- **Goal**: To design an exploration algorithm for infinite-horizon discounted MDPs that is simple, compatible with arbitrary Bayesian models, and achieves near-minimax optimal regret bounds and sample complexity.
- **Key Insight**: **[Epistemic Guidance]** Decouple exploration and exploitation using probabilistic inference. By introducing a binary "uncertainty variable" $U$ and marginalizing it to obtain a lower bound on the stepwise likelihood, the reward is rewritten as a convex combination of an "exploitation term" and an "epistemic uncertainty term" weighted by the uncertainty probability. This biases the agent toward exploration when uncertain and toward exploitation when confident.

## Method

### Overall Architecture
EUBRL iterates alternately within the mean-MDP framework: at each step, it updates belief $b$ in closed form using conjugate priors to obtain the posterior predictive transition $P_b$ and reward $r_b$. The reward is then replaced by the "epistemic directed reward" $r_b^{\text{EUBRL}}$ to construct an MDP $\mathcal M=(S,A,P_b,r_b^{\text{EUBRL}},\gamma)$, and the policy is solved via value iteration. The primary difference lies not in the algorithmic skeleton, but in the **reward formulation**, which is derived from probabilistic inference rather than heuristic bonuses.

```mermaid
flowchart LR
    A[Prior b0] --> B[Closed-form Belief Update]
    B --> C[Posterior Predictive Pb, rb]
    C --> D[Compute Epistemic Uncertainty Eb]
    D --> E[Epistemic Directed Reward r_EUBRL]
    E --> F[Construct mean-MDP Value Iteration for π]
    F --> G[Interaction & Sampling]
    G --> B
```

### Key Designs

**1. Generalized Epistemic Uncertainty $E_b$: Unifying Environmental Unfamiliarity.** Epistemic uncertainty measures the "degree of disagreement" of model parameters $w$ within a belief. For transitions, it is defined as $E_T(s,a)=f\circ g(P_b(s'|s,a))-\mathbb E_{w\sim b}[f\circ g(P(s'|s,a,w))]$. By selecting different $f,g$, this simplifies to common metrics like **variance** ($f(x)=-x^2$) or **mutual information** ($f=H$); $E_R$ is similarly defined for rewards. Both sources are aggregated into a unified measure $E_b(s,a):=h(E_T,E_R)$ where $h(x,y)=\eta(\sqrt x+\sqrt y)$. This abstraction allows the algorithm to integrate with any uncertainty metric.

**2. Probabilistic Inference + Epistemic Directed Reward: Decoupling Exploration and Exploitation.** Standard RL can be framed as an inference problem by introducing a binary "optimality" variable $O_t$, where $P(O_t=1|s_t,a_t)\propto\exp(r(s_t,a_t))$, followed by maximizing $\log\prod_t P(O_t=1)$. EUBRL further introduces a binary "uncertainty variable" $U_t$. By marginalizing $U_t$ and applying Jensen's inequality to obtain a lower bound $\log P(O_t=1)\ge \mathbb E_{U_t}[\log P(O_t=1|s_t,a_t,U_t)]$, the core epistemic directed reward is derived:

$$r_b^{\text{EUBRL}}(s,a):=(1-P(U=1|s,a))\,r_b(s,a)+P(U=1|s,a)\,E_b(s,a).$$

This is a **convex combination**—not a summation—of the "exploitation term $r_b$" and "exploration term $E_b$" weighted by the uncertainty probability $P_U$. When uncertain, the agent ignores reward estimates to focus on exploration; when confident, it commits to exploiting learned knowledge, making it more robust to unreliable reward estimates. Here, $P(U=1|s,a)=E_b(s,a)/E_{\max}$ naturally transitions from "initial indifference to rewards" to "eventual firm exploitation" as evidence accumulates.

**3. General Algorithm Recipe: Adaptive Combination of Resets and Policy Updates.** Algorithm 1 alternates between "posterior updates ↔ policy learning," leveraging conjugacy for closed-form belief updates, epistemic uncertainty, and posterior predictions. By adjusting the combination of "when to reset" and "when to update policy," the **same recipe covers both infinite-horizon discounted MDPs (stepwise updates, no resets) and finite-horizon episodic MDPs (updates every $H$ steps with resets)**. It avoids complex designs like knownness or customized bonuses found in optimistic methods and can be paired with any Bayesian model.

### Theoretical Analysis (Core Conclusions)
Stepwise regret is decomposed as $V^\star-V^{\pi_t}=\underbrace{V^\star-\tilde V^t}_{\text{quasi-optimism}}+\underbrace{\tilde V^t-V^t}_{\text{complexity}}+\underbrace{V^t-V^{\pi_t}}_{\text{accuracy}}$. The paper proves that stepwise regret is bounded by a term involving "**epistemic resistance** $\mathcal R_t$"; higher action uncertainty leads to lower stepwise regret (Theorem 1). For infinite-horizon discounted MDPs, the frequentist regret bound is $\tilde O\big(\sqrt{SAT}/(1-\gamma)^{1.5}+S^2A/(1-\gamma)^2\big)$ (Theorem 2, improving upon He et al. 2021) and the sample complexity is $\tilde O\big(SA/(\epsilon^2(1-\gamma)^3)+\cdots\big)$ (Theorem 3). These results extend to a class of "decomposable/weakly informative" priors (satisfied by Dirichlet+Normal conjugates), reaching near-minimax optimality (Theorem 4 / Corollary 1). The authors also identify failure cases: Normal-Gamma priors may result in epistemic uncertainty decaying to zero in near-deterministic environments, violating quasi-optimism (Prop. 1), and convergence may fail under severe prior misspecification (Theorem 5).

## Key Experimental Results

### Main Results (Chain and Loop, 500 seeds × 1000 steps)

| Algorithm | Chain Avg. Reward | Loop(2) Avg. Reward |
|---|---|---|
| PSRL | 3158 | 377 |
| RMAX | 3090 | 394 |
| Mean-MDP | 3078 | 233 |
| BEB | 3430 | 386 |
| MBIE-EB | 3462 | — |
| VBRB | 3465 | — |
| **EUBRL** | **3473 (SE 16)** | **395 (SE 0.04)** |

EUBRL achieves the highest rewards with extremely low variance across classic tasks. Mean-MDP (without bonuses) consistently performs worst, confirming that efficient exploration requires reward bonuses.

### Ablation Study

| Setting | Observation |
|---|---|
| Increasing Loop Cycles (Sparser) | Even with perfect priors, RMAX scales poorly compared to EUBRL, suggesting a "smoothing" effect from Bayesian priors. |
| DeepSea (Deterministic/Stochastic) | EUBRL shows superior sample efficiency, scalability, and consistency; **EUBRL+ fully solves the stochastic variant with zero failures** (unprecedented). |
| LazyChain (Long-horizon+Sparse) | EUBRL leads consistently and remains robust under heavy noise injection. |
| Tied Prior (Global Shared Dirichlet) | Requires fewer samples for convergence and achieves higher success rates. |
| MI vs. Variance for Uncertainty | MI requires slightly more steps but achieves the **highest overall success rate**, being more exploratory. |

### Key Findings
- Formulating uncertainty within the **objective function** (convex combination) rather than the **reward** (additive bonus) is more robust to unreliable estimations. PSRL fluctuates near convergence and scales poorly due to excessive sampling frequency.
- Bayesian priors provide a smoothing effect in sparse environments, allowing the method to scale gracefully with problem size.

## Highlights & Insights
- **Simplicity with Strong Theory**: A single "convex combination reward" achieves near-minimax optimal frequentist regret and sample complexity for infinite horizons. It is claimed to be the **first online algorithm to reach this sample complexity without generative model assumptions**.
- **True Decoupling**: The uncertainty probability $P_U$ naturally manages the phase transition from "ignoring rewards early on" to "firm exploitation later," replacing manual annealing.
- **Honesty in Failure Modes**: The authors proactively provide counterexamples for Normal-Gamma degradation and prior misspecification, emphasizing the criticality of $\eta$ and prior selection.

## Limitations & Future Work
- Experiments are limited to tabular environments (Chain, Loop, DeepSea); large-scale/continuous states require tree search or function approximation, which remain unverified.
- Dependency on conjugate priors for closed-form updates; prior misspecification (low epistemic uncertainty) can lead to non-convergence.
- Open Question: How to capture epistemic uncertainty across multiple levels (hierarchies) to further reduce reliance on manual rewards.

## Related Work & Insights
- **Bayesian RL**: BAMDP (Duff 2002), mean-MDP (Poupart 2006), PSRL (Strens 2000), VBRB (Sorg 2012; most similar to EUBRL but uses only variance and lacks epistemic guidance).
- **Provably Efficient RL**: knownness/PAC-MDP (Kakade 2003; Strehl & Littman 2008), He et al. 2021 (Regret optimality for infinite horizon), quasi-optimism (Lee & Oh 2025; the foundation of this paper's analysis).
- **Uncertainty Quantification**: Variance (Kendall & Gal 2017) and Mutual Information (Hüllermeier & Waegeman 2021) as epistemic measures, echoing curiosity and surprise in cognitive science.
- **Insight**: Reformulating "intrinsic motivation" from a reward bonus into a probabilistic inference term in the objective is an elegant paradigm for handling the conflict between uncertainty driving exploration and contaminating estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The convex combination form of the objective using probabilistic inference is novel and achieves theoretical optimality for infinite horizons.
- Experimental Thoroughness: ⭐⭐⭐ — Tasks are well-targeted (sparse/long-horizon), but limited to tabular settings without large-scale or Deep RL validation.
- Writing Quality: ⭐⭐⭐⭐ — Derivations are clear, motivation is progressive, and the inclusion of failure modes strengthens theoretical integrity.
- Value: ⭐⭐⭐⭐ — Provides a simple recipe with both theoretical guarantees and empirical advantages, holding methodological value for Bayesian RL exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[ICLR 2026\] Learning to Be Uncertain: Pre-training World Models with Horizon-Calibrated Uncertainty](learning_to_be_uncertain_pre-training_world_models_with_horizon-calibrated_uncer.md)
- [\[ICLR 2026\] Geometry of Uncertainty: Learning Metric Spaces for Multimodal State Estimation in RL](geometry_of_uncertainty_learning_metric_spaces_for_multimodal_state_estimation_i.md)
- [\[ICML 2026\] PAC-Bayesian Reinforcement Learning Trains Generalizable Policies](../../ICML2026/reinforcement_learning/pac-bayesian_reinforcement_learning_trains_generalizable_policies.md)
- [\[ICLR 2026\] DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty](dr-sac_distributionally_robust_soft_actor-critic_for_reinforcement_learning_unde.md)

</div>

<!-- RELATED:END -->

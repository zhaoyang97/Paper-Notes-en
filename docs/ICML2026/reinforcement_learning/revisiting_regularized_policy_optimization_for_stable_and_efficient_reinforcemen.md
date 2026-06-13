---
title: >-
  [Paper Note] Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games
description: >-
  [ICML 2026][Reinforcement Learning][Regularized Policy Optimization] KLENT recombines three established components—reverse-KL regularization (controlling policy update magnitude)…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Regularized Policy Optimization"
  - "Reverse-KL"
  - "Entropy Regularization"
  - "$\\lambda$-return"
  - "Search-free AlphaZero"
date: 2026-05-08
content_hash: d758f670c12dabb3
---

# Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games

**Conference**: ICML 2026  
**arXiv**: [2602.10894](https://arxiv.org/abs/2602.10894)  
**Code**: TBD  
**Area**: Reinforcement Learning / Self-Play / Board Games  
**Keywords**: Regularized Policy Optimization, Reverse-KL, Entropy Regularization, $\lambda$-return, Search-free AlphaZero  

## TL;DR
KLENT recombines three established components—reverse-KL regularization (controlling policy update magnitude), entropy regularization (maintaining exploration), and $\lambda$-return (balancing bias and variance)—into model-free self-play RL. It achieves $4\times$ higher training efficiency than Gumbel AlphaZero across 5 board games and provides convergence proofs for both normal-form and finite-length scenarios.

## Background & Motivation
**Background**: Two-player zero-sum board games are predominantly dominated by the AlphaZero family (AlphaZero / MuZero / Gumbel AlphaZero / TRPO-AlphaZero, etc.), which follows the "network + MCTS" paradigm. These methods rely on look-ahead search to generate strong policy targets, but training costs are extremely high (e.g., AlphaZero requires 10+ GPU-years to converge), making reproduction difficult.

**Limitations of Prior Work**: Methods like Muesli and Gumbel AlphaZero only "shorten/shallow" MCTS; the most expensive component—search—remains. Conversely, pure model-free methods (PPO, DQN) have long been considered "unstable" in self-play, and few have systematically compared them on board games.

**Key Challenge**: Search provides stability but is expensive; removing search leads to instability. This instability stems from the non-stationary nature of self-play (changing opponents) and test-time distribution shifts. Aggressive policy updates lead to collapse, while insufficient exploration causes overfitting to oneself.

**Goal**: (i) Design a model-free self-play algorithm that does not rely on MCTS; (ii) Provide provable stability guarantees explaining why "reverse-KL + entropy" works; (iii) Empirically demonstrate higher efficiency than search-based methods across various board games.

**Key Insight**: The authors leverage the insight from Grill et al. (2020) that AlphaZero is essentially solving a KL-regularized policy optimization problem implicitly. They reverse this equivalence: if KL regularization is the essence of AlphaZero's stability, explicitly using KL-regularized policy optimization (with entropy regularization for distribution shift) can eliminate the expensive MCTS "approximate solver."

**Core Idea**: Treat "self-play = a closed-form policy improvement with reverse-KL + entropy regularization" as the foundation. Neural networks are used to approximate the policy and $Q$-function, and $\lambda$-return is employed to control variance in value learning, resulting in a search-free yet stable self-play model-free algorithm, KLENT.

## Method

### Overall Architecture
KLENT simultaneously parameterizes a policy $\pi_\theta(a|s)$ and an action-value function $Q_\theta(s,a)$ (in contrast to AlphaZero, which only learns $V(s)$ and estimates $Q$ via MCTS). Training cycles between two phases: (i) **Self-play phase**: The network calculates $\pi'$ (the closed-form regularized best response, see Equation 3), samples actions according to $\pi'$ to complete games, and records $(S_t, A_t, \{\pi'(a|S_t)\}_a, G_t^\lambda)$ at each step, storing the $\lambda$-return as the value target in buffer $\mathcal{D}$. (ii) **Fitting phase**: $\mathcal{D}$ is used to simultaneously update the policy (fitting $\pi'$ as the target) and $Q$ (fitting $G^\lambda$). **No MCTS** is used during training—MCTS is only optionally used during test-time evaluation.

### Key Designs

1. **Closed-form Policy Update with Dual Reverse-KL + Entropy Regularization**:
    - **Function**: Addresses the two core pain points of self-play—non-stationarity and distribution shift—using two regularization terms, providing a closed-form $\pi'$ to avoid iterative solving.
    - **Mechanism**: At each state $s$, solve $\max_{\pi'} \mathbb{E}_{A\sim\pi'}[Q^\pi(s,A)] - \beta D_{\text{KL}}(\pi'(\cdot|s)\|\pi(\cdot|s)) + \alpha H(\pi'(\cdot|s))$. Reverse-KL pins $\pi'$ near the current $\pi$ (gradual update against the "changing opponent"), while entropy regularization spreads probability mass (against unseen test-time opponents). Given the finite action space, the analytical solution is $\pi'(a|s)=\frac{1}{Z(s)}\exp\big(\frac{Q^\pi(s,a)+\beta\log\pi(a|s)}{\alpha+\beta}\big)$, where $Z(s)$ is the normalization constant. The policy network distills towards $\pi'$ by minimizing cross-entropy $-\sum_a \pi'(a|s)\log\pi_\theta(a|s)$.
    - **Design Motivation**: Reverse-KL is mode-seeking (unlike mean-seeking forward-KL), making it better for finding the best response; entropy regularization ensures $\pi'$ does not collapse to a deterministic policy, preventing cycles or overfitting in self-play.

2. **$\lambda$-return as the $Q$ Learning Target**:
    - **Function**: Balances bias and variance in value estimation under sparse $\pm 1$ terminal rewards.
    - **Mechanism**: Instead of Monte Carlo returns ($\lambda=1$, high variance) or TD(0) ($\lambda=0$, high bias), KLENT uses the $\lambda$-return $G^\lambda$ as the target for $Q_\theta$. Bias-variance experiments on 9x9 Go confirmed an intermediate $\lambda$ minimizes the sum of squared bias and variance. The final loss is $L(\theta)=\mathbb{E}_{\mathcal{D}}\big[-\sum_a \pi'(a|S)\log\pi_\theta(a|S) + (Q_\theta(S,A)-G^\lambda)^2\big]$, with $\lambda=e^{-1/8}$ used across all 5 games.
    - **Design Motivation**: Long trajectories and terminal-only rewards lead to variance explosion in MC returns, while TD(0) suffers from high bias due to bootstrapping and policy drift. $\lambda$-return, a classic solution, is brought back as a key efficiency driver.

3. **Convergence Proofs in Dual Scenarios**:
    - **Function**: Theoretically explains why this combination is stable.
    - **Mechanism**: (a) In normal-form two-player zero-sum games, the update rule achieves **local linear convergence** to a unique fixed point when $\alpha(\alpha+2\beta) > \|R\|_2^2/4$, proven by analyzing the Jacobian spectral radius < 1 at the fixed point. (b) In finite-length games, KLENT's policy **converges to an entropy-regularized optimal policy** $\pi(a|s)=\frac{1}{Z(s)}\exp(Q^\pi(s,a)/\alpha)$, proven via backward induction from terminal states. As $\alpha\to 0$, this equilibrium approaches the Nash equilibrium.
    - **Design Motivation**: The authors emphasize that their contribution lies in the new theoretical and empirical characterization of this specific combination in two-player games.

### Loss & Training
Hyperparameters $(\alpha, \beta, \lambda) = (0.03, 0.1, e^{-1/8})$ are shared across all games. Models use 6-block ResNets (20-block for 19x19 Go). Efficiency is compared using "simulator evaluations" as the x-axis to ensure fairness. Test-time evaluation uses reactive policies (no MCTS) to isolate performance gains.

## Key Experimental Results

### Main Results
Baselines: AlphaZero (AZ), TRPO-AlphaZero, Gumbel AlphaZero (Gumbel AZ), DQN, PPO. Games: Animal Shogi, Gardner Chess, 9x9 Go, Hex, Othello. Win rates against an anchored baseline after 800M simulator evaluations (with 800-rollout MCTS at test-time):

| Game | AZ | Gumbel AZ | **KLENT** |
|------|----|-----------|-----------|
| Animal Shogi | 31±2% | 67±5% | 63±4% |
| Gardner Chess | 64±3% | 70±1% | **81±1%** |
| 9x9 Go | 7±2% | 37±2% | **89±1%** |
| Hex | 8±5% | 47±5% | **98±1%** |
| Othello | 51±2% | 47±3% | **55±6%** |
| **Average** | 32.2% | 53.6% | **77.2%** |

Efficiency: KLENT reaches a 50% win rate in ~75M simulator evaluations on average, whereas Gumbel AlphaZero requires ~300M—a **4× efficiency gain**. KLENT's advantage is most pronounced in games with high branching factors (9x9 Go: 42.3, Hex: 90.6).

### Ablation Study
| Variant | Change | Result |
|------|------|------|
| **KLENT (full)** | All components active | Optimally consistent across 5 games |
| KL Only ($\alpha=0$) | No entropy reg | Win rate on Animal Shogi peaks at 75% then **crashes**; entropy drops to zero. |
| ENT Only ($\beta=0$) | No KL reg | $D_{\text{KL}}(\pi'\|\pi)$ spikes; policy fluctuates violently; significant drop on 9x9 Go. |
| 1-Step KLENT ($\lambda=0$) | TD(0) | Significant performance drop on 9x9 Go / Hex. |
| Monte Carlo KLENT ($\lambda=1$) | MC return | Performance drop on 9x9 Go / Hex due to variance. |

### Key Findings
- Without entropy regularization, the policy entropy in Animal Shogi collapses, leading to an unstable "rise and fall" curve—refuting the idea that self-play does not need explicit exploration.
- Without KL regularization, $D_{\text{KL}}(\pi'\|\pi)$ escapes control, verifying that reverse-KL is essential for maintaining gradual updates in non-stationary self-play.
- The $\lambda$ ablation shows that $\lambda$-return is an essential component for efficient model-free self-play, not just a minor trick.
- Efficiency gains correlate with branching factor: larger branching factors favor KLENT as MCTS budgets are stretched too thin.

## Highlights & Insights
- While Grill et al. (2020) characterized AlphaZero as implicit KL regularization, KLENT operationalizes this by explicitly using KL to remove MCTS, turning a theoretical insight into an engineering victory.
- The paper acknowledges its components are not new, but the packaging through convergence proofs (covering a wider $(\alpha,\beta)$ range for normal-form games and using backward induction for finite-length games) elevates the work to a substantial contribution.
- Learning $\pi$ and $Q$ simultaneously using $\pi'$ distillation is essentially the two-player version of MPO, suggesting that MPO-style frameworks are highly effective for board games.

## Limitations & Future Work
- The focus is on "efficiency"; whether AlphaZero-style methods retain superior asymptotic performance when given infinite compute remains an open question.
- Convergence proofs are split: local linear convergence for normal-form and backward induction for finite-length games (assuming acyclic state graphs). More complex rules (e.g., repeating positions) require further rigor.
- The study focuses on perfect-information games. Extending this to imperfect-information settings (e.g., Poker) may require adaptive regularization strengths.

## Related Work & Insights
- **vs AlphaZero / Gumbel AlphaZero**: These use MCTS as an approximate policy improvement operator; KLENT replaces this with a closed-form $\pi'$, explicitly realizing the implicit KL regularization.
- **vs MPO (Abdolmaleki 2018)**: KLENT is effectively a two-player zero-sum variant of MPO, with added entropy regularization and game-theoretic convergence proofs.
- **vs SAC / Soft Q-Learning**: These focus on entropy regularization ($\beta=0$). KLENT's "ENT Only" ablation shows that self-play requires both terms for stability.
- **vs Muesli / TRPO-AlphaZero**: These attempt to reduce search; KLENT reaches the logical conclusion of this trajectory: zero search.

## Rating
- Novelty: ⭐⭐⭐ (Existing components, novel combination + dual-scenario proofs + engineering execution).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 games + 19x19 Go + detailed ablations + fair test-time MCTS comparison).
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear mapping of regularization to failure modes and transparent discussion of contributions).
- Value: ⭐⭐⭐⭐ (Significant for resource-constrained labs by demystifying the need for MCTS in board games).

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICML 2026\] Learning to Route Languages for Multilingual Policy Optimization](learning_to_route_languages_for_multilingual_policy_optimization.md)
- [\[ICML 2026\] Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning](convergence_of_two-timescale_markovian_stochastic_approximations_with_applicatio.md)
- [\[ICML 2026\] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)

</div>

<!-- RELATED:END -->

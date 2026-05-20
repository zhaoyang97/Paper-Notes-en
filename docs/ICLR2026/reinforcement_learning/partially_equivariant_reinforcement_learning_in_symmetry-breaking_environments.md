---
title: >-
  [Paper Note] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments
description: >-
  [ICLR 2026][Reinforcement Learning][Partial equivariance] This paper proposes the Partially Invariant MDP (PI-MDP) framework, which employs a learnable gating function $\lambda(s…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Partial equivariance"
  - "symmetry breaking"
  - "group-invariant MDP"
  - "gated policy"
  - "Bellman error propagation"
date: 2026-05-08
content_hash: ffba62141d07f822
---

# Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments

**Conference**: ICLR 2026
**arXiv**: [2512.00915](https://arxiv.org/abs/2512.00915)  
**Code**: [Project Page](https://pranaboy72.github.io/perl_page/)  
**Area**: Reinforcement Learning / Equivariance
**Keywords**: Partial equivariance, symmetry breaking, group-invariant MDP, gated policy, Bellman error propagation

## TL;DR

This paper proposes the Partially Invariant MDP (PI-MDP) framework, which employs a learnable gating function $\lambda(s,a)$ to pointwise switch between equivariant and standard Bellman updates across the state-action space. The paper theoretically proves that local symmetry breaking propagates through discounted backup and amplifies global value function error by a factor of $1/(1-\gamma)$, while PI-MDP provably confines the error strictly within the breaking region. The framework is instantiated as PE-DQN and PE-SAC, achieving comprehensive improvements over strictly equivariant and approximately equivariant baselines on Grid-World, MuJoCo locomotion, and robotic manipulation tasks.

## Background & Motivation

**Background**: Group equivariance provides a powerful inductive bias for reinforcement learning. By constructing group-invariant MDPs — requiring that reward function $R(s,a)=R(gs,ga)$ and transition kernel $P(s'|s,a)=P(gs'|gs,ga)$ hold for all group elements $g \in G$ — equivariant networks enable zero-shot generalization across symmetric states, substantially improving sample efficiency. Existing equivariant RL works (e.g., EMLP-based RPP, equivariant DQN) are built on the premise that the environment fully satisfies the group-invariant assumption.

**Limitations of Prior Work**: Real-world control tasks can almost never fully satisfy group-invariant conditions. In robotic control, ground contact forces break vertical symmetry, actuator torque limits break joint symmetry, and the presence of obstacles breaks spatial rotational symmetry. The critical issue is that even when symmetry is broken only in a local region of the state-action space, conventional equivariant RL produces incorrect value estimates in that region, and this local error propagates and amplifies throughout the entire space via Bellman backup, ultimately causing global policy degradation or training failure.

**Key Challenge**: Strictly equivariant methods introduce uncontrollable errors in breaking regions; existing approximately equivariant methods (e.g., RPP, which globally relaxes equivariance constraints via residual pathways) provide some robustness, but their "globally uniform relaxation" strategy either sacrifices sample efficiency in fully symmetric regions or remains unstable under severe breaking — because such methods cannot distinguish "where symmetry holds and where it does not."

**Goal**: (1) Quantify how local symmetry breaking propagates through the Bellman operator into global value function error; (2) Design a framework that can pointwise select between equivariant and standard updates across the state-action space; (3) Automatically detect symmetry-breaking regions in a data-driven manner without prior knowledge.

**Key Insight**: The authors observe that the deviation between the group-invariant MDP $\mathcal{M}_E$ and the true MDP $\mathcal{M}_N$ can be precisely described by pointwise reward deviation $\epsilon_R(s,a)$ and transition deviation $\epsilon_P(s,a)$. If one can revert to standard updates in regions where $\epsilon > 0$, error propagation can be blocked at the source.

**Core Idea**: A learnable binary gating function $\lambda(s,a)$ automatically selects between equivariant and standard Bellman updates at each state-action pair, preserving sample efficiency in symmetric regions while preventing errors in breaking regions from propagating outward.

## Method

### Overall Architecture

PERL (Partially Equivariant RL) maintains two parallel sets of value function/policy networks — one satisfying group equivariance constraints $(Q_E, \pi_E)$ and one unconstrained standard network $(Q_N, \pi_N)$ — along with a gating function $\lambda_\omega(s,a) \in \{0,1\}$ that determines whether each state-action pair lies in a symmetry-breaking region. The final Q-value and policy perform hard switching between the two networks via $\lambda$: equivariant networks are used in symmetric regions, and standard networks in breaking regions. Training is conducted in the true environment $\mathcal{M}_N$, and the gating function receives supervision from the disagreement between two one-step predictors.

### Key Designs

1. **Theoretical Analysis of Local-to-Global Error Propagation**:

    - **Function**: Provides the theoretical foundation for why selective equivariance is necessary.
    - **Mechanism**: Defines pointwise deviations between the true MDP and the group-invariant MDP: $\epsilon_R(s,a) = |R_N(s,a) - R_E(s,a)|$ and $\epsilon_P(s,a) = \frac{1}{2}\int|P_N(s'|s,a) - P_E(s'|s,a)|ds'$. Lemma 1 proves that the single-step Bellman error is $\leq \epsilon_R(s,a) + 2\gamma V_{\max}\epsilon_P(s,a)$. Proposition 1 further proves that the global error of the optimal value function satisfies $\|Q_N^* - Q_E^*\|_\infty \leq \frac{1}{1-\gamma}\|\delta\|_\infty$, i.e., local error is amplified by $(1-\gamma)^{-1}$ through backup to affect the global value function.
    - **Design Motivation**: This theoretical result clearly identifies the root cause of strictly equivariant RL's failure in breaking environments — not that equivariance itself is harmful, but that local MDP mismatch is amplified into a global problem via Bellman backup.

2. **Partially Invariant MDP (PI-MDP) Framework**:

    - **Function**: Formally defines the concept of "selective equivariance" at the MDP level.
    - **Mechanism**: Introduces gating function $\lambda: \mathcal{S}\times\mathcal{A} \to [0,1]$ and defines mixed reward $R_H = (1-\lambda)R_E + \lambda R_N$ and transition kernel $P_H = (1-\lambda)P_E + \lambda P_N$. Theorem 1 proves that the PI-MDP Bellman operator $\mathcal{T}_H$ satisfies an affine decomposition (a convex combination of equivariant and standard operators) and remains a $\gamma$-contraction, guaranteeing a unique fixed point. Corollary 1 gives the key bound: $\|Q_H^* - Q_N^*\|_\infty \leq \frac{1}{1-\gamma}\|(1-\lambda)\delta\|_\infty$, which reduces to zero when $\lambda = 1$ in breaking regions.
    - **Design Motivation**: Elevates the intuition of "use equivariance where it holds, standard otherwise" into an MDP framework with rigorous theoretical guarantees. The convex combination preserves MDP validity, $\gamma$-contraction ensures convergence, and the error bound prescribes how $\lambda$ should be designed.

3. **Symmetry-Breaking Detection via Predictor Disagreement**:

    - **Function**: Automatically identifies whether each $(s,a)$ lies in a symmetry-breaking region without prior knowledge.
    - **Mechanism**: Two one-step predictors are trained — an equivariant predictor $\hat{P}_E$ subject to group constraints and an unconstrained standard predictor $\hat{P}_N$. In symmetric regions both predictors agree (low disagreement); in breaking regions, $\hat{P}_E$ can only represent a group-averaged surrogate dynamics while $\hat{P}_N$ approximates the true dynamics, resulting in high disagreement. A disagreement score $d(s,a) = D(\hat{P}_E, \hat{P}_N)$ is computed, high-disagreement samples are treated as anomalies (upper-tail distribution), pseudo-labels $y \in \{0,1\}$ are generated, and the gating network $\lambda_\omega$ is trained with binary cross-entropy loss. The gating network is frozen during RL updates and receives no RL gradients.
    - **Design Motivation**: Directly measuring $\epsilon_R, \epsilon_P$ requires knowledge of the group-invariant MDP (typically unavailable), while predictor disagreement provides an indirect but practical surrogate signal. Anomaly detection avoids the need to set hard thresholds.

### Loss & Training

**Critic Loss**: Gated mixed Q-value $Q_\theta(s,a) = (1-\lambda_\omega)Q_{E,\theta}(s,a) + \lambda_\omega Q_{N,\theta}(s,a)$, trained with standard TD targets (hard max for DQN, soft max for SAC). $\lambda_\omega$ is treated with stop-gradient when computing TD targets.

**Actor Loss (SAC variant)**: Introduces a state-level gate $\lambda_\zeta(s)$; the policy takes a Product of Experts (PoE) form $\pi_\phi \propto \pi_E^{1-\lambda_\zeta} \cdot \pi_N^{\lambda_\zeta}$. $\lambda_\zeta$ is aggregated from $\lambda_\omega(s,a)$ via expectile regression — using expectile loss with $\tau \to 1$ to approximate $\max_a \lambda_\omega(s,a)$, ensuring that as long as any action triggers a breaking signal at a given state, the entire policy switches to standard mode (conservative policy).

**Predictor Loss**: $\hat{P}_E$ and $\hat{P}_N$ fit one-step transitions using equivariant and standard networks respectively, optionally augmented with reward prediction heads $\hat{R}_i(s,a)$ for detecting reward-level symmetry breaking.

**Overall Training Loop**: Each step proceeds as: collect data → update predictors → compute disagreement → update gate → update critic → update actor → soft update target networks. Each component (critic, actor, predictors, gate) uses independent trunks to ensure training stability.

## Key Experimental Results

### Main Results: Grid-World Discrete Control ($C_4$ Rotational Symmetry + Obstacle Breaking)

| Method | 0 Obstacles | 10 Obstacles | 20 Obstacles | 30 Obstacles | 40 Obstacles |
|--------|-------------|--------------|--------------|--------------|--------------|
| Vanilla DQN | Moderate | Moderate | Moderate | Moderate | Moderate |
| Equivariant DQN | **Highest** | Rapid decline | Large degradation | Severe degradation | Near failure |
| RPP-DQN (approx. equivariant) | High | Slightly above Vanilla | Slightly above Vanilla | Slightly above Vanilla | Slightly above Vanilla |
| Approx. Equivariant DQN | High | Slightly above Vanilla | Slightly above Vanilla | Moderate | Moderate |
| **PE-DQN** | **Highest** | **Highest** | **Highest** | **Highest** | **Highest** |

As the number of obstacles increases, the gap between PE-DQN and the second-best method continues to widen, validating the theoretical prediction that "more severe breaking → greater importance of selective equivariance."

### Main Results: Continuous Control (MuJoCo + Robotic Arm)

| Environment | SAC | Equi-SAC | RPP-SAC | Approx-SAC | **PE-SAC** | Source of Symmetry Breaking |
|-------------|-----|----------|---------|------------|-----------|---------------------------|
| Hopper | Moderate | Moderate | Moderate | Moderate | **Highest learning speed** | Ground contact |
| Ant | Moderate | Moderate | Moderate | Moderate | **Highest (efficiency + final performance)** | Asymmetric leg torques |
| Swimmer | Moderate | **Highest** | High | High | Near highest | Nearly no breaking |
| Fetch Reach | Moderate | High | High | High | **Highest** | Ground constraint |
| UR5e Reach | Moderate | Unstable/collapse | Unstable | Unstable | **Highest and stable** | Dynamics + free orientation |

The effect is most pronounced on the UR5e Reach task: strictly equivariant and approximately equivariant SAC variants become unstable or collapse due to extensive symmetry breaking from real robot arm dynamics, while PE-SAC is the only method that maintains stable high performance.

### Ablation Study

| Configuration | Grid-World (30 obs.) | Notes |
|---------------|----------------------|-------|
| PE-DQN (full) | **Highest** | Hard gate + predictor disagreement |
| Soft gate ($\lambda \in [0,1]$) | Decrease | Less stable than hard gate |
| Shared trunk (critic) | Slight decrease | Occasionally affects stability |
| Shared trunk (actor) | Decrease | Equivariant/standard networks interfere |
| Remove reward head (transition disagreement only) | Decrease in reward-breaking scenarios | Cannot detect purely reward-level breaking |
| Sampled max ($K=4$) replacing $\lambda_\zeta$ | Near full | Lightweight alternative, slightly weaker under sparse breaking |
| Sampled max ($K=8$) replacing $\lambda_\zeta$ | Near full | Comparable to learned state gate |

### Key Findings

- **Breaking severity vs. performance curve**: In Grid-World, systematically increasing the number of obstacles (0→40) shows that PE-DQN's relative advantage increases monotonically with the degree of breaking. In fully symmetric environments, $\lambda$ rapidly converges to approximately 0 (pure equivariant mode), matching the performance of strictly equivariant DQN with no additional overhead.
- **Gate visualization**: The learned $\lambda$ in Grid-World closely aligns with obstacle positions — $\lambda \approx 0$ (equivariant) in open regions far from obstacles and $\lambda = 1$ (standard) near obstacles, validating the effectiveness of the detection mechanism.
- **Hard gate outperforms soft gate**: Experiments show that hard switching with $\lambda \in \{0,1\}$ is more stable than soft interpolation with $\lambda \in [0,1]$, likely because soft gating introduces gradient coupling that causes mutual interference between the two networks.
- **Robustness to complex dynamics**: In the Grid-World variant with 40 obstacles and stochastic transitions, PE-DQN still achieves optimal performance, demonstrating that predictor-disagreement detection remains effective under noisy dynamics.
- **Reward-level breaking**: In the variant where some obstacles are passable but incur negative rewards, PE-DQN with a reward prediction head remains optimal, showing the ability to simultaneously handle symmetry breaking at both the transition and reward levels.

## Highlights & Insights

- **Error propagation theory fills a conceptual gap**: Prior work only empirically observed that equivariant RL is unstable in real environments. This paper is the first to rigorously prove the propagation mechanism — "local symmetry breaking → amplified by $(1-\gamma)^{-1}$ through Bellman backup → global value function bias." This theory not only explains the phenomenon but precisely identifies the solution direction: errors must be blocked at the local level.
- **Gate design bridges theory and practice**: The convex-combination form of PI-MDP guarantees MDP validity and contractivity, while Corollary 1's error bound directly prescribes that $\lambda$ should be set to 1 in breaking regions — the correspondence between theory and algorithm design is exceptionally tight. In practice, the true $\epsilon_R, \epsilon_P$ need not be known; predictor disagreement provides a viable surrogate signal.
- **Transferable "selective inductive bias" paradigm**: The paradigm of "apply the prior when needed, relax it when not" is not limited to equivariance. Any method exploiting structural priors (e.g., sparsity, smoothness, causal structure) in settings where the prior partially fails can draw inspiration from this gated switching approach.

## Limitations & Future Work

- **Computational overhead**: Maintaining dual networks (equivariant + standard) alongside additional predictor and gating networks results in training times approximately 2–3× that of standard RL. This overhead may be prohibitive for tasks with already large parameter counts (e.g., high-dimensional visual inputs).
- **Degradation under pervasive breaking**: When symmetry is severely broken throughout the entire state space (e.g., omnidirectional motion under strong gravity), $\lambda$ is nearly everywhere 1, and the framework degrades to standard RL — the benefits of equivariance disappear while the additional architectural overhead remains.
- **Limited to state-based inputs**: The current equivariant networks (EMLP-based) operate on state vectors and have not yet been extended to visual observations (images/point clouds). Extending PI-MDP to visual RL is identified by the authors as the primary direction for future work.
- **Gate accuracy depends on predictor quality**: The accuracy of disagreement-based detection depends on the quality of both predictors. In high-dimensional, complex dynamics settings, predictors may be insufficiently accurate, leading to noisy gating signals. Ensembling multiple predictors or leveraging stronger world models could improve detection reliability.

## Related Work & Insights

- **vs. RPP (Finzi et al., 2021)**: RPP globally relaxes equivariance constraints by adding a residual pathway alongside equivariant layers, essentially allowing each parameter to independently determine its degree of equivariance. PE-RL instead makes pointwise decisions in the state-action space about whether to apply equivariance. RPP's relaxation is continuous and uniformly applied across the entire network, whereas PE-RL's gating is binary and spatially adaptive. Experiments show that PE-RL substantially outperforms RPP in scenarios with severe breaking (e.g., 25+ obstacles, UR5e).
- **vs. Approx. Equivariant RL (Park et al., 2025)**: Similarly attempts to handle approximate symmetry but does so via global architectural modifications. Performance is comparable to PE-RL under mild breaking, but significantly inferior under severe breaking (e.g., Grid-World with 30+ obstacles), as global relaxation cannot distinguish "good regions" from "bad regions."
- **Inspiration**: The error propagation analysis framework can be directly applied to analyze other RL methods exploiting structural priors (e.g., causal RL, options frameworks in hierarchical RL), enabling quantitative understanding of how partial violation of prior assumptions affects performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The PI-MDP framework, error propagation theory, and gated detection mechanism form a coherent trinity with a complete theory–algorithm–experiment chain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Broad coverage across discrete/continuous/manipulation task categories; the systematic analysis of breaking severity is persuasive. Lacks visual-input and real-robot experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The derivation logic from theory to algorithm is clear; the hierarchical structure of theorems, corollaries, and algorithms is well organized.
- **Value**: ⭐⭐⭐⭐⭐ Represents a fundamental advance toward deploying equivariant RL in real-world settings; the "selective inductive bias" paradigm has broad transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../NeurIPS2025/reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[NeurIPS 2025\] To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/to_distill_or_decide_understanding_the_algorithmic_trade-off_in_partially_observ.md)

</div>

<!-- RELATED:END -->

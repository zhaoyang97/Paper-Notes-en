---
title: >-
  [Paper Note] Action-Free Offline-to-Online RL via Discretised State Policies
description: >-
  [ICLR 2026][AI Safety][Action-free offline RL] This paper formalises the "action-free offline-to-online RL" setting for the first time and proposes the OSO-DecQN algorithm. By discretising continuous state differences in…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "Action-free offline RL"
  - "state policy"
  - "state discretisation"
  - "DecQN"
  - "guided online learning"
date: 2026-05-08
content_hash: f34f7e554094d564
---

# Action-Free Offline-to-Online RL via Discretised State Policies

**Conference**: ICLR 2026
**arXiv**: [2602.00629](https://arxiv.org/abs/2602.00629)  
**Code**: Available (provided in supplementary material)  
**Area**: AI Safety
**Keywords**: Action-free offline RL, state policy, state discretisation, DecQN, guided online learning

## TL;DR

This paper formalises the "action-free offline-to-online RL" setting for the first time and proposes the OSO-DecQN algorithm. By discretising continuous state differences into ternary tokens $\{-1, 0, 1\}$, a state policy $Q(s, \Delta s)$ is pretrained on action-free $(s, r, s')$ tuples to predict the expected direction of next-state change rather than actions. A policy-switching mechanism combined with an online-trained inverse dynamics model (IDM) then translates the state policy into executable actions, guiding online agents to accelerate learning. The approach consistently improves both convergence speed and asymptotic performance on D4RL and DeepMind Control Suite (including 78-dimensional state spaces).

## Background & Motivation

**Background**: Offline RL has demonstrated the ability to learn policies from static datasets, but nearly all existing methods assume complete action labels $(s, a, r, s')$ in the dataset.

**Limitations of Prior Work**: In many real-world scenarios, action information is inherently absent — treatment decisions in medical records are removed for privacy, specific operations in financial transactions are withheld due to proprietary strategy protection, and control signals in robot sensor logs are not recorded due to storage constraints. Such $(s, r, s')$-only datasets are widespread yet cannot be exploited by standard offline RL methods.

**Limitations of Existing Attempts**: (1) The $Q_{SS'}$ methods of Edwards et al. (2020) and Hepburn et al. (2024) estimate the value of state transitions but still rely on action labels during training; (2) Zhu et al. (2023) use a Decision Transformer to learn action-free state policies, but the approach is computationally expensive, yields unstable online guidance, and has not been validated across diverse environments; (3) The action discretisation approach of Seyde et al. (2022) requires bounded action ranges and is incompatible with action-free datasets. **No existing method simultaneously satisfies: learning from action-free offline data + high-dimensional scalability + effective guidance of online learning.**

**Key Challenge**: Action-free data cannot be directly used by any standard value-based or policy-based RL algorithm. Directly regressing to predict the next state in continuous space leads to instability and overfitting.

**Core Idea**: Rather than learning "which action to take," the method learns "in which direction the state should change" — by discretising continuous state differences into directional tokens $\{-1, 0, 1\}$, an ill-posed continuous regression problem is transformed into a structured classification problem, circumventing the action space while retaining sufficient decision-relevant information.

## Method

### Overall Architecture

A two-stage pipeline: **offline pretraining** — learning a state policy $Q(s, \Delta s)$ from action-free $(s, r, s')$ datasets, where $\Delta s \in \{-1, 0, 1\}^M$ denotes the discrete direction of change for each state dimension; **guided online learning** — using the pretrained state policy via policy switching and an IDM to translate $\Delta s$ into executable actions, accelerating the online agent.

### Key Designs

1. **State Discretisation Transformation**:

    - Function: Converts continuous state differences $s' - s$ into discrete directional tokens, transforming state prediction into a classification problem.
    - Mechanism: After z-score normalisation of state differences, each dimension is mapped to $\{-1, 0, 1\}$ (decrease / unchanged / increase) according to threshold $\epsilon$. Formally:
    $$\delta_i^\epsilon(s,s') = \begin{cases} -1 & s'_i - s_i < -\epsilon \\ 1 & s'_i - s_i > \epsilon \\ 0 & \text{otherwise} \end{cases}$$
    - Design Motivation: (1) Continuous regression of $s'$ or $s'-s$ is highly unstable (experiments show performance near that of a random policy); (2) z-score normalisation achieves scale invariance without per-dimension tuning; (3) Theoretical guarantee: the value function error of $k$-bin discretisation is $O(H\sqrt{M}/k)$, which is controllable and decreases as precision improves.

2. **OSO-DecQN Offline Pretraining Algorithm**:

    - Function: Pretrains a high-quality state policy from action-free data.
    - Mechanism: The value decomposition of DecQN is transferred from the action dimension to the state-difference dimension — $Q_\theta(s, \Delta s) = \frac{1}{M}\sum_{j=1}^M U^j_{\theta_j}(s, \Delta s_j)$ — where each state dimension independently maintains a utility branch and the total Q-value is the mean across all dimensions. An ensemble variant with double-Q learning stabilises training.
    - Conservative Regularisation: $R_\theta = \sum_{(s,\Delta s)\sim D} \log \| \exp(Q_\theta(s, \cdot)) \|_1 - Q_\theta(s, \Delta s)$, which is equivalent to a CQL penalty in the discrete setting and simultaneously addresses both overestimation bias and state reachability constraints.
    - Design Motivation: The decomposition structure of DecQN reduces the combinatorially explosive $3^M$ space to linear $3M$ complexity, making the method scalable to 78-dimensional state spaces.

3. **Guided Online Learning Mechanism**:

    - Function: Transfers knowledge from the offline state policy to the online agent.
    - Mechanism: (1) **Policy switching** — with probability $\beta$, use the offline-guided action; otherwise use the online policy: $a = I_\phi(s, \arg\max_{\Delta s} Q(s, \Delta s))$ (when guided) or $\pi_{on}(s)$ (when exploring); (2) **Online IDM training** — a lightweight inverse dynamics model $I_\phi$ is trained with L1 loss on online-collected $(s, a, s')$ tuples to map $\Delta s$ to actions; (3) **Offline data augmentation** — pseudo-actions are annotated for offline samples using the online policy $\pi_{on}(s_{off})$ to augment IDM training data.
    - Design Motivation: The IDM is intentionally kept simple (fixed architecture across all environments) to ensure that performance gains are attributable to the quality of the state policy rather than the capability of the translator.

### Loss & Training

- **Offline phase**: $\theta \leftarrow \arg\min_\theta \sum (y_1 - Q_\theta(s, \Delta s))^2 + \alpha R_\theta$, where $y_1 = r + \gamma \bar{Q}_\theta(s', \Delta s')$ and $\Delta s'$ is sampled according to the softmax of $Q_\theta(s', \cdot)$.
- **IDM training**: $L(\phi) = \|a_{on,off} - I_\phi(s, \Delta s_{off})\|_1 + \|a - I_\phi(s, \Delta s)\|_1$; the L1 loss is more robust to outliers due to its approximation of the median.

## Key Experimental Results

### Main Results: Offline Pretraining Performance (Selected from Table 1)

Normalised mean return of OSO-DecQN versus multiple baselines on D4RL and DeepMind Control Suite (mean ± standard error over 5 seeds × 10 episodes):

| Dataset | BC (w/ actions) | BC $s'$ | BC $s'-s$ | BC $\Delta s$ | DecQN_N (no regularisation) | **OSO-DecQN** |
|--------|-----------|---------|-----------|--------------|-------------------|---------------|
| Hopper-medium-replay | 26.6 | 5.8 | 4.9 | 29.2±3.7 | 7.7±1.3 | **65.7±2.6** |
| Hopper-expert | 110.6 | 2.2 | 9.9 | 106.9±2.4 | 1.9±0.45 | **111.6±0.08** |
| HalfCheetah-medium-expert | 60.1 | -0.25 | -0.25 | 54.7±3.9 | -1.6±0.13 | **87.8±2.7** |
| Walker2d-medium-replay | 23.5 | 6.4 | -0.32 | 37.5±6.1 | -0.41±0.26 | **84.8±2.2** |
| Walker2d-medium-expert | 107.7 | 2.3 | -0.71 | 84.4±3.4 | -0.24±0.06 | **108.8±0.13** |
| Cheetah-Run-medium-expert | 61.6 | 1.7 | 1.7 | 48.0±3.4 | 1.2±0.32 | **90.0±3.8** |
| Quadruped-Walk-expert (78-dim) | 97.7 | 6.6 | 6.6 | 96.7±6.4 | 0.1±0.04 | **100.7±1.2** |

**Key observations**: (1) Continuous regression methods (BC $s'$, BC $s'-s$) achieve near-zero or negative performance on most tasks; (2) Discretised BC $\Delta s$ already approaches the performance of action-conditioned BC; (3) OSO-DecQN substantially surpasses imitation learning through RL, with especially pronounced advantages on mixed-quality datasets such as medium-replay and medium-expert; (4) The unregularised DecQN_N almost completely fails.

### Guided Online Learning Experiments

Over 1M steps of online learning, OSO-DecQN-guided TD3/DecQN_N versus unguided baselines:

| Environment | State Dim. | Action Dim. | Online Baseline | OSO-DecQN Guidance Effect |
|------|---------|---------|---------|-------------------|
| HalfCheetah | 17 | 6 | TD3 | Significant improvement in both convergence speed and asymptotic performance |
| Walker2D | 17 | 6 | TD3 | Notable early-stage acceleration; improved asymptotic performance |
| Hopper | 11 | 3 | TD3 | Modest improvement (task is inherently simple) |
| Quadruped-Walk | 78 | 12 | DecQN_N | Sustained early-stage improvement, validating high-dimensional scalability |
| Cheetah-Run | 17 | 6 | DecQN_N | Improved convergence speed and final performance |

Compared with AF-Guide (Zhu et al., 2023): AF-Guide performs **below** the unguided TD3 baseline on all three D4RL environments, whereas OSO-DecQN performs **above** the baseline on all environments.

### Ablation Study

| Ablation | Result | Conclusion |
|--------|------|------|
| Remove discretisation (use $s'$ or $s'-s$) | Normalised return drops sharply to near-random-policy level | Discretisation is essential; continuous regression is completely infeasible |
| Remove regularisation (DecQN_N) | Near-total failure on all tasks (return ≈ 0) | Regularisation prevents overestimation and ensures state reachability |
| Sensitivity to threshold $\epsilon$ | Performance stable over a wide range | Method is insensitive to $\epsilon$ |
| 2-bin vs. 3-bin discretisation | Comparable performance | Coarse discretisation is already sufficiently effective |
| Variation in IDM architecture / batch size | Performance largely unchanged | Improvement stems from the state policy, not the IDM |
| Sensitivity to guidance ratio $\beta$ | Improvement consistent across reasonable range | Hyperparameter is robust |
| TD3 → SAC as online agent | Equally effective | Method is decoupled from the specific online algorithm |

### Key Findings

- **Continuous regression completely fails**: The discrete difference prediction error of BC $s'$ and BC $s'-s$ approaches that of a random policy (e.g., error ≈ 16–17 in Walker2d vs. 17 for a random policy), confirming the infeasibility of continuous prediction.
- **Minimal information loss from discretisation**: BC $\Delta s$ already matches the performance of action-conditioned BC, and OSO-DecQN further significantly surpasses it via RL.
- **Regularisation is a necessary condition**: The unregularised variant not only yields near-zero returns but also exhibits sharply elevated state prediction error, simultaneously suffering from overestimation bias and state unreachability.
- **First validation of action-free offline-to-online RL on a 78-dimensional state space** (Quadruped-Walk).

## Highlights & Insights

- **Value of problem formalisation**: The paper is the first to define "action-free offline-to-online RL" as an independent research problem and to provide a complete framework, opening a new pathway for RL applications in action-absent domains such as medicine and finance.
- **Elegance of the directional token design**: Discretising state differences into $\{-1, 0, 1\}$ appears simplistic but strikes precisely the optimal balance between information retention and prediction stability — it offers a controllable error bound in theory and matches or exceeds action-conditioned baselines in practice.
- **Cross-domain architectural transfer**: The DecQN value decomposition is transferred from multi-agent RL to state-space decomposition, and CQL regularisation is transferred from action constraints to state reachability constraints — both cross-domain adaptations are highly natural.
- **Highly convincing ablation study**: Tables 1 and 2 jointly demonstrate the necessity of discretisation and regularisation from two complementary perspectives — return and prediction error — forming a closed logical chain.

## Limitations & Future Work

- **Discretisation granularity is fixed at 3-bin $\{-1,0,1\}$**: While theory predicts reduced error with finer granularity, adaptive discretisation (e.g., progressive approaches as in Growing Q-Networks) is not experimentally explored.
- **IDM assumes a locally smooth inverse mapping from states to actions**: In strongly discontinuous dynamics or highly multimodal inverse mapping scenarios, a simple IDM may fail.
- **Not extended to visual observations**: All experiments use vector-based states; image-based environments would require additional encoders and reward extraction mechanisms.
- **Theoretical analysis is limited to the discretisation error bound**: An end-to-end convergence guarantee for the full framework (pretraining + guided online learning) is absent.
- **Guidance ratio $\beta$ is a fixed hyperparameter**: Adaptive decay strategies based on the online agent's learning progress are not explored.
- Directions for improvement: (1) Incorporate categorical value functions (Farebrother et al., 2024) in place of regression; (2) Adaptive discretisation; (3) More expressive IDM architectures for complex dynamics; (4) Extension to visual RL.

## Related Work & Insights

- **vs. $Q_{SS'}$ (Edwards et al., 2020; Hepburn et al., 2024)**: Both estimate state-transition values but still require action labels during training. This paper eliminates the action dependency entirely.
- **vs. AF-Guide (Zhu et al., 2023)**: The only direct competitor, which uses a Decision Transformer to learn action-free state policies with reward shaping to guide online learning, but is computationally expensive and experiments show that its online guidance performs worse than an unguided TD3 baseline.
- **vs. DecQN (Seyde et al., 2022)**: The original DecQN is used for continuous action discretisation; OSO-DecQN transfers the same value decomposition idea from the action space to the state-difference space and adds conservative regularisation to suit the offline setting.
- **Insights**: (1) The concept of a state policy is generalisable — any signal capable of predicting the expected direction of state transitions can serve as guidance for online learning; (2) Discretisation as a stabilisation tool can be applied to other RL scenarios where continuous prediction is unstable.

## Rating

- Novelty: ⭐⭐⭐⭐ — Pioneering problem formalisation with an elegantly designed state-discretisation solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five environments × four dataset quality levels; ablations are exceptionally complete with every design decision thoroughly validated.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and in-depth analysis; the sections explaining why discretisation and why regularisation are particularly strong.
- Value: ⭐⭐⭐⭐ — Opens a practical pathway for RL in action-absent settings, though application validation remains limited to simulated environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)
- [\[ICLR 2026\] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](../../ICML2026/ai_safety/actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[NeurIPS 2025\] Fairness-Regularized Online Optimization with Switching Costs](../../NeurIPS2025/ai_safety/fairness-regularized_online_optimization_with_switching_costs.md)
- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](../../CVPR2026/ai_safety/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][multi-agent RL] Through 82,620 large-scale experiments, this work systematically investigates robustness and resilience in cooperative multi-agent RL…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "multi-agent RL"
  - "robustness"
  - "resilience"
  - "hyperparameter tuning"
  - "cooperative MARL"
date: 2026-05-08
content_hash: cf6c31df24ae51d0
---

# Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.11824](https://arxiv.org/abs/2510.11824)  
**Code**: [github.com/BUAA-TrustworthyMARL/adv_marl_benchmark](https://github.com/BUAA-TrustworthyMARL/adv_marl_benchmark)  
**Area**: Reinforcement Learning
**Keywords**: multi-agent RL, robustness, resilience, hyperparameter tuning, cooperative MARL

## TL;DR
Through 82,620 large-scale experiments, this work systematically investigates robustness and resilience in cooperative multi-agent RL, demonstrating that hyperparameter tuning matters more than algorithm selection, and revealing that commonly adopted practices such as parameter sharing, GAE, and PopArt are harmful under uncertainty. A set of practical hyperparameter recommendations is proposed.

## Background & Motivation

**Background**: Cooperative MARL algorithms (MADDPG, MAPPO, HAPPO) have achieved remarkable success in simulators by maximizing cooperative performance through hyperparameter tuning. Real-world deployment introduces challenges such as observation noise, action perturbations, and environmental uncertainty.

**Limitations of Prior Work**: (a) Robustness and resilience are frequently conflated in the MARL literature—the former refers to maintaining performance under sustained uncertainty, while the latter refers to the ability to recover from perturbations. (b) The impact of hyperparameters on robustness and resilience has received little attention. (c) Existing robust MARL studies are confined to simple simulated environments and lack real-world validation.

**Key Challenge**: Hyperparameters tuned for cooperative performance may severely degrade robustness and resilience (as illustrated in Figure 1), yet researchers generally do not know which hyperparameters are harmful.

**Goal**: (a) Formally distinguish robustness from resilience in MARL; (b) systematically evaluate the relationships among cooperation, robustness, and resilience; (c) assess the effects of 15 hyperparameters on all three objectives and provide practical recommendations.

**Key Insight**: A large-scale controlled empirical study—4 real-world environments, 13 uncertainty types, 15 hyperparameters, 3 algorithms, 18 tasks, and 5 random seeds.

**Core Idea**: Through the largest empirical study on MARL robustness to date, this work demonstrates that hyperparameter tuning is more consequential than algorithm selection and uncovers several counterintuitive findings.

## Method

### Overall Architecture
Based on the Dec-POMDP $\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \mathcal{O}, O, \mathcal{A}, \mathcal{P}, R, \gamma \rangle$, a three-stage pipeline is adopted: (1) train cooperative models; (2) evaluate robustness $J^{\text{robust}}(\pi)$ under uncertainty; (3) evaluate recovery capability $J^{\text{resilience}}(\pi)$ from perturbed states.

### Key Designs

1. **Formal Definition of Robustness**:

    - Function: Measures the expected return of a fixed policy under sustained uncertainty.
    - Mechanism: $J^{\text{robust}}(\pi) = \mathbb{E}_{u\sim\mathcal{U}}[\mathbb{E}_{s_0\sim\rho_0}\mathbb{E}_{\pi,u}[\sum_{t=0}^{\infty}\gamma^t r_t | s_0]]$, where $\mathcal{U}$ denotes the uncertainty set.
    - Design Motivation: Uncertainty acts continuously throughout the episode while the policy $\pi$ remains fixed.

2. **Formal Definition of Resilience**:

    - Function: Measures the ability of a policy to recover normal functionality from a perturbed state.
    - Mechanism: $J^{\text{resilience}}(\pi) = \mathbb{E}_{u\sim\mathcal{U}}[\mathbb{E}_{s_u\sim\rho_u}\mathbb{E}_\pi[\sum_{t=t_u}^{\infty}\gamma^t r_t | s_0=s_u]]$—starting from the post-perturbation state $s_u$, with no further uncertainty thereafter.
    - Design Motivation: Distinguishes resilience from robustness—resilience concerns recovery after a shock rather than continuous resistance. An analogy is the ability of a power grid to restore supply after an earthquake.
    - Distinction from Robustness: A system may be resilient but not robust (recovers quickly from perturbations but performs poorly under sustained noise), or robust but not resilient (withstands sustained noise but recovers slowly from anomalous initial states).

3. **Systematic Coverage of 13 Uncertainty Types**:

    - Observation uncertainty (6 types): Gaussian noise / greedy attack / learned attack × all agents ($\epsilon=0.1$) / single agent ($\epsilon=0.2$).
    - Action uncertainty (6 types): random policy / greedy policy / learned policy × all agents / single agent, modeled as $\epsilon\hat{\pi} + (1-\epsilon)\pi$.
    - Environmental uncertainty (1 type): 50 rollouts sampled uniformly from a parameter uncertainty set, with the worst-case result reported.

4. **Systematic Evaluation of 15 Hyperparameters**:

    - General hyperparameters (10): hidden layer size, discount factor, activation function, initialization method, network type, learning rate, critic learning rate, feature normalization, parameter sharing, early stopping.
    - MADDPG-specific (2): N-step TD, exploration noise.
    - MAPPO/HAPPO-specific (3): entropy coefficient, GAE, PopArt.
    - One hyperparameter is varied at a time with all others fixed at default values, yielding 34 model configurations.

### Experimental Environments
Four real-world environments are used: DexHand (dexterous hand manipulation, Isaac Gym), Quads (quadrotor formation, continuous control), Traffic (traffic signal control, discrete actions), and Voltage (voltage control, continuous control), comprising 18 tasks in total.

### Experimental Protocol
5 random seeds × 27 uncertainty settings (1 cooperative + 13 robustness + 13 resilience) × 18 tasks × 34 hyperparameter configurations = **82,620 experiments**, totaling approximately 230K GPU hours (GTX 4090).

## Key Experimental Results

### Key Finding 1: Cooperation–Robustness–Resilience Correlations

| Perturbation Severity | Cooperation↔Robustness | Cooperation↔Resilience |
|----------------------|------------------------|------------------------|
| Mild | High ($r > 0.8$) | High ($r > 0.8$) |
| Moderate | Moderate ($r \approx 0.6$) | Moderate |
| Severe | Low ($r < 0.4$) | Low |

Linear regression: cooperation–robustness $r=0.85, p<.001$; cooperation–resilience $r=0.76, p<.001$ (both decline linearly with attack severity).

### Key Finding 2: Hyperparameter Recommendations

| Hyperparameter | Recommended Setting | Cooperation | Robustness | Resilience | Statistical Test |
|----------------|-------------------|-------------|------------|------------|-----------------|
| Early stopping | Enabled | ✓ | ✓ | ✓ | $t(161)=6.16, p<.001$ |
| Critic LR | > Actor LR | ✓ | ✓ | ✓ | $t(161)=10.02, p<.001$ |
| Activation function | Leaky ReLU | ✓ | ✓ | ✓ | $t(161)=6.31, p<.001$ |
| GAE | Disabled | ✓ | ✓ | ✓ | $t(107)=7.44, p<.001$ |
| PopArt | Disabled | ✓ | ✓ | ✓ | $t(107)=4.84, p<.001$ |
| Parameter sharing | Disabled (heterogeneous) | ✓ | ✓ | ✓ | $t(161)=7.01, p<.001$ |
| Exploration (PPO) | High entropy | ✓ | ✓ | ✓ | $t(107)=6.64, p<.001$ |

### Overall Performance Gains

| Metric | Default → Optimal Hyperparameters | Robust Method + Optimal Hyperparameters |
|--------|-----------------------------------|-----------------------------------------|
| Cooperation | +52.60% | +89.43% |
| Robustness | +34.78% | +65.83% |
| Resilience | +60.34% | +82.96% |

### Key Findings
- **Hyperparameters matter more than algorithms**: Two-way ANOVA shows that in 9 out of 18 tasks, the hyperparameter effect is significantly stronger than the algorithm effect ($p<0.001$).
- **Uncertainty types are not interchangeable**: Correlations among observation, action, and environmental uncertainty are low (one-way ANOVA for robustness: $F(2,153)=9.53, p<.001$); robustness to one type does not imply robustness to another.
- **Agent scope is not interchangeable**: Defenses targeting all agents may fail under single-agent attacks ($F(1,142)=4.36, p<.05$).
- **Algorithms have complementary strengths**: MADDPG is most robust to action uncertainty (sustained exploration noise), while MAPPO/HAPPO perform best against observation uncertainty (centralized critic).
- **Why GAE is harmful**: In real-world environments with sparse and high-variance rewards, the bootstrapping error of GAE is amplified; it is only effective in reward-dense environments such as MuJoCo.

## Highlights & Insights
- **Unprecedented empirical scale**: 82,620 experiments and ~230K GPU hours represent a scale unseen in the MARL literature, with every conclusion supported by systematic statistical testing.
- **Formal introduction of resilience**: This is the first work to formally introduce the concept of resilience from control theory and ecology into MARL, with a clear formal definition that distinguishes sustained resistance from post-shock recovery.
- **Multiple counterintuitive findings**: Parameter sharing, GAE, and PopArt are recommended as key tricks in the MAPPO paper; this work demonstrates their potential harm across a broader range of real-world environments, challenging consensus in the MARL community.
- **Open-source benchmark**: A modular codebase is provided that supports custom algorithm and environment integration, enabling direct assessment of the trustworthiness of new MARL methods.

## Limitations & Future Work
- **Only policy gradient methods are covered**: The three evaluated algorithms (MADDPG, MAPPO, HAPPO) are all policy-gradient-based; value decomposition methods (QMIX, QPLEX, etc.) are not included.
- **One hyperparameter varied at a time**: Interaction effects among hyperparameter combinations are not fully explored, though post-hoc analysis partially compensates.
- **Limited uncertainty models**: Observation and action noise models are relatively simple (Gaussian noise, $\epsilon$-replacement policy) and do not cover more complex adversarial scenarios.
- **Simplifying assumption in resilience evaluation**: Perturbations in practice may be gradual rather than abrupt; the assumption of starting from a perturbed state with perturbation-free rollout thereafter is a simplification.

## Related Work & Insights
- **vs. Engstrom et al. (2020, RL Implementation Matters)**: Similar in spirit—demonstrating that hyperparameters matter more than the algorithm itself—but this work extends the finding to the robustness and resilience dimensions of MARL at a substantially larger scale.
- **vs. MAPPO (Yu et al. 2021)**: Hyperparameter recommendations such as parameter sharing and GAE from MAPPO are effective on SMAC/MuJoCo, but this work reveals their limitations across more diverse real-world environments.
- **vs. RRLS/Robust Gymnasium**: These works provide benchmark codebases but focus primarily on single-agent or small-scale evaluations; this work provides a systematic relational analysis in real-world MARL settings.

## Rating
- Novelty: ⭐⭐⭐ The introduction of resilience and the large-scale empirical study are valuable, though the methodological contribution is relatively limited.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 82,620 experiments, rigorous statistical testing, and four real-world environments make this exceptionally thorough.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and findings are summarized in practical recommendation tables for easy practitioner reference.
- Value: ⭐⭐⭐⭐ The work provides important guidance for the MARL community and challenges several widely held assumptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Extending NGU to Multi-Agent RL: A Preliminary Study](extending_ngu_to_multi-agent_rl_a_preliminary_study.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[ACL 2026\] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](../../ACL2026/reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)

</div>

<!-- RELATED:END -->

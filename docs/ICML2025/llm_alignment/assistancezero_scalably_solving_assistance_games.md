---
title: >-
  [Paper Note] AssistanceZero: Scalably Solving Assistance Games
description: >-
  [ICML 2025][LLM Alignment][assistance game] AssistanceZero is proposed, scaling assistance games to complex environments (Minecraft building assistance with $10^{400}$ possible goals) for the first time. By extending AlphaZero with a reward prediction head and a human action prediction head to perform planning under uncertainty via MCTS, the method significantly outperforms PPO and imitation learning baselines. Human experiments demonstrate that AssistanceZero effectively red…
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "assistance game"
  - "AlphaZero"
  - "MCTS"
  - "human modeling"
  - "Minecraft"
  - "cooperative AI"
  - "POMDP"
date: 2026-05-08
content_hash: d35550458ba9dbb2
---

# AssistanceZero: Scalably Solving Assistance Games

**Conference**: ICML 2025  
**arXiv**: [2504.07091](https://arxiv.org/abs/2504.07091)  
**Code**: [https://github.com/cassidylaidlaw/minecraft-building-assistance-game](https://github.com/cassidylaidlaw/minecraft-building-assistance-game)  
**Area**: RLHF Alignment  
**Keywords**: assistance game, AlphaZero, MCTS, human modeling, Minecraft, cooperative AI, POMDP

## TL;DR

AssistanceZero is proposed, scaling assistance games to complex environments (Minecraft building assistance with $10^{400}$ possible goals) for the first time. By extending AlphaZero with a reward prediction head and a human action prediction head to perform planning under uncertainty via MCTS, the method significantly outperforms PPO and imitation learning baselines. Human experiments demonstrate that AssistanceZero effectively reduces user actions and exhibits emergent behaviors such as digging foundations, inferring roofs, and learning from corrections.

## Background & Motivation

### Limitations of RLHF

The current mainstream AI assistant training paradigm (Pre-training + SFT + RLHF) presents structural issues:

**Deception Incentives**: Annotators can be deceived into giving positive feedback, which incentivizes models to produce deceptive or manipulative behaviors.

**Discouraging Uncertainty Maintenance**: Single-turn high-rating objectives discourage assistants from asking clarifying questions or hedging answers.

**Non-cooperativeness**: Autocomplete assistants (such as Copilot) cannot consider the complementarity of human-AI collaboration—the helper's actions should complement the user's actions rather than simply replace them.

### Advantages of Assistance Games

An assistance game is a two-player game: the helper and the user act in a shared environment and share a reward function, but the helper **cannot observe** the goal parameter $\theta$. This framework:

- Eliminates deception incentives (as the reward depends on the true latent reward rather than human feedback)
- Incentivizes the helper to resolve uncertainty through interaction
- Generates optimal joint behaviors that complement user actions

### Why Have Assistance Games Not Been Widely Studied Before?

Two major challenges: (1) Decision-making under uncertainty is computationally intractable; (2) An accurate human behavior model is required. Prior work was limited to simple environments with ≤10 discrete reward parameters.

## Method

### Overall Architecture

**Environment Design: Minecraft Building Assistance Game (MBAG)**

- State: 3D block grid (11×10×10) + player position + inventory
- Action space: no-op, movement in 6 directions, placing blocks, breaking blocks (>20,000 possible actions)
- Goal parameter $\theta$: the block grid of the target building (based on the CraftAssist dataset)
- $|\Theta| \approx 10^{400}$—far exceeding the fewer than 20 goals in prior work
- Reward $R(s, a_H, a_R; \theta) = d(s', \theta) - d(s, \theta)$ (change in edit distance)

### Key Designs

**PPO Failure Analysis**: PPO barely works on MBAG (assistant goal % ≈ 0%). Reasons:

- The reward signal is highly noisy (rewards depend on both human and helper actions simultaneously)
- Even expected-beneficial actions can receive negative rewards (due to goal uncertainty)
- Long-horizon decision-making further amplifies the noise in reward-to-go
- The primary signal PPO learns is "placing/breaking = negative reward" $\rightarrow$ converging to doing nothing

**Core Idea of AssistanceZero**: Decoupling goal prediction from action selection

The recurrent neural network features four heads:
1. **Policy head** $\pi_\phi(a_R | h)$: Selects the helper's action
2. **Value head** $\hat{V}_\phi(h)$: Estimates the state value
3. **Reward parameter prediction head** $\hat{p}_\phi(\theta | h)$: Predicts the block type distribution at each position of the target building
4. **Human action prediction head** $\hat{p}_\phi(a_H | h)$: Predicts the human's next action

MCTS simulates future trajectories by sampling reward parameters and human actions, achieving planning under uncertainty.

### Loss & Training

The complete loss function of AssistanceZero:

$$L(\phi) = \frac{1}{n} \sum_{t=1}^{n} \left[ \lambda_{\text{policy}} D_{\text{KL}}(\pi_t^{\text{MCTS}} \| \pi_\phi(\cdot|h_t)) + \lambda_{\text{value}} (\hat{V}_\phi(h_t) - \sum_{t'=t}^{T} \gamma^{t'-t} R_{t'})^2 - \lambda_{\text{reward}} \log \hat{p}_\phi(\theta|h_t) + \lambda_{\text{prev-rew}} D_{\text{KL}}(\hat{p}_\phi(\theta|h_t) \| \hat{p}_t(\theta)) - \lambda_{\text{action}} \log \hat{p}_\phi(a_H^t | h_t) \right]$$

Five loss terms train the four heads, where the $\lambda_{\text{prev-rew}}$ term prevents the reward prediction head from overfitting to the most recently observed goals.

**Reward estimation in MCTS** uses a low-variance trick: utilizing the fact that the reward can be decomposed into $R = R_H + R_R$, the helper's reward is estimated at the current step, and the human's reward is estimated at the next step.

**Human Modeling**:
- Reward-based (PPO/AlphaZero): Poor prediction, builds too fast
- BC (Behavior Cloning): Accurate prediction but suffers from compounding error
- **piKL (Optimal Choice)**: MCTS + BC prior policy, balancing prediction accuracy and task performance

## Key Experimental Results

### Main Results

**Table 1: Evaluation with a Fixed Human Model**

| Method | Total Goal Completion Rate | Human Action Count | Helper Completion Rate |
|------|------------|-----------|-----------|
| PPO baseline | 71.6% | 203 | 0.0% |
| PPO + reward engineering | 74.0% | 200 | 3.5% |
| PPO + aux loss | 74.1% | 191 | 7.2% |
| **AssistanceZero** | **79.8%** | **158** | **27.0%** |
| Human Model Alone | 70.8% | 200 | — |

AssistanceZero reduces human actions by 42, with the helper independently completing 27% of the goal.

**Table 3: Comparison of Different Training Paradigms**

| Method | Total Goal Completion Rate | Human Action Count | Helper Completion Rate |
|------|------------|-----------|-----------|
| Pretraining (Copilot-like) | 89.8% | 240 | 2.3% |
| SFT (RLHF Phase 1-like) | 90.4% | 241 | 2.9% |
| **Assistance Game** | **92.6%** | **179** | **26.0%** |

### Ablation Study

- Removing LSTM: The goal completion rate drops sharply from 77.5% to 69.0%, and the helper completion rate falls from 25.2% to -0.6%.
- Removing KL regularization ($\lambda_{\text{prev-rew}}$): The goal completion rate drops from 77.5% to 76.8%, and the helper completion rate falls from 25.2% to 18.1%.
- Removing MCTS at test time: Performance does not degrade (80.2% vs 79.8%), indicating the advantage does not stem from extra inference-time computation.

### Key Findings

**Human Experiments (16 participants)**:
- AssistanceZero helper helpfulness score is 3.1/5 vs. SFT 1.7/5 vs. human helper 4.0/5
- Significantly reduces the placing/breaking actions performed by the participants ($p < 0.05$)
- **Emergent Behaviors**:
    - Digging foundations: Automatically clearing the interior after observing the human outline the boundaries.
    - Inferring roofs: Inferring the roof structure and completing it from just a few blocks placed by the human.
    - Learning from corrections: When a wall is built too high and the human breaks one block, the helper automatically breaks the remaining excess blocks.

## Highlights & Insights

1. **Scales assistance games to complex environments ($10^{400}$ goals) for the first time**, proving its feasibility.
2. **Insightful analysis of the underlying causes of PPO failure**: reward noise + coupling of goal prediction and action selection is identified as the core bottleneck.
3. **Design philosophy of decoupling prediction and action**: The AlphaZero framework is naturally suited for belief maintenance in POMDPs.
4. **Empirical findings on human modeling**: Pure reward-based models fail to predict human actions, BC suffers from compounding errors, and piKL serves as the best compromise.
5. **Emergent behaviors demonstrate the intrinsic strength of the assistance game framework**: The helper learns pragmatic communication rather than simple imitation.
6. **Future vision for LLM post-training**: Treating dialogue as a multi-turn assistance game can resolve the deception incentives and uncertainty avoidance issues in RLHF.

## Limitations & Future Work

1. **Environment Simplification**: MBAG is a highly simplified version of Minecraft, whereas the real-world complexity is far higher.
2. **Limited Dataset for Human Modeling**: The BC model is trained on only 18 episodes from 5 participants.
3. **Lack of Comparison with Full RLHF**: Since RLHF is difficult to directly apply in multi-agent environments, comparisons are only made against SFT.
4. **Computational Overhead**: AssistanceZero requires MCTS simulations (100 rollouts/step during training), which is computationally expensive.
5. **Scale of Human Experiments**: 16 participants and only 1 human helper, limiting the statistical power.
6. **A Long Way to LLM Scaling**: The migration path from Minecraft to LLM dialogue remains conceptual.

## Related Work & Insights

- **The assistance game theoretical lineage**: Fern et al. 2014 (hidden-goal MDP) $\rightarrow$ Hadfield-Menell et al. 2016 (CIRL) $\rightarrow$ Ours (first large-scale solution)
- **Human Modeling**: BC suffers from compounding error (DAgger problem), and piKL (Jacob et al. 2022) is a reasonable hybrid approach.
- **AlphaZero Extensions**: Differs from MuZero by handling partial observability and stochasticity; differs from POMCP by utilizing a learned model.
- **Learned Belief Search** (Hu et al. 2021): A similar approach of learning belief distributions from rollouts.
- **Insights**: Assistance games could be the "next generation" alignment paradigm for RLHF, but they require better human models and more efficient planning algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The first large-scale assistance game solution, showing a completely new methodological path)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Simulation + human experiments + rich ablation studies, though human experiments are limited in scale)
- Writing Quality: ⭐⭐⭐⭐⭐ (The paper structure is clear, figures are elegant, and the demonstration of emergent behaviors is highly compelling)
- Value: ⭐⭐⭐⭐⭐ (Pioneers the large-scale application of assistance games, providing profound insights for AI alignment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Diverging Preferences: When do Annotators Disagree and do Models Know?](diverging_preferences_when_do_annotators_disagree_and_do_models_know.md)
- [\[ICML 2025\] Improving Model Alignment through Collective Intelligence of Open-Source LLMs](improving_model_alignment_through_collective_intelligence_of_open-source_llms.md)
- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ICML 2025\] AlphaPO: Reward Shape Matters for LLM Alignment](alphapo_reward_shape_matters_for_llm_alignment.md)

</div>

<!-- RELATED:END -->

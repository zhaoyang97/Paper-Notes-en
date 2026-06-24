---
title: >-
  [Paper Note] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning
description: >-
  [ACL 2025][Reinforcement Learning][Multi-Agent RL] Proposes MAPoRL—a post-training paradigm based on multi-agent reinforcement learning. By co-training multiple LLMs within a debate framework, integrated with verifier scoring and collaborative incentive mechanisms, it significantly enhances the effectiveness of multi-LLM collaboration and demonstrates cross-task generalization capabilities.
tags:
  - "ACL 2025"
  - "Reinforcement Learning"
  - "Multi-Agent RL"
  - "LLM Collaboration"
  - "Post-Training"
  - "Multi-Turn Debate"
  - "Collaborative Reward Shaping"
date: 2026-05-08
content_hash: ec1cad456f78afbd
---

# MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning

**Conference**: ACL 2025  
**arXiv**: [2502.18439](https://arxiv.org/abs/2502.18439)  
**Code**: [https://github.com/chanwoo-park-official/MAPoRL](https://github.com/chanwoo-park-official/MAPoRL)  
**Authors**: Chanwoo Park, Seungju Han, Xingzhi Guo, Asuman Ozdaglar, Kaiqing Zhang, Joo-Kyung Kim  
**Institutions**: MIT, Stanford, Amazon, UMD  
**Area**: Reinforcement Learning / Multi-Agent Collaboration  
**Keywords**: Multi-Agent RL, LLM Collaboration, Post-Training, Multi-Turn Debate, Collaborative Reward Shaping

## TL;DR

Proposes MAPoRL—a post-training paradigm based on multi-agent reinforcement learning. By co-training multiple LLMs within a debate framework, integrated with verifier scoring and collaborative incentive mechanisms, it significantly enhances the effectiveness of multi-LLM collaboration and demonstrates cross-task generalization capabilities.

## Background & Motivation

**Background**: Multi-LLM collaboration (such as multi-agent debate) has attracted significant attention in recent years, but existing methods primarily rely on prompting off-the-shelf LLMs, expecting them to "naturally" possess collaborative abilities.

**Limitations of Prior Work**:
   - Multi-turn debates do not always guarantee performance benefits, and may even degrade performance for less capable models (Huang et al., 2024).
   - LLMs are never explicitly trained to collaborate during pre-training; relying solely on prompts cannot stimulate genuine cooperative behavior.
   - Single-agent training (such as SFT or individual RL) is insufficient to produce effective collaboration—an untrained, non-strategic opponent cannot drive the emergence of cooperative behavior.

**Core Motivation**: Needs an interactive training environment where multiple LLMs are trained simultaneously and dynamically optimize their respective policies, thereby explicitly learning collaborative behaviors. Game-theoretic analysis demonstrates that co-trained agents can reach an equilibrium showing cooperative behavior.

## Method

### Overall Architecture

MAPoRL is built upon a Collaborative Debate framework, with the following workflow:

1. **Independent Generation**: Each LLM agent independently generates an initial response to the prompt.
2. **Multi-Turn Discussion**: The agents engage in $T$ turns of discussion, where each turn generates a new response based on the historical responses of all agents.
3. **Verifier Scoring**: The MAPoRL verifier simultaneously evaluates answer correctness and discussion quality.
4. **Multi-Agent RL**: Using the verifier's score as the reward, multi-agent PPO is applied to maximize each agent's value function.

### Key Designs

#### Key Design 1: Influence-aware Verification Reward

Standard rewards only evaluate the correctness of the current answer, whereas the reward function of MAPoRL takes into account the agent's impact on all future responses from other agents:

$$R_{\theta}(q, s_{ta}) = \mathbb{E}\left[\frac{1}{\sum_{t' \in [t,T]} \gamma^{t'-t}} \left(\text{Verifier}(q, s_{ta}) + \sum_{t' \in [t+1,T]} \sum_{j \in [A]} \frac{1}{A} \gamma^{t'-t} \text{Verifier}(q, s_{t'j})\right)\right]$$

- It considers not only the score of the current response but also incorporates the discounted scores of all future agent responses.
- It encourages the agent to focus beyond immediate scores and provide valuable information for subsequent discussions.
- $\gamma$ is the discount factor controlling the importance assigned to future influence.

#### Key Design 2: Collaborative Incentive Mechanism (Reward Shaping)

Four incentive parameters are introduced to reinforce different collaborative behaviors:

| Parameter | Meaning |
|------|------|
| $\alpha_0$ | Reward extracting useful information from incorrect answers (critical reasoning) |
| $\alpha_1$ | Reward being persuaded by correct information (persuadability) |
| $\beta_0$ | Reward providing incorrect yet useful answers (constructive influence) |
| $\beta_1$ | Reward persuading others with correct answers (persuasiveness) |

Detailed implementation: Positive or negative incentives are distributed based on the changes in answer correctness of the agent across consecutive turns, as well as the direction of the majority vote change.

#### Key Design 3: Multi-Agent PPO

- Each agent possesses its own independent policy network and value network for each turn.
- The state is defined as the full multi-agent interaction history.
- Loss function = PPO surrogate loss + value function MSE loss.
- KL regularization is incorporated to prevent the policy from deviating too far from the reference model.
- GAE (Generalized Advantage Estimation) is utilized to estimate the advantage function.

### Game-Theoretic Analysis

The paper proves using a simplified model (2 turns, 2 agents, 2 actions: cooperate / act independently) that:
- **Observation 1**: In single-agent training, if the opponent's probability of cooperation is not high enough, the optimal strategy is not to cooperate.
- **Observation 2**: When two agents are co-trained, as long as the synergistic payoff $R_{syn}$ is sufficiently large, the equilibrium naturally leads to cooperation.

## Key Experimental Results

### Datasets
- **GSM8K**: Grade school math reasoning (7,463 training + 12,800 MAPoRL training + 1,319 testing)
- **ANLI**: Adversarial Natural Language Inference (10,000 verifier training + 12,800 MAPoRL training + 1,200 testing)

### Models
- Primary model: Phi-3-mini-128k-instruct (3.4B), fine-tuned with QLoRA quantization.
- Auxiliary models: Qwen2.5-3B-instruct, Llama-3-8B-instruct.

### Experiment 1: Off-the-Shelf vs. MAPoRL Training

| Setting | GSM8K (T1→T2→T3) | ANLI (T1→T2→T3) |
|------|-------------------|------------------|
| Phi-3 Off-the-shelf | No performance gain or even degradation | No performance gain or even degradation |
| MAPoRL Training | Continuous improvement over turns | Continuous improvement over turns |

Key Finding: When testing the MAPoRL-trained model individually (without collaboration), its performance is on par with the original model (GSM8K: 0.609 vs. 0.604/0.611), which proves that the improvement stems from **collaborative capabilities** rather than task-specific knowledge.

### Experiment 2: Incentive Parameter Analysis

- Increasing $\alpha_1=2$ leads to a 9.5% improvement in $\Delta_1$—enhancing the ability to follow the correct majority opinion.
- Increasing $\beta_0=2$ leads to a 17.2% improvement in $\Delta_0$—significantly enhancing the ability to provide "constructive incorrect answers".
- Increasing $\beta_1$ is conversely ineffective, indicating that directly rewarding "persuading with correct answers" is less effective than rewarding "providing useful information".

### Experiment 3: Cross-Domain Transfer

| Training $\rightarrow$ Evaluation | Turn 1 | Turn 2 | Turn 3 |
|-----------|--------|--------|--------|
| ANLI $\rightarrow$ GSM8K (Off-the-shelf) | 0.677 | 0.688 | 0.640 |
| ANLI $\rightarrow$ GSM8K (MAPoRL) | 0.677 | 0.712 | **0.720** |
| GSM8K $\rightarrow$ ANLI (Off-the-shelf) | 0.482 | 0.486 | 0.468 |
| GSM8K $\rightarrow$ ANLI (MAPoRL) | 0.482 | 0.499 | **0.507** |

Demonstrates that collaborative capabilities can generalize across tasks.

### Experiment 4: Heterogeneous Model Collaboration

Heterogeneous pairings of Phi-3 (3.4B) + Qwen2.5 (3B) and Phi-3 + Llama-3 (8B), when co-trained via MAPoRL, outperform single models, indicating that models with differing capabilities collaborate more effectively.

## Highlights & Insights

1. **First to systematically** train a multi-LLM collaborative system using multi-agent RL (instead of SFT), supported by game-theoretic analysis.
2. **Counter-intuitive findings in incentive design**: Rewarding "constructive incorrect answers" ($\beta_0$) is more effective than rewarding "persuading with the correct answer" ($\beta_1$).
3. **Collaborative capability as a meta-skill**: MAPoRL does not learn task-specific knowledge, but rather cross-domain collaborative strategies.
4. **High practical value**: Applicable to any multi-LLM system equipped with a verifier.

## Limitations & Future Work

1. Due to computational resource constraints, experiments are primarily conducted on quantized 3-4B models; the effectiveness on larger models remains unverified.
2. Only the debate framework was tested; other multi-agent collaboration frameworks (e.g., consensus, division of labor) remain unexplored.
3. The optimal configurations of incentive parameters require manual tuning, lacking an automated selection mechanism.
4. The quality of the verifier directly impacts training efficacy, but the training of the verifier itself has not been studied in depth.

## Related Work & Insights

- **Multi-LLM Collaboration**: Du et al. (2024) multi-agent debate, Li et al. (2023) prompt-based multi-agent systems
- **LLM Post-Training**: Single-agent post-training in RLHF/PPO paradigms
- **Contemporaneous Work**: Subramaniam et al. (2025) and Zhao et al. (2025) train multi-LLM collaboration using iterative SFT (rather than RL)

## Rating

⭐⭐⭐⭐ (4/5)

The combination of theoretical analysis (from a game-theoretic perspective) and empirical validation is solid, proposing for the first time a post-training paradigm of multi-agent RL co-training. However, the scale of experiments is limited by computational resources (only 3-4B models with QLoRA), and verification is conducted on only two benchmarks. The analysis of the incentive mechanism is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](../../NeurIPS2025/reinforcement_learning/repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ACL 2025\] Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient](bypass_back-propagation_optimization-based_structural_pruning_for_large_language.md)
- [\[ACL 2026\] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](../../ACL2026/reinforcement_learning/why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)
- [\[ICLR 2026\] Representation-Based Exploration for Language Models: From Test-Time to Post-Training](../../ICLR2026/reinforcement_learning/representation-based_exploration_for_language_models_from_test-time_to_post-trai.md)

</div>

<!-- RELATED:END -->

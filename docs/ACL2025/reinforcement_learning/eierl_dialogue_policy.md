---
title: >-
  [Paper Note] An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals
description: >-
  [ACL 2025][Reinforcement Learning][task-oriented dialogue] This paper is the first to apply Evolutionary Reinforcement Learning (ERL) to the task-oriented dialogue policy task. It proposes the EIERL method, which combines the global exploration of Evolutionary Algorithms (EA) with the local optimization of Deep Reinforcement Learning (DRL). It addresses the slow evolution of EA in the large search space of natural language through an Elite Individual Injection (EII) mechanism…
tags:
  - "ACL 2025"
  - "Reinforcement Learning"
  - "task-oriented dialogue"
  - "evolutionary algorithm"
  - "elite injection"
  - "exploration-exploitation"
date: 2026-05-08
content_hash: 59decb0cedc83ce4
---

# An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals

**Conference**: ACL 2025  
**arXiv**: [2506.03519](https://arxiv.org/abs/2506.03519)  
**Code**: [GitHub](https://github.com/niulinbiao/eierl)  
**Area**: Reinforcement Learning  
**Keywords**: task-oriented dialogue, reinforcement learning, evolutionary algorithm, elite injection, exploration-exploitation

## TL;DR

This paper is the first to apply Evolutionary Reinforcement Learning (ERL) to the task-oriented dialogue policy task. It proposes the EIERL method, which combines the global exploration of Evolutionary Algorithms (EA) with the local optimization of Deep Reinforcement Learning (DRL). It addresses the slow evolution of EA in the large search space of natural language through an Elite Individual Injection (EII) mechanism, achieving a more efficient exploration-exploitation balance across four datasets.

## Background & Motivation

Task-oriented dialogue (TOD) systems aim to understand user intent, generate responses, and guide the dialogue to achieve goals, where the dialogue policy (DP) module is responsible for selecting the optimal action in each turn. Deep reinforcement learning (DRL) is the mainstream method to optimize DP, but it faces a core challenge—the **exploration-exploitation balance**:

1. **The Dilemma of DRL**: Insufficient exploration leads to local optima (learning only suboptimal policies), while excessive exploration results in low training efficiency. Conventional exploration strategies such as $\epsilon$-greedy have limited effectiveness in high-dimensional dialogue state-action spaces.
2. **Methods to directly boost exploration** (e.g., ICM_DQN, NOISY_DQN): These involve high design costs and are often limited to specific domains. Encouraging "curiosity-driven exploration of new states" in highly goal-driven dialogue tasks is counterproductive.
3. **Methods to indirectly boost exploration** (expert knowledge guidance, high-quality user simulators): These require additional construction costs, and data quality and simulator realism are difficult to guarantee.
4. **LLM-based solutions**: Although they possess strong language capabilities, they have limited decision-making abilities in DP tasks and high fine-tuning costs. Experiments also demonstrate that GPT-4's success rate in DP tasks is inferior to a fully trained DQN.

**Evolutionary Algorithms (EA)** are theoretically suitable for addressing the exploration-exploitation balance because they naturally possess global search capabilities by maintaining population diversity. However, the lack of gradient information in EA leads to low exploitation efficiency. Furthermore, the **search space of dialogue tasks is far larger than that of game tasks** (due to the flexibility of natural language), causing EA to take an excessively long time to evolve effective policies.

This paper proposes EIERL, which combines the complementary advantages of EA and DRL, and utilizes an EII mechanism to solve the core bottleneck of slow evolution of EA in dialogue tasks.

## Method

### Overall Architecture

EIERL comprises two major modules that form a collaborative exploration-exploitation loop:

- **Exploitation Module**: A DRL agent performs gradient optimization on experiences sampled from the replay buffer. After training, it is replicated into a DRL population (multiple agent copies).
- **Exploration Module**: This consists of an EA submodule and an EII submodule. The EA submodule performs selection, crossover, and mutation operations on the EA population and the DRL population to generate a new EA population. The EII submodule adaptively injects the optimal individual into the EA population to accelerate evolution.

Both populations jointly interact with the dialogue environment to generate experiences, which are stored in a shared experience replay buffer.

### Key Designs

1. **DRL-EA Population Collaboration Mechanism**:

    - **Function**: Achieve complementary exploration and exploitation, where the DRL population provides high-quality policies (exploitation-driven) and the EA population maintains policy diversity (exploration-driven).
    - **Mechanism**: The DRL agent is trained using the standard DQN algorithm to minimize the TD loss $\mathcal{L}(\theta_Q) = \mathbb{E}[(y_i - Q_{\theta_Q}(s,a))^2]$, and is then replicated to form the DRL population. The EA population retains individuals in high-fitness regions through tournament selection, followed by genetic crossover and probabilistic mutation (perturbing network weights with a normal distribution) to generate new policies.
    - **Experience Sharing**: To keep training costs consistent, experiences from each population individual are sampled at a ratio of $1/M$ (where $M$ is the total number of individuals) and stored in the shared replay buffer.
    - **Design Motivation**: DRL excels at local optimization using gradients but easily falls into local optima; EA excels at global search through population diversity but lacks gradient information. Combining the two compensates for their respective weaknesses.

2. **Elite Individual Injection (EII) Mechanism**:

    - **Function**: Solve the problem of slow EA evolution in the large search space of dialogue tasks, providing a clear evolutionary direction for the EA population.
    - **Mechanism**: An elite discriminator is established to maintain the historical maximum fitness threshold $f_{max}$ (initialized to $-\infty$). In each iteration, the fitness (cumulative dialogue reward) of all individuals is evaluated. When the fitness of an individual exceeds $f_{max}$, elite injection is triggered: the optimal individual $\pi_{max}$ is injected into the EA population, and $f_{max}$ is updated to the new maximum value.
    - **Adaptive Characteristic**: As training progresses, $f_{max}$ continuously increases, making the injection criteria increasingly stringent—easily triggered in the early stage (rapidly guiding the direction) and hard to trigger in the late stage (avoiding over-intervention).
    - **Injection Effect**: The elite individual participates in subsequent EA crossover operations, diffusing its high-quality genes into the EA population to guide the entire population to evolve in a better direction.
    - **Design Motivation**: When traditional ERL is directly migrated to dialogue tasks, since the EA starts from a population with low fitness, it explores a vast number of ineffective regions in the huge search space, leading to excessively long evolution times. Elite injection acts as a "lighthouse" for the EA.

3. **Fitness Evaluation**:

    - **Function**: Provide decision-making bases for the elite discriminator and population ranking.
    - **Mechanism**: Each individual interacts with the dialogue environment for $\xi$ complete dialogues, and the cumulative reward of all dialogue turns is used as fitness. An $\epsilon$-greedy strategy (small probability for random actions, large probability for selecting the action with the maximum Q-value) is adopted.
    - **Reward Design**: A success reward of $+2L$, a failure penalty of $-L$, and a fixed turn cost of $-1$ (to encourage concise dialogues), where $L$ is the maximum number of dialogue turns.

### Loss & Training

**DRL Part**: Standard TD loss for DQN

$$\mathcal{L}(\theta_Q) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}[(y_i - Q_{\theta_Q}(s,a))^2]$$

where $y_i = r + \gamma \max_{a'} Q'_{\theta_{Q'}}(s', a')$, and $\gamma = 0.99$.

**EA Part**: Gradient-free—evolving the policy network via tournament selection + genetic crossover + probabilistic mutation (weight perturbation with a normal distribution $\mathcal{N}(0, \sigma)$).

**Training Configuration**:
- Network structure: A two-layer MLP with 80 hidden units per layer and ReLU activation.
- Single-domain task: EA population size $P=3$, DRL population size = 1, mutation strength $\sigma=0.1$, 500 epochs.
- Multi-domain task (MultiWOZ): EA population size $P=10$, DRL population size = 5, 10,000 epochs.
- Warm start: 120 epochs to pre-fill the replay buffer.
- Mini-batch size 16, learning rate 0.001, buffer capacity 5000.
- Averaged over 5 random seeds.

## Key Experimental Results

### Main Results

Success Rate at Epoch=500:

| Method | Movie | Restaurant | Taxi |
|------|-------|-----------|------|
| DQN_ε=0.0 | 0.5553 | 0.5671 | 0.5879 |
| DQN_ε=0.05 | 0.7668 | 0.5817 | 0.6683 |
| NOISY_DQN | 0.7280 | 0.2988 | 0.2615 |
| ICM_DQN | 0.5311 | 0.0082 | 0.0706 |
| LLM_DP (GPT-4) | 0.4156 | 0.3896 | 0.3496 |
| LLM_DP_NLG | 0.2564 | 0.2498 | 0.2395 |
| **EIERL** | **0.8552** | **0.7935** | **0.8159** |

Average Reward at Epoch=500:

| Method | Movie | Restaurant | Taxi |
|------|-------|-----------|------|
| DQN_ε=0.05 | 43.42 | 12.79 | 20.19 |
| **EIERL** | **55.29** | **34.99** | **35.39** |

### Ablation Study

| Configuration | Key Conclusion | Explanation |
|------|---------|------|
| EIERL Complete | Optimal | EA+DRL+EII all components |
| ERL (without EII) | Slow convergence, unstable | Proves the critical role of EII in accelerating evolution |
| DQN Only | Converges to suboptimal policies | Insufficient exploration |
| EA Only | Almost no improvement | Lacks gradients, directionless in large search space |

### Key Findings

1. **EIERL comprehensively and significantly leads**: Success rate improves by 8.8–21.2 percentage points over the best DRL baseline on three single-domain tasks, and by 41–44 percentage points over GPT-4 DP.
2. **EII mechanism is critical**: Compared to ERL (without EII), EIERL has smoother learning curves and faster convergence, with more pronounced advantages in the complex Restaurant and Taxi domains.
3. **NOISY_DQN and ICM_DQN collapse in complex domains**: Success rates in the Restaurant domain are only 29.88% and 0.82%, indicating that general exploration strategies are unsuitable for goal-oriented dialogue tasks.
4. **LLMs perform poorly on DP tasks**: GPT-4's success rate is only 35–42%, far lower than trained DQN, proving that LLM language capability $\neq$ decision-making capability.
5. **Sensitivity to EA hyperparameters is controllable**: Population size $P=3$ and mutation strength $\sigma=0.1$ are the optimal default values. Extremely large or small values degrade performance, but the overall framework is robust.
6. **MultiWOZ multi-domain validation**: EIERL consistently performs the best on the 7-domain multi-domain task, proving the generalizability of the framework.

## Highlights & Insights

- **First to apply ERL to dialogue policies**: While ERL research in the game domain is mature, the search space for dialogue tasks is larger (due to natural language flexibility). This work successfully adapts it to DP tasks and solves the issue of slow evolution.
- **Simple and elegant adaptive threshold design of EII**: Utilizing only a dynamically updated fitness threshold, it achieves the effect of "frequent injections in the early stage to guide direction, and sparse injections in the late stage to avoid interference" without requiring extra hyperparameters.
- **Re-proving the value of lightweight RL in decision-making tasks in the LLM era**: GPT-4's language capability cannot compensate for its deficiencies in sequential decision-making, showing that dialogue policy optimization still requires specialized RL methods.

## Limitations & Future Work

- **Evaluation limited to user simulators**: All experiments are evaluated on simulated dialogue environments (Microsoft Dialogue Challenge, ConvLab) without verification through real-user interactions.
- **Increased computational cost**: Multiple individuals in the EA population need to interact with the environment. Although the buffer cost is controlled via $1/M$ sampling, the total number of interactions increases.
- **Using only DQN as the DRL foundation**: More advanced RL algorithms like PPO and SAC, which inherently feature better exploration-exploitation characteristics, have not been investigated.
- **Limited coverage of dialogue tasks**: Only information-querying tasks (booking tickets/restaurants/taxis) are involved. It has not been verified in more complex dialogue scenarios (e.g., negotiation, dialogue mixing chit-chat and tasks).
- **Insufficient comparison with recent LLM-as-agent work**: GPT-4 is only used as a simple DP replacement without considering more elaborate LLM agent frameworks (such as ReAct, ToT).

## Related Work & Insights

- **vs Standard DRL (DQN/PPO)**: EIERL enhances exploration via EA population diversity to avoid local optima in high-dimensional dialogue spaces.
- **vs Game-domain ERL (ERL-Re2, AERL, etc.)**: This work is the first to adapt ERL to dialogue tasks. The core contribution is using EII to solve the slow evolution of EA caused by the dialogue search space.
- **vs LLM dialogue agents**: In DP tasks that require precise decision-making, lightweight RL remains a better choice—LLMs are strong in comprehension but weak in decision-making.
- **Insights**: (1) Cross-domain technology migration (games $\rightarrow$ dialogue) requires domain-specific adaptation mechanisms; (2) Adaptive mechanisms (like the dynamic threshold of EII) are more suitable for complex tasks than fixed hyperparameters.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce ERL into dialogue policies, with an innovative EII mechanism design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 4 datasets (3 single-domain + 1 multi-domain), multiple baselines (including LLM), with comprehensive ablation and hyperparameter analyses.
- Writing Quality: ⭐⭐⭐ Detailed but slightly verbose algorithm description; many symbols increase the cognitive load for reading.
- Value: ⭐⭐⭐ Contributes to RL research in dialogue systems, but the impact is somewhat limited due to evaluation on simulators and specific DP scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TreeRL: LLM Reinforcement Learning with On-Policy Tree Search](treerl_tree_search_rl.md)
- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](../../ICLR2026/reinforcement_learning/efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[ACL 2026\] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](../../ACL2026/reinforcement_learning/breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)
- [\[ICLR 2026\] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](../../ICLR2026/reinforcement_learning/helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [\[ICLR 2026\] Asynchronous Policy Gradient Aggregation for Efficient Distributed Reinforcement Learning](../../ICLR2026/reinforcement_learning/asynchronous_policy_gradient_aggregation_for_efficient_distributed_reinforcement.md)

</div>

<!-- RELATED:END -->

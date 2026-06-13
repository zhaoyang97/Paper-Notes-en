---
title: >-
  [Paper Note] MAT-Agent: Adaptive Multi-Agent Training Optimization
description: >-
  [NeurIPS 2025][LLM Agent][multi-agent systems] This paper proposes MAT-Agent, a multi-agent framework consisting of four autonomous agents responsible for data augmentation, optimizer selection, learning rate scheduling…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "multi-agent systems"
  - "training optimization"
  - "multi-label classification"
  - "reinforcement learning"
  - "dynamic configuration"
date: 2026-05-08
content_hash: 1f35c2187923ec94
---

# MAT-Agent: Adaptive Multi-Agent Training Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.17845](https://arxiv.org/abs/2510.17845)  
**Code**: None  
**Area**: Agent
**Keywords**: multi-agent systems, training optimization, multi-label classification, reinforcement learning, dynamic configuration

## TL;DR
This paper proposes MAT-Agent, a multi-agent framework consisting of four autonomous agents responsible for data augmentation, optimizer selection, learning rate scheduling, and loss function selection, respectively. The framework dynamically adjusts training configurations during the training process, employing DQN to learn policies as a replacement for conventional static hyperparameter configurations, and achieves state-of-the-art performance on multi-label image classification tasks.

## Background & Motivation
**Background**: Multi-label image classification (MLIC) training typically fixes hyperparameter configurations—including data augmentation, optimizer, learning rate schedule, and loss function—prior to training, or applies heuristic adjustments only at predefined milestones.

**Limitations of Prior Work**: Static configurations are incapable of adapting to the evolving label co-occurrence patterns, class difficulty, and feature-label mappings that emerge during training, resulting in training instability, premature convergence, and limited performance.

**Key Challenge**: The training process is inherently non-stationary—different stages require different strategy combinations (more exploration in early stages, fine-grained tuning in later stages, and specialized handling for long-tail classes)—yet conventional methods treat configuration search as a one-shot static decision. Furthermore, nonlinear interactions exist among components, and independent tuning neglects synergistic effects.

**Goal**: (1) How to adaptively adjust multiple training components in real time during training; (2) How to capture synergistic effects among components for joint optimization; (3) How to strike a balance between exploring new strategies and exploiting known effective ones.

**Key Insight**: The training optimization problem is reformulated as a multi-agent sequential decision-making problem, where each agent is responsible for one training component and learns optimal policies online through interaction with the training process.

**Core Idea**: Four DQN agents collaboratively select data augmentation, optimizer, learning rate, and loss function combinations in real time during training, transforming static hyperparameter search into dynamic policy learning.

## Method

### Overall Architecture
MAT-Agent comprises four autonomous agents, each controlling one training component: Agent_AUG (data augmentation), Agent_OPT (optimizer), Agent_LRS (learning rate scheduling), and Agent_LOSS (loss function). At each decision step $t$, the system perceives the current training state $s_t$, and each agent selects an action according to its policy network. These actions are combined into a global configuration $\mathbf{C}_t = (a_t^{\text{AUG}}, a_t^{\text{OPT}}, a_t^{\text{LRS}}, a_t^{\text{LOSS}})$ and applied to the next training iteration. After training, the outcome is evaluated to produce a reward signal, which is used to update the agents' policies, forming a closed loop of "perception → decision → execution → evaluation → learning."

### Key Designs

1. **State Representation $s_t$**:

    - Function: Encodes the current training state as shared input for all agents.
    - Mechanism: The state vector $s_t = [s_t^{\text{perf}}; s_t^{\text{dyn}}; s_t^{\text{data}}]$ incorporates three categories of information—performance metrics (validation mAP), training dynamics (training/validation loss, loss delta, gradient L2 norm, relative update magnitude), and data characteristics (e.g., sample texture richness). An extended representation $\mathcal{I}_t$ that concatenates historical states is also constructed to support temporal reasoning.
    - Design Motivation: A comprehensive state representation enables agents to perceive the global training situation rather than merely the current loss, while historical information facilitates trend inference.

2. **DQN-Based Agent Decision Making**:

    - Function: Each agent independently learns a Q-function to select optimal actions.
    - Mechanism: Each Agent_k approximates $Q_k(\mathcal{I}_t, a; \theta_k)$ using a deep Q-network and employs an $\epsilon$-greedy policy to balance exploration and exploitation. Experience replay and target networks are used to stabilize training. The TD loss is $L_j(\theta_k) = (y_j - Q_k(\mathcal{I}_j, a_j^k; \theta_k))^2$, where $y_j = R_{j+1} + \gamma \max_{a'} Q_k(\mathcal{I}_{j+1}, a'; \theta_k^-)$. Curiosity-driven intrinsic rewards based on state transition prediction error are further introduced to enhance exploration.
    - Design Motivation: DQN can efficiently learn value functions over finite discrete action spaces. The decay schedule of $\epsilon$-greedy ensures sufficient exploration of the strategy space in early stages and convergence to effective strategies in later stages.

3. **Composite Reward Function**:

    - Function: Evaluates the overall effectiveness of the joint configuration.
    - Mechanism: $R_{t+1} = w_{\text{mAP}} \cdot f(\Delta\text{mAP}_t) + w_{\text{stab}} \cdot \text{Stability}_t + w_{\text{conv}} \cdot \text{Convergence}_t - w_{\text{pen}} \cdot \text{Penalty}_t$, balancing accuracy improvement, training stability, convergence speed, and computational cost.
    - Design Motivation: Using accuracy alone as the reward would bias agents toward selecting strategies that are effective in the short term but unstable. The multi-objective reward design guides agents to simultaneously attend to convergence quality.

4. **Inter-Agent Coordination Mechanism**:

    - Function: Shared reward signals and state representations promote cooperation.
    - Mechanism: All four agents receive the same global reward $R_{t+1}$ (rather than independent individual rewards) and share the state $\mathcal{I}_t$. This encourages each agent to optimize the global objective rather than a local one, indirectly achieving joint policy optimization.
    - Design Motivation: Independent rewards may lead to conflicting policies among agents (e.g., one agent selecting aggressive augmentation while another selects a conservative loss). A global reward enables natural coordination.

### Loss & Training
- Dual-rate exponential moving average (EMA) is adopted to smooth strategy transitions and avoid training instability caused by abrupt configuration changes.
- Mixed-precision training is supported to improve efficiency.
- The configuration space is the Cartesian product of individual agent action spaces, $|\mathcal{C}| = \prod_{k \in \mathcal{K}} |\mathcal{A}_k|$. By decomposing the search into independent agents, the exponential search space is reduced to a linear one.

## Key Experimental Results

### Main Results
Comparisons against 8 state-of-the-art methods on three datasets—Pascal VOC, MS-COCO, and VG-256:

| Method | Pascal VOC mAP | COCO mAP | VG-256 mAP | COCO OF1 | COCO CF1 |
|--------|---------------|----------|------------|----------|----------|
| ML-GCN | 94.0 | 83.0 | 52.3 | 80.3 | 78.0 |
| ASL | 95.8 | 86.6 | 56.3 | 81.9 | 81.4 |
| HSQ-CvN | 96.4 | 92.0 | - | 87.5 | 86.6 |
| PAT-T | 96.2 | 91.8 | 59.5 | 87.6 | 86.4 |
| **MAT-Agent** | **97.4** | **92.8** | **60.9** | **88.2** | **87.1** |

Cross-domain transfer experiments (MS-COCO → other datasets, zero-shot mAP):

| Method | → VOC | → NUS-WIDE | → OpenImages |
|--------|-------|------------|--------------|
| DARTS | 73.8 | 59.7 | 50.8 |
| **MAT-Agent** | **76.2** | **62.5** | **53.4** |

### Ablation Study
Component ablation on Pascal VOC:

| Configuration | mAP | Note |
|---------------|-----|------|
| Full MAT-Agent | **97.4** | Complete model |
| w/o AUG | ~95.5 | Removing augmentation agent; long-tail robustness degrades |
| w/o OPT | ~95.8 | Removing optimizer agent; convergence slows |
| w/o LRS | ~96.0 | Removing LR agent; late-stage performance limited |
| w/o LOSS | ~95.3 | Removing loss agent; class imbalance worsens |
| w/o AUG+OPT | ~93.5 | Removing two agents; performance drops sharply |
| w/o All Agents | 91.7 | Degenerates to static configuration |
| w/o Agent Coordination | ~96.2 | Agents present but uncoordinated; below full model |

### Key Findings
- Removing any single agent leads to a performance drop of 0.8–2.1 points, indicating that adaptive control of all four components is necessary.
- Removing multiple agents simultaneously causes nonlinear performance degradation (w/o AUG+OPT drops more than the sum of individual removals), confirming synergistic effects among agents.
- Training convergence is approximately 47% faster than standard training (reaching the performance level of 80 standard epochs at epoch 47).
- Across different target domains, agents adaptively shift their focus: long-tail domains place greater emphasis on CB Loss, high visual complexity domains favor CutMix, and AdamW+OneCycleLR remains consistently stable across all domains.

## Highlights & Insights
- **Modeling training optimization as multi-agent decision making is an interesting perspective**: It shifts from the conventional AutoML "one-shot search" paradigm to online adaptation throughout the training process, better addressing non-stationarity. This framework can be transferred to any task requiring dynamic hyperparameter tuning.
- **Cross-domain analysis of attention distributions is insightful**: By observing differences in agent policy distributions across domains (Figure 3), one can intuitively understand what training strategies are required for different scenarios, providing a degree of interpretability.
- **Dual-rate EMA for smooth strategy transitions**: A simple yet effective engineering technique that prevents abrupt policy switches by agents from causing training oscillations.

## Limitations & Future Work
- **Validation limited to multi-label classification**: Although the framework is general in principle, experiments are restricted to MLIC tasks; effectiveness on other tasks such as detection, segmentation, or generation remains undemonstrated.
- **Overhead of agents not thoroughly analyzed**: While the training and inference cost of four DQN agents (GPU memory, time overhead) is more efficient than grid search, the additional cost relative to direct training is not quantitatively analyzed.
- **Predefined action space is restrictive**: The candidate actions for each agent (augmentation strategies, optimizers, etc.) are manually predefined as finite sets, precluding the discovery of entirely novel strategies.
- **Agents coordinate only indirectly through shared rewards**: The absence of an explicit communication mechanism prevents agents from directly observing other agents' decisions, potentially leading to suboptimal joint policies.

## Related Work & Insights
- **vs AutoML/NAS (ENAS, DARTS)**: AutoML methods search for optimal configurations before or early in training; this work dynamically adjusts throughout the entire training process, offering greater flexibility at the cost of increased complexity.
- **vs PBT (Population-Based Training)**: PBT tunes hyperparameters through population-based evolution, whereas this work employs RL agents to learn policies online, achieving faster convergence but requiring careful design of state representations and reward functions.
- **vs Manual Tuning**: MAT-Agent automates the tuning process and surpasses manually tuned results on all three datasets.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-agent training optimization framework is a novel perspective, though the core components (DQN, $\epsilon$-greedy) are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, ablation studies, transfer experiments, and convergence analysis are provided, but validation on other tasks is absent.
- Writing Quality: ⭐⭐⭐ The mathematical notation system is complete, but certain descriptions are redundant and the depth of experimental analysis is insufficient.
- Value: ⭐⭐⭐⭐ Offers a new perspective on training optimization, but practical overhead and generalizability require further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Group-in-Group Policy Optimization for LLM Agent Training](groupingroup_policy_optimization_for_llm_agent_training.md)
- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)
- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](../../ACL2026/llm_agent/coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ICLR 2026\] Efficient Agent Training for Computer Use](../../ICLR2026/llm_agent/efficient_agent_training_for_computer_use.md)
- [\[NeurIPS 2025\] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)

</div>

<!-- RELATED:END -->

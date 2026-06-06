---
title: >-
  [Paper Note] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents
description: >-
  [ACL 2026][Reinforcement Learning][Parallel Exploration] The authors propose a new paradigm of "Parallel Exploration" where an agent interacts with $K$ environments synchronously and shares experiences across trajectorie…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Parallel Exploration"
  - "GRPO"
  - "Diversity-Driven Reward"
  - "ALFWorld"
  - "ScienceWorld"
date: 2026-05-08
content_hash: 42ae0447a5c4afed
---

# DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents

**Conference**: ACL 2026  
**arXiv**: [2604.24320](https://arxiv.org/abs/2604.24320)  
**Code**: https://github.com/LePanda026/Code-for-DPEPO  
**Area**: Reinforcement Learning / LLM Agent / GRPO Improvement  
**Keywords**: Parallel Exploration, GRPO, Diversity-Driven Reward, ALFWorld, ScienceWorld

## TL;DR
The authors propose a new paradigm of "Parallel Exploration" where an agent interacts with $K$ environments synchronously and shares experiences across trajectories. They introduce the corresponding RL algorithm **DPEPO**: it utilizes Cold-start SFT to learn parallel reasoning, followed by GRPO training with hierarchical rewards consisting of "trajectory-level success + step-level Diverse Action / Diverse State Transition." DPEPO achieves SOTA results on all splits of ALFWorld and ScienceWorld (98.2% and 61.4% respectively on Qwen2.5-7B), with token growth significantly lower than "multi-sampling" baselines as $K$ increases.

## Background & Motivation

**Background**: LLM-based autonomous agents mostly follow the ReAct paradigm—"Think → Act → Observe → Think Again." On long-horizon tasks like ALFWorld and ScienceWorld, RL post-training methods such as GRPO, GiGPO, RLVMR, and SPEAR have pushed success rates to over 90%.

**Limitations of Prior Work**: In ReAct, observing only a single trajectory per step leads to a **narrow and one-sided** understanding of the environment (termed a "narrow, linear view"). The most intuitive extension—sampling multiple independent trajectories for the same task—suffers from two fatal flaws: (1) **Lack of Diversity**: despite increased output entropy, actions from multiple samplings tend to converge to similar choices; (2) **Efficiency and Isolation**: there is no experience sharing between trajectories, and sequential sampling causes token consumption and time to expand linearly.

**Key Challenge**: Traditional RL uses "multi-sampling" for exploration in LLM agents, but explored samples neither learn from each other nor avoid computational waste. Agents are forced to choose between a "narrow view vs. high cost," making it impossible to understand the environment both broadly and quickly.

**Goal**: To allow an agent to **synchronously** interact with multiple environments of the same initial state within a single episode, share intermediate observations across environments, and design an RL reward that **actively counters redundancy**.

**Key Insight**: Upgrade from "one-to-one agent-environment" to "one-to-many agent-environment set $\mathcal{E}=\{E_1,\ldots,E_K\}$." At each step, the agent autonomously selects a subset $\mathcal{E}'_t\subseteq\mathcal{E}$ and initiates parallel actions $A_t=\{(E_i, a_t)\}$, obtaining parallel states $S_t$ concurrently.

**Core Idea**: **Parallel Exploration as a Paradigm + Diversity as a Reward**. By treating parallel exploration as a paradigm-level extension of ReAct and using step-level diversity rewards to encode "do not repeat behaviors" explicitly into the policy gradient, the agent becomes both broad and precise.

## Method

### Overall Architecture
DPEPO is a parallel extension of GRPO. Given a task and $K$ parallel environments $\mathcal{E}$ sharing an initial state, the agent selects $\mathcal{E}'_t$ at each step $t$, generates actions for each selected environment to form $A_t$, and executes them to obtain $S_t$, resulting in a trajectory $\tau=\{(S_t, A_t)\}_{t=1}^T$. Training consists of two stages: (1) Cold-start SFT, using 5 human-annotated ground-truth parallel trajectories to guide DeepSeek-V3.2 in synthesizing 500/1000 SFT data points (ALFWorld/ScienceWorld) so the model learns the "parallel thinking output format"; (2) RL stage, optimized using hierarchical rewards—parallel trajectory-level success reward + two step-level diversity rewards—combined with GRPO's group-relative advantage. During inference, group size $N=8/4$, $K=4$, $T_{\max}=25$, and temperature is 0.4.

### Key Designs

1. **Parallel Trajectory-level Success Reward ("OR" Semantics for Parallel Success)**:

    - **Function**: Extends the traditional ReAct 0/1 task success reward to the parallel setting.
    - **Mechanism**: $R_{traj}(\tau)=1$ if $S_T\cap\mathcal{G}\neq\emptyset$ (success if any parallel environment reaches the goal state), otherwise 0. Like GRPO, it uses group-relative advantage based on the mean and standard deviation of $N$ group trajectories: $\Phi_{traj}(\tau_i)=(R(\tau_i)-\text{mean})/\text{std}$.
    - **Design Motivation**: The "OR" semantics of "any environment success counts as success" encourages the agent to treat multiple environments as redundant backups and information sources—exploring with one hand and betting with the other, consistent with the nature of parallel computing. Group-relative advantage eliminates dependency on a critic model.

2. **Diverse Action Reward (Bi-directional De-duplication of Depth × Width)**:

    - **Function**: Directly penalizes action repetition within a step, encouraging the agent to "deal different cards" across environments.
    - **Mechanism**: Defined as $R_{action}(A_t)=\frac{1}{|\mathcal{E}'_t|}\sum_{E_i\in\mathcal{E}'_t}\alpha^{C_{depth}(E_i, a_t)}+\omega^{C_{width}(A_t)}$. Here, $C_{depth}$ is the repetition count of action $a_t$ in the history of environment $E_i$ (Depth = repeated attempts in the same environment); $C_{width}$ is the count of repeated actions within the current step $A_t$ (Width = same action sent to multiple environments); $\alpha, \omega\in(0,1]$ are discount factors. Higher repetition leads to rewards closer to 0.
    - **Design Motivation**: Transforms "diversity" from a vague concept into a differentiable dense reward signal. The depth dimension avoids "spinning in place," while the width dimension prevents "copy-pasting," forcing $K$ parallel environments to explore complementary state spaces.

3. **Diverse State Transition Reward + Hierarchical Advantage Combination**:

    - **Function**: Further penalizes repetitive transitions (doing the same action after seeing the same state) and embeds step-level rewards into trajectory-level advantages.
    - **Mechanism**: Defines transition $p_t = s_t\to a_t$, $R_{transition}=\frac{1}{|\mathcal{E}'_t|}\sum_i\gamma^{M_{depth}(E_i, p_t)} + \frac{1}{|\mathcal{E}'_t|}\sum_i\beta^{M_{width}(E_i, p_t)}$, where $M_{depth}$ counts transition repetitions in $E_i$ and $M_{width}$ counts identical transitions in other selected environments. The step-level rewards are averaged: $R_{step}=(R_{action}+R_{transition})/2$. Combined advantage: if $\Phi_{traj}(\tau_i)>0$, then $\Phi_{step}=R_{step}$; otherwise, $\Phi_{step}=2-R_{step}$. The final advantage is $\Phi(A_{i,t})=\Phi_{step}\cdot\Phi_{traj}$.
    - **Design Motivation**: Evaluating actions alone is insufficient—the same action has different semantic meanings in different states. Transitions are the minimal units of actual "experience." Flipping to $2-R_{step}$ is clever: in successful trajectories, diversity is a bonus, but in failed trajectories, high step-level diversity imply the agent was guessing randomly and should be down-weighted. This enables the model to be "bold in exploration" while "knowing how to converge."

### Loss & Training
The RL stage directly adopts the GRPO policy objective (critic-free, relative advantage). Each task uses 500 RL samples, 125 training steps, 1 epoch. Group size $N$ is 8 for ALFWorld and 4 for ScienceWorld. Max steps is 25, and parallel environment count $K=4$. Cold start uses SFT for behavioral cloning on 500/1000 synthetic parallel trajectories.

## Key Experimental Results

### Main Results
**ALFWorld (Success Rate %) + ScienceWorld (Success Rate %) on Qwen2.5-Instruct**:

| Model / Method | ALFWorld In-Domain | ALFWorld OOD | ALFWorld Avg | SW L0 | SW L1 | SW L2 | SW Avg |
|------------|-------------------|--------------|--------------|-------|-------|-------|--------|
| GPT-4o (closed) | 48.0 | 66.0 | 57.0 | 45.4 | 49.2 | 41.0 | 45.2 |
| DeepSeek-R1 (closed) | 75.0 | 85.1 | 80.1 | 22.2 | 31.4 | 29.1 | 27.6 |
| **1.5B**: GRPO | 72.8 | 71.1 | 72.0 | 21.1 | 13.7 | 10.9 | 15.2 |
| **1.5B**: GiGPO | 86.7 | 83.2 | 85.0 | 25.8 | 15.2 | 4.7 | 15.2 |
| **1.5B**: RLVMR | 89.1 | 87.9 | 88.5 | 46.9 | 34.3 | 26.5 | 35.9 |
| **1.5B**: SPEAR | 93.2 | - | - | - | - | - | - |
| **1.5B**: **DPEPO** | **95.7** | **92.5** | **94.1** | **59.8** | **58.1** | **34.2** | **50.7** |
| **7B**: GRPO | 77.6 | 77.3 | 77.5 | 49.1 | 30.1 | 26.6 | 35.3 |
| **7B**: GiGPO | 90.8 | 90.2 | 90.5 | 53.4 | 25.2 | 25.8 | 34.8 |
| **7B**: RLVMR | 91.4 | 91.8 | 91.6 | 67.2 | 43.0 | 32.2 | 47.5 |
| **7B**: SPEAR | 94.7 | - | - | - | - | - | - |
| **7B**: **DPEPO** | **98.6** | **97.8** | **98.2** | **66.6** | **66.5** | **51.0** | **61.4** |

The 7B DPEPO is 3.5 percentage points higher than the second-best SPEAR (94.7) on ALFWorld, and 13.9 points higher than RLVMR (47.5) on ScienceWorld. Notably, the 1.5B DPEPO on ScienceWorld (50.7) outperforms the 7B RLVMR (47.5).

**Inference Efficiency (ALFWorld, same experimental setup)**:

| Method | Tokens | Steps | Time (s) |
|------|--------|-------|----------|
| DeepSeek-V3 | 950.0 | 20.5 | 62.4 |
| DeepSeek-R1 | 1667.9 | 24.8 | 237.0 |
| GiGPO | 1115.1 | 15.2 | 70.8 |
| **DPEPO** | 2283.4 | **12.3** | **44.7** |

Although token usage is twice that of GiGPO, the wall-clock time is fastest because the number of steps decreases and parallel execution does not increase single-step time.

### Ablation Study

| Configuration | ALFWorld In | ALFWorld OOD | SW L0 | SW L1 | SW L2 |
|------|-------------|--------------|-------|-------|-------|
| ColdStart SFT Only | 93.6 | 97.8 | 66.5 | 62.2 | 48.1 |
| **DPEPO Full** | **98.6** | 97.8 | **66.6** | **66.5** | **51.0** |
| w/o Diverse Action Reward | 97.1 | 97.0 | 65.1 | 62.8 | 49.0 |
| w/o Diverse State Trans. Reward | 96.4 | 98.5 | 64.4 | 63.6 | 49.7 |
| w/o DAR & DTR | 96.4 | 98.5 | 66.3 | 65.7 | 49.9 |

Full DPEPO gains +5.0 on ALFWorld In-Domain and +2.9 on ScienceWorld L2 compared to ColdStart. Removing either diversity reward leads to a drop, but removing both is slightly better than removing just one—indicating that the two rewards are an **interdependent** holistic design.

### Key Findings
- **Parallel Exploration > Multi-sampling**: Figure 4 shows that as $K$ increases on ALFWorld, GiGPO's tokens expand linearly while DPEPO's tokens barely grow. DPEPO outperforms GiGPO across all $K$. This is a paradigm-level (not trick-level) victory.
- **Small Model Comeback**: 1.5B DPEPO exceeds 7B RLVMR on ScienceWorld, showing that paradigm dividends can outweigh model parameter dividends, which is highly valuable for compute-constrained scenarios.
- **Dual Reward Interdependency**: Removing DAR or DTR individually causes performance drops, but removing both is slightly better. This suggests they are coupled designs where independent use introduces misleading gradients.
- **Faster Inference**: DPEPO is 37% faster in wall-clock time than GiGPO and 5× faster than DeepSeek-R1, as parallel actions don't increase step time and the step count is halved.
- **OOD Robustness**: DPEPO remains best on ALFWorld OOD and ScienceWorld L2 (unseen variants + categories), suggesting the "comprehensive cognition" from parallel exploration provides generalization capability beyond merely overfitting the training distribution.
- **4× Training Efficiency**: DPEPO reached SOTA in 24 hours using 500 RL samples, whereas GiGPO took 96 hours. Parallel environment reuse and step-level dense rewards significantly boost sample efficiency.

## Highlights & Insights
- **Paradigm Innovation**: Expanding ReAct from "one environment at a time" to "multiple environments at a time" is a fundamental change, akin to the first systematic implementation of batched RL in the LLM agent field. Future RL-for-agent work will likely need to consider this axis.
- **Diversity-as-Reward Formalization**: Translating the vague intuition of "do not repeat" into two independent dimensions—depth (intra-environment + self-looping) × width (inter-environment + consistency)—and quantifying both with exponential discounts is a clean reward shaping design.
- **Reward Flipping for Failed Trajectories**: The handling of $\Phi_{step}=2-R_{step}$ in failures is clever—encouraging exploration while penalizing "reckless exploration." This outcome-conditioned reward shaping can be transferred to any RLHF task.
- **OR-Semantics Task Success**: Embedding the OR semantics of "any parallel environment success equals overall success" into RL training allows the agent to treat parallel environments as robust hedging rather than requiring success in all. This can be generalized to "multi-model ensemble" scenarios.
- **Constant Tokens, Halved Wall-clock Time**: In LLM agent engineering, the goals of "reducing steps," "reducing tokens," and "reducing time" are often conflated. DPEPO clearly decouples these via inference time data, offering high engineering reference value.

## Limitations & Future Work
- **Dependency on Clonable Environments**: All parallel environments must share an initial state, meaning real-world physical or web environments (with irreversible operations or external API dependencies) are difficult to implement. The authors propose "training agents to complete different tasks in parallel" as future work.
- **Small $K$**: Experiments used $K=4$ and did not explore the marginal returns or token expansion at $K=16/32$. Analysis on whether group size needs to increase with $K$ is missing.
- **SFT Data Quality**: Relying on 5 human trajectories + LLM expansion might lead to varied quality in long-tail tasks. The relationship between cold-start data quality and DPEPO convergence was not detailed.
- **Hypersensitivity of Rewards**: Hyperparameters $\alpha, \omega, \gamma, \beta$ are all in $(0,1]$, but specific values and sensitivity tables are missing.
- **Text-only Environments**: Tested only on ALFWorld and ScienceWorld; transferability to multimodal or real GUI environments remains unverified.

## Related Work & Insights
- **vs. GiGPO (Feng et al. 2025)**: GiGPO uses an anchor-state group to estimate step-level advantage but stays within a single ReAct environment. DPEPO upgrades step rewards from "relative to anchor" to "anti-redundancy" and breaks the single-environment constraint.
- **vs. RLVMR (Zhang et al. 2025)**: RLVMR uses meta-reasoning rewards for ineffective exploration, which is an extension of reward scope. DPEPO changes the paradigm (parallel) rather than just the reward category; the two are orthogonal and combinable.
- **vs. SPEAR (Qin et al. 2025)**: SPEAR uses self-imitation + intrinsic rewards to balance exploration-exploitation. DPEPO uses parallel structures to broaden the exploration space directly, without needing imitation memory.
- **vs. DeepSeek-R1**: R1 is a powerhouse in single-environment CoT reasoning but has the highest step count and time in agent tasks. DPEPO achieves higher performance at the 7B scale and is 5× faster in wall-clock time.
- **vs. Richens et al. 2025 (General agents need world models)**: This work argues that agents need comprehensive world models. DPEPO is an **algorithmic implementation** of this argument—parallel exploration helps agents build more complete environment cognition.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Parallel exploration is a rare paradigm-level innovation in the agent RL field. The diverse reward design is also well-formed. Expect significant follow-up research.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 2 model sizes × 2 benchmarks × 5 baselines + 4 ablations + scaling/efficiency/training-cost. The only gap is verification in real GUI/multimodal environments.
- **Writing Quality**: ⭐⭐⭐⭐ Figure 1 perfectly captures the cognitive gap between ReAct and Parallel. Mathematical notation is rigorous, though SFT synthetic details are slightly brief.
- **Value**: ⭐⭐⭐⭐⭐ High practical value (SOTA with small models + faster wall-clock) + paradigm-level academic contribution (reshaping ReAct) + open-source code. Directly actionable for both agent RL research and industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)
- [\[ICML 2026\] LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation](../../ICML2026/reinforcement_learning/labo_llm-accelerated_bayesian_optimization_through_broad_exploration_and_selecti.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](visually-guided_policy_optimization_for_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning
description: >-
  [ACL2026][LLM Agent][GUI Agent] SOLAR-RL utilizes offline trajectory reconstruction, failure point detection, and goal-aligned reward shaping to process static GUI data into long-horizon training signals with pseudo-onli…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "GUI Agent"
  - "Semi-online Reinforcement Learning"
  - "Long-horizon Task"
  - "Credit Assignment"
  - "Reward Shaping"
date: 2026-05-08
content_hash: 1fa019fa30ef09ae
---

# SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning

**Conference**: ACL2026  
**arXiv**: [2604.22558](https://arxiv.org/abs/2604.22558)  
**Code**: No public code (the paper states it is implemented based on verl)  
**Area**: GUI Agent / Reinforcement Learning / Robotics & Embodied Intelligence  
**Keywords**: GUI Agent, Semi-online Reinforcement Learning, Long-horizon Task, Credit Assignment, Reward Shaping

## TL;DR
SOLAR-RL utilizes offline trajectory reconstruction, failure point detection, and goal-aligned reward shaping to process static GUI data into long-horizon training signals with pseudo-online feedback. This enables Qwen2.5-VL-7B scale GUI agents to achieve stable performance that matches or exceeds strong offline baselines on Android Control, GUI-Odyssey, and Android World.

## Background & Motivation
**Background**: GUI agents are evolving from single-step clicks and element localization toward cross-app, multi-step, and long-horizon tasks. A segment of existing strong methods relies on SFT/behavior cloning to learn from expert demonstrations, while another part uses online RL and environment interaction to collect new trajectories to mitigate covariate shift during deployment.

**Limitations of Prior Work**: Pure SFT is prone to learning "local reactions on expert paths." Once the interface state slightly deviates from the training distribution, the model lacks recovery capabilities. Online RL can obtain real dynamic feedback, but GUI environment interaction is expensive and unstable. Furthermore, tasks exceeding 30 steps often only provide terminal success/failure signals, leading to high training variance, sparse rewards, and policy collapse. While standard offline RL is safe and inexpensive, it often segments static data into local step transitions, discarding global information such as "whether this trajectory succeeded as a whole and where it began to fail."

**Key Challenge**: Long-horizon GUI tasks require online-style credit assignment, yet actual training aims to maintain the controllability and low cost of offline data. The critical problem is not simply increasing the number of trajectories, but how to recover "which prefixes are valid, which action first caused the task to deviate, and how subsequent actions should be penalized" from existing static trajectories.

**Goal**: The authors aim to design a semi-online RL mechanism that transforms static GUI data into multiple trainable candidate trajectories and assigns dense rewards consistent with global completion quality to each step without real environment access.

**Key Insight**: The paper views long-horizon failure as a credit assignment problem. If the first breakdown point can be detected, the valid prefix before the breakdown can be rewarded, the breakdown and subsequent actions can be penalized, and the total return can be calibrated to the trajectory-level quality.

**Core Idea**: Use offline data to simulate branches of online rollouts, and then transform sparse terminal signals into goal-aligned step-wise rewards through failure-point based retroactive credit assignment.

## Method
The core of SOLAR-RL is not changing the GUI agent architecture, but reconstructing the long-horizon optimization problem in terms of training data and reward signals. It uses Qwen2.5-VL-7B-Instruct as the initial policy to generate multiple candidate rollouts on static trajectories, judges the validity of each action through expert labels or rules, and finally trains the policy with shaped rewards.

### Overall Architecture
The method consists of two main modules. First, Offline Trajectory Reconstruction: for each step of the same task, $N$ candidate responses are generated and concatenated into $N$ reconstructed trajectories based on candidate indices; if a trajectory is judged invalid at step $t^*$, it is truncated at that point. Second, Trajectory-Aware Reward Shaping: a step validity score is first calculated based on the action type, and then the valid prefix, invalid suffix, and trajectory-level success/length/quality are synthesized into the final step-wise reward. The training follows a two-stage approach: atomic adaptation followed by trajectory optimization to improve long-horizon stability.

### Key Designs
1.  **Offline Trajectory Reconstruction**:
    - **Function**: Simulates multiple possible online execution paths on static data to expand the exploration space.
    - **Mechanism**: Given a task, $N=8$ candidate rollouts are run at each time step; candidate actions with the same index are concatenated into a trajectory candidate. Although candidates are generated offline, ground-truth validity assessment can determine if a candidate path remains semantically consistent.
    - **Design Motivation**: Standard offline RL only considers expert trajectories or local transitions and cannot observe possible outcomes after deviation. Reconstructing candidate trajectories allows the training to see "different choices starting from the same context," approximating online exploration without real environment costs.

2.  **Failure Point Detection & Prefix Credit Assignment**:
    - **Function**: Locates the position of the first failure in a long-horizon trajectory and concentrates credit on the valid actions before the failure.
    - **Mechanism**: Different validity criteria are used for actions like Click, Scroll, Type, Launch, and Wait/Back—for example, spatial similarity for coordinate actions, F1 for text actions, and exact matching for system actions. If step $t^*$ is the first invalid step, $0$ to $t^*-1$ are considered the valid prefix and receive positive rewards; the breakdown step and subsequent invalid actions receive negative penalties.
    - **Design Motivation**: In long-horizon GUI tasks, terminal failure is often triggered by a single critical early error. If only a negative score is given to the entire failed trajectory, the model cannot identify which preceding actions were actually correct; if every step is rewarded based on local similarity, it may encourage meaningless long sequences. Failure points separate these two aspects.

3.  **Goal-Aligned Reward Shaping**:
    - **Function**: Ensures that the sum of step-wise rewards is consistent with the trajectory-level execution quality while maintaining dense feedback.
    - **Mechanism**: The trajectory-level reward $R_{traj}$ is composed of the average step raw score, the current length relative to the reference length $T/N_{ref}$, and a success indicator. At the step level, valid actions retain positive scores, while invalid actions become $-(1-s_{raw})$, followed by normalization of positive and negative parts. Finally, the reward gap $\Delta=R_{target}-\sum_t r_t^{base}$ is calculated and distributed evenly among the positive steps in the valid prefix.
    - **Design Motivation**: This step addresses two problems: first, the local reward scale is incomparable across different trajectory lengths; second, the model might "game the reward" by lengthening sequences or repeating locally correct actions. Target alignment pulls step-wise rewards back to global goals.

### Loss & Training
The paper trains within the RL framework based on GRPO/verl, with the main change being the reward definition. The policy is initialized with Qwen2.5-VL-7B-Instruct using 15k high-quality static trajectories (approximately 94k steps). The trajectory reconstruction temperature is 1.0, with 8 candidates per step. Training utilizes 32 NVIDIA L40S GPUs, a global batch size of 128, a maximum context length of 6,144 tokens, and 650 update steps over approximately 60 hours. The GRPO baseline uses the same training budget as SOLAR-RL, with the primary difference being sparse trajectory rewards versus trajectory-aware shaped rewards.

## Key Experimental Results

### Main Results

| Model | Training Paradigm | Android Control Low SR | Android Control High SR | GUI-Odyssey TM / EM | Android World SR | Training Data |
|--------|------|------|------|------|------|------|
| Qwen2.5-VL-7B | Generalist | 85.05 | 61.40 | 61.89 / 47.92 | Not reported | No specialized GUI training |
| UI-TARS-7B-SFT | Online specialized | 94.81 | 77.99 | 86.94 / 68.82 | 33.3 | 145K trajectories |
| AgentCPM-GUI-8B | Offline specialized | 88.60 | 67.93 | 90.82 / 74.84 | Not reported | >470K steps, >55K trajectories |
| UI-Venus-Navi-7B | Offline specialized | 86.16 | 68.61 | 87.30 / 71.09 | 49.1 | 350K steps |
| SOLAR-RL | Offline / semi-online shaping | 88.57 | 69.27 | 87.60 / 68.20 | 33.7 | 94K steps, 15K trajectories |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Direct GRPO, Super Long Low | Hard to optimize after 200 steps | Sparse terminal reward causes late-stage collapse |
| Direct SOLAR-RL, Super Long Low | Higher and more stable action SR | Dense reward alleviates long-horizon credit assignment |
| 2-stage GRPO, High Long | Saturates quickly after SR ~0.66-0.67 | Good initialization cannot fully solve sparse long-horizon feedback |
| 2-stage SOLAR-RL, High Long | SR ~0.70 | Trajectory-aware shaping continues to provide gains |
| 2-stage GRPO, High Super Long | SR ~0.58-0.60 with oscillations | Policy tends to stagnate in ultra-long paths |
| 2-stage SOLAR-RL, High Super Long | Peak SR ~0.66 | Advantages in long-horizon tasks are more pronounced |
| PressBack primitive | Accuracy >0.8 and faster convergence | More stable learning of error recovery actions |

### Key Findings
- SOLAR-RL achieves 69.27% SR on Android Control High, the highest in the offline category, surpassing UI-Venus's 68.61% and AgentCPM's 67.93%. This indicates its advantage lies primarily in splits requiring multi-step reasoning.
- On GUI-Odyssey, SOLAR-RL's TM is 87.60, lower than AgentCPM's 90.82; however, AgentCPM uses over 55k trajectories, while SOLAR-RL uses only 15k, highlighting superior sample efficiency.
- On Android World, SOLAR-RL achieves 33.7% SR with 94k steps, slightly higher than UI-TARS-7B-SFT's 33.3%, without requiring online interaction or 145k trajectories.
- Training dynamics show that GRPO's mean action reward experiences policy collapse after approximately 600 steps, while SOLAR-RL improves monotonically and converges around 0.75.

## Highlights & Insights
- This paper most clearly captures the "long-horizon failure attribution" problem in GUI agents. While many GUI RL works emphasize online exploration or reward models, SOLAR-RL focuses on the failure point structure within static data.
- The idea of target-aligned reward shaping is highly practical: dense rewards are not just about spreading the terminal reward across steps, but explicitly constraining total returns to match trajectory quality, preventing local rewards from inducing incorrect goals.
- The semi-online paradigm is suitable for agent tasks where costs are high and real environments are unstable. Similar ideas could be transferred to web automation, desktop operations, robotic offline demonstration learning, and tool-calling agents.
- The results suggest that data scale is not the only variable. Better reward attribution allows 15k trajectories to achieve results comparable to much larger training sets.

## Limitations & Future Work
- Semi-online feedback is still limited by the coverage of the offline dataset. Unseen pop-ups, latencies, rare app states, and cross-platform events cannot be generated from static trajectories.
- The current validity filter relies on ground-truth labels and action type rules. Replacing these with a learned verifier or process reward model could introduce reward noise, calibration drift, and reward hacking.
- Experiments focused on the Android mobile environment. Desktop and browser environments involve hovers, right-clicks, shortcuts, drag-and-drops, multi-windowing, and asynchronous page changes, requiring redesigned validity criteria.
- The paper does not provide interaction evaluations in real online deployments. While SOLAR-RL is effective on static and dynamic benchmarks, it still needs validation on real app version changes and system state drift.
- Ablations are mainly presented as curves and qualitative analysis; providing more final numerical values for ultra-long tasks in table format would facilitate reproduction and horizontal comparison.

## Related Work & Insights
- **vs SFT / Behavior Cloning**: SFT learns expert actions but lacks recovery mechanisms after deviation; SOLAR-RL allows the model to see deviation structures through candidate trajectories and failure points.
- **vs Online RL**: Online RL provides real dynamic feedback but is expensive and has high variance; SOLAR-RL simulates feedback using static data, sacrificing some coverage for stability and low cost.
- **vs UI-S1 / semi-online GUI RL**: UI-S1 uses a patch module to correct bias, while SOLAR-RL emphasizes outcome-aware credit assignment and reward shaping.
- **vs VAGEN / Bi-Level GAE**: VAGEN rewards explicit world modeling and performs hierarchical credit propagation; SOLAR-RL does not rely on an internal world model but constructs rewards from trajectory validity and breakdown locations.

## Rating
- Novelty: ⭐⭐⭐⭐ Semi-online GUI RL is not entirely new, but the combination of failure-point and target-aligned shaping is highly targeted.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three GUI benchmarks and training dynamics analysis, though real-world environment validation remains insufficient.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive illustrations, though readability of some tables and appendix formulas is average in HTML.
- Value: ⭐⭐⭐⭐ Highly practical for training GUI agents at low cost, especially in scenarios with existing offline demonstrations but difficult large-scale online interaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](timem_temporal-hierarchical_memory_consolidation_for_long-horizon_conversational.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ICML 2026\] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](../../ICML2026/llm_agent/on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)

</div>

<!-- RELATED:END -->

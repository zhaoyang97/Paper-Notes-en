---
title: >-
  [Paper Note] SkillMimic: Learning Basketball Interaction Skills from Demonstrations
description: >-
  [CVPR 2025][Reinforcement Learning][Human-Object Interaction Mimicking] SkillMimic is proposed, a purely data-driven framework that learns diverse basketball interaction skills from motion capture data using a unified HOI imitation reward (especially the innovative contact graph reward), and composes these skills using a high-level controller to complete complex long-horizon tasks such as continuous scoring.
tags:
  - "CVPR 2025"
  - "Reinforcement Learning"
  - "Human-Object Interaction Mimicking"
  - "Basketball Skills"
  - "Contact Graph"
  - "Unified Reward Function"
  - "Hierarchical Policy"
date: 2026-05-08
content_hash: d3bbe830273342d2
---

# SkillMimic: Learning Basketball Interaction Skills from Demonstrations

**Conference**: CVPR 2025  
**arXiv**: [2408.15270](https://arxiv.org/abs/2408.15270)  
**Code**: [https://ingrid789.github.io/SkillMimic/](https://ingrid789.github.io/SkillMimic/)  
**Area**: Reinforcement Learning / Human-Object Interaction  
**Keywords**: Human-Object Interaction Mimicking, Basketball Skills, Contact Graph, Unified Reward Function, Hierarchical Policy

## TL;DR

SkillMimic is proposed, a purely data-driven framework that learns diverse basketball interaction skills from motion capture data using a unified HOI imitation reward (especially the innovative contact graph reward), and composes these skills using a high-level controller to complete complex long-horizon tasks such as continuous scoring.

## Background & Motivation

**Background**: In physics-based character animation, imitation learning (e.g., DeepMimic, AMP, ASE) has achieved great success in motion skill learning. However, these methods primarily focus on pure locomotion skills like walking and running, with insufficient attention paid to human-object interaction (HOI).

**Limitations of Prior Work**: Existing HOI methods (such as playing tennis, rope climbing, etc.) require hand-designed reward functions customized for each specific interaction skill, which is labor-intensive and fails to generalize to new interaction modes. In scenarios with highly diverse skills like basketball (dribbling, shooting, layups, etc.), manually designing rewards to cover all skill variants is nearly impossible.

**Key Challenge**: Kinematic-level imitation rewards (which only measure joint position/velocity matching) are insufficient for HOI—they cannot distinguish between "controlling the ball with the hand" and "controlling the ball with the head", causing the humanoid to frequently fall into local optima that are kinematically similar but physically incorrect.

**Goal**: To design a unified HOI imitation learning framework that does not require skill-specific parameter tuning, whilst being capable of learning multiple basketball interaction skills and combining them to complete complex tasks.

**Key Insight**: To introduce a Contact Graph to explicitly model the contact relationships in the interaction, making the contact information a core component of the imitation reward. Concurrently, a multiplicative rather than additive combination of sub-rewards is used to prevent unbalanced learning.

**Core Idea**: Explicitly constrain contact patterns using a contact graph reward + combine multidimensional rewards multiplicatively to achieve unified HOI imitation learning.

## Method

### Overall Architecture

The system consists of three components: (1) HOI Data Collection—including the vision-based BallPlay-V (8 basic skills) and the mocap-based BallPlay-M (35 minutes of diverse basketball interactions, recorded at 120fps); (2) Interaction Skill (IS) Policy Training—imitating the HOI data through RL, where the input is the HOI state + skill label (one-hot), and the output is target joint angles driven by a PD controller; (3) High-Level Controller (HLC)—taking the current state + task observations (e.g., hoop position) as input, and outputting skill selection labels to drive the frozen IS policy to complete complex tasks.

### Key Designs

1. **Contact Graph (CG) Reward**:

    - **Function**: Explicitly models contact relationships during interactions, ensuring the humanoid contacts objects with the correct body parts.
    - **Mechanism**: Defines objects/body parts in the interaction scene as graph nodes (e.g., both hands, non-hand body parts, the basketball), where edges represent the contact status (0/1) between two nodes. For each frame, the edge set $\mathcal{E}$ of the contact graph is calculated, and the matching degree to the reference contact pattern is measured by $r_t^{cg} = \exp(-\sum_j \lambda^{cg}[j] \cdot |s_t^{cg}[j] - \hat{s}_t^{cg}[j]|)$.
    - **Design Motivation**: Without the contact graph reward, the humanoid often falls into kinematic local optima, such as heading the ball, touching the ball with the wrist, or failing to grasp objects. The contact graph explicitly penalizes incorrect contact patterns. Ablation studies show that adding the CG reward boosts accuracy from 7.5% to 82.4%.

2. **Multiplicative Unified HOI Imitation Reward**:

    - **Function**: Integrates multi-dimensional imitation signals into a single reward, preventing learning failures caused by unbalanced rewards.
    - **Mechanism**: The total reward is $r_t = r_t^b \times r_t^o \times r_t^{rel} \times r_t^{reg} \times r_t^{cg}$, corresponding to body kinematics, object kinematics, relative motion, velocity regularization, and the contact graph, respectively. Each sub-reward is formulated as $\exp(-\lambda \cdot \text{MSE})$, and the multiplicative combination ensures that a mismatch in any single dimension will drastically reduce the overall reward.
    - **Design Motivation**: Additive combinations allow high scores in a single dimension (e.g., body movement) to mask low scores in other dimensions (e.g., contact), leading to unbalanced learning. Ablations show an accuracy of 95.4% for multiplicative vs 38.6% for additive combination.

3. **Hierarchical Controller (HLC) for Skill Composition**:

    - **Function**: Trains a high-level policy on top of the learned interaction skills to accomplish complex long-horizon tasks such as continuous scoring.
    - **Mechanism**: The IS policy is frozen. The HLC takes the current HOI state and task-specific observations (e.g., hoop position) as input, and outputs a discrete skill embedding vector to select which skill to execute. The HLC is trained using task-specific rewards (e.g., distance to the hoop, throwing height).
    - **Design Motivation**: Decouples skill acquisition and task planning; the IS policy is responsible for "how to execute a skill", while the HLC focuses on "which skill to choose".

### Loss & Training

The system is trained using the PPO algorithm. The IS policy is a 3-layer MLP [1024, 512, 512], outputting a Gaussian distribution (with fixed variance). The humanoid model has 52-53 joints and 156 degrees of freedom (including 30x3 DOF for the hands). Training starts with random initialization from reference clips and is optimized using the unified HOI imitation reward. It supports mixture training of multiple skills (the same policy concurrently learning dribbling, layups, etc.), differentiated by one-hot skill labels.

## Key Experimental Results

### Main Results

Skill learning success rate comparison (BallPlay-M):

| Method | Picking Up | Forward Dribbling | Layup | Shooting |
|------|------|---------|------|------|
| DeepMimic* | 19.6% | 68.5% | 98.9% | 97.8% |
| AMP* | 0.0% | 13.6% | 0.0% | 1.6% |
| **SkillMimic** | **86.7%** | **79.6%** | **99.1%** | **97.9%** |

High-level task success rate comparison:

| Method | Dribbling Forward | Dribbling in Circles | Throwing | Scoring |
|------|---------|---------|------|------|
| PPO (From Scratch) | 0.70% | 11.14% | 0.00% | 0.00% |
| ASE* (With Interaction Prior) | 0.31% | 7.21% | 0.00% | 0.00% |
| **SkillMimic + HLC** | **93.04%** | **79.92%** | **93.40%** | **80.25%** |

### Ablation Study

| Configuration | Accuracy | Contact Error $E_{cg}$ | Description |
|------|-------|------------------|------|
| Full Model | 82.4% | 0.087 | — |
| W/o Contact Graph Reward | 7.5% | 0.306 | Completely incorrect contacts |
| Additive Reward | 38.6% | — | Unbalanced learning |
| Multiplicative Combination | 95.4% | — | GRAB Dataset |

### Key Findings
- **The Contact Graph reward is the most critical innovation**: Without it, the accuracy drops to 7.5%, as the humanoid contacts objects using incorrect parts such as the head or wrists.
- **Performance scales with data volume**: The success rate of the picking-up skill scales from 0.5% with 1 clip to 85.6% with 131 clips, demonstrating the scalability of the data-driven approach.
- **Mixture training enhances individual skills**: Training 4 skills simultaneously yields better performance than training a single skill in isolation (e.g., left dribble: 4.1% → 67.9%), and supports zero-shot skill switching.
- **Robustness to physical properties**: Success rates remain stable under perturbations of 0.5-1.5x of the ball's radius and 0.1-6x of its density.

## Highlights & Insights

- **Simplicity and Generality of the Contact Graph**: In the basketball scenario, only 3 nodes (hands, body, ball) are needed to model the contact patterns of all skills. This abstraction is simple yet highly effective, and is transferable to other HOI scenarios (e.g., kitchen manipulation, tool use).
- **Insights into the Multiplicative Reward**: Viewed from an information-theoretic perspective, multiplication corresponds to addition in the log space, imposing a stricter "veto power" on each dimension. Any dimension close to 0 will drag down the overall reward, thereby preventing "false success".
- **Data-Driven Scalability**: It eliminates the need to customize reward designs for new skills—simply expanding the data allows new skills to be learned, unlocking scalability in human-object interaction animation.
- **Decoupled Hierarchical Architecture**: The IS policy and HLC are trained hierarchically; the IS policy handles "how to execute a skill", while the HLC focuses on "which skill to choose".

## Limitations & Future Work

- **Limited to Basketball Scenarios**: While the contact graph is a general design, experiments were validated only on basketball tasks. More complex multi-object interactions (such as cooking or assembly) would require more nodes and edges.
- **Single-Object Constraint**: Currently only handles interaction with a single ball; simultaneous manipulation of multiple objects remains a greater challenge.
- **High Data Collection Costs**: BallPlay-M requires optical motion capture + inertial sensors, which restricts the further scaling of the dataset.
- **No Sim-to-Real Validation**: All experiments were conducted in the Isaac Gym simulator, and transfer to real-world robots would require addressing the sim-to-real domain gap.
- **HLC Still Requires Task Rewards**: The high-level controller still needs a hand-designed reward function for each specific task (e.g., scoring, dribbling).

## Related Work & Insights

- **vs DeepMimic**: DeepMimic's kinematic imitation reward fails in HOI settings—achieving only a 19.6% success rate in picking up the ball—because it does not model contact. SkillMimic's contact graph reward addresses this gap.
- **vs AMP/ASE**: Adversarial Motion Priors (AMP) perform poorly in HOI (0-13.6%), indicating that GAN-style reward signals lack the fine-grained resolution needed to guide precise contact learning.
- **vs Interaction Graph Methods**: Prior InteractionGraph approaches only consider kinematic relationships (distance/velocity) without considering physical contacts, leading to unstable learning.

## Rating
- Novelty: ⭐⭐⭐⭐ The contact graph and multiplicative reward combination are simple yet effective innovations, achieving unified multi-skill HOI imitation for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets, multiple skills, comprehensive ablations and comparisons, and physical property robustness testing.
- Writing Quality: ⭐⭐⭐⭐ Clear illustrations, complete structure, though many symbols require cross-referencing.
- Value: ⭐⭐⭐⭐ Provides a simple and unified baseline for HOI imitation learning, with data-driven scalability as a key advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](../../NeurIPS2025/reinforcement_learning/learning_from_demonstrations_via_capability-aware_goal_sampling.md)
- [\[ICML 2025\] Learning Utilities from Demonstrations in Markov Decision Processes](../../ICML2025/reinforcement_learning/learning_utilities_from_demonstrations_in_markov_decision_processes.md)
- [\[ICML 2025\] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](../../ICML2025/reinforcement_learning/leveraging_skills_from_unlabeled_prior_data_for_efficient_online_exploration.md)
- [\[ICML 2026\] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](../../ICML2026/reinforcement_learning/interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)
- [\[ICLR 2026\] From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones](../../ICLR2026/reinforcement_learning/from_fx_and_gx_to_fgx_llms_learn_new_skills_in_rl_by_composing_old_ones.md)

</div>

<!-- RELATED:END -->

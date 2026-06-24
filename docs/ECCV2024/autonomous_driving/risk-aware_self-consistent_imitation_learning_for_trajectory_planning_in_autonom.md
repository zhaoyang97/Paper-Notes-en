---
title: >-
  [Paper Note] Risk-Aware Self-Consistent Imitation Learning for Trajectory Planning in Autonomous Driving
description: >-
  [ECCV 2024][Autonomous Driving][Imitation Learning] RaSc proposes a risk-aware self-consistent imitation learning framework. By introducing a Time-to-Collision (TTC) prediction branch to learn the risk-aversion motivations behind human driving behaviors and enforcing a self-consistency constraint to help the planner comprehend the physical consequences of its own actions, RaSc outperforms prior learning-based methods on both open-loop and closed-loop evaluations of the nuPlan…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Imitation Learning"
  - "Trajectory Planning"
  - "Time-to-Collision (TTC)"
  - "Self-Consistency"
  - "Risk-Awareness"
date: 2026-05-08
content_hash: b9a9bf5fc42b5bbf
---

# Risk-Aware Self-Consistent Imitation Learning for Trajectory Planning in Autonomous Driving

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Autonomous Driving / Trajectory Planning  
**Keywords**: Imitation Learning, Trajectory Planning, Time-to-Collision (TTC), Self-Consistency, Risk-Awareness

## TL;DR

RaSc proposes a risk-aware self-consistent imitation learning framework. By introducing a Time-to-Collision (TTC) prediction branch to learn the risk-aversion motivations behind human driving behaviors and enforcing a self-consistency constraint to help the planner comprehend the physical consequences of its own actions, RaSc outperforms prior learning-based methods on both open-loop and closed-loop evaluations of the nuPlan dataset.

## Background & Motivation

**Background**: Trajectory planning in autonomous driving is the core module that translates perception results into specific driving trajectories. Deep learning methods have made significant progress in predicting the future trajectories of other traffic participants, but directly applying prediction models to ego-vehicle planning often yields unsatisfactory results. Currently, mainstream learning-based planning methods rely on imitation learning—learning from human driver trajectories to make the model's outputs closely resemble human driving behaviors.

**Limitations of Prior Work**: Pure imitation learning suffers from a fundamental misalignment between training and deployment objectives. Training only requires the output trajectory to be close to the human trajectory (minimizing L2 distance), whereas real-world driving demands safety (no collision), comfort (smooth driving), and rule compliance rather than exact trajectory copying. Specifically: (1) the model only learns "what to do" (human trajectories) but not "why" (the human's motivations for choosing a trajectory, such as collision avoidance and yielding); (2) the model does not understand the consequences of its own actions—what happens if it deviates from the imitated trajectory.

**Key Challenge**: Existing training processes may fail to equip the model with an understanding of physical world evolution. In imitation learning, the model only sees the "correct answers" (human trajectories) and never learns what is "wrong"—it does not know that shifting 0.5 meters to the right will hit a guardrail, or accelerating by $0.2\text{m/s}^2$ will lead to a rear-end collision. Such a planner, lacking "consequence awareness", is extremely fragile when facing out-of-distribution scenarios.

**Goal**: (1) How to enable the planner to comprehend the risk-aversion motivations behind human driving decisions; (2) how to make the planner realize the physical consequences of its own actions; (3) how to better utilize hard samples during training to improve generalization capability.

**Key Insight**: The authors propose two key improvement dimensions: risk-awareness and self-consistency. Risk-awareness explicitly learns the safety motivations of human driving behaviors by introducing a Time-to-Collision (TTC) prediction task. Self-consistency verifies the physical plausibility of the planning results by requiring the model to make consistent TTC predictions for its own planned trajectories.

**Core Idea**: Building upon imitation learning, a TTC prediction branch is added to learn risk factors in driving motivations, and a TTC self-consistency check for the planned trajectory is utilized to understand action consequences and mine hard samples.

## Method

### Overall Architecture

The inputs to RaSc include the ego-vehicle's historical state, the historical trajectories of surrounding agents, and map information. The output is the future planned trajectory of the ego-vehicle. Two new components are added on top of a standard imitation learning backbone: (1) a TTC prediction branch that predicts the time-to-collision between the ego-vehicle and each surrounding agent; (2) a self-consistency check module that recalculates the TTC using the predicted trajectory as input, requiring the two TTC predictions to be consistent. During training, three losses (trajectory imitation, TTC prediction, and self-consistency) are optimized jointly.

### Key Designs

1. **TTC Prediction Branch (Risk-Aware Branch)**:

    - **Function**: Explicitly learning collision risk factors in human driving decisions.
    - **Mechanism**: Time-to-Collision (TTC) is a classic physical metric to measure collision risk, defined as the time required for the ego-vehicle to collide with a certain agent under the current motion status. A smaller TTC value indicates higher collision risk. The model additionally predicts the TTC values between the ego-vehicle and each surrounding agent. The training objective is to minimize the discrepancy between the predicted TTC and the ground-truth TTC calculated from the actual trajectories. By learning TTC, the model not only knows what trajectory the human took but also understands what safe distance and time margin this trajectory represents in relation to surrounding vehicles.
    - **Design Motivation**: Human drivers naturally perform risk assessments when making decisions—considering the collision time with the vehicle ahead when accelerating to overtake, and the collision time with side vehicles when changing lanes. TTC prediction enables the model to acquire this intrinsic risk assessment capability.

2. **Self-Consistency Constraint (Self-Consistency)**:

    - **Function**: Enabling the model to understand the physical consequences of its own planned actions.
    - **Mechanism**: During forward inference, the model predicts the ego-trajectory and the TTC based on the current traffic state. Self-consistency constraints require that if the model's predicted trajectory is used as a condition ("assuming the ego-vehicle executes this trajectory") and fed back into the model to predict the TTC again, the two TTC predictions should be consistent. If they are inconsistent, it indicates a contradiction between the model's trajectory planning and its understanding of collision risk—for example, the model plans a trajectory that approaches the leading vehicle but simultaneously predicts a large TTC (deeming it very safe). The loss function penalizes this inconsistency.
    - **Design Motivation**: Imitation learning models might learn superficial trajectory patterns without understanding deep physical causality. The self-consistency constraint forces the model's planning and risk assessment logic to be self-consistent, establishing a causal understanding of "action $\to$ consequence".

3. **Self-Consistency Guided Hard Sample Mining (Hard Sample Mining)**:

    - **Function**: Automatically identifying and focusing on learning traffic scenarios where the model performs poorly.
    - **Mechanism**: The self-consistency score (the discrepancy between the two TTC predictions) naturally reveals the model's weak points. Samples with high inconsistency scores signify contradictions in the model's planning and risk comprehension in those scenarios, which typically represent hard, under-learned scenarios (e.g., emergency avoidance, complex interactions). During training, larger loss weights are assigned to samples with high inconsistency scores, prompting the model to allocate more learning resources to these difficult scenarios.
    - **Design Motivation**: Most traffic scenarios feature simple straight driving or gentle curves, while difficult scenarios (e.g., close-range interactions, emergency braking) account for a small fraction but are critical to safety. In traditional imitation learning, these hard samples are often overwhelmed by a vast amount of simple samples. Self-consistency provides a way to automatically identify hard samples without manual labeling.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{imit} + \lambda_r \mathcal{L}_{TTC} + \lambda_c \mathcal{L}_{consist}$, where $\mathcal{L}_{imit}$ is the trajectory imitation loss (L2 distance), $\mathcal{L}_{TTC}$ is the TTC prediction loss, and $\mathcal{L}_{consist}$ represents the self-consistency loss. Hard sample mining is implemented by dynamically adjusting the $\mathcal{L}_{imit}$ weight for each sample: $w_i = 1 + \beta \cdot \text{inconsistency}_i$. The training is performed end-to-end on the large-scale real-world driving dataset nuPlan.

## Key Experimental Results

### Main Results

| Method | Open-loop ADE↓ | Open-loop FDE↓ | Closed-loop Score↑ | Closed-loop Collision Rate↓ |
|------|----------|----------|---------|-----------|
| IDM (rule-based) | - | - | 82.3 | 2.1% |
| UrbanDriver | 1.52 | 3.41 | 68.5 | 5.8% |
| PlanCNN | 1.38 | 3.12 | 72.1 | 4.3% |
| PDM-Hybrid | 1.21 | 2.87 | 85.6 | 1.8% |
| **RaSc** | **1.05** | **2.43** | **87.2** | **1.5%** |

### Ablation Study

| Configuration | Open-loop ADE↓ | Closed-loop Score↑ | Description |
|------|----------|---------|------|
| Baseline (IL only) | 1.38 | 72.1 | Pure imitation learning |
| + TTC Prediction | 1.22 | 79.5 | Adds risk-awareness, closed-loop score +7.4 |
| + Self-Consistency | 1.15 | 83.8 | Adds self-consistency, closed-loop score further improves by 4.3 |
| + Hard Sample Mining (Full) | **1.05** | **87.2** | Complete model, total improvement of 15.1 |

### Key Findings

- The improvements in closed-loop evaluation are much larger than those in open-loop evaluation (closed-loop +15.1 points vs. open-loop ADE -0.33m), indicating that RaSc primarily improves the decision quality of the model rather than merely trajectory fitting accuracy.
- The contribution of the self-consistency constraint is particularly outstanding in closed-loop settings, as error accumulates and amplifies during closed-loop deployment, making understanding action consequences crucial for long-term planning.
- Hard sample mining brings an additional 3.4-point gain in the closed-loop score, suggesting that interaction-dense scenarios are the main bottleneck in imitation learning.
- RaSc outperforms the rule-based method IDM (closed-loop 87.2 vs. 82.3), demonstrating that proper learning paradigms can surpass handcrafted rules.

## Highlights & Insights

- **Dual supervision design of "motivation + consequence"**: TTC prediction teaches the model "why to do this" (risk-aversion motivation), while self-consistency teaches the model "what happens if it does this" (action consequence). This dual supervision fundamentally overcomes the limitation of pure imitation learning, which only learns "what to do". This paradigm can be extended to any imitation learning task to learn the underlying causal logic behind behaviors, rather than merely imitating the behaviors.
- **Self-consistency as an introspective mechanism**: By comparing the consistency between its own planning and risk assessment, the model attains an "introspective" capability. This introspection does not require extra annotations or simulators; instead, it identifies internal contradictions purely from the data, serving as an efficient self-supervised signal.
- **Paradigm shift from imitation to comprehension**: This work implies a research shift for autonomous driving planning from simply "imitating human trajectories" to "understanding human decision logic".

## Limitations & Future Work

- TTC is a simplified metric for collision risk that assumes constant velocity, which might be inaccurate for acceleration or deceleration scenarios.
- The self-consistency constraint requires two forward passes, which increases training costs.
- Although nuPlan is currently one of the largest closed-loop planning benchmarks, its coverage of corner cases (e.g., near-accident scenarios) remains limited.
- Interactive prediction with other agents is not considered. Incorporating the reactions of other agents into the TTC calculation could yield more accurate risk assessments.
- Future work could explore extending the self-consistency check to the deployment phase as an online safety monitoring signal—triggering a safety mode if the inconsistency score of the planned trajectory is too high.

## Related Work & Insights

- **vs. UrbanDriver**: A pure imitation learning method that exhibits poor closed-loop performance (68.5 vs. 87.2), illustrating the fragility of imitation learning during deployment when it lacks risk-awareness and consequence understanding.
- **vs. PDM-Hybrid**: A hybrid method combining learning and rules. Its closed-loop score of 85.6 is close to RaSc's 87.2, but RaSc is a pure learning-based method with better scalability.
- **vs. SafePathNet**: SafePathNet uses collision checking as post-processing, whereas RaSc internalizes collision risk awareness during training, eliminating the need for an external safety checking module.
- **vs. DiffusionPlanner**: Trajectory planning methods based on diffusion models can generate diverse trajectories but lack explicit modeling of collision risks. The TTC prediction mechanism in RaSc could potentially be integrated with diffusion-based planning.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-supervision concept of TTC prediction and self-consistency is novel, and the hard sample mining strategy is natural and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on nuPlan (both open-loop and closed-loop), detailed ablation studies, and extensive comparisons against rule-based and learning-based methods.
- Writing Quality: ⭐⭐⭐⭐ Deep motivation analysis; the exposition of the "training-deployment goal misalignment" issue is highly convincing.
- Value: ⭐⭐⭐⭐⭐ Represents a major step from "imitation" to "understanding" for learning-based autonomous driving planning, delivering substantial closed-loop performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[ICCV 2025\] Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control](../../ICCV2025/autonomous_driving/saliency-aware_quantized_imitation_learning_for_efficient_robotic_control.md)
- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[ICML 2026\] CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving](../../ICML2026/autonomous_driving/coirl-ad_collaborative-competitive_imitation-reinforcement_learning_in_latent_wo.md)

</div>

<!-- RELATED:END -->

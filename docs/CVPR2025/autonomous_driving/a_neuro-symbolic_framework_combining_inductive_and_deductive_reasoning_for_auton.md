---
title: >-
  [Paper Note] A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning
description: >-
  [CVPR 2025][Autonomous Driving][Neuro-Symbolic Systems] This paper proposes the first neuro-symbolic framework that directly embeds ASP symbolic reasoning decisions as learnable embeddings into the trajectory decoding of an end-to-end planner. It dynamically extracts scene rules using LLMs, performs logical arbitration via the Clingo solver, generates physically feasible trajectories via a differentiable KBM, and refines them with neural residuals. On nuScenes…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Neuro-Symbolic Systems"
  - "LLM Reasoning"
  - "ASP Logic Programming"
  - "Kinematic Bicycle Model"
  - "Interpretable Planning"
date: 2026-05-08
content_hash: a58e88ee6bc78420
---

# A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning

**Conference**: CVPR 2025  
**arXiv**: [2603.12421](https://arxiv.org/abs/2603.12421)  
**Code**: None  
**Area**: Autonomous Driving / Neuro-Symbolic Systems  
**Keywords**: Neuro-Symbolic Systems, LLM Reasoning, ASP Logic Programming, Kinematic Bicycle Model, Interpretable Planning

## TL;DR
This paper proposes the first neuro-symbolic framework that directly embeds ASP symbolic reasoning decisions as learnable embeddings into the trajectory decoding of an end-to-end planner. It dynamically extracts scene rules using LLMs, performs logical arbitration via the Clingo solver, generates physically feasible trajectories via a differentiable KBM, and refines them with neural residuals. On nuScenes, it comprehensively outperforms MomAD with an L₂ error of 0.57m, a collision rate of 0.075%, and a TPC of 0.47m.

## Background & Motivation
**Background**: End-to-end autonomous driving (UniAD, VAD, SparseDrive, MomAD) unifies perception, prediction, and planning, achieving inductive reasoning through large-scale data training, with performance continuously improving.

**Limitations of Prior Work**: (1) Pure "black-box" neural networks lack common-sense reasoning capabilities and explicit physical/logical constraints; (2) they are prone to generating common-sense-violating or even dangerous trajectories in out-of-distribution (OOD) long-tail scenarios; (3) internal decision-making processes are untraceable, making root-cause analysis difficult when planning fails.

**Key Challenge**: Methods that directly regress coordinate displacements lack physical semantics (acceleration intent/steering motivation); post-processing collision re-scoring/data augmentation cannot alter the purely inductive nature of the planner. Meanwhile, pure symbolic methods are limited by the capacity for manual rule enumeration and cannot handle complex semantics.

**Goal**: (a) How to meaningfully apply discrete decisions from deductive reasoning to the continuous trajectory decoding of neural planners? (b) How to ensure that trajectory representations possess sufficient physical semantics to allow the target speed to directly constrain the velocity profile?

**Key Insight**: Human driving relies not only on intuition (induction) but also on traffic rules and common sense as logical foundations (deduction)—inductive reasoning and deductive reasoning are complementary.

**Core Idea**: LLM inductive rule extraction + ASP deductive logic arbitration + differentiable KBM physical trajectories + neural residual correction, offering full-link traceability.

## Method

### Overall Architecture
Three core modules run in parallel: (1) Perception fact extraction (symmetric sparse perception network) $\to$ structured scene representation; (2) dynamic deductive reasoning engine (LLM rule extraction $\to$ Clingo logical arbitration $\to$ discrete decision); (3) decision-conditioned physical residual planner (decision embedding $\to$ KBM physical trajectory + neural residual $\to$ final trajectory).

### Key Designs

1. **Dynamic Rule Extraction via LLM (Inductive Reasoning)**

    - **Function**: Inputs the structured perception outputs (in ASP predicate format) into the LLM to dynamically generate 3-6 scene-specific ASP logic rules.
    - **Mechanism**: The System Prompt defines the LLM's role as an "expert in autonomous driving rules and ASP programming", outputting `suggestion(Action, TargetSpeed, Type)`, containing 9 Actions, 6 TargetSpeed levels, and 3 Navigation types.
    - **Design Motivation**: Manual rule bases cannot cover long-tail scenarios, whereas LLMs can generate customized rules tailored to specific scene contexts; strictly constraining LLM outputs to a 9×6×3 discrete space ensures parseability.

2. **ASP Logical Arbitration (Deductive Reasoning)**

    - **Function**: Inputs the LLM rules and predefined safety axioms into the Clingo solver to output a unique deterministic decision `final_decision(Action, Speed)`.
    - **Mechanism**: A five-level priority mapping: Emergency Avoidance > Safety Precaution > Legal Compliance > Efficiency Goal, formally encoding the principle that "safety strictly prioritizes efficiency".
    - **Design Motivation**: It does not rely on hyperparameters requiring network optimization, serving as a rigid, formal constraint hierarchy to guarantee decision uniqueness and safety.

3. **Dual-Path Decision Embedding**

    - **Function**: Maps discrete Action/Speed into high-dimensional feature vectors to condition trajectory decoding via two paths.
    - **Path 1 (Semantic Offset)**: Decision features are injected as semantic offsets into the planning query $\mathbf{Q}'_\text{plan} = \mathbf{Q}_\text{plan} \oplus \mathbf{d}$, constraining the trajectory space shape.
    - **Path 2 (Velocity Bias)**: Learns a velocity bias parameter to directly correct the initial KBM velocity $v'_0 = v_0 + b_v$, forcing the network to match the target speed of the logical decision.

4. **Differentiable Kinematic Bicycle Model (KBM) + Neural Residuals**

    - **Function**: Uses KBM to generate physical baseline trajectories that satisfy kinematic constraints, and uses neural residuals to compensate for complex interactions.
    - **Mechanism**: $\dot{x}=v\cos\psi, \dot{y}=v\sin\psi, \dot{v}=a, \dot{\psi}=\frac{v\tan\delta}{L}$, integrated via second-order Runge-Kutta 2 (RK2), yielding the final trajectory $\boldsymbol{\tau}_\text{final} = \boldsymbol{\tau}_\text{physics} + \lambda \cdot \tanh(\boldsymbol{\tau}_\text{residual})$.
    - **Design Motivation**: $\tanh$ limits the residual amplitude to prevent violations of physical priors; the anisotropic residual loss applies a 10 times penalty to lateral residuals to suppress physically inconsistent "sideslip".

### Loss & Training
Two-stage training: Stage 1 trains the KBM physical planner to establish physical priors; Stage 2 introduces ASP deductive conditioning to fine-tune for 10 epochs. It utilizes an anisotropic residual loss + control smoothing loss + auxiliary action classification loss. ASP decisions are pre-computed offline to accelerate training.

## Key Experimental Results

### Main Results (nuScenes validation)

| Method | L₂ Avg (m)↓ | Collision Rate (%)↓ | TPC Avg (m)↓ |
|------|------------|-----------|-------------|
| UniAD | 0.73 | 0.61 | 0.68 |
| SparseDrive | 0.61 | 0.08 | 0.57 |
| MomAD | 0.60 | 0.09 | 0.54 |
| **Ours** | **0.57** | **0.075** | **0.47** |

### Ablation Study

| Configuration | L₂ Avg | Collision Rate | TPC Avg |
|------|--------|--------|---------|
| MomAD baseline | 0.60 | 0.090 | 0.54 |
| + KBM only | 0.58 | 0.110 | 0.47 |
| + KBM + ASP (full) | **0.57** | **0.075** | **0.47** |

### Key Findings
- **KBM Only**: Identifies significant improvements in L₂ and TPC (physical constraints improve spatial accuracy and temporal consistency), but the collision rate increases instead (0.09 $\to$ 0.11). This proves that pure physical priors are insufficient in high-risk scenarios and require a logical reasoning path.
- **ASP Conditioning**: Decreases the collision rate by 31.8% (0.110 $\to$ 0.075) on top of KBM, validating that deductive reasoning effectively constrains unsafe behavior.
- Improvements are more prominent in short time horizons (1s: 12.9%, 2s: 5.3%) compared to long time horizons (3s: 1.1%), aligning with the semantic scope of ASP inferring immediate action intentions.
- The complete reasoning chain is fully traceable, starting from perception facts $\to$ logic rules $\to$ deductive decisions $\to$ physical control $\to$ executed trajectories.
- Case Study: In a pedestrian jaywalking scenario, a TTC of 890ms triggers a safety axiom $\to$ (yield, zero) decision $\to$ negative velocity bias ($\sim$-2m/s) $\to$ braking trajectory. While MomAD can also decelerate, it cannot explain "why".

## Highlights & Insights
- **First to directly embed ASP decisions into the trajectory decoding of end-to-end planners**: Different from post-processing filtering, the decisions directly participate in the trajectory generation process, presenting a fundamental design difference.
- **KBM provides a physical interface for logical decisions**: The target velocity directly constrains the KBM integration via velocity bias, where physical semantics guarantee consistency across logic, control, and trajectory.
- **LLM acts as a rule extractor rather than a decision-maker**: This avoids the uncontrollable issue of LLM logical inconsistency, utilizing a formal ASP solver to ensure reasoning rigor.
- **Asynchronous dual-rate mechanism**: The LLM extracts macro-rules at a low frequency ($\sim$1s/frame), while Clingo performs logical arbitration frame-by-frame at a high frequency ($<$5ms/frame), bridging the gap between reasoning latency and real-time control requirements.

## Limitations & Future Work
- Based on the nuScenes open-loop evaluation protocol, it may underestimate closed-loop collision risks.
- LLM rule extraction takes $\sim$1s/frame; actual deployment requires model acceleration or distillation into lightweight networks.
- Limited improvement in the 3s long time horizon (only 1.1%), showing that long-range interaction still primarily relies on the inductive generalization of neural networks.
- It has not been validated in closed-loop simulation environments like CARLA, nor has it built a dedicated dataset for long-tail scenarios.

## Related Work & Insights
- **vs MomAD**: MomAD introduces trajectory momentum to improve temporal consistency but lacks logical guarantees; Ours utilizes ASP logical arbitration + KBM physical constraints to provide dual guarantees.
- **vs DriveVLM**: DriveVLM employs VLMs for chain reasoning but cannot guarantee logical consistency; Ours limits the LLM to a rule extractor, with logical consistency guaranteed by Clingo.
- **vs SparseDrive**: SparseDrive relies on post-processing with heuristic collision re-scoring; Ours directly embeds safety constraints into the decoding process.
- **vs DeepProbLog**: DeepProbLog embeds logical rules into neural network training but has not been validated in real-time planning scenarios driven by complex 3D perception.
- **vs Sharifi et al.**: Combined symbolic logic programming with deep reinforcement learning for highway safety constraints, but was limited to structured environments; Ours targets complex urban traffic.
- **Significance for interpretable autonomous driving**: The reasoning chain provided in this paper is not just for technical novelty, but is also a core requirement for safety certification—allowing precise backtracking to triggered rules and facts when planning fails.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to directly embed ASP-LLM deductive reasoning into end-to-end planner decoding; the concept is novel and highly scalable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively benchmarked, with ablations and case studies on nuScenes, but lacks closed-loop validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain of the paper is clear, delivering a cohesive flow from motivation to method and experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a paradigmatic solution for interpretable safety planning in autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving](t2sg_traffic_topology_scene_graph_for_topology_reasoning_in_autonomous_driving.md)
- [\[NeurIPS 2025\] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning](../../NeurIPS2025/autonomous_driving/unimotion_a_unified_motion_framework_for_simulation_prediction_and_planning.md)
- [\[CVPR 2026\] ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving](../../CVPR2026/autonomous_driving/colavla_leveraging_cognitive_latent_reasoning_for_hierarchical_parallel_trajecto.md)
- [\[CVPR 2025\] SparseAlign: A Fully Sparse Framework for Cooperative Object Detection](sparsealign_a_fully_sparse_framework_for_cooperative_object_detection.md)
- [\[ECCV 2024\] Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving](../../ECCV2024/autonomous_driving/reason2drive_towards_interpretable_and_chainbased_reasoning.md)

</div>

<!-- RELATED:END -->

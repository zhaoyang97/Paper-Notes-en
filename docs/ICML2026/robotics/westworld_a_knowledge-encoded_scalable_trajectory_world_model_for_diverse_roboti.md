---
title: >-
  [Paper Note] WestWorld: Scalable Trajectory World Models with Knowledge Encoding
description: >-
  [ICML 2026][Robotics][World Models] WestWorld enables scalable prediction across diverse robotic systems with a single model by explicitly encoding general robot dynamics knowledge into a trajectory world model—achieving…
tags:
  - "ICML 2026"
  - "Robotics"
  - "World Models"
  - "Knowledge Encoding"
  - "Trajectory Prediction"
  - "Robot Diversity"
  - "Cross-Embodiment"
date: 2026-05-08
content_hash: b564090946ba7d03
---

# WestWorld: Scalable Trajectory World Models with Knowledge Encoding

**Conference**: ICML 2026  
**arXiv**: [2603.14392](https://arxiv.org/abs/2603.14392)  
**Code**: https://github.com/511205787/WestWorld  
**Area**: Robotics / Embodied AI / World Models  
**Keywords**: World Models, Knowledge Encoding, Trajectory Prediction, Robot Diversity, Cross-Embodiment

## TL;DR
WestWorld enables scalable prediction across diverse robotic systems with a single model by explicitly encoding general robot dynamics knowledge into a trajectory world model—achieving an average 14.3% improvement over specialized models across 7 robot morphologies and supporting zero-shot transfer to novel robot configurations.

## Background & Motivation

**Background**: World models, as fundamental components of robot learning, are primarily divided into pixel-level generative models (e.g., DreamerV3) and state-level dynamics models (e.g., PlaNet). Both face challenges in generalizing from a single robot to diverse robotic systems.

**Limitations of Prior Work**: (1) Existing world models are mostly robot-specific—requiring training from scratch for each new morphology; (2) The trade-off between model capacity and data diversity is difficult to balance during cross-morphology generalization; (3) Existing methods do not utilize robot physics knowledge, resulting in low efficiency for purely data-driven approaches; (4) Evaluations lack a unified scalability benchmark.

**Key Challenge**: The trend of robot diversification (arms, quadrupeds, humanoids, etc.) requires a unified world model, but the performance of existing architectures drops sharply when adding new robot morphologies.

**Goal**: Design a scalable trajectory world model that maintains high-quality predictions across diverse robots through explicit knowledge encoding.

**Key Insight**: Although robots vary in morphology, they share common laws of dynamics (Newtonian mechanics, joint kinematics, etc.). If these universal laws are encoded into the model, only a few parameters are needed for robot-specific adaptation.

**Core Idea**: Decompose the world model into a **General Knowledge Module** + **Robot-Specific Adapters**—the general module captures physical laws, while adapters learn the kinematics and morphological characteristics of specific robots.

## Method

### Overall Architecture
A two-layer architecture—(1) **General Knowledge Layer**: Encodes universal laws such as kinematic equations, dynamics constraints, and joint limits based on physical priors, with parameters shared across all robots; (2) **Robot Adaptation Layer**: Each robot is equipped with an independent lightweight adapter (kinematic parameters + morphological embeddings + shape parameters), interacting with the general layer via cross-attention.

### Key Designs

1. **Physics-Aware Knowledge Encoding**:

    - Function: Injects Newtonian mechanics and kinematic laws into the world model as inductive biases.
    - Mechanism: The general layer includes a **forward dynamics subnet** predicting $\dot{q} = f_{\text{dyn}}(q, \tau, M, K)$ (where $q$ is the joint state, $\tau$ is the control torque, $M$ is the mass matrix, and $K$ is stiffness), and a **kinematics constraint subnet** predicting the end-effector position $x = f_{\text{kin}}(q, L)$ ($L$ is link lengths); the model is constrained to follow physical laws via a physics consistency loss $\mathcal{L}_{\text{phys}} = \|M\ddot{q} + C\dot{q} + g - \tau\|^2$.
    - Design Motivation: Purely data-driven models require large amounts of data to learn physical laws; explicit encoding allows the model to converge quickly on small datasets and provides interpretable predictions.

2. **Robot Morphology Adapters**:

    - Function: Learns lightweight adapters for each robot morphology to capture specific kinematic and dynamic parameters.
    - Mechanism: Each robot has independent $\{M_i, L_i, K_i, e_i\}$ (mass matrix, link lengths, stiffness, morphological embedding), interacting with the general layer via cross-attention; morphological embeddings $e_i \in \mathbb{R}^{128}$ learn high-level morphological features (e.g., quadruped/biped/arm).
    - Design Motivation: Avoids training full models for every robot; adapter parameters are significantly smaller than the general layer (approx. 1%), supporting rapid adaptation to new robots.

3. **Multi-Robot Joint Training + Knowledge Sharing**:

    - Function: Enables the general layer to learn cross-morphology shared physical knowledge through multi-robot joint training.
    - Mechanism: Robots are randomly sampled in batches during training, with each batch containing multiple morphologies; the general layer learns universal physical laws via backward passes; adapters learn robot specifics via forward passes; the loss function is $\mathcal{L} = \sum_i \mathcal{L}_{\text{pred}}^i + \lambda_1 \mathcal{L}_{\text{phys}}^i + \lambda_2 \|\theta_{\text{adapter}}^i\|^2$.
    - Design Motivation: Joint training forces the general layer to learn features useful for all robots; adapter regularization prevents over-specialization.

## Key Experimental Results

### Main Results

| Robot | Specialized Model (PSNR) | DreamerV3 (PSNR) | **Ours (PSNR)** | Gain |
|--------|--------------|---------------|----------|------|
| Franka Panda (7-DoF arm) | 24.3 | 22.1 | **27.5** | +3.2 |
| Spot (Quadruped) | 22.8 | 20.5 | **26.1** | +3.3 |
| H1 (Humanoid) | 21.5 | 19.2 | **25.4** | +3.9 |
| UR5 (6-DoF arm) | 23.7 | 21.3 | **26.8** | +3.1 |
| ANYmal (Quadruped) | 22.2 | 20.0 | **25.6** | +3.4 |
| Atlas (Humanoid) | 20.9 | 18.5 | **24.8** | +3.9 |
| Pepper (Bimanual) | 22.4 | 20.2 | **25.9** | +3.5 |

Average gain is 14.3%, with a maximum of 3.9 PSNR (Humanoid robot).

### Zero-Shot Cross-Robot Generalization

| Unseen Robot | DreamerV3 | **Ours** | Gain |
|-----------|---------|----------|------|
| KUKA iiwa (New arm) | 12.3 | **22.5** | +83% |
| Cassie (New biped) | 13.7 | **23.1** | +69% |
| iCub (New humanoid) | 11.8 | **21.4** | +81% |

### Ablation Study

| Config | Avg PSNR | Description |
|------|----------|------|
| w/o Physics Encoding | 21.2 | Purely data-driven |
| w/o Robot Adapters | 19.5 | Single universal model |
| Physics + Adapters (Full) | **25.9** | Ours |
| Increased Adapter Scale (2×) | 26.0 | Diminishing returns |
| Reduced General Layer (50%) | 23.1 | General Knowledge Layer is key |

### Key Findings
- Physics knowledge encoding contributes 4.7 PSNR; robot adapters contribute 6.4 PSNR.
- Zero-shot cross-robot generalization is prominent—PSNR for new robots still reaches 21-23, far exceeding DreamerV3's 12-14.
- Performance is optimal when the scale of the general layer > adapter scale.

## Highlights & Insights
- **Elegant integration of physical knowledge and neural networks**: Explicitly encoding physical laws provides inductive biases while maintaining neural network flexibility.
- **Lightweight design of morphology adapters**: Achieves robot-specific adaptation with only 1% of parameters, ensuring strong scalability.
- **Outstanding zero-shot cross-robot generalization**: The physical laws learned by the general knowledge layer are truly transferable.
- **Multi-robot joint training paradigm**: Opens the direction for unified world models.

## Limitations & Future Work
- Physics knowledge encoding depends on manual design—more complex physics (fluids, soft bodies) requires expansion.
- Adapter scale grows linearly with the number of robots; ultra-large scales may face limitations.
- Current evaluation is limited to rigid-body robots; soft or flexible robots require further study.
- Improvements: Introduce learnable physics priors; explore more compact adapter architectures; extend to manipulation tasks rather than only motion prediction.

## Related Work & Insights
- **vs DreamerV3**: DreamerV3 is a robot-specific world model; Ours unifies multiple robots + physical priors.
- **vs PlaNet**: PlaNet learns state-level dynamics; Ours explicitly encodes physical laws.
- **vs RobotGPT**: RobotGPT is a generative model for a single robot; Ours provides a unified multi-robot architecture.
- **Insight**: The combination of physical priors and data-driven approaches is an effective paradigm for robot learning, scalable to manipulation and navigation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to explicitly encode physical priors into a multi-robot world model, with outstanding cross-morphology generalization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 training robots + 3 zero-shot test robots + detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and detailed method description.
- Value: ⭐⭐⭐⭐⭐ Provides a unified solution for the trend of robot diversification, with breakthrough significance for zero-shot cross-robot capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[ICLR 2026\] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models](../../ICLR2026/robotics/synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge.md)
- [\[ICML 2026\] Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](optimal_and_scalable_mapf_via_multi-marginal_optimal_transport_and_schrödinger_b.md)
- [\[ICLR 2026\] Building Spatial World Models from Sparse Transitional Episodic Memories](../../ICLR2026/robotics/building_spatial_world_models_from_sparse_transitional_episodic_memories.md)
- [\[ICLR 2026\] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](../../ICLR2026/robotics/test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] QUAR-VLA: Vision-Language-Action Model for Quadruped Robots
description: >-
  [ECCV 2024][Robotics][quadruped robot] This paper proposes the first vision-language-action (QUAR-VLA) paradigm for quadruped robots, constructing a multi-task dataset QUARD with 259K episodes and the QUART model based on a pretrained multimodal large model, achieving unified control of multi-tasks such as perception, navigation, and whole-body manipulation.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "quadruped robot"
  - "vision-language-action"
  - "imitation learning"
  - "sim-to-real"
  - "multi-task"
date: 2026-05-08
content_hash: 030887bd982987fc
---

# QUAR-VLA: Vision-Language-Action Model for Quadruped Robots

**Conference**: ECCV 2024  
**arXiv**: [2312.14457](https://arxiv.org/abs/2312.14457)  
**Code**: Not released yet  
**Area**: Multimodal VLM  
**Keywords**: quadruped robot, vision-language-action, imitation learning, sim-to-real, multi-task

## TL;DR
This paper proposes the first vision-language-action (QUAR-VLA) paradigm for quadruped robots, constructing a multi-task dataset QUARD with 259K episodes and the QUART model based on a pretrained multimodal large model, achieving unified control of multi-tasks such as perception, navigation, and whole-body manipulation.

## Background & Motivation
**Background**: Quadruped robot learning typically processes visual perception (QUAR-VA) and language interaction (QUAR-LA) separately. VA methods leverage first/third-person images to guide actions but use only coarse-grained target image instructions; LA methods use language to execute fine-grained tasks but lack visual perception and cannot perform autonomous navigation.

**Limitations of Prior Work**:
   - VA methods rely on a single coarse-grained image instruction, making it difficult to handle compositional tasks (e.g., "first... then...").
   - LA methods lack the visual modality, preventing them from perceiving environmental obstacles.
   - There is a lack of large-scale multi-task datasets for quadruped robots.

**Key Challenge**: Quadruped robots need to simultaneously understand visual scenes and language instructions to make autonomous decisions, but existing paradigms split the two; moreover, the flexible gaits and whole-body control of quadrupeds make the action space design more complex.

**Key Insight**: Define a 12-dimensional high-level command action space (velocity + gait + pose + termination signal), discretize continuous actions into 256 bins, and utilize a pretrained VLM to directly output action tokens.

**Core Idea**: Fine-tune a pretrained multimodal large model into a unified policy for quadruped robots, taking image + language instructions as inputs and outputting discretized action tokens.

## Method

### Overall Architecture
The QUAR-VLA system consists of three components: (1) the QUARD large-scale multi-task dataset (simulation 256K + real 3K episodes); (2) the QUART model based on a pretrained 8B VLM, which receives first-person images and language instructions to output 12-dimensional discretized actions; (3) a low-level command tracking controller that translates high-level commands into joint actions.

### Key Designs

#### 1. **Action Space Design**
    - **Function**: Define a high-level command space suitable for quadruped robots, balancing flexibility and computational efficiency.
    - **Mechanism**:
      - 12-dimensional action vector: $[v_x, v_y, \omega_z, \theta_1, \theta_2, \theta_3, f, h_z, \phi, s_y, h_z^f, t]$
      - Meanings: $v_x, v_y$ represent x/y-axis velocities, $\omega_z$ is the z-axis angular velocity, $\theta_{1,2,3}$ denote gait patterns, $f$ is frequency, $h_z$ is robot height, $\phi$ is pitch angle, $s_y$ is foot width, $h_z^f$ is foot swing height, and $t$ is the termination signal.
      - Continuous dimensions are discretized into 256 uniform bins.
    - **Design Motivation**: Avoid overly simplified 2D navigation velocity output (which lacks whole-body control) and avoid directly outputting joint motor controls (as the required frequency is too high). High-level commands are executed by a pretrained low-level policy, decoupling decision-making and execution.

#### 2. **QUARD Dataset (QUAdruped Robot Dataset)**
    - **Function**: Build the first large-scale quadruped robot dataset containing vision, language instructions, and robot commands.
    - **Mechanism**:
      - 7 tasks divided into 3 difficulty levels: Easy (letter recognition, 10K), Medium (navigation to objects, sim 72K + real 3K), Hard (tunnel crawling 48K / obstacle avoidance navigation 63K / pole climbing 1K / unloading objects 52K).
      - Simulation data are collected in parallel via Isaac Gym, converting target velocities using A*/D* path planning + PD controllers.
      - Real-world data are collected in a lab using teleoperation on a WR-2 quadruped robot.
      - Diversity design: Multiple colors (green/red/blue/yellow), indoor/outdoor objects (bookshelves/ovens/trash cans, etc.), three velocity levels, and multiple gaits.
    - **Consistency Constraints**: The simulation and real-world starting/target position ranges are consistent (target $x \in [2.7, 3.3]$m, $y \in [0.9, 1.1]$m).

#### 3. **QUART Model (QUAdruped Robotic Transformer)**
    - **Function**: Fine-tune a pretrained vision-language model into a unified control policy for quadruped robots.
    - **Mechanism**:
      - Based on an 8B pretrained VLM (fuyu-8b), integer tokens (0-255) are associated with action bins, which is a form of symbol tuning.
      - Input: Single first-person RGB image $s$ + language instruction $w$.
      - Tokenizer converts inputs into a token sequence: $\tau(t|s,w)$.
      - Decoder-only Transformer outputs discretized action tokens: $p(a_d|t)$.
      - Policy: $\text{QUART}(a_d|s,w) = p(a_d|t)\tau(t|s,w)$.
      - Action Detokenize: $a_c = \text{Detokenize}(a_d)$ converts discrete tokens back into continuous action values.
    - **Training**: Standard categorical cross-entropy + causal masking, lr=2e-5, batch=256, 100K steps.
    - Inference speed: 2Hz, meeting the frequency requirements of high-level command control.
    - **Design Motivation**: Leverage the pre-trained vision-language alignment capabilities and world knowledge of the VLM to directly output actions via symbol tuning, without requiring an additional action head.

### Loss & Training
- **Loss**: Standard categorical cross-entropy (behavioral cloning loss), performing next-token prediction on 12-dimensional action tokens.
- **Sim-to-Real Joint Training**: Co-training with a large amount of simulation data + a small amount of real-world data, where simulation data provides diversity and real-world data guarantees applicability.

## Key Experimental Results

### Main Results: Multi-task Success Rate

| Methods | Distinguish | Go to | Go avoid | Go through | Crawl | Unload |
|------|-----------|-------|---------|-----------|-------|--------|
| CLIP | 0.44 | 0.43 | 0.45 | 0.19 | 0 | 0 |
| R3M | 0.58 | 0 | 0 | 0 | 0 | 0 |
| VC-1 | 0.46 | 0.43 | 0.45 | 0.31 | 0 | 0 |
| **QUART** | **0.66** | **0.60** | **0.53** | **0.41** | **0.32** | **0.12** |

QUART outperforms the baselines on all tasks, being the only method to achieve success on highly difficult tasks (Crawl/Unload).

### Generalization

| Methods | Unseen Object | Unseen Verbal |
|------|-------------|---------------|
| CLIP | 0.11 | 0.14 |
| R3M | 0 | 0 |
| VC-1 | 0.29 | 0.19 |
| **QUART** | **0.35** | **0.33** |

### Sim-to-Real Scaling Experiment

| Simulation Data : Real Data | Success Rate |
|-------------------|--------|
| 0K : 3K | 3/20 |
| 25.6K : 3K | 7/20 |
| **256K : 3K** | **13/20** |

### Key Findings
- **R3M lacks language alignment**: Although it has some capability in simple perception tasks (letter discrimination), the lack of language semantic alignment leads to total failure in other tasks.
- **Limitations of VLM baselines**: CLIP and VC-1 can complete basic navigation, but fail completely on complex mechanical movements (crawling, unloading), indicating that while VLMs can understand abstract concepts of the world, they cannot directly translate them into mechanical task executions.
- **Source of QUART's advantage**: The decoder-only VLA architecture allows implicit learning of dependencies across different action dimensions, which is impossible with a single-layer MLP policy head.
- **Generalization to unseen instructions**: Leveraging the language capability of large models, QUART can understand semantic variations not present in the training set (e.g., "navigate to target" vs "go to object"), and even complex/compositional instructions ("first... then...") and spatial relationship instructions.
- **Effective scaling of simulation data**: Increasing simulation data from 0K to 256K boosts real-world deployment success rate from 15% to 65%.

## Highlights & Insights
- **Clearly defined QUAR-VLA paradigm**: Clarifies the differences and respective limitations of the three paradigms: vision-action (VA), language-action (LA), and vision-language-action (VLA). The introduction of VLA is meaningful and self-consistent.
- **Engineering insight of action space design**: The 12-dimensional high-level command includes both velocity control and gait/pose parameters. A 2Hz inference frequency combined with a low-level controller represents an elegant engineering compromise between flexibility and feasibility, analogous to the decoupling of planning and control layers in autonomous driving.
- **Scaling curve of simulation scale**: Clearly demonstrates the positive correlation between the volume of simulation data and the success rate of real-world deployment, providing an empirical reference for sim-to-real.

## Limitations & Future Work
- The visual fidelity of the simulation environment is insufficient, and experiments were only conducted on flat terrains without considering complex terrains.
- Real-world data comprises only 3K episodes collected in lab environments; the Sim2Real gap remains significant.
- The success rate of the Unload task is only 12%; highly difficult whole-body manipulation tasks remain challenging.
- The 2Hz inference speed might be insufficient for scenarios requiring rapid response.
- Language templates in the dataset are relatively monotone (predefined format), lacking natural language diversity.

## Related Work & Insights
- **vs RT-2**: RT-2 applies VLMs to robotic arm manipulation; QUART transfers a similar idea to quadruped robots. However, quadruped kinematics are more complex, requiring a richer action space.
- **vs Tang et al. (QUAR-LA)**: The first language-control work for quadruped robots, but the lack of visual perception prevents the robot from autonomous navigation. QUAR-VLA integrates both vision and language.
- **vs VC-1/R3M**: These visual representation models are effective for simple perception tasks but lack end-to-end action generation capabilities. QUART's VLA architecture is significantly stronger.

## Rating
- Novelty: ⭐⭐⭐⭐ First to define the quadruped VLA paradigm + first large-scale dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multi-task, generalization, and sim-to-real, but lacks comparison with more VLA methods.
- Writing Quality: ⭐⭐⭐ Clear structure, but some expressions are repetitive.
- Value: ⭐⭐⭐⭐ Blazes the path for the VLA direction in quadruped robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model](../../ICLR2026/robotics/x-vla_soft-prompted_transformer_as_scalable_cross-embodiment_vision-language-act.md)
- [\[NeurIPS 2025\] Bridging Embodiment Gaps: Deploying Vision-Language-Action Models on Soft Robots](../../NeurIPS2025/robotics/bridging_embodiment_gaps_deploying_vision-language-action_models_on_soft_robots.md)
- [\[ICLR 2026\] Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process](../../ICLR2026/robotics/unified_diffusion_vla_vision-language-action_model_via_joint_discrete_denosing_d.md)
- [\[CVPR 2026\] ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models](../../CVPR2026/robotics/acot-vla_action_chain-of-thought_for_vision-language-action_models.md)
- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](../../ICML2026/robotics/from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)

</div>

<!-- RELATED:END -->

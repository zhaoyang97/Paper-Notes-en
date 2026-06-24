---
title: >-
  [Paper Note] GraspXL: Generating Grasping Motions for Diverse Objects at Scale
description: >-
  [ECCV 2024][Robotics][Grasp motion generation] GraspXL is proposed, an RL-based grasp motion generation framework that generalizes to over 500,000 unseen objects after training on only 58 objects, while simultaneously supporting multi-objective motion control (grasp region, heading, wrist rotation, and hand position) and multiple dexterous hand platforms.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Grasp motion generation"
  - "dexterous manipulation"
  - "reinforcement learning"
  - "large-scale generalization"
  - "multi-objective control"
date: 2026-05-08
content_hash: b666e7cdce6cb047
---

# GraspXL: Generating Grasping Motions for Diverse Objects at Scale

**Conference**: ECCV 2024  
**arXiv**: [2403.19649](https://arxiv.org/abs/2403.19649)  
**Code**: [Available](https://eth-ait.github.io/graspxl/)  
**Area**: Robotics  
**Keywords**: Grasp motion generation, dexterous manipulation, reinforcement learning, large-scale generalization, multi-objective control

## TL;DR

GraspXL is proposed, an RL-based grasp motion generation framework that generalizes to over 500,000 unseen objects after training on only 58 objects, while simultaneously supporting multi-objective motion control (grasp region, heading, wrist rotation, and hand position) and multiple dexterous hand platforms.

## Background & Motivation

Human hands possess extraordinary dexterity—the ability to grasp objects of arbitrary shapes, target specific regions, approach from specific directions, and do so without object-specific training. Replicating this ability is a core challenge in both computer animation and robotic grasping.

Existing grasp motion synthesis methods face **three key bottlenecks**:

**Dependence on expensive 3D hand-object interaction data**: Data-driven methods (e.g., D-Grasp) require precise 3D annotated sequences as training data, which are extremely costly to collect, and the models are constrained by the training distribution, limiting generalization.

**Limited generalization**: Most methods can only handle objects seen during training (DexVIP evaluates 0 novel objects) or support only a small number of test objects (UniDexGrasp covers about 100), falling far short of real-world requirements.

**Lack of multi-objective control**: Existing methods typically support only a single objective (e.g., controlling only the heading direction) and cannot simultaneously satisfy multiple motion objectives such as grasp region, heading, wrist rotation, and position.

The core challenge lies in: how to enable a policy to generate stable grasping motions for **arbitrary 3D shapes** that satisfy **multi-objective motions** without **requiring 3D hand-object interaction data**? Key difficulties include: (1) highly diverse object shapes require a general shape-perception mechanism; (2) multiple objectives can conflict (satisfying an objective may cause contact that moves the object, undermining grasping stability); (3) joint configurations of different dexterous hands vary significantly.

## Method

### Overall Architecture

GraspXL operates within a reinforcement learning framework. Given an object and a hand model, the policy network $\boldsymbol{\pi}$ takes the motion target $\mathcal{T}$ and the physical state $\mathbf{s}$ as inputs. After transformation by the feature extraction layer $\Phi$, it outputs PD control targets as actions $\mathbf{a}$, which are executed by a physics simulator (RaiSim). The hand model consists of $L$ links, and the object is represented as a 3D point cloud.

### Key Designs

1. **Universal Feature Extraction $\Phi(\mathbf{s}, \mathcal{T})$**:

    - Inputs include joint angles $\mathbf{q}$, PD tracking errors $\mathbf{d}$, hand/object velocities $\mathbf{u}_h, \mathbf{u}_o$, contact vector $\mathbf{c}$, and contact force $\mathbf{f}$.
    - **Target differences**: $\tilde{\mathbf{v}}, \tilde{\mathbf{m}}, \tilde{\omega}$ represent the current differences from the target heading, position, and rotation, respectively.
    - **Link distance features** $\mathbf{l}^+ \in \mathbb{R}^{L \times 3}$: The vectors from each link to the closest point in the **graspable region**; $\mathbf{l}^- \in \mathbb{R}^{L \times 3}$: The vectors to the closest point in the **non-graspable region**.
    - Design Motivation: Link distance features are generic to object shapes (independent of specific point cloud encoders) and capture the spatial relationships with graspable/non-graspable regions, enabling the policy to generalize to arbitrary shapes.

2. **Multi-Objective Reward Function**:

    - The total reward is $r = r_{\text{goal}} + r_{\text{grasp}}$, decoupling target satisfaction and grasping stability.
    - Target reward: $r_{\text{goal}} = r_{\text{dis}} + r_\mathbf{v} + r_\omega + r_\mathbf{m}$
        - $r_{\text{dis}} = -\sum_i [w_d^+(i)\|\mathbf{h}_i - \mathbf{o}_i^+\|^2 - w_d^-(i)\|\mathbf{h}_i - \mathbf{o}_i^-\|^2]$: Encourages approaching the graspable region and staying away from the non-graspable region.
        - $r_\mathbf{v} = -w_\mathbf{v}\|\mathbf{v} - \bar{\mathbf{v}}\|^2$: Heading alignment.
    - Grasping reward: $r_{\text{grasp}} = r_\mathbf{c} + r_\mathbf{f} + r_{\text{anatomy}} + r_{\text{reg}}$
        - Contact reward $r_\mathbf{c}$ and force reward $r_\mathbf{f}$ (both distinguishing between graspable and non-graspable regions), anatomical constraints $r_{\text{anatomy}}$ (for the MANO hand), and regularization $r_{\text{reg}}$.
    - Design Motivation: The reward function is entirely independent of specific hand structures, relying only on the geometric relationships between link positions and the object surface, allowing direct transfer to different dexterous hands.

3. **Curriculum Learning**:

    - **Stage 1**: Train on **static objects** with higher weight on $r_{\text{goal}}$ to learn precise finger movements to satisfy the targets.
    - **Stage 2**: Fine-tune on **movable objects** with higher weight on $r_{\text{grasp}}$ to learn stable grasping.
    - **Objective-driven Guidance**: The difference between the target and current values is directly added as a bias to the wrist's 6DoF PD controller, accelerating exploration.
    - Design Motivation: Directly learning both target alignment and grasping on movable objects leads to **local optima**—the contact forces generated while satisfying targets can cause the object to flip, resulting in failed grasps. The curriculum design decouples these two learning tasks.

### Loss & Training

- Standard RL training using the PPO algorithm in the RaiSim physics simulator.
- The training set contains only 58 objects (26 from ShapeNet + 32 from PartNet).
- During training, targets are randomly sampled: random heading $\bar{\mathbf{v}}$, random rotation $\bar{\omega} \in [0, 2\pi)$, ensuring the graspable region width $\le 12$cm.
- Single Nvidia RTX 6000 GPU + 128 CPU cores.

## Key Experimental Results

### Main Results

| Method | Success Rate↑ | Mid. Error↓ | Head. Error↓ | Rot. Error↓ | Contact Ratio↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| SynH2R-PD | 26.5% | 4.30cm | 0.767rad | 0.857rad | 13.0% |
| SynH2R | 82.3% | 4.06cm | 0.522rad | 0.568rad | 53.4% |
| **GraspXL** | **95.0%** | **2.85cm** | **0.270rad** | **0.306rad** | **86.7%** |

ShapeNet test set (completely unseen objects):

| Method | Success Rate↑ | Mid. Error↓ | Head. Error↓ | Rot. Error↓ |
|------|:---:|:---:|:---:|:---:|
| SynH2R | 65.8% | 4.49cm | 0.642rad | 0.688rad |
| **GraspXL** | **81.0%** | **3.22cm** | **0.292rad** | **0.338rad** |

Large-scale Objaverse generalization (503k objects):

| Object Size | Success Rate↑ | Mid. Error↓ | Head. Error↓ | Rot. Error↓ |
|----------|:---:|:---:|:---:|:---:|
| Small | 85.9% | 3.20cm | 0.311rad | 0.362rad |
| Medium | 84.5% | 3.16cm | 0.274rad | 0.315rad |
| Large | 79.0% | 3.50cm | 0.271rad | 0.306rad |
| **Average** | **82.2%** | **3.32cm** | **0.279rad** | **0.319rad** |

### Ablation Study

| Configuration | Suc. Rate↑ | Mid. Err↓ | Head. Err↓ | Rot. Err↓ | Description |
|------|:---:|:---:|:---:|:---:|------|
| w/o Guidance | 90.0% | 3.22 | 0.394 | 0.425 | Without guidance, exploration efficiency decreases |
| w/o Distance | 81.6% | 2.90 | 0.419 | 0.475 | Without distance features, shape perception degrades |
| w/o Curriculum | 96.2% | 4.12 | 0.381 | 0.462 | Without curriculum, high success rate but poor target accuracy |
| **Full Model** | **95.0%** | **2.85** | **0.270** | **0.306** | Best overall performance |

Generalization to different dexterous hands (PartNet):

| Hand Model | Suc. Rate↑ | Mid. Error↓ | Head. Error↓ | No. of Joints |
|--------|:---:|:---:|:---:|:---:|
| MANO | 95.0% | 2.85cm | 0.270rad | 45 |
| Allegro | 95.3% | 4.38cm | 0.291rad | 16 |
| Shadow | 94.0% | 3.57cm | 0.317rad | 22 |
| Faive | 95.8% | 2.85cm | 0.228rad | 30 |

### Key Findings

- Generalizes to over 500k unseen objects after training on only 58 objects, achieving an 82.2% success rate and demonstrating remarkable generalization capability.
- SynH2R requires a week to generate reference poses for the ShapeNet test set, whereas GraspXL achieves real-time inference.
- Although the success rate with the curriculum is slightly lower than without it, the target accuracy is significantly improved (Heading Error dropped from 0.381 to 0.270), highlighting the necessity of decoupled learning.
- Link distance features are crucial for generalization—they directly extract shape intelligence from geometric distances without relying on point cloud encoders.
- Highly effective on both reconstructed objects (with noisy meshes) and AI-generated objects, presenting no barriers to real-world deployment.

## Highlights & Insights

- **Breakthrough in Scale**: Generalizing from 58 training objects to over 500,000 test objects is an unprecedented scale in grasp motion synthesis.
- **Zero Data Dependency**: It completely eliminates the need for 3D hand-object interaction data, learning solely through RL + physical simulation.
- **Exquisite Curriculum Design**: Static-to-movable object state transitions paired with target-to-grasp reward weight switches effectively resolve multi-objective conflicts.
- **Generality**: The same framework supports four different dexterous hands (MANO, Shadow, Allegro, Faive) without any architecture modification.

## Limitations & Future Work

- Only supports **rigid bodies** and does not handle deformable objects (e.g., cloth, ropes).
- The success rate drops when objects are excessively large or heavy (79% for Large objects), requiring torque constraints to be considered.
- Currently only generates **approach-and-grasp** motions, excluding subsequent **manipulation** (e.g., rotation, placement).
- Link distance features are sensitive to occlusion and self-occlusion; future work could integrate point cloud encoders to enhance shape understanding.
- Inference requires a physics simulator; direct deployment onto real robots necessitates sim-to-real transfer.

## Related Work & Insights

- **SynH2R** (Christen et al., ICRA 2024): A two-stage approach that generates reference poses via optimization and follows them using RL. However, the optimization process is time-consuming and the control objectives are limited.
- **UniDexGrasp/UniDexGrasp++**: Also employs RL for grasping but requires pre-generated reference poses and generalizes to only about 100 objects.
- **D-Grasp** (Christen et al., CVPR 2022): Utilizes reference grasp poses to generate physically plausible grasping motions.
- **Application of Curriculum Learning in RL**: The static-to-movable curriculum design in this work can be extended to other contact-rich RL tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The multi-objective control framework and curriculum learning design are creative, and the large-scale generalization is highly impressive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive, with evaluation spanning multiple datasets, multiple hand models, thorough ablations, and a test scale of over 500k objects.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition, consistent notation, and well-structured presentation.
- **Value**: ⭐⭐⭐⭐⭐ — Demonstrates grasp motion generation for 500k+ objects for the first time. The code and dataset are open-source, offering direct value to the robotics and computer animation communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to Control Physically-simulated 3D Characters via Generating and Mimicking 2D Motions](../../CVPR2026/robotics/learning_to_control_physically-simulated_3d_characters_via_generating_and_mimick.md)
- [\[ICLR 2026\] Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning](../../ICLR2026/robotics/emergent_dexterity_via_diverse_resets_and_large-scale_reinforcement_learning.md)
- [\[CVPR 2026\] AffordGen: Generating Diverse Demonstrations for Generalizable Object Manipulation with Affordance Correspondence](../../CVPR2026/robotics/affordgen_generating_diverse_demonstrations_for_generalizable_object_manipulatio.md)
- [\[CVPR 2025\] ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping](../../CVPR2025/robotics/manivideo_generating_hand-object_manipulation_video_with_dexterous_and_generaliz.md)
- [\[ECCV 2024\] Hierarchically Structured Neural Bones for Reconstructing Animatable Objects from Casual Videos](hierarchically_structured_neural_bones_for_reconstructing_animatable_objects_fro.md)

</div>

<!-- RELATED:END -->

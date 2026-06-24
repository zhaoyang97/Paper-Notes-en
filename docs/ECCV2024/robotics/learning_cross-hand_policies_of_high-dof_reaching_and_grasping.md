---
title: >-
  [Paper Note] Learning Cross-Hand Policies of High-DOF Reaching and Grasping
description: >-
  [ECCV 2024][Robotics][Dexterous Grasping] A two-stage hierarchical framework is proposed, which uses semantic keypoints and the Interaction Bisector Surface (IBS) as hand-agnostic state representations. Combined with a Transformer policy network and hand-specific adaptation models, it achieves zero-shot transfer of dexterous grasping policies across different high-DOF robotic hands.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Dexterous Grasping"
  - "Policy Transfer"
  - "Cross-Hand Generalization"
  - "Transformer"
  - "Interaction Bisector Surface (IBS)"
date: 2026-05-08
content_hash: aa75efae1da0536f
---

# Learning Cross-Hand Policies of High-DOF Reaching and Grasping

**Conference**: ECCV 2024  
**arXiv**: [2404.09150](https://arxiv.org/abs/2404.09150)  
**Code**: None  
**Area**: Robotics  
**Keywords**: Dexterous Grasping, Policy Transfer, Cross-Hand Generalization, Transformer, Interaction Bisector Surface (IBS)

## TL;DR

A two-stage hierarchical framework is proposed, which uses semantic keypoints and the Interaction Bisector Surface (IBS) as hand-agnostic state representations. Combined with a Transformer policy network and hand-specific adaptation models, it achieves zero-shot transfer of dexterous grasping policies across different high-DOF robotic hands.

## Background & Motivation

Robot "reaching-and-grasping" is a fundamental skill in manipulation. Existing learning methods typically train models for a single gripper; changing to another gripper requires recollecting data and training from scratch, which is highly expensive. Although prior works have explored cross-object generalization, **policy transfer across dexterous hand morphologies** remains largely unexplored.

Most existing cross-hand grasping methods (such as UniGrasp and GenDexGrasp) only generate static grasp poses and cannot perform online adjustments during physical execution. The core hypothesis of this work is that **grasping skills across different dexterous hands share commonalities**; what limits generalization is the representation of states and actions, rather than the skills themselves. Therefore, the key challenge lies in finding a hand-agnostic geometric representation to eliminate the influence of two factors:

**Hand Morphology Differences**: Distinct hands exhibit significant variations in joint space dimensions and topological structures.

**Hand Geometry Differences**: Representations like point clouds or images may cause the policy to overfit to the geometric appearance of specific hands.

## Method

### Overall Architecture

The method adopts a two-stage hierarchical model:

1. **Unified Policy Model**: Shared across all hand types, taking hand-agnostic features as input and predicting displacements of semantic keypoints.
2. **Specific Adaptation Model**: Converts keypoint displacements into joint angle changes of the specific hand type.

| Module | Input | Output | Characteristics |
|------|------|------|------|
| Hand-Agnostic Feature Extraction | Scene Point Cloud + Hand Configuration | IBS Point Features + Semantic Keypoints | Unified representation, independent of specific hands |
| Unified Policy Model | IBS + Semantic Keypoints | Keypoint Displacements + Global Translation/Rotation + Stop Signal | Transformer architecture, shared across hands |
| Adaptation Model | Keypoint Displacements + Current Joint Angles | Joint Angle Changes | Lightweight MLP, trained separately for each hand |

### Key Designs

**1. Semantic Keypoints**

Inspired by the IK Rig in animation systems, two semantic keypoints are selected on each finger (the fingertip and the middle joint), along with the palm root point, to construct a hand-agnostic state representation. Keypoint positions are computed through forward kinematics relative to the hand's local coordinate system. The complete semantic keypoint input consists of $6(K+1)$ dimensions, where $K$ is the number of fingers.

**2. Interaction Bisector Surface (IBS)**

IBS is the Voronoi diagram between the hand and the scene, encoding their spatial interaction. By voxelizing a spherical region around the palm center, the approximate IBS points are computed and downsampled to 4096 points as network input. Each IBS point contains rich features: coordinates, distance to the scene, distance to the hand, a foreground indicator, a one-hot encoding of hand part assignment, and a hand surface orientation indicator.

**3. Transformer Policy Network**

The network consists of three components:
- **Per-Finger Encoder**: Uses MLP and PointNet to encode the semantic points and IBS points of each finger, respectively.
- **Transformer Encoder**: Fuses information across different fingers and different representations via a self-attention mechanism.
- **Per-Finger/Global Decoder**: Predicts the keypoint displacement for each finger and the global motion.

This design allows the model to naturally adapt to hands with varying numbers of fingers (e.g., transferring from a five-finger hand to a four-finger hand).

### Loss & Training

**Training occurs in two stages:**

1. **Joint Training**: The policy model and the adaptation model are trained simultaneously with independent loss functions, and gradients do not propagate across models. Training lasts for 800k steps.
2. **Transfer Training**: The policy model is frozen, and the adaptation model is trained from scratch for the new hand, requiring only 50k steps.

The **policy model** is trained using reinforcement learning (Soft Actor-Critic algorithm) with a reward function including:
- Task Reward: Successful and stable grasping.
- Approaching Reward: Avoiding collisions between the hand and the scene.

The **adaptation model** is trained using a self-supervised cycle loss:

$$L_{point}(\theta) = \frac{1}{2}\sum_{k=1}^{K}\sum_{i=0}^{1}(e_k^i - p_k^i - \Delta p_k^i)^2$$

where $e_k^i = FK_k^i(j + \Delta j)$ represents the expected keypoint position computed via differentiable forward kinematics. Additionally, a self-collision loss is added to avoid inter-finger penetration.

## Key Experimental Results

### Main Results

Our experiments are evaluated on 5 types of dexterous hands: Shadow, Schunk, Mano, Rutgers, and Allegro. The policy is trained on the Shadow Hand, and the other hand types are used for transfer testing.

| Method | Shadow SR | Schunk SR | Mano SR | Rutgers SR | Allegro SR |
|------|-----------|-----------|---------|------------|------------|
| Single (End-to-End) | 72.2% | - | - | - | - |
| UNI+OCM | 50.1% | 41.2% | 45.5% | 38.9% | - |
| UNI+GCM | 64.0% | 45.5% | 50.4% | 41.4% | - |
| UNI+IBS | 68.0% | 54.6% | 61.2% | 42.6% | - |
| **Ours** | **71.3%** | **65.3%** | **65.2%** | **54.8%** | **55.0%** |

### Ablation Study

| Comparison | Findings |
|--------|------|
| Two-Stage vs End-to-End | The two-stage framework performs close to the end-to-end model on the source hand (71.3% vs 72.2%) but is capable of transferring to other hands. |
| IBS vs OCM | OCM overfits to the training objects on the source hand (50.1%), whereas IBS significantly outperforms OCM (68.0% vs 50.1%). |
| IBS vs GCM | GCM performing reasonably well on the source hand (64.0%) but transfers poorly due to overfitting to hand geometry. |
| Transformer vs Naive Concatenation | The Transformer policy outperforms feature concatenation across all hands and is able to adapt to the four-fingered Allegro hand. |

### Key Findings

1. **IBS is the most effective spatial interaction representation**: It balances both object and hand geometry, significantly outperforming the sole use of Object Contact Maps (OCM) or Grasp Contact Maps (GCM).
2. **Transformer architecture is critical**: It not only improves transfer performance but also enables the model to adapt to hands with different finger counts.
3. **High transfer efficiency**: The adaptation model for a new hand requires only 50k steps of training, which is far fewer than the 800k steps required for joint training.
4. **Excellent real-time performance of the adaptation model**: Compared to optimization-based methods, the neural network adaptation model is several times faster and collision-free.

## Highlights & Insights

- **Core Insight**: "Commonalities in grasping skills exist across different hand types; the key is finding an appropriate representation." This concept can be generalized to other cross-morphology skill transfer tasks.
- **Cross-Domain Effectiveness of IBS**: As a product of the Voronoi diagram, IBS is naturally robust to the geometry of both interacting entities, which could also prove useful in other contact-rich tasks beyond grasping.
- **Keypoints as a Unified Action Space**: Similar to IK control in animation systems, semantic keypoints provide an intuitive and unified action interface.

## Limitations & Future Work

1. **Evaluated only in simulation**: Based on PyBullet, sim-to-real transfer is not yet validated on physical hardware.
2. **Limited variations in hand morphology**: All five tested hands are anthropomorphic (human-like); non-anthropomorphic configurations (such as soft hands) are not tested.
3. **Insufficient object diversity**: Test objects are from limited datasets like YCB; generalization to more irregular shapes is not fully validated.
4. **Adaptation model still requires training data**: Although transfer training is rapid, it still requires collecting data for each new hand morphology.

## Related Work & Insights

- **Difference from GraspXL**: GraspXL trains independent models for each hand type and does not allow direct policy transfer.
- **Relation to Motion Retargeting**: While motion retargeting can visually generate similar motions, it performs poorly in dynamic physical execution. This work directly learns executable policies.
- **Possible Extension Directions**: Combined with large-scale pre-training, it may enable zero-shot transfer across more diverse morphologies.
- **Insights**: The concept of unified representation + hierarchical architecture can be applied to other multi-morphology robot tasks (such as locomotion and manipulation).

## Rating

| Dimension | Rating (1-5) | Comment |
|------|-----------|------|
| Novelty | 4 | Achieves policy transfer between dexterous hands for the first time; the IBS + keypoints representation design is elegant. |
| Technical Depth | 4 | The two-stage framework is well-designed, and the Transformer network architecture is highly tailored. |
| Experimental Thoroughness | 3.5 | Ablation studies are sufficient, though physical robot experiments are lacking. |
| Writing Quality | 4 | Clear structure with well-articulated motivation. |
| Value | 3.5 | Highly valuable reference for multi-hand robotic systems, but practical deployment requires sim-to-real validation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GraspGen-X: Cross-Embodiment 6-DOF Diffusion-based Grasping](../../CVPR2026/robotics/graspgen-x_cross-embodiment_6-dof_diffusion-based_grasping.md)
- [\[ECCV 2024\] An Economic Framework for 6-DoF Grasp Detection](an_economic_framework_for_6-dof_grasp_detection.md)
- [\[ECCV 2024\] GraspXL: Generating Grasping Motions for Diverse Objects at Scale](graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)
- [\[CVPR 2025\] Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References](../../CVPR2025/robotics/learning_physics-based_full-body_human_reaching_and_grasping_from_brief_walking_.md)
- [\[ICML 2026\] Fourier Features Let Agents Learn High Precision Policies with Imitation Learning](../../ICML2026/robotics/fourier_features_let_agents_learn_high_precision_policies_with_imitation_learnin.md)

</div>

<!-- RELATED:END -->

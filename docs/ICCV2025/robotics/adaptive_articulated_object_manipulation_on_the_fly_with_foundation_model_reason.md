---
title: >-
  [Paper Note] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding
description: >-
  [ICCV 2025][Robotics][articulated object manipulation] This paper proposes AdaRPG, a framework that leverages foundation vision-language models for part-level segmentation and affordance reasoning on articulated objects, and employs GPT-4o to generate high-level control code for adaptively scheduling atomic manipulation skills, achieving cross-category zero-shot generalization in both simulation and real-world environments.
tags:
  - "ICCV 2025"
  - "Robotics"
  - "articulated object manipulation"
  - "foundation models"
  - "part segmentation"
  - "affordance prediction"
  - "adaptive policy"
date: 2026-05-08
content_hash: f1539bc6b166d82d
---

# Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding

**Conference**: ICCV 2025
**arXiv**: [2507.18276](https://arxiv.org/abs/2507.18276)  
**Code**: None  
**Area**: Robotic Manipulation
**Keywords**: articulated object manipulation, foundation models, part segmentation, affordance prediction, adaptive policy

## TL;DR
This paper proposes AdaRPG, a framework that leverages foundation vision-language models for part-level segmentation and affordance reasoning on articulated objects, and employs GPT-4o to generate high-level control code for adaptively scheduling atomic manipulation skills, achieving cross-category zero-shot generalization in both simulation and real-world environments.

## Background & Motivation
Articulated objects (e.g., bottles, doors, safes, microwaves) consist of multiple movable parts and joints, making their manipulation a core challenge in robotics. In real-world scenarios, robots must handle complex adaptive manipulation tasks — for instance, a safe must be unlocked before it can be opened, yet the lock state is not directly observable, requiring iterative attempts and strategy adjustment based on feedback.

Existing approaches face two major bottlenecks: (1) the geometric diversity of real articulated objects is enormous, making it difficult for visual perception and affordance learning to generalize to novel categories; and (2) the manipulation mechanisms of different object categories (joint constraints, locking mechanisms, etc.) vary substantially, preventing direct transfer of manipulation policies. Together, these factors hinder the construction of a unified, cross-category adaptive manipulation policy.

The root cause lies in the following: whole-object-level geometric variation is too large to learn effectively, yet local parts across different categories (e.g., handles, knobs, buttons) share similar geometric characteristics. This key insight motivates the use of *parts* as an intermediate representation to improve the generalizability of affordance prediction.

**Key Insight**: Integrate the powerful generalization capabilities of foundation models in visual perception and language reasoning to build a complete pipeline from part segmentation to affordance reasoning to high-level policy generation. **Core Idea**: part-level affordance modeling + foundation model reasoning = cross-category adaptive manipulation.

## Method

### Overall Architecture
AdaRPG comprises three core components: (1) part-level affordance dataset construction and learning; (2) foundation-model-guided part localization and segmentation; and (3) GPT-4o-driven high-level control code generation with atomic skill execution. The overall pipeline is: RGB-D image acquisition → GPT-4o generates part descriptions → GroundingDINO localizes parts → SAM performs fine-grained segmentation → affordance model inference → atomic skill execution → GPT-4o generates adaptive control code.

### Key Designs

1. **Part-Level Affordance Dataset and Learning**:

    - **Function**: Extracts functional parts (handles, buttons, knobs, etc.) from 11 object categories in the PartNet-Mobility dataset to construct a part-level point cloud–affordance annotation dataset.
    - **Mechanism**: Given a part point cloud $O_i$, an automatic algorithm annotates high-affordance regions. High-affordance points are located near the part center, determined via centroid estimation and bounding-box center refinement. PointNet++ is used as the feature extractor, outputting a per-point affordance score $V(O_i, p_i) \in [0,1]$, trained with binary cross-entropy loss: $L_V = \text{BCELoss}(r_i, V(O_i, p_i))$.
    - **Design Motivation**: Part-level modeling substantially reduces geometric variability compared to whole-object-level modeling, enabling better generalization of affordance prediction to novel object categories. The dataset contains only part point clouds rather than complete objects, so the full shape is never exposed during training.

2. **Foundation-Model-Guided Part Localization and Segmentation**:

    - **Function**: Three frozen foundation models operate in cascade to achieve precise localization and segmentation of parts on unseen objects.
    - **Mechanism**: RGB image → GPT-4o generates detailed part descriptions (no more than three sentences, distinguishing functional components such as fixed vs. movable handles) → descriptions serve as prompts to GroundingDINO for bounding box generation → bounding boxes serve as prompts to SAM for fine-grained segmentation → back-projection onto the depth map yields 3D part point clouds.
    - **Design Motivation**: Detailed textual descriptions significantly improve GroundingDINO detection accuracy compared to single-word prompts. All three foundation models are used in a frozen manner without any additional fine-tuning; the only trainable component is the affordance model.

3. **Atomic Skill Functions and High-Level Code Generation**:

    - **Function**: Defines six atomic manipulation functions in the end-effector coordinate frame (push/pull along z-axis, rotation, translation along y-axis), with GPT-4o generating Python control code.
    - **Mechanism**: Affordance scores determine the end-effector pose (the average of high-scoring points as the translation position; point cloud surface normals for orientation). GPT-4o generates structured Python code from natural language input, forming an adaptive control loop (e.g., grasp → iterative rotation → probabilistic pull attempts → sustained pulling upon success until completion).
    - **Design Motivation**: Abstracting manipulation into atomic functions allows GPT-4o to perform high-level reasoning rather than low-level control. All motions are executed via impedance control to ensure smooth and stable operation; real-time angular deviation correction enhances robustness.

### Loss & Training
The affordance model is trained with binary cross-entropy loss. Impedance control is applied to all motions for dynamic force-compliance regulation. GPT-4o uses a single unified prompt to generate control code for all object categories, requiring no object-specific scripting.

## Key Experimental Results

### Main Results: Manipulation Success Rate in Simulation

| Method | Bottle | Pen | PC | CM | Window | Door | Lamp |
|--------|--------|-----|----|----|--------|------|------|
| SAGE | 0.21 | 0.40 | 0.00 | 0.30 | 0.38 | 0.39 | 0.40 |
| CoPa | 0.58 | 0.47 | 0.17 | 0.40 | 0.08 | 0.39 | 0.20 |
| AdaManip | 0.46 | 0.53 | 0.50 | 0.60 | 0.46 | 0.44 | 0.30 |
| **AdaRPG** | **0.84** | **0.73** | **1.00** | **0.80** | **0.84** | **0.78** | **0.70** |

### Ablation Study

| Configuration | Bottle | Pen | PC | CM | Window | Door | Lamp | Note |
|---------------|--------|-----|----|----|--------|------|------|------|
| AdaRPG (full) | 0.84 | 0.73 | 1.00 | 0.80 | 0.84 | 0.78 | 0.70 | Full model |
| w/o prompt | 0.11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | Without GPT-guided descriptions; near-total failure |
| w/o affordance | 0.78 | 0.58 | 0.83 | 0.60 | 0.57 | 0.63 | 0.40 | Uniform affordance scores; average drop of ~15% |

### Key Findings
- AdaRPG surpasses all baseline methods across all 7 categories, achieving 100% success rate on Pressure Cooker.
- Part segmentation IoU exceeds 80% across all categories (reaching 0.99 for Bottle).
- Part-level affordance F1 score substantially outperforms the whole-object-level method VAT-MART (average 0.74 vs. 0.27).
- Real-world success rates even exceed simulation results, as the foundation models are pretrained on real-world data.

## Highlights & Insights
- **Parts as intermediate representations** constitute an elegant decoupling design: local part geometry is more similar across categories, naturally supporting generalization.
- **Three-stage frozen foundation model cascade** (GPT-4o → GroundingDINO → SAM) achieves high-quality part segmentation with zero training.
- **Real-world performance surpassing simulation** reveals the potential of foundation models in bridging the sim-to-real gap.

## Limitations & Future Work
- The atomic skill set is relatively simple (6 types) and may be insufficient for more complex tool-use scenarios.
- Inference latency from GPT-4o may limit real-time applicability.
- The affordance model still requires training on a specific dataset, precluding fully zero-shot operation.

## Supplementary Results: Real-World Performance

| Method | Bottle | Pressure Cooker | Microwave | Lamp |
|--------|--------|-----------------|-----------|------|
| SAGE | 5/10 | 6/10 | 3/10 | 6/10 |
| CoPa | 4/10 | 2/10 | 3/10 | 1/10 |
| AdaManip | 8/10 | 5/10 | 7/10 | 5/10 |
| **AdaRPG** | **9/10** | **10/10** | **9/10** | **8/10** |

Real-world performance comprehensively exceeds simulation results, attributed to the fact that all foundation models are pretrained on real-world data, while the simulation environment introduces a domain gap that slightly degrades performance. This finding suggests that incorporating more foundation models into robotic systems can better address real-world challenges.

## Related Work & Insights
- **vs. AdaManip**: AdaManip trains diffusion policies via imitation learning, requiring large amounts of expert demonstrations and exhibiting limited cross-category generalization; AdaRPG achieves zero-shot cross-category generalization through foundation models.
- **vs. SAGE**: SAGE relies on GAPartNet for part pose estimation, which is less accurate than affordance representations, and errors propagate through the pipeline.
- **vs. CoPa**: CoPa employs GraspNet for general-purpose grasping, but such grasps fail to align with functional parts of articulated objects, leading to early execution failures.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The framework combining part-level affordance with foundation models is elegantly designed and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 7 simulation + 4 real-world object categories, with complete ablations and comprehensive baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Figures and text are well coordinated; the pipeline is clearly presented.
- **Value**: ⭐⭐⭐⭐ Provides a practical zero-shot generalization solution for adaptive robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PA3FF: Part-Aware Dense 3D Feature Fields for Generalizable Articulated Object Manipulation](../../ICLR2026/robotics/pa3fflearning_part-aware_dense_3d_feature_field_for_generalizable_articulated_ob.md)
- [\[CVPR 2026\] Physically Ground Commonsense Knowledge for Articulated Object Manipulation with Analytic Concepts](../../CVPR2026/robotics/physically_ground_commonsense_knowledge_for_articulated_object_manipulation_with.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[ICCV 2025\] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)
- [\[ICLR 2026\] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](../../ICLR2026/robotics/from_spatial_to_actions_grounding_vision-language-action_model_in_spatial_founda.md)

</div>

<!-- RELATED:END -->

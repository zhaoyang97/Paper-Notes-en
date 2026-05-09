---
title: >-
  [Paper Note] Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation
description: >-
  [ICLR 2026][3D Vision][3D feature field] This paper proposes PA3FF (Part-Aware 3D Feature Field), a natively 3D dense part-aware feature representation. By combining a Sonata pre-trained backbone with geometric and semantic contrastive learning, PA3FF yields zero-shot part-level features. Paired with a Part-Aware Diffusion Policy (PADP), the system achieves few-shot, highly generalizable articulated object manipulation, substantially outperforming baselines such as CLIP, DINOv2, and GenDP in both simulation and real-world settings.
tags:
  - ICLR 2026
  - 3D Vision
  - 3D feature field
  - part-aware
  - articulated object manipulation
  - contrastive learning
  - diffusion policy
date: 2026-05-08
content_hash: 656c20d64bb03bee
---

# Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation

**Conference**: ICLR 2026
**arXiv**: [2602.14193](https://arxiv.org/abs/2602.14193)
**Code**: [https://pa3ff.github.io/](https://pa3ff.github.io/)
**Area**: 3D Vision / Robot Manipulation
**Keywords**: 3D feature field, part-aware, articulated object manipulation, contrastive learning, diffusion policy

## TL;DR

This paper proposes PA3FF (Part-Aware 3D Feature Field), a natively 3D dense part-aware feature representation. By combining a Sonata pre-trained backbone with geometric and semantic contrastive learning, PA3FF yields zero-shot part-level features. Paired with a Part-Aware Diffusion Policy (PADP), the system achieves few-shot, highly generalizable articulated object manipulation, substantially outperforming baselines such as CLIP, DINOv2, and GenDP in both simulation and real-world settings.

## Background & Motivation

**Background**: Robotic manipulation of articulated objects (e.g., microwave doors, drawers, faucets) requires understanding functional parts (handles, knobs, etc.) to determine where and how to interact. Current approaches predominantly leverage 2D foundation models (CLIP, DINOv2, SigLIP) to extract semantic features from images for policy learning.

**Limitations of Prior Work**: 2D features inherently lack 3D geometric information and spatial continuity. Methods that lift 2D features into 3D (multi-view fusion, NeRF distillation, etc.) suffer from three core problems: (a) long inference times (up to several minutes); (b) multi-view feature inconsistency — 2D features from different viewpoints conflict when aggregated in 3D; and (c) the patch-based design of ViT architectures reduces spatial resolution by a factor of 14, causing fine-grained parts (e.g., refrigerator handles) to be lost.

**Key Challenge**: How to obtain 3D features that simultaneously satisfy geometric precision, semantic part-awareness, and cross-object generalizability, while supporting real-time feed-forward inference.

**Goal**: (a) Construct a natively 3D, part-level dense feature field; (b) enable few-shot (30 demonstrations) generalizable manipulation policies built upon this representation.

**Key Insight**: Rather than distilling from 2D, the paper exploits Sonata/PTv3, a self-supervised model pre-trained on 140K point clouds, to provide 3D priors. Part semantics are then injected via contrastive learning — points within the same part are pulled together in feature space, while points from different parts are pushed apart.

**Core Idea**: Construct a dense 3D feature field using a natively 3D pre-trained model and part-level contrastive learning, enabling robotic policies to generalize to unseen articulated objects with minimal demonstrations.

## Method

### Overall Architecture

The pipeline consists of three stages:

- **Stage 1 (Pre-training)**: Sonata (self-supervised Point Transformer V3) pre-trained on 140K point clouds to acquire general 3D geometric priors. PTv3 is architecturally adapted by removing most downsampling layers and stacking additional transformer blocks to preserve fine-grained details.
- **Stage 2 (Contrastive Learning)**: PA3FF features are refined on large-scale annotated datasets (PartNet-Mobility, 3DCoMPaT, PartObjaverse-Tiny) using geometric and semantic contrastive losses to inject part-awareness.
- **Stage 3 (Policy Learning)**: The frozen PA3FF backbone extracts point cloud features → a Transformer encoder aggregates them → a diffusion action head generates continuous action sequences.
- **Input**: Point clouds from multi-view depth cameras + robot proprioceptive state.
- **Output**: An action sequence over a future horizon $H$ (end-effector pose + gripper state).

### Key Designs

1. **Geometric Contrastive Loss $\mathcal{L}_{Geo}$**:

   - **Function**: Pulls features of points belonging to the same part together in feature space while pushing apart points from different parts.
   - **Mechanism**: Based on Supervised Contrastive Loss. Given feature-label pairs $\{f_k, a_k\}$ for $N$ points, a softmax similarity loss is computed over positive pairs sharing the same label.
   - **Design Motivation**: The purely geometric constraint ensures that features reflect 3D part structure without relying on any 2D information.

2. **Semantic Contrastive Loss $\mathcal{L}_{Sem}$**:

   - **Function**: Aligns point features with text embeddings of part names.
   - **Mechanism**: A SigLIP text encoder encodes part names (e.g., "handle", "knob") into semantic vectors $\mathbf{x}_k = \text{SigLIP}(s_k)$; an InfoNCE loss then aligns point features with the corresponding part-name text features.
   - **Design Motivation**: Semantic alignment ensures features encode not only spatial relationships but also functional meaning — "handle" and "knob" are distinctly separated in feature space.

3. **Part-Aware Diffusion Policy (PADP)**:

   - **Function**: Leverages PA3FF features for diffusion-based policy learning.
   - **Mechanism**: The frozen PA3FF backbone extracts point cloud embeddings → semantic embeddings of task-relevant part names serve as CLS tokens to guide a Transformer encoder in feature aggregation → the aggregated features are concatenated with proprioceptive state and compressed via MLP → actions are generated using DDPM training and DDIM inference.
   - **Design Motivation**: The part-name semantic CLS token steers attention toward task-relevant parts (e.g., "handle" for opening a microwave), preventing the policy from being distracted by irrelevant regions.

### Loss & Training

- Total feature learning loss: $\mathcal{L}_{total} = \mathcal{L}_{Geo} + \mathcal{L}_{Sem}$
- Policy training: standard DDPM MSE loss $\mathcal{L}(\phi) = \text{MSE}(\mathbf{a}_t, D_\theta(\mathbf{o}_t, \tilde{\mathbf{a}}_t, k))$
- The PA3FF backbone is fully frozen during policy learning; only the Transformer encoder, MLP, and diffusion head are trained.
- Real-world experiments require only 30 human teleoperation demonstrations per task.

## Key Experimental Results

### Main Results

Simulation results (PartInstruct 5-level generalization benchmark, success rate %):

| Method | Test1 (OS) | Test2 (OI) | Test3 (TP) | Test4 (TC) | Test5 (OC) | Avg. |
|--------|-----------|-----------|-----------|-----------|-----------|------|
| Act3D | 6.25 | 5.68 | 4.55 | 0.0 | 2.08 | 3.88 |
| DP (Diffusion Policy) | 7.27 | 8.64 | 8.18 | 3.75 | 6.67 | 5.96 |
| DP3 | 23.18 | 23.18 | 18.18 | 7.73 | 6.67 | 15.40 |
| GenDP | 24.34 | 23.36 | 24.53 | 10.00 | 14.61 | 19.36 |
| **PADP (Ours)** | **36.76** | **34.33** | **32.45** | **13.75** | **26.67** | **28.79** |

PADP achieves an average success rate of 28.79%, representing an absolute improvement of **9.4%** over the strongest baseline GenDP.

Real-world results (8 tasks, 10 trials each for train/test splits):

| Method | Pot Lid | Drawer | Box | Laptop | Microwave | Bottle | Kettle Lid | Dispenser | Mean (Test) |
|--------|---------|--------|-----|--------|-----------|--------|------------|-----------|------------|
| DP | 2/10 | 1/10 | 1/10 | 3/10 | 0/10 | 2/10 | 0/10 | 0/10 | 11.25% |
| GenDP | 6/10 | 5/10 | 3/10 | 4/10 | 3/10 | 4/10 | 2/10 | 1/10 | 35.0% |
| **PADP** | **6/10** | **6/10** | **5/10** | **7/10** | **5/10** | **6/10** | **5/10** | **3/10** | **58.75%** |

### Ablation Study

Generalization analysis on the Open Bottle task (real-world, completion rate % over 10 trials):

| Method | Original | Spatial Perturbation | Object Generalization | Environment Generalization |
|--------|----------|---------------------|----------------------|---------------------------|
| DP | 50 | 40 | 10 | 0 |
| DP3 | 40 | 20 | 20 | 10 |
| GenDP | 50 | 20 | 30 | 30 |
| **PADP** | **80** | **70** | **60** | **60** |

### Key Findings

- PA3FF substantially outperforms 2D-based methods in spatial resolution: DINOv2's ViT patch design reduces resolution by 14×, completely losing fine parts such as refrigerator handles, whereas PA3FF operates at the per-point level and is free of this issue.
- PA3FF feature fields are smoother and more consistent than those of DINOv2/SigLIP, exhibiting no multi-view stitching artifacts — this is especially notable on symmetric objects such as faucets.
- Generalization to unseen objects and environments is achieved with only 30 demonstrations, demonstrating that the part features extracted by PA3FF are genuinely cross-object transferable.
- Under the most challenging environment generalization setting (added distractors + background change), PADP maintains a 60% success rate while all baselines degrade substantially.

## Highlights & Insights

- **Fundamental advantage of native 3D over 2D lifting**: The approach avoids multi-view fusion inconsistency and resolution loss, representing a paradigm shift from "projecting into 3D" to "being natively 3D." This perspective is broadly applicable to other tasks requiring fine-grained 3D understanding.
- **Dual contrastive learning for joint geometric and semantic alignment**: One loss governs spatial structure; the other governs functional semantics. These two objectives are complementary. The dual-loss design is highly general and transferable to any scenario requiring features that are simultaneously structure-aware and semantics-aware.
- **Part-name CLS token for attention guidance**: Using task-relevant part names as CLS tokens within the policy network is a lightweight yet effective mechanism for task conditioning.

## Limitations & Future Work

- **Dependency on part annotations**: Training PA3FF requires datasets with part-level labels such as PartNet-Mobility, limiting coverage of novel object categories.
- **Moderate absolute success rates in simulation**: The best PADP variant achieves only 28.79% average success, indicating that articulated object manipulation remains an extremely challenging problem.
- **No support for deformable objects**: PA3FF assumes rigid parts and is not applicable to deformable objects such as cloth or ropes.
- **High real-world deployment cost**: Point cloud fusion from three depth cameras is required; single-camera monocular depth estimation could be explored to reduce hardware requirements.

## Related Work & Insights

- **vs. GenDP**: GenDP also targets category-level generalization but relies on DINOv2 for 2D semantic fields, suffering from multi-view inconsistency and insufficient resolution. PADP addresses both root problems with native 3D features, achieving a 23.75% absolute improvement in real-world performance.
- **vs. DP3**: DP3 uses point clouds but lacks part-level semantic features, resulting in significantly weaker generalization than PADP.
- **vs. 2D lifting methods (LERF, F3RM)**: These methods require per-scene NeRF optimization, making inference slow and incompatible with real-time use; PA3FF performs feed-forward inference. Transfer insight: any task requiring fine-grained 3D semantics (e.g., industrial inspection, surgical robotics) could benefit from native 3D features as an alternative to 2D lifting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of a native 3D part feature field with dual contrastive learning is novel, though individual components (Sonata, contrastive learning, diffusion policy) are not new in themselves.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 16 simulation tasks with 5-level generalization, 8 real-world tasks, ablation studies, multi-dimensional generalization analysis, and downstream applications (correspondence, segmentation) — extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and experiments are presented in detail, though the method section is notation-heavy and somewhat dense to read.
- **Value**: ⭐⭐⭐⭐⭐ Provides a strong 3D representation framework for robotic manipulation; code and project page are publicly available, offering high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation](ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)
- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [\[ICLR 2026\] PD²GS: Part-Level Decoupling and Continuous Deformation of Articulated Objects via Gaussian Splatting](pd2gs_part-level_decoupling_and_continuous_deformation_of_articulated_objects_vi.md)
- [\[ICLR 2026\] Splat Feature Solver](splat_feature_solver.md)
- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](../../ICCV2025/3d_vision/constraint-aware_feature_learning_for_parametric_point_cloud.md)

<!-- RELATED:END -->

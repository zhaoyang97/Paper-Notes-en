---
title: >-
  [Paper Note] 3D Single-Object Tracking in Point Clouds with High Temporal Variation
description: >-
  [ECCV 2024][3D Vision][3D Single-Object Tracking] HVTrack is the first to explore 3D single-object tracking under high temporal variation scenarios. It addresses coordinate-wise cloud shape variations, distractor interference, and background noise via three modules: Relative-Pose-Aware Memory (RPM), Base-Expansion Feature Cross-Attention (BEA), and Contextual Point Guided Self-Attention (CPA). On the KITTI-HV dataset with a 5-frame interval, it improves Success/Precision by 1…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Single-Object Tracking"
  - "Point Clouds"
  - "High Temporal Variation"
  - "Memory Module"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: 42d7a51532eb340f
---

# 3D Single-Object Tracking in Point Clouds with High Temporal Variation

**Conference**: ECCV 2024  
**arXiv**: [2408.02049](https://arxiv.org/abs/2408.02049)  
**Code**: No public code  
**Area**: 3D Vision / Autonomous Driving  
**Keywords**: 3D Single-Object Tracking, Point Clouds, High Temporal Variation, Memory Module, Attention Mechanism  

## TL;DR
HVTrack is the first to explore 3D single-object tracking under high temporal variation scenarios. It addresses coordinate-wise cloud shape variations, distractor interference, and background noise via three modules: Relative-Pose-Aware Memory (RPM), Base-Expansion Feature Cross-Attention (BEA), and Contextual Point Guided Self-Attention (CPA). On the KITTI-HV dataset with a 5-frame interval, it improves Success/Precision by 11.3%/15.7% over the state-of-the-art (SOTA).

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Existing 3D SOT methods are based on a key assumption: the changes in point cloud shape and object motion between adjacent frames are smooth. Consequently, they crop a small search region around the predicted location of the previous frame for tracking. However, high temporal variation (HV) occurs in real-world scenarios due to limited sensor frame rates, high-speed object motion, or frame-skipping tracking for computation savings. Under these conditions, the point cloud shape changes drastically due to rapid shifts in viewpoint, and the target might move out of the original search region, leading to a sharp decline in the performance of existing methods (e.g., the Success of CXTrack drops from 69.1 to 38.6 under a 5-frame interval).

### Solution Strategy

**Goal**: How to achieve robust 3D single-object tracking in high temporal variation scenarios where point cloud shapes change drastically and the search region needs to be expanded significantly (which introduces more distractors and noise)?

## Method

### Overall Architecture
Local features are extracted using DGCNN as the backbone, followed by an $L=2$ layer Transformer. Each layer contains three core modules: RPM extracts temporal templates $\rightarrow$ BEA performs template-search area cross-attention $\rightarrow$ CPA suppresses background noise. Finally, the RPN regresses the 3D bounding box, foreground mask, and viewing angle.

### Key Designs
1. **Relative-Pose-Aware Memory (RPM)**: Three memory banks are maintained: (1) layer feature memory, which stores historical Transformer features as templates (avoiding re-extraction for each frame); (2) mask memory, which contains foreground information; and (3) viewing angle memory, which stores historical viewing angles (encoded with sin/cos). The key innovation is the introduction of viewing angles: observing the same object from different angles yields completely different point cloud distributions. By recording the historical sequence of viewing angles, the model can implicitly learn transition patterns of point cloud distributions under changing poses. The three memories are concatenated and fused via a linear layer and self-attention.

2. **Base-Expansion Feature Cross-Attention (BEA)**: This modules splits the $H$ heads of multi-head attention into two groups: (1) $H/2$ heads perform standard cross-attention (base scale, local features); (2) $H/2$ heads first utilize EdgeConv to enlarge the receptive field for extracting more abstract features and then perform cross-attention (expansion scale, environmental context). This preserves local accuracy while exploiting spatial context to distinguish similar objects, with minimal extra computational overhead.

3. **Contextual Point Guided Self-Attention (CPA)**: This module uses the base and expansion attention maps from BEA to calculate the importance of each point. All points are sorted by importance and divided into $G=3$ groups, with each group aggregated into a different number of "contextual points": the low-importance group is allocated fewer contextual points (4), and the high-importance group is allocated more (32). This essentially compresses insignificant background features and reduces the key-value (KV) length in self-attention to lower computational cost.

### Loss & Training
- Five losses are jointly trained: coarse center $L_2$ + foreground mask cross-entropy + viewing angle Huber + targetness mask cross-entropy + bounding box Huber.
- The sequence length is 8 frames during training with a memory bank size of $K=2$ (due to GPU memory limits), and $K=6$ during testing.
- The KITTI-HV dataset is constructed by sampling from KITTI with frame intervals of $[2, 3, 5, 10]$, where the search region expands as the frame interval increases.

## Key Experimental Results
**KITTI-HV (5-Frame Interval)**

| Method | Car | Pedestrian | Cyclist | Mean |
|------|-----|-----------|---------|------|
| CXTrack | 38.6/42.4 | 34.1/49.6 | 25.7/32.9 | 35.3/42.8 |
| M2-Track | 52.6/61.6 | 35.9/51.3 | 49.3/63.6 | 44.1/55.2 |
| **HVTrack** | **60.3/68.9** | **35.1/52.1** | **58.2/71.7** | **46.6/58.5** |

**Standard Tracking**

| Dataset | HVTrack | CXTrack (SOTA) |
|--------|---------|---------------|
| KITTI Mean | 65.5/83.1 | 67.5/85.3 |
| Waymo Mean | **43.0/58.1** | 42.2/56.7 |
| NuScenes Mean | **51.1/62.2** | 42.0/51.8 |

Inference speed is 31 FPS, with a parameter size of 5.60 MB (versus 18.27 MB for CXTrack).

### Ablation Study
- Removing the Viewing Angle Memory (OM): Mean drops from 46.6/58.5 to 45.1/56.5, demonstrating the effectiveness of viewing angle information.
- Removing BEA (using vanilla CA): Mean drops from 46.6/58.5 to 46.0/57.5. While beneficial for small and large objects, there is a minor negative effect on medium-sized objects (Car) due to the introduction of more noise.
- Removing CPA (using vanilla SA): Mean drops from 46.6/58.5 to 45.8/57.5. CPA is effective for small and medium objects but detrimental to large objects (e.g., Van), where more foreground points are misclassified as low importance and thereby compressed.
- Memory bank size: Performance peaks at $K=6$, after which it decreases due to the accumulated error of historical information.

## Highlights & Insights
- **Viewing angle is a neglected yet crucial cue**: The root cause of point cloud shape variation is relative pose change. Encoding viewing angles into memory enables the model to predict trend variations in point cloud distribution.
- **Dual-scale (base + expansion) attention grouping** is an elegant solution to balance local precision and global context without significantly increasing computation.
- **Allocating computational resources based on importance** (where highly important points in CPA obtain more contextual points) represents a general-purpose efficiency optimization strategy.
- **KITTI-HV dataset construction methodology**: Simply sampling with frame intervals simulates high temporal variation scenarios, which is straightforward yet effective.

## Limitations & Future Work
- CPA uses fixed hyperparameters for grouping, which is detrimental for tracking large objects. Large objects have more foreground points, so under a fixed threshold, more foreground points are misclassified as low-importance.
- It ranks second in standard KITTI tracking with a 2% gap behind CXTrack, suggesting that optimization for high temporal variation might sacrifice some performance in smooth scenarios.
- The memory bank size during training is limited to $K=2$ due to GPU memory constraints, which does not fully match the testing size of $K=6$.
- The authors suggest using learnable functions instead of fixed hyperparameters in the future.

## Related Work & Insights
- **CXTrack**: Employs the same backbone and RPN, but HVTrack achieves an 11.3% higher Success under a 5-frame interval, proving the effectiveness of the feature correlation module design.
- **M2-Track**: A motion-based, match-free method that also degrades drastically under high temporal variation. HVTrack is more robust due to its temporal memory.
- **TAT**: Also utilizes temporal information but relies on a simple concatenation of RNN features, whereas HVTrack features a more elaborately designed RPM (integrating masks and viewing angles).
- **M3SOT**: The latest SOTA but achieves a Mean of only 29.4/37.2 under a 5-frame interval, compared to 46.6/58.5 for HVTrack.

## Connection to My Research
- High temporal variation tracking scenarios are directly related to the demands for frame-skipping inference and edge device deployment in autonomous driving.
- The importance-guided computational resource allocation strategy (the concept behind CPA) can be transferred to other Point Transformer scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to systematically define and solve the high temporal variation 3D SOT problem, featuring a novel viewing angle memory design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on four datasets (KITTI-HV, KITTI, Waymo, NuScenes) with comprehensive ablation studies and a highly detailed appendix.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems, with a logical mapping between three challenges and three corresponding modules.
- Value: ⭐⭐⭐ Although 3D tracking is not the primary focus of this research, the designs of temporal memory and importance-guided attention are highly valuable for reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SEED: A Simple and Effective 3D DETR in Point Clouds](seed_a_simple_and_effective_3d_detr_in_point_clouds.md)
- [\[ECCV 2024\] FLAT: Flux-Aware Imperceptible Adversarial Attacks on 3D Point Clouds](flat_flux-aware_imperceptible_adversarial_attacks_on_3d_point_clouds.md)
- [\[ECCV 2024\] Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds](heterogeneous_graph_learning_for_scene_graph_prediction_in_3d_point_clouds.md)
- [\[ECCV 2024\] MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)
- [\[ECCV 2024\] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)

</div>

<!-- RELATED:END -->

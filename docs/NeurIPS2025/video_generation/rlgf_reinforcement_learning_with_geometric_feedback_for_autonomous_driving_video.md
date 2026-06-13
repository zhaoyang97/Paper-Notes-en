---
title: >-
  [Paper Note] RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation
description: >-
  [NeurIPS 2025][Video Generation][Geometric Consistency] This paper is the first to systematically quantify geometric distortions in autonomous driving video generation. It proposes the RLGF framework…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Geometric Consistency"
  - "Reinforcement Learning"
  - "Video Diffusion Models"
  - "Autonomous Driving Data Generation"
  - "3D Perception"
date: 2026-05-08
content_hash: e6302649e80efc5b
---

# RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation

**Conference**: NeurIPS 2025
**arXiv**: [2509.16500](https://arxiv.org/abs/2509.16500)  
**Code**: None  
**Area**: Autonomous Driving / Video Generation
**Keywords**: Geometric Consistency, Reinforcement Learning, Video Diffusion Models, Autonomous Driving Data Generation, 3D Perception

## TL;DR

This paper is the first to systematically quantify geometric distortions in autonomous driving video generation. It proposes the RLGF framework, which leverages hierarchical geometric rewards (vanishing point → lane lines → depth → occupancy) combined with a latent-space sliding window optimization strategy to improve 3D object detection mAP by 12.7 absolute percentage points (25.75→31.42), substantially closing the performance gap between synthetic and real data.

## Background & Motivation

Autonomous driving systems demand large-scale, diverse training data. Diffusion-model-based video generation methods have achieved highly realistic driving videos with low FVD scores. However, these generated videos suffer from a critical yet underexplored flaw: geometric inconsistency.

The empirical evidence is striking: when using BEVFusion for 3D object detection on synthetic videos, mAP is only 25.7 (vs. 35.5 on real data), while 2D object detection (YOLOv5) mAP remains nearly on par (43.8 vs. 44.7). This reveals that current generative models preserve 2D appearance while degrading 3D scene structure. Three categories of failure are identified: vanishing point drift, lane topology inconsistency, and depth errors.

The root cause lies in the standard MSE training objective of diffusion models, which treats each pixel independently and thus fails to model higher-order geometric relationships such as perspective consistency and 3D spatial structure.

The core idea of this paper is to use pretrained autonomous driving perception models as reward providers, injecting geometric constraints into the video generation process via reinforcement learning.

## Method

### Overall Architecture

RLGF builds upon a pretrained video diffusion model $\epsilon_\theta$ fine-tuned via LoRA. During multi-step denoising, a latent-space sliding window extracts noisy latent features $z_k$ at intermediate steps, which are fed into frozen perception models (geometric perception model $\mathcal{P}_{geo}$ and occupancy prediction model $\mathcal{P}_{occ}$) to compute multi-level rewards. Gradients derived from comparing generated features against real video features $z_v$ update the LoRA parameters.

### Key Designs

1. **GeoScores Evaluation Metrics**:

    - Vanishing Point Error (VP Error): normalized distance between the predicted VP and the pseudo-ground-truth VP
    - Lane Topology Score (Lane F1-Score): F1 score of lane marking semantic segmentation
    - Depth Error (Depth RMSE): depth RMSE over road surface regions
    - Geometric discrepancies are quantified by running perception models separately on synthetic and real videos
    - Provides the first quantitative geometric quality assessment tool for autonomous driving video generation

2. **Latent-Space Sliding Window Optimization**:

    - Key observation: geometric structure emerges progressively during diffusion denoising—early steps establish coarse global geometry while later steps refine details
    - Instead of backpropagating through the full sampling chain (prohibitively expensive), rewards are provided within a randomly sampled window of $w$ steps
    - Advantages: (1) significantly reduces the computational graph and GPU memory usage; (2) enables targeted correction signals for both global structure formation and detail refinement stages
    - Optimization objective: $J(\theta_{LoRA}) = \mathbb{E}[R(z_k, z_v)]$

3. **Micro-Decode Module**:

    - A lightweight decoding module built from shallow layers of the VAE decoder, avoiding the high cost of full VAE decoding at every step
    - Receives noisy latent features $z_k$ and timestep $k$ (via Fourier Embedding), outputting enhanced frame features suitable for perception tasks
    - Real video latent features $z_v$ are processed with $k=0$ to produce reference features

4. **Hierarchical Geometric Reward**:

    - **Point-level (VP Reward)**: $r_{vp} = -\|p_{vp} - v_{ref}\|_2^2$, enforcing vanishing point consistency
    - **Line-level (Lane Reward)**: $r_{lane} = \text{F1-Score}(L_{pred}, L_{ref})$, ensuring valid lane topology
    - **Surface-level (Depth Reward)**: $r_{depth}$, computing depth consistency separately for road and vehicle regions using pixel masks to isolate critical areas
    - **Feature Alignment Reward**: $r_{align} = -D_{KL}(p(\text{feat}^{real}) \| p(\text{feat}^{gen}))$, aligning intermediate occupancy feature distributions
    - **3D Occupancy IoU Reward**: $r_{iou} = \text{IoU}(O^{gen}, O^{real})$, ensuring correct volumetric object layout

5. **Perception Model Architecture**:

    - Geometric perception model $\mathcal{P}_{geo}$: based on DINOv2 + DepthAnything-v2, trained with multi-task learning for VP/lane/depth
    - Occupancy prediction model $\mathcal{P}_{occ}$: based on FlashOcc architecture, inferring 3D occupancy grids from frame-level features
    - Both models operate in latent space with weights frozen after training

### Loss & Training

LoRA is applied to the Q/K/V projections of the DiT backbone attention layers; training is conducted on 8×A100 GPUs. The reward function is $R = R_{geo} + R_{occ}$, where $R_{geo}$ is a weighted combination of VP, lane, and depth terms. Pseudo-labels are generated using DepthAnything-v2 (depth), Grounded-SAM-2 (lane/road/vehicle segmentation), and geometric computation (VP).

## Key Experimental Results

### Main Results

**nuScenes validation set performance**:

| Method | FVD↓ | 3DOD mAP↑ | 3DOD NDS↑ | VP Error↓ | Lane F1↑ | Depth RMSE↓ |
|--------|------|-----------|-----------|-----------|----------|-------------|
| Real Data | - | 35.53 | 41.20 | - | - | - |
| DiVE | 68.4 | 25.75 | 33.61 | 0.086 | 0.792 | 1.822 |
| **DiVE+RLGF** | **67.6** | **31.42** | **36.07** | **0.068** | **0.879** | **0.772** |
| MagicDrive-v2 | 101.2 | 18.95 | 21.10 | 0.092 | 0.787 | 1.732 |
| **MagicDrive-v2+RLGF** | **99.8** | **23.21** | **27.80** | **0.079** | **0.854** | **0.983** |

### Ablation Study

| ID | $r_{vp}$ | $r_{lane}$ | $r_{depth}$ | $r_{align}$ | $r_{iou}$ | mAP↑ | NDS↑ |
|----|----------|-----------|------------|------------|---------|------|------|
| DiVE Baseline | - | - | - | - | - | 25.75 | 33.61 |
| 1 | ✓ | | | | | 26.31 | 33.66 |
| 2 | ✓ | ✓ | | | | 26.93 | 33.98 |
| 3 (Point–Line–Surface) | ✓ | ✓ | ✓ | | | 27.12 | 34.82 |
| 4 (Occupancy) | | | | ✓ | ✓ | 28.06 | 35.11 |
| **Full** | ✓ | ✓ | ✓ | ✓ | ✓ | **31.42** | **36.07** |

### Key Findings
- VP error reduced by 21% (0.086→0.068); depth RMSE reduced by 57% (1.822→0.772), indicating substantial geometric quality improvement
- 3DOD mAP improves from 25.75 to 31.42 (+12.7 absolute), NDS from 33.61 to 36.07, approaching the real-data level of 35.53
- FVD not only does not degrade but improves slightly (68.4→67.6), demonstrating that geometric correction does not compromise visual quality
- Occupancy rewards ($r_{align} + r_{iou}$) contribute most (yielding +2.31 mAP when used alone); point–line–surface rewards provide complementary gains (stacking them pushes mAP from 28.06 to 31.42)
- RLGF is plug-and-play, delivering significant improvements on two distinct baselines

## Highlights & Insights

- **High-value problem identification**: This is the first work to systematically expose the overlooked discrepancy between "visually realistic" and "geometrically correct," with the large gap between 2D and 3D detection performance serving as compelling evidence
- **GeoScores evaluation tool**: Provides a dedicated geometric quality assessment standard for the field, filling the gap left by FVD/FID metrics that cannot measure geometric consistency
- **Latent-space perception models**: Running perception models directly on noisy latent features is an elegant design choice that avoids the overhead of full decoding
- **Hierarchical rewards from point to volume**: The point-level (VP) → line-level (lane) → surface-level (depth) → volumetric-level (occupancy) hierarchy systematically covers geometric constraints at different scales

## Limitations & Future Work

- The quality of pseudo-labels for perception models directly affects reward signal accuracy; errors in the perception models themselves will propagate
- Depth estimation RMSE of the latent-space perception model (2.596) is notably higher than that of the pixel-space counterpart (1.798), indicating information loss from latent-space operation
- Validation is currently limited to the nuScenes dataset; generalization to other driving datasets remains unknown
- Computational overhead remains substantial, as it requires training additional perception and occupancy models
- Temporal geometric consistency (e.g., cross-frame object motion consistency) is not addressed

## Related Work & Insights

- **vs. VADER**: VADER applies RL training over the entire sampling chain, whereas RLGF's sliding window strategy is more efficient and enables stage-specific targeted feedback
- **vs. DPO-based video fine-tuning**: DPO methods use holistic preference scores and lack the fine-grained local geometric feedback provided by RLGF
- **Implications for data generation paradigms**: The shift from "pixel-level loss optimization" to "perception-driven geometric reward optimization" represents an important paradigm change that can be generalized to other data generation tasks requiring physical consistency

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic quantification of geometric distortions in video generation and proposal of a perception-driven RL correction framework
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-baseline validation, comprehensive ablation, and multi-dimensional evaluation, though limited to a single dataset
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation, detailed technical descriptions, and intuitive illustrations
- Value: ⭐⭐⭐⭐⭐ Significant importance to the field of synthetic data generation for autonomous driving; both GeoScores and RLGF have direct practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](../../CVPR2026/video_generation/goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)
- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](../../ICCV2025/video_generation/magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](../../ICCV2025/video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/video_generation/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](../../ICCV2025/video_generation/steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)

</div>

<!-- RELATED:END -->

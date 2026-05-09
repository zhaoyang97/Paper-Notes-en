---
title: >-
  [Paper Note] Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation
description: >-
  [CVPR 2026][Robotics][bimanual manipulation] This work leverages the pretrained 3D geometric foundation model π3 as a perception backbone, fuses 3D geometric, 2D semantic, and proprioceptive features, and jointly predicts future action chunks and future 3D Pointmaps via a diffusion model. Using only RGB inputs, the proposed method comprehensively surpasses point-cloud-based approaches on the RoboTwin bimanual benchmark.
tags:
  - CVPR 2026
  - Robotics
  - bimanual manipulation
  - 3D geometric foundation model
  - joint action–geometry prediction
  - π3
  - diffusion policy
date: 2026-05-08
content_hash: 0f30aa7d2d895eba
---

# Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation

**Conference**: CVPR 2026
**arXiv**: [2602.23814](https://arxiv.org/abs/2602.23814)
**Code**: [https://github.com/Chongyang-99/GAP.git](https://github.com/Chongyang-99/GAP.git)
**Area**: Robotics / Embodied Intelligence
**Keywords**: bimanual manipulation, 3D geometric foundation model, joint action–geometry prediction, π3, diffusion policy

## TL;DR
This work leverages the pretrained 3D geometric foundation model π3 as a perception backbone, fuses 3D geometric, 2D semantic, and proprioceptive features, and jointly predicts future action chunks and future 3D Pointmaps via a diffusion model. Using only RGB inputs, the proposed method comprehensively surpasses point-cloud-based approaches on the RoboTwin bimanual benchmark.

## Background & Motivation

### State of the Field

**State of the Field**: Bimanual manipulation demands precise 3D spatial reasoning and inter-arm coordination. Existing 2D methods (ACT, DP) lack spatial awareness, while 3D methods (DP3) are effective but rely on point cloud acquisition (requiring calibration, sensitive to noise, and difficult to obtain reliably in real-world settings). Meanwhile, 3D geometric foundation models (DUSt3R, π3, etc.) can already reconstruct high-quality 3D structures directly from RGB images. The key question is: can 3D foundation models be used directly as perception priors to achieve—or even surpass—point-cloud-level 3D perception using only RGB inputs?

### Starting Point

**Paper Goals**: Can a pretrained 3D geometric foundation model replace explicit point cloud pipelines to realize an RGB-only, 3D-aware bimanual manipulation policy, while also gaining predictive planning capability through joint future 3D geometry prediction?

## Method

### Overall Architecture
Three parallel encoders are fused: a π3 encoder processes sequential RGB frames to extract 3D geometric features; DINOv3 encodes the current frame for 2D semantic features; and an MLP encodes proprioception. The three 1024-dimensional features are fused by a 4-layer DETR Encoder into a unified semantic–geometric context $\mathbf{f}_c$. Conditioned on $\mathbf{f}_c$, a diffusion decoder jointly denoises and generates: (1) a future action chunk $a_{t:t+N}$ (7-DoF per arm); and (2) a future 3D latent $\mathbf{f}_{t+N}$, decoded by a Dense Head into a dense Pointmap $P_{t+N} \in \mathbb{R}^{H \times W \times 4}$.

### Key Designs
1. **π3 Geometric Encoder**: A pretrained π3 backbone extracts 3D geometric features from a temporal sequence of 6 frames (5 past frames + the current frame). π3 is a permutation-equivariant multi-view 3D reconstruction model capable of inferring dense geometry directly from RGB. Features from the last two layers are concatenated to form a 1024-dimensional representation. Crucially, π3 is frozen throughout training.

2. **Joint Action–Geometry Prediction**: The diffusion decoder simultaneously predicts actions and the 3D Pointmap latent $\mathbf{f}_{t+N}$ at a future timestep. This compels the policy to "imagine" the 3D scene state after action execution, forming implicit look-ahead planning. Ablation studies show that removing geometry imagination reduces success rate from 25.1% to 23.6%; removing both the 3D geometry module and geometry imagination reduces it further to 21.0%.

3. **Semantic–Geometry Fusion**: 2D semantics (DINOv3) and 3D geometry (π3) are complementary: geometry provides spatial structure while semantics provides task-relevant object understanding. Removing 2D semantics alone yields only a ~1% drop, whereas removing 3D geometry and geometry imagination causes a ~4% drop, indicating that 3D perception is the primary contributor.

### Loss & Training
$$\mathcal{L} = \|a - \hat{a}\|_1 + \lambda\|\mathbf{f}_{t+N} - \hat{\mathbf{f}}_{t+N}\|_1 + \gamma\|P_{t+N} - \hat{P}_{t+N}\|_1$$
3D latents are pre-extracted from all demonstrations using π3 as pseudo ground truth (stabilized via temporal windowing). Training runs for 200–600 epochs with batch size 32 on an RTX 4090.

## Key Experimental Results

| RoboTwin 2.0 | Metric | Ours | DP3 | ACT | DP | RDT |
|---|---|---|---|---|---|---|
| Dominant-select (16 tasks) | Avg SR (%) | **63.2** | 61.2 | 34.1 | 44.4 | 44.5 |
| Sync-bimanual (8 tasks) | Avg SR (%) | **51.3** | 40.7 | 32.4 | 37.1 | 47.0 |
| Seq-coordinate (8 tasks) | Avg SR (%) | **50.4** | 41.1 | 29.4 | 33.6 | 42.3 |
| Real-world (4 tasks) | Avg SR (%) | **40.0** | — | 23.8 | 25.0 | — |

### Ablation Study
- Remove 2D semantic module: 25.1% → 24.4% (−0.7%); semantics plays a supporting role.
- Remove geometry imagination: 25.1% → 23.6% (−1.5%); future 3D prediction is important for planning.
- Remove 3D geometry + geometry imagination: 25.1% → 21.0% (−4.1%); 3D perception is the core contribution.
- Data efficiency: with only 10 demonstrations, the proposed method already exhibits a learning signal, whereas the 2D method DP completely fails (0%).
- Real-world Hang Mug task: ACT and DP both achieve 0%, while the proposed method achieves 20%, demonstrating the value of 3D reasoning for complex tasks.

## Highlights & Insights
- Feeding RGB images directly into a 3D foundation model already surpasses explicit point cloud methods, eliminating the engineering overhead of calibration and point cloud acquisition.
- The design of predicting future 3D Pointmaps is elegant—it simultaneously serves as an auxiliary training signal and as implicit look-ahead planning.
- The evaluation scale of 32 RoboTwin tasks plus 4 real-world tasks is uncommon in the bimanual manipulation literature.
- The data efficiency advantage is notable: pretrained features deliver substantially better low-data performance compared to 2D methods trained from scratch.

## Limitations & Future Work
- Only a single-step future 3D Pointmap is predicted (at step $N$), lacking multi-step 3D trajectory prediction and persistent 3D memory.
- The method depends on the pretraining quality of π3 and may degrade on scenes not seen during π3 pretraining.
- Real-world experiments use only 50 demonstrations, limiting scale.
- The option to skip Pointmap decoding at inference time suggests room for improving inference efficiency.

## Related Work & Insights
- **DP3**: Relies on explicit point clouds requiring calibration and noise handling; the proposed method achieves superior 3D perception using only RGB via π3, comprehensively outperforming DP3.
- **G3Flow**: Projects 2D features into 3D, relying on DINOv2 and semantic flow; the proposed method works directly in the 3D latent space via π3.
- **RDT**: A 1.2B-parameter foundation model achieving 42.3% on Seq-coordinate vs. 50.4% for the proposed method, demonstrating that 3D prediction is more effective than scaling model size.
- **Xu et al.**: Jointly predicts actions and future image frames (2D); the proposed method predicts 3D Pointmaps instead, yielding stronger geometric consistency.

## Related Work & Insights
- Using 3D foundation models such as π3 as plug-and-play geometric backbones for manipulation policies is a promising direction.
- The joint action–geometry prediction paradigm can be extended to single-arm manipulation and navigation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ First to apply 3D geometric foundation models such as π3 to bimanual manipulation with joint geometry prediction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 32 simulation tasks + 4 real-world tasks + data efficiency analysis + ablation studies; evaluation is exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; experimental presentation is well-structured.
- Value: ⭐⭐⭐⭐ Provides a practical paradigm for RGB-only, 3D-aware bimanual manipulation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](../../ICLR2026/robotics/twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[CVPR 2026\] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)

<!-- RELATED:END -->

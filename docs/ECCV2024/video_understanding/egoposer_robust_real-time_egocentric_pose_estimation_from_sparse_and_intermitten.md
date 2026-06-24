---
title: >-
  [Paper Note] EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere
description: >-
  [ECCV 2024][Video Understanding][Egocentric Pose Estimation] Proposes EgoPoser to robustly estimate full-body poses from sparse and intermittent tracking signals of the head and hands from head-mounted devices. Through four core designs—global motion decomposition, realistic field-of-view modeling, SlowFast temporal fusion, and body-shape-aware pose optimization—it achieves state-of-the-art performance in large-scale real-world scenarios while running at over 600 fps.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Egocentric Pose Estimation"
  - "Mixed Reality"
  - "Sparse Tracking"
  - "Global Motion Decomposition"
  - "SlowFast"
date: 2026-05-08
content_hash: b6e5cb7d56d17827
---

# EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere

**Conference**: ECCV 2024  
**arXiv**: [2308.06493](https://arxiv.org/abs/2308.06493)  
**Code**: [https://github.com/siplab-gt/EgoPoser](https://github.com/siplab-gt/EgoPoser)  
**Area**: Video Understanding / Human Pose Estimation  
**Keywords**: Egocentric Pose Estimation, Mixed Reality, Sparse Tracking, Global Motion Decomposition, SlowFast

## TL;DR

Proposes EgoPoser to robustly estimate full-body poses from sparse and intermittent tracking signals of the head and hands from head-mounted devices. Through four core designs—global motion decomposition, realistic field-of-view modeling, SlowFast temporal fusion, and body-shape-aware pose optimization—it achieves state-of-the-art performance in large-scale real-world scenarios while running at over 600 fps.

## Background & Motivation

Current mixed reality (MR) systems (such as Meta Quest, Apple Vision Pro, HoloLens) provide tracking signals from only a three-point sparse input of the head and hands. Recovering full-body poses from these signals is a highly underdetermined problem. Existing methods suffer from three core limitations:

**Global Position Overfitting**: Prior methods (AvatarPoser, AGRoL, AvatarJLM) directly use the global pose in the world coordinate system as the network input. This causes the model to severely overfit to actions near the origin in the training data. When users move in large scenes (even with a displacement of just a few meters), prediction accuracy drops drastically.

**Assumption of Continuous Hand Visibility**: Existing methods assume that hand tracking signals are continuously available. However, in practice, the inward-outward tracking cameras of head-mounted devices (HMDs) have a limited field of view (FoV), and hands frequently leave the FoV, resulting in tracking interruptions.

**Neglecting Body Shape Variations**: Existing methods assume a uniform mean body shape skeleton, failing to adapt to body size variations among different users, which results in motion artifacts such as floating and ground penetration.

**Core Idea**: Enable position-agnostic pose prediction via global motion decomposition, handle intermittent hand signals with realistic FoV modeling, and adapt to different users using body-shape-aware pose optimization.

## Method

### Overall Architecture

EgoPoser takes the global positions $\mathbf{p}$ and orientations $\boldsymbol{\Theta}$ of the head and hands (3 tracking points in total) from HMDs. Through realistic FoV modeling $\to$ global motion decomposition $\to$ SlowFast feature fusion $\to$ Transformer encoder $\to$ human motion decoder, it outputs the global root orientation $\theta_{\text{global}}$, local joint rotations $\theta_{\text{local}}$, and body shape parameters $\beta$. The input window is the most recent 80 frames (fused into 40 frames via SlowFast). Finally, the 3D positions of 22 full-body joints are generated through forward kinematics of the SMPL model.

### Key Designs

1. **Realistic Field-of-View (FoV) Modeling**: Unlike previous works that use random frame dropping, EgoPoser realistically simulates the view frustum of the HMD cameras based on the spatial relationship between head pose and hand relative positions. The horizontal FoV angle $\alpha_h$ and vertical FoV angle $\alpha_v$ determine the hand visibility in the head coordinate system. When a hand is outside the FoV, its corresponding input features are zeroed out while preserving temporal continuity. This modeling captures the realistic temporal dependencies of hands entering and leaving the FoV.

2. **Global Motion Decomposition (GMD)**: Combining the advantages of global and local representations, a position-agnostic pose estimation strategy is proposed, containing two core operations:

    - **Temporal Normalization (TN)**: Subtracts the first frame's translation from the trajectory of each joint within the time window to extract the relative global trajectory: $\mathbf{p}_{\text{TN}}^{t_i,j} = \mathbf{p}_W^{t_i,j} - \mathbf{p}_W^{t_0,j}$
    - **Spatial Normalization (SN)**: Only normalizes the horizontal translation of the hands relative to the head, while keeping the global vertical translation as a crucial feature for encoding motion priors: $\mathbf{p}_{\text{SN},h}^{t_i,\text{hand}} = \mathbf{p}_{W,h}^{t_i,\text{hand}} - \mathbf{p}_{W,h}^{t_i,\text{head}}$
    - **Design Motivation**: Purely global representations lead to position overfitting, while purely local ones (using the head as reference) lose information and are sensitive to head rotation. GMD smartly balances both by retaining vertical displacement information.

3. **SlowFast Feature Fusion Module**: Inspired by the SlowFast network for video recognition, given an input window of $\tau$ frames, the FAST branch takes the most recent $\tau/2$ frames to maintain high temporal resolution, while the SLOW branch samples the entire window with a stride of 2 to obtain $\tau/2$ frames for capturing long-range context. After concatenation, the features are fed into the Transformer. The input sequence length is cut in half, covering double the time span without increasing computational overhead (Transformer self-attention complexity is $O(n^2)$).

4. **Body-Shape-Aware Pose Optimization**: Two solutions are proposed to resolve body shape discrepancies:

    - **Option 1**: Data augmentation + T-pose calibration—augment training data with real body shape parameters, and scale the output using height and arm-length scale factors during inference.
    - **Option 2** (Calibration-free): Jointly estimate pose and body shape parameters $\beta$. Optimize $\beta$ indirectly via forward kinematics of a differentiable SMPL model: $\mathcal{L}_{\text{pos}} = \|\text{FK}(\theta, \beta) - \text{FK}(\theta_{GT}, \beta_{GT})\|_1$, with L1 regularization imposed on $\beta$ to encourage sparsity.

### Loss & Training

The total loss function is a weighted sum of four L1 losses:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{ori}} \mathcal{L}_{\text{ori}} + \lambda_{\text{rot}} \mathcal{L}_{\text{rot}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}} + \lambda_{\beta} \|\beta\|_1$$

where the weights are $\lambda_{\text{ori}}=0.05$, $\lambda_{\text{rot}}=1$, $\lambda_{\text{pos}}=1$, and $\lambda_{\beta}=0.01$, respectively. An Adam optimizer is used with a batch size of 256, initial learning rate of $1\times10^{-4}$ decaying by 0.5 every $2\times10^4$ iterations, trained on a single RTX 3090 GPU.

## Key Experimental Results

### Main Results

Performance compared with SOTA methods on the large-scale real-world HPS dataset (all models are trained on CMU/BMLrub/HDM05 subsets of AMASS):

| Scenario | Metric | EgoPoser | AvatarPoser | AGRoL | AvatarJLM |
|------|------|----------|-------------|-------|-----------|
| BIB_EG_Tour | MPJPE(cm) | **9.55** | 22.53 | 28.95 | 41.27 |
| BIB_EG_Tour | MPJVE(cm/s) | **49.39** | 60.25 | 166.34 | 82.92 |
| MPI_EG | MPJPE(cm) | **11.05** | 16.54 | 19.41 | 12.91 |
| Working_Standing | MPJPE(cm) | **8.70** | 19.08 | 17.67 | 17.26 |
| Go_Around | MPJPE(cm) | **6.90** | 19.50 | 14.16 | 11.57 |

### Ablation Study

**Ablation of Global Motion Decomposition** (on AMASS dataset):

| Configuration | MPJPE(cm) | MPJVE(cm/s) | Note |
|------|-----------|-------------|------|
| Mean normalization of all features | 6.25 | 42.69 | Large information loss due to removing the mean |
| Spatial normalization (horizontal displacement) | 4.45 | 27.56 | Keeping vertical information is helpful |
| Temporal normalization only | 4.58 | 28.01 | Extracting relative trajectories |
| **Full GMD (Temporal + Spatial)** | **4.14** | **25.95** | Best performance when both complement each other |

**Ablation of SlowFast Design**:

| Configuration | MPJPE(cm) | MPJVE(cm/s) | FLOPs | Params |
|------|-----------|-------------|-------|--------|
| 40-frame input | 4.36 | 28.12 | 0.33G | 4.12M |
| 80-frame input | 4.11 | 29.27 | 0.65G | 4.12M |
| 80-frame downsampled by 2x | 4.13 | 30.02 | 0.33G | 4.12M |
| **SlowFast Fusion** | **4.14** | **25.95** | **0.33G** | **4.12M** |

**Comparison of Hand Visibility Strategies under Different FoVs**:

| Strategy | FoV=180° MPJPE | FoV=120° MPJPE | FoV=90° MPJPE |
|------|---------------|---------------|--------------|
| Assume fully visible | 24.75 | 38.99 | 41.24 |
| Random occlusion (FLAG) | 7.09 | 13.29 | 14.84 |
| Improved random occlusion | 6.52 | 11.88 | 12.83 |
| **Realistic FoV Modeling** | **5.31** | **6.07** | **6.60** |

### Key Findings

- The overfitting issue caused by global input representations is very severe: shifting just 5 meters away from the origin causes AvatarPoser's MPJPE to soar from ~5cm to 25cm+.
- Preserving vertical displacement information is crucial for pose estimation (evident from spatial normalization horizontal vs full).
- SlowFast achieves a significantly lower velocity error (MPJVE) with the same computational cost (25.95 vs 28-30 cm/s).
- Body shape estimation reduces MPJPE from 6.36cm (mean body shape) to 4.79cm, and ground penetration distance from 3.87cm to 2.31cm.

## Highlights & Insights

- **Deep Insights into the Problem**: For the first time, the overfitting issue caused by global position representations is systematically revealed, providing valuable references for the entire field.
- **Simple Yet Effective Design**: The global motion decomposition strategy is simple and elegant, achieving position-agnostic representation purely through coordinate transformations.
- **High Engineering Value**: A running time speed of 600+ fps far exceeds similar methods. It has been validated on-device (Quest 2), exhibiting strong deployment feasibility.
- **Comprehensiveness**: Simultaneously addresses three practical problems: generalization in large scenes, hand occlusion, and body shape adaptation.

## Limitations & Future Work

- Assumes users move on the same level (since global vertical position is encoded); crossing floors requires resetting the origin.
- Uses a simple Transformer backbone; more sophisticated model architectures (e.g., joint-level modeling) could yield further improvements.
- Lacks a post-processing step (such as physical constraints, collision detection), which could further secure physical plausibility.
- Finger poses are not modeled (only 22 SMPL-H joints are estimated), limiting its usage in fine interaction scenarios.

## Related Work & Insights

- AvatarPoser (CVPR 2022): Established the baseline framework for full-body pose estimation from 3-point HMD tracking.
- AGRoL (CVPR 2023): A diffusion-model-based method yielding high-quality results, but with slow inference speed and dependence on future frames.
- FLAG: Proposed a random occlusion data augmentation method to handle hand invisibility, but did not consider spatial relationships.
- SlowFast Networks (ICCV 2019): The dual-rate design for video recognition is successfully adapted to temporal signal processing.

## Rating

- Novelty: ⭐⭐⭐⭐ Global motion decomposition is the most prominent contribution of this work, revealing and addressing global representation overfitting for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive experimental design with multiple datasets, diverse-method comparisons, detailed ablations, and on-device validation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic description of method, and excellent coordination between figures and text.
- Value: ⭐⭐⭐⭐⭐ Solves three core practical issues in MR pose estimation; the 600 fps inference speed makes it directly applicable for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Benchmarks and Challenges in Pose Estimation for Egocentric Hand Interactions with Objects](benchmarks_and_challenges_in_pose_estimation_for_egocentric_hand_interactions_wi.md)
- [\[CVPR 2026\] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](../../CVPR2026/video_understanding/egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)
- [\[ECCV 2024\] Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation](motion-prior_contrast_maximization_for_dense_continuous-time_motion_estimation.md)
- [\[ECCV 2024\] AMEGO: Active Memory from Long EGOcentric Videos](amego_active_memory_from_long_egocentric_videos.md)
- [\[ECCV 2024\] LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow](layeredflow_a_real-world_benchmark_for_non-lambertian_multi-layer_optical_flow.md)

</div>

<!-- RELATED:END -->

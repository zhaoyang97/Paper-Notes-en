---
title: >-
  [Paper Note] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span
description: >-
  [NeurIPS 2025][3D Vision][Egocentric Vision] This paper proposes EgoSpanLift, a method that lifts egocentric 2D gaze predictions into 3D space…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Egocentric Vision"
  - "Gaze Prediction"
  - "3D Visual Span"
  - "SLAM"
  - "Voxel Prediction"
date: 2026-05-08
content_hash: 0b264ababd7dea01
---

# Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span

**Conference**: NeurIPS 2025
**arXiv**: [2511.18470](https://arxiv.org/abs/2511.18470)
**Code**: [Available](https://hs-yn.github.io/GazeBeyondFrame/)
**Area**: 3D Vision
**Keywords**: Egocentric Vision, Gaze Prediction, 3D Visual Span, SLAM, Voxel Prediction

## TL;DR

This paper proposes EgoSpanLift, a method that lifts egocentric 2D gaze predictions into 3D space, constructing multi-level volumetric visual span representations. Combined with a 3D U-Net and a causal Transformer, the framework forecasts future 3D regions of visual attention.

## Background & Motivation

People accomplish daily activities through continuous perception and interaction with their environment, with visual perception serving as the foundation guiding human behavior — "look before you act." Existing egocentric understanding research focuses primarily on action prediction and contact-based interaction, leaving **prediction of visual perception itself** largely unexplored.

Limitations of existing 2D gaze prediction:

**Ambiguous definition in dynamic scenes**: Modeling both the user's self-motion and attention simultaneously is required, yet both are naturally expressed in 3D space.

**Information loss from 2D projection**: Gaze direction and self-motion inherently point to specific locations in 3D space, not arbitrary regions on a 2D image plane.

**Inability to predict beyond the current field of view**: Users may turn their heads to attend to regions outside the current frame.

This paper introduces the novel problem of **egocentric 3D visual span forecasting**: predicting where in the 3D environment a user's visual attention will be directed in the future.

## Method

### Overall Architecture

The system consists of three components:
1. **EgoSpanLift**: Lifts 2D gaze from the image plane into volumetric regions in the 3D scene.
2. **Prediction Network**: A 3D U-Net (spatial encoder) combined with a causal Transformer (temporal modeling) for future visual span prediction.
3. **Benchmark**: A 364.6K-sample benchmark curated from raw egocentric multi-sensor data.

### Key Designs

#### 1. EgoSpanLift — Lifting Gaze from 2D to 3D

**Inputs**: Semi-dense 3D keypoints $\mathcal{P}$ from SLAM (each containing position $\mathbf{p}_i \in \mathbb{R}^3$, confidence $\sigma_i$, and observation timestamp $t_i$), and localization information $\mathcal{E}$ (SE(3) transformation matrices $\mathbf{E}_t$).

**Observation-based keypoint selection**:

$$\mathcal{P}_t = \{p_i \in \mathcal{P} \mid t_i = t, \|\mathbf{p}_i - \mathbf{t}_t\|_1 < D/2, \mathcal{I}_f(p_i; \mathcal{P}_t) = 1\}$$

Triple filtering: temporal window filtering (retaining only the most recent seconds), spatial filtering (within a $D=3.2$m cube), and statistical outlier filtering (preserving dynamic objects such as hands and moving people while removing invalid points).

**Gaze-based keypoint classification**: After transforming keypoints into the local coordinate frame, a 3D gaze cone determines whether each point falls within the visual span:

$$Q_t^{\theta, \mathbf{g}_t} = \left\{p_i \in \mathcal{P}_t \mid \frac{\langle \mathbf{E}_t^{-1}\mathbf{p}_i, \mathbf{g}_t \rangle}{\|\mathbf{E}_t^{-1}\mathbf{p}_i\| \|\mathbf{g}_t\|} > \cos\theta \right\}$$

where $\theta$ is the eccentricity angle threshold and $\mathbf{g}_t$ is the gaze direction.

#### 2. Multi-Level Volumetric Region Localization

Inspired by the visual science literature, four hierarchical levels of visual span are defined:

| Level | Eccentricity $\theta$ | Meaning |
|------|:--------------:|------|
| **Foveal** | 2° | Region targeted by conventional 2D gaze |
| **Central** | 8° | Compensates for sparse coverage of semi-dense keypoints |
| **Near Peripheral** | 30° | Broader peripheral perception range |
| **Orientation** | 55° | Field of view centered on head orientation |

Classified keypoints are voxelized into a 3D grid (resolution $R$, side length $D$), with binary occupancy computed as:

$$V_{[t_b,t_e]}^{\theta,\mathbf{g}_t}(i,j,k) = \mathcal{I}(|\{p_i \in \cup_{t} Q_t^{\theta,\mathbf{g}_t} \mid \text{falls in corresponding voxel}\}| > 0)$$

#### 3. Prediction Network

**Autoregressive Encoder**:
- Input: $T_p \times (4+1) \times R \times R \times R$ voxel grids (4 visual span levels + full scene)
- A 3D U-Net encoder compresses the spatial dimensions (reduction factor $R$), yielding $T_p \times C$ temporal features
- A global embedding is appended as a prediction token, forming a $(T_p+1) \times C$ feature sequence
- A **causal Transformer** learns temporal dependencies, ensuring information flows toward the final global embedding

**Decoder**:
- The output embedding is upsampled via the U-Net decoder
- Residual connections from intermediate encoder features are incorporated
- Sigmoid activation produces a $4 \times R \times R \times R$ soft occupancy map in $[0,1]$

### Loss & Training

Since visual spans occupy only a tiny fraction of the total space (foveal span < 1%), standard cross-entropy fails to learn meaningful signals. **Dice Loss** is adopted:

$$\mathcal{L} = 1 - \frac{2 \times \sum \tilde{Y}_{ijk} \odot Y_{ijk}}{\sum \tilde{Y}_{ijk} + \sum Y_{ijk} + 1}$$

Joint multi-level training outperforms single-task training, with particularly significant improvements in foveal span prediction.

## Key Experimental Results

### Main Results

**FoVS-Aria** test set (daily activities, 23.2K samples):

| Method | Orientation IoU | Peripheral IoU | Central IoU | Foveal IoU |
|------|:--------------:|:-------------:|:-----------:|:----------:|
| CSTS + EgoSpanLift | — | 0.457 | 0.234 | 0.139 |
| EgoChoir | 0.496 | 0.430 | 0.261 | 0.199 |
| **Ours (full)** | **0.584** | **0.489** | **0.351** | **0.284** |

**FoVS-EgoExo** test set (skilled activities, 341.4K samples):

| Method | Orientation IoU | Peripheral IoU | Central IoU | Foveal IoU |
|------|:--------------:|:-------------:|:-----------:|:----------:|
| CSTS + EgoSpanLift | — | 0.498 | 0.287 | 0.156 |
| EgoChoir | 0.329 | 0.285 | 0.198 | 0.127 |
| **Ours (full)** | **0.523** | **0.511** | **0.421** | **0.369** |

3D foveal localization error (distance distribution):

| Method | Min (cm) | Mean (cm) | Max (cm) |
|------|:-------:|:-------:|:-------:|
| CSTS | 59.71 | 73.79 | 87.68 |
| **Ours** | **19.04** | **34.85** | **51.23** |

### Ablation Study

| Configuration | Orientation IoU | Central IoU | Foveal IoU |
|------|:--------------:|:-----------:|:----------:|
| w/o prior spans | 0.342 | 0.107 | 0.059 |
| BCE loss | 0.573 | 0.284 | 0.206 |
| Single-task training | 0.583 | 0.335 | 0.249 |
| w/o global embedding | 0.560 | 0.324 | 0.262 |
| **Full model** | **0.584** | **0.351** | **0.284** |

### Key Findings

1. The proposed method substantially outperforms all baselines across all levels, exceeding prior work by 50%+ on foveal span prediction.
2. Prior visual span information is critical — removing it causes a dramatic performance drop.
3. Joint multi-level training significantly outperforms single-task training, exploiting mutual cues between peripheral and foveal attention.
4. Back-projecting 3D predictions to 2D (without any 2D-specific training) matches the performance of dedicated 2D gaze methods.
5. Inference latency is only 71.2 ms, satisfying real-time requirements.

## Highlights & Insights

- **Novel problem formulation**: The first work to formally define egocentric 3D visual span forecasting.
- **Visual science grounding**: The multi-level visual span hierarchy is derived from established visual science taxonomy.
- **Practical efficiency**: Built on SLAM keypoints (not dense reconstruction), with low latency (71 ms), making it suitable for AR/VR applications.
- **Bidirectional validation**: Effectiveness in both 2D→3D and 3D→2D directions demonstrates the superiority of 3D modeling.
- **Large-scale benchmark**: A 364.6K-sample evaluation platform is curated and released.

## Limitations & Future Work

1. Performance depends on the quality of semi-dense SLAM keypoints, which may be insufficient in certain scenes.
2. Absolute foveal span IoU remains moderate (~0.28–0.37), leaving room for improvement in fine-grained prediction.
3. Large-scale dynamic scenes (e.g., soccer, basketball) are excluded, limiting generalizability.
4. Voxelization is the primary computational bottleneck (45 ms) and requires optimization for more constrained devices.
5. Non-visual modalities (auditory, proprioceptive) are not incorporated, potentially missing important cues.

## Related Work & Insights

- **CSTS**: Current state-of-the-art 2D gaze prediction method based on multimodal contrastive spatiotemporal fusion.
- **EgoChoir**: Predicts 3D interaction hotspots from synthetic geometry and motion; serves as the strongest baseline.
- **Ego-Exo4D / Aria Everyday Activities**: Sources of training and evaluation data.
- Insight: Lifting 2D tasks into 3D is a general and effective strategy that may extend to other egocentric prediction tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Defines a novel problem and proposes a systematic solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets (364.6K samples), multiple baselines, comprehensive ablations, and 2D back-projection validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with natural integration of visual science background.
- Value: ⭐⭐⭐⭐ Direct applicability to AR/VR and assistive technologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation](../../ICCV2025/3d_vision/uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](../../ICCV2025/3d_vision/benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[NeurIPS 2025\] IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants](indego_a_dataset_of_industrial_scenarios_and_collaborative_work_for_egocentric_a.md)
- [\[NeurIPS 2025\] 3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)
- [\[NeurIPS 2025\] Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting](mesh_interpolation_graph_network_for_dynamic_and_spatially_irregular_global_weat.md)

</div>

<!-- RELATED:END -->

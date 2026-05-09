---
title: >-
  [Paper Note] TAPIP3D: Tracking Any Point in Persistent 3D Geometry
description: >-
  [NeurIPS 2025][3D Vision][3D point tracking] This paper proposes TAPIP3D, which represents video as a camera-stabilized spatiotemporal 3D feature point cloud and iteratively refines multi-frame point trajectories in persistent 3D geometric space via a 3D Neighborhood-to-Neighborhood (N2N) attention mechanism, substantially outperforming existing 3D point tracking methods.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D point tracking
  - feature point cloud
  - world coordinate system
  - neighborhood attention
  - camera motion elimination
date: 2026-05-08
content_hash: 516e6954b9a44070
---

# TAPIP3D: Tracking Any Point in Persistent 3D Geometry

**Conference**: NeurIPS 2025
**arXiv**: [2504.14717](https://arxiv.org/abs/2504.14717)
**Code**: [Project Page](https://tapip3d.github.io/)
**Area**: 3D Vision
**Keywords**: 3D point tracking, feature point cloud, world coordinate system, neighborhood attention, camera motion elimination

## TL;DR

This paper proposes TAPIP3D, which represents video as a camera-stabilized spatiotemporal 3D feature point cloud and iteratively refines multi-frame point trajectories in persistent 3D geometric space via a 3D Neighborhood-to-Neighborhood (N2N) attention mechanism, substantially outperforming existing 3D point tracking methods.

## Background & Motivation

Long-term point tracking in video is an important tool for robotics and action recognition. Particle-level motion estimation provides a unified framework for capturing object pose changes, articulated motion, and deformable structures. However, existing methods exhibit critical limitations:

**Fundamental problems with 2D tracking**: Most point trackers operate in pixel space (e.g., the CoTracker series) or pixel space augmented with depth (e.g., SpatialTracker, DELTA). However, most apparent motion in video originates from camera movement rather than object motion; tracking in 2D space conflates the two, increasing tracking difficulty.

**Limitations of UVD coordinates**: Existing 3D trackers (SpatialTracker, DELTA) use UVD coordinates (pixel coordinates + depth), a 2.5D representation that does not genuinely exploit 3D geometric structure. Under large camera motion, trajectories in UVD space become complex and irregular.

**Limitations of feature extraction**: SpatialTracker characterizes 3D point clouds via triplane projection—fast but at the cost of geometric information. DELTA simply stacks depth as an extra channel onto 2D correlation maps, remaining essentially an extension of 2D methods.

The paper's core insight is: by lifting video into a world-coordinate 3D feature point cloud using depth and camera poses, camera motion is effectively "eliminated," leaving only the true 3D motion of scene objects, making trajectories smoother and easier to track.

## Method

### Overall Architecture

TAPIP3D takes an RGB-D video as input (depth from sensors, estimators, or GT) and outputs 3D trajectories for query points. The main pipeline: (1) lift per-frame video features into a 3D feature point cloud; (2) optionally transform to a world coordinate system using camera poses to eliminate camera motion; (3) extract local 3D context from the feature point cloud via 3D N2N attention; (4) iteratively refine trajectory estimates via a Transformer.

### Key Designs

1. **3D feature point cloud video representation**: Traditional 2D feature maps are "augmented" into a representation with 3D coordinates—each 2D cell stores both a $C$-dimensional feature vector and its corresponding 3D coordinate $(X,Y,Z)$ obtained by back-projecting depth. Multi-scale features are constructed as:
    $\mathcal{F}_l = \{\mathbf{F}^{l,t} \in \mathbb{R}^{\frac{H}{\ell 2^{l-1}} \times \frac{W}{\ell 2^{l-1}} \times (3+C)}\}_{t=1}^T$
   Critically, coordinates are downsampled via nearest-neighbor interpolation while features use average pooling, ensuring 3D geometry is not blurred.

2. **3D Neighborhood-to-Neighborhood (N2N) attention**: This is the core innovation. Replacing conventional 2D patch correlation windows, 3D k-NN is used to establish neighborhood relationships based on true 3D distances.

    - **Support tokens**: For each query point, $K$ 3D nearest neighbors are found in the feature point cloud of its initial frame, forming a "support" group that captures the local shape context of the query point.
    - **Context tokens**: At each timestep, $K$ 3D nearest neighbors of the current trajectory estimate $\tau_q^t$ are found in that frame's feature point cloud, forming a "context" group.
    - **Bidirectional cross-attention**: Bidirectional cross-attention is applied between support and context groups, then compressed into a per-query, per-timestep summary vector via attention pooling.
    - Positional encodings based on 3D relative offsets are incorporated into the cross-attention to enhance spatial awareness.
    - Executed in parallel across all scales to obtain multi-scale neighborhood features $\mathcal{N}_q^t$.

3. **3D trajectory update Transformer**: N2N attention features are combined with trajectory information into tokens:
    $G_q^t = [\mathcal{N}_q^t, \gamma(\tau_q^t - \tau_q^{t-1}), \gamma(\tau_q^{t+1} - \tau_q^t), \gamma(\pi_t(\tau_q^t)), o_q^t, \gamma(t)]$
   The 2D projection coordinate $\pi_t(\tau_q^t)$ is explicitly included to help the model identify points outside the image boundary. A Transformer with proxy tokens handles spatiotemporal attention, outputting incremental updates to position and visibility.

4. **World-coordinate tracking vs. camera-coordinate tracking**:

    - TAPIP3D-camera: tracks in camera coordinate space (no camera poses required).
    - TAPIP3D-world: transforms point clouds from all frames into the first frame's camera coordinate system (i.e., "world" coordinates) using camera poses, eliminating camera motion.
    - Both variants share the same trained weights; only the coordinate system differs at inference.

### Loss & Training

The training loss consists of a depth-adaptive position loss and a visibility cross-entropy loss:
$$\mathcal{L} = \sum_{q=1}^Q \sum_{t=1}^T \frac{1}{d_q^t} \|\tau_q^t - \tilde{\tau}_q^t\|_2 + \alpha_{vis} \text{CE}(o_q^t, \tilde{o}_q^t)$$

The $1/d_q^t$ depth scaling reduces the loss contribution of distant points, preventing imprecise far-field depth from causing gradient instability. Training uses 4 iterative refinement steps, each supervised (with a discount factor $\gamma=0.8$ decreasing the weight of earlier iterations). Training data is the Kubric MOVi-F synthetic dataset; training runs for 200K iterations on 8× L40S GPUs over approximately 4.2 days.

## Key Experimental Results

### Main Results

**TAPVid-3D real-world benchmark (depth and poses estimated via MegaSaM):**

| Method | ADT AJ↑ | ADT APD↑ | DriveTrack AJ↑ | PStudio AJ↑ | Mean AJ↑ |
|--------|---------|---------|----------------|-------------|---------|
| CoTracker3 + M-SaM | 20.4 | 30.1 | 14.1 | 17.4 | 17.3 |
| SpatialTracker + M-SaM | 15.9 | 23.8 | 7.7 | 15.3 | 13.0 |
| DELTA + M-SaM | 21.0 | 29.3 | 14.6 | 17.7 | 17.8 |
| TAPIP3D-camera + M-SaM | 21.6 | 31.0 | 14.6 | 18.1 | 18.1 |
| **TAPIP3D-world + M-SaM** | **23.5** | **32.8** | **14.9** | **18.1** | **18.8** |

**LSFOdyssey synthetic benchmark (GT Depth comparison):**

| Method | AJ3D↑ | APD3D↑ | AJ2D↑ |
|--------|-------|--------|-------|
| DELTA + GT | 37.7 | 50.1 | 72.4 |
| TAPIP3D-camera + GT | 68.3 | 83.2 | 76.0 |
| **TAPIP3D-world + GT** | **72.2** | **85.8** | **78.5** |

### Ablation Study

| Configuration | LSFOdyssey AJ3D↑ | APD3D↑ | Note |
|---------------|------------------|--------|------|
| UV+D coordinates | 63.4 | 77.0 | Commonly used by prior methods |
| UV+log(D) coordinates | 62.9 | 77.9 | Log-depth offers no clear benefit |
| XYZ (camera) coordinates | 67.1 | 81.6 | Significantly better than UVD |
| **XYZ (world) coordinates** | **70.7** | **84.1** | Best |
| Camera coords, w/o N2N attention | 59.4 | 72.7 | Baseline |
| **Camera coords, w/ N2N attention** | **67.1** | **81.6** | Large gain from N2N |
| World coords, w/o N2N attention | 62.1 | 75.1 | World coords alone are beneficial |
| **World coords, w/ N2N attention** | **70.7** | **84.1** | Combined effect is best |

### Key Findings

- On synthetic data with GT depth, TAPIP3D-world's AJ3D is nearly twice that of DELTA (72.2 vs. 37.7), demonstrating the substantial advantage of the 3D feature point cloud representation.
- As depth quality improves (MegaSaM → GT), TAPIP3D's performance gain far exceeds that of baseline methods, indicating the model can more effectively exploit high-quality depth.
- 3D k-NN improves AJ3D on DexYCB-Pt from 27.7 to 29.8 compared to fixed 2D neighborhoods.
- The advantage of world-coordinate tracking is most pronounced on the ADT dataset (large camera motion): 23.5 vs. 21.6.
- Inference speed is 10 FPS with approximately 2.6 GB VRAM (tracking 1024 query points over 32 frames).

## Highlights & Insights

- **Pioneer of world-coordinate tracking**: TAPIP3D is the first method capable of 3D point tracking in a world coordinate system where camera motion is eliminated, leveraging recent advances in monocular depth estimation and camera pose estimation.
- **Elegant 3D N2N attention design**: Support groups capture the local shape of query points; context groups capture the target region's context; bidirectional cross-attention fuses both, effectively resolving matching ambiguities.
- **Memory-efficient training strategy**: Detaching gradients and back-propagating immediately after each iteration reduces memory consumption from 48 GB+ to approximately 20 GB without sacrificing performance.
- **Ecology-friendly generalization**: Training exclusively on synthetic data achieves state-of-the-art performance on real-world benchmarks.

## Limitations & Future Work

- Performance is significantly affected by depth map quality; depth flickering may occur in scenes with extreme depth variation or small, distant, blurry objects.
- When high-quality depth is unavailable, 2D metrics may be weaker than those of pure 2D or UVD-space trackers, since 2D trajectories are obtained by projecting 3D estimates and thus depend on geometric consistency.
- Depth quality issues can be mitigated through preprocessing with depth completion and noise filtering.
- As 3D visual reconstruction models advance, the robustness of TAPIP3D will improve naturally.

## Related Work & Insights

TAPIP3D builds on the iterative refinement framework of CoTracker3, replacing 2D feature maps with 3D feature point clouds and 2D correlation windows with N2N attention. Compared to SpatialTracker's triplane representation, directly using 3D feature point clouds encodes geometric structure more explicitly; although slightly slower, accuracy improves substantially. The paper's N2N attention generalizes the region-to-region matching idea from LocoTrack's 2D CNN to cross-attention over 3D feature point clouds.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to propose world-coordinate 3D point tracking with an elegant N2N attention design
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 benchmarks (real/synthetic, multiple depth sources) with detailed ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear; figures intuitively contrast UVD and XYZ trajectory differences
- Value: ⭐⭐⭐⭐⭐ Achieves a qualitative leap by leveraging advances in depth estimation, establishing a new paradigm for 3D point tracking

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Online Segment Any 3D Thing as Instance Tracking](online_segment_any_3d_thing_as_instance_tracking.md)
- [\[ICCV 2025\] TAPNext: Tracking Any Point (TAP) as Next Token Prediction](../../ICCV2025/3d_vision/tapnext_tracking_any_point_tap_as_next_token_prediction.md)
- [\[ICCV 2025\] Multi-View 3D Point Tracking](../../ICCV2025/3d_vision/multi-view_3d_point_tracking.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](../../ICCV2025/3d_vision/alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)

<!-- RELATED:END -->

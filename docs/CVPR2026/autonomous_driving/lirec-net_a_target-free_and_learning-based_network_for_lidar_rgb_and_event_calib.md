---
title: >-
  [Paper Note] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration
description: >-
  [CVPR 2026][Autonomous Driving][multi-sensor calibration] This paper proposes LiREC-Net, the first unified framework for simultaneously performing target-free extrinsic calibration between LiDAR–RGB and LiDAR–Event camera pairs. Through a shared LiDAR representation that fuses 3D point features with projected depth features, and pairwise cost volumes for cross-modal alignment, LiREC-Net achieves calibration accuracies of 1.80 cm/0.11° on KITTI, and 2.51 cm/0.14° (LiDAR–RGB) and 1.18 cm/0.07° (LiDAR–Event) on DSEC.
tags:
  - CVPR 2026
  - Autonomous Driving
  - multi-sensor calibration
  - target-free calibration
  - tri-modal fusion
  - event camera
  - extrinsic estimation
date: 2026-05-08
content_hash: b17963e49dbdc527
---

# LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration

**Conference**: CVPR 2026
**arXiv**: [2602.21754](https://arxiv.org/abs/2602.21754)
**Code**: Unavailable
**Area**: Autonomous Driving
**Keywords**: multi-sensor calibration, target-free calibration, tri-modal fusion, event camera, extrinsic estimation

## TL;DR

This paper proposes LiREC-Net, the first unified framework for simultaneously performing target-free extrinsic calibration between LiDAR–RGB and LiDAR–Event camera pairs. Through a shared LiDAR representation that fuses 3D point features with projected depth features, and pairwise cost volumes for cross-modal alignment, LiREC-Net achieves calibration accuracies of 1.80 cm/0.11° on KITTI, and 2.51 cm/0.14° (LiDAR–RGB) and 1.18 cm/0.07° (LiDAR–Event) on DSEC.

## Background & Motivation

Autonomous driving systems rely on multi-sensor fusion to construct a consistent perception of the environment. Accurate **extrinsic calibration**—knowing the relative pose of each sensor in a common coordinate frame—is a prerequisite for sensor fusion. In practice, however, vehicle vibrations, temperature changes, minor collisions, and routine maintenance cause gradual drift in sensor poses, invalidating the initial calibration.

Traditional **target-based calibration** methods (checkerboards, ArUco markers, etc.) achieve high accuracy but require controlled scenes, careful target placement, repeated data collection, and manual supervision, making them **impractical for frequent in-operation recalibration**. **Target-free methods** calibrate directly from natural driving scenes without any special setup and can be applied at any time.

Learning-based target-free calibration methods (LCCNet, RegNet, CalibNet, etc.) have made notable progress, but share a critical limitation: **they handle only a single modality pair**. LCCNet, for example, addresses only LiDAR–RGB, while MULiEv handles only LiDAR–Event. When a system is equipped with all three sensor types, **two independent networks must be trained separately**, introducing computational redundancy and potentially causing **calibration inconsistency**—the LiDAR–RGB and LiDAR–Event transformations estimated by two independent networks may not be mutually consistent in 3D space.

The core idea of LiREC-Net is to **design a shared LiDAR representation that serves both calibration pathways simultaneously**, reducing redundancy and ensuring consistency.

## Method

### Overall Architecture

LiREC-Net adopts a dual-pathway architecture consisting of:

1. **Shared LiDAR branch**: extracts a unified representation from the point cloud
2. **RGB encoder** + **Event encoder**: extract visual and event features, respectively
3. **Two pairwise cost volumes**: one for LiDAR–RGB and one for LiDAR–Event
4. **Two context modules + prediction heads**: output extrinsic parameters for each pair

### Key Designs

1. **Shared LiDAR Representation (Point + Depth Fusion)**: LiDAR features are extracted in parallel through **two complementary encoders**:

    - **Point encoder**: employs Point-Transformer-V3 (PTV3) to directly process unordered 3D points, serialized via space-filling curves for efficient local attention, capturing **fine-grained geometric structure**
    - **Depth encoder**: projects the point cloud onto the image plane to produce a single-channel depth map, from which MViTV2 extracts **dense spatial context**

   The two feature streams are aligned to the same resolution via **Scaled Feature Projection (SFP)** and then concatenated:

    $\mathbf{F}^{\text{Li}} = \text{Concat}(\text{SFP}(\mathbf{F}^{\text{point}}), \mathbf{F}^{\text{depth}})$

   Ablation experiments confirm that neither feature stream is dispensable—using point features alone causes the LiDAR–RGB translation error to surge from 2.51 cm to 14.43 cm.

2. **Scaled Depth Projection (SDP) and Scaled Feature Projection (SFP)**: A critical engineering detail. When projecting a point cloud onto the image plane, the conventional approach projects with the original intrinsics and then resizes. However, resizing introduces **blurring artifacts** that disrupt fine-grained feature alignment. SDP avoids this by scaling the intrinsic matrix before projection:

    $R' = \text{diag}\left(\frac{W'}{W}, \frac{H'}{H}, 1\right), \quad \mathbf{K'}_{\text{Cam}} = R' \mathbf{K}_{\text{Cam}}$

   Ablations show that removing both SDP and SFP degrades LiDAR–Event error from 1.18 cm/0.07° to 3.35 cm/0.30°.

3. **Pairwise Cost Volume**: Inspired by PWC-Net and LCCNet, LiREC-Net computes pixel-wise local correlations between LiDAR and camera features. For pixel $\mathbf{p}=(x,y)$ and displacement $(\Delta x, \Delta y)$:

    $\mathcal{C}(y,x,\Delta x,\Delta y) = \frac{1}{C}\sum_{c=1}^{C} \mathbf{F}^{\text{Li}}_{c,y,x} \cdot \mathbf{F}^{\text{Cam}}_{c,y+\Delta y,x+\Delta x}$

   The cost volume has dimension $H'' \times W'' \times (2d+1)^2$, measuring cross-modal local similarity via channel-wise inner products within a sliding window.

4. **Iterative Refinement**: Multiple independent models are trained, each targeting a different error range in a coarse-to-fine manner (from large to small: ±20°/150 cm → ±1°/10 cm), and applied in cascade at evaluation time:

    $\hat{\mathbf{T}}^{v,(k)} = \Delta\hat{\mathbf{T}}^{v,(k)} \hat{\mathbf{T}}^{v,(k-1)}$

   This coarse-to-fine strategy allows each stage to focus on corrections within a specific precision range.

### Loss & Training

The total loss is the sum of losses from both modality pairs: $\mathcal{L}_{\text{total}} = \mathcal{L}^{\text{Li-RGB}} + \mathcal{L}^{\text{Li-Ev}}$

Each pairwise loss comprises three terms:

$$\mathcal{L}^v = (1-w)(\lambda_t \mathcal{L}^v_{\text{trans}} + \lambda_r \mathcal{L}^v_{\text{rot}}) + w \mathcal{L}^v_{\text{pcd}}$$

- **Translation loss**: Smooth L1 loss on $\hat{\mathbf{t}}^v$
- **Rotation loss**: angular distance between predicted and ground-truth quaternions $\theta(\hat{\mathbf{q}}^v, \mathbf{q}^v)$
- **Point cloud distance loss**: L2 distance between the transformed point cloud and the ground-truth transformed point cloud, enforcing geometric consistency

Training details: Adam optimizer, lr = 3e-4, milestone decay (×0.5); first stage on DSEC for 150 epochs, subsequent stages for 70 epochs; batch size 64, 4× A6000/L40S GPUs.

## Key Experimental Results

### Main Results — KITTI Dataset

| Method | LiDAR–RGB Error | LiDAR–Event Error |
|--------|----------------|------------------|
| RegNet | 6.00 cm / 0.28° | — |
| CalibNet | 4.34 cm / 0.41° | — |
| LCCNet | 1.59 cm / 0.16° | — |
| PseudoCal | **1.18 cm / 0.05°** | — |
| **LiREC-Net** | 1.80 cm / 0.11° | **1.82 cm / 0.12°** |

### Main Results — DSEC Dataset (5 stages)

| Method | LiDAR–RGB Error | LiDAR–Event Error |
|--------|----------------|------------------|
| MULiEv (2 stages) | — | 0.81 cm / 0.10° |
| LiREC-Net (2 stages) | 2.62 cm / 0.30° | 2.05 cm / 0.25° |
| **LiREC-Net (5 stages)** | **2.51 cm / 0.14°** | **1.18 cm / 0.07°** |

### Ablation Study — Feature Fusion (DSEC)

| Point Features | Depth Features | LiDAR–RGB | LiDAR–Event |
|---------------|---------------|-----------|-------------|
| ✓ | ✗ | 14.43 cm / 0.70° | 14.05 cm / 0.64° |
| ✗ | ✓ | 2.97 cm / 0.70° | 2.16 cm / 0.60° |
| ✓ | ✓ | **2.51 cm / 0.14°** | **1.18 cm / 0.07°** |

### Tri-modal vs. Bi-modal Efficiency Comparison

| Configuration | Inference Time (s) | Parameters (B) | Memory (GiB) |
|---------------|--------------------|---------------|--------------|
| Bi-modal (KITTI, two separate networks) | 0.51 | 1.9 | 14.6 |
| **Tri-modal (KITTI, unified network)** | **0.33** | **1.7** | **11.1** |
| Bi-modal (DSEC) | 0.44 | 1.9 | 14.6 |
| **Tri-modal (DSEC)** | **0.31** | **1.7** | **11.1** |

### Key Findings

- **Point and depth features are mutually indispensable**: Using only point features causes a 5.8× increase in translation error (2.51 → 14.43 cm) due to the lack of dense spatial context; using only depth features yields a 5× increase in rotation error (0.14 → 0.70°) due to the loss of fine-grained 3D structure.
- **Tri-modal outperforms bi-modal**: The unified model achieves comparable or superior accuracy (the shared LiDAR representation provides implicit regularization) while improving efficiency (35% faster inference, 24% less memory).
- **Projection accuracy via SDP and SFP is critical**: The two scaled projection modules contribute complementary accuracy improvements.
- **MViTV2 > ResNet**: The global feature modeling capability of Transformers proves more effective for cross-modal alignment.

## Highlights & Insights

- **The tri-modal unification paradigm has practical value**: As event cameras become increasingly prevalent in autonomous driving (e.g., Prophesee sensors), unified multi-sensor calibration is a genuine operational need.
- **The shared LiDAR representation provides both efficiency and accuracy benefits**: It eliminates the inconsistency that would arise from two independent LiDAR encoders.
- **The SDP/SFP engineering insight is broadly applicable**: Blurring artifacts introduced by naive resizing are a critical bottleneck in fine-grained alignment tasks.
- LiREC-Net establishes the first LiDAR–RGB calibration baseline on DSEC, as no prior method had reported such results on this dataset.

## Limitations & Future Work

- **Assumes a pre-calibrated RGB–Event pair**: i.e., $\mathbf{T}^{\text{Ev} \to \text{RGB}}$ is assumed to be known. The authors note that joint estimation of this transformation within the framework could be explored.
- The method is limited to LiDAR, RGB, and event modalities and does not extend to thermal imaging, millimeter-wave radar, or other sensor types.
- Iterative refinement requires training a separate model per error range (5 stages = 5 models), incurring high training costs.
- Point clouds are sampled to a fixed number of points ($N$ = 20,000 for KITTI, 5,000 for DSEC), lacking flexibility.
- The effect of the cost volume sliding window size $d$ on different error ranges has not been thoroughly analyzed.

## Related Work & Insights

- LiREC-Net is directly inspired by LCCNet, extending its core ideas (cost volume + iterative refinement) from bi-modal to tri-modal calibration.
- The choice of Point-Transformer-V3 as the point cloud encoder is noteworthy—space-filling curve serialization makes it considerably more efficient than PointNet++.
- MULiEv is the only prior work on LiDAR–Event calibration, but addresses only a single modality pair.
- Unlike CalibNet's self-supervised calibration paradigm, LiREC-Net employs supervised regression and relies on artificially perturbed pseudo-misalignments for training data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Tri-modal unified calibration is a novel and meaningful problem formulation; the shared LiDAR representation is a creative design choice.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluations on two datasets with comprehensive ablations and efficiency analysis; however, validation on real-world misalignment (rather than artificial perturbations) is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rigorous formulation, and intuitive figures and tables.
- **Value**: ⭐⭐⭐⭐ Directly applicable to multi-sensor autonomous driving systems; the tri-modal baseline serves as a meaningful reference for future work.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)
- [\[CVPR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[CVPR 2026\] Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks](mind_the_hitch_dynamic_calibration_and_articulated_perception_for_autonomous_tru.md)
- [\[CVPR 2026\] FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)

</div>

<!-- RELATED:END -->

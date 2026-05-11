---
title: >-
  [Paper Note] AnthroTAP: Learning Point Tracking with Real-World Motion
description: >-
  [CVPR 2026][3D Vision][point tracking] AnthroTAP proposes an automated pipeline that generates large-scale pseudo-labeled point tracking data from real-world human motion videos via SMPL fitting and optical flow filterin…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "point tracking"
  - "human motion"
  - "pseudo-labels"
  - "SMPL"
  - "optical flow consistency"
date: 2026-05-08
content_hash: 33bb2c3807d1ebe2
---

# AnthroTAP: Learning Point Tracking with Real-World Motion

**Conference**: CVPR 2026
**arXiv**: [2507.06233](https://arxiv.org/abs/2507.06233)
**Code**: [Project Page](https://cvlab-kaist.github.io/AnthroTAP/)
**Area**: 3D Vision / Point Tracking
**Keywords**: point tracking, human motion, pseudo-labels, SMPL, optical flow consistency

## TL;DR
AnthroTAP proposes an automated pipeline that generates large-scale pseudo-labeled point tracking data from real-world human motion videos via SMPL fitting and optical flow filtering. Using only 1.4K videos and 4 GPUs for one day of training, it achieves state-of-the-art performance on the TAP-Vid benchmark, surpassing BootsTAPIR which uses 15M videos.

## Background & Motivation
**Background**: Point tracking (tracking any point) is a fundamental computer vision task with broad applications in robotics, 3D reconstruction, and video editing.

**Limitations of Prior Work**:
- Large-scale training data relies almost entirely on synthetic sources (e.g., Kubric), which fail to capture the complex visual characteristics of the real world;
- Manual annotation of point trajectories is extremely time- and labor-intensive and cannot be scaled;
- Self-training methods (BootsTAPIR, CoTracker3) require massive amounts of video (15M+) and large-scale computation (256 GPUs), and suffer from confirmation bias.

**Key Challenge**: Real-world data is critical for generalization, yet the cost of obtaining annotations is prohibitively high. The central challenge is how to efficiently acquire high-quality real-world point tracking training data.

**Key Insight**: Human motion naturally encompasses complex phenomena such as non-rigid deformation, articulated movement, and frequent occlusion, while the SMPL model can automatically establish point correspondences.

**Core Idea**: Leverage the SMPL body model to automatically generate pseudo-labeled trajectories from real videos, combined with optical flow consistency filtering, yielding high-quality, low-cost real-world training data.

## Method

### Overall Architecture
Input: human motion video → HMR model (TokenHMR) fits SMPL meshes → mesh vertices projected to 2D to obtain initial trajectories → ray casting for visibility estimation → optical flow consistency filtering → pseudo-label dataset → training of point tracking model.

### Key Designs
1. **SMPL-Based Pseudo-Label Generation**:

    - **Function**: Automatically extract pseudo-labels for 2D point tracking from video.
    - **Mechanism**:
        - Apply pretrained TokenHMR to fit the SMPL model to each detected person per frame, yielding $N_v$ 3D vertices;
        - Each SMPL vertex corresponds to a fixed anatomical location, ensuring temporal consistency;
        - Project to 2D: $\mathbf{x}_{p,t,j} = \Pi(\mathbf{v}_{p,t,j})$.
    - **Design Motivation**: The parametric representation of SMPL reduces complex human motion to low-dimensional pose and shape parameters, allowing HMR models to produce reliable reconstructions even under motion blur and extreme poses. The fixed topology of the 3D mesh provides natural point correspondences.

2. **Ray Casting Visibility Prediction**:

    - **Function**: Determine whether each trajectory point is visible in each frame.
    - **Mechanism**: Cast a ray from the camera center toward the target vertex $\mathbf{v}_{p,t,j}$ and use the Möller–Trumbore algorithm to detect intersections with any triangular face of the body mesh. If an occlusion is detected, $v_{p,t,j} = 0$.
    - **Scope**: Handles self-occlusion and inter-person occlusion, but cannot account for occlusions caused by non-human scene elements (e.g., furniture).
    - **Design Motivation**: Accurate visibility labels are critical for training point trackers; erroneous visibility annotations introduce noisy supervision signals.

3. **Optical Flow Consistency Filtering**:

    - **Function**: Remove unreliable trajectory segments caused by SMPL fitting errors or occlusions from non-human objects.
    - **Mechanism**:
        - Compute forward-backward optical flow consistency between adjacent frames to identify reliable flow regions;
        - Compare SMPL-predicted displacements with optical flow displacements, and flag transition frames where divergence exceeds a threshold;
        - Compute the error ratio per trajectory: discard the entire trajectory if the ratio exceeds the threshold, otherwise remove only the inconsistent frames.
    - **Design Motivation**: SMPL does not model occlusions by scene objects; when a person is occluded by furniture, SMPL still predicts a normal position. Optical flow naturally reflects true image motion, and deviations from SMPL predictions can identify these unreliable segments.

### Loss & Training
- The original training loss of the downstream point tracking model is used directly.
- Data: pseudo-labels generated from 1,400 videos (compared to 15M videos used by BootsTAPIR).
- Training setup: 4 GPUs × 1 day.

## Key Experimental Results

### Main Results (TAP-Vid Benchmark, 256×256 Resolution)

| Method | Training Data | DAVIS First AJ | DAVIS Strided AJ | Kinetics First AJ | Notes |
|--------|--------------|---------------|-----------------|-------------------|-------|
| LocoTrack | Kubric | 63.0 | 67.8 | 52.9 | Synthetic data baseline |
| BootsTAPIR | Kubric+15M | 61.4 | 66.2 | **54.6** | Self-training with 15M videos |
| **Anthro-LocoTrack** | Kubric+1.4K | **64.8** | **69.0** | 53.9 | Only 1.4K real videos |
| TAPNext | Kubric | 62.4 | 65.4 | - | Baseline |
| BootsTAPNext | Kubric+15M | 65.2 | 68.9 | - | Self-training |
| **Anthro-TAPNext** | Kubric+1.4K | **66.1** | **71.4** | - | Surpasses 10,000× more data |

### Ablation Study

| Configuration | DAVIS AJ | Notes |
|--------------|---------|-------|
| Kubric only | 63.0 | Synthetic data baseline |
| + SMPL trajectories (no filtering) | 63.5 | Limited gain due to noise |
| + Ray casting visibility | 64.1 | Visibility labels are important |
| + Optical flow filtering | **64.8** | Full pipeline achieves best results |

### Key Findings
- Human motion pseudo-labels from only 1.4K videos surpass self-training methods using 15M videos.
- The method achieves state-of-the-art results on general (non-human) object tracking benchmarks (DAVIS, Kinetics, including animals and vehicles).
- Optical flow filtering is critical: approximately 15% of trajectories are discarded, yet quality improves significantly.
- The complexity of human motion trajectories—measured by trajectory complexity and diversity—far exceeds that of driving datasets such as DriveTrack.

## Highlights & Insights
- **A thought-provoking core finding**: The structured complexity of human motion serves as the optimal training signal for general-purpose point tracking.
- Exceptional data efficiency: the method surpasses CoTracker3 with 11× fewer videos, and BootsTAPIR with 10,000× fewer frames.
- The pipeline is simple yet effective, combining only off-the-shelf components (HMR + optical flow).
- The dataset is non-proprietary and openly contributed to the community.

## Limitations & Future Work
- Only human motion is exploited; other valuable motion types (e.g., animals, fluids) are not utilized.
- SMPL does not model hand and facial details, resulting in the loss of fine-grained trajectories in these regions.
- HMR models remain limited in robustness under crowded scenes and extreme occlusion.

## Related Work & Insights
- Complementary to DriveTrack (pseudo-labels from driving scenes): driving motion is simple (predominantly rigid bodies), whereas human motion is complex.
- The approach is extensible: any object category with a parametric model (e.g., animals via SMAL) can benefit from the same pseudo-label generation strategy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using human motion as a training signal for general-purpose point tracking is an elegant insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple benchmarks × multiple trackers × extensive ablations × comparisons with several SOTA methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear; the pipeline design logic is rigorous.
- Value: ⭐⭐⭐⭐⭐ Efficient, reproducible, and likely to have lasting impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects](ictpolarreal_a_polarized_reflection_and_material_dataset_of_real_world_objects.md)
- [\[CVPR 2026\] DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)
- [\[CVPR 2026\] Dark3R: Learning Structure from Motion in the Dark](dark3r_learning_structure_from_motion_in_the_dark.md)
- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](../../ICCV2025/3d_vision/revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos
description: >-
  [CVPR 2025][3D Vision][hand motion reconstruction] HaWoR achieves the first reconstruction of 3D hand motion in the world coordinate system from egocentric videos. By decoupling the task into camera-space hand reconstruction and adaptive SLAM camera trajectory estimation, and introducing a motion infilling network to handle out-of-view hand scenarios, it achieves state-of-the-art global trajectory accuracy (ATE 3.36mm) and hand reconstruction quality (PA-MPJPE 4.79mm) on the…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "hand motion reconstruction"
  - "world coordinate system"
  - "egocentric video"
  - "SLAM"
  - "motion infilling"
date: 2026-05-08
content_hash: 5f15b699cad78390
---

# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

**Conference**: CVPR 2025  
**arXiv**: [2501.02973](https://arxiv.org/abs/2501.02973)  
**Code**: [https://github.com/ZJHTerry18/HaWoR](https://github.com/ZJHTerry18/HaWoR)  
**Area**: 3D Vision  
**Keywords**: hand motion reconstruction, world coordinate system, egocentric video, SLAM, motion infilling

## TL;DR
HaWoR achieves the first reconstruction of 3D hand motion in the world coordinate system from egocentric videos. By decoupling the task into camera-space hand reconstruction and adaptive SLAM camera trajectory estimation, and introducing a motion infilling network to handle out-of-view hand scenarios, it achieves state-of-the-art global trajectory accuracy (ATE 3.36mm) and hand reconstruction quality (PA-MPJPE 4.79mm) on the HOT3D dataset.

## Background & Motivation

1. **Background**: Significant progress has been made in 3D hand pose estimation, but existing methods almost exclusively operate in the camera coordinate system, neglecting hand motion trajectories in the world space.

2. **Limitations of Prior Work**: In egocentric videos, both hands and the camera move concurrently, meaning camera-space reconstruction alone cannot reflect real-world motion. Furthermore, frequent out-of-view occurrences, severe occlusions, and rapid motions make world-space reconstruction extremely challenging.

3. **Key Challenge**: World-space hand reconstruction faces two unique difficulties: first, egocentric hand trajectory scales are inherently complex (unlike third-person full-body trajectories); second, frequent out-of-view hand occurrences lead to incomplete motion sequences. Additionally, while human body movements can be constrained by motion priors, establishing priors for hand movements remains exceptionally difficult.

4. **Goal**: Given an egocentric video, how to accurately reconstruct the complete 3D hand motion trajectory in the world coordinate system?

5. **Key Insight**: Decouple the problem into two simpler sub-tasks—hand motion reconstruction in camera space and camera trajectory estimation in world space—and then synthesize the world-space hand motion.

6. **Core Idea**: Develop the first end-to-end world-space hand motion reconstruction system through a temporal-attention-enhanced hand reconstruction network, adaptive hands-off SLAM, and a transformer-based motion infilling network.

## Method

### Overall Architecture
The input is an egocentric video sequence. The pipeline consists of three steps: (1) The hand motion estimation network $\mathcal{M}$ reconstructs the camera-space MANO parameter sequence from video frames; (2) The adaptive SLAM module estimates the world-space camera trajectory (utilizing a hands-off mask and Metric3D scale calibration); (3) The motion infilling network $\mathcal{F}$ transforms incomplete camera-space hand motions into a canonical space, infills missing frames, and maps them back into world space.

### Key Designs

1. **Hand Motion Estimation Network with Temporal Attention**:

    - **Function**: Reconstruct high-fidelity camera-space hand motion from video frame sequences.
    - **Mechanism**: Frame-level features are extracted using a pretrained ViT backbone based on WiLoR. A two-level temporal attention module is introduced: **IAM (Image Attention Module)** performs cross-frame fusion at the ViT feature level to enhance feature robustness in truncated hand regions; **PAM (Pose Attention Module)** performs temporal self-attention at the MANO parameter level to directly learn hand motion priors that constrain the temporal consistency of the reconstruction. Each frame outputs MANO pose $\tilde{\Theta}_t$, shape $\tilde{\beta}_t$, global orientation $\tilde{\Phi}_t$, and camera-space translation $\tilde{\Gamma}_t$.
    - **Design Motivation**: Single-frame methods lack temporal consistency, causing jitter, and are fragile to hand truncation or occlusion. The two-level attention injects temporal information at both the feature and parameter levels, complementarily solving these issues.

2. **Adaptive Egocentric SLAM + Metric Scale Estimation**:

    - **Function**: Estimate the world-space camera trajectory from egocentric videos.
    - **Mechanism**: Built on DROID-SLAM, which degrades under egocentric setups due to large hand-occupied areas. A dual masking strategy is introduced: the reconstructed hand is projected into image space to generate a hand mask $\mathbf{M}_t$, filtering out hand areas in both the input image and the SLAM confidence map via $\hat{w}_t = (1-\mathbf{M}_t) \cdot w_t$, ensuring only background pixels participate in bundle adjustment. Metric3D is utilized to predict metric depth $\mathbf{D}_t$, alongside a proposed Adaptive Sampling Module (AdaSM) which excludes hand regions and excessively far/near points, optimizing the scale factor $\alpha$ only within a reliable intermediate depth range: $E(\alpha) = \sum_{p \in S_t} \mathcal{L}_{GM}(\mathbf{D}_t(p) - \alpha \cdot \mathbf{d}_t(p))$.
    - **Design Motivation**: Standard SLAM degrades severely in egocentric videos with active hand motion (where the hand is the largest dynamic object). Over-reliance on metric depth maps is also inaccurate due to close-range and far-range biases. The dynamic sampling strategy significantly enhances the robustness of scale estimation.

3. **Motion Infiller**:

    - **Function**: Infill missing motion frames when hands leave the field of view.
    - **Mechanism**: First, incomplete MANO sequences are transformed from camera space to canonical space (with the first frame's hand pose as the origin) to eliminate camera motion interference. A transformer encoder structure is used to process sequences with positional encoding, where missing frames are initialized with SLERP (spherical linear interpolation) and linear interpolation. The transformer learns to predict missing MANO parameters from context frames. Training utilizes the HOT3D dataset (which provides both ego and third-person views, making it easy to label hand visibility) and performs data augmentation via random masking.
    - **Design Motivation**: In egocentric videos, hands are out-of-view for 30-50% of the duration; neglecting infilling causes broken trajectories. Canonical space transformation normalizes the input, lowering learning difficulty. SLERP initialization significantly reduces network computational demand.

### Loss & Training
The hand reconstruction loss is formulated as $\mathcal{L}_\mathcal{M}$: 3D joint L1 + 2D joint L1 + MANO parameter L2. The motion infilling loss is defined as $\mathcal{L}_\mathcal{F}$: world translation L1 + global rotation L1 + hand pose L1 + shape L1. Inference runs at 40ms/frame, which is 75% faster than the optimization-based method HMP-SLAM (160ms/frame).

## Key Experimental Results

### Main Results

| Dataset | Metric | HaWoR | Prev. SOTA | Gain |
|--------|------|-------|---------|------|
| DexYCB | PA-MPJPE↓ | **4.76** | 5.01 (WiLoR) | -5.0% |
| DexYCB (75-100% occlusion) | PA-MPJPE↓ | **5.07** | 5.68 (WiLoR) | -10.7% |
| HOT3D | ATE↓ (Camera) | **3.36** | 3.80 (DROID) | -11.6% |
| HOT3D | ATE-S↓ (with scale) | **14.61** | 21.07 (DROID+M3D) | -30.7% |
| HOT3D | W-MPJPE↓ (world) | **33.20** | 119.41 (HMP-SLAM) | -72.2% |
| HOT3D | PA-MPJPE↓ | **4.79** | 6.00 (WiLoR-SLAM) | -20.2% |

### Ablation Study

| Configuration | PA-MPJPE | W-MPJPE | Accel | Description |
|------|---------|---------|-------|------|
| Full model | 4.79 | 33.20 | 5.41 | Full HaWoR |
| w/o Pretrained ViT | 7.59 | 86.80 | 9.09 | Pretraining is crucial |
| w/o IAM & PAM | 5.07 | 44.60 | 8.42 | Missing temporal modules |
| w/o PAM | 4.80 | 36.32 | 6.03 | PAM is crucial for temporal consistency |
| Infiller: Last Pose | - | 116.79 | - | Simplest baseline |
| Infiller: LERP | - | 75.01 | - | Interpolation baseline |
| Infiller: Proposed | - | **66.25** | - | Learned infilling performs best |

### Key Findings
- Pretrained ViT is the most significant individual factor; omitting it causes PA-MPJPE to degrade from 4.79 to 7.59.
- The IAM+PAM two-level temporal attention reduces W-MPJPE from 44.60 to 33.20 and acceleration error from 8.42 to 5.41, verifying the critical nature of temporal information at both levels.
- Adaptive SLAM (hand masking) reduces ATE from 3.80 to 3.36mm. While seemingly minor, the improvement is substantial on ATE-S (with scale) (21.07 -> 14.61).
- The motion infilling network further reduces W-MPJPE by 12% compared to simple interpolation (LERP) (75.01 -> 66.25).
- HaWoR is 4x faster than the optimization-based HMP-SLAM (40ms vs 160ms/frame) while leading significantly in accuracy.

## Highlights & Insights
- **Smart Decoupling Strategy**: Decomposing difficult world-space hand reconstruction into two sub-problems supported by established methods lowers the complexity of end-to-end learning.
- **Adaptive Hands-off SLAM**: Simple yet effective—masking out the hand enables SLAM to work reliably in egocentric videos. This insight can be extended to any SLAM scenario with high-concentration dynamic foregrounds.
- **Canonical Space Motion Infilling**: Removing camera motion before infilling is equivalent to normalizing varying coordinate systems, which greatly simplifies the learning problem.

## Limitations & Future Work
- Relies on off-the-shelf detectors and trackers; failures in these components cascade and affect the entire system.
- The motion infilling network's accuracy may degrade during excessively long gaps (e.g., > dozens of frames).
- World-space performance is validated only on the HOT3D laboratory dataset; its generalization to in-the-wild scenarios remains unverified.
- Does not model bimanual interaction; both hands are reconstructed independently.

## Related Work & Insights
- **vs WHAM/TRAM**: These are methods for global human body motion reconstruction. HaWoR transfers similar ideas to hand reconstruction but faces hand-specific challenges (more frequent occlusions, smaller scales).
- **vs HaMeR/WiLoR**: State-of-the-art in single-frame hand reconstruction. HaWoR builds upon them by adding temporal modeling and global trajectory capabilities.
- **vs SLAHMR**: Optimization-based global human motion reconstruction. HaWoR achieves 4x faster speed through a feed-forward inference paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ First world-space hand motion reconstruction method, featuring a pioneering problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional ablations (hand/SLAM/infilling) with comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear problem decomposition and systematic organization of experiments.
- Value: ⭐⭐⭐⭐⭐ Unlocks egocentric global hand motion understanding, highly valuable for AR/VR and activity analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [\[CVPR 2025\] HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)
- [\[CVPR 2025\] Estimating Body and Hand Motion in an Ego-sensed World](estimating_body_and_hand_motion_in_an_ego-sensed_world.md)
- [\[CVPR 2025\] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)
- [\[CVPR 2026\] DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](../../CVPR2026/3d_vision/duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)

</div>

<!-- RELATED:END -->

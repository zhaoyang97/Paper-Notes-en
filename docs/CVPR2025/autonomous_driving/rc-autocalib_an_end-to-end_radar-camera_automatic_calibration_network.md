---
title: >-
  [Paper Note] RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network
description: >-
  [CVPR 2025][Autonomous Driving][Radar-Camera Calibration] RC-AutoCalib is proposed as the first end-to-end online automatic geometric calibration method for 3D Radar and Camera. By utilizing a dual-perspective (front view + bird's-eye view) feature representation, a selective fusion mechanism, and a noise-resistant matcher, it effectively addresses the sparsity and high uncertainty of Radar data, significantly outperforming existing LiDAR-Camera calibration methods on the nuS…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Radar-Camera Calibration"
  - "Online Self-Calibration"
  - "Dual-Perspective Representation"
  - "Feature Matching"
  - "Noise Resistance"
date: 2026-05-08
content_hash: f900cb2da060e0cb
---

# RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network

**Conference**: CVPR 2025  
**arXiv**: [2505.22427](https://arxiv.org/abs/2505.22427)  
**Code**: [https://github.com/nycu-acm/RC-AutoCalib](https://github.com/nycu-acm/RC-AutoCalib)  
**Area**: Autonomous Driving / Sensor Calibration  
**Keywords**: Radar-Camera Calibration, Online Self-Calibration, Dual-Perspective Representation, Feature Matching, Noise Resistance

## TL;DR

RC-AutoCalib is proposed as the first end-to-end online automatic geometric calibration method for 3D Radar and Camera. By utilizing a dual-perspective (front view + bird's-eye view) feature representation, a selective fusion mechanism, and a noise-resistant matcher, it effectively addresses the sparsity and high uncertainty of Radar data, significantly outperforming existing LiDAR-Camera calibration methods on the nuScenes dataset.

## Background & Motivation

**Background**: Radar and cameras are increasingly popular in ADAS systems due to their low cost and all-weather operational capability. Accurate calibration between sensors is the foundation of multi-modal fusion. Existing calibration methods are mainly divided into offline calibration (requiring targets like calibration boards, which is time-consuming, labor-intensive, and cannot handle sensor drift during operation) and online calibration (utilizing natural scene features, which is more flexible in adapting to dynamic changes).

**Limitations of Prior Work**: (1) Online self-calibration for Radar-Camera has barely been explored, with only Schöller et al. using deep learning for rotational calibration without addressing translational calibration; (2) Although mature solutions exist for online LiDAR-Camera calibration (e.g., LCCNet, CalibDepth), Radar data presents two unique challenges: **sparsity** (far fewer points than LiDAR) and **high uncertainty** (Radar has extremely poor measurement accuracy in the height dimension, leading to significant noise in depth values when projected to the front view); (3) Existing methods primarily extract features from a single front-view perspective, whereas Radar points projected onto the front view are sparser and noisier.

**Key Challenge**: The sparsity and high uncertainty of Radar data make directly applying traditional LiDAR-Camera calibration schemes to Radar-Camera calibration highly ineffective.

**Goal**: Design an end-to-end online self-calibration network that efficiently processes sparse and noisy Radar data, while simultaneously estimating both 6-DoF rotation and translation parameters.

**Key Insight**: The Bird's-Eye View (BEV) is unaffected by height uncertainty (since BEV only uses X and Z coordinates) and can provide features robust to height noise. Therefore, the two views complement each other—the front view preserves rich semantics but is affected by noise, while BEV provides stable geometry but loses semantics.

**Core Idea**: Dual-perspective representation + attention-based selective fusion + explicit feature matching supervision (via a noise-resistant matcher to provide cleaner matching ground truth).

## Method

### Overall Architecture

The inputs are an RGB image, Radar point clouds, and initial calibration parameters $T_{init}$. The data transformation module converts them into four representations: front-view depth maps (estimated by the camera + projected from Radar) and BEV maps (pseudo-BEV images + Radar BEV projection). After feature extraction (ResNet), they enter the feature matching module (including multi-modal cross-attention and explicit matching supervision). Then, a selective fusion mechanism merges the dual-perspective features. Finally, a regression head (LSTM sequence decoder) predicts the rotation and translation vectors. Iterative refinement is supported: the predicted $T_{pred}^i$ updates $T_{init}$ to re-input into the network.

### Key Designs

1. **Dual-Perspective Feature Representation**:

    - **Function**: Extract Radar and camera features from two complementary perspectives: front-view (FV) and Bird's-Eye View (BEV).
    - **Mechanism**: **Radar data**: Use the initial calibration parameters to transform Radar 3D points $\mathcal{P}_r$ to the camera coordinate system, and project them to the front view (recording depth $Z_r^c$) and BEV (recording height $Y_r^c$) respectively. **Camera data**: Use DepthAnything+ZoeDepth to estimate metric depth from the RGB image as the front-view feature; unproject the depth map into a pseudo-point-cloud and project it to BEV to generate a pseudo-BEV image. This yields four pairs of feature maps: $I_R^{FV}, I_I^{FV}$ (front view) and $I_R^{BEV}, I_I^{BEV}$ (BEV).
    - **Design Motivation**: In the BEV perspective, Radar data is unaffected by height uncertainty (only X, Z coordinates are used), providing more stable geometric features. The front-view perspective preserves rich semantic and structural information but is corrupted by height noise. The two perspectives are complementary.

2. **Multi-Modal Cross-Attention + Explicit Feature Matching Supervision**:

    - **Function**: Explicitly establish correspondences between Radar and camera features within each perspective.
    - **Mechanism**: Multi-Modal Cross-Attention (MCA) allows mutual attention between Radar and camera features, computing the attention score $a_{IR} = K_I^\top K_R$ to obtain attended features $m_{I\leftarrow R}$ and $m_{R\leftarrow I}$. On this basis, a Residual Conv Block aggregates them into a unified feature $F_{view}$. An extra matching branch is set up during training: an assignment matrix $P$ is calculated via softmax-normalized similarity matrix $S$ and matchability scores $\sigma_*$, which is then supervised by the matching ground truth matrix $\mathcal{M}$ using a matching loss.
    - **Design Motivation**: Previous methods used concatenation and convolution for implicit matching, which was only indirectly supervised by the final calibration loss and failed to explicitly learn corresponding point pairs. Explicit matching supervision allows the network to truly understand the geometric correspondence between Radar and images.

3. **Noise-Resistant Matcher**:

    - **Function**: Filter out unreliable Radar points caused by height uncertainty in front-view matching, providing cleaner matching ground truth.
    - **Mechanism**: Utilize LiDAR data (only used during training) to identify unreliable Radar points. For each Radar 3D point, an adaptive 3D bounding box $B$ (with height $h_B$, width $w_B$, depth $d_B$ adaptively computed according to the Radar's elevation $\phi$, azimuth $\theta$, range $R$, and tolerance margin $\delta$) is constructed. If the number of LiDAR points within the box exceeds a threshold $\tau$, the Radar point is considered reliable; otherwise, it is removed from the matching ground truth $\mathcal{M}$.
    - **Design Motivation**: Due to poor Radar height measurement accuracy, reflection signals far from the Radar plane generate unreliable 3D positions. Directly using these noisy points as matching ground truth would mislead the network's learning.

### Loss & Training

The total loss is $L_{total} = L_{calib} + \beta L_{matching}$. The matching loss is $L_{matching} = L_{M_{bev}} + L_{M_{fv}}$, where the matching loss for each perspective includes a positive loss (log-likelihood of matching pairs) and a negative loss (no-matchable scores for non-matching points). The calibration loss $L_{calib}$ adopts the iterative calibration loss of CalibDepth. The nuScenes dataset is used, with 12,610 samples for training, 1,628 for validation, and 1,623 for testing. The depth range is 0-200m, and the input resolution is 400×192. The regression head uses an LSTM sequence decoder for multi-step autoregressive prediction.

## Key Experimental Results

### Main Results

Miscalibration Range R1 (±10°, ±0.25m):

| Method | Rotation Error (°) Mean | Roll | Pitch | Yaw | Translation Error (cm) Mean | X | Y | Z |
|------|----------|------|-------|-----|----------|---|---|---|
| LCCNet-1 | 1.603 | 0.123 | 3.130 | 1.556 | 16.531 | 22.99 | 17.65 | 8.95 |
| CalibDepth | 0.807 | 0.390 | 0.345 | 1.686 | 12.608 | 12.86 | 12.25 | 12.72 |
| **Ours** | **0.427** | **0.130** | **0.198** | **0.953** | **9.498** | 12.56 | **3.30** | 12.64 |

Miscalibration Range R2 (±20°, ±1.5m):

| Method | Rotation Error (°) Mean | Translation Error (cm) Mean |
|------|----------|----------|
| CalibDepth | 1.686 | 55.380 |
| **Ours** | **0.852** | **47.537** |

### Ablation Study

| FV | BEV | SF | MCA | EMS | NR | Rot Mean (°) | Trans Mean (cm) |
|----|-----|----|----|-----|----|----|-----|
| ✓ | | | | | | 0.657 | 12.602 |
| | ✓ | | | | | 0.689 | 12.605 |
| ✓ | ✓ | | | | | 0.575 | 12.315 |
| ✓ | ✓ | ✓ | | | | 0.529 | 11.842 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **0.427** | **9.498** |

### Key Findings

- Combining dual perspectives (FV+BEV) reduces the rotation error by 12.5% and 16.5% compared to using single perspectives respectively.
- Selective Fusion (SF) further reduces the rotation error by 8% on top of the dual perspectives, indicating that adaptive selection is more effective than simple merging.
- Explicit Matching Supervision (EMS) contributes most to translational calibration, reducing the Y-direction translation error from 9.98cm to 3.30cm.
- The Noise-Resistant Matcher (NR) successfully filters out noisy matching pairs in the front view, further improving accuracy.
- Under the large miscalibration range (R2), the advantage is even more pronounced, with a rotation error of only 0.852°, far lower than CalibDepth's 1.686°.

## Highlights & Insights

- **First complete online self-calibration scheme for Radar-Camera**: It addresses both rotation and translation calibration, filling a gap in the field and outperforming LiDAR-camera methods (meaning comparable or better calibration accuracy can be achieved using cheaper Radar).
- **Insight on avoiding height uncertainty via the BEV perspective**: It cleverly exploits the characteristic of Radar being highly accurate on the X-Z plane but poor on the Y-axis, completely avoiding the problematic dimension in the BEV perspective.
- **Adaptive 3D bounding box design**: Rather than using simple thresholds, it dynamically adjusts the bounding box size based on the angle and range of each Radar point to estimate reliability, which is physically more reasonable.
- **LiDAR assisted during training, but not required during inference**: It cleverly leverages the presence of LiDAR in nuScenes to generate cleaner training data.

## Limitations & Future Work

- Training relies on LiDAR data to construct the ground truth for the noise-resistant matcher, which limits training on systems with only Radar-Camera setups.
- Using DepthAnything+ZoeDepth for depth estimation is a fixed pre-processing step, and estimation errors will propagate to subsequent modules.
- The number of iterative refinement steps needs to be manually set; an adaptive termination strategy would be an interesting direction.
- Currently only validated on the nuScenes dataset; robustness across more driving datasets (e.g., Waymo, ONCE) and extreme weather conditions remains to be verified.

## Related Work & Insights

- **vs CalibDepth (LiDAR-Camera)**: The current SOTA LiDAR-Camera method; RC-AutoCalib achieves better rotational accuracy and comparable translational accuracy while replacing LiDAR with Radar.
- **vs LCCNet**: An early LiDAR-Camera method that uses cost volumes for feature matching but lacks explicit matching supervision, showing poor performance in Radar scenarios.
- **vs Schöller et al.**: The only prior deep learning-based work on Radar-Camera calibration, but it only handles rotation and uses fixed traffic Radar instead of vehicle-mounted Radar.
- The dual-perspective concept can be transferred to other scenarios involving sparse 3D data and 2D image alignment (e.g., ToF camera calibration).

## Rating

- Novelty: ⭐⭐⭐⭐ The first complete online self-calibration scheme for Radar-Camera; the dual-perspective and noise-resistant matcher designs are highly clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient ablation studies, though conducted on only one dataset.
- Writing Quality: ⭐⭐⭐⭐ Thorough analysis of the problems and detailed descriptions of the methods.
- Value: ⭐⭐⭐⭐ Direct practical value for autonomous driving Radar-Camera fusion systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2025\] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [\[CVPR 2025\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)

</div>

<!-- RELATED:END -->

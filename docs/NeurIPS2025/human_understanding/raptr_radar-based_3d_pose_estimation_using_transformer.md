---
title: >-
  [Paper Note] RAPTR: Radar-Based 3D Pose Estimation Using Transformer
description: >-
  [NeurIPS 2025][Human Understanding][Radar Perception] This paper proposes RAPTR, the first Transformer framework for radar-based 3D human pose estimation using weak supervision (3D bounding boxes + 2D keypoint labels). Through pseudo-3D deformable attention and structured loss functions, RAPTR substantially outperforms baselines on two indoor datasets.
tags:
  - NeurIPS 2025
  - Human Understanding
  - Radar Perception
  - 3D Human Pose Estimation
  - Transformer
  - Weak Supervision
  - Deformable Attention
date: 2026-05-08
content_hash: d9a50af4b1f9c750
---

# RAPTR: Radar-Based 3D Pose Estimation Using Transformer

**Conference**: NeurIPS 2025
**arXiv**: [2511.08387](https://arxiv.org/abs/2511.08387)
**Code**: [GitHub](https://github.com/merlresearch/radar-pose-transformer)
**Area**: Human Understanding
**Keywords**: Radar Perception, 3D Human Pose Estimation, Transformer, Weak Supervision, Deformable Attention

## TL;DR

This paper proposes RAPTR, the first Transformer framework for radar-based 3D human pose estimation using weak supervision (3D bounding boxes + 2D keypoint labels). Through pseudo-3D deformable attention and structured loss functions, RAPTR substantially outperforms baselines on two indoor datasets.

## Background & Motivation

Radar offers unique advantages for indoor human perception: privacy preservation, through-wall sensing, and robustness to lighting and smoke. However, existing methods rely on expensive fine-grained 3D keypoint annotations (typically collected via motion capture systems such as VICON), making annotation prohibitively costly and difficult to scale in cluttered indoor environments with occlusions and multiple persons.

In contrast, 2D keypoint labels (obtained from camera images) and coarse 3D bounding box labels (obtained from depth sensors or radar) are far cheaper to acquire. The core motivation of this paper is: can high-quality radar 3D pose models be trained using only these low-cost weak supervision signals? The key challenge is depth ambiguity — 2D labels provide no depth information, and 3D bounding boxes provide only positional information without joint-level precision.

## Method

### Overall Architecture

RAPTR takes multi-view radar heatmaps as input — horizontal-depth views $\mathbf{Y}_{\text{hor}} \in \mathbb{R}^{T \times W \times D}$ and vertical-depth views $\mathbf{Y}_{\text{ver}} \in \mathbb{R}^{T \times H \times D}$ — extracts multi-scale features via a shared backbone, fuses dual-view information through a cross-view encoder, and progressively estimates 3D human poses via a two-stage decoder (pose decoder + joint decoder).

### Key Designs

1. **Pseudo-3D Deformable Attention**: The core innovation. Reference points and sampling offsets are defined in 3D radar space $(x, y, z)$ and projected onto two 2D radar views for feature extraction: $\mathbf{f}_{\text{hor}}^{(i)} = \mathbf{F}_{\text{hor}}(x+\Delta x_i, z+\Delta z_i)$ and $\mathbf{f}_{\text{ver}}^{(i)} = \mathbf{F}_{\text{ver}}(y+\Delta y_i, z+\Delta z_i)$. Compared to QRFPose's per-view independent 2D attention, this approach handles offsets uniformly in 3D space, eliminates redundant per-view offset estimation, and scales more effectively as the number of views increases. Multi-view attention weights are obtained via linear projection of queries followed by softmax.

2. **Two-Stage Decoder Architecture**: Inspired by the RGB-based pose estimation method PETR. The **pose decoder** processes $N$ pose queries and iteratively updates reference poses at each layer: $\tilde{\mathbf{P}}_{\text{radar}}^{(l)} = \sigma(\sigma^{-1}(\tilde{\mathbf{P}}_{\text{radar}}^{(l-1)}) + \Delta \tilde{\mathbf{P}}_{\text{radar}}^{(l-1)})$, outputting initial 3D poses and confidence scores. The **joint decoder** further refines predictions per joint for each matched pose, taking the pose decoder outputs as input.

3. **Structured Loss Functions (Core)**: Carefully designed to exploit weak supervision labels.

    - **3D Template Loss (T3D)**: Applied at the pose decoder. The centroid $\mathbf{g}_{\text{world}}$ is computed from 3D bounding box labels and combined with a predefined keypoint template $\mathbf{K}_{\text{world}}$ to generate a template pose $\mathbf{T}_{\text{world}} = \mathbf{K}_{\text{world}} + \mathbf{1}^\top \mathbf{g}_{\text{world}}$, constraining initial pose predictions to align with the template and alleviating depth ambiguity.
    - **3D Gravity Loss (G3D)**: Applied at the joint decoder, constraining the centroid of the refined pose to be consistent with the bounding box centroid.
    - **2D Keypoint Loss (K2D)**: Projects the refined 3D pose onto the image plane and computes Euclidean distance and OKS loss against 2D keypoint labels.
    - Total loss: $\mathcal{L} = \frac{1}{N'}\sum(\lambda_1 \mathcal{L}_{\text{template}} + \lambda_2 \mathcal{L}_{\text{gravity}} + \lambda_3 \mathcal{L}_{\text{kpt2D}} + \lambda_4 \mathcal{L}_{\text{OKS}}) + \lambda_5 \mathcal{L}_{\text{cls}}$

### Loss & Training

DETR-style bipartite matching is used to associate predictions with ground truth. The backbone is ResNet. The input uses $T=4$ frames of temporal context. The number of pose queries is $N=10$. Focal loss is used as the classification loss.

## Key Experimental Results

### Main Results

| Dataset | Method | Overall MPJPE (cm) | Horizontal Error | Vertical Error | Depth Error |
|---|---|---|---|---|---|
| HIBER-WALK | Person-in-WiFi 3D | 58.25 | 25.60 | 23.94 | 36.20 |
| HIBER-WALK | QRFPose | 38.20 | 14.78 | 13.40 | 26.76 |
| HIBER-WALK | HRRadarPose | 33.96 | 15.14 | 13.13 | 19.85 |
| HIBER-WALK | **RAPTR** | **22.32** | **8.41** | **4.85** | **17.73** |
| HIBER-MULTI | HRRadarPose | 33.19 | 16.77 | 10.75 | 21.84 |
| HIBER-MULTI | **RAPTR** | **18.99** | **7.80** | **4.38** | **14.54** |

### Ablation Study

| Configuration | Metric | Notes |
|---|---|---|
| Without T3D template loss | MPJPE increases | Depth-direction error increases significantly |
| Without G3D gravity loss | MPJPE increases | 3D positional constraint lost |
| Without K2D 2D loss | MPJPE increases significantly | Fine-grained joint position information lost |
| 2D attention (QRFPose-style) vs. pseudo-3D attention | Pseudo-3D is superior | Unified 3D offset estimation is more efficient |
| Single-stage vs. two-stage decoder | Two-stage is superior | Coarse-to-fine progressive estimation is more effective |

### Key Findings

- RAPTR reduces MPJPE by 34.3% on WALK (vs. HRRadarPose) and by 42.7% on MULTI.
- On the MMVR dataset, joint position error is reduced by 76.9%.
- Vertical error is only 4.85 cm on WALK, far below the 13.13 cm of competing methods, demonstrating that 2D labels effectively constrain vertical position.
- In multi-person scenarios (MULTI), RAPTR maintains consistent performance while baselines degrade noticeably.

## Highlights & Insights

- This is the first systematic exploration of training radar 3D pose models with cheap weak supervision (2D keypoints + 3D bounding boxes), offering significant practical value.
- The structured loss design is elegant: T3D provides an initial 3D skeletal prior via templates, while G3D and K2D constrain complementary dimensions, collectively resolving depth ambiguity.
- Pseudo-3D attention handles sampling points uniformly in 3D space, yielding a more principled and scalable design than per-view independent attention.
- Strong robustness in multi-person scenarios (18.99 cm vs. 33.19 cm) demonstrates that DETR-style query matching is well-suited for radar-based human perception.

## Limitations & Future Work

- The 3D template assumes a fixed human skeletal proportion, which does not generalize to children or atypical body types.
- Evaluation is limited to two indoor datasets; outdoor or uncontrolled environments remain untested.
- Depth estimation under weak supervision remains the largest source of error (17.73 cm vs. 4.85 cm vertically); temporal consistency constraints could be explored.
- Robustness to severe occlusion scenarios (e.g., obstruction by large furniture) requires further evaluation.
- Computational overhead relative to simpler CNN-based baselines warrants further discussion.

## Related Work & Insights

- The method inherits the two-stage query-based pose estimation paradigm from PETR and adapts it to the radar modality.
- Weakly supervised 3D pose estimation has been extensively studied in the RGB domain (2D-to-3D lifting); this paper is the first to realize it on radar.
- The multi-view radar feature fusion strategy may inspire other multi-modal sensor fusion approaches.
- The method has direct application prospects for privacy-preserving indoor intelligence (e.g., elderly monitoring, smart buildings).
- The key distinction from QRFPose lies in pseudo-3D attention performing unified 3D offset estimation vs. per-view independent 2D attention.
- The template loss design is generalizable: any scenario with coarse 3D annotations can leverage human skeletal priors as regularization.
- The bidirectional cross-attention in the cross-view encoder fuses complementary view information more effectively than simple concatenation.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic use of weak supervision for radar 3D pose estimation, with a novel loss design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, multiple baselines, detailed ablations, and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear figures and thorough architectural descriptions.
- Value: ⭐⭐⭐⭐ Reduces annotation cost for radar pose estimation, closely aligned with practical needs.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](../../ICCV2025/human_understanding/bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)
- [\[ICCV 2025\] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation](../../ICCV2025/human_understanding/perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](../../ICCV2025/human_understanding/fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[NeurIPS 2025\] Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization](cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space
description: >-
  [NeurIPS 2025][Human Understanding][3D human pose estimation] This paper proposes PandaPose, which propagates 2D pose priors into a 3D anchor space as a unified intermediate representation. By combining joint-wise adapti…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "3D human pose estimation"
  - "anchor mechanism"
  - "depth estimation"
  - "self-occlusion"
  - "pose lifting"
date: 2026-05-08
content_hash: 075488c3a18ad4e9
---

# PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space

**Conference**: NeurIPS 2025
**arXiv**: [2602.01095](https://arxiv.org/abs/2602.01095)  
**Code**: N/A  
**Area**: 3D Human Pose Estimation / Object Detection
**Keywords**: 3D human pose estimation, anchor mechanism, depth estimation, self-occlusion, pose lifting

## TL;DR

This paper proposes PandaPose, which propagates 2D pose priors into a 3D anchor space as a unified intermediate representation. By combining joint-wise adaptive 3D anchor setting with joint-wise depth distribution estimation, PandaPose achieves robust single-frame 3D human pose lifting against occlusion and 2D pose errors.

## Background & Motivation

**Background**: Monocular 3D human pose estimation is a core task in 3D vision. Single-frame image-based methods have attracted significant attention due to their real-time advantages; current state-of-the-art approaches leverage 2D poses and image features for pose lifting.

**Limitations of Prior Work**:
- Existing methods establish direct joint-to-joint mappings from 2D to 3D, making them overly sensitive to 2D pose accuracy — even minor noise causes substantial deviations in 3D predictions.
- Most approaches rely primarily on in-plane features, lacking explicit modeling of the depth dimension, making it difficult to handle self-occlusion and depth ambiguity.

**Key Challenge**: Simultaneously addressing 2D pose error propagation and depth ambiguity under self-occlusion is required, yet no existing solution handles both effectively.

**Goal**: To propose a framework that propagates 2D pose priors into a 3D anchor space as an intermediate representation.

**Key Insight**: Drawing inspiration from anchor mechanisms explored in hand pose estimation, the paper upgrades fixed global anchors to adaptive joint-wise local anchors.

**Core Idea**: Replace direct joint regression with weighted aggregation over a set of 3D anchors, and introduce joint-wise depth distributions to resolve occlusion.

## Method

### Overall Architecture

A two-stage pipeline: (1) a pretrained 2D pose estimator (HRNet) generates 2D poses and intermediate feature maps; (2) 3D anchor setting → depth-aware feature lifting → anchor-feature interaction decoder → anchor-to-joint aggregation prediction.

### Key Designs

1. **Joint-wise Adaptive 3D Anchor Setting**:

    - **Function**: Dynamically generates a set of 3D anchors per joint as coarse initializations.
    - **Design Motivation**: Fixed global anchors produce excessively large anchor-to-joint offsets (average 154.6 mm), degrading accuracy and robustness.
    - **Mechanism**: The normalized 2D pose $P_J^{2D}$ is passed through a linear layer to generate $K$ 3D offsets $\delta_J = \text{Linear}(P_J^{2D})$ per joint, which are added to the joint's 3D position $(j_x, j_y, 0)$ to form local anchors:
    $$A_{local} = \{a \mid P_a = (j_x, j_y, 0) + \delta_{j,k},\ j \in J,\ k \in K\}$$
    These are supplemented by 256 uniformly distributed global fixed anchors $A_{global}$, yielding $A = A_{global} \cup A_{local}$.
    - **Novelty**: Adaptive anchors reduce the average offset from 154.6 mm to 69.7 mm and exhibit strong resilience to locally inaccurate 2D joints.

2. **Joint-wise Depth Distribution Estimation**:

    - **Function**: Predicts an independent depth distribution map per joint, rather than a single global depth map.
    - **Design Motivation**: A single depth map cannot handle cases where joints with large 3D distances overlap in the 2D projection plane, introducing training ambiguity.
    - **Mechanism**: A lightweight depth network operates on $H/8 \times W/8$ resolution features, discretizing the depth range $[-d_{min}, d_{max}]$ into $K_{bin}$ bins and treating depth prediction as a classification task. Sparse supervision is provided by 3D pose annotations:
    $$\mathcal{L}_{depth} = \frac{1}{N}\sum_{n=1}^{N}\left(\sum_{k=0}^{l_n-1}\log P_{(n,0)}^k + \sum_{k=l_n}^{K_{bin}}\log P_{(n,1)}^k\right)$$
    - **Novelty**: Joint-wise depth distributions provide each occluded joint with independent depth information, overcoming the fundamental limitations of a single depth map.

3. **Anchor-Feature Interaction Decoder**:

    - **Function**: Fuses 3D anchors, depth features, and visual features into unified anchor queries.
    - **Mechanism**: Each layer contains depth cross-attention, inter-anchor self-attention, and 3D deformable cross-attention. The 2D features are lifted to 3D via an outer product $F_{3D} = Dist_D \otimes F_I$, and aggregated using 3D deformable cross-attention:
    $$DCA(a) = \sum_{n \in N} W_n \phi(F_{3D}, P_a + \Delta S_n)$$

4. **Anchor-to-Joint Prediction**:

    - **Function**: Predicts offsets and weights from anchor queries and computes joint positions via weighted summation.
    - **Mechanism**: $P_j^{3D} = \sum_{a \in A} \tilde{W}_{a,j}(P_a + O_{a,j})$

### Loss & Training

$$\mathcal{L} = \lambda_1 \mathcal{L}_{pose} + \lambda_2 \mathcal{L}_{depth}$$

$\lambda_1 = 2$, $\lambda_2 = 0.1$. MPJPE supervises the 3D pose, and binary cross-entropy supervises the joint-wise depth distributions.

## Key Experimental Results

### Main Results

Comparison on Human3.6M (single-frame methods):

| Method | Conference | MPJPE↓ | PA-MPJPE↓ |
|--------|------------|--------|-----------|
| CA-PF | NeurIPS'23 | 41.4 | 33.5 |
| HiPART | CVPR'25 | 42.0 | - |
| **PandaPose (Ours)** | - | **39.8 (1.6↓)** | **32.7 (0.8↓)** |

**Challenging subset** (heavy occlusion / large 2D pose errors):

| Method | MPJPE↓ | PA-MPJPE↓ |
|--------|--------|-----------|
| CA-PF | 82.4 | 82.0 |
| **PandaPose** | **73.1 (9.3↓)** | **69.9 (12.1↓)** |

MPI-INF-3DHP challenging subset: PCK +9.8%, AUC +9.3%, MPJPE −14.8 mm.

Cross-dataset 3DPW: MPJPE −2.3 mm, PA-MPJPE −1.9 mm.

### Ablation Study

Anchor setting ablation (Human3.6M):

| Configuration | MPJPE (Full) | MPJPE (Challenging) |
|---------------|-------------|---------------------|
| No anchors | 42.1 | 81.9 |
| Global fixed anchors only | 40.8 | 76.2 |
| Adaptive local anchors only | 40.1 | 74.0 |
| Global + Local | **39.8** | **73.1** |

Depth distribution ablation:

| Configuration | MPJPE (Full) | MPJPE (Challenging) |
|---------------|-------------|---------------------|
| 2D features (no depth) | 40.9 | 80.8 |
| 3D + single depth map | 40.3 | 75.9 |
| 3D + joint-wise depth | **39.8** | **73.1** |

Depth discretization strategy: classification (64 bins) outperforms regression (39.8 vs. 41.6 MPJPE).

### Key Findings

- Improvements are most pronounced in challenging scenarios (severe occlusion / inaccurate 2D poses), with MPJPE reduced by 9.3 mm.
- Single-frame method performance matches or surpasses state-of-the-art temporal methods that use 243-frame sequences.
- Adaptive anchors exhibit inherent robustness to 2D pose errors — anchor distributions derived from inaccurate 2D poses closely approximate the ground-truth distribution.
- The 2D pose prior feature sampling strategy reduces GPU memory usage by 37% while maintaining performance.

## Highlights & Insights

- **Anchor-to-joint aggregation prediction paradigm**: Transitioning from direct regression to weighted aggregation essentially provides a "soft voting" mechanism with natural error tolerance.
- **Joint-wise depth distribution**: Sparse depth supervision derived from 3D pose annotations is elegantly repurposed to replace hard-to-obtain dense depth ground truth.
- **Multi-use of 2D pose priors**: Anchor generation, feature sampling, and depth estimation all leverage the 2D pose prior, yielding a highly coherent overall design.
- **Substantial gains in challenging scenarios**: PA-MPJPE is reduced by 14.7% under heavy occlusion, demonstrating strong practical value.

## Limitations & Future Work

- Relies on a pretrained 2D pose estimator (HRNet); end-to-end training may yield further improvements.
- The number of anchors requires manual tuning; adaptive anchor count selection could be a natural extension.
- Effectiveness in multi-person scenarios has not been validated.
- Performance is sensitive to the number of depth bins (128 bins degrades performance), warranting further analysis.

## Related Work & Insights

- The anchor mechanism originates from hand pose estimation (A2J-Transformer); this paper is the first to fully introduce it into 3D human pose lifting and upgrade it to an adaptive variant.
- The depth distribution classification strategy draws from BEV-based 3D detection (e.g., BEVDet series), converting dense depth into joint-wise sparse depth.
- Insight: The anchor-to-target aggregation prediction paradigm may generalize to other 3D tasks such as 3D object detection and hand pose estimation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of adaptive anchors and joint-wise depth distributions is novel, though individual components have precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets + challenging subsets + extensive ablations + cross-dataset generalization.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, detailed illustrations, and complete method descriptions.
- **Value**: ⭐⭐⭐⭐ Substantial improvements in challenging scenarios with significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](raptr_radar-based_3d_pose_estimation_using_transformer.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](../../ICCV2025/human_understanding/posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[ICCV 2025\] HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation](../../ICCV2025/human_understanding/hcceposebf_predicting_front_back_surfaces_to_construct_ultra-dense_2d-3d_corresp.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->

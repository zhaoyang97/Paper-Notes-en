---
title: >-
  [Paper Note] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection
description: >-
  [CVPR 2026][Autonomous Driving][4D millimeter-wave radar] This paper proposes R4Det, which systematically addresses three core challenges in 4D radar-camera fusion—inaccurate depth estimation, pose-free temporal fusion…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "4D millimeter-wave radar"
  - "radar-camera fusion"
  - "3D object detection"
  - "depth estimation"
  - "temporal fusion"
date: 2026-05-08
content_hash: 28618b93a95da57a
---

# R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.11566](https://arxiv.org/abs/2603.11566)  
**Code**: N/A  
**Area**: Autonomous Driving
**Keywords**: 4D millimeter-wave radar, radar-camera fusion, 3D object detection, depth estimation, temporal fusion

## TL;DR

This paper proposes R4Det, which systematically addresses three core challenges in 4D radar-camera fusion—inaccurate depth estimation, pose-free temporal fusion, and small object detection—through three plug-and-play BEV modules: Panoramic Depth Fusion (PDF), Deformable Gated Temporal Fusion (DGTF), and Instance-Guided Dynamic Refinement (IGDR). R4Det achieves 47.29% 3D mAP (+5.47%) on TJ4DRadSet and 66.69% mAP on VoD.

## Background & Motivation

**Background**: 4D millimeter-wave radar has emerged as a critical sensor for autonomous driving perception due to its all-weather operability, long range, and low cost. However, its point clouds are sparse and noisy, necessitating fusion with cameras. Existing methods (CRN, SGDet3D, CVFusion, etc.) have made preliminary progress in multimodal fusion within BEV space.

**Challenge 1 — Inaccurate Depth Estimation**: Existing frameworks (SGDet3D, RCBEVDet) apply absolute depth supervision only to foreground points, resulting in sparse supervision, poor panoramic depth estimation quality, and inaccurate 3D localization. Although powerful relative depth models (Metric3D) offer strong generalization, how to effectively leverage their capabilities for accurate panoramic absolute depth remains unresolved.

**Challenge 2 — Pose-Free Temporal Fusion**: Temporal information is critical for detecting occluded objects, but mainstream datasets such as TJ4DRadSet lack ego-vehicle pose. Existing methods rely on simple BEV feature concatenation with limited effectiveness.

**Challenge 3 — Small Object Detection**: Distant small objects such as cyclists may be visible in images yet produce no radar returns, making it necessary to rely on visual priors. Existing Transformer-based approaches extract instance proposals but are incompatible with CNN frameworks.

## Method

### Overall Architecture

R4Det is a progressive BEV feature purification pipeline: (1) **PDF** generates high-quality BEV features from multimodal inputs; (2) **DGTF** performs pose-free temporal alignment and gated aggregation; (3) **IGDR** refines BEV features using 2D instance prototypes before the 3D detection head. The backbone follows the BEV paradigm of SGDet3D (Neighborhood Cross-Attention + LSS).

### Key Designs

1. **Panoramic Depth Fusion (PDF)**:

    - **Function**: Comprehensively improves depth estimation quality through triple supervision, ensuring both accuracy and structural coherence.
    - **Probabilistic Supervision**: Constructs a Gaussian target distribution from sparse LiDAR depth and minimizes KL divergence:
    $\mathcal{L}_{prob} = \frac{1}{|\mathcal{M}_{\text{sparse}}|} \sum_{i \in \mathcal{M}_{\text{sparse}}} \text{KL}(\mathcal{G}(d_{g_i}^{\text{sparse}}) \| \mathcal{P}_i)$
    - **Foundation Model-Guided Supervision**: Applies Smooth L1 absolute depth loss using both sparse radar and dense Metric3D pseudo-GT, balancing keypoint accuracy and full-scene coverage.
    - **Structural Ranking Supervision** (core innovation): Pairwise relative depth ranking loss $\mathcal{L}_{pair}(i,j) = \text{Softplus}(-s_{ij}(\hat{d}_i - \hat{d}_j))$, combined with a depth-adaptive dynamic threshold to filter noise in flat regions:
    $\tau_{ij} = \max(\tau_{abs},\, \tau_{rel} \cdot (d_{g_i}^{\text{dense}} + d_{g_j}^{\text{dense}})/2)$
    - **Foreground-Biased Sampling**: $\mathcal{L}_{edge}$ samples between the dilated mask ring (outside object boundaries) and object interiors, enforcing the network to learn sharp depth discontinuities.
    - **Design Motivation**: Probabilistic or absolute supervision alone provides only local guidance; combining ranking constraints yields structurally coherent panoramic depth.

2. **Deformable Gated Temporal Fusion (DGTF)**:

    - **Function**: Achieves temporal BEV feature alignment and fusion without relying on ego-vehicle pose.
    - **Mechanism**: Spatial alignment and temporal updating are explicitly decoupled into two branches.
    - **Motion-Aware Alignment Branch**: Employs DCNv2 to learn deformable offsets $\Delta p$ and modulation masks $m$, predicted from $X_t$ and $H_{t-1}$:
    $\tilde{H}_{t-1} = \text{DCNv2}(H_{t-1}, \Delta p, m)$
      The learned offsets implicitly reconstruct relative motion flow, while the modulation masks suppress unreliable background regions.
    - **Gated Temporal Update Branch**: Follows a GRU-style design—reset gate $r_t$ filters historical information, update gate $z_t$ balances old and new information:
    $H_t = (1 - z_t) \odot X_t + z_t \odot \tilde{H}_t$
    - **Design Motivation**: Conventional RNNs handle alignment and updating simultaneously with reduced efficiency; the decoupled design of DCN for spatial correction and GRU for temporal evolution is more precise.

3. **Instance-Guided Dynamic Refinement (IGDR)**:

    - **Function**: Dynamically calibrates BEV features using clean 2D instance semantic priors to address instance overlap contamination and ambiguity in distant small objects.
    - **Instance Semantic Prior Construction**: Instance features $E_{features}$ are extracted from a 2D RPN, pooled and projected to obtain instance prototypes $E_{proj}$, then broadcast into BEV space via Softmax-weighted fusion over the LSS-projected spatial distribution $S_{BEV}$:
    $E_{BEV} = \text{BMM}(\text{Softmax}(S_{BEV}/\tau),\, E_{proj})$
    - **Prototype-Guided Dynamic Calibration** (core innovation): $E_{BEV}$ is passed through a Conv layer to predict per-location affine parameters $(\gamma_{BEV}, \beta_{BEV})$, which perform feature-wise affine transformation on the potentially noisy $F_{RC}$:
    $F_{calibrated} = F_{RC} \odot \gamma_{BEV} + \beta_{BEV}$
    - **Foreground-Gated Fusion**: The sum of $S_{BEV}$ across all instances is passed through a Gate-conv and Sigmoid to produce gate $G_{bg}$, applying calibration only within instance regions:
    $F_{final} = (1 - G_{bg}) \odot F_{RC} + G_{bg} \odot F_{calibrated}$
    - **Design Motivation**: Directly fusing instance features introduces background noise; the indirect approach of generating calibration parameters from instance prototypes is more robust.

### Loss & Training

- **Depth Loss**: $\mathcal{L}_{depth} = \lambda_1 \mathcal{L}_{prob} + \lambda_2 \mathcal{L}_{found} + \lambda_3 \mathcal{L}_{relative}$, with weights $\lambda_1=0.1, \lambda_{abs}=0.01, \lambda_{dense}=0.03, \lambda_3=0.05$
- **Two-Stage Training**: (i) 15 epochs of spatially-aware pretraining (DGTF/IGDR/detection head frozen) to initialize PDF and the 2D instance branch; (ii) 15 epochs of full end-to-end fine-tuning.
- **Optimizer**: AdamW, lr=4e-4, cosine decay.
- **IGDR Training Strategy**: Strictly uses dynamically generated proposals from the 2D detector rather than GT bounding boxes, avoiding exposure bias.

## Key Experimental Results

### Main Results

**TJ4DRadSet Test Set**:

| Method | Modality | mAP$_{3D}$ | mAP$_{BEV}$ | Cyclist AP | Gain |
|---|---|---|---|---|---|
| SGDet3D | R+C | 41.82 | 47.16 | 51.30 | Baseline |
| CVFusion | R+C | 40.00 | 44.07 | 49.41 | - |
| **R4Det** | **R+C** | **47.29** | **54.07** | **62.84** | **+5.47/+6.91** |

**VoD Validation Set**:

| Method | Modality | mAP$_{EAA}$ | mAP$_{DC}$ | FPS |
|---|---|---|---|---|
| SGDet3D | R+C | 59.75 | 77.42 | 9.2 |
| CVFusion | R+C | 65.41 | 82.42 | 5.4 |
| **R4Det** | **R+C** | **66.69** | **83.68** | **8.3** |

### Ablation Study

**Incremental Module Stacking (TJ4DRadSet Val)**:

| PDF | DGTF | IGDR | mAP$_{BEV}$ | mAP$_{3D}$ | Note |
|---|---|---|---|---|---|
| | | | 45.15 | 39.86 | SGDet3D baseline |
| ✓ | | | 46.86 | 41.41 | +1.71 (depth improvement) |
| ✓ | ✓ | | 50.41 | 44.86 | +3.55 (temporal fusion) |
| ✓ | ✓ | ✓ | **54.07** | **47.29** | +3.66 (instance refinement) |

**DGTF Module Ablation**:

| Configuration | BEV mAP | 3D mAP | Note |
|---|---|---|---|
| No temporal | 46.86 | 41.41 | Baseline |
| +Concat | 47.82 | 42.01 | Simple concatenation |
| +DCN | 48.86 | 43.32 | Deformable alignment |
| **+DCN+ConvGRU** | **50.41** | **44.86** | Full DGTF |

### Key Findings

- Cyclist (small object) achieves the most significant improvement: **+11.54 AP** (51.30→62.84), validating the effectiveness of IGDR for small objects.
- All three modules are fully plug-and-play: applying them to BEVFusion/RCBEVDet yields improvements of **+6.34/+5.34 mAP**, respectively.
- ConvGRU in DGTF contributes the largest single gain (+3.45 3D mAP), while SE modules prove detrimental.
- Conv calibrator in IGDR outperforms Attention and MLP calibrators, indicating that local spatial patterns are more effective than global attention.
- The edge ranking loss (boundary sampling) in PDF is critical for sharp depth boundary estimation.

## Highlights & Insights

1. **Problem-Driven Modular Design**: Three clearly defined technical challenges map to three decoupled modules, offering both engineering and research value.
2. **Pose-Free Temporal Fusion**: The DCN+GRU decoupled design elegantly resolves the challenge of temporal fusion without ego-vehicle pose.
3. **Boundary Sampling in Structural Ranking Loss**: The dilated ring sampling strategy compels the network to focus on depth discontinuity boundaries, representing a practically valuable technique.
4. **Thorough Plug-and-Play Validation**: Effectiveness is verified not only within the proposed framework but also through successful transfer to BEVFusion and RCBEVDet, enhancing credibility.

## Limitations & Future Work

1. The approach relies on Metric3D as pseudo-GT; its intrinsic errors propagate into depth supervision.
2. The GRU-style recurrence in DGTF may suffer from information decay over long temporal horizons; Transformer-based temporal modeling warrants exploration.
3. The 2D instance branch in IGDR depends on RPN quality; a weaker detector may limit refinement effectiveness.
4. Evaluation is conducted only on two relatively small-scale datasets (TJ4DRadSet and VoD); performance on large-scale benchmarks such as nuScenes remains unverified.

## Related Work & Insights

- **SGDet3D**: Direct baseline; R4Det adds three modules on top of its BEV framework.
- **Metric3D**: Provides dense pseudo-depth GT, enabling panoramic depth supervision.
- **BEVFormer**: Performs temporal fusion in BEV but relies on ego pose, complementing DGTF's pose-free approach.
- **Insights**: (a) "Calibrating the main feature stream using clean parallel features" (IGDR) is a general strategy for handling BEV feature contamination; (b) Combining absolute, relative, and structural ranking supervision for depth estimation is generalizable to other depth-related tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (Each of the three modules contributes innovations, with DGTF and IGDR being particularly elegant in design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (State-of-the-art on two datasets + plug-and-play validation + detailed per-module ablation)
- Writing Quality: ⭐⭐⭐⭐ (Clear problem-solution correspondence with well-structured ablation design)
- Value: Pending evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[CVPR 2026\] Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection](look_before_you_fuse_2d-guided_cross-modal_alignment_for_robust_3d_detection.md)
- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[CVPR 2026\] A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)
- [\[AAAI 2026\] Exploring Surround-View Fisheye Camera 3D Object Detection](../../AAAI2026/autonomous_driving/exploring_surround-view_fisheye_camera_3d_object_detection.md)

</div>

<!-- RELATED:END -->

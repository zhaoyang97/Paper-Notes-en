---
title: >-
  [Paper Note] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection# R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection
description: >-
  [CVPR 2026][3D Vision][4D mmWave Radar] Proposes R4Det, addressing inaccurate depth estimation, ego-pose-dependent temporal fusion, and small object detection challenges in 4D radar-camera fusion via three plug-and-play modules: Panoramic Depth Fusion (PDF), Deformable Gated Temporal Fusion (DGTF), and Instance-Guided Dynamic Refinement (IGDR), achieving SOTA on TJ4DRadSet and VoD.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D mmWave Radar
  - Camera-Radar Fusion
  - 3D Object Detection
  - BEV Perception
  - Depth Estimation
date: 2026-05-08
content_hash: 679bd6f2ed68dea9
---
**Conference**: CVPR 2026
**Area**: Object Detection / 3D Vision / Autonomous Driving
**Keywords**: 4D millimeter-wave radar, camera-radar fusion, 3D object detection, BEV perception, depth estimation
**arXiv**: [2603.11566](https://arxiv.org/abs/2603.11566)
**Code**: N/A

## TL;DR

R4Det proposes three plug-and-play modules — Panoramic Depth Fusion (PDF), Deformable Gated Temporal Fusion (DGTF), and Instance-Guided Dynamic Refinement (IGDR) — to address the key challenges in 4D radar-camera fusion: inaccurate depth estimation, ego-pose-dependent temporal fusion, and poor small-object detection. State-of-the-art results are achieved on TJ4DRadSet and VoD.

## Background & Motivation

4D millimeter-wave radar has attracted increasing attention in autonomous driving as a cost-effective, all-weather alternative to LiDAR, offering long-range perception capabilities. However, radar point clouds are inherently sparse and noisy, making standalone high-accuracy 3D detection infeasible. Camera-radar fusion is therefore necessary. Existing 4D radar-camera fusion methods suffer from three key limitations: (1) depth estimation modules lack robustness — sparse supervision is applied only to foreground points, resulting in poor panoramic depth quality and inaccurate 3D localization; (2) temporal fusion relies heavily on ego-vehicle pose, which is unavailable or unreliable in datasets such as TJ4DRadSet and in real-world scenarios with GPS dropout, causing temporal fusion to degrade to naive channel concatenation; (3) distant small objects (e.g., cyclists) are visible in images but produce no radar returns, so detection must rely on visual priors that existing methods exploit insufficiently.

## Core Problem

How can a BEV-based 4D radar-camera fusion framework simultaneously address three core challenges: geometric corruption from poor depth estimation, temporal feature alignment without ego pose, and BEV feature degradation for small objects not covered by sparse radar?

## Method

### Overall Architecture

R4Det is a progressive BEV feature purification pipeline. Multi-view camera images and 4D radar point clouds are encoded by an image backbone and a radar encoder, respectively. The pipeline proceeds through three stages: (1) the PDF module uses sparse radar features as queries to aggregate image semantics via neighborhood cross-attention, generates high-quality depth maps under triple depth supervision, and projects features into BEV space via LSS; the resulting camera BEV and radar BEV are concatenated to form the initial fused BEV $X_t$; (2) the DGTF module performs ego-pose-free temporal alignment and gated update on $X_t$, producing temporally consistent features $F_{RC}$; (3) the IGDR module uses 2D instance-level semantic prototypes to dynamically calibrate $F_{RC}$, yielding the refined feature $F_{final}$ that is fed into the 3D detection head.

### Key Designs

1. **Panoramic Depth Fusion (PDF)**: The core innovation is a triple depth supervision scheme. *Probability supervision* applies KL divergence between the predicted depth distribution and a Gaussian centered at sparse LiDAR measurements, encouraging sharp and accurate depth distributions for the LSS splat operation. *Foundation-model-guided supervision* combines sparse radar depth and dense pseudo-GT from Metric3D via Smooth L1 regression, balancing precision at key points with full-scene coverage. *Structural ranking supervision* introduces pairwise relative depth ordering loss with a depth-dependent dynamic threshold to filter uninformative pairs, and employs a foreground-biased dual sampling strategy — boundary sampling draws pixel pairs between the dilated mask periphery and the object interior to enforce sharp depth transitions at object edges, while background sampling maintains global structural consistency. Together, these three losses improve depth accuracy in terms of probabilistic correctness, metric precision, and structural continuity.

2. **Deformable Gated Temporal Fusion (DGTF)**: Temporal fusion is decoupled into two sub-problems — spatial alignment and temporal update. In the motion-aware alignment branch, the current BEV feature $X_t$ and the previous hidden state $H_{t-1}$ are concatenated to predict sampling offsets $\Delta p$ and modulation masks $m$; DCNv2 is then applied to deformably align $H_{t-1}$, with learned offsets implicitly encoding inter-frame relative motion without requiring ego pose. In the gated update branch, a GRU-style reset gate $r_t$ filters the relevance of aligned historical features, while an update gate $z_t$ adaptively balances current observations with historical context; a convolutional layer produces the final output $F_{RC}$. The key insight of decoupling "alignment via DCN" from "selective update via GRU" yields substantially better performance than naive concatenation or standard recurrent units.

3. **Instance-Guided Dynamic Refinement (IGDR)**: The central idea is to use clean 2D instance-level features as semantic priors to actively calibrate the BEV features that may be corrupted by radar noise or poor depth. Instance prototypes are obtained by global average pooling and channel projection over RoI features from a 2D instance segmentation head. These prototypes are spatially distributed into BEV space via Softmax-weighted aggregation over the LSS spatial allocation map $S_{BEV}$, producing a clean instance feature map $E_{BEV}$. Critically, $E_{BEV}$ is not directly fused with $F_{RC}$; instead, it serves as a conditional generator that predicts spatially varying affine transformation parameters (scale $\gamma$ and bias $\beta$) through convolutions to calibrate $F_{RC}$. A foreground gate $G_{bg}$ ensures that calibration is applied only within instance regions, preserving background structure. Training uses dynamic proposals from the 2D detector rather than GT bounding boxes to avoid exposure bias.

### Loss & Training

The total depth loss is $\mathcal{L}_{depth} = \lambda_1 \mathcal{L}_{prob} + \lambda_2 \mathcal{L}_{found} + \lambda_3 \mathcal{L}_{relative}$, with $\lambda_1=0.1$, $\lambda_{abs}=0.01$, $\lambda_{dense}=0.03$, and $\lambda_3=0.05$. A two-stage training strategy is adopted: stage 1 trains PDF and the 2D instance branch for 15 epochs with DGTF, IGDR, and the detection head frozen (spatial-aware pre-training); stage 2 fine-tunes all parameters end-to-end for 15 epochs. The optimizer is AdamW with an initial learning rate of $4\times10^{-4}$ and cosine decay.

## Key Experimental Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|---------|--------|------|------------|------|
| TJ4DRadSet test | 3D mAP | 47.29% | 41.82% (SGDet3D) | +5.47% |
| TJ4DRadSet test | BEV mAP | 54.07% | 47.16% (SGDet3D) | +6.91% |
| TJ4DRadSet test | Cyclist AP3D | 62.84% | 54.93% (RCFusion) | +7.91% |
| VoD val | mAP$_\text{EAA}$ | 66.69% | 65.41% (CVFusion) | +1.28% |
| VoD val | mAP$_\text{DC}$ | 83.68% | 82.42% (CVFusion) | +1.26% |
| BEVFusion+Ours | mAP$_\text{EAA}$ | 55.59% | 49.25% (BEVFusion) | +6.34% |
| RCBEVDet+Ours | mAP$_\text{EAA}$ | 55.33% | 49.99% (RCBEVDet) | +5.34% |

### Ablation Study

- **Sequential module ablation** (TJ4DRadSet val): Baseline 39.86 → +PDF 41.41 (+1.55) → +DGTF 44.86 (+3.45) → +IGDR 47.29 (+2.43); all three modules contribute complementarily.
- **PDF internal ablation**: Dense metric loss contributes +0.93 mAP; structural ranking loss adds a further +0.78 mAP.
- **DGTF internal ablation**: Simple concatenation +0.60, +DCN alignment +1.31, +GRU gating +1.54; SE attention provides no additional gain.
- **IGDR internal ablation**: Direct Softmax fusion is nearly ineffective (+0.88); foreground gating is critical (+1.19); convolutional affine generator achieves the best overall gain (+1.55), outperforming MLP and attention-based alternatives.
- **Temporal depth sensitivity**: Using only the $t{-}1$ frame is optimal; including $t{-}2$ and $t{-}3$ introduces accumulated noise and degrades performance.

## Highlights & Insights

- The triple depth supervision framework (probability + metric + ranking) constitutes a comprehensive approach to panoramic depth estimation; the boundary cross-sampling strategy is particularly elegant, directly supervising the network to learn sharp depth transitions at object contours.
- DGTF achieves temporal fusion entirely without ego pose by decoupling alignment from update, making it immediately applicable to pose-unreliable scenarios such as indoor environments and underground parking.
- IGDR uses instance semantics to perform conditional affine calibration rather than direct feature replacement — a clean and principled formulation in which clean 2D priors serve as conditions for 3D feature self-correction.
- All three modules are plug-and-play and can be seamlessly integrated into other BEV-based detection frameworks.

## Limitations & Future Work

- Evaluation is limited to TJ4DRadSet and VoD; validation on larger-scale datasets with 4D radar configurations (e.g., nuScenes) is absent.
- PDF relies on Metric3D to generate pseudo-GT depth; the quality ceiling of PDF is thus bounded by the pre-trained depth foundation model.
- The two-stage training strategy adds training complexity; whether a single-stage end-to-end alternative is viable remains unexplored.
- Inference speed is approximately 8–9 FPS on an RTX 3090, which may be insufficient for real-time deployment.
- IGDR's effectiveness depends on the quality of the 2D instance segmentation head; missed detections by the 2D detector directly limit IGDR's contribution.

## Related Work & Insights

- **SGDet3D**: The direct baseline for R4Det, with only sparse depth supervision and simple multimodal fusion. R4Det adds panoramic depth, temporal fusion, and instance refinement, yielding a +5.47% 3D mAP improvement.
- **CVFusion**: A concurrent method employing point- and grid-guided multi-view fusion, which is strong on VoD but lacks temporal modeling. R4Det surpasses it on both datasets.
- **HyDRa**: Fuses features in both perspective and BEV spaces but similarly lacks autonomous temporal fusion and instance-level refinement.

The triple depth supervision paradigm of PDF is transferable to any BEV perception framework requiring view transformation. The ego-pose-free temporal fusion design of DGTF offers general reference value for all BEV temporal methods, especially in pose-unreliable settings. The "instance-semantic conditional affine calibration" paradigm of IGDR is a generic feature refinement strategy transferable to downstream tasks such as semantic and panoptic segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ Each module addresses a concrete pain point; the combination is well-grounded, though no single module represents a fundamental breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ State-of-the-art results on two datasets, plug-and-play verification on two additional frameworks, and detailed per-module/per-component ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with explicit motivation-method-experiment correspondence and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Direct contribution to the 4D radar perception field; the plug-and-play nature of all three modules confers strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] 4C4D: 4 Camera 4D Gaussian Splatting](4c4d_4_camera_4d_gaussian_splatting.md)
- [\[CVPR 2026\] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)
- [\[ICCV 2025\] G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection](../../ICCV2025/3d_vision/g2sf_geometry-guided_score_fusion_for_multimodal_industrial_anomaly_detection.md)
- [\[CVPR 2026\] SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](span_spatial_projection_alignment_mono3d.md)
- [\[CVPR 2026\] A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection](a_semantically_disentangled_unified_model_for_multi-category_3d_anomaly_detectio.md)

<!-- RELATED:END -->

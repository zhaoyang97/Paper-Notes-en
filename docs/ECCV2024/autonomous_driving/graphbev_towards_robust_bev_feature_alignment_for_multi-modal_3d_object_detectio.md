---
title: >-
  [Paper Note] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection
description: >-
  [ECCV 2024][Autonomous Driving][3D Object Detection] To address the feature misalignment caused by calibration errors between LiDAR and cameras in multi-modal BEV fusion, this paper proposes the GraphBEV framework. It introduces two modules: LocalAlign (KD-Tree-based neighborhood depth graph matching) and GlobalAlign (global alignment via learnable offsets). GraphBEV achieves 70.1% mAP on nuScenes (outperforming BEVFusion by 1.6%) and outperforms BEVFusion by 8.3% in noisy mi…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "3D Object Detection"
  - "Multi-Modal Fusion"
  - "BEV Feature Alignment"
  - "LiDAR-Camera"
  - "Graph Matching"
date: 2026-05-08
content_hash: c12265ade60c7ca3
---

# GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2403.11848](https://arxiv.org/abs/2403.11848)  
**Code**: [https://github.com/adept-thu/GraphBEV](https://github.com/adept-thu/GraphBEV)  
**Area**: Autonomous Driving  
**Keywords**: 3D Object Detection, Multi-Modal Fusion, BEV Feature Alignment, LiDAR-Camera, Graph Matching

## TL;DR
To address the feature misalignment caused by calibration errors between LiDAR and cameras in multi-modal BEV fusion, this paper proposes the GraphBEV framework. It introduces two modules: LocalAlign (KD-Tree-based neighborhood depth graph matching) and GlobalAlign (global alignment via learnable offsets). GraphBEV achieves 70.1% mAP on nuScenes (outperforming BEVFusion by 1.6%) and outperforms BEVFusion by 8.3% in noisy misalignment scenarios.

## Background & Motivation

**Background**: Multi-modal 3D object detection is a core task in autonomous driving. Current mainstream solutions adopt the BEV fusion paradigm (e.g., BEVFusion) to project LiDAR and camera information into a unified BEV space for fusion, achieving promising performance on clean datasets such as nuScenes.

**Limitations of Prior Work**: Methods like BEVFusion neglect a critical issue: **feature misalignment**. In real-world scenarios, the calibration matrix between LiDAR and cameras is typically obtained through manual calibration, which inevitably introduces projection errors. Combined with road vibration, these errors further escalate during runtime and cannot be fully eliminated through online calibration.

**Key Challenge**: The camera-to-BEV process in BEVFusion relies on the explicit LiDAR-to-camera depth supervision from BEVDepth, which assumes accurate depth projections. Projection errors lead to:

**Local Misalignment**: Inaccurate depth from LiDAR-to-camera projection maps the depths of surrounding neighborhoods incorrectly to the current pixel depth, causing local misalignment in camera BEV features.

**Global Misalignment**: During the fusion of LiDAR BEV and camera BEV features, global spatial offsets caused by inaccurate depths are ignored by simple concatenation-based fusion strategies.

**Key Insight**: Addressing local and global misalignment in the camera-to-BEV stage and the LiDAR-camera BEV fusion stage, respectively, starting from the root cause of calibration errors.

**Core Idea**: Utilizing graph matching to construct neighborhood depth awareness for correcting local depth errors, while addressing global BEV feature misalignment via simulated offset noise during training and learnable offsets.

## Method

### Overall Architecture
GraphBEV is built upon the BEVFusion framework, consisting of a LiDAR branch and a Camera branch:
- **LiDAR Branch**: Extracts 3D features using a SECOND encoder and compresses them along the Z-axis to obtain LiDAR BEV features.
- **Camera Branch**: Extracts multi-view image features using Swin-Transformer, and converts camera features to camera BEV features via the **LocalAlign module**.
- **Fusion Stage**: Aligns LiDAR and camera BEV features via the **GlobalAlign module**, followed by a detection head for 3D detection.

### Key Designs

#### 1. LocalAlign Module: Graph Matching-based Local Alignment

**Function**: To resolve the camera-to-BEV local misalignment caused by inaccurate depth projections during LiDAR-to-camera mapping.

**Mechanism**: For each LiDAR pixel projected onto the camera, instead of only using its projected depth, the KD-Tree algorithm is utilized to find the depth information of its $K_{graph}$ nearest neighbor pixels, constructing neighborhood-aware depth features.

**Key Steps**:
1. LiDAR-to-camera projection yields the projected depth $D_S \in \mathbb{R}^{B_S \times N_C \times 1 \times H \times W}$ and projected pixel coordinates $M_{Coords} \in \mathbb{R}^{N_P \times 2}$.
2. Each projected pixel's $K_{graph}=8$ nearest neighbors are retrieved using the KD-Tree algorithm to obtain neighborhood coordinates $M_{K_{Coords}} \in \mathbb{R}^{N_P \times K_{graph} \times 2}$.
3. The neighborhood depth $D_K \in \mathbb{R}^{B_S \times N_C \times K_{graph} \times H \times W}$ is indexed using the neighborhood coordinates.
4. $D_S$ and $D_K$ are encoded into dual depth features $D_{SK}$ through a **Dual Transform module** (Conv+BN+ReLU).
5. $D_{SK}$ is fused with the camera FPN features $F_{Cam}$ in DepthNet, which splits into new depth features $\hat{F_D}$ and image context features $\hat{F_C}$.
6. $\hat{F_D}$ undergoes a softmax operation and is element-wise multiplied by $\hat{F_C}$ to obtain depth-aware image features, and camera BEV features are finally generated via BEV Pooling.

**Design Motivation**: When the projected depth is inaccurate due to calibration errors, the ground-truth depth is highly likely to exist within the depths of neighboring projection points. Introducing neighborhood depth information via graph matching provides an "error-tolerance space" for depth estimation.

#### 2. GlobalAlign Module: Learnable Offset-based Global Alignment

**Function**: To address global spatial offsets between LiDAR BEV and camera BEV features.

**Mechanism**: Random offset noise is added to camera BEV features during training to simulate global misalignment, followed by training an offset predictor to restore alignment.

**Key Steps**:
1. Concatenate LiDAR BEV features $F_B^L$ and camera BEV features $F_B^C$ to form $F_B^{MM}$.
2. Pass $F_B^{MM}$ through convolution to obtain clean fused features $\hat{F_B}$ (acting as the supervision signal during training).
3. Add random offset noise to the camera portion of $F_B^{MM}$ during training to obtain noisy features $F_N^{MM}$.
4. Train a CBR module with $F_N^{MM}$ to predict offsets $F^O \in \mathbb{R}^{B_S \times 2 \times H_B \times W_B}$.
5. Perform Grid Sampling on $F_B^L$ using $F^O$ to get deformation weights $F_W^D$, which are multiplied by $F_B^L$ and passed through CBR to obtain the Deform BEV $F_B^D$.
6. No noise is added during inference; forward inference directly uses the learned offsets.

**Design Motivation**: LiDAR BEV features compressed directly from point clouds are relatively more accurate. Simulating various levels of offset noise during training forces the network to learn to predict and compensate for spatial offsets from fused features.

### Loss & Training

**GlobalAlign Alignment Loss**:
$$L_{Align} = \frac{1}{N_B} \sum_{i=1}^{N_B} (\hat{F_B}_i - {F_B^D}_i)^2$$

This formulation represents the MSE loss between clean fused features and the denoised Deform BEV features. The total loss consists of the detection head loss and the alignment loss.

**Training Strategy**: Offset noise is only added during training. No noise is added during inference, and the learned offsets are utilized directly. The model is trained for 10 epochs using CBGS data resampling with the Adam optimizer and a maximum learning rate of 0.001.

## Key Experimental Results

### Main Results

**nuScenes Validation Set 3D Object Detection**:

| Method | Modality | mAP | NDS | Car | Barrier | Bike | Ped. |
|------|------|------|------|-------|---------|-------|-------|
| TransFusion-L | L | 65.1 | 70.1 | 86.5 | 74.1 | 56.0 | 86.6 |
| BEVFusion-MIT | LC | 68.5 | 71.4 | 89.2 | 72.0 | 65.3 | 88.2 |
| ObjectFusion | LC | 69.8 | 72.3 | 89.7 | 75.2 | 65.0 | 89.3 |
| **GraphBEV** | **LC** | **70.1** | **72.9** | **89.9** | **76.0** | **67.5** | **89.2** |
| Gain (vs BEVFusion) | - | **+1.6** | **+1.5** | +0.7 | **+4.0** | **+2.2** | +1.0 |

**nuScenes Noisy Misalignment Settings**:

| Method | mAP | NDS | Margin to BEVFusion |
|------|------|------|-----------------|
| BEVFusion (Noisy) | 60.8 | 65.7 | - |
| GraphBEV (Noisy) | 69.1 | 72.0 | **+8.3 / +6.3** |
| GraphBEV Degradation (Clean→Noisy) | -1.0 | -0.9 | Minimal degradation |
| BEVFusion Degradation (Clean→Noisy) | -7.7 | -5.7 | Significant degradation |

### Ablation Study

**Module Contributions (Clean / Noisy Settings)**:

| Configuration | mAP(Clean) | mAP(Noisy) | NDS(Clean) | NDS(Noisy) | Latency (ms) |
|------|-----------|-----------|-----------|-----------|---------|
| Baseline (BEVFusion) | 68.5 | 60.8 | 71.4 | 65.7 | 133.2 |
| +LocalAlign only | 69.7(+1.2) | 67.0(+6.2) | 72.4(+1.0) | 70.1(+4.4) | 136.3 |
| +GlobalAlign only | 68.9(+0.4) | 63.1(+2.3) | 71.7(+0.3) | 67.2(+1.5) | 138.1 |
| GraphBEV (Both) | **70.1(+1.6)** | **69.1(+8.3)** | **72.9(+1.5)** | **72.0(+6.3)** | 140.9 |

**$K_{graph}$ Hyperparameter Ablation Study (Noisy Setting)**:

| $K_{graph}$ | mAP | NDS | Latency (ms) |
|-------------|-------|-------|---------|
| Baseline | 60.8 | 65.7 | 132.9 |
| 5 | 67.1 | 70.9 | 138.2 |
| **8** | **69.1** | **72.0** | 141.0 |
| 12 | 69.8 | 72.2 | 143.4 |
| 16 | 68.8 | 70.5 | 145.3 |

### Key Findings

1. **LocalAlign is the primary contributor to performance improvement**: Particularly in noisy scenarios, LocalAlign alone contributes a +6.2 mAP gain.
2. **Small objects benefit the most**: Barriers (+4.0%) and Bikes (+2.2%) show the most significant improvement due to their high sensitivity to misalignment.
3. **Extreme robustness**: GraphBEV degrades by only ~1% mAP from clean to noisy settings, whereas BEVFusion drops by ~8% mAP.
4. **Low computational overhead**: Adds only ~8ms of latency compared to BEVFusion, which is still lower than TransFusion.
5. **Noticeable improvement in night scenes**: Night mAP increases from 42.8% to 45.1% (+2.3%).
6. **Greatest improvements on distant and small objects**: Distant objects improve by +2.1 mAP, and small objects improve by +5.1 mAP.

## Highlights & Insights

1. **Precise Problem Definition**: Characterizes feature misalignment in BEV fusion into local (depth projection error) and global (BEV spatial offset) levels, designing specific modules for each.
2. **Ingenious KD-Tree Neighborhood Depth**: Employs graph matching to correct depth estimations via neighborhood depths when projection depth is inaccurate, functioning as a robust depth estimation strategy.
3. **Adding noise during training while omitting it during inference** is analogous to data augmentation or adversarial training, which is simple and highly effective.
4. **Preserves the BEVFusion Paradigm**: Integrated as a plug-and-play module without modifying the underlying BEVFusion structure, ensuring high practicality.
5. **Highly Valuable for Practical Deployment**: Addresses a real-world engineering bottleneck where sensor calibration errors are inevitable in actual driving scenarios.

## Limitations & Future Work

1. KD-Tree neighborhood search may hit efficiency bottlenecks under large-scale point clouds; more efficient neighborhood construction strategies could be explored.
2. The random offset noise in GlobalAlign follows a uniform distribution. Real calibration errors, however, might follow specific patterns, and modeling a more realistic error distribution could be beneficial.
3. Evaluation is restricted to the nuScenes dataset; generalization should be further verified on other datasets like Waymo.
4. Potential extensions of LocalAlign to Transformer-based BEV methods (e.g., BEVFormer) could be investigated.
5. Temporal information has not been incorporated; misalignment correction in multi-frame fusion scenarios warrants further investigation.

## Related Work & Insights

- **BEVFusion (MIT/PKU)**: A standard unified LiDAR and Camera BEV framework, upon which GraphBEV enhances robustness.
- **ObjectFusion**: Also focuses on BEV alignment but changes the fusion paradigm (using RoI Pooling), whereas GraphBEV maintains the original paradigm for better generalizability.
- **MetaBEV**: Utilizes Cross Deformable Attention for alignment but ignores the depth errors occurring during view transformation.
- **Deformable Convolution / Grid Sampling**: The offset learning mechanism in GlobalAlign draws inspiration from deformable convolution.
- **Insight**: In multi-sensor fusion systems, robustness to sensor calibration errors is an underestimated yet extremely critical challenge.

## Rating
- Novelty: ⭐⭐⭐⭐ The diagnosis of feature misalignment in BEV fusion and the dual-level design of solutions are novel. The KD-Tree neighborhood depth idea is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering clean/noisy settings comparisons, module ablations, hyperparameter ablations, and robustness analysis under various weather conditions, distances, and object sizes.
- Writing Quality: ⭐⭐⭐⭐ Motivations are clear, methodology is extensively detailed, and figures are well-designed.
- Value: ⭐⭐⭐⭐ Addresses real-world engineering issues in actual deployment, serving as practical, plug-and-play modules.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection](fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)
- [\[ECCV 2024\] OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)
- [\[ECCV 2024\] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)
- [\[ECCV 2024\] MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection](monowad_weather-adaptive_diffusion_model_for_robust_monocular_3d_object_detectio.md)
- [\[ECCV 2024\] Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)

</div>

<!-- RELATED:END -->

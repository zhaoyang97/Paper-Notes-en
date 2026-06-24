---
title: >-
  [Paper Note] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision
description: >-
  [CVPR 2025][3D Vision][Hand Pose Estimation] EgoPressure introduces the first egocentric dataset for hand tactile pressure and pose estimation, containing 5 hours of RGB-D interaction data from 21 participants, high-fidelity MANO hand mesh annotations based on multi-view optimization, and ground-truth pressure mapping from pressure sensors. It also establishes benchmark models for estimating hand pressure and pose from RGB images.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Hand Pose Estimation"
  - "Pressure Sensing"
  - "Egocentric Vision"
  - "Multi-view Annotation"
  - "Hand-Object Interaction"
date: 2026-05-08
content_hash: 99d1eef66d8e8d61
---

# EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision

**Conference**: CVPR 2025  
**arXiv**: [2409.02224](https://arxiv.org/abs/2409.02224)  
**Code**: [https://yiming-zhao.github.io/EgoPressure/](https://yiming-zhao.github.io/EgoPressure/)  
**Area**: 3D Vision  
**Keywords**: Hand Pose Estimation, Pressure Sensing, Egocentric Vision, Multi-view Annotation, Hand-Object Interaction

## TL;DR

EgoPressure introduces the first egocentric dataset for hand tactile pressure and pose estimation, containing 5 hours of RGB-D interaction data from 21 participants, high-fidelity MANO hand mesh annotations based on multi-view optimization, and ground-truth pressure mapping from pressure sensors. It also establishes benchmark models for estimating hand pressure and pose from RGB images.

## Background & Motivation

**Background**: Understanding tactile contact and pressure information in hand-object interactions is crucial for AR/VR and robotic manipulation. Existing pressure estimation methods primarily rely on sensor gloves or robotic tactile sensors to obtain pressure labels, but these devices interfere with natural tactile feedback. Vision-based methods require no hand-worn devices and are easier to deploy on devices like smart glasses. The PressureVision dataset is one of the few vision datasets providing ground-truth pressure annotations, but it is limited to static third-person camera views.

**Limitations of Prior Work**: Existing datasets suffer from three critical limitations. First, they lack egocentric perspective—while cameras on smart glasses and VR headsets are egocentric, existing pressure datasets only cover static third-person views, making the models unable to generalize to egocentric scenarios. Second, they lack accurate hand pose annotations—contact information in most hand-object interaction datasets is inferred from the intersection of coarse hand poses and object meshes, which is inaccurate. Third, there is a lack of 3D localization for pressure information—existing methods predict the 2D projection of pressure on the image plane, which cannot localize the exact position of pressure on the 3D surface of the hand.

**Key Challenge**: Visual pressure estimation requires a large amount of training data with pressure labels, but precise pressure annotation requires expensive sensing equipment; meanwhile, hand self-occlusion is severe in egocentric views, and single-view hand pose estimation accuracy is insufficient to support precise pressure-to-hand mapping.

**Goal**: (1) Construct the first dataset featuring egocentric views, ground-truth pressure labels, and precise hand pose/mesh annotations; (2) develop a multi-view marker-less hand pose annotation method; (3) establish a benchmark for joint estimation of hand pressure and pose.

**Key Insight**: The authors design a capture platform consisting of 8 Azure Kinect RGB-D cameras (7 static + 1 head-mounted) and a Sensel Morph pressure touchpad. Accurate MANO hand meshes are obtained through multi-view differentiable rendering optimization, and pressure is then back-projected onto the UV texture map of the hand mesh using a virtual orthographic camera.

**Core Idea**: To build the first egocentric pressure dataset, obtain high-fidelity hand mesh annotations via multi-view differentiable rendering optimization, and map pressure as hand UV textures to achieve pressure localization in 3D space.

## Method

### Overall Architecture

The construction pipeline of EgoPressure comprises three stages: (1) Data Collection—21 participants perform 31 gestures on an 8-camera + pressure sensor platform; (2) Hand Pose Annotation—accurate MANO hand meshes are obtained using HaMeR initialization followed by multi-view differentiable rendering optimization; (3) Pressure Mapping—pressure data captured by the touchpad is back-projected onto the UV texture map of the hand mesh via a virtual orthographic camera. Benchmark evaluation is established across three types of models: RGB-only, RGB + hand pose, and PressureFormer, which jointly estimates hand mesh and pressure.

### Key Designs

1. **Multi-view Marker-less Annotation**:

    - **Function**: Obtain sub-millimeter level accuracy of MANO hand mesh annotations from 7 static camera views.
    - **Mechanism**: First, utilize HaMeR to estimate initial poses independently for each view, and resolve scale-translation ambiguity by triangulating root joints across the 7 views. Then perform two-stage optimization: (1) **Pose Optimization**—fix hand shape parameters $\beta$ and optimize pose $\theta$ and translation $t$ with the objective function $\mathcal{L}_{\text{pose}} = \mathcal{L}_{\mathcal{R}} + \mathcal{L}_{\text{insec}}$, where $\mathcal{L}_{\mathcal{R}}$ is the differentiable rendering loss across all views and $\mathcal{L}_{\text{insec}}$ prevents mesh self-intersection; (2) **Shape Refinement**—fix pose and introduce vertex displacement $D_{\text{vert}}$ to offset each vertex along the normal direction, incorporating geometric regularization and virtual rendering loss.
    - **Design Motivation**: Single-view hand pose estimation suffers from insufficient accuracy under self-occlusion, whereas joint multi-view optimization offers complementary observations from different angles. Vertex displacement allows the model to capture soft tissue deformations that cannot be represented by the parametric MANO model.

2. **Virtual Render for Contact and Pressure**:

    - **Function**: Map 2D pressure data from the touchpad plane to 3D pressure textures in the hand mesh UV space.
    - **Mechanism**: Place a virtual orthographic camera beneath the touchpad, pointing upwards and aligning the rendering plane with the touchpad surface. The optimization objective has two terms: (1) MSE loss between rendered pressure and ground-truth pressure; (2) depth consistency constraint from the hand mesh to the touchpad surface within contact regions. The loss function is: $\mathcal{L}_{\mathcal{R}^v} = \text{MSE}(\mathcal{R}^v_P(\Theta^*, \mathcal{T}_P), P_{\text{gt}}) + |\mathbb{I}(P_{\text{gt}}>0) \odot (\mathcal{R}^v_D(\Theta^*)[z] - Z_{v2p})|_1$
    - **Design Motivation**: Traditional pressure datasets only provide planar pressure maps, which cannot reveal which part of the hand is exerting how much pressure. Mapping the pressure back to the hand UV texture via a virtual camera can precisely localize pressure on each hand region in 3D space, providing more valuable information for applications like robotic grasping.

3. **PressureFormer**:

    - **Function**: Jointly estimate hand mesh and pressure UV maps from a single RGB image.
    - **Mechanism**: Built as an extension of the HaMeR architecture, it exploits HaMeR's ViT to extract image feature tokens. Taking hand vertices $V_{\text{hand}}$ as input tokens to a Transformer decoder, it performs cross-attention with the ViT features. Each output token is a $D$-dimensional feature vector, mapped to MANO's UV coordinate system to obtain a UV pressure feature map, which is then interpolated via two convolutional layers to predict the quantized pressure UV map $U_{\text{pred}}$. A differentiable renderer projects the pressure back to the image plane to calculate the pressure loss in the image space.
    - **Design Motivation**: Predicting pressure directly in the UV space is more meaningful than in the 2D image plane, as the UV space maps one-to-one with the hand surface and is invariant to camera perspective.

### Loss & Training

Loss of PressureFormer: $\mathcal{L}_{\mathcal{PF}} = w_1 \mathcal{L}_c + w_2 \mathcal{L}_p$, where $\mathcal{L}_c$ represents the coarse pressure classification loss in UV space, and $\mathcal{L}_p$ represents the pressure loss after projection to the image plane via differentiable rendering, both using cross-entropy. During training, data augmentation (translation, scaling, rotation) is performed on center-cropped hand images.

## Key Experimental Results

### Main Results

Comparison of pressure estimation with different input modalities (full dataset of 21 subjects):

| Model | Training View | Testing View | Input Modality | Contact IoU↑ | Vol. IoU↑ | MAE(Pa)↓ |
|------|---------|---------|---------|-------------|-----------|----------|
| PressureVisionNet | Ego | Ego | RGB | 55.73 | 38.64 | 53.60 |
| + HaMeR pose | Ego | Ego | RGB+pred pose | 56.25 | 40.52 | 55.23 |
| + GT pose | Ego | Ego | RGB+GT pose | 58.80 | 41.39 | 53.79 |
| PressureVisionNet | Exo(2345) | Exo(2345) | RGB | 62.11 | 44.73 | 43.15 |
| + GT pose | Exo(2345) | Exo(2345) | RGB+GT pose | 64.39 | 47.58 | 41.72 |
| PressureVisionNet | Exo(2345) | Exo(167) | RGB | 36.82 | 25.05 | 62.22 |
| + GT pose | Exo(2345) | Exo(167) | RGB+GT pose | 43.04 | 31.39 | 49.45 |

### Ablation Study

| Model | Contact IoU↑ | Vol. IoU↑ | MAE(kPa)↓ | Description |
|------|-------------|-----------|-----------|------|
| PressureVisionNet | 55.73 | 38.64 | 53.60 | RGB-only baseline |
| + HaMeR pose (2.5D) | 56.25 | 40.52 | 55.23 | Predicted hand pose as additional channel |
| + GT pose | 58.80 | 41.39 | 53.79 | Upper bound with GT hand pose |
| PressureFormer | **Best** | Comparable | Comparable | UV space pressure prediction |

### Key Findings
- Hand pose information indeed benefits pressure estimation: GT pose improves Vol. IoU by 5%+ (third-person view) and 2.75% (egocentric view).
- Hand pose also significantly enhances the model's cross-view generalization capability: Contact IoU of RGB+GT pose on unseen views improves from 36.82 to 43.04 (+6.22).
- Even when using the predicted HaMeR pose (non-GT), clear improvements are attained, indicating that the accuracy of current SOTA hand estimators is sufficient to provide benefits.
- PressureFormer is the first to achieve joint estimation of the hand mesh and 3D pressure distribution from a single RGB image, achieving the best performance in Contact IoU.
- Hand-pad contact is observed in approximately 45.1% of the frames, and a pressure threshold of 0.5 kPa is used to filter out diffused noise readings.

## Highlights & Insights
- **The capture platform design combining multi-view and pressure sensors** is highly systematic—featuring 8 synchronized Kinects + Sensel Morph + infrared PTP time synchronization + four tabletop textures, covering various dimensions of data quality.
- **The idea of mapping pressure as UV textures** pioneeringly elevates 2D sensor data into 3D space—no longer asking "where on the image is there pressure", but "which part of the hand exerts how much pressure". This is highly valuable for robots learning human grasping.
- **The complementarity between hand pose and pressure** is an important experimental finding—pressure provides "intensity" information while pose provides "spatial location" information. Combining the two enables a complete understanding of hand-object interactions, paving the way for future multimodal interaction interpretation.

## Limitations & Future Work
- It is restricted to planar touchpad interaction scenarios and cannot capture pressure distribution when grasping 3D objects, which calls for the development of curved pressure sensors.
- The data diversity from 21 participants is limited, with a narrow range of hand shape variations (concentrated MANO $\beta$ distribution) and an imbalanced gender ratio (6 females, 15 males).
- The annotation pipeline relies on 7 static cameras, and the capture platform is complex and expensive, making large-scale deployment and scaling difficult.
- PressureFormer quantizes pressure into discrete categories, thereby losing the precision of continuous pressure values; additionally, it depends on the hand estimation quality of HaMeR as an upper bound.
- All gestures are performed in controlled laboratory environments, showing a significant gap with real-use scenarios (such as kitchens, office desks, etc.).

## Related Work & Insights
- **vs PressureVision**: PressureVision is the first visual pressure dataset, but is confined to third-person static cameras and lacks hand pose annotations. EgoPressure introduces the egocentric perspective and high-precision hand meshes, serving as a natural and important extension.
- **vs ContactPose**: ContactPose obtains contact maps via thermal imaging but lacks pressure intensity information, and its contact annotations are based on thermal residue rather than real-time measurements. EgoPressure provides real-time, high-resolution pressure readings.
- **vs ARCTIC/GRAB**: The contact information in these datasets is inferred from intersections of hand-object meshes, which is limited by pose estimation errors. EgoPressure uses pressure sensors to obtain ground-truth contact and pressure.

## Rating
- Novelty: ⭐⭐⭐⭐ First egocentric pressure dataset with innovative annotation methods and UV pressure mapping
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage with multiple baseline models, cross-view generalization experiments, and joint estimation models
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely detailed dataset description, transparent collection details, and high reproducibility of methods
- Value: ⭐⭐⭐⭐⭐ Fills the blank of egocentric pressure datasets, providing a significant boost to research in AR/VR hand interactions and robotic manipulation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](../../CVPR2026/3d_vision/ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [\[CVPR 2025\] HD-EPIC: A Highly-Detailed Egocentric Video Dataset](hd-epic_a_highly-detailed_egocentric_video_dataset.md)
- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)
- [\[CVPR 2025\] MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond](motionpro_exploring_the_role_of_pressure_in_human_mocap_and_beyond.md)

</div>

<!-- RELATED:END -->

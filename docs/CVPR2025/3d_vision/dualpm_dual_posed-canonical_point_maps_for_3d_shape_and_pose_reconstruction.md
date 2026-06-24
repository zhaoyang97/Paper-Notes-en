---
title: >-
  [Paper Note] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction
description: >-
  [3D Vision] This work proposes the Dual Point Maps (DualPM) representation, which predicts a pair of point maps (camera-space $P$ and canonical-space $Q$) from a single image. This simplifies the 3D shape and pose reconstruction of deformable objects into a point map prediction problem. Additionally, a layered amodal point map is introduced to achieve complete shape recovery (including self-occluded parts), generalizing to real-world images while training with only 1–2 synthe…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 1363b207e363d03b
---

# DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction

## TL;DR

This work proposes the Dual Point Maps (DualPM) representation, which predicts a pair of point maps (camera-space $P$ and canonical-space $Q$) from a single image. This simplifies the 3D shape and pose reconstruction of deformable objects into a point map prediction problem. Additionally, a layered amodal point map is introduced to achieve complete shape recovery (including self-occluded parts), generalizing to real-world images while training with only 1–2 synthetic 3D models.

## Background & Motivation

In 3D computer vision, the choice of data representation is crucial for the success of deep learning. DUSt3R has recently demonstrated the power of point maps—mapping pixels to 3D points allows unifying matching, camera estimation, and triangulation in static scene reconstruction into point map prediction.

**Core Problem**: Can a similar idea be applied to a completely different problem—monocular 3D shape and pose reconstruction of **deformable objects**?

Key challenges:
- A single point map $P$ only gives 3D positions, failing to encode the deformation or pose information of the object.
- Recovering the deformation field requires knowing the position of each point in the "neutral pose."
- Existing methods (e.g., MagicPony, 3D-Fauna, Farm3D) rely on weakly-supervised learning of category-level templates or massive collections of real image data.

**Key Insight**: If **two** point maps are predicted—one representing the 3D position $P$ in camera space, and the other representing the position $Q$ in the canonical space (neutral pose)—then the deformation field is simply the difference $P - Q$, making pose recovery a trivial operation.

## Method

### Overall Architecture

Given an input image $I$ and an object mask $M$, the framework first extracts features $F$ using a pretrained feature extractor $\Psi$ (such as DINOv2). It then: (1) predicts the canonical point map $Q = \Phi_Q(\Psi(I))$ from $F$; (2) predicts the posed point map $P = \Phi_P(Q)$ conditioned on $Q$. The representation is also extended to a layered amodal format, which reconstructs self-occluded regions via multi-layer point maps to achieve complete 3D reconstruction.

### Key Designs

#### 1. Dual Point Maps

- **Function**: Unifies the encoding of both 3D shape and pose information of deformable objects, rendering pose recovery and cross-image matching trivial operations.
- **Mechanism**: Each pixel $\boldsymbol{u}$ is mapped to two 3D points: $\boldsymbol{p} = P(\boldsymbol{u})$ (the position in camera space under the current pose) and $\boldsymbol{q} = Q(\boldsymbol{u})$ (the position in canonical space under the neutral pose). The deformation field is defined as $P - Q$, and cross-image matching is achieved by comparing $Q$ (since $Q$ is invariant to viewpoints and poses).
- **Design Motivation**: A single point map cannot distinguish between rigid motion and deformation. Dual point maps decouple the two through a canonical frame of reference, similar to NOCS (Normalized Object Coordinates) but extended to deformable objects.

#### 2. Layered Amodal Point Maps

- **Function**: Reconstructs the complete 3D shape of objects, including self-occluded, invisible parts.
- **Mechanism**: Each pixel is associated with **multiple** intersection points along the ray passing through the object's surface, forming an ordered sequence $(\boldsymbol{p}_1, \boldsymbol{p}_2, \ldots)$. A layered representation is utilized where the first layer encodes visible points, and subsequent layers encode points occluded by preceding layers, while also predicting opacity $\sigma$. Each pixel outputs $2K$ layers of points (paired as entry and exit rays), totaling $3 \times 2 + (3+1) \times 2 \times (K-1)$ scalars.
- **Design Motivation**: Standard point maps only reconstruct visible parts. Incomplete reconstructions limit downstream applications such as animation and skeleton fitting. The layered representation is analogous to depth peeling in computer graphics, offering a natural and efficient solution.

#### 3. Cascaded Prediction Architecture (Q → P)

- **Function**: Enhances model generalizability, particularly on out-of-distribution images.
- **Mechanism**: The canonical point map $Q$ is predicted first (leveraging pretrained DINOv2 features), and the posed point map $P$ is subsequently predicted conditioned on $Q$, rather than predicting both directly from raw image features.
- **Design Motivation**: Predicting $Q$ is essentially a pixel labeling problem (since $Q$ is pose-invariant), which is relatively simpler and generalizes easily. Using $Q$ as a conditioning input for $P$, rather than raw image features, reduces overfitting to appearance and improves generalization to novel poses and appearances. Ablation studies validate that this design is critical for out-of-distribution generalization.

### Loss & Training

A self-calibrated L2 loss (from DUSt3R) is used:

$$\mathcal{L}_P = \frac{1}{|M|}\sum_{\boldsymbol{u} \in M} c_P(\boldsymbol{u})\|\hat{P}(\boldsymbol{u}) - P(\boldsymbol{u})\|^2 - \alpha \log c_P(\boldsymbol{u})$$

where $c_P(\boldsymbol{u}) > 0$ is the pixel-wise confidence. $\mathcal{L}_Q$ is defined similarly. An additional loss $\mathcal{L}_\sigma$ is introduced to supervise the opacity prediction of the amodal layers.

## Key Experimental Results

### Main Results (PASCAL VOC + Animodel)

| Method | PCK@0.1 (Horse) ↑ | Chamfer (Horse, cm) ↓ | Chamfer (Cow) ↓ | Chamfer (Sheep) ↓ |
|------|-------------------|----------------------|-----------------|-------------------|
| A-CSM | 32.9 | 11.75 | 9.52 | 9.24 |
| MagicPony | 42.9 | 11.19 | 10.29 | — |
| Farm3D | 49.1 | 11.34 | 9.63 | 11.01 |
| 3D-Fauna | 53.9 | 11.86 | 10.54 | 9.61 |
| Trellis | — | 6.93 | 6.80 | 5.91 |
| **DualPM (ours)** | **73.2** | **4.30** | **3.18** | **3.30** |

### Ablation Study

| Design Choices | PCK (Horse) |
|---------|-------------|
| P predicted from image features (w/o Q) | Significant drop |
| P predicted conditioned on Q (Full Model) | **73.2** |
| Without amodal extension | Reconstructs visible parts only |

### Key Findings

- **PCK Improvement of 20 Percentage Points**: Performance increases to 73.2% compared to the strongest baseline, 3D-Fauna (53.9%).
- **Halved Chamfer Distance**: The Chamfer distance for Horses is significantly reduced from 6.93cm (Trellis) to 4.30cm.
- **Extremely Low Synthetic Data Training Requirements**: Trained using synthetic data rendered from only 1–2 3D models per category, yet successfully generalizes to real images and out-of-distribution samples (such as foals and donkeys).
- **Crucial Role of Cascaded Prediction**: The sequential prediction from Q to P significantly improves out-of-distribution generalization compared to directly predicting both from the image.
- **Skeleton Fitting**: The deformation field produced by DualPM can be directly utilized to fit 3D articulated skeletons, facilitating motion transfer and animation.

## Highlights & Insights

1. **Conceptual Elegance**: Generalizes the point map concept of DUSt3R to deformable object reconstruction, where the dual point map design is natural—pose is simply the delta between the two spaces.
2. **Power of Problem Reduction**: Simplifies the complex pipeline of "3D reconstruction + pose estimation + matching + skeleton fitting" into predicting two dense maps, which is highly network-friendly and unified.
3. **Amodal Extension**: The layered depth peeling approach elegantly resolves the self-occlusion problem, integrating seamlessly with the training pipeline.
4. **Exceptional Data Efficiency**: Training with just 1–2 synthetic models outperforms prior works trained on massive real-world data, proving that the choice of representation is more critical than raw data scale.
5. **Canonical Point Map as Features**: The canonical map $Q$ itself serves as an outstanding feature map, which is more suitable than raw DINOv2 features as a conditioning input for predicting $P$.

## Limitations & Future Work

1. **Category-Specific Models**: Separate training is required for each animal category. Although it exhibits preliminary zero-shot cross-category capability, it is not yet fully mature.
2. **Validated Only on Quadrupeds**: The method has not yet been extended to other deformable object categories, such as humans or birds.
3. **Reliance on Mask Inputs**: Object segmentation masks must be provided beforehand.
4. **Synthetic-to-Real Domain Gap**: Despite strong generalization, it may still fail in extreme scenarios, such as heavily occluded scenes.
5. **Fixed Number of Amodal Layers**: The pre-defined value of $K$ limits the capacity to model objects with complex topologies.

## Related Work & Insights

- **DUSt3R/MASt3R**: The originator of the point map concept, proving that point map prediction can simplify various stages of SfM.
- **NOCS**: The predecessor of canonical space coordinates, but limited to rigid objects.
- **3D-Fauna/Farm3D/MagicPony**: Prior SOTA methods in deformable object reconstruction, which rely heavily on weakly-supervised learning or massive real-world datasets.
- **Insight**: Great representation design can translate hard problems into simpler ones—DualPM makes the learning task substantially easier for the network through dual point maps, allowing it to surpass complex weakly-supervised methods even when trained on extremely sparse synthetic data.

## Rating

⭐⭐⭐⭐⭐

Conceptually elegant, reducing a highly complex problem into a concise representation prediction task. Quantitative results dramatically surpass all prior SOTA methods (PCK +20pp, halved Chamfer distance). The data efficiency is remarkable. Both the mathematical intuition and the engineering implementation of the dual point maps are exceptional.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)
- [\[CVPR 2025\] Multi-View Pose-Agnostic Change Localization with Zero Labels](multi-view_pose-agnostic_change_localization_with_zero_labels.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](dune_distilling_a_universal_encoder_from_heterogeneous_2d_and_3d_teachers.md)
- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)

</div>

<!-- RELATED:END -->

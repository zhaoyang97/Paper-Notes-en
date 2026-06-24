---
title: >-
  [Paper Note] HandOS: 3D Hand Reconstruction in One Stage
description: >-
  [CVPR 2025][3D Vision][Hand Reconstruction] HandOS proposes an end-to-end, single-stage 3D hand reconstruction framework that unifies hand detection, 2D pose estimation, and 3D mesh reconstruction into a single pipeline. By freezing a pre-trained detector and introducing an interactive 2D-3D decoder, it eliminates the redundant calculation and cumulative errors of classical multi-stage methods, achieving state-of-the-art performance with a PA-MPJPE of 5.0 on FreiHand.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Hand Reconstruction"
  - "One-Stage Detection"
  - "2D-3D Interactive Decoding"
  - "End-to-End Pose Estimation"
  - "DETR"
date: 2026-05-08
content_hash: 66c21e0b3cdd5c49
---

# HandOS: 3D Hand Reconstruction in One Stage

**Conference**: CVPR 2025  
**arXiv**: [2412.01537](https://arxiv.org/abs/2412.01537)  
**Code**: [https://github.com/idea-research/HandOS](https://github.com/idea-research/HandOS)  
**Area**: 3D Vision  
**Keywords**: Hand Reconstruction, One-Stage Detection, 2D-3D Interactive Decoding, End-to-End Pose Estimation, DETR

## TL;DR
HandOS proposes an end-to-end, single-stage 3D hand reconstruction framework that unifies hand detection, 2D pose estimation, and 3D mesh reconstruction into a single pipeline. By freezing a pre-trained detector and introducing an interactive 2D-3D decoder, it eliminates the redundant calculation and cumulative errors of classical multi-stage methods, achieving state-of-the-art performance with a PA-MPJPE of 5.0 on FreiHand.

## Background & Motivation

1. **Background**: Existing hand reconstruction methods commonly adopt multi-stage frameworks: detecting hand locations first, then classifying left/right hands, and finally estimating pose. This pipelined design has been widely used for years.

2. **Limitations of Prior Work**: The multi-stage pipeline brings two serious problems: computational redundancy and error accumulation. For instance, on the HInt test set, the error rate of detection and left/right classification is as high as 11.2%, meaning these samples are doomed to fail before pose estimation even begins.

3. **Key Challenge**: Hands typically occupy very small regions in an image, necessitating a detector for localization and zooming; additionally, hand pose representations are symmetric rather than identical for left and right hands, which usually requires left/right classification to flip and unify. Ironically, these "necessities" introduce substantial complexity to the pipeline.

4. **Goal**: How to design an end-to-end framework to directly perform hand detection and 3D reconstruction from a full image without relying on left/right hand classification?

5. **Key Insight**: Freeze a pre-trained object detector as a foundation model, and append lightweight modules to "upgrade" detection capabilities into 3D reconstruction capabilities. Use a unified keypoint representation (rather than MANO parameters) to represent both left and right hands.

6. **Core Idea**: Utilize the detection queries provided by the frozen detector and jointly learn 2D joints, 3D vertices, and camera translation in an interactive decoder through instance-to-joint expansion and 2D-to-3D lifting.

## Method

### Overall Architecture
The input is a single full image (long edge resized to 1280), which is passed through a frozen Grounding DINO 1.5 detector to obtain multi-scale features, detection bounding boxes, and category scores. Then, Side Tuning is employed to generate supplementary features, which are finally fed into an interactive 2D-3D decoder to simultaneously output 2D joint coordinates $\mathbf{J}^{2D} \in \mathbb{R}^{J \times 2}$, 3D vertices $\mathbf{V} \in \mathbb{R}^{V \times 3}$ ($J=21, V=778$), and camera translation $\mathbf{t} \in \mathbb{R}^3$. The 3D joints are obtained from the vertices using the joint regression matrix of MANO.

### Key Designs

1. **Side Tuning**:

    - **Function**: To provide keypoint-related supplementary features for the frozen detector.
    - **Mechanism**: The detector weights are entirely frozen to preserve detection capability; however, its features are insufficient for keypoint estimation. Thus, a learnable lightweight network is designed to take the shallow features of the vision backbone as input to generate supplementary features $\mathbf{F}^s$, which are then concatenated with the encoded detector features.
    - **Design Motivation**: The freezing + side-tuning strategy retains object detection performance, speeds up training convergence, and facilitates training across different datasets.

2. **Instance-to-Joint Query Expansion + 2D-to-3D Query Lifting**:

    - **Function**: To progressively expand instance-level detection queries into 2D joint queries, and subsequently lift them into 3D vertex queries.
    - **Mechanism**: First, a SimOTA assigner selects positive samples from $Q$ detection queries. Then, a learnable joint embedding $\mathbf{e}^Q \in \mathbb{R}^{J \times d^q}$ is added to each instance query, expanding it into 21 2D-joint queries. The reference boxes are also derived from the detection boxes. Next, a learnable lifting matrix $\mathbf{L} \in \mathbb{R}^{(V+1) \times J}$ (initialized using MANO skinning weights) is designed to lift the 2D joint queries into 778 3D vertex queries and 1 camera translation query via a linear combination through Einstein summation.
    - **Design Motivation**: Detection queries already embody the semantic localization of hands, making deriving joint queries from them far more efficient than learning from scratch. The lifting matrix initialized with MANO skinning weights provides a strong structural prior, facilitating a physically meaningful 2D-to-3D transformation.

3. **Hierarchical Attention**:

    - **Function**: To coordinate attention interaction among three types of queries (2D joints, 3D vertices, and camera translation) within the interactive decoder layers.
    - **Mechanism**: The decoder consists of 6 layers, where the first 2 are 2D layers (handling only 2D joints) and the remaining 4 are interaction layers. The core of the interaction layer is the hierarchical attention mechanism: 2D joint and 3D vertex queries can attend to each other (exchanging complementary features) but do not interact with the camera translation query; the camera translation query can attend to both 2D and 3D queries (since it requires location and geometric information). This asymmetric interaction is enforced via an attention mask.
    - **Design Motivation**: 2D joints and 3D vertices should be invariant to absolute translation and scale, whereas camera parameters are sensitive to both. Enabling full-attention connectivity among all three would conflate these distinct invariance properties.

### Loss & Training
2D supervision: L1 loss + OKS (Object Keypoint Similarity). 3D supervision: vertex L1, joint L1, normal consistency $\mathcal{L}^{normal}$, and edge length consistency $\mathcal{L}^{edge}$—the latter two are critical for maintaining plausible geometry during model-free inference (without MANO). Weak supervision: 2D annotations constrain mesh learning through projection error and normal consistency, addressing the lack of 3D annotations in in-the-wild datasets. An elegant design uses the normal direction as a left/right hand indicator—the normal directions reverse when the right-hand face topology is applied to left-hand vertices.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|--------|----------|------|
| FreiHand | PA-MPJPE↓ | **5.0** | 5.7 (MobRecon/Zhou) | -12.3% |
| FreiHand | PA-MPVPE↓ | **5.3**| 5.5 (Hamba) | -3.6% |
| FreiHand | F@5↑ | **0.812** | 0.798 (Hamba) | +1.8% |
| HO3Dv3* | PA-MPJPE↓ | **6.8** | 6.9 (Hamba*) | -1.4% |
| DexYCB | PA-MPJPE↓ | **5.2** | 5.5 (Zhou) | -5.5% |
| HInt-Ego4D | PCK@0.05↑ | **64.6** | 51.6 (HaMeR) | +25.2% |

### Ablation Study

| Configuration | PA-MPJPE | PA-MPVPE | Description |
|------|---------|---------|------|
| Full model | 5.0 | 5.3 | Full HandOS |
| Training with flipped left-hand | 5.3 | 5.6 | Joint left/right training yields only marginal performance decrease |
| Using GT bounding boxes | - | - | Multi-stage methods rely heavily on ideal detection assumptions |

### Key Findings
- HandOS achieves substantial gains on HInt-Ego4D (+13pp PCK), demonstrating that the end-to-end paradigm is far superior to multi-stage methods in complex scenarios (egocentric view, hand-object interaction).
- Jointly training left and right hands only incurs marginal performance loss (5.0 to 5.3), validating the feasibility of a unified representation.
- Normal consistency and edge-length losses are crucial for maintaining mesh quality during model-free inference (without MANO parameters).
- Evaluation on HO3Dv3 highlights that the GT bounding box assumption in multi-stage methods is unrealistic: actual detected boxes are significantly smaller than GT boxes under severe occlusion.

## Highlights & Insights
- **Frozen detector + side-tuning**: This strategy retains original detection capabilities while efficiently adapting to new tasks. It can be generalized to any scenario where new output heads are added to pre-trained detectors.
- **Normal direction as left/right hand indicator**: Leverages mesh geometric properties to avoid explicit classification. When the right-hand face topology template is applied to left-hand vertices, the normal vectors reverse. This observation is highly elegant.
- **Two-step paradigm of query expansion and lifting**: The progressive refinement from instances to 2D joints and then to 3D vertices is transferrable to any task requiring structured outputs derived from object detection results.

## Limitations & Future Work
- Reliance on the heavy Grounding DINO 1.5 detector limits inference speed.
- Only single hand reconstruction is demonstrated; two-hand interactive scenarios are not addressed.
- The training data scale (204K) is much smaller than HaMeR (2,749K), indicating that the architectural design compensates for the data gap, though more data may yield further improvements.
- The frozen detector strategy might limit feature adaptation specifically for hand tasks.

## Related Work & Insights
- **vs HaMeR**: HaMeR uses a multi-stage framework with a large ViT and 2.7M data. HandOS achieves comparable or superior results with only 1/13 of the data size via a single-stage design.
- **vs Hamba**: Hamba introduces graph-guided Mamba for a parametric hand framework, which is also multi-stage, whereas HandOS directly regresses vertices in a more straightforward manner.
- **vs EDPose/PETR**: HandOS borrows the single-stage concept (query expansion) from 2D human pose estimation, and extends it to 3D hand mesh reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐ The single-stage hand reconstruction concept and the interactive 2D-3D decoder design are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 datasets, including ablation studies and visual analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed method descriptions.
- Value: ⭐⭐⭐⭐ High practical value for end-to-end hand reconstruction, especially for AR/VR applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MV-DUSt3R(+): Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds](mv-dust3r_single-stage_scene_reconstruction_from_sparse_views_in_2_seconds.md)
- [\[CVPR 2025\] Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass](fast3r_towards_3d_reconstruction_of_1000_images_in_one_forward_pass.md)
- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)
- [\[CVPR 2025\] StageDesigner: Artistic Stage Generation for Scenography via Theater Scripts](stagedesigner_artistic_stage_generation_for_scenography_via_theater_scripts.md)
- [\[CVPR 2026\] OMGTex: One-stage Multi-style Facial Texture Reconstruction without Geometry Guidance](../../CVPR2026/3d_vision/omgtex_one-stage_multi-style_facial_texture_reconstruction_without_geometry_guid.md)

</div>

<!-- RELATED:END -->

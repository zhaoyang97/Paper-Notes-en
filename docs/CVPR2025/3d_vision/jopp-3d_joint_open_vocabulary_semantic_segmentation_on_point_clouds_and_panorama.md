---
title: >-
  [Paper Note] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas
description: >-
  [CVPR 2025][3D Vision][Open-vocabulary segmentation] This work proposes the JOPP-3D framework, which achieves the first joint open-vocabulary semantic segmentation of 3D point clouds and panoramic images by tangentially decomposing panoramas into perspective images and leveraging SAM+CLIP for 3D instance-semantic alignment. It outperforms existing methods on the Stanford-2D-3D-s and ToF-360 datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Open-vocabulary segmentation"
  - "point cloud semantic segmentation"
  - "panoramic images"
  - "vision-language models"
  - "cross-modal alignment"
date: 2026-05-08
content_hash: 97cb4df42e3876c8
---

# JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas

**Conference**: CVPR 2025  
**arXiv**: [2603.06168](https://arxiv.org/abs/2603.06168)  
**Code**: To be confirmed  
**Area**: 3D Vision  
**Keywords**: Open-vocabulary segmentation, point cloud semantic segmentation, panoramic images, vision-language models, cross-modal alignment

## TL;DR
This work proposes the JOPP-3D framework, which achieves the first joint open-vocabulary semantic segmentation of 3D point clouds and panoramic images by tangentially decomposing panoramas into perspective images and leveraging SAM+CLIP for 3D instance-semantic alignment. It outperforms existing methods on the Stanford-2D-3D-s and ToF-360 datasets.

## Background & Motivation
**Background**: Semantic segmentation methods are often limited to a single modality of 2D images or 3D point clouds, relying heavily on large-scale annotated data with pre-defined categories.

**Limitations of Prior Work**: (a) Fixed-label models fail to generalize to novel categories; (b) VLMs (e.g., CLIP, SAM) perform well on perspective views but cannot be directly applied to panoramas (due to geometric distortion) or 3D point clouds; (c) existing methods do not jointly address open-vocabulary segmentation on panoramas and 3D point clouds.

**Key Challenge**: Panoramas provide 360° coverage but suffer from distortion, while 3D point clouds offer geometric-fidelity but lack texture semantics. The two modalities are complementary but have not been unified.

**Goal**: To simultaneously perform language-driven semantic segmentation on both panoramas and 3D point clouds without annotated data.

**Key Insight**: Decomposing panoramas into 20 perspective sub-images using regular icosahedron projection allows VLMs to be directly applied, followed by propagating 3D semantics back to the panoramic domain via depth correspondences.

**Core Idea**: Tangential decomposition resolving panoramic distortion + 3D instance masks aggregating CLIP features + depth correspondence propagating semantic labels.

## Method

### Overall Architecture
The inputs are a set of panoramic RGB-D images of a scene (with poses), and the outputs are dense semantic segmentation maps of the 3D point cloud and the panoramic images. The workflow consists of three steps: (1) Tangential Decomposition: panorama $\rightarrow$ 20 perspective subviews + 3D point cloud; (2) 3D Instance Extraction and CLIP Semantic Alignment; (3) 3D-to-Panoramic Semantic Back-projection.

### Key Designs

1. **Tangential Decomposition**:

    - **Function**: Projects panoramic images onto the 20 faces of a regular icosahedron to generate perspective subviews.
    - **Mechanism**: Employs perspective projection with a 100° FoV (wider than the 73.1° in prior work) to map the panoramic sphere onto each face: $(x_r,y_r,z_r) = R_{r,local}[(i_r-W/2)/f, (j_r-H/2)/f, 1]^T$, which is then converted back to spherical coordinates to sample the panorama and depth. Depth is corrected via Z-depth to ensure geometric accuracy. Overlap between views is modeled to reduce boundary discontinuities.
    - **Design Motivation**: VLMs are native to perspective images and perform poorly on panoramic distortions; the 20 faces of a regular icosahedron provide uniform coverage and overlapping regions.

2. **3D Instance Extraction and Semantic Alignment**:

    - **Function**: Extracts class-agnostic instances from 3D point clouds and assigns open-vocabulary semantics using CLIP.
    - **Mechanism**: Extracting 3D instance masks $\mathcal{M}_j$ using Mask3D (supervised) or SAM3D (unsupervised) $\rightarrow$ projecting 3D points onto tangential perspective views to obtain 2D coordinates $\rightarrow$ selecting the Top-K views with the most matching points $\rightarrow$ segmenting instance crops in 2D using SAM $\rightarrow$ encoding the masked crops using CLIP $\rightarrow$ averaging the CLIP features across the K views to obtain the instance semantic embedding $\mathbf{e}_j^{3D}$.
    - **Design Motivation**: Point-wise CLIP embeddings are highly noisy, whereas instance-level aggregation is more stable. Masked cropping avoids semantic contamination from surrounding objects. Experiments show that full cropping results in a 5.1 point drop in mIoU compared to masked cropping, indicating that background noise severely interferes with CLIP features. Multi-view aggregation (across K views) further improves robustness, illustrating the intuition that "seeing more leads to higher accuracy".

3. **3D-to-Panoramic Semantic Propagation**:

    - **Function**: Back-projects 3D semantic labels onto panoramic images.
    - **Mechanism**: Panoramic depth pixels are back-projected to 3D coordinates via $\mathbf{X}^p(u,v) = R^p \Pi^{-1}(u,v,D^p(u,v)) + \mathbf{t}^p$, and nearest neighbor matching is performed with the semantic point cloud to obtain labels. For depth-discontinuous regions like doorways and corridors, depth correspondence consistency across neighboring panoramas is introduced to match pixels with a depth difference $< \delta_d$ to propagate labels.
    - **Design Motivation**: Direct nearest neighbor mapping produces incomplete segmentation at depth discontinuities. Cross-scene depth correspondences effectively fill empty regions.

### Loss & Training
Two paradigms are proposed: (1) a weakly-supervised version (using pretrained Mask3D for instance extraction); (2) an unsupervised version (using SAM3D for zero-training instance extraction). Semantic alignment relies entirely on the zero-shot capability of CLIP and does not require any semantic labels. During inference, given a list of text categories, text embeddings are extracted via the CLIP text encoder and cosine similarities are computed with the instance semantic embeddings to achieve open-vocabulary classification. The Top-K view selection uses $K=5$, the resolution of the tangential face is $480 \times 480$, and the depth correspondence threshold $\delta_d = 0.1$m.

## Key Experimental Results

### Main Results (Stanford 2D-3D-s)

| Method | 3D mIoU↑ | 2D mIoU↑ | Type |
|------|---------|---------|------|
| PointNet (Closed-set) | 41.1 | — | Supervised |
| OpenScene | 46.5 | — | Open-vocabulary |
| **JOPP-3D (Mask3D)** | **57.2** | **48.3** | Weakly-supervised |
| **JOPP-3D (SAM3D)** | **51.8** | **43.7** | Unsupervised |
| OPS (Panoramic) | — | 37.2 | Open-vocabulary |

### ToF-360 Dataset

| Method | 3D mIoU↑ | 2D mIoU↑ |
|------|---------|----------|
| OpenScene | 38.7 | — |
| JOPP-3D (Mask3D) | 44.9 | 39.5 |
| JOPP-3D (SAM3D) | 41.2 | 36.8 |

### Ablation Study

| Configuration | 3D mIoU | Description |
|------|---------|------|
| Full (masked crop) | 57.2 | Full model |
| w/o mask (full crop) | 52.1 | No masked cropping $\rightarrow$ semantic contamination |
| K=1 view | 49.8 | Reduced view aggregation $\rightarrow$ unstable features |
| Cube projection | 50.3 | vs. regular icosahedron $\rightarrow$ uneven coverage |

### Key Findings
- Joint 3D and panoramic segmentation outperforms individual 3D or 2D methods—the cross-modal information is complementary, where 3D provides geometric structure and 2D provides textured semantics.
- The difference between masked crop and full crop is significant (+5.1 mIoU), demonstrating that masked cropping is crucial for the quality of CLIP embeddings.
- Tangential decomposition on a regular icosahedron outperforms cube projection—the 20 faces offer more uniform coverage, and the 100° FoV provides better context coverage, whereas the 6 faces of a cube suffer from severe distortion at corners and edges.
- The weakly-supervised version (Mask3D) delivers superior performance but requires pretrained data, whereas the unsupervised version (SAM3D) is fully training-free but performs slightly lower. These two paradigms satisfy different deployment requirements.
- Depth correspondence propagation is highly effective at depth discontinuities such as doorways and corridors, filling in around 8% of label holes.

## Highlights & Insights
- **Tangential decomposition** is an elegant solution to handle panoramic images with VLMs: it does not require training a deformation adapter, relying purely on geometric transformations, and is training-free. This approach can be generalized to any scenario where perspective-pretrained models need to be applied to panoramic images.
- The **3D $\rightarrow$ 2D $\rightarrow$ CLIP $\rightarrow$ 3D loop** is elegantly designed: 3D provides spatial structure, while 2D provides semantic recognition capabilities, and both are connected via projection and back-projection.
- **Depth correspondence propagation** resolves semantic discontinuities at boundary regions between neighboring scenes. It is simple yet highly practical.

## Limitations & Future Work
- It is dependent on the accuracy of the input depth maps—depth estimation errors propagate to both 3D reconstruction and semantic propagation.
- 3D instance extraction remains a bottleneck—Mask3D requires pretraining, while SAM3D requires careful hyperparameter tuning.
- Nearest neighbor semantic propagation (Eq. 13-14) can be inaccurate in geometrically complex areas.
- Evaluation is limited to indoor scenes; applicability to large-scale outdoor environments remains unverified.
- Tangential decomposition generates 20 subviews, each requiring forward passes through SAM and CLIP, leading to high inference overhead and making real-time applications challenging.

## Related Work & Insights
- **vs. OpenScene**: Both perform 3D open-vocabulary segmentation, but OpenScene does not handle panoramic images, and its point-wise CLIP embeddings suffer from high noise. JOPP-3D achieves higher mIoU through instance-level aggregation and joint panoramic segmentation.
- **vs. OPS**: Performs panoramic open-vocabulary segmentation, but requires training a deformation adapter and does not support 3D. JOPP-3D is training-free and performs joint 3D+2D segmentation.
- **vs. OpenMask3D**: Focuses on 3D instance segmentation, whereas ours focuses on scene-level semantic segmentation covering large structures like walls and floors.
- **Difference from LEGaussians**: LEGaussians achieves open-vocabulary 3D understanding via language-embedded Gaussians and requires training the 3D representation; JOPP-3D is completely training-free and directly leverages 2D VLM capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ First joint panoramic and 3D open-vocabulary segmentation, with engineering innovation in the tangential decomposition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on two datasets, extensive ablation studies, and comparisons between weakly-supervised and unsupervised paradigms.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and comprehensive formulation.
- Value: ⭐⭐⭐⭐ High practical value for indoor scene understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation](mosaic3d_foundation_dataset_and_model_for_open-vocabulary_3d_segmentation.md)
- [\[CVPR 2025\] Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](../../NeurIPS2025/3d_vision/segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation
description: >-
  [CVPR 2025][Segmentation][Multi-view Human Pose Estimation] MV-SSM introduces State Space Models (Mamba) to multi-view 3D human pose estimation for the first time. By explicitly modeling joint spatial sequences at both the feature and keypoint levels through the Projective State Space (PSS) block, combined with Grid Token-guided Bidirectional Scanning (GTBS), it achieves 93.5 AP25 on CMU Panoptic and significantly outperforms prior SOTA methods in cross-camera and cross-scene…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Multi-view Human Pose Estimation"
  - "State Space Model"
  - "Mamba"
  - "Cross-view Generalization"
  - "Projective Attention"
date: 2026-05-08
content_hash: e934c7f7b7667b57
---

# MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2509.00649](https://arxiv.org/abs/2509.00649)  
**Code**: [https://aviralchharia.github.io/MV-SSM](https://aviralchharia.github.io/MV-SSM) (Project Page)  
**Area**: Human Understanding  
**Keywords**: Multi-view Human Pose Estimation, State Space Model, Mamba, Cross-view Generalization, Projective Attention

## TL;DR
MV-SSM introduces State Space Models (Mamba) to multi-view 3D human pose estimation for the first time. By explicitly modeling joint spatial sequences at both the feature and keypoint levels through the Projective State Space (PSS) block, combined with Grid Token-guided Bidirectional Scanning (GTBS), it achieves 93.5 AP25 on CMU Panoptic and significantly outperforms prior SOTA methods in cross-camera and cross-scene generalization tests.

## Background & Motivation

1. **Background**: End-to-end methods for multi-view 3D human pose estimation (e.g., MvP, MVGFormer) use attention-based Transformers to fuse multi-view features, achieving decent accuracy. Traditional multi-stage methods first detect 2D keypoints and then perform triangulation, whose accuracy is limited by matching algorithms and error accumulation.

2. **Limitations of Prior Work**: (a) Attention-based methods tend to overfit to specific camera configurations and visual scenes during training, causing a severe drop in performance when applied to new camera counts/locations; (b) MvP fails almost completely in cross-camera generalization tests (AP25 drops to 0); (c) Cross-attention operations over all tokens are computationally expensive and do not sufficiently model joint spatial relations under occluded scenarios.

3. **Key Challenge**: The attention mechanism of existing methods lacks explicit modeling of the inherent structure of joint spatial sequences, leading to poor generalization under unseen camera configurations.

4. **Goal**: Design a multi-view 3D pose estimation framework with robust generalization capability against variations in camera configurations.

5. **Key Insight**: State Space Models (SSMs) are naturally adept at capturing long-range dependencies of sequential elements. The authors observe that there is an inherent spatial sequence relationship among joints (such as the kinematic chain of the human body), and SSMs can model this sequence relationship at both the feature and keypoint levels.

6. **Core Idea**: Replace pure attention with Mamba's selective scanning mechanism to model multi-view joint spatial sequence relationships, while integrating projective attention for multi-view feature fusion.

## Method

### Overall Architecture
Given RGB images from $T$ camera views as input, multi-scale features are extracted using a ResNet-50 backbone, and keypoint predictions are progressively refined through multi-layer stacked Projective State Space (PSS) blocks. Each layer of the PSS block outputs 2D joint offsets and confidence scores, followed by differentiable algebraic triangulation to obtain 3D keypoints. A hierarchical token scheme is adopted to reduce the search space.

### Key Designs

1. **Projective State Space (PSS) Block**:

    - **Function**: Jointly use projective attention and state space modeling to learn joint spatial sequences and fuse cross-view information.
    - **Mechanism**: Each PSS block consists of two parts: the SS2D block (a visual adaptation of Mamba) and projective attention. Projective attention projects 3D keypoints to each view to obtain anchor points, and samples deformable points around anchor points to aggregate local context (much more efficient than cross-attention). Then, the SS2D block performs state space modeling on the sampled tokens to capture the intrinsic spatial relations among joints. Features from both parts are integrated via residual connections and an FFN. Unlike the original Mamba block and VMamba's VSS block, the PSS block is specifically designed for joint spatial sequences, scanning along the joint dimension rather than image patches.
    - **Design Motivation**: Projective attention provides efficient multi-view feature fusion, but using it alone cannot fully exploit inter-joint relationships. SSM complements this capability—experiments show that removing the Mamba block (Row 3) drops AP25 from 93.5 to 92.3, and removing GTBS+Mamba (Row 4) drops AP25 to 87.7.

2. **Grid Token-guided Bidirectional Scanning (GTBS)**:

    - **Function**: Perform efficient bidirectional scanning on projectively sampled tokens to encode local context and joint spatial sequences.
    - **Mechanism**: Unlike naive full-patch scanning (which is computationally expensive and contains many redundant background tokens), GTBS performs token-level bidirectional scanning only on feature points sampled by projective attention. Adapted from VMamba's SS2D, it transfers the scanning dimension from image space to joint space. As layers increase and keypoints are progressively refined, features scanned by GTBS become increasingly relevant, forming a positive feedback loop.
    - **Design Motivation**: Avoiding computational waste from scanning all image tokens; the spatial sequence of joints is more suitable for SSM modeling than image patch sequences (the human skeleton is naturally a sequence/tree structure).

3. **Progressive Regression and Hierarchical Tokens**:

    - **Function**: Progressively refine 3D keypoint predictions through multi-layer PSS blocks.
    - **Mechanism**: Initialize $N$ candidate tokens (each containing a visual feature term $\mathbf{V}_n \in \mathbb{R}^{J \times L}$ and a geometric term $\mathbf{K}_n \in \mathbb{R}^{J \times 3}$). The geometric term is initialized as a T-pose on the ground plane. Each layer uses an MLP to predict 2D offsets and confidence, and obtains 3D keypoints via differentiable algebraic triangulation $\mathbf{k}' = \text{AlgTriangulation}(\mathbf{u}'_t, \mathbf{c}_t, \mathbf{\Pi}_t)$, which serve as the input geometric terms for the next layer. Meanwhile, a linear classifier is used to filter out low-confidence candidate tokens (threshold $\epsilon = 0.1$), followed by NMS for de-duplication.
    - **Design Motivation**: Predicting all at once is inaccurate; progressive regression allows each layer to sample features at more accurate projection locations, creating a loop where quality is incrementally improved.

### Loss & Training
- Pose Loss: $\mathcal{L}_{\text{pose}} = \sum_{w=1}^{W} (\mathcal{L}_1(\mathbf{K}_{z(w)}, \mathbf{H}_z) + \sum_{t=1}^{T} \mathcal{L}_1(\hat{\mathbf{U}}_{z(w),t}, \mathbf{U}_{z,t}))$, which consists of 3D keypoint L1 loss + 2D projection L1 loss in each view.
- Classification Loss: Cross-entropy, used to distinguish positive and negative candidate tokens.
- The aforementioned losses are applied to each layer.
- ResNet-50 pre-trained on COCO is used, with a learning rate of 4e-4, trained for 40 epochs with early stopping.

## Key Experimental Results

### Main Results

| Method | Conference | CMU Panoptic AP25↑ | MPJPE↓ |
|------|------|-------------------|--------|
| VoxelPose | ECCV 20 | 84.0 | 17.7 |
| MvP | NeurIPS 21 | 92.3 | 15.8 |
| MVGFormer | CVPR 24 | 92.3 | 16.0 |
| **MV-SSM** | **Ours** | **93.5** | **15.7** |

Cross-camera Generalization (CMU0, 3 cameras only):

| Method | AP25↑ | mAP↑ |
|------|-------|------|
| MvP | 12.3 | 57.1 |
| MVGFormer | 44.6 | 83.4 |
| **MV-SSM** | **55.4** | **90.3** |

Cross-scene Generalization (Campus, without fine-tuning):

| Method | A1 PCP | A2 PCP | A3 PCP | Average |
|------|--------|--------|--------|---------|
| MvP | 0.0 | 0.0 | 0.0 | 0.0 |
| MVGFormer | 40.2 | 61.0 | 73.1 | 58.1 |
| **MV-SSM** | **55.5** | **65.5** | **79.9** | **67.3** |

### Ablation Study

| Configuration | AP25↑ | MPJPE↓ | Description |
|------|-------|--------|------|
| w Mean (Replace PSS) | 36.2 | 71.8 | Mean operation causes severe information loss |
| w Cross-attention | 90.4 | 16.8 | Pure attention is insufficient |
| w/o Mamba (SS2D+LN+FFN) | 92.3 | 16.0 | Degenerates to pure projective attention |
| w/o GTBS + Mamba | 87.7 | 18.6 | Both are necessary |
| Full MV-SSM | **93.5** | **15.7** | Full model |

### Key Findings
- MV-SSM improves over MVGFormer by **+10.8 AP25 (+24%)** in the most challenging 3-camera setup, representing the most outstanding display of generalization.
- In cross-camera configurations (CMU1-4 varying camera IDs and quantities), the average AP25 is improved by **+3.9**.
- MvP fails almost completely to generalize to new camera configurations (AP25=0), demonstrating that pure attention mechanisms heavily overfit to training cameras.
- Ablation proves both Mamba block and GTBS are indispensable; removing Mamba alone drops 1.2 AP25, while removing both drops 5.8 AP25.

## Highlights & Insights
- **First use of SSM in multi-view geometric modeling**: While Mamba was previously used primarily for temporal modeling in image classification and video understanding, this paper novelly applies it to model joint spatial sequences in multi-view static frames, offering a fresh approach.
- **Significantly superior generalization over Transformers**: The sequence modeling capability of SSM makes it more robust to camera configuration changes. This finding is highly insightful—SSMs could be applicable to other multi-view tasks requiring cross-domain generalization.
- **GTBS's "scan-only-useful-tokens" strategy**: By scanning only on a small number of key tokens after projective sampling, it simultaneously reduces computation and avoids background noise. This strategy can be transferred to any task requiring sequence modeling over sparse keypoints.

## Limitations & Future Work
- Still relies on known camera intrinsic and extrinsic parameters for projection and triangulation.
- Not trained/evaluated on larger-scale datasets (such as Human3.6M).
- Limited improvement on the Shelf dataset (PCP is nearly saturated: 88.0 vs 87.9).
- The scanning order of GTBS might not be optimal—human joints resemble a tree structure rather than a linear sequence.
- Missing computational efficiency analysis (parameter size and inference speed comparisons are not reported).

## Related Work & Insights
- **vs MVGFormer**: MVGFormer uses hierarchical queries + attention fusion, reaching similar in-domain accuracy (92.3 vs 93.5), but showing much poorer cross-domain generalization. The SSM modeling in this paper provides a better inductive bias for generalization.
- **vs MvP**: MvP also uses projective attention but severely overfits to camera configurations, with its cross-domain AP25 dropping to zero. This paper addresses this issue by adding SSM on top of MvP's projective attention.
- **vs VMamba/Vim**: These works adapt Mamba to 2D image classification, scanning image patches. This paper adapts it to 3D multi-view geometry, scanning joint tokens, which is a completely different application direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces SSM to multi-view 3D human pose estimation for the first time, offering a novel direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four types of generalization tests (in-domain, cross-camera, cross-configuration, and cross-scene) with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Logically clear with intuitive illustrations.
- Value: ⭐⭐⭐⭐ Generalization ability is key to practical deployment, providing a valuable new paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GroupMamba: Efficient Group-Based Visual State Space Model](groupmamba_efficient_group-based_visual_state_space_model.md)
- [\[CVPR 2025\] DefMamba: Deformable Visual State Space Model](defmamba_deformable_visual_state_space_model.md)
- [\[CVPR 2026\] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation](../../CVPR2026/segmentation/rs-ssm_refining_forgotten_specifics_in_state_space_model_for_video_semantic_segm.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2025\] 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)

</div>

<!-- RELATED:END -->

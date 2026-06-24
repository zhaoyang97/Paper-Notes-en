---
title: >-
  [Paper Note] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation
description: >-
  [CVPR 2025][3D Vision][Weakly-Supervised Semantic Segmentation] Rewis3d leverages feed-forward 3D reconstruction (MapAnything) to obtain 3D point clouds from 2D videos as auxiliary supervision signals. Utilizing a dual Student-Teacher architecture and weighted cross-modal consistency (CMC) loss, it improves weakly-supervised 2D semantic segmentation performance by 2-7% mIoU under sparse annotations (points/scribbles/coarse labels), while remaining purely 2D during inference.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Weakly-Supervised Semantic Segmentation"
  - "3D Reconstruction"
  - "Cross-Modal Consistency"
  - "Point Annotation"
  - "Scribble Annotation"
  - "Mean Teacher"
date: 2026-05-08
content_hash: 56b90836395579bc
---

# Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2603.06374](https://arxiv.org/abs/2603.06374)  
**Code**: To be released  
**Area**: 3D Vision / Semantic Segmentation  
**Keywords**: Weakly-Supervised Semantic Segmentation, 3D Reconstruction, Cross-Modal Consistency, Point Annotation, Scribble Annotation, Mean Teacher

## TL;DR
Rewis3d leverages feed-forward 3D reconstruction (MapAnything) to obtain 3D point clouds from 2D videos as auxiliary supervision signals. Utilizing a dual Student-Teacher architecture and weighted cross-modal consistency (CMC) loss, it improves weakly-supervised 2D semantic segmentation performance by 2-7% mIoU under sparse annotations (points/scribbles/coarse labels), while remaining purely 2D during inference.

## Background & Motivation
**Background**: Semantic segmentation heavily relies on dense pixel-level annotations. While sparse annotations can significantly cut down annotation costs, they incur a performance gap. Existing weakly-supervised methods like SASFormer (self-attention propagation) and TEL (minimum spanning tree pseudo-labels) struggle to fully bridge this gap in complex scenes.

**Limitations of Prior Work**: Pure 2D methods rely solely on single-frame appearance cues to propagate sparse supervision, which are limited by occlusions, scale variations, and long-range dependencies in geometrically complex outdoor scenes.

**Key Challenge**: 3D geometric structures can provide strong scene constraints to achieve cross-view consistency, but traditional methods require 3D sensors like LiDAR, which limits their applicability.

**Goal**: How to enhance weakly-supervised semantic segmentation using 3D geometric priors obtained solely from 2D video sequences, while requiring no 3D data during inference?

**Key Insight**: Leveraging the latest feed-forward 3D reconstruction models (MapAnything) to reconstruct dense 3D point clouds from videos, using 3D as an auxiliary supervision signal during training.

**Core Idea**: 3D geometric structures provide cross-view consistency constraints; sparse annotations from one viewpoint can be propagated to all visible viewpoints via 3D reconstruction.

## Method

### Overall Architecture
Input 2D video $\rightarrow$ Feed-forward reconstruction with MapAnything to obtain dense 3D point cloud + confidence $\rightarrow$ Independent training of 2D and 3D Mean Teacher branches (Base Training) $\rightarrow$ Cross-Modal Consistency (CMC): 2D teacher $\rightarrow$ 3D student, 3D teacher $\rightarrow$ 2D student $\rightarrow$ Output pure 2D segmentation model.

### Key Designs

1. **3D Scene Reconstruction and View-Aware Sampling**:

    - **Function**: Reconstructs dense 3D point clouds from videos using MapAnything, obtaining point-wise reconstruction confidence $c_i^{\text{rec}}$.
    - Generates a view-specific 120K point subsampling for each target image—60% from the viewpoint's own points (ensuring dense 2D-3D correspondence) and 40% from the spatial neighborhood (providing global context).
    - **Design Motivation**: Random sampling over the entire scene ($60\text{M}+ \rightarrow 120\text{K}$) results in only about 140 corresponding points per frame, which is insufficient for training the CMC loss.

2. **Dual Student-Teacher + Cross-Modal Consistency (CMC)**:

    - **Function**: Employs SegFormer-B4 for the 2D branch and Point Transformer V3 for the 3D branch, each equipped with an EMA Teacher.
    - Core of CMC: The teacher of one modality supervises the student of the other modality. 3D teacher $\rightarrow$ 2D student: $\mathcal{L}_C^{2D} = -\sum_j w_i \cdot \log(S_{2D}^{y_i}(I_j))$
    - **Dual Confidence Weighting**: $w_i = \max(\text{softmax}(T_{3D}(p_i))) \cdot c_i^{\text{rec}}$, which is the prediction confidence multiplied by the reconstruction confidence.
    - **Design Motivation**: Dual filtering ensures that only points with reliable predictions and reliable geometry provide supervision.

3. **3D Label Propagation**:

    - **Function**: Projects 2D sparse annotations back into 3D points to accumulate multi-view labels.
    - **Mechanism**: Each 3D point originates from a specific 2D pixel, mapping labels 1:1. These are accumulated across all source images to form a unified sparse 3D label map.

### Loss & Training
$$\mathcal{L}_{\text{Total}} = \sum_{m \in \{2D, 3D\}} (\mathcal{L}_S^m + \mathcal{L}_U^m) + \lambda_{2D} \mathcal{L}_C^{2D} + \lambda_{3D} \mathcal{L}_C^{3D}$$

Base Training runs for 15 epochs $\rightarrow$ CMC linearly ramps up over 5 epochs to $\lambda=0.1$.

## Key Experimental Results

### Main Results (Scribble Supervision, mIoU%)

| Method | Waymo | KITTI-360 | NYUv2 | SS/FS Ratio |
|------|-------|-----------|-------|---------|
| Fully Supervised | 59.0 | 68.4 | 51.1 | — |
| EMA Baseline | 49.4 | 60.3 | 42.9 | 83.7% |
| SASFormer | 37.8 | 46.4 | 44.7 | 64.1% |
| TEL | 42.4 | 59.2 | 38.3 | 71.9% |
| **Rewis3d (Ours)** | **53.3** | **63.4** | **46.1** | **90.3%** |

### Different Annotation Types (Cityscapes)

| Annotation Type | EMA | Rewis3d | Gain |
|---------|-----|---------|------|
| Point | 50.5 | **56.5** | +6.0 |
| Scribble | 61.2 | **68.1** | +6.9 |
| Coarse | 66.5 | **68.6** | +2.1 |

### Ablation Study (Waymo)

| Configuration | mIoU |
|------|------|
| No Filtering | 51.9 |
| + Prediction Confidence | 52.7 |
| + Dual Confidence | **53.3** |
| Random Sampling | 51.9 |
| View-Aware | **53.3** (+1.4) |
| Single-Frame Reconstruction | 52.1 |
| Multi-View Reconstruction | **53.3** (+1.2) |

### Key Findings
- **Reconstructed 3D outperforms real LiDAR** (+1.5 mIoU): Reconstructed point clouds are denser and contain confidence scores for filtering.
- Dual confidences are complementary (Prediction +0.8, Reconstruction +0.2, Combined +1.4).
- The sparser the annotation, the larger the geometric supervision gain (Point +6.0 > Coarse +2.1).

## Highlights & Insights
- **"No 3D at Inference" Design**: 3D is only used during training. Inference is fully 2D, which offers high practicality.
- **Counter-Intuitive Finding (Reconstructed 3D > Real 3D)**: Dense reconstruction combined with confidence filtering performs better than sparse physical sensor data.
- **View-Aware Sampling**: Solves the practical challenge of managing 2D-3D correspondence density in large-scale point clouds.

## Limitations & Future Work
- Relies on video sequences for 3D reconstruction—limited applicability to single-image datasets.
- MapAnything can produce noise when reconstructing dynamic objects.
- High 3D preprocessing overhead (200+ images $\rightarrow$ 60M+ points).

## Related Work & Insights
- **vs SASFormer/TEL**: Pure 2D methods are constrained in complex outdoor scenarios, whereas Rewis3d introduces 3D constraints to yield a 7-15% improvement.
- **vs 2DPASS**: 2DPASS requires physical LiDAR, whereas Rewis3d only requires 2D video sequences and performs pure 2D inference.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing 3D reconstruction as an auxiliary signal for weak supervision is a novel concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 4 datasets, 3 annotation types, with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and logically clear.
- Value: ⭐⭐⭐⭐⭐ Highly practical, providing a significant boost to weakly-supervised segmentation.
- **vs Pure 2D Weakly-Supervised**: Lacks spatial constraints; Rewis3d provides additional geometric priors through 3D reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel approach of auxiliary weakly-supervised learning via reconstruction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets with extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Introduces a new weakly-supervised paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GenFusion: Closing the Loop between Reconstruction and Generation via Videos](genfusion_closing_the_loop_between_reconstruction_and_generation_via_videos.md)
- [\[CVPR 2025\] ARM: Appearance Reconstruction Model for Relightable 3D Generation](arm_appearance_reconstruction_model_for_relightable_3d_generation.md)
- [\[CVPR 2025\] Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors](pow3r_empowering_unconstrained_3d_reconstruction_with_camera_and_scene_priors.md)
- [\[CVPR 2025\] Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction](touch2shape_touch-conditioned_3d_diffusion_for_shape_exploration_and_reconstruct.md)
- [\[CVPR 2025\] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation](generative_multiview_relighting_for_3d_reconstruction_under_extreme_illumination.md)

</div>

<!-- RELATED:END -->

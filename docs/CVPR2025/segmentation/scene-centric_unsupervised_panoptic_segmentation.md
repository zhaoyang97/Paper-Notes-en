---
title: >-
  [Paper Note] Scene-Centric Unsupervised Panoptic Segmentation
description: >-
  [CVPR 2025][Segmentation][Unsupervised Panoptic Segmentation] CUPS is the first unsupervised panoptic segmentation method trained directly on scene-centric images (such as autonomous driving scenarios). By fusing self-supervised visual features, stereo depth, and optical flow motion cues to generate high-quality pseudo-labels, it outperforms the previous SOTA, U2Seg, by 9.4% PQ on Cityscapes.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Unsupervised Panoptic Segmentation"
  - "Scene-Centric"
  - "Pseudo-Labels"
  - "Motion Segmentation"
  - "Depth Guidance"
date: 2026-05-08
content_hash: 629d6a7953083aea
---

# Scene-Centric Unsupervised Panoptic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2504.01955](https://arxiv.org/abs/2504.01955)  
**Code**: [https://visinf.github.io/cups](https://visinf.github.io/cups)  
**Area**: Segmentation  
**Keywords**: Unsupervised Panoptic Segmentation, Scene-Centric, Pseudo-Labels, Motion Segmentation, Depth Guidance

## TL;DR

CUPS is the first unsupervised panoptic segmentation method trained directly on scene-centric images (such as autonomous driving scenarios). By fusing self-supervised visual features, stereo depth, and optical flow motion cues to generate high-quality pseudo-labels, it outperforms the previous SOTA, U2Seg, by 9.4% PQ on Cityscapes.

## Background & Motivation

**Background**: Panoptic segmentation unifies semantic and instance segmentation into a single task. Currently, mainstream methods heavily rely on extensive pixel-level annotations. Recently, advancements in self-supervised representation learning (e.g., DINO) have driven the development of unsupervised segmentation—with STEGO addressing unsupervised semantic segmentation and CutLER addressing unsupervised instance segmentation.

**Limitations of Prior Work**: The only prior unsupervised panoptic segmentation method, U2Seg, suffers from three major limitations: (1) It relies on MaskCut to generate instance pseudo-labels, which assumes "object-centric" inputs (large, clear foreground objects) and performs poorly on complex, scene-centric images (with a mask precision of only 6.5%); (2) It cannot be trained directly on scene-centric target datasets, forcing the use of numerous pseudo-classes to bypass the thing/stuff distinction; (3) It relies on low-resolution STEGO semantic predictions, which yields limited effectiveness on high-resolution scene data.

**Key Challenge**: Existing unsupervised methods originate from "object-centric" data, whereas real-world applications (e.g., autonomous driving) face complex "scene-centric" images—characterized by small, dense objects, complex backgrounds, and coexisting thing and stuff categories. A fundamental gap exists between the object-centric assumption and scene-centric reality.

**Goal**: To build the first unsupervised panoptic segmentation method capable of training directly on scene-centric images, while simultaneously addressing the two key challenges of high-resolution semantic prediction and moving object instance discovery.

**Key Insight**: Inspired by Gestalt laws of perceptual organization—where humans naturally group visual elements based on similarity, constancy, and common fate—this work leverages depth to provide spatial 3D information and motion to furnish physical priors of "objects as movable entities." Together with visual appearance cues, they help disambiguate complex scenes.

**Core Idea**: Leveraging stereo video to extract depth-guided semantic information and motion-guided instance information, fusing them to generate high-quality panoptic pseudo-labels, and then training a panoptic network using pseudo-label guidance and self-training strategies.

## Method

### Overall Architecture

CUPS consists of three stages: (1) Pseudo-label generation—extracting scene flow and depth from stereo video frames, obtaining instance pseudo-labels via motion segmentation and semantic pseudo-labels via depth-enhanced inference, and fusing them into panoptic pseudo-labels; (2) Panoptic guidance—training the panoptic network with pseudo-labels guided by DropLoss and self-enhanced copy-paste; (3) Panoptic self-training—generating self-labels through multi-scale/multi-flip ensemble predictions from a momentum network to further improve performance. Inputs are stereo video pairs (only during the pseudo-label generation stage), while inference requires only a single monocular image.

### Key Designs

1. **Motion-based Instance Pseudo-label Generation**:

    - **Function**: Discovers moving objects from stereo video and generates high-accuracy instance masks.
    - **Mechanism**: Uses the unsupervised optical flow model SMURF to estimate forward/backward optical flow and disparity, computing scene flow $\mathbf{F}$ and an occlusion/consistency mask $\mathbf{O}$. The scene flow is processed by SF2SE3 to perform SE(3) rigid motion clustering. To address the inconsistency caused by the random initialization of SF2SE3, it is run $n$ times to obtain $m$ potentially overlapping masks. Masks appearing in fewer than 80% of the runs are filtered out using a consistency score $c_i$. Finally, Matrix NMS is applied to eliminate overlaps, and connected components are separated to obtain high-precision moving object masks.
    - **Design Motivation**: Motion cues naturally provide physical priors for "thing" categories (what moves is an object), which are more reliable than pure visual features for discovering instances in complex scenes. The ensemble-filtering strategy effectively improves pseudo-label accuracy (boosting mask precision from 6.5% for MaskCut to 59.6%).

2. **Depth-guided Semantic Pseudo-label Generation**:

    - **Function**: Generates high-quality semantic segmentation pseudo-labels on high-resolution scene images.
    - **Mechanism**: Distills DINO features based on DepthG to obtain low-dimensional semantic representations, followed by depth-guided multi-resolution fusion inference. Specifically, it obtains low-resolution predictions $\mathbf{P}^{low}$ (scaled full-image inputs) and high-resolution predictions $\mathbf{P}^{high}$ (stitched sliding windows). It calculates pixel-wise weights $\alpha_{h,w} = (D_{h,w}+1)^{-1}$ using depth $D$, and fuses them via $\mathbf{P}^* = \alpha \odot \mathbf{P}^{low} + (1-\alpha) \odot \mathbf{P}^{high}$. Nearby pixels (lower depth) rely heavily on low-resolution predictions, while distant pixels (higher depth) rely on high-resolution predictions.
    - **Design Motivation**: Self-supervised features extracted at low resolutions are more reliable for large, nearby objects, whereas small, distant objects require high-resolution details. Depth naturally encodes the distance between objects and the camera, acting as a bridge for resolution selection and elegantly solving the resolution limitations of SSL features.

3. **Panoptic Pseudo-label Fusion and Automatic Thing/Stuff Distinction**:

    - **Function**: Fuses semantic and instance pseudo-labels into unified panoptic pseudo-labels, while automatically distinguishing between "thing" and "stuff" categories.
    - **Mechanism**: It computes the ratio of a pseudo-semantic category's pixel coverage inside instance masks to its global occurrence frequency. Categories with ratios exceeding a threshold $\psi^{ts}$ are classified as "thing", while others are labeled as "stuff". The most frequent semantic ID within an instance mask is assigned to that instance; "thing" regions with no corresponding instance mask are labeled as dynamic ignore regions.
    - **Design Motivation**: While U2Seg is forced to use numerous pseudo-classes to bypass the difficulty of thing/stuff distinction, this work utilizes the prior that moving object masks naturally correspond to "thing" categories, achieving automatic categorization through simple statistics.

### Loss & Training

- **Stage 2 (Guided Learning)**: Employs DropLoss to supervise only "thing" predictions that have sufficient IoU overlap with pseudo-masks (without penalizing predictions lacking corresponding pseudo-labels, allowing the network to discover static objects autonomously) + standard cross-entropy loss for semantics (ignoring pixels in the ignore class) + Self-enhanced Copy-Paste augmentation (pasting reliable model predictions updated during training back into the training images). Optimized via AdamW for 4000 steps.
- **Stage 3 (Self-training)**: Maintains an EMA momentum network, applying flip and multi-scale augmentations to synthesize ensemble predictions. Self-labels are obtained by filtering predictions with an instance confidence threshold $\gamma$ and class-dependent semantic thresholds $\zeta_k$. Only heads are trained while normalization layers are frozen. Optimized via AdamW for 1500 steps.

## Key Experimental Results

### Main Results

| Dataset | Metric | CUPS | U2Seg (prev SOTA) | Gain |
|--------|------|------|----------|------|
| Cityscapes val | PQ | 27.8 | 18.4 | +9.4 |
| Cityscapes val | SQ | 57.4 | 55.8 | +1.6 |
| Cityscapes val | RQ | 35.2 | 22.7 | +12.5 |
| Cityscapes val | PQ_Th | 17.7 | 10.2 | +7.5 |
| Cityscapes val | PQ_St | 35.1 | 24.3 | +10.8 |
| KITTI | PQ | 25.5 | 20.6 | +4.9 |
| BDD | PQ | 19.9 | 15.8 | +4.1 |
| Waymo | PQ | 26.4 | 19.8 | +6.6 |
| MOTS (OOD) | PQ | 67.8 | 50.7 | +17.1 |

In unsupervised semantic segmentation, CUPS achieves an mIoU of 26.8% and an Acc of 83.2%, likewise outperforming all previous approaches.

### Ablation Study

| Configuration | PQ | Description |
|------|------|------|
| Full CUPS | 27.8 | Full model |
| w/o Depth-guided inference | ~24 | Removing depth-adaptive fusion degrades semantic quality |
| w/o Motion instance masks | ~20 | Cannot obtain reliable thing instances without motion cues |
| w/o Self-training (Stage 3) | ~24 | Self-training yields significant improvement |
| w/o Self-enhanced copy-paste | ~25 | Copy-paste augmentation helps discover static objects |

### Key Findings

- Motion cues are the most critical contribution of CUPS: replacing motion segmentation with MaskCut causes the mask precision to plunge from 59.6% to 6.5%, proving that object-centric methods fail completely on scene-centric data.
- Depth-guided inference effectively improves the semantic segmentation accuracy in high-resolution scenes, particularly for small, distant objects.
- CUPS demonstrates excellent generalization on the OOD dataset MOTS (PQ 67.8 vs. U2Seg 50.7), showing that motion and depth priors provide more robust transferability compared to pure visual cues.
- The gap to supervised methods is reduced from 43.9 PQ (with U2Seg) to 34.5, representing an approximate 20% reduction.

## Highlights & Insights

- **Depth-guided resolution-adaptive inference** is an elegant design. Using depth as a "resolution router," it automatically utilizes low-resolution global features for nearby elements and high-resolution local features for distant elements, bypassing the resolution bottleneck of SSL features without introducing extra parameters.
- **Systematically introducing Gestalt principles to unsupervised segmentation** provides an inspiring perspective. By aligning motion with common fate and depth with spatial proximity/similarity, the ensemble of these cues can be generalized to other unsupervised scene understanding tasks.
- The **ensemble-filtering strategy** is straightforward yet highly effective. Filtering random algorithm runs using consistency scores can be transferred to any perceptual modules incorporating random initializations.

## Limitations & Future Work

- Dependence on stereo video data for pseudo-label generation limits its application scenarios—monocular video or static image datasets cannot deploy this method directly.
- Motion cues only help discover "moving" objects; static "things" (e.g., parked cars) still rely on bootstrapping and self-training for discovery, offering limited coverage.
- The choice of 27 pseudo-categories is an empirical hyperparameter that needs tuning when transferring to other datasets.
- Future Directions: Integrating monocular depth estimation (e.g., DepthAnything) and video optical flow to alleviate the dependency on stereo videos; utilizing foundation models (e.g., SAM, DINOv2) to further enhance pseudo-label quality.

## Related Work & Insights

- **vs. U2Seg**: U2Seg relies on CutLER's MaskCut for instance masks and STEGO for semantics, both of which assume object-centric, low-resolution inputs and fail on scene-centric data. CUPS bypasses these limits by leveraging motion and depth, proving physical cues are more robust than pure visual cues for scene understanding.
- **vs. CutLER/MaskCut**: MaskCut achieves only 6.5% precision on scene-focused images (as Normalized Cut focuses on semantic regions rather than distinct instances in complex scenes), whereas CUPS's motion segmentation achieves 59.6%—a 9-fold improvement.
- **vs. DepthG**: CUPS's semantic module builds on DepthG but introduces depth-guided multi-resolution inference, improving mIoU from 23.1% to 26.8% on Cityscapes.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to systematically utilize motion and depth cues for unsupervised panoptic segmentation, representing a significant paradigm breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, covering evaluations on six datasets, cross-domain generalization, sub-task evaluations, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Exceptionally logical, with tight coherence from motivation to methodology and experimentation.
- **Value**: ⭐⭐⭐⭐ Pushes unsupervised panoptic segmentation towards real-world usability, particularly for autonomous driving scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene](learning_4d_panoptic_scene_graph_generation_from_rich_2d_visual_scene.md)
- [\[CVPR 2025\] Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning](hierarchical_compact_clustering_attention_coca_for_unsupervised_object-centric_l.md)
- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[CVPR 2025\] Towards Generalizable Scene Change Detection](towards_generalizable_scene_change_detection.md)
- [\[CVPR 2026\] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](../../CVPR2026/segmentation/dsflash_panoptic_scene_graph_realtime.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] 3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation
description: >-
  [CVPR 2025][Autonomous Driving][Auto-Vocabulary] Ours proposes 3D-AVS, the first **auto-vocabulary segmentation** method specifically tailored for LiDAR point clouds. Without requiring users to specify target categories, the system automatically identifies semantic entities in the scene from both images and point clouds to generate a vocabulary, and then finishes point-wise semantic segmentation with an open-vocabulary segmenter. It demonstrates the capability to generate fin…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Auto-Vocabulary"
  - "LiDAR Segmentation"
  - "CLIP"
  - "open-vocabulary"
  - "TPSS"
date: 2026-05-08
content_hash: 58c29425386d7026
---

# 3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2406.09126](https://arxiv.org/abs/2406.09126)  
**Code**: [https://github.com/ozzyou/3D-AVS](https://github.com/ozzyou/3D-AVS)  
**Area**: Autonomous Driving / 3D Point Cloud Segmentation / Open-Vocabulary  
**Keywords**: Auto-Vocabulary, LiDAR Segmentation, CLIP, open-vocabulary, TPSS

## TL;DR
Ours proposes 3D-AVS, the first **auto-vocabulary segmentation** method specifically tailored for LiDAR point clouds. Without requiring users to specify target categories, the system automatically identifies semantic entities in the scene from both images and point clouds to generate a vocabulary, and then finishes point-wise semantic segmentation with an open-vocabulary segmenter. It demonstrates the capability to generate fine-grained semantic categories on nuScenes and ScanNet200.

## Background & Motivation
Autonomous driving perception systems typically rely on pre-defined category sets for training and inference, which fail to recognize objects not covered in the training set. Although open-vocabulary segmentation (OVS) can detect arbitrary categories, it still requires users to manually specify the query vocabulary—this is impractical in real-world driving scenes since the category names of unknown objects are also unknown. While 2D auto-vocabulary methods like AutoSeg have emerged, they have not yet been extended to 3D point clouds. The key advantage of LiDAR data lies in its illumination invariance, which can complement cameras under harsh environmental conditions.

## Core Problem
How to automatically discover all semantic categories present in a LiDAR point cloud scene and complete point-wise segmentation **without any human-specified categories**? This setup is far more challenging than standard open-vocabulary segmentation, as the system itself must decide "what to look for".

## Method

### Overall Architecture
The pipeline of 3D-AVS consists of two phases: **Vocabulary Generation** $\to$ **Open-Vocabulary Segmentation**.
1. Given a LiDAR point cloud and its corresponding multi-view images, the image and point cloud branches first automatically identify semantic entities in the scene.
2. The identified category names from both branches are merged and de-duplicated to form a scene-specific vocabulary.
3. This vocabulary is fed as a query into an open-vocabulary 3D segmenter (OpenScene) to complete point-wise semantic segmentation.

### Key Designs
1. **Image-Based Recognition**: This branch utilizes the AutoSeg method (a training-free method based on BLIP) to process multi-view images via BLIP-Cluster-Caption. It first extracts multi-scale embeddings using BLIP, clusters them for enhancement, and then uses the BLIP decoder to generate captions, which are parsed into noun phrases as candidate categories.

2. **Point-Based Recognition with SMAP**: Ours introduces a Sparse Masked Attention Pooling (SMAP) module to directly recognize semantic entities from LiDAR point cloud features. Geometric information is more reliable under poor lighting conditions, enabling the discovery of objects that are hard to identify solely through images and enhancing vocabulary diversity.

3. **Open-Vocabulary 3D Segmenter (OpenScene)**: Utilizing a point encoder pre-aligned to the CLIP space, this module matches LiDAR point features with the text embeddings of candidate categories via similarity computation to complete point-wise label assignment. No additional training is required.

4. **TPSS Metric**: Ours proposes Text-Point Semantic Similarity (TPSS) to measure the semantic alignment between the automatically generated vocabulary and the point clouds in the CLIP space. For each point, the maximum similarity with all candidate labels is computed, and the global average is calculated. This metric does not rely on fixed annotations, allowing a fair evaluation of different vocabulary qualities.

3. **Optimization Strategy**

    - Function: Improves training stability and convergence speed
    - Mechanism: Adopts appropriate learning rate schedules, gradient clipping, and regularization strategies
    - Design Motivation: Ensures training efficiency of the model on large-scale data

### Implementation Details
- The framework is implemented based on PyTorch.
- Standard data augmentation strategies are applied to improve generalization.
- Both training and inference are executed efficiently on GPUs.

### Loss & Training
The entire approach is **zero-shot and training-free**. No training is required—by leveraging pre-trained BLIP (for image captioning), OpenScene (for point feature extraction, pre-aligned to the CLIP space), and CLIP (for text encoding), all modules are frozen during direct inference.

## Key Experimental Results

### nuScenes Validation Set (after LAVE mapping)

| Method | Zero-Shot| User-Free | mIoU | mAcc |
|------|-----------|-----------|------|------|
| LidarMultiNet (Fully Supervised) | ✗ | ✗ | 82.0 | - |
| OpenScene (OVS) | ✓ | ✗ | 42.1 | 61.8 |
| 3D-AVS (Auto-Vocabulary) | ✓ | ✓ | 30.6 | 44.1 |

### TPSS Metric (higher is better)

| Method | TPSS |
|------|------|
| OpenScene (using GT categories) | 8.71 |
| 3D-AVS (auto-generated categories) | **9.26** (+6.3%) |

### Ablation Study
- TPSS shows that the automatically generated vocabulary matches the point cloud semantics in the CLIP space better than hand-crafted pre-defined categories (9.26 vs 8.71).
- The decrease in mIoU after LAVE mapping is mainly due to: (1) fine-grained categories of the auto-vocabulary losing information when mapped back to coarse-grained GT categories; (2) LLM mapping itself introducing noise.
- Qualitative results display that 3D-AVS can recognize fine-grained categories such as building, pole, and sign, whereas the GT annotation only provides a broad "manmade" label.

## Highlights & Insights
- **Pioneering Task Definition**: Extends auto-vocabulary segmentation from 2D to 3D LiDAR point clouds for the first time, eliminating the dependency on human-specified queries.
- **Point-Image Dual-Branch Recognition**: Fuses visual and geometric information, ensuring greater robustness under adverse lighting conditions.
- **TPSS Metric**: Ingeniously leverages semantic alignment in the CLIP space to evaluate vocabulary quality, bypassing the limitation of traditional mIoU relying on fixed annotations.
- **Completely Training-Free**: All components utilize pre-trained models, allowing direct deployment without any fine-tuning.

## Limitations & Future Work
- The mIoU gap (30.6) compared to fully supervised (82.0) and OVS methods (42.1) remains large, illustrating substantial room for improvement in vocabulary accuracy and segmentation quality.
- Vocabulary generation depends on the captioning quality of BLIP, which may miss rare or small objects.
- LAVE evaluation requires LLM mapping, which introduces extra noise; although TPSS is annotation-free, it is bound by the quality of CLIP alignment.
- The current integration is relatively straightforward (an AutoSeg + OpenScene pipeline), lacking end-to-end optimization.

## Related Work & Insights
- **OpenScene**: Requires users to provide target categories. 3D-AVS outperforms OpenScene by 6% on TPSS, indicating higher-quality auto-generated categories, though its mIoU (requiring LAVE mapping) is 11.5 lower due to the much harder task setting.
- **AutoSeg** (2D AVS): 3D-AVS is its 3D extension, where the main innovation lies in introducing the point cloud branch SMAP and adaptation to 3D segmenters.
- **CLIP2Scene / ULIP**: Distills CLIP knowledge into point encoders, serving as the underlying technical foundation for 3D-AVS.

## Inspirations & Connections
- The auto-vocabulary concept can be extended to other 3D understanding tasks (3D object detection, 3D scene graph generation) without pre-defined category sets.
- The design of the TPSS metric can be generalized to evaluate alignment quality in other modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ Extends auto-vocabulary to 3D for the first time; the task definition is valuable.
- Experimental Thoroughness: ⭐⭐⭐ Conducted on two datasets, but the experiments are relatively sparse, lacking detailed ablation and more baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Problem definition and motivation are clear; the evaluation metric is well-designed.
- Value: ⭐⭐⭐ The direction is promising, but the current performance gap is large, making it more of a proof-of-concept.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [\[CVPR 2025\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2025\] Zero-Shot 4D Lidar Panoptic Segmentation](zero-shot_4d_lidar_panoptic_segmentation.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[NeurIPS 2025\] Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation](../../NeurIPS2025/autonomous_driving/leveraging_depth_and_language_for_open-vocabulary_domain-generalized_semantic_se.md)

</div>

<!-- RELATED:END -->

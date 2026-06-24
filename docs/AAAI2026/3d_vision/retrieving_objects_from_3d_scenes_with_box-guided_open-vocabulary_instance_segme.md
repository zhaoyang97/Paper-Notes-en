---
title: >-
  [Paper Note] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation
description: >-
  [AAAI 2026][3D Vision][Open-Vocabulary 3D Instance Segmentation] A Box-Guided approach is proposed, which leverages detection boxes from the 2D open-vocabulary detector YOLO-World to guide the construction of 3D instance masks from superpoints. Eliminating the need for SAM and CLIP, it achieves high efficiency (< 1 minute per scene) while significantly improving the retrieval capability for low-frequency target categories.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Open-Vocabulary 3D Instance Segmentation"
  - "3D Object Retrieval"
  - "Superpoints"
  - "YOLO-World"
  - "2D-to-3D Lifting"
date: 2026-05-08
content_hash: 37e25696baad93fa
---

# Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation

**Conference**: AAAI 2026  
**arXiv**: [2512.19088](https://arxiv.org/abs/2512.19088)  
**Code**: [https://github.com/ndkhanh360/BoxOVIS](https://github.com/ndkhanh360/BoxOVIS)  
**Area**: 3D Vision  
**Keywords**: Open-Vocabulary 3D Instance Segmentation, 3D Object Retrieval, Superpoints, YOLO-World, 2D-to-3D Lifting

## TL;DR

A Box-Guided approach is proposed, which leverages detection boxes from the 2D open-vocabulary detector YOLO-World to guide the construction of 3D instance masks from superpoints. Eliminating the need for SAM and CLIP, it achieves high efficiency (< 1 minute per scene) while significantly improving the retrieval capability for low-frequency target categories.

## Background & Motivation

### Background

Open-vocabulary 3D instance segmentation (OV-3DIS) aims to retrieve objects of arbitrary categories in 3D point clouds based on text queries, which is a core problem in robotics and augmented reality. Existing methods are mainly divided into two categories:

- **SAM+CLIP-based methods** (OpenMask3D, Open3DIS, OVIR-3D): Use SAM to generate 2D masks -> lift to 3D -> classify with CLIP. Although accurate, they are **extremely slow** (5-10 minutes per scene), making them impractical for deployment.
- **Efficient methods** (Open-YOLO 3D): Use a pre-trained 3D segmenter Mask3D to generate class-agnostic masks + YOLO-World for classification, taking about 22 seconds per scene by eliminating SAM and CLIP.

### Limitations of Prior Work

Although Open-YOLO 3D is fast, it has a **critical drawback**: it completely relies on Mask3D (a pre-trained 3D segmenter) to generate 3D candidate masks. Due to the limited size of 3D training data (incomplete category coverage in datasets like ScanNet), Mask3D often misses **low-frequency/rare categories** (such as calendars, thermometers, etc.). While 2D detectors (YOLO-World) can recognize these objects, Open-YOLO 3D only utilizes them for classification rather than for generating new masks.

### Key Challenge & Key Insight

Challenge: The generalization ability of 3D segmenters is limited vs. 2D detectors possess rich world knowledge. The core idea is: **using detection boxes from 2D detectors to guide the construction of new instance masks from 3D superpoints**, thereby inheriting the generalization capabilities of 2D models without relying on SAM (maintaining high efficiency).

## Method

### Overall Architecture

Input: 3D point cloud $P$ + multi-view RGB-D images + camera intrinsics and extrinsics + text query.  
Output: 3D instance masks matching the query.

Pipeline:
1. Generate 3D superpoints (geometrically consistent regions) using a graph segmentation algorithm.
2. Generate point-based masks using Mask3D (traditional path).
3. Generate 2D bounding boxes on RGB images using YOLO-World.
4. **Box-Guided RGBD-Based Mask Generation**: Lift 2D boxes to 3D, and assemble new instance masks using superpoints.
5. Merge the two types of masks and perform classification using detection results.

### Key Designs

#### 1. Box-Guided RGBD-Based Mask Generation (New Instance Discovery Guided by Boxes)

**Function**: Generate 3D masks for rare objects missed by the 3D segmenter.

**Core Pipeline**:

**(a) 2D-to-3D Box Lifting**:
- For each RGB frame, YOLO-World generates detection boxes $B_i = \{(b_{ij}, c_{ij})\}$
- Project pixels within the bounding box into 3D using depth information and camera parameters.
- Compute the 3D oriented bounding box $b_{ij}^{3D}$ containing all projected points using Open3D.

**(b) Redundancy Filtering**:
- If the intersection of the 3D box with an existing point-based mask is > $\tau_{\text{box}}\%$, it indicates that the object has already been detected by the 3D segmenter, and the box is discarded.

**(c) Superpoint Assembly**:
- Extract superpoints within the box: if a superpoint has $\geq \tau_{\text{spp}}\%$ of its points inside the box, it is assigned to that box.
- Obtain a coarse mask $S_{ij}$ for each box.

**(d) Cross-Frame Merging**:
- Process frame-by-frame: if the IoU between the new mask and an existing candidate is $\geq \tau_{\text{merge}}$ and they share the same category, merge the superpoints; otherwise, add it as a new candidate.
- Perform a final round of filtering: new masks with an IoU > $\tau_{\text{filter}}$ with a point-based mask are discarded (prioritizing the retention of point-based masks which have higher geometric quality).

**Design Motivation**:
- Using superpoint assembly instead of SAM: Superpoints are based on an efficient graph segmentation algorithm (Felzenszwalb), making the computational cost much lower than SAM.
- Redundancy filtering ensures that new masks supplement rather than replace the output of the 3D segmenter—objects already detected retain their more accurate point-based masks.

#### 2. Box-Based Mask Classification (Box-Based Classification)

**Function**: Assign category labels to each 3D candidate mask.

Following the approach of Open-YOLO 3D, entirely without CLIP:
- **Construct Label Map**: For each frame, fill the detection box regions with corresponding category labels, filling large boxes first and overwriting them with smaller boxes (intuition: if a small object is visible, it is likely closer to the camera than a larger object).
- **Calculate Visibility**: Project all 3D points to all frames at once to calculate in-frame visibility and occlusion visibility.
- **Aggregate Category Distribution**: For each 3D mask, count the category labels that the projected points fall into across the top-$k$ visible frames, and then assign the most frequently occurring category.

### Loss & Training

This paper presents a training-free/zero-shot method and does not require training. The pre-trained models used are:
- Mask3D: A class-agnostic 3D instance segmenter pre-trained on ScanNet.
- YOLO-World extra-large: An open-vocabulary 2D detector.
- Graph Segmentation: The classic algorithm by Felzenszwalb & Huttenlocher (2004).

Inference Settings:
- ScanNet200: 1 frame out of every 10 frames is used for YOLO-World detection.
- Replica: All frames are detected.
- Images are downsampled by 5 times during superpoint generation to increase efficiency.

## Key Experimental Results

### Main Results

**ScanNet200 Validation Set**:

| Method | SAM | CLIP | mAP | mAP50 | mAP25 | mAP_tail | Time/Scene |
|------|-----|------|-----|-------|-------|----------|----------|
| OpenMask3D | ✓ | ✓ | 15.4 | 19.9 | 23.1 | 14.9 | 553.87s |
| Open3DIS | ✓ | ✓ | 23.7 | 29.4 | 32.8 | 21.8 | 360.12s |
| Open-YOLO 3D | × | × | 24.7 | 31.7 | 36.2 | 21.6 | **21.8s** |
| **Ours** | **×** | **×** | **24.9** | **32.1** | **36.8** | **22.4** | 55.9s |

- Compared with Open-YOLO 3D: mAP +0.2, mAP50 +0.4, mAP25 +0.6, and **tail class mAP +0.8**.
- Although slower than Open-YOLO 3D (55.9s vs. 21.8s), it is far faster than SAM/CLIP-based methods (360s+).

**Replica Dataset**:

| Method | mAP | mAP50 | mAP25 | Time/Scene |
|------|-----|-------|-------|----------|
| OpenMask3D | 13.1 | 18.4 | 24.2 | 547.32s |
| Open3DIS | 18.5 | 24.5 | 28.2 | 187.97s |
| Open-YOLO 3D | 23.7 | 28.6 | 34.8 | **16.6s** |
| **Ours** | **24.0** | **31.8** | **37.4** | 43.7s |

On Replica, mAP50 improves by +3.2 and mAP25 by +2.6, showing a more pronounced improvement.

### Ablation Study

While the paper does not show a formal ablation table, key ablation insights can be extracted from method comparisons and discussion:

| Configuration | Key Change | Effect / Explanation |
|------|---------|---------|
| Point-based mask only (Open-YOLO 3D) | No RGBD-based mask | Tail class mAP of 21.6, missing rare objects |
| + Box-guided RGBD mask (Ours) | Add new instance discovery | Tail class mAP of 22.4 (+0.8), capable of detecting rare objects like "calendar" |
| RGBD mask quality | Superpoint-based instead of SAM | Large gains in IoU 50/25, but slightly lower quality under strict IoU thresholds |

### Key Findings

1. **Tail categories are the key gap**: There is little difference in head categories on ScanNet200 (even slightly lower by -0.2), but there is a clear improvement in tail categories (+0.8). This validates the core hypothesis that "3D segmenters generalize poorly to rare objects."
2. **Gains are larger at lower IoU thresholds**: The improvement magnitude decreases in the order of mAP25 > mAP50 > mAP, as the mask boundaries assembled by superpoints are not as detailed as those from SAM.
3. The visualization clearly demonstrates that the "calendar" object, which is completely missed by Open-YOLO 3D, is successfully retrieved by the proposed method.

## Highlights & Insights

1. **Clear design philosophy**: Rather than pursuing the absolute optimum for every single component, it finds a practical trade-off between efficiency and generalization capability.
2. **No extra training needed**: The entire pipeline is zero-shot, requiring no training on 3D data, simply combining existing pre-trained models.
3. **Superpoints replacing SAM**: This elegantly solves the efficiency issue of lifting 2D masks to 3D. Superpoints are based on a classic graph segmentation algorithm, which is far more efficient than SAM.
4. **Incremental design**: New masks supplement rather than replace point-based masks, preserving the geometric accuracy advantages of 3D segmenters on common categories.

## Limitations & Future Work

1. **Speed bottleneck**: Primarily in Open3D calculating the 3D oriented bounding boxes. The authors mention that developing a GPU-accelerated implementation is a critical next step.
2. **Mask quality**: The masks assembled from superpoints lack precision at high IoU thresholds. Future work could explore using SAM refinement solely on final candidates.
3. **Insufficient ablation**: Lacks sensitivity analysis for various hyperparameters ($\tau_{\text{box}}, \tau_{\text{spp}}, \tau_{\text{merge}}, \tau_{\text{filter}}$).
4. **Indoor only**: Both ScanNet200 and Replica are indoor datasets. Performance in outdoor scenarios (e.g., autonomous driving) has not been verified.
5. The absolute improvement in mAP is limited (+0.2), with the primary value lying in the improvements for tail categories.

## Related Work & Insights

- **Open-YOLO 3D** (ICLR 2025): Direct predecessor of this paper, which proved the feasibility of eliminating SAM/CLIP.
- **Open3DIS** (CVPR 2024): A pioneer in merging point-based and RGBD-based masks, but relies heavily on SAM.
- **YOLO-World** (CVPR 2024): A real-time open-vocabulary detector, acting as the core 2D module of the proposed approach.
- **Felzenszwalb Graph Segmentation (2004)**: A classic algorithm from 20 years ago finding new life in modern 3D understanding pipelines.
- Future work could consider replacing 2D detector classification with 3D open-vocabulary classifiers such as OpenShape or DuoMamba.

## Rating

- Novelty: ⭐⭐⭐ — Direct and effective approach, representing a component-level optimization rather than a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐ — Verified on two datasets but lacks deep ablation; the performance gains are relatively small.
- Writing Quality: ⭐⭐⭐⭐ — Clear and concise, with well-formulated problem motivation.
- Value: ⭐⭐⭐⭐ — Highly practical, providing an efficient solution for retrieving rare objects in 3D scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] OVSeg3R: Learn Open-vocabulary Instance Segmentation from 2D via 3D Reconstruction](../../ICLR2026/3d_vision/ovseg3r_learn_open-vocabulary_instance_segmentation_from_2d_via_3d_reconstructio.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [\[CVPR 2026\] OVI-MAP: Open-Vocabulary Instance-Semantic Mapping](../../CVPR2026/3d_vision/ovi-map_open-vocabulary_instance-semantic_mapping.md)
- [\[CVPR 2025\] Sketchy Bounding-Box Supervision for 3D Instance Segmentation](../../CVPR2025/3d_vision/sketchy_bounding-box_supervision_for_3d_instance_segmentation.md)
- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)

</div>

<!-- RELATED:END -->

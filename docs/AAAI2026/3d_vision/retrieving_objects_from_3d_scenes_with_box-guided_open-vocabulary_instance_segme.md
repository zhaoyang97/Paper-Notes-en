---
title: >-
  [Paper Note] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation
description: >-
  [AAAI 2026][3D Vision][Open-vocabulary 3D instance segmentation] This paper proposes a box-guided approach that leverages 2D bounding boxes from the open-vocabulary detector YOLO-World to guide the assembly of 3D instanc…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Open-vocabulary 3D instance segmentation"
  - "3D object retrieval"
  - "superpoints"
  - "YOLO-World"
  - "2D-to-3D lifting"
date: 2026-05-08
content_hash: 85eede20057cb6ec
---

# Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation

**Conference**: AAAI 2026
**arXiv**: [2512.19088](https://arxiv.org/abs/2512.19088)
**Code**: [https://github.com/ndkhanh360/BoxOVIS](https://github.com/ndkhanh360/BoxOVIS)
**Area**: 3D Vision
**Keywords**: Open-vocabulary 3D instance segmentation, 3D object retrieval, superpoints, YOLO-World, 2D-to-3D lifting

## TL;DR

This paper proposes a box-guided approach that leverages 2D bounding boxes from the open-vocabulary detector YOLO-World to guide the assembly of 3D instance masks from superpoints, without relying on SAM or CLIP. The method achieves high efficiency (<1 min/scene) while substantially improving retrieval performance on rare-category objects.

## Background & Motivation

### State of the Field

Open-vocabulary 3D instance segmentation (OV-3DIS) aims to retrieve objects of arbitrary categories from 3D point clouds given text queries, and is a core problem in robotics and augmented reality. Existing methods fall into two main categories:

- **SAM+CLIP-based methods** (OpenMask3D, Open3DIS, OVIR-3D): generate 2D masks with SAM → lift to 3D → classify with CLIP. Reasonable accuracy but **extremely slow** (5–10 min/scene), making deployment impractical.
- **Efficient method** Open-YOLO 3D: uses the class-agnostic 3D segmentor Mask3D to generate candidate masks and YOLO-World for classification, ~22 s/scene, eliminating SAM and CLIP.

### Limitations of Prior Work

Despite its speed, Open-YOLO 3D has a **critical weakness**: it relies entirely on Mask3D (a pretrained 3D segmentor) to generate 3D candidate masks. Due to limited 3D training data (e.g., incomplete category coverage in ScanNet), Mask3D frequently **misses low-frequency or rare categories** (e.g., calendars, thermometers). Although YOLO-World can recognize such objects, Open-YOLO 3D uses it only for classification and not for generating new masks.

### Root Cause & Starting Point

The tension lies between the limited generalization of 3D segmentors and the rich world knowledge of 2D detectors. The paper's core idea is to **use 2D detection boxes to guide the assembly of new 3D instance masks from superpoints**—inheriting the generalization capacity of 2D models while avoiding SAM and thus preserving efficiency.

## Method

### Overall Architecture

**Input**: 3D point cloud $P$ + multi-view RGB-D images + camera intrinsics/extrinsics + text query.
**Output**: 3D instance masks matching the query.

**Pipeline**:
1. Generate 3D superpoints (geometrically consistent regions) via graph-based segmentation.
2. Generate point-based masks using Mask3D (conventional path).
3. Run YOLO-World on RGB frames to produce 2D bounding boxes.
4. **Box-Guided RGBD-Based Mask Generation**: lift 2D boxes to 3D and assemble new instance masks from superpoints.
5. Merge both types of masks; use detection box results for classification.

### Key Designs

#### 1. Box-Guided RGBD-Based Mask Generation

**Function**: Generate 3D masks for rare objects missed by the 3D segmentor.

**Core pipeline**:

**(a) Lifting 2D boxes to 3D**:
- For each RGB frame, YOLO-World produces detections $B_i = \{(b_{ij}, c_{ij})\}$.
- Pixels within each box are projected to 3D via depth maps and camera parameters.
- Open3D is used to compute a 3D oriented bounding box $b_{ij}^{3D}$ enclosing all projected points.

**(b) Redundancy filtering**:
- If the overlap between a 3D box and any existing point-based mask exceeds $\tau_{\text{box}}\%$, the object has already been detected by the 3D segmentor and the box is discarded.

**(c) Superpoint assembly**:
- Superpoints are assigned to a box if $\geq \tau_{\text{spp}}\%$ of their points fall inside the box.
- This yields a coarse mask $S_{ij}$ for each box.

**(d) Cross-frame merging**:
- Frames are processed sequentially; if a new mask has IoU $\geq \tau_{\text{merge}}$ with an existing candidate of the same category, their superpoints are merged; otherwise the new mask is added as a new candidate.
- A final filtering pass discards new masks whose IoU with any point-based mask exceeds $\tau_{\text{filter}}$, preserving the geometrically superior point-based masks for already-detected objects.

**Design Motivation**:
- Superpoint assembly instead of SAM: superpoints are computed via the efficient Felzenszwalb graph-based algorithm, incurring far lower computational cost than SAM.
- Redundancy filtering ensures the new masks complement rather than replace the 3D segmentor's output.

#### 2. Box-Based Mask Classification

**Function**: Assign category labels to each 3D candidate mask.

Following Open-YOLO 3D, CLIP is not used:
- **Label map construction**: for each frame, box regions are filled with detected category labels; larger boxes are filled first, then overwritten by smaller ones (intuition: a visible small object is closer to the camera than a larger one).
- **Visibility computation**: all 3D points are projected to all frames at once to compute in-frame and occlusion-aware visibility.
- **Category aggregation**: for each 3D mask, projected points in the top-$k$ most visible frames are polled for their label map entries, and the most frequent category is assigned.

### Loss & Training

This is a training-free / zero-shot method requiring no additional training. Pretrained components used:
- **Mask3D**: class-agnostic 3D instance segmentor pretrained on ScanNet.
- **YOLO-World extra-large**: open-vocabulary 2D detector.
- **Graph-based segmentation**: the classical Felzenszwalb & Huttenlocher (2004) algorithm.

Inference settings:
- ScanNet200: YOLO-World is applied to every 10th frame.
- Replica: all frames are processed.
- Images are downsampled by a factor of 5 during superpoint generation for efficiency.

## Key Experimental Results

### Main Results

**ScanNet200 validation set**:

| Method | SAM | CLIP | mAP | mAP50 | mAP25 | mAP_tail | Time/scene |
|--------|-----|------|-----|-------|-------|----------|------------|
| OpenMask3D | ✓ | ✓ | 15.4 | 19.9 | 23.1 | 14.9 | 553.87s |
| Open3DIS | ✓ | ✓ | 23.7 | 29.4 | 32.8 | 21.8 | 360.12s |
| Open-YOLO 3D | × | × | 24.7 | 31.7 | 36.2 | 21.6 | **21.8s** |
| **Ours** | **×** | **×** | **24.9** | **32.1** | **36.8** | **22.4** | 55.9s |

- Compared to Open-YOLO 3D: mAP +0.2, mAP50 +0.4, mAP25 +0.6, **tail mAP +0.8**.
- Speed (55.9s) is slower than Open-YOLO 3D (21.8s) but far faster than SAM/CLIP-based methods (360s+).

**Replica dataset**:

| Method | mAP | mAP50 | mAP25 | Time/scene |
|--------|-----|-------|-------|------------|
| OpenMask3D | 13.1 | 18.4 | 24.2 | 547.32s |
| Open3DIS | 18.5 | 24.5 | 28.2 | 187.97s |
| Open-YOLO 3D | 23.7 | 28.6 | 34.8 | **16.6s** |
| **Ours** | **24.0** | **31.8** | **37.4** | 43.7s |

Gains on Replica are more pronounced: mAP50 +3.2, mAP25 +2.6.

### Ablation Study

No formal ablation table is presented; key observations are extracted from comparisons and discussion:

| Configuration | Key Change | Effect |
|---------------|-----------|--------|
| Point-based masks only (Open-YOLO 3D) | No RGBD-based masks | tail mAP 21.6; rare objects missed |
| + Box-guided RGBD masks (Ours) | Novel instance discovery added | tail mAP 22.4 (+0.8); rare objects such as "calendar" detectable |
| RGBD mask quality | Superpoint assembly vs. SAM | Larger gains at IoU 50/25 thresholds; lower quality under strict IoU |

### Key Findings

1. **Tail categories are the key gap**: differences on head categories are minimal (slightly negative, −0.2), while tail categories show clear improvement (+0.8), confirming the core hypothesis that 3D segmentors generalize poorly to rare objects.
2. **Gains are larger at lower IoU thresholds**: improvement diminishes from mAP25 > mAP50 > mAP, since superpoint-assembled masks have coarser boundaries than SAM-based masks.
3. Qualitative results explicitly demonstrate that objects such as "calendar"—completely undetected by Open-YOLO 3D—are successfully retrieved by the proposed method.

## Highlights & Insights

1. **Clear design philosophy**: rather than optimizing every component, the method strikes a practical balance between efficiency and generalization.
2. **No additional training required**: the entire pipeline is zero-shot, combining existing pretrained models without any training on 3D data.
3. **Superpoints as a SAM substitute**: an elegant solution to the 2D-to-3D mask lifting efficiency problem, relying on the classical and highly efficient Felzenszwalb graph-based segmentation.
4. **Incremental design**: new masks supplement rather than replace point-based masks, preserving the geometric accuracy of the 3D segmentor for common categories.

## Limitations & Future Work

1. **Speed bottleneck**: primarily caused by Open3D's computation of 3D oriented bounding boxes; GPU-accelerated implementations are identified as an important direction.
2. **Mask quality**: superpoint-assembled masks are insufficiently precise under high IoU thresholds; future work could explore applying SAM refinement exclusively to final candidates.
3. **Insufficient ablation**: sensitivity analyses for hyperparameters ($\tau_{\text{box}}, \tau_{\text{spp}}, \tau_{\text{merge}}, \tau_{\text{filter}}$) are absent.
4. **Indoor-only evaluation**: both ScanNet200 and Replica are indoor datasets; generalization to outdoor scenes (e.g., autonomous driving) remains unvalidated.
5. The absolute mAP gain is modest (+0.2); the primary contribution lies in improving tail-category performance.

## Related Work & Insights

- **Open-YOLO 3D** (ICLR 2025): the direct predecessor of this work, demonstrating the feasibility of eliminating SAM and CLIP.
- **Open3DIS** (CVPR 2024): pioneered the fusion of point-based and RGBD-based masks but relies on SAM.
- **YOLO-World** (CVPR 2024): real-time open-vocabulary detector serving as the core 2D module in this method.
- **Felzenszwalb graph segmentation (2004)**: a 20-year-old classical algorithm finds renewed relevance in modern 3D understanding pipelines.
- Future work could explore replacing the 2D detector-based classifier with 3D open-vocabulary classifiers such as OpenShape or DuoMamba.

## Rating

- Novelty: ⭐⭐⭐ — Straightforward yet effective; a component-level improvement rather than a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐ — Validated on two datasets but lacks ablations; performance gains are modest.
- Writing Quality: ⭐⭐⭐⭐ — Clear and concise, with well-motivated problem framing.
- Value: ⭐⭐⭐⭐ — Strong practical utility; provides an efficient solution for retrieving rare objects in 3D scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [\[AAAI 2026\] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation](assist-3d_adapted_scene_synthesis_for_class-agnostic_3d_instance_segmentation.md)
- [\[CVPR 2026\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](../../CVPR2026/3d_vision/jopp3d_joint_open_vocabulary_semantic_segmentation.md)
- [\[ICLR 2026\] GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation](../../ICLR2026/3d_vision/geopurify_a_data-efficient_geometric_distillation_framework_for_open-vocabulary_.md)

</div>

<!-- RELATED:END -->

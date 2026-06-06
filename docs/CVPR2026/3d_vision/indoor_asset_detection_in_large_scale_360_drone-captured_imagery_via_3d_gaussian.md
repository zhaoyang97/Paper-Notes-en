---
title: >-
  [Paper Note] Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][indoor asset detection] This paper proposes a pipeline based on a 3D Object Codebook that associates 2D segmentation masks into consistent 3D object instances within 3DGS using semantic and spatial…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "indoor asset detection"
  - "3DGS segmentation"
  - "multi-view consistency"
  - "object codebook"
  - "drone 360° imaging"
date: 2026-05-08
content_hash: 8f0c870b4d74f65b
---

# Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2604.05316](https://arxiv.org/abs/2604.05316)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: indoor asset detection, 3DGS segmentation, multi-view consistency, object codebook, drone 360° imaging

## TL;DR
This paper proposes a pipeline based on a 3D Object Codebook that associates 2D segmentation masks into consistent 3D object instances within 3DGS using semantic and spatial constraints, enabling object-level detection on large-scale indoor 360° drone imagery. It achieves a 65% improvement in F1 score and 11% improvement in mAP over the state-of-the-art method GAGA.

## Background & Motivation
1. **Background**: 3DGS has become the mainstream approach for indoor scene reconstruction. Semantic understanding (object detection/segmentation) is critical for downstream applications such as facility management and safety assessment. 2D foundation models (SAM/Grounded SAM) can produce high-quality segmentation masks.
2. **Limitations of Prior Work**: 2D masks suffer from inconsistent IDs across viewpoints — the same object is assigned different labels in different frames. Existing 3DGS segmentation methods (e.g., GAGA) use video tracking or 3D memory banks to associate masks, but perform poorly in large-scale indoor scenes with multi-video streams, repeated object appearances, and dense occlusions. Many methods adopt a "segment everything" paradigm (including stuff classes such as walls and floors), which is ill-suited for asset-specific applications.
3. **Key Challenge**: 2D segmentation masks are of high quality but inconsistent across viewpoints; 3DGS provides 3D consistency but lacks semantic information. A robust multi-view mask association mechanism is therefore required.
4. **Goal**: To achieve consistent multi-view object detection and segmentation for user-defined indoor assets within large-scale 3DGS scenes.
5. **Key Insight**: The paper leverages the spatial consistency of 3DGS Gaussian primitives to construct a 3D Object Codebook — each entry records a unique ID, a semantic label, and the associated set of Gaussians. Masks are progressively merged via semantic and spatial constraints.
6. **Core Idea**: Extract 3D Gaussians corresponding to each mask using adaptive depth tolerance, and construct a multi-view consistent object codebook through a two-stage semantic-then-spatial merging process.

## Method

### Overall Architecture
The pipeline consists of three stages: (1) **Preprocessing** — OWLv2 detection followed by SAM segmentation to generate labeled 2D masks per frame; (2) **Codebook construction** — per-frame mask processing involving adaptive-depth Gaussian extraction → semantic-constrained merging → low-weight Gaussian filtering → spatial merging → second-round filtering → confidence-weighted voting for final label assignment; (3) **Post-processing** — filtering of low-confidence objects and HDBSCAN-based spatial outlier removal.

### Key Designs
1. **Gaussian Extraction with Adaptive Depth Tolerance**:
    - Function: Precisely determine the 3D Gaussians corresponding to each 2D mask, preventing background Gaussians from being incorrectly included due to foreground objects.
    - Mechanism: For each Gaussian center $\mu$ projected onto the mask region at $(x_p, y_p)$, the method checks whether its depth $d_\mu$ falls within an adaptive tolerance $\delta(x_p, y_p)$ of the rendered depth map $D(x_p, y_p)$. The tolerance is defined as the maximum depth difference within a $7\times7$ neighborhood: $\delta(x,y) = \max\{|D(x,y) - D(n_x,n_y)| : (n_x,n_y) \in \mathcal{N}_{(x,y)}, |...| \leq T\}$, with an upper bound $T=0.5$.
    - Design Motivation: GAGA uses a global depth interval, which is too wide for objects with severe perspective foreshortening (e.g., ceiling lights), inadvertently including ceiling Gaussians. The adaptive tolerance adjusts to local surface geometry — stricter for flat regions and more permissive for curved surfaces.

2. **Two-Stage Merging: Semantic Constraint followed by Spatial Merging**:
    - Function: Associate 2D masks from different viewpoints into consistent 3D objects within the 3DGS representation.
    - Mechanism: In the first stage (semantic-constrained merging), a new mask is compared only against codebook entries sharing the **same semantic label**; if the Gaussian overlap ratio exceeds $\tau_{overlap}=0.2$, the mask is merged into that entry, otherwise a new entry is created. In the second stage (spatial merging), an undirected graph is constructed where entries with bidirectional overlap ratios exceeding $\tau_{spatial}=0.3$ form connected components that are merged; confidence-weighted voting then determines the final semantic label.
    - Design Motivation: The two-stage ordering — semantic first, spatial second — is critical. Without prior semantic constraint, small objects (e.g., windows on doors) would be absorbed by larger ones (e.g., doors). The spatial merging stage corrects cross-frame label inconsistencies produced by the 2D detector.

3. **Multi-Level Filtering (Low-Weight Gaussian Filtering + Object Filtering + HDBSCAN)**:
    - Function: Progressively eliminate noise at three levels — Gaussian level, object level, and spatial outlier level.
    - Mechanism: The weight of each Gaussian is defined as the sum of (confidence / estimated depth) across all associated masks — Gaussians observed at long range, with low confidence, or infrequently are down-weighted and removed. Object confidence is defined as $\log(|M|) \cdot \frac{1}{|M|}\sum c_m$, requiring objects to be both frequently and confidently observed. HDBSCAN removes spatially anomalous Gaussians.
    - Design Motivation: In 360° drone data, each object is observed from a large number of viewpoints; genuine objects accumulate high weights and high confidence scores, whereas noisy masks and Gaussians do not. The $\log|M|$ term rewards multi-view observation frequency.

### Loss & Training
The pipeline involves no training — it is entirely based on pretrained 2D detectors (OWLv2), segmentation models (SAM), and standard 3DGS reconstruction (30K iterations). All thresholds are determined via ablation studies.

## Key Experimental Results

### Main Results (Multi-View Mask Consistency)

| Scene | Method | mIoU | Precision | Recall | F1 | Unique Objects | Time |
|-------|--------|------|-----------|--------|-----|----------------|------|
| Cory Corridor | GAGA | 9.28 | 73.33 | 9.73 | 17.19 | 15 (GT: 113) | 39 min |
| Cory Corridor | **Ours** | **75.84** | **82.61** | **84.07** | **83.33** | **115** | **10 min** |
| Cory Office | GAGA | 2.10 | 33.33 | 1.64 | 3.12 | 12 (GT: 244) | 3 hr 43 min |
| Cory Office | **Ours** | **66.01** | **67.05** | **72.54** | **69.69** | **264** | **1 hr 33 min** |

### Object Detection Results

| Scene | Method | mAP↑ | mLAMR↓ |
|-------|--------|------|--------|
| Cory Corridor | OWLv2 | 28.15 | 78.30 |
| Cory Corridor | **Ours** | **41.78** | **69.77** |
| Cory Office | OWLv2 | 33.27 | 73.19 |
| Cory Office | **Ours** | **41.62** | **63.92** |

### Ablation Study (Cory Corridor / Cory Office F1)

| Configuration | Cory Corridor F1 | Cory Office F1 | Notes |
|---------------|-----------------|----------------|-------|
| Full pipeline | **83.33** | **69.69** | Complete model |
| w/o depth processing | 64.21 | 61.30 | Global depth interval causes erroneous inclusion of background Gaussians |
| w/o semantic constraint | 80.72 | 59.72 | Objects of different semantics incorrectly merged |
| w/o 1st-round filtering | 83.33 | 69.05 | No effect in corridor; slight drop in office |
| w/o spatial merging | 82.61 | 63.47 | Cross-frame label inconsistency leads to duplicate objects |
| w/o 2nd-round filtering | 83.33 | 69.69 | Minimal impact in corridor |
| w/o object filtering | 61.77 | — | Large influx of spurious objects |
| w/o HDBSCAN | — | 69.69 | No notable impact in corridor |

### Key Findings
- F1 score improves by 65% (17.19 → 83.33). GAGA's extremely low Recall (9.73%) indicates severe mask association failure in large-scale scenes due to over-merging: GAGA detects only 15 unique objects against a ground truth of 113.
- mAP on the object detection task also improves by 11% (28.15 → 41.78), demonstrating that the 3D codebook enhances 2D-level detection coverage.
- Processing speed is substantially faster (10 min vs. 39 min), as semantic constraints reduce unnecessary overlap computations.
- Adaptive depth tolerance is the most critical component: removing it causes F1 on Cory Corridor to drop sharply from 83.33 to 64.21 (−23%), since the global depth interval fails severely on foreshortened objects.
- Semantic constraints are more critical in the complex 29-category office scene: Cory Office F1 drops from 69.69 to 59.72 (−14%) without them.
- Object filtering has the largest impact on Cory Corridor (F1 drops from 83.33 to 61.77), indicating serious spurious detection issues in multi-class scenes.
- GAGA exhibits temporal degradation: batch-level F1 decreases monotonically as the sequence progresses, as the same object receives the same ID at both early and late stages, causing global evaluation to fall far below local batch evaluation.

## Highlights & Insights
- **The two-stage merging design**: Constraining semantics first and then relaxing to spatial overlap prevents objects with different semantics from being incorrectly merged. This strategy is transferable to any 3D scene segmentation task requiring multi-view instance association.
- **Adaptive depth tolerance**: A simple yet effective fix for the critical failure mode of global depth intervals. Leveraging local depth variation rather than global statistics yields substantial gains from a minimal implementation change.
- **The Object Codebook as a practical framework**: Provides an end-to-end solution for "defining and locating specific asset categories in 3DGS," directly serving real-world applications such as facility management and safety inspection.

## Limitations & Future Work
- The pipeline relies on OWLv2's open-vocabulary detection capability — objects may be missed when category names are ambiguous or appearances are atypical.
- Thresholds ($\tau_{overlap}$, $\tau_{spatial}$, etc.) are determined via ablation and may require adjustment across different scenes.
- The 360° drone capture ensures high observation coverage — generalization to scenes captured from conventional forward-facing viewpoints has not been validated.
- The method targets thing classes only and does not handle stuff classes; segmentation of large-area surfaces such as walls and ceilings is out of scope.

## Related Work & Insights
- **vs. GAGA**: GAGA's global 3D memory bank degrades severely in large-scale multi-stream scenes (Recall of only 9.73%); the proposed semantic-then-spatial two-stage codebook is substantially more robust.
- **vs. Gaussian Grouping**: Video tracking-based methods cannot handle repeated object appearances across multiple video streams.
- **vs. direct application of Grounded SAM**: 2D segmentation quality is high but cross-frame inconsistency remains; this paper achieves consistency via the 3D Gaussian space.

## Rating
- Novelty: ⭐⭐⭐ The method is a well-engineered combination of existing components; no single component is entirely novel, but the combination is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two large-scale real-world scenes, detailed ablations, and a fair comparison with GAGA.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is described clearly, with figures aiding comprehension of each step.
- Value: ⭐⭐⭐⭐ Provides a viable solution for practical object detection in large-scale indoor 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IM360: Large-scale Indoor Mapping with 360 Cameras](../../ICCV2025/3d_vision/im360_large-scale_indoor_mapping_with_360_cameras.md)
- [\[CVPR 2026\] Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments](few-shot_incremental_3d_object_detection_in_dynamic_indoor_environments.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [\[CVPR 2026\] SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations](scenescribe-1m_a_large-scale_video_dataset_with_comprehensive_geometric_and_sema.md)

</div>

<!-- RELATED:END -->

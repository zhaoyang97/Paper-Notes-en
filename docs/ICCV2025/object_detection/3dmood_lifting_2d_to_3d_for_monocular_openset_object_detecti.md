---
title: >-
  [Paper Note] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection
description: >-
  [ICCV 2025][Object Detection][Monocular 3D object detection] This paper proposes 3D-MOOD, the first end-to-end monocular open-set 3D object detector…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Monocular 3D object detection"
  - "open-set detection"
  - "2D-to-3D lifting"
  - "geometry-aware query"
  - "canonical image space"
date: 2026-05-08
content_hash: c79fa90aff8c9d46
---

# 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection

**Conference**: ICCV 2025
**arXiv**: [2507.23567](https://arxiv.org/abs/2507.23567)  
**Code**: [royyang0714.github.io/3D-MOOD](https://royyang0714.github.io/3D-MOOD)  
**Area**: Object Detection
**Keywords**: Monocular 3D object detection, open-set detection, 2D-to-3D lifting, geometry-aware query, canonical image space

## TL;DR
This paper proposes 3D-MOOD, the first end-to-end monocular open-set 3D object detector, which lifts open-set 2D detections into 3D space via geometry-aware 3D query generation and a canonical image space design, achieving state-of-the-art performance on both the Omni3D closed-set benchmark and the Argoverse 2 / ScanNet open-set benchmarks.

## Background & Motivation

Monocular 3D object detection (3DOD) infers and localizes 3D objects from a single RGB image, offering low cost but posing significant challenges. Existing methods operate almost exclusively under the **closed-set** assumption, where training and test sets share the same scenes and categories. However, in real-world applications such as robotics and AR/VR, models frequently encounter objects from **novel environments** and **novel categories**, to which closed-set methods cannot generalize.

Unlike the 2D domain—where abundant image-text pairs enable open-vocabulary classification—3D data lacks rich vision-language correspondences, making open-set classification in 3D extremely difficult. Monocular depth estimation also faces inherent generalization challenges across datasets. The combination of these two issues has left open-set monocular 3DOD largely unsolved.

## Core Problem
**How can a monocular 3D detector simultaneously achieve: (1) the ability to recognize novel categories unseen during training (open-vocabulary classification), and (2) accurate 3D localization/size/orientation estimation in unseen scenes (cross-domain 3D regression generalization)?**

The key challenges are: 3D annotations are scarce and lack textual correspondence, making it impossible to achieve open-set classification through vision-language alignment as in 2D. Furthermore, cross-dataset training introduces significant variation in image resolution and camera intrinsics; conventional resize-and-padding strategies cause ambiguity between intrinsics and image dimensions.

## Method

### Overall Architecture
3D-MOOD builds upon the open-set 2D detector Grounding DINO (G-DINO). Given a monocular image $\mathbf{I}$ and a language prompt $\mathbf{T}$ describing object categories of interest, the model outputs 3D bounding boxes $\mathbf{D}^{3D}$ (comprising 3D center coordinates, dimensions, and 6D orientation) along with category predictions $\hat{\mathbf{C}}$.

The core mechanism is **2D-to-3D lifting**: G-DINO's vision-language fusion capability first produces open-set 2D detections; a 3D bounding box head then estimates lifting parameters from 2D object queries to differentiably map 2D boxes to 3D boxes. The entire pipeline is end-to-end trainable, enabling joint optimization of 2D and 3D objectives.

Pipeline:
1. An image encoder (Swin-T/B) extracts image features $\mathbf{q}_{\text{image}}$; a text backbone (BERT) extracts text features $\mathbf{q}_{\text{text}}$.
2. A cross-modal Transformer decoder fuses image and text features across layers, producing 2D object queries $\mathbf{q}_{2d}^i$.
3. A 2D box head predicts 2D detections; categories are determined by the similarity between queries and text embeddings (open-set classification).
4. A Geometry-aware 3D Query Generation module fuses 2D queries with camera embeddings and depth features to produce 3D queries $\mathbf{q}_{3d}^i$.
5. A 3D bounding box head predicts lifting parameters (projected 3D center offset, log depth, log dimensions, 6D rotation) from 3D queries; the final 3D detections are obtained by combining these with the 2D boxes and camera intrinsics $\mathbf{K}$.

### Key Designs

1. **3D Bounding Box Head (Differentiable 2D→3D Lifting)**: Predicts 12-dimensional 3D attributes from 2D object queries. 3D localization is achieved by predicting the offset $[\hat{u}, \hat{v}]$ between the projected 3D center and the 2D box center, together with a scaled log depth $\hat{d}$: $\hat{z} = \exp(\hat{d}/s_{\text{depth}})$. 3D dimensions are predicted as scaled log values without relying on category-specific size priors, since such priors are unavailable in open-set settings. Orientation is parameterized using 6D rotation (rather than yaw-only), enabling generalization to both indoor and outdoor scenes. The entire lifting process is differentiable: $\hat{\mathbf{D}}^{3D}_i = \mathbf{Lift}(\text{MLP}^{3D}_i(\mathbf{q}_{3d}^i), \hat{\mathbf{D}}^{2D}_i, \mathbf{K})$.

2. **Canonical Image Space**: Addresses the core problem of inconsistent image resolutions and camera intrinsics across datasets. Prior methods (e.g., Cube R-CNN resizing by short edge with bottom-right padding; G-DINO resizing by long edge with bottom-right padding) suffer from: (a) wasteful zero-padding that consumes GPU memory; (b) camera intrinsics $\mathbf{K}$ not being updated after resizing, causing train-inference inconsistency; and (c) violation of the central projection assumption. 3D-MOOD fixes the input resolution to $[H_c \times W_c]$ (800×1333), resizes images while preserving aspect ratio, applies **center padding**, and adjusts intrinsics accordingly. This ensures a consistent observation space across training and inference while reducing GPU memory consumption (17 GB vs. 21–23 GB per batch of 2).

3. **Auxiliary Metric Depth Estimation**: An FPN extracts depth features $\mathbf{F}$; a Transformer block generates depth features $\mathbf{F}^d_{16}$, which are conditioned on camera embeddings $\mathbf{E}$ (following UniDepth's design) and upsampled to 1/8 resolution to predict full-image log depth $\hat{d}_{\text{full}}$, sharing the depth scale factor $s_{\text{depth}}$. Supervised with a scale-invariant log loss, this auxiliary branch provides global scene geometry understanding and serves as a conditioning signal for subsequent query generation.

4. **Geometry-aware 3D Query Generation**: To improve cross-domain generalization of 3D estimation, geometric priors are injected into 2D object queries in two steps: (1) cross-attention with camera embeddings $\mathbf{E}$ to make queries aware of scene-specific camera properties; (2) cross-attention with depth features $\mathbf{F}^d_8|\mathbf{E}$ to align depth estimation with 3D box estimation. Notably, gradients are **detached** in the depth feature cross-attention to stabilize training. Ablations show this module yields the largest improvement in open-set settings (+1.3 ODS), demonstrating that encoding geometric priors is critical for generalization.

### Loss & Training
- **2D loss $L_{2D}$**: L1 + GIoU (box regression) + contrastive loss (classification between predicted objects and language tokens, inherited from GLIP).
- **3D loss $L_{3D}$**: L1 loss supervising each 3D attribute independently (projected center offset, depth, dimensions, orientation).
- **Auxiliary depth loss $L^{aux}_{depth}$**: Scale-invariant log loss with weight $\lambda_{\text{depth}}=10$.
- Total loss: $L_{\text{final}} = \sum_{i=0}^{l}(L^i_{2D} + L^i_{3D}) + \lambda_{\text{depth}} L^{aux}_{\text{depth}}$
- Each decoder layer has independent 2D and 3D heads (multi-layer supervision).
- Depth ground truth is sourced from Omni3D sub-dataset depth annotations, projected LiDAR points, or SfM points.
- Training: 120 epochs, batch size 128, lr = 0.0004; ablations use 12 epochs, batch size 64.
- Backbone: Swin-T (29M) / Swin-B (88M).

## Key Experimental Results

### Open-Set Results (Trained on Omni3D → Tested on AV2/ScanNet)

| Dataset | Metric | 3D-MOOD (Swin-B) | Cube R-CNN | OVM3D-Det | Gain |
|---------|--------|------------------|------------|-----------|------|
| Argoverse 2 | ODS | **23.8** | 8.9 | 8.8 | +14.9 vs. Cube R-CNN |
| Argoverse 2 | ODS(N) Novel | **14.8** | 0.0 | 1.7 | Zero → functional |
| ScanNet | ODS | **31.5** | 19.5 | 16.3 | +12.0 vs. Cube R-CNN |
| ScanNet | ODS(N) Novel | **15.7** | 0.0 | 8.8 | +6.9 vs. OVM3D-Det |

### Closed-Set Results (Omni3D test)

| Method | AP3D omni↑ |
|--------|-----------|
| Cube R-CNN | 23.3 |
| Uni-MODE* | 28.2 |
| **3D-MOOD (Swin-T)** | **28.4** |
| **3D-MOOD (Swin-B)** | **30.0** |

### Ablation Study

| Setting | CI | Depth | GA | AP3D omni (closed-set) | ODS open (open-set) |
|---------|-----|-------|-----|------------------------|---------------------|
| Baseline | - | - | - | 24.1 | 23.6 |
| +Canonical Image | ✓ | - | - | 25.5 (+1.4) | 24.5 (+0.9) |
| +Depth | ✓ | ✓ | - | 26.2 (+0.7) | 24.7 (+0.2) |
| +Geometry-aware | ✓ | ✓ | ✓ | **26.8 (+0.6)** | **26.0 (+1.3)** |

- Canonical Image Space benefits both closed-set and open-set performance while reducing GPU memory (21→17 GB).
- The auxiliary depth head yields larger gains on closed-set (+0.7) than open-set (+0.2); the authors attribute this to limited diversity in Omni3D depth annotations.
- Geometry-aware query generation provides the most significant open-set improvement (+1.3 ODS), demonstrating that geometric priors are critical for cross-domain generalization.
- Geometry-aware query generation converges faster than Cube R-CNN's virtual depth (26.8 vs. 21.6 at 12 epochs).

## Highlights & Insights
- **First formulation and solution of open-set monocular 3DOD**: Establishes complete benchmarks (Omni3D→AV2/ScanNet) and the ODS evaluation metric.
- **Elegant inheritance of open-set capability from 2D to 3D**: No vision-language pairs in 3D are required; classification ability is entirely borrowed from the 2D open-set detector via a differentiable 2D→3D lifting module.
- **Canonical Image Space design**: Conceptually simple yet effective—fixed resolution + center padding + synchronized intrinsic adjustment, with the added benefit of reduced GPU memory.
- **ODS evaluation metric**: Uses normalized distance rather than IoU for matching, yielding fairer evaluation for small or thin objects; incorporates TP error terms (ATE/ASE/AOE) for a more comprehensive assessment than AP alone.
- **Advantage of end-to-end joint training**: OVM3D-Det relies on a pipeline approach (G-DINO + SAM + UniDepth + LLM for pseudo-GT generation) that precludes end-to-end optimization, resulting in substantially inferior performance.

## Limitations & Future Work
- **Slower inference**: 3D-MOOD (Swin-T) runs at 17 FPS vs. 68 FPS for Cube R-CNN (DLA-34), reflecting the cost of a heavy backbone and multiple heads.
- **Limited open-set generalization from the auxiliary depth head**: Depth training data lacks diversity (Omni3D only); incorporating more diverse depth data (e.g., MiDaS/UniDepth pretraining) could further improve performance.
- **Depth estimation accuracy remains inferior**: Absolute relative error of 9.1% on KITTI Eigen-split, well behind UniDepth (4.21%) and Metric3Dv2 (4.4%).
- **IoU-based AP remains very low in open-set settings**: Even when ODS is high, IoU-based AP3D is still very low (e.g., 14.7% on AV2), indicating substantial room for improvement in 3D localization accuracy.
- Future work could explore using a stronger depth foundation model as a frozen feature provider rather than training the depth head from scratch.
- Extending open-set capability to 3D tracking is a natural direction (the authors have prior work in 3D tracking, e.g., CC-3DT).

## Related Work & Insights

| Method | Key Difference | Advantage | Disadvantage |
|--------|---------------|-----------|--------------|
| **Cube R-CNN** | Closed-set unified 3DOD; uses category priors for size prediction and virtual depth | 3D-MOOD surpasses Cube R-CNN by 6.7 AP on closed-set and detects novel categories | Cube R-CNN is faster (68 FPS) |
| **OVM3D-Det** | Pipeline approach: uses foundation models to generate pseudo 3D GT | 3D-MOOD is end-to-end trainable with significantly better performance; OVM3D-Det's depth model is also trained on target-domain data | OVM3D-Det does not require 3D annotations |
| **Uni-MODE** | Domain confidence + joint BEV detector for indoor/outdoor | 3D-MOOD achieves slightly better closed-set performance on Omni3D (30.0 vs. 28.2) with additional open-set capability | Uni-MODE is not open-sourced, preventing direct open-set comparison |

The **2D→3D lifting paradigm** is broadly applicable: given a strong 2D open-set detector, learning lifting parameters to extend predictions to 3D is a transferable approach for open-set 3D segmentation, tracking, and related tasks. The **canonical image space** strategy of center padding with synchronized intrinsic adjustment can be adopted by any cross-dataset detection or depth estimation method. The **ODS metric** provides a standardized evaluation protocol for future work in this area. The paper demonstrates that, in the absence of 3D vision-language data, borrowing semantic alignment capability from 2D foundation models is a viable and effective strategy, inspiring analogous approaches in other 3D tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First end-to-end open-set monocular 3DOD with significant contributions in problem formulation and benchmarking; individual technical modules (depth head, canonical space) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation covering closed-set, open-set, cross-domain, ablation, backbone comparison, and metric analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with clear motivation; some tables and equations could be formatted more compactly.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new research direction in open-set monocular 3DOD, provides benchmarks and evaluation metrics, and is likely to serve as a foundational reference for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)
- [\[AAAI 2026\] MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection](../../AAAI2026/object_detection/monoclue_object-aware_clustering_enhances_monocular_3d_object_detection.md)
- [\[CVPR 2026\] Towards Intrinsic-Aware Monocular 3D Object Detection](../../CVPR2026/object_detection/towards_intrinsic-aware_monocular_3d_object_detection.md)
- [\[ICCV 2025\] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)
- [\[CVPR 2026\] MonoSAOD: Monocular 3D Object Detection with Sparsely Annotated Label](../../CVPR2026/object_detection/monosaod_monocular_3d_object_detection_with_sparsely_annotated_label.md)

</div>

<!-- RELATED:END -->

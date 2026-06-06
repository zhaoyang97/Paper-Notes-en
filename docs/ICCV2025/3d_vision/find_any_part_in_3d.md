---
title: >-
  [Paper Note] Find Any Part in 3D
description: >-
  [ICCV 2025][3D Vision][3D part segmentation] This paper proposes Find3D, an automated 3D data annotation engine driven by 2D foundation models (SAM + Gemini) that generates 2.1 million part annotations. The resulting mod…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D part segmentation"
  - "open-world"
  - "data engine"
  - "contrastive learning"
  - "zero-shot generalization"
date: 2026-05-08
content_hash: 2bd1d7f18cd1181d
---

# Find Any Part in 3D

**Conference**: ICCV 2025
**arXiv**: [2411.13550](https://arxiv.org/abs/2411.13550)  
**Code**: [Project Page](https://ziqi-ma.github.io/find3dsite/)  
**Area**: 3D Vision / 3D Part Segmentation
**Keywords**: 3D part segmentation, open-world, data engine, contrastive learning, zero-shot generalization

## TL;DR

This paper proposes Find3D, an automated 3D data annotation engine driven by 2D foundation models (SAM + Gemini) that generates 2.1 million part annotations. The resulting model is the first to simultaneously achieve open-world, cross-category, part-level, and feed-forward inference capabilities in 3D segmentation, yielding a 260% zero-shot mIoU improvement and inference speeds 6–300× faster than existing methods.

## Background & Motivation

- **Why has a 3D foundation model not yet emerged?** The key bottleneck is data scarcity.
- **Limitations of existing 3D part segmentation datasets**:
    - ShapeNet-Part: only 16 categories and 41 part types, with all chairs sharing the same orientation.
    - PartNet-E: 45 categories but only 40 unique part types, limited to simple household objects.
    - Combined, these datasets cover only 71 unique part types.
- **Problems with existing methods**:
    - 2D aggregation methods (PointCLIP, PartSLIP++): lack 3D geometric information, exhibit cross-view inconsistency, and have slow inference.
    - Closed-set methods (PointNeXt): cannot generalize to unseen categories.
    - Test-time optimization methods (LERF, Feature3DGS): require per-scene optimization, taking minutes per object.
    - Distillation methods (PartDistill): distill per category, precluding zero-shot inference.
- **Core Idea**: Once the data challenge is resolved, a simple and general training strategy suffices to yield a powerful model.

## Method

### Overall Architecture

Find3D consists of two components:
1. **A scalable data engine**: automatically annotates 3D assets sourced from the internet.
2. **An open-world 3D part model**: a point cloud Transformer trained with contrastive learning.

### Data Engine

The pipeline for automatically annotating Objaverse 3D assets:

1. **Multi-view rendering**: each 3D asset is rendered from multiple camera angles.
2. **Canonical view selection**: Gemini selects the most natural viewpoint from 10 candidates.
3. **SAM segmentation**: grid-point prompts are used to segment each rendered image.
4. **Filtering**: masks that are too small (<350 pixels), too large (>20% of pixels), or low-confidence are discarded.
5. **Gemini labeling**: each mask is overlaid on the original image and Gemini is queried to name the corresponding part.
6. **Merging**: masks sharing the same label are merged.
7. **3D back-projection**: masks are mapped to 3D points in the point cloud via projection geometry.
8. **Text embedding**: label text is embedded using SigLIP as the supervision signal.

**Data scale**: processing 30K Objaverse objects (761 categories) yields 2.1 million part annotations covering 124,615 unique part types — **1,775×** more than existing datasets.

### Open-World 3D Part Model

**Architecture**: The model adopts the PT3 (Point Transformer V3) architecture, voxelizing and serializing point clouds via space-filling curves. A 4-layer MLP is appended after the final Transformer layer to align point features to SigLIP's 768-dimensional embedding space. The model has 46.2M parameters in total.

**Contrastive learning training**: Since a single point may carry multiple labels (location, material, function, etc.) and many points are unannotated, contrastive learning is employed:

$$l_i = -\log \frac{\exp(f(C_i) \cdot T(\text{label}_i))}{\sum_{j=1}^{|\mathcal{B}|} \exp(f(C_i) \cdot T(\text{label}_j))}$$

where $f(C_i)$ is the mean feature of the point set corresponding to the label (pooled for denoising), and $T(\text{label}_i)$ is the SigLIP text embedding. Each batch contains 64 objects with approximately 3,000 positive pairs.

**Inference**: For an arbitrary text query $s$, its SigLIP embedding is computed and the cosine similarity to each point feature is calculated; each point is assigned to the query with the highest similarity.

**Data augmentation**: Random rotations (all three axes), scaling, flipping, jittering, and color augmentation are applied to prevent over-reliance on pose and color.

## Key Experimental Results

### Main Results: Open-World Method Comparison (Objaverse-General + ShapeNet-Part)

| Method | Obj-Gen Seen mIoU | Obj-Gen Unseen mIoU | ShapeNet-Part Canonical mIoU | ShapeNet-Part Rotated mIoU |
|------|-------------------|---------------------|------------------------------|---------------------------|
| **Find3D** | **34.10** | **27.41** | **28.39** | **29.64** |
| PointCLIPV2 | 11.27 | 11.09 | 20.22 | 18.19 |
| PartSLIP++ | 15.03 | 10.43 | 6.46 | 6.03 |
| OpenMask3D | 11.93 | 10.31 | 10.37 | 14.56 |

Key finding: on zero-shot unseen categories, Find3D achieves a **260%** mIoU improvement (27.41 vs. 11.09). Even in zero-shot evaluation, it surpasses PointCLIPV2 trained on ShapeNet-Part.

### Inference Speed and Model Property Comparison

| Method | Time | Open-World | Cross-Category | Part-Level | Feed-Forward |
|------|------|---------|--------|--------|---------|
| **Find3D** | **0.9s** | ✓ | ✓ | ✓ | ✓ |
| PointCLIPV2 | 5.4s | ✓ | ✓ | ✓ | ✗ |
| PartSLIP++ | 174.3s | ✓ | ✗ | ✓ | ✗ |
| OpenMask3D | 296.5s | ✓ | ✓ | ✗ | ✗ |
| PointNeXt | 1.4s | ✗ | ✓ | ✓ | ✓ |

Find3D is the only method satisfying all four properties simultaneously, running **6–300×** faster than other open-world methods.

### Generalization Comparison (Closed-Set Methods on ShapeNetPart-V2)

| Method | Training Data | ShapeNet-Part mIoU | ShapeNetPart-V2 mIoU |
|------|----------|-------------------|---------------------|
| PointNeXt | ShapeNet-Part | 80.44 | 28.70 (↓64%) |
| PartDistill | ShapeNet-Part | 63.9 | N/A |
| **Find3D** | Data Engine | 28.39 (zero-shot) | **42.15** |

Key finding: PointNeXt suffers a 64% performance drop on out-of-domain data, while Find3D's zero-shot performance exceeds it by 1.5×.

### Data Scale Analysis

A clear positive scaling trend is observed between the number of training categories and zero-shot mIoU:
- 16 categories (ShapeNet scale) → ~14% mIoU
- 45 categories (PartNet-E scale) → ~18% mIoU
- 761 categories (full Find3D data) → ~27% mIoU

### Robustness Comparison

| Condition | PointCLIPV2 Change | Find3D Change |
|----------|-------------------|---------------|
| Modified query prompt | ↓64% | ↓1% |
| Random object rotation | ↓46% | ↑3% |
| Domain shift (ShapeNetPart-V2) | ↓56% | ↑20% |

## Highlights & Insights

1. **Data engine paradigm**: The core contribution is not architectural innovation but the construction of an automated pipeline from 2D foundation models to 3D annotations, scaling dataset size by 1,775×.
2. **Scale as generalization**: Data scaling analysis demonstrates that generalization capability derives directly from training data diversity, echoing scaling laws observed in NLP and CV.
3. **Simple yet effective training strategy**: Contrastive learning combined with data augmentation suffices — no per-category fine-tuning, multi-pass inference, or predefined part ordering logic is required.
4. **Flexible query capability**: Supports text queries at varying granularities (e.g., "limbs" vs. "arms" + "legs") and across different dimensions (body parts vs. clothing).

## Limitations & Future Work

- The voxel sampling resolution (0.02) limits recognition of inconspicuous fine-grained parts (e.g., buttons on a surface).
- Rotation-equivariant training causes the model to tend to predict identical labels for symmetric parts.
- The model relies solely on the point cloud modality, lacking the fine-grained detail provided by 2D image inputs.
- The data engine depends on the quality of SAM and Gemini; annotation errors propagate to downstream performance.

## Related Work & Insights

- **Alignment with scaling law intuitions**: Analogous to the success of the GPT series in NLP, this paper validates the "data scale → generalization" pathway in the 3D domain.
- **Reproducibility of the data engine**: The pipeline employs off-the-shelf SAM + Gemini components and is clearly replicable, offering a methodological reference for other 3D tasks.
- **Lessons from closed-set to open-world transition**: Over-engineering for small datasets should be avoided; data diversity and training generality should be prioritized instead.

## Rating ⭐⭐⭐⭐⭐

An excellent piece of work. The data engine design philosophy carries broad inspirational value, and the experimental design is comprehensive and convincing. A 260% mIoU improvement and 300× speedup are impressive results. The scaling analysis further substantiates the value of the proposed data engine.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] From One to More: Contextual Part Latents for 3D Generation](from_one_to_more_contextual_part_latents_for_3d_generation.md)
- [\[ICCV 2025\] SAS: Segment Any 3D Scene with Integrated 2D Priors](sas_segment_any_3d_scene_with_integrated_2d_priors.md)
- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[ICCV 2025\] AnyI2V: Animating Any Conditional Image with Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)
- [\[ICCV 2025\] TAPNext: Tracking Any Point (TAP) as Next Token Prediction](tapnext_tracking_any_point_tap_as_next_token_prediction.md)

</div>

<!-- RELATED:END -->

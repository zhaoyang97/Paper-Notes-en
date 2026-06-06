---
title: >-
  [Paper Note] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation
description: >-
  [AAAI 2026][3D Vision][class-agnostic 3D instance segmentation] This paper proposes ASSIST-3D, a synthetic data pipeline that generates high-quality annotated data for class-agnostic 3D instance segmentation through thre…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "class-agnostic 3D instance segmentation"
  - "3D scene synthesis"
  - "synthetic data"
  - "point cloud"
  - "LLM-guided layout"
date: 2026-05-08
content_hash: 61cbb0d6b3ee072b
---

# ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation

**Conference**: AAAI 2026
**arXiv**: [2512.09364](https://arxiv.org/abs/2512.09364)  
**Code**: Not released  
**Area**: 3D Vision
**Keywords**: class-agnostic 3D instance segmentation, 3D scene synthesis, synthetic data, point cloud, LLM-guided layout

## TL;DR

This paper proposes ASSIST-3D, a synthetic data pipeline that generates high-quality annotated data for class-agnostic 3D instance segmentation through three stages: heterogeneous object selection, LLM-guided scene layout generation, and realistic point cloud construction, significantly improving model generalization.

## Background & Motivation

**Class-agnostic 3D instance segmentation** aims to segment all object instances in a scene without relying on semantic categories, including categories unseen during training. Existing methods are hampered by data scarcity and noise from 2D segmentation.

**Traditional class-aware methods** depend on annotations over predefined categories, covering only dozens of classes, and thus fail to handle the thousands of unseen objects encountered in the real world.

**Methods based on 2D foundation models** (e.g., SAM) exhibit strong generalization but suffer from inherent 2D segmentation errors and multi-view fusion inconsistencies, making reliable lifting to 3D segmentation difficult.

**Real 3D data acquisition is costly**, annotations are incomplete, and scene complexity is limited, directly constraining data diversity and the generalization ceiling of trained models.

**Existing 3D scene synthesis methods are insufficient**: Holodeck uses LLMs to select objects, biasing toward common categories with insufficient geometric diversity and contextual complexity; RandomRooms places objects randomly, resulting in implausible layouts.

**Core Insight**: Data diversity is the key driver for improving generalization in class-agnostic segmentation, and must simultaneously satisfy three principles: **geometric diversity**, **contextual complexity**, and **layout plausibility**.

## Method

### Overall Architecture

ASSIST-3D consists of three stages: (1) heterogeneous object selection → (2) scene layout generation → (3) realistic point cloud construction. Synthetic data is jointly trained with real data (ScanNetV2) using Mask3D (with the multi-class head replaced by a binary objectness classifier), and the optimization objective combines real and synthetic losses with weight $\alpha=0.5$.

### Key Design 1: Heterogeneous Object Selection

- Uses a subset of Objaverse (50,000 3D models across 800 categories) as the asset pool, partitioned into three groups by placement: $\mathcal{O}_{\text{floor}}$ (floor objects such as furniture), $\mathcal{O}_{\text{wall}}$ (wall-mounted objects such as paintings), and $\mathcal{O}_{\text{obj}}$ (objects placeable on top of the former two).
- Each scene uniformly samples $M_1=100$ floor objects and $M_2=50$ wall objects, with 5 small objects further sampled per large object, yielding approximately $5(M_1+M_2)$ additional objects.
- **Breaking conventional category co-occurrence patterns** enhances contextual complexity; an alternating complementary sampling strategy (with probability 0.7 to prioritize categories present in real data) mitigates incomplete annotation issues.

### Key Design 2: Scene Layout Generation

- **GPT-4** is used to infer plausible spatial relationships (orientation and relative position) between objects, but absolute coordinates are not directly output due to the limited spatial reasoning capacity of LLMs.
- A **depth-first search (DFS)** strategy sequentially places objects: the floor is discretized into a uniform grid, and starting from the first object, feasible grid cells are iteratively found for each subsequent object, with backtracking upon constraint violation.
- Among all feasible configurations, the one that **accommodates the most objects** is selected; wall-mounted and surface objects are handled by the same procedure.

### Key Design 3: Realistic Point Cloud Construction

- Rather than directly sampling points from mesh surfaces (which produces overly uniform distributions lacking noise and occlusion), the pipeline simulates the acquisition process of a real SLAM system.
- The mid-height plane of the scene is uniformly divided into $0.1 \times 0.1\text{m}^2$ grids, from which 5 optimal viewpoints are selected via FPS.
- From each viewpoint, 12 RGB-D images are rendered at $30°$ intervals (60 images total); the final point cloud and instance annotations are generated via depth projection, coordinate transformation, and voxel downsampling.

### Loss & Training

- Built on the Mask3D framework, using a combination of **binary cross-entropy loss + dice loss + mask loss** as the optimization objective.
- Real and synthetic data are trained jointly, with synthetic data loss weight $\alpha=0.5$.
- Total training: 600 epochs, batch size 36, distributed across 6 A100 GPUs.
- The synthetic dataset contains 2,000 scenes with approximately 134,000 object instances, averaging 67 objects per scene.

## Key Experimental Results

### Main Results: Comparison with SOTA Methods (Class-Agnostic 3D Instance Segmentation)

| Method | ScanNet++ AP | ScanNet++ AP50 | S3DIS AP | S3DIS AP50 | ScanNetV2 AP | ScanNetV2 AP50 |
|--------|:-----------:|:--------------:|:--------:|:----------:|:------------:|:--------------:|
| Baseline (Mask3D) | 12.0 | 21.7 | 13.6 | 23.2 | 46.6 | 69.0 |
| SA3DIP | 19.6 | 32.4 | 25.7 | 42.4 | 41.6 | 64.6 |
| SAI3D | 17.1 | 31.1 | 24.8 | 42.4 | 30.8 | 50.5 |
| **ASSIST-3D** | **22.2** | **35.5** | **29.0** | **43.9** | **48.1** | **70.7** |

ASSIST-3D outperforms all SOTA methods across all three benchmarks, with particularly notable improvements in cross-domain generalization (ScanNet++ / S3DIS).

### Comparison with Other 3D Scene Synthesis Methods

| Method | Geometric Diversity | Contextual Complexity | Layout Plausibility | ScanNet++ AP | S3DIS AP |
|--------|:------------------:|:--------------------:|:------------------:|:-----------:|:--------:|
| Holodeck | ✗ (0.85) | ✗ (0.38) | ✓ (72) | 14.2 | 18.2 |
| RandomRooms | ✓ (4.37) | ✓ (0.04) | ✗ (23) | 16.6 | 23.5 |
| **ASSIST-3D** | **✓ (4.15)** | **✓ (0.08)** | **✓ (62)** | **22.2** | **29.0** |

ASSIST-3D is the only method that satisfies all three principles simultaneously, substantially outperforming both Holodeck and RandomRooms.

### Ablation Study

- **Geometric diversity**: Expanding from 1 cluster to 5 clusters improves ScanNet++ AP from 14.6 to 22.2 (+52%).
- **Contextual complexity**: Reducing co-occurrence probability from 100% to 0% improves ScanNet++ AP from 17.2 to 22.2.
- **Realistic point cloud construction**: Direct mesh sampling yields AP of only 14.2, while the rendering-based approach achieves 22.2 (+56%), substantially closing the domain gap.
- **Data scalability**: More object categories and more synthetic scenes consistently yield further performance gains.

## Highlights & Insights

1. **Systematic framework**: The paper explicitly proposes three principles that synthetic 3D data should satisfy (geometric diversity, contextual complexity, layout plausibility) and designs corresponding technical solutions for each.
2. **LLM + DFS layout generation** cleverly combines LLM commonsense reasoning with physics-constraint checking via search algorithms, yielding layouts that are both plausible and controllable.
3. **Realistic point cloud construction** simulates a real SLAM acquisition pipeline, effectively bridging the synthetic-to-real domain gap — a design with broad applicability to synthetic data training in general.
4. **Thorough ablation study**: Each of the three principles is validated with fine-grained quantitative experiments, making the contribution of each component clearly measurable.
5. **Strong cross-domain generalization**: The improvements on ScanNet++ and S3DIS are particularly prominent, validating the generalization benefit of synthetic data for unseen categories.

## Limitations & Future Work

1. **Reliance on GPT-4 inference** constrains the cost and speed of scene layout generation due to LLM API calls.
2. **Asset pool remains limited**: Although 50,000 models across 800 categories surpass real annotated data in diversity, coverage of long-tail real-world scenes remains incomplete.
3. **Only indoor scenes are evaluated**; applicability to outdoor or larger-scale scenes (e.g., city-level) is not explored.
4. **Texture and material realism of synthetic point clouds** is not thoroughly discussed; domain gaps at the color level may still persist.
5. **Scalability of DFS-based layout**: As the number of objects per scene increases, the search space grows exponentially; practical efficiency is not reported in detail.

## Related Work & Insights

- **Class-agnostic 3D segmentation**: OpenMask3D replaces the classification head with a binary classifier but has limited generalization; SAI3D and SA3DIP leverage SAM for 2D→3D lifting but are affected by 2D errors; Segment3D uses pseudo-label pretraining. ASSIST-3D addresses the problem from a data generation perspective and is complementary to the above methods.
- **3D scene synthesis**: Holodeck uses LLMs for end-to-end generation of high-quality scenes but lacks object diversity; RandomRooms introduces diversity through randomization but at the cost of layout plausibility. ASSIST-3D combines the strengths of both.
- **Synthetic data training**: Simulating real sensor acquisition pipelines to construct point clouds is an established practice in autonomous driving; this paper introduces the approach to indoor 3D segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Reframes 3D scene synthesis as a data augmentation strategy for class-agnostic segmentation; the three-principle formulation and LLM+DFS layout design are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation across three benchmarks with fine-grained ablations analyzing each component and principle.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rigorous problem definition and principled motivation.
- Value: ⭐⭐⭐⭐ — The three principles for synthetic data generation and the realistic point cloud construction pipeline offer broadly applicable reference value for the 3D vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [\[AAAI 2026\] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](../../ICCV2025/3d_vision/cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](../../ICCV2025/3d_vision/disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)

</div>

<!-- RELATED:END -->

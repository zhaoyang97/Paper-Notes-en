---
title: >-
  [Paper Note] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation
description: >-
  [AAAI 2026][3D Vision][class-agnostic 3D instance segmentation] This paper proposes the ASSIST-3D synthetic data pipeline, which generates high-quality annotated data for class-agnostic 3D instance segmentation through three stages: heterogeneous object selection, LLM-guided scene layout generation, and realistic point cloud construction, significantly improving model generalization.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "class-agnostic 3D instance segmentation"
  - "3D scene synthesis"
  - "synthetic data"
  - "point cloud"
  - "LLM-guided layout"
date: 2026-05-08
content_hash: 95debbb83969fe7a
---

# ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation

**Conference**: AAAI 2026  
**arXiv**: [2512.09364](https://arxiv.org/abs/2512.09364)  
**Code**: Not open-sourced  
**Area**: 3D Vision  
**Keywords**: class-agnostic 3D instance segmentation, 3D scene synthesis, synthetic data, point cloud, LLM-guided layout

## TL;DR

This paper proposes the ASSIST-3D synthetic data pipeline, which generates high-quality annotated data for class-agnostic 3D instance segmentation through three stages: heterogeneous object selection, LLM-guided scene layout generation, and realistic point cloud construction, significantly improving model generalization.

## Background & Motivation

**Class-agnostic 3D instance segmentation** aims to segment all object instances in a scene (including unseen categories during training) without relying on semantic categories. However, existing methods suffer from data scarcity and 2D segmentation noise.

**Traditional class-aware methods** rely on annotations of predefined categories, which only cover dozens of classes and fail to handle thousands of unseen objects in the real world.

**Based on 2D foundation models** (such as SAM), although possessing strong generalization capabilities, existing methods have inherent flaws in 2D segmentation errors and multi-view consistency fusion, making it difficult to reliably map them to 3D segmentation results.

**Collecting real 3D data is expensive**, resulting in incomplete annotations and limited scene complexity, which directly restricts the model's data diversity and generalization upper bound.

**Existing 3D scene synthesis methods do not meet requirements**: Holodeck leverages LLMs to select objects, which biases toward common categories (lacking geometric diversity and contextual complexity), while RandomRooms places objects randomly, leading to unrealistic layouts.

**Key Insight**: Data diversity is the key driving force for improving the generalization ability of class-agnostic segmentation, which must simultaneously satisfy three principles: **geometric diversity**, **contextual complexity**, and **layout plausibility**.

## Method

### Overall Architecture

ASSIST-3D consists of three phases: (1) heterogeneous object selection $\rightarrow$ (2) scene layout generation $\rightarrow$ (3) realistic point cloud construction. The synthetic data and real data (ScanNetV2) are jointly trained on Mask3D (replacing the multi-class classification head with a binary objectness classifier), and the optimization objective merges real and synthetic losses with a weight ($\alpha=0.5$).

### Key Design 1: Heterogeneous Object Selection

- A subset of Objaverse (50,000 3D models across 800 categories) is used as the asset library, divided into three groups based on placement: $\mathcal{O}_{\text{floor}}$ (floor objects such as furniture), $\mathcal{O}_{\text{wall}}$ (wall objects such as paintings), and $\mathcal{O}_{\text{obj}}$ (objects that can be placed on top of the first two groups).
- For each scene, $M_1=100$ floor objects and $M_2=50$ wall objects are uniformly sampled. For each large object, 5 small objects are further sampled, resulting in approximately $5(M_1+M_2)$ additional objects.
- Contextual complexity is enhanced by **breaking conventional category co-occurrence patterns**; complementary sampling strategies are used alternately (prioritizing categories that appear in real data with a probability of 0.7) to compensate for incomplete annotations.

### Key Design 2: Scene Layout Generation

- **GPT-4** is utilized to infer reasonable spatial relationships (orientations and relative positions) between objects, but it does not directly output absolute coordinates (due to the LLM's limited spatial reasoning capability).
- A **depth-first search (DFS)** strategy is adopted to place objects one by one: the floor is discretized into a uniform grid, starting from the first object, and feasible grids are progressively found to place the current object, backtracking if constraints are not met.
- Among all feasible solutions, the one that **places the most objects** is selected. Wall and surface objects are processed using the same pipeline.

### Key Design 3: Realistic Point Cloud Construction

- Instead of directly sampling point clouds from mesh surfaces (which are overly uniform and lack noise and occlusion), the acquisition process of a real SLAM system is simulated.
- The middle height plane of the scene is uniformly partitioned into a grid of $0.1 \times 0.1\text{m}^2$, and 5 optimal observation points are selected using FPS (Farthest Point Sampling).
- At each observation point, 12 RGB-D images are rendered at $30°$ intervals (60 images in total). The final point cloud and instance annotations are generated through depth projection, coordinate transformation, and voxel downsampling.

### Loss & Training

- Based on the Mask3D framework, a combination of **binary cross-entropy loss + dice loss + mask loss** is used as the optimization objective.
- Joint training is conducted on real and synthetic data, with the synthetic data loss weight set to $\alpha=0.5$.
- The model is trained for 600 epochs with a batch size of 36, using distributed training across 6 A100 GPUs.
- The synthetic dataset contains 2,000 scenes with approximately 134,000 object instances, averaging 67 objects per scene.

## Key Experimental Results

### Table 1: Comparison with SOTA methods (Class-Agnostic 3D Instance Segmentation)

| Method | ScanNet++ AP | ScanNet++ AP50 | S3DIS AP | S3DIS AP50 | ScanNetV2 AP | ScanNetV2 AP50 |
|------|:-----------:|:--------------:|:--------:|:----------:|:------------:|:--------------:|
| Baseline (Mask3D) | 12.0 | 21.7 | 13.6 | 23.2 | 46.6 | 69.0 |
| SA3DIP | 19.6 | 32.4 | 25.7 | 42.4 | 41.6 | 64.6 |
| SAI3D | 17.1 | 31.1 | 24.8 | 42.4 | 30.8 | 50.5 |
| **ASSIST-3D** | **22.2** | **35.5** | **29.0** | **43.9** | **48.1** | **70.7** |

ASSIST-3D comprehensively outperforms SOTA methods across three datasets, showing particularly remarkable improvements in cross-domain generalization (ScanNet++/S3DIS).

### Table 2: Comparison with other 3D scene synthesis methods

| Method | Geometric Diversity | Contextual Complexity | Layout Plausibility | ScanNet++ AP | S3DIS AP |
|------|:---------:|:----------:|:---------:|:-----------:|:--------:|
| Holodeck | ✗ (0.85) | ✗ (0.38) | ✓ (72) | 14.2 | 18.2 |
| RandomRooms | ✓ (4.37) | ✓ (0.04) | ✗ (23) | 16.6 | 23.5 |
| **ASSIST-3D** | **✓ (4.15)** | **✓ (0.08)** | **✓ (62)** | **22.2** | **29.0** |

ASSIST-3D is the only method that satisfies all three principles simultaneously, significantly outperforming Holodeck and RandomRooms.

### Ablation Study Highlights

- **Geometric Diversity**: Expanding from 1 cluster to 5 clusters improves ScanNet++ AP from 14.6 to 22.2 (+52%).
- **Contextual Complexity**: Reducing the co-occurrence probability from 100% to 0% improves ScanNet++ AP from 17.2 to 22.2.
- **Realistic Point Cloud Construction**: Sampling directly from the mesh yields an AP of only 14.2, whereas the rendering method achieves 22.2 (+56%), greatly narrowing the domain gap.
- **Data Scalability**: More object categories and synthetic scenes continuously bring performance gains.

## Highlights & Insights

1. **Systematic Framework**: Explicitly proposes three principles for 3D synthetic data (geometric diversity, contextual complexity, and layout plausibility) and designs corresponding technical solutions.
2. **LLM + DFS Layout Generation** cleverly combines the common-sense reasoning of LLMs with the physical constraint checking of search algorithms, achieving both plausibility and controllability.
3. **Realistic Point Cloud Construction** simulates the real SLAM acquisition process to effectively bridge the synthetic-to-real domain gap, serving as a general reference for training on synthetic data.
4. **Thorough Ablation Experiments**: Detailed quantitative verification is performed for each of the three principles, making the contribution of each component clear and measurable.
5. **Strong Cross-Domain Generalization**: The improvement on ScanNet++/S3DIS is particularly prominent, validating the generalization gains of synthetic data on unseen categories.

## Limitations & Future Work

1. **Reliance on GPT-4 Reasoning**: The cost and speed of scene layout generation are constrained by LLM API usage.
2. **Limited Asset Library**: Although 50,000 models across 800 categories are richer than real data, there is still a gap in covering long-tail real-world scenarios.
3. **Indoor-Only Verification**: The applicability to outdoor or larger-scale scenes (e.g., city-scale) is yet to be explored.
4. **Realism of Texture and Material in Synthetic Point Clouds**: Not fully discussed; domain gaps in terms of color may still persist.
5. **Scalability of DFS Layouts**: As the number of objects in a scene increases, the search space grows exponentially, and its practical efficiency is not reported in detail.

## Related Work & Insights

- **Class-agnostic 3D Segmentation**: OpenMask3D replaces the classification head with a binary classifier, but its generalization is limited; SAI3D/SA3DIP utilize SAM for 2D$\rightarrow$3D uplifting but are susceptible to 2D errors; Segment3D uses pseudo-labels for pre-training. ASSIST-3D addresses the problem from the perspective of data generation, complementing these methods.
- **3D Scene Synthesis**: Holodeck uses LLMs to generate high-quality scenes end-to-end but lacks object diversity; RandomRooms introduces diversity through randomization but suffers from unrealistic layouts. ASSIST-3D merges the advantages of both.
- **Synthetic Data Training**: The practice of simulating real sensor acquisition workflows to construct point clouds has precedents in autonomous driving; this paper introduces this concept to indoor 3D segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Repositions 3D scene synthesis as a data augmentation solution for class-agnostic segmentation; the extraction of the three principles and the LLM+DFS layout design are highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensively validated across three benchmarks, with fine-grained ablation studies analyzing each component and principle.
- Writing Quality: ⭐⭐⭐⭐ — Structurally clear, with rigorous logic in problem formulation and derivation of principles.
- Value: ⭐⭐⭐⭐ — The three principles for synthetic data generation and the realistic point cloud construction provide universal reference value for the 3D vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](../../CVPR2025/3d_vision/any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)
- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [\[AAAI 2026\] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[AAAI 2026\] 3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation](3dteethsam_taming_sam2_for_3d_teeth_segmentation.md)

</div>

<!-- RELATED:END -->

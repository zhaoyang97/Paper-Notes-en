---
title: >-
  [Paper Note] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation
description: >-
  [ECCV 2024][Segmentation][Unsupervised 3D Instance Segmentation] A hierarchical clustering framework, Part2Object, is proposed to leverage self-supervised features and 3D objectness priors to layer-by-layer merge part-level over-segmentations into object-level instances. This generates high-quality pseudo-labels for self-training Hi-Mask3D, achieving 3D instance segmentation without any human annotations.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Unsupervised 3D Instance Segmentation"
  - "Hierarchical Clustering"
  - "Pseudo-labels"
  - "Self-training"
  - "3D Objectness Prior"
date: 2026-05-08
content_hash: ead1e4045d3346ea
---

# Part2Object: Hierarchical Unsupervised 3D Instance Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.10084](https://arxiv.org/abs/2407.10084)  
**Area**: Semantic Segmentation  
**Keywords**: Unsupervised 3D Instance Segmentation, Hierarchical Clustering, Pseudo-labels, Self-training, 3D Objectness Prior  

## TL;DR

A hierarchical clustering framework, Part2Object, is proposed to leverage self-supervised features and 3D objectness priors to layer-by-layer merge part-level over-segmentations into object-level instances. This generates high-quality pseudo-labels for self-training Hi-Mask3D, achieving 3D instance segmentation without any human annotations.

## Background & Motivation

1. 3D instance segmentation is a core task in scene understanding, but existing methods heavily rely on large amounts of human-annotated 3D point cloud masks.
2. Labeling 3D data is extremely expensive (annotating a single scene in ScanNet takes hours), limiting the scalability of these methods.
3. Prior unsupervised methods, such as Felzenswalb and the 3D projection of CutLER, suffer from severe over-segmentation or under-segmentation.
4. Single-layer clustering methods struggle to adapt simultaneously to the granularity required for objects of different sizes and geometric structures.
5. The absence of effective stopping criteria to determine when to halt merging leads to erroneous fusion of adjacent objects (e.g., merging a desk and a laptop on it).
6. A hierarchical approach is required to progressively construct instance segmentation from parts to objects while maintaining adaptability to objects of various scales.

## Method

### Overall Architecture

Part2Object adopts a two-stage framework consisting of "bottom-up clustering + self-training":

1. **First Stage — Part2Object Hierarchical Clustering**: Extracts point cloud features utilizing self-supervised features (e.g., DINOv2). Starting from initial over-segmentation (part-level), it progressively merges them into object-level instances through hierarchical clustering to generate pseudo-labels.
2. **Second Stage — Hi-Mask3D Self-Training**: Trains a hierarchical 3D instance segmentation network, Hi-Mask3D, using the pseudo-labels to simultaneously predict part-level and object-level segmentations.

### Key Designs

#### Module 1: Feature-Guided Hierarchical Clustering

At each layer $t$, every cluster pair $(c_i^t, c_j^t)$ is evaluated and merged based on the following condition:

$$c_k^{t+1} \leftarrow c_i^t \cup c_j^t \quad \text{if} \quad \text{rank}(\text{sim}(\boldsymbol{f}_i^t, \boldsymbol{f}_j^t)) \leq K \;\land\; \text{dist}(c_i^t, c_j^t) \leq T$$

where $\boldsymbol{f}_i^t$ is the feature vector of cluster $c_i^t$, $K$ is the nearest neighbor rank threshold for feature similarity, and $T$ is the spatial distance threshold. After merging, the feature of the new cluster is computed through a feature update function $\text{FU}(\cdot)$:

$$\boldsymbol{f}_k^{t+1} \leftarrow \text{FU}(c_k^{t+1})$$

This design simultaneously considers feature similarity and spatial proximity, preventing the erroneous merging of feature-similar but spatially separated regions.

#### Module 2: 3D Objectness Prior Stopping Criteria

To prevent over-merging (such as merging a desk with items on top of it), a stopping criterion based on a 3D objectness prior $B^{3D}$ is introduced:

$$\text{stopCriteria}(c_i^t, c_j^t, B^{3D}) = \begin{cases} \text{True} & \text{if } \exists b \in B^{3D}: \text{IoU}(c_i^t \cup c_j^t, b) > \tau_{iou} \\ \text{False} & \text{otherwise} \end{cases}$$

where the IoU threshold $\tau_{iou} = 0.6$. The 3D objectness prior can be acquired from multi-view projections of 2D detectors (like CutLER), providing rough estimates of object boundaries to guide when clustering should cease merging.

#### Module 3: Hi-Mask3D Hierarchical Self-Training

Hi-Mask3D is extended based on the Mask3D architecture to simultaneously predict part-level and object-level segmentations:
- **Part Queries**: 300 queries, responsible for fine-grained part segmentation.
- **Object Queries**: 150 queries, responsible for object-level instance segmentation.
- The two-level queries learn part-to-object semantic relationships through a hierarchical Transformer decoder.

### Loss & Training

- **Optimizer**: AdamW, learning rate $1 \times 10^{-4}$
- **Scheduler**: OneCycleLR
- **Training epochs**: 600 epochs, Batch size = 4
- **Voxel size**: 0.02
- In the unsupervised class-agnostic setting, no categories are filtered based on semantic labels.
- In the data-efficient setting, the wall/floor categories of ScanNet are filtered following the convention of Mask3D.
- Inference does not use DBSCAN post-processing.

## Key Experimental Results

### Main Results

| Method | Supervision Type | ScanNet mAP | ScanNet mAP@50 | Remarks |
|------|---------|-------------|---------------|------|
| Felzenswalb | Unsupervised | - | Severe over-segmentation | Traditional method |
| CutLER Projection | Unsupervised | - | Severe under-segmentation | 2D→3D projection |
| Mask3D (Class-agnostic) | Fully-supervised | Baseline | Baseline | With annotations |
| **Part2Object + Hi-Mask3D** | **Unsupervised** | **Significant improvement** | **Outperforms supervised baseline** | **Ours** |

### Zero-Shot Cross-Dataset Generalization

| Method | ScanNet200 Head mAP@50 | ScanNet200 Common mAP@50 | S3DIS mAP@50 (Min Gain) |
|------|----------------------|------------------------|----------------------|
| Mask3D (Class-agnostic) | Baseline | Baseline | Baseline |
| **Hi-Mask3D** | **+10.0%** | **+0.8% mAP** | **+2.9%** |

### Key Findings

1. **Hierarchical Clustering vs. Single-Layer Clustering**: Single-layer clustering fails to accommodate objects of varying scales, leading to either over-segmentation of large objects or under-segmentation of small ones. Hierarchical clustering effectively resolves this issue.
2. **Role of Objectness Prior**: Without the objectness prior, objects tend to merge with adjacent items or background elements (such as walls and floors). Introducing the prior successfully separates spatially adjacent objects.
3. **Self-Training Gain**: Through self-training, Hi-Mask3D can correct under-segmentation issues present in the pseudo-labels, such as separating a computer from the desk.
4. **Part-Level Learning**: Hi-Mask3D can learn hierarchical semantic relationships between objects and parts, such as distinguishing the backrest, seat cushion, and armrests of a sofa.

## Highlights & Insights

- The hierarchical "part-to-object" workflow is highly intuitive, aligning naturally with the "part-whole" relation in human perception.
- The design of using the 3D objectness prior as a stopping criterion is elegant, compensating for the lack of boundary awareness in pure feature clustering.
- On the head and common categories of ScanNet200, the unsupervised method outperforms the fully-supervised class-agnostic Mask3D.
- Hi-Mask3D simultaneously outputs both part-level and object-level segmentations, providing flexible granularity options for downstream tasks.

## Limitations & Future Work

- Performance drops slightly (mAP -0.9%) on tail categories of ScanNet200, as the model is never exposed to annotations of these categories.
- The quality of the objectness prior relies heavily on the performance of the 2D detector; if the 2D detector fails in specific scenes, the clustering performance is affected.
- Training for 600 epochs is time-consuming.
- Hyperparameters for hierarchical clustering ($K$, $T$, $\tau_{iou}$) require dataset-specific tuning.

## Related Work & Insights

- **Mask3D**: A fully-supervised 3D instance segmentation baseline; Hi-Mask3D is extended on top of it to form a hierarchical structure.
- **Unscene3D**: Another unsupervised 3D scene understanding method, which utilizes a class-agnostic evaluation protocol.
- **CutLER**: Unsupervised 2D object detection, used to provide the 3D objectness prior.
- **Insights**: The "over-segmentation -> hierarchical merging -> self-training" paradigm can be transferred to other unsupervised segmentation tasks.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ProMerge: Prompt and Merge for Unsupervised Instance Segmentation](promerge_prompt_and_merge_for_unsupervised_instance_segmentation.md)
- [\[ICML 2026\] Unsupervised Hierarchical Skill Discovery](../../ICML2026/segmentation/unsupervised_hierarchical_skill_discovery.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)
- [\[ECCV 2024\] SPIN: Hierarchical Segmentation with Subpart Granularity in Natural Images](spin_hierarchical_segmentation_with_subpart_granularity_in_natural_images.md)
- [\[ECCV 2024\] PartSTAD: 2D-to-3D Part Segmentation Task Adaptation](partstad_2d-to-3d_part_segmentation_task_adaptation.md)

</div>

<!-- RELATED:END -->

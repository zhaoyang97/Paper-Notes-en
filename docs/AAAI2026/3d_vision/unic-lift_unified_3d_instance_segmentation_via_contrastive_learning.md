---
title: >-
  [Paper Note] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning
description: >-
  [AAAI 2026][3D Vision][3D Instance Segmentation] Ours proposes UniC-Lift, a unified single-stage 3D instance segmentation framework. By learning optimizable vector embeddings in 3DGS primitives and training them with contrastive and triplet losses, it directly decodes consistent 3D segmentation labels through a simple "Embedding-to-Label" process. This eliminates post-processing clustering steps like HDBSCAN, reducing training time from over 15 hours to less than 40 minutes.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Instance Segmentation"
  - "3D Gaussian Splatting"
  - "Contrastive Learning"
  - "Multi-view Consistency"
  - "Embedding-to-Label"
date: 2026-05-08
content_hash: 2a0cbe69498aa2b2
---

# UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning

**Conference**: AAAI 2026  
**arXiv**: [2512.24763](https://arxiv.org/abs/2512.24763)  
**Code**: [github.com/val-iisc/UniC-Lift](https://github.com/val-iisc/UniC-Lift)  
**Area**: 3D Vision  
**Keywords**: 3D Instance Segmentation, 3D Gaussian Splatting, Contrastive Learning, Multi-view Consistency, Embedding-to-Label

## TL;DR

Ours proposes UniC-Lift, a unified single-stage 3D instance segmentation framework. By learning optimizable vector embeddings in 3DGS primitives and training them with contrastive and triplet losses, it directly decodes consistent 3D segmentation labels through a simple "Embedding-to-Label" process. This eliminates post-processing clustering steps like HDBSCAN, reducing training time from over 15 hours to less than 40 minutes.

## Background & Motivation

### Problem Definition

3D scene understanding is a key task in fields such as AR/VR, autonomous driving, and path planning. Existing methods usually achieve 3D segmentation by "lifting" 2D segmentation labels to 3D representations (e.g., NeRF, 3DGS). However, 2D segmentation models generate **inconsistent** instance labels across different views—the same object may be assigned different instance IDs in different views.

### Limitations of Prior Work

**Two-stage methods (preprocessing + segmentation)**: Such as Panoptic-Lifting and DM-NeRF, rely on the Linear Assignment Problem to match 2D predictions with 3D representations. The computational overhead is extremely high, requiring over 20 hours to train a single scene.

**Two-stage methods (contrastive learning + clustering)**: Such as Contrastive-Lift, optimize feature embeddings via contrastive learning and then require post-processing with the HDBSCAN clustering algorithm to assign labels. This introduces sensitivity to hyperparameters and takes over 15 hours to train.

**Feature distillation methods**: Such as DFF, distill high-dimensional features like CLIP/DINO into 3D representations, but suffer from extremely slow training speed (approx. 2 days).

### Core Motivation

Can contrastive learning and label decoding be unified into a single-stage process? The authors observe that when the embedding space is constrained to $[0,1]^d$ (via sigmoid), the contrastive loss naturally drives the embeddings of different instances to converge to different corners of the hypercube. Each corner corresponds to a unique binary code, which can be directly decoded into discrete labels. This insight completely eliminates the need for post-processing clustering.

## Method

### Overall Architecture

UniC-Lift is built upon 3DGS. The Mechanism is to append a $d$-dimensional learnable vector embedding $\boldsymbol{v} \in \mathbb{R}^d$ to each 3D Gaussian primitive. By rendering 2D embedding maps via differentiable rendering, optimization is performed using contrastive and triplet losses. Finally, instance labels are directly obtained through thresholding and binary decoding.

Overall pipeline:
- Input: Multi-view RGB images + camera poses + 2D segmentation masks (which can be inconsistent)
- 3DGS optimization: Simultaneously optimize color parameters and vector embeddings
- Loss: Rendering loss + cluster contrastive loss + triplet loss + 3D neighborhood regularization
- Inference: Pass rendered embeddings through sigmoid $\rightarrow$ thresholding $\rightarrow$ binary decoding $\rightarrow$ instance labels

### Key Designs

#### 1. **Rendering of Learnable Vector Embeddings**

Each 3D Gaussian primitive is associated with a $d$-dimensional vector embedding $\boldsymbol{v}$ (view-independent), which is rendered onto a 2D plane using alpha blending in the same manner as color:

$$\boldsymbol{\mathcal{V}} = \sum_{i \in T} \boldsymbol{v}_i \alpha'_i \prod_{j=1}^{i-1}(1-\alpha'_j)$$

*Design Motivation*: Utilizing the differentiable rendering framework of 3DGS allows embedding optimization to naturally incorporate 3D geometric relationships, achieving multi-view consistency.

#### 2. **Cluster Contrastive Loss (Cluster Loss)**

For the rendered embedding map $\mathbb{V} \in \mathbb{R}^{H \times W \times d}$ of each view, pixels are partitioned into $K$ disjoint sets $\{\Omega_1, ..., \Omega_K\}$ based on 2D segmentation masks. The centroid $\boldsymbol{m}_{\Omega_i}$ of each set is calculated, and then the distance from intraclass embeddings to their centroids is minimized while maximizing the distance between different class centroids:

$$\mathcal{L}_{cluster} = \sum_{\Omega_i} \sum_{u \in \Omega_i} \|\mathbb{V}(u) - \boldsymbol{m}_{\Omega_i}\|_2^2 - \sum_{i \neq j} \|\boldsymbol{m}_{\Omega_i} - \boldsymbol{m}_{\Omega_j}\|_2^2$$

*Design Motivation*: Pulling embeddings of the same instance closer while pushing embeddings of different instances apart is a standard contrastive learning objective.

#### 3. **Triplet Loss and Boundary Hard Sampling**

Directly applying cluster loss to rendered embeddings cannot guarantee consistent penalties. Therefore, embeddings are first constrained to the $[0,1]$ range via sigmoid, and then projected through a **linear layer** $\mathbb{W}$ where the triplet loss is calculated in the projected space:

$$\mathcal{L}_{triplet} = \sum_{(a,p,n) \in \Delta} \max(0, \|a-p\|_2^2 - \|a-n\|_2^2 + \delta)$$

***Key Designs***: Positive and negative samples are sampled from **segmentation boundaries** instead of random sampling. The reason is that triplets at the boundaries provide non-zero gradients, which are far more informative than random triplets and can accelerate convergence (achieving the same quality in 25k iterations vs. 50k iterations).

*Role of the Linear Layer*: Direct hard mining on feature embeddings has been shown to be unstable. Applying triplet loss after a linear transformation on the rendered embeddings stabilizes training and significantly boosts performance.

#### 4. **3D Neighborhood Regularization**

For each Gaussian primitive $i$, a spatial neighborhood $\mathcal{N}(i) = \{\|\mu_i - \mu_j\|_2^2 \leq \tau\}$ is constructed to penalize embedding differences between neighboring Gaussians:

$$\mathcal{L}_{3D} = \sum_{i=1}^{|\mathcal{G}|} \sum_{j \in \mathcal{N}(i)} \|\boldsymbol{v}_i - \boldsymbol{v}_j\|_2^2$$

This loss is activated only after 15,000 iterations (once adaptive density control stabilizes), preventing interference with the splitting/cloning process of Gaussian primitives.

#### 5. **Embedding-to-Label Process**

This is the core innovation of Ours. The entire decoding process is extremely succinct:

1. Apply sigmoid to the rendered embeddings: $\hat{\boldsymbol{\mathcal{V}}} = \sigma(\boldsymbol{\mathcal{V}})$
2. Threshold into binary vectors: $\tilde{\boldsymbol{\mathcal{V}}} = \mathbf{1}[\hat{\boldsymbol{\mathcal{V}}} > 0.5]$
3. Decode binary vectors into labels: $l = \sum_k \tilde{\boldsymbol{\mathcal{V}}}_k \cdot 2^{k-1}$

This reduces the inference time complexity for label prediction to $O(n)$ (where $n$ is the number of pixels), whereas Contrastive-Lift requires $O(n \log c)$ (where $c$ is the number of clusters), significantly improving inference speed.

### Loss & Training

Total loss:

$$\mathcal{L}_{total} = \mathcal{L}_{rendering} + \lambda_{cluster} \mathcal{L}_{cluster} + \lambda_{triplet} \mathcal{L}_{triplet} + \lambda_{3D} \mathcal{L}_{3D}$$

Hyperparameter settings: $\lambda_{cluster} = \lambda_{triplet} = \lambda_{3D} = 0.1$, triplet margin $\delta = 1$, neighborhood threshold $\tau = 0.01$. Embedding dimension $d = 12$, maximum number of triplets is 3,000.

Training strategy:
- Uses the ADAM optimizer with a learning rate of $1 \times 10^{-4}$
- Total training of 30k iterations on a single RTX A6000 GPU
- Gradients of the segmentation loss do not participate in the adaptive density control of 3DGS
- Triplet loss and 3D loss are enabled only after adaptive density control is completed

## Key Experimental Results

### Main Results

**ScanNet and Replica3D (PQ^scene metric)**:

| Dataset | DM-NeRF | Panoptic-Lifting | Contrastive-Lift | Gaussian-Grouping | **Ours** |
|--------|---------|-------------------|------------------|-------------------|---------------|
| ScanNet | 41.7 | 58.9 | 62.3 | 61.83 | **63.0** |
| Replica3D | 44.1 | 57.9 | 59.1 | 66.52 | **88.7** |

On Replica3D, UniC-Lift outperforms Gaussian Grouping by 1.3x (88.7 vs 66.52)!

**Messy-Rooms Dataset (PQ^scene metric, different number of objects)**:

| Method | 25obj(Old) | 50obj(Old) | 100obj(Old) | 500obj(Old) | Mean |
|------|------------|------------|-------------|-------------|------|
| Panoptic-Lifting | 73.2 | 69.9 | 64.3 | 51.0 | 63.2 |
| Contrastive-Lift | 78.9 | 75.8 | 69.1 | 55.0 | 69.0 |
| **Ours** | **86.0** | **79.1** | **70.8** | **57.4** | **71.5** |

Achieves top performance in 6 out of 8 scenes.

### Ablation Study

**Effect of each loss function (Replica3D)**:

| Configuration | PQ^scene | mIoU | Description |
|------|----------|------|------|
| CL + 3D Reg | 88.0 | 94.4 | No triplet loss |
| CL only | 83.7 | 91.8 | No triplet loss, no 3D regularization |
| CL + Triplet(MLP) | 89.0 | 95.2 | No 3D regularization |
| CL + Triplet(no MLP) + 3D | 88.0 | 94.0 | No linear projection layer |
| **All (CL+Triplet(MLP)+3D)** | **89.0** | **95.4** | Ours final configuration |

**Training time comparison (Replica single scene, NVIDIA A6000)**:

| Method | Training Time |
|------|----------|
| Panoptic-Lifting | >20 hours |
| Contrastive-Lift | >15 hours |
| **Ours** | **<40 minutes** |

### Key Findings

1. **Embedding-to-Label process eliminates clustering post-processing**: Compared to Contrastive-Lift+3DGS, UniC-Lift completely eliminates the clustering step while maintaining comparable quantitative metrics (42 minutes vs. 85 minutes).
2. **Boundary hard sampling significantly accelerates convergence**: Using boundary triplets achieves the same quality in 25k iterations as random triplets do in 50k iterations.
3. **Low-resolution mask training does not affect results**: Training with 0.5x resolution masks shows no visually obvious difference from full-resolution results.
4. **Training with few masks is possible**: Using only 5% of the segmentation masks yields results close to training with the full set of masks.

## Highlights & Insights

1. **Elegance of the core innovation**: The observation that contrastive learning naturally drives embeddings to the corners of a sigmoid-constrained space is highly ingenious, making label decoding as simple as binary encoding. This is a very natural and elegant mathematical finding.
2. **From $O(n \log c)$ to $O(n)$**: The reduction in inference complexity is of practical significance, especially for large-scale scenes.
3. **20-30x Training Speedup**: Reducing training time from 15-20 hours to 40 minutes makes the method highly practical.
4. **Downstream Applications**: High-quality 3D segmentation directly supports object extraction and scene editing, demonstrating the practical value of the method.

## Limitations & Future Work

1. **Embedding dimension limits the number of instances**: A $d=12$ dimensional embedding theoretically supports up to $2^{12} = 4096$ instances, which may be insufficient for extremely large-scale scenes.
2. **Limited to static scenes**: Not yet extended to dynamic scenes.
3. **Dependency on 2D segmentation quality**: Although multi-view consistency is not strictly required, reasonable 2D segmentations are still needed as inputs.
4. **Artifacts still exist in boundary areas**: Although the hard sampling strategy mitigates boundary issues, the authors note that this problem is not fully resolved.

## Related Work & Insights

- **Contrastive-Lift**: The most direct baseline for optimization/improvement in this work; UniC-Lift merges its two-stage pipeline into a single stage.
- **3DGS series**: Leverages the high-efficiency rendering and scalable attribute storage of 3DGS.
- **Binary encoding concept**: The concept of driving embeddings to corners can be generalized to other scenarios requiring discrete representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The Embedding-to-Label concept is highly elegant and novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Tested on three datasets with comprehensive ablations, but lacks trials on larger-scale scenes.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation; toy experiments intuitively demonstrate the core idea.
- **Value**: ⭐⭐⭐⭐⭐ — The 20-30x speedup makes real-world application feasible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion](../../ECCV2024/3d_vision/pcf-lift_panoptic_lifting_by_probabilistic_contrastive_fusion.md)
- [\[AAAI 2026\] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation](assist-3d_adapted_scene_synthesis_for_class-agnostic_3d_instance_segmentation.md)
- [\[AAAI 2026\] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)
- [\[CVPR 2026\] CompetitorFormer: Mitigating Query Conflicts for 3D Instance Segmentation via Competitive Strategy](../../CVPR2026/3d_vision/competitorformer_mitigating_query_conflicts_for_3d_instance_segmentation_via_com.md)
- [\[ICLR 2026\] Learning Unified Representation of 3D Gaussian Splatting](../../ICLR2026/3d_vision/learning_unified_representation_of_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

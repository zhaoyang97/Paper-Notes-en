---
title: >-
  [Paper Note] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning
description: >-
  [CVPR 2025][3D Vision][Unsupervised Semantic Segmentation] This paper proposes P-SLCR, an unsupervised point cloud semantic segmentation method driven by a prototype library. By separating points into "consistent" and "ambiguous" categories, aligning consistent points with prototypes through consistent structure learning, and constraining two prototype libraries via semantic relation consistent reasoning, it achieves an unsupervised mIoU of 47.1% on S3DIS…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Unsupervised Semantic Segmentation"
  - "Point Cloud"
  - "Prototype Learning"
  - "Consistent Reasoning"
  - "Structure Learning"
date: 2026-05-08
content_hash: 1a8ce6ad431f81a2
---

# P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning

**Conference**: CVPR 2025  
**arXiv**: [2603.06321](https://arxiv.org/abs/2603.06321)  
**Code**: [https://github.com/lixinzhan98/P-SLCR](https://github.com/lixinzhan98/P-SLCR)  
**Area**: 3D Vision  
**Keywords**: Unsupervised Semantic Segmentation, Point Cloud, Prototype Learning, Consistent Reasoning, Structure Learning

## TL;DR
This paper proposes P-SLCR, an unsupervised point cloud semantic segmentation method driven by a prototype library. By separating points into "consistent" and "ambiguous" categories, aligning consistent points with prototypes through consistent structure learning, and constraining two prototype libraries via semantic relation consistent reasoning, it achieves an unsupervised mIoU of 47.1% on S3DIS, outperforming the fully supervised PointNet.

## Background & Motivation
**Background**: Point cloud semantic segmentation mainly relies on large-scale annotated data (e.g., PointNet, PTv2), but annotating 3D data is significantly more time-consuming and labor-intensive than 2D data.

**Limitations of Prior Work**: Unsupervised methods (e.g., GrowSP, U3DS3) generate pseudo-labels through clustering to supervise training. However, pseudo-labels contain substantial noise, and directly using all of them degrades the discriminability of features. Moreover, the prototype features are not representative enough.

**Key Challenge**: Fully untrusted clustering pseudo-labels vs. the necessity to use pseudo-labels to supervise network learning—how to leverage high-quality pseudo-labels while progressively improving low-quality ones?

**Goal**: Design an effective training strategy to extract reliable signals from unreliable pseudo-labels and gradually achieve complete feature space partitioning.

**Key Insight**: Categorize points into "consistent" points (where pseudo-labels, network predictions, and confidence are mutually consistent) and "ambiguous" points, establish separate prototype libraries for each, and use consistent prototypes to guide the learning of ambiguous prototypes.

**Core Idea**: Reliable points $\rightarrow$ consistent prototype learning $\rightarrow$ guide ambiguous prototypes $\rightarrow$ progressively incorporate ambiguous points into the consistent set.

## Method

### Overall Architecture
The input point cloud undergoes feature extraction via SparseConv $\rightarrow$ clustering to obtain pseudo-labels $\rightarrow$ separation into consistent/ambiguous points based on reliability $\rightarrow$ establishment of dual prototype libraries (updated via EMA) $\rightarrow$ consistent structure learning to pull consistent points closer to prototypes $\rightarrow$ semantic relation consistent reasoning to constrain the semantic relations between the two libraries $\rightarrow$ iterative optimization, with ambiguous points progressively transforming into consistent points.

### Key Designs

1. **Separation of Reliable Points**:

    - **Function**: Filter high-confidence "consistent points" from noisy pseudo-labels.
    - **Mechanism**: Triple consistency condition—clustering pseudo-label $\mathbf{l}$ = network prediction $\bar{\mathbf{p}}$ and confidence $\mathbf{p}_c \geq \tau$. A binary mask $R$ is used to separate the consistent point cloud $P^c$ and the ambiguous point cloud $P^a$.
    - **Design Motivation**: Introducing unreliable pseudo-labels into training destroys the feature discriminability between classes. Learning prototypes solely using high-confidence points is more secure.

2. **Consistent Structure Learning**:

    - **Function**: Pull the feature representations of consistent points closer to their corresponding consistent prototypes.
    - **Mechanism**: Structural error matrix $M_k^c = \|\boldsymbol{\mu}_k^c - G(\mathbf{p}_j^c)\|_2$, minimizing $\mathcal{L}_{sl} = \sum_k M_k^c$. The dual prototype libraries are updated from the intra-class cluster centers of each batch via EMA.
    - **Design Motivation**: Learning compact representations within classes renders the prototypes progressively more representative.

3. **Semantic Relation Consistent Reasoning**:

    - **Function**: Constrain the consistent prototype library and the ambiguous prototype library to maintain consistent inter-class semantic relations.
    - **Mechanism**: Compute and normalize similarity matrices between consistent and ambiguous prototypes, $e_{ij}^c = \boldsymbol{\mu}_i^c \cdot (\boldsymbol{\mu}_j^c)^T$ and $e_{ij}^a$, respectively. A KL-divergence constraint is applied: $\mathcal{L}_{cr} = \frac{1}{K^2}\sum_{ij} \bar{e}_{ij}^c \log(\bar{e}_{ij}^c / \bar{e}_{ij}^a)$.
    - **Design Motivation**: Inter-class relations of consistent prototypes are more accurate. Using them to constrain ambiguous prototypes indirectly improves the segmentation quality of ambiguous points.

### Loss & Training
The total loss is $\mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda_1 \mathcal{L}_{sl} + \lambda_2 \mathcal{L}_{cr}$, where $\lambda_1=0.5$ and $\lambda_2=1.0$. The EMA parameter $\alpha=0.99$, and the confidence threshold $\tau=0.9$. Initial clustering utilizes K-means on the 512-dimensional features extracted by SparseConv, with $K=13$ on S3DIS. During training, ambiguous points are progressively converted into consistent points, eventually completing the feature space partitioning. Pseudo-labels are re-clustered and updated every 5 epochs to avoid training instability caused by frequent updates.

## Key Experimental Results

### Main Results (S3DIS Area-5)

| Method | OA (%) | mAcc (%) | mIoU (%) | Type |
|------|--------|----------|----------|------|
| PointNet (Fully Supervised) | 77.5 | 59.1 | 44.6 | Supervised |
| GrowSP | 78.4 | 57.2 | 44.5 | Unsupervised |
| U3DS3 | 75.5 | 55.8 | 42.8 | Unsupervised |
| **P-SLCR** | **80.2** | **61.3** | **47.1** | **Unsupervised** |

Meanwhile, P-SLCR achieves unsupervised state-of-the-art results on SemanticKITTI and ScanNet. On SemanticKITTI, P-SLCR reaches 38.6% mIoU, outperforming GrowSP's 35.9%.

### Ablation Study

| Configuration | mIoU (%) | OA (%) | Description |
|------|----------|--------|------|
| Baseline (Clustering + Direct Training) | 42.3 | 76.1 | No quality filtering |
| + Consistent Structure Learning | 44.8 | 78.5 | Pulls consistent points closer to prototypes |
| + Semantic Consistent Reasoning | 45.9 | 79.1 | Constrains relations between two libraries |
| Full P-SLCR | 47.1 | 80.2 | Synergy of three components |

### Key Findings
- The unsupervised mIoU of 47.1% outperforms the fully supervised PointNet (44.6%), demonstrating that high-quality pseudo-labels paired with structure learning can compensate for the lack of annotations.
- The proportion of consistent points increases progressively during training. This dynamic transition from ambiguous to consistent refines the feature space, with initial ~40% consistent points increasing to ~75%.
- Visualization of consistent structure learning shows that intra-class features become progressively compact, and class boundaries in the t-SNE plot change from blurry to distinct.
- The proposed method also achieves unsupervised state-of-the-art on the outdoor SemanticKITTI dataset, indicating that the approach is not limited to indoor scenes.
- The EMA update strategy for the dual prototype libraries renders the model more robust to intra-batch noise, avoiding the accumulation of single-clustering errors.
- Removing semantic consistent reasoning in the ablation study causes a drop of 1.1% in mIoU, indicating that consistent structure learning alone is insufficient for ambiguous prototypes to learn correct inter-class relations.
- Sensitivity analysis of $\lambda_1$ and $\lambda_2$ shows that model performance is stable within the ranges of $\lambda_1 \in [0.3, 0.7]$ and $\lambda_2 \in [0.5, 1.5]$.

## Highlights & Insights
- The dual-prototype strategy of **"sorting gold from noisy pseudo-labels"** is highly inspiring: instead of trying to correct all pseudo-labels, it first learns from reliable ones only and then gradually expands the reliable set.
- **Using KL-divergence to constrain inter-class relations** is more global than directly constraining feature distances, as it preserves the semantic structure rather than individual prototype positions.
- The dynamic separation of consistent/ambiguous points serves as a natural curriculum learning process from easy to hard.
- The dual prototype library design can be transferred to other pseudo-label training scenarios (such as semi-supervised detection or domain adaptation), where the core concept is "separating reliable/unreliable samples and using the reliable ones to guide the unreliable ones."
- At the end of training, the proportion of consistent points reaches ~75%. The remaining 25% of ambiguous points are still assigned labels via nearest neighbor search from consistent prototypes, yielding complete segmentation.

## Limitations & Future Work
- Requiring a predefined number of classes $K$ makes the method inapplicable to scenarios with an unknown number of categories. Future work could incorporate methods that automatically determine the number of classes (e.g., cluster number estimation in DINO).
- The quality of the initial clustering directly affects subsequent learning; if the error rate of initial pseudo-labels is too high, the model may fail to converge, creating an implicit requirement on initial feature quality.
- Only 3D geometry and color features are utilized, leaving the rich semantic priors of 2D images or language models untapped. Integrating VLMs could further boost performance.
- The choice of the confidence threshold $\tau$ affects the quality of the consistent/ambiguous separation. If it is too high, the initial consistent points are too sparse, causing insufficient learning; if too low, noise is introduced.
- The feature dimension is fixed at 512. The effect of higher dimensions on improving prototype discriminability remains unexplored.

## Related Work & Insights
- **vs GrowSP**: Superpoint growing + clustering, but lacks quality filtering and structure learning. P-SLCR enhances performance by +2.6% mIoU through the dual prototype libraries and consistent reasoning.
- **vs U3DS3**: Spatial clustering + iterative training, but uses all pseudo-labels. P-SLCR's selective usage is more effective.
- **vs PointDC**: Deep contrastive learning, but does not explicitly model prototype structures.
- **Relationship with STRL**: Spatiotemporal contrastive learning is used for point cloud pre-training but still requires annotated fine-tuning for downstream tasks; P-SLCR directly achieves unsupervised segmentation.

## Rating
- Novelty: ⭐⭐⭐⭐ The training strategy of dual prototype libraries + consistent/ambiguous separation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three datasets with ablations and visualizations.
- Writing Quality: ⭐⭐⭐ Rich in formulas, logically clear but somewhat wordy.
- Value: ⭐⭐⭐⭐ Unsupervised performance outperforming fully supervised PointNet is highly compelling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ProtoDepth: Unsupervised Continual Depth Completion with Prototypes](protodepth_unsupervised_continual_depth_completion_with_prototypes.md)
- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[CVPR 2025\] ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)
- [\[CVPR 2026\] PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting](../../CVPR2026/3d_vision/pointgs_semantic-consistent_unsupervised_3d_point_cloud_segmentation_with_3d_gau.md)
- [\[CVPR 2025\] Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation](relation3d_enhancing_relation_modeling_for_point_cloud_instance_segmentation.md)

</div>

<!-- RELATED:END -->

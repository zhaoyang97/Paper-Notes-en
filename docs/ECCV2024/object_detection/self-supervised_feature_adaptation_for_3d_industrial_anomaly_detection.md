---
title: >-
  [Paper Note] Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection
description: >-
  [ECCV 2024][Object Detection][3D anomaly detection] This paper proposes the LSFA (Local-to-global Self-supervised Feature Adaptation) framework. It performs task-oriented adaptation of pretrained features through two self-supervised strategies: Intramodal Feature Compactness (IFC) optimization and Cross-modal Local-to-global Consistency (CLC) alignment. LSFA achieves 97.1% I-AUROC on MVTec-3D AD, outperforming the state-of-the-art (SOTA) by +3.4%.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "3D anomaly detection"
  - "multimodal feature adaptation"
  - "self-supervised learning"
  - "memory bank"
  - "industrial inspection"
date: 2026-05-08
content_hash: 7f88e33f440573a2
---

# Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection

**Conference**: ECCV 2024  
**arXiv**: [2401.03145](https://arxiv.org/abs/2401.03145)  
**Code**: Not released  
**Area**: Human Understanding  
**Keywords**: 3D anomaly detection, multimodal feature adaptation, self-supervised learning, memory bank, industrial inspection

## TL;DR

This paper proposes the LSFA (Local-to-global Self-supervised Feature Adaptation) framework. It performs task-oriented adaptation of pretrained features through two self-supervised strategies: Intramodal Feature Compactness (IFC) optimization and Cross-modal Local-to-global Consistency (CLC) alignment. LSFA achieves 97.1% I-AUROC on MVTec-3D AD, outperforming the state-of-the-art (SOTA) by +3.4%.

## Background & Motivation

Industrial anomaly detection is typically conducted as an unsupervised task, where models are trained using only normal samples. Existing 2D anomaly detection methods have achieved excellent performance, but relying solely on RGB data is insufficient for identifying subtle geometric surface anomalies. Therefore, it is necessary to utilize both RGB images and 3D point clouds for multimodal anomaly detection.

Existing multimodal methods (such as PatchCore+FPFH, M3DM) directly construct feature databases using models pretrained on ImageNet. However, they suffer from two core limitations:

**Domain bias issue**: There is a large gap between pre-trained knowledge and industrial scenarios, which easily leads the model to misclassify anomalous regions as normal (false negatives).

**Texture complexity issue**: For categories with complex textures (such as "cookie"), models struggle to identify subtle anomaly patterns.

The key insight of this paper is: **Pretrained features cannot be used directly and require self-supervised adaptation targeted for the anomaly detection task**, enhancing feature quality from the two dimensions of intramodal compactness and cross-modal consistency.

## Method

### Overall Architecture

LSFA takes an RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and a point cloud $P \in \mathbb{R}^{N \times 3}$ as inputs. A pretrained ViT-B/8 (DINO-pretrained) is used to extract RGB features, and PointMAE is used to extract 3D features. Lightweight Transformer encoder layers are connected after the pretrained feature extractors as adaptors $\Psi_I$ / $\Psi_P$. The parameters of these adaptors are optimized via two self-supervised objectives: IFC and CLC. In the inference stage, only the adapted local features are used to build the PatchCore memory bank to compute anomaly scores.

### Key Designs

1. **Cross-modal Local-to-global Consistency alignment (CLC)**: Addresses the poor integration caused by unaligned features between the two modalities.

    - **Feature Projection**: 3D point features are remapped to 2D patch space via distance-weighted interpolation, so that the two modalities share the same number of patches $N_m$, naturally establishing local correspondences.
    - **Local Alignment (LA)**: Computes a contrastive loss between the RGB and point cloud patch features at the same location, maximizing the cross-modal feature similarity at the same location and minimizing the similarity at different locations:
    $\mathcal{L}_{LA} = -\log \frac{\exp(\langle F_{I_i}^{\prime j}, F_{P_i}^{\prime j}\rangle)}{\sum_{t,k} \exp(\langle F_{I_i}^{\prime t}, F_{P_i}^{\prime k}\rangle)}$
    - **Global Alignment (GA)**: Applies k-means clustering to local features to obtain instance-level global features $G_{I_i}$/$G_{P_i}$, performing contrastive alignment across samples within a batch:
    $\mathcal{L}_{GA} = -\log \frac{\exp(\langle G_{I_i}^{\prime}, G_{P_i}^{\prime}\rangle)}{\sum_{t,x} \exp(\langle G_{I_t}^{\prime}, G_{P_x}^{\prime}\rangle)}$
    - **Design Motivation**: Local alignment ensures fine-grained spatial consistency, while global alignment ensures the interaction of object-level structural information, rendering the two complementary.

2. **Intramodal Feature Compactness optimization (IFC)**: Solves the issue where normal and anomalous feature distributions are difficult to distinguish in pretrained features.

    - **Local Compactness (LC)**: Maintains a dynamically updated patch-level memory bank $M_I^L$ for each modality, minimizing the distance between the current patch feature and its nearest neighbor in the memory bank:
    $\mathcal{L}_{LC} = \sum_{i=1}^{N_b} \sum_{j=1}^{N_m} \min_{Q \in M_I^L} \|F_{I_i}^j - Q\|_2$
    - **Global Compactness (GC)**: Maintains a prototype-level memory bank $M_I^G$ and optimizes the compactness of global features in a similar manner.
    - The memory bank dynamically updates using a FIFO queue mechanism, enqueuing current batch features and dequeuing the oldest features at each iteration.
    - **Design Motivation**: By narrowing the distribution range of normal features, anomalous features can be more easily detected as outliers.

3. **Inference pipeline**: After adaptation, only local features are used (global features are discarded). PatchCore is used to calculate anomaly scores for RGB and 3D modalities separately, and the final anomaly score is obtained by averaging the scores of both modalities.

### Loss & Training

The total loss is the weighted sum of two components:

$$\mathcal{L}_{LSFA} = \mathcal{L}_{IFC} + \lambda \mathcal{L}_{CLC}$$

where $\mathcal{L}_{IFC} = \mathcal{L}_{LC} + \mathcal{L}_{GC}$ and $\mathcal{L}_{CLC} = \mathcal{L}_{LA} + \mathcal{L}_{GA}$.

- Optimizer: AdamW, learning rate $2 \times 10^{-3}$, cosine warm-up.
- Batch size: 8.
- RGB feature extractor: ViT-B/8 (DINO-pretrained), 768-dimensional outputs pooled to 56×56.
- 3D feature extractor: Point Transformer (ShapeNet-pretrained), outputs from layers 3/7/11 concatenated.

## Key Experimental Results

### Main Results

**MVTec-3D AD I-AUROC (RGB+3D Multimodal)**

| Method | Bagel | Cookie | Dowel | Tire | Mean |
|------|-------|--------|-------|------|------|
| PatchCore+FPFH | 0.918 | 0.883 | 0.932 | 0.886 | 0.865 |
| AST | 0.983 | 0.971 | 0.932 | 0.797 | 0.937 |
| M3DM | 0.994 | 0.972 | 0.942 | 0.850 | 0.945 |
| **LSFA** | **1.000** | **0.989** | **0.961** | **0.951** | **0.971** |

**MVTec-3D AD AUPRO (RGB+3D Multimodal)**

| Method | Bagel | Cookie | Mean |
|------|-------|--------|------|
| PatchCore+FPFH | 0.976 | 0.973 | 0.959 |
| M3DM | 0.970 | 0.950 | 0.964 |
| **LSFA** | **0.986** | **0.946** | **0.968** |

Under 3D modality only, LSFA's Mean I-AUROC reaches 0.921, outperforming M3DM's 0.874 by +4.7%.

### Ablation Study

**Ablation of IFC and CLC Components**

| IFC | CLC | I-AUROC | AUPRO | P-AUROC |
|-----|-----|---------|-------|---------|
| ✗ | ✗ | 0.929 | 0.953 | 0.987 |
| ✗ | ✓ | 0.957 | 0.963 | 0.990 |
| ✓ | ✗ | 0.959 | 0.964 | 0.992 |
| ✓ | ✓ | **0.971** | **0.968** | **0.993** |

**Ablation of Internal Losses of CLC**

| $\mathcal{L}_{GA}$ | $\mathcal{L}_{LA}$ | I-AUROC | AUPRO |
|---|---|---------|-------|
| ✗ | ✗ | 0.929 | 0.953 |
| ✗ | ✓ | 0.949 | 0.961 |
| ✓ | ✗ | 0.952 | 0.961 |
| ✓ | ✓ | **0.959** | **0.964** |

### Key Findings

- IFC and CLC contribute approximately +2.8%/+3.0% I-AUROC respectively, and their combination further boosts the performance to +4.2%.
- Both local and global levels of alignment/compactness are necessary; using either alone is less effective than the combination.
- Best results are also achieved on the Eyecandies dataset (RGB I-AUROC 87.5%, AUPRO 97.8%).
- The improvement is particularly significant for challenging categories such as cable gland and tire.

## Highlights & Insights

1. **Well-founded problem analysis**: Clearly points out two deficiencies of directly utilizing pretrained features (overestimating normalcy + missed detection under complex textures) and proposes targeted solutions.
2. **Self-supervised adaptation approach**: Requires no anomaly annotations and fully leverages intramodal/cross-modal self-supervised signals among normal samples for feature fine-tuning.
3. **Local-to-global design philosophy**: Both cross-modal alignment and intramodal compactness are optimized from local and global levels of granularity, ensuring that multi-scale information is fully utilized.
4. **Zero inference overhead**: Compared to M3DM, LSFA introduces no additional modules in the inference phase and merely replaces the adaptor weights after the feature extractor.

## Limitations & Future Work

1. The adaptor structure only utilizes a single-layer Transformer encoder, which may lack expressiveness in more complex scenarios.
2. The size of the memory bank needs to be manually tuned, and the optimal values may vary across different categories.
3. Evaluation is only performed on two relatively small-scale 3D anomaly detection datasets; the generalizability to larger-scale scenarios remains unknown.
4. Simple averaging of anomaly scores from two modalities might not be the optimal fusion strategy.

## Related Work & Insights

- **PatchCore**: A memory-bank-based feature embedding method, serving as the foundation for the inference pipeline in this paper.
- **M3DM**: A preceding work on multimodal 3D anomaly detection; this paper builds upon it by adding intramodal optimization and multi-granularity alignment.
- **CFA**: Proposes a coupled-hypersphere fine-tuning framework for feature adaptation, shares a similar idea with this work, but was only applied to 2D.
- **Insights**: The concept of feature adaptation can be generalized to other downstream tasks using pretrained features, and the local-to-global self-supervised alignment strategy is of general applicability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing self-supervised feature adaptation to 3D anomaly detection, and the local-to-global design of IFC+CLC is reasonable and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, multiple modalities, and multiple metrics, with meticulous ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-elaborated motivation, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Reaching a new SOTA in 3D industrial anomaly detection, with a simple and practical approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection](../../AAAI2026/object_detection/casl_curvature-augmented_self-supervised_learning_for_3d_anomaly_detection.md)
- [\[ECCV 2024\] WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [\[CVPR 2025\] AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios](../../CVPR2025/object_detection/anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)
- [\[ECCV 2024\] DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning](dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)

</div>

<!-- RELATED:END -->

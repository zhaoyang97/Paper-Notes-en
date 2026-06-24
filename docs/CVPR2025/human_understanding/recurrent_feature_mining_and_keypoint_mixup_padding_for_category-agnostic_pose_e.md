---
title: >-
  [Paper Note] Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation
description: >-
  [CVPR 2025][Human Understanding][Category-Agnostic Pose Estimation] This paper proposes the FMMP framework, which substantially outperforms state-of-the-art methods (+3.2% PCK@0.05) in category-agnostic pose estimation (CAPE) via recurrent mining of fine-grained structure-aware (FGSA) features based on deformable attention, combined with a keypoint mixup padding strategy.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Category-Agnostic Pose Estimation"
  - "Deformable Attention"
  - "Keypoint Mixup Padding"
  - "Few-Shot Learning"
  - "Structure-Aware Features"
date: 2026-05-08
content_hash: 6da5466cb7e82f1f
---

# Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.21140](https://arxiv.org/abs/2503.21140)  
**Code**: [https://github.com/chenbys/FMMP](https://github.com/chenbys/FMMP)  
**Area**: Human/Object Pose Estimation  
**Keywords**: Category-Agnostic Pose Estimation, Deformable Attention, Keypoint Mixup Padding, Few-Shot Learning, Structure-Aware Features

## TL;DR

This paper proposes the FMMP framework, which substantially outperforms state-of-the-art methods (+3.2% PCK@0.05) in category-agnostic pose estimation (CAPE) via recurrent mining of fine-grained structure-aware (FGSA) features based on deformable attention, combined with a keypoint mixup padding strategy.

## Background & Motivation

**Background**: Category-agnostic pose estimation (CAPE) aims to locate keypoints on query images of arbitrary unseen categories given only a few annotated support images. Existing approaches like POMNet and CapeFormer usually perform support feature pooling from heatmaps and interact support-query features through cross-attention.

**Limitations of Prior Work**: Current methods perform feature pooling or cross-attention at a single-layer feature level, yielding coarse feature granularity that fails to fully mine fine-grained and structure-aware (FGSA) features. Since keypoint localization is inherently a pixel-level precision task, finer features are crucial. In addition, different categories have different numbers of keypoint annotations; existing methods align these numbers using "zero padding", which carries empty semantics and provides highly limited supervision signals.

**Key Challenge**: Pixel-level keypoint localization demands fine-grained and structure-aware features, but existing pooling/attention operations on single-scale feature maps are too coarse. Meanwhile, the zero-padding strategy wastes many learning opportunities for locations.

**Goal**: (1) Design a module capable of simultaneously mining FGSA features from both support and query images; (2) Design a better keypoint padding strategy to provide richer supervision.

**Key Insight**: The authors observe that the deformable attention mechanism natively supports multi-scale feature extraction and flexibly aggregates information centered on reference points. Furthermore, using keypoint links (relationships) to control the reference point locations of attention heads can naturally perceive pose structures.

**Core Idea**: Replace heatmap pooling/cross-attention with a deformable-attention-based FGSA module, repeatedly mining fine-grained and structure-aware features from support and query images via recurrent layers. Meanwhile, use a Mixup strategy to interpolate new keypoints between linked keypoint pairs instead of zero padding.

## Method

### Overall Architecture

The input consists of a query image and a support image (with annotated keypoints and category-specific links), and the output corresponds to the target keypoint coordinates on the query image. The framework consists of $L$ recurrent layers, where each layer executes three steps: (1) align the number of keypoints to a unified $K$ using Mixup padding; (2) mine FGSA features from the multi-scale feature pyramid of the support image using $f_{miner-s}$ to obtain support features; (3) mine FGSA features from the query image using $f_{miner-q}$ to obtain keypoint features and predict keypoint coordinates. By recurrently updating the support features, keypoint features, and estimated keypoints layer-by-layer, precise localization results are obtained.

### Key Designs

1. **FGSA Feature Mining Module (Based on Deformable Attention)**:

    - **Function**: Mine fine-grained and structure-aware features from multi-scale feature pyramids with keypoints serving as reference points.
    - **Mechanism**: Two modifications are introduced based on deformable attention. First, keypoints are utilized as reference points to perform deformable sampling on multi-scale feature maps (feature pyramids), obtaining fine-grained features. Second, utilizing keypoint links (searched via BFS in the pose graph), the reference point offsets of different attention heads are shifted to the linked keypoint positions, enabling the features of each keypoint to aggregate information from its neighboring keypoints in the pose structure. Formally, for the $k$-th keypoint, the reference points of $M$ attention heads $\mathcal{P}_k \in \mathbb{R}^{M \times 2}$ are derived from the link graph via BFS, where each head samples features using deformable attention with a different linked keypoint as the reference point.
    - **Design Motivation**: Single-scale heatmap pooling has a bottleneck in accuracy, whereas multi-scale deformable attention is naturally suited for pixel-level localization tasks. Utilizing links to assign reference points to different heads naturally captures structural information, which is more direct and efficient than feature-level post-processing.

2. **Recurrent Feature Mining Framework (Recurrent Pipeline)**:

    - **Function**: Alternately extract and refine features and keypoint estimations from support and query images.
    - **Mechanism**: The $l$-th layer first uses $f_{miner-s}$ (with the previous layer's keypoint features $F_q^{l-1}$ as the query and the support image's feature pyramid as the key/value) to extract support features $F_s^l$. Then, it uses $f_{miner-q}$ (with $F_s^l$ as the query and the query image's feature pyramid as the key/value) to obtain keypoint features $F_q^l$. Finally, an MLP + sigmoid is applied to predict incremental keypoint coordinates: $P_q^l = \sigma(\sigma^{-1}(P_q^{l-1}) + f_{mlp}(F_q^l))$. Consequently, the support features are no longer static, but are dynamically updated as the understanding of the query image deepens, establishing a bidirectional interaction of "query-guided support feature extraction".
    - **Design Motivation**: In conventional methods, support feature extraction is executed in a single-pass manner and cannot be adjusted based on the query image context. The recurrent architecture allows support features to dynamically adapt to the current query, thereby providing more targeted support information.

3. **Keypoint Mixup Padding Strategy (Keypoint Mixup Padding)**:

    - **Function**: Align the differing keypoint counts of various categories to a unified $K$, while providing richer supervision signals.
    - **Mechanism**: Randomly sample $K-K_c$ pairs from all linked keypoint pairs. For each pair of keypoints $P^*[i]$ and $P^*[j]$, convex combinations are computed with $\lambda \sim \text{Beta}(\alpha, \alpha)$ as $P[k] = \lambda \cdot P^*[i] + (1-\lambda) \cdot P^*[j]$ to generate new padded keypoints. Concurrently, the link matrix is updated to connect all newly generated keypoints sequentially along the original link. During training, the same $\lambda$ is used for both support and ground-truth query keypoints to maintain consistency; during inference, uniform division points are used instead of random mixup.
    - **Design Motivation**: Zero padding only provides non-semantic empty spaces. In contrast, Mixup padding places keypoints along the physical structure of the object, enabling the model to learn denser pose semantics. Moreover, the new keypoints participate in the loss computation ($\mathcal{L}_{mixup}$), introducing additional valid supervision.

### Loss & Training

The overall loss is defined as $\mathcal{L}_{full} = \mathcal{L}_{raw} + \beta \cdot \mathcal{L}_{mixup}$. Here, $\mathcal{L}_{raw}$ is the L1 loss of the original $K_c$ keypoints (averaged across all $L$ layers), and $\mathcal{L}_{mixup}$ represents the L1 loss for the padded keypoints. The hyperparameter $\beta$ is set to $0.5$. A ResNet-50 backbone is used to extract the multi-scale feature pyramids, with channels compressed to 256 dimensions. The default settings are $K=70$, $M=8$, $L=3$, and $\alpha=1$. The model is optimized using Adam with a learning rate of $1e^{-5}$ for 200 epochs at a batch size of 16.

## Key Experimental Results

### Main Results

| Method | 1-shot mPCK (AVG) | 5-shot mPCK (AVG) |
|------|-------------------|-------------------|
| POMNet | 64.53 | 68.28 |
| CapeFormer | 70.58 | 75.45 |
| MetaPoint | 72.23 | 76.90 |
| SCAPE | 72.36 | 77.18 |
| **FMMP (Ours)** | **73.42** | **78.02** |

### Ablation Study

| Configuration | Split1 mPCK |
|------|-------------|
| Base (w/o FGSA, w/o mixup) | 69.82 |
| + Recurrent Support Feature Mining | 73.18 (+3.36) |
| + Structure-Aware Reference Points | 76.23 (+3.05) |
| + Mixup Padding | 77.41 (+1.18) |
| + Mixup Loss | **78.72** (+1.31) |

### Key Findings

- Recurrently mining support features ($f_{miner-s}$) brings the largest gain (+3.36%), indicating that dynamic feature extraction from the support image is highly critical.
- The structure-aware reference point design contributes a +3.05% improvement, verifying the effectiveness of utilizing link relationships to guide the attention heads.
- Under the stringent PCK@0.05 threshold, the performance gap is even wider (+3.2% vs SCAPE), indicating that the proposed method primarily enhances precise localization ability.
- Replacing the link matrix with FC (fully connected) or self-connections only leads to a performance drop, validating the importance of structural information.
- Utilizing uniform padding during inference outperforms both mixup padding and zero padding.

## Highlights & Insights

- This work unifies the feature extraction processes from both support and query images in CAPE using deformable attention, resulting in a simple and elegant design.
- The recurrent architecture enables bidirectional information flow between support and query features, avoiding the separation between support feature extraction and query feature interaction found in traditional methods.
- The Mixup padding strategy cleverly exploits existing link relationships to generate meaningful new keypoints, presenting an intuitive and effective concept.
- The entire approach relies on variations of a single generic module $f_{miner}$, keeping the architecture clean and highly reproducible.

## Limitations & Future Work

- Method performance depends on correct keypoint link information, which restricts its applicability to scenarios where links are unknown.
- Setting a unified keypoint count of $K=70$ may introduce redundancy for categories that have fewer keypoints.
- More general graph relationships (e.g., hierarchical structures, symmetrical structures) have not been explored; BFS only utilizes topological distances.
- Future work could extend the method to 3D pose estimation or unsupervised keypoint discovery tasks.

## Related Work & Insights

- The comparison with MetaPoint demonstrates that using deformable attention solely on the query side is insufficient; fine-grained feature mining is also required on the support side.
- From a data augmentation perspective, Mixup provides a novel perspective on keypoint padding, potentially inspiring other tasks that require aligning sets of different sizes.
- The advantages of deformable attention in pixel-level localization tasks are worth referencing for other fine-grained tasks.

## Rating

| Area | Rating (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Overall Evaluation | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SCAPE: A Simple and Strong Category-Agnostic Pose Estimator](../../ECCV2024/human_understanding/scape_a_simple_and_strong_category-agnostic_pose_estimator.md)
- [\[ICLR 2026\] EdgeCAPE: Edge Weight Prediction for Category-Agnostic Pose Estimation](../../ICLR2026/human_understanding/edgecape_edge_weight_prediction_for_category-agnostic_pose_estimation.md)
- [\[ICLR 2026\] GenCape: Structure-Inductive Generative Modeling for Category-Agnostic Pose Estimation](../../ICLR2026/human_understanding/gencape_structure-inductive_generative_modeling_for_category-agnostic_pose_estim.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)

</div>

<!-- RELATED:END -->

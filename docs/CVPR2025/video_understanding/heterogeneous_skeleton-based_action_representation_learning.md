---
title: >-
  [Paper Note] Heterogeneous Skeleton-Based Action Representation Learning
description: >-
  [CVPR 2025][Video Understanding][Skeleton-based action recognition] This work is the first to investigate the heterogeneity of human skeleton data (varying joint numbers and coordinate dimensions). It proposes three core components: a 3D pose estimation module to unify dimensions, skeleton-specific prompts to unify topologies, and semantic motion encoding to introduce semantic information. Combined with a self-supervised unified representation learning framework…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Skeleton-based action recognition"
  - "heterogeneous data"
  - "unified representation learning"
  - "semantic motion encoding"
  - "self-supervised learning"
date: 2026-05-08
content_hash: 28fd13fa7c048268
---

# Heterogeneous Skeleton-Based Action Representation Learning

**Conference**: CVPR 2025  
**arXiv**: [2506.03481](https://arxiv.org/abs/2506.03481)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Skeleton-based action recognition, heterogeneous data, unified representation learning, semantic motion encoding, self-supervised learning

## TL;DR

This work is the first to investigate the heterogeneity of human skeleton data (varying joint numbers and coordinate dimensions). It proposes three core components: a 3D pose estimation module to unify dimensions, skeleton-specific prompts to unify topologies, and semantic motion encoding to introduce semantic information. Combined with a self-supervised unified representation learning framework, this approach achieves significant improvements on NTU-60/120 and PKU-MMD II.

## Background & Motivation

Skeleton data originating from different sensors and algorithms naturally exhibit **heterogeneity**:

- **Kinect V2 Depth Sensor**: 3D coordinates $\times$ 25 joints (containing rich hand joints)
- **RGB Video Estimation**: 2D coordinates $\times$ 17 joints (containing more facial joints)

Existing skeleton action recognition methods (e.g., GCNs, Transformers) assume homogeneous skeleton data and train models independently for each skeleton type. This leads to two major issues: (1) they cannot utilize complementary information between heterogeneous skeletons (e.g., hand joints in 25-joint vs. face joints in 17-joint data); (2) the models lack transferability across datasets.

The core motivation of this paper is: **Can a unified model be designed to simultaneously handle heterogeneous skeletons with different dimensions and topologies, utilizing their complementarity to enhance action recognition performance?**

## Method

### Overall Architecture

The framework consists of two main modules:
1. **Heterogeneous Skeleton Processing**: Converts skeletons of different dimensions and topologies into a unified format.
2. **Unified Representation Learning**: Uses a shared Transformer backbone to learn unified action representations from multiple heterogeneous skeletons.

### Key Designs

1. **3D Pose Estimation Module**:
    - Function: Lifts 2D 17-joint skeletons to 3D coordinates, unifying the coordinate dimension.
    - Mechanism: First, three spine joints are added to the 17-joint skeleton via linear interpolation: $p_{spine} = (p_{left\_shoulder} + p_{right\_shoulder})/2$, $p_{base} = (p_{left\_hip} + p_{right\_hip})/2$, and $p_{middle} = (p_{spine} + p_{base})/2$. Then, a 4-layer MLP is employed to regress 3D coordinates from the 2D coordinates. The training loss is defined as $\mathcal{L}_{rec} = \frac{1}{|\mathcal{B}|} \sum \|u_i^C - u_i^J\|_2^2$ (calculated over shared joints).
    - Design Motivation: 3D skeletons contain richer motion information (depth dimension); unifying them into a 3D space allows effective fusion with 25-joint skeletons. Interpolated spine joints increase corresponding points between 2D and 3D, aiding the regression training.

2. **Skeleton-specific Prompt + Unified Skeleton**:
    - Function: Unifies skeletons with different joint counts into a standardized format of 30 joints.
    - Mechanism: The union of 25-joint and 17-joint skeletons yields 30 unique joints. For each skeleton type, trainable prompts are used to fill in missing joints: the 25-joint skeleton lacks 5 facial joints, so $prompt_J \in \mathbb{R}^{5 \times 3}$ is added; the 17-joint skeleton lacks 10 hand/foot joints, so $prompt_C \in \mathbb{R}^{10 \times 3}$ is added. This yields a unified skeleton $u \in \mathbb{R}^{m \times t \times 30 \times 3}$, with joints ordered as "face (1-5) $\rightarrow$ common joints (6-20) $\rightarrow$ hand/foot joints (21-30)".
    - Design Motivation: Zero padding is a simple solution for missing joints but provides no meaningful information. Trainable prompts allow the model to learn reasonable "virtual joint" positions, unifying the topology while reserving skeleton specificity.

3. **Semantic Motion Encoding**:
    - Function: Introduces action semantic information, overcoming the limitation where coordinate joints only represent physical motion and lack semantics.
    - Mechanism: The motion direction of each joint at each timestep is discretized into 7 direction words (right/left/up/down/front/back/unmove). These are encoded into high-dimensional features $e \in \mathbb{R}^{7 \times l}$ using a pre-trained language model (CLIP ViT-B/32 text encoder), and then mapped to a 1D embedding via a dimensionality reduction module. This builds a semantic motion encoding of the same size as the skeleton: $m_{t,j}^x = \tilde{e}_{left}$ if $s_{t,j}^x - s_{t-1,j}^x < 0$ (and similarly for other directions).
    - Design Motivation: Although skeletons with different topologies differ physically, they are semantically consistent (both representing a human body). Coordinating motion direction semantics via a language model provides a "common semantic anchor" for heterogeneous skeletons.

### Loss & Training

The total loss is defined as $\mathcal{L} = \lambda \mathcal{L}_{con} + \mathcal{L}_{reg} + \mathcal{L}_{rec}$:

- **Feature Consistency Loss $\mathcal{L}_{con}$**: Enforces MSE alignment between fused features $Z'_i$ and single-modality features $Z_i$, as well as between different modality features, within a skeleton-specific projection space.
- **VC Regularization $\mathcal{L}_{reg}$**: Variance-covariance regularization of VICReg to prevent representation collapse.
- **3D Pose Reconstruction Loss $\mathcal{L}_{rec}$**: $L_2$ loss for regressing 3D coordinates on common joints.

The evaluation paradigm follows self-supervised pre-training followed by linear probing (frozen encoder + trained linear classifier). The backbone is a dual-head Transformer (one spatial and one temporal, hidden size = 1024), trained on 2 RTX 4090 GPUs.

## Key Experimental Results

### Main Results

**Skeleton Action Recognition (Linear Probing)**:

| Method | Modality | FLOPs (G) | NTU-60 x-sub | NTU-60 x-view | NTU-120 x-sub | NTU-120 x-set | PKU-MMD |
|------|------|----------|-------------|--------------|--------------|--------------|---------|
| 3s-UmURL | J+M+B | 5.22 | 84.4 | 91.4 | 75.9 | 77.2 | 54.3 |
| USDRL | J+M+B | - | 87.1 | 93.2 | 79.3 | 80.6 | 59.7 |
| **Ours** | **J+C+S** | **2.54** | **87.8** | **93.7** | **78.9** | **82.2** | **58.2** |

**Skeleton Action Retrieval (NTU-60)**:

| Method | Modality | x-sub | x-view |
|------|------|-------|--------|
| UmURL | J+M+B | 72.0 | 88.9 |
| **Ours (J+C+S)** | J+C+S | **72.7** | **90.9** |

### Ablation Study

| Configuration | PKU-MMD II | Description |
|------|-----------|------|
| w/o 3D pose estimation | 55.8 | Only 2D skeleton used |
| w/o semantic motion | 57.9 | Replaced semantic encoding with numerical values (1/-1/0) |
| w/o skeleton-specific prompt | 57.2 | Replaced trainable prompts with zero padding |
| **Full Model** | **58.2** | All three modules contribute |

### Key Findings

- The complementarity of heterogeneous skeletons is significant: J+C+S (87.8%) substantially outperforms single skeletons J (80.2%) or C (84.4%).
- When used individually, the 17-joint skeleton (C, 84.4%) outperforms the 25-joint skeleton (J, 80.2%), likely because facial joints provide additional descriptive information.
- While semantic motion encoding (S) on its own only yields 70.1%, it serves as a highly effective auxiliary modality when combined with skeletons.
- In transfer learning experiments, transferring from NTU-60 to PKU-MMD II reaches 64.3%, significantly higher than UmURL's 59.7%.
- Through transfer learning on the FineGYM 2D skeleton dataset, the model achieves 75.3%, outperforming various RGB-based methods.

## Highlights & Insights

- **Novel Problem Definition**: This work is the first to define and address the "heterogeneity" of skeleton data, a common but overlooked issue in practical scenarios.
- **Topological Unification via Prompts**: Drawing on the concept of prompt learning in NLP to handle missing joints, which is more elegant than simple zero padding.
- **Semantic Motion Encoding**: Textualizes motion directions and then encodes them via a language model, successfully establishing a bridge between physical motion and semantics.
- **Self-Supervised + Heterogeneous**: The self-supervised paradigm naturally fits heterogeneous data (no label alignment required), and the learned representations demonstrate robust transferability.
- Skeletons of different types show preferences for certain actions (e.g., 25-joint for hand actions, 17-joint for facial actions); the unified model effectively combines these strengths.

## Limitations & Future Work

- Currently, the model only supports skeleton data for up to two people, limiting its applicability in multi-person scenarios.
- Only two typical skeletons (25-joint and 17-joint) have been validated, leaving other heterogeneous formats (e.g., SMPL, hand skeletons) unexplored.
- The 3D pose estimation module is implemented with a simple MLP, whose accuracy might not match specialized methods (e.g., MotionBERT).
- The semantic motion encoding uses only seven direction words, resulting in relatively coarse-grained motion semantics.

## Related Work & Insights

- **Relationship with UmURL**: UmURL relies on J+M+B (joint, motion, bone) trimodal representations, whereas this work adopts J+C+S (25-joint, 17-joint, semantics), effectively introducing actual heterogeneous data.
- **Relationship with InfoGCN, etc.**: These methods focus on enhancing representations of a single skeleton type, whereas this work focuses on leveraging complementary characteristics across different skeleton types.
- **Insights**: (1) The paradigm of unifying heterogeneous data can be extended to other modalities (e.g., depth maps with different resolutions, point clouds from different sensors). (2) Encoding "motion vocabularies" with language models deserves further exploration using richer descriptions.

## Rating

- Novelty: ⭐⭐⭐⭐ Both the problem definition and solutions for heterogeneous skeletons are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensively covers recognition, retrieval, semi-supervised, and transfer tasks, accompanied by comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and visually intuitive, though some mathematical formulas could be simplified.
- Value: ⭐⭐⭐⭐ Provides the first systematic framework for processing heterogeneous skeletons, offering high practical value for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](../../ICCV2025/video_understanding/adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)

</div>

<!-- RELATED:END -->

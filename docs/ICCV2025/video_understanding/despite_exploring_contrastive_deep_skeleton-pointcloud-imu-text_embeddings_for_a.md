---
title: >-
  [Paper Note] DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition
description: >-
  [ICCV 2025][Video Understanding][Multimodal contrastive learning] DeSPITE proposes a privacy-preserving multimodal contrastive pre-training framework that aligns four modalities — LiDAR point clouds, skeleton poses…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Multimodal contrastive learning"
  - "LiDAR point cloud"
  - "IMU"
  - "skeleton pose"
  - "joint embedding space"
date: 2026-05-08
content_hash: 97a0e9282f53347d
---

# DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition

**Conference**: ICCV 2025  
**arXiv**: [2506.13897](https://arxiv.org/abs/2506.13897)  
**Code**: Coming soon  
**Area**: Video Understanding  
**Keywords**: Multimodal contrastive learning, LiDAR point cloud, IMU, skeleton pose, joint embedding space

## TL;DR

DeSPITE proposes a privacy-preserving multimodal contrastive pre-training framework that aligns four modalities — LiDAR point clouds, skeleton poses, IMU signals, and text — into a unified embedding space, enabling cross-modal matching, retrieval, and a pre-training paradigm for human activity recognition.

## Background & Motivation

Existing multimodal contrastive learning methods (e.g., ImageBind, IMU2CLIP, MotionCLIP) have achieved notable success in human activity understanding, yet they universally rely on RGB video as the primary visual modality. However, deploying RGB cameras in privacy-sensitive domains such as healthcare and surveillance raises serious ethical and legal concerns.

LiDAR, as an inherently privacy-preserving sensor, has demonstrated strong capabilities in human activity recognition (HAR) and human pose estimation (HPE), yet its role within multimodal contrastive learning spaces remains unexplored. Specifically:

**RGB dependency**: All existing methods anchor other modalities (e.g., IMU, skeleton) to RGB video/images, severely limiting applicability in privacy-sensitive scenarios.

**Cross-modal matching gap**: Correspondences between LiDAR point clouds ↔ IMU and LiDAR point clouds ↔ skeletons have never been investigated.

**Insufficient pre-training**: Due to limited data, no general-purpose pre-trained model for LiDAR-based HAR exists; existing pre-training relies mainly on self-supervision within small individual datasets.

The core research question of this paper is: **What happens if RGB is entirely abandoned and LiDAR is adopted as the primary visual modality for multimodal contrastive learning?**

## Method

### Overall Architecture

DeSPITE learns a joint embedding space that aligns point cloud sequences $X_{pc}$, IMU sequences $X_{imu}$, skeleton pose sequences $X_{pose}$, and text descriptions $X_{text}$. Each modality is mapped to an $\mathbb{R}^e$-dimensional embedding vector by an independent encoder:

- **Point cloud encoder** $f_{pc}$: PST-Transformer + SimCLR projection head
- **IMU encoder** $f_{imu}$: 2-layer LSTM
- **Skeleton encoder** $f_{pose}$: ACTOR encoder
- **Text encoder**: Frozen CLIP text encoder

### Key Designs

1. **Two-level contrastive loss design**: Rather than simply aligning all modalities to the text space, DeSPITE employs a two-level loss strategy:

    - **Text alignment loss** $\mathcal{L}_{text}$: Aligns each sensor modality with CLIP text embeddings on the subset with text annotations (using a boolean mask $tm$ to handle unannotated samples).
    - **Inter-sensor alignment loss** $\mathcal{L}_M$: Directly aligns all pairwise combinations of the three sensor modalities: $(pc, imu)$, $(pc, pose)$, $(imu, pose)$.
    - **Design Motivation**: The primary objective is not text alignment per se (for which existing methods suffice), but rather to leverage the natural correspondences among sensor modalities to enable cross-modal applications of LiDAR point clouds.

2. **Flexible modality combination training**: All possible modality subsets (DeSPIE, DeSPE, DePIE, etc.) are systematically trained by modifying the modality set $M$ and pairing set $M^*$, enabling analysis of each modality's contribution to the joint embedding space.

3. **LIPD-Babel dataset construction**: A temporally aligned dataset is constructed by matching the LIPD dataset with Babel text annotations (downsampling Babel from 30 FPS to 10 FPS), yielding the first large-scale four-modality dataset comprising point clouds, IMU, skeletons, and text. It is divided into v1 (for matching/retrieval evaluation) and v2 (for HAR evaluation).

### Loss & Training

The total loss function is:

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{text} + \beta \mathcal{L}_M$$

where $\alpha = \beta = 0.5$. The contrastive loss in each direction is based on InfoNCE:

$$\mathcal{L}_{a \to b}^i = -\log \frac{\exp(\text{sim}(z_a^i, z_b^i) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(z_a^i, z_b^j) / \tau)}$$

Training configuration: 512-dimensional embeddings, Adam optimizer (lr=1e-4), batch size 1024, 145 epochs, 24-frame windows, 256-point FPS downsampling.

## Key Experimental Results

### Main Results — MSR-Action3D HAR

| Method | Pre-training | Acc@1 (%) |
|--------|-------------|-----------|
| PST-Transformer (baseline) | None | 93.73 |
| PST-Transformer† (reproduced) | None | 92.33 |
| PSTNet + PointCMP | Unimodal self-supervised | 93.27 |
| PST-Transformer + MaST-Pre | Unimodal self-supervised | 94.08 |
| PST-Transformer + M2PSC | Unimodal self-supervised | 94.84 |
| PST-Transformer + DePITE | **Multimodal contrastive (Ours)** | 95.12 |
| PST-Transformer + DeSPIE | **Multimodal contrastive (Ours)** | **95.47** |
| PST-Transformer + DeSPITE | **Multimodal contrastive (Ours)** | **95.47** |

DeSPITE/DeSPIE surpass all existing pre-training methods, outperforming the strongest unimodal method M2PSC by 0.63%, and approaching the performance of KAN-HyperpointNet (95.59%).

### Ablation Study — HMPEAR Dataset Results

| Method | Modality | Acc(Seg) (%) |
|--------|----------|-------------|
| PST-Transformer† | PC | 65.94 |
| PEAR-Proj (BestAR) | RGB+PC | 66.0 |
| PST-Transformer + DeSPITE | PC | 69.18 (+3.24) |
| PST-Transformer + DeSPIE | PC | 70.26 (+4.32) |
| PST-Transformer + DePITE | PC | **70.65 (+4.71)** |

The DeSPITE family achieves a new state of the art on HMPEAR, outperforming all prior point-cloud-only, RGB-only, and multimodal methods, with pre-training yielding nearly 4% improvement.

### Key Findings

1. **Text hurts matching/retrieval but benefits HAR**: Models incorporating the text modality (DeSPITE, DePITE, etc.) almost consistently underperform text-free models (DeSPIE, DeSPE, etc.) on matching and temporal retrieval tasks, yet achieve better performance when fine-tuned for HAR.
2. **More modalities = better HAR pre-training**: DeSPITE (four modalities), DeSPIE (three modalities, no text), and DePITE (three modalities, no skeleton) consistently achieve the best performance on both MSR-Action3D and HMPEAR.
3. **Cross-modal matching feasibility**: IMU ↔ skeleton matching is the easiest; point cloud ↔ skeleton is intermediate; IMU ↔ point cloud is the most challenging.

## Highlights & Insights

- This work is the first to incorporate LiDAR point cloud sequences into a multimodal contrastive learning framework, opening a new direction for privacy-preserving multimodal human activity understanding.
- The finding that text embeddings play opposite roles in different downstream tasks (harmful for matching/retrieval, beneficial for HAR) reveals the complexity of joint embedding spaces.
- Exhaustive training and evaluation across all modality subset combinations provides an exceptionally systematic experimental analysis.

## Limitations & Future Work

- The LIPD dataset contains limited real LiDAR data, with the majority being synthetic, which may affect generalization.
- The 24-frame window (approximately 2.4 seconds) limits the model's capacity to capture long-horizon activities.
- IMU ↔ point cloud matching performance is relatively weak; stronger alignment strategies warrant future exploration.

## Related Work & Insights

- The key distinction from ImageBind lies in shifting the visual anchor from RGB to LiDAR.
- This work can inspire the development of general-purpose LiDAR foundation models oriented toward privacy preservation.
- It provides a new tool for interpretability of IMU signals in AR/VR contexts by enabling retrieval of skeletons or point clouds to visualize the semantic content of IMU data.

## Rating

- Novelty: ⭐⭐⭐⭐ (First exploration of LiDAR-centric multimodal contrastive learning)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Exhaustive modality combinations + multiple datasets + multiple tasks)
- Writing Quality: ⭐⭐⭐⭐ (Clear and systematic)
- Value: ⭐⭐⭐⭐ (Opens a new direction for privacy-preserving human activity understanding)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections](adaptive_hyper_graph_convolution_network_skeleton_action_recognition.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)

</div>

<!-- RELATED:END -->

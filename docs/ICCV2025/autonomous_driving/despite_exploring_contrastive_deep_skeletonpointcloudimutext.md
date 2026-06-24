---
title: >-
  [Paper Note] DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding
description: >-
  [ICCV 2025][Autonomous Driving][multimodal contrastive learning] This paper proposes DeSPITE, a contrastive learning framework that aligns four modalities—LiDAR point clouds, skeletal poses, IMU signals, and text—into a joint embedding space. It is the first to adopt LiDAR (rather than RGB) as the primary visual modality, enabling previously infeasible tasks such as cross-modal matching and retrieval, while also serving as an effective HAR pretraining strategy that achieves s…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "multimodal contrastive learning"
  - "LiDAR point cloud"
  - "human activity recognition"
  - "cross-modal retrieval"
  - "privacy preservation"
date: 2026-05-08
content_hash: 156fedb3e6ce86b0
---

# DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding

**Conference**: ICCV 2025
**arXiv**: [2506.13897](https://arxiv.org/abs/2506.13897)  
**Code**: None (the paper states that pretrained encoders, code, and data will be released publicly)  
**Area**: Other
**Keywords**: multimodal contrastive learning, LiDAR point cloud, human activity recognition, cross-modal retrieval, privacy preservation

## TL;DR

This paper proposes DeSPITE, a contrastive learning framework that aligns four modalities—LiDAR point clouds, skeletal poses, IMU signals, and text—into a joint embedding space. It is the first to adopt LiDAR (rather than RGB) as the primary visual modality, enabling previously infeasible tasks such as cross-modal matching and retrieval, while also serving as an effective HAR pretraining strategy that achieves state-of-the-art performance on MSR-Action3D and HMPEAR.

## Background & Motivation

RGB cameras are restricted in privacy-sensitive domains such as healthcare and surveillance, whereas LiDAR is inherently privacy-preserving. Prior multimodal contrastive learning methods (ImageBind, IMU2CLIP, MotionCLIP, LAVIMO, etc.) uniformly rely on RGB as the primary visual modality to "bind" other modalities. However, contrastive alignment between LiDAR point clouds and modalities such as IMU and skeleton has received virtually no attention. The authors pose a central research question: **What happens if we rely solely on LiDAR as the primary visual modality in multimodal contrastive learning?**

## Core Problem

1. Cross-modal matching and retrieval among the three privacy-preserving modalities of LiDAR point clouds, skeletal poses, and IMU signals have been entirely unexplored.
2. The point cloud HAR field lacks general-purpose pretrained models; existing self-supervised methods are pretrained only on small datasets.
3. No large-scale dataset simultaneously contains point cloud, skeleton, IMU, and text annotations.

## Method

### Overall Architecture

The core idea of DeSPITE is straightforward: human motion exhibits a natural correspondence across sensor modalities—the same person performing the same action is described equivalently by the LiDAR point cloud sequence, the worn IMU signals, and the extracted skeletal poses. DeSPITE exploits this correspondence by mapping all four modalities into a shared 512-dimensional embedding space via the InfoNCE contrastive loss.

Each modality is encoded by a dedicated encoder:
- **Point cloud**: PST-Transformer with a SimCLR projection head
- **IMU**: 2-layer LSTM
- **Skeleton**: ACTOR encoder (Transformer VAE)
- **Text**: Frozen CLIP text encoder (serving as the "anchor" modality)

All inputs are standardized to a 24-frame window; each frame's point cloud is downsampled to 256 points via farthest point sampling.

### Key Designs

1. **Flexible modality combinations**: All possible modality subsets are trained (e.g., DeSPIE = Skeleton + Point cloud + IMU + Text minus T; DeSPE = Skeleton + Point cloud; DePIE = Point cloud + IMU), systematically examining each modality's contribution.

2. **LIPD-Babel dataset construction**: This represents a significant engineering contribution. The authors merge the LIPD dataset (containing point clouds, IMU, and skeleton but no activity labels) with the Babel dataset (containing text annotations for AMASS motion sequences) via sequence ID mapping. Frame rate differences between the two datasets (Babel at 30 FPS vs. LIPD at 10 FPS) require downsampling for alignment. Two versions are constructed:
    - LIPD-Babel-v1: for matching and retrieval evaluation (502K/85K train/test windows)
    - LIPD-Babel-v2: for HAR classification evaluation (403K/58K train/test windows, with text annotations)

3. **Text as an optional binding modality**: Not all training samples carry text annotations. A boolean mask $tm$ handles samples with missing text—the text contrastive loss is computed only on the subset with text pairings.

4. **An interesting finding**: The text modality is harmful for matching and retrieval tasks but beneficial for HAR fine-tuning. This suggests that the semantic structure of the CLIP text embedding space aids the learning of more discriminative activity features, while its coarse granularity undermines fine-grained spatiotemporal alignment.

### Loss & Training

The total loss consists of two terms:

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{text} + \beta \mathcal{L}_{M}$$

where $\alpha = \beta = 0.5$.

- $\mathcal{L}_{text}$: InfoNCE contrastive loss between each sensor modality and text (computed only on samples with text annotations)
- $\mathcal{L}_{M}$: InfoNCE contrastive loss between all pairwise sensor modalities (point cloud↔IMU, point cloud↔skeleton, IMU↔skeleton)

Each pairwise loss is bidirectionally symmetric: $\mathcal{L}_{a,b} = \frac{1}{2}(\mathcal{L}_{a \to b} + \mathcal{L}_{b \to a})$

Training runs for 145 epochs using the Adam optimizer with lr=1e-4, batch size 1024, and a learnable temperature parameter $\tau$. Random translation, scaling, and Gaussian noise augmentations are applied to prevent overfitting. HAR fine-tuning uses SGD with warmup to lr=0.01 over 35 epochs.

## Key Experimental Results

### MSR-Action3D (point cloud HAR, clip-level accuracy)

| Method | Accuracy |
|--------|----------|
| PST-Transformer (baseline) | 93.73 |
| PST-Transformer + MaST-Pre | 94.08 |
| PST-Transformer + M2PSC | 94.84 |
| PvNext | 94.77 |
| KAN-HyperpointNet | 95.59 |
| **PST-Transformer + DeSPIE (Ours)** | **95.47** |
| **PST-Transformer + DeSPITE (Ours)** | **95.47** |

### HMPEAR (point cloud HAR, segment-level accuracy)

| Method | Modality | Acc(Seg) |
|--------|----------|----------|
| PST-Transformer | PC | 65.94 |
| PEAR-Proj (BestAR) | RGB+PC | 66.0 |
| **PST-Transformer + DePITE (Ours)** | **PC** | **70.65 (+4.71)** |

### LIPD-Babel-v2 (multimodal HAR)

| Method | Modality | Acc(Seg) |
|--------|----------|----------|
| PST-Transformer (scratch) | PC | 67.38 |
| LSTM (scratch) | IMU | 65.62 |
| ACTOR (scratch) | Skeleton | 68.23 |
| PST-Transformer + DeSPITE | PC | 69.00 (+1.62) |
| LSTM + DeSPIE | IMU | 69.21 (+3.59) |
| ACTOR + DeSPITE | Skeleton | 70.64 (+2.41) |

### Ablation Study

1. **The double-edged effect of the text modality**: Models trained with text (DeSPITE, DePITE, etc.) consistently underperform their text-free counterparts (DeSPIE, DePIE, etc.) on matching and temporal retrieval tasks, yet perform better on HAR fine-tuning. This indicates that semantic information from the CLIP text space aids classification but harms fine-grained alignment.

2. **Impact of modality combinations**: Involving more modalities in pretraining generally benefits downstream HAR—DeSPITE/DeSPIE/DePITE consistently outperform two-modality variants.

3. **Cross-modal difficulty**: IMU↔skeleton matching and retrieval is easiest, point cloud↔skeleton is intermediate, and IMU↔point cloud is hardest—consistent with intuition, as IMU and skeleton both directly describe joint motion while point clouds are more abstract.

4. **Frozen vs. fine-tuned encoders**: Detailed ablations show that fine-tuning generally outperforms linear/nonlinear probing, though probing results are also competitive, confirming that pretraining learns meaningful representations.

## Highlights & Insights

1. **Well-framed research question**: Positioning LiDAR rather than RGB as the central visual modality in multimodal contrastive learning is a concise yet unexplored direction. The privacy-preservation motivation is practically grounded.

2. **Systematic experimental design**: Training all modality combinations (DeSPE, DeSIE, DePIE, …) entails substantial experimental effort but yields a clear panoramic view of each modality's contribution.

3. **Engineering value of dataset construction**: Although LIPD-Babel is not a technical novelty per se, its construction via clever sequence ID mapping and frame rate alignment provides the community with a four-modality training resource that previously did not exist.

4. **The "text is harmful yet useful" finding**: The observation that the text modality is detrimental for matching/retrieval but beneficial for classification appears paradoxical and warrants deeper reflection—it reflects a mismatch between the granularity of semantic alignment and the granularity of downstream tasks.

## Limitations & Future Work

1. **Straightforward methodology**: The framework comprises four independent encoders with InfoNCE contrastive loss and introduces limited architectural innovation. Each modality relies on an existing encoder (PST-Transformer, LSTM, ACTOR), making the contribution more exploratory than methodological.

2. **Reliance on synthetic data**: The LIPD dataset depends heavily on synthetic LiDAR point clouds and IMU data; noise and occlusion challenges in real-world scenarios are not adequately discussed.

3. **Suboptimal text alignment**: Text retrieval performance lags considerably behind TMR++ (R-Top-1: 42.55 vs. 55.54), indicating substantial room for improvement in aligning the CLIP text space with motion modalities.

4. **Single-person scenario limitation**: Although multi-person scenes are artificially simulated, the underlying data originate from single-person motion capture; occlusion and interference in real multi-person interaction scenarios are not considered.

5. **Encoder selection not optimized**: Using an LSTM for IMU encoding appears dated; more modern architectures (e.g., Transformer or Mamba) may yield superior IMU representations.

## Related Work & Insights

| Method | Primary Visual Modality | Aligned Modalities | Main Application |
|--------|------------------------|--------------------|-----------------|
| CLIP | RGB | Text | General vision-language |
| ImageBind | RGB | 6 modalities | General multimodal |
| IMU2CLIP | RGB (CLIP) | IMU | IMU retrieval |
| MotionCLIP | RGB (CLIP) | Skeleton | Motion generation |
| LAVIMO | RGB | Skeleton+Text | Skeleton retrieval |
| **DeSPITE** | **LiDAR PC** | **Skeleton+IMU+Text** | **Privacy-preserving HAR, cross-modal retrieval** |

The core differentiator of DeSPITE lies in completely abandoning RGB and constructing the multimodal embedding space centered on LiDAR point clouds.

## Relevance to My Research

- The contrastive learning framework for aligning multiple modalities is transferable to other privacy-sensitive domains (e.g., aligning different sensor modalities in medical imaging).
- The finding that "text aids classification but harms retrieval" is worth bearing in mind when designing multimodal pretraining strategies.

## Rating

- **Novelty**: ⭐⭐⭐ — The research question is novel (LiDAR replacing RGB), but the methodology follows standard contrastive learning practice
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive: all modality combinations, multi-task, multi-dataset, and extensive ablations
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-articulated motivation, and systematic experimental presentation
- **Value**: ⭐⭐ — The research direction is relatively distant from my own, but the experimental methodology for multimodal contrastive alignment is worth referencing

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration](mixed_signals_a_diverse_point_cloud_dataset_for_heterogeneous_lidar_v2x_collabor.md)
- [\[ICCV 2025\] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking](trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)
- [\[CVPR 2025\] Unlocking Generalization Power in LiDAR Point Cloud Registration](../../CVPR2025/autonomous_driving/unlocking_generalization_power_in_lidar_point_cloud_registration.md)
- [\[ICML 2025\] Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration](../../ICML2025/autonomous_driving/geometry-to-image_synthesis-driven_generative_point_cloud_registration.md)
- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)

</div>

<!-- RELATED:END -->

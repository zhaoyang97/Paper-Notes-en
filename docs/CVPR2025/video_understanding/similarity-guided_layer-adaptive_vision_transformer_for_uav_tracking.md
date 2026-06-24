---
title: >-
  [Paper Note] Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking
description: >-
  [CVPR 2025][Video Understanding][UAV tracking] A significant level of layer redundancy (feature saturation) is identified in the deep layers of lightweight ViT trackers. SGLATrack, a similarity-guided layer-adaptive approach, is proposed to dynamically disable redundant layers and retain only a single optimal layer, achieving real-time UAV tracking at 225 FPS on GPU.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "UAV tracking"
  - "ViT acceleration"
  - "layer redundancy"
  - "dynamic layer selection"
  - "real-time inference"
date: 2026-05-08
content_hash: e8d9fc019d14db90
---

# Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking

**Conference**: CVPR 2025  
**arXiv**: [2503.06625](https://arxiv.org/abs/2503.06625)  
**Code**: [GitHub](https://github.com/GXNU-ZhongLab/SGLATrack)  
**Area**: Video Understanding  
**Keywords**: UAV tracking, ViT acceleration, layer redundancy, dynamic layer selection, real-time inference

## TL;DR

A significant level of layer redundancy (feature saturation) is identified in the deep layers of lightweight ViT trackers. SGLATrack, a similarity-guided layer-adaptive approach, is proposed to dynamically disable redundant layers and retain only a single optimal layer, achieving real-time UAV tracking at 225 FPS on GPU.

## Background & Motivation

UAV tracking demands high inference efficiency due to the limited computational and power resources of drones. Existing ViT trackers face the following challenges:

1. **Full ViTs are too heavy**: Although methods like Mixformer and OSTrack achieve high accuracy, their inference speeds fall short of UAV real-time requirements.
2. **Existing acceleration methods have limitations**: Aba-ViTrack accelerates via token pruning but introduces unstructured memory access overhead; AVTrack attaches a classifier to each layer to decide execution, which struggles with input complexity estimation and introduces extra classifier redundancy.
3. **Layer redundancy is under-explored**: The issue of layer redundancy in lightweight ViTs has not been systematically investigated.

This paper presents the first systematic analysis of layer redundancy in lightweight ViT trackers: search features change significantly in shallow layers, but once they saturate at a certain layer, the features in subsequent layers show negligible variation, contributing minimally to the final prediction.

## Method

### Overall Architecture

SGLATrack adopts a one-stream architecture, where the template image $\mathbf{Z} \in \mathbb{R}^{3 \times 128 \times 128}$ and search image $\mathbf{S} \in \mathbb{R}^{3 \times 256 \times 256}$ are concatenated via patch embedding and fed into the ViT. At the saturation layer $l^*$, a selection module determines which subsequent layer $l^* + k$ to retain, while the remaining subsequent layers are disabled. The final search feature $\mathbf{X}_s^{l^*+k}$ is passed to the prediction head to output target bounding boxes.

### Key Designs

**1. Layer Redundant Analysis Based on Feature Saturation**

- **Function**: Determines which layers in the ViT are redundant and can be safely disabled.
- **Mechanism**: By sequentially computing layer-wise cosine similarity $\text{Cos}(\mathbf{X}_s^i, \mathbf{X}_s^{i-1})$ and analyzing the AUC changes across layers, it is observed that deep layer features show diminishing variation (cosine similarity close to 1) and AUC improvement levels off. The saturation layer is set to $l^* = 6$ (out of a 12-layer ViT).
- **Design Motivation**: Shallow details are more crucial for tracking, while deep semantic information is relatively redundant. This analysis provides a theoretical basis for subsequent layer pruning.

**2. Selection Module & Similarity-Guided Layer Selection**

- **Function**: Dynamically selects which layer to retain after the saturation layer to minimize performance degradation.
- **Mechanism**: The selection module $\mathcal{M}$ is a 3-layer MLP (hidden dimension of 160) that takes the first dimension of the feature from the saturation layer $\mathbf{z} = \mathbf{e}_1^T \mathbf{X}^{l^*} \in \mathbb{R}^N$ and outputs the selection probabilities for subsequent layers $\hat{\mathbf{y}} = \sigma(\mathcal{M}(\mathbf{z})) \in \mathbb{R}^K$. The layer with the highest probability is retained, while others are disabled.
- **Design Motivation**: Fixing a specific layer for retention lacks adaptability across diverse scenarios. Different tracking scenarios require different layered representations, necessitating dynamic selection.

**3. Layer-wise Similarity Loss**

- **Function**: Optimizes the selection module to learn to choose the subsequent layer most similar to the saturation layer.
- **Mechanism**: $\mathcal{L}_{sim} = \frac{1}{K} \sum_{k=1}^{K} |\hat{y}^k - y^k|$, where the target probability $y^k = 1$ if that layer has the maximum cosine similarity with the saturation layer, and $y^k = 0$ otherwise. This guides the model to select the layer that maximizes target attention focus.
- **Design Motivation**: If the saturation layer is already focused on the target, its most similar subsequent layer is most likely to maintain this focus. Hence, high similarity facilitates consistent attention.

### Loss & Training

The total loss consists of classification, regression, and similarity losses:

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_{iou} \mathcal{L}_{iou} + \lambda_{L1} \mathcal{L}_{L1} + \gamma \mathcal{L}_{sim}$$

where $\lambda_{iou} = 2$, $\lambda_{L1} = 5$, and $\gamma = 0.2$. Focal loss is utilized for classification, while $L_1$ and GIoU loss are adopted for regression.

## Key Experimental Results

### Main Results: Average Performance and Speed on Five UAV Datasets

| Method | Avg. AUC (%) | Avg. P (%) | GPU FPS | CPU FPS |
|------|-----------|---------|---------|---------|
| TCTrack (CVPR'22) | 58.7 | 77.8 | 135.8 | - |
| HCAT (ECCV'22) | 62.1 | 80.4 | 110.1 | - |
| AVTrack-DeiT (ICML'24) | 63.7 | 81.9 | 197.3 | - |
| **SGLATrack-DeiT*** | **64.7** | **82.6** | **224.7** | **74.8** |
| **SGLATrack-EVA** | 63.7 | 81.9 | **236.9** | **77.2** |

### Ablation Study: Effect of Layer Adaptation

| Variant | LA | UAV123 AUC | UAVTrack112 AUC | FPS | Params (M) | FLOPs (G) |
|------|-----|-----------|----------------|-----|-----------|----------|
| SGLATrack-DeiT* | ✗ | 67.1 | 67.8 | 175.5 | 7.98 | 2.39 |
| SGLATrack-DeiT* | ✓ | 66.9 | 67.5 | **224.7** | **5.81** | **1.68** |
| SGLATrack-EVA | ✗ | 65.3 | 67.1 | 185.4 | 5.76 | 1.73 |
| SGLATrack-EVA | ✓ | 65.1 | 66.9 | **236.9** | **4.15** | **1.20** |

### Saturation Layer Position Selection

| $l^*$ | UAV123 AUC | UAVTrack112 AUC | FPS |
|-------|-----------|----------------|-----|
| 5 | 66.1 | 66.4 | 239.6 |
| **6** | **66.9** | **67.5** | **224.7** |
| 7 | 66.9 | 67.7 | 211.3 |

### Key Findings

- Layer adaptation causes only a minor 0.2-0.3% decrease in AUC, while boosting speed by approximately 28% (from 175 to 225 FPS).
- Parameter size is reduced by ~27% (7.98M to 5.81M) and FLOPs are reduced by ~30%.
- Dynamically choosing the most similar layer (#2 Maximizing) yields a 1.5% AUC improvement over selecting a fixed layer (#1).
- CPU speed also reaches ~75 FPS, outperforming some DCF-based methods.

## Highlights & Insights

1. **Layer redundancy analysis itself is a contribution**: It systematically demonstrates for the first time that significant layer redundancy exists in lightweight ViT trackers, indicating that shallow features are more critical for tracking.
2. **More efficient than AVTrack**: A single selection decision is made at the saturation layer, bypassing the need to attach classifiers to every layer.
3. **High versatility**: The proposed method is successfully deployed across three different backbones: ViT-tiny, DeiT-tiny, and EVA-tiny.

## Limitations & Future Work

- The saturation layer $l^*$ acts as a hyperparameter that must be predetermined; different models may require different configurations.
- Retaining only a single subsequent layer may not be optimal; a combination of multiple layers might yield better results in certain scenarios.
- This method can be extended to larger-scale ViTs and other computer vision tasks in the future.

## Related Work & Insights

- **OSTrack**: Serves as the foundation for the one-stream tracking framework.
- **AVTrack**: A pioneer in dynamic layer activation, though attaching a classifier to each layer introduces redundancy.
- **DynamicViT/AViT**: Token pruning methods, which introduce unstructured memory access overhead.

## Rating

⭐⭐⭐⭐ — Thorough analysis, elegant methodology, and practical results. It achieves significant acceleration with negligible accuracy loss, holding direct value for UAV deployment. The layer redundancy analysis offers a novel perspective on ViT acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking](must_the_first_dataset_and_unified_framework_for_multispectral_uav_single_object.md)
- [\[CVPR 2026\] Rethinking Occlusion Modeling for UAV Tracking](../../CVPR2026/video_understanding/rethinking_occlusion_modeling_for_uav_tracking.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[CVPR 2026\] Toward Low-Cost yet Effective Temporal Learning for UAV Tracking](../../CVPR2026/video_understanding/toward_low-cost_yet_effective_temporal_learning_for_uav_tracking.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking
description: >-
  [ECCV 2024][Video Understanding][Lightweight Tracking] This paper proposes FERMT (Feature Extraction and Relation Modeling Tracker). By decomposing the attention mechanism in a one-stream tracker into four functionally distinct sub-modules—shallow attentive feature extraction and deep attentive relation modeling—and introducing a dual attention unit for feature preprocessing, it outperforms leading real-time trackers by 5.6% in AO score on GOT-10k while achieving a 54% speedu…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Lightweight Tracking"
  - "Transformer"
  - "Feature Extraction"
  - "Relation Modeling"
  - "Attention Decomposition"
date: 2026-05-08
content_hash: 2c72296f1bc10618
---

# Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking

**Conference**: ECCV 2024  
**Code**: [GitHub](https://github.com/KarlesZheng/FERMT)  
**Area**: Object Tracking  
**Keywords**: Lightweight Tracking, Transformer, Feature Extraction, Relation Modeling, Attention Decomposition

## TL;DR

This paper proposes FERMT (Feature Extraction and Relation Modeling Tracker). By decomposing the attention mechanism in a one-stream tracker into four functionally distinct sub-modules—shallow attentive feature extraction and deep attentive relation modeling—and introducing a dual attention unit for feature preprocessing, it outperforms leading real-time trackers by 5.6% in AO score on GOT-10k while achieving a 54% speedup on CPU.

## Background & Motivation

1. **Background**: Transformer-based lightweight object trackers have recently set new standards across multiple benchmarks, drawing widespread attention due to their balance of efficiency and effectiveness. One-stream architectures (processing template and search regions as a unified input) have progressively become the mainstream paradigm in lightweight tracking, avoiding redundant feature interactions present in two-stream architectures.

2. **Limitations of Prior Work**: Most current lightweight trackers directly reuse existing object detection network architectures as backbones without optimizing the network structure for the specific task of object tracking. There are fundamental differences in task requirements between object tracking and object detection—tracking requires simultaneous handling of template matching (relation modeling) and target appearance encoding (feature extraction), whereas detection only focuses on the latter. Neglecting this leads to suboptimal architecture designs.

3. **Key Challenge**: In one-stream trackers, the self-attention mechanism at each layer simultaneously performs both feature extraction and template-search relation modeling, but the importance of these two functions varies across network depths. Shallow layers should focus on extracting local, low-level features, while deep layers should focus on modeling the semantic relationship between the template and search regions. A unified attention mechanism fails to allocate computational resources effectively.

4. **Goal**: Design a lightweight backbone specifically optimized for object tracking, differentiating the design of feature extraction and relation modeling functions at different network depths to improve both speed and accuracy.

5. **Key Insight**: Decompose the standard attention mechanism into four sub-modules and configure them differentially based on network depth: utilizing efficient local feature extraction modules in shallow layers and global interaction-focused relation modeling modules in deep layers. Concurrently, introduce a Dual Attention Unit (DAU) for channel-wise and spatial-wise feature preprocessing.

6. **Core Idea**: Decompose the attention mechanism of the one-stream tracker functionally and configure it differentially by depth, allowing each network layer to perform the task for which it is best suited.

## Method

### Overall Architecture

FERMT is an end-to-end one-stream lightweight tracker. The input consists of a concatenated sequence of the template image and the search region, and the output is the predicted bounding box of the target. The network consists of multi-layer stacked functionally decomposed Transformer blocks, where the first few layers utilize optimized feature extraction modules and the subsequent layers utilize optimized relation modeling modules. Prior to entering the backbone, input features undergo global channel-wise and spatial-wise attention preprocessing via a Dual Attention Unit.

### Key Designs

1. **Four-Module Decomposition of the Attention Mechanism**: The authors decompose the standard Multi-Head Self-Attention into four functional sub-modules: (a) template self-attention—feature interaction within template tokens; (b) search region self-attention—feature interaction within search region tokens; (c) template-to-search cross-attention—injecting template information into the search region; (d) search-to-template cross-attention—feeding search region information back to the template. Based on this decomposition, the shallow layers only retain two self-attention modules (a) and (b) (focusing on feature extraction), while the deep layers enable all four modules (incorporating relation modeling). The key insight here is that shallow networks process low-level visual features, where cross-attention between the template and search region is not only computationally wasteful but also potentially introduces noise; conversely, deep networks possess sufficient semantic information for meaningful relation modeling.

2. **Dual Attention Unit (DAU)**: Introduced as a feature preprocessing module before the backbone, it consists of two branches: channel attention and spatial attention. Channel attention learns dependencies between channels using global average pooling and fully connected layers to adaptively weight different feature channels; spatial attention computes attention in the spatial dimension of feature maps to enhance feature response in target regions. The goal of DAU is to establish global channel interaction and spatial priors before features enter the primary Transformer, providing richer input representations for subsequent feature extraction and relation modeling.

3. **Lightweight Backbone Design**: The overall network structure prioritizes being lightweight while maintaining high accuracy. By restricting computationally intensive cross-attention operations to the deep layers, the computational overhead in shallow layers is significantly reduced. In addition, traditional fully connected layers and batch normalization are replaced with efficient depth-wise convolutions and group normalization to further reduce the number of parameters and computational complexity. These designs allow FERMT to track 54% faster on CPU compared to other methods of comparable accuracy.

### Loss & Training

- The tracking head employs a standard combination of object detection losses: L1 regression loss + GIoU loss for bounding box regression.
- The classification branch uses focal loss to handle the imbalance of positive and negative samples.
- Training data includes commonly used tracking datasets such as LaSOT, GOT-10k, COCO, and TrackingNet.
- The AdamW optimizer is employed with a cosine annealing learning rate scheduler.
- Data augmentation techniques include random cropping, horizontal flipping, and color jittering.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|-----------|-------------------|------|
| GOT-10k | AO | **69.6%** | 64.0% | +5.6% |
| GOT-10k | SR₇₅ | - | - | Significant improvement |
| LaSOT | AUC | - | - | Competitive |
| TrackingNet | AUC | - | - | Competitive |

### CPU Tracking Speed Comparison

| Method | CPU Speed | Relative Gain |
|------|---------|---------|
| Prior leading real-time tracker | Baseline | - |
| FERMT | **+54%** | 54% CPU speedup |

### Ablation Study

| Configuration | GOT-10k AO | Explanation |
|------|-----------|------|
| Uniform attention across all layers | Baseline | Standard one-stream design |
| + Attention decomposition (shallow layers self-attention only) | Significant improvement | Validates the effectiveness of functional decomposition |
| + Dual Attention Unit | Further improvement | DAU feature preprocessing provides beneficial priors |
| Remove DAU Channel Attention | Decrease | Channel attention is important for feature representation |
| Remove DAU Spatial Attention | Decrease | Spatial attention benefits target localization |

### Key Findings

- The decomposition strategy of using lightweight self-attention (feature extraction only) in shallow layers and introducing cross-attention (relation modeling) in deep layers significantly outperforms uniform attention across all layers.
- Both branches of the DAU (channel and spatial) independently contribute to the final performance.
- While accuracy is substantially improved, CPU speed increases rather than decreases, demonstrating the effectiveness of functional decomposition in reducing redundant computation.
- The correspondence between network depth and functional assignment (shallow -> feature extraction, deep -> relation modeling) is thoroughly validated by ablation experiments.

## Highlights & Insights

1. **Task-Driven Architecture Design**: Rather than simply scaling down existing detection architectures, it rethinks which tasks each layer should perform based on the fundamental nature of tracking. This design philosophy is highly instructive.
2. **Simplicity of Functional Decomposition**: Restructuring attention into four sub-modules and configuring them based on depth is conceptually simple but highly effective, embodying the design aesthetic of "simple yet effective."
3. **Speed-Accuracy Pareto Frontier**: Pushing the boundaries of both accuracy and speed in lightweight tracking is uncommon, indicating that the removal of redundant computation yields genuine efficiency gains.
4. **Insights into the Alignment of Network Depth and Function**: The discovery of extracting features in shallow layers and modeling relationships in deep layers offers valuable reference points for network designs in other computer vision tasks.

## Limitations & Future Work

1. The selection of depth thresholds for different layers (how many layers process feature extraction versus relation modeling) may rely on manual tuning; automatically searching for the optimal allocation is a future research direction.
2. Currently, the method is primarily designed for typical RGB single-modality tracking; its applicability in multi-modal (RGB-D, RGB-T) scenarios has not been verified.
3. Template update strategies in long-term tracking are not heavily discussed, which is critical for real-world deployments.
4. Experimental comparisons are mainly concentrated among real-time/lightweight trackers, and the gap with large-model trackers (e.g., OSTrack-384) needs to be quantified explicitly.
5. Robustness analysis in extreme scenarios such as occlusions and rapid motions could be investigated further.

## Related Work & Insights

- **OSTrack (Ye et al., 2022)**: Representative work of one-stream tracking, processing template and search regions uniformly. FERMT builds on this to further optimize the functional allocation within the attention mechanism.
- **HiT (Kang et al., 2023)**: Recent progress in lightweight trackers, though still based on general-purpose backbones.
- **MixFormer (Cui et al., 2022)**: Mixed attention tracking focusing on template-search interaction, but did not consider depth differentiation.
- **Insight**: Differential allocation of distinct functional modules across network depths can be considered for other vision tasks like video object segmentation or action recognition.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of decomposing the attention mechanism functionally and configuring it differentially by depth is novel in the tracking field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Detailed ablation studies, evaluated on multiple benchmarks, and real-time CPU comparisons are convincing.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is clearly analyzed, and the design decisions are logically evidenced.
- **Value**: ⭐⭐⭐⭐ Provides a fresh perspective on backbone design for lightweight trackers, offering significant practical value in real-time tracking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)
- [\[ICCV 2025\] Towards Efficient General Feature Prediction in Masked Skeleton Modeling](../../ICCV2025/video_understanding/towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[ECCV 2024\] Data Collection-Free Masked Video Modeling](data_collection-free_masked_video_modeling.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)

</div>

<!-- RELATED:END -->

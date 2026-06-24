---
title: >-
  [Paper Note] OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels
description: >-
  [CVPR 2025][Segmentation][Convolutional Neural Networks] OverLoCK is proposed, which is the first pure convolutional backbone network that explicitly incorporates a top-down attention mechanism. Through a deep-stage decomposition strategy (DDS) and context-mixing dynamic convolution (ContMix), it surpasses ConvNeXt-B on ImageNet-1K using only 1/3 of the FLOPs, achieving comprehensive leadership in detection and segmentation tasks.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Convolutional Neural Networks"
  - "Top-Down Attention"
  - "Dynamic Convolution"
  - "Long-Range Dependencies"
  - "Backbone"
date: 2026-05-08
content_hash: 9a05bac1319affd9
---

# OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels

**Conference**: CVPR 2025  
**arXiv**: [2502.20087](https://arxiv.org/abs/2502.20087)  
**Code**: [https://bit.ly/OverLoCK](https://bit.ly/OverLoCK)  
**Area**: Segmentation/Vision Backbone  
**Keywords**: Convolutional Neural Networks, Top-Down Attention, Dynamic Convolution, Long-Range Dependencies, Backbone

## TL;DR

OverLoCK is proposed, which is the first pure convolutional backbone network that explicitly incorporates a top-down attention mechanism. Through a deep-stage decomposition strategy (DDS) and context-mixing dynamic convolution (ContMix), it surpasses ConvNeXt-B on ImageNet-1K using only 1/3 of the FLOPs, achieving comprehensive leadership in detection and segmentation tasks.

## Background & Motivation

The top-down attention mechanism in the human visual system—first obtaining a global overview to discover salient cues, and then closely examining details—has been largely neglected in modern vision backbones.

The Key Challenge faced by current backbone networks:

1. **Lack of Feedback in Pyramidal Architectures**: Existing ConvNet/ViT/Mamba backbones adopt a step-by-step downsampling pyramidal structure. Intermediate layers can only rely on prior features, lacking explicit top-down semantic guidance.
2. **Experimental Verification**: Visualizations of Class Activation Maps (CAM) of Swin-T, ConvNeXt-T, and VMamba-T reveal that even at Stage 4 (close to the classifier), these models still struggle to accurately locate target objects, with performance being even worse at Stage 3.
3. **Limitations of Prior Work**: Recurrent top-down architectures introduce excessive computational overhead, resulting in a poor performance-complexity trade-off. Task-specific feedback designs are unsuitable for building general-purpose backbones.

Another key challenge is: how to equip pure convolutions with dynamic global modeling capabilities (analogous to Transformers/Mamba) while preserving the inherent local inductive bias of convolutions? The receptive field of large-kernel convolutions relatively shrinks as the resolution increases, while deformable convolutions sacrifice local inductive biases.

## Method

### Overall Architecture

OverLoCK decomposes the network into three collaborative sub-networks: Base-Net encodes mid-to-low-level features (Stage 1 to the first half of Stage 3); a lightweight Overview-Net rapidly generates coarse-grained global semantic overviews (Stages 3-4); a powerful Focus-Net performs detailed perception guided by top-down signals (Stages 3-4). The output of Overview-Net is injected into every building block of Focus-Net as a context prior.

### Key Designs

**Design 1: Deep-stage Decomposition Strategy (DDS)**

- **Function**: Explicitly encode the "overview first, look closely next" human visual mechanism into the network architecture.
- **Mechanism**: Base-Net downsamples images to $H/16 \times W/16$. Overview-Net further downsamples them to $H/32 \times W/32$ to rapidly obtain the semantic overview (context prior). Focus-Net receives features from Base-Net and the context prior, progressively refining features under top-down guidance. Sharing Base-Net between the two sub-backbones minimizes additional overhead.
- **Design Motivation**: Realizing top-down attention via a branched architecture rather than a recurrent one avoids the computational redundancy of recurrent structures. During pre-training, both sub-networks have individual classification heads, while only the Focus-Net output is used in downstream tasks.

**Design 2: Context-Mixing Dynamic Convolution (ContMix)**

- **Function**: Equip fixed-size convolution kernels with adaptive long-range dependency modeling capabilities while retaining the local inductive bias.
- **Mechanism**: For an input feature map, an affinity matrix $A^g \in \mathbb{R}^{HW \times S^2}$ is computed between each token and the $S \times S$ regional centers. The affinity values are aggregated into spatially-varying dynamic convolution kernels $D^g = \text{softmax}(A^g W_d)$ via a learnable linear layer $W_d$. Since the weights of each kernel encode global context, sliding-window convolutions can capture long-range dependencies.
- **Design Motivation**: The receptive field of large-kernel convolutions is static, shrinking relatively as resolution increases. By mixing global context into kernel weights, ContMix enables global perception even under a fixed kernel size, while maintaining the local structure of convolutions.

$$D^g = \text{softmax}(A^g W_d) \in \mathbb{R}^{HW \times K^2}$$

**Design 3: Context Flow and Gated Dynamic Spatial Aggregator (GDSA)**

- **Function**: Continuously update and utilize top-down semantic guidance within Focus-Net.
- **Mechanism**: The context prior $P_i$ and feature map $Z_i$ are concatenated and fed into the Dynamic Block. Within ContMix, $P_i$ is used to compute keys (regional centers) while $Z_i$ is used to compute queries, achieving the effect of "context-guided kernel weights". After separating the outputs, the context prior is updated via $P_{i+1} = \alpha P_i' + \beta P_o$ to prevent the dilution of context.
- **Design Motivation**: Top-down guidance should not be a one-time injection but should continuously affect the feature extraction process within each block. A residual connection to the initial context prior is leveraged to prevent info decay.

### Loss & Training

During ImageNet pre-training, Focus-Net and Overview-Net are each connected to an individual classification head, optimized with the same cross-entropy classification loss. In downstream tasks, Overview-Net no longer requires auxiliary supervision.

## Key Experimental Results

### ImageNet-1K Image Classification (224×224)

| Method | Type | FLOPs(G) | Params(M) | Top-1 Acc(%) |
|------|------|----------|-----------|-------------|
| ConvNeXt-T | ConvNet | 4.5 | 29 | 82.1 |
| UniRepLKNet-T | ConvNet | 4.9 | 31 | 83.2 |
| VMamba-T | Mamba | 4.9 | 30 | 82.6 |
| Swin-T | Transformer | 4.5 | 29 | 81.3 |
| **OverLoCK-T** | **ConvNet** | **4.6** | **29** | **84.2** |
| ConvNeXt-B | ConvNet | 15.4 | 89 | 83.8 |
| **OverLoCK-T vs ConvNeXt-B** | — | **~1/3 FLOPs** | **~1/3 Params** | **+0.4** |

### COCO Object Detection (Mask R-CNN 3x)

| Method | FLOPs(G) | $AP^b$ | $AP^m$ |
|------|----------|--------|--------|
| ConvNeXt-S | 348 | 49.7 | 43.8 |
| MogaNet-B | 373 | 49.9 | 44.2 |
| **OverLoCK-S** | **345** | **50.9** | **44.8** |

### ADE20K Semantic Segmentation (UperNet)

| Method | FLOPs(G) | mIoU |
|------|----------|------|
| UniRepLKNet-T | 946 | 48.6 |
| MogaNet-S | 946 | 49.2 |
| **OverLoCK-T** | **930** | **50.3** |

### Key Findings

1. OverLoCK-T achieves 84.2% Top-1 accuracy with ~4.6G FLOPs, surpassing ConvNeXt-B which requires 15.4G FLOPs.
2. Effective Receptive Field (ERF) visualizations demonstrate that OverLoCK-T enjoys a larger ERF at Stages 3 & 4 than VMamba-T, despite being a pure ConvNet.
3. Class Activation Maps indicate that OverLoCK can accurately locate targets as early as Stage 3, validating the effectiveness of top-down guidance.
4. ContMix ablation analysis indicates that concurrent usage of large and small kernel groups (multi-scale) yields the best performance.

## Highlights & Insights

1. **Biologically-inspired Architectural Innovation**: Demonstrates the first explicit realization of top-down attention in a pure ConvNet, without relying on recurrent structures or introducing Transformer blocks.
2. **Core Insight of ContMix**: By encoding global context into convolution kernel weights, it cleverly endows fixed-size convolutions with "resolution-adaptive" long-range modeling capability.
3. **Excellent Efficiency-Accuracy Trade-off**: OverLoCK-T outperforms ConvNeXt-B's accuracy with only approximately 1/3 of its computational cost, demonstrating the immense potential of the architectural design.

## Limitations & Future Work

1. Overview-Net introduces extra branch computations, which is lightweight but still incurs some overhead.
2. The three-subnet architecture increases design complexity and the hyperparameter space.
3. The number of regional centers $S=7$ in ContMix is fixed; the possibility of adaptive adjustment was not explored.
4. Exploring the extension of the DDS strategy to Transformer or Mamba architectures is worth exploring.

## Related Work & Insights

- **ConvNeXt/RepLKNet/UniRepLKNet**: Evolution path of large-kernel ConvNets; OverLoCK addresses long-range modeling from a different perspective (dynamic kernels).
- **InternImage**: Achieves dynamic modeling via deformable convolutions but sacrifices inductive bias.
- **AbsViT**: Feedback-driven ViT backbone, but relies on recurrent structures. OverLoCK avoids recurrence through a branched design.
- Insight: The "context-to-kernel" concept of ContMix can be applied to other convolutional scenarios requiring long-range dependencies.

## Rating

⭐⭐⭐⭐⭐ — Implements a major breakthrough in pure ConvNet architectures. The core innovations (DDS + ContMix) feature solid theoretical foundations, comprehensive experiments, and outstanding performance. The efficiency-accuracy trade-off, which surpasses ConvNeXt-B with only 1/3 of the computational budget, is highly impressive. Underpins landmark progress for vision backbone designs in 2025.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Frequency Dynamic Convolution for Dense Image Prediction](frequency_dynamic_convolution_for_dense_image_prediction.md)
- [\[CVPR 2025\] Paint by Inpaint: Learning to Add Image Objects by Removing Them First](paint_by_inpaint_learning_to_add_image_objects_by_removing_them_first.md)
- [\[ICML 2025\] ConText: Driving In-context Learning for Text Removal and Segmentation](../../ICML2025/segmentation/context_driving_in-context_learning_for_text_removal_and_segmentation.md)
- [\[CVPR 2025\] The Power of Context: How Multimodality Improves Image Super-Resolution](the_power_of_context_how_multimodality_improves_image_super-resolution.md)
- [\[CVPR 2025\] ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting](rocket-1_mastering_open-world_interaction_with_visual-temporal_context_prompting.md)

</div>

<!-- RELATED:END -->

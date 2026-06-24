---
title: >-
  [Paper Note] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance
description: >-
  [CVPR 2025][Segmentation][RGB-D Scene Understanding] An efficient RGB-D multi-task scene understanding network is proposed. It accelerates inference by utilizing redundant features in an improved fusion encoder, introduces a Normalized Focus Channel Layer (NFCL) and Context Feature Interaction Layer (CFIL) for cross-dimensional feature guidance, and designs a multi-task adaptive loss function to dynamically adjust task weights, achieving SOTA performance on NYUv2/SUN RGB-D/Ci…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "RGB-D Scene Understanding"
  - "Multi-Task Adaptive Learning"
  - "Cross-Dimensional Feature Guidance"
  - "Panoptic Segmentation"
  - "Efficient Fusion Encoder"
date: 2026-05-08
content_hash: 3870cb517abefc5c
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance

**Conference**: CVPR 2025  
**arXiv**: [2603.07570](https://arxiv.org/abs/2603.07570)  
**Code**: To be confirmed  
**Area**: Image Segmentation / RGB-D Scene Understanding / Multi-Task Learning  
**Keywords**: RGB-D Scene Understanding, Multi-Task Adaptive Learning, Cross-Dimensional Feature Guidance, Panoptic Segmentation, Efficient Fusion Encoder

## TL;DR
An efficient RGB-D multi-task scene understanding network is proposed. It accelerates inference by utilizing redundant features in an improved fusion encoder, introduces a Normalized Focus Channel Layer (NFCL) and Context Feature Interaction Layer (CFIL) for cross-dimensional feature guidance, and designs a multi-task adaptive loss function to dynamically adjust task weights, achieving SOTA performance on NYUv2/SUN RGB-D/Cityscapes.

## Background & Motivation

### 1. Background
Scene understanding, which enables robots to accurately perceive environments, identify objects, and classify scenes, is the foundation of autonomous robotic decision-making. Multi-task learning achieves mutual reinforcement and collaborative optimization among multiple tasks through information sharing and learning mechanisms.

### 2. Limitations of Prior Work
- **Dual encoder efficiency issue**: Seichter et al. extract RGB and depth features separately using a dual encoder, but fail to fully integrate complementary information.
- **Transformer encoder speed issue**: Fischedick et al. jointly extract RGB-D information using Swin Transformer v2, but suffer from high matrix calculation and memory access overheads, leading to slow inference.
- **MLP decoder limitation**: The MLP decoder has a simple structure, but error information from shallow features can mislead it, and it primarily focuses on global features while ignoring local information.
- **Fixed loss weight issue**: Different tasks have significantly different learning difficulties and data distributions. Fixed weights cannot adapt to the dynamically changing task relationships during training.

### 3. Key Challenge
How to efficiently fuse RGB-D information while ensuring speed, how to guide cross-dimensional features to integrate local and global information, and how to adaptively adjust multi-task learning priorities.

### 4. Mechanism
Three innovations: (1) An efficient fusion encoder utilizing channel redundancy; (2) Cross-dimensional feature guidance via NFCL and CFIL; (3) A multi-task adaptive loss based on historical performance.

### 5. Prior Attempts and Limitations
- Lin et al. used random loss and weighting to avoid human bias, but introduced performance instability.
- Liu et al. calculated task weights based on training loss to improve stability, but only adjusted weights in the first batch of each iteration, lacking real-time adaptability.
- Bottleneck modules lead to information loss and decreased feature diversity due to dimensionality reduction.

### 6. Solution Overview
A unified multi-task network is designed to handle five tasks: semantic segmentation, instance segmentation, orientation estimation, panoptic segmentation, and scene classification. It is collaboratively optimized through the fusion encoder, NFCL, CFIL, and adaptive loss.

## Method

### Overall Architecture
RGB-D input $\rightarrow$ improved fusion encoder (4 stages, based on FasterNet-M) $\rightarrow$ three-branch output: scene classification head (fully connected layer), semantic decoder (MLP + NFCL + CFIL), instance decoder (Non-bottleneck 1D three-layer structure). Semantic segmentation provides foreground masks to instance segmentation, and both are combined to achieve panoptic segmentation. A multi-task adaptive loss is used during training.

### Key Design 1: Efficient Fusion Encoder
- **Function**: Simultaneously extract complementary features from RGB and depth data to improve inference speed.
- **Mechanism**: Exploit the high similarity (redundancy) of feature channels, performing convolutions on only 1/4 of the channels, then concatenating them with the remaining channels, reducing FLOPs to 1/16 of standard convolutions.
- **Design Motivation**: Features from different channels are highly similar, eliminating the need to perform convolutions on all channels; reducing memory access frequency significantly improves inference speed.
- **Depth Weight Initialization**: Sum the ImageNet pre-trained RGB three-channel weights to initialize depth weights as $D=(R+G+B)/2$, avoiding additional pre-training.
- **4-Stage Design**: Each stage contains 3/4/18/3 fusion blocks, respectively. The number of blocks is increased in later stages since the image size is smaller.

### Key Design 2: Normalized Focus Channel Layer (NFCL)
- **Function**: Enhance the representational capacity of shallow encoder features in the semantic decoder.
- **Mechanism**: Utilize the absolute value of the scaling factor $\gamma$ learned in BatchNorm as an indicator of channel importance. After normalization, channel features are re-weighted and re-ordered.
- **Design Motivation**: MLP decoders are easily misled by noise/error information in shallow features. NFCL automatically identifies important channels using the BN $\gamma$ parameter (greater $\gamma$ = greater variance = contains more key information).
- **Placement**: Applied to the first 3 layers of skip connections in the semantic decoder (stage 4 encoder features are already sufficient and require no extra guidance).

### Key Design 3: Context Feature Interaction Layer (CFIL)
- **Function**: Compensate for the inadequacy of the MLP semantic decoder in fusing local and global information.
- **Mechanism**: Capture multi-scale context information using multi-scale adaptive average pooling ($1\times1$ and $5\times5$), compress channels to $C/2$, perform bilinear upsampling to unify resolution, and concatenate with original features for fusion.
- **Design Motivation**: MLP decoders excel at non-linear mapping but primarily focus on global features. CFIL integrates features of different resolutions through multi-scale pooling, improving the ability to distinguish fine structures and boundaries.

### Key Design 4: Non-bottleneck 1D Instance Decoder
- **Function**: Feature extraction for instance segmentation and orientation estimation.
- **Mechanism**: Decompose a $3\times3$ 2D convolution into two 1D convolutions ($3\times1 + 1\times3$) with non-linear activation functions inserted in between.
- **Design Motivation**: 3D decomposition reduces parameters by 30% (when kernel size is 3) while increasing non-linear capacity.

### Key Design 5: Multi-Task Adaptive Loss
- **Function**: Dynamically adjust the loss weights of each task.
- **Mechanism**: Calculate the relative loss $RL_k = L_k/\sum L_t$ for each task after each batch, maintain a historical running average of relative loss $AvgRL_k$, and update the weights as $W_k = \max(\bar{W}_k \times (AvgRL_k)^\alpha, W_{\min})$ using an adjustment factor $\alpha$.
- **Design Motivation**: The learning difficulty of different tasks changes dynamically, and fixed weights cannot adapt. Using $\alpha=0.01$ achieves fine-grained adjustment, and $W_{\min}=0.1$ prevents any task from being completely ignored.

### Loss & Training
- Semantic segmentation: Cross-entropy loss
- Instance center: MSE loss
- Instance offset: MAE loss
- Orientation estimation: Continuous probability distribution loss based on cos/sin vectors
- Scene classification: Cross-entropy loss
- Panoptic segmentation: No additional loss is computed; evaluated during validation

## Key Experimental Results

### Encoder Comparison (Table 1, NYUv2)

| Encoder | Instance PQ↑ | MAAE↓ | Semantic mIoU↑ | Inference Speed |
|--------|-------------|-------|----------------|---------|
| Swin v2 | 58.49 | 21.09 | 49.76 | Slow |
| ConvNeXt v2 | 41.04 | 31.24 | 27.69 | Medium |
| MPViT | 57.77 | 21.18 | 47.44 | Relatively Slow |
| MetaFormer | 53.31 | 23.69 | 43.27 | Medium |
| **Ours** | **58.59** | **18.67** | 46.83 | **Fast** |

### Model Complexity Comparison (Table 8)

| Method | Params | FLOPs | FPS | VRAM | Sem. mIoU | Inst. PQ |
|------|--------|-------|-----|------|-----------|----------|
| EMSAFormer | 72.08M | 50.66G | 16.32 | 3188M | 49.76 | 58.49 |
| **Ours** | **71.82M** | 75.28G | **20.33** | 3293M | **49.82** | **59.90** |

### Semantic Segmentation SOTA Comparison

| Dataset | Method | Backbone | mIoU↑ |
|--------|------|----------|-------|
| NYUv2 | EMSAFormer | Swin v2 | 49.76 |
| NYUv2 | **Ours** | FasterNet-M | **49.82** |
| SUN RGB-D | EMSAFormer | Swin v2 | 44.13 |
| SUN RGB-D | **Ours** | FasterNet-M | **45.56** |
| Cityscapes | EMSAFormer | Swin v2 | 60.76 |
| Cityscapes | **Ours** | FasterNet-M | **65.11** |

### Ablation Study (Table 7, Stepwise Addition of Components to Framework)

| Config | Inst. PQ↑ | Pan. mIoU↑ | Sem. mIoU↑ | bAcc↑ |
|------|-----------|-----------|-----------|-------|
| Baseline (Swin v2) | 58.49 | 50.51 | 49.76 | 77.11 |
| + Fusion Encoder | 58.59 | 47.37 | 46.83 | 74.67 |
| + Adaptive Loss | 59.37 | 48.39 | 47.72 | 76.23 |
| + CFIL | 59.25 | 50.16 | 49.72 | 77.00 |
| + NFCL | **59.90** | **50.21** | **49.82** | 76.57 |

### Key Findings
1. **Fusion Encoder is 24.6% faster than Swin v2**: FPS increases from 16.32 to 20.33 with fewer parameters (71.82M vs 72.08M).
2. **CFIL achieves significant effects**: Semantic mIoU improves by 2.0 percentage points (from 47.72% to 49.72%), outperforming context modules like ASPP, SPPELAN, RFB, etc.
3. **Applying NFCL to the first 3 layers is optimal**: Stage 4 encoder features are already sufficient and require no extra guidance.
4. **Multi-task adaptive loss outperforms fixed weights**: The model is most balanced when the adjustment factor $\alpha=0.01$. The adaptive loss model converges faster and more stably.
5. **Non-bottleneck 1D outperforms other extraction modules**: Achieves an Instance PQ of 59.25%, outperforming BasicBlock, Bottleneck, MobileBottleneck, and GhostBottleneck.
6. **Indoor-to-outdoor generalization**: Achieves an mIoU of 65.11% on Cityscapes, surpassing Lovász (63.06%) and EMSAFormer (60.76%).

## Highlights & Insights
1. **Exploiting channel redundancy** is simple yet highly efficient: performing convolutions on only 1/4 of the channels reduces FLOPs by 16 times, significantly accelerating inference.
2. **Using BN $\gamma$ as channel importance** is an elegant design: it introduces no extra parameters and reuses the learning results of existing BN layers.
3. **Adaptive loss is based on historical performance** rather than a single batch, making it more stable than Liu et al.'s approach.
4. **Unified multi-task framework** handles 5 tasks simultaneously (semantic segmentation, instance segmentation, orientation estimation, panoptic segmentation, and scene classification), offering high practicality.
5. The depth weight initialization $D=(R+G+B)/2$ is simple but effectively leverages ImageNet pre-training.

## Limitations & Future Work
1. **Balance between accuracy and speed**: The fusion encoder only samples 1/4 of the channels, which might lose information. Automatically choosing the optimal ratio via Neural Architecture Search (NAS) is worth exploring.
2. **Scalability to high resolutions**: The current implementation is difficult to handle ultra-high resolution images/videos.
3. **Assumption of high-quality RGB-D input**: Noise in consumer-grade depth sensors such as reflection, transparent surfaces, and boundary sparseness are not handled.
4. **Frame-by-frame independent processing**: Temporal consistency is not utilized, which may lead to segmentation flickering in video scenes.
5. Conflicts might exist in the requirements of different tasks on the fusion encoder; the current shared encoder scheme may not be optimal.

## Related Work & Insights
- Relation to EMSAFormer: This paper uses EMSAFormer as the main baseline, replacing its Swin v2 encoder with the faster fusion encoder, and adding NFCL, CFIL, and adaptive loss.
- Relation to FasterNet: The fusion encoder is based on the partial convolution concept of FasterNet-M, extending it to RGB-D fusion scenarios.
- The concept of utilizing BN $\gamma$ in NFCL is inspired by Network Slimming (channel pruning) but applied in a different context (feature enhancement vs. pruning).

## Rating
- Novelty: ⭐⭐⭐ (The design of individual components is reasonable, but the innovation margin is limited; concepts like channel redundancy utilization, BN $\gamma$ as channel weights, and adaptive loss have been explored previously.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated on three datasets, with ablation studies covering the encoder, CFIL, NFCL, decoder, and loss; however, comparisons with other multi-task baselines are relatively sparse.)
- Writing Quality: ⭐⭐⭐⭐ (Complete and clear structure, rigorous formulas, with discussions on ethics and limitations; some parts could be more concise.)
- Value: ⭐⭐⭐⭐ (Provides a practical multi-task RGB-D scene understanding solution that balances speed and accuracy, comprehensively surpassing the baseline on three datasets.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene](learning_4d_panoptic_scene_graph_generation_from_rich_2d_visual_scene.md)
- [\[CVPR 2025\] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation](m3-vos_multi-phase_multi-transition_and_multi-scenery_video_object_segmentation.md)
- [\[CVPR 2025\] Towards Generalizable Scene Change Detection](towards_generalizable_scene_change_detection.md)
- [\[CVPR 2025\] Task-driven Image Fusion with Learnable Fusion Loss](task-driven_image_fusion_with_learnable_fusion_loss.md)
- [\[CVPR 2025\] Scene-Centric Unsupervised Panoptic Segmentation](scene-centric_unsupervised_panoptic_segmentation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] GroupMamba: Efficient Group-Based Visual State Space Model
description: >-
  [CVPR 2025][Segmentation][state space model] This paper proposes the Modulated Group Mamba layer, which divides input channels into four groups to perform unidirectional SSM scans in four distinct directions. It enhances cross-group channel interaction via Channel Affinity Modulation (CAM) and employs a distillation training objective to address instability in large models, achieving 83.3% Top-1 accuracy on ImageNet-1K with only 23M parameters.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "state space model"
  - "Mamba"
  - "group convolution"
  - "channel modulation"
  - "knowledge distillation"
  - "ImageNet"
date: 2026-05-08
content_hash: a04344b6ed7444c9
---

# GroupMamba: Efficient Group-Based Visual State Space Model

**Conference**: CVPR 2025  
**arXiv**: [2407.13772](https://arxiv.org/abs/2407.13772)  
**Code**: [GitHub](https://github.com/Amshaker/GroupMamba)  
**Area**: Image Segmentation  
**Keywords**: state space model, Mamba, group convolution, channel modulation, knowledge distillation, ImageNet

## TL;DR

This paper proposes the Modulated Group Mamba layer, which divides input channels into four groups to perform unidirectional SSM scans in four distinct directions. It enhances cross-group channel interaction via Channel Affinity Modulation (CAM) and employs a distillation training objective to address instability in large models, achieving 83.3% Top-1 accuracy on ImageNet-1K with only 23M parameters.

## Background & Motivation

**Background**: Visual State Space Models (Visual SSMs) such as VMamba and Vision Mamba leverage Mamba's linear complexity in handling long sequences, showing promise in vision tasks.

**Limitations of Prior Work**:
- **Low parameter efficiency**: The standard VSS block performs 4-way full scanning on all channels. The parameter counts of input/output projections and depthwise convolutions scale proportionally with the number of channels, causing parameter redundancy.
- **Unstable training**: Mamba-based models exhibit training instability when scaling up to larger models (e.g., SiMBA-L with an MLP channel mixer achieves only 49% accuracy).
- **Scanning redundancy**: Scanning all channels in every direction results in computational waste.

**Key Challenge**: Improving the parameter efficiency and training stability of Visual SSMs while maintaining effective modeling of local and global information.

**Key Insight**: Inspired by group convolution, this paper divides channels into four groups and scans each group along only one direction, which significantly reduces parameters. Meanwhile, a CAM mechanism is designed to compensate for the insufficient channel interaction caused by grouping.

## Method

### Overall Architecture

A four-stage hierarchical structure similar to Swin Transformer is adopted:
1. Patch Embedding (two $3\times3$ convolutions, stride=2) generates initial features of size $H/4\times W/4$.
2. Each stage consists of $N$ Modulated Group Mamba blocks and a downsampling layer.
3. The feature resolutions of the four stages are $H/4$, $H/8$, $H/16$, and $H/32$, respectively.

### Key Designs

**1. Visual Single Selective Scan (VSSS) Block**
- **Function**: A Mamba-based token-channel mixer composed of a Mamba block and an FFN, each preceded by LayerNorm.
- **Mechanism**: The input $\mathbf{Z}_{in}$ first undergoes token mixing (sequence modeling) via Mamba SSM, and then channel mixing via an FFN, both leveraging residual connections.
- **Design Motivation**: As the basic unit of grouped scanning, each VSSS block only processes a single-direction scan for $C/4$ channels.

**2. Grouped Mamba Operator (Grouped Scanning)**
- **Function**: Divides the input $C$ channels into 4 groups (each with $C/4$ channels), which are flattened into 1D sequences along four directions (left-to-right, right-to-left, top-to-bottom, bottom-to-top), independently processed by a VSSS block, and finally concatenated.
- **Mechanism**: 
$$\mathbf{X}_{GM} = \text{Concat}(\text{VSSS}(\mathbf{X}_{LR}), \text{VSSS}(\mathbf{X}_{RL}), \text{VSSS}(\mathbf{X}_{TB}), \text{VSSS}(\mathbf{X}_{BT}))$$
- **Design Motivation**: Each group only handles $C/4$ channels and a single scanning direction, which substantially reduces the parameter count and computational complexity (parameters reduced by approximately 26%). The four directions collectively cover complete spatial dependencies.

**3. Channel Affinity Modulation (CAM)**
- **Function**: Performs channel recalibration on the output of Grouped Mamba to enhance cross-group channel information exchange.
- **Mechanism**: 
    - Global average pooling $\to$ two FC layers (similar to an SE block) $\to$ Sigmoid to obtain channel weights.
    - $\mathbf{X}_{CAM} = \mathbf{X}_{GM} \cdot \text{Affinity}(\mathbf{X}_{in})$
- **Design Motivation**: The grouping operation restricts cross-channel interactions (as each group only accesses $C/4$ channels). CAM recalibrates the output using affinity weights calculated from the input features, mitigating information isolation.

### Loss & Training

Joint distillation loss:

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{CE}(Z_s, y) + (1-\alpha) \mathcal{L}_{CE}(Z_s, y_t)$$

- $Z_s$: student model logits, $y$: ground-truth label, $y_t$: teacher hard label.
- Teacher model: RegNetY-16G (84M parameters, 82.9% Top-1).
- The distillation objective aims to alleviate training instability in large models (as SiMBA demonstrated that combining an MLP channel mixer with a large Mamba can lead to divergence).
- Label smoothing 0.1, 300 epochs, AdamW optimizer, initial $lr=1e-3$.

## Key Experimental Results

### Main Results (ImageNet-1K Classification)

| Model | Params | FLOPs | Top-1 |
|---|---|---|---|
| Swin-T | 28M | 4.6G | 81.3 |
| VMamba-T | 31M | 4.9G | 82.5 |
| LocalVMamba-T | 26M | 5.7G | 82.7 |
| **GroupMamba-T** | **23M** | **4.5G** | **83.3** |
| VMamba-S | 50M | 8.7G | 83.6 |
| **GroupMamba-S** | **34M** | **7.0G** | **83.9** |
| VMamba-B | 89M | 15.4G | 83.9 |
| **GroupMamba-B** | **57M** | **14G** | **84.5** |

Downstream Tasks:
- COCO Detection (Mask R-CNN): $AP^b = 47.6$, $AP^m = 42.9$ (outperforms Swin-T and ConvNeXt-T)
- ADE20K Semantic Segmentation (UperNet): $mIoU = 48.6$ (SS) / 49.2 (MS)

### Ablation Study

| Configuration | Params | Throughput | Top-1 |
|---|---|---|---|
| 4-D scanning (baseline) | 22M | 803 | 82.30 |
| + Grouped 1-D scanning | 22M | 1125 | 82.20 |
| + CAM | 22M | 1069 | 82.50 |
| + Distillation loss | 23M | 1069 | **83.30** |

### Key Findings

1. **Grouped scanning incurs negligible accuracy loss**: Moving from 4-D full scanning to grouped 1-D scanning drops performance by only 0.1%, while throughput increases by 40% (803 $\to$ 1125).
2. **CAM effectively compensates for channel isolation**: Offers a +0.3% accuracy gain with minimal overhead.
3. **Distillation is key to stabilizing training**: Enhances accuracy by +0.8% and resolves the divergence issue in large SSM models.
4. **Significant parameter efficiency**: GroupMamba-T (23M) outperforms VMamba-T (31M) while using 26% fewer parameters.
5. **GroupMamba-B vs. VMamba-B**: Achieves 36% fewer parameters and a +0.6% accuracy gain.

## Highlights & Insights

- Simple and effective grouped scanning: Leverages the well-established concept of group convolutions to address channel redundancy in SSMs.
- Although the design of CAM is similar to the SE block, it holds unique value in the context of grouped SSMs.
- The distillation-based solution to address SSM training instability is highly generalizable.
- The three variants (T/S/B) establish a comprehensive trade-off spectrum between accuracy and efficiency.

## Limitations & Future Work

- Distillation relies on an external teacher model (RegNetY-16G), which increases training complexity.
- The evaluation is limited to image classification, object detection, and semantic segmentation, without extension to video understanding or sequential tasks.
- The fixed four-group division may not be optimal; adaptive grouping strategies could be explored.
- CAM is essentially an application of the SE block, presenting limited incremental novelty.
- No comparison was made with contemporaneous architectures like Mamba-2.

## Related Work & Insights

- VMamba pioneered 4-way 2D scanning but suffered from computational redundancy, which is effectively resolved in this work via grouping.
- The distillation token concept in DeiT is simplified to a distillation loss in this work.
- Insights: The parameter efficiency and training stability of SSM methods are crucial bottlenecks. The combined strategy of grouping and distillation is highly worth promoting.

## Rating

⭐⭐⭐⭐ — The grouped scanning design is simple and elegant, and the experiments thoroughly cover multiple downstream tasks. Although the combination of distillation and CAM is not entirely unique, it is highly practical and effective, offering significant parameter efficiency advantages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DefMamba: Deformable Visual State Space Model](defmamba_deformable_visual_state_space_model.md)
- [\[CVPR 2025\] 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[ICLR 2026\] Enabling True Global Perception in State Space Models for Visual Tasks](../../ICLR2026/segmentation/enabling_true_global_perception_in_state_space_models_for_visual_tasks.md)

</div>

<!-- RELATED:END -->

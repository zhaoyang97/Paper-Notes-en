---
title: >-
  [Paper Note] GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention
description: >-
  [ECCV2024][Object Detection][Oriented Object Detection] A lightweight Group-wise Rotating and Attention (GRA) module is proposed. By grouping and rotating convolution kernels and applying group-wise spatial attention, it outperforms the previous SOTA method ARC with nearly 50% fewer parameters, achieving new state-of-the-art performance on DOTA-v2.0.
tags:
  - "ECCV2024"
  - "Object Detection"
  - "Oriented Object Detection"
  - "Group-wise Rotating"
  - "Spatial Attention"
  - "Dynamic Neural Networks"
date: 2026-05-08
content_hash: 31c626082b739a2a
---

# GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention

**Conference**: ECCV2024  
**arXiv**: [2403.11127](https://arxiv.org/abs/2403.11127)  
**Code**: [wangjiangshan0725/GRA](https://github.com/wangjiangshan0725/GRA)  
**Area**: Object Detection  
**Keywords**: Oriented Object Detection, Group-wise Rotating, Spatial Attention, Dynamic Neural Networks  
**Institution**: Tsinghua University, DAMO Academy (Alibaba), ModelTC

## TL;DR

A lightweight Group-wise Rotating and Attention (GRA) module is proposed. By grouping and rotating convolution kernels and applying group-wise spatial attention, it outperforms the previous SOTA method ARC with nearly 50% fewer parameters, achieving new state-of-the-art performance on DOTA-v2.0.

## Background & Motivation

Oriented object detection aims to locate and identify objects under arbitrary orientations using oriented bounding boxes, which is widely applied in remote sensing, autonomous driving, text recognition, and other scenarios. Recent studies have advanced this field from various aspects (such as bounding box representations, loss functions, network architectures, and label assignment strategies), but the latest trend focuses on developing rotation-aware detection backbones.

The previous SOTA method, Adaptive Rotated Convolution (ARC), utilizes $m$ independent convolutional kernels, where each kernel is rotated to a different angle to extract features separately, followed by a weighted summation to aggregate the output. Although this method improves performance, it introduces two critical issues:

1. **Excessive Parameter Size**: Using $m$ convolutional kernels leads to an $m$-fold increase in the number of parameters. A standard ResNet-50 has about 23.5M parameters, which swells to 57.2M after integrating ARC ($m=4$). This poses a serious challenge for deployment scenarios with limited storage resources, such as remote sensing equipment.
2. **Imprecise Features**: Mixing features extracted by convolutional kernels at different angles via weighted summation couples target features with noise. Experiments indicate that a convolutional kernel at a specific rotation angle mainly captures object features aligned with its own orientation, while generating unwanted noise for objects with other orientations. Weighted summation consequently leads to a large number of low-confidence detections.

## Core Problem

How to **simultaneously achieve model effectiveness and parameter efficiency** in backbones for oriented object detection? Specifically:

- How to capture fine-grained features from multiple rotational directions without copying multiple full convolutional kernels?
- How to avoid noise interference caused by mixing features of different orientations?

## Method

### Overall Architecture

The GRA module consists of two core components: **Group-wise Rotating** and **Group-wise Attention**. It is designed to replace the $3 \times 3$ convolutions in the last three stages of ResNet in the backbone network.

### 1. Group-wise Rotating

This component consists of three steps:

**Angle Generator**: A lightweight network predicts $n$ rotation angles and scaling factors from the input feature map $\boldsymbol{x} \in \mathbb{R}^{C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$. Specifically: depth-wise separable convolution $\to$ ReLU $\to$ LayerNorm $\to$ global pooling $\to$ two linear layers (each outputting $n$ dimensions), yielding $\{\theta_j\}$ and $\{\lambda_j\}$ respectively.

**Grouping**: The convolutional kernel $\boldsymbol{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$ is uniformly divided into $n$ groups along the $C_{\text{out}}$ dimension, with each group containing $C_{\text{out}}/n$ sub-kernels.

**Rotating**: Each group of kernels is rotated by the corresponding predicted angle $\theta_j$ and multiplied by the scaling factor $\lambda_j$:

$$\widetilde{\boldsymbol{W}}_j = \{\lambda_j \times \text{Rotate}(\boldsymbol{w}_{j,l}, \theta_j)\}$$

Rotation is implemented via bilinear interpolation. All rotated group kernels are concatenated and convolved with the input using standard convolution to generate the output feature $\boldsymbol{y}$.

**Difference from Group Convolution**: Group convolution divides input features into groups along the $C_{\text{in}}$ dimension and conducts independent convolutions; whereas GRA groups convolutional kernels along the $C_{\text{out}}$ dimension for rotations of different angles, and the convolution itself remains standard.

### 2. Group-wise Attention

The convolution output $\boldsymbol{y}$ is naturally split into $n$ groups, where each group $\boldsymbol{y}_j \in \mathbb{R}^{C_{\text{out}}/n \times H_{\text{out}} \times W_{\text{out}}}$ primarily captures object features close to the angle $\theta_j$ but contains noise for other orientations. The process of the group-wise attention mechanism is as follows:

1. Apply Max Pooling and Avg Pooling separately to each group of features and concatenate them to obtain $\boldsymbol{S}_j \in \mathbb{R}^{2 \times H_{\text{out}} \times W_{\text{out}}}$
2. Adjust channels through a convolutional layer $F$ followed by Sigmoid to obtain the attention map $\widetilde{\boldsymbol{S}}_j \in \mathbb{R}^{1 \times H_{\text{out}} \times W_{\text{out}}}$
3. Element-wise multiply by the original feature group: $\widetilde{\boldsymbol{y}}_j = \boldsymbol{y}_j \odot \widetilde{\boldsymbol{S}}_j$

This mechanism enhances the target regions aligned with the corresponding rotation angles in each group while suppressing noise in irrelevant regions.

### Design Advantages

- **Finer-grained Angle Modeling**: $n$ can be set to a larger value ($n=32$ in experiments) to predict more angles with minimal parameter increase
- **Plug-and-play**: It can be seamlessly embedded into any CNN
- **Reusable Pre-trained Weights**: Only standard ResNet pre-trained weights need to be loaded; the ResNet parameters can be frozen, and only the GRA module needs to be trained

## Key Experimental Results

### DOTA-v1.0 (Single-scale training and testing, 12 epochs)

Taking Oriented R-CNN + ResNet-50 as an example:

| Backbone | Params (M) | mAP (%) |
|---------|-----------|---------|
| R50 (baseline) | 41.37 | 75.81 |
| R50_ARC | 75.06 | 77.35 |
| **R50_GRA** | **41.65 (↓43%)** | **77.63** |

GRA consistently outperforms ARC across 6 different detectors while reducing parameters by 43%–46%.

### DOTA-v2.0 (Contains many small objects)

| Method | mAP (%) |
|------|---------|
| Oriented R-CNN + R50_ARC | 55.91 |
| Oriented R-CNN + R50_GRA | 56.63 |
| **R50_GRA (40 epochs)** | **57.95 (SOTA)** |

### HRSC2016

Oriented R-CNN + R50_GRA: mAP 72.59%, outperforming ARC's 72.39%.

### Ablation Study

**Number of Groups** ($n$): As $n$ increases from 2 to 32, mAP improves from 76.82% to 77.63%, while parameters only increase by 0.23M and FLOPs by 1.2G.

**Component Contribution** (Oriented R-CNN, DOTA-v1.0):

| Group-wise Rotating | Scaling Factor $\lambda$ | Group-wise Attention | mAP (%) |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 75.81 |
| ✓ | ✗ | ✗ | 76.73 |
| ✓ | ✓ | ✗ | 77.25 |
| ✓ | ✓ | ✓ | **77.63** |

### Pre-training Strategy

Loading public ResNet pre-trained weights and training only the GRA module yields a 77.39% mAP, which is close to the 77.64% mAP of from-scratch pre-training, bypassing the need for additional ImageNet training resources.

## Highlights & Insights

1. **Extremely Parameter-Efficient**: Under the group-wise rotation mechanism instead of multi-kernel replication, model parameters are reduced by nearly 50%, while performance surpasses ARC.
2. **Group-wise Attention Denoising**: Successfully addresses the critical drawback of noise in ARC's weighted summation, avoiding mutual interference among features of different orientations.
3. **High Flexibility**: A plug-and-play module compatible with various single-stage and two-stage detectors; supports the reuse of public pre-trained weights to reduce training costs.
4. **Detailed Analysis**: Clearly reveals the root cause of feature degradation in ARC's weighted summation through confidence distribution visualization.

## Limitations & Future Work

1. **Kernel Size Limitation**: Currently, only $3 \times 3$ convolution kernels are replaced, and the effectiveness on larger kernels (e.g., $7 \times 7$) remains unverified.
2. **Limited to ResNet Architecture**: Has not been tested on modern architectures such as ConvNeXt or ViT, leaving its generalizability to be validated.
3. **Sample-wise Angle Prediction**: The angle is predicted by fully connected layers after global pooling, operating at the sample level (rather than the spatial level), which leaves room for improvement in fine-grained processing of multi-oriented objects within a single image.
4. **Redundancy in Rotation Angles**: When $n$ is excessively large, groups may predict similar angles, leading to saturation.

## Related Work & Insights

| Method | Mechanism | Parameter Efficiency | Feature Quality |
|------|---------|---------|---------|
| **ARC** | Weighted sum after rotating $m$ full kernels | Poor ($m\times$ parameters) | Weighted sum introduces noise |
| **ReDet** | Rotation-equivariant operations (group theory) | Medium | Maintains rotation equivariance |
| **LSKNet** | Spatial attention with adaptive kernel size selection | Medium | Rotation information is not modeled |
| **GRA (Ours)** | Single-kernel group-wise rotation + group-wise spatial attention | **High** (only +0.28M) | Group-wise attention denoising |

### Insights & Connections

1. **Efficiency of Group-wise Strategies**: Converting "multi-kernel independent operations" into "single-kernel group-wise operations" is a general parameter compression approach that can be extended to other dynamic network scenarios.
2. **Refined Feature Denoising**: Applying attention separately after grouping by function is more targeted than global attention, offering valuable lessons for multi-task learning scenarios.
3. **Complementarity with Rotation-Equivariant Networks**: GRA is a data-driven dynamic rotation, while ReDet provides structural equivariance; whether they can be combined is worth exploring.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combined design of group-wise rotating and group-wise attention is elegant and efficient, solving the two major pain points of ARC.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive experiments across multiple datasets, detectors, and ablation studies, backed by convincing visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem analysis with well-designed motivation charts.
- Value: ⭐⭐⭐⭐ — Performance is enhanced while reducing parameters by half, showing strong practicality; however, generalizability to other architectures is yet to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ElasticFormer: Detecting Objects in HRW Shots via Elastic Computing Vision Transformer](../../CVPR2026/object_detection/elasticformer_detecting_objects_in_hrw_shots_via_elastic_computing_vision_transf.md)
- [\[CVPR 2025\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](../../CVPR2025/object_detection/show_dont_tell_detecting_novel_objects_by_watching_human_videos.md)
- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](../../CVPR2026/object_detection/detecting_unknown_objects_via_energy-based_separation.md)
- [\[ECCV 2024\] Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)
- [\[ECCV 2024\] Visible and Clear: Finding Tiny Objects in Difference Map](visible_and_clear_finding_tiny_objects_in_difference_map.md)

</div>

<!-- RELATED:END -->

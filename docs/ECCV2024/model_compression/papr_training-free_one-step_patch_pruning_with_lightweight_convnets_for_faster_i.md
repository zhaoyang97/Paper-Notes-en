---
title: >-
  [Paper Note] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference
description: >-
  [ECCV 2024][Model Compression][Patch Pruning] This paper proposes PaPr, which leverages convolutional feature maps from lightweight ConvNets to generate Patch Significance Maps (PSMs). It performs **one-step** patch pruning on ViT/ConvNet/hybrid architectures **without any retraining**, achieving significant computation reductions (up to 3.7× FLOPs reduction in video scenarios) with minimal loss in accuracy.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Patch Pruning"
  - "Vision Transformer"
  - "ConvNet"
  - "Token Reduction"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: b2862d03eafe6299
---

# PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference

**Conference**: ECCV 2024  
**arXiv**: [2403.16020](https://arxiv.org/abs/2403.16020)  
**Code**: [GitHub](https://github.com/tanvir-utexas/PaPr)  
**Area**: Model Compression  
**Keywords**: Patch Pruning, Vision Transformer, ConvNet, Token Reduction, Inference Acceleration

## TL;DR

This paper proposes PaPr, which leverages convolutional feature maps from lightweight ConvNets to generate Patch Significance Maps (PSMs). It performs **one-step** patch pruning on ViT/ConvNet/hybrid architectures **without any retraining**, achieving significant computation reductions (up to 3.7× FLOPs reduction in video scenarios) with minimal loss in accuracy.

## Background & Motivation

As vision models evolve from ConvNets to ViTs, the computational complexity continues to climb. High-resolution images contain a large amount of redundant background patches; thus, efficiently removing these redundant patches is key to accelerating inference.

Existing patch pruning methods suffer from three major limitations:

**Requires Additional Training**: Most methods (DynamicViT, A-ViT, SP-ViT) require training additional mask prediction modules at intermediate layers. As baseline models are continuously updated, retraining becomes highly costly.

**Multi-Step Progressive Pruning**: Gradually reducing tokens across multiple intermediate layers of the network means early layers still perform redundant computation, which is particularly detrimental to deep models.

**Architecture Binding**: Many methods rely on specific architectural features (e.g., class tokens, attention maps), making them non-generalizable to ConvNets or hybrid models.

The core insight of PaPr is: **although lightweight ConvNets have low top-1 accuracy, their convolutional layers possess exceptional discriminative region localization capabilities**. Experiments show that as $k$ in top-k increases, the accuracy gap between lightweight ConvNets and large ViTs significantly narrows, indicating that shallow models can already accurately locate discriminative regions in an image. The fully connected (FC) layer is the bottleneck for fine-grained classification in ConvNets, rather than the localization capability of the convolutional layers.

## Method

### Overall Architecture

PaPr adopts a two-stage paradigm: "lightweight ConvNet proposal + large model inference". An ultra-lightweight ConvNet (such as MobileOne-S0, with only 0.27 GFLOPs) is first used to extract the Patch Significance Map (PSM). Then, immediately after the patch extraction stage, low-significance patches are pruned, and the large model only processes the remaining discriminative patches in subsequent computations.

### Key Designs

#### 1. FC Layer Weight Recalibration

**Function**: Extract discriminative region proposals from the final convolutional layer of the ConvNet.

**Mechanism**: Traditional CAM methods rely on class activation weights $w_c^k$ to weight feature maps, which is limited by model classification accuracy. PaPr replaces the FC layer weights with uniform weights $w_c^k = \frac{1}{KC}$, eliminating the interference of the FC layer on region localization:

$$\mathcal{R}(x,y) = \frac{1}{K}\sum_{k} f_k(x,y)$$

where $f_k(x,y)$ is the feature response of the $k$-th convolutional kernel at location $(x,y)$. The resulting discriminative region proposal $\mathcal{R}$ depends only on the spatial perception capability of the convolutional layers and is independent of the model classification accuracy.

**Design Motivation**: The convolutional layers of lightweight ConvNets are inherently good at locating discriminative regions (with top-10 accuracy close to that of large models), whereas the FC layer is the classification bottleneck. By suppressing the FC layer, high-quality region localization can be obtained from an extremely small ConvNet.

#### 2. Patch Significance Map (PSM) Generation and Pruning

**Function**: Map low-resolution region proposals to the target feature map resolution, generating patch-level significance scores.

**Mechanism**: Bicubically upsample $\mathcal{R} \in \mathbb{R}^{h \times w}$ to the target size $(h', w')$ to obtain the PSM $\mathcal{P}$, and then sort to obtain the pruning mask:

$$\mathcal{M} = \text{reshape}(\text{argsort}(\text{flatten}(\mathcal{P})))$$

Keep the top-$z\%$ patches and discard the rest directly.

**Design Motivation**: ConvNets naturally preserve spatial location info (inductive bias). Thus, the PSM can directly establish spatial correspondences with the target model's patches without requiring additional learning.

#### 3. Adaption to Different Architectures

**ViT Integration**: Immediately after patch extraction and positional encoding, and before entering the transformer blocks, low-significance tokens are directly deleted based on the PSM. All subsequent transformer blocks only process the remaining tokens.

**Hierarchical Model Integration (ConvNet/Swin)**: Hierarchical models use window operations, making it impossible to directly delete patches. PaPr performs selective computation targeting **pixel operators** ($1 \times 1$ convolution/linear layers, which account for 96.2% of ConvNext and 63.3% of Swin computation):

- Divide features into foreground $A_f$ and background $A_b$ according to the PSM
- Only perform linear operations on $A_f$, and pad $A_b$ with zeros
- Re-assemble and pass to the next window operator

### Loss & Training

PaPr **requires no training at all**. The proposal ConvNet uses pre-trained weights, and the large model uses its original pre-trained weights, making the entire pipeline plug-and-play. PaPr can also be utilized during the training phase (similar to masked training), achieving 1.6× training acceleration.

## Key Experimental Results

### Main Results

Performance comparison of MAE pre-trained ViT on ImageNet-1k:

| Model | Method | Top-1 Acc (%) | GFLOPs | Throughput (img/s) |
|------|------|:---:|:---:|:---:|
| ViT-B-16 | Baseline | 83.74 | 17.59 | 307 |
| ViT-B-16 | ToMe | 78.82 | 8.78 | 615 |
| ViT-B-16 | TokenFusion | 79.23 | 8.78 | 618 |
| ViT-B-16 | **PaPr (z=0.5)** | **82.40** | 8.98 | 605 |
| ViT-L-16 | Baseline | 85.95 | 61.61 | 91 |
| ViT-L-16 | ToMe | 84.24 | 30.99 | 180 |
| ViT-L-16 | **PaPr (z=0.5)** | **85.06** | 30.83 | 183 |
| ViT-H-16 | Baseline | 86.89 | 167.4 | 36 |
| ViT-H-16 | ToMe | 85.48 | 82.53 | 72 |
| ViT-H-16 | **PaPr (z=0.5)** | **86.40** | 83.04 | 71 |

The advantage of PaPr is particularly pronounced on MAE models (achieving **4.5%** higher accuracy than ToMe on ViT-B), as MAE pre-training inherently uses masking, naturally adapting to masked inference.

### Ablation Study

Impact of different ConvNet proposal models on final accuracy (z=0.5):

| Proposal Model | GFLOPs | ViT-B Acc (%) | ViT-L Acc (%) |
|----------|:---:|:---:|:---:|
| MobileOne-S0 | 0.27 | 82.24 | 84.06 |
| MobileOne-S2 | 1.35 | 82.28 | 84.16 |
| ResNet-18 | 1.81 | 81.10 | 83.84 |
| ResNet-50 | 4.09 | 82.33 | 84.09 |
| ResNet-152 | 11.51 | 82.51 | 84.08 |

The accuracy variance across different ConvNets is only **0.3%** (ViT-B), demonstrating the strong model-agnostic nature of PSMs. Although the computational cost of MobileOne-S0 is only $1/42$ of ResNet-152, their localization accuracies are almost identical.

Ablation on video recognition (Kinetics-400, ViT-B MAE):

| Method | Top-1 Acc (%) | GFLOPs |
|------|------|:---:|
| Baseline | 81.21 | 180 |
| PaPr (z=0.45) | 81.18 | 76 |
| PaPr (z=0.35) | 80.15 | 59 |

In the video scenario, about 70% of the patches are pruned, reducing FLOPs to 1/3 of the original, with only a 0.8% loss in accuracy.

### Key Findings

1. Combining PaPr with ToMe achieves **Pareto optimality**, improving accuracy by 4.5% over ToMe alone on ViT-B AugReg.
2. It is equally effective on hierarchical models like ConvNeXt and Swin, and can easily adapt to model updates without training.
3. In videos, PaPr can suppress background patches in redundant frames through spatio-temporal understanding, achieving up to 3.7× acceleration.

## Highlights & Insights

- **Highly Simple Core Insight**: Lightweight ConvNets are not poor at localization; they are poor at classification. By suppressing the FC layer, the localization potential of convolutional layers is unlocked.
- **Strong Practicality**: Training-free, architecture-agnostic, and supporting batch processing, it can be directly applied to ViT, ConvNet, and hybrid models.
- **Natural Synergy with MAE**: MAE pre-training inherently masks images. PaPr provides a "meaningful mask" to replace the random mask.

## Limitations & Future Work

1. Only validated on discriminative tasks (classification, video recognition); dense prediction tasks (detection/segmentation) remain unexplored.
2. The PSM is static (computed once per image) and does not utilize dynamic information from the intermediate layers of the large model.
3. For datasets with uneven foreground/background distributions (e.g., medical images), a fixed ratio $z$ may not be optimal.

## Related Work & Insights

- **ToMe (ICLR 2023)**: Token merging; PaPr can serve as a front-end step to enhance its performance.
- **DynamicViT (NeurIPS 2021)**: Trains an additional predictor for token sparsification, which requires retraining.
- **CAM Methods**: Rely on gradients and final predictions, and do not support batch processing; PaPr replaces this with a simple mean.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The decoupled approach of "lightweight ConvNet localization + large model inference" is novel and intuitively reasonable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers various architectures (ViT/ConvNet/Swin), image/video dual-tasks, and multiple pre-training methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic, rich diagrams, and practical PyTorch pseudo-code.
- **Value**: ⭐⭐⭐⭐⭐ — A truly plug-and-play method with high value for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](../../NeurIPS2025/model_compression/one-step_diffusion-based_image_compression_with_semantic_distillation.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](isomorphic_pruning_for_vision_models.md)
- [\[ECCV 2024\] Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search](auto-das_automated_proxy_discovery_for_training-free_distillation-aware_architec.md)
- [\[ICLR 2026\] Faster Vision Transformers with Adaptive Patches](../../ICLR2026/model_compression/faster_vision_transformers_with_adaptive_patches.md)

</div>

<!-- RELATED:END -->

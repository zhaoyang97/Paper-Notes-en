---
title: >-
  [Paper Note] "A Regularization-Guided Equivariant Approach for Image Restoration"
description: >-
  [CVPR2025][Image Restoration][Paper Note] Academic paper note for "A Regularization-Guided Equivariant Approach for Image Restoration".
tags:
  - CVPR2025
  - Image Restoration
date: 2025-05-26
content_hash: a46d5662c7319416
---
# EQ-Reg: A Regularization-Guided Equivariant Approach for Image Restoration

**Authors**: Lu Yu, Jiahao Li, Yutong Zhang et al.  
**Affiliations**: Xi'an Jiaotong University / Macau UST / Pengcheng Lab  
**Conference**: CVPR 2025  
**Code**: [https://github.com/yulu919/EQ-REG](https://github.com/yulu919/EQ-REG)  

## Background & Motivation

Image restoration is a classic problem in low-level vision, encompassing tasks such as denoising, deraining, deblurring, super-resolution, and CT artifact removal. Although modern deep learning methods have achieved remarkable progress, they still suffer from the following fundamental limitations:

**Insufficient Generalization**: Models perform exceptionally well on the training distribution but suffer dramatic performance drops on out-of-distribution (OOD) data. For instance, a deraining model trained on synthetic rain patterns may fail in real-world rain scenarios.

**Lack of Equivariance**: An ideal image restoration model should satisfy geometric equivariance—meaning that when a rotation or flipping transformation is applied to the input, the output should also transform accordingly. However, standard CNNs and Transformers lack this intrinsic constraint.

**Inadequate Regularization**: Existing regularization methods (e.g., dropout, weight decay) primarily target overfitting issues and fail to constrain model behavior from the perspective of geometric symmetry.

**Isotropic Assumption in Feature Space**: The feature representations in the intermediate layers of the network lack explicit symmetry constraints, leading to insufficient coupling of features across different channels.

Mathematical definition of equivariance: For a function $f$ and a transformation $T$, if $f(T(x)) = T(f(x))$, then $f$ is equivariant to $T$. EQ-Reg aims to inject equivariance constraints into standard networks through regularization.

## Method

### Overall Architecture

EQ-Reg is a plug-and-play regularization module that can be applied to any image restoration network. The core idea is to guide the network to learn equivariant feature representations by imposing equivariance losses on various layers of the network during training.

### Equivariant Transformation Groups

EQ-Reg considers two types of transformations:

| Transformation Type | Mathematical Description | Group Structure |
|----------|----------|--------|
| Rotational Transformation | $R_{\theta}: x \mapsto R(\theta) \cdot x$, $\theta \in \{0°, 90°, 180°, 270°\}$ | Cyclic Group $C_4$ |
| Channel Cyclic Shift | $\sigma_k: (c_1,...,c_n) \mapsto (c_{k+1},...,c_n,c_1,...,c_k)$ | Cyclic Group $C_n$ |

### Layer-wise Equivariance Loss

For the feature map $f_l$ at the $l$-th layer of the network, the equivariance loss is defined as:

$$\mathcal{L}_{eq}^{(l)} = \mathbb{E}_{x, T} \left[ \| f_l(T(x)) - T(f_l(x)) \|_2^2 \right]$$

where $T$ is uniformly sampled from the transformation group.

### Total Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \sum_{l=1}^{L} w_l \cdot \mathcal{L}_{eq}^{(l)}$$

- $\mathcal{L}_{task}$ is the task-specific loss (e.g., L1 or L2)
- $\lambda$ is the global regularization weight
- $w_l$ represents the weight coefficient for each layer, where shallower layers are typically assigned larger weights

### Channel Cyclic Shift Equivariance

In addition to spatial rotation, EQ-Reg imposes a cyclic shift equivariance constraint on the channel dimension:

$$\mathcal{L}_{ch}^{(l)} = \mathbb{E}_{x, k} \left[ \| f_l(\sigma_k(x)) - \sigma_k(f_l(x)) \|_2^2 \right]$$

This encourages the network to learn more uniform and decoupled feature representations across channels.

### Implementation Details

- The equivariance loss is calculated only during the training phase, incurring no extra overhead during inference.
- A transformation is randomly sampled per mini-batch to enforce the equivariance constraint.
- Gradient clipping is employed to prevent the equivariance loss from dominating the optimization during the early stages of training.

## Key Experimental Results

### CT Artifact Removal

| Method | PSNR ↑ | SSIM ↑ |
|------|--------|--------|
| FBPConvNet | 38.42 | 0.9621 |
| RED-CNN | 39.15 | 0.9673 |
| DuDoNet | 40.28 | 0.9712 |
| Baseline (w/o EQ-Reg) | 41.03 | 0.9738 |
| **+ EQ-Reg (ours)** | **42.07** | **0.9781** |

### Image Deraining (Rain100L)

| Method | PSNR ↑ | SSIM ↑ |
|------|--------|--------|
| DerainNet | 32.16 | 0.9363 |
| PReNet | 37.10 | 0.9799 |
| MPRNet | 39.47 | 0.9825 |
| Baseline (w/o EQ-Reg) | 39.68 | 0.9831 |
| **+ EQ-Reg (ours)** | **40.33** | **0.9856** |

### Auxiliary Validation on Image Classification (CIFAR-100)

| Method | Top-1 Accuracy |
|------|---------------|
| ResNet-50 Baseline | 55.21% |
| + Group Equivariant CNN | 56.03% |
| + Augerino | 56.78% |
| **+ EQ-Reg (ours)** | **57.56%** |

### Ablation Study

| Configuration | CT PSNR | Rain100L PSNR |
|------|---------|---------------|
| Baseline | 41.03 | 39.68 |
| + Rotation equivariance only | 41.52 | 39.95 |
| + Channel shift equivariance only | 41.28 | 39.82 |
| + Both (EQ-Reg) | **42.07** | **40.33** |

## Highlights & Insights

1. **Layer-wise Equivariance Regularization**: It is proposed for the first time to impose equivariance constraints independently on each layer of the network, rather than solely on the input-output mapping.
2. **Channel Cyclic Shift Equivariance**: The equivariance constraint is innovatively extended from the spatial dimension to the channel dimension.
3. **Plug-and-Play Design**: As a regularization loss term, it can be seamlessly integrated into any image restoration network with zero additional computation during inference.
4. **Cross-Task Effectiveness**: Consistent improvements are achieved across multiple tasks, including CT artifact removal, deraining, denoising, and classification.

## Theoretical Analysis

The authors analyze the regularization effect of EQ-Reg from the perspective of group theory:

- The equivariance constraint is equivalent to restricting the effective dimension of the function space, thereby reducing the Rademacher complexity of the model.
- Layer-wise constraints are stronger than constraining only the input-output mapping, preventing intermediate layers from learning features that break symmetry.

## Limitations & Future Work

- Currently, only discrete transformations (the $C_4$ rotation group) are considered, without extension to the continuous rotation group $SO(2)$.
- The physical meaning of channel cyclic shift equivariance is less intuitive than that of spatial rotation.
- For tasks that inherently possess strong directional characteristics (such as text recognition), rotational equivariance constraints might have a negative impact.
- Training time increases by approximately 15-20% due to the computation of the equivariance loss.

## Related Work

- Group Equivariant CNN (G-CNN): Achieves strict equivariance through group convolutions.
- E(2)-Steerable CNN: A continuous rotation equivariant convolutional network.
- Augerino: Approximates equivariance by learning data augmentation policies.
- SwinIR / Restormer: Transformer-based image restoration methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](../../NeurIPS2025/image_restoration/latent_harmony_synergistic_unified_uhd_image_restoration_with_pre-trained_diffus.md)
- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[CVPR 2025\] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach](pixel-level_and_semantic-level_adjustable_super-resolution_a_dual-lora_approach.md)
- [\[ICLR 2026\] Efficient Degradation-agnostic Image Restoration via Channel-Wise Functional Decomposition and Manifold Regularization](../../ICLR2026/image_restoration/efficient_degradation-agnostic_image_restoration_via_channel-wise_functional_dec.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](../../ECCV2024/image_restoration/seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)

</div>

<!-- RELATED:END -->

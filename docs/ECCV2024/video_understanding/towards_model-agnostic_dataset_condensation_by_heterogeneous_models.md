---
title: >-
  [Paper Note] Towards Model-Agnostic Dataset Condensation by Heterogeneous Models
description: >-
  [ECCV 2024][Video Understanding][Dataset Condensation] The Heterogeneous Model Dataset Condensation (HMDC) method is proposed. By simultaneously using two structurally different models (such as ConvNet and ViT) for dataset condensation, and designing a Gradient Balancing Module (GBM) and a Mutual Distillation (MD) mechanism, it generates condensed images that are universally applicable to various models, addressing the limitation where conventional methods overfit to a single…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Dataset Condensation"
  - "Model-Agnostic"
  - "Heterogeneous Models"
  - "Gradient Balancing"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: ffa950de45e759ba
---

# Towards Model-Agnostic Dataset Condensation by Heterogeneous Models

**Conference**: ECCV 2024  
**arXiv**: [2409.14538](https://arxiv.org/abs/2409.14538)  
**Code**: [Available](https://github.com/KHU-AGI/HMDC)  
**Area**: Video Understanding  
**Keywords**: Dataset Condensation, Model-Agnostic, Heterogeneous Models, Gradient Balancing, Knowledge Distillation

## TL;DR

The Heterogeneous Model Dataset Condensation (HMDC) method is proposed. By simultaneously using two structurally different models (such as ConvNet and ViT) for dataset condensation, and designing a Gradient Balancing Module (GBM) and a Mutual Distillation (MD) mechanism, it generates condensed images that are universally applicable to various models, addressing the limitation where conventional methods overfit to a single model.

## Background & Motivation

Dataset Condensation (DC) aims to generate a very small number of synthetic images from large-scale training data, such that models trained on these images can achieve performance close to that of the full dataset. However, **existing methods suffer from a severe model-dependency issue**:

- Traditional methods almost exclusively use a 3-layer ConvNet for both condensation and evaluation (ConvNet-to-ConvNet).
- The condensed images exhibit excellent performance on ConvNets, but when transferred to widely used models such as ResNet and ViT, their performance drops drastically.
- This implies that the condensed images are **over-condensed** to a specific model.

As shown, methods like IDC and DREAM far outperform random sampling on ConvNet, but fall short of even random sampling on ResNet and ViT. This severely limits the practical value of dataset condensation—introducing a new model requires regenerating the condensed data, which defeats the original purpose of reducing storage and computation.

## Method

### Overall Architecture

HMDC simultaneously uses two heterogeneous models $f_{\theta_1}$ (e.g., ConvNet) and $f_{\theta_2}$ (e.g., ViT-tiny) to perform gradient matching-based dataset condensation. The core mechanism is to extract more generalized features through cross-constraints of two completely different structures, thereby avoiding overfitting to a single model.

The framework faces two core challenges:

| Challenge | Cause | Solution |
|------|------|---------|
| Gradient Magnitude Discrepancy | Models with different architectures/depths generate significantly different gradient magnitudes on synthetic images | Gradient Balancing Module (GBM) |
| Semantic Distance | Feature semantics learned by the two models gradually diverge during training | Mutual Distillation + Spatial-Semantic Decomposition (MD + SSD) |

### Key Designs

**Gradient Balancing Module (GBM)**: GBM maintains a gradient magnitude accumulator $\mathcal{A}$ for each optimization target, recording the historical maximum of each loss gradient. It scales each loss by the normalized reciprocal of the accumulator to ensure that different models exert comparable influence on the synthetic images:

$$\mathcal{L}_{\text{target}} = \textbf{L} \cdot \min(\mathcal{A}) \mathcal{A}^{\mathrm{R}}$$

where $\mathcal{A}^{\mathrm{R}}$ is the element-wise reciprocal of the accumulator. In practice, sampling is performed every 10 steps to reduce computational overhead.

**Spatial-Semantic Decomposition (SSD)**: Since features from different models vary in dimensions, spatial sizes, and layers, a unified representation is required for distillation. SSD decomposes features into:
- **Semantic features**: Global information representing the entire image (e.g., CLS token of ViT / global average pooling output of CNN).
- **Spatial features**: Local information at each spatial location (e.g., image tokens of ViT / feature maps of CNN).

Spatial sizes are aligned via bilinear interpolation, feature dimensions are aligned through learnable affine transformations, and layer counts are aligned using a softmax-based layer matching matrix.

**Mutual Distillation (MD)**: During training, an MSE constraint is applied to the SSD-aligned features of both models to reduce semantic distance:

$$\mathcal{L}_{\text{MD}}(\mathbf{x}) = \text{MSE}(\text{SSD}(f_{\theta_1}(\mathbf{x}), f_{\theta_2}(\mathbf{x})))$$

The mutual distillation loss is used for both model training and synthetic image optimization (by matching its gradients).

### Loss & Training

The total loss contains 3 optimization targets, which are automatically balanced by GBM:

1. $\mathcal{L}^1$: Gradient matching loss of Model 1 (gradient discrepancy between real and synthetic images on $f_{\theta_1}$)
2. $\mathcal{L}^2$: Gradient matching loss of Model 2
3. $\text{MSE}(\nabla \mathcal{L}_{\text{MD}}(\mathbf{x}^t), \nabla \mathcal{L}_{\text{MD}}(\mathbf{x}^s))$: Gradient matching for mutual distillation

Training details:
- Heterogeneous model pair: ConvNet + ViT-tiny
- Number of iterations: 100 outer loops $\times$ 100 inner loops
- Model learning rate: 0.001, affine layer/layer matching matrix learning rate: 0.01
- SGD optimizer, batch size: 128

## Key Experimental Results

### Main Results

**Test accuracy of different models on CIFAR-10 (IPC=10)**:

| Method | ConvNet | ResNet18 | ResNet50 | ResNet101 | ViT-tiny | ViT-small | ViT-base | Average |
|------|---------|----------|----------|-----------|----------|-----------|----------|------|
| Random | 36.45 | 56.59 | 69.55 | 82.03 | 59.41 | 90.11 | 81.26 | 67.91 |
| IDC | 46.67 | 56.35 | 67.53 | 60.94 | 50.23 | 68.67 | 64.32 | 59.24 |
| DREAM | 49.23 | 57.97 | 67.85 | 64.12 | 55.40 | 71.77 | 68.88 | 62.18 |
| **HMDC** | **47.54** | **69.75** | **77.99** | **82.25** | **73.60** | **89.02** | **85.58** | **75.10** |

### Ablation Study

**Contribution of each component (CIFAR-10, IPC=10)**:

| GBM | MD | ConvNet | ResNet18 | ResNet50 | ViT-tiny | ViT-small | Average |
|-----|-------|---------|----------|----------|----------|-----------|------|
| ✗ | ✗ | 49.23 | 57.97 | 67.85 | 55.40 | 71.77 | 62.18 |
| ✗ | ✗ (Two models without components) | 46.42 | 72.11 | 76.46 | 74.09 | 90.01 | 74.09 |
| ✓ | ✗ | 47.43 | 71.62 | 76.92 | 73.67 | 86.62 | 73.21 |
| ✗ | ✓ | 47.30 | 72.01 | 78.05 | 70.96 | 86.25 | 73.40 |
| ✓ | ✓ | 47.54 | 69.75 | 77.99 | 73.60 | 89.02 | 75.10 |

### Key Findings

1. **Existing methods exhibit surprising model dependency**: Methods like IDC and DREAM underperform even random sampling on ResNet/ViT, indicating that synthetic images are severely overfitted to ConvNet.
2. **HMDC shows the greatest advantage at IPC=1**: In extremely low-sample scenarios, the value of generalized features is more pronounced.
3. **GBM and MD exhibit synergistic effects**: When used individually, they each show bias (GBM favors smaller models, while MD favors larger models); combining them achieves a balance.
4. **HMDC requires only 100 iterations**: Whereas other methods typically require 1,200–20,000 iterations, yielding high computational efficiency.
5. **Slight performance trade-off on ConvNet**: In exchange for cross-model generalization, HMDC performs slightly below methods specifically optimized for ConvNet.

## Highlights & Insights

- **Uncovering a major blind spot in dataset condensation**: The existing ConvNet-to-ConvNet evaluation paradigm conceals the model-dependency issue.
- **Complementary nature of heterogeneous models**: Two structurally distinct models mutually constrain each other, allowing the extracted "consensus" features to be more generalizable.
- **Gradient balancing without hyperparameter search**: GBM adaptively adjusts the loss weights using historical gradients, eliminating the need for manual tuning.
- **Generality of SSD**: Spatial-Semantic Decomposition can be adapted to any model architecture with spatial features (not limited to CNNs and ViTs).

## Limitations & Future Work

1. Sufficient experiments are only conducted on CIFAR-10; verification on larger-scale datasets (such as ImageNet) is needed.
2. Currently limited to two models; combinations of three or more heterogeneous models can be explored.
3. The use of pre-trained models (ImageNet-1K) makes the evaluation of large models somewhat unfair.
4. There is a lack of visual analysis and interpretability for the synthetic images.
5. Integrating HMDC with other condensation paradigms, like trajectory matching, represents a promising future direction.

## Related Work & Insights

- Core paradigms of dataset condensation include gradient matching (DC, IDC, DREAM) and distribution matching (CAFE, IDM). This work adds heterogeneous model constraints based on gradient matching.
- Knowledge distillation is repurposed from "large model teaching small model" to "reducing the semantic distance between two models."
- Gradient balancing methods in multi-task learning inspired the design of GBM.

## Rating

| Metric | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | First to propose model-agnostic dataset condensation; the heterogeneous model strategy is novel. |
| Technical Depth | 4 | The combined design of GBM + SSD + MD is reasonable and effectively addresses the practical issue. |
| Experimental Thoroughness | 3.5 | Experiments are only on CIFAR-10; although model coverage is broad, the dataset is single. |
| Practicality | 4 | Resolves the core practicality issue of dataset condensation. |
| Overall | 4 | A significant contribution to the field of dataset condensation, worthy of attention. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos](actionswitch_class-agnostic_detection_of_simultaneous_actions_in_streaming_video.md)
- [\[ECCV 2024\] SemTrack: A Large-Scale Dataset for Semantic Tracking in the Wild](semtrack_a_large-scale_dataset_for_semantic_tracking_in_the_wild.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](../../CVPR2025/video_understanding/heterogeneous_skeleton-based_action_representation_learning.md)
- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)

</div>

<!-- RELATED:END -->

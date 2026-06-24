---
title: >-
  [Paper Note] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression
description: >-
  [ICML 2025][Model Compression][Network Compression] The FGFP framework is proposed, which integrates fractional calculus with Gaussian functions to construct a fractional Gaussian filter (FGF), requiring only 7 parameters per kernel. In conjunction with adaptive unstructured pruning (AUP), it achieves an 85.2% model compression rate with only a 1.52% accuracy drop for ResNet-20 on CIFAR-10, and a 69.1% compression rate with a 1.63% accuracy drop for ResNet-50 on ImageNet.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Network Compression"
  - "Fractional Gaussian Filter"
  - "Unstructured Pruning"
  - "CNN"
  - "Edge Deployment"
date: 2026-05-08
content_hash: 52089f9ed2dba217
---

# FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression

**Conference**: ICML 2025  
**arXiv**: [2507.22527](https://arxiv.org/abs/2507.22527)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: Network Compression, Fractional Gaussian Filter, Unstructured Pruning, CNN, Edge Deployment

## TL;DR

The FGFP framework is proposed, which integrates fractional calculus with Gaussian functions to construct a fractional Gaussian filter (FGF), requiring only 7 parameters per kernel. In conjunction with adaptive unstructured pruning (AUP), it achieves an 85.2% model compression rate with only a 1.52% accuracy drop for ResNet-20 on CIFAR-10, and a 69.1% compression rate with a 1.63% accuracy drop for ResNet-50 on ImageNet.

## Background & Motivation

### 1. Challenges of Edge Deployment

Deep neural networks exhibit outstanding performance in tasks such as image classification and detection, but their massive parameter size and computational demand make them difficult to deploy on resource-constrained devices like mobile phones and embedded systems.

### 2. Limitations of Prior Work

- **Unstructured Pruning**: Maintains accuracy but remains computationally complex (limited by the efficiency of sparse matrix operations).
- **Structured Pruning**: Directly removes channels/kernels, which is computationally efficient but often leads to significant accuracy loss.
- **Low-Rank Decomposition**: Decomposes convolutional kernels into low-rank forms, but the resulting structures may lack flexibility.

### 3. Core Idea

Classical filters in computer vision, such as Gaussian and Laplacian filters, can be extended to more flexible forms using fractional-order derivatives. Integrating these classical filters with CNNs and representing convolutional kernels with minimal parameters (7 per kernel) yields extremely high compression rates.

## Method

### Overall Architecture

1. Select the largest layers in the pretrained model and convert their convolutional weights into FGF representations.
2. Repeat this process layer by layer until all selected layers are converted.
3. Perform adaptive unstructured pruning (AUP) on the remaining unconverted layers.
4. Obtain the final sparse FGF model.

### Key Designs

#### 1. Fractional Gaussian Filter (FGF)

- **Mechanism**: Approximates fractional differential equations using the Grünwald-Letnikov fractional derivative and applies it to Gaussian functions.
- **Parameter Efficiency**: Each $3\times3$ convolutional kernel requires only 7 learnable parameters (the fractional order $\alpha$, Gaussian parameter $\sigma$, etc.).
- **Two Variants**:
    - **CA-FGF (Channel Attention FGF)**: Shares filter parameters across channels + employs a channel attention mechanism to compensate for accuracy.
    - **3D-FGF (Three-Dimensional FGF)**: Applies fractional Gaussian simultaneously in both spatial and channel dimensions.

#### 2. Grünwald-Letnikov Approximation

- Simplifies the fractional differential equation into a three-term polynomial, significantly reducing computational complexity.
- Renders both forward and backward propagation of FGF highly efficient and feasible.

#### 3. Adaptive Unstructured Pruning (AUP)

- Performs unstructured pruning on the dense layers remaining after FGF conversion.
- Adaptively determines the pruning rate of each layer to maximize the overall compression rate.

## Key Experimental Results

### Main Results: CIFAR-10 and ImageNet

| Dataset | Model | Method | Accuracy Drop | Compression Rate |
|--------|------|------|---------|--------|
| CIFAR-10 | ResNet-20 | FGFP | 1.52% | 85.2% |
| CIFAR-10 | ResNet-20 | Structured Pruning SOTA | ~3% | 50-60% |
| CIFAR-10 | ResNet-20 | Low-rank Decomposition | ~2% | 60-70% |
| ImageNet | ResNet-50 | FGFP | 1.63% | 69.1% |
| ImageNet | ResNet-50 | Hybrid Compression SOTA | ~2% | 50-60% |

### Ablation Study

| Configuration | CIFAR-10 Accuracy Drop | Compression Rate | Description |
|------|-----------------|--------|------|
| CA-FGF only | ~1.8% | 75% | Filter replacement only |
| 3D-FGF only | ~2.0% | 70% | 3D variant |
| FGF + AUP (FGFP) | 1.52% | 85.2% | Full framework |
| AUP only | ~3.5% | 60% | Pruning only |

### Key Findings

- The 7-parameter/kernel representation of FGF is sufficient to maintain high accuracy, proving the effectiveness of the inductive bias of classical filters.
- CA-FGF compensates for the accuracy loss of cross-channel sharing using channel attention.
- AUP and FGF are complementary: FGF compresses large layers, while AUP handles small layers.

## Highlights & Insights

- **Elegant Integration of Classical CV and Deep Learning**: Extends traditional Gaussian filters through fractional calculus to act as efficient parameterizations of CNN kernels.
- **Extreme Parameter Efficiency**: 7 parameters per kernel is one of the lowest parameterization schemes known for convolutional kernels.
- **Solid Mathematical Foundation**: Based on the Grünwald-Letnikov fractional derivative theory with clear mathematical derivations.
- **Practicality of Hybrid Strategy**: The combination of FGF + AUP covers different types of layers effectively.

## Limitations & Future Work

- Cache truncation occurs in the latter part of the methodology, preventing full ablation data from being obtained.
- Verified only on CNNs (ResNet); applicability to Transformer architectures has not been explored.
- The actual hardware acceleration effect of unstructured pruning depends heavily on sparse computing support.
- The 7-parameter constraint of FGF may lack flexibility for certain complex features.
- Future work could explore extending fractional-order concepts to the compression of attention mechanisms.

## Related Work & Insights

- **vs. Structured Pruning**: Directly removes channels; simple but causes significant accuracy loss. FGFP uses parametric replacement to achieve higher compression rates.
- **vs. Low-Rank Decomposition**: SVD and other methods decompose weight matrices, whereas FGFP employs physically-inspired filter parameterization.
- **vs. Zamora et al. (2021)**: First integrated fractional filters with CNNs; FGFP further optimizes this and incorporates AUP.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The CNN compression scheme combining fractional calculus and Gaussian filtering is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully validated on CIFAR-10 and ImageNet, though some details are limited due to cache truncation.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations and intuitive framework diagrams.
- Value: ⭐⭐⭐⭐ Holds practical value for the compression of CNNs in edge deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)
- [\[NeurIPS 2025\] QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](../../NeurIPS2025/model_compression/quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](../../NeurIPS2025/model_compression/spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[CVPR 2026\] Neural Differentiation in Deep Networks: A Theoretical Framework for Expressivity and Representational Diversity](../../CVPR2026/model_compression/neural_differentiation_in_deep_networks_a_theoretical_framework_for_expressivity.md)

</div>

<!-- RELATED:END -->

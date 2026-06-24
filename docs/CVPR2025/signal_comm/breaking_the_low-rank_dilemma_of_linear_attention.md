---
title: >-
  [Paper Note] Breaking the Low-Rank Dilemma of Linear Attention
description: >-
  [CVPR 2025][Signal & Communication][Linear Attention] This paper theoretically reveals that the fundamental cause of linear attention's performance lagging behind Softmax attention is the low-rank bottleneck of output features. It proposes Rank-Augmented Linear Attention (RALA), which utilizes two complementary strategies—enhancing KV buffer rank and output feature rank—to match or even surpass the performance of Softmax attention while maintaining linear complexity.
tags:
  - "CVPR 2025"
  - "Signal & Communication"
  - "Linear Attention"
  - "Rank Augmentation"
  - "Vision Transformer"
  - "Efficient Attention"
  - "RALA"
date: 2026-05-08
content_hash: 298317c9b394ecbf
---

# Breaking the Low-Rank Dilemma of Linear Attention

**Conference**: CVPR 2025  
**arXiv**: [2411.07635](https://arxiv.org/abs/2411.07635)  
**Code**: [https://github.com/qhfan/RALA](https://github.com/qhfan/RALA)  
**Authors**: Qihang Fan, Huaibo Huang, Ran He  
**Institution**: CRIPAC, Institute of Automation, CAS; University of CAS  
**Area**: Signal Processing and Communications  
**Keywords**: Linear Attention, Rank Augmentation, Vision Transformer, Efficient Attention, RALA

## TL;DR
This paper theoretically reveals that the fundamental cause of linear attention's performance lagging behind Softmax attention is the low-rank bottleneck of output features. It proposes Rank-Augmented Linear Attention (RALA), which utilizes two complementary strategies—enhancing KV buffer rank and output feature rank—to match or even surpass the performance of Softmax attention while maintaining linear complexity.

## Background & Motivation
**Background**: Softmax attention in Transformers suffers from quadratic complexity $O(N^2)$, incurring massive computational overhead when processing high-resolution visual tasks. Linear attention reduces the complexity to $O(N)$ through kernel function approximation, standing as an important research direction for efficient Transformers. In recent years, linear attention vision models such as EfficientViT, FLatten Transformer, and CastlingViT have emerged.

**Limitations of Prior Work**: Despite its theoretical efficiency advantages, the practical performance of linear attention is typically significantly lower than that of Softmax attention. Taking DeiT-T as an example, typical linear attention only achieves around 72% Top-1 accuracy, whereas Softmax attention achieves 72.2%. The gap is even wider on downstream tasks such as detection and segmentation. Existing methods (e.g., the focused function of FLatten, the hybrid strategy of CastlingViT) only alleviate the issue but fail to fundamentally explain and resolve this performance gap.

**Key Challenge**: The efficiency of linear attention stems from decomposing $\text{softmax}(QK^T)V$ into $\kappa(Q)(\kappa(K)^TV)$, which avoids the explicit computation of the $N \times N$ attention matrix. However, this decomposition restricts the rank of the KV buffer matrix $B = \sum \kappa(K_j)^T V_j$ to $\min(d_k, d_v)$ (typically very small), thereby limiting the upper bound of the rank of the output features $Y = \kappa(Q) B$. A low-rank output implies restricted expressiveness of features, making it difficult to model complex spatial relationships.

**Key Insight**: Theoretical analysis is conducted on linear attention from the perspective of matrix rank. It is discovered that the upper bound of the rank of linear attention output $Y$ is $\text{rank}(Y) \leq \min(N, d_k, d_v)$, whereas for Softmax attention, it is $\min(N, d_v)$. When $d_k \ll N$ (typically $d_k = 32$ or 64, while $N$ can reach thousands), the rank upper bound of linear attention is significantly lower than that of Softmax. This theoretical finding precisely explains the root cause of the performance gap.

**Core Idea**: The rank is enhanced from two dimensions: (1) Augmenting the rank of the KV buffer $B$: introducing context-aware weights $\alpha_j$ to perform a weighted sum over KV terms so that the rank of $B$ is no longer restricted by $d_k$; (2) Augmenting the rank of output features: multiplying the input features with the attention output via the Hadamard product to raise the upper bound of the output rank, leveraging the property $\text{rank}(A \odot B) \leq r \cdot s$.

## Method

### Overall Architecture
RALA is a plug-and-play linear attention enhancement method containing two complementary rank augmentation strategies. Built upon this, RAVLT (Rank-Augmented Vision Linear Transformer) adopts a 4-stage hierarchical architecture, using conditional position encoding (CPE, 3×3 depthwise convolution) in each stage. The model series covers four scales from Tiny to Large.

### Key Designs

1. **Augment KV Buffer Rank**:

    - **Function**: Elevate the rank of the KV buffer matrix $B$, liberating it from the limitation of $d_k$.
    - **Mechanism**: Compute a global query $Q_g = \text{mean}(Q_i)$, and then compute Softmax attention weights $\alpha_j = \text{softmax}(Q_g^T \kappa(K_j))$ for all keys. The KV buffer is modified from a simple summation $B = \sum \kappa(K_j)^T V_j$ to a weighted summation $B = \sum \alpha_j \kappa(K_j)^T V_j$. Because $\alpha_j$ introduces degrees of freedom related to $N$, the upper bound of the rank of $B$ is elevated from $\min(d_k, d_v)$ to $\min(N, d_k \cdot d_v)$.
    - **Design Motivation**: The computational overhead of the global query $Q_g$ is extremely small (taking the mean of $Q$ and then performing a single Softmax of size $N$), maintaining the overall $O(N)$ complexity. This context-aware weighting assigns different weights to KV pairs at different positions, breaking the rank limitation caused by simple summation.
    - **Mathematical Guarantee**: $\text{rank}(\sum \alpha_j \kappa(K_j)^T V_j) \leq \sum \text{rank}(\alpha_j \kappa(K_j)^T V_j) \leq N \cdot \min(d_k, d_v)$

2. **Augment Output Features Rank**:

    - **Function**: Further elevate the rank of the final output features via the Hadamard product.
    - **Mechanism**: Perform a Hadamard product between the attention output and a non-linear transformation of the input features: $Y_i = \phi(X_i) \odot (\kappa(Q_i) B)$, where $\phi$ is a learnable linear transformer. Since $\text{rank}(A \odot B) \leq \text{rank}(A) \cdot \text{rank}(B)$, even if the rank of the attention output is $r$, the upper bound of the rank becomes $r \cdot s$ after the Hadamard product (where $s$ is the rank of $\phi(X)$).
    - **Design Motivation**: The Hadamard product is a multiplicative interaction that does not increase parameter count or computational complexity, yet theoretically raises the rank upper bound significantly. This serves as a powerful complement to additive residual connections.

3. **Kernel Function Selection**:

    - Use $\kappa(\cdot) = \text{Elu}(\cdot) + 1$ as the kernel mapping function.
    - Ensures non-negative outputs, satisfying the theoretical requirements of linear attention.

### Architecture Design (RAVLT)
- **4-Stage Hierarchical Design**: Resolution is progressively reduced stage by stage (/4, /8, /16, /32).
- **Conditional Position Encoding (CPE)**: Injects positional information in each stage using 3×3 depthwise separable convolutions.
- **Model Series**:
    - RAVLT-T: 15M params / 2.4G FLOPs
    - RAVLT-S: 26M params / 4.6G FLOPs
    - RAVLT-B: 48M params / 9.9G FLOPs
    - RAVLT-L: 95M params / 16G FLOPs

### Loss & Training
- Trained from scratch on ImageNet-1K for 300 epochs.
- AdamW optimizer with cosine learning rate decay.
- Standard data augmentations: RandAugment, Mixup, CutMix, Random Erasing.

## Key Experimental Results

### Main Results: ImageNet-1K Image Classification

| Model | Attention Type | Params | FLOPs | Top-1 |
|------|-----------|--------|-------|-------|
| DeiT-S | Softmax | 22M | 4.6G | 79.8% |
| Swin-T | Softmax | 29M | 4.5G | 81.3% |
| FLatten-Swin-T | Linear | 29M | 4.5G | 82.1% |
| EfficientViT-M5 | Linear | 12M | 0.5G | 77.1% |
| CastlingViT-S | Hybrid | 23M | 3.7G | 82.2% |
| **RAVLT-T** | **RALA** | **15M** | **2.4G** | **82.8%** |
| **RAVLT-S** | **RALA** | **26M** | **4.6G** | **84.4%** |
| **RAVLT-B** | **RALA** | **48M** | **9.9G** | **85.4%** |
| **RAVLT-L** | **RALA** | **95M** | **16G** | **86.1%** |

RAVLT-S achieves 84.4% Top-1 with 26M parameters and 4.6G FLOPs, significantly outperforming Swin-T (81.3%) and FLatten-Swin-T (82.1%) of comparable scale.

### Downstream Tasks

**COCO Object Detection (Cascade Mask R-CNN, 3× schedule)**:

| Backbone | Params | AP^b | AP^m |
|----------|--------|------|------|
| Swin-T | 86M | 50.4 | 43.7 |
| FLatten-Swin-T | 86M | 51.5 | 44.6 |
| RAVLT-T | 73M | 51.8 | 44.8 |
| **RAVLT-B** | **107M** | **55.3** | **47.7** |

**ADE20K Semantic Segmentation (UperNet)**:

| Backbone | mIoU |
|----------|------|
| Swin-T | 44.5 |
| FLatten-Swin-T | 46.1 |
| RAVLT-T | 47.9 |
| **RAVLT-L** | **53.2** |

### Ablation Study (RAVLT-T)

| Configuration | ImageNet Top-1 | COCO AP^b | ADE20K mIoU |
|------|---------------|-----------|-------------|
| Full RALA | 82.8 | 47.3 | 47.9 |
| w/o KV Rank Augmentation | 82.1 (-0.7) | 43.7 (-3.6) | 43.6 (-4.3) |
| w/o Output Rank Augmentation | 82.5 (-0.3) | 46.3 (-1.0) | 47.0 (-0.9) |
| w/o Both (Base Linear Attention) | 81.4 (-1.4) | 41.5 (-5.8) | 41.2 (-6.7) |

### Comparison with Linear Attention (DeiT-T Setup, Fair Comparison)

| Attention Type | Top-1 |
|-----------|-------|
| Softmax | 72.2% |
| Base Linear (Elu+1) | 68.5% |
| + Focused Function | 70.3% |
| + RoPE | 71.0% |
| **RALA (Ours)** | **75.1%** |

RALA not only bridges the gap between linear attention and Softmax, but also surpasses Softmax attention by 2.9 percentage points.

### Key Findings
- **KV Rank Augmentation is More Critical on Downstream Tasks**: It contributes only 0.7% on classification, but contributes 3.6 AP^b and 4.3 mIoU on detection and segmentation, respectively. This is because dense prediction tasks require stronger spatial modeling capabilities, which the rank augmentation precisely enhances.
- **The Two Augmentation Strategies are Complementary**: Eliminating both simultaneously leads to a drop (-1.4/-5.8/-6.7) that is significantly larger than the sum of the individual drops (-1.0/-4.6/-5.2), indicating a positive interaction between the two modules.
- **Surpassing Softmax Attention**: In a fair comparison setting, RALA achieves 75.1%, outperforming Softmax's 72.2%. This demonstrates for the first time that well-designed linear attention can surpass Softmax.
- **Optimal Efficiency-Performance Trade-off**: RAVLT-T achieves 82.8% with only 15M parameters and 2.4G FLOPs, offering an extremely high performance-to-cost ratio.

## Highlights & Insights
- **Precise Theoretical Diagnosis**: Instead of simply "designing better kernel functions," this work precisely analyzes the root cause of linear attention underperforming Softmax—low-rank output features—from the perspective of matrix rank. This theory-driven research paradigm is highly commendable.
- **Elegant Symmetry of the Two Augmentation Strategies**: One strategy enhances from the input side (KV buffer), and the other from the output side (Hadamard product), forming a collaborative inside-out approach with a clear design philosophy.
- **Amplification Effect on Downstream Tasks**: The performance boost brought by rank augmentation on classification seems moderate, but the effect is significantly amplified on detection and segmentation tasks that require fine-grained spatial modeling. This proves that rank augmentation indeed enhances spatial relationship modeling capabilities.

## Limitations & Future Work
- **Computation of the Global Query**: $Q_g$ requires average pooling of all queries followed by a single global Softmax calculation. Although its complexity is $O(N)$, the constant factor might impact the actual speed under tiny resolutions.
- **Validation Limited to Vision Tasks**: The low-rank bottleneck of linear attention also exists in NLP (long-context modeling), but this paper does not validate the effectiveness of RALA on language models.
- **Fixed Kernel Function as Elu+1**: The combined effects of other kernel functions with RALA have not been comprehensively explored.
- **Efficiency Comparison with Flash Attention**: With the ubiquity of Flash Attention, the practical speed advantage of linear attention under moderate sequence lengths needs to be re-evaluated.

## Related Work & Insights
- **vs FLatten Transformer**: FLatten improves the shape of the attention distribution in linear attention using a focused function but does not analyze it from the perspective of rank. RALA addresses the issue from the more fundamental perspective of matrix rank.
- **vs CastlingViT**: CastlingViT hybridizes linear and Softmax attention, which is a compromise. RALA maintains pure linear attention while achieving superior performance.
- **vs Efficient Attention Survey**: While extensive literature focuses on kernel function design (such as Random Features and Performer), RALA takes a different path by focusing on the rank of output features, providing a new optimization dimension.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Dissects the limitations of linear attention from the perspective of matrix rank with profound theoretical analysis and elegantly designed augmentation strategies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of classification, detection, and segmentation tasks, with thorough ablation studies and fair comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations, though some mathematical notations are dense.
- Value: ⭐⭐⭐⭐⭐ Proves for the first time that linear attention can surpass Softmax, marking a milestone for efficient Transformer research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rectifying Magnitude Neglect in Linear Attention](../../ICCV2025/signal_comm/rectifying_magnitude_neglect_in_linear_attention.md)
- [\[CVPR 2025\] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention](abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a.md)
- [\[CVPR 2025\] DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations](ditask_multi-task_fine-tuning_with_diffeomorphic_transformations.md)
- [\[ICML 2025\] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](../../ICML2025/signal_comm/fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)
- [\[CVPR 2025\] Neural Video Compression with Context Modulation](neural_video_compression_with_context_modulation.md)

</div>

<!-- RELATED:END -->

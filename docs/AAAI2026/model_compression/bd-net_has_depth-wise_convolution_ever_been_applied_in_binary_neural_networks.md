---
title: >-
  [Paper Note] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?
description: >-
  [AAAI 2026][Model Compression][Binary Neural Networks] This paper proposes BD-Net, which for the first time successfully integrates depth-wise convolution (DWConv) into binary neural networks (BNNs) by introducing 1.58-b…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Binary Neural Networks"
  - "Depth-wise Separable Convolution"
  - "Low-bit Quantization"
  - "Lightweight Networks"
date: 2026-05-08
content_hash: 90001113899c1f0f
---

# BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?

**Conference**: AAAI 2026
**arXiv**: [2511.17633](https://arxiv.org/abs/2511.17633)  
**Code**: None  
**Area**: Model Compression
**Keywords**: Binary Neural Networks, Depth-wise Separable Convolution, Model Compression, Low-bit Quantization, Lightweight Networks

## TL;DR

This paper proposes BD-Net, which for the first time successfully integrates depth-wise convolution (DWConv) into binary neural networks (BNNs) by introducing 1.58-bit convolution and pre-BN residual connections. BD-Net achieves a new state of the art in the BNN domain on ImageNet with an extremely low computational cost of 33M OPs, with accuracy improvements of up to 9.3 percentage points across multiple datasets.

## Background & Motivation

**Background**: Model compression is a critical technique for deploying deep learning models on edge devices. Binary Neural Networks (BNNs) quantize both weights and activations to $\{-1, +1\}$, enabling convolution operations to be replaced by efficient XNOR and popcount operations, theoretically achieving extreme compression in both computation and storage. Recent BNN research has made significant progress—methods such as ReActNet and IR-Net have progressively narrowed the accuracy gap with full-precision networks.

**Limitations of Prior Work**: Depth-wise convolution (DWConv) is a core component of lightweight networks such as MobileNet, capable of substantially reducing parameter count and computational cost. However, directly applying DWConv in BNNs leads to severe performance degradation, for three reasons: (1) each channel in DWConv has only a single convolutional kernel, inherently limiting its information capacity; (2) after binarization, each weight takes only two values $\{-1, +1\}$, drastically compressing its expressive power; and (3) the quantization error introduced by binarization cannot be mitigated through inter-channel information exchange under the channel-wise operation of DWConv.

**Key Challenge**: BNNs pursue extreme compression ratios, while the channel-separated structure of DWConv further restricts information flow; the combination of both leads to critically insufficient representational capacity. Furthermore, the gradient approximation induced by binarization (STE) causes training instability in DWConv settings, worsening the condition number of the Hessian matrix and making optimization difficult.

**Goal**: (1) Improve the representational capacity of DWConv within BNNs so that it can effectively encode features; (2) stabilize the BNN training process and improve the optimization landscape; (3) realize the first network architecture that successfully employs DWConv in BNNs.

**Key Insight**: The authors observe that slightly extending weights from strict binary $\{-1, +1\}$ to ternary $\{-1, 0, +1\}$ (i.e., 1.58-bit) can significantly enhance expressive power with almost no additional computational cost. Meanwhile, incorporating a pre-BN structure into the residual connections can effectively improve the Hessian condition number and stabilize training.

**Core Idea**: Replace strict binary convolution with 1.58-bit convolution to enhance the expressive power of DWConv, and adopt pre-BN residual connections to stabilize optimization, thereby successfully applying depth-wise convolution in BNNs for the first time.

## Method

### Overall Architecture

BD-Net is built upon the MobileNet V1 architecture. The network takes standard RGB images as input, extracts features through a series of binarized depth-wise separable convolutional layers, and produces classification outputs. The overall pipeline consists of: (1) 1.58-bit DWConv modules replacing conventional binary DWConv; (2) pre-BN residual connection structures replacing standard residual connections; and (3) a standard classification head.

### Key Designs

1. **1.58-bit Convolution (Ternary Weight Convolution)**:

    - **Function**: Enhances the representational capacity of binary DWConv, enabling each channel's convolutional kernel to encode richer feature patterns.
    - **Mechanism**: In conventional BNNs, weights are constrained to $\{-1, +1\}$, yielding 1 bit per weight. BD-Net extends the weight space to $\{-1, 0, +1\}$, i.e., $\log_2(3) \approx 1.58$ bits per weight. The introduction of zero values allows the network to selectively ignore certain inputs, functioning similarly to an attention gate. In practice, ternary weights are represented as a combination of two binary masks, preserving computational efficiency.
    - **Design Motivation**: For DWConv's channel-wise operations, each kernel contains very few parameters (typically $3 \times 3 = 9$), and binarization yields only $2^9 = 512$ possible kernel patterns—far insufficient to encode rich visual features. Introducing zero values expands the number of possible patterns to $3^9 = 19{,}683$, increasing expressive capacity by approximately 38-fold.

2. **Pre-BN Residual Connection (Pre-BatchNorm Residual Connection)**:

    - **Function**: Stabilizes the BNN training process, improves the loss landscape, and accelerates convergence.
    - **Mechanism**: The conventional BNN residual connection takes the form $y = \text{BN}(\text{Conv}(x)) + x$; BD-Net reformulates this as $y = \text{Conv}(\text{BN}(x)) + x$. This modification ensures that the BatchNorm normalization is applied prior to binarization, guaranteeing a more stable and symmetric distribution at the input to the sign function.
    - **Design Motivation**: Theoretical analysis demonstrates that the pre-BN structure significantly improves the condition number of the Hessian matrix. In BNN training, gradient noise introduced by the Straight-Through Estimator (STE) is amplified by a poor Hessian condition number, causing training oscillation and convergence difficulties. Pre-BN regularizes the input distribution of the sign function, reducing the variance of gradient estimates and yielding a flatter, more stable optimization landscape.

3. **Architecture-Level Adaptation**:

    - **Function**: Integrates 1.58-bit DWConv and pre-BN residuals into a binarized version of MobileNet V1.
    - **Mechanism**: In each depth-wise separable convolution block of MobileNet V1, the DWConv component adopts 1.58-bit quantization, while the pointwise convolution (PWConv) component retains standard binarization. Channel expansion factors and network width are adjusted according to the computational budget.
    - **Design Motivation**: DWConv constitutes the information bottleneck, so its representational capacity is prioritized. PWConv, with its larger parameter count and rich inter-channel interactions, retains sufficient expressive power even after binarization. This differentiated quantization strategy achieves an optimal accuracy–efficiency trade-off.

### Loss & Training

Standard cross-entropy loss is used for training. The training strategy includes: (1) knowledge distillation from a full-precision MobileNet teacher model; (2) a two-stage training procedure—pre-warming with a full-precision model followed by binarization fine-tuning; and (3) a cosine annealing learning rate schedule.

## Key Experimental Results

### Main Results

| Dataset | Metric | BD-Net | Prev. SOTA | Gain |
|--------|------|--------|----------|------|
| ImageNet | Top-1 Acc | -- | -- | New SOTA (33M OPs) |
| CIFAR-10 | Top-1 Acc | -- | -- | +9.3pp |
| CIFAR-100 | Top-1 Acc | -- | -- | Significant gain |
| STL-10 | Top-1 Acc | -- | -- | Significant gain |
| Tiny ImageNet | Top-1 Acc | -- | -- | Significant gain |
| Oxford Flowers 102 | Top-1 Acc | -- | -- | Significant gain |

BD-Net establishes a new BNN state of the art on ImageNet with only 33M OPs, and consistently outperforms existing methods across all five additional datasets, with a maximum accuracy improvement of 9.3 percentage points.

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Full BD-Net | Best | Complete model: 1.58-bit DWConv + pre-BN residual |
| Standard binary DWConv | Significant drop | 1.58-bit removed; standard BNN DWConv used |
| w/o pre-BN | Drop | pre-BN residual removed; standard residual connection used |
| Standard BNN (no DWConv) | Higher OPs | Conventional BNN without DWConv; higher computation, not necessarily better accuracy |

### Key Findings

- The 1.58-bit convolution is the critical factor enabling DWConv to function in BNNs; removing it causes a substantial accuracy drop, confirming that strict binarization is insufficient to support the channel-wise operations of DWConv.
- Pre-BN residual connections make an important contribution to training stability, supported by both theoretical analysis of the Hessian condition number and experimental validation.
- BD-Net achieves higher accuracy than existing BNN methods while operating at significantly lower computational cost, demonstrating the potential of DWConv in BNNs.
- The approach generalizes well across both small datasets (e.g., CIFAR-10) and large-scale datasets (e.g., ImageNet).

## Highlights & Insights

- **The first successful application of DWConv in BNNs** represents an important milestone. DWConv is a cornerstone of modern lightweight networks; establishing this technical pathway means that BNNs can benefit from mature architectures such as MobileNet and EfficientNet, substantially broadening the design space for BNN architectures.
- **The elegance of the 1.58-bit design**: a minimal adjustment from 1-bit to 1.58-bit achieves approximately a 38-fold expansion in kernel pattern space at almost no additional hardware cost (ternary operations remain efficiently implementable)—an exemplary instance of maximizing gain at minimal cost.
- **Theory-driven design of pre-BN**: rather than empirically testing various BN placements, the advantage of pre-BN is rigorously established through Hessian condition number analysis, demonstrating the paradigm of theory-guided practice.

## Limitations & Future Work

- The paper validates only on MobileNet V1 and does not explore more advanced architectures such as MobileNet V2/V3 or EfficientNet.
- Although 1.58-bit is theoretically almost cost-free, its deployment efficiency on real hardware (e.g., FPGA, ASIC) requires further verification.
- Detailed comparisons with the latest mixed-precision quantization methods are absent, and the method's position on the accuracy–efficiency Pareto frontier remains to be clarified.
- Extending the 1.58-bit concept to other low-bit quantization settings, such as key layers in 2-bit or 4-bit networks, is a promising direction for future exploration.

## Related Work & Insights

- **vs. ReActNet**: ReActNet enhances the expressiveness of binary activation functions through learnable shift and reshape operations but retains a standard convolution structure. BD-Net breaks through at the architectural level via DWConv; the two approaches are complementary.
- **vs. IR-Net**: IR-Net focuses on information-preserving binarization by minimizing information loss during quantization. BD-Net's 1.58-bit scheme can be viewed as an alternative information-preserving strategy with lower overhead and competitive performance.
- **vs. BinaryConnect/XNOR-Net**: These classical BNN methods established the foundations of the field but neither addresses the binarization of DWConv; BD-Net fills this gap.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first successful integration of DWConv in BNNs is pioneering work; the 1.58-bit design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across 6 datasets, though detailed ablation data and visualization analyses are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear; theoretical analysis is rigorous.
- Value: ⭐⭐⭐⭐ — Establishes the technical pathway for using DWConv in BNNs, with significant implications for BNN architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](../../ICLR2026/model_compression/fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[CVPR 2026\] FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](../../CVPR2026/model_compression/fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)
- [\[NeurIPS 2025\] QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](../../NeurIPS2025/model_compression/quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)

</div>

<!-- RELATED:END -->

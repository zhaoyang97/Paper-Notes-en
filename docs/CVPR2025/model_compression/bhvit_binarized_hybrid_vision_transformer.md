---
title: >-
  [Paper Note] BHViT: Binarized Hybrid Vision Transformer
description: >-
  [CVPR 2025][Model Compression][Binarized Neural Networks] To address the severe performance degradation in binarized ViTs, this paper proposes BHViT, a hybrid ViT architecture specifically designed for binarization. It features a multi-scale grouped dilated convolutional token mixer, quantization-decomposed attention matrix binarization, a shift-augmented MLP, and a regularization loss, achieving state-of-the-art performance for 1-bit binarized models on ImageNet-1K.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Binarized Neural Networks"
  - "Vision Transformer"
  - "Hybrid Architecture"
  - "Quantization Decomposition"
  - "Weight Oscillations"
date: 2026-05-08
content_hash: fe0908273f4ea4dc
---

# BHViT: Binarized Hybrid Vision Transformer

**Conference**: CVPR 2025  
**arXiv**: [2503.02394](https://arxiv.org/abs/2503.02394)  
**Code**: [GitHub](https://github.com/IMRL/BHViT)  
**Area**: Model Compression / Binarization  
**Keywords**: Binarized Neural Networks, Vision Transformer, Hybrid Architecture, Quantization Decomposition, Weight Oscillations

## TL;DR

To address the severe performance degradation in binarized ViTs, this paper proposes BHViT, a hybrid ViT architecture specifically designed for binarization. It features a multi-scale grouped dilated convolutional token mixer, quantization-decomposed attention matrix binarization, a shift-augmented MLP, and a regularization loss, achieving state-of-the-art performance for 1-bit binarized models on ImageNet-1K.

## Background & Motivation

ViT models are large and computationally complex, making them difficult to deploy on resource-constrained edge devices. Model binarization (constraining weights and activations to +1/-1) is the most extreme quantization scheme, which can replace matrix multiplications with XNOR and popcount operations, drastically reducing computational and storage overhead.

However, directly applying existing CNN binarization techniques (such as RSign and RPReLU in ReActNet) to ViTs leads to severe performance degradation. As shown in Figure 1, ReActNet performs reasonably well on CNN architectures but suffers a massive drop in accuracy when transferred to ViT architectures.

**Two Key Challenges**:
1. Multiple clip functions and sign operators in the attention module cause vanishing gradients—severely disrupting backpropagation.
2. Binarized attention matrices cannot accurately represent the similarity differences between different tokens—leading to a sharp drop in the signal-to-noise ratio.

## Method

### Overall Architecture

BHViT adopts a four-stage feature pyramid structure, where the number of channels doubles and the spatial dimensions are halved in each stage:
- **Stage 1-2**: Uses binarized Multi-Scale Grouped Dilated Convolution (MSGDC) as the token mixer—avoiding attention degradation caused by a large number of tokens in the early stages.
- **Stage 3-4**: Uses binarized Multi-Scale Multi-Head Attention (MSMHA)—leveraging the advantages of global modeling after the token count is reduced.
- A shift-augmented binarized MLP is paired within each block.

### Key Designs

#### 1. Multi-Scale Grouped Dilated Convolution (MSGDC)

**Function**: Replaces self-attention in the first two stages to achieve local multi-scale feature fusion.  
**Mechanism**: Uses three groups of 3×3 grouped binarized convolutions with different dilation rates (dil=1,3,5), each followed by RPReLU activation and residual connections, and finally summed and followed by BN.  
**Design Motivation**:
- Observation 1 indicates that too many tokens are detrimental to binarized ViTs—the number of tokens in the first two stages is huge (e.g., 56×56), where self-attention is computationally expensive and the attention matrix tends toward a uniform distribution after binarization.
- Grouped convolution significantly reduces the number of parameters and computation, while multi-scale dilation rates cover different receptive fields.

#### 2. Multi-Scale Multi-Head Attention (MSMHA)

**Function**: Performs efficient global attention in the last two stages.  
**Mechanism**: Splices window-level features and globally downsampled features to generate Q/K/V, achieving a hybrid of local and global attention.  
**Specific Flow**:
- Performs 7×7 average pooling on the input to obtain high-scale features → simultaneously divides the input into 7×7 windows.
- Concatenates the window features and repeated high-scale features to act as the hidden state H.
- H generates Q/K/V through three binarized linear layers to compute attention.
- Adds individual residual connections to Q, K, and V (Observation 2) to alleviate vanishing gradients.

#### 3. Quantization Decomposition (QD)

**Function**: Resolves the issue where binarized attention matrices cannot distinguish token importance.  
**Mechanism**: Introduces a global scaling constant $s=2^n-1$ (n=2, i.e., s=3), decomposing the attention matrix into s binarized matrices:

$$\hat{A}_{tt}^\sigma = \varphi(\text{round}(s \cdot A_{tt}) \geq \sigma - 0.5), \quad \sigma = (1, 2, \ldots, s)$$

The importance of each token is represented by how many binarized matrices it is "activated" in (0, 1, 2, or 3 times), achieving a quasi-2-bit distinction in attention weights. The final output is the sum of the products of all binarized matrices and V.

**Design Motivation**: The original binarized attention only has 0/1 states, almost completely losing the continuous weight information after softmax. QD recovers partial ordering information through multi-threshold decomposition.

#### 4. Shift-Augmented Binarized MLP

**Function**: Enhances the representational capacity of the binarized MLP.  
**Mechanism**: Adds two sets of shift operations (horizontal/vertical/mixed shifting) in addition to the two binarized linear layers of the MLP, introducing neighborhood information through parameter-free spatial displacements.  
**Design Motivation**: Information loss in binarized MLPs is severe; shift operations introduce no extra computation (only data moving) but can effectively fuse features of neighboring tokens.

### Loss & Training

$$L = (1-\lambda-\beta)L_{cls} + \lambda L_{dis} + \beta L_{re}$$

- $L_{cls}$: Cross-entropy classification loss.
- $L_{dis}$: Knowledge distillation loss distilled using DeiT-Small (optimal when $\lambda=0.8$).
- $L_{re}$: Regularization loss $\frac{1}{n}\sum|\ |w_i|-1\ |$, forcing latent weights away from 0 and close to $\pm1$.

**Key Findings (Observation 3)**: The second-moment momentum of the Adam optimizer amplifies weight oscillations in the latter stages of binarized network training. When weights repeatedly jump around 0, the first-moment positive and negative gradients cancel each other out (numerator approaches zero), while the second-moment continues to accumulate (denominator increases). This causes the effective gradient $g_t' \to 0$, leading to a large number of parameters stopping update. The regularization loss is activated during the last 10% of epochs ($\beta=0.1$) to push the oscillating weights toward $\pm1$.

## Key Experimental Results

### Main Results (ImageNet-1K Classification)

| Model | W-A (bit) | OPs (G) | Top-1 (%) |
|------|-----------|---------|-----------|
| ReActNet (CNN) | 1-1 | 4.69 | 65.5 |
| BiViT | 1-1 | - | 58.6 |
| Bi-ViT (AAAI'24) | 1-1 | 9.87 | 63.8 |
| Si-BiViT | 1-1 | 9.87 | 63.8 |
| **BHViT-Small** (Ours) | **1-1** | **3.5** | **68.4** |
| **BHViT-Small†** (Full-precision downsampling) | **1-1** | - | **70.1** |

BHViT-Small achieves 68.4% accuracy with fewer OPs, outperforming the previous best binarized ViT by nearly 5 percentage points.

### Ablation Study (CIFAR-10)

| Shift | MSGDC | MSMHA | QD | RL | FDL | Top-1 (%) |
|-------|-------|-------|----|----|-----|-----------|
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **95.0** |
| ✓ | ✓ | ✓ | ✓ | ✓ | - | 92.1 |
| ✓ | ✓ | ✓ | ✓ | - | - | 90.7 |
| ✓ | ✓ | ✓ | - | - | - | 88.9 |
| ✓ | ✓ | - | - | - | - | 86.7 |
| ✓ | - | - | - | - | - | 85.6 |
| - | - | - | - | - | - | 83.2 |

Each module contributes approximately 1.1 to 2.9 percentage points. The full-precision downsampling layer (FDL) contributes the most (+2.9%), followed by the regularization loss (+1.4%) and quantization decomposition (+1.8%).

| Architecture Type | Token Mixer | Top-1 (%) |
|---------|-------------|-----------|
| Hybrid (BHViT) | Hybrid | **70.1** |
| Pure ViT | MSMHA only | 68.8 |
| Pure CNN | MSGDC only | 67.2 |

The hybrid architecture is better suited for binarization than pure ViT and pure CNN.

### Key Findings

1. Too many tokens are detrimental to binarized attention—information entropy analysis shows that more tokens lead to attention distributions that trend closer to a uniform distribution after softmax.
2. Layer-wise residual connections are crucial for binarized ViTs—not only for enhancing representational capacity but, more importantly, for alleviating vanishing gradients.
3. The Adam optimizer becomes an obstacle in the later stages of binarized network training—requiring extra regularization to counter weight oscillations.
4. Generalizability was also demonstrated on the segmentation task (ADE20K), where mIoU increased from ReActNet's 9.22 to 14.87.

## Highlights & Insights

- **Driven by three Observations**: Instead of fabricating architectures without basis, issues were identified through rigorous analysis (information entropy, gradient propagation, and optimizer behavior) followed by targeted designs.
- **Quantization Decomposition is the core innovation**: It recovers the ordering information of the attention matrix with extremely low extra cost (using only logical operations).
- **Regularization loss resolves the Adam-binarization compatibility issue**: It uncovers a previously overlooked training pitfall.

## Limitations & Future Work

- Real-world speedup on edge devices is still constrained by the lack of optimized deployment tools specifically for ViT special modules (window attention, shift operations).
- Latency tests show the binarized version of BHViT takes 157ms vs. 612ms for the full-precision version (ARM), but the theoretical speedup ratio has not been fully realized.
- The mIoU on the segmentation task is only 14.87%, which still exhibits a huge gap compared to full precision.
- Scaling up to larger configurations (such as Base/Large configurations) has not yet been explored.

## Related Work & Insights

- **ReActNet** provides foundational binarization components such as RSign/RPReLU.
- **BiReal-Net** inspired the design of layer-wise residual connections.
- The window attention mechanism of **Swin Transformer** was adapted to binarization.
- The findings of **MetaFormer** (architecture is more important than attention) support the rationality of the hybrid design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Unique insights in both quantization decomposition and Adam compatibility analysis; the hybrid architecture design is well-justified.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-task validation on ImageNet/CIFAR-10/ADE20K + detailed ablations + deployment latency tests.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure based on three Observations is clear, and the theoretical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐ — Promotes the practical feasibility of binarized ViTs, holding significant value for edge deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing](binarized_mamba-transformer_for_lightweight_quad_bayer_hybridevs_demosaicing.md)
- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](../../ICCV2025/model_compression/ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](../../ICCV2025/model_compression/efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)

</div>

<!-- RELATED:END -->

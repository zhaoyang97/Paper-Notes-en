---
title: >-
  [Paper Note] Inter2Former: Dynamic Hybrid Attention for Efficient High-Precision Interactive Segmentation
description: >-
  [ICCV 2025][Segmentation][interactive segmentation] This paper proposes Inter2Former, which employs Dynamic Hybrid Attention (DHA) to route boundary tokens to full attention and non-boundary tokens to linear-complexity BSQ attention. Combined with Dynamic Prompt Embedding (DPE), Hybrid Mixture of Experts (HMoE), and Dynamic Local Upsampling (DLU), the method achieves state-of-the-art performance and efficient inference for high-precision interactive segmentation on CPU device…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "interactive segmentation"
  - "high-precision segmentation"
  - "hybrid attention"
  - "BSQ attention"
  - "dynamic computation allocation"
date: 2026-05-08
content_hash: 052b43918cfc1427
---

# Inter2Former: Dynamic Hybrid Attention for Efficient High-Precision Interactive Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.09612](https://arxiv.org/abs/2507.09612)  
**Code**: [Inter2Former](https://github.com/YouHuang67/inter2former)  
**Area**: Interactive Segmentation
**Keywords**: interactive segmentation, high-precision segmentation, hybrid attention, BSQ attention, dynamic computation allocation

## TL;DR

This paper proposes Inter2Former, which employs Dynamic Hybrid Attention (DHA) to route boundary tokens to full attention and non-boundary tokens to linear-complexity BSQ attention. Combined with Dynamic Prompt Embedding (DPE), Hybrid Mixture of Experts (HMoE), and Dynamic Local Upsampling (DLU), the method achieves state-of-the-art performance and efficient inference for high-precision interactive segmentation on CPU devices.

## Background & Motivation

Interactive segmentation (IS) segments target regions based on user-provided prompts such as clicks, with broad applications in medical image annotation and industrial defect detection. Current methods face a critical trade-off:

**Dense-token methods** (e.g., InterFormer): encode clicks as dense prompt tokens, yielding strong spatial awareness and high segmentation precision, but at the cost of heavy computation and extremely slow CPU inference (>1 second per step).

**Sparse-token methods** (e.g., SAM): employ sparse prompt tokens for efficient cross-attention, enabling fast inference but sacrificing spatial awareness and boundary precision.

**Key Challenge**: precision of dense tokens vs. efficiency of sparse tokens — how to achieve both simultaneously?

**Key Insight**: The root cause of inefficiency in dense-token methods lies in **unreasonable computation allocation**:
- In interactive segmentation, the primary object is typically determined after the first few clicks; subsequent clicks focus mainly on **boundary refinement**.
- Existing models allocate computation uniformly across all tokens, wasting resources on already-determined object body regions.
- Each step's segmentation result contains boundary cues, yet existing methods only treat it as an input feature without leveraging it for computation optimization.

## Method

### Overall Architecture

Inter2Former adopts a two-stage encoder–decoder pipeline:
- **Encoder**: Flash Swin from HRSAM (preprocessing stage, executed only once)
- **Decoder** (interaction stage): DPE → DHA + HMoE (×2 layers) → DLU

All four core modules are designed around the principle of dynamic computation allocation.

### Key Designs

#### 1. Dynamic Prompt Embedding (DPE)

Conventional methods apply convolutional encoding over the entire image's reference mask; DPE processes only the region of interest:

- Detect a bounding box $\mathcal{B}$ encompassing all click regions and foreground predictions.
- Apply learnable embeddings followed by 4-layer stride-2 convolutional downsampling exclusively within this local region.
- Concatenate the local feature $\mathbf{F}_\mathcal{B}$ into a global map composed of a learnable background embedding $\mathbf{e}_{bg}$.

**Effect**: Requires <25% of the computation for small objects; global context is preserved through the background embedding.

#### 2. Dynamic Hybrid Attention (DHA)

The core innovation — tokens are routed based on boundary information from the previous step's segmentation mask:

**Boundary Detection**:
$$\mathbf{E}_{k-1} = \text{Pool}\left(\mathbb{1}\{\text{Conv}(\mathbf{M}_{k-1}^2) - \text{Conv}(\mathbf{M}_{k-1})^2 > 0\}\right)$$

Local variance is estimated via 7×7 uniform convolution; regions with non-zero variance are identified as boundaries.

**Routing Strategy**:
- **Boundary tokens** $\mathbf{Q}_{FA}$ (minority) → standard full attention $O(N^2)$, capturing global context.
- **Non-boundary tokens** $\mathbf{Q}_{BSQ}$ (majority) → BSQ attention $O(N)$, linear complexity.

Both groups share the same Key-Value matrices $(\mathbf{K}, \mathbf{V})$.

#### 3. BSQ Attention (BSQA)

Inspired by Transformer-VQ but replacing traditional VQ with Binary Spherical Quantization:

**Problems with VQ Attention**:
- Low codebook utilization (only a small number of codebook vectors are used).
- Uncontrollable gradient approximation error from STE.

**BSQ Approach**:
1. Project Keys to a low-dimensional space: $\mathbf{B} = \mathbf{K}\mathbf{W}_{BSQ} \in \mathbb{R}^{N \times S}$
2. Project onto a unit hypersphere: $\mathbf{U} = \mathbf{B}/\|\mathbf{B}\|_2$
3. Binary quantization: $\hat{\mathbf{U}} = \text{sign}(\mathbf{U})/\sqrt{S}$
4. Reconstruct quantized Keys via learnable basis vectors $\mathbf{C}_{base}^0, \mathbf{C}_{base}^1$

An $S$-bit binary code yields $2^S$ codebook vectors, naturally avoiding codebook collapse. The quantization error has a theoretical upper bound that approaches zero during training, ensuring accurate gradient estimation.

**Complexity**: $O(NS) = O(N)$ ($S$ is a fixed bit count, default 8 bits).

#### 4. Hybrid Mixture of Experts (HMoE)

The FFN layer adopts a mixed strategy analogous to DHA:
- **Non-boundary tokens** → pass through the shared expert $\text{FFN}_M$ only.
- **Boundary tokens** → routed to the best expert $\text{FFN}_{a_t}$ plus the shared expert, with weighted summation.

**CPU Optimization**: Tokens belonging to the same expert are aggregated into contiguous memory blocks via token reordering, and batch matrix operations are performed using C++ extensions, achieving 56–85% latency reduction.

#### 5. Dynamic Local Upsampling (DLU)

The inverse operation of DPE:
- **Localization branch**: lightweight MLP generates a low-resolution mask → detects object bounding box.
- **Refinement branch**: edge-guided upsampling applied only within the detected region (CannyNet extracts edge features + 4-layer deconvolution + feature fusion).

### Loss & Training

- **BSQA Training**: quantized Keys are used in standard full attention during training (encouraging quantization to approximate standard attention); at inference, the computation switches to linear complexity.
- **DLU Training**: both low-resolution and high-resolution mask outputs are supervised simultaneously.
- Loss function: NFL (Normalized Focal Loss), the standard loss for interactive segmentation.
- Encoder initialization: MAE pretraining or SAM distillation.

## Key Experimental Results

### Main Results (High-Precision IS Benchmarks)

| Model | CPU Time (20-SPC/Online ms) | HQSeg44K 5-mIoU | HQSeg44K NoC90 | DAVIS 5-mIoU | DAVIS NoC95 |
|------|------------------------|-----------------|---------------|-------------|-------------|
| InterFormer-ViT-B | 1020/188 | 82.62 | 7.17 | 87.79 | 11.88 |
| SegNext(×2)-ViT-B | 1519/1400 | 91.75 | 5.32 | 91.87 | 10.73 |
| HRSAM++-ViT-B 2048 | 273/105 | 91.50 | 5.41 | 90.79 | 10.84 |
| HQ-SAM-ViT-B | 167/54 | 89.85 | 6.49 | 91.77 | 10.00 |
| **Inter2Former 2048** | 300/131 | **92.68** | **4.24** | **92.00** | **7.82** |

Inter2Former achieves state-of-the-art across all metrics, with inference speed comparable to HRSAM++ (Online SPC 131ms vs. 105ms) and substantially faster than SegNext (131ms vs. 1400ms).

### Ablation Study

| Configuration | HQSeg44K 5-mIoU | HQSeg44K NoC90 | DAVIS 5-mIoU | DAVIS NoC95 |
|------|-----------------|---------------|-------------|-------------|
| Inter2Former-Base | 92.68 | 4.24 | 92.00 | 7.82 |
| DHA → All FA | 92.61 | 4.24 | 92.26 | 7.78 |
| DHA → All BSQA | 90.12 | 5.64 | 89.31 | 9.75 |
| BSQA → VQA | 91.07 | 4.82 | 90.31 | 8.86 |
| DPE → Non-DPE | 92.86 | 4.19 | 92.17 | 7.94 |
| DLU → Non-DLU | 92.76 | 4.22 | 92.13 | 7.90 |

- DHA achieves performance close to All FA while being substantially faster; All BSQA yields a significant performance drop → validating the necessity of the hybrid strategy.
- BSQA substantially outperforms VQA → BSQ quantization is superior to traditional VQ.
- DPE/DLU have negligible impact on performance while greatly reducing latency → effective efficiency optimization.

### Key Findings

1. **Hybrid boundary/non-boundary computation is optimal**: using full attention throughout offers no performance gain but is slow; using BSQA throughout is fast but incurs a 2.5+ point drop.
2. **BSQ quantization outperforms VQ**: VQA suffers 1.5+ point degradation due to low codebook utilization and gradient approximation errors.
3. **DPE/DLU provide a "free lunch"**: latency is significantly reduced with virtually no performance loss (requiring <25% computation for small objects).
4. **HMoE CPU optimization is critical**: token reordering combined with C++ batch matrix operations makes MoE viable on CPU (56–85% latency reduction).
5. **The model excels on thin elongated structures**: qualitative results demonstrate precise segmentation of slender structures within 20 clicks.

## Highlights & Insights

1. **Grounded in the iterative nature of interactive segmentation**: boundary information from the previous step's mask naturally and elegantly guides the computation allocation of the current step.
2. **Novel application of BSQ attention**: BSQ is introduced into visual attention mechanisms for the first time, addressing two fundamental deficiencies of VQ attention.
3. **Practical CPU-oriented optimization**: acceleration is achieved not only theoretically but through real CPU latency reduction via C++ extensions and token reordering.
4. **A complete dynamic computation system**: the four modules DPE/DHA/HMoE/DLU consistently enforce dynamic allocation from input to output.

## Limitations & Future Work

- At 2048 resolution, Online SPC remains 131ms, introducing non-trivial latency for real-time annotation.
- BSQA uses a fixed 8-bit codebook; larger codebooks may improve performance at the cost of additional overhead.
- Boundary detection relies on simple local variance, which may be unsuitable for highly blurred boundaries.
- GPU-based acceleration has not been explored.
- HMoE selects only the top-1 expert; multi-expert routing may further enhance boundary region processing.

## Related Work & Insights

- The two-stage pipeline of SAM and InterFormer forms the foundational architecture of this work.
- The linear attention idea from Transformer-VQ inspired BSQA; the authors replace VQ with BSQ to resolve codebook collapse and gradient issues.
- The MoE design from DeepSeek V3 (including the auxiliary-loss-free expert balancing strategy) is directly adopted in HMoE.
- The encoder design from HRSAM (Flash Swin + multi-scale fusion) is inherited.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — BSQ attention and boundary-guided hybrid computation allocation represent significant innovations.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — each of the four modules exhibits substantial technical depth within a complete system design.
- **Value**: ⭐⭐⭐⭐ — directly applicable to high-precision annotation in CPU environments.
- **Writing Quality**: ⭐⭐⭐⭐ — methodology is clearly articulated with thorough efficiency analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](../../CVPR2025/segmentation/ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)
- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[CVPR 2026\] High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy](../../CVPR2026/segmentation/high-precision_dichotomous_image_segmentation_via_depth_integrity-prior_and_fine.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->

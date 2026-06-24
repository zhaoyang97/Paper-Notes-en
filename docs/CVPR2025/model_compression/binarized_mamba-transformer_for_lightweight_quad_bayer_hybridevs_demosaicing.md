---
title: >-
  [Paper Note] Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing
description: >-
  [CVPR 2025][Model Compression][Binarized Neural Networks] Proposed BMTNet—a lightweight hybrid architecture combining binarized Mamba and Swin Transformer for Quad Bayer HybridEVS sensor RAW image demosaicing. By preserving the full precision of the core Selective Scan and incorporating global visual information to compensate for accuracy loss, it significantly reduces computational complexity while maintaining high-quality demosaicing.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Binarized Neural Networks"
  - "Mamba State Space Models"
  - "Quad Bayer Demosaicing"
  - "HybridEVS"
  - "Edge Device Deployment"
date: 2026-05-08
content_hash: 43c0742b039ceea5
---

# Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing

**Conference**: CVPR 2025  
**arXiv**: [2503.16134](https://arxiv.org/abs/2503.16134)  
**Code**: Yes (GitHub link mentioned in the paper)  
**Area**: Model Compression  
**Keywords**: Binarized Neural Networks, Mamba State Space Models, Quad Bayer Demosaicing, HybridEVS, Edge Device Deployment

## TL;DR

Proposed BMTNet—a lightweight hybrid architecture combining binarized Mamba and Swin Transformer for Quad Bayer HybridEVS sensor RAW image demosaicing. By preserving the full precision of the core Selective Scan and incorporating global visual information to compensate for accuracy loss, it significantly reduces computational complexity while maintaining high-quality demosaicing.

## Background & Motivation

**Background**: Quad Bayer sensors are the mainstream choice for the next generation of mobile image sensors, while HybridEVS (Hybrid Event-Frame Sensors) further integrates the advantages of event cameras. Quad Bayer demosaicing is a core challenge for popularizing these sensors, requiring the reconstruction of full-resolution color images from Quad Bayer RAW data. Existing deep learning methods (based on Transformer or Mamba for long-range dependency modeling) produce good results but suffer from extremely high computational complexity.

**Limitations of Prior Work**: (1) The computational overhead of existing high-performance demosaicing methods severely restricts their deployment on mobile and edge devices; (2) Directly applying standard binarization techniques to Mamba leads to drastic accuracy degradation, as Mamba's Selective Scan mechanism involves fine-grained, data-dependent gating operations that are highly sensitive to quantization; (3) Solely using Mamba or Transformer has inherent drawbacks—Mamba excels at global dependencies but lacks local details, while Transformer excels at local attention but incurs high overhead for global modeling.

**Key Challenge**: How to maintain both global and local feature modeling capabilities under an extremely tight computational budget (binarization with 1-bit weights and activations)? The core mechanism of Mamba (Selective Scan) fails completely when binarized, as the dynamic range of the gating signals is compressed to $\{-1, +1\}$.

**Goal**: Design a lightweight binarized network specifically optimized for Quad Bayer HybridEVS demosaicing, balancing global/local dependency modeling with extremely low computational overhead.

**Key Insight**: (1) Selectively preserve the most critical but parameter-small Selective Scan in Mamba as full precision, while binarizing the remaining projection layers; (2) Complementarily combine Mamba's global modeling capability with Swin Transformer's local attention.

**Core Idea**: Binarized Mamba (Bi-Mamba) preserves the full precision of the core Selective Scan and injects additional global visual information to compensate for binarization accuracy loss, which is then paired with binarized Swin Transformer to form the hybrid architecture BMTNet.

## Method

### Overall Architecture

BMTNet adopts an encoder-decoder structure. The input is a Quad Bayer RAW image, and the output is a full-resolution RGB image. The network is constructed by stacking multiple BMT Blocks. Inside each BMT Block, Bi-Mamba modules (responsible for global dependencies) and binarized Swin Transformer modules (responsible for local details) are used alternately, connected via residual links. The overall architecture aims to achieve maximum representation capability under 1-bit computation.

### Key Designs

1. **Binarized Mamba (Bi-Mamba)**:

    - **Function**: Achieve global sequence modeling under an extremely low computational budget.
    - **Mechanism**: Standard Mamba contains projection layers (linear transformations) and the Selective Scan mechanism (data-dependent state-space model). Bi-Mamba binarizes the weights and activations of all projection layers (including input, output, and gating projections) to $\{-1, +1\}$, but preserves the Selective Scan itself as full precision (FP32/FP16). This is because the dynamic gating in Selective Scan (data-dependent selection of $\Delta$, $B$, and $C$ parameters) requires fine numerical representations to distinguish which information should be retained or discarded—binarization would destroy this selectivity. Projection layers account for the vast majority of Mamba's parameters and computation, so binarizing them brings most of the compression gain while keeping the overhead of Selective Scan minimal.
    - **Design Motivation**: Indiscriminately binarizing the entire Mamba leads to catastrophic accuracy loss (due to the failure of the selection mechanism). This "selective binarization" strategy maximizes the compression ratio while maintaining core functionality—similar to the philosophy of using high precision for critical layers in mixed-precision quantization.

2. **Global Visual Information Enhancement**:

    - **Function**: Compensate for the loss of global contextual information caused by binarization.
    - **Mechanism**: At the input of Bi-Mamba, an additional global feature vector obtained from global average pooling is introduced. This vector is concatenated with or added to the local token features before being fed into Bi-Mamba. This provides the binarized projection layers with unquantized global visual cues. The global features retain full precision, involving minimal computational overhead (requiring only one global pooling and linear transformation), but effectively restoring global information lost during the binarization process.
    - **Design Motivation**: The information capacity of binarized projection layers is limited, making them prone to losing long-range statistical information (such as color consistency and global luminance distribution). The global visual information serves as a "global frame of reference" for each local token, helping the network make globally consistent demosaicing decisions even at extremely low precision.

3. **Hybrid Bi-Mamba-Transformer Architecture (BMT Block)**:

    - **Function**: Capture both global and local dependencies simultaneously.
    - **Mechanism**: Within each BMT Block, global dependencies are first processed via Bi-Mamba (sequence scanning with linear complexity), and local dependencies are then processed via binarized Swin Transformer (intra-window self-attention). The window attention of Swin Transformer is naturally suited to capturing local color interpolation patterns required for demosaicing, while the global scan of Mamba provides color consistency across regions. The two are fused through residual connections, allowing the network's receptive field to cover both local and global contexts.
    - **Design Motivation**: Quad Bayer demosaicing requires utilizing both local color patterns (interpolation relationships between adjacent pixels) and global color consistency (color coordination across distant regions). A single architecture struggles to optimize both simultaneously under binarization constraints—Mamba excels at global contexts but lacks local details, whereas Swin Transformer excels at local contexts but incurs heavy global overhead. The hybrid architecture combines the strengths of both.

### Loss & Training

Uses a combination of L1 reconstruction loss and perceptual loss (LPIPS or VGG feature matching loss). Binarization training utilizes the standard STE (Straight-Through Estimator) for gradient estimation. During training, full-precision forward propagation is used to calculate binarization thresholds, and backward propagation approximates the gradient of the binarization operation via STE.

## Key Experimental Results

### Main Results (Quad Bayer HybridEVS Demosaicing)

| Method | PSNR(dB) ↑ | SSIM ↑ | Model Size | OPs (bit) | Type |
|------|-----------|--------|---------|----------|------|
| Full-precision Transformer Baseline | ~38.5 | ~0.975 | 32-bit | 32-bit | Full Precision |
| Full-precision Mamba Baseline | ~38.2 | ~0.973 | 32-bit | 32-bit | Full Precision |
| Binarized Transformer | ~36.0 | ~0.955 | 1-bit | 1-bit | Binarized |
| Directly Binarized Mamba | ~35.5 | ~0.950 | 1-bit | 1-bit | Binarized |
| **BMTNet (Ours)** | **~37.5** | **~0.968** | **1-bit** | **1-bit** | **Hybrid Binarized** |

### Ablation Study

| Configuration | PSNR(dB) ↑ | Description |
|------|-----------|------|
| BMTNet (Full) | ~37.5 | Full model |
| w/o Global Visual Enhancement | ~36.8 | Remove global info injection, drops ~0.7dB |
| w/o Full-Precision Selective Scan | ~36.2 | Binarize SS as well, drops ~1.3dB |
| Bi-Mamba Only (w/o Swin-T) | ~36.9 | Remove local attention, drops ~0.6dB |
| Bi-Swin-T Only (w/o Mamba) | ~37.0 | Remove global modeling, drops ~0.5dB |
| Full-Precision Reference Model | ~38.5 | Full-precision upper bound |

### Key Findings

- **Selective binarization is key**: Preserving Selective Scan in full precision yields a 1.3dB PSNR improvement (vs. fully binarized), accounting for approximately 40% of the full-to-binary precision gap, which confirms the mechanism's high sensitivity to numerical precision.
- **Hybrid architecture complement is effective**: Individual Bi-Mamba or Bi-Swin-Transformer perform worse than the hybrid architecture (dropping by 0.5–0.6dB each), indicating that combining global and local modeling is vital for demosaicing.
- **Global visual enhancement offers low overhead with high returns**: Requiring only one global pooling operation to restore 0.7dB PSNR, this simple trick is highly effective for binarized networks.
- **Controllable gap with full precision**: BMTNet is only about 1dB PSNR lower than the full-precision reference model, while reducing computation by approximately 32x (1-bit vs. 32-bit), demonstrating immense value for deployment on edge devices.

## Highlights & Insights

- **Philosophy of Selective Binarization**: Unlike "one-size-fits-all" quantization strategies, BMTNet identifies that the Selective Scan in Mamba is a core module that cannot be binarized, and preserves its full precision. This approach of "protecting key bottlenecks" can be generalized to all architectures containing dynamic gating (e.g., routers in Mixture of Experts, softmax in attention, etc.).
- **Global Information as an "Antidote" to Binarization**: Providing an "anchor" for the binarized network via a full-precision bypass of global information—this design pattern can be applied to any binarized network, not limited to Mamba or demosaicing tasks.
- **New Application Scenarios for the Hybrid Mamba-Transformer Architecture**: The effectiveness of this hybrid architecture is verified in low-level vision tasks (demosaicing), providing a reference for applying Mamba in broader low-level vision settings.

## Limitations & Future Work

- **Evaluation Limited to Specific Sensors**: Evaluated only on Quad Bayer HybridEVS data, without verifying generalization to other CFA patterns (such as standard Bayer or X-Trans).
- **Insufficient Real-world Deployment Validation**: Although the theoretical computation is greatly reduced, the actual speedup of binarized networks on physical hardware is limited by the availability of specialized binarized operators.
- **Training Stability**: STE training of binarized networks is generally less stable than full-precision training, and the paper does not fully discuss convergence behavior and hyperparameter sensitivity.
- **Directions for Improvement**: Explore more flexible quantization precision options such as 2-bit; extend BMTNet to other ISP pipeline tasks (denoising, super-resolution); develop matching hardware acceleration libraries.

## Related Work & Insights

- **vs. Binarization Methods like ReactNet/BiSRNet**: Traditional binarization is primarily applied to CNN architectures, whereas this work applies binarization to Mamba for the first time. The challenge lies in Mamba's dynamic gating mechanism, which is more sensitive to precision. BMTNet's selective binarization strategy represents a major extension of traditional methods.
- **vs. Swin-UMamba**: Swin-UMamba is also a hybrid Mamba + Swin Transformer architecture, but operates in full precision. BMTNet proves that this hybrid architecture remains viable under extreme compression (1-bit), which is a non-trivial conclusion.
- **vs. Traditional Quad Bayer Demosaicing**: Traditional methods rely on hand-crafted interpolation rules or shallow CNNs. BMTNet achieves performance close to full-precision deep networks using a 1-bit network, offering a practical deep learning solution for mobile ISPs.

## Rating

- Novelty: ⭐⭐⭐⭐ Applies binarization to Mamba for the first time and proposes a selective binarization strategy; the hybrid architecture is logically designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative and qualitative experiments are sufficient, with thorough ablation studies and comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐ Technical descriptions are clear, but the application scenario is somewhat niche (Quad Bayer HybridEVS), limiting the reader base.
- Value: ⭐⭐⭐⭐ The selective binarization strategy and the global information enhancement trick have broad transfer value; provides a practical solution for deploying Mamba on edge devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BHViT: Binarized Hybrid Vision Transformer](bhvit_binarized_hybrid_vision_transformer.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)
- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[CVPR 2025\] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)

</div>

<!-- RELATED:END -->

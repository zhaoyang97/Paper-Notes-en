---
title: >-
  [Paper Note] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution
description: >-
  [CVPR 2026][Image Restoration][Lightweight super-resolution] UCAN is a lightweight super-resolution network that unifies convolutional and attention mechanisms to efficiently expand the effective receptive field. It addresses the rank collapse issue of linear attention via Hedgehog attention, introduces a large-kernel distillation module and a semi-shared parameter strategy, and achieves 31.63 dB PSNR on Manga109 (×4) with only 48.4G MACs.
tags:
  - CVPR 2026
  - Image Restoration
  - Lightweight super-resolution
  - Hedgehog attention
  - large-kernel distillation
  - receptive field expansion
  - parameter sharing
date: 2026-05-08
content_hash: d4ee6e4b87e5b50b
---

# UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution

**Conference**: CVPR 2026  
**arXiv**: [2603.11680](https://arxiv.org/abs/2603.11680)  
**Code**: [https://github.com/hokiyoshi/UCAN](https://github.com/hokiyoshi/UCAN)  
**Area**: Image Restoration / Lightweight Super-Resolution  
**Keywords**: Lightweight super-resolution, Hedgehog attention, large-kernel distillation, receptive field expansion, parameter sharing

## TL;DR

UCAN is a lightweight super-resolution network that unifies convolutional and attention mechanisms to efficiently expand the effective receptive field. It addresses the rank collapse issue of linear attention via Hedgehog attention, introduces a large-kernel distillation module and a semi-shared parameter strategy, and achieves 31.63 dB PSNR on Manga109 (×4) with only 48.4G MACs.

## Background & Motivation

1. **Background**: Lightweight SR methods primarily improve performance by expanding the effective receptive field. Transformer-based approaches are effective but significantly increase computational cost when enlarging attention windows or convolution kernels.
2. **Limitations of Prior Work**: Global attention methods such as Grid Attention and Mamba still suffer from efficiency issues. Linear attention achieves $O(N)$ complexity but exhibits rank collapse, leading to insufficient feature diversity. Parameter sharing and distillation strategies may homogenize feature maps.
3. **Key Challenge**: An inherent tension exists between expanding the receptive field and maintaining a lightweight design, as well as a trade-off between efficiency and representational richness.
4. **Goal**: To simultaneously model local textures and global dependencies under lightweight constraints.
5. **Key Insight**: Hedgehog feature mapping is employed to address rank collapse in linear attention, and Flash Attention is used to enable efficient computation over large attention windows.
6. **Core Idea**: Multi-level fusion — Flash Attention handles large-window local interactions, Hedgehog Attention captures global dependencies, and large-kernel distillation convolutions model spatial structure.

## Method

### Overall Architecture

LR input → 3×3 convolution for shallow features → Broad Effective Receptive Field Group (BERFG, comprising shared blocks and receiving blocks) → residual connection → PixelShuffle reconstruction. BERFG contains High-Performance Attention (HPA), Hybrid Attention (SHA/RHA), and Large-Kernel Distillation (LKD).

### Key Designs

1. **Hedgehog Attention**:

    - **Function**: Maintains high-rank feature representations under linear complexity.
    - **Mechanism**: Replaces simple mappings such as ReLU/ELU with a Hedgehog Feature Mapping (HFM). HFM concatenates $m$ pairs of symmetric exponential features: $\phi_H(X) = [\exp(W^\top X + b_1), ..., \exp(-W^\top X - b_m)]$. Symmetric pairing preserves information in both positive and negative directions, avoiding the information loss caused by ReLU discarding negative values and the extreme variations introduced by ELU+1. Empirically, linear attention with HFM recovers rank to 46 out of 64 (full rank), far exceeding ReLU/ELU.
    - **Design Motivation**: Low-rank output matrices in linear attention compress features into few directions, resulting in insufficient representational diversity. The trainable MLP-style structure of HFM offers greater flexibility than fixed mappings.

2. **Semi-Shared Mechanism**:

    - **Function**: Maintains representational updates within parameter sharing.
    - **Mechanism**: BERFG is divided into Shared Blocks (SB) and Receiving Blocks (RB). The shared hybrid attention in SB computes full attention and caches $A_{qk}^{(a)}, A_{map}^{(a)}$. The receiving hybrid attention in RB directly reuses the softmax attention maps from SB, while the dynamic feature mappings ($\phi(Q), \phi(K)$) of Hedgehog attention are independently recomputed at each layer.
    - **Design Motivation**: Full sharing leads to representational homogenization. Semi-sharing shares the window attention component (reducing computation) while independently updating the global attention component (preserving diversity).

3. **Large-Kernel Distillation Module (LKD)**:

    - **Function**: Expands spatial receptive field with low parameter overhead.
    - **Mechanism**: Channels are split into a fine-grained subset $F_{fg}$ ($\max(C/4, 16)$ channels) and a coarse-grained subset $F_{cg}$. A three-branch feature extractor (TFE) is applied only to $F_{fg}$: a channel attention branch, a 1×1→3×3→1×1 bottleneck local branch, and a hierarchical large-kernel branch using depthwise separable and dilated convolutions. $F_{cg}$ is passed through directly.
    - **Design Motivation**: Confining heavy computation to a small subset of channels proportionally reduces overall cost, while the large-kernel path efficiently expands the receptive field via dilation and depthwise separable convolutions.

### Loss & Training

L1 reconstruction loss + LDL loss + Wavelet loss. Adam optimizer ($\beta_1=0.9, \beta_2=0.99$), 64×64 crop, batch size 16. Trained on 2 × RTX 3090. ×2 scale trained from scratch for 800K iterations; ×3/×4 scales fine-tuned from ×2 for 400K iterations.

## Key Experimental Results

### Main Results

| Method | Manga109 4× PSNR | Params | MACs |
|--------|------------------|--------|------|
| UCAN-L | 31.63 | 902K | 48.4G |
| MambaIRV2-light | 31.24 | 790K | 75.6G |
| ATD-light | 31.48 | 769K | 100.1G |
| ESC | 31.54 | 968K | 149.2G |
| RCAN | 31.22 | 15592K | 917.6G |

### Ablation Study

| Configuration | Set5 PSNR | Urban100 PSNR | Notes |
|---------------|-----------|---------------|-------|
| w/o HPA | 38.27 | 32.90 | Missing large-window local attention |
| HPA 16×16 window | 38.32 | 33.04 | Default 32×32 is superior |
| ReLU mapping | 38.33 | 33.16 | Low rank |
| Hedgehog mapping | 38.34 | 33.22 | High rank, +0.06 dB |
| Full sharing | 38.29 | 32.89 | Representational homogenization |
| Semi-sharing | 38.34 | 33.22 | Maintained diversity, +0.33 dB |

### Key Findings

- UCAN outperforms MambaIRV2 by 0.39 dB on Manga109 (×4) while reducing MACs by 36%.
- Hedgehog feature mapping recovers rank to 46/64; ReLU and ELU reach only ~20 and ~30, respectively.
- ERF visualization demonstrates that UCAN's effective receptive field coverage is substantially larger than that of MambaIR/MambaIRv2.
- LAM analysis shows that UCAN aggregates repeated patterns and similar structures from a broader context.

## Highlights & Insights

- **Hedgehog Attention Resolves Rank Collapse**: Symmetric exponential feature mapping restores the rank of linear attention, directly improving representational diversity.
- **Multi-Level Receptive Field Fusion**: Flash Attention (32×32 local) + Hedgehog Attention (global) + Large-Kernel Distillation (spatial structure) form a complementary three-way design.
- **Extreme Efficiency**: With only 705K parameters and 38.1G MACs, UCAN achieves performance comparable to RCAN (15.6M parameters, 918G MACs).

## Limitations & Future Work

- Flash Attention relies on specific CUDA implementations and may not be available on certain hardware platforms.
- The number of feature pairs $m$ in the Hedgehog feature mapping requires tuning.
- Validation is limited to SR tasks; generalizability to other image restoration tasks remains to be verified.

## Related Work & Insights

- **vs. OmniSR**: OmniSR expands the receptive field via Grid Attention but with limited efficiency; UCAN achieves greater efficiency.
- **vs. MambaIRv2**: MambaIRv2 combines Swin Transformer with SSM; UCAN replaces SSM with Hedgehog linear attention.
- **vs. ATD-light**: ATD employs an adaptive token dictionary, while UCAN uses large-kernel distillation and Hedgehog attention, achieving lower MACs.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of Hedgehog attention in SR with rank recovery analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks, three scales, ERF/LAM analysis, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with in-depth analysis of attention mechanisms.
- Value: ⭐⭐⭐⭐ Establishes a new SOTA direction for lightweight SR.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks
description: >-
  [NEURIPS2025][Image Restoration][Spiking Neural Networks] This paper proposes SpikeSR, the first attention-based spiking neural network (SNN) framework for remote sensing image super-resolution. By incorporating Spiking Attention Blocks (SAB) that combine Hybrid Dimensional Attention (HDA) and Deformable Similarity Attention (DSA), SpikeSR achieves state-of-the-art performance on AID/DOTA/DIOR while maintaining high computational efficiency.
tags:
  - NEURIPS2025
  - Image Restoration
  - Spiking Neural Networks
  - Remote Sensing Super-Resolution
  - Attention Mechanism
  - Deformable Similarity Attention
  - Energy-Efficient AI
date: 2026-05-08
content_hash: ab567e804c33466f
---

# Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks

**Conference**: NEURIPS2025
**arXiv**: [2503.04223](https://arxiv.org/abs/2503.04223)
**Code**: [https://github.com/XY-boy/SpikeSR](https://github.com/XY-boy/SpikeSR)
**Area**: Image Restoration
**Keywords**: Spiking Neural Networks, Remote Sensing Super-Resolution, Attention Mechanism, Deformable Similarity Attention, Energy-Efficient AI

## TL;DR
This paper proposes SpikeSR, the first attention-based spiking neural network (SNN) framework for remote sensing image super-resolution. By incorporating Spiking Attention Blocks (SAB) that combine Hybrid Dimensional Attention (HDA) and Deformable Similarity Attention (DSA), SpikeSR achieves state-of-the-art performance on AID/DOTA/DIOR while maintaining high computational efficiency.

## Background & Motivation

**Background**: High-resolution remote sensing images (RSI) are critical for downstream tasks, yet sensor-imposed resolution limits remain a fundamental constraint. Deep learning-based SR methods (CNN/Transformer) have achieved notable progress but incur substantial computational overhead, making large-scale deployment in remote sensing scenarios difficult.

**Limitations of Prior Work**:
   - CNN-based SR methods (EDSR, RCAN, etc.) focus on network design but exhibit high computational complexity, particularly in exhaustive non-local modeling operations;
   - Transformer-based SR methods (SwinIR, HiT-SR, etc.) offer global modeling capacity but still carry large parameter counts and FLOPs;
   - SNNs, as third-generation neural networks, offer inherent energy efficiency advantages but remain almost entirely unexplored for pixel-level regression tasks such as SR.

**Key Challenge**: The binary spike signals of SNNs inevitably cause per-pixel information loss (spiking degradation), and insufficiently optimized membrane potential dynamics limit the representational capacity of SNNs for SR.

**Goal**:
   - Introduce SNNs into remote sensing SR to leverage their energy efficiency advantages
   - Optimize membrane potentials via attention mechanisms to enhance SNN representational capacity
   - Achieve or surpass ANN-level performance while maintaining low FLOPs

**Key Insight**: A key observation—even in severely degraded remote sensing images, LIF neurons maintain vigorous membrane potential fluctuations (active learning state), suggesting that SNNs possess an inherent sensitivity to high-frequency information (Figure 1a).

**Core Idea**: Regulate SNN membrane potentials through attention mechanisms (temporal-channel and deformable spatial), enabling spiking neural networks to achieve state-of-the-art performance in remote sensing SR for the first time with greater efficiency.

## Method

### Overall Architecture

- **Input**: LR remote sensing image replicated along the temporal dimension for $T$ steps (default $T=4$)
- **Shallow Feature Extraction**: $3\times3$ convolution
- **Deep Feature Extraction**: $m$ Spiking Attention Groups (SAGs), each containing $n$ SABs with residual connections
- **Fusion**: Fusion Block (FB) converts discrete spike sequences into continuous-valued features
- **Reconstruction**: PixelShuffle + $3\times3$ convolution to generate the SR output

### Key Designs

1. **Spiking Attention Block (SAB)**:

    - **Function**: Optimizes feature representation within the SNN framework
    - **Mechanism**: Dual-branch parallel structure — Branch 1 uses two stacked SCBs (SNN Convolutional Blocks: LIF neuron → spiking convolution → tdBN); Branch 2 uses a standard CNN convolution. The two branches are summed and processed through HDA and DSA with a residual connection: $\mathbf{X}^{t,n} = \mathbf{X}^{t,n-1} + \text{DSA}(\text{HDA}(\bar{\mathbf{X}}_1^{t,n} + \bar{\mathbf{X}}_2^{t,n}))$
    - **Design Motivation**: The CNN branch compensates for information loss caused by binary SNN signals (a core challenge in SNN-based SR); the attention modules optimize membrane potentials to make spiking activity more informative.

2. **Hybrid Dimensional Attention (HDA)**:

    - **Function**: Jointly modulates spike responses along the temporal and channel dimensions
    - **Mechanism**: Employs temporal-channel joint attention (TJCA). Unlike prior approaches that treat temporal and channel attention independently, HDA bridges dependencies across both dimensions, enabling joint feature correlation learning.
    - **Design Motivation**: SNN spike signals inherently possess a temporal dimension ($T$ time steps), necessitating selective enhancement of useful signals simultaneously across temporal and channel dimensions.

3. **Deformable Similarity Attention (DSA)**:

    - **Function**: Exploits global self-similarity in remote sensing images as an SR prior
    - **Mechanism**: (1) Multi-scale feature pyramid via bilinear interpolation downsampling; (2) Patch-level self-similarity computation: average-pool each patch → reshape → dot-product similarity matrix → concatenate multi-scale similarity scores; (3) Deformable convolution to correct geometric misalignment among the most similar patches: $\mathbf{F}^D(p_0) = \sum_{p_m \in \mathcal{R}} \omega(p_m) \cdot \mathbf{F}(p_0 + p_m + \Delta p_m)$; (4) Cross-attention fusion: $Q$ from deformed features, $K, V$ from original features.
    - **Design Motivation**: Remote sensing images exhibit repetitive patterns of the same scene type (e.g., building clusters, farmland) across different spatial locations, making self-similarity a strong prior. However, exhaustive pixel-wise non-local attention is computationally prohibitive; patch-level operations are both efficient and effective. Deformable convolution handles geometric transformations between matched patches.

4. **Fusion Block (FB)**:

    - **Function**: Adaptively aggregates discrete spike sequences into continuous pixel values
    - **Mechanism**: First, temporal attention-weighted aggregation: $\mathbf{Y}_1 = \sigma(\text{TA}(\mathbf{Y})) \otimes \mathbf{Y}$; then spatial attention processes residual information: $\mathbf{Y}_2 = \sigma(\text{SA}(\mathbf{Y})) \otimes (1 - \mathbf{Y}_1)$; final output: $\mathbf{Y}_1 + \mathbf{Y}_2$.
    - **Design Motivation**: Naïve temporal averaging retains only first-order statistics; adaptive attention weighting preserves richer spatial-temporal details.

### Loss & Training

- L1 loss (pixel-level reconstruction)
- Gumbel-Softmax for non-differentiable argmax in DSA patch matching
- $T=4$ time steps for training/inference ($T=1$ used for fair FLOPs comparison)

## Key Experimental Results

### Main Results — Remote Sensing SR Performance (×4)

| Method | Params | FLOPs | AID PSNR | DOTA PSNR | DIOR PSNR | Mean PSNR |
|--------|--------|-------|----------|-----------|-----------|-----------|
| EDSR | 1518K | 50.77G | 30.65 | 33.64 | 30.63 | 31.64 |
| SwinIR-light | 897K | 23.56G | 30.83 | 33.94 | 30.85 | 31.87 |
| HiT-SR | 792K | 21.04G | 30.87 | 33.93 | 30.89 | 31.90 |
| Omni-SR | 2803K | 70.98G | 30.89 | 33.94 | 30.89 | 31.91 |
| **SpikeSR** | **1042K** | **33.05G** | **30.91** | **33.98** | **30.95** | **31.95** |
| SpikeSR-S | 472K | 15.21G | 30.86 | 33.89 | 30.89 | 31.88 |

### Ablation Study

| Configuration | PSNR↑ | Notes |
|---------------|-------|-------|
| Full SpikeSR | 31.95 | Complete model |
| w/o CNN branch | Significant drop | Pure SNN suffers severe information loss |
| w/o HDA | Drop | Temporal-channel joint attention is important |
| w/o DSA | Drop | Global self-similarity prior is critical |
| w/o Deformable Conv | Drop | Geometric correction is necessary for patch matching |

### Key Findings

- **SpikeSR comprehensively outperforms ANN methods**: Achieves state-of-the-art on all three datasets (AID/DOTA/DIOR); mean PSNR of 31.95 surpasses Omni-SR (31.91) with only 47% of its FLOPs.
- **SpikeSR-S approaches SOTA at minimal cost**: With only 472K parameters and 15.21G FLOPs, it achieves 31.88 PSNR, competitive with SwinIR-light (31.87/23.56G) but with 35% fewer FLOPs.
- **CNN branch is indispensable**: Removing the CNN branch (pure SNN) leads to a substantial performance drop, confirming that CNN compensation is necessary to address SNN information loss.
- **Value of deformable convolution in DSA**: Patch matching without geometric correction introduces hallucinated textures.

## Highlights & Insights

- **First successful application of SNNs to pixel-level regression**: Prior SNN work mainly targets classification/detection; this paper demonstrates that, through attention-based membrane potential optimization, SNNs can match or exceed ANNs on pixel-level regression (SR). This is pioneering for applying SNNs to broader low-level vision tasks.
- **Observation that spiking signals preserve high-frequency sensitivity**: Figure 1a shows that while pixel intensities in degraded images are smooth, LIF neurons still exhibit vigorous fluctuations, providing intuitive justification for SNN-based SR.
- **Patch-level non-local attention**: Reduces exhaustive pixel-wise non-local attention to patch-level similarity computation followed by deformable convolution correction, substantially lowering computational cost while preserving self-similarity modeling capacity. This design is generalizable to other tasks requiring non-local priors.
- **CNN-SNN hybrid architecture**: Rather than a pure SNN, the CNN branch pragmatically compensates for SNN information loss—a practical and effective design choice.

## Limitations & Future Work

- **Time step configuration**: $T=1$ minimizes FLOPs but sacrifices performance; $T=4$ achieves SOTA but FLOPs scale linearly with time steps. The paper lacks sufficient analysis of the trade-off across different values of $T$.
- **Remote sensing datasets only**: Generalizability is unknown, as the method is not validated on natural image SR benchmarks (e.g., DIV2K, Urban100).
- **Absence of energy efficiency quantification**: Energy efficiency is claimed as a key advantage of SNNs, but no actual power consumption or deployment results on neuromorphic hardware are reported.
- **Only ×4 super-resolution**: Performance at other upscaling factors (×2, ×8) is not evaluated.
- **Future directions**:
    - Deploy on neuromorphic chips to empirically verify energy efficiency
    - Extend to natural image SR benchmarks
    - Explore larger SNN backbones (the current maximum of 1042K parameters is relatively small)

## Related Work & Insights

- **vs. SwinIR/HiT-SR**: Transformer-based SR methods offer strong global modeling but at high FLOPs. SpikeSR achieves better performance with lower FLOPs via SNN + DSA.
- **vs. Efficient SR (IMDN/RFDN/FMEN)**: These methods reduce CNN overhead through pruning/distillation, but generally achieve lower performance than SpikeSR.
- **vs. Direct ANN-to-SNN Conversion**: Conversion methods suffer from accuracy gaps and high latency; SpikeSR employs direct training (surrogate gradients + BPTT), yielding superior results.

## Rating

- Novelty: ⭐⭐⭐⭐ First successful application of SNNs to remote sensing SR; SAB/DSA designs are innovative
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison on 3 remote sensing datasets with detailed ablations and per-scene-category analysis across 30 scene types
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and method figures are detailed, though some formulations could be more concise
- Value: ⭐⭐⭐⭐ Opens a new direction for SNNs in low-level vision tasks with practical deployment value for the remote sensing community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](../../ICCV2025/image_restoration/emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[NeurIPS 2025\] Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](../../CVPR2026/image_restoration/ucan_unified_convolutional_attention_lightweight_sr.md)
- [\[ICCV 2025\] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention](../../ICCV2025/image_restoration/consistent_time-of-flight_depth_denoising_via_graph-informed_geometric_attention.md)
- [\[NeurIPS 2025\] FIPER: Factorized Features for Robust Image Super-Resolution and Compression](fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)

</div>

<!-- RELATED:END -->

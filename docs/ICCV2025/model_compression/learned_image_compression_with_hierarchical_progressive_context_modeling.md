---
title: >-
  [Paper Note] Learned Image Compression with Hierarchical Progressive Context Modeling
description: >-
  [ICCV 2025][Model Compression][Learned image compression] This paper proposes a Hierarchical Progressive Context Model (HPCM) that partitions the latent representation into multi-scale sub-representations and encodes them sequentially from coarse to fine, combined with a cross-attention-based progressive context fusion mechanism across coding steps, enabling more efficient long-range dependency modeling and more accurate entropy parameter estimation while achieving a better trade-off between compression performance and computational complexity.
tags:
  - ICCV 2025
  - Model Compression
  - Learned image compression
  - context modeling
  - entropy coding
  - hierarchical coding
  - progressive fusion
date: 2026-05-08
content_hash: ca54fb94b03f987e
---

# Learned Image Compression with Hierarchical Progressive Context Modeling

**Conference**: ICCV 2025
**arXiv**: [2507.19125](https://arxiv.org/abs/2507.19125)
**Code**: [github.com/lyq133/LIC-HPCM](https://github.com/lyq133/LIC-HPCM)
**Area**: Image Compression
**Keywords**: Learned image compression, context modeling, entropy coding, hierarchical coding, progressive fusion

## TL;DR

This paper proposes a Hierarchical Progressive Context Model (HPCM) that partitions the latent representation into multi-scale sub-representations and encodes them sequentially from coarse to fine, combined with a cross-attention-based progressive context fusion mechanism across coding steps, enabling more efficient long-range dependency modeling and more accurate entropy parameter estimation while achieving a better trade-off between compression performance and computational complexity.

## Background & Motivation

- The core of learned image compression lies in the entropy model: more accurate probability distribution estimation → fewer bits → better compression performance.
- Conditional entropy models jointly use a hyperprior and a context model (encoding latents in groups sequentially) to capture contextual information.
- Two key issues with existing methods:
    - **Inefficient long-range dependency modeling**: Transformer architectures can capture long-range dependencies but introduce high complexity.
    - **Insufficient utilization of diverse context across coding steps**: Each step only uses the hyperprior and already-coded latents, without fully exploiting contextual information accumulated in previous steps.
- **Paper Goals**: Achieve more efficient context information acquisition through hierarchical coding schedules and progressive context fusion.

## Method

### Overall Architecture

The paper follows the standard learned image compression pipeline: analysis transform → quantization → entropy coding → decoding → synthesis transform. The core improvement lies in the context modeling component of the entropy model—HPCM is proposed to replace the conventional single-scale context model.

### Key Designs

1. **Hierarchical Coding Schedule**: The quantized latent $\hat{y}$ is divided via specialized sampling into three sub-representations at different scales: $\hat{y}^{S1}$ (4× downsampled), $\hat{y}^{S2}$ (2× downsampled), and $\hat{y}^{S3}$ (original scale). Encoding proceeds sequentially from the smallest scale, progressively modeling dependencies from long-range to short-range. Key advantage: large receptive fields can be obtained at smaller scales with lightweight operations (ERF visualizations confirm that the effective receptive field of $g_{ep}^{S1}$ is significantly larger than that of $g_{ep}^{S3}$), avoiding the high complexity of global attention at the original resolution. After encoding $\hat{y}^{S1}$, its values are filled back into the corresponding positions of $\hat{y}^{S2}$, and so on. Different channel groups use different spatial partitioning strategies to enable spatial-channel interaction. The number of coding steps per scale is (2, 3, 6), totaling 11 steps.

2. **Progressive Context Fusion (PCF)**: The core innovation—integrating contextual information from the previous coding step into the current step. At step $i$, the context $C_i^{S2}$ is obtained by fusing the previous step's $C_{i-1}^{S2}$ with the entropy parameter state $\psi_{i-1}$. Fusion is implemented via cross-attention: $Q = \text{Linear}(\psi_i)$, $K,V = \text{Linear}(C_i^{S2})$, $C_{i+1}^{S2} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V + \text{Linear}(\psi_i)$. Cross-scale propagation: the small-scale context $C_6^{S2}$ is filled into corresponding positions of the large-scale $C_6^{S3}$, with the remaining positions filled using hyperprior information.

3. **Parameter-Efficient Design and Other Improvements**: Advanced network architectures (deep residual connections, non-local attention) are employed to construct the transform and context model networks. Network parameters are shared across coding steps, significantly reducing model size. The Python-C data exchange interface is optimized for more efficient arithmetic coding.

### Loss & Training

- Rate-distortion loss: $L = \mathcal{R}(\hat{y}) + \mathcal{R}(\hat{z}) + \lambda \cdot \mathcal{D}(x, \hat{x})$
- Probability distribution model: generalized Gaussian model $\mathcal{N}_\beta(\mu, \alpha)$ with $\beta=1.5$
- MSE optimization: $\lambda \in \{0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483\}$
- MS-SSIM optimization: $\lambda \in \{2.40, 4.58, 8.73, 16.64, 31.73, 60.50\}$
- Training data: Flicker2W, 256×256 patches, batch size = 32
- 2 million training steps, Adam optimizer, learning rate decayed from 1e-4 to 1e-6 in stages

## Key Experimental Results

### Main Results

**BD-Rate Performance (relative to VTM-22.0, PSNR; negative values indicate bit savings)**:

| Method | Params (M) | kMACs/pixel | Kodak | CLIC Pro | Tecnick |
|--------|------------|-------------|-------|----------|---------|
| ELIC (CVPR'22) | 36.93 | 573.88 | -3.22% | -3.89% | -4.57% |
| TCM (CVPR'23) | 76.57 | 1823.58 | -10.70% | -8.32% | -11.84% |
| MLIC++ (ICML'23) | 116.72 | 1282.81 | -15.15% | -14.05% | -17.90% |
| FLIC (ICLR'24) | 70.96 | 1096.04 | -13.20% | -9.88% | -15.27% |
| **HPCM-Base** | 68.50 | 918.57 | **-15.31%** | -14.23% | -18.16% |
| **HPCM-Large** | 89.71 | 1261.29 | **-19.19%** | **-18.37%** | **-22.20%** |

**Comparison of Different Entropy Models under the Same Transform**:

| Entropy Model | kMACs/pixel | Kodak BD-Rate |
|---------------|-------------|---------------|
| CHARM | 495.75 | +0.86% |
| DCVC-DC intra | 542.14 | -9.18% |
| **HPCM-Base** | 918.57 | **-15.31%** |

### Ablation Study

**Ablation on Hierarchical Coding Schedule**:

| Configuration | kMACs/pixel | BD-Rate |
|---------------|-------------|---------|
| HPCM-Base (2,3,6) default | 918.57 | 0.00% |
| Remove hierarchical extraction (model at original scale) | 1107.48 | +1.07% |
| Coding steps (2,3,3) | 663.90 | +2.39% |
| Coding steps (2,3,12) | 1427.91 | -2.55% |
| Coding steps (4,3,6) | 925.59 | +0.35% |

**Ablation on Progressive Context Fusion**:

| Configuration | kMACs/pixel | BD-Rate |
|---------------|-------------|---------|
| HPCM-Base (with PCF) | 918.57 | 0.00% |
| Remove progressive fusion | 872.80 | +4.71% |
| Use $\psi_i$ as progressive context | 872.80 | +1.17% |

### Key Findings

- **Progressive context fusion is the most critical component**: removing it leads to a 4.71% performance drop, indicating that cross-step context accumulation is essential for entropy estimation.
- The hierarchical coding schedule effectively enables long-range dependency modeling: larger effective receptive fields are naturally obtained at smaller scales (verified via ERF visualizations).
- Cross-attention fusion outperforms simply using the entropy parameter state (4.71% vs. 1.17%), highlighting the importance of accurate context integration mechanisms.
- HPCM-Large saves approximately 19–22% in bit-rate over VTM-22.0 on Kodak/Tecnick, achieving academic state-of-the-art.
- The number of coding steps has the most significant impact at the largest scale S3 (more latents → greater benefit), but increasing to (2,3,12) yields only an additional 2.55% gain at substantially higher computational cost.
- The attention maps of the PCF module concentrate on high-bit-rate regions (complex textures), indicating that the model learns to allocate more context modeling effort to difficult regions.

## Highlights & Insights

- **Elegant design of the hierarchical schedule**: long-range dependencies are modeled without requiring global attention as in Transformers—lightweight operations at smaller scales naturally yield large effective receptive fields.
- **The progressive fusion mechanism** shares conceptual similarity with hidden state propagation in RNNs, but achieves more flexible information selection via cross-attention.
- The cross-scale context propagation is elegantly designed: accumulated context from smaller scales is directly filled into corresponding positions of larger scales, with hyperprior information supplementing the remaining positions.
- Parameter sharing across coding steps reduces model size while preserving context-aware adaptability.
- Optimizing the Python-C interface for arithmetic coding is also a practical engineering contribution.

## Limitations & Future Work

- Encoding/decoding time (~80–90 ms) is competitive but still has room for improvement, particularly due to the serial bottleneck of arithmetic coding.
- The (2,3,6) step allocation is manually chosen as a trade-off; automated learning-based allocation could be explored.
- Optimization is currently focused on MSE/MS-SSIM; perceptual quality and subjective evaluation remain largely unexplored.
- Integration with variable-rate techniques to achieve multi-rate compression with a single model is a promising direction.

## Related Work & Insights

- The checkerboard model (He et al.) and channel-conditional context model (Minnen et al.) form the foundation of context modeling in this field.
- MLIC++'s multi-reference context model inspired the idea of diverse context utilization, but HPCM achieves this more efficiently through hierarchical coding and progressive fusion.
- The hierarchical coding concept shares philosophical roots with multi-resolution approaches in traditional coding (e.g., wavelet decomposition in JPEG2000).
- The cross-attention progressive fusion mechanism is potentially transferable to other tasks requiring multi-step context accumulation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of hierarchical coding and progressive fusion is novel, and the cross-attention fusion design is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation on 3 datasets with comparisons against VTM and multiple state-of-the-art methods; well-designed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Figures are clear (especially the coding schedule diagram in Fig. 2); the method is described in a well-organized manner.
- **Value**: ⭐⭐⭐⭐⭐ State-of-the-art compression performance with a favorable performance-complexity trade-off; valuable for both academic research and industrial applications.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[NeurIPS 2025\] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](../../NeurIPS2025/model_compression/4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)
- [\[AAAI 2026\] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](../../AAAI2026/model_compression/dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)
- [\[ICCV 2025\] DLF: Extreme Image Compression with Dual-generative Latent Fusion](dlf_extreme_image_compression_with_dual-generative_latent_fusion.md)
- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](../../AAAI2026/model_compression/hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)

<!-- RELATED:END -->

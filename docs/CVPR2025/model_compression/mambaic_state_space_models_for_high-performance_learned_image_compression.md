---
title: >-
  [Paper Note] MambaIC: State Space Models for High-Performance Learned Image Compression
description: >-
  [CVPR 2025][Model Compression][learned image compression] Integrates SSM into both the non-linear transform and the context model of learned image compression for the first time. It enhances channel-spatial context modeling through the VSS block and eliminates spatial redundancy using window-based local attention, saving 12.52% BD-rate over VVC on the Kodak dataset, with even more pronounced advantages in high-resolution image compression.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "learned image compression"
  - "state space model"
  - "Mamba"
  - "entropy model"
  - "context modeling"
  - "window-based local attention"
date: 2026-05-08
content_hash: f29234dc4ba9d1a1
---

# MambaIC: State Space Models for High-Performance Learned Image Compression

**Conference**: CVPR 2025  
**arXiv**: [2503.12461](https://arxiv.org/abs/2503.12461)  
**Code**: [GitHub](https://github.com/AuroraZengfh/MambaIC)  
**Area**: Model Compression  
**Keywords**: learned image compression, state space model, Mamba, entropy model, context modeling, window-based local attention

## TL;DR

Integrates SSM into both the non-linear transform and the context model of learned image compression for the first time. It enhances channel-spatial context modeling through the VSS block and eliminates spatial redundancy using window-based local attention, saving 12.52% BD-rate over VVC on the Kodak dataset, with even more pronounced advantages in high-resolution image compression.

## Background & Motivation

**Background**: Learned image compression (LIC) has developed rapidly, with CNN and Transformer methods surpassing traditional coding standards (BPG/VVC), but efficiency issues remain prominent in high-resolution scenarios.

**Limitations of Prior Work**:
1. Transformer methods (e.g., Contextformer) perform better than CNNs, but their computational complexity increases quadratically with the number of pixels, resulting in high latency for high resolution.
2. CNN methods (e.g., ELIC) are highly efficient but lack global modeling capabilities.
3. Existing SSM attempts (MambaVC) merely replace base blocks without adaptation for compression characteristics, leading to poor performance.
4. The context model is crucial for compression performance, but the efficiency and effectiveness of context modeling in existing methods still have room for improvement.

**Key Challenge**: How to achieve linear complexity while maintaining a global receptive field? How to fully leverage the advantages of SSM in image compression?

**Key Insight**: Tailor-design a context modeling mechanism + local attention complement for SSM to achieve a win-win in efficiency and performance.

## Method

### Overall Architecture

Standard LIC framework: The encoder $g_a$ compresses the image into a latent representation $\mathbf{y}$, the hyperprior encoder/decoder $h_a/h_s$ learns the distribution parameters, and the arithmetic encoder/decoder (AE/AD) executes the actual coding. The core components are the VSS block-based non-linear transform, the context entropy model, and the window-based local attention.

### Key Designs

**1. SSM Context Entropy Model**
- **Function**: Embeds the VSS (Visual State Space) block in channel-spatial context modeling to enhance side information representation.
- **Mechanism**:
    - Channel context $\Psi_k$: Extract channel features $\mathcal{F}_c$ from encoded channels $\hat{\mathbf{y}}^{<k}$ using VSS block + Conv.
    - Spatial context $\Phi$: Extract spatial features $\mathcal{F}_s$ from the encoded spatial neighborhood $\hat{\mathbf{y}}^k_{<i}$ using VSS block + Conv.
    - Inside the VSS block: 2D Selective Scan (SS2D) scans along 4 traversal paths $\rightarrow$ passes through SSM individually $\rightarrow$ merges back to 2D, effectively building a global receptive field.
    - Uses checkerboard mask for parallel spatial modeling (anchor/non-anchor grouping).
- **Design Motivation**: SSM balances efficiency and global information capture better than CNN/Transformer in context models (ablation shows SSM improves BD-rate by 8.71% over CNN and 5.33% over Transformer).

**2. Window-based Local Attention (WLA)**
- **Function**: Adds intra-window local attention after parameter aggregation to complement the global modeling of SSM.
- **Mechanism**: Divide patches into $w \times w$ small windows $\rightarrow$ compute attention within windows $\rightarrow$ restore original arrangement.
- **Optimal Window Size**: $8 \times 8$ (experimentally compared with $6 \times 6$, $8 \times 8$, $10 \times 10$).
- **Design Motivation**: SSM excels at global receptive fields, but local spatial redundancy needs local attention to eliminate; their complementarity makes the bitstream more compact.

**3. SSM Non-linear Transform**
- **Function**: Replaces base blocks (including residual bottleneck structures) in encoder and decoder with VSS blocks.
- **Mechanism**: VSS block = LayerNorm + Linear + DW Conv + SiLU + SS2D, integrating 2D spatial information through cross-scan and merge.
- **Design Motivation**: Establishes global dependencies right at the encoding and decoding stages to improve the quality of latent representations.

### Loss & Training

- Rate-distortion optimization: $\mathcal{L} = \lambda \mathcal{D}(\mathbf{x}, \hat{\mathbf{x}}) + \mathcal{R}(\hat{\mathbf{y}}) + \mathcal{R}(\hat{\mathbf{z}})$
- Distortion metric: MSE
- $\lambda \in \{0.0035, 0.0067, 0.013, 0.025, 0.05\}$ controls different bitrates
- Trained for 250 epochs on the Flickr30k dataset (31,783 images)
- Channel numbers $N=128$ ($\mathbf{z}$) and $M=320$ ($\mathbf{y}$), with channel grouping $K=5$

## Key Experimental Results

### Main Results — BD-Rate (Relative to VVC, lower is better)

| Method | BD-Rate | Encoding Latency (ms) | Decoding Latency (ms) |
|---|---|---|---|
| ELIC (CNN) | -3.95% | 40.76 | 45.34 |
| Contextformer (Trans.) | -5.05% | 40.00 | 44.00 |
| MambaVC (SSM) | -7.31% | 60.45 | 41.67 |
| Mixed (Trans.+CNN) | -7.39% | 127.36 | 91.44 |
| **MambaIC (Ours)** | **-12.52%** | 60.73 | **39.42** |

### Ablation Study

| Configuration | Decoding Latency (ms) | BD-Rate |
|---|---|---|
| w/o CAM (channel autoregressive) | 16.72 | -6.73% |
| w/o spatial context | 32.73 | -8.54% |
| w/o WLA (window attention) | 35.14 | -9.17% |
| **Full MambaIC** | 39.42 | **-12.52%** |

### Base Block Comparison

| Block | Decoding Latency (ms) | BD-Rate|
|---|---|---|
| CNN | 35.53 | -3.81% |
| Transformer | 48.74 | -7.19%|
| **SSM (Ours)** | 39.42 | **-12.52%** |

### Bitstream Comparison (PSNR ≈ 34.2 dB on Kodak)

| Method | Receptive Field | Bpp | ΔBpp |
|---|---|---|---|
| ELIC | Local | 0.4683 | - |
| Contextformer | Global | 0.4596 | 1.86% |
| MambaVC | Global | 0.4482 | 4.29% |
| **Ours (8×8)** | Global + Local | **0.4404** | **5.95%** |

### Key Findings

1. **Significant advantages at high resolutions**: From Kodak (768×512) to Tecnick (1200×1200) to CLIC (2048×1440), the advantages of MambaIC progressively expand, while other methods exhibit varying degrees of degradation.
2. **SSM is superior to CNN/Transformer in compression**: Under the same framework, the SSM block improves BD-rate by 8.71% compared to CNN and by 5.33% compared to Transformer.
3. **Context modeling and local attention are complementary**: Each component makes a significant contribution (5.79% + 3.98% + 3.35%), and the additional latency is controllable.
4. Comparison with Mixed (SOTA Trans.+CNN): BD-rate is 5.13% better, while encoding time is only 47.7% and decoding time is only 43.1%.

## Highlights & Insights

- Systematically introduces SSM into the context model of LIC for the first time, not as a simple replacement but with a tailored design.
- The complementary strategy of "global SSM + local attention" is intuitively verified in attention map visualizations.
- Stability in high-resolution scenarios is a key selling point for industrial applications.
- Attention map visualization clearly demonstrates how WLA helps focus on semantically relevant local regions.

## Limitations & Future Work

- Training data only uses Flickr30k (31K images), which is limited in scale.
- Encoding latency (60.73ms) is higher than ELIC (40.76ms), leaving room for improvement in encoding-side efficiency.
- Only MSE is used as the distortion metric; perceptual quality (such as LPIPS) is not considered.
- Variable bitrate (single model for multiple bitrates) schemes have not been explored.
- The theoretical analysis of SSM's advantages in compression is not deep enough.

## Related Work & Insights

- The channel-spatial hybrid context model proposed by ELIC is the foundation of the context modeling in this work.
- MambaVC made the first attempt to use SSM for compression but lacked adaptation; this work demonstrates that tailored design is crucial.
- Window attention (Swin Transformer style) is highly effective in eliminating local redundancy in compression scenarios.
- Insight: Applying new architectures (SSM) to compression is not simply about "replacing the backbone"; it requires rethinking the adaptation from the perspective of entropy models.

## Rating

⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)

</div>

<!-- RELATED:END -->

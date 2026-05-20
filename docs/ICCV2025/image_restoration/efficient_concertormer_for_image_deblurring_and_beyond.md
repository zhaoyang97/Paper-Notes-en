---
title: >-
  [Paper Note] Efficient Concertormer for Image Deblurring and Beyond
description: >-
  [ICCV 2025][Image Restoration][Image deblurring] This paper proposes Concertormer, which decomposes self-attention into a global Concertino component and a local Ripieno component…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Image deblurring"
  - "self-attention"
  - "linear complexity"
  - "Transformer"
  - "feed-forward network"
date: 2026-05-08
content_hash: 913b60b8a4964f61
---

# Efficient Concertormer for Image Deblurring and Beyond

**Conference**: ICCV 2025
**arXiv**: [2404.06135](https://arxiv.org/abs/2404.06135)  
**Code**: Coming soon  
**Area**: Image Restoration
**Keywords**: Image deblurring, self-attention, linear complexity, Transformer, feed-forward network

## TL;DR

This paper proposes Concertormer, which decomposes self-attention into a global Concertino component and a local Ripieno component, and further introduces a Cross-Dimensional Communication module and a Gated Depthwise Convolution MLP. The method achieves global-local feature modeling at linear complexity, attaining state-of-the-art performance on image deblurring and other restoration tasks.

## Background & Motivation

Transformers have achieved remarkable success in high-level vision and NLP, but the quadratic complexity of self-attention with respect to image resolution renders it prohibitively expensive for high-resolution image restoration. Existing approaches fall into two categories, each with notable limitations:

**Window Multi-Head Self-Attention (W-MSA)**: partitions feature maps into non-overlapping $k \times k$ blocks and computes attention only within each block. While this reduces complexity, it entirely ignores inter-block relationships, resulting in insufficient global modeling capacity. Even with shifted window techniques, a sufficient number of layers must be stacked to indirectly achieve a global receptive field.

**Transposed Self-Attention (Transposed SA)**: computes attention along the channel dimension rather than the spatial dimension, reducing complexity to $\mathcal{O}(hw)$. However, this approach **discards spatial connectivity information**—the paper provides an elegant argument showing that randomly permuting the columns of the Q and K matrices does not affect the result of transposed self-attention, demonstrating that it is fundamentally insensitive to spatial positional relationships.

The core motivation is: can one design a self-attention mechanism with **linear complexity** that **simultaneously captures local and global relationships**? Concertormer draws on musical terminology—the soloist group (Concertino) and the full ensemble (Ripieno)—to decompose attention into two complementary components that address this challenge.

## Method

### Overall Architecture

Concertormer adopts a multi-scale U-Net architecture. The input image is bilinearly downsampled to produce four scales ($\mathbf{I}_0$ through $\mathbf{I}_3$), each of which is fed into the encoder after channel expansion via $3 \times 3$ convolutions. Skip connections between the encoder and decoder are realized through cross-attention rather than simple addition or concatenation. The encoder reduces resolution and increases channels via stride-2 $2 \times 2$ convolutions; the decoder increases resolution via $1 \times 1$ convolutions followed by pixel-shuffle. The fundamental building block at each stage consists of **Concerto Self-Attention (CSA)** and **Gated-Dconv MLP (gdMLP)**, merged into a single-stage design.

### Key Designs

1. **Concerto Self-Attention (CSA)**:

    - **Function**: Decomposes self-attention into a globally shared Concertino component and a locally specific Ripieno component, computing attention along both the spatial and channel dimensions.
    - **Mechanism**: After partitioning Q, K, V into $k \times k$ blocks, the Concertino component $C$ **sums/averages** attention maps across all blocks to capture general global spatial relationships:
    $C = \text{softmax}\left(\sum_i Q_i^c K_i^{c\top} / \beta\right)$
      The Ripieno component $R_i$ computes the **deviation** of each block from the average to compensate for information loss:
    $R_i = \text{softmax}\left((Q_i^r K_i^{r\top} - \overline{Q_i^r K_i^{r\top}}) / \alpha\right)$
      Channels are split into two halves dedicated to the Concertino and Ripieno computations, respectively, and the results are concatenated for output. This decomposition propagates global information to each local block through $C$, while each block retains its distinctive local details. The attention maps are represented as tensors: $\mathbf{R}^s \in \mathbb{R}^{n \times k^2 \times k^2}$ and $\mathbf{C}^s \in \mathbb{R}^{d_s/2 \times k^2 \times k^2}$, naturally introducing additional dimensions.
    - **Design Motivation**: W-MSA entirely ignores inter-block relationships, while transposed SA discards spatial information. CSA addresses both limitations by providing global context through Concertino and locally differentiated details through Ripieno, achieving complexity of $\mathcal{O}(hw)$, linear in image size.

2. **Cross-Dimensional Communication (CDC)**:

    - **Function**: Establishes connections across the additional dimensions introduced by CSA, enhancing the expressiveness of the attention maps.
    - **Mechanism**: For the Ripieno tensor, it is reshaped into the form $t \times h/k \times w/k \times k^4$, and a $3 \times 3 \times 1$ convolution $\mathbf{W}^{r_s}$ is applied to linearly combine information across the block dimension:
    $\mathbf{R}^s = \text{softmax}\left(\mathbf{W}^{r_s}(\mathbf{Q}^{r_s} \times \mathbf{K}^{r_s\top})\right)$
      For the Concertino tensor, a fully connected layer $\mathbf{W}_p^{c_s}$ performs a linear projection along its constant dimension. A beneficial side effect is that the convolution operation replaces the global mean with a **local mean** (the neighborhood average covered by the kernel), which is more suitable for modeling local details.
    - **Design Motivation**: In CSA, attention maps across different heads and blocks are computed independently. CDC allows information to flow across these dimensions, substantially enlarging the receptive field (diffusion index: 39.15 vs. 20.51).

3. **Channel CSA**:

    - **Function**: Extends Concerto Self-Attention to the channel dimension.
    - **Mechanism**: Symmetrically to spatial CSA, the channel dimension is also decomposed into Ripieno $\mathbf{R}^c$ and Concertino $\mathbf{C}^c$ components. Since positional information is encoded in the $n$ dimension of $\mathbf{R}^c$ and the $k^2$ dimension of $\mathbf{C}^c$, respectively, Channel CSA can perceive spatial positions, overcoming the limitation of the original transposed self-attention.
    - **Design Motivation**: Transposed SA is efficient but lacks spatial awareness. By introducing the Concerto decomposition along the channel dimension, the method retains efficiency while resolving the spatial invariance issue.

4. **Gated Depthwise Convolution MLP (gdMLP)**:

    - **Function**: Replaces the conventional two-stage Transformer design (SA + FFN) by merging self-attention and the feed-forward network into a single stage.
    - **Mechanism**:
    $\text{gdMLP}(\mathbf{X}) = \mathbf{W}_p^g\left((\text{SCA}(\mathbf{X}^A) + \mathbf{U}) \odot \mathbf{Z}\right)$
      Here, $\mathbf{U} = \mathbf{W}_d^u(\mathbf{W}_p^u(\mathbf{X}))$ extracts features via depthwise convolution, $\mathbf{Z} = \mathbf{W}_p^z \mathbf{X}$ serves as a gating signal, and $\mathbf{X}^A$ is the CSA output weighted by Simplified Channel Attention (SCA). The depthwise convolution also compensates for boundary discontinuities introduced by non-overlapping block partitioning.
    - **Design Motivation**: The role of FFN in vision tasks remains unclear, and the two-stage design limits flexibility. gdMLP integrates attention and feed-forward computation through a gating mechanism, reducing overall complexity.

### Loss & Training

- $\ell_1$ loss in both the spatial and frequency domains, computed simultaneously across all four scales
- AdamW optimizer ($\beta_1 = \beta_2 = 0.9$, weight decay $10^{-3}$)
- Progressive training: from $128 \times 128$ to $256 \times 256$ to $320 \times 320$, with 200K iterations per stage
- Test-time Local Converter (TLC) applied at inference to further boost performance

## Key Experimental Results

### Main Results

| Dataset | Metric | Concertormer | FFTformer (Prev. SOTA) | Gain |
|--------|------|-------------|---------------------|------|
| GoPro | PSNR/SSIM | **34.42/0.971** | 34.21/0.969 | +0.21 dB |
| HIDE | PSNR/SSIM | **32.12/0.951** | 31.62/0.946 | +0.50 dB |
| RealBlur-R | PSNR/SSIM | **40.78/0.977** | 40.11/0.973 | +0.67 dB |
| RealBlur-J | PSNR/SSIM | **33.51/0.945** | 32.62/0.933 | +0.89 dB |

State-of-the-art performance is also achieved on image deraining: average 34.60/0.943 (vs. Restormer 34.16/0.937).

### Ablation Study

| Configuration | PSNR | SSIM | FLOPs(G) | Notes |
|------|------|------|----------|------|
| Model 1 (gdMLP baseline) | 32.35 | 0.951 | 41.22 | No self-attention |
| +Spatial Ripieno | 32.58 | 0.953 | - | +0.23 dB |
| +Spatial CSA (R+C) | 33.11 | 0.958 | 119.34 | Concertino contributes +0.53 dB |
| +Channel CSA | 33.20 | 0.958 | 118.33 | Spatial + channel fusion |
| +SCA | 33.31 | 0.959 | 118.57 | +0.11 dB with only 0.2% more FLOPs |
| +CDC (full model) | **33.53** | **0.961** | 116.79 | CDC adds +0.22 dB |
| Two-stage FFN design | 31.90 | 0.945 | 116.81 | 1.6 dB below gdMLP |

### Key Findings

- Replacing Restormer's transposed SA with CSA improves PSNR on GoPro by 0.4 dB (32.92→33.32) while reducing FLOPs by 0.5%
- CDC increases the diffusion index (a measure of receptive field size) from 20.51 to 39.15, nearly doubling it
- The single-stage gdMLP design outperforms the conventional two-stage design by 1.6 dB PSNR

## Highlights & Insights

- **Musical metaphor design**: The intuition of decomposing attention into Concertino (soloist/global) and Ripieno (ensemble/local) is elegant and intuitive, analogous to signal decomposition into mean and residual in the frequency domain
- **Linear complexity**: Through block-wise computation and channel splitting, global modeling capacity is preserved without sacrificing linear complexity
- **Elegance of CDC**: By introducing a learnable linear combination along additional dimensions, CDC substantially enlarges the receptive field at negligible cost (0.36% FLOPs)
- **Single-stage design**: Challenges the necessity of FFN in vision Transformers, demonstrating that a gated MLP can more effectively serve this role
- **Plug-and-play capability**: CSA can be used as a drop-in replacement for the self-attention module in existing methods (e.g., Restormer), offering broad generalizability

## Limitations & Future Work

- The block size $k=8$ is fixed; adaptive partitioning strategies could be explored
- The global averaging operation in Concertino may be insufficiently flexible for extreme spatially non-uniform degradations
- Multi-scale inputs require additional downsampling and convolution, increasing encoder-side complexity
- The paper primarily validates on deblurring and deraining tasks; evaluation on other restoration tasks (e.g., super-resolution, denoising) is insufficient

## Related Work & Insights

- Connected to NAFNet's SCA—both use simplified channel attention to balance different components
- The Concerto decomposition idea can be generalized to other tasks requiring global-local modeling, such as video understanding and 3D point cloud processing
- The practice in CDC of replacing the global mean with convolution inspires a "hierarchical aggregation" paradigm for attention design

## Rating

- Novelty: ⭐⭐⭐⭐ The Concerto decomposition is a novel idea, though the overall contribution remains within the scope of Transformer improvements
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablation studies are exceptionally detailed, including diffusion index analysis and LAM visualizations
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, though the notation is dense and requires careful cross-referencing
- Value: ⭐⭐⭐⭐ The plug-and-play nature of CSA and its linear complexity give the method solid practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Blind Noisy Image Deblurring Using Residual Guidance Strategy](blind_noisy_image_deblurring_using_residual_guidance_strateg.md)
- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](../../CVPR2026/image_restoration/beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] BluRef: Unsupervised Image Deblurring with Dense-Matching References](../../CVPR2026/image_restoration/bluref_unsupervised_image_deblurring_with_dense-matching_references.md)

</div>

<!-- RELATED:END -->

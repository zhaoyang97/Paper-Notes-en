---
title: >-
  [Paper Note] Progressive Focused Transformer for Single Image Super-Resolution
description: >-
  [CVPR 2025][Image Restoration][Super-Resolution] PFT proposes a Progressive Focused Attention (PFA) mechanism, which transfers the Hadamard product of attention maps between adjacent Transformer layers to filter out irrelevant tokens layer-by-layer and enhance the weights of key tokens, achieving state-of-the-art performance on super-resolution tasks while significantly reducing computational overhead.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Super-Resolution"
  - "Sparse Attention"
  - "Transformer"
  - "Progressive Focused"
  - "Attention Transfer"
date: 2026-05-08
content_hash: bcdb7e67542fe149
---

# Progressive Focused Transformer for Single Image Super-Resolution

**Conference**: CVPR 2025  
**arXiv**: [2503.20337](https://arxiv.org/abs/2503.20337)  
**Code**: [https://github.com/LabShuHangGU/PFT-SR](https://github.com/LabShuHangGU/PFT-SR)  
**Area**: Image Restoration  
**Keywords**: Super-Resolution, Sparse Attention, Transformer, Progressive Focused, Attention Transfer  

## TL;DR

PFT proposes a Progressive Focused Attention (PFA) mechanism, which transfers the Hadamard product of attention maps between adjacent Transformer layers to filter out irrelevant tokens layer-by-layer and enhance the weights of key tokens, achieving state-of-the-art performance on super-resolution tasks while significantly reducing computational overhead.

## Background & Motivation

**Background**: Transformer-based super-resolution methods (such as SwinIR, HAT, ATD) utilize self-attention mechanisms to capture long-range dependencies for restoring high-resolution details. Due to the quadratic complexity of self-attention, most methods restrict attention within local windows.

**Limitations of Prior Work**: Existing methods face a dilemma. One category of methods (HAT, ATD) attempts to expand the window size or introduce external information to obtain more token interactions, but more tokens bring larger computational overhead. Another category of methods (NLSA, DRSformer) uses sparse attention to filter out irrelevant tokens; however, they still need to calculate the similarity of all token pairs first before selecting the top-k, failing to exclude irrelevant tokens before computation.

**Key Challenge**: Identifying irrelevant tokens before similarity calculation and skipping their computation remains a critical unsolved problem. Both standard and sparse attention require full calculation of the similarity matrix, which limits the potential of using larger windows.

**Goal**: Design an attention mechanism that can filter out irrelevant tokens before computing similarity, thereby achieving better aggregation effects with less computation over larger windows.

**Key Insight**: The authors observe a key fact—if a token is deemed irrelevant in previous layers (with a very small attention weight), it is highly likely to remain irrelevant in subsequent layers. Therefore, the attention map from previous layers can be used to guide subsequent layers in skipping computation.

**Core Idea**: Connect the attention maps of adjacent layers via Hadamard product, allowing the attention to "progressively focus" across layers—consistently highly relevant tokens have their weights enhanced layer-by-layer, while low-relevance tokens have their weights decayed to zero, thereby achieving pre-computation filtering.

## Method

### Overall Architecture

PFT follows an encoder-reconstructor architecture similar to SwinIR, HAT, etc., consisting of 6 PFA Blocks. Unlike standard Transformer Blocks, multiple attention layers within the PFA Block share and transfer attention maps, forming a progressive dense-to-sparse process. The input is a low-resolution image, and the output is the high-resolution reconstruction result. The window size is 32×32 (much larger than SwinIR's 8×8), allowing the model to utilize a wider range of information.

### Key Designs

1. **Progressive Attention Across Layers**:

    - **Function**: Transfer the attention weights from the previous layer to the current layer, performing cross-layer cumulative filtering of attention.
    - **Mechanism**: The final attention map of the current layer is obtained by taking the Hadamard product of the calculated attention map $\mathbf{A}_{cal}^l$ and the previous layer's attention map $\mathbf{A}^{l-1}$, followed by normalization: $\mathbf{A}^l = Norm(\mathbf{A}^{l-1} \odot \mathbf{A}_{cal}^l)$. This means only token pairs that consistently exhibit high similarity across multiple layers can maintain large weights, while a small weight in any single layer will be amplified (attenuated) during the multiplication process.
    - **Design Motivation**: Standard self-attention determines weights solely based on single-step similarity computation, with limited capability to distinguish between highly relevant and lowly relevant tokens. Through multi-layer accumulation, PFA makes a more comprehensive evaluation of token relationships.

2. **Sparse Matrix Multiplication**:

    - **Function**: Skip similarity computation in the current layer using zeroed positions in the previous layer's attention map.
    - **Mechanism**: Since the final attention map will be multiplied by the previous layer's attention map, positions that already have zero weight in the previous layer do not need to be computed. By maintaining a sparse index matrix $\mathbf{I}^{l-1}$, the SMM operation only computes the dot product of $Q^l(i,:)$ and $K^l(j,:)^T$ for positions where $\mathbf{I}^{l-1}(i,j)=1$. Each layer retains top-$K^l$ non-zero values, with $K^l = \alpha K^{l-1}$ ($\alpha < 1$), achieving a layer-by-layer decreasing focus range.
    - **Design Motivation**: This directly achieves "filtering before computation"—instead of computing all similarities first and then filtering, it skips computations directly based on irrelevant positions already determined in previous layers, reducing computational complexity exponentially from $O(W^2)$. A specialized CUDA kernel was developed to efficiently implement sparse multiplication.

3. **Progressive Focused Resource Allocation Strategy**:

    - **Function**: Systematically arrange computational resources for each layer, with dense shallow layers and sparse deep layers.
    - **Mechanism**: The first layer starts with $K^1 = N$ (all tokens in the window), using standard self-attention to compute the complete attention map as an initial foundation. Subsequent layers progressively reduce the retained count according to $K^l = \alpha K^{l-1}$. Specifically, it is configured to retain [1024, 256, 128, 64, 32, 16] attention values across the 6 blocks respectively.
    - **Design Motivation**: Shallow layers need wide exploration to avoid early exclusion of important tokens, while deep layers have gathered enough information to focus boldly. This resource allocation allows PFT to use an ultra-large window of 32×32, while maintaining a computational cost comparable to small-window methods.

### Loss & Training

PFT is trained using standard L1 pixel loss. The model follows the classic SR training pipeline: trained on the DF2K dataset with input LR patch size of 64×64. PFT adopts a shift-window strategy similar to SwinIR, with attention alternately transferred between odd and even layers. LePE (Locally-enhanced Positional Encoding) is incorporated into the attention calculation.

## Key Experimental Results

### Main Results

| Method | Parameters | FLOPs | Set5 (×2) | Urban100 (×2) | Manga109 (×2) |
|------|--------|-------|-----------|-------------|-------------|
| SwinIR | 11.8M | 3.04T | 38.42 | 33.81 | 39.92 |
| HAT | 20.6M | 5.81T | 38.63 | 34.45 | 40.26 |
| ATD | 20.1M | 6.07T | 38.61 | 34.70 | 40.37 |
| **Ours** | **19.6M** | **5.03T** | **38.68** | **34.90** | **40.49** |

At ×3 scale: PFT reaches 30.56 dB on Urban100, surpassing ATD (30.46) and IPG (30.36).

### Ablation Study

| Configuration | Description | PSNR Impact |
|------|------|----------|
| Standard Self-Attention | Without PFA | Baseline |
| Top-k Sparse Attention | Use top-k but no transfer | Better than standard SA |
| Progressive Attention (No sparsity) | Product transfer but do not skip computation | Better than top-k |
| PFA (Full) | Progressive focused + Sparse Matrix Multiplication | Best |

### Key Findings

- PFT achieves the best PSNR/SSIM on all 5 benchmark datasets while requiring fewer parameters (19.6M) and FLOPs (5.03T) than HAT and ATD.
- The gain brought by progressive attention transfer is greater than simple top-k sparse selection—validating the superiority of cross-layer information accumulation compared to single-layer selection.
- The improvement is most significant on texture-rich/structurally diverse datasets like Urban100 (at ×2, ours is 0.20 dB higher than ATD), indicating that PFA is more adept at utilizing long-range structural similarities.
- Setting $\alpha=0.5$ reduces the computational complexity to 6.25% after 4 steps of decay, making the extremely large 32×32 window feasible.

## Highlights & Insights

- **Filtering before computation** is the most ingenious design of this work. Unlike top-k which computes first and then selects, PFA directly skips unnecessary computations, representing a qualitative leap in the concept of sparse attention. Essentially, it uses "cheap" information from previous layers to guide the "expensive" computations of subsequent layers.
- The concept of **attention map product transfer** can be generalized to other vision Transformer architectures. Any task that requires attention over a large range (e.g., video understanding, dense prediction) can benefit from this "coarse-to-fine" progressive focus strategy.
- The SMM implemented via a custom CUDA kernel is a key engineering contribution that translates theoretical advantages into practical speedups.

## Limitations & Future Work

- The progressive focusing strategy of PFA assumes that once a token is marked as irrelevant, it is permanently excluded. However, some tokens might become important in deeper layers (e.g., distant semantically relevant patches).
- The focusing ratio $\alpha$ is globally fixed, whereas different image regions (texture-rich vs. smooth areas) may require different decay rates.
- The paper only validates the method on image super-resolution; whether it can be effectively transferred to other low-level vision tasks like denoising or deblurring requires further validation.
- Although the 32×32 window is large, it is still fixed. Adaptive window sizes could be explored in the future.

## Related Work & Insights

- **vs HAT**: HAT expands the receptive field by combining channel attention and window attention, but still uses dense attention within the window. PFT achieves more efficient information interaction within a larger window through sparsification.
- **vs ATD**: ATD introduces an external token dictionary to compensate for the insufficient information of local windows. PFT optimizes the attention mechanism itself from within, reducing redundant computations to support larger windows.
- **vs DRSformer**: DRSformer uses learnable top-k selection, but still requires computing the complete similarity matrix before sparsifying. PFT filters before computation, making it more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ Cross-layer transfer of progressive focused attention is a powerful improvement over sparse attention.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-scale and multi-dataset comparisons, detailed complexity analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear description of methods and complete mathematical formulation.
- Value: ⭐⭐⭐⭐ Elevates the SR SOTA, and the PFA mechanism possesses good generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Augmenting Perceptual Super-Resolution via Image Quality Predictors](augmenting_perceptual_super-resolution_via_image_quality_predictors.md)
- [\[CVPR 2025\] PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution](pidsr_complementary_polarized_image_demosaicing_and_super-resolution.md)
- [\[CVPR 2025\] Gyro-based Neural Single Image Deblurring](gyro-based_neural_single_image_deblurring.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](../../ICCV2025/image_restoration/enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)

</div>

<!-- RELATED:END -->

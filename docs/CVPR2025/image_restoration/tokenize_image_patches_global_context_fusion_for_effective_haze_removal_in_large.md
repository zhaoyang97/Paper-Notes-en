---
title: >-
  [Paper Note] Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images
description: >-
  [CVPR 2025][Image Restoration][Large image dehazing] DehazeXL proposes an end-to-end framework for large image dehazing. By splitting the input image into fixed-size patches and encoding them into tokens, it leverages an efficient global attention module to fuse contextual information. This enables inference on 10240×10240 images with only 21GB of VRAM and achieves state-of-the-art (SOTA) performance on a self-built 8K dehazing dataset.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Large image dehazing"
  - "global attention"
  - "patch tokenization"
  - "high resolution"
  - "attribution analysis"
date: 2026-05-08
content_hash: 314f8acd6fbd09f9
---

# Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images

**Conference**: CVPR 2025  
**arXiv**: [2504.09621](https://arxiv.org/abs/2504.09621)  
**Code**: [https://github.com/CastleChen339/DehazeXL](https://github.com/CastleChen339/DehazeXL)  
**Area**: Image Restoration / Dehazing  
**Keywords**: Large image dehazing, global attention, patch tokenization, high resolution, attribution analysis

## TL;DR

DehazeXL proposes an end-to-end framework for large image dehazing. By splitting the input image into fixed-size patches and encoding them into tokens, it leverages an efficient global attention module to fuse contextual information. This enables inference on 10240×10240 images with only 21GB of VRAM and achieves state-of-the-art (SOTA) performance on a self-built 8K dehazing dataset.

## Background & Motivation

**Background**: Image dehazing has made significant progress, in which methods based on CNNs, GANs, Transformers, and diffusion models perform exceptionally well on small images (256-512px). However, with the advancement of sensor technologies, image resolutions in practical applications are continuously increasing (4K, 8K, or even larger), posing a VRAM bottleneck for existing methods on GPUs.

**Limitations of Prior Work**: When handling large images, mainstream methods have to compromise—either downsampling (losing high-frequency details) or cropping into slices (cutting off global context, which leads to blocking artifacts and color inconsistency). Dehazing is heavily dependent on global context (e.g., haze distribution, color reference from clear regions, and brightness consistency), and slice-based inference directly degrades the dehazing quality.

**Key Challenge**: High-resolution images require global context to accurately estimate haze distribution and scene depth, but the computational complexity of global attention scales quadratically with resolution, making it impossible to run within limited VRAM.

**Goal**: Design an end-to-end method that preserves both global context and local details when processing ultra-large images, while maintaining a manageable VRAM footprint.

**Key Insight**: Inspired by long-context handling in Large Language Models (e.g., locality-sensitive hashing, low-rank decomposition), image patches can be treated as tokens. Applying efficient global attention at the token level decouples the image size from the inputs of the encoder and decoder.

**Core Idea**: Crop the image into fixed-size patches, encode them into tokens via a shared encoder, perform efficient global attention in the token space (instead of the pixel space), and then reconstruct patch-by-patch using a decoder, thereby decoupling VRAM consumption from the image size.

## Method

### Overall Architecture

The input hazy image is split into equal-sized patches, each encoded into a feature token by a shared Encoder (Swin Transformer V2). All tokens are fed into an efficient Transformer in the bottleneck for global information fusion, allowing each token to "see" information from all other tokens. The fused tokens are then reconstructed back into clear patches one by one via a Decoder (also based on Swin Transformer V2), which are stitched to produce the final output. The Encoder and Decoder adopt an asynchronous processing strategy—processing patches sequentially in multiple mini-batches rather than simultaneously—to minimize VRAM usage.

### Key Designs

1. **Decoupled Patch Tokenization for Input Dimensions**:

    - **Function**: Decouples the input size of the encoder/decoder from the original image size, confining the VRAM footprint to the size of a single patch.
    - **Mechanism**: Splits an arbitrary-sized input image into fixed-size patches (using the same patch size for training and inference), where each patch independently passes through the shared encoder. It adopts an asynchronous mini-batch strategy, processing only a few patches at a time to significantly reduce peak VRAM.
    - **Design Motivation**: Conventional methods suffer from VRAM consumption that scales quadratically with input resolution. The key of DehazeXL is to restrict the impact of image size to the number of tokens (which scales linearly) rather than the size of the feature maps. A uniform patch size also facilitates training stability and convergence.

2. **Efficient Global Attention Bottleneck**:

    - **Function**: Fuses global contextual information (haze distribution, color consistency, and brightness) in the token space, enhancing the scene understanding of local features.
    - **Mechanism**: Constructs a Transformer block to process all patch tokens. It employs RMSNorm instead of LayerNorm to speed up computation and introduces Hyper Attention (inspired by long-context LLM techniques), which combines locality-sensitive hashing with low-rank decomposition to reduce the time and space complexity of self-attention.
    - **Design Motivation**: Dehazing requires global information to distinguish sky from dense haze regions and to preserve global color consistency. Conducting attention at the token (rather than pixel) level dramatically reduces the token count—for instance, an 8K image split into 512px patches yields only 256 tokens, making full self-attention highly feasible.

3. **Dehazing Attribution Map (DAM)**:

    - **Function**: Quantifies the contribution of each region toward the dehazing result, providing an interpretable attribution analysis.
    - **Mechanism**: Based on the integrated gradients method, it uses the clear image as a baseline and integrates gradients along a linear interpolation path from clear to hazy states, calculating the contribution of each pixel to the dehazing effect of a specific target region. A pixel intensity detector $D_{xy}(I) = \sum I_{ij}$ is used as the metric for dehazing effectiveness.
    - **Design Motivation**: Existing dehazing methods often lack interpretability, as it is unclear which regions the model relies on to remove haze. DAM reveals the effectiveness of global context modeling: DehazeXL can exploit distant, haze-free areas to assist in reconstructing nearby hazy regions, whereas slice-based methods are restricted to local information.

### Loss & Training

The L1 loss function is used along with the Adam optimizer, with an initial learning rate of 0.001 and cosine annealing decay over 500 epochs. Input images are randomly cropped to 2048×2048 during training. Competing methods are trained with 512 crops since they cannot be directly trained on a size of 2048.

## Key Experimental Results

### Main Results

| Method | 8KDehaze PSNR | 8KDehaze SSIM | 4KID PSNR | O-HAZE PSNR | Inference Time (s) |
|------|--------------|--------------|-----------|------------|------------|
| 4KDehazing (Direct) | 20.41 | 0.8664 | 18.68 | 19.3 | 1.350 |
| DehazeFormer-b | 26.83 | 0.9657 | 21.25 | 20.22 | 15.013 |
| ConvIR-b | 26.93 | 0.9775 | 21.92 | 19.61 | 8.709 |
| MixDehazeNet-b | 23.16 | 0.9284 | 23.22 | 20.67 | 13.154 |
| **DehazeXL** | **32.35** | **0.9863** | **26.62** | **21.49** | **4.617** |

### Ablation Study

| Backbone | Bottleneck Depth | PSNR | SSIM | Inference Time (s) |
|----------|-----------------|------|------|------------|
| Swin-T | 1 | 31.61 | 0.9719 | 4.511 |
| Swin-T | 2 (default) | 32.35 | 0.9863 | 4.617 |
| Swin-T | 4 | 32.40 | 0.9857 | 4.810 |
| Swin-L | 4 | 33.30 | 0.9911 | 18.25 |

### Key Findings

- On the 8KDehaze dataset, DehazeXL dominates the second-best method by over 5 dB in PSNR (32.35 dB vs. ConvIR-b's 26.93 dB) while offering faster inference.
- With FP16 inference, it can process 10240×10240 images using only 21GB of VRAM, saving 65%–80% compared to other methods.
- Global attention is highly critical: Slice-based inference methods tend to fail in sky and heavy haze regions (failing to distinguish sky from haze), whereas DehazeXL leverages global context for accurate distinction.
- DAM attribution analysis confirms that DehazeXL effectively leverages the color and spectral information of distant, haze-free regions to guide the reconstruction of nearby hazy regions.
- It also achieves the best performance on the real-world hazy dataset O-HAZE, showing excellent generalization capability.
- Increasing the bottleneck depth from 1 to 2 yields a significant performance boost, but going from 2 to 4 shows diminishing returns, suggesting that a 2-layer global attention mechanism is sufficient.

## Highlights & Insights

- **Dimensional transition from pixel space to token space**: This core idea is exceptionally ingenious—transforming global attention over millions of pixels into attention over a few hundred tokens reduces the complexity by three to four orders of magnitude. This mechanism is not only applicable to dehazing but can also benefit any low-level vision task requiring global reasoning on large images (e.g., denoising, super-resolution, enhancement).
- **Asynchronous mini-batch processing**: Utilizing asynchronous processing on the encoder/decoder side trade-offs a small amount of speed for a completely decoupled VRAM footprint, which is a highly practical engineering technique.
- **Self-built 8KDehaze dataset**: Fills the gap in ultra-high-resolution dehazing datasets, providing 10,000 remote sensing images of 8192×8192 resolution, which holds significant value for future research.

## Limitations & Future Work

- Asynchronous processing sacrifices some speed; more efficient parallel strategies can be explored in the future.
- Fixed patch size implies that boundaries between patches may suffer from discontinuity—although global attention alleviates this issue, it is not completely eliminated.
- Synthesis of hazy data might lead to a domain gap with real-world hazy conditions; the atmospheric scattering model used to generate haze in 8KDehaze may not capture the full complexity.
- Lacks comparison with diffusion-based dehazing methods.

## Related Work & Insights

- **Ours vs. 4KDehazing**: 4KDehazing is the only baseline that can directly infer on large images, but its 3-CNN architecture suffers heavy performance degradation on large-scale images (20.41 dB). DehazeXL maintains high quality via token-level global attention.
- **Ours vs. DehazeFormer**: DehazeFormer performs well on dehazing using standard Transformers but requires slice-based inference, which results in block artifacts. The innovation of DehazeXL is elevating self-attention from the pixel level to the patch token level.
- **Ours vs. Large image inference methods (Gupta et al.)**: Gupta et al. designed a similar "slice-then-attend" framework for high-level vision tasks. DehazeXL successfully adapts this concept to the low-level vision task of dehazing.

## Rating

- Novelty: ⭐⭐⭐⭐ Transferring long-context LLM concepts to image dehazing is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on three datasets with various metrics, comprehensive ablations, and insightful attribution analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear figures and tables, with natural flow of motivation.
- Value: ⭐⭐⭐⭐ Provides general insights for low-level vision processing of large images, and the dataset is highly valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mechanism of Task-oriented Information Removal in In-context Learning](../../ICLR2026/image_restoration/mechanism_of_task-oriented_information_removal_in_in-context_learning.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)
- [\[CVPR 2025\] Detail-Preserving Latent Diffusion for Stable Shadow Removal](detail-preserving_latent_diffusion_for_stable_shadow_removal.md)
- [\[CVPR 2025\] SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)
- [\[ICCV 2025\] MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices](../../ICCV2025/image_restoration/mobileie_an_extremely_lightweight_and_effective_convnet_for_real-time_image_enha.md)

</div>

<!-- RELATED:END -->

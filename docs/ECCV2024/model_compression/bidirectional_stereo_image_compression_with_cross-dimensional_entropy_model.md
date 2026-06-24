---
title: >-
  [Paper Note] Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model
description: >-
  [ECCV2024][Model Compression][stereo image compression] Proposes a bidirectionally symmetric stereo image compression framework, BiSIC, using a 3D convolutional joint codec and a cross-dimensional entropy model. It outperforms both traditional standards and existing learned methods on PSNR and MS-SSIM, while eliminating the reconstruction quality imbalance between the left and right views inherent in unidirectional approaches.
tags:
  - "ECCV2024"
  - "Model Compression"
  - "stereo image compression"
  - "bidirectional coding"
  - "3D convolution"
  - "cross-dimensional entropy model"
  - "learned image compression"
date: 2026-05-08
content_hash: 114a2b36821b7939
---

# Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model

**Conference**: ECCV2024  
**arXiv**: [2407.10632](https://arxiv.org/abs/2407.10632)  
**Code**: [GitHub](https://github.com/LIUZhening111/BiSIC)  
**Area**: Model Compression  
**Keywords**: stereo image compression, bidirectional coding, 3D convolution, cross-dimensional entropy model, learned image compression

## TL;DR

Proposes a bidirectionally symmetric stereo image compression framework, BiSIC, using a 3D convolutional joint codec and a cross-dimensional entropy model. It outperforms both traditional standards and existing learned methods on PSNR and MS-SSIM, while eliminating the reconstruction quality imbalance between the left and right views inherent in unidirectional approaches.

## Background & Motivation

Stereo vision simulates human binocular vision and is widely applied in scenarios such as 3D cinema, autonomous driving, and AR/VR. With the popularity of stereo cameras, the volume of stereo image data has grown rapidly, making efficient compression a critical requirement.

Existing stereo image compression methods suffer from two major issues:

1. **Quality imbalance caused by unidirectional compression**: Mainstream methods compress one view first and then compress the other view using it as a reference. This unidirectional dependency leads to a significant difference in reconstruction quality between the two views (the two-view PSNR gap of VVC reaches 2.58 dB in the experiments), which is detrimental to human perception and downstream tasks.
2. **Limited performance of existing bidirectional methods**: The only bidirectional method, BCSIC, uses 2D convolutions to process the two views separately, failing to capture aligned features between the views; its entropy model only utilizes spatial context, neglecting the rich dependency information in the channel dimension.

## Core Problem

How to design a symmetrically bidirectional stereo image compression framework that can fully exploit the inter-view correlation to achieve a high compression rate while ensuring balanced reconstruction quality for both views?

## Method

### Overall Architecture

BiSIC consists of two core components: a Joint Codec and a Cross-Dimensional Entropy Model. The entire pipeline is completely symmetric—the left and right views are treated equally with no master-slave relationship.

### 1. 3D Convolutional Codec

Unlike previous methods that use 2D convolutions to process each view separately, BiSIC adopts 3D convolution as the codec backbone. 3D convolution stacks the left and right images along the view dimension for joint processing, naturally possessing the capability to extract inter-view correlations.

- **Encoder**: 4 layers of 3D convolutions downsample the stereo image pair from $B \times 3 \times H \times W$ to a compact latent representation of $B \times N \times \frac{H}{16} \times \frac{W}{16}$.
- **Decoder**: A symmetric 3D transposed convolution structure recovers the quantized latents to reconstructed images.
- **Hyper Codec**: Also based on 3D convolutions, generating auxiliary hyperprior information for the entropy model.

### 2. Bidirectional Mutual Attention Block

Convolutional layers excel at local modeling but are limited in capturing long-range dependencies. Therefore, a mutual attention block is inserted after the 2nd and 4th convolutional layers of the encoder to capture global features. This block contains two stages:

- **Cross-Key Attention**: Generates an attention map using the Key from one view and the Value from the other view, which is then queried by the current view's Query—used to discover aligned features and common patterns across views.
- **Cross-Query Attention**: Key and Value come from the same view, which is queried by the Query from the other view—retaining the individual features of each view while allocating attention weights with the help of the other view.

The outputs of the two stages are enhanced by self-attention and then fused through a parameter-shared Combine Block. To avoid the $(H \times W)^2$ computational complexity, Efficient Attention is employed, reducing the attention map size to $C_K \times C_V$, which is independent of the input size.

### 3. Cross-Dimensional Entropy Model

The entropy model is key to compression performance—it estimates the probability distribution of the latents; the more accurate the distribution estimation, the more compact the bitstream generated by entropy coding. BiSIC proposes a cross-dimensional entropy model that aggregates four types of conditional information:

- **Hyperprior**: Auxiliary information generated by the hyper decoder, providing global statistical characteristics.
- **Channel Context**: The latent is evenly split along the channel into $K$ slices. During slice-by-slice decoding, previously decoded slices (containing both views) are used as references and aggregated via the mutual attention block.
- **Spatial Context**: Within each slice, masked 3D convolutions extract spatial dependencies from the already-encoded causal regions (the mask ensures only decoded locations are accessed).
- **Stereo Dependency**: The aforementioned channel and spatial contexts both simultaneously consider information from both views, with 3D convolutions naturally capturing cross-view correlations.

Finally, an aggregation network $\mathbf{G}_{ag}$ fuses all conditions to estimate the mean $\mu$ and variance $\sigma^2$ of the Gaussian distribution.

### 4. Fast Variant BiSIC-Fast

Pixel-by-pixel spatial autoregressive inference is extremely time-consuming. BiSIC-Fast extends the Checkerboard structure to Stereo-Checkerboard:

- The latent is split into two parts: stereo anchor and stereo non-anchor.
- The anchor part is decoded relying solely on the hyperprior and channel context.
- The non-anchor part additionally utilizes the decoded anchors as spatial conditions.
- This simplifies the pixel-by-pixel autoregression into a two-step operation, achieving significant acceleration.

### Loss & Training

Standard rate-distortion loss: $\mathcal{L} = \lambda D + R$, where $D$ is the MSE or MS-SSIM distortion, $R$ is the bitrate of the latent and hyperprior, and $\lambda$ controls the rate-distortion trade-off.

## Key Experimental Results

### Datasets

- **InStereo2K**: 2,060 indoor stereo image pairs (2,010 for training / 50 for testing), with a resolution of 1080x860.
- **Cityscapes**: 5,000 outdoor urban scene pairs (2,975 for training / 1,525 for testing), with a resolution of 2048x1024.

### RD Performance (BDBR, anchored on BPG, lower is better)

| Method | InStereo2K PSNR | InStereo2K MS-SSIM | Cityscapes PSNR | Cityscapes MS-SSIM |
|------|----------------|-------------------|----------------|-------------------|
| VVC | -35.31% | -31.05% | -56.25% | -44.04% |
| ECSIC | -43.71% | -55.65% | -52.06% | -64.96% |
| **BiSIC** | **-48.07%** | **-61.13%** | **-57.49%** | **-67.98%** |
| BiSIC-Fast | -45.35% | -59.36% | -51.96% | -65.56% |

- Compared to VVC, BiSIC achieves an additional bitrate saving of 12.76% (PSNR) and 30.08% (MS-SSIM) on InStereo2K.
- Compared to the strongest bidirectional baseline BCSIC, it achieves an additional bitrate saving of 6.5%–15%.

### Running Time

| Method | BPG | HEVC | VVC | SASIC | BCSIC | BiSIC | BiSIC-Fast |
|------|------|------|------|-------|-------|-------|------------|
| Time | 16.17s | 28.16s | 190.27s | 20.24s | 89.44s | 167.25s | **22.82s** |

The running time of BiSIC-Fast is only 22.82s, which is close to the level of BPG/SASIC and much faster than BiSIC and VVC.

### Ablation Study (BD-PSNR, relative to BiSIC)

- Replacing 3D Conv with 2D Conv: -0.32 dB (demonstrates that 3D convolution effectively captures inter-view features).
- Replacing the cross-dimensional entropy model with the Minnen entropy model: -0.35 dB (validates the necessity of multi-dimensional conditions).
- Removing the mutual attention block: -0.79 dB (shows the greatest impact, indicating that global feature exchange is crucial).

## Highlights & Insights

1. **Symmetric bidirectional design** eliminates the inherent view quality imbalance problem of unidirectional methods; in the experiments, the two-view PSNR difference of BiSIC is negligible, whereas VVC has a gap of 2.58 dB.
2. **3D convolutional backbone** is a simple and effective design choice, naturally fitting the view dimension of stereo images, which is more elegant than processing with 2D convolutions separately and then fusing them.
3. **Cross-dimensional entropy model** systematically aggregates four conditions: hyperprior, spatial, channel, and stereo, being both comprehensive and symmetric.
4. **Stereo-Checkerboard fast variant** reduces the running time from 167s to 23s at the cost of only minor performance degradation, significantly improving practicality.

## Limitations & Future Work

1. The encoding/decoding time of BiSIC (167.25s) is still relatively long, limiting actual deployment; BiSIC-Fast solves the speed issue but sacrifices some performance.
2. Validated only on two datasets (InStereo2K and Cityscapes), lacking testing on more diverse scenarios (such as remote sensing, medical imaging, etc.).
3. The choice of hyperparameters in the entropy model, such as the number of slices $K$ and convolution kernel sizes, is not analyzed in depth.
4. The parameter size and GPU memory overhead of 3D convolution are larger than 2D convolution, and the trade-off between model complexity and compression performance is not discussed.
5. The relationship with perceptual quality metrics (such as LPIPS) or downstream task performance has not been explored.

## Related Work & Insights

| Method | Direction | Encoder | View Interaction | Entropy Model Conditions |
|------|------|--------|----------|-----------|
| HESIC+ | Unidirectional | 2D Conv | Homography warp | Hyperprior + Spatial |
| SASIC | Unidirectional | 2D Conv | Horizontal displacement | Hyperprior + Spatial |
| ECSIC | Unidirectional | 2D Conv | No explicit warp | Unidirectional conditional entropy |
| BCSIC | Bidirectional | 2D Conv (Separate) | Contextual transfer | Hyperprior + Spatial |
| LDMIC | Bidirectional | 2D Conv | Attention | Hyperprior + Spatial + Channel |
| **BiSIC** | **Bidirectional** | **3D Conv (Joint)** | **Mutual attention** | **Hyperprior + Spatial + Channel + Stereo** |

The core advantage of BiSIC lies in utilizing 3D convolution to achieve joint processing (rather than separate processing followed by interaction), as well as the systematic integration of multiple conditions by the cross-dimensional entropy model.

## Insights & Connections

- The idea of using 3D convolution to process stereo pairs can be extended to multi-view compression and joint multi-frame coding in video compression.
- The design philosophy of the cross-dimensional entropy model (multi-source condition aggregation) is also valuable for improving entropy models in single-image compression.
- The acceleration strategy of Stereo-Checkerboard reflects a general approach of "finding a balance between autoregressive steps and condition richness".
- The balanced compression characteristics of the bidirectionally symmetric architecture are friendly to downstream stereo vision tasks (depth estimation, 3D reconstruction).

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The combined design of the 3D convolutional backbone and the cross-dimensional entropy model is novel, though individual components are not entirely new)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Ablation is complete, and the comparison with multiple baselines is thorough, but the number of datasets is limited)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, intuitive diagrams, and standard mathematical formulations)
- **Value**: ⭐⭐⭐⭐ (Achieves SOTA in the field of stereo image compression, with the bidirectionally symmetric design holding practical significance)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MambaSIC: Mamba-based Stereo Image Compression with Bi-directional Multi-reference Entropy Model](../../CVPR2026/model_compression/mambasic_mamba-based_stereo_image_compression_with_bi-directional_multi-referenc.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](../../CVPR2025/model_compression/learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[CVPR 2026\] FreqSIC: Frequency-aware Stereo Image Compression with Bi-directional Checkerboard Context Model](../../CVPR2026/model_compression/freqsic_frequency-aware_stereo_image_compression_with_bi-directional_checkerboar.md)
- [\[ICML 2026\] Efficient Learned Image Compression without Entropy Coding](../../ICML2026/model_compression/efficient_learned_image_compression_without_entropy_coding.md)
- [\[ECCV 2024\] BaSIC: BayesNet Structure Learning for Computational Scalable Neural Image Compression](basic_bayesnet_structure_learning_for_computational_scalable_neural_image_compre.md)

</div>

<!-- RELATED:END -->

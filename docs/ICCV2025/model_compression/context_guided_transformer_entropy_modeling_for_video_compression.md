---
title: >-
  [Paper Note] Context Guided Transformer Entropy Modeling for Video Compression
description: >-
  [ICCV 2025][Model Compression][Video Compression] This paper proposes the Context Guided Transformer (CGT) conditional entropy model, which reduces entropy modeling time by approximately 65% while achieving an 11% BD-Rate improvement in video compression. This is accomplished via a Temporal Context Resampler that reduces computational overhead and a Dependency-Weighted Spatial Context Assigner that explicitly models spatial dependencies.
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Video Compression"
  - "Entropy Model"
  - "Transformer"
  - "Spatiotemporal Context"
  - "Conditional Coding"
date: 2026-05-08
content_hash: 351fd16333a4fcf4
---

# Context Guided Transformer Entropy Modeling for Video Compression

**Conference**: ICCV 2025
**arXiv**: [2508.01852](https://arxiv.org/abs/2508.01852)  
**Code**: [https://github.com/EIT-NLP/CGT](https://github.com/EIT-NLP/CGT)  
**Area**: Model Compression / Video Compression
**Keywords**: Video Compression, Entropy Model, Transformer, Spatiotemporal Context, Conditional Coding

## TL;DR
This paper proposes the Context Guided Transformer (CGT) conditional entropy model, which reduces entropy modeling time by approximately 65% while achieving an 11% BD-Rate improvement in video compression. This is accomplished via a Temporal Context Resampler that reduces computational overhead and a Dependency-Weighted Spatial Context Assigner that explicitly models spatial dependencies.

## Background & Motivation
- **Background**: Deep neural network-driven video compression methods, particularly conditional entropy models, have emerged as a dominant paradigm by leveraging spatiotemporal context to estimate the probability mass function (PMF) of video frames.
- **Limitations of Prior Work 1**: In the temporal dimension, incorporating additional temporal context inevitably increases computational overhead and inference latency — e.g., VCT requires self-attention over two frames of temporal context.
- **Limitations of Prior Work 2**: In the spatial dimension, existing methods (autoregressive, checkerboard, minimum-entropy decoding, etc.) adopt predefined fixed-order decoding strategies and lack explicit modeling of spatial positional dependencies.
- **Key Challenge**: How to effectively exploit spatiotemporal context simultaneously without significantly increasing computational cost.
- **Key Insight**: Temporal context compression via learnable queries combined with a teacher–student network that explicitly models spatial dependency weights.
- **Core Idea**: Compact learnable queries are used to resample temporal context, reducing subsequent processing overhead, while a teacher–student Swin Transformer network balances token importance and prediction certainty to determine the optimal spatial decoding order.

## Method

### Overall Architecture
CGT is built upon a contextual-based video codec. The encoder maps RGB frames to latent space features, and the CGT entropy model exploits temporal context (historical frame information from the latent buffer) and spatial context (already-decoded tokens of the current frame) to estimate the PMF of the current latent representation for entropy coding. The overall framework consists of a frame codec and the CGT entropy model.

### Key Designs
1. **Temporal Context Resampler (TCR)**:

    - **Function**: Extracts effective features from multiple types and scales of temporal context, producing a fixed-length compact token sequence.
    - **Mechanism**: A set of small learnable window queries is predefined and interacts with the temporal context via window cross-attention in a Swin Transformer, performing local information compression within each window between the compact queries and the larger temporal context.
    - **Design Motivation**: Not all temporal context information is equally important, and increased information volume significantly impacts decoding speed. Resampling via compact queries captures key temporal dependencies while substantially reducing the computational cost of subsequent processing.

2. **Dependency-Weighted Spatial Context Assigner (DWSCA)**:

    - **Function**: Explicitly models the positional dependencies of spatial context to identify the most informative context for undeoded tokens.
    - **Mechanism**: A shared-parameter teacher–student Swin Transformer decoder is employed. The teacher network generates an attention map (representing token importance) and an entropy map (reflecting prediction certainty) from a randomly masked input, and computes a dependency score via a weighted combination: $Score = \alpha H + (1-\alpha) A$, where $A$ is the normalized attention map and $H$ is the normalized entropy map. Soft top-$k$ selection then identifies positions with the highest dependency scores for decoding, providing context to the student network.
    - **Design Motivation**: Prior methods (autoregressive/checkerboard/minimum-entropy) do not explicitly model spatial dependencies, making it difficult to provide the most relevant context for undeoded tokens. The teacher–student structure ensures training–inference consistency.

3. **Random Masking Proxy Task**:

    - **Function**: Addresses the problem that the already-decoded content in the current frame cannot be predefined during training of the teacher network.
    - **Mechanism**: A random mask $y_t + M$ is applied to the input latent representation, with unmasked regions simulating already-decoded content. The teacher network generates attention and entropy maps from the masked representation to guide the student network's decoding.
    - **Design Motivation**: Inspired by masked image modeling, random masking simulates the progressive decoding process to ensure training–inference consistency.

### Loss & Training
Rate-distortion loss: $\mathcal{L}_{RD} = R(\hat{y}_t) + R(\hat{z}_t) + R(\hat{v}_t) + \lambda \cdot d(x_t - \hat{x}_t)$

where $R$ denotes rate terms, $d$ denotes the distortion term, and $\lambda \in \{256, 512, 1024, 2048\}$ controls the rate-distortion trade-off. The model is trained on Vimeo-90k with random crops to $265\times256$ and random flipping augmentation. Decoding employs an 8-step sinusoidal scheduling strategy.

## Key Experimental Results

### Main Results (BD-Rate, PSNR, anchor = VTM)

| Dataset | MCL-JCV | UVG | HEVC-B | Average |
|---------|---------|-----|--------|---------|
| VTM | 0 | 0 | 0 | 0 |
| DMC | -24.5 | -26.1 | -49.4 | -33.3 |
| MIMT | -33.0 | -34.9 | -57.1 | -41.7 |
| **CGT** | **-43.8** | **-45.5** | **-62.5** | **-50.6** |

BD-Rate (MS-SSIM): CGT achieves an average of −74.7%, substantially outperforming MIMT (−65.3%) and DMC (−55.4%).

### Ablation Study
**Temporal Context Resampler Ablation**:

| Model | BD-Rate Change | Entropy Modeling Time | Encoding Time | Decoding Time |
|-------|--------------|----------------------|---------------|---------------|
| CGT-w/o TCR | anchor | 1305ms | 1682ms | 1576ms |
| CGT-w/ TCR | +1.8% | 488ms (↓63%) | 1073ms (↓35%) | 984ms (↓38%) |

**Spatial Context Assigner Ablation** (anchor = minimum-entropy decoding):

| Model | MCL-JCV | UVG | HEVC-B | Average |
|-------|---------|-----|--------|---------|
| CGT-DWSCA (Ours) | -11.3 | -7.8 | -14.6 | -11.2 |
| CGT-min-entropy | 0 | 0 | 0 | 0 |
| CGT-checkerboard | +17.7 | +15.1 | +19.2 | +17.3 |
| CGT-autoregressive | +19.3 | +16.6 | +22.8 | +19.5 |

**Weight Coefficient α Analysis** (λ=256, MCL-JCV):
- α=0 (attention only, importance): PSNR 35.88, Bpp 0.019
- α=1 (entropy only, certainty): PSNR 35.3, Bpp 0.017
- α=0.5 (balanced): PSNR 35.82, Bpp 0.018

### Key Findings
- TCR incurs only a 1.8% BD-Rate increase while reducing entropy modeling time by 63%, encoding time by 35%, and decoding time by 38%.
- Explicit dependency modeling outperforms proxy-task-based (random masking) modeling due to reduced training–inference mismatch.
- CGT maintains strong performance when the frame codec is replaced (DCVC → DCVC-DC), demonstrating good generalizability.
- Compared to the VTM anchor, CGT achieves an average BD-Rate reduction of 50.6% on the PSNR metric.

## Highlights & Insights
- The temporal context resampling strategy is highly efficient — a small set of learnable queries with cross-attention achieves effective information compression, substantially reducing subsequent computation while preserving coding performance.
- The teacher–student + soft top-$k$ spatial decoding scheme outperforms both fixed-order strategies (autoregressive/checkerboard) and heuristic-order strategies (minimum-entropy), validating the necessity of explicit dependency modeling.
- α=0 (importance only) reduces distortion, while α=1 (certainty only) reduces bitrate; their complementary nature is effectively exploited by the balanced formulation.

## Limitations & Future Work
- A fixed α=0.5 may not be optimal across all scenarios; adaptive α could yield further gains.
- The 8-step decoding schedule is fixed to a sinusoidal function; more flexible scheduling strategies may improve performance.
- The training set Vimeo-90k has limited resolution (448×256), and generalization to high-resolution video remains to be verified.
- No comprehensive speed–performance comparison is provided against the latest implicit-representation-based methods (NVRC, MVC).

## Related Work & Insights
- The minimum-entropy principle from MIMT serves as an important baseline; this work advances it by introducing explicit dependency modeling.
- The learnable query + cross-attention information compression paradigm is broadly applicable to scenarios requiring reduced computational overhead.
- The training–inference consistency design via the teacher–student network offers a useful reference for other tasks requiring progressive decoding.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of temporal resampling and explicit spatial dependency modeling is novel, particularly the teacher–student soft top-$k$ scheme.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are comprehensive, covering TCR, DWSCA, α, generalization, and explicit modeling across multiple dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clear and method description is detailed, though notation could be made more consistent in places.
- **Value**: ⭐⭐⭐⭐ A favorable balance between compression efficiency and computational cost is achieved; the practical significance of 65% entropy modeling speedup combined with 11% BD-Rate improvement is substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](../../ACL2026/model_compression/latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](../../CVPR2025/model_compression/learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](../../ICML2025/model_compression/core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ICCV 2025\] MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion](motionfollower_editing_video_motion_via_score-guided_diffusion.md)

</div>

<!-- RELATED:END -->

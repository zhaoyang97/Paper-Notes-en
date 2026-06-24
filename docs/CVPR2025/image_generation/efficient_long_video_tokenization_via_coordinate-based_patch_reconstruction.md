---
title: >-
  [Paper Note] Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction
description: >-
  [CVPR 2025][Image Generation][Video Tokenizer] This paper proposes CoordTok, a scalable video tokenizer that encodes video into a factorized triplane representation. The decoder learns the mapping from randomly sampled $(x,y,t)$ coordinates to the corresponding patch pixels (rather than reconstructing all frames at once). This design enables direct training of a large tokenizer on 128-frame long videos, compressing a 128-frame video into only 1280 tokens (compared to 6144-819…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Video Tokenizer"
  - "Triplane Representation"
  - "Coordinate Reconstruction"
  - "Long Video Encoding"
  - "Video Compression"
  - "Diffusion Generation"
date: 2026-05-08
content_hash: 4561728f637a437b
---

# Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2411.14762](https://arxiv.org/abs/2411.14762)  
**Code**: [https://huiwon-jang.github.io/coordtok](https://huiwon-jang.github.io/coordtok)  
**Area**: Image/Video Generation  
**Keywords**: Video Tokenizer, Triplane Representation, Coordinate Reconstruction, Long Video Encoding, Video Compression, Diffusion Generation

## TL;DR

This paper proposes CoordTok, a scalable video tokenizer that encodes video into a factorized triplane representation. The decoder learns the mapping from randomly sampled $(x,y,t)$ coordinates to the corresponding patch pixels (rather than reconstructing all frames at once). This design enables direct training of a large tokenizer on 128-frame long videos, compressing a 128-frame video into only 1280 tokens (compared to 6144-8192 tokens required by baselines), and driving a DiT to achieve one-shot 128-frame video generation (with a SOTA FVD of 369.3).

## Background & Motivation

**Background**: Video tokenizers are fundamental components of video generation models, responsible for compressing high-dimensional videos into compact token representations. Existing tokenizers (such as TATS-AE, MAGVIT-AE, and PVDM-AE) continuously improve compression ratios but require reconstructing all frames during training, causing computation and memory costs to grow linearly with video length.

**Limitations of Prior Work**: (1) Training cost issues—on a single 4090 GPU, PVDM-AE directly encounters Out-Of-Memory (OOM) errors when training on 128-frame videos, forcing most tokenizers to train only on 16-frame short videos; (2) Temporal consistency issues—tokenizers that can only encode short segments fail to fully exploit the temporal coherence of videos. When multiple short segments are stitched to encode long videos, pixel value inconsistencies occur at the boundaries of the segments (as shown in Figure 1b).

**Key Challenge**: Temporal coherence in videos is a key prior for efficient compression (similar to video codecs utilizing keyframe plus residual coding). However, existing tokenizers are restricted to training on short segments due to training cost limitations and cannot exploit this prior.

**Goal**: How to design a tokenizer that can be trained directly on long videos, thereby fully utilizing temporal coherence to achieve more efficient compression?

**Key Insight**: Inspired by 3D generative models (e.g., NeRF/triplane) that avoid full-coordinate all-at-once training by learning the mapping from randomly sampled coordinates to RGB/density values, the video reconstruction problem is similarly reformulated as learning the mapping from $(x,y,t)$ coordinates to corresponding patches.

**Core Idea**: Encode the video into a compact representation of three 2D planes (triplane). During decoding, only a small number of randomly sampled coordinates are needed to reconstruct the corresponding patches. This decouples the training cost from the video length, allowing direct training on 128-frame long videos to yield highly efficient tokenization.

## Method

### Overall Architecture

CoordTok consists of an encoder and a decoder: the encoder divides the video $\mathbf{x}$ into spatiotemporal patches, processes them via a Transformer, and projects the video features into a factorized triplane representation $\mathbf{z} = [\mathbf{z}^{xy}, \mathbf{z}^{yt}, \mathbf{z}^{xt}]$ (which capture global content, y-axis motion, and x-axis motion, respectively) through cross-self attention layers. The decoder receives $N$ randomly sampled normalized coordinates $(i,j,k) \in [0,1]^3$, queries coordinate features from the triplane using bilinear interpolation, processes them via self-attention, and projects them into pixel values of the corresponding patches. During training, reconstructing only 3% of the patches is sufficient to achieve strong performance.

### Key Designs

1. **Factorized Triplane Encoder**:

    - **Function**: Compresses the video into a compact representation of three 2D planes, avoiding the high memory overhead of 3D latents.
    - **Mechanism**: Introduces learnable embeddings $\mathbf{z}_0 = [\mathbf{z}_0^{xy}, \mathbf{z}_0^{yt}, \mathbf{z}_0^{xt}]$ (with shapes $H' \times W'$, $W' \times T'$, and $H' \times T'$, respectively) and aggregates video features $\mathbf{e}$ (obtained by processing spatiotemporal patches with ViT) into these three planes via cross-self attention layers. Here, $\mathbf{z}^{xy}$ captures global content across time (scene layout, appearance), whereas $\mathbf{z}^{yt}$ and $\mathbf{z}^{xt}$ capture motion details along the two spatial axes. Each learnable embedding is split into 4 sub-embeddings to increase sequence length and improve model utilization.
    - **Design Motivation**: Replacing a 3D latent with three 2D planes reduces the token count from $H' \times W' \times T'$ to $H'W' + W'T' + H'T'$. For a 128-frame video, this difference reduces the number of tokens from thousands to just over one thousand. The factorization of the triplane naturally disentangles content and motion, which benefits downstream generative models in modeling them separately.

2. **Coordinate Sampling + Patch Reconstruction Decoder**:

    - **Function**: Decouples the training cost from video length, enabling direct training on long videos.
    - **Mechanism**: Divides the video into non-overlapping spatiotemporal patches and converts each patch index into normalized coordinates $(i,j,k) \in [0,1]^3$. During training, $N$ coordinates (only 3% of the patches) are randomly sampled. Coordinate features $\mathbf{h} = \text{Concat}(\mathbf{h}^{xy}, \mathbf{h}^{yt}, \mathbf{h}^{xt})$ are queried from the triplanes using bilinear interpolation, and after information interaction among coordinates via self-attention layers, they are mapped to the RGB pixels of the corresponding patches using a linear projection layer. The loss used is the $\ell_2$ reconstruction loss.
    - **Design Motivation**: Traditional tokenizers must reconstruct all frames at once, which makes the memory and computation scale linearly with the number of frames, limiting training to short videos. Sampling only 3% of the patches allows effective training, allowing the training batch size for 128-frame videos to remain at 256 (compared to PVDM-AE which encounters OOM at 128 frames).

3. **Frame Sampling Fine-tuning + LPIPS Loss**:

    - **Function**: Enhances the perceptual quality of the reconstruction based on coordinate-sampling pre-training.
    - **Mechanism**: After the main training phase is completed (after 1M iterations), the training switches to frame-sampling mode—randomly sampling several frames, reconstructing all coordinates for these frames, and fine-tuning for 50K iterations using a joint $\ell_2$ and LPIPS loss. Since LPIPS requires full frames to compute, it cannot be used during the coordinate sampling phase.
    - **Design Motivation**: Using frame sampling from the very beginning of training performs poorly due to insufficient sampling diversity (validated in Table 4). However, fine-tuning after sufficient coordinate-sampling training successfully enhances the perceptual quality. This two-stage strategy combines the efficiency of coordinate sampling with the high quality of frame sampling.

### Loss & Training

- **Main Training Phase**: $\ell_2$ reconstruction loss, $\mathcal{L} = \|\hat{\mathbf{x}}_{ijk} - \mathbf{x}_{ijk}\|_2^2$, computed only on the randomly sampled $N=1024$ coordinates.
- **Fine-tuning Phase**: Joint $\ell_2 + $ LPIPS loss, computed on randomly sampled full frames ($N=4096$ coordinates).

## Key Experimental Results

### Long Video Reconstruction Quality (128 frames, 128×128)

| Method | Token Type | Token Count | Training Frames | PSNR↑ | LPIPS↓ | rFVD↓ |
|:--|:--|:--|:--|:--|:--|:--|
| OmniTok-CV | Continuous | 8192 | 17 | 28.3 | 0.081 | 49.5 |
| CosmosTokenizer* | Continuous | 8192 | 17 | 28.5 | 0.119 | 87.8 |
| PVDM-AE | Continuous | 6144 | 16 | 26.5 | 0.120 | 66.5 |
| OmniTok-CV | Continuous | 1024 | 17 | 23.2 | 0.175 | 396.7 |
| PVDM-AE | Continuous | 1152 | 16 | 19.1 | 0.333 | 1270.1 |
| **CoordTok** | **Continuous** | **1280** | **128** | **28.6** | **0.066** | **102.9** |

### Video Generation (128 frames, UCF-101)

| Method | FVD↓ | Generation Time (s) | GPU Memory (GB) |
|:--|:--|:--|:--|
| StyleGAN-V | 1773.4 | - | - |
| PVDM-L | 505.0 | 116.9 | 4.0 |
| HVDM | 549.7 | 52.1 | 3.9 |
| Latte-L/2 | 1901.8 | 21.4 | 3.1 |
| **CoordTok-SiT-L/2** | **369.3** | **9.8** | **4.5** |

### Ablation Study and Analysis

| Analysis Item | Key Conclusion |
|:--|:--|
| Coordinate sampling ratio | 3% of patches is sufficient; more does not significantly improve performance |
| Model scale | Large > Base > Small; larger models continuously improve |
| Triplane spatial dimension | 16×16 is optimal; 8×8 is insufficient, 32×32 is redundant |
| Triplane temporal dimension | 32 is optimal; 16 is insufficient, 64 is redundant |
| Frame sampling vs. Coordinate sampling | Pure frame sampling (from scratch) is inferior to coordinate sampling due to insufficient diversity |

### Key Findings

- **Huge compression advantages brought by long-video training**: The reconstruction quality achieved by CoordTok with 1280 tokens (rFVD 102.9) is comparable to or better than that achieved by baselines using 6144-8192 tokens.
- **Triplanes are more sensitive to dynamic videos**: Pearson correlation analysis shows a stronger correlation (r=0.617) between CoordTok's reconstruction quality and video dynamics, suggesting that motion factorization is its primary pressure point.
- **Efficient tokenization improves downstream generation**: SiT trained with 1280 tokens outperforms that trained with 3072 tokens (FVD is ~50+ lower), because fewer tokens reduce the learning difficulty of the generative model.
- **Extremely fast generation speed**: One-shot generation of 128 frames takes only 9.8s (compared to 116.9s for PVDM-L, a 12x speedup).

## Highlights & Insights

1. **Cross-domain inspiration transfer**: Seamlessly migrating the coordinate-sampling training concept from 3D generation/NeRF to video tokenizer design is highly intuitive and effective.
2. **Decoupled training cost and video length**: This is a key methodological breakthrough—by reconstructing only a random 3% of patches, the training cost for 128 frames is comparable to that of 16 frames.
3. **Fewer tokens = Better generation**: This counter-intuitive finding is highly inspiring—the sweet spot between token count and reconstruction quality does not necessarily equal the sweet spot for the generative model. Instead, a more compact representation reduces generation difficulty.
4. **Content-motion separation of the triplane**: The natural separation where the xy-plane captures global content and the yt/xt-planes capture motion is validated in the visualization.

## Limitations & Future Work

1. **Resolution limitations**: All experiments were conducted at 128×128 resolution, and the effectiveness at higher resolutions (e.g., 256×256, 512×512) has not been verified.
2. **Experiments limited to UCF-101**: The dataset scale and diversity are limited, and generalization has not been verified on large-scale datasets.
3. **Triplane limitations on dynamic videos**: Analysis shows that more dynamic videos are harder to reconstruct—fast motion makes content-motion factorization more difficult.
4. **Unconditional generation**: Downstream generation only evaluated unconditional models and has not been integrated with text-conditional generation.
5. **Coordinate reconstruction vs. pixel-level details**: Patch-level reconstruction may perform worse on fine textures compared to pixel-by-pixel decoders.

## Related Work & Insights

- **PVDM**: Also utilizes a triplane representation but decodes all frames at once, serving as a direct baseline to CoordTok—demonstrating the crucial importance of the decoder design.
- **TiTok**: An image tokenizer utilizing 1D tokens; this work extends the concept of factorized representation to the video dimension.
- **NeRF / 3D Triplane Generation**: The source of inspiration for coordinate-sampling training (e.g., 3D generation works such as LRM, Instant3D, etc.).
- **Video Codecs (HEVC, AV1, etc.)**: Keyframes + residual coding is a classic strategy to exploit temporal redundancy; CoordTok's triplane can be viewed as its learned counterpart.
- **Insights**: The core characteristic of video is temporal redundancy—any video processing model should ponder how to exploit this prior instead of treating each frame independently. There is an interesting trade-off between training length and inference efficiency.

## Rating

⭐⭐⭐⭐ — The idea is novel, clean, and elegant, seamlessly transferring the coordinate-sampling concept from 3D generation to video tokenizers, achieving a significant breakthrough in training efficiency (direct training on 128 frames) with outstanding downstream generation performance. However, evaluations are limited by low resolution and dataset scale; verification on high-resolution and large-scale scenarios remains to be demonstrated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Language-Guided Image Tokenization for Generation](language-guided_image_tokenization_for_generation.md)
- [\[CVPR 2026\] Spk2VidNet: A Hierarchical Recurrent Architecture for High-Fidelity Video Reconstruction from Long Spike-Camera Streams](../../CVPR2026/image_generation/spk2vidnet_a_hierarchical_recurrent_architecture_for_high-fidelity_video_reconstr.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](../../ICCV2025/image_generation/efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[ICLR 2026\] WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction](../../ICLR2026/image_generation/wetok_powerful_discrete_tokenization_for_high-fidelity_visual_reconstruction.md)
- [\[CVPR 2025\] OFER: Occluded Face Expression Reconstruction](ofer_occluded_face_expression_reconstruction.md)

</div>

<!-- RELATED:END -->

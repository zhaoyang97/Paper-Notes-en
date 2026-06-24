---
title: >-
  [Paper Note] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation
description: >-
  [CVPR 2025][Image Generation][Video Frame Interpolation] EDEN is proposed to comprehensively enhance the role of diffusion models in video frame interpolation from three dimensions: input representation, model architecture, and training paradigm. By compressing intermediate frames into semantically rich 1D token representations using a Transformer tokenizer, replacing the U-Net architecture with DiT, and introducing a dual-stream context integration mechanism (temporal attent…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Video Frame Interpolation"
  - "Diffusion Model"
  - "DiT"
  - "Transformer tokenizer"
  - "Large Motion"
  - "Temporal Attention"
date: 2026-05-08
content_hash: ac01ed695f0ca90e
---

# EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation

**Conference**: CVPR 2025  
**arXiv**: [2503.15831](https://arxiv.org/abs/2503.15831)  
**Code**: [https://github.com/bbldCVer/EDEN](https://github.com/bbldCVer/EDEN)  
**Area**: Image/Video Generation  
**Keywords**: Video Frame Interpolation, Diffusion Model, DiT, Transformer tokenizer, Large Motion, Temporal Attention

## TL;DR

EDEN is proposed to comprehensively enhance the role of diffusion models in video frame interpolation from three dimensions: input representation, model architecture, and training paradigm. By compressing intermediate frames into semantically rich 1D token representations using a Transformer tokenizer, replacing the U-Net architecture with DiT, and introducing a dual-stream context integration mechanism (temporal attention + frame difference embedding), EDEN reduces LPIPS by nearly 10% on large-motion benchmarks like DAVIS, achieving high-quality generation with only 2 denoising steps.

## Background & Motivation

**Background**: Video Frame Interpolation (VFI) aims to synthesize intermediate frames between starting and ending frames. Traditional methods rely on optical flow estimation to warp intermediate frames. Recently, diffusion models have been applied to VFI tasks to directly generate intermediate frames in the latent space, avoiding explicit optical flow warping.

**Limitations of Prior Work**: Although diffusion-based methods have theoretical advantages, they still perform poorly in large-motion and non-linear motion scenarios, where generated frames often suffer from motion blur and temporal inconsistency. Crucially, the authors find that the diffusion process in existing diffusion-based VFI methods has a negligible impact on final generation quality—sampling directly from random noise versus decoding from denoised latents yields minimal perceptual difference, suggesting the diffusion process is largely underutilized.

**Key Challenge**: The diffusion process is supposed to be the core source of generative capability. However, in existing VFI frameworks, it is severely "bypassed" or underutilized due to non-semantic latent representations, architectures poorly suited for temporal modeling, and training paradigms lacking motion awareness.

**Goal**: How can the contribution of the diffusion process in VFI be fundamentally amplified to truly handle large and complex motion scenarios?

**Key Insight**: The authors approach this from three improvable dimensions: (1) replacing 2D VAE with a Transformer tokenizer to obtain superior latent representations; (2) replacing U-Net with DiT to attain better temporal modeling capabilities; and (3) introducing a dual-stream context integration mechanism to enhance the training paradigm.

**Core Idea**: By improving latent representation quality, migrating to a DiT architecture better suited for VFI, and incorporating a dual-stream context mechanism with frame difference embeddings, the proposed approach ensures that the diffusion process truly acts as the decisive factor for VFI generation quality.

## Method

### Overall Architecture

The pipeline of EDEN is trained in two stages: (1) training a Transformer tokenizer to compress the intermediate frame into compact 1D latent tokens. Both the encoder and decoder contain 4 Transformer blocks, each featuring a pyramid feature fusion module and a temporal attention module. (2) training a DiT diffusion model to perform denoising generation based on these latent tokens, incorporating dual-stream context integration with temporal attention and frame difference embedding in each DiT block. During inference, DiT generates latent tokens from noise, and the tokenizer decoder reconstructs the intermediate frame.

### Key Designs

1. **Transformer Tokenizer and Pyramid Feature Fusion**:

    - **Function**: Compresses the intermediate frame into semantically rich 1D token sequences, replacing the grid representations of traditional 2D VAEs.
    - **Mechanism**: The encoder splits the input image into patches to obtain large-scale tokens $I_t^l$, which are then average-pooled into small-scale tokens $I_t^s$. These tokens are concatenated and passed through self-attention for multi-scale feature fusion (Pyramid Feature Fusion Module, PFFM), after which the outputs at small-scale positions are passed to the subsequent layers. The decoder operates in reverse: starting from small-scale tokens, interpolating to obtain large-scale tokens, and then performing fusion. Both the encoder and decoder consist of 4 Transformer blocks with a latent dimension of 768.
    - **Design Motivation**: TiTok has demonstrated that 1D sequence representations capture more high-level semantic information in compact latent spaces compared to 2D grids. The pyramid fusion module leverages the traditional benefits of multi-scale features to capture fine-grained details across different motion scales, where the ratio of large-scale to small-scale tokens is $m = 4n$.

2. **DiT Diffusion Model + Dual-Stream Context Integration**:

    - **Function**: Effectively integrates the contextual information of start and end frames during the diffusion process to enhance motion dynamics modeling.
    - **Mechanism**: A 12-layer DiT block serves as the backbone. In each self-attention layer, a temporal attention layer (Temporal Context) is inserted: tokens of the start and end frames are concatenated with the current latent token at their corresponding spatial positions, and temporal attention is applied to integrate complete inter-frame information. Simultaneously, a difference context integration (Difference Context) is introduced: the cosine similarity between the start and end frames is calculated, normalized, and transformed into a difference embedding via an MLP, which is then added to the timestep embedding as an input condition to adaptive layer normalization (adaLN).
    - **Design Motivation**: Temporal attention implicitly aligns the spatial correspondence between frames, offering better robustness on unseen resolutions than cross-attention. The frame difference embedding explicitly encodes the prior of motion magnitude, enabling the model to adaptively adjust its generation strategy according to the extent of motion.

3. **Multi-Resolution Multi-Frame Interval Fine-Tuning**:

    - **Function**: Enhances the model's generalization capabilities across different resolutions and motion magnitudes.
    - **Mechanism**: In the first stage, the model is trained to convergence on a fixed low resolution (256×448) and small frame intervals (1 to 5 frames). In the second stage, it is fine-tuned using randomly selected frame pairs with varying interval lengths and resolutions. Position embeddings are interpolated to accommodate different sizes.
    - **Design Motivation**: Real-world video resolutions and motion magnitudes vary significantly, and transferring ViT position embeddings across different resolutions is a well-known vulnerability. Training on fixed low resolutions to fully learn motion patterns before multi-scale fine-tuning to improve generalization is a simple yet effective strategy.

### Loss & Training

- **Tokenizer Training Loss**: $\mathcal{L}_{tok} = \lambda_1 \mathcal{L}_1 + \lambda_p \mathcal{L}_p + \lambda_G \mathcal{L}_G + \lambda_{kl} \mathcal{L}_{kl}$, including L1 reconstruction loss, perceptual loss (LPIPS), patch adversarial loss, and a minor KL regularization, with weights set to 1.0, 1.0, 0.5, and 1e-6 respectively.
- **DiT Training Loss**: Employs the Flow Matching paradigm, where the forward process is defined as $x_t = (1-t)x_0 + t\varepsilon$. The velocity field prediction loss is formulated as $\mathcal{L}_{dit} = \mathbb{E}_{t,p_t(z)} \|v_\Theta(z,t) - u_t(z)\|_2^2$.

## Key Experimental Results

### Main Results

| Method | DAVIS LPIPS↓ | DAVIS FloLPIPS↓ | DAIN-HD LPIPS↓ | SNU-Extreme LPIPS↓ | Inference Time (s) |
|:--|:--|:--|:--|:--|:--|
| VFIMamba | 0.1084 | 0.1486 | 0.1426 | 0.1154 | 0.230 |
| SGM-VFI | 0.1140 | 0.1571 | 0.1423 | 0.1205 | 0.136 |
| LBBDM | 0.0963 | 0.1313 | 0.1471 | 0.1101 | 1.689 |
| **EDEN (Ours)** | **0.0874** | **0.1201** | **0.1321** | **0.0986** | 0.130 |

### Ablation Study

| Ablation Item | DAVIS LPIPS | DAIN-HD LPIPS | Description |
|:--|:--|:--|:--|
| w/o Temporal Attention | 0.1731 | - | Performance drops significantly |
| Added to encoder only | 0.0773 | - | Shows improvement but insufficient |
| Added to both encoder and decoder | 0.0150 | - | Optimal |
| Cross-Attn vs Temp-Attn | 0.0547 vs 0.0548 | 0.0718 vs 0.0515 | Temporal attention is more robust on unseen resolutions |
| w/o Frame Difference Embedding | 0.0976 / 0.1425 | 0.1327 / 0.2376 | Frame difference embedding brings significant improvement |
| w/o PFFM | 0.0564 / 0.0596 | 0.0799 / 0.0926 | Pyramid fusion is effective |
| Latent dim 16 vs 24 (w/ DiT) | 0.1538 | 0.1641 | 16-dimensional representation unexpectedly outperforms 24-dimensional |

### Key Findings

- **Minimal Diffusion Steps Required**: High-quality generation is achieved with only 2 denoising steps, making EDEN's inference speed multiple times faster than previous diffusion methods, and even faster than some optical-flow-based methods.
- **Temporal Attention > Cross-Attention**: Both perform comparably at the training resolution, but temporal attention is significantly superior on unseen resolutions (DAIN-HD: 0.0515 vs 0.0718).
- **Higher Latent Dimension Is Not Always Better**: While a 24-dimensional tokenizer yields stronger reconstruction capabilities, its performance drops below that of the 16-dimensional variant when combined with DiT, as the generative capability of DiT struggles to navigate excessively high-dimensional latent spaces.

## Highlights & Insights

1. **Diagnostic Findings Pave the Way**: The authors first identify the issue of the diffusion process being "bypassed" in existing diffusion VFI methods through comparative experiments (very minimal difference between random noise vs. denoised latent decoding). This diagnosis itself is highly valuable.
2. **Comprehensive Enhancement Across Three Dimensions**: Simultaneous improvements in representation (tokenizer), architecture (DiT), and training paradigm (dual-stream context) present a neat, complete, and complementary strategy.
3. **Extreme Efficiency of 2-Step Denoising**: Dispensing with multi-step iterations to generate high-quality results makes a diffusion-based VFI approach practically viable for the first time (inference speed of 0.13s/frame, comparable to optical-flow methods).
4. **Clever Design of Frame Difference Embedding**: Injecting motion magnitude information into the adaLN condition as "cosine similarity $\to$ normalization $\to$ MLP embedding" is simple yet highly effective.

## Limitations & Future Work

1. **High Training Cost**: Requiring the training of the tokenizer first followed by the DiT increases the overall training overhead through a two-stage process.
2. **Evaluating Only Perceptual Metrics**: The paper completely bypasses reconstruction metrics like PSNR/SSIM. While justifications are provided, it still leaves questions regarding fine-grained detail fidelity.
3. **Lack of Arbitrary-Time Interpolation**: The current design is tailored for fixed intermediate timestamps and does not demonstrate controllability for arbitrary timestamps $t \in (0,1)$.
4. **Resolution Scalability**: Although multi-resolution fine-tuning is effective, the computational overhead of the Transformer tokenizer on ultra-high resolutions remains a concern.

## Related Work & Insights

- **TiTok**: Proves 1D token sequences outperform 2D grid representations, directly inspiring the design of the Transformer tokenizer in this paper.
- **LBBDM**: Continuous Brownian Bridge Diffusion Model was previously the strongest diffusion-based VFI method. EDEN reduces its LPIPS from 0.0963 to 0.0874 (approx. 9.2%↓) on DAVIS.
- **Flow Matching**: Employs rectified flow to define the forward process via linear paths, yielding highly efficient training and serving as the theoretical foundation for 2-step denoising.
- **Insight**: The efficacy of diffusion models in video tasks heavily relies on the quality of latent representations. Simply "adding diffusion" is insufficient; one must ensure that the diffusion process operates within a semantically rich, motion-aware space.

## Rating

⭐⭐⭐⭐ — Systematic from problem diagnosis to the three-dimensional solution. The efficiency advantage of 2-step denoising is highly significant for practical use. However, the two-stage training cost and evaluation limited to perceptual metrics are minor drawbacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hierarchical Flow Diffusion for Efficient Frame Interpolation](hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)
- [\[ICCV 2025\] TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation](../../ICCV2025/image_generation/tlb-vfi_temporal-aware_latent_brownian_bridge_diffusion_for_video_frame_interpol.md)
- [\[ECCV 2024\] DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](../../ECCV2024/image_generation/dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)
- [\[CVPR 2025\] EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)
- [\[ICCV 2025\] Video Motion Graphs](../../ICCV2025/image_generation/video_motion_graphs.md)

</div>

<!-- RELATED:END -->

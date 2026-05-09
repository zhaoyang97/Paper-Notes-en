---
title: >-
  [Paper Note] Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking
description: >-
  [NeurIPS 2025][Video Generation][Graphical Watermarking] Safe-Sora is the first method to embed **graphical watermarks** (e.g., logo images) directly into the video generation pipeline. It employs a coarse-to-fine adaptive matching strategy to assign watermark patches to visually similar frames and regions, and designs a 3D wavelet transform-enhanced Mamba architecture for spatiotemporal fusion. The method substantially outperforms all baselines in both video quality (FVD 3.77 vs. the second-best 154.35) and watermark fidelity.
tags:
  - NeurIPS 2025
  - Video Generation
  - Graphical Watermarking
  - Mamba
  - 3D Wavelet Transform
  - Adaptive Matching
date: 2026-05-08
content_hash: 5ac3d56da7a20604
---

# Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking

**Conference**: NeurIPS 2025
**arXiv**: [2505.12667](https://arxiv.org/abs/2505.12667)
**Code**: [https://github.com/Sugewud/Safe-Sora](https://github.com/Sugewud/Safe-Sora)
**Area**: Video Generation / Digital Watermarking / Copyright Protection
**Keywords**: Graphical Watermarking, Video Generation, Mamba, 3D Wavelet Transform, Adaptive Matching

## TL;DR
Safe-Sora is the first method to embed **graphical watermarks** (e.g., logo images) directly into the video generation pipeline. It employs a coarse-to-fine adaptive matching strategy to assign watermark patches to visually similar frames and regions, and designs a 3D wavelet transform-enhanced Mamba architecture for spatiotemporal fusion. The method substantially outperforms all baselines in both video quality (FVD 3.77 vs. the second-best 154.35) and watermark fidelity.

## Background & Motivation
The rapid proliferation of generative video models (e.g., Sora, VideoCrafter2, Open-Sora) has intensified the demand for copyright protection of AI-generated content. While mature invisible watermarking schemes exist for image generation, video generation watermarking remains severely underdeveloped. The few existing video watermarking methods (VideoShield, LVMark) embed only low-capacity binary strings, failing to exploit the inherently high information bandwidth of video. Videos possess far greater embedding capacity than images, making them well-suited for **graphical watermarks** (logos/icons)—a form of copyright identifier that is more intuitive, visually verifiable, and reliable in practical authentication scenarios.

The authors identify a key empirical phenomenon: **watermark embedding quality is highly correlated with the visual similarity between the watermark and the cover content**. Testing a classical image steganography network on 1,000 image pairs reveals that higher visual similarity between the cover image and the watermark (lower LPIPS) consistently leads to improved embedding quality (higher PSNR) and better watermark extraction. This observation directly motivates the core design of the proposed method.

## Core Problem
1. **How to embed graphical watermarks during video generation**—not a simple binary string but a complete logo image, which demands significantly higher information capacity and embedding precision.
2. **How to address poor embedding quality caused by large visual discrepancies between watermark and cover**—requiring identification of spatiotemporally optimal positions within the video for watermark placement.
3. **How to fuse and extract watermark information distributed across frames and regions**—since watermark patches are scattered across different spatial locations in different frames, effective spatiotemporal modeling is essential.

## Method

### Overall Architecture
The Safe-Sora pipeline consists of three stages:

**Input**: Video latent (from the latent space of a video generation model) + graphical watermark image
**Output**: Watermark-embedded generated video and the watermark image extracted from the embedded video

1. **Coarse-to-Fine Adaptive Patch Matching**: The watermark image is divided into patches, which are adaptively assigned to the most visually similar frames and spatial regions.
2. **Watermark Embedding**: A UNet-structured 2D SFMamba block fuses watermark features with multi-scale video features, followed by a 3D SFMamba block for spatiotemporal interaction, producing the watermark-embedded video.
3. **Watermark Extraction**: The embedded video passes through a degradation layer simulation, 3D SFMamba blocks, and position recovery to reconstruct the original watermark image.

### Key Designs

1. **Coarse-to-Fine Adaptive Patch Matching (CFAPM)**: Each watermark patch is first augmented with a **position channel** (binary-encoded patch index with redundant copies for robustness). Matching then proceeds in two stages:

    - **Coarse stage (frame-level)**: Features are extracted from patches and video frame latents via convolution + ReLU + GAP; dot-product similarity scores are computed to assign each patch to its most similar frame (subject to a capacity limit, with overflow redirected to the next-best frame).
    - **Fine stage (region-level)**: Within the selected frame, spatial regions are defined and patch-region similarities are computed to place each patch at its most compatible spatial location.

   A key design elegance is the direct reuse of the video generation model's latent representations as features, requiring only a single convolutional layer for feature extraction at virtually zero additional computational cost.

2. **Spatial-Frequency Mamba (SFMamba) Block**: A dual-stream design that processes spatial and frequency domain information simultaneously:

    - **Spatial branch**: LayerNorm → two parallel paths (SiLU activation path + Conv1×1 → Mamba path) → element-wise multiplication.
    - **Frequency branch**: 2D DWT decomposes features into four subbands (LL/LH/HL/HH) → rearrangement to restore resolution → FreqMamba scanning → 2D IDWT → multiplication with the SiLU path.
    - Outputs of both branches are concatenated and fused via a 1×1 convolution.

   The 2D variant is used for spatial fusion within the UNet; the 3D variant handles spatiotemporal interaction.

3. **3D Spatiotemporal Local Scanning Strategy**: The frequency branch of the 3D SFMamba uses 3D DWT to decompose features into eight subbands (LLL through HHH), with a proposed **hierarchical bidirectional scanning** scheme:

    - Forward: LLL → LLH → LHL → HLL → LHH → HLH → HHL → HHH (low to high frequency).
    - Backward: HHH → LLL (high to low frequency).
    - Within each subband, a **space-first, time-second** scanning order is adopted.

   This represents the first application of state space models to the watermarking domain, with hierarchical frequency scanning effectively capturing long-range spatiotemporal dependencies.

### Loss & Training
- **Video reconstruction loss**: $\mathcal{L}_{\text{video}} = \text{MSE}(\mathbf{V}, \hat{\mathbf{V}})$
- **Watermark reconstruction loss**: $\mathcal{L}_{\text{watermark}} = \text{MSE}(\mathbf{W}, \hat{\mathbf{W}})$
- **Total loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{video}} + \lambda \mathcal{L}_{\text{watermark}}$, with $\lambda=0.75$
- During training, ground-truth positions are provided for watermark reconstruction; at test time, the embedded position channels are used to predict patch arrangement.
- The degradation layer simulates H.264 compression (approximated by a 3D CNN to handle non-differentiability), rotation, and other real-world transformations.
- Position recovery employs a **confidence-guided greedy assignment algorithm** to resolve conflicts.
- AdamW optimizer; initial lr 5e-4 with cosine decay to 1e-6; 30 epochs; 4× RTX 4090.

## Key Experimental Results

| Method | W-PSNR↑ | W-SSIM↑ | V-PSNR↑ | V-SSIM↑ | V-LPIPS↓ | tLP↓ | FVD↓ |
|--------|---------|---------|---------|---------|---------|------|------|
| Balujanet | 25.28 | 0.91 | 25.26 | 0.87 | 0.25 | 1.32 | 512.22 |
| UDH | 22.90 | 0.77 | 27.75 | 0.73 | 0.32 | 2.09 | 1075.62 |
| PUSNet | 28.86 | 0.93 | 29.98 | 0.92 | 0.11 | 0.98 | 154.35 |
| Safe-SD | 24.24 | 0.84 | 22.32 | 0.75 | 0.24 | 1.87 | 849.83 |
| WengNet | 33.18 | 0.96 | 28.09 | 0.85 | 0.21 | 1.27 | 265.82 |
| **Safe-Sora** | **37.71** | **0.97** | **42.50** | **0.98** | **0.01** | **0.38** | **3.77** |

FVD decreases from the second-best 154.35 to 3.77 (**a 97.6% reduction**); watermark PSNR improves by more than 4.5 dB; video PSNR improves by more than 12.5 dB. The method is equally effective on the Open-Sora backbone (FVD 3.04).

In terms of robustness, Safe-Sora achieves state-of-the-art performance under various attacks including random erasure (5–20%), Gaussian blur (kernel sizes 3/5/7), Gaussian noise ($\sigma \sim U(0, 0.2)$), rotation (±30°), and H.264 compression (CRF=24). Notably, while all baselines suffer severe degradation under H.264 compression, Safe-Sora maintains high watermark quality.

### Ablation Study

| Variant | W-PSNR | V-PSNR | FVD |
|---------|--------|--------|-----|
| w/o CFAPM (remove adaptive matching) | 36.71 | 39.68 | 16.87 |
| w/o RtL (no latent-based routing) | 36.36 | 40.23 | 6.37 |
| w/o FS (remove fine stage) | 36.88 | 41.25 | 4.82 |
| w/o SLS (remove spatiotemporal local scanning) | 35.96 | 38.42 | 13.16 |
| w/o SFS (time-first scanning) | 36.41 | 42.21 | 5.24 |
| w/o MSFI (remove multi-scale feature injection) | 36.56 | 39.39 | 14.11 |
| **Full model** | **37.71** | **42.50** | **3.77** |

- **CFAPM contributes most**: Removing adaptive matching causes FVD to spike from 3.77 to 16.87, with notable drops in both watermark and video quality.
- **Spatiotemporal local scanning is critical**: Replacing it with vanilla 3D scanning pushes FVD to 13.16.
- **Multi-scale feature injection**: Leveraging VAE multi-scale features significantly improves video quality (FVD 14.11 → 3.77).
- Space-first scanning outperforms time-first scanning in overall quality; the latter yields only marginal gains in tLP.

## Highlights & Insights
- **Observation-driven design philosophy**: The coarse-to-fine matching mechanism is motivated by a clear and convincing empirical observation—that embedding quality correlates with visual similarity between watermark and cover—resulting in a well-motivated and coherent design.
- **First application of SSM/Mamba to watermarking**: The combination of 3D wavelet transform-based Mamba with hierarchical frequency scanning constitutes a highly novel architectural contribution.
- **Position channel design**: Binary positional encodings appended directly to each patch provide a simple yet effective solution for reassembling dispersed patches at extraction time.
- **Reuse of latent features** for similarity matching incurs virtually no additional computation, representing an elegant engineering choice.
- **Cross-backbone generalization**: The method works effectively on both UNet-based (VideoCrafter2) and DiT-based (Open-Sora) backbones.

## Limitations & Future Work
- **Static graphical watermarks only**: The current method supports only static logo/icon watermarks and does not accommodate dynamic video watermarks (e.g., animated sequences or temporally dynamic patterns), a limitation explicitly acknowledged by the authors.
- **Generalization to resolution and length**: Experiments are conducted at a fixed resolution of 320×512 with 8 frames; performance on high-resolution, long-duration videos remains to be validated.
- **Fidelity of the degradation layer simulation**: Approximating H.264 with a 3D CNN is a common practice, but real-world video degradation (e.g., social platform recompression, screen recording) is considerably more complex.
- **Potential for misuse**: The authors note in the broader impact discussion that the method could be exploited for fraudulent copyright claims.
- **Future directions**: Extending graphical watermarking to video-in-video embedding (i.e., embedding a video sequence into another video), or incorporating text prompts for conditional watermarking.

## Related Work & Insights
- **vs. VideoShield/LVMark**: These existing video generation watermarking methods embed only binary strings, yielding low information capacity. Safe-Sora embeds complete graphical watermarks, offering substantially greater information density and intuitive interpretability. The two approaches are not directly compared in experiments due to the fundamental task difference (binary vs. graphical).
- **vs. Safe-SD**: Safe-SD is a graphical watermarking method designed for image generation. Safe-Sora extends this paradigm to video. Quantitatively, Safe-SD achieves only 22.32 video PSNR and 849.83 FVD, far inferior to Safe-Sora, demonstrating that naively applying frame-by-frame image methods to video results in severe temporal inconsistency.
- **vs. PUSNet**: As a state-of-the-art image steganography method, PUSNet achieves acceptable per-frame quality (video PSNR 29.98) but its FVD of 154.35 reveals a fatal lack of inter-frame consistency. Safe-Sora's 3D Mamba-based spatiotemporal modeling fundamentally addresses this issue.

The **similarity-driven embedding** principle is transferable to other information hiding tasks: identifying the most visually similar region before embedding is likely to improve performance in any task requiring information fusion into a carrier. The frequency-domain Mamba pattern (3D DWT + hierarchical scanning) is broadly applicable to tasks requiring spatiotemporal frequency modeling, such as video understanding and video restoration. The binary position channel encoding is also generically applicable to multi-patch or multi-token recovery scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Two legitimate "firsts"—introducing graphical watermarking to video generation and applying Mamba to watermarking—provide sufficient novelty, though the overall framework is a novel combination of existing ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five baseline comparisons, robustness evaluation under multiple attacks, detailed ablations, and validation on two backbones provide comprehensive coverage; however, the test scale (100 prompts) is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ The narrative flow from empirical observation to method design is smooth and the figures are clear; some formulas and notation are somewhat redundant.
- **Value**: ⭐⭐⭐⭐ Opens a new direction in graphical watermarking for video generation with practical copyright protection applications; further work is needed to address robustness under real-world degradation scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](../../ICCV2025/video_generation/stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICCV 2025\] BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation](../../ICCV2025/video_generation/badvideo_stealthy_backdoor_attack_against_text-to-video_generation.md)
- [\[NeurIPS 2025\] Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models](video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)

<!-- RELATED:END -->

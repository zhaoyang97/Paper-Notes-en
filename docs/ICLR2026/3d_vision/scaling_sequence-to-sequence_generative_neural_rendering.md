---
title: >-
  [Paper Note] Scaling Sequence-to-Sequence Generative Neural Rendering
description: >-
  [ICLR 2026][3D Vision][Neural Rendering] This paper presents Kaleido, a family of decoder-only rectified flow transformers that treats 3D as a special subdomain of video. Through Unified Positional Encoding, a masked autoregressive framework, and a video pretraining strategy, Kaleido achieves "any-to-any" 6-DoF novel view synthesis without any explicit 3D representation. It **is the first generative method to match per-scene optimization (InstantNGP) in rendering quality under multi-view settings**, and scales resolution from 512/576px to 1024px.
tags:
  - ICLR 2026
  - 3D Vision
  - Neural Rendering
  - Novel View Synthesis
  - Rectified Flow Transformer
  - Masked Autoregression
  - Unified Positional Encoding
  - Video-3D Unification
date: 2026-05-08
content_hash: 3b2fbf23442b3601
---

# Scaling Sequence-to-Sequence Generative Neural Rendering

**Conference**: ICLR 2026
**arXiv**: [2510.04236](https://arxiv.org/abs/2510.04236)
**Code**: [Project Page](https://shikun.io/projects/kaleido)
**Area**: 3D Vision
**Keywords**: Neural Rendering, Novel View Synthesis, Rectified Flow Transformer, Masked Autoregression, Unified Positional Encoding, Video-3D Unification

## TL;DR

This paper presents Kaleido, a family of decoder-only rectified flow transformers that treats 3D as a special subdomain of video. Through Unified Positional Encoding, a masked autoregressive framework, and a video pretraining strategy, Kaleido achieves "any-to-any" 6-DoF novel view synthesis without any explicit 3D representation. It **is the first generative method to match per-scene optimization (InstantNGP) in rendering quality under multi-view settings**, and scales resolution from 512/576px to 1024px.

## Background & Motivation

Novel View Synthesis (NVS) is a core task in 3D vision: given a set of reference views, generate images at arbitrary target viewpoints. Existing methods face clear technical bottlenecks:

| Paradigm | Representative Works | Core Limitations |
|---------|---------|---------|
| Per-scene optimization | 3DGS, NeRF, InstantNGP | Requires many views + per-scene optimization taking minutes; cannot generalize zero-shot |
| Feed-forward reconstruction | PixelNeRF, LRM | Relies on explicit 3D representations (point clouds/tri-planes); limited generalization |
| Diffusion-based generation | Zero123++, SV3D, SEVA | U-Net architecture limits resolution to 512/576px and scalability; most require SDS two-stage refinement |
| Video-controlled generation | MotionCtrl, CameraCtrl, GEN3C | Essentially 4D temporal prediction; constrained by single reference frames/fixed trajectories; cannot handle "any-to-any" spatial queries |

**Core Insight**: 3D can be viewed as a special subdomain of video—both are fundamentally sequences of images, differing only in whether the inter-frame camera transformations are known. However, directly fine-tuning video models does not work: video models rely on temporal VAEs that assume high temporal correlation between frames, an assumption that breaks down in sparse-view 3D tasks.

**Key Motivations**:
- Camera-annotated 3D data is extremely scarce, while video data is orders of magnitude larger
- SEVA, the previous strongest general NVS model, uses a U-Net architecture with poor scalability and is limited to 576px
- A scalable pure Transformer architecture is needed that can learn spatial priors ("visual commonsense") from video and transfer them to 3D

## Method

### Overall Architecture

Kaleido formulates novel view synthesis entirely as a **sequence-to-sequence image synthesis task**: given $N$ reference views with 6-DoF camera poses $\{(I_i, P_i)\}_{i=1}^{N}$ and $M$ target poses $\{P_j\}_{j=1}^{M}$, it directly generates $M$ target views $\{I_j\}_{j=1}^{M}$. The entire process uses a single **decoder-only rectified flow transformer** with no explicit 3D representation (no point clouds, no NeRF, no 3DGS, no depth estimation).

Core architectural components:
- **Tokenization**: Each image is encoded into latent tokens via a VAE
- **Pose conditioning**: Camera pose information is injected into the token sequence via Unified Positional Encoding
- **Masking strategy**: Reference view tokens are unmasked (conditioned), and target view tokens are masked (to be generated)
- **Generation process**: Masked tokens are denoised via rectified flow

### Key Designs

**1. Unified Positional Encoding**

This is the key architectural innovation enabling a single Transformer to handle both video and 3D data simultaneously, **requiring no additional trainable parameters**:

- For **video data**: RoPE encodes the temporal position $t$ and spatial position $(h, w)$ of each frame
- For **3D data**: RoPE encodes Plücker ray coordinates $(\mathbf{o}, \mathbf{d})$ (computed from camera intrinsics and extrinsics), where $\mathbf{o}$ is the ray origin and $\mathbf{d}$ is the ray direction
- This unified design allows temporal consistency priors from video to naturally transfer as spatial consistency in 3D, and the architecture **switches between video and 3D training with zero modifications**

**2. Masked Autoregressive Framework**

Enables flexible "any-to-any" inference:
- Arbitrary $N$ reference views → arbitrary $M$ target views ($N$ and $M$ can vary at both training and inference time)
- Causal masking distinguishes conditioned views (clean tokens) from target views (noisy tokens)
- Autoregressive iteration: already-generated high-quality views can be added as new conditions, progressively expanding coverage
- Supports extreme inference: extrapolation from 12-view training length to 480-frame autoregressive generation (40× training length)

**3. Systematic Resolution of Scaling Bottlenecks**

Through extensive ablation studies, the paper identifies and resolves non-obvious bottlenecks that prevent pure seq2seq models from scaling:
- **Activation overflow** (Sec 2.2.2): Massive activation overflow occurring during large model training, resolved through specific activation function choices
- **Suboptimal SNR sampler** (Sec 2.2.3): Standard diffusion model SNR sampling strategies are suboptimal for 3D rendering; noise-biased sampling is proposed to improve training
- These "scaling recipe" findings are themselves core contributions, making simple pure-Transformer designs viable for generative NVS for the first time

### Loss & Training

**Two-stage training paradigm**:

1. **Video pretraining stage**: Training on large-scale video data with the rectified flow matching loss $\mathcal{L} = \mathbb{E}_{t}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$, learning spatiotemporal consistency priors
2. **3D fine-tuning stage**: Fine-tuning on camera-annotated 3D datasets; the Unified Positional Encoding automatically switches to Plücker ray mode to learn precise geometric correspondences

Key training strategies:
- Rectified Flow offers better training stability and sampling efficiency than standard DDPM
- Classifier-Free Guidance (CFG) is supported to enhance generation quality
- Noise-biased sampling strategy optimizes the SNR distribution for 3D generation
- No architectural modifications are required for the transfer from video to 3D

## Key Experimental Results

### Main Results: NVS Benchmark Comparison

| Setting | Comparison Methods | Kaleido Performance | Key Metric |
|------|---------|-------------|---------|
| Few-view (1–3 views) | Zero123++, SV3D, SEVA, EscherNet | Zero-shot; significantly outperforms all generative methods | PSNR substantially higher |
| Many-view (>10 views) | InstantNGP (per-scene optimization) | **First to match** optimization-based quality | Comparable PSNR |
| Single-view 3D reconstruction | All prior methods | SOTA | CD=1.83, VIoU=70% |
| Resolution | SEVA (576px), CAT3D (512px) | **First 1024px** generative rendering model | Resolution doubled |

### Comparison with Video-Controlled Generation Models (Appendix H, added in rebuttal)

| Method | Type | Camera Accuracy ($R_{err}$↓ / $T_{err}$↓) | Visual Quality (LPIPS↓) |
|------|------|------|------|
| Wonderland (CVPR 2025) | Video→3DGS pipeline | Worse | Comparable |
| ViewCrafter | Video generation | Worse | Worse |
| VD3D | Video diffusion | Worse | Worse |
| MotionCtrl (SIGGRAPH 2024) | Video control | Worse | Worse |
| **Kaleido** | Seq2Seq NVS | **Best** | **Best or comparable** |

On DL3DV and Tanks & Temples, Kaleido significantly outperforms all video baseline models in camera accuracy.

### Ablation Study

| Ablation Configuration | Impact | Notes |
|---------|------|------|
| Remove video pretraining | Significant degradation in spatial consistency | Validates the "3D ≈ video subdomain" hypothesis |
| Standard PE vs. Unified PE | Performance drop | Unified PE is critical for video→3D transfer |
| Encoder-decoder vs. Decoder-only | Performance drop | Decoder-only better suits variable-length sequence tasks |
| Standard SNR sampling vs. Noise-biased sampling | Performance drop | SNR distribution optimization is critical for 3D rendering |
| Activation overflow not fixed | Unstable training | Key bottleneck for large-model scaling |
| Reduced model scale | Consistent degradation | 3D geometric understanding requires sufficient model capacity |

### Key Findings

1. **3D ≈ a special subdomain of video**: The hypothesis is empirically validated—video pretraining substantially improves 3D rendering quality, and temporal consistency effectively transfers as spatial consistency
2. **Clear scaling law**: Increasing model scale consistently improves rendering quality; a clear scaling law exists for NVS with pure Transformers
3. **Implicit geometric understanding**: The model learns meaningful geometry without explicit 3D representations, as evidenced by 3D reconstruction quality (CD=1.83)
4. **Video models cannot be naively substituted**: Comparative experiments show that the temporal VAE assumption of video models fails on sparse 3D tasks; methods such as GEN3C collapse under large viewpoint changes due to "empty cache" issues

## Highlights & Insights

- **Paradigm breakthrough**: The first demonstration that a pure 2D sequence model can match per-scene optimization (e.g., InstantNGP) in rendering quality, with no SDS two-stage refinement required
- **Architectural innovation**: Unified Positional Encoding enables a single architecture to handle both video and 3D with zero modifications—an elegant and practical design
- **Data scale leverage**: Video pretraining elegantly compensates for the scarcity of 3D data with clearly transferable benefits—offering a paradigm reference for other 3D tasks facing data scarcity
- **Scaling recipe**: Systematic identification and resolution of seq2seq 3D rendering scaling bottlenecks (activation overflow, SNR sampling) yields broadly applicable insights
- **512px to 1024px**: The first generative rendering model to surpass U-Net limitations and achieve 1024px resolution

## Limitations & Future Work

1. **Inference efficiency**: Large models with long sequences incur significant inference costs; zero-shot inference, while faster than per-scene optimization, remains far from real-time
2. **Geometric precision ceiling**: Implicit 3D understanding may be insufficient for applications requiring sub-pixel-level geometric accuracy (e.g., AR/VR)
3. **Camera parameter dependency**: Known 6-DoF camera parameters for target views are still required as conditions
4. **Long-sequence degradation**: Extreme autoregressive generation at 480 frames still exhibits artifacts; consistency over very long trajectories remains to be improved
5. **High training cost**: The combined cost of video pretraining and 3D fine-tuning is substantial
6. **No 4D capability**: The current design targets static scenes and does not address dynamic scene reconstruction

## Related Work & Insights

- **NeRF / 3DGS**: Per-scene optimization methods serving as quality upper bounds; Kaleido is the first to match InstantNGP in multi-view settings
- **SEVA (ICCV 2025)**: A U-Net-based general NVS model; Kaleido comprehensively surpasses it in scalability and resolution
- **CAT3D / ReconFusion / ZeroNVS**: Generative methods requiring SDS two-stage refinement; Kaleido achieves higher quality in a single stage
- **GEN3C (NVIDIA)**: A video model based on explicit depth reprojection that collapses under large viewpoint changes due to empty caches; Kaleido's implicit priors are more robust
- **Wonderland (CVPR 2025)**: A video→3DGS pipeline; Kaleido significantly outperforms it in camera accuracy
- **Insights**: The approach of unifying specialized domain tasks as sequence-to-sequence problems and leveraging large-scale pretraining on adjacent data can be generalized to 3D editing, dynamic scene reconstruction, robotic manipulation, and other data-scarce settings

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Triple innovation: "3D-as-Video" unified paradigm + Unified Positional Encoding + scaling recipe)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive multi-benchmark evaluation with thorough ablations; comparison with video models was missing from the initial submission and added in the rebuttal)
- Writing Quality: ⭐⭐⭐⭐ (Concepts are clearly articulated and systematically organized; the distinguishing arguments in the rebuttal are more precise than those in the main paper)
- Value: ⭐⭐⭐⭐⭐ (Opens a new direction for generative neural rendering; first to match optimization-based quality; scaling insights have broad guiding significance)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](../../CVPR2026/3d_vision/longstream_long-sequence_streaming_autoregressive_visual_geometry.md)
- [\[ICCV 2025\] LONG3R: Long Sequence Streaming 3D Reconstruction](../../ICCV2025/3d_vision/long3r_long_sequence_streaming_3d_reconstruction.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](../../CVPR2026/3d_vision/aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)
- [\[AAAI 2026\] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](../../AAAI2026/3d_vision/radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)
- [\[CVPR 2026\] Scaling View Synthesis Transformers (SVSM)](../../CVPR2026/3d_vision/scaling_view_synthesis_transformers.md)

<!-- RELATED:END -->

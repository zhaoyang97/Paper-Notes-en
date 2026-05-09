---
title: >-
  [Paper Note] Generative Neural Video Compression via Video Diffusion Prior
description: >-
  [CVPR 2026][Video Generation][Video Compression] This paper proposes GNVC-VD, the first DiT-based generative neural video compression framework. By leveraging a video diffusion transformer as a video-native generative prior, GNVC-VD performs spatiotemporal latent compression and sequence-level generative refinement within a unified codec. At extremely low bitrates (<0.03 bpp), it substantially surpasses both traditional and learned codecs in perceptual quality while significantly reducing the flickering artifacts prevalent in prior generative approaches.
tags:
  - CVPR 2026
  - Video Generation
  - Video Compression
  - Video Diffusion Model
  - Flow Matching
  - Perceptual Quality
  - Temporal Consistency
date: 2026-05-08
content_hash: 08934f314d6f6268
---

# Generative Neural Video Compression via Video Diffusion Prior

**Conference**: CVPR 2026
**arXiv**: [2512.05016](https://arxiv.org/abs/2512.05016)
**Code**: N/A
**Area**: Video Generation
**Keywords**: Video Compression, Video Diffusion Model, Flow Matching, Perceptual Quality, Temporal Consistency

## TL;DR

This paper proposes GNVC-VD, the first DiT-based generative neural video compression framework. By leveraging a video diffusion transformer as a video-native generative prior, GNVC-VD performs spatiotemporal latent compression and sequence-level generative refinement within a unified codec. At extremely low bitrates (<0.03 bpp), it substantially surpasses both traditional and learned codecs in perceptual quality while significantly reducing the flickering artifacts prevalent in prior generative approaches.

## Background & Motivation

1. **Background**: Neural video compression (NVC) has advanced rapidly in recent years, with learned codecs such as the DCVC series surpassing traditional standards like HEVC and VVC in rate-distortion performance. In the image domain, generative compression has successfully recovered high-frequency textures via pretrained GANs or diffusion models, producing visually compelling reconstructions at very low bitrates.
2. **Limitations of Prior Work**: When bitrates fall into the extremely low regime (<0.03 bpp), distortion-driven objectives (e.g., MSE) excessively smooth textures and erase fine structures. More critically, existing perceptual video codecs (e.g., GLC-Video, DiffVC) integrate image-domain generative priors that are inherently static and lack temporal modeling capacity.
3. **Key Challenge**: Video imposes strict requirements on temporal consistency. Even when conditioned on adjacent frames, image-based generative priors cannot capture long-range temporal structure, causing the recovered appearance to drift over time and producing severe perceptual flickering, particularly at extremely low bitrates.
4. **Goal**: (a) How to introduce a video-native generative prior into neural video compression? (b) How to perform sequence-level refinement in the spatiotemporal latent space rather than per-frame enhancement? (c) How to adapt the diffusion prior to compression-induced degradations?
5. **Key Insight**: Video diffusion models (especially DiT architectures) learn spatiotemporal latent representations from large-scale video data, capturing appearance, motion, and long-range dependencies. This makes VDMs ideal generative priors for video compression, reframing decoding as a sequence-level conditional denoising process.
6. **Core Idea**: A pretrained VideoDiT (Wan2.1) is used as a video-native prior. Rather than denoising from pure Gaussian noise, the model performs flow-matching refinement starting from the compressed spatiotemporal latent representation, learning a correction term to adapt to compression-induced degradations.

## Method

### Overall Architecture

GNVC-VD processes an input video $V \in \mathbb{R}^{(1+T) \times H \times W \times 3}$ as follows: (1) A 3D causal VAE encoder $\mathcal{E}$ (from Wan2.1) encodes the video into a spatiotemporal latent sequence $\boldsymbol{x}_1 = \{l_t\}_{t=1}^{1+T/4}$; (2) a contextual transform coding module compresses the latents and generates a bitstream; (3) a VideoDiT-based flow-matching latent refinement module applies sequence-level generative denoising to the decoded latent sequence; (4) a 3D causal decoder $\mathcal{D}$ reconstructs the video. The pipeline tightly couples transform coding compression with diffusion-based generative refinement.

### Key Designs

1. **Contextual Latent Codec**:

    - **Function**: Exploits temporal correlations to compress spatiotemporal latent representations.
    - **Mechanism**: The latent sequence is partitioned along the temporal axis. The anchor latent $l_1$ (corresponding to an I-frame) is coded with an independent transform coding module. Each predicted latent $\{l_t\}_{t>1}$ is conditioned on the previously decoded result $\hat{l}_{t-1}$ to reduce temporal redundancy: $\hat{y}_t = \text{Quant}(g_a(l_t | f_{t-1}))$, $\hat{l}_t = g_s(\hat{y}_t, f_{t-1})$, where $f_{t-1}$ denotes temporal context features extracted from $\hat{l}_{t-1}$. Quantized latents are entropy-coded via a learned probability model.
    - **Design Motivation**: Following the conditional coding philosophy of DCVC-RT, this design produces compact, motion-aware latent representations that maintain temporal continuity and provide a solid foundation for subsequent diffusion refinement.

2. **Flow-Matching Latent Refinement Module**:

    - **Function**: Leverages a pretrained VideoDiT as a video-native prior to jointly enhance the entire frame sequence in the 3D latent space.
    - **Mechanism**: The compressed latent $\boldsymbol{x}_c$ can be viewed as a perturbed version of the original latent $\boldsymbol{x}_1$: $\boldsymbol{x}_c = \boldsymbol{x}_1 + \boldsymbol{e}$. Rather than starting from pure noise, partial noise is injected into $\boldsymbol{x}_c$: $\boldsymbol{x}_{t_N} = t_N \boldsymbol{x}_c + (1-t_N)\boldsymbol{x}_0$ (with $t_N=0.7$), defining a continuous probability flow path from $t_N$ to 1 for refinement. The target velocity field is decomposed as $\boldsymbol{v}_\tau = \underbrace{(\boldsymbol{x}_1 - \boldsymbol{x}_0)}_{\boldsymbol{v}_{\text{pre-train}}} - \underbrace{\frac{t_N}{1-t_N}(\boldsymbol{x}_c - \boldsymbol{x}_1)}_{\Delta \boldsymbol{v}_{\text{fine}}}$, where $\boldsymbol{v}_{\text{pre-train}}$ is the pretrained diffusion model's velocity field and $\Delta \boldsymbol{v}_{\text{fine}}$ is a correction term adapting to compression degradation. Refinement is completed via $L=5$ steps of deterministic flow integration.
    - **Design Motivation**: The key innovation is to perform "short-path" refinement starting from the compressed latent rather than denoising from scratch, efficiently exploiting the fact that $\boldsymbol{x}_c$ already lies close to the data manifold. The velocity field decomposition cleanly decouples pretrained knowledge from compression-specific adaptation.

3. **Compression-Aware Conditioning Adapter**:

    - **Function**: Injects compression-domain contextual information into intermediate layers of the VideoDiT.
    - **Mechanism**: Conditioning adapter layers are inserted into the transformer blocks of the VideoDiT, receiving the context feature sequence $\{f_t\}_{t=1}^{1+T/4}$ as conditional input to modulate intermediate DiT representations. These adapters estimate the correction term $\Delta \boldsymbol{v}_{\text{fine}}$, aligning the generative prior with the compressed latent distribution.
    - **Design Motivation**: Directly applying a pretrained VideoDiT to denoise compressed latents is suboptimal due to the distributional gap between compressed and natural video latents. The adapter provides compression-domain prior knowledge, enabling the diffusion model to perceive compression artifacts and perform targeted restoration.

### Loss & Training

A two-stage training strategy is employed:

- **Stage I — Latent-Level Alignment**: $\mathcal{L}_{\text{latent}} = R(\hat{y}) + \lambda_r \|\tilde{\boldsymbol{x}}_1 - \boldsymbol{x}_1\|_2^2 + \mathcal{L}_{\text{CFM}}$, where $\mathcal{L}_{\text{CFM}}$ is the conditional flow matching loss. This ensures that refined latents are consistent with ground-truth latents on the diffusion manifold.
- **Stage II — Pixel-Level Fine-Tuning**: $\mathcal{L}_{\text{pixel}} = R(\hat{y}) + \lambda_r(\|V - \tilde{V}\|_2^2 + \lambda_{\text{lpips}}\mathcal{L}_{\text{LPIPS}}(V,\tilde{V}) + \|\boldsymbol{x}_c - \boldsymbol{x}_1\|_2^2 + \|\tilde{\boldsymbol{x}}_1 - \boldsymbol{x}_1\|_2^2)$, incorporating LPIPS perceptual loss for end-to-end pixel-domain optimization.

This progressive strategy first bridges the gap between the codec latent space and the diffusion manifold before fine-tuning for perceptual quality.

## Key Experimental Results

### Main Results

**Perceptual Quality Comparison (BD-Rate %, anchored to VVC; lower is better):**

| Method | HEVC-B LPIPS | MCL-JCV LPIPS | UVG LPIPS | UVG DISTS |
|--------|-------------|--------------|----------|----------|
| GLC-Video | -79.1% | -74.8% | -60.0% | -10.3% |
| **GNVC-VD** | **-89.4%** | **-90.8%** | **-86.5%** | **-96.1%** |

GNVC-VD achieves the best perceptual quality across all benchmarks and metrics, outperforming GLC-Video by a further 10–26 percentage points in BD-Rate.

**Temporal Consistency Comparison (HEVC-B):**

| Method | $E_{\text{warp}} \downarrow$ | CLIP-F $\uparrow$ |
|--------|--------------------------|-------------------|
| GLC-Video | 86.5 | 0.979 |
| GNVC-VD | 66.6 | 0.982 |
| HEVC | 23.3 | 0.982 |

### Ablation Study

| Configuration | HEVC-B BD-LPIPS | UVG BD-LPIPS | Note |
|---------------|----------------|-------------|------|
| Full model | 0 | 0 | Baseline |
| W/o Latent Refinement | +0.181 | +0.159 | Removing diffusion refinement causes severe over-smoothing |
| W/o Stage I Loss | +0.016 | +0.016 | Removing latent alignment degrades detail recovery |
| W/o Stage II Loss | +0.252 | +0.242 | Removing pixel-level fine-tuning causes the most severe degradation |

### Key Findings

- **The diffusion refinement module contributes most**: Its removal causes a BD-LPIPS degradation of +0.181, producing severely over-smoothed results, confirming that the video diffusion prior is central to perceptual quality recovery.
- **Stage II pixel-level fine-tuning is indispensable**: Its removal leads to the worst degradation (+0.252), demonstrating that latent-space alignment alone is insufficient for optimal perceptual reconstruction.
- **Source of temporal consistency gains**: GNVC-VD achieves an $E_{\text{warp}}$ of 66.6, far below GLC-Video's 86.5. Inter-frame texture drift and flickering in GLC-Video are clearly visible in spatiotemporal visualizations.
- Traditional codecs (HEVC/VVC) achieve the lowest $E_{\text{warp}}$ due to excessive smoothing, which constitutes a form of "false stability" rather than genuine temporal coherence.

## Highlights & Insights

- **First introduction of a video-native diffusion prior into NVC**: This work bypasses the limiting pathway of "image prior → video compression" and directly employs a video diffusion model to capture spatiotemporal dependencies, fundamentally addressing inter-frame flickering. The principle of "solving a sequence-level problem with a sequence-level prior" is both natural and effective.
- **Partial denoising from compressed latents**: Starting from the compressed latent rather than pure noise substantially reduces the number of denoising steps (only 5 are required) while preserving the generative model's capacity for detail recovery. The formal decomposition of the velocity field into pretrained and correction components is also elegant.
- **Design rationale of the two-stage training strategy**: Direct end-to-end training is unstable; the progressive approach of latent alignment followed by pixel-level fine-tuning resolves the distributional mismatch between the diffusion manifold and compressed latents. This paradigm is transferable to other settings where pretrained generative models are adapted to downstream tasks.

## Limitations & Future Work

- **Computational efficiency**: Diffusion refinement requires multiple denoising steps (5 steps), making decoding several times slower than traditional codecs and hindering practical deployment.
- **Further optimization of the transform coding module**: The authors themselves acknowledge that the current contextual transform coding module can be improved in terms of efficiency.
- **Training data and sequence length limitations**: Training uses only 13-frame Vimeo sequences; generalization to longer videos remains unvalidated.
- **Evaluation limited to extremely low bitrates (<0.03 bpp)**: Whether the approach remains advantageous at moderate bitrates is not discussed.
- Accelerating diffusion refinement (e.g., via distillation or consistency models) is an important future direction.

## Related Work & Insights

- **vs. GLC-Video**: GLC-Video applies an image diffusion prior for per-frame enhancement, leading to texture drift and flickering. GNVC-VD uses a video diffusion prior for sequence-level refinement, fundamentally resolving temporal inconsistency and achieving consistently superior BD-Rate performance.
- **vs. DCVC-RT**: DCVC-RT is among the strongest learned codecs, yet it over-smooths at extremely low bitrates. GNVC-VD augments it with diffusion refinement, achieving up to 98% BD-DISTS improvement on UVG.
- This work demonstrates the substantial value of video generative foundation models in compression tasks and opens a new direction for "generative codecs."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First introduction of a video diffusion prior into NVC; the flow-matching refinement design starting from compressed latents is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-benchmark comparisons with complete ablations; analysis of moderate bitrates and complex motion scenarios is lacking.
- Writing Quality: ⭐⭐⭐⭐ Technical pipeline is clearly presented, mathematical derivations are rigorous, and figures are informative.
- Value: ⭐⭐⭐⭐⭐ Points the way toward next-generation perceptual video compression; the paradigm of video diffusion prior + codec has broad impact.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior](dreamshot_storyboard_synthesis.md)
- [\[CVPR 2026\] LightMover: Generative Light Movement with Color and Intensity Controls](lightmover_generative_light_movement_with_color_and_intensity_controls.md)
- [\[ICLR 2026\] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](../../ICLR2026/video_generation/javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)
- [\[ICLR 2026\] Arbitrary Generative Video Interpolation](../../ICLR2026/video_generation/arbitrary_generative_video_interpolation.md)
- [\[CVPR 2026\] Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)

<!-- RELATED:END -->

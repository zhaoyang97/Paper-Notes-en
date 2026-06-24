---
title: >-
  [Paper Note] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution
description: >-
  [CVPR 2025][Video Generation][Video Super-Resolution] PatchVSR is the first to employ a pre-trained video diffusion model (T2V) for patch-wise video super-resolution. By leveraging a dual-branch adapter (local patch branch + global context branch) and a training-free multi-patch joint modulation scheme, it achieves high-fidelity 4K video super-resolution based on a 512×512 resolution base model while significantly improving computational efficiency.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Super-Resolution"
  - "Diffusion Models"
  - "Patch Processing"
  - "Arbitrary Resolution"
  - "Dual-Branch Adapter"
date: 2026-05-08
content_hash: b2e73fff1081731a
---

# PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution

**Conference**: CVPR 2025  
**arXiv**: [2509.26025](https://arxiv.org/abs/2509.26025)  
**Code**: None  
**Area**: Diffusion Models / Video Super-Resolution  
**Keywords**: Video Super-Resolution, Diffusion Models, Patch Processing, Arbitrary Resolution, Dual-Branch Adapter

## TL;DR

PatchVSR is the first to employ a pre-trained video diffusion model (T2V) for patch-wise video super-resolution. By leveraging a dual-branch adapter (local patch branch + global context branch) and a training-free multi-patch joint modulation scheme, it achieves high-fidelity 4K video super-resolution based on a 512×512 resolution base model while significantly improving computational efficiency.

## Background & Motivation

**Background**: Video super-resolution (VSR) has long been a core challenge in computer vision. Traditional CNN/Transformer methods are limited by model capacity and data coverage, making it difficult to generate realistic details and textures. Recently, diffusion generative models have brought new opportunities to VSR due to their powerful generation capabilities, with methods like VEnhancer and Upscale-A-Video utilizing pre-trained diffusion models for video enhancement.

**Limitations of Prior Work**: Existing diffusion-based VSR methods process videos at the full-frame level, inheriting the fixed resolution limitations of the base models. Due to the full-attention nature of Transformers, pre-trained models typically support only a fixed number of tokens (such as various aspect ratios of 512×512). Scaling up to higher resolutions requires substantial training resources and high-quality, high-resolution datasets, which are currently unavailable. Consequently, current methods cannot support arbitrary-resolution outputs, resulting in low inference efficiency and high storage/VRAM requirements.

**Key Challenge**: Feature attention in super-resolution tasks is more localized compared to generation tasks (since low-resolution references are available, details can be generated based on neighborhood semantics, eliminating the need for global consistency). However, current methods still perform expensive full-attention calculations across the entire frame, wasting substantial resources.

**Goal**: Leverage the generative priors of pre-trained video diffusion models for patch-wise video super-resolution, achieving high-fidelity VSR at arbitrary resolutions without modifying the resolution limit of the base model.

**Key Insight**: The insight that attention in super-resolution is more localized makes patch-wise processing possible—namely, slicing the input video into patches of sizes compatible with the pre-trained model, enhancing them independently, and then stitching them together. However, a major challenge is that pre-trained models are trained on full frames, and patch-level generation leads to severe performance degradation.

**Core Idea**: Inject local content fidelity (via the patch branch) and global semantic context (via the global branch) into patch-level generation using a dual-branch adapter, adapting the pre-trained full-frame T2V model for patch-level detail generation.

## Method

### Overall Architecture

Given a low-resolution video $\mathbf{V}_l \in \mathbb{R}^{F \times H \times W \times 3}$ and an upsampling factor $k$, it is first upsampled to the target resolution using bicubic interpolation. The upsampled result is then cropped into patches $\{\mathbf{P}_i \in \mathbb{R}^{F \times h \times w \times 3}\}_{i=1}^N$, where the patch size $(h, w)$ matches the generation size of the pre-trained T2V model. Simultaneously, the full video is scaled down to $\mathbf{G} \in \mathbb{R}^{F \times h \times w \times 3}$ to serve as global guidance. Each patch is enhanced via the dual-branch adapter, and then stitched together using multi-patch joint modulation to produce the final output.

### Key Designs

1. **Patch Condition Branch**:

    - Function: Extract features from the input patch to guide the base model in synthesizing details while maintaining content fidelity.
    - Mechanism: An adapter composed of several Transformer blocks is used to extract features from the input patch $\mathbf{P}_i$ and inject them into the output of each block in the base model. LoRA is also used to fine-tune the base model to adapt it to the patch-level data distribution (as the local patch data distribution differs significantly from full video frames). The text prompt is replaced with a fixed prompt in the patch branch to prevent mismatches between the patch and global semantic information.
    - Design Motivation: A lightweight approach is needed to inject input patch information into the pre-trained model to guide detail synthesis based on the input content; compared to direct concatenation in the noise space, the adapter approach is more lightweight and only requires training the newly added branch.

2. **Global Context Branch**:

    - Function: Extract context information from the full input video to bridge the gap in generation caused by incomplete patch semantics.
    - Mechanism: A Transformer encoder processes the downscaled full video into context features with a quarter of the token count, which are integrated into each block of the base model via newly added cross-attention modules: $\{\mathbf{Q}, \mathbf{K}_g, \mathbf{V}_g\}$, where $\mathbf{Q}$ shares the existing text cross-attention. A key design is to concatenate a binary position mask $\mathbf{M}_i$ in the global input, indicating the target patch's position within the full frame to make the context guidance more targeted.
    - Design Motivation: Pre-trained models are trained on full frames, whereas patches often contain incomplete or ambiguous semantics (such as only a portion of an object); the global context helps the model understand the overall environment of the patch, thereby generating more natural details.

3. **Multi-Patch Joint Modulation**:

    - Function: Ensure that independently enhanced patches maintain visual consistency after stitching and eliminate boundary artifacts.
    - Mechanism: Improved based on MultiDiffusion—first, the video is divided into non-overlapping patches, and then auxiliary patches are created at the boundaries of adjacent patches (combining half of each adjacent patch) to form a new set of patches with 50% overlap. During each denoising step, a weighted fusion is applied to the overlapping regions. Instead of simple averaging (which leads to black holes or seams), a spatial weight map is used, where the influence of the auxiliary patch gradually decays from the center split line to both sides.
    - Design Motivation: Due to the randomness of the generation process, multiple plausible high-resolution solutions may be yielded in degraded or uncertain areas, leading to inconsistent texturing at the boundaries between adjacent patches.

### Loss & Training

- The Rectified Flow framework is utilized, where the forward process is defined as a straight path between the data distribution and a standard Gaussian distribution: $z_t = (1-t)z_0 + t\epsilon$.
- The training data consists of 460K self-collected high-quality video-text pairs with resolutions ranging from 1024×1024 to 2K.
- Degradation processing: HR videos are first bilinearly downsampled and then upsampled back to the original resolution, followed by random cropping of 512×512 regions.
- Noise augmentation: Noise is injected into the input latent (timestep 200-300) to preserve structure.
- The downsampling factor, cropping position, and noise timestep are all used as conditioning encodings.

## Key Experimental Results

### Main Results

SynVideo30 Dataset $\times 4$ Super-Resolution (2K):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | MUSIQ↑ | Aesthetics↑ |
|------|-------|-------|--------|--------|-------------|
| RealBasicVSR | 33.507 | 0.776 | 0.185 | 49.557 | 0.496 |
| Upscale-A-Video | 33.432 | 0.728 | 0.205 | 49.839 | 0.494 |
| VEnhancer | 28.856 | 0.697 | 0.199 | 43.538 | 0.503 |
| **PatchVSR** | 30.857 | 0.732 | **0.183** | **50.695** | **0.520** |

Computational Efficiency (2K Video):

| Method | Time(s)↓ | GPU Memory (GB)↓ |
|------|---------|---------|
| LaVie-SR | 2261 | 68 |
| Upscale-A-Video | 2743 | 47 |
| VEnhancer | 1562 | 62 |
| **PatchVSR** | **680** | **40** |

### Ablation Study

Ablation study of components on VideoGen30:

| Component | DOVER↑ | MUSIQ↑ | Aesthetics↑ |
|------|--------|--------|-------------|
| w/o global branch | 0.502 | 46.074 | 0.589 |
| w/o LoRA | 0.582 | 50.496 | 0.600 |
| w/o location embed | 0.574 | 50.084 | 0.601 |
| w/o fixed prompt | 0.562 | 48.133 | 0.597 |
| Full PatchVSR | **0.590** | **50.559** | **0.602** |

The removal of the global branch leads to the most significant performance drop (DOVER drops from 0.590 to 0.502).

### Key Findings

- PatchVSR achieves the best performance in LPIPS (perceptual fidelity), but does not yield the highest PSNR/SSIM, since generative methods tend to produce rich details rather than pixel-level alignment.
- The patch processing strategy confines self-attention to individual patches, reducing the computational complexity from $n^2$ for full frames to $n \times m$ (where $m$ is the number of patch tokens), securing a 2-4$\times$ efficiency improvement.
- Position embedding within the global branch is critical for restoring complete patch semantics.
- The model exhibits good robustness toward semantically incomplete patches (such as those containing only a part of an object).

## Highlights & Insights

- First work to leverage pre-trained video diffusion models for patch-wise VSR, elegantly addressing the fixed resolution limitation.
- Highly significant efficiency advantages: achieving 4K output based on a 512×512 model, with a substantial reduction in both inference time and VRAM footprint.
- Key insight—that attention in VSR is more localized than in generation tasks—provides a sound theoretical foundation for patch processing.
- The multi-patch joint modulation design (auxiliary patches + spatial weight maps) effectively eliminates stitching artifacts.

## Limitations & Future Work

- Notable performance degradation on real-world low-resolution images, as the training data lacks specialized degradation augmentation.
- Inherits the iterative inference scheme of the base model; while already more efficient than full-frame methods, there remains room for optimization.
- While results with different upsampling factors demonstrate flexibility, the controllability of detail generation under extreme factors requires further exploration.
- Future work could integrate techniques like step distillation to further compress inference times.

## Related Work & Insights

- The multi-region joint sampling idea from MultiDiffusion is successfully adapted and applied to video super-resolution in this work.
- Complementary to full-frame methods like VEnhancer and Upscale-A-Video—while those methods pursue global consistency, PatchVSR focuses on high efficiency and resolution flexibility.
- The effectiveness of LoRA fine-tuning in adapting to patch distributions offers a solid reference for similar transfer learning paradigms.
- The dual-branch adapter paradigm (local + global) can be generalized to other generative tasks requiring multi-scale guidance.

## Rating

- **Novelty**: 8/10 — The first exploration of patch-wise video diffusion super-resolution, representing a novel direction.
- **Experimental Thoroughness**: 8/10 — Thorough comparison across multiple datasets and metrics, with detailed ablation studies.
- **Writing Quality**: 8/10 — Clear motivation and detailed methodological description.
- **Value**: 9/10 — High practical value; the efficiency advantage for 4K VSR offers direct benefits for industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatia.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [\[ICLR 2026\] UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers](../../ICLR2026/video_generation/ultravico_breaking_extrapolation_limits_in_video_diffusion_transformers.md)
- [\[CVPR 2025\] Generative Inbetweening through Frame-wise Conditions-Driven Video Generation](generative_inbetweening_through_frame-wise_conditions-driven_video_generation.md)

</div>

<!-- RELATED:END -->

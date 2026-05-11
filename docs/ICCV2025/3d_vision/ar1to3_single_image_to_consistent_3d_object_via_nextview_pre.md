---
title: >-
  [Paper Note] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction
description: >-
  [ICCV 2025][3D Vision][Single-image 3D generation] This paper proposes AR-1-to-3, an autoregressive next-view prediction framework built upon diffusion models. By adopting a progressive "near-to-far" generation strategy…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Single-image 3D generation"
  - "autoregressive multi-view synthesis"
  - "diffusion models"
  - "next-view prediction"
  - "multi-view consistency"
date: 2026-05-08
content_hash: 909ba93e5980580d
---

# AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction

**Conference**: ICCV 2025
**arXiv**: [2503.12929](https://arxiv.org/abs/2503.12929)
**Code**: Available (project page on arXiv)
**Area**: 3D Vision / Single-Image 3D Reconstruction / Novel View Synthesis
**Keywords**: Single-image 3D generation, autoregressive multi-view synthesis, diffusion models, next-view prediction, multi-view consistency

## TL;DR
This paper proposes AR-1-to-3, an autoregressive next-view prediction framework built upon diffusion models. By adopting a progressive "near-to-far" generation strategy, combined with two conditioning mechanisms—Stacked-LE (Stacked Local-feature Encoding) and LSTM-GE (LSTM-based Global-feature Encoding)—the method significantly improves multi-view consistency in single-image-to-multi-view generation. On the GSO dataset, it achieves a PSNR of 13.18 (a 23.5% improvement over InstantMesh's 10.67) and reduces Chamfer Distance to 0.063 (compared to 0.117 for InstantMesh).

## Background & Motivation
Single-image 3D object generation is a fundamental problem in computer vision and graphics. Current mainstream approaches fall into two categories:
1. **Individual view generation** (e.g., Zero123): fine-tunes a diffusion model on (input view, camera pose, target view) triplets and generates each target view independently;
2. **Joint multi-view generation** (e.g., Zero123++): arranges 6 target views into a 3×2 grid image and generates all views simultaneously with a diffusion model.

Both categories share a core limitation: **when the camera pose of a target view deviates significantly from that of the input view, severe geometric and texture inconsistencies arise between the generated and input views**. For example, Zero123++ produces noticeable shape deformation and texture errors for bowls and shoes at views far from the input.

The authors identify the root cause as these methods treating all target views with **equal priority**, without leveraging the observation that views closer to the input view are generated with higher fidelity. This motivates a progressive generation strategy.

## Core Problem
How can the contextual information from already-generated views be fully exploited in single-image-to-multi-view generation, so that target views far from the input also maintain high consistency? The core challenges are: (1) establishing an ordering among target views; and (2) encoding the partially generated view sequence as effective conditioning signals for the diffusion model.

## Method

### Overall Architecture
AR-1-to-3 adopts the same 3×2 grid layout as Zero123++, but replaces single-pass generation with a **three-step autoregressive generation** process. The 6 target views are divided row-wise into 3 pairs (each pair at elevations of 20° and −10°, with 60° azimuth spacing), with 120° azimuth separation between adjacent pairs. Generation proceeds from the pair closest to the input view, progressively expanding to more distant views. At each step, previously generated views are incorporated as additional conditioning. The final 6 views are fed into a pre-trained InstantMesh for 3D reconstruction.

### Key Designs
1. **Stacked-LE (Stacked Local-feature Encoding)**: A local conditioning strategy that handles **pixel-level spatial correspondences**. At autoregressive step $k$, there are $2k-1$ reference views. Each reference view is individually passed through the denoising UNet to extract its key/value matrices at self-attention layers. During denoising of the target views, the key and value matrices from all reference views are concatenated along the spatial dimension and injected into the self-attention layers. Key advantages: (1) it can encode an arbitrary number of reference features; (2) it directly reuses UNet weights without additional network modules.

2. **LSTM-GE (LSTM-based Global-feature Encoding)**: A global conditioning strategy that captures **high-level semantic information**. The CLIP image features of conditioning views are split into two groups by elevation angle, each fed into a separate LSTM module. The hidden state at step $k$ is taken as output, concatenated, and transformed via an MLP and trainable global weights into cross-attention conditioning embeddings. Compared to a simple matrix multiplication (matmul) scheme, LSTM better models sequential dependencies and captures high-level 3D object semantics.

3. **Near-to-far generation order**: Views are generated in order of increasing camera pose distance from the input view. Experiments show this order (Normal) outperforms reversed (Reverse) and random (Random) orderings, with random being worst (PSNR 17.36 vs. 20.28), demonstrating the importance of modeling ordinal relationships among views.

### Loss & Training
- Standard diffusion training objective (v-prediction loss + linear noise schedule, inherited from Zero123++)
- During training, $k \in \{1, 2, 3\}$ is sampled randomly to construct the autoregressive configuration; the first $2k-1$ views serve as conditions, and the last two as targets
- Conditioning images are resized to a random resolution between 128 and 512 to handle varying resolutions; target views are fixed at 320
- Trained for 150k steps, batch size 32, on 8×A100 (80G), learning rate $1 \times 10^{-5}$, with CosineAnnealingWarmRestarts schedule

## Key Experimental Results

### Comparison with SOTA Methods (Image-to-3D, OOD Datasets)

| Method | GSO-PSNR↑ | GSO-LPIPS↓ | GSO-CD↓ | GSO-F-Score↑ | Omni3D-PSNR↑ | Omni3D-CD↓ |
|---|---|---|---|---|---|---|
| Michelangelo | 9.323 | 0.408 | 0.165 | 0.105 | 9.969 | 0.174 |
| SyncDreamer | 10.82 | 0.332 | 0.108 | 0.125 | 9.485 | 0.196 |
| LGM | 9.139 | 0.429 | 0.157 | 0.075 | 10.02 | 0.152 |
| InstantMesh | 10.67 | 0.338 | 0.117 | 0.135 | 9.91 | 0.178 |
| **AR-1-to-3** | **13.18** | **0.232** | **0.063** | **0.258** | **10.25** | **0.148** |

### Ablation Study (Objaverse Test Set, Novel View Synthesis)

| Configuration | PSNR↑ | LPIPS↓ | SSIM↑ |
|---|---|---|---|
| Baseline (Zero123++) | 14.83 | 0.201 | 0.815 |
| + Stacked-LE | 17.59 | 0.174 | 0.833 |
| + LSTM-GE | 17.91 | 0.170 | 0.836 |
| + Both (Full AR-1-to-3) | **20.28** | **0.121** | **0.857** |

### Ablation Study
- **Both encoding strategies are individually effective and complementary**: Stacked-LE alone improves PSNR by 2.76; LSTM-GE alone by 3.08; combined, the improvement is 5.45.
- **Generation order matters**: Normal (20.28) ≈ Reverse (20.19) >> Random (17.36); the reversed order performs comparably to the normal order because, on a circular camera trajectory, reversing the order is equivalent to a near-to-far traversal in the opposite direction.
- **LSTM outperforms matmul**: The matmul scheme introduces biases in global semantic understanding (e.g., shape distortion), whereas LSTM better captures sequential semantics.

## Highlights & Insights
- **Concise and effective core insight**: By leveraging the empirical observation that views closer to the input are generated with higher fidelity, the paper transforms parallel generation into a near-to-far autoregressive process without requiring complex 3D priors.
- **Plug-and-play design**: Both Stacked-LE and LSTM-GE introduce minimal additional parameters. Stacked-LE directly reuses UNet weights, while LSTM-GE requires only two lightweight LSTM modules.
- **Large gains over baseline**: Compared to the Zero123++ baseline, PSNR improves from 14.83 to 20.28 (+36.7%) on novel view synthesis.
- **Strong generalization on OOD datasets**: AR-1-to-3 outperforms all compared methods across all metrics on GSO and Omni3D.

## Limitations & Future Work
- **Inference speed**: Three-step autoregressive generation is approximately 3× slower than single-pass generation, as each step requires a full diffusion sampling process.
- **Fixed view layout**: The method is strictly tied to Zero123++'s 3×2 grid and specific camera configurations (20°/−10° elevation, 60° azimuth spacing), making it difficult to generalize to arbitrary viewpoints.
- **Six-view constraint**: Only 6 views are generated, which may be insufficient for 3D reconstruction of complex objects.
- **Dependency on InstantMesh**: Reconstruction quality is bounded by the downstream reconstruction model; no joint optimization is performed.
- **Limited to SD 1.5**: The method is built on an SD 1.5-based UNet architecture; the potential of newer architectures such as DiT remains unexplored.

## Related Work & Insights
- **vs. Zero123++**: Both use the same grid layout and 6-view configuration, but Zero123++ generates all views in parallel in a single pass, whereas AR-1-to-3 uses three autoregressive steps. AR-1-to-3 achieves +2.51 PSNR and −0.106 LPIPS on GSO, with particularly pronounced advantages under large viewpoint changes.
- **vs. Cascade-Zero123**: The most closely related work, which also conditions on previously generated views. However, Cascade-Zero123 first uses a multi-view diffusion model to generate a large number of auxiliary views, then uses a second diffusion model to generate specific target views. AR-1-to-3 instead models the distance relationship between target and input views, progressively generating with a single model.
- **vs. SyncDreamer**: SyncDreamer establishes correspondences across 16 views via 3D-aware attention but still generates all views simultaneously. AR-1-to-3 requires no 3D-aware modules and implicitly establishes consistency through autoregressive ordering.

### Potential Extensions
The near-to-far strategy of AR-1-to-3 is potentially extensible to video diffusion models (e.g., SV3D), where temporal autoregression could be replaced by spatial autoregression ordered by viewpoint distance, addressing consistency in long-sequence multi-view generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The core insight is concise and effective (near-view priority); Stacked-LE and LSTM-GE are well-motivated, though of moderate technical complexity.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 3 datasets with ablations covering key components and generation orderings; however, inference speed comparisons and additional SOTA baselines are missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rich figures and tables, convincing motivation; some mathematical notation could be made more concise.
- **Value**: ⭐⭐⭐⭐ The proposed autoregressive view generation paradigm is inspiring and yields large performance gains, though its tight coupling to the Zero123++ framework limits generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MV-Adapter: Multi-view Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)
- [\[ICCV 2025\] FlexGen: Flexible Multi-View Generation from Text and Image Inputs](flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)
- [\[ICCV 2025\] TAPNext: Tracking Any Point (TAP) as Next Token Prediction](tapnext_tracking_any_point_tap_as_next_token_prediction.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[ICCV 2025\] WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions](wonderplay_dynamic_3d_scene_generation_from_a_single_image_and_actions.md)

</div>

<!-- RELATED:END -->

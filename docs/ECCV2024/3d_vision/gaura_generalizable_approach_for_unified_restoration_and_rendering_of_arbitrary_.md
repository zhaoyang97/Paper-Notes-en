---
title: >-
  [Paper Note] GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views
description: >-
  [ECCV 2024][3D Vision][Novel View Synthesis] GAURA is proposed, a unified restoration and rendering framework based on generalizable NeRF. By utilizing learnable degradation-aware latent codes, it dynamically adapts to different image degradation types during the feature aggregation and rendering stages, enabling the rendering of clean novel views from degraded images without scene-specific optimization.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Image Restoration"
  - "Generalizable Degradation Handling"
  - "Generalizable NeRF"
  - "Transformer"
date: 2026-05-08
content_hash: d6e034c733f5da86
---

# GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views

**Conference**: ECCV 2024  
**arXiv**: [2407.08221](https://arxiv.org/abs/2407.08221)  
**Code**: [https://vinayak-vg.github.io/GAURA](https://vinayak-vg.github.io/GAURA)  
**Area**: 3D Vision  
**Keywords**: Novel View Synthesis, Image Restoration, Generalizable Degradation Handling, Generalizable NeRF, Transformer

## TL;DR

GAURA is proposed, a unified restoration and rendering framework based on generalizable NeRF. By utilizing learnable degradation-aware latent codes, it dynamically adapts to different image degradation types during the feature aggregation and rendering stages, enabling the rendering of clean novel views from degraded images without scene-specific optimization.

## Background & Motivation

**Background**: Neural rendering methods such as NeRF can achieve photorealistic novel view synthesis from multi-view images, but they require perfect input image quality.

**Limitations of Prior Work**: Real-world images are often corrupted by degradations such as low light, motion blur, haze, rain, and snow. Existing methods typically require explicitly modeling the physical degradation process for each specific degradation type (e.g., SeaThru-NeRF for underwater imaging, Deblur-NeRF for blur kernels). This not only demands substantial custom design but also limits applicability to specific degradation types.

**Key Challenge**: Explicitly modeling degradation processes leads to a contradiction between **more difficult inverse problems** and **zero generalization capability**—each new degradation type requires redesigning the method, restricting practical applications.

**Goal**: Build a 3D restoration and rendering framework that can simultaneously generalize to different scenes and different degradation types without requiring custom designs for each degradation or scene-specific optimization.

**Key Insight**: Drawing inspiration from the success of "all-in-one" methods in 2D image restoration, the degradation information is implicitly encoded into learnable parameters, and combined with a generalizable NeRF framework (GNT) to achieve cross-scene inference.

**Core Idea**: Condition the epipolar feature aggregation and rendering process with learnable degradation-aware latent codes, allowing the network to implicitly learn the image formation process under different degradations.

## Method

### Overall Architecture

GAURA is built upon GNT (Generalizable NeRF Transformer) and consists of three core stages:
1. **Feature Extraction**: UNet extracts convolutional features from the degraded input views.
2. **Epipolar Feature Aggregation**: View Transformer aggregates multi-view features along epipolar lines.
3. **Ray Rendering**: Ray Transformer accumulates point features along rays to obtain pixel colors.

A Degradation-Aware Latent Module (DLM) is inserted into each stage to dynamically adjust network behavior using degradation latent codes.

### Key Designs

1. **Degradation-Aware Latent Module (DLM)**: Learnable latent codes $\{\boldsymbol{L}_i\}_{i=1}^M$ are maintained for $M$ degradation types. Inspired by HyperNetwork, the latent code is mapped to MLP weights to perform degradation-specific transformations on input features:

$$\text{DLM}(\boldsymbol{X}, \boldsymbol{D}) = W \cdot \boldsymbol{X}, \quad W = \mathcal{F}_{\text{latent}}(\boldsymbol{L}_D | \{\boldsymbol{L}_i\}_{i=1}^M)$$

Design Motivation: The image formation processes of different degradations share commonalities. Sharing the backbone network parameters while using degradation-specific latent codes is more efficient than cloning the entire network for each degradation type.

2. **Adaptive Residual Module (ARM)**: The latent codes of DLM are independent of the actual input and cannot capture variations within the same degradation type. ARM extracts residual features $\boldsymbol{S}$ from the input view closest to the target view to enhance adaptability to varying degradation intensities:

$$\text{DLM}_{\text{w/ residue}}(\boldsymbol{X}, \boldsymbol{D}) = \text{DLM}(\boldsymbol{X}, \boldsymbol{D}) + \boldsymbol{S}, \quad \boldsymbol{S} = \text{pool}(\mathcal{F}_{\text{residue}}(\boldsymbol{I}_{\text{nearest}}))$$

where $\mathcal{F}_{\text{residue}}$ is a small convolutional network and pool is global average pooling.

3. **Degradation-Aware Three-Stage Conditioning**: 

    - **Feature UNet**: DLMs are inserted before upsampling in each decoder layer to implicitly enrich features with degradation information.
    - **View Transformer**: All Q/K/V projections of the vanilla MLP are replaced with DLMs to inject degradation priors during the scene representation stage (as degradation causes multi-view feature inconsistency, calibration is required).
    - **Ray Transformer**: Only the V projection is replaced with DLM (since Q/K are responsible for computing attention scores as geometric blending weights, which are independent of the degradation type).

Formal description:

$$\boldsymbol{f}(t) = \mathcal{F}_{view}(\{\text{DLM}(\boldsymbol{G})_q, \text{DLM}(\boldsymbol{G})_k, \text{DLM}(\boldsymbol{G})_v\})$$

$$\boldsymbol{c}(r) = \mathcal{F}_{point}(\{\boldsymbol{H}_q, \boldsymbol{H}_k, \text{DLM}(\boldsymbol{H})_v\})$$

### Loss & Training

- **Loss**: $\mathcal{L} = \mathcal{L}_{\text{MSE}}(\hat{I}_{target}, I_{target}) + \mathcal{L}_{\text{LPIPS}}(\hat{I}_{target}, I_{target})$
- **Training Data**: Using IBRNet and LLFF training sets, 4 types of degradations (low light, motion blur, haze, rain) are synthetically generated. During training, one degradation and its intensity are randomly sampled at each iteration.
- **Initialization**: Reusing pre-trained GNT checkpoints, and adding degradation-aware modules on top.
- **Optimization**: Adam optimizer, initial learning rate $5 \times 10^{-4}$, trained for 400K steps.
- **Inference**: Accepts 10 source views, requiring no scene-specific optimization.
- **Finetuning to New Degradations**: Requires adding only one new latent code (<5% extra parameters), freezing the rest of the parameters, and finetuning on 8 scenes for about 10K steps.

## Key Experimental Results

### Main Results (Real-world Degraded Scenes)

| Model | Generalizable (Scene/Degradation) | Low-light PSNR↑ | Motion Blur PSNR↑ | Haze PSNR↑ |
|------|:---:|:---:|:---:|:---:|
| NeRF-Restore | ✗/✗ | 15.42 | 23.27 | 13.87 |
| 3D Restore | ✗/✗ | 17.64 | **25.65** | - |
| GNT-Restore | ✓/✗ | 16.36 | 21.97 | 14.16 |
| GNT-(AIO) Restore | ✓/✓ | 17.90 | 20.88 | 16.68 |
| **GAURA (Ours)** | ✓/✓ | **19.91** | 22.12 | **16.82** |

### LLFF-Corrupted Benchmark

| Model | Low-light PSNR↑ | Motion Blur PSNR↑ | Haze PSNR↑ | Rain PSNR↑ |
|------|:---:|:---:|:---:|:---:|
| GNT-AirNet | 18.20 | 21.08 | 14.55 | 20.71 |
| GNT-PromptIR | 17.67 | 21.01 | 15.81 | 20.73 |
| GNT-DA-CLIP | 12.46 | 21.96 | 8.36 | 20.24 |
| **GAURA** | **21.98** | **22.61** | **18.95** | **22.61** |

### Ablation Study (AlethNeRF Low-light Enhancement)

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Description |
|------|:---:|:---:|:---:|------|
| Vanilla GNT | 17.69 | 0.704 | 0.406 | No degradation-aware module |
| +DLM in $\mathcal{F}_{conv}$ | 17.98 | 0.680 | 0.427 | DLM only in feature extraction |
| +DLM in $\mathcal{F}_{conv}$, $\mathcal{F}_{view}$ | 18.73 | 0.727 | 0.394 | +View Transformer |
| +DLM in all three | 19.37 | 0.714 | 0.363 | +Ray Transformer |
| **+ARM (Ours)** | **19.91** | **0.736** | **0.352** | Adaptive Residual Module |

### Finetuning to Unseen Degradations

| Model | Desnowing PSNR↑ | Defocus Deblurring PSNR↑ |
|------|:---:|:---:|
| Vanilla-GNT | 21.96 | 20.95 |
| GNT-Restore | 20.24 | 21.13 |
| **GAURA (Finetuned)** | **22.61** | **21.34** |

### Key Findings

- Although GAURA is a general-purpose method, it outperforms the specially designed 3D Restore method (AlethNeRF) on low-light tasks, indicating that implicit learning is superior to explicit degradation modeling.
- It comprehensively outperforms all 2D All-in-One + GNT pipelines on LLFF-Corrupted, with a maximum gain of up to 10+ dB (dehazing task).
- Finetuning to a new degradation requires only <5% parameters, 8 scenes, and 10K steps of training.
- Multiple simultaneous degradations can be handled by interpolating latent codes: $\boldsymbol{L} = \alpha \boldsymbol{L}_{D_1} + (1-\alpha) \boldsymbol{L}_{D_2}$.

## Highlights & Insights

- **Elegant Design of a Unified Framework**: Encoding degradation information into lightweight latent codes decoupled from the main network achieves the goal of "one model to handle all degradations".
- **Implicit vs. Explicit Degradation Modeling**: This work demonstrates that in 3D scene restoration, implicit learning of degradation processes can outperform explicit physical modeling, mirroring trends in the 2D image restoration domain.
- **Asymmetric Design (Replacing Q/K/V in View Transformer vs. Only V in Ray Transformer)**: Reflects a deep understanding of geometric consistency (which is degradation-independent) versus appearance restoration (which is degradation-dependent).
- **Practical Finetuning Mechanism**: The decoupled design of latent codes allows expansion to new degradation types with minimal data and computation.

## Limitations & Future Work

- Requires prior knowledge of the degradation type (non-blind restoration); future work could explore automatic identification of degradation types.
- Based on epipolar geometry, performance is limited on sparse 360-degree scenes and complex light transport scenes.
- The degradation-aware modules could be integrated into faster representations such as 3D Gaussian Splatting.
- Lack of evaluation benchmarks for real-world multi-degradation scenes.

## Related Work & Insights

- GNT provides a powerful Transformer infrastructure for generalizable NeRF; this work demonstrates its suitability as a backbone for unified restoration frameworks.
- Directly applying 2D All-in-One restoration methods (AirNet, PromptIR) to multi-view setups leads to inconsistency, indicating that 3D consistency must be resolved at the representation level.
- The HyperNetwork-styled approach (latent $\rightarrow$ network weights) is highly effective in this context.
- Future work can combine the explicit representation capabilities of 3DGS with the degradation-aware design of GAURA.

## Rating

- Novelty: ⭐⭐⭐⭐ The first 3D restoration method that simultaneously generalizes across scenes and degradation types, featuring a clever DLM+ARM design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 degradation types, multiple benchmarks, complete ablations, finetuning verification, and multi-degradation combinations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-explained motivation, and standard mathematical formulations.
- Value: ⭐⭐⭐⭐ Provides important insights for the 3D scene restoration field, and the unified framework holds high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Flying with Photons: Rendering Novel Views of Propagating Light](flying_with_photons_rendering_novel_views_of_propagating_light.md)
- [\[ECCV 2024\] CaesarNeRF: Calibrated Semantic Representation for Few-Shot Generalizable Neural Rendering](caesarnerf_calibrated_semantic_representation_for_few-shot_generalizable_neural_.md)
- [\[ECCV 2024\] A Direct Approach to Viewing Graph Solvability](a_direct_approach_to_viewing_graph_solvability.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->

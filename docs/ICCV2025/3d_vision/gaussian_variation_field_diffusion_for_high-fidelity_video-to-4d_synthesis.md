---
title: >-
  [Paper Note] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis
description: >-
  [ICCV 2025][3D Vision][4D generation] This paper proposes a video-to-4D generation framework that encodes animation data directly into a compact Gaussian variation field latent space via a Direct 4DMesh-to-GS Variation F…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "4D generation"
  - "video-to-4D"
  - "Gaussian variation field"
  - "diffusion model"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: 63c6d517f41ae74c
---

# Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis

**Conference**: ICCV 2025
**arXiv**: [2507.23785](https://arxiv.org/abs/2507.23785)  
**Code**: [GVFDiffusion.github.io](https://GVFDiffusion.github.io)  
**Area**: 3D Vision
**Keywords**: 4D generation, video-to-4D, Gaussian variation field, diffusion model, 3D Gaussian Splatting

## TL;DR

This paper proposes a video-to-4D generation framework that encodes animation data directly into a compact Gaussian variation field latent space via a Direct 4DMesh-to-GS Variation Field VAE, and trains a temporally-aware diffusion model to generate dynamic 3D content. The framework achieves high-fidelity 4D synthesis in 4.5 seconds and demonstrates strong generalization to real-world video inputs.

## Background & Motivation

4D generation—creating dynamic 3D content—represents the next frontier following image, video, and 3D generation. Real-world phenomena inherently combine spatial and temporal dynamics, yet training robust 4D diffusion models faces two major technical challenges:

**High cost of large-scale 4D dataset construction**: Direct approaches require fitting an independent dynamic Gaussian splatting (4DGS) representation to each 3D animation sequence, typically taking tens of minutes per instance (6 minutes for 4DGaussians, 30+ minutes for K-planes), making them computationally expensive and difficult to scale.

**Difficulty of directly modeling high-dimensional representations**: Simultaneously representing 3D shape, appearance, and motion typically requires over 100K tokens, making direct diffusion modeling extremely challenging.

**Limitations of prior work**:
- Optimization-based methods (Consistent4D, STAG4D, etc.) rely on SDS distillation, requiring over an hour and suffering from spatiotemporal inconsistencies.
- Feed-forward methods (L4GM) reconstruct 4DGS from 2D-generated multi-view images, but multi-view inconsistencies in 2D generation degrade quality.
- Native 4D diffusion models are absent; existing approaches indirectly leverage 2D/3D priors.

**Mechanism**: The paper decomposes 4D generation into canonical 3DGS generation (leveraging existing 3D models) and Gaussian Variation Field modeling. By directly encoding 3D animation data, per-instance fitting is bypassed, and high-dimensional motion information is compressed into a compact latent space.

## Method

### Overall Architecture

Given an input video $\mathcal{I} = \{I_t\}_{t=1}^T$, the goal is to generate a 3DGS sequence $\mathcal{G} = \{G_t\}_{t=1}^T$. This is decomposed into:
- **Canonical GS** $G_1$: a static 3DGS generated from the first frame using a pretrained 3D model.
- **Gaussian Variation Field** $\mathcal{V} = \{\Delta G_t\}_{t=1}^T$: temporal variations of each Gaussian attribute relative to $G_1$.

The framework comprises two main components: (1) the Direct 4DMesh-to-GS Variation Field VAE, and (2) the Gaussian Variation Field diffusion model.

### Key Designs

1. **Direct 4DMesh-to-GS Variation Field VAE**:

   **Encoding**:
   - Converts mesh animation sequences into point clouds $\mathcal{P} = \{P_t \in \mathbb{R}^{N \times 3}\}_{t=1}^T$ ($N = 8192$).
   - Computes displacement fields $\Delta P_t = P_t - P_1$.
   - Obtains canonical GS via a pretrained Mesh-to-GS encoder: $G_1 = \mathcal{D}_{GS}(\mathcal{E}_{GS}(M_1))$.

   **Mesh-guided Interpolation** (key innovation): For each canonical Gaussian position $\bm{p}_1^i$, K nearest neighbors are identified and the displacement field is interpolated using adaptive radius-based weighting:

   $\bm{w}_{i,k} = \exp(-\frac{\beta \bm{d}_{i,k}}{r_i^2}), \quad r_i = \sqrt{\frac{1}{K}\sum_{k=1}^K \bm{d}_{i,k}}$

   $\Delta \bm{p}_{t,i}^{interp} = \sum_{k=1}^K \frac{\bm{w}_{i,k}}{\sum_k \bm{w}_{i,k}} \Delta P_{t,n(i,k)}$

   FPS sampling is then applied to the interpolated displacements to obtain motion-aware queries $\Delta \bm{p}_t^{fps} \in \mathbb{R}^{L \times 3}$, which are encoded into a latent representation $\bm{z} \in \mathbb{R}^{T \times L \times C}$ ($L = 512$, $C = 16$) via cross-attention, compressing the sequence length from $N = 8192$ to $L = 512$.

   **Decoding**: The latent representation is processed via self-attention, then decoded into variation fields $\Delta G_t = \{\Delta \bm{p}_t, \Delta \bm{s}_t, \Delta \bm{q}_t, \Delta \bm{c}_t, \Delta \alpha_t\}$ using all canonical GS parameters as queries through cross-attention.

2. **Gaussian Variation Field Diffusion Model**:

   Built on a Diffusion Transformer (DiT) architecture, with the following core innovations:
   - **Temporal self-attention layers**: Added alongside standard spatial self-attention to capture inter-frame motion coherence.
   - **Dual conditioning injection**: Visual features $\mathcal{C}^v$ (extracted via DINOv2) and canonical GS geometric features $\mathcal{C}^{GS}$ are injected via cross-attention.
   - **Positional prior embedding**: Positional encodings based on canonical GS positions $\bm{p}_1^{fps}$ to enhance the model's awareness of spatial position-to-variation-field correspondences.

   Velocity prediction parameterization is adopted, with training objective:

   $\mathcal{L}_{simple} = \mathbb{E}_{s, \bm{z}^0, \bm{\epsilon}} \left[ \|\hat{\bm{v}}_\theta(\alpha_s \bm{z}^0 + \sigma_s \bm{\epsilon}, s, \mathcal{C}) - \bm{v}^s\|_2^2 \right]$

3. **Mesh-guided Loss**:

   Aligns predicted Gaussian displacements with pseudo-GT displacements from mesh interpolation, serving as a critical supervision signal for motion reconstruction:

   $\mathcal{L}_{mg} = \sum_{t=1}^T \|\Delta \mathbf{p}_t - \Delta \bm{p}_t^{interp}\|_2^2$

### Loss & Training

**VAE training**: Two-stage pipeline—first fine-tuning the canonical GS decoder for 150K iterations, then jointly training all modules for 200K iterations. Total loss: $\mathcal{L}_{total} = \mathcal{L}_{img} + \lambda_{mg}\mathcal{L}_{mg} + \lambda_{kl}\mathcal{L}_{kl}$, where $\mathcal{L}_{img}$ combines L1 + LPIPS + SSIM.

**Diffusion model training**: Trained on 24-frame sequences for 1,300K iterations using a cosine noise schedule with 1,000 timesteps. Inference supports 32-frame generation.

## Key Experimental Results

### Main Results (Video-to-4D Generation)

| Method | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP↑ | FVD↓ | Time↓ |
|---|---|---|---|---|---|---|
| Consistent4D | 16.20 | 0.146 | 0.880 | 0.910 | 935.19 | ~1.5hr |
| SC4D | 15.93 | 0.164 | 0.872 | 0.870 | 833.15 | ~20min |
| STAG4D | 16.85 | 0.144 | 0.887 | 0.893 | 1008.40 | ~1hr |
| DreamGaussian4D | 15.24 | 0.162 | 0.868 | 0.904 | 799.56 | ~15min |
| L4GM | 17.03 | 0.128 | 0.891 | 0.930 | 529.10 | 3.5s |
| **Ours** | **18.47** | **0.114** | **0.901** | **0.935** | **476.83** | **4.5s** |

The proposed method achieves the best performance on all quality metrics: PSNR improves by 1.44 dB over L4GM, FVD decreases by 9.9% (476.83 vs. 529.10), and the total generation time is only 4.5 seconds (3.0s for canonical GS + 1.5s for variation field diffusion).

### Ablation Study

**VAE component ablation**:

| Config | Encoder Query | Mesh-guided Loss | Variation Attrs | PSNR↑ | LPIPS↓ | SSIM↑ |
|---|:---:|:---:|:---:|---|---|---|
| A. Baseline | $\bm{p}_t^{fps}$ | ✗ | $\Delta\bm{p},\Delta\bm{s},\Delta\bm{q}$ | 23.25 | 0.0678 | 0.936 |
| B. +mesh loss | $\bm{p}_t^{fps}$ | ✓ | $\Delta\bm{p},\Delta\bm{s},\Delta\bm{q}$ | 26.17 | 0.0544 | 0.950 |
| C. +motion query | $\Delta\bm{p}_t^{fps}$ | ✓ | $\Delta\bm{p},\Delta\bm{s},\Delta\bm{q}$ | 28.58 | 0.0478 | 0.958 |
| D. Full (Ours) | $\Delta\bm{p}_t^{fps}$ | ✓ | All 5 attrs | **29.28** | **0.0439** | **0.964** |

**Diffusion model ablation**:

| Method | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP↑ | FVD↓ |
|---|---|---|---|---|---|
| w/o positional embedding | 17.86 | 0.121 | 0.897 | 0.931 | 547.20 |
| Full model | **18.47** | **0.114** | **0.901** | **0.935** | **476.83** |

### Key Findings

1. **Mesh-guided loss is critical for motion learning**: Config A→B yields a 2.92 dB PSNR gain, addressing the core challenge of lacking GT Gaussian motion supervision.
2. **Motion-aware queries substantially outperform static positional queries**: Config B→C yields an additional 2.41 dB PSNR gain.
3. **Color and opacity variations also contribute meaningfully**: Adding $\Delta\bm{c}_t$ and $\Delta\alpha_t$ provides a further 0.70 dB improvement.
4. **Positional prior embedding is essential for the diffusion model**: It strengthens the model's awareness of spatial position-to-variation-field correspondences.

## Highlights & Insights

- **Bypassing per-instance fitting**: By encoding Gaussian variation fields directly from mesh animations in a single forward pass, the high cost of 4DGS reconstruction is entirely avoided.
- **Efficient 4D decomposition**: Decomposing the 4D problem into canonical 3DGS generation and variation field modeling reduces the dimensionality burden on the diffusion model.
- **Elegant motion-aware encoding design**: Mesh-guided interpolation bridges mesh motion and Gaussian motion, and motion-aware queries substantially improve encoding quality.
- **Synthesis-to-real generalization**: The model is trained exclusively on synthetic data yet generalizes to in-the-wild videos, demonstrating the transferability of the learned motion priors.

## Limitations & Future Work

- The training set contains only 34K animated objects, with scale limited by the scarcity of high-quality animation assets.
- Inference requires canonical GS generation from a pretrained 3D model, introducing a dependency on third-party model quality.
- Current support is limited to 32-frame animation; long-sequence generation may require autoregressive or sliding-window strategies.
- The capability to model complex topological changes (e.g., object splitting or merging) remains to be validated.

## Related Work & Insights

- The perceiver-based encoding in 3DShape2VecSet inspired the cross-attention encoding architecture adopted in this work.
- Trellis's structured latent representation provides a strong foundation for high-quality canonical GS generation.
- The decomposition strategy for 4D problems (canonical + variation) is generalizable to other dynamic 3D tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A pioneering contribution toward native 4D diffusion models; the VAE-based direct encoding strategy circumvents per-instance fitting.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative comparisons and clear ablations, though the test set covers only 100 objects.
- **Writing Quality**: ⭐⭐⭐⭐ — Framework is clearly described with complete technical details and well-designed ablation experiments.
- **Value**: ⭐⭐⭐⭐⭐ — A practical solution for 4D generation with real application potential given the 4.5-second generation time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Not All Frame Features Are Equal: Video-to-4D Generation via Decoupling Dynamic-Static Features](not_all_frame_features_are_equal_video-to-4d_generation_via_decoupling_dynamic-s.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Arbitrary-Scale 3D Gaussian Super-Resolution
description: >-
  [AAAI 2026][3D Vision][3DGS] This paper proposes Arbi-3DGSR, an integrated framework that, for the first time, enables a single 3DGS model to support arbitrary-scale (including non-integer) high-resolution rendering through three core components: scale-aware rendering, generative prior-guided optimization, and progressive super-resolving. At ×5.7 scale, PSNR improves by 6.59 dB over vanilla 3DGS while maintaining real-time rendering at 85 FPS.
tags:
  - AAAI 2026
  - 3D Vision
  - 3DGS
  - arbitrary-scale super-resolution
  - scale-aware rendering
  - generative prior
  - progressive training
date: 2026-05-08
content_hash: e54ffb20b1383c6b
---

# Arbitrary-Scale 3D Gaussian Super-Resolution

**Conference**: AAAI 2026
**arXiv**: [2508.16467](https://arxiv.org/abs/2508.16467)
**Code**: [https://github.com/huimin-zeng/Arbi-3DGSR](https://github.com/huimin-zeng/Arbi-3DGSR)
**Area**: 3D Vision / 3D Gaussian Splatting / Super-Resolution
**Keywords**: 3DGS, arbitrary-scale super-resolution, scale-aware rendering, generative prior, progressive training

## TL;DR
This paper proposes Arbi-3DGSR, an integrated framework that, for the first time, enables a single 3DGS model to support arbitrary-scale (including non-integer) high-resolution rendering through three core components: scale-aware rendering, generative prior-guided optimization, and progressive super-resolving. At ×5.7 scale, PSNR improves by 6.59 dB over vanilla 3DGS while maintaining real-time rendering at 85 FPS.

## Background & Motivation

**State of the Field**: High-resolution novel view synthesis (HRNVS) requires reconstructing 3D models from low-resolution sparse views and rendering HR outputs. Recent 3DGS methods achieve real-time rendering via explicit point cloud representations, but existing 3DGS super-resolution methods (SuperGS, SRGS, GaussianSR, etc.) only handle fixed integer scale factors (e.g., ×2, ×4), requiring separate models trained for each scale.

**Limitations of Prior Work**: (1) Fixed-scale constraints limit flexibility and ignore the intrinsic continuity of the 3D world; (2) directly rendering at arbitrary scales with vanilla 3DGS introduces aliasing artifacts due to the lack of scale awareness; (3) cascading a 2D super-resolver after 3DGS can support arbitrary scales but increases framework complexity and severely degrades rendering efficiency (StableSR achieves only 0.13 FPS).

**Root Cause**: Arbitrary-scale rendering requires simultaneously addressing three interrelated challenges—anti-aliased rendering across scales, detail supervision without HR ground truth, and cross-scale structural consistency—while existing methods handle at most one of these.

**Paper Goals**: Enable a single 3DGS model to produce high-quality HR renderings at arbitrary scales (including non-integer scales such as ×3.5 and ×5.7 within the range ×1 to ×8), while preserving structural consistency and real-time speed.

**Starting Point**: The authors observe that both the Gaussian bandwidth and pixel integration window in 3DGS should adapt to the target resolution. By injecting the scale factor into two key stages of the rendering pipeline—3D filtering and 2D Mip filtering—anti-aliased multi-scale rendering becomes achievable. Additionally, the generative prior of a diffusion model provides detail supervision in latent space, eliminating the need for explicit HR supervision.

**Core Idea**: Treat the scale factor as a first-class citizen injected into both 3D filtering and 2D Mip filtering of the 3DGS rendering pipeline, complemented by latent distillation from generative priors and progressive training, enabling arbitrary-scale super-resolution within a single model.

## Method

### Overall Architecture
The input is a set of low-resolution views; the output is a high-resolution rendering at an arbitrary target scale $s$. The framework consists of three core components: scale-aware rendering (used in both training and inference) enables 3DGS to adaptively adjust rendering behavior according to the target resolution; generative prior-guided optimization (training only) leverages the denoising process of StableSR to provide detail supervision for HR rendering; progressive super-resolving (training only) divides training into multiple stages with gradually increasing target scales to maintain cross-scale consistency.

### Key Designs

1. **Scale-Aware Rendering**:

    - **Function**: Enables the same set of Gaussian primitives to adaptively adjust rendering behavior according to the target resolution, avoiding aliasing and blurring at different scales.
    - **Mechanism**: Operates via two-level filtering in 3D and 2D. The 3D scale-aware smoothing filter introduces the scale factor $s$ into the maximum sampling rate computation $\hat{r}_i(s) = \max(\mathbb{I}_k(G_i^{3D}) \cdot f_k \cdot s_k / d_k)$, thereby adaptively constraining Gaussian bandwidth. The 2D scale-aware Mip filter sets the integration window size to $\varepsilon_k = \varepsilon / s_k$, aligning the pixel shading integration window with the actual pixel footprint. A 1D approximation error analysis demonstrates that fixed windows accumulate errors across scales, whereas adaptive windows maintain consistently low error.
    - **Design Motivation**: The original filters in Mip-Splatting use fixed parameters and cannot adapt to varying target resolutions. High-scale rendering requires narrower signal bandwidth and smaller integration windows, while low-scale rendering requires the opposite.

2. **Generative Prior-Guided Optimization**:

    - **Function**: Provides texture detail supervision using a pretrained diffusion model (StableSR) in the absence of HR ground truth.
    - **Mechanism**: Comprises two sub-modules. (a) *Latent Distillation Sampling (LDS Loss)*: Conditional diffusion processes are applied separately to LR views and the current SR rendering; noise prediction differences at asynchronous timesteps are computed in latent space as $\nabla_\theta \mathcal{L}_{LDS} = \mathbb{E}_{\hat{n}}[w(\hat{n}) \cdot (\epsilon_\phi(z_{SR}^{\hat{n}}) - \epsilon_\phi(z_{LR}^n)) \cdot \partial I_{SR}^t / \partial \theta]$, driving SR latents to approach LR latents that carry rich structural information. Unlike SDS Loss, LDS compares noise predictions between asynchronous latents rather than at the same timestep, providing structural supervision while tolerating pixel-level misalignment introduced by generative priors. (b) *Orthogonal Reference Refinement*: A subset of nearly orthogonal views is selected from the scene; full denoising is performed on these views to obtain HR reference images, and a pixel-level texture loss $\mathcal{L}_{tex} = \mathbb{I}_{ortho} \cdot \|I_{SR}^t - I_{Ref}^t\|^2$ is applied.
    - **Design Motivation**: Applying pixel-level supervision directly from generated HR references causes blurring and artifacts due to generation inconsistencies across adjacent views. LDS operates in latent space to avoid pixel-level misalignment; the orthogonal view strategy ensures non-overlapping regions among reference images, preventing conflicting supervision signals.

3. **Progressive Super-Resolving**:

    - **Function**: Divides training into multiple stages with progressively increasing target scales, ensuring cross-scale structural consistency.
    - **Mechanism**: Training proceeds in three stages: ×2 → ×4 → ×8. Each stage initializes from the Gaussian primitives of the previous stage and randomly samples from the accumulated set of scale factors during training. A structural loss $\mathcal{L}_{str}$ is applied between stages, aligning the current-stage HR rendering (after downsampling) with the previous-stage rendering using a weighted combination of MSE and D-SSIM.
    - **Design Motivation**: Directly mixing arbitrary scales during training (w/o PSR) leads to unstable optimization due to conflicting requirements between small and large scales. The progressive strategy ensures the model first masters low-scale details before gradually extending to higher scales.

### Loss & Training
The total loss is a weighted sum of three terms: $\mathcal{L} = \lambda_1 \mathcal{L}_{LDS} + \lambda_2 \mathcal{L}_{tex} + \lambda_3 \mathcal{L}_{str}$. Training takes approximately 57 minutes per scene on a single A6000 GPU with ~7 GB memory. No additional computational overhead is incurred during rendering. LR inputs are obtained by applying ×8 bicubic downsampling to original images; no original HR images are used during training.

## Key Experimental Results

### Main Results

Comparison against 7 methods across 4 benchmark datasets (Blender, Mip-NeRF360, Tanks&Temples, Deep Blending) at both integer and non-integer scales:

| Method | Blender ×4 PSNR↑ | Blender ×4 FID↓ | MipNeRF360 ×8 PSNR↑ | MipNeRF360 ×5.7 PSNR↑ | T&T ×4 PSNR↑ |
|--------|-------------------|-----------------|---------------------|----------------------|--------------|
| 3DGS | 17.84 | 208.17 | 19.92 | 20.33 | 16.24 |
| Mip-Splatting | 22.25 | 109.44 | 24.51 | 25.02 | 20.97 |
| Analytic-Splatting | 23.57 | 141.30 | 23.04 | 23.41 | 19.42 |
| GaussianSR | 23.03 | 118.02 | 24.10 | 24.20 | 20.63 |
| **Ours** | **24.32** | **86.27** | **24.85** | **24.99** | **21.14** |

### Ablation Study (Mip-NeRF360)

| Configuration | ×2 PSNR | ×4 PSNR | ×8 PSNR | ×2 FID |
|---------------|---------|---------|---------|--------|
| Full model | 26.23 | 25.18 | 24.85 | 36.52 |
| w/o 3D-SASF | 26.13 | 24.85 | 24.39 | 41.58 |
| w/o 2D-SAMF | 25.53 | 24.83 | 24.61 | 36.86 |
| w/o PSR | 26.03 | 24.51 | 23.91 | 37.92 |
| w/o GPO | 25.23 | 24.51 | 24.27 | 99.69 |
| Pseudo HR | 23.96 | 23.36 | 23.19 | 111.15 |
| SDS loss | 23.52 | 22.91 | 22.71 | 72.64 |

### Key Findings
- GPO contributes most significantly: removing it decreases PSNR by 1 dB at ×2 and causes FID to surge from 36.52 to 99.69, demonstrating that generative priors are critical for perceptual quality.
- Progressive super-resolving has a substantial impact at high scales: w/o PSR degrades ×8 PSNR by 0.94 dB.
- LDS Loss substantially outperforms conventional alternatives: Pseudo HR and SDS Loss degrade ×2 PSNR by 2.27 dB and 2.71 dB, respectively.
- Efficiency advantage is pronounced: 85 FPS vs. StableSR's 0.13 FPS (908× faster), with a storage footprint of only 0.79 GB.

## Highlights & Insights
- **Unified model for arbitrary scales**: This is the first work to introduce arbitrary-scale super-resolution into the 3DGS domain, with a single model covering both integer and non-integer scales; the approach is transferable to NeRF and other 3D representations.
- **Elegant LDS Loss design**: By comparing latent noise predictions at asynchronous timesteps rather than pixel-level differences, the method leverages diffusion model generative priors while avoiding view inconsistency—outperforming SDS Loss by 2.71 dB in PSNR.
- **Orthogonal view strategy**: Using geometric constraints (orthogonal views cover non-overlapping regions) to address generation consistency is a generalizable mechanism for ensuring multi-view consistency.

## Limitations & Future Work
- Only static scenes are addressed; extension to dynamic 3DGS (e.g., 4D-GS) remains unexplored.
- The generative prior depends on the pretraining quality of StableSR, and may introduce unrealistic textures at very high scales (>×8).
- Training takes 57 min/scene, with the primary overhead from diffusion model inference; lighter prior sources (e.g., ESRGAN-based models) could be explored.
- Cross-scene generalization is not investigated—each scene still requires independent training.

## Related Work & Insights
- **vs. GaussianSR**: Although GaussianSR achieves faster rendering (126 FPS), it relies on mixed random-scale training without structural consistency constraints and underperforms on all metrics. GaussianSR has a smaller storage footprint (0.56 GB vs. 0.79 GB) but requires 4.5× longer training (256 min vs. 57 min).
- **vs. Mip-Splatting**: Mip-Splatting provides a strong anti-aliasing baseline but does not address super-resolution. This work builds upon its filtering design by injecting scale factors to achieve scale-aware rendering, outperforming Mip-Splatting by 2.13 dB PSNR at Blender ×3.5.
- **vs. Analytic-Splatting**: Despite theoretically more accurate pixel integration, Analytic-Splatting produces high-frequency artifacts in practical HR rendering; this work surpasses it by 1.81 dB PSNR on Mip-NeRF360 ×8.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to define the Arbi-3DGSR problem; scale-aware rendering and LDS Loss are novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 benchmarks, 7 baselines, 5 scale factors, comprehensive ablation and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clear, technical descriptions are complete, and mathematical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐ Real-time rendering with flexible scale factors has significant practical implications for 3DGS deployment.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution](ie-srgs_an_internal-external_knowledge_fusion_framework_for_high-fidelity_3d_gau.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[ICCV 2025\] A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting](../../ICCV2025/3d_vision/a3gs_arbitrary_artistic_style_into_arbitrary_3d_gaussian_spl.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[AAAI 2026\] Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction](splat-sap_feed-forward_gaussian_splatting_for_human-centered_scene_with_scale-aw.md)

<!-- RELATED:END -->

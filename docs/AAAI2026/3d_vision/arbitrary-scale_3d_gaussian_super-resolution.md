---
title: >-
  [Paper Note] Arbitrary-Scale 3D Gaussian Super-Resolution
description: >-
  [AAAI 2026][3D Vision][3DGS] This paper proposes the Arbi-3DGSR integrated framework. Comprising three core components—scale-aware rendering, generative-prior-guided optimization, and progressive super-resolution—it achieves, for the first time, high-resolution rendering at arbitrary (including non-integer) scales using a single 3DGS model. It improves the PSNR by 6.59 dB compared to vanilla 3DGS at a $\times 5.7$ scale while maintaining a real-time rendering speed of 85 FPS.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3DGS"
  - "arbitrary-scale super-resolution"
  - "scale-aware rendering"
  - "generative prior"
  - "progressive training"
date: 2026-05-08
content_hash: bffe7d19ba29af6a
---

# Arbitrary-Scale 3D Gaussian Super-Resolution

**Conference**: AAAI 2026  
**arXiv**: [2508.16467](https://arxiv.org/abs/2508.16467)  
**Code**: [https://github.com/huimin-zeng/Arbi-3DGSR](https://github.com/huimin-zeng/Arbi-3DGSR)  
**Area**: 3D Vision / 3D Gaussian Splatting / Super-Resolution  
**Keywords**: 3DGS, arbitrary-scale super-resolution, scale-aware rendering, generative prior, progressive training

## TL;DR
This paper proposes the Arbi-3DGSR integrated framework. Comprising three core components—scale-aware rendering, generative-prior-guided optimization, and progressive super-resolution—it achieves, for the first time, high-resolution rendering at arbitrary (including non-integer) scales using a single 3DGS model. It improves the PSNR by 6.59 dB compared to vanilla 3DGS at a $\times 5.7$ scale while maintaining a real-time rendering speed of 85 FPS.

## Background & Motivation

**Background**: High-Resolution Novel View Synthesis (HRNVS) requires reconstructing 3D models from low-resolution sparse views and rendering high-resolution (HR) views. Recently, 3DGS methods have achieved real-time rendering thanks to their explicit point cloud representation. However, existing 3DGS super-resolution methods (such as SuperGS, SRGS, and GaussianSR) can only handle fixed, integer upscaling factors (e.g., $\times 2$, $\times 4$), requiring independent models to be trained for different scales.

**Limitations of Prior Work**: (1) Fixed upscaling factors limit flexibility and ignore the inherent continuity of the 3D world; (2) Directly rendering at arbitrary scales using vanilla 3DGS produces aliasing artifacts due to the lack of scale-awareness; (3) Cascading a 2D super-resolver after 3DGS can support arbitrary scales but increases framework complexity and severely degrades rendering efficiency (e.g., StableSR achieves only 0.13 FPS).

**Key Challenge**: Arbitrary-scale rendering requires simultaneously addressing three interrelated challenges: anti-aliased rendering at different scales, detail constraint in the absence of HR ground truth, and cross-scale structural consistency—whereas existing methods can address at most one of these.

**Goal**: To achieve high-quality HR rendering at arbitrary scales (including non-integer scales like $\times 3.5$ and $\times 5.7$ between $\times 1$ and $\times 8$) using a single 3DGS model, while maintaining structural consistency and real-time speed.

**Key Insight**: The authors observe that both the Gaussian bandwidth and the pixel integration window of 3DGS should adaptively adjust with the target resolution. By injecting the scale factor into two key stages of the rendering pipeline (3D filtering and 2D Mip-filtering), anti-aliased multi-scale rendering can be achieved. Meanwhile, the generative prior of diffusion models can be utilized to provide detail supervision in the latent space, avoiding explicit HR supervision.

**Core Idea**: Inject the scale factor as a first-class citizen into both the 3D filtering and 2D Mip-filtering of the 3DGS rendering pipeline, which, combined with latent distillation of generative priors and progressive training, enables arbitrary-scale super-resolution in a single model.

## Method

### Overall Architecture
The input is a set of low-resolution views, and the output is the high-resolution rendering result at an arbitrary target scale $s$. The framework comprises three core components: scale-aware rendering (used during both training and inference), which enables 3DGS to adaptively adjust its rendering behavior based on the target resolution; generative-prior-guided optimization (used during training), which leverages the denoising process of StableSR to provide detail supervision for HR rendering; and progressive super-resolution (used during training), which divides the training process into multiple stages to gradually increase the target scale and maintain cross-scale consistency.

### Key Designs

1. **Scale-Aware Rendering**:

    - **Function**: Enables the same set of Gaussian primitives to adaptively adjust their rendering behavior according to the target resolution, avoiding aliasing and blurriness at different scales.
    - **Mechanism**: Divided into two-stage filtering: 3D and 2D. 3D scale-aware low-pass filtering introduces the scale factor $s$ into the maximum sampling rate calculation $\hat{r}_i(s) = \max(\mathbb{I}_k(G_i^{3D}) \cdot f_k \cdot s_k / d_k)$ to adaptively constrain the Gaussian bandwidth. 2D scale-aware Mip-filtering sets the integration window size to $\varepsilon_k = \varepsilon / s_k$, matching the integration window of pixel shading with the actual pixel area. Through a 1D approximation error analysis, the authors prove that a fixed window accumulates errors at different scales, whereas the adaptive window consistently maintains low error.
    - **Design Motivation**: The original filters of Mip-Splatting use fixed parameters, which cannot adapt to different target resolutions. High-scale rendering requires narrower signal bandwidth and a smaller integration window, whereas low-scale rendering requires the opposite.

2. **Generative Prior-Guided Optimization**:

    - **Function**: Leverages a pre-trained diffusion model (StableSR) to provide texture detail supervision in the absence of HR ground truth.
    - **Mechanism**: Comprises two sub-modules. (a) Latent Distillation Sampling (LDS Loss): conditional diffusion processes are applied to the LR views and current SR renderings separately. The noise prediction difference at asynchronous timesteps is calculated in the latent space as $\nabla_\theta \mathcal{L}_{LDS} = \mathbb{E}_{\hat{n}}[w(\hat{n}) \cdot (\epsilon_\phi(z_{SR}^{\hat{n}}) - \epsilon_\phi(z_{LR}^n)) \cdot \partial I_{SR}^t / \partial \theta]$, forcing the SR latent to approximate the LR latent featuring rich structural information. Unlike SDS Loss, LDS compares noise differences of asynchronous latents rather than at the same timestep, which provides structural supervision while tolerating pixel-level misalignments introduced by the generative prior. (b) Orthogonal Reference Refinement: a subset of views in the scene that are nearly mutually orthogonal is selected, and complete denoising is performed on these views to obtain HR reference maps, imposing a pixel-level texture loss $\mathcal{L}_{tex} = \mathbb{I}_{ortho} \cdot \|I_{SR}^t - I_{Ref}^t\|^2$.
    - **Design Motivation**: Direct pixel-level supervision using generated HR references causes blurring and artifacts due to the generative inconsistencies among neighboring views. Operating LDS in the latent space avoids the pixel-level misalignment issue; the orthogonal views strategy ensures no overlapping regions exist between reference images, thereby preventing conflicting information.

3. **Progressive Super-Resolving**:

    - **Function**: Divides the training process into multiple stages to gradually increase the target scale, ensuring cross-scale structural consistency.
    - **Mechanism**: Training is divided into three stages: $\times 2 \rightarrow \times 4 \rightarrow \times 8$. Each stage is initialized from the Gaussian primitives of the previous stage, and training is conducted by randomly sampling from the existing scale set. A structural loss $\mathcal{L}_{str}$ is applied between stages to align the downsampled HR rendering of the current stage with the rendering result of the previous stage, using a weighted combination of MSE and D-SSIM.
    - **Design Motivation**: Directly training with randomly mixed scales (w/o PSR) leads to optimization instability, as the requirements for small and large scales conflict with each other. The progressive strategy ensures that the model first learns low-scale details before gradually expanding.

### Loss & Training
The total loss is a weighted sum of three terms: $\mathcal{L} = \lambda_1 \mathcal{L}_{LDS} + \lambda_2 \mathcal{L}_{tex} + \lambda_3 \mathcal{L}_{str}$. Training takes approximately 57 minutes per scene on a single A6000 GPU, with a memory footprint of about 7 GB. There is no additional computational overhead during the rendering phase. LR inputs are obtained by applying an 8x Bicubic downsampling to the original images, and the original HR images are not used during the training process.

## Key Experimental Results

### Main Results

Compared with 7 methods on 4 benchmark datasets (Blender, Mip-NeRF360, Tanks&Temples, Deep Blending), evaluating both integer and non-integer scales:

| Method | Blender $\times 4$ PSNR↑ | Blender $\times 4$ FID↓ | MipNeRF360 $\times 8$ PSNR↑ | MipNeRF360 $\times 5.7$ PSNR↑ | T&T $\times 4$ PSNR↑ |
|------|-------------------|-----------------|---------------------|----------------------|--------------|
| 3DGS | 17.84 | 208.17 | 19.92 | 20.33 | 16.24 |
| Mip-Splatting | 22.25 | 109.44 | 24.51 | 25.02 | 20.97 |
| Analytic-Splatting | 23.57 | 141.30 | 23.04 | 23.41 | 19.42 |
| GaussianSR | 23.03 | 118.02 | 24.10 | 24.20 | 20.63 |
| **Ours** | **24.32** | **86.27** | **24.85** | **24.99** | **21.14** |

### Ablation Study (Mip-NeRF360)

| Configuration | $\times 2$ PSNR | $\times 4$ PSNR | $\times 8$ PSNR | $\times 2$ FID |
|------|---------|---------|---------|--------|
| Full model | 26.23 | 25.18 | 24.85 | 36.52 |
| w/o 3D-SASF | 26.13 | 24.85 | 24.39 | 41.58 |
| w/o 2D-SAMF | 25.53 | 24.83 | 24.61 | 36.86 |
| w/o PSR | 26.03 | 24.51 | 23.91 | 37.92 |
| w/o GPO | 25.23 | 24.51 | 24.27 | 99.69 |
| Pseudo HR | 23.96 | 23.36 | 23.19 | 111.15 |
| SDS loss | 23.52 | 22.91 | 22.71 | 72.64 |

### Key Findings
- GPO contributes the most: removing it drops PSNR by 1 dB at $\times 2$ and balloons FID from 36.52 to 99.69, indicating that the generative prior is crucial for perceptual quality.
- Progressive super-resolution has a significant impact on high-scale upscaling: w/o PSR degrades the PSNR by 0.94 dB at $\times 8$.
- LDS Loss is far superior to traditional alternatives: Pseudo HR and SDS Loss result in a PSNR drop of 2.27 dB and 2.71 dB (at $\times 2$) respectively.
- Obvious efficiency advantage: 85 FPS vs 0.13 FPS for StableSR (908 times faster), with a storage footprint of only 0.79 GB.

## Highlights & Insights
- **Unified Model for Arbitrary Scales**: This work brings arbitrary-scale super-resolution into the 3DGS field for the first time, covering both integer and non-integer scales with a single model; this idea can be transferred to NeRF or other 3D representations.
- **Ingenious LDS Loss Design**: By comparing latent noise at asynchronous timesteps instead of pixel differences, it utilizes the generative prior of diffusion models while avoiding view inconsistency, outperforming SDS Loss by 2.71 dB in PSNR.
- **Orthogonal Views Strategy**: Addressing generative consistency issues via geometric constraints (orthogonal views having no overlapping regions) presents a generalizable scheme to ensure multi-view consistency.

## Limitations & Future Work
- Only handles static scenes and has not been extended to dynamic 3DGS (e.g., 4D-GS).
- The generative prior depends on the pre-training quality of StableSR, which might introduce unrealistic textures at extremely high scales ($> \times 8$).
- Training time is 57 min per scene, with the main overhead coming from diffusion model inference; lighter sources of priors (such as the ESRGAN series) could be explored.
- Cross-scene generalization ability has not been explored—each scene still requires independent training.

## Related Work & Insights
- **vs GaussianSR**: Although GaussianSR renders faster (126 FPS), it employs randomly mixed scale training and lacks structural consistency constraints, lagging behind the proposed method across all metrics. GaussianSR has a smaller storage footprint (0.56 GB vs 0.79 GB) but requires 4.5 times longer training time (256 min vs 57 min).
- **vs Mip-Splatting**: Mip-Splatting provides an anti-aliasing foundation but does not involve super-resolution. This work introduces the scale factor on top of its filters to realize scale-awareness, yielding a 2.13 dB higher PSNR on Blender at $\times 3.5$ scale.
- **vs Analytic-Splatting**: Theoretically more precise pixel integration, but introduces high-frequency artifacts in actual HR rendering; the proposed method achieves 1.81 dB higher PSNR at $\times 8$ scale on Mip-NeRF360.

## Rating
- Novelty: ⭐⭐⭐⭐ First to define the Arbi-3DGSR problem, with novel designs for scale-aware rendering and LDS Loss.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 benchmarks, 7 baselines, 5 scales, as well as comprehensive ablation and efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete technical description, and rigorous mathematical derivations.
- Value: ⭐⭐⭐⭐ Real-time rendering combined with flexible scaling factors is highly significant for the practical deployment of 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution](ie-srgs_an_internal-external_knowledge_fusion_framework_for_high-fidelity_3d_gau.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting](../../CVPR2026/3d_vision/splatsure_selective_super-resolution_for_multi-view_consistent_3d_gaussian_splat.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](../../CVPR2025/3d_vision/s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[ECCV 2024\] SuperGaussian: Repurposing Video Models for 3D Super Resolution](../../ECCV2024/3d_vision/supergaussian_repurposing_video_models_for_3d_super_resolution.md)

</div>

<!-- RELATED:END -->

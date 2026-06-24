---
title: >-
  [Paper Note] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation
description: >-
  [CVPR 2025][3D Vision][Multiview Relighting] This paper proposes using a multiview relighting diffusion model to first unify images captured under different illumination conditions into a reference lighting condition, and then reconstruct the 3D representation using a robust NeRF model with "shading embedding". It achieves high-fidelity appearance reconstruction under extreme illumination variations, significantly outperforming existing methods…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multiview Relighting"
  - "3D Reconstruction"
  - "Diffusion Models"
  - "Extreme Illumination Variation"
  - "NeRF"
date: 2026-05-08
content_hash: c3ba827d9e07c5e8
---

# Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation

**Conference**: CVPR 2025  
**arXiv**: [2412.15211](https://arxiv.org/abs/2412.15211)  
**Code**: [https://relight-to-reconstruct.github.io/](https://relight-to-reconstruct.github.io/) (Project Page)  
**Area**: 3D Vision / 3D Reconstruction  
**Keywords**: Multiview Relighting, 3D Reconstruction, Diffusion Models, Extreme Illumination Variation, NeRF

## TL;DR
This paper proposes using a multiview relighting diffusion model to first unify images captured under different illumination conditions into a reference lighting condition, and then reconstruct the 3D representation using a robust NeRF model with "shading embedding". It achieves high-fidelity appearance reconstruction under extreme illumination variations, significantly outperforming existing methods, especially in recovering specular/highlight effects.

## Background & Motivation

1. **Background**: 3D reconstruction and novel view synthesis (such as NeRF) typically assume that input images are captured under the same illumination condition. Common strategies for handling illumination variations include: (a) per-image appearance embedding (e.g., NeRF-W), which allows the model to learn the appearance changes of each image; (b) inverse rendering, which explicitly recovers materials and per-image illumination.
2. **Limitations of Prior Work**: Appearance embeddings often "explain away" view-dependent appearances (such as specular reflections), leading to diffuse reconstruction results. Inverse rendering models materials and lighting through low-frequency parameterizations (spherical harmonics, spherical Gaussians), failing to recover high-frequency specular reflections. Although recent single-image relighting diffusion models (such as IllumiNeRF and Neural Gaffer) leverage strong priors, relighting each image independently leads to inconsistent material interpretations.
3. **Key Challenge**: Recovering object appearance (particularly view-dependent specular highlights) from images with "extreme illumination variations" is a highly underdetermined problem—appearance changes can originate from either illumination changes or viewpoint changes, causing severe ambiguity between the two.
4. **Goal**: (a) How to reconstruct faithful 3D appearances from images under extremely different lighting conditions; (b) how to correctly recover specular/highlight effects instead of degrading to diffuse results.
5. **Key Insight**: Inspired by photometric stereo, observations of the same material under multiple lighting conditions can reduce material/illumination ambiguity. This insight is introduced into diffusion models to jointly relight all images rather than processing them independently.
6. **Core Idea**: Use a multiview diffusion model to jointly relight all input images to a reference illumination, and then use a shading embedding mechanism to robustly fit residual inconsistencies.

## Method

### Overall Architecture
Two-stage pipeline: (1) The multiview relighting diffusion model takes $N$ input images under different illumination conditions along with their camera poses, simultaneously relighting all images to the illumination condition of the reference image; (2) The NeRF-Casting-based 3D reconstruction model recovers geometry and view-dependent appearance from the relighted images, utilizing per-image shading embeddings to absorb residual inconsistencies from the diffusion model outputs.

### Key Designs

1. **Multiview Relighting Diffusion Model**:
    - **Function**: Unifies $N$ images with different illumination conditions into the illumination condition of a reference image.
    - **Mechanism**: Extending single-image relighting models to multiview scenarios. The architecture is based on a latent diffusion model similar to Stable Diffusion 1.5, denoising $N$ latent codes $z_1,...,z_N$ simultaneously. When denoising the latent of the $i$-th image, the original "clean" image $I_i$ and the camera pose $\pi_i$ are provided as input for geometric information, while 3D self-attention blocks allow interaction between latents of different views, and cross-attention processes the camera poses. The reference image is identified via a reference map (all-ones channel), while other images use an all-zeros channel.
    - **Design Motivation**: Joint processing of multiple views exploits the complementary information of "different illumination on the same material" (similar to photometric stereo), which significantly reduces material/lighting ambiguity compared to independent image processing, yielding more consistent relighting results.

2. **Shading Embedding**:
    - **Function**: Absorbs residual specular position shifts in the output of the diffusion model.
    - **Mechanism**: Learns an embedding vector $\mathbf{v}_i$ for each training image, which is used together with spatial features to predict image-specific normal perturbations $\mathbf{n}_i(\mathbf{x}) = \text{normalize}(\text{MLP}(\mathbf{f}(\mathbf{x}), \mathbf{v}_i))$ via an MLP. The perturbed normal changes the secondary specular reflection beam direction, thereby adjusting the position of specular highlights. A consistency constraint with the density field normal prevents excessively large offsets.
    - **Design Motivation**: Minor errors in the implicit shape estimation of the diffusion model lead to shifts in highlight locations. Traditional appearance embeddings absorb highlights as per-image diffuse colors (leading to a diffuse-only result), whereas shading embeddings only adjust the normal direction, preserving the physically correct mechanism of highlight generation.

3. **Purely Synthetic Training Data and Specular Augmentation**:
    - **Function**: Provides large-scale ground-truth data for training the multiview relighting model.
    - **Mechanism**: Training data is rendered from approximately 300k high-quality 3D object assets using around 700 environment maps (Poly Haven) and augmented with random azimuthal rotations. A key innovation is additionally rendering a replica where all materials are replaced with perfect mirrors, creating a mixed training set of standard and mirror-like materials. The model is trained progressively in four stages (8 → 16 → 32 → 64 frames) on 64 TPU v5s for about two weeks.
    - **Design Motivation**: Specular data augmentation not only improves the reconstruction quality of specular objects, but interestingly, also enhances the results of ordinary diffuse objects. This is likely because mirror-like materials provide richer training signals regarding the relationship between lighting and surface normals.

### Loss & Training
Relighting model: Standard diffusion denoising loss + classifier-free guidance (CFG = 3), with random masking of reference image attention for unconditional training. 3D reconstruction: Standard loss of NeRF-Casting, optimized for ~30 minutes per scene on 16 A100 GPUs, rendering a single 512×512 image in ~0.5 seconds. During inference, $N=64$ images are relighted simultaneously.

## Key Experimental Results

### Main Results

**Synthetic Data (Objaverse)**

| Method | Standard PSNR↑ | Standard LPIPS↓ | Shiny PSNR↑ | Shiny LPIPS↓ |
|------|------|------|------|------|
| NeROIC | 26.13 | 0.088 | 22.14 | 0.113 |
| NeRFCast+AE | 27.53 | 0.067 | 21.80 | 0.108 |
| IllumiNeRF (w/ GT Lighting) | 29.22 | 0.057 | 23.46 | 0.095 |
| **Ours** | **31.34** | **0.053** | **26.54** | **0.090** |

**Real Images (NAVI)**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|------|------|
| NeRFCast+AE | 22.67 | 0.906 | 0.074 |
| NeROIC | 24.01 | 0.918 | 0.079 |
| **Ours** | **25.55** | **0.929** | **0.060** |

### Ablation Study

**Frame Count Ablation (Shiny Assets)**

| Jointly Processed Frames $N$ | PSNR↑ | SSIM↑ |
|------|---------|------|
| 1 frame | 23.85 | 0.889 |
| 8 frames | 25.62 | 0.901 |
| 32 frames | 26.14 | 0.908 |
| 64 frames (ours) | **26.54** | **0.911** |

**Per-image Embedding Ablation**

| Method | PSNR↑ | LPIPS↓ |
|------|---------|------|
| No embeddings | 26.02 | 0.094 |
| Appearance embeddings (NeRF-W) | 24.38 | 0.096 |
| **Shading embeddings (ours)** | **26.54** | **0.090** |

### Key Findings
- The more frames processed simultaneously, the better the results. PSNR improves by nearly 3dB from 1 frame to 64 frames, proving the value of joint relighting.
- Appearance embedding performs even worse than having no embedding (24.38 vs 26.02) because it absorbs highlights as per-image variation.
- Shading embedding outperforms no embedding (26.54 vs 26.02) by correctly adjusting only the normal direction.
- Including purely specular training data not only improves shiny objects (+0.57 PSNR) but also improves standard objects (+0.10 PSNR).
- Without access to ground-truth environment maps on real images, this method still outperforms IllumiNeRF which uses ground-truth environment maps.

## Highlights & Insights
- **The comparison between shading embedding and appearance embedding reveals an important insight**: When addressing illumination inconsistency, it is better to adjust the input to the shading calculation (normal direction) rather than directly adjusting the output (color). This preserves the correctness of the physical rendering pipeline. This principle can be generalized to other 3D reconstruction tasks involving inconsistent inputs.
- **Multiview joint relighting**: Introduces the concept of photometric stereo into diffusion models. By utilizing joint observations from multiple views and illuminations, it resolves ambiguity and yields more consistent results than processing each image independently.
- **Specular material data augmentation**: An unexpected finding—rendering additional training data with perfect mirror materials helps reconstruct not only specular objects but also diffuse ones.

## Limitations & Future Work
- Requires input object masks and accurate camera poses. Pose estimation for highly reflective objects is notoriously difficult (due to the lack of reliable keypoints).
- Completely relies on synthetic data for training. Although experiments demonstrate generalizability to real images, a domain gap may still exist.
- The relighting model processes $N=64$ frames at a time. Scaling up to larger scales (e.g., hundreds of images from internet photo collections) warrants further research.
- High training cost (two weeks on 64 TPU v5s) limits fast iteration.
- Currently only handles object-level reconstruction. Extending this to scene-level reconstruction (e.g., large indoor/outdoor scenes) is a future direction.

## Related Work & Insights
- **vs IllumiNeRF**: IllumiNeRF independently relights each image, causing severe inconsistencies (resulting in blurry reconstructions), and requires ground-truth environment maps as input. This method eliminates inconsistencies through joint processing and only requires a reference image.
- **vs NeROIC**: Inverse-rendering-based methods model materials/lighting with low-frequency parameterizations, which inherently limits their ability to recover highlights. This method bypasses inverse rendering via generative relighting.
- **vs NeRF-W**: Appearance embedding is a common practice for handling illumination changes, but this work demonstrates that it is actually counterproductive for specular objects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Multiview joint relighting and shading embedding are both highly innovative designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic + real data, detailed ablations, and each component is individually validated.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and well-explained intuitive mechanisms.
- Value: ⭐⭐⭐⭐⭐ Resolves a key obstacle for high-quality 3D reconstruction from online image collections.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ROGR: Relightable 3D Objects using Generative Relighting](../../NeurIPS2025/3d_vision/rogr_relightable_3d_objects_using_generative_relighting.md)
- [\[CVPR 2025\] Extreme Rotation Estimation in the Wild](extreme_rotation_estimation_in_the_wild.md)
- [\[CVPR 2025\] Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction](touch2shape_touch-conditioned_3d_diffusion_for_shape_exploration_and_reconstruct.md)
- [\[CVPR 2025\] Decompositional Neural Scene Reconstruction with Generative Diffusion Prior](decompositional_neural_scene_reconstruction_with_generative_diffusion_prior.md)
- [\[CVPR 2025\] ReCap: Better Gaussian Relighting with Cross-Environment Captures](recap_better_gaussian_relighting_with_cross-environment_captures.md)

</div>

<!-- RELATED:END -->

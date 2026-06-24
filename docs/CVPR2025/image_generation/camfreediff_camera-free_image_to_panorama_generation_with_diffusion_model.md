---
title: >-
  [Paper Note] CamFreeDiff: Camera-free Image to Panorama Generation with Diffusion Model
description: >-
  [CVPR 2025][Image Generation][Panorama generation] CamFreeDiff is proposed to achieve 360° panorama generation from a single camera-free image by integrating a lightweight 3-DoF homography estimator into a multi-view diffusion framework. This reduces the FID from 42.4 (MVDiffusion) to 27.0 and generalizes to out-of-domain data without fine-tuning.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Panorama generation"
  - "camera-free"
  - "homography estimation"
  - "multi-view diffusion"
  - "correspondence-aware attention"
date: 2026-05-08
content_hash: 99c14593f634675a
---

# CamFreeDiff: Camera-free Image to Panorama Generation with Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2407.07174](https://arxiv.org/abs/2407.07174)  
**Code**: None (not mentioned)  
**Area**: Diffusion Models / 3D Vision  
**Keywords**: Panorama generation, camera-free, homography estimation, multi-view diffusion, correspondence-aware attention

## TL;DR
CamFreeDiff is proposed to achieve 360° panorama generation from a single camera-free image by integrating a lightweight 3-DoF homography estimator into a multi-view diffusion framework. This reduces the FID from 42.4 (MVDiffusion) to 27.0 and generalizes to out-of-domain data without fine-tuning.

## Background & Motivation

**Background**: 360° panorama generation has important applications in AR/VR. MVDiffusion achieves multi-view consistent panoramic outpainting on frozen pre-trained diffusion models via Correspondence-Aware Attention (CAA), but requires known camera intrinsic and extrinsic parameters of the input image. PanoDiffusion fine-tunes diffusion models to generate panoramas, but destroys pre-trained priors, leading to poor generalization capability.

**Limitations of Prior Work**: Existing methods rely on the assumption that the camera parameters of the input image—including field of view (FOV) and rotation matrix—are known. This severely limits the ability to generate panoramas from arbitrary images (e.g., web images, phone photos). Without camera parameters, establishing pixel correspondences between the input view and target panoramic views becomes impossible, thereby failing to ensure multi-view consistency via CAA.

**Key Challenge**: The CAA mechanism of MVDiffusion requires precise pixel correspondences to ensure panoramic consistency, which in turn require camera parameters. Given that camera parameter estimation inherently contains errors, the key challenge is how to maintain generation quality in the presence of such errors.

**Goal**: To generate high-quality, multi-view consistent 360° panoramas from a single image without knowing its camera parameters.

**Key Insight**: Under the panorama generation scenario, the degrees of freedom of the homography matrix $H = K_2 R K_1^{-1}$ can be simplified from the standard 8-DoF to 3-DoF (FOV, x-axis rotation, z-axis rotation), as many parameters are known constants. This greatly reduces estimation difficulty, and predicting via classification instead of regression further improves accuracy.

**Core Idea**: To estimate the transformation from the input image to the canonical view using a 3-DoF homography classifier, inject the estimated correspondences into the multi-view diffusion framework via correspondence-aware attention, and employ an independent conditional branch to prevent the propagation of estimation errors.

## Method

### Overall Architecture
The 360° scene is split into 8 perspective views (90° FOV, 45° horizontal overlap). The homography matrix from the input image to the canonical view is estimated to obtain pixel correspondences between the input and each target view. Three variant strategies are designed to inject these correspondences into the multi-view diffusion model: Variant 1 (unwarped image), Variant 2 (unwarped latent), and Variant 3 (independent conditional branch + CAA). Based on the Stable Diffusion inpainting model, the weights of VAE and U-Net are frozen, training only the MLP classifier and the CAA modules.

### Key Designs

1. **3-DoF Homography Parameterization and Classification Estimation**

    - **Function**: To estimate the homography transformation from a camera-free input image to the canonical view.
    - **Mechanism**: Under the panorama generation scenario, the intrinsics $K_2$ of the canonical view are known (FOV=90°, center-aligned), the input image is assumed to be captured by a pinhole camera (zero distortion, principal point at the center), and the y-axis rotation $\theta$ is meaningless for a single image (as it can be mapped to any canonical view). Thus, the homography matrix only needs to predict 3 degrees of freedom: FOV $f$, x-axis rotation $\phi$, and z-axis rotation $\psi$. A frozen SD U-Net encoder extracts image features, which are fed into a 3-layer MLP classifier (5120→2560→1280) to perform classification predictions for each of the three parameters (using cross-entropy loss). Classification outperforms regression: the FOV MAE is reduced from 10.6° to 7.9°.
    - **Design Motivation**: The standard 8-DoF parameterization mixes rotation and translation terms, making optimization difficult. The 3-DoF design leverages the task characteristics of panorama generation to substantially simplify the problem.

2. **Variant 3: Independent Conditional Branch + Correspondence-Aware Attention (CAA)**

    - **Function**: To ensure panoramic generation quality even when homography estimation contains errors.
    - **Mechanism**: Unlike Variant 1/2 (which directly unwarp the image/latent to the canonical view before inpainting), Variant 3 designs 1 conditional branch and 8 generation branches. The conditional branch receives the original input image (without unwarping), while the 8 generation branches each handle one canonical view. Pixel correspondences between the conditional branch and each generation branch are established via the predicted homography, and a CAA module is used to transfer information using cross-attention within a $K \times K$ neighborhood. Consequently, estimation errors only affect the selection of corresponding positions, without directly destroying the input image content (as in Variant 1) or distorting textures (as in Variant 2).
    - **Design Motivation**: Experiments reveal that unwarping schemes degrade severely when homography estimation is inaccurate—Variant 1 suffers from inconsistent scene layout, and Variant 2 exhibits texture distortion. By decoupling the conditioning and generation processes, Variant 3 enables the model to self-learn how to extract useful information from rough correspondences.

3. **Multi-View Correspondence-Aware Attention (CAA)**

    - **Function**: To ensure geometric consistency across the 8 panoramic views without modifying the pre-trained model weights.
    - **Mechanism**: Following the design of MVDiffusion, for point $p_s$ in the source view $I_s$ and its corresponding point $p_t$ in the target view $I_t$, information is aggregated from the $K \times K$ neighborhood of $p_s$ to $p_t$ (cross-attention, where Query comes from $p_t$, and Key/Value come from the neighborhood). The extension in CamFreeDiff is to additionally establish CAA connections between the conditional branch and all generation branches.
    - **Design Motivation**: Relying solely on the CAA between generation branches is insufficient to diffuse the input image information to all views; direct bridging with the conditional branch is necessary.

### Loss & Training
Homography estimation uses cross-entropy classification losses (separately for FOV, $\phi$, $\psi$). Panorama generation uses standard diffusion denoising loss. VAE and U-Net are frozen, and only the MLP + CAA modules are trained. It is trained on the Matterport3D dataset for 30 epochs with a learning rate of $2 \times 10^{-4}$. The training data is augmented using random transformations (FOV 60°-110°, rotation ±15°).

## Key Experimental Results

### Main Results

| Method | FID↓ | IS↑ | CLIP Score↑ | PSNR↑ |
|------|------|-----|-------------|-------|
| PanoDiffusion | 48.7 | 3.1 | — | — |
| MVDiffusion (given camera parameters) | 42.4 | 5.4 | 21.9 | — |
| CamFreeDiff V1 (unwarp image) | 35.2 | 5.5 | 23.6 | 18.7 |
| CamFreeDiff V2 (unwarp latent) | 34.3 | 5.6 | 22.4 | 15.6 |
| **CamFreeDiff V3 (new view)** | **27.0** | **5.6** | **24.4** | **19.3** |

Zero-shot Structured3D: FID **31.1** (vs PanoDiffusion 35.3, which was trained on this dataset)

### Ablation Study

| Configuration | FOV MAE↓ | phi MAE↓ | psi MAE↓ |
|------|---------|---------|---------|
| MSE Regression | 10.6° | 2.5° | 2.4° |
| **CE Classification** | **7.9°** | **1.8°** | **1.5°** |

| Homography Estimator Architecture | FID↓ | PSNR↑ |
|-----------------|------|-------|
| HomographyNet | 29.2 | 19.2 |
| **SD encoder + MLP** | **27.0** | **19.3** |

### Key Findings
- Variant 3 (independent conditional branch) significantly outperforms V1/V2: FID 27.0 vs 35.2/34.3, PSNR 19.3 vs 18.7/15.6, indicating that avoiding direct unwarping of the input is crucial.
- Classification outperforms regression, significantly reducing estimation errors, especially for FOV (7.9° vs 10.6°), which corresponds to improved panorama quality.
- Reusing SD encoder features outperforms an independent HomographyNet (FID 27.0 vs 29.2), with zero extra encoding overhead.
- Zero-shot generalization on Structured3D even surpasses PanoDiffusion trained on the same dataset (FID 31.1 vs 35.3), indicating that freezing pre-trained weights preserves generalization ability.

## Highlights & Insights
- **3-DoF simplification** is an exemplar of task-driven parameterization: leveraging the symmetry and constraints of panorama generation to reduce 8-DoF to 3-DoF, with classification replacing regression to further reduce difficulty.
- The **decoupled design of Variant 3** elegantly handles estimation errors: instead of directly altering the input, it allows the model to self-learn to exploit coarse correspondences via correspondence attention, ensuring robustness to errors.
- The strategy of **freezing the pre-trained model + lightweight training modules** is highly effective in preserving generalization capability.

## Limitations & Future Work
- Training and evaluation are conducted only on indoor scenes (Matterport3D/Structured3D), leaving outdoor generalization unverified.
- The pinhole camera model is assumed (zero distortion, centered principal point), which does not support special lenses like fisheyes.
- Y-axis rotation is not predicted (set to 0), making it impossible to determine the absolute horizontal orientation of the input image.
- Increasing the CAA neighborhood size ($K=5,7$) yields limited performance gains while significantly increasing computational overhead.

## Related Work & Insights
- **vs MVDiffusion**: MVDiffusion requires known camera parameters, while CamFreeDiff removes this constraint. Even when MVDiffusion is provided with estimated parameters, CamFreeDiff remains more robust (FID 27.0 vs 42.4).
- **vs PanoDiffusion**: PanoDiffusion fine-tunes the entire diffusion model, leading to poor generalization. CamFreeDiff freezes pre-trained weights and achieves zero-shot superiority on out-of-domain data.
- **vs PanoDiff**: PanoDiff estimates latitude and longitude angles but still assumes known FOV and roll angles, whereas CamFreeDiff estimates the full 3-DoF.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel problem definition (camera-free panorama); 3-DoF parameterization and Variant 3 design are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-variant comparison + out-of-domain generalization + ablation study, though limited to indoor scenes and a narrow dataset range.
- Writing Quality: ⭐⭐⭐⭐ Clear description of methods, extensive illustrations, and intuitive comparison of the three variants.
- Value: ⭐⭐⭐⭐ Solves practical bottlenecks of panorama generation (obviating camera parameters), showing value for AR/VR content creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Panorama Generation From NFoV Image Done Right](panorama_generation_from_nfov_image_done_right.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] TKG-DM: Training-Free Chroma Key Content Generation Diffusion Model](tkg-dm_training-free_chroma_key_content_generation_diffusion_model.md)
- [\[ICCV 2025\] What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?](../../ICCV2025/image_generation/what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)
- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)

</div>

<!-- RELATED:END -->

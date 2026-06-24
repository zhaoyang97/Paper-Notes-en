---
title: >-
  [Paper Note] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction
description: >-
  [AAAI 2026][3D Vision][Dynamic Scene Reconstruction] This paper proposes Sparse4DGS, the first 4D dynamic scene reconstruction method designed for sparse-frame inputs. Through two core modules, Texture-Aware Deformation Regularization (TADR) and Texture-Aware Canonical Optimization (TACO), it guides the Gaussian distribution to focus on texture-rich areas, achieving high-quality dynamic novel view synthesis with only 5–30 sparse input frames.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Dynamic Scene Reconstruction"
  - "4D Gaussian Splatting"
  - "Sparse Frames"
  - "Texture-Aware"
  - "Stochastic Gradient Langevin Dynamics"
date: 2026-05-08
content_hash: 12b8e19813e75b1b
---

# Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction

**Conference**: AAAI 2026  
**arXiv**: [2511.07122](https://arxiv.org/abs/2511.07122)  
**Code**: [Project Page](https://ChangyueShi.github.io/Sparse4DGS)  
**Area**: 3D Vision  
**Keywords**: Dynamic Scene Reconstruction, 4D Gaussian Splatting, Sparse Frames, Texture-Aware, Stochastic Gradient Langevin Dynamics  

## TL;DR

This paper proposes Sparse4DGS, the first 4D dynamic scene reconstruction method designed for sparse-frame inputs. Through two core modules, Texture-Aware Deformation Regularization (TADR) and Texture-Aware Canonical Optimization (TACO), it guides the Gaussian distribution to focus on texture-rich areas, achieving high-quality dynamic novel view synthesis with only 5–30 sparse input frames.

## Background & Motivation

Dynamic Gaussian Splatting methods have made significant progress in 4D scene reconstruction. However, existing methods such as Deformable3DGS and 4DGaussians heavily rely on dense-frame video sequences (typically requiring hundreds of frames). In the real world, due to device limitations (e.g., low-frame-rate cameras), only sparse frames are often available.

The authors discover that when the input frames are reduced from dense to sparse, existing methods suffer from severe degradation in **texture-rich regions**. This is because:
1. **Deformation Space Degradation**: Sparse inputs provide insufficient temporal constraints, making it impossible for the deformation network to accurately model geometric changes in high-frequency texture regions.
2. **Canonical Space Degradation**: The canonical Gaussian field lacks adequate supervision signals, leading to geometric collapse in complex texture regions.

The core insight is that since sparse frame inputs naturally provide limited information, high-frequency texture signals become the main source of rich details and dynamic cues. Therefore, guiding Gaussians to focus on texture-rich regions can help model the underlying structure more effectively.

## Method

### Overall Architecture

Sparse4DGS is based on the dynamic reconstruction paradigm of a canonical Gaussian field combined with a deformation network. Given a sparse sequence of input frames:
1. A Sobel operator is used to extract the 2D Texture Intensity (TI) map for each frame.
2. A monocular depth estimator (DPT) is used to obtain depth maps.
3. Texture intensity is embedded into the 3D Gaussian attributes.
4. The deformation network is regularized via TADR.
5. The canonical Gaussian field is optimized via TACO.

### Key Designs

#### 1. **TI Gaussian Field (Texture Intensity Gaussian Field)**: Embedding Texture Richness into 3D Gaussians

First, the horizontal and vertical gradient maps $TI_x$ and $TI_y$ of each input RGB image are computed using a Sobel operator. Then, the pixel-wise gradient magnitude is obtained as an explicit measure of texture intensity:

$$TI_{gt}(i,j) = \sqrt{TI_x(i,j)^2 + TI_y(i,j)^2}$$

To represent texture richness in 3D space, a new attribute $TI$ is introduced for each Gaussian, which is rendered into a texture map $TI_{render}$ using a differentiable rasterizer.

**Key Innovation**: The Pearson Correlation Coefficient (PCC) is utilized instead of the conventional L1 loss to align the rendered texture map with the ground-truth texture map. This choice stems from the fact that applying the Sobel operator independently to each image leads to spatial inconsistencies, whereas PCC focuses on relative rates of change, effectively mitigating this issue:

$$L_{tex} = 1 - \text{PCC}(TI_{gt}, TI_{render})$$

#### 2. **Texture-Aware Deformation Regularization (TADR)**: Constraining the Geometry of the Deformation Network

The core idea of TADR is to constrain the deformation field using the texture consistency of depth maps. Traditional methods directly compare the image-level PCC between rendered depth and monocular depth, but they fail to capture local depth variations.

TADR addresses this by:
- First extracting texture intensity maps from the rendered depth $D_{render}$ and DPT depth $D_{dpt}$ using a Sobel operator.
- Then computing the PCC loss between these two depth-texture maps.

$$L_{tadr} = 1 - \text{PCC}(TI_{gt}^{depth}, TI_{render}^{depth})$$

This "textured" depth alignment method focuses more on the consistency of local depth changes rather than the global depth distribution.

#### 3. **Texture-Aware Canonical Optimization (TACO)**: Reconstructing the Gradient Descent of Canonical Gaussians

TACO is based on Stochastic Gradient Langevin Dynamics (SGLD). It introduces a texture-intensity-based noise term in each iteration to drive the Gaussians to converge towards texture-rich regions:

$$g = g - \alpha_g \cdot \nabla_g \mathbb{E}[L(g;I)] + \alpha_{noise} \cdot (\epsilon_{tex} + \epsilon_o)$$

where the texture noise term is defined as:
$$\epsilon_{tex} = \sigma(-k(TI - t)) \cdot \sum \eta$$

When a Gaussian enters a texture-rich region, its $TI$ value approaches 1, $\epsilon_{tex}$ approaches 0, and the noise naturally ceases. This means the noise continuously perturbs the optimization process until the Gaussians converge to texture-rich regions. $\epsilon_o$ is used to reduce blur-inducing low-opacity Gaussians (floaters).

### Loss & Training

The total training loss is defined as:
$$L = L_{rgb} + \lambda_1 \cdot L_{tex} + \lambda_2 \cdot L_{tadr}$$

where $L_{rgb}$ is the standard MSE + SSIM loss. The optimal hyperparameters are $\lambda_1 = \lambda_2 = 0.01$.

The training process uses TACO instead of standard SGD to update the canonical Gaussian parameters. This method is applicable to videos of various frame rates from 5 FPS to 30 FPS.

## Key Experimental Results

### Main Results

| Dataset | Metric | Sparse4DGS | Deformable3DGS | 4DGaussians | CoRGS | Gain |
|--------|------|------------|----------------|-------------|-------|------|
| NeRF-Synthetic (20 frames) | PSNR↑ | **25.31** | 22.65 | 22.47 | 20.15 | +2.66 |
| NeRF-Synthetic (20 frames) | SSIM↑ | **0.944** | 0.927 | 0.931 | 0.920 | +0.013 |
| NeRF-DS (20 frames) | PSNR↑ | **22.34** | 20.81 | 19.70 | 19.86 | +1.53 |
| NeRF-DS (20 frames) | LPIPS↓ | **0.233** | 0.301 | 0.350 | 0.319 | -0.068 |
| HyperNeRF (30 frames) | PSNR↑ | **23.91** | 22.41 | 20.64 | 20.50 | +1.50 |
| iPhone-4D (30FPS) | PSNR↑ | **29.81** | 27.01 | 28.79 | 21.58 | +1.02 |
| iPhone-4D (5FPS) | PSNR↑ | **27.51** | 21.12 | 16.37 | 16.81 | +6.39 |

It significantly outperforms prior methods across all datasets, particularly in the extremely sparse 5 FPS scenarios, where the PSNR gains exceed 6dB.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Description |
|------|-------|-------|--------|------|
| Baseline (w/o TADR+TACO) | 20.81 | 0.753 | 0.301 | Baseline method |
| w/o TADR | 21.89 | 0.792 | 0.245 | Remove deformation regularization, PSNR drops by 0.45 |
| w/o TACO | 21.33 | 0.773 | 0.271 | Remove canonical optimization, PSNR drops by 1.01 |
| **Full Method** | **22.34** | **0.801** | **0.233** | TACO contributes more |
| TACO w/o $\epsilon_o$ | 21.81 | 0.792 | 0.246 | Remove opacity noise term |
| TACO w/o $\epsilon_{tex}$ | 21.57 | 0.783 | 0.260 | Remove texture noise term |
| $L_{tex}$ w/o PCC | 21.71 | 0.789 | 0.245 | Replace PCC with L1, drops by 0.6 |
| w/o texture-aware depth | 21.46 | 0.775 | 0.277 | Standard depth regularization |

### Key Findings

1. The contribution of TACO is greater than that of TADR (1.01 vs 0.45 PSNR gain), indicating that canonical space optimization is the primary bottleneck in sparse-frame reconstruction.
2. The PCC loss shows a significant advantage over the L1 loss in both texture embedding and depth alignment.
3. The texture-aware depth loss improves PSNR by 0.88 compared to direct depth PCC alignment.
4. The performance gain is most pronounced in extremely sparse 5 FPS scenarios (+6.39 PSNR).

## Highlights & Insights

1. **Novel Problem Formulation**: This work is the first to define and systematically investigate the problem of 4D dynamic scene reconstruction from sparse frames.
2. **Texture-Driven Optimization Strategy**: Based on the observation that degradation under sparse frames is concentrated in texture-rich regions, a complete solution is designed to tackle this issue.
3. **Innovative Application of SGLD**: Stochastic Gradient Langevin Dynamics is introduced into dynamic Gaussian optimization, with an elegant and effective texture-guided noise design.
4. **Replacing L1 with PCC**: In scenes featuring spatial inconsistencies, PCC serves as a more robust correlation metric than L1 loss.
5. **Real-World Validation**: The introduction of the iPhone-4D dataset demonstrates the potential for practical applications on videos captured by mobile phones.

## Limitations & Future Work

1. When texture information in the scene is extremely scarce (e.g., solid-colored walls), the method's effectiveness may be limited.
2. The performance relies on the accuracy of the DPT monocular depth estimator, meaning errors from the pre-trained depth model may propagate.
3. The scale of the iPhone-4D dataset is relatively small (only 4 scenes), limiting the scope of validation.
4. Very short sequences (e.g., 2–3 frames) have not yet been explored.
5. The noise hyperparameters of TACO may require scene-specific tuning.

## Related Work & Insights

- **Dynamic Gaussian Splatting**: Deformable3DGS and 4DGaussians establish the standard paradigm of a canonical field combined with a deformation network.
- **Few-Shot Gaussian Splatting**: DNGaussian introduces depth regularization, CoRGS refines the training process, and FSGS addresses sparse initialization.
- **Application of SGLD in 3DGS**: Kheradmand et al. first introduced SGLD into Gaussian Splatting optimization.
- **Insights**: Credit to the texture-guided optimization scheme, which can potentially be extended to other 3D reconstruction tasks with sparse inputs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to study sparse-frame dynamic reconstruction, introducing a novel texture-aware strategy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Extensive validation on four datasets, accompanied by detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and mathematically rigorous formulations.
- **Value**: ⭐⭐⭐⭐ — Directly links to practical value for the dynamic reconstruction of low-frame-rate videos.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos](dynamic_gaussian_scene_reconstruction_from_unsynchronized_videos.md)
- [\[ICLR 2026\] Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements](../../ICLR2026/3d_vision/implicit_4d_gaussian_splatting_for_fast_motion_with_large_inter-frame_displaceme.md)
- [\[ICLR 2026\] Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting](../../ICLR2026/3d_vision/mango-gs_enhancing_spatio-temporal_consistency_in_dynamic_scenes_reconstruction_.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](../../CVPR2026/3d_vision/retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[CVPR 2026\] Layered 4D-Rotor Gaussian Splatting: A Compressed Representation for Long Dynamic Scenes](../../CVPR2026/3d_vision/layered_4d-rotor_gaussian_splatting_a_compressed_representation_for_long_dynamic.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction
description: >-
  [AAAI 2026][3D Vision][dynamic scene reconstruction] Sparse4DGS is proposed as the first method for sparse-frame dynamic scene reconstruction, achieving high-fidelity 4D scene reconstruction from sparse video frames via two core modules: Texture-Aware Deformation Regularization (TADR) and Texture-Aware Canonical Optimization (TACO).
tags:
  - AAAI 2026
  - 3D Vision
  - dynamic scene reconstruction
  - 4D Gaussian splatting
  - sparse frames
  - texture-awareness
  - stochastic gradient Langevin dynamics
date: 2026-05-08
content_hash: 612add3f6060e8f4
---

# Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2511.07122](https://arxiv.org/abs/2511.07122)
**Code**: [Project Page](https://ChangyueShi.github.io/Sparse4DGS)
**Area**: 3D Vision
**Keywords**: dynamic scene reconstruction, 4D Gaussian splatting, sparse frames, texture-awareness, stochastic gradient Langevin dynamics

## TL;DR

Sparse4DGS is proposed as the first method for sparse-frame dynamic scene reconstruction, achieving high-fidelity 4D scene reconstruction from sparse video frames via two core modules: Texture-Aware Deformation Regularization (TADR) and Texture-Aware Canonical Optimization (TACO).

## Background & Motivation

Existing dynamic Gaussian splatting methods such as Deformable3DGS and 4DGaussians have demonstrated significant progress in 4D scene reconstruction, but they rely heavily on dense-frame video sequences (typically hundreds of frames). In practice, only sparse frames are often available due to hardware constraints such as low-frame-rate cameras.

The authors find that when the number of input frames is reduced from dense to sparse, existing methods suffer severe degradation in **texture-rich regions**, due to two factors:
1. **Deformation space degradation**: Insufficient temporal constraints from sparse inputs prevent the deformation network from accurately modeling geometric changes in high-frequency texture regions.
2. **Canonical space degradation**: The canonical Gaussian field lacks adequate supervision signals, leading to geometric collapse in regions with complex texture.

The core intuition is that sparse-frame inputs inherently provide limited information, making high-frequency texture signals the primary source of rich detail and dynamic cues. Accordingly, Gaussians should be guided to focus on texture-rich regions to better model the underlying structure.

## Method

### Overall Architecture

Sparse4DGS follows the canonical Gaussian field plus deformation network paradigm for dynamic reconstruction. Given a sparse frame sequence, the pipeline proceeds as follows:
1. Apply the Sobel operator to extract 2D texture intensity (TI) maps for each frame.
2. Obtain depth maps using a monocular depth estimator (DPT).
3. Embed texture intensity into 3D Gaussian attributes.
4. Regularize the deformation network via TADR.
5. Optimize the canonical Gaussian field via TACO.

### Key Designs

#### 1. **Texture Intensity Gaussian Field**: Embedding texture richness into 3D Gaussians

Horizontal and vertical gradient maps $TI_x$ and $TI_y$ are computed for each input RGB image using the Sobel operator, yielding per-pixel gradient magnitudes as an explicit measure of texture intensity:

$$TI_{gt}(i,j) = \sqrt{TI_x(i,j)^2 + TI_y(i,j)^2}$$

To represent texture richness in 3D space, each Gaussian is assigned a new attribute $TI$, which is rendered into a texture map $TI_{render}$ via a differentiable rasterizer.

**Key innovation**: Pearson Correlation Coefficient (PCC) is used instead of the conventional L1 loss to align the rendered texture map with the ground-truth texture map. Since the Sobel operator is applied independently to each image, spatial inconsistencies arise; PCC, which measures relative variation rather than absolute differences, effectively mitigates this issue:

$$L_{tex} = 1 - \text{PCC}(TI_{gt}, TI_{render})$$

#### 2. **Texture-Aware Deformation Regularization (TADR)**: Constraining geometric structure in the deformation network

TADR leverages the textural consistency of depth maps to constrain the deformation field. Conventional methods compute image-level PCC between rendered depth and monocular depth, which fails to capture local depth variations. TADR instead:
- Applies the Sobel operator to both the rendered depth $D_{render}$ and the DPT depth $D_{dpt}$ to extract their respective texture intensity maps.
- Computes the PCC loss between these two depth texture maps:

$$L_{tadr} = 1 - \text{PCC}(TI_{gt}^{depth}, TI_{render}^{depth})$$

This texture-based depth alignment focuses on the consistency of local depth variations rather than the global depth distribution.

#### 3. **Texture-Aware Canonical Optimization (TACO)**: Restructuring gradient descent for canonical Gaussians

TACO is based on Stochastic Gradient Langevin Dynamics (SGLD) and introduces texture-intensity-guided noise at each iteration to drive Gaussians toward texture-rich regions:

$$g = g - \alpha_g \cdot \nabla_g \mathbb{E}[L(g;I)] + \alpha_{noise} \cdot (\epsilon_{tex} + \epsilon_o)$$

The texture noise term is defined as:

$$\epsilon_{tex} = \sigma(-k(TI - t)) \cdot \sum \eta$$

When a Gaussian reaches a texture-rich region, its $TI$ value approaches 1 and $\epsilon_{tex}$ approaches 0, causing the noise to vanish naturally. This means the noise continuously perturbs the optimization until Gaussians converge to texture-rich regions. The opacity noise term $\epsilon_o$ serves to reduce low-opacity floaters.

### Loss & Training

The total training loss is:

$$L = L_{rgb} + \lambda_1 \cdot L_{tex} + \lambda_2 \cdot L_{tadr}$$

where $L_{rgb}$ is the standard MSE + SSIM loss. The optimal hyperparameters are $\lambda_1 = \lambda_2 = 0.01$.

TACO replaces the standard SGD update for canonical Gaussian parameters during training. The method generalizes across different frame rates ranging from 5 FPS to 30 FPS.

## Key Experimental Results

### Main Results

| Dataset | Metric | Sparse4DGS | Deformable3DGS | 4DGaussians | CoRGS | Gain |
|--------|------|------------|----------------|-------------|-------|------|
| NeRF-Synthetic (20 frames) | PSNR↑ | **25.31** | 22.65 | 22.47 | 20.15 | +2.66 |
| NeRF-Synthetic (20 frames) | SSIM↑ | **0.944** | 0.927 | 0.931 | 0.920 | +0.013 |
| NeRF-DS (20 frames) | PSNR↑ | **22.34** | 20.81 | 19.70 | 19.86 | +1.53 |
| NeRF-DS (20 frames) | LPIPS↓ | **0.233** | 0.301 | 0.350 | 0.319 | −0.068 |
| HyperNeRF (30 frames) | PSNR↑ | **23.91** | 22.41 | 20.64 | 20.50 | +1.50 |
| iPhone-4D (30 FPS) | PSNR↑ | **29.81** | 27.01 | 28.79 | 21.58 | +1.02 |
| iPhone-4D (5 FPS) | PSNR↑ | **27.51** | 21.12 | 16.37 | 16.81 | +6.39 |

Sparse4DGS achieves substantial improvements across all datasets, with the most prominent gain of over 6 dB PSNR in the extremely sparse 5 FPS setting.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Note |
|------|-------|-------|--------|------|
| Baseline (w/o TADR & TACO) | 20.81 | 0.753 | 0.301 | Base method |
| w/o TADR | 21.89 | 0.792 | 0.245 | Remove deformation regularization, −0.45 PSNR |
| w/o TACO | 21.33 | 0.773 | 0.271 | Remove canonical optimization, −1.01 PSNR |
| **Full model** | **22.34** | **0.801** | **0.233** | TACO contributes more |
| TACO w/o $\epsilon_o$ | 21.81 | 0.792 | 0.246 | Remove opacity noise term |
| TACO w/o $\epsilon_{tex}$ | 21.57 | 0.783 | 0.260 | Remove texture noise term |
| $L_{tex}$ w/o PCC | 21.71 | 0.789 | 0.245 | Replace PCC with L1, −0.6 PSNR |
| w/o texture-aware depth | 21.46 | 0.775 | 0.277 | Standard depth regularization |

### Key Findings

1. TACO contributes more than TADR (1.01 vs. 0.45 PSNR gain), indicating that canonical space optimization is the primary bottleneck for sparse-frame reconstruction.
2. PCC loss yields significant advantages over L1 loss for both texture embedding and depth alignment.
3. Texture-aware depth regularization improves PSNR by 0.88 over direct depth PCC alignment.
4. The advantage is most pronounced in the extreme 5 FPS sparse setting (+6.39 PSNR).

## Highlights & Insights

1. **Novel problem formulation**: The paper is the first to formally define and systematically study sparse-frame 4D dynamic scene reconstruction.
2. **Texture-driven optimization strategy**: The observation that sparse-frame degradation concentrates in texture-rich regions motivates a complete and principled solution.
3. **Creative application of SGLD**: Stochastic Gradient Langevin Dynamics is introduced into dynamic Gaussian optimization, with an elegant texture-guided noise term design.
4. **PCC as a substitute for L1**: PCC proves more robust than L1 as a correlation measure in the presence of spatial inconsistencies.
5. **Real-world validation**: The proposed iPhone-4D dataset demonstrates practical applicability to smartphone-captured videos.

## Limitations & Future Work

1. The method may be limited in scenes with extremely sparse texture (e.g., uniformly colored surfaces).
2. The approach depends on the accuracy of the DPT monocular depth estimator; errors from pre-trained depth models may propagate.
3. The iPhone-4D dataset is small in scale (only 4 scenes), limiting the breadth of validation.
4. Extremely short sequences (e.g., 2–3 frames) are not explored.
5. The noise hyperparameters in TACO may require tuning for different scenes.

## Related Work & Insights

- **Dynamic Gaussian splatting**: Deformable3DGS and 4DGaussians establish the standard canonical field plus deformation network framework.
- **Few-shot Gaussian splatting**: DNGaussian introduces depth regularization, CoRGS improves the training process, and FSGS addresses sparse initialization.
- **SGLD in 3DGS**: Kheradmand et al. first introduce SGLD into Gaussian splatting optimization.
- **Inspiration**: The texture-guided optimization strategy is potentially transferable to other 3D reconstruction tasks with sparse inputs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First sparse-frame dynamic reconstruction method; texture-aware strategy is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and rigorous method derivation.
- **Value**: ⭐⭐⭐⭐ — Direct practical value for dynamic reconstruction from low-frame-rate video.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos](dynamic_gaussian_scene_reconstruction_from_unsynchronized_videos.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](../../CVPR2026/3d_vision/retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[ICLR 2026\] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](../../ICLR2026/3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians
description: >-
  [ECCV 2024][3D Vision][Sparse View Synthesis] CoherentGS is proposed to introduce a structured representation (one Gaussian per pixel) for 3DGS, establishing single-view and multi-view consistency constraints using an implicit convolutional decoder and total variation loss. Combined with a monocular depth-based initialization strategy, it achieves high-quality novel view synthesis under extremely sparse inputs (e.g., 3 images), outperforming existing NeRF methods significantl…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Sparse View Synthesis"
  - "3D Gaussian Splatting"
  - "Implicit Decoder"
  - "Depth Initialization"
  - "Regularized Optimization"
date: 2026-05-08
content_hash: 5d72089d6f77aef9
---

# CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians

**Conference**: ECCV 2024  
**arXiv**: [2403.19495](https://arxiv.org/abs/2403.19495)  
**Code**: [https://people.engr.tamu.edu/nimak/Papers/CoherentGS](https://people.engr.tamu.edu/nimak/Papers/CoherentGS) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Sparse View Synthesis, 3D Gaussian Splatting, Implicit Decoder, Depth Initialization, Regularized Optimization

## TL;DR

CoherentGS is proposed to introduce a structured representation (one Gaussian per pixel) for 3DGS, establishing single-view and multi-view consistency constraints using an implicit convolutional decoder and total variation loss. Combined with a monocular depth-based initialization strategy, it achieves high-quality novel view synthesis under extremely sparse inputs (e.g., 3 images), outperforming existing NeRF methods significantly in terms of LPIPS.

## Background & Motivation

3D Gaussian Splatting (3DGS) exhibits significant advantages in training speed and rendering quality compared to NeRF. However, it suffers from severe overfitting under extremely sparse inputs (e.g., 2-4 images), presenting a "needle-like clutter" appearance from novel viewpoints.

Limitations of prior sparse-view methods:

**NeRF-based methods**: Although methods like RegNeRF, FreeNeRF, and SparseNeRF propose various regularizations, their constraints are insufficient, still leading to noticeable blurriness and artifacts.

**Difficulty in directly applying NeRF regularizations to 3DGS**: NeRF methods rely on the implicit continuity of MLPs to propagate sparse constraints, whereas 3DGS is a discrete, unstructured explicit representation where constraints cannot propagate naturally.

**Contemporaneous 3DGS methods (FSGS, SparseGS, DNGaussian)**: These methods fail to enforce consistency between neighboring Gaussians, thereby generating floating artifacts.

Key Challenge: The unstructured nature of 3DGS makes it difficult to introduce effective regularization constraints under sparse inputs.

Core Idea: Convert unstructured 3D Gaussians into a structured 2D pixel-level representation, imposing consistency constraints in the 2D image space so that neighboring Gaussians move collaboratively rather than being optimized independently.

## Method

### Overall Architecture

1. **Preprocessing**: Obtain monocular depth estimation (Depth Anything) and dense optical flow correspondences (FlowFormer++).
2. **Initialization**: Initialize one 3D Gaussian for each pixel of every input image using monocular depth, and align depths across different views using optical flow.
3. **Regularized Optimization**: Perform joint optimization using an implicit decoder (single-view constraint) + TV loss (multi-view constraint) + optical flow regularization.

### Key Designs

1. **Structured Gaussian Representation**:

    - Assign one Gaussian to **each pixel** of every input image, totaling N×H×W Gaussians.
    - The position of each Gaussian is constrained along the ray connecting the camera center and the corresponding pixel, controlled by a scalar depth value.
    - Position updates utilize residual depth: $\mathbf{x} = g(D_n^{\text{init}}[\mathbf{p}] + \Delta D_n[\mathbf{p}], \mathbf{p})$
    - **Design Motivation**: Constraining Gaussian movement along ray directions dramatically reduces degrees of freedom during optimization, making regularization feasible.

2. **Implicit Convolutional Decoder (Single-view Constraint)**:

    - Instead of directly optimizing each pixel's residual depth, an implicit decoder is used to predict the residual depth map for the entire image.
    - The input is the normalized view index n, and the output is the residual depth map: $\Delta D_n = f_\phi(n)$
    - Optimizing the decoder parameters $\phi$ instead of pixel-wise depth ensures smooth residual depths and collaborative surface deformation.
    - **Handling Depth Discontinuities**: The image is partitioned into C=5 regions based on monocular depth values. The decoder outputs C-channel residual depths, allowing different regions to deform independently through a segmentation mask.
    - A similar constraint is applied to the global opacity $\alpha$.

3. **Total Variation (TV) Regularization (Multi-view Constraint)**:

    - Apply TV loss on depth maps rendered from all Gaussians to ensure geometric smoothness across views.
    $\mathcal{L}_{\text{TV}} = \left\|\nabla\left(\frac{1}{1+R_{\Sigma,\alpha,\mathbf{x},d}}\right)\right\|_1$
    - Masked TV loss $\mathcal{L}_{\text{MTV}}$ preserves structural details.
    - Adopt a progressive strategy: first minimize global $\mathcal{L}_{\text{TV}}$ to obtain globally smooth and continuous geometry, then gradually increase $\mathcal{L}_{\text{MTV}}$ to recover details.
    $\mathcal{L}_{\text{multi}} = (1-\lambda_s)\mathcal{L}_{\text{TV}} + \lambda_s\mathcal{L}_{\text{MTV}}$
   where $\lambda_s$ gradually increases from 0 to 1.

4. **Flow-based Regularization**:

    - Core Idea: Dynamic correspondence points in two input images originate from the same 3D point, so the Gaussian positions of corresponding pixels should be similar.
    $\mathcal{L}_{\text{flow}} = \sum_{(i,j)}\sum_\mathbf{p} \|M_{i\to j} \odot (g(D_i[\mathbf{p}],\mathbf{p}) - g(D_j[\mathbf{q}],\mathbf{q}))\|_1$
    - Use FlowFormer++ to compute correspondences, and perform forward-backward consistency checks to obtain reliable masks.

5. **Monocular Depth-Based Initialization**:

    - Use Depth Anything to obtain high-quality but scale-inconsistent relative depths across views.
    - Optimize only the global scale and offset of each depth map via flow-based loss for coarse alignment.
    $\mathbf{s}^*, \mathbf{o}^* = \arg\min_{\mathbf{s},\mathbf{o}} \sum_{(i,j)} \|M_{i\to j} \odot (g(s_i \cdot D_i^m + o_i, \mathbf{p}) - g(s_j \cdot D_j^m + o_j, \mathbf{q}))\|_1$
    - Gaussian scaling factors are computed from depth: $r = f \cdot D^{\text{init}} / H$, ensuring each Gaussian exactly covers its corresponding pixel.
    - Initial opacity decreases with the number of input views: 0.6 for 2 views, 0.5 for 3 views, 0.35 for 4 views.

### Loss & Training

Total optimization objective:
$$\Sigma^*,\phi^*,\mathbf{c}^* = \arg\min \sum_{\mathbf{p}} \mathcal{L}(R(\mathbf{p}), \hat{R}(\mathbf{p})) + \beta_m \mathcal{L}_{\text{multi}} + \beta_f \mathcal{L}_{\text{flow}}$$

- $\beta_m = 5$, $\beta_f = 0.1$
- Initialization phase: 1000 iterations for coarse alignment.
- Optimization phase: 13,000 iterations in total. The first 8000 iterations fix the rotation as identity matrix and scale as computed by the formula, while the last 5000 iterations freely optimize rotation and scaling.
- **Multisampling**: Sample multiple points within each pixel and average them, addressing the semi-transparency issue caused by Gaussians not fully covering pixels under sparse inputs.

## Key Experimental Results

### Main Results

**LLFF Dataset (3 input views)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3DGS | 14.99 | 0.483 | 0.362 |
| RegNeRF | 19.41 | 0.627 | 0.306 |
| FreeNeRF | 19.97 | 0.652 | 0.280 |
| SparseNeRF | 20.33 | 0.657 | 0.302 |
| **CoherentGS** | **20.33** | **0.725** | **0.180** |

**NVS-RGBD ZED2 Dataset (3 input views)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| FreeNeRF | 25.60 | 0.817 | 0.166 |
| SparseNeRF | 26.56 | 0.835 | 0.154 |
| **CoherentGS** | 24.93 | 0.840 | **0.135** |

- Inference speed: 278 fps (LLFF with 3 inputs, V100 GPU) vs. 0.08 fps for NeRF-based methods, which is **3475x faster**.

### Ablation Study

| Setting | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| w/o alignment | 19.06 | 0.679 | 0.217 |
| w/o implicit decoder | 16.68 | 0.477 | 0.331 |
| w/o tv reg. | 20.20 | 0.724 | 0.186 |
| w/o flow reg. | 20.32 | 0.723 | 0.185 |
| w/o multisampling | 19.99 | 0.718 | 0.194 |
| **Full Model** | **20.33** | **0.725** | **0.180** |

### Key Findings

- **Implicit decoder is the most critical component**: Removing it drops PSNR by 3.65 and degrades LPIPS by 0.151, demonstrating that the structured constraint is vital for sparse inputs.
- Depth alignment initialization is crucial for complex scenes (PSNR drops by 1.27 without it).
- The advantage in LPIPS is the most prominent, signifying an overwhelming advantage in perceptual quality.
- Occluded regions can be identified and inpainted using diffusion models, which is an additional benefit of constraining Gaussian movement.
- The advantage is even more pronounced with 2 input views (LPIPS: 0.220 vs. 0.376 for the second best), proving the robustness of the method under extremely sparse conditions.

## Highlights & Insights

1. **The core idea of "Structuring the Unstructured"**: By utilizing one Gaussian per pixel combined with ray constraints, the uncontrollable Gaussian motion in 3D space is transformed into a scalar depth optimization problem in 2D image space, representing an extremely elegant design.
2. **Clever utilization of the implicit decoder**: Leveraging the implicit smoothness of the network to replace explicit regularization successfully overcomes the core challenge of 3DGS's unstructured representation.
3. **Occluded region identification capability**: As a byproduct of constraining Gaussian motion, the unreconstructed areas correspond precisely to the occluded regions from all views, allowing them to be identified and recovered.
4. **Multisampling strategy**: Simple yet effective, it resolves the semi-transparency issue caused by Gaussians only covering pixel centers under sparse inputs.
5. **Progressive TV strategy**: Starting with global smoothness and then recovering details is a good practice for coarse-to-fine structured optimization.

## Limitations & Future Work

1. **Single Gaussian per pixel limitation**: Unable to handle transparent/semi-transparent objects (such as glass) because there is only one Gaussian along each ray.
2. Dependence on the quality of monocular depth; results are limited if the depth estimation is severely inaccurate.
3. The modeling of Gaussians in occluded regions is not considered. Although they can be patched via diffusion models, this increases the pipeline complexity.
4. Future work could explore adaptive multi-Gaussian-per-pixel strategies to handle scenes with complex depth configurations.
5. Semantic information could be integrated to guide region-based depth segmentation (currently simply divided into C=5 regions based on depth values).

## Related Work & Insights

- Comparison with sparse-view NeRF methods shows that the intrinsic smoothness of implicit representations is a double-edged sword: it helps propagate regularization but sacrifices rendering speed and quality.
- The implicit decoder concept originates from dynamic scene reconstruction methods like D-NeRF, but here it is utilized for regularization rather than modeling deformation.
- The initialization strategy combining optical flow and depth alignment could be generalized to other tasks requiring multi-view alignment.
- The application potential of monocular depth estimation (Depth Anything) in 3D reconstruction: High-quality relative depth + global alignment > Low-quality multi-view stereo matching.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combined design of structured Gaussians + implicit decoder is novel, though the individual components are not pioneered here)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, multiple input configurations, detailed ablations, visual and quantitative comparisons)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, well-explained motivation, information-rich figures and tables)
- Value: ⭐⭐⭐⭐ (Sparse-view 3DGS has high practical value, and the 278fps inference speed offers great application prospects)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis](thermal3d-gs_physics-induced_3d_gaussians_for_thermal_infrared_novel-view_synthe.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](../../CVPR2025/3d_vision/comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[ECCV 2024\] Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This paper proposes CoCoGaussian, which utilizes physical photographic defocus principles (Circle of Confusion) to model defocus blur within the 3D Gaussian Splatting framework, enabling accurate 3D scene reconstruction and sharp novel view rendering using only defocused images.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "defocus blur"
  - "Circle of Confusion"
  - "novel view synthesis"
  - "depth estimation"
date: 2026-05-08
content_hash: fdffbc22e4570ed8
---

# CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images

**Conference**: CVPR 2025  
**arXiv**: [2412.16028](https://arxiv.org/abs/2412.16028)  
**Code**: [https://Jho-Yonsei.github.io/CoCoGaussian](https://Jho-Yonsei.github.io/CoCoGaussian)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, defocus blur, Circle of Confusion, novel view synthesis, depth estimation

## TL;DR

This paper proposes CoCoGaussian, which utilizes physical photographic defocus principles (Circle of Confusion) to model defocus blur within the 3D Gaussian Splatting framework, enabling accurate 3D scene reconstruction and sharp novel view rendering using only defocused images.

## Background & Motivation

In real-world photography, defocus blur caused by the camera's limited depth of field (DoF) is unavoidable—small apertures require longer exposure times (triggering motion blur), while large apertures result in shallow DoF defocus. Existing 3DGS and NeRF methods assume sharp input images, which is unrealistic in real-world scenarios. Most existing methods for handling blurry images (e.g., Deblur-NeRF, BAGS) rely heavily on pure learning strategies while ignoring physical photographic principles. While DoF-NeRF introduces the CoC concept, it is based on implicit representations and suffers from slow rendering speeds. The core motivation of this paper is to **combine the physical Circle of Confusion (CoC) principles with the explicit representation of 3DGS**, accurately modeling defocus effects while maintaining real-time rendering speeds.

## Method

### Overall Architecture

Based on the 3DGS framework, CoCoGaussian generates $M$ groups of CoC Gaussians ($\mathbf{G}_{CoC}$) for each base Gaussian ($\mathbf{G}_B$), yielding a total of $(M+1)$ groups of Gaussians. An MLP $h_\theta$ is employed to predict the aperture parameter $K$, direction vector $\mathbf{d}$, scaling factor $\beta$, and scale/rotation offsets $\delta\mathbf{s}_{CoC}$ and $\delta\mathbf{q}_{CoC}$. Finally, the $(M+1)$ images rendered separately from all Gaussian groups are weighted and summed using pixel-level weights calculated by a shallow CNN, producing the final defocused image.

### Key Designs

1. **Based on 3D Gaussian CoC Diameter Calculation**:
    - **Function**: Calibrating the circle of confusion diameter from Gaussian depth and learnable aperture information.
    - **Mechanism**: Deriving a simplified CoC diameter formula: $\sigma(\mu_B) \approx K \times |1/d(\mu_B) - 1/d_F|$, where $K = f \times D$ is a learnable scalar (focal length $\times$ aperture diameter), $d_F$ is the learnable focal plane distance, and $d(\mu_B)$ is the Euclidean distance from the camera to the Gaussian mean.
    - **Design Motivation**: Utilizing physical photographic priors to constrain the CoC size, avoiding overfitting associated with pure learning methods. Making $K$ and $d_F$ learnable allows the model to adaptively fit different scenarios.

2. **Adaptive CoC Gaussian Generation**:
    - **Function**: Generating CoC-shaped Gaussians surrounding the base Gaussian and handling unreliable depths.
    - **Mechanism**: Introducing a learnable scaling factor $\beta \in (0,1]$ to correct the offset as $\Delta\mu_{CoC;m} = \frac{\sigma(\mu_B)}{2}\beta_m \mathbf{d}_m$. Here, $\beta$ ensures that the CoC Gaussians remain inside the CoC boundary, while adaptively shrinking them when inaccurate depths lead to oversized CoC estimations.
    - **Design Motivation**: Refractive/reflective surfaces in a scene can cause unreliable Gaussian depths, which makes calculations solely dependent on depth prone to errors. Specifying $\beta$ decreases this strong dependency on depth.

3. **Weighted Summation Rendering & Customizable DoF**:
    - **Function**: Blending multiple sets of rendered images into the final defocused image and supporting runtime adjustments of depth of field and the focal plane.
    - **Mechanism**: A shallow CNN $\mathcal{F}$ computes softmax-normalized pixel-level weights $\mathcal{W}$ across $(M+1)$ images, which are then used in a weighted sum to yield the final image. Because the model learns $K$ and $d_F$, adjusting these parameters at render time achieves various depth-of-field effects.
    - **Design Motivation**: Inspired by blind deblurring strategies, the CoC Gaussian groups are treated as defocus kernels. Having a customizable DoF is highly valuable for AR/VR applications.

### Loss & Training

The loss function is a weighted combination of standard L1 loss and D-SSIM loss:

$$\mathcal{L}_{rgb} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D\text{-}SSIM}$$

where $\lambda = 0.3$. During training, COLMAP is used to obtain camera poses and initial point clouds. The focal plane $d_F$ is initialized as the average distance from the camera to the SfM point cloud to ensure stable training.

## Key Experimental Results

### Main Results

| Dataset | Metric | CoCoGaussian | BAGS (Prev. SOTA) | Gain |
|--------|------|-------------|-----------------|------|
| Deblur-NeRF Synthetic | PSNR/SSIM/LPIPS | 30.84/0.9212/0.0478 | 30.65/0.9128/0.0631 | +0.19/+0.008/-0.015 |
| Deblur-NeRF Real | PSNR/SSIM/LPIPS | 23.70/0.7531/0.0825 | 23.48/0.7408/0.0962 | +0.22/+0.012/-0.014 |
| DoF-NeRF Real | PSNR/SSIM/LPIPS | 30.14/0.9127/0.0701 | 29.87/0.8816/0.1100 | +0.27/+0.031/-0.040 |

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS | Description |
|------|------|------|-------|------|
| Baseline (3DGS) | 25.72 | 0.8291 | 0.1817 | No defocus handling |
| w/o CoC | 28.29 | 0.8778 | 0.0927 | Without physical CoC constraints, overfitting occurs |
| w/o CoC Direction Vector | 28.91 | 0.8893 | 0.0896 | Restricted by fixed directions |
| w/o CoC Scale Factor | 29.46 | 0.9058 | 0.0793 | Over-reliance on depth |
| w/o Aperture Parameter K | 27.37 | 0.8510 | 0.1113 | Unable to accurately compute CoC size |
| **Full Model** | **30.14** | **0.9127** | **0.0701** | All components working together |

### Key Findings

- The physical CoC prior is essential: removing CoC drops PSNR by 1.85, demonstrating that relying solely on parameter learning leads to overfitting.
- Aperture parameter $K$ contributes the most: removing it drops PSNR by 2.77, indicating it is critical for determining CoC size.
- It also outperforms the baseline 3DGS on all-in-focus images (NeRF-LLFF) (PSNR 27.76 vs 27.10), indicating that modeling even minuscule CoC values benefits representation accuracy.

## Highlights & Insights

- Elegant combination of **physical priors + learning**: constraining the framework with photographic optics formulas while leveraging learning to compensate for uncertainty.
- The **depth of field and focal plane of the scene can be freely adjusted at render time**, enabling creative control capabilities not present in traditional 3DGS.
- The method is also effective on all-in-focus images, demonstrating strong generalization.

## Limitations & Future Work

- The adaptive scaling factor $\beta$ currently only handles scenarios where the CoC diameter is overestimated ($\beta \leq 1$), leaving underestimations uncompensated.
- The model generates $M \times N$ additional Gaussians, increasing the computational overhead of training and rendering.
- The Deblur-NeRF real-world dataset exhibits illumination discrepancies between sharp and defocused images, which affects evaluation reliability.

## Related Work & Insights

- Unlike learning-based deblurring methods such as Deblur-NeRF/BAGS, this work **embeds physical optical priors directly into 3DGS**, serving as a prime example of a physical + learning hybrid paradigm.
- Although DoF-NeRF also uses CoC, its reliance on implicit representations leads to slow speeds. CoCoGaussian achieves real-time rendering while maintaining physical plausibility.
- The ability to customize depth of field offers straightforward application value for AR/VR content creation.

## Rating

- Novelty: ⭐⭐⭐⭐ Integrating physical CoC modeling into 3DGS is an interesting concept, though its core lies in applying known optical formulas.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across three datasets, featuring complete ablation studies, all-in-focus generalization experiments, and customizable DoF demonstrations.
- Writing Quality: ⭐⭐⭐⭐ The physics principles are explained with clarity, and the mathematical derivations are fully fleshed out.
- Value: ⭐⭐⭐⭐ Successfully addresses the 3D reconstruction problem from defocused images in real scenes while offering custom depth-of-field control.

## Related Papers

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[CVPR 2025\] Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries](blurry-edges_photon-limited_depth_estimation_from_defocused_boundaries.md)
- [\[CVPR 2025\] Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[CVPR 2025\] Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries](blurry-edges_photon-limited_depth_estimation_from_defocused_boundaries.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

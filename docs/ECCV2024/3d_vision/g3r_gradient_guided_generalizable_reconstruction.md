---
title: >-
  [Paper Note] G3R: Gradient Guided Generalizable Reconstruction
description: >-
  [ECCV 2024][3D Vision][Generalizable 3D Reconstruction] G3R is proposed as a gradient-guided generalizable reconstruction method. It iteratively updates a 3D Neural Gaussian representation using 3D gradient feedback from differentiable rendering via a learned reconstruction network. It achieves reconstruction of large-scale scenes (>10,000m²) in under 2 minutes, accelerating the process by at least 10x while maintaining comparable or superior rendering quality compared to 3DG…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Generalizable 3D Reconstruction"
  - "3D Gaussian Splatting"
  - "Gradient Guided"
  - "Learning-based Optimizer"
  - "Large-scale Scene Neural Rendering"
date: 2026-05-08
content_hash: 845fc7eab558b457
---

# G3R: Gradient Guided Generalizable Reconstruction

**Conference**: ECCV 2024  
**arXiv**: [2409.19405](https://arxiv.org/abs/2409.19405)  
**Code**: [https://waabi.ai/g3r](https://waabi.ai/g3r) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Generalizable 3D Reconstruction, 3D Gaussian Splatting, Gradient Guided, Learning-based Optimizer, Large-scale Scene Neural Rendering

## TL;DR

G3R is proposed as a gradient-guided generalizable reconstruction method. It iteratively updates a 3D Neural Gaussian representation using 3D gradient feedback from differentiable rendering via a learned reconstruction network. It achieves reconstruction of large-scale scenes (>10,000m²) in under 2 minutes, accelerating the process by at least 10x while maintaining comparable or superior rendering quality compared to 3DGS.

## Background & Motivation

**Background**: Neural rendering methods such as NeRF and 3DGS achieve high-quality rendering for large-scale scene reconstruction, but require time-consuming per-scene optimization (taking hours). Generalizable methods (e.g., Generalizable NVS, Large Reconstruction Models) are fast but mainly suitable for small scenes/objects and yield lower rendering quality.

**Limitations of Prior Work**: Per-scene optimization methods (e.g., 3DGS) are time-consuming and prone to overfitting to source views, leading to artifacts under large viewpoint changes. Generalizable methods typically extract features from a small number of source views ($\le$ 5) for feedforward prediction; they are limited by network capacity and memory overhead, making them unable to handle hundreds of high-resolution input images required for large-scale scenes.

**Key Challenge**: Per-scene optimization methods produce high-quality rendering but are slow and lack generalization capability; feedforward prediction methods are fast but suffer from low quality and struggle to scale to large scenes. How to combine the strengths of both?

**Goal**: Achieve the first generalizable fast 3D reconstruction method for large-scale scenes, generating editable 3D representations for scenes >10,000m² within 2 minutes, while guaranteeing high-quality novel-view rendering.

**Key Insight**: Leverage gradients from differentiable rendering as an information bridge between 2D and 3D. A "learned optimizer" network is trained to iteratively update the 3D representation using gradient feedback, replacing traditional gradient descent optimization.

**Core Idea**: Render $\rightarrow$ compute gradients $\rightarrow$ use a neural network instead of traditional gradient descent to interpret gradients and update the 3D representation—learning how to optimize is faster and better than direct optimization.

## Method

### Overall Architecture

The G3R paradigm (Fig. 2c) sits between generalizable methods and per-scene optimization. Given source images and initial 3D points (from LiDAR/MVS), the 3D Neural Gaussians $\mathcal{S}^{(0)}$ are initialized, followed by $T$ iterative steps. In each step, differentiable rendering is performed first $\rightarrow$ reconstruction loss is computed $\rightarrow$ backpropagation yields 3D gradients $\nabla_{\mathcal{S}^{(t)}}$ $\rightarrow$ the reconstruction network $G_\theta$ takes gradients and the current representation $\rightarrow$ representation updates are predicted $\rightarrow$ yielding $\mathcal{S}^{(t+1)}$. Only 24 iteration steps are required to match or even surpass the performance of 3DGS optimization with thousands of steps.

### Key Designs

1. **3D Neural Gaussians**: Based on the standard 3DGS representation $g_i \in \mathbb{R}^{14}$ (position 3 + scale 3 + rotation 4 + color 3 + opacity 1), each Gaussian point is augmented with a latent feature vector $h_i \in \mathbb{R}^C$. A MLP then decodes $g_i = f_{\text{mlp}}(h_i)$ to transform them into standard Gaussians for rendering. The first 14 channels are fixed as standard attributes with skip connections to ensure stable optimization. The scene is decomposed into a static background $\mathcal{S}^{\mathcal{B}}$, dynamic objects $\mathcal{S}^{\mathcal{A}}$ (assuming rigid body motion $\mathcal{T}(\mathcal{S}^{\mathcal{A}}, \boldsymbol{\xi}^{\mathcal{A}})$), and far-field regions $\mathcal{S}^{\mathcal{Y}}$. The rendering formula is $f_{\text{render}}(\mathcal{S}; \Pi) = f_{\text{rast}}(f_{\text{mlp}}(\mathcal{S}^{\mathcal{B}}, \mathcal{S}^{\mathcal{Y}}, \mathcal{T}(\mathcal{S}^{\mathcal{A}}, \boldsymbol{\xi}^{\mathcal{A}})); \Pi)$. **Design Motivation**: Latent features grant the network extra expressive capacity, allowing it to store implicit information about the scene during iterative updates.

2. **Gradient-Guided 2D-to-3D Lifting**: Unlike traditional methods that process source images individually, G3R aggregates all 2D image information into the 3D space via "rendering + backpropagation". Specifically, the current 3D representation is rendered onto all source views $\hat{\mathbf{I}}^{\text{src}} = f_{\text{render}}(\mathcal{S}; \Pi^{\text{src}})$, and the reconstruction loss is computed as $L = \sum_i \| \mathbf{I}_i^{\text{src}} - f_{\text{render}}(\mathcal{S}; \Pi_i^{\text{src}}) \|_2$. Backpropagation then produces the 3D gradients $\nabla_\mathcal{S} = \frac{\partial L}{\partial \mathcal{S}}$. These 3D gradients are naturally occlusion-aware, can efficiently aggregate information from an arbitrary number of images, and leverage the current 3D representation as a proxy. **Design Motivation**: Traditional methods process each image independently using networks, where memory scales linearly with the number of source images. In contrast, the gradient-based method naturally aggregates information through backpropagation, decoupling the process from the number of images.

3. **Iterative Reconstruction Network (G3R-Net)**: The core update formula is $\mathcal{S}^{(t+1)} = \mathcal{S}^{(t)} + \gamma(t) \cdot G_\theta(\mathcal{S}^{(t)}, \nabla_{\mathcal{S}^{(t)}} L; t)$, where $\gamma(t)$ is scheduled using a DDIM cosine decay. The network $G_\theta$ employs a 3D UNet with sparse convolutions to process the Neural Gaussians. Gradients are normalized channel-wise before input. For dynamic scenes, three independent networks are used to handle the background, objects, and far-field, respectively. During training, $T=24$ steps are used, training on approximately 1000 scene iterations. **Design Motivation**: The iterative mechanism enables the network to predict a coarse representation first and then progressively refine it, reducing the difficulty of single-step prediction through progressive updates.

### Loss & Training

- **Total Loss**: $\mathcal{L} = \mathcal{L}_{\text{mse}}(\hat{\mathbf{I}}, \mathbf{I}) + \lambda_{\text{lpips}} \mathcal{L}_{\text{lpips}}(\hat{\mathbf{I}}, \mathbf{I}) + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}(\mathcal{G})$
- **Regularization**: $\mathcal{L}_{\text{reg}}(\mathcal{G}) = \sum_i \max(0, d_i^{\min} - \epsilon)$, which encourages the minimum scale channel of each Gaussian to be flat, encouraging alignment with the surface.
- **Key Training Strategies**: After each step update, rendering is performed on both source **and novel views** to compute loss, but only the gradients from source views are used as input for the next step of $G_\theta$ (since novel views are unavailable during testing). Novel-view supervision helps the network learn more generalizable 3D representations rather than overfitting to the source views.
- Trained for approximately 30 hours on 2x RTX 3090. A warm-up strategy progressively increases the number of iteration steps.

## Key Experimental Results

### Main Results: PandaSet (Driving Scene)

| Model | Type | PSNR↑ | SSIM↑ | LPIPS↓ | Recon. Time | Rendering FPS |
|---|---|---|---|---|---|---|
| ENeRF | Generalizable | 24.43 | 0.736 | 0.306 | 0.057s/view | 6.93 |
| PixelSplat | Generalizable | 23.21 | 0.653 | 0.490 | 0.74s/view | 147 |
| Instant-NGP | Per-scene | 24.34 | 0.729 | 0.436 | 7min 16s | 3.24 |
| 3DGS | Per-scene | 25.14 | 0.747 | 0.372 | 50min 14s | 121 |
| **G3R (turbo)** | **ours** | 24.76 | 0.720 | 0.438 | **31s** | 121 |
| **G3R** | **ours** | **25.22** | 0.742 | **0.371** | **123s** | 121 |

### Main Results: BlendedMVS (Large-scale UAV Scenes, Larger Viewpoint Variations)

| Model | Type | PSNR↑ | SSIM↑ | LPIPS↓ | Recon. Time | Rendering FPS |
|---|---|---|---|---|---|---|
| PixelSplat++ | Generalizable | 19.60 | 0.404 | 0.601 | 69s | 158 |
| Instant-NGP | Per-scene | 24.86 | 0.639 | 0.459 | 26min 48s | 1.65 |
| 3DGS | Per-scene | 25.12 | 0.668 | 0.462 | 39.5min | 97.0 |
| **G3R (turbo)** | **ours** | 24.56 | 0.674 | 0.421 | **98s** | 97.0 |
| **G3R** | **ours** | **25.22** | **0.707** | **0.390** | **210s** | 97.0 |

### Ablation Study (PandaSet)

| Configuration | PSNR | SSIM | LPIPS | Description |
|------|------|------|-------|------|
| G3R (Full) | **25.22** | **0.742** | **0.371** | Baseline |
| −3D Neural Gaussian representation | 24.72 | 0.718 | 0.420 | Using standard 3DGS representation, latent features are highly important |
| −Iterative reconstruction | 20.03 | 0.510 | 0.623 | Single-step prediction is highly insufficient |
| −Novel-view training | 24.59 | 0.715 | 0.419 | Source views only $\rightarrow$ overfitting; novel-view supervision is key to generalization |
| −Update schedule $\gamma(t)$ | 25.03 | 0.732 | 0.400 | Constant update $\rightarrow$ performance drops |

### Cross-dataset Generalization (PandaSet $\rightarrow$ BlendedMVS)

| Configuration | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| Zero-shot transfer | 24.11 | 0.653 | 0.448 |
| Fine-tuned on 2 scenes | 24.99 | 0.676 | 0.428 |

### Key Findings

- **Speed Advantage**: G3R requires only 2 minutes to complete reconstruction, achieving a 25x speedup compared to 3DGS (50 minutes) and 3.5x compared to Instant-NGP (7 minutes).
- **Iteration is Key**: Disabling iteration (single-step prediction) leads to a drastic drop of 5.2dB in PSNR, validating the necessity of "progressive refinement" in the learning-based optimizer paradigm.
- **Gradients > Feature Projection**: 3D gradients as a 2D-to-3D bridge outperform traditional feature projection, as they are naturally occlusion-aware and can aggregate information from an arbitrary number of images.
- **Strong Generalizability**: A model trained on PandaSet achieves zero-shot transfer to BlendedMVS (driving $\rightarrow$ UAV) and already outperforms all generalizable baselines trained on BlendedMVS.
- **Robustness**: G3R significantly outperforms 3DGS in large viewpoint extrapolation tests (where 3DGS suffers from black holes and incorrect colors) because novel-view training supervision prevents overfitting.

## Highlights & Insights

- **"Learning to Optimize" Paradigm**: Replacing a traditional optimizer (such as Adam) with a neural network. The network utilizes not only gradient direction information but also spatial correlation priors learned across scenes—this represents a successful application of meta-learning in large-scale 3D reconstruction.
- **Gradient as Representation**: Cleverly reinterpreting gradients from differentiable rendering as information carriers for 2D-to-3D lifting, rather than merely optimization signals. Gradients naturally aggregate information from all source views while accounting for occlusions.
- **Scene Decomposition**: Decomposing dynamic scenes into background + objects + far-field, which are processed by independent networks. Combined with rigid body motion assumptions, this allows the method to handle complex dynamics in real-world driving scenes.
- **High Practicality**: The reconstructed output is in standard 3D Gaussian format, which can be directly used for real-time rendering (>90 FPS), scene editing, and multi-camera simulation.

## Limitations & Future Work

- Large-range extrapolation still exhibits artifacts, which may require scene completion or better surface regularization.
- Performance relies on the density of the initialization point cloud; performance degrades with sparse point initialization (though this can be mitigated by LiDAR or fast MVS).
- Non-rigid deformations (e.g., pedestrian movement) and lighting effects are not modeled.
- View-dependent spherical harmonics are disabled to reduce memory overhead, which may limit the rendering quality of specular highlights.
- The learning-based optimizer paradigm can be extended to other 3D inverse problems, such as TriPlane + NeRF rendering or generalizable surface reconstruction.

## Related Work & Insights

- **3DGS**: G3R can be viewed as "accelerating 3DGS with a learned optimizer", where the reconstruction network takes over the role of the Adam optimizer.
- **PixelSplat**: A representative feedforward 3D Gaussian prediction method, but limited by low-resolution image pairs and small-scale scenes.
- **Learning to Optimize (Andrychowicz et al., 2016)**: G3R directly inherits the idea of "learning to gradient descent by gradient descent".
- **DeepView**: Also uses an iterative network + gradient-guided reconstruction for MPI, but employs different CNNs in each step and can only handle a small number of images.
- **Insight**: The idea of using gradient feedback signals as a 2D-3D bridge is highly versatile and can be applied to any scenario requiring 3D reconstruction from a large number of 2D observations.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The paradigm of gradient-guided learning-based optimizers is pioneering and highly inspiring in the field of large-scale 3D reconstruction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broadly covers both driving and UAV large-scale scenes, with comprehensive ablation studies, cross-dataset generalization, and scene editing demonstrations.
- Writing Quality: ⭐⭐⭐⭐⭐ The comparison diagram of the three paradigms (Fig. 2) is extremely clear, the formulation is mathematically rigorous, and the arguments are progressively articulated.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm in large-scale scene reconstruction, with direct application value for autonomous driving simulation and digital twins.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)
- [\[CVPR 2026\] Energy-GS: Image Energy-guided Pose Alignment Gaussian Splatting with redesigned pose gradient flow](../../CVPR2026/3d_vision/energy-gs_image_energy-guided_pose_alignment_gaussian_splatting_with_redesigned_.md)
- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](../../CVPR2025/3d_vision/depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)

</div>

<!-- RELATED:END -->

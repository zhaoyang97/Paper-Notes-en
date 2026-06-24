---
title: >-
  [Paper Note] Dynamic Neural Radiance Field from Defocused Monocular Video
description: >-
  [ECCV 2024][3D Vision][Dynamic NeRF] Proposed $D^2RF$, the first method to recover sharp dynamic NeRFs from defocused monocular videos, which unifies Depth of Field (DoF) rendering with volume rendering and introduces layered DoF volume rendering to model defocus blur and recover sharp novel views.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Dynamic NeRF"
  - "Defocus Blur"
  - "Depth of Field Rendering"
  - "Volume Rendering"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 07a6adc9e7527dd0
---

# Dynamic Neural Radiance Field from Defocused Monocular Video

**Conference**: ECCV 2024  
**arXiv**: [2407.05586](https://arxiv.org/abs/2407.05586)  
**Code**: [Yes](https://github.com/xianrui-luo/D2RF)  
**Area**: 3D Vision  
**Keywords**: Dynamic NeRF, Defocus Blur, Depth of Field Rendering, Volume Rendering, Novel View Synthesis

## TL;DR

Proposed $D^2RF$, the first method to recover sharp dynamic NeRFs from defocused monocular videos, which unifies Depth of Field (DoF) rendering with volume rendering and introduces layered DoF volume rendering to model defocus blur and recover sharp novel views.

## Background & Motivation

### Background
Dynamic NeRFs have achieved outstanding results in spatio-temporal novel view synthesis from monocular videos. Existing methods (NSFF, HyperNeRF, DVS, etc.) reconstruct high-quality dynamic scenes by establishing spatio-temporal consistency.

### Limitations of Prior Work
Existing dynamic NeRF methods assume that the input is an all-in-focus image sequence. However, in real video capture, **defocus blur** is almost inevitable due to dynamic scene depth changes, large aperture settings, and unstable focusing by the camera operator. Defocus blur leads to:
1. A lack of sharp details, which hinders dynamic object motion modeling.
2. Inability to establish temporal consistency among input views.
3. Severe degradation of current dynamic NeRF methods under defocused inputs, rendering them unable to recover sharp content.

### Limitations of Prior Work
- Static defocus methods (Deblur-NeRF, DP-NeRF, DoF-NeRF): Target only static multi-view inputs and cannot handle dynamic scenes.
- These methods model blur in the final step of NeRF (after volume rendering) — serving as a post-processing step.
- 2D deblurring + dynamic NeRF pipeline: Deblurring individual views lacks consistency across views, leading to unstable performance.

### Key Insight
The authors found that **layer visibility in DoF rendering shares the same physical meaning as opacity in volume rendering** — both describe the degree to which light is absorbed or blocked at a certain point. Based on this, the DoF blurring process can be seamlessly embedded into the volume rendering pipeline of NeRF, rather than being treated as a post-processing step.

## Method

### Overall Architecture

$D^2RF$ takes a defocused monocular video as input and outputs a sharp dynamic scene NeRF representation. The core workflow is:
1. Define a blur-kernel template $\rightarrow$ Predict optimized sparse rays and weights using an MLP.
2. Feed the rays into static MLP $G_\theta^{st}$ and dynamic MLP $G_\theta^{dy}$ respectively to model the scene.
3. Fuse blur through layered DoF volume rendering and supervise the training.
4. During rendering, directly render rays (without DoF blurring) to output sharp novel views.

### Key Designs

#### 1. Layered DoF Volume Rendering

**Function**: Elevates the defocus blur modeling process from post-processing into the volume rendering stage to achieve blur-aware NeRF training.

**Mechanism**: The alpha value $\alpha_i = 1 - \exp(-\sigma\delta_i)$ of a sampling point in volume rendering represents its opacity, which shares the same physical meaning as the layer visibility $W_i$ in DoF rendering. Furthermore, NeRF's discrete sampling is naturally compatible with the layer discretization of DoF rendering.

Converting the traditional DoF rendering formula from the image level to the single-pixel/single-ray level, the layered DoF volume rendering formula is proposed:

$$\hat{C}_{dof}(\mathbf{r}_p) = \frac{\sum_{i=1}^{k} (T_i * K(\mathbf{r}))(1-\exp(-\sigma\delta_i)) \mathbf{c}(\mathbf{r}(t_i), \mathbf{d}) * K(\mathbf{r})}{\sum_{i=1}^{k} (T_i * K(\mathbf{r}))(1-\exp(-\sigma\delta_i)) * K(\mathbf{r})}$$

where $T_i$ is the accumulated transmittance and $K(\mathbf{r})$ is the ray-level blur kernel.

**Design Motivation**: Integrates blur modeling deeply into the volume rendering sampling process instead of post-hoc blurring, enabling the NeRF network to learn sharp scene representations from defocused inputs.

#### 2. Ray-based Optimized Sparse Kernel

**Function**: Converts the layer-level kernel $K(\gamma_i)$ of DoF rendering into a ray-level kernel $K(\mathbf{r})$, and replaces dense kernels with sparse points to reduce computational cost.

**Mechanism**: Uses an MLP $G_\theta^k$ to predict the offsets and weights of kernel points:

$$(\Delta\mathbf{j}, g_j) = G_\theta^k((u,v), \mathbf{j}, t_l)$$

where $(u,v)$ are the planar coordinates of the kernel center, $\mathbf{j}$ is the original ray of the kernel template, and $t_l$ is the time embedding. The final optimized ray is $\mathbf{r}_j = \mathbf{j} + \Delta\mathbf{j}$.

Sparse kernel convolution: $\mathbf{b}_p = \sum_{j \in S(p)} \mathbf{c}_j g_j$, with the constraint $\sum g_j = 1$ to ensure color consistency.

**Design Motivation**: DoF rendering is originally image-based, while NeRF is based on ray sampling, requiring the kernel to be converted from layer-level to ray-level. The sparse kernel (5 points, radius 10) significantly reduces computational overhead, while deformable learning adapts to real-world spatially-varying blur.

#### 3. Dynamic-Static Scene Fusion and Cross-Time Rendering

**Function**: Models static and dynamic scenes with two independent MLPs and establishes temporal consistency through cross-time rendering.

- **Static MLP** $G_\theta^{st}$: Outputs color $\mathbf{c}$, density $\sigma$, and blending weight $\eta$.
- **Dynamic MLP** $G_\theta^{dy}$: Outputs color $\mathbf{c}_t$, density $\sigma_t$, scene flow $f_t$, and occlusion weight $\mathcal{W}_t$.
- Fusion rendering weights static and dynamic colors via $\eta(t)$.
- **Cross-time rendering**: Translates sampling points from adjacent frames to the target frame using scene flow, computes cross-time color through layered DoF volume rendering, and establishes temporal consistency.

### Loss & Training

The total loss consists of:
- **Blending rendering loss** $\mathcal{L}_{color}^b$: L2 loss between the fused result and ground truth (GT).
- **Dynamic rendering loss** $\mathcal{L}_{color}^t$: Constrains the standalone dynamic rendering result.
- **Cross-time loss** $\mathcal{L}_{cross}$: Weighted L2 loss between warped rendering of adjacent frames and GT, where weights are controlled by occlusion confidence.
- **Data prior loss** $\mathcal{L}_{data}$: Scale-invariant monocular depth loss + scene flow consistency and L1 regularization.

Training details: Adam optimizer, learning rate $5 \times 10^{-4}$, 250k iterations per scene, taking about two days on a single RTX 3090 GPU. COLMAP is used to estimate camera parameters, and RAFT and DPT provide optical flow and depth priors.

## Key Experimental Results

### Datasets
Collected 8 dynamic scenes from the VDW stereo dataset, using the BokehMe pipeline to synthesize defocus blur. The image resolution is $940 \times 360$, and the focal length changes progressively with scene parallax to simulate real focusing. Defocused images from the left view are used for training, while sharp images from the right view are used for evaluation.

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| DVS [Gao et al.] | 25.43 | 0.764 | 0.242 |
| RoDynRF [Liu et al.] | 26.18 | 0.770 | 0.227 |
| HyperNeRF [Park et al.] | 26.96 | 0.780 | 0.208 |
| NSFF [Li et al.] | 27.01 | 0.803 | 0.209 |
| [Lee]+RoDynRF (2D Deblurring) | 25.79 | 0.776 | 0.196 |
| [Lee]+DVS (2D Deblurring) | 24.52 | 0.757 | 0.208 |
| **$D^2RF$ (Ours)** | **27.30** | **0.816** | **0.130** |

$D^2RF$ outperforms all methods across all metrics, with a particularly significant improvement in LPIPS (0.130 vs. 0.196 for the runner-up), indicating greatly improved perceptual quality. Pre-processing with 2D deblurring can introduce multi-view inconsistencies and degrade performance.

### Ablation Study

| Configuration | PSNR↑ (Full Image) | SSIM↑ | LPIPS↓ | Explanation |
|------|-------------|-------|--------|------|
| w/o cross-time | 22.61 | 0.725 | 0.232 | Cross-time rendering is critical for temporal consistency |
| w/o layered volume | 27.11 | 0.811 | 0.211 | Layered volume rendering delivers more precise blur modeling |
| w/o optimized kernel | 27.25 | 0.795 | 0.216 | Optimized kernel provides an efficient blur fitting pipeline |
| w/o static | 26.20 | 0.769 | 0.177 | Independent static representation stabilizes training |
| **Full (Ours)** | **27.30** | **0.816** | **0.130** | All components synergize to achieve the best performance |

### Key Findings

1. Removing cross-time rendering causes a sharp PSNR drop of 4.7dB (in dynamic regions), indicating that temporal consistency is crucial for dynamic scenes.
2. Layered DoF volume rendering is more accurate than post-processing blur modeling, reducing LPIPS from 0.211 to 0.130.
3. Modeling defocus blur in 3D space (as in this work) performs significantly better than a two-stage pipeline of 2D frame-by-frame deblurring + dynamic NeRF.
4. 2D deblurring preprocessing can sometimes degrade performance (RoDynRF + deblurring < original RoDynRF) because independent frame deblurring disrupts multi-view consistency.

## Highlights & Insights

1. **Elegant Theoretical Link**: Discovered the equivalence between layer visibility in DoF rendering and opacity in volume rendering, naturally unifying the two rendering frameworks.
2. **From Post-Processing to Embedded**: Shifting blur modeling from post-rendering to the rendering process itself represents a paradigmatic improvement.
3. **Problem Definition Value**: First to define and solve the practical and critical problem of "defocused dynamic NeRF".

## Limitations & Future Work

1. Cannot handle extreme defocus blur.
2. Long training times (about 2 days per scene) and slow inference speed (13 seconds per frame).
3. Evaluated on synthetic datasets; not yet validated on real-world defocused videos.
4. Relies on COLMAP camera parameter estimation, where defocus blur might compromise feature matching accuracy.
5. Future work could explore integration with 3D Gaussian Splatting for acceleration.

## Related Work & Insights

- **Deblur-NeRF** (CVPR 2022): Proposed deformable sparse kernels to model defocus, but limited to static scenes.
- **NSFF** (CVPR 2021): Used scene flow to establish temporal consistency in dynamic NeRFs; this work incorporates blur processing on top of it.
- **BokehMe** (ECCV 2022): High-quality defocus synthesis, utilized in this work for dataset construction.
- Insight: Embedding physical imaging processes into the neural rendering pipeline is an effective paradigm for handling degraded inputs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to unify DoF rendering and NeRF volume rendering, offering a novel problem definition.
- **Experimental Thoroughness**: ⭐⭐⭐ — Thorough ablation studies on an 8-scene synthetic dataset, but lacks validation on real-world data.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear theoretical derivations, intuitive diagrams, with parallel formulation and intuitive explanations.
- **Value**: ⭐⭐⭐⭐ — Addresses defocus blur, which is unavoidable in actual filming, providing high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[ECCV 2024\] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)
- [\[ECCV 2024\] BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Ray-Distance Volume Rendering for Neural Scene Reconstruction
description: >-
  [ECCV 2024][3D Vision][Indoor Scene Reconstruction] The RS-Recon method is proposed, which replaces the traditional SDF with a ray-direction-dependent Signed Ray Distance Function (SRDF) to parameterize the density function in volume rendering. Combined with an SRDF-SDF consistency loss and a self-supervised visibility task, it achieves more accurate surface reconstruction and view synthesis in multi-object indoor scenes.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Indoor Scene Reconstruction"
  - "Neural Implicit Surfaces"
  - "SRDF"
  - "Volume Rendering"
  - "Visibility Prediction"
date: 2026-05-08
content_hash: 59825d676aa42583
---

# Ray-Distance Volume Rendering for Neural Scene Reconstruction

**Conference**: ECCV 2024  
**arXiv**: [2408.15524](https://arxiv.org/abs/2408.15524)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Indoor Scene Reconstruction, Neural Implicit Surfaces, SRDF, Volume Rendering, Visibility Prediction  

## TL;DR

The RS-Recon method is proposed, which replaces the traditional SDF with a ray-direction-dependent Signed Ray Distance Function (SRDF) to parameterize the density function in volume rendering. Combined with an SRDF-SDF consistency loss and a self-supervised visibility task, it achieves more accurate surface reconstruction and view synthesis in multi-object indoor scenes.

## Background & Motivation

- **Background**: NeRF-based neural implicit scene reconstruction methods (e.g., VolSDF, NeuS, MonoSDF) typically parameterize the volume density function using a learnable transformation of SDF, yielding excellent performance in single-object scenes.
- **Limitations of Prior Work**: In multi-object indoor scenes, the SDF values of sampled points along a camera ray can fluctuate due to the influence of neighboring object surfaces, creating multiple local minima and leading to incorrect local maxima and high weights in the density function.
- **Key Challenge**: SDF computes the shortest distance from a point to **all surfaces in the entire scene**, but along the ray direction in volume rendering, only the intersection surface of the ray is truly relevant. Distant surfaces not on the ray should not affect the density distribution of this ray.
- **Goal**: Resolve the issue of SDF generating spurious density peaks in multi-object scenes, enabling the weight distribution of volume rendering to more accurately reflect actual 2D observations.
- **Key Insight**: Introduce ray-dependent SRDF (Signed Ray Distance Function), which computes only the shortest distance from a point to the surface along the ray direction, thereby eliminating interference from surfaces not along the ray.
- **Core Idea**: Model the density function using SRDF to achieve more accurate volume rendering, while describing the 3D surface geometry using SDF, coupling the two via a consistency loss and a visibility task.

## Method

### Overall Architecture

The network of RS-Recon consists of three MLP branches:
1. **Geometry MLP $f_g$**: Predicts the SDF $d_\Omega$ and geometric features $\mathbf{F}_g$ from the encoded position.
2. **SRDF MLP $f_s$**: Predicts the SRDF $\tilde{d}_\Omega$ and visibility probability from geometric features + viewing direction + position.
3. **Color MLP $f_c$**: Predicts color from geometric features + viewing direction + normal.

Volume rendering uses the density function derived from SRDF; surface extraction uses the position-only SDF (via Marching Cubes).

### Key Designs

**Module 1: SRDF Density Function**

The traditional SDF is defined as the shortest distance from a point to all surfaces in the scene:

$$d_\Omega(\mathbf{p}) = (-1)^{\mathbf{1}_\Omega(\mathbf{p})} \min_{\mathbf{p}^* \in \mathcal{M}} \|\mathbf{p} - \mathbf{p}^*\|_2$$

SRDF is defined as the shortest distance from a point to the surface along the ray direction:

$$\tilde{d}_\Omega(\mathbf{p}, \mathbf{r}) = (-1)^{\mathbf{1}_\Omega(\mathbf{p})} \min_{(\mathbf{p}+\rho\mathbf{r}) \in \mathcal{M}; \rho \in \mathbb{R}} |\rho|$$

SRDF is view-dependent, so it only produces density peaks near the intersection of the ray and the surface, without interference from adjacent objects. The density function derived from SRDF is:

$$\sigma^{\text{SRDF}}(\mathbf{p}, \mathbf{r}) = \alpha \Psi_\beta(-\tilde{d}_\Omega(\mathbf{p}, \mathbf{r}))$$

where $\Psi_\beta$ is the cumulative distribution function (CDF) of a Laplace distribution with zero mean and scale $\beta$, and $\alpha, \beta$ are learnable parameters.

**Module 2: SRDF-SDF Consistency Loss**

Although SRDF and SDF have different definitions, their sign meanings are consistent (positive = outside the surface, negative = inside the surface). Since they are predicted by different branches, sign consistency cannot be automatically guaranteed. A sigmoid function is used to approximate the sign function to realize a differentiable constraint:

$$\mathcal{L}_{con} = \frac{1}{N_r} \sum_{\mathbf{p}, \mathbf{r}} M_{con} \|\varsigma(\tilde{d}_\Omega) - \varsigma(d_\Omega)\|_2$$

$$\varsigma(d) = \text{Sigmoid}(k \cdot d), \quad M_{con} = [\tilde{d}_\Omega \cdot d_\Omega < 0]$$

Penalties are only applied to points with sign inconsistency. The gradient of this loss has two advantages: (1) the penalty strength is adjusted according to the degree of inconsistency; (2) the derivative of the sigmoid is largest near zero, providing the strongest supervision for points near the surface.

**Module 3: Self-Supervised Visibility Task**

Along the ray direction, sample points before the first surface intersection are visible, and those after are occluded. The first surface is located by detecting sign changes of SRDF/SDF between adjacent sample points. To reduce noise, information from both SRDF and SDF is used:

$$V_{gt} = \begin{cases} 1, & \text{if } V^{\text{SRDF}}=1 \text{ and } V^{\text{SDF}}=1 \\ 0, & \text{if } V^{\text{SRDF}}=0 \text{ and } V^{\text{SDF}}=0 \end{cases}$$

When the predictions of SRDF and SDF are inconsistent, they do not participate in training. Binary cross-entropy loss $\mathcal{L}_{vis}$ is used to supervise the visibility prediction.

### Loss & Training

Total loss function:

$$\mathcal{L} = \mathcal{L}_c + \lambda_n \mathcal{L}_n + \lambda_d \mathcal{L}_d + \lambda_e \mathcal{L}_e + \lambda_s \mathcal{L}_s + \lambda_{con} \mathcal{L}_{con} + \lambda_{vis} \mathcal{L}_{vis}$$

Includes: RGB loss $\mathcal{L}_c$, normal loss $\mathcal{L}_n$, depth loss $\mathcal{L}_d$, Eikonal loss $\mathcal{L}_e$ (constraining the SDF gradient norm to 1), smoothness loss $\mathcal{L}_s$, consistency loss $\mathcal{L}_{con}$, and visibility loss $\mathcal{L}_{vis}$. During training, colors from both SRDF and SDF densities are rendered to acquire gradient signals for SDF.

## Key Experimental Results

### Main Results

**ScanNet (Real-world Indoor Dataset)**:

| Method | Acc ↓ | Comp ↓ | Prec ↑ | Recall ↑ | F-score ↑ |
|------|-------|--------|--------|----------|-----------|
| MonoSDF_MLP | 0.035 | 0.048 | 0.799 | 0.681 | 0.733 |
| HelixSurf | 0.038 | 0.044 | 0.786 | 0.727 | 0.755 |
| Occ_SDF_Hybrid | 0.040 | 0.041 | 0.783 | 0.748 | 0.765 |
| **Ours_MLP** | **0.040** | **0.040** | **0.809** | **0.779** | **0.794** |

**Replica (Synthetic Indoor Dataset) / Tanks and Temples (Large-scale Real Dataset)**:

| Dataset | Method | Key Metrics |
|--------|------|---------|
| Replica (MLP) | MonoSDF → Ours | F-score: 86.18 → **91.72** |
| Tanks and Temples (Grid) | MonoSDF → Ours | F-score: 6.58 → **7.73** |

**View Synthesis PSNR**:

| Dataset | MonoSDF_MLP | Occ_SDF_Hybrid | **Ours_MLP** |
|--------|------------|----------------|-------------|
| ScanNet | 26.40 | 26.98 | **27.77** |
| Replica | 34.45 | 35.50 | **36.06** |
| Tanks and Temples | 24.13 | 24.72 | **25.47** |

### Ablation Study

Ablation on MLP representation on ScanNet (F-score ↑):

| Configuration | F-score |
|------|---------|
| (a) Baseline (MonoSDF) | 0.733 |
| (b) + SRDF density | 0.745 (+1.2%) |
| (c) + SRDF-SDF consistency loss | 0.776 (+3.1%) |
| (d) + visibility (SDF only) | 0.789 |
| (e) + visibility (SRDF only) | 0.788 |
| **(f) + visibility (SRDF+SDF)** | **0.794** (+6.1%) |

### Key Findings

1. Simply replacing the density function with SRDF (without extra constraints) improves the F-score by 1.2%, verifying the effectiveness of the SRDF density.
2. The SRDF-SDF consistency loss contributes the most (+3.1%), indicating that sign alignment is crucial for the two-branch architecture.
3. Using both SRDF and SDF for visibility labels outperforms using either individually, as the complementary priors help filter out noisy labels.
4. In qualitative analysis, MonoSDF produces spurious surfaces near white walls (since SDF is affected by adjacent surfaces), while the proposed method is more accurate.
5. In rendered images, MonoSDF generates inaccurate colors due to double density peaks, whereas the proposed method's single-peak weights are more precise.

## Highlights & Insights

- **Excellent Problem Analysis**: A toy example is used to clearly demonstrate the density issues of SDF in multi-object scenes, making the motivation highly convincing.
- **Clear Division of Labor between SRDF and SDF**: SRDF handles density modeling (view-dependent), while SDF handles surface extraction (view-independent), with each fulfilling its distinct role.
- **Self-Supervised Visibility**: It does not rely on multi-view geometry or extra annotations, utilizing the network's own SRDF/SDF predictions to generate pseudo-labels.
- **High Versatility**: It can be applied to reconstruction methods based on VolSDF or NeuS, utilizing either Grid or MLP representations.

## Limitations & Future Work

- The SRDF MLP introduces additional network parameters and computational overhead.
- For single-object scenes, the difference between SRDF and SDF is minor, leading to limited gains.
- Visibility pseudo-labels may introduce noise during the early stages of training (when SDF/SRDF are inaccurate).
- Similar ideas could be explored within the 3D Gaussian Splatting framework.
- There is still room for improvement in the degree of enhancement on large-scale outdoor scenes (e.g., Tanks and Temples).

## Related Work & Insights

- **VolSDF**: Converts SDF into density using Laplace CDF → Ours points out its limitations in multi-object scenes.
- **MonoSDF**: Leverages monocular depth and normal priors to enhance reconstruction → Ours uses this as a baseline to achieve further improvements.
- **VolRecon (CVPR 2023)**: Uses SRDF for generalizable multi-view reconstruction → Ours applies SRDF to volume rendering density modeling for per-scene optimization.
- **VIP-NeRF**: Constructs visibility labels using plane-sweeping volumes → Our self-supervised method is more lightweight and does not require multi-view geometric computations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — SRDF density modeling presents a novel perspective in neural scene reconstruction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, two representations, and detailed ablation and qualitative analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The motivation of the problem is exceptionally clear via a toy example.
- **Value**: ⭐⭐⭐⭐ — Serves as a plug-and-play module compatible with existing SDF-based methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A Probability-guided Sampler for Neural Implicit Surface Rendering](a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)
- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene (FA-NeRF)
description: >-
  [CVPR 2025][3D Vision][NeRF] FA-NeRF proposes a frequency-aware neural radiance field framework. It analyzes the scene's frequency distribution using a 3D frequency quantification method, combining frequency grids, frequency-aware feature re-weighting, and adaptive ray marching. It captures both the overall scene structure and high-definition tiny details within a single model, significantly outperforming all baseline methods on multi-frequency datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "NeRF"
  - "Frequency-aware"
  - "Multi-frequency scene"
  - "Tiny-detail rendering"
  - "Adaptive sampling"
date: 2026-05-08
content_hash: aa0f7cd21759fcaf
---

# LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene (FA-NeRF)

**Conference**: CVPR 2025  
**arXiv**: [2503.18513](https://arxiv.org/abs/2503.18513)  
**Code**: None  
**Area**: 3D Vision / Neural Radiance Fields  
**Keywords**: NeRF, Frequency-aware, Multi-frequency scene, Tiny-detail rendering, Adaptive sampling

## TL;DR

FA-NeRF proposes a frequency-aware neural radiance field framework. It analyzes the scene's frequency distribution using a 3D frequency quantification method, combining frequency grids, frequency-aware feature re-weighting, and adaptive ray marching. It captures both the overall scene structure and high-definition tiny details within a single model, significantly outperforming all baseline methods on multi-frequency datasets.

## Background & Motivation

**Background**: NeRF has achieved great success in novel view synthesis. However, existing methods either focus on modeling high-frequency details in local scenes or handle low-frequency structures in large-scale scenes, making it difficult to accommodate both in a single model.

**Limitations of Prior Work**: Although Mip-NeRF 360 introduces cone tracing for anti-aliasing, it performs poorly when multi-frequency signals coexist because it treats all pixels uniformly, ignoring the frequency distribution in the scene. Methods like BungeeNeRF handle large-scale changes by progressively enabling high-frequency features, but exhibit poor generalization in complex scenes. Spatial partitioning methods (such as adaptive octrees) partition based on spatial relationships rather than frequency distributions, failing to align with actual high-frequency regions.

**Key Challenge**: In immersive scenes, users need to both overlook the entire scene (low-frequency structure) and zoom in to observe petal textures or butterfly wings (high-frequency details). However, images from different perspectives and resolutions cause the frequency of 3D signals to vary by orders of magnitude, posing a fundamental challenge to NeRF.

**Goal**: How to accurately quantify the frequency distribution of a 3D scene in a single NeRF model, and adaptively allocate network capacity, adjust sample density, and adjust feature weights based on this?

**Key Insight**: It is assumed that the frequency of 3D content can be inferred from the degraded 2D image space—by finding the lowest sufficient frequency through progressive image regression, and then projecting it back to the 3D space based on focal length and depth, thereby obtaining the 3D frequency distribution of the entire scene.

**Core Idea**: Quantify 3D frequencies through progressive image regression and store them in a frequency grid, using the frequency information to guide feature re-weighting and adaptive sampling, achieving high-fidelity rendering of both scene structures and tiny details in a single model.

## Method

### Overall Architecture

The input to FA-NeRF is a multi-frequency dataset containing overall/normal-resolution images (scene structure) and high-resolution images (detailed regions). The entire framework is built on the Hash Grid architecture of Instant-NGP. First, the 3D frequency distribution of the scene is quantified via progressive image regression and stored in a frequency grid. During training, three key operations are executed based on the frequency information: (1) frequency-aware feature re-weighting for each level of the Hash Grid; (2) frequency-balanced sampling to increase the training probability in high-frequency regions; (3) adaptive ray marching to adjust the sampling interval according to the frequency. The entire system achieves a rendering speed of 20 FPS on a single RTX 4090.

### Key Designs

1. **3D Frequency Quantification (Patch-based 3D Frequency Quantification)**:

    - **Function**: Analyzes the frequency level of each 3D point in the scene.
    - **Mechanism**: Progressive image regression—for each 2D image patch, the frequency components of NeRF encoding are progressively increased until the rendered result's SSIM compared to the ground truth exceeds a threshold $t$. The frequency at this point is defined as the 2D frequency $f_{2D}$ of the patch. Then, the 2D frequency is projected to 3D space via $f_{3D} = f_{2D} \cdot fl / d$ (where $fl$ is the focal length and $d$ is depth). For a 3D point with multiple observing patches, the median of all projected frequencies is taken as its 3D frequency. Experiments prove that different frequency contents require different lowest NeRF frequency levels, and the estimated 3D frequency accurately reflects the true frequency.
    - **Design Motivation**: The required frequency representation capability varies greatly among different objects in a scene (e.g., rough walls vs. fine patterns). Without quantifying frequency, the network capacity cannot be allocated reasonably.

2. **Frequency Grid + Frequency-Aware Feature Re-weighting**:

    - **Function**: Stores the frequency distribution of the entire scene and adaptively adjusts the weight of each feature level based on the frequency.
    - **Mechanism**: A frequency voxel grid $V^{(\text{frequency})} \in \mathbb{R}^{N_x \times N_y \times N_z \times 1}$ is used to store spatial frequency information, initialized by point clouds and updated during training. In the multi-level Hash Grid encoding of Instant-NGP, the feature at level $\ell$ is multiplied by the weight $\omega_\ell = \text{erf}\left(\sqrt{(\ell_{max} - \ell_{min})^2 / \text{Clip}[(\ell_{max} - \ell + 1)^2]}\right)$. This is a single-sided attenuation function—automatically reducing the weights of high-level features in low-frequency regions, avoiding wasting high-frequency feature space on low-frequency content.
    - **Design Motivation**: High-resolution levels in Hash Grid contribute minimally to low-frequency content but waste capacity. Through re-weighting, the network can more efficiently utilize the limited feature space to serve different frequency contents.

3. **Adaptive Ray Marching**:

    - **Function**: Adaptively adjusts ray sampling intervals according to the frequency of the content.
    - **Mechanism**: High-frequency regions require denser sampling points to avoid over-smoothing. According to the frequency value $f$ in the frequency grid, the sampling frequency is set to $f_{sample} = 2f$ following the sampling theorem. This automatically determines the appropriate sampling interval without manual tuning.
    - **Design Motivation**: Traditional methods use a fixed sampling interval, which causes sampling points to be far from the surface on high-frequency surfaces, leading to incorrect colors (over-smoothing), while wasting computational resources on low-frequency surfaces. Frequency-aware adaptive sampling achieves the optimal balance between accuracy and efficiency.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{recon}(\hat{c}, c_{gt}) + \lambda_{dist}\mathcal{L}_{dist} + \lambda_{depth}\mathcal{L}_{depth}$, where the reconstruction loss uses the Charbonnier formulation $\sqrt{(\hat{c} - c_{gt})^2 + \epsilon}$, $\mathcal{L}_{dist}$ regularizes the density distribution to encourage thin surfaces, and $\mathcal{L}_{depth}$ uses the depth of sparse point clouds to prevent geometry errors in early training stages. In addition, a Frequency-balanced Sampling (FAS) strategy is applied, which evenly distributes the training batch across $N$ frequency intervals to increase the sampling probability of high-frequency regions.

## Key Experimental Results

### Main Results

Multi-Frequency Dataset (constructed by the authors):

| Method | Structure PSNR↑ | Structure SSIM↑ | Detail PSNR↑ | Detail SSIM↑ | Detail LPIPS↓ |
|------|-----------------|-----------------|--------------|--------------|---------------|
| TensoRF | 28.88 | 0.854 | 22.76 | 0.781 | 0.430 |
| iNGP-Base | 30.27 | 0.893 | 23.63 | 0.784 | 0.408 |
| iNGP-Big | 30.97 | 0.909 | 24.00 | 0.786 | 0.398 |
| Mip-NeRF360 | 30.79 | 0.906 | 24.16 | 0.792 | 0.383 |
| 3D-GS | 30.85 | 0.897 | 24.29 | 0.802 | 0.390 |
| **FA-NeRF** | **32.44** | **0.929** | **26.29** | **0.843** | **0.332** |

Standard Dataset (MipNeRF-360 + Tanks&Temples):

| Method | MipNeRF-360 PSNR↑ | T&T PSNR↑ |
|------|-------------------|-----------|
| Mip-NeRF360 | 31.49 | 22.22 |
| 3D-GS | 30.95 | 24.36 |
| **FA-NeRF** | 31.20 | **24.45** |

### Ablation Study

Music Room Scene (Multi-Frequency Dataset):

| Configuration | normal-res PSNR↑ | high-res PSNR↑ | high-res LPIPS↓ |
|------|-------------------|----------------|-----------------|
| w/o Frequency Grid (A) | 31.95 | 24.90 | 0.316 |
| w/o Feature Re-weighting (B) | 33.58 | 26.73 | 0.256 |
| w/o FAS (C) | 33.50 | 25.84 | 0.268 |
| w/o adaptive RM (D) | 32.30 | 25.42 | 0.255 |
| **Complete Model (E)** | 33.52 | **26.97** | **0.250** |

### Key Findings

- Removing the frequency grid (Model A) results in the largest performance drop, proving that frequency awareness is the foundation of the entire framework.
- Removing Adaptive Ray Marching (ARM) causes a 1.55 drop in high-resolution PSNR, which is the most significant impact from a single component, because high-frequency content requires denser sampling.
- Disabling feature re-weighting slightly improves low-resolution performance (33.58 vs 33.52) but degrades high-resolution performance, indicating that when capacity is limited, low-frequency signals tend to "drown out" high-frequency signals.
- Simply enlarging the Hash Table (iNGP-Big vs. iNGP-Base) yields limited improvement, demonstrating that increasing capacity alone cannot solve the multi-frequency problem.
- Improvements are also observed on standard datasets with smaller frequency spans, indicating that the multi-frequency problem is ubiquitous.

## Highlights & Insights

- **Universality of the Frequency Quantification Method**: Quantifying the abstract concept of "scene frequency" into concrete values via progressive image regression can be transferred to other representations like 3D-GS. This paradigm of "quantifying frequency first, then performing frequency-aware design" can inspire many scene representation tasks.
- **3D Rendering Application of Sampling Theorem**: The Nyquist sampling theorem is cleverly applied to ray marching—setting the sampling frequency to twice the content frequency, which is both theoretically supported and eliminates the pain point of manual parameter tuning.
- **Dataset Design Idea**: Mixing panorama low-resolution images and local high-resolution images to construct multi-frequency datasets aligns well with practical application scenarios (e.g., long-shot + close-up requirements in virtual tourism), offering a new evaluation perspective for the community.

## Limitations & Future Work

- The progressive image regression preprocessing stage requires extra computational cost (although the authors state that subsequent update costs are negligible).
- The initialization of the frequency grid depends on the quality of SfM point clouds; frequency estimation in sparse regions might be inaccurate.
- Lacks comparison with recent 3D-GS variants (such as anti-aliased 3D-GS).
- Scene frequency may depend on viewpoints (e.g., reflective surfaces), and a simple static frequency grid may not fully capture this.

## Related Work & Insights

- **vs Mip-NeRF 360**: Mip-NeRF 360 achieves a certain level of multi-scale rendering through cone tracing + IPE, but it treats all pixels uniformly. FA-NeRF explicitly quantifies 3D frequency and adaptively adjusts based on it, outperforming Mip-NeRF 360 by 1.65-2.13 dB in PSNR on the multi-frequency dataset.
- **vs BungeeNeRF**: BungeeNeRF progressively enables high-frequency features, but partitions based on spatial locations rather than frequency distributions, behaving similarly to Mip-NeRF in multi-frequency scenes. FA-NeRF's frequency-aware design is more targeted.
- **vs 3D-GS**: 3D-GS exhibits spiky artifacts under large frequency spans, appearing "sharp" but failing to capture real details. FA-NeRF comprehensively leads in PSNR, SSIM, and LPIPS.

## Rating

- Novelty: ⭐⭐⭐⭐ The 3D frequency quantification and frequency-aware framework have strong novelty, though the individual components (re-weighting, adaptive sampling) themselves are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Custom multi-frequency dataset + standard datasets + detailed ablation study, but lacks comparison with more baselines.
- Writing Quality: ⭐⭐⭐⭐ The method flow is clear, and the explanation of the toy example is intuitive.
- Value: ⭐⭐⭐⭐ Addresses practical demands in real-world scenes (long-shot + close-up), with good framework generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction](nerfprior_learning_neural_radiance_field_as_a_prior_for_indoor_scene_reconstruct.md)
- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](../../ECCV2024/3d_vision/taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)

</div>

<!-- RELATED:END -->

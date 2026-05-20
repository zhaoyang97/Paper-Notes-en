---
title: >-
  [Paper Note] ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars
description: >-
  [CVPR 2026][3D Vision][Progressive 3D representation] This paper proposes ProgressiveAvatars, a progressive avatar representation that constructs hierarchical 3DGS via adaptive implicit subdivision on a template mesh…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Progressive 3D representation"
  - "animatable head avatars"
  - "3D Gaussian splatting"
  - "streaming"
  - "adaptive subdivision"
date: 2026-05-08
content_hash: 043a923721c56f2b
---

# ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars

**Conference**: CVPR 2026
**arXiv**: [2603.16447](https://arxiv.org/abs/2603.16447)  
**Code**: [GitHub](https://ustc3dv/ProgressiveAvatars)  
**Area**: 3D Vision
**Keywords**: Progressive 3D representation, animatable head avatars, 3D Gaussian splatting, streaming, adaptive subdivision

## TL;DR
This paper proposes ProgressiveAvatars, a progressive avatar representation that constructs hierarchical 3DGS via adaptive implicit subdivision on a template mesh, enabling progressive transmission and rendering under varying bandwidth and compute constraints. With only 5% of the data (2.6 MB), a usable avatar is immediately renderable, and incremental loading smoothly improves quality to a level comparable with state-of-the-art methods.

## Background & Motivation
**Background**: High-fidelity real-time head avatars are a critical technology for immersive interaction. 3DGS has become the dominant explicit representation due to its efficient rendering. Methods such as GaussianAvatars, FlashAvatar, and MeGA have achieved high-quality animatable avatars.

**Limitations of Prior Work**:
- In multi-user dynamic scenarios such as social VR, transmitting high-fidelity avatars as traditional static assets causes severe startup latency and bandwidth spikes; users must wait for the complete download before any rendering is possible.
- Existing 3DGS avatars lack incremental loading mechanisms and cannot smoothly accumulate detail during transmission.
- Existing LOD methods (LoDAvatar, ArchitectHead) rely on a discrete LOD switching paradigm, requiring multiple independent model copies, which introduces significant storage redundancy and resource-switching latency.
- Uniform subdivision (e.g., LoDAvatar) over-refines smooth regions while under-refining high-frequency regions, wasting resources.

**Key Challenge**: How to achieve progressive transmission and rendering within a single unified asset—supporting immediately animatable rendering at arbitrary transmission ratios—without introducing discrete asset switching or storage redundancy.

**Key Insight**: Construct hierarchical 3DGS using face-local coordinate systems on the FLAME template mesh, with adaptive implicit subdivision for on-demand detail growth and importance-ordered sorting for continuous streaming.

**Core Idea**: Build a hierarchical forest via adaptive implicit subdivision on FLAME mesh faces, and leverage per-face importance scores to enable progressive transmission and rendering through incremental loading.

## Method

### Overall Architecture
Input: head video → FLAME mesh tracking. Training: bind 3D Gaussians to FLAME faces, drive implicit subdivision via screen-space gradients to construct a multi-level hierarchy. Inference: precompute importance scores, transmit Gaussians in descending importance order, and incrementally add and render on the receiver side.

### Key Designs

1. **Implicit Subdivision**

    - Function: Recursively generate child faces on each triangular face of the FLAME template mesh, forming a per-face hierarchical tree structure.
    - Mechanism: For a parent face $f = (i,j,k)$, new vertices are created via barycentric interpolation: $\mathbf{p} = \beta_1 \mathbf{v}_i + \beta_2 \mathbf{v}_j + \beta_3 \mathbf{v}_k$. Barycentric coordinates are initialized to $(1/3, 1/3, 1/3)$ and optimized under simplex constraints during training. Subdivision point positions under different expressions and poses are recomputed via the same barycentric mapping.
    - Meaning of "Implicit": Rather than explicitly creating new vertices and topology for child faces, new points are implicitly defined within the parent face via learnable barycentric coordinates, allowing them to move freely within the triangle to adapt to the optimal position for different facial regions.
    - Design Motivation: Compared with explicit uniform subdivision, implicit subdivision adapts to different effective scales and shapes across facial regions of varying sizes and structures through learnable barycentric coordinates.

2. **Face-Local Gaussian Binding**

    - Function: Bind 3D Gaussians to the local coordinate system of each face in the hierarchy.
    - Mechanism: Each Gaussian's rotation $\mathbf{R} = \Delta\mathbf{R}\mathbf{r}$, scale $\mathbf{S} = \Delta\mathbf{S} s$, and center $\boldsymbol{\mu} = s\mathbf{r}\Delta\boldsymbol{\mu} + \mathbf{t}$, where $\mathbf{r}$ is the rotation aligned with the face normal, $\mathbf{t}$ is the face centroid, $s$ is the mean edge length, and $\Delta\mathbf{R}, \Delta\mathbf{S}, \Delta\boldsymbol{\mu}$ are trainable residuals.
    - Design Motivation: Face-local parameterization ensures that Gaussians deform jointly with the face (expressions/head motion), maintaining consistent appearance across different levels of the hierarchy.

3. **Adaptive Growing**

    - Function: Expand the hierarchy on demand during training, concentrating detail in regions that need it most.
    - Mechanism: Screen-space gradients $g_i$ are accumulated only at the current finest level $\ell_{\max}$. Every $k$ iterations, leaf faces satisfying $g_i > \varepsilon$ are selected for subdivision and new Gaussians are bound to child faces. This process repeats until the maximum depth $L$ is reached.
    - Design Motivation: Uniform subdivision expands all regions indiscriminately, wasting computation and storage. The adaptive strategy focuses limited resources on high-frequency regions such as facial features and hair, while smooth regions (forehead, cheeks) require fewer Gaussians.

4. **Importance Scoring and Progressive Transmission**

    - Function: Determine the transmission priority of Gaussians within each level.
    - Mechanism: The importance score of each face is defined as the total rendering contribution of its bound Gaussians across all pixels: $W_i = \sum_{j \in \mathcal{G}_i} \sum_p \alpha_{j,p} T_{j,p}$. Gaussians are transmitted in descending order of score, so high-contribution Gaussians arrive first.
    - Design Motivation: Transmitting high-importance Gaussians first minimizes color drift between partial and complete renders. Experiments (Fig. 3) demonstrate that importance-first transmission significantly outperforms random transmission.

### Loss & Training
- Multi-level joint supervision: $\mathcal{L}_{\text{rgb}} = \sum_{\ell \in \mathcal{S}} w_\ell [(1-\lambda_s)\mathcal{L}_1 + \lambda_s \mathcal{L}_{\text{ssim}}]$
- Coarse-to-fine optimization: depth limit initialized to 1, increased every 50k iterations to trigger adaptive subdivision.
- Regularization: $\mathcal{L}_{\text{scale}}$ (scale constraint) + $\mathcal{L}_{\text{pos}}$ (position constraint), preventing Gaussians from deviating from their bound faces.
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{scale}}\mathcal{L}_{\text{scale}} + \lambda_{\text{pos}}\mathcal{L}_{\text{pos}}$
- Adam optimizer, 60k iterations, adaptive expansion every 2k iterations.

## Key Experimental Results

### Main Results (NeRSemble dataset, varying transmission budgets)

| Transmission Ratio | NVS PSNR↑ | NVS SSIM↑ | NVS LPIPS↓ | #Gaussians | Data Size | FPS |
|--------------------|-----------|-----------|-----------|------------|-----------|-----|
| 5% (Base) | 27.89 | 0.851 | 0.186 | 10,144 | 2.60 MB | 291 |
| 25% | 29.14 | 0.892 | 0.080 | 37,302 | 9.56 MB | 278 |
| 50% | 30.03 | 0.904 | 0.073 | 84,132 | 21.56 MB | 258 |
| 100% | 31.47 | 0.929 | 0.068 | 169,438 | 43.42 MB | 260 |
| GaussianAvatars | 31.10 | 0.937 | 0.064 | 163,829 | 41.90 MB | 271 |

### Comparison with SOTA

| Method | NVS PSNR↑ | NVS LPIPS↓ | NES PSNR↑ | NES LPIPS↓ |
|--------|-----------|-----------|-----------|-----------|
| PointAvatar | 25.8 | 0.097 | 23.4 | 0.102 |
| GaussianAvatars | 31.1 | 0.064 | 25.8 | 0.076 |
| Ours (5%) | 27.9 | 0.186 | 25.1 | 0.176 |
| **Ours (100%)** | **31.5** | 0.068 | **25.9** | 0.080 |

### Key Findings
- Only 5% of the data (2.6 MB) yields a usable avatar (PSNR 27.89), whereas GaussianAvatars requires nearly all data before any rendering is possible.
- At 100% transmission, PSNR 31.47 surpasses GaussianAvatars (31.10); novel expression synthesis (NES) is also marginally better.
- Frame rate is consistently maintained at 258–291 FPS (RTX 4090, 550×802); increasing the number of Gaussians does not cause a notable frame rate drop.
- Adaptive subdivision outperforms uniform subdivision: reconstruction quality is higher at the same Gaussian count (Fig. 6), and high-frequency regions (e.g., beard) receive deeper subdivision.
- Multi-level supervision is critical for progressive transmission: when only the finest level is supervised, coarser levels fail to learn a complete avatar (Tab. 3 shows PSNR dropping from 29.87 to 20.06 at 35% budget without multi-level supervision).

## Highlights & Insights
- **From discrete LOD to continuous progressive streaming**: a fundamental paradigm shift. Traditional LOD requires multiple independent models and incurs switching latency, whereas ProgressiveAvatars' single continuous asset supports immediate rendering at any transmission ratio—directly valuable for latency-sensitive scenarios such as social VR.
- **Adaptive implicit subdivision is substantially more efficient than uniform subdivision**: high-frequency regions (eyes, mouth, beard) receive deeper subdivision while smooth regions remain at shallow levels, achieving a better quality–cost trade-off.
- **Importance-based ordering guarantees monotonic quality improvement during progressive streaming**: transmitting high-contribution Gaussians first ensures each incremental step maximally improves rendering quality.
- **Face-local binding combined with barycentric mapping** ensures that the avatar is animatable at any transmission stage, which is a key consequence of the technical design choices.

## Limitations & Future Work
- LPIPS at 100% transmission (0.068) is slightly worse than GaussianAvatars (0.064), leaving a small perceptual quality gap.
- The maximum subdivision depth $D=4$ limits the finest detail achievable.
- Validation is conducted only on the NeRSemble dataset; generalization to more subjects and more complex scenes remains unknown.
- Importance scores are precomputed at training time and fixed, precluding dynamic adjustment based on runtime viewpoint.
- Priority strategies for progressive transmission of multiple simultaneous avatars in multi-user scenes are not discussed.

## Related Work & Insights
- **vs. GaussianAvatars**: Both share a similar face-binding Gaussian design, but GaussianAvatars lacks hierarchical structure and progressive transmission capability. ProgressiveAvatars delivers a comparable-quality progressive experience starting from just 2.6 MB.
- **vs. LoDAvatar**: LoDAvatar also builds LOD on a template mesh but uses uniform subdivision with hand-crafted masks for selective densification. ProgressiveAvatars' adaptive growing is more principled and achieves higher quality.
- **vs. GA+LightGaussian (discrete compression)**: GA+LG requires 227.2 MB to store 10 discrete LOD levels, whereas ProgressiveAvatars uses a single 43.4 MB asset supporting continuous rendering at arbitrary transmission ratios.

## Rating
- Novelty: ⭐⭐⭐⭐ — The paradigm shift from discrete LOD to continuous progressive streaming is innovative, and the adaptive implicit subdivision is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Progressive transmission simulation and ablations are thorough, though validation is limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated and the method is described in detail.
- Value: ⭐⭐⭐⭐ — Significant practical value for latency-sensitive 3D avatar transmission scenarios such as VR and telepresence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[CVPR 2026\] Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image](zero-shot_reconstruction_of_animatable_3d_avatars_with_cloth_dynamics_from_a_sin.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [\[CVPR 2026\] PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)

</div>

<!-- RELATED:END -->

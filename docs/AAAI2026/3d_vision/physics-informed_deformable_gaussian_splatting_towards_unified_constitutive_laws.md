---
title: >-
  [Paper Note] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] By treating each 3D Gaussian as a Lagrangian material point and introducing a time-evolving material field to predict particle velocity and the constitutive stress tensor, this work incorporates the Cauchy momentum residual as a physical constraint alongside Lagrangian particle flow matching as a data-fitting term. This approach achieves physical consistency and cross-scene generalization in monocular dynamic novel view synthesis…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Dynamic Scene Reconstruction"
  - "Physics-Informed Neural Networks"
  - "Continuum Mechanics"
  - "Optical Flow Supervision"
date: 2026-05-08
content_hash: c25aeb8ac717c1d8
---

# Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field

**Conference**: AAAI 2026  
**arXiv**: [2511.06299](https://arxiv.org/abs/2511.06299)  
**Code**: [https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Dynamic Scene Reconstruction, Physics-Informed Neural Networks, Continuum Mechanics, Optical Flow Supervision

## TL;DR
By treating each 3D Gaussian as a Lagrangian material point and introducing a time-evolving material field to predict particle velocity and the constitutive stress tensor, this work incorporates the Cauchy momentum residual as a physical constraint alongside Lagrangian particle flow matching as a data-fitting term. This approach achieves physical consistency and cross-scene generalization in monocular dynamic novel view synthesis, achieving state-of-the-art (SOTA) performance on both a self-constructed physics-driven dataset and the HyperNeRF dataset.

## Background & Motivation
**Background**: 3DGS has become the mainstream method for dynamic novel view synthesis due to its explicit representation and real-time rendering capabilities. Dynamic modeling approaches include incremental methods (DynamicGS), deformation field methods (D-3DGS, Grid4D), and low-rank decomposition methods (SC-GS), which, when combined with 4D decomposed hash encoding, can efficiently represent spatiotemporal information.

**Limitations of Prior Work**: Existing approaches simplify motion to rigid body transformations, ignoring the constitutive physical laws of different materials (as the motion patterns of fluids, elastomers, and fabrics are vastly different). Furthermore, they rely solely on 2D visual supervision (RGB loss), which fails to provide unified constraints on the physical states of 3D particles, causing Gaussian particles to deviate from physically realistic motion patterns.

**Key Challenge**: Purely data-driven deformation fields lack physical inductive bias and cannot distinguish the motion laws of different materials (fluid vs. elastomer vs. rigid body). Existing physics-embedded methods (such as PhysGaussian and those based on PINNs) rely on strict boundary conditions, fixed material properties, or RGB-D/multi-view inputs, making them fail to generalize to monocular dynamic scenes.

**Goal**: (1) How to model the locations and time-evolving deformations of Gaussian particles without prior knowledge of particle motion? (2) What kind of boundary conditions or alternative supervisions can realize physically consistent and generalizable dynamic material modeling?

**Key Insight**: Starting from Lagrangian mechanics, this work employs the Cauchy momentum equation as a unified constitutive law, where the velocity and stress of each particle are independently predicted through a time-evolving material field. Meanwhile, optical flow decomposition is leveraged to provide motion flow as pseudo-ground truth to guide the convergence of the velocity field.

**Core Idea**: Embed the Cauchy momentum equation from continuum mechanics into the 3DGS framework. Each Gaussian particle acts as a Lagrangian material point that evolves within a time-evolving material field. Generalization across materials is achieved through dual supervision using physical residuals and optical flow alignment.

## Method

### Overall Architecture
PIDG consists of three core modules: (1) Dynamic modeling in canonical hash space, which uses a 4D decomposed hash encoding to efficiently represent spatiotemporal deformation and decouples static and dynamic regions; (2) Physics-informed Gaussian representation, which treats each Gaussian as a Lagrangian particle and predicts velocity and stress tensors via a time-evolving material field, employing the Cauchy momentum residual as a physical constraint; (3) Lagrangian particle flow matching, which decomposes optical flow into camera flow and motion flow, using the motion flow as pseudo-ground truth to supervise the Gaussian flow and velocity flow of particles. The entire pipeline is fully differentiable and trained end-to-end from monocular video inputs.

### Key Designs

1. **4D Decomposed Hash Encoding + Static-Dynamic Decoupling**:

    - Function: Efficiently encodes 4D spatiotemporal coordinates into features, decoupling static and dynamic regions.
    - Mechanism: Map $(x,y,z,t)$ to four independent 3D hash grids $G_{xyz}, G_{xyt}, G_{yzt}, G_{xzt}$, reducing memory complexity from $\mathcal{O}(n^4)$ to $\mathcal{O}(n^3)$. A spatial MLP extracts directional attention weights $a = 2\sigma(f_s(G_{xyz})) - 1$ to modulate the output features from a temporal MLP $h = a \odot f_t(G_{xyt}, G_{yzt}, G_{xzt})$. A multi-head MLP then decodes deformation parameters $D(h) = \{R_x, T_x, \Delta r, \Delta s\}$.
    - Design Motivation: Compared to 4D MLPs or low-rank plane decompositions, hash encoding significantly reduces memory overhead while maintaining accuracy. A two-stage optimization strategy first jointly optimizes geometry and motion, and then freezes static regions using a dynamic mask, allowing the physical modeling to focus entirely on the dynamic parts.

2. **Time-Evolving Material Field**:

    - Function: Predicts the time-varying velocity and constitutive stress tensor for each Gaussian particle.
    - Mechanism: Embed normalized 4D coordinates into 6 learnable spatial/temporal plane tensors $\mathbf{F}_{\text{Hash}}$, concatenate them with a Fourier temporal encoding $T(t)$ and a learnable particle index embedding $\mathbf{e}_i$ to obtain the feature vector $\mathbf{F} = [\mathbf{F}_{\text{Hash}}, T(t), \mathbf{e}_i]$. A multi-head MLP $f_\theta$ jointly predicts the velocity $\bm{v} \in \mathbb{R}^3$ and the 6 independent components of the stress tensor $\bm{\sigma} \in \mathbb{R}^6$. The Cauchy momentum residual is defined as $\mathbf{r}(x,t) = \rho(\frac{\partial \bm{v}}{\partial t} + (\bm{v} \cdot \nabla)\bm{v}) - \nabla \cdot \bm{\sigma}$, and minimizing its L2 norm yields the physics loss $\mathcal{L}_{\text{CMR}}$.
    - Design Motivation: Modeling velocity and stress as independent intrinsic attributes allows each Gaussian particle to not only be encoded in a deformation field but also evolve continuously over time. By changing the form of the constitutive stress tensor, the Cauchy momentum equation can uniformly describe fluid, elastomer, and rigid body dynamics.

3. **Lagrangian Particle Flow Matching**:

    - Function: Uses optical flow decomposition to provide motion supervision, guiding the velocity and stress fields to converge to physically plausible solutions.
    - Mechanism: **Backward Optical Flow Decomposition**—calculates motion flow backward from $I_{t+1}$ and transforms it into the $I_t$ coordinate system to avoid banding artifacts caused by bilinear interpolation in forward strategies. The Gaussian flow $flow_g$ is obtained by the weighted sum of 2D displacements of tracked top-K Gaussian particles, while the velocity flow $flow_v$ is obtained by advecting Gaussian particles using the predicted velocity. Both are aligned with the ground-truth motion flow: $\mathcal{L}_{\text{LPFM}} = \lambda_g \|flow_g - flow_{gt}\|_1 + \lambda_v \|flow_v - flow_{gt}\|_1$.
    - Design Motivation: Relying solely on the Cauchy momentum residual as a physical constraint makes it difficult for velocity and stress predictions to converge to a physically plausible solution (as the equations are underdetermined). Introducing optical flow as a data-fitting term serves as an alternative boundary condition in the PINN framework, anchoring particle trajectories and providing a clear convergence direction for optimization.

### Loss & Training
The total loss includes: rendering loss $\mathcal{L}_{\text{renders}} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c\mathcal{L}_{\text{D-SSIM}}$ ($\lambda_c=0.2$), Cauchy momentum residual $\mathcal{L}_{\text{CMR}}$ ($\lambda_{\text{CMR}}=0.1$), and Lagrangian particle flow matching $\mathcal{L}_{\text{LPFM}}$ ($\lambda_{\text{LPFM}}=0.01$). It is trained for 50K iterations (synthetic) / 40K iterations (real-world). To avoid GPU memory explosion, a chunked sampling strategy is adopted to calculate the Cauchy momentum residual—particles are divided into chunks to compute physical residuals and the computational graph is released before aggregation. The scaling threshold is adjusted from 0.1 to 0.015 to filter out large-scale noisy Gaussians.

## Key Experimental Results

### Main Results (PIDG Physics-driven Dataset + HyperNeRF)

| Method | PIDG-PSNR↑ | PIDG-SSIM↑ | PIDG-LPIPS↓ | HyperNeRF-PSNR↑ | HyperNeRF-MS-SSIM↑ |
|------|-----------|-----------|------------|---------|-----------|
| D-NeRF | 23.45 | 0.866 | 0.124 | 25.7 | 0.726 |
| D-3DGS | 29.54 | 0.951 | 0.066 | - | - |
| GaussianPredict | 30.17 | 0.957 | 0.062 | 26.6 | 0.884 |
| Grid4D | 30.32 | 0.956 | 0.061 | 27.3 | 0.899 |
| D-2DGS | 29.23 | 0.944 | 0.061 | 17.7 | 0.509 |
| **PIDG (Ours)** | **30.96** | **0.967** | **0.058** | **27.8** | **0.906** |

### Ablation Study

| Configuration | PIDG-PSNR↑ | PIDG-SSIM↑ | D-NeRF-PSNR↑ | Description |
|------|-----------|-----------|-------------|------|
| w/o ($\mathcal{L}_{\text{LPFM}} + \mathcal{L}_{\text{CMR}}$) | 30.46 | 0.956 | 42.00 | Static-dynamic decoupling + hash encoding only |
| w/o $\mathcal{L}_{\text{LPFM}}$ | 30.78 | 0.957 | 42.14 | Cauchy momentum residual constraint only |
| **Full model** | **30.96** | **0.967** | - | Full model (D-NeRF cannot use flow matching due to a lack of continuous views) |
| Grid4D + $\mathcal{L}_{\text{CMR}}$ | - | - | 42.10 | Plug-and-play improvement +0.10 PSNR |
| SC-GS + $\mathcal{L}_{\text{CMR}}$ | - | - | 41.85 | Plug-and-play improvement +0.20 PSNR |

### Key Findings
- Full flow matching yields the most significant improvements in complex physical scenes like fluid smoke (Dry Ice: 25.34 $\rightarrow$ 26.12 PSNR) and elastic collisions (Balls: 32.79 $\rightarrow$ 33.31), indicating that optical flow supervision is particularly effective for complex physical motion.
- As a plug-and-play module, $\mathcal{L}_{\text{CMR}}$ consistently brings improvements when applied to GaussianPredict, SC-GS, and Grid4D, validating the universality of the physical constraints.
- Without the stress tensor, $\mathcal{L}_{\text{CMR}}$ degenerates into $\nabla \cdot \bm{v} = 0$ (continuity constraint). The t-SNE visualization shows a distinct decrease in the discriminative capacity of dynamic particle features.
- High training efficiency: 72 minutes / ~85K Gaussians / 250 FPS, with a memory overhead of 6.2GB, outperforming most baselines.

## Highlights & Insights
- **Unified Constitutive Law Framework**: This work derives the unified degradation process of the Cauchy momentum equation to rigid bodies, elastic solids, and fluids through Effective Field Theory (EFT). The network adaptively learns material properties without manual specification, demonstrating high theoretical elegance. The derivation in Appendix A presents the complete link from the scalar Goldstone field to the Lamé parameters and then to the Navier-Stokes equations.
- **Backward Optical Flow Decomposition**: Applying the motion mask to pixel-level motion flow instead of Gaussian flow preserves the true displacement from being corrupted by stripe artifacts. This is a crucial improvement over the forward strategy of MotionGS, yielding a noticeable difference in visual quality on the HyperNeRF real-world dataset.
- **Lagrangian Particle Identity Inheritance**: During densification, child particles inherit the index embedding of their parent particles, which avoids expensive nearest-neighbor searches to recover identity, making it simple and efficient.
- **Plug-and-play CMR**: The physical residual module can be inserted into various existing dynamic 3DGS methods as a regularizer with zero extra cost, offering strong practicality.

## Limitations & Future Work
- **Computational Overhead**: Training still takes hours (72 minutes on an A800) and consumes a significant amount of GPU memory (6.2GB), leaving a considerable gap for real-time reconstruction. The authors plan to develop a lightweight feed-forward network architecture to bypass expensive optimization cycles.
- **Limited Material Models**: Linear constitutive assumptions cannot capture complex behaviors such as non-linear elastoplasticity or viscoelasticity. Combining finite element or hybrid particle methods could enhance the richness of material modeling.
- **Dependency on Optical Flow**: The approach requires pre-trained models for optical flow, depth, and segmentation (e.g., UniMatch, Distill Any Depth, SAMv2). Flow matching cannot be applied to scenes with discontinuous views. Exploring self-supervised motion priors as alternatives is a potential path forward.
- **Limitations of 2D Evaluation**: The authors call on the community to establish a comprehensive evaluation system that includes geometric, temporal, and physical metrics, such as introducing velocity field consistency or stress field rationality metrics.

## Related Work & Insights
- **vs Grid4D**: Grid4D models dynamics efficiently using 4D hash encoding but lacks physical constraints; PIDG improves PSNR by 0.64 on top of it by adding the material field (on the PIDG dataset). PIDG maintains fewer Gaussians (~85K vs. ~100K) and achieves a higher FPS (250 vs. 240), indicating that the physical constraints also introduce an implicit regularization effect.
- **vs PhysGaussian/PhysDreamer**: MPM-based methods rely on grid discretization and fixed material properties, and require RGB-D or multi-view inputs. PIDG bypasses the grid discretization limit via the PINN framework, renders material parameters fully learnable, and requires only monocular video input.
- **vs MotionGS**: Although both utilize optical flow supervision, MotionGS suffers from stripe artifacts due to its forward decomposition strategy. PIDG's backward decomposition is more robust, exceeding SOTA in PSNR by 2.6 on HyperNeRF with more consistent flow visualization alignment.
- **vs GaussianFlow**: PIDG reuses GaussianFlow's CUDA rasterizer but incorporates efficiency optimizations (merging gradient computations, removing redundant backpropagation) to serve as the foundation for Lagrangian particle-flow calculation.

## Rating
- Novelability: ⭐⭐⭐⭐ Embedding the Cauchy momentum equation from continuum mechanics into the 3DGS framework provides a fresh theoretical perspective, though PINNs themselves are already widely applied.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Valuted on three platforms—the self-constructed 5-scene physical dataset, D-NeRF, and HyperNeRF. The ablations are extremely detailed, containing plug-and-play experiments, future predictions, t-SNE analysis, etc.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivations are complete (with the appendix including the EFT derivation of the unified constitutive laws), though the density of mathematical formulas in the main text is slightly high.
- Value: ⭐⭐⭐⭐ Provides a valuable physical inductive bias paradigm for dynamic 3DGS. The plug-and-play $\mathcal{L}_{\text{CMR}}$ module is highly practical, and the self-built PIDG dataset is also valuable to the community.

<!-- Note written on 2026-04-10, based on paper_cache/AAAI2026/2511.06299.txt full-text cache -->

<!-- Note: PIDG dataset contains 5 scenes (Balls Reaction/Mechanics Cloth/Motion Kuro/Rubber Duck/Dry Ice), each with 150 frames and 1600x900 resolution, generated by Blender physics solver. -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Gaussian Mapping for Evolving Scenes](../../CVPR2026/3d_vision/gaussian_mapping_for_evolving_scenes.md)
- [\[CVPR 2026\] Seele: A Unified Acceleration Framework for Real-Time Gaussian Splatting on Mobile Devices](../../CVPR2026/3d_vision/seele_a_unified_acceleration_framework_for_real-time_gaussian_splatting_on_mobil.md)
- [\[ICLR 2026\] DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics](../../ICLR2026/3d_vision/diffwind_physics-informed_differentiable_modeling_of_wind-driven_object_dynamics.md)
- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](../../CVPR2026/3d_vision/retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] Each 3D Gaussian is treated as a Lagrangian material point. A time-evolving material field predicts per-particle velocities and constitutive stress tensors…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Dynamic Scene Reconstruction"
  - "Physics-Informed Neural Networks"
  - "Continuum Mechanics"
  - "Optical Flow Supervision"
date: 2026-05-08
content_hash: 2928af2a89906655
---

# Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field

**Conference**: AAAI 2026
**arXiv**: [2511.06299](https://arxiv.org/abs/2511.06299)  
**Code**: [https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Dynamic Scene Reconstruction, Physics-Informed Neural Networks, Continuum Mechanics, Optical Flow Supervision

## TL;DR
Each 3D Gaussian is treated as a Lagrangian material point. A time-evolving material field predicts per-particle velocities and constitutive stress tensors; the Cauchy momentum residual serves as a physics constraint while Lagrangian particle flow matching provides a data-fitting term. The approach achieves physical consistency and cross-scene generalization in monocular dynamic view synthesis, reaching state-of-the-art performance on both a self-constructed physics-driven dataset and the HyperNeRF real-world benchmark.

## Background & Motivation

**Background**: 3DGS has become the dominant approach for dynamic novel-view synthesis owing to its explicit representation and real-time rendering capability. Dynamic modeling strategies include incremental methods (DynamicGS), deformation-field methods (D-3DGS, Grid4D), and low-rank decomposition methods (SC-GS), with 4D decomposed hash encoding enabling efficient spatiotemporal representation.

**Limitations of Prior Work**: Existing methods reduce motion to rigid-body transformations, ignoring the material-specific constitutive laws governing different substances (fluids, elastic bodies, cloth, etc. exhibit fundamentally different motion patterns). Relying solely on 2D visual supervision (RGB loss) is insufficient to constrain the physical state of 3D particles, causing Gaussian particles to deviate from physically plausible motion modes.

**Key Challenge**: Purely data-driven deformation fields lack physical inductive bias and cannot distinguish the motion laws of different materials (fluid vs. elastic solid vs. rigid body). Existing physics-embedding methods (PhysGaussian, PINNs-based approaches) depend on strict boundary conditions, fixed material properties, or RGB-D/multi-view inputs, precluding generalization to monocular dynamic scenes.

**Goal**: (1) How to model the positions and time-varying deformations of Gaussian particles without prior knowledge of particle motion? (2) What boundary conditions or alternative supervision signals enable physically consistent and generalizable dynamic material modeling?

**Key Insight**: Starting from Lagrangian mechanics, the Cauchy momentum equation is adopted as a unified constitutive law. Each particle's velocity and stress are independently predicted by a time-evolving material field. Optical flow decomposition provides motion flow as pseudo ground truth to guide velocity field convergence.

**Core Idea**: Embed the Cauchy momentum equation from continuum mechanics into the 3DGS framework; each Gaussian particle evolves as a Lagrangian material point within a time-evolving material field, with dual supervision from physics residuals and optical flow alignment achieving cross-material generalization.

## Method

### Overall Architecture
PIDG comprises three core modules: (1) *Dynamic modeling in a canonical hash space* — 4D decomposed hash encoding efficiently represents spatiotemporal deformations with static/dynamic region decoupling; (2) *Physics-informed Gaussian representation* — each Gaussian is treated as a Lagrangian particle whose velocity and stress tensor are predicted by a time-evolving material field, with the Cauchy momentum residual enforcing physical consistency; (3) *Lagrangian particle flow matching* — optical flow is decomposed into camera flow and motion flow, with the motion flow serving as pseudo ground truth to supervise both Gaussian flow and velocity flow. The entire pipeline is fully differentiable and trained end-to-end from monocular video.

### Key Designs

1. **4D Decomposed Hash Encoding + Static/Dynamic Decoupling**

    - **Function**: Efficiently encode 4D spatiotemporal coordinates into features and decouple static and dynamic regions.
    - **Mechanism**: Map $(x,y,z,t)$ to four independent 3D hash grids $G_{xyz}, G_{xyt}, G_{yzt}, G_{xzt}$, reducing memory from $\mathcal{O}(n^4)$ to $\mathcal{O}(n^3)$. A spatial MLP extracts directional attention weights $a = 2\sigma(f_s(G_{xyz})) - 1$, which modulate the output features of a temporal MLP: $h = a \odot f_t(G_{xyt}, G_{yzt}, G_{xzt})$. A multi-head MLP then decodes deformation parameters $D(h) = \{R_x, T_x, \Delta r, \Delta s\}$.
    - **Design Motivation**: Compared with 4D MLPs or low-rank planar decompositions, hash encoding substantially reduces memory while maintaining accuracy. A two-stage optimization strategy jointly optimizes geometry and motion first, then freezes static regions via a dynamic mask so that physics modeling focuses exclusively on dynamic parts.

2. **Time-Evolving Material Field**

    - **Function**: Predict time-varying velocities and constitutive stress tensors for each Gaussian particle.
    - **Mechanism**: Normalized 4D coordinates are embedded into six learnable spatial/temporal plane tensors $\mathbf{F}_{\text{Hash}}$, concatenated with a Fourier temporal encoding $T(t)$ and a learnable per-particle index embedding $\mathbf{e}_i$ to form the feature vector $\mathbf{F} = [\mathbf{F}_{\text{Hash}}, T(t), \mathbf{e}_i]$. A multi-head MLP $f_\theta$ jointly predicts velocity $\bm{v} \in \mathbb{R}^3$ and the six independent components of the stress tensor $\bm{\sigma} \in \mathbb{R}^6$. The Cauchy momentum residual is $\mathbf{r}(x,t) = \rho\!\left(\frac{\partial \bm{v}}{\partial t} + (\bm{v} \cdot \nabla)\bm{v}\right) - \nabla \cdot \bm{\sigma}$, and its $L_2$ norm yields the physics loss $\mathcal{L}_{\text{CMR}}$.
    - **Design Motivation**: Modeling velocity and stress as independent intrinsic attributes allows each Gaussian particle not only to encode deformation but also to evolve continuously over time. By varying the form of the constitutive stress tensor, the Cauchy momentum equation uniformly describes fluid, elastic, and rigid-body dynamics.

3. **Lagrangian Particle Flow Matching**

    - **Function**: Provide motion supervision via optical flow decomposition to guide the velocity and stress fields toward physically plausible solutions.
    - **Mechanism**: *Backward optical flow decomposition* — motion flow is computed backward from $I_{t+1}$ and transformed into the coordinate frame of $I_t$, avoiding the streak artifacts caused by bilinear interpolation in the forward strategy. Gaussian flow $flow_g$ is obtained by weighted summation of 2D displacements of the top-$K$ tracked Gaussian particles; velocity flow $flow_v$ is derived by advecting Gaussian particles using predicted velocities. Both are aligned to the motion flow ground truth: $\mathcal{L}_{\text{LPFM}} = \lambda_g \|flow_g - flow_{gt}\|_1 + \lambda_v \|flow_v - flow_{gt}\|_1$.
    - **Design Motivation**: The Cauchy momentum residual alone is underdetermined, making it difficult to converge to physically plausible velocity and stress predictions. Introducing optical flow as a data-fitting term serves as a boundary condition substitute within the PINN framework, anchoring particle trajectories and providing a clear convergence direction for optimization.

### Loss & Training
The total loss comprises: rendering loss $\mathcal{L}_{\text{renders}} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c\mathcal{L}_{\text{D-SSIM}}$ ($\lambda_c=0.2$), Cauchy momentum residual $\mathcal{L}_{\text{CMR}}$ ($\lambda_{\text{CMR}}=0.1$), and Lagrangian particle flow matching $\mathcal{L}_{\text{LPFM}}$ ($\lambda_{\text{LPFM}}=0.01$). Training runs for 50K iterations (synthetic) / 40K iterations (real). A chunked sampling strategy for the Cauchy momentum residual is adopted to avoid GPU memory explosion — particles are partitioned into chunks, physical residuals are computed and the computation graph released before aggregation. The scaling threshold is adjusted from 0.1 to 0.015 to filter large-scale noisy Gaussians.

## Key Experimental Results

### Main Results (PIDG Physics-Driven Dataset + HyperNeRF)

| Method | PIDG-PSNR↑ | PIDG-SSIM↑ | PIDG-LPIPS↓ | HyperNeRF-PSNR↑ | HyperNeRF-MS-SSIM↑ |
|---|---|---|---|---|---|
| D-NeRF | 23.45 | 0.866 | 0.124 | 25.7 | 0.726 |
| D-3DGS | 29.54 | 0.951 | 0.066 | - | - |
| GaussianPredict | 30.17 | 0.957 | 0.062 | 26.6 | 0.884 |
| Grid4D | 30.32 | 0.956 | 0.061 | 27.3 | 0.899 |
| D-2DGS | 29.23 | 0.944 | 0.061 | 17.7 | 0.509 |
| **PIDG (Ours)** | **30.96** | **0.967** | **0.058** | **27.8** | **0.906** |

### Ablation Study

| Configuration | PIDG-PSNR↑ | PIDG-SSIM↑ | D-NeRF-PSNR↑ | Notes |
|---|---|---|---|---|
| w/o ($\mathcal{L}_{\text{LPFM}} + \mathcal{L}_{\text{CMR}}$) | 30.46 | 0.956 | 42.00 | Static/dynamic decoupling + hash encoding only |
| w/o $\mathcal{L}_{\text{LPFM}}$ | 30.78 | 0.957 | 42.14 | Cauchy momentum residual constraint only |
| **Full model** | **30.96** | **0.967** | - | Complete model (D-NeRF lacks consecutive views for flow matching) |
| Grid4D + $\mathcal{L}_{\text{CMR}}$ | - | - | 42.10 | Plug-and-play gain of +0.10 PSNR |
| SC-GS + $\mathcal{L}_{\text{CMR}}$ | - | - | 41.85 | Plug-and-play gain of +0.20 PSNR |

### Key Findings
- Full flow matching yields the largest improvements on physically complex scenes such as fluid smoke (Dry Ice: 25.34→26.12 PSNR) and elastic collisions (Balls: 32.79→33.31), demonstrating that optical flow supervision is especially effective for physically complex motion.
- $\mathcal{L}_{\text{CMR}}$ as a plug-and-play module consistently improves GaussianPredict, SC-GS, and Grid4D, validating the universality of the physics constraint.
- Removing the stress tensor degrades $\mathcal{L}_{\text{CMR}}$ to $\nabla \cdot \bm{v} = 0$ (continuity constraint); t-SNE visualization shows a marked decrease in the discriminability of dynamic particle features.
- Training efficiency is competitive: 72 minutes / ~85K Gaussians / 250 FPS, with 6.2 GB memory consumption, outperforming most baselines.

## Highlights & Insights
- **Unified Constitutive Law Framework**: The paper derives, via effective field theory (EFT), the unified reduction of the Cauchy momentum equation to rigid bodies, elastic solids, and fluids. The network adaptively learns material properties without manual specification — a theoretically elegant formulation. Appendix A presents the complete derivation chain from scalar Goldstone fields through Lamé parameters to Navier–Stokes equations.
- **Backward Optical Flow Decomposition**: Applying the motion mask to pixel-level motion flow rather than Gaussian flow preserves true displacements without streak artifacts. This is a key improvement over the forward decomposition strategy in MotionGS, with visually notable differences on HyperNeRF real-world data.
- **Lagrangian Particle Identity Inheritance**: Child particles produced during densification inherit the index embedding of their parent, avoiding expensive nearest-neighbor searches to recover identity — a concise and efficient design.
- **Plug-and-Play CMR**: The physics residual module can be inserted as a regularizer into various existing dynamic 3DGS methods at negligible cost, offering strong practical utility.

## Limitations & Future Work
- **Computational Overhead**: Training still requires several hours (72 minutes on an A800) and substantial GPU memory (6.2 GB), leaving a significant gap to real-time reconstruction. The authors plan to develop lightweight feed-forward network architectures to avoid costly optimization loops.
- **Limited Material Models**: The linear constitutive assumption cannot capture complex behaviors such as nonlinear elastoplasticity or viscoelasticity. Hybrid finite-element/particle methods could enrich material modeling capability.
- **Optical Flow Dependency**: The method relies on pretrained optical flow, depth, and segmentation models (UniMatch, Distill Any Depth, SAMv2); scenes with non-consecutive views cannot employ flow matching. Self-supervised motion priors could serve as an alternative.
- **Limitations of 2D Evaluation**: The authors call on the community to establish comprehensive evaluation protocols incorporating geometric, temporal, and physical metrics — e.g., velocity field consistency or stress field plausibility indices.

## Related Work & Insights
- **vs. Grid4D**: Grid4D efficiently models dynamics via 4D hash encoding but lacks physics constraints; adding the material field in PIDG improves PSNR by 0.64 on the PIDG dataset. PIDG uses fewer Gaussians (~85K vs. ~100K) and achieves higher FPS (250 vs. 240), suggesting that the physics constraint also acts as an implicit regularizer.
- **vs. PhysGaussian / PhysDreamer**: MPM-based methods rely on grid discretization and fixed material properties and require RGB-D or multi-view inputs. PIDG circumvents grid discretization via the PINN framework, with fully learnable material parameters and only monocular video as input.
- **vs. MotionGS**: Both employ optical flow supervision, but MotionGS uses a forward decomposition strategy that introduces streak artifacts. PIDG's backward decomposition is more robust, achieving +2.6 PSNR on HyperNeRF with more consistent flow visualization.
- **vs. GaussianFlow**: PIDG reuses GaussianFlow's CUDA rasterizer with efficiency optimizations (merged gradient computation, removal of redundant backward passes) as the basis for Lagrangian particle flow computation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Embedding the continuum-mechanics Cauchy momentum equation into the 3DGS framework is a theoretically novel perspective, though PINNs themselves are already widely adopted.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three evaluation platforms (self-constructed 5-scene physics dataset, D-NeRF, HyperNeRF), detailed ablations including plug-and-play experiments, future prediction, and t-SNE analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are complete (Appendix provides EFT unified constitutive law derivation), though the main text has a somewhat high equation density.
- **Value**: ⭐⭐⭐⭐ — Provides a valuable physics-inductive-bias paradigm for dynamic 3DGS; the plug-and-play $\mathcal{L}_{\text{CMR}}$ module is highly practical; the PIDG dataset also contributes community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[AAAI 2026\] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](../../CVPR2026/3d_vision/retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[ICLR 2026\] DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics](../../ICLR2026/3d_vision/diffwind_physics-informed_differentiable_modeling_of_wind-driven_object_dynamics.md)
- [\[ICML 2026\] PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](../../ICML2026/3d_vision/physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)

</div>

<!-- RELATED:END -->

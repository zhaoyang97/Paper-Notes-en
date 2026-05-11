---
title: >-
  [Paper Note] DGH: Dynamic Gaussian Hair
description: >-
  [NeurIPS 2025][3D Vision][Dynamic hair modeling] This paper proposes Dynamic Gaussian Hair (DGH), a data-driven coarse-to-fine framework that learns hair dynamics via a volumetric implicit deformation model…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Dynamic hair modeling"
  - "3D Gaussian Splatting"
  - "hair dynamics"
  - "novel view synthesis"
  - "digital avatars"
date: 2026-05-08
content_hash: 0edc42cc53f9cd0d
---

# DGH: Dynamic Gaussian Hair

**Conference**: NeurIPS 2025
**arXiv**: [2512.17094](https://arxiv.org/abs/2512.17094)
**Code**: [Project Page](https://junyingw.github.io/paper/dgh)
**Area**: 3D Vision
**Keywords**: Dynamic hair modeling, 3D Gaussian Splatting, hair dynamics, novel view synthesis, digital avatars

## TL;DR

This paper proposes Dynamic Gaussian Hair (DGH), a data-driven coarse-to-fine framework that learns hair dynamics via a volumetric implicit deformation model, and achieves photorealistic novel-view rendering of dynamic hair by combining cylindrical Gaussian representations with a curvature blending strategy.

## Background & Motivation

**Dynamic hair modeling is a core challenge in digital avatars**: The complex motion, occlusion, and light scattering of hair make realistic dynamic hair generation extremely difficult, yet it is critical for AR/VR avatar applications.

**Physics-based simulation methods are not scalable**: Traditional physics-based hair simulation (e.g., XPBD) requires manual per-hairstyle parameter tuning (stiffness, density, damping, etc.) and incurs prohibitive computational costs, making it unsuitable for AR/VR devices.

**Existing NeRF/3DGS avatar methods neglect hair dynamics**: Hair in 3DGS-based head avatars is only rigidly transformed, lacking gravitational effects and secondary motion, which produces unnatural results especially for long hair.

**Hair rendering is extremely expensive**: Photorealistic path-traced rendering must account for translucency, occlusion, and light scattering, typically requiring GPU render farms; real-time engines can only produce approximations.

**Static reconstruction methods cannot handle dynamic appearance**: Methods such as GaussianHair and Gaussian Haircut reconstruct static hair geometry and then animate it with an external physics engine, resulting in a fragmented pipeline that requires additional parameter tuning.

**Real dynamic hair data is scarce**: No real-world dynamic hair capture dataset with accurate strand tracking exists, necessitating the construction of synthetic datasets to advance research.

## Method

### Overall Architecture

DGH consists of two stages. **Stage I** learns the hair dynamics deformation (coarse-to-fine), deforming the static canonical hair under given head motion into a dynamic hair point cloud. **Stage II** optimizes the dynamic hair appearance, representing the deformed hair as cylindrical 3D Gaussians and generating photorealistic images via differentiable rendering. At inference time, deformation is predicted frame-by-frame in a recurrent manner and then rendered.

### Key Design 1: Coarse Stage — Pose-Driven Volumetric Implicit Deformation

- **Function**: Rigidly transforms the canonical hair point cloud to the current head pose, then predicts per-point displacements via an MLP to obtain the initially deformed hair.
- **Mechanism**: The hair and upper-body mesh are respectively encoded as SDF voxels; a 3D U-Net encoder ($\mathcal{E}_\text{hair}$ and $\mathcal{E}_\text{pose}$) extracts and concatenates their features. For each hair point, features are interpolated from the voxel grid and, together with positional encodings and head pose, fed into an MLP to predict displacement $\Delta p$.
- **Design Motivation**: The volumetric implicit representation handles diverse hairstyles (long, curly, ponytail, etc.) in a mesh-free manner without explicit physical simulation. An SDF regularization term ($\mathcal{L}_\text{SDF}$) penalizes points that penetrate the body mesh to ensure physical plausibility. This stage is time-independent and provides a stable initialization for subsequent dynamics refinement.

### Key Design 2: Fine Stage — Temporal Flow Refinement with Cross-Frame Attention

- **Function**: On top of the coarse deformation, predicts 3D flow vectors between adjacent frames to model dynamic effects such as inertia, oscillation, and damping.
- **Mechanism**: The deformed hair voxels from the previous two frames, $V^{t-2}$ and $V^{t-1}$, are encoded and fused via cross-attention ($Q = V^{t-2}$, $K/V = V^{t-1}$). The aggregated temporal features are concatenated with the current pose and the previous frame's flow, then fed into an MLP to predict the current frame's flow.
- **Design Motivation**: The single-frame pose used in the coarse stage alone cannot capture temporally dependent hair motion (e.g., inertial swinging after a head shake). Cross-frame attention leverages information from the preceding two frames to infer motion trends and achieve temporal consistency, replacing the multi-step iterative solvers of traditional physics simulation with a data-driven approach.

### Key Design 3: Cylindrical Gaussian Representation and Strand-Guided Appearance Optimization

- **Function**: Represents each hair strand as a series of cylindrical Gaussian primitives, and optimizes appearance parameters (color SH coefficients, scale, opacity) via a lightweight MLP conditioned on the motion state.
- **Mechanism**: MLP $\mathcal{D}$ takes as input the deformed hair voxel features and positional encodings of position, tangent, and view direction, and outputs refined SH color, scale, and opacity. The strand tangent vector $t$ is used to model anisotropic hair scattering. Final images are rendered via a differentiable tile rasterizer.
- **Design Motivation**: Naively propagating canonical frame colors over time leads to appearance gaps due to dynamic occlusion. By incorporating tangent information, the MLP can learn the anisotropic effects of the hair BSDF, enabling view-consistent photorealistic rendering under motion.

### Key Design 4: Curvature-Adaptive Gaussian Blending

- **Function**: Adaptively blends the colors and opacities of neighboring Gaussians according to local strand curvature, eliminating shading discontinuities in high-curvature regions.
- **Mechanism**: The tangent $t_i$ and curvature $\kappa_i$ of each strand segment are computed and normalized as blending weights $w_i$, which are used to interpolate the SH coefficients and opacity of adjacent Gaussians: $\text{SH}_\text{blended} = \text{SH}_i \cdot (1 - w_i) + \text{SH}_{i-1} \cdot w_i$.
- **Design Motivation**: When strands are discretized with a fixed number of segments, large differences in adjacent tangents at high-curvature regions cause shading discontinuities. Increasing segment density is costly, whereas curvature blending smooths the transitions at zero additional geometric cost and is jointly optimized during training.

## Loss & Training

- **Stage I Coarse**: $\mathcal{L}_\text{total} = \lambda_p \cdot \mathcal{L}_\text{point} + \lambda_\text{SDF} \cdot \mathcal{L}_\text{SDF}$ ($\lambda_p = 1.0$, $\lambda_\text{SDF} = 0.01$). $\mathcal{L}_\text{point}$ is the MSE between predicted and GT displacements; $\mathcal{L}_\text{SDF}$ penalizes points that penetrate the body.
- **Stage I Fine**: $\mathcal{L}_\text{flow} = \text{MSE}(\mathcal{F}_\text{flow}^t, \mathcal{F}_\text{GT}^t)$, supervising consistency between predicted and GT flow.
- **Stage II**: $\mathcal{L} = \lambda_\text{rgb} \cdot L_\text{rgb} + \lambda_\text{ssim} \cdot L_\text{ssim} + \lambda_\text{lpips} \cdot L_\text{lpips}$ ($\lambda_\text{rgb} = 1.0$, $\lambda_\text{ssim} = 0.1$, $\lambda_\text{lpips} = 0.1$).
- Training is performed on a single A100 GPU using Adam (lr = 1e-4), with 200K hair points sampled per iteration.

## Key Experimental Results

### Dynamic Hair Appearance Quality Comparison (Table 1)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| **DGH (Ours)** | **27.01** | **0.871** | **0.127** |
| Gaussian Haircut | 23.07 | 0.847 | 0.131 |
| 3DGS | 20.91 | 0.822 | 0.164 |

DGH achieves the highest PSNR and SSIM across all 5 hairstyle subjects, improving over 3DGS by approximately 6 dB in PSNR.

### Hair Deformation Accuracy Comparison (Table 2)

| Method | L2 Error↓ | Chamfer↓ |
|--------|-----------|----------|
| **DGH (Ours)** | **0.0832** | **0.0266** |
| Rigged hair (rigid transform) | 0.1639 | 0.0424 |

DGH reduces deformation error by approximately 49% (L2) and 37% (Chamfer), effectively modeling gravity and collision effects.

### Ablation Study (Table 3 & 4)

- **Dynamics ablation**: Removing the SDF constraint increases L2 to 0.1269; removing the fine stage increases it to 0.0964; removing cross-attention increases it to 0.0909; the full model achieves 0.0832.
- **Appearance ablation**: Without tangents or blending, PSNR = 20.89; adding tangents yields PSNR = 25.08; the full model achieves PSNR = 28.12, with both tangents and curvature blending each contributing significantly.

## Highlights & Insights

- **Fully data-driven replacement for physics simulation**: Generalizes to diverse hairstyles and motions without manual parameter tuning; end-to-end differentiable training.
- **Decoupled coarse-to-fine design**: Pose-driven initialization combined with temporal flow refinement ensures both stability and capture of high-frequency dynamics.
- **Simple yet effective curvature blending**: Eliminates discretization artifacts at zero additional geometric cost, improving PSNR by approximately 3 dB.
- **Seamless integration with Gaussian avatar frameworks**: Can be directly merged with body Gaussian primitives for full avatar animation.
- **High-quality synthetic dynamic hair dataset**: Contains multiple hairstyles, 10K frames of geometry, and 12K multi-view images; planned for public release.

## Limitations & Future Work

- **Validation on synthetic data only**: Evaluation on real captured data is lacking, and a domain gap exists between synthetic and real hair.
- **Per-hairstyle model training required**: Cross-hairstyle generalization has not been achieved; scaling to large numbers of hairstyles is costly.
- **Recurrent inference may accumulate errors**: Flow prediction errors may compound over long sequences.
- **Fixed lighting conditions**: Training is performed under a single illumination setting; robustness to complex lighting variations is not verified.
- **Requires pre-existing strand geometry**: The input requires a point cloud or strand model of the canonical hairstyle; direct inference from images is not supported.

## Related Work & Insights

- **vs. HVH / NeuWigs**: HVH and NeuWigs also employ neural volumetric representations for dynamic hair, but focus on capture-based reconstruction; DGH focuses on animating hair from a static hairstyle and additionally incorporates 3DGS-based appearance optimization.
- **vs. GaussianHair / Gaussian Haircut**: These methods reconstruct only static hair geometry and rely on an external physics engine for animation, resulting in a fragmented pipeline. DGH learns dynamic deformation and time-varying appearance end-to-end, achieving substantially higher dynamic rendering quality (PSNR +4–6 dB).
- **vs. 3DGS avatar methods** (GaussianAvatars, Rig3DGS, etc.): Hair in these methods only follows the head rigidly; DGH complements them with dynamic non-rigid deformation capability.
- **vs. physics simulation** (XPBD, Quaffure): Physics-based methods offer high accuracy but are non-differentiable, require parameter tuning, and are computationally intensive. DGH achieves comparable results in a data-driven manner while supporting gradient-based optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ — Coarse-to-fine dynamics learning combined with curvature Gaussian blending represents a valuable and novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-hairstyle comparisons and comprehensive ablations, though limited to synthetic data.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-formulated equations, and detailed appendix.
- Value: ⭐⭐⭐⭐ — Provides a scalable data-driven solution for dynamic hair modeling with practical impact on the digital avatar field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)
- [\[CVPR 2026\] CGHair: Compact Gaussian Hair Reconstruction with Card Clustering](../../CVPR2026/3d_vision/cghair_compact_gaussian_hair_reconstruction_with_card_clustering.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](../../ICCV2025/3d_vision/haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[NeurIPS 2025\] IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

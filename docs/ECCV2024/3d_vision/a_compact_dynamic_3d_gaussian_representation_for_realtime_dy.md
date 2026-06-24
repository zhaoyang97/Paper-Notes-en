---
title: >-
  [Paper Note] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This work models the position and rotation parameters in 3DGS as continuous functions of time (Fourier approximation for position, linear approximation for rotation), reducing the storage complexity of dynamic scenes from $O(TN)$ to $O(LN)$. It achieves rendering quality comparable to NeRF-based methods on the D-NeRF, DyNeRF, and HyperNeRF datasets while maintaining real-time rendering speeds over 118 FPS.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Dynamic Scenes"
  - "Fourier Approximation"
  - "Real-time Rendering"
  - "Compact Representation"
date: 2026-05-08
content_hash: 06b4e50d370abd3d
---

# A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2311.12897](https://arxiv.org/abs/2311.12897)  
**Code**: [https://github.com/seoha-kim/Compact3D](https://github.com/seoha-kim/Compact3D)  
**Area**: 3D Vision / Dynamic Novel View Synthesis  
**Keywords**: 3D Gaussian Splatting, Dynamic Scenes, Fourier Approximation, Real-time Rendering, Compact Representation  

## TL;DR
This work models the position and rotation parameters in 3DGS as continuous functions of time (Fourier approximation for position, linear approximation for rotation), reducing the storage complexity of dynamic scenes from $O(TN)$ to $O(LN)$. It achieves rendering quality comparable to NeRF-based methods on the D-NeRF, DyNeRF, and HyperNeRF datasets while maintaining real-time rendering speeds over 118 FPS.

## Background & Motivation
3DGS has demonstrated exceptional performance in static novel view synthesis. However, directly extending it to dynamic scenes (as in Dynamic 3D Gaussians) requires storing all Gaussian parameters at each individual timestep, which results in two severe limitations: (1) memory consumption scales linearly with video length, requiring gigabytes of storage for long sequences; (2) massive multi-view observations are necessary at each timestep to constrain optimization, failing to handle monocular or sparse-view scenarios. On the other hand, NeRF-based dynamic methods (e.g., K-Planes, V4D, TiNeuVox), despite their high rendering quality, exhibit extremely slow rendering speeds (0.1–1.2 FPS) due to the heavy query cost of coordinate-based MLPs, which falls far short of real-time requirements. How to achieve a compact, monocular-friendly representation for dynamic scenes while preserving the real-time execution benefits of 3DGS is the key question.

## Core Problem
The core challenge in dynamic 3DGS is that storing independent Gaussian parameters across timesteps is both memory-inefficient and highly reliant on rich multi-view observations. A mechanism is needed to "compress" and formulate the continuous temporal motions of 3D Gaussians with a small set of parameters, allowing observations from all timesteps to jointly constrain the same global parameter set. This is essential for both compact storage and sparse-view robustness.

## Method
The mechanism is intuitive: physical motion in a dynamic scene is modeled as temporal variations in position and orientation. Thus, the position and rotation of Gaussians are defined as continuous, explicit functions of time, while other physical attributes (scale, color, and opacity) are kept static. This allows the entire dynamic scene to be represented by a single set of shared functional parameters rather than individual parameters for each discrete frame.

### Overall Architecture
Input multi-view/monocular video and camera parameters → Initialize 3D Gaussian points (using SfM for real scenes, random initialization for synthetic scenes) → Static phase optimization (3K iterations, fixing temporal coefficients of position/rotation while training only intercepts and time-invariant attributes) → Dynamic phase optimization (27K iterations, releasing all parameters to learn trajectories) → Obtain compact dynamic Gaussian representation → Given target time $t$ and camera parameters, analytical position and rotation are calculated to perform standard 3DGS splatting for real-time rendering.

### Key Designs
1. **Fourier Approximation on Position**: The 3D center coordinates $x(t), y(t), z(t)$ of each Gaussian are parameterized via Fourier series: $x(t) = w_0 + \sum_{i=1}^{L} w_{2i-1}\sin(2i\pi t) + w_{2i}\cos(2i\pi t)$. Fourier basis functions are naturally suited for periodic or smooth motion and do not overfit at higher frequencies compared to high-degree polynomials. Consequently, each Gaussian requires only $3(2L+1)$ positional parameters, decoupling storage scale from sequence length $T$. On D-NeRF, $L=2$ (15 positional parameters), whereas DyNeRF/HyperNeRF use $L=5$ (33 positional parameters).

2. **Linear Approximation on Rotation**: 3D rotations are represented via quaternions. Due to the unit sphere constraints of quaternions, complex function approximations underperform or fail. Thus, a simple linear formulation is adopted: $q_x(t) = w_0 + w_1 t$. Each Gaussian requires only 8 parameters for rotation (4 quaternion components $\times$ 2 coefficients).

3. **Time-invariant Attributes**: Scales, spherical harmonics (SH) coefficients, and opacity remain static across time, given that physical objects rarely expand or shrink within standard dynamic sequences. This design prevents parameter explosion, and ablation studies confirm that opting for time-varying scaling yields negligible qualitative improvement while substantially inflating storage.

4. **Optical Flow Supervision Loss**: To address temporal ambiguities in monocular/sparse-view scenarios, RAFT is utilized to precompute forward/backward optical flows between neighboring frames as ground truth. Pseudo-optical flows are then analytically derived from the Gaussian representations (by calculating $\mu(t+\Delta t) - \mu(t)$ to obtain 3D scene flow, projecting it into 2D, and applying alpha-blending depth rendering), and optimized using an L1 loss. This loss incurs zero rendering overhead and strengthens temporal consistency, effectively eliminating ghosting artifacts.

5. **Two-stage Optimization Strategy**: The static stage optimizes the scene using all training frames for 3K steps (learning only time-invariant parameters and intercepts) to establish a reasonable spatial prior. Then, all parameters are unlocked for 27K steps of dynamic optimization. This proves more stable than direct end-to-end training.

### Loss & Training
- **Reconstruction Loss**: $\mathcal{L}_{recon} = (1-\lambda)||\hat{I}-I|| + \lambda\mathcal{L}_{D-SSIM}$, where $\lambda=0.2$
- **Flow Loss**: $\mathcal{L} = \mathcal{L}_{recon} + \lambda_{flow}\mathcal{L}_{flow}$, where $\lambda_{flow}=1000$
- The D-NeRF dataset does not utilize the optical flow loss due to discontinuous motion across frames caused by camera teleportation.
- Adaptive density control for Gaussians (splitting, cloning, pruning) follows standard 3DGS strategies.

## Key Experimental Results

| Dataset | Metrics | Ours | K-Planes | V4D | TiNeuVox | 3DGS | D-3DGS |
|--------|------|------|----------|-----|----------|------|--------|
| D-NeRF | PSNR↑ | 32.19 | 31.61 | **33.72** | 30.75 | 20.51 | 17.22 |
| D-NeRF | FPS↑ | **150** | 0.54 | 1.23 | 0.32 | 170 | 173 |
| D-NeRF | Mem↓ | 159MB | 497MB | 1.2GB | 8MB | 50MB | 913MB |
| DyNeRF | PSNR↑ | 31.65 | **31.63** | 28.96 | - | 20.94 | 24.36 |
| DyNeRF | FPS↑ | **118** | 0.31 | 0.11 | - | 109 | 119 |
| DyNeRF | Mem↓ | ~110MB | ~309MB | 1.2GB | - | ~198MB | ~2.3GB |
| HyperNeRF | PSNR↑ | **25.6** | - | 24.3 | 24.3 | - | - |
| HyperNeRF | FPS↑ | **188** | - | 0.15 | 0.14 | - | - |

### Ablation Study
- **Fourier Order $L$**: $L=2$ is optimal on D-NeRF (Mean PSNR 32.19). Higher $L$ values perform better in specific complex motion scenarios (e.g., Jumping Jacks, T-Rex) but degrade the overall average, indicating that the choice of $L$ is scene-dependent.
- **Approximation Functions Comparison**: Fourier ($L=2$) > Spline (5/6) $\approx$ Fourier (but Spline only achieves 91 FPS) > Cubic > Quadratic > Linear. Polynomial approximations tend to underfit at low orders and overfit at high orders.
- **Time-varying Scaling**: Allowing scales to linear-vary with time slightly changes the Mean PSNR from 32.19 to 31.94, but increases storage, which is not worth it.
- **Optical Flow Loss**: Qualitative comparison on DyNeRF reveals that adding the flow loss significantly eliminates ghostly artifacts, leading to more accurate color reconstruction.

## Highlights & Insights
- **Extremely Simple Core Idea**: Directly modeling dynamic scenes with only Fourier approximation for position and linear approximation for rotation without auxiliary neural networks or deformation fields, fully preserving the rendering speed advantages of pure 3DGS.
- **Decoupling Storage from Sequence Length**: The $O(LN)$ complexity renders the model representation size completely independent of the video length, making it highly beneficial for long video sequences.
- **Monocular/Sparse-View Friendliness**: Since parameters are shared across all timesteps, observations from each frame constrain the global model, inherently serving as a regularization in the temporal dimension.
- **Elegant Use of Optical Flow Loss**: By leveraging the explicit analytical position function of Gaussians, pseudo-optical flow can be directly and analytically derived without extra rendering, obtaining temporal constraints at zero cost.
- **Direct Editing Support**: It maintains a pure 3DGS-compatible representation, enabling direct scene composition (the paper demonstrates composition of two dynamic scenes).
- **300$\times$ to 100$\times$ Speedup**: It is 300$\times$ faster than K-Planes and 100$\times$ faster than V4D, achieving 118 FPS even at $1352 \times 1014$ resolution.

## Limitations & Future Work
- **Inability to Model Topological Changes**: All Gaussians persist across all timesteps, making it impossible to represent the appearance, disappearance, or splitting of objects (e.g., fluids). The authors suggest incorporating start and end time parameters to model the "lifespan" of Gaussians.
- **Degradation on Long Sequences**: With fixed parameters, the capacity of Fourier basis functions is bounded. The rendering quality drops for extremely long videos or highly complex motions. Adaptive determination of complexity for each Gaussian is required.
- **Oversimplified Rotation Modeling**: Employing only linear approximation for rotation quaternions leaves insufficient expressiveness for complex rotations (e.g., flipping or spinning objects). Spherical Linear Interpolation (SLERP) or higher-order quaternion approximations could be considered.
- **Inherent Limitations of 3DGS**: It is sensitive to inaccurate camera poses and exhibits worse generalization than continuous neural fields of NeRF. The authors mention that distilling from NeRF to this representation (akin to PlenOctree) could alleviate this.
- **Static Scale Assumption**: The assumption that objects do not expand or contract holds for most scenes, but fails in scenarios such as balloon inflation or paper unfolding.

## Related Work & Insights
- **vs Dynamic 3D Gaussians (Luiten et al.)**: D-3DGS stores positions and rotations independently at each timestep with a space complexity of $O(TN)$. This work compresses it to $O(LN)$ via functional approximation, boosting PSNR on D-NeRF from 17.22 to 32.19 (D-3DGS completely fails in monocular setups) while reducing memory from 913MB to 159MB.
- **vs K-Planes / V4D**: These two NeRF-based methods exhibit slightly higher rendering quality (V4D yields 33.72 PSNR on D-NeRF vs. 32.19 in this work), but are 100$\times$ to 300$\times$ slower in rendering speed (V4D: 1.23 FPS vs. Ours: 150 FPS), failing to support real-time rendering. This work achieves an excellent trade-off between speed and quality.
- **vs 4D Rotor GS / Spacetime GS**: Concurrent works extend 3DGS to dynamic scenes in different ways. 4D-Rotor models local temporal intervals via time slicing, while SpacetimeGS incorporates MLPs. This work maintains a pure 3DGS representation with analytical functions, which is more concise and directly supports scene editing.
- **Generality of Function Approximation**: The idea of using Fourier series to approximate positions can be transferred to other tasks requiring compact temporal modeling (e.g., dynamic point cloud compression, 4D occupancy prediction). The core insight is that "encoding the temporal dimension with basis functions is more efficient than storing parameters per frame."
- **Optical Flow Supervision as a Generic Utility**: Deriving pseudo-optical flow from explicit point representations for temporal regularization is a design pattern applicable to any explicit-point-based dynamic representation method.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of replacing per-timestep storage with Fourier approximation is elegant and effective, though the core technique is not overly complex—highly representative of "why didn't anyone think of this before".
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets cover synthetic/real/monocular/multi-view scenarios. Ablation studies are thorough (covering function types, orders, time-varying attributes, and flow loss), but comparisons against more concurrent 4DGS works are lacking.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rigorous mathematical derivations and highly informative figures and tables.
- Value: ⭐⭐⭐⭐⭐ This is the first work to achieve high-quality real-time rendering of dynamic scenes (118 FPS @ $1352 \times 1014$), providing significant inspiration for subsequent 4DGS research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)
- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [\[ECCV 2024\] NGP-RT: Fusing Multi-Level Hash Features with Lightweight Attention for Real-Time Novel View Synthesis](ngp-rt_fusing_multi-level_hash_features_with_lightweight_attention_for_real-time.md)
- [\[ECCV 2024\] Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)
- [\[CVPR 2026\] Learning Compact 3D Representations from Feed-Forward Novel View Synthesis](../../CVPR2026/3d_vision/learning_compact_3d_representations_from_feed-forward_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->

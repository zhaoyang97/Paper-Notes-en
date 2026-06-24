---
title: >-
  [Paper Note] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity
description: >-
  [CVPR 2025][3D Vision][3D Physics Learning] Ours proposes FreeGave, a general framework for learning 3D scene geometry, appearance, and physical velocity from multi-view dynamic videos. By introducing a learnable physics code for each 3D Gaussian kernel and designing a divergence-free velocity field parameterization, FreeGave achieves accurate future frame extrapolation without relying on PINN losses or target priors.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Physics Learning"
  - "Gaussian Velocity Field"
  - "Divergence-free Velocity Field"
  - "Dynamic 3DGS"
  - "Future Frame Prediction"
date: 2026-05-08
content_hash: b75eeea2c6e3894e
---

# FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity

**Conference**: CVPR 2025  
**arXiv**: [2506.07865](https://arxiv.org/abs/2506.07865)  
**Code**: [https://github.com/vLAR-group/FreeGave](https://github.com/vLAR-group/FreeGave)  
**Area**: 3D Computer Vision / Dynamic Scene Reconstruction  
**Keywords**: 3D Physics Learning, Gaussian Velocity Field, Divergence-free Velocity Field, Dynamic 3DGS, Future Frame Prediction

## TL;DR
Ours proposes FreeGave, a general framework for learning 3D scene geometry, appearance, and physical velocity from multi-view dynamic videos. By introducing a learnable physics code for each 3D Gaussian kernel and designing a divergence-free velocity field parameterization, FreeGave achieves accurate future frame extrapolation without relying on PINN losses or target priors.

## Background & Motivation

**Background**: 3DGS and its dynamic variants (e.g., Deformable 3DGS, 4DGS) have achieved high standards in novel view rendering of dynamic scenes, but most can only interpolate within the training time period and cannot predict the future—because they do not explicitly learn physical properties but merely fit visual observations.

**Limitations of Prior Work**: (1) PINN-based methods convert PDEs into loss functions as soft constraints, but the learned physics in boundary regions is inaccurate, and dense sampling in the spatial-temporal dimension is required, leading to low training efficiency; (2) Explicit physical model methods (such as spring systems, graph neural networks) require target priors (e.g., object masks, types), showing poor generalization, and can usually only handle specific types of motion (fluids or elastic bodies).

**Key Challenge**: The trade-off between the generality and accuracy of physics learning—PINNs are general but inaccurate at boundaries, while explicit physical models are accurate but require domain priors.

**Goal**: Starting from pure RGB multi-view videos, to learn the 3D velocity fields of all objects/parts in the scene without knowing the number, type, or mask of the objects, thereby achieving physically plausible future frame prediction.

**Key Insight**: Treat each 3D Gaussian kernel as a rigid particle, learn a latent "physics code" for it to describe its motion type (such as an abstraction of mass, force, etc.), and then decode the velocity field from the code using a carefully designed divergence-free parameterization to structurally satisfy physical constraints instead of relying on "soft" losses.

**Core Idea**: Replace PINN losses with learnable physics codes and structured divergence-free velocity field parameterization to achieve zero-prior 3D physics learning.

## Method

### Overall Architecture
A three-module pipeline: (1) The Canonical 3DGS module learns the static geometry and appearance of the scene at $t=0$; (2) The Neural Divergence-free Gaussian Velocity module learns a physics code for each Gaussian and decodes a divergence-free 6-DOF velocity field; (3) The Deformation-aided Optimization module introduces an auxiliary deformation field to aid optimization convergence, using the midpoint method to integrate the velocity field, propagate the Gaussian kernel positions, and compare them with multi-view images to generate training signals.

### Key Designs

1. **Physics Code**:

    - Function: Models a latent vector for each Gaussian kernel to abstractly describe its time-invariant motion type/physical properties.
    - Mechanism: Predicts an $L$-dimensional code $\mathbf{z} = f_{code}(\mathbf{p}_0)$ from the canonical position $\mathbf{p}_0$ of the Gaussian using an MLP $f_{code}$. The code $\mathbf{z}$ is shared across all timestamps—it describes the "motion mode" rather than the "state at a certain moment." Compared to learning an independent free code vector for each Gaussian, position-based MLP prediction introduces spatial smoothness regularization, so neighboring Gaussians tend to obtain similar codes.
    - Design Motivation: Mapping solely from position to velocity cannot distinguish adjacent objects with completely different motions (such as a ball rolling on a static table). Introducing the physics code breaks this limitation of "continuous position implies continuous velocity," allowing the network to generate different velocity fields for different motion modes.

2. **Divergence-free Velocity Parameterization**:

    - Function: Ensures that the estimated velocity field satisfies divergence-free physical constraints without requiring PINN losses.
    - Mechanism: Decomposing the velocity of each particle into 6 basic components $\mathbb{V}_t = [v_t^x, v_t^y, v_t^z, w_t^z, w_t^y, w_t^x]$ (3 linear velocities + 3 angular velocities) and a position-dependent basis matrix $\mathcal{B}(\mathbf{p}_t)$, the velocity is $\mathbf{v} = \mathbb{V}_t \cdot \mathcal{B}(\mathbf{p}_t)$. Crucially, $\mathbb{V}_t$ is designed to be independent of position $\mathbf{p}_t$—obtained via $\mathbb{V}_t = f_{neck}(\mathbf{z}) \cdot f_{weight}(t)$: the physics code is decoded into $K$-dimensional motion modes by an MLP, and the timestamp generates a $K \times 6$ weight matrix through another MLP to select/mix the motion modes.
    - Design Motivation: The form of the basis matrix $\mathcal{B}$ guarantees the divergence-free nature of the velocity field (the divergence is identically zero), which is a fundamental physical constraint of rigid body motion. Divergence-free is structurally satisfied rather than acting as a soft penalty like PINN. The multiplicative form of $f_{neck} \cdot f_{weight}$ decouples physics codes from temporal dynamics—the code defines "what motion to perform," and the temporal weight defines "when to perform it."

3. **Deformation-aided Optimization**:

    - Function: Introduces an auxiliary deformation field to help training convergence and connect the velocity field with rendering supervision.
    - Mechanism: The auxiliary deformation field $f_{deform}(\mathbf{p}_0, t, \mathbf{z})$ directly predicts displacement $\delta\mathbf{p}$, rotation $\delta\mathbf{r}$, and scaling $\delta\mathbf{s}$, translating the canonical Gaussian to time $t'$. Starting from $t'$, the velocity field is integrated using interleaved mid-point integration to propagate to $t$, rendered, and compared with ground-truth images. The two paths (deformation field + velocity field) share the physics code $\mathbf{z}$. The deformation field provides a reasonable "starting point" for the velocity field, resolving the issue that direct end-to-end training is difficult to converge.
    - Design Motivation: Directly training the velocity field requires an accurate position -> velocity -> new position chain, but in the initial phase, all parameters are inaccurate, leading to high gradient noise. The deformation field acts as "training wheels," first allowing the model to learn the general motion modes, and then letting the velocity field handle the precise physically-consistent propagation.

### Loss & Training
The loss function is the standard 3DGS $\ell_1 + \ell_{ssim}$ rendering loss. Training consists of two phases: first training the canonical 3DGS, and then jointly training the physics code + velocity module + deformation field. In each iteration, time $t$ and interval $\Delta t$ are sampled, the deformation field is used to locate the Gaussian at $t'$, and then the mid-point method is used to integrate to $t$.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Dynamic Object | NVFi | 27.594 | 0.972 | 0.036 |
| Dynamic Object | DefGS_nvfi | 28.749 | 0.984 | 0.013 |
| Dynamic Object | **FreeGave** | **31.987** | **0.990** | **0.007** |
| Dynamic Indoor | NVFi | 29.745 | 0.876 | 0.204 |
| Dynamic Indoor | DefGS_nvfi | 31.096 | 0.945 | 0.077 |
| Dynamic Indoor | **FreeGave** | **35.019** | **0.966** | **0.051** |
| ParticleNeRF | NVFi | 18.173 | 0.867 | 0.119 |
| ParticleNeRF | DefGS_nvfi | 22.730 | 0.931 | 0.050 |
| ParticleNeRF | **FreeGave** | **26.657** | **0.956** | **0.030** |

### Ablation Study

| Configuration | Extrapolation PSNR | Explanation |
|------|----------|------|
| Full model | Optimal | Full framework |
| No physics code (direct velocity regression) | Decreased | Unable to distinguish adjacent differently-moving objects |
| Independent learnable code (no MLP) | Decreased | Too many parameters, overfitting |
| Direct MLP decoding of $\mathbb{V}_t$ (no multiplicative decomposition) | Decreased | Too flexible, coupling of physics and time |
| No deformation assistance | Hard to converge | Missing "training wheels" |

### Key Findings
- FreeGave significantly outperforms all baselines in all extrapolation tasks—by ~4 dB PSNR over the strongest baseline on Dynamic Indoor.
- On the newly collected FreeGave-GoPro real-world dataset, the extrapolation PSNR reaches 28.094, close to the interpolation level (28.451), demonstrating robust generalization to real-world scenes.
- Visualization of the physics codes reveals meaningful clustering—Gaussians sharing the same motion pattern obtain similar codes, achieving implicit motion segmentation without any labels.
- The extrapolation performance of DefGS (Deformable GS without velocity fields) lags far behind, confirming the necessity of learning a velocity field over merely fitting deformations.

## Highlights & Insights
- **Structured Divergence-free Parameterization over PINN**: Rather than using PDEs as soft constraints, physical laws are "hardcoded" directly into the parameterized form of the velocity field—this represents an elegant design paradigm. A similar structured encoding approach can be attempted for any known physical invariance.
- **Emergent Semantics of Physics Codes**: The physics codes spontaneously cluster motion patterns in unsupervised training and can be directly used for motion segmentation—this indicates that 3D velocity field learning can be used for perception in addition to prediction. It can serve as an emergent representation for downstream tasks (e.g., robotic manipulation, scene understanding).
- **Design Philosophy of Multiplicative Decomposition**: The design of $f_{neck}(\mathbf{z}) \cdot f_{weight}(t)$ explicitly decouples physics from temporal dynamics—the code explains "what motion to perform," while the temporal weight controls "when to start/stop." This underlying physical modeling approach is elegant and concise.

## Limitations & Future Work
- Assuming each Gaussian is a rigid particle—which may not be accurate enough for highly deformable objects (e.g., cloth, fluids).
- The divergence-free assumption of the velocity field is inapplicable to scenes with mass sources/sinks (e.g., smoke dissipation, object appearance/disappearance).
- Training requires multi-view videos; monocular video scenes are not covered.
- The GoPro dataset has only 6 scenes; the scale of real-world data validation is relatively small.
- Improvements: extend to non-rigid motion (introducing strain fields or compressible velocity fields); combine with diffusion models for conditional future generation; apply physics codes to downstream robot planning.

## Related Work & Insights
- **vs NVFi**: NVFi also learns velocity fields but relies on PINN losses, showing poor performance in boundary regions. FreeGave completely bypasses PINN through structured parameterization, significantly outperforming it on all datasets.
- **vs Deformable 3DGS (DefGS)**: DefGS learns deformation fields but not physics, resulting in poor extrapolation performance. FreeGave combines an auxiliary deformation field and a velocity field, possessing both fitting and prediction capabilities.
- **vs PhysGaussian/PAC-NeRF**: These methods require explicit physical models and object mask/type priors, which limits their generality. FreeGave does not require any priors, making it more suitable for complex real-world scenes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The physics code + divergence-free velocity field parameterization is a brand-new design that is both theoretically elegant and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient evaluation across 4 datasets + ablation study, but the scale of the real-world dataset is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear and rigorous technical descriptions, though the heavy use of mathematical notation might slightly affect readability.
- Value: ⭐⭐⭐⭐⭐ Opens up a new paradigm for 3D physics learning; the emergent semantics of the physics code have broad value for downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Recovering Dynamic 3D Sketches from Videos](recovering_dynamic_3d_sketches_from_videos.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)
- [\[CVPR 2025\] Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos](stereo4d_learning_how_things_move_in_3d_from_internet_stereo_videos.md)
- [\[CVPR 2025\] PGC: Physics-Based Gaussian Cloth from a Single Pose](pgc_physics-based_gaussian_cloth_from_a_single_pose.md)
- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] PhysHead: Simulation-Ready Gaussian Head Avatars
description: >-
  [CVPR 2026][3D Vision][Head Avatar Reconstruction] This paper proposes PhysHead—the first method to integrate physics-driven hair dynamics with animatable 3DGS head avatars. It models expressive faces via FLAME mesh + 3DGS, represents hair appearance via strands + 3DGS, drives hair animation through a physics engine, and enables layered optimization of hair and face through VLM-generated bald images.
tags:
  - CVPR 2026
  - 3D Vision
  - Head Avatar Reconstruction
  - 3D Gaussian Splatting
  - Hair Strand Physics Simulation
  - Layered Representation
  - VLM-Assisted
date: 2026-05-08
content_hash: ca68223a9dedd0e2
---

# PhysHead: Simulation-Ready Gaussian Head Avatars

**Conference**: CVPR 2026
**arXiv**: [2604.06467](https://arxiv.org/abs/2604.06467)
**Code**: [https://phys-head.github.io](https://phys-head.github.io)
**Area**: 3D Vision / Head Avatar
**Keywords**: Head Avatar Reconstruction, 3D Gaussian Splatting, Hair Strand Physics Simulation, Layered Representation, VLM-Assisted

## TL;DR
This paper proposes PhysHead—the first method to integrate physics-driven hair dynamics with animatable 3DGS head avatars. It models expressive faces via FLAME mesh + 3DGS, represents hair appearance via strands + 3DGS, drives hair animation through a physics engine, and enables layered optimization of hair and face through VLM-generated bald images.

## Background & Motivation

**State of the Field**: Existing data-driven animatable head avatar methods (e.g., GA, GHA) achieve high-quality rendering but universally treat hair as a rigid shell attached to the head—hair does not move naturally when the head turns. Recent work has begun to separate hair from the head (DELTA, MeGA), yet hair remains static. A concurrent work (HairCUP) supports compositional decomposition but uses unstructured Gaussians that are incompatible with physics simulation.

**Limitations of Prior Work**: (1) In most avatar methods, hair behaves as a rigid body, precluding simulation of dynamic effects such as wind or head shaking. (2) Data-driven hair dynamics methods that learn dynamics from observations cannot generalize to unseen dynamic scenarios—capturing all possible hair motions is infeasible. (3) Strand-level representations used in film and games support physics engine simulation but typically focus only on geometry, with appearance crafted manually by artists and lacking photorealistic expressions.

**Root Cause**: Two requirements must be satisfied simultaneously—(1) photorealistic facial avatars driven by a 3DMM, and (2) strand-level hair representations compatible with physics engines for dynamic hair. No existing method satisfies both.

**Paper Goals**: Construct a head avatar with parametric facial control (expression + head pose) and photorealistic hair animation driven by a physics engine.

**Starting Point**: A layered representation—FLAME + 3DGS for the face and strands + 3DGS for the hair—where the two layers are modeled separately and coupled dynamically through a physics engine.

**Core Idea**: Build a simulatable head avatar using a layered 3DGS representation (FLAME facial layer + strand-based hair layer), and propose VLM-assisted bald image generation along with a color consistency regularization to address occlusion challenges in layered optimization.

## Method

### Overall Architecture
The inputs are multi-view video and a 360° static capture. Optimization proceeds in two stages: (1) **Head layer optimization**: VLM-edited bald training images are generated first, and a FLAME-bound 3DGS facial model is optimized on these images. (2) **Hair layer optimization**: On top of the head layer, strand geometry is initialized via NeuralHaircut, and 3DGS primitives attached to strand segments are optimized for appearance. Hair animation is then achieved through guiding strands + sparse-to-dense propagation within a physics engine.

### Key Designs

1. **VLM-Assisted Bald Image Generation**:

    - **Function**: Generates training data for the head layer and resolves reconstruction of hair-occluded regions such as the ears and neck.
    - **Mechanism**: The Nano-Banana VLM is used to automatically remove hair from the first frame, and multi-view consistent views are selected. Based on these sparse views, a shared FLAME texture map $T \in \mathbb{R}^{2048 \times 2048 \times 3}$ is optimized via differentiable rendering. For each training frame, the texture is used to render an appearance proxy at the corresponding pose, which is then blended with the facial region of the original image via Poisson image editing to produce hair-free training images.
    - **Design Motivation**: Unlike HairCUP, which uses SDS distillation to obtain bald images, the VLM-based approach produces higher-quality results and generalizes across different skin tones. The shared texture map combined with Poisson blending avoids strong boundary artifacts.

2. **Strand-Level 3DGS Appearance Model**:

    - **Function**: Attaches photorealistic appearance to physically simulatable strand geometry.
    - **Mechanism**: Strand geometry is obtained from NeuralHaircut and uniformly resampled to $m=60000$ strands with $n=16$ points each. A 3DGS primitive is assigned to each strand segment, with rotations computed using the Frenet-Serret (TNB) frames. Gaussians are configured as elongated ellipsoids along the strand direction: $\mathbf{g}_{\text{mean}}=(p_1+p_2)/2$, $\mathbf{g}_{\text{scale}}=(\|p_2-p_1\|, k, k)$, $k=0.0001$.
    - **Design Motivation**: Strand-level representation (as opposed to unstructured Gaussians) is directly compatible with physics engines, enabling collision detection, gravity, and wind forces to be applied to the hair.

3. **Color Consistency Regularization**:

    - **Function**: Ensures that invisible inner strands also acquire reasonable colors.
    - **Mechanism**: For each strand $i$ and its neighboring strands $j \in \mathcal{N}(i)$, the color discrepancy is regularized as: $\mathcal{L}_{\text{consistency}} = \sum_{i \in \mathcal{S}} \sum_{j \in \mathcal{N}(i)} \|\mathbf{c}_i - \mathbf{c}_j\|_2^2$. This regularization is activated after 3,000 iterations of outer RGB optimization, allowing colors learned on visible strands to diffuse naturally to hidden strands.
    - **Design Motivation**: Since the photometric loss for the hair layer is computed only within the hair mask, inner or back-facing invisible strands receive random colors, which produce visual artifacts when exposed during animation.

4. **Sparse Guiding Strand Physics Simulation**:

    - **Function**: Enables physically-based hair animation via a physics engine.
    - **Mechanism**: Sparse guiding strands are selected from the dense set to form a hair particle system. Dynamics are simulated using a physics engine (semi-implicit Euler integration + iterative constraint solver). The relative displacements of the sparse guiding strands are propagated to dense strands via k-nearest-neighbor strand skinning with inverse-distance weighting.
    - **Design Motivation**: Directly simulating 60,000 dense strands is computationally prohibitive; sparse guiding strands with interpolation is a standard practice in computer graphics.

### Loss & Training
Head layer loss: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}} + \lambda_{\text{pos}}\mathcal{L}_{\text{pos}} + \lambda_{\text{scaling}}\mathcal{L}_{\text{scaling}}$, with $\lambda=0.2$. Loss is computed only within the facial mask region.

Hair layer loss: $\mathcal{L}_{\text{hair}} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{consistency}}\mathcal{L}_{\text{consistency}}$, where the RGB loss is computed only within the hair mask region. Strand scale and rotation are initialized from TNB frames and fixed; opacity is fixed at 1; only color and spherical harmonic coefficients are optimized.

## Key Experimental Results

### Main Results (Comparison with GaussianHaircut)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| GaussianHaircut | 30.55 | 0.915 | 0.071 |
| Ours w/ $\mathcal{L}_{\text{consistency}}$ | 28.24 | 0.893 | 0.078 |
| **Ours (Full)** | **30.28** | **0.946** | **0.061** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Without color consistency loss | Inner strands exhibit random colors, causing severe artifacts during animation |
| With color consistency loss | Hidden strands acquire reasonable colors, suitable for physics simulation |
| Single-layer representation (no layering) | Skin peeling and poor handling of occluded regions during animation |
| Layered representation (head + hair) | Independent control of expression and hair physics simulation |

Note: The color consistency loss performs best when activated after 5,000 iterations, allowing visible strands to first learn accurate colors before diffusing them to hidden strands.

### Key Findings
- **The core advantage of PhysHead lies not in static reconstruction quality (comparable to GaussianHaircut) but in animation capability**—it is the only method that supports physics-driven hair animation.
- GaussianHaircut strands penetrate the head and learn skin colors, making it unsuitable as an appearance model for physics simulation.
- Compared to GA/GHA: baseline methods achieve comparable quality under small head motion ($t=0$), but the gap widens significantly under large motion (head turns, nodding, wind)—baseline hair always moves as a rigid body.
- Compared to HairCUP: HairCUP's SDS-based approach introduces artifacts in the head layer and tends to produce uniform skin tones; PhysHead's VLM-based approach is more robust and generalizes better.

## Highlights & Insights
- The combination of **VLM + differentiable rendering + Poisson blending** is an elegant pipeline—the VLM performs coarse hair removal, differentiable rendering generates a shared texture as a proxy for occluded regions, and Poisson blending achieves seamless compositing. This pipeline is transferable to other scenarios requiring foreground/background separation.
- **Strand-bound 3DGS**: Assigning elongated Gaussians to strand segments using TNB frames is concise and effective—geometry is defined and fixed by the strands, with only color and spherical harmonic coefficients optimized, greatly simplifying the optimization.
- **Decoupling appearance from simulation**: The approach of separating "hair appearance + physics simulation"—where the physics engine handles motion and 3DGS handles rendering, connected through strand geometry—is a generalizable paradigm that can extend to deformable objects such as clothing and fluids.

## Limitations & Future Work
- Appearance quality depends on the quality of foreground/hair masks; imperfect masks lead to incorrect layer separation.
- Strand geometry is obtained from NeuralHaircut, inheriting its limitations (e.g., poor reconstruction of curly hair).
- The fidelity of physics simulation depends on the quality of propagation from sparse guiding strands to dense strands, which may be inaccurate for complex hairstyles.
- No quantitative evaluation of animation quality is presented (e.g., comparison against real dynamic hair video).
- Training requires multi-view video and 360° static capture, imposing relatively high data acquisition costs.

## Related Work & Insights
- **vs. Gaussian Avatars (GA)**: GA uses FLAME-bound 3DGS for facial control, but hair treated as a 3DMM offset behaves as a rigid body. PhysHead combines GA's facial layer scheme with strand-based physics simulation.
- **vs. GaussianHaircut**: Focuses on strand geometry and appearance reconstruction, but suffers from strand penetration into the head and skin color leakage, and does not support animation. PhysHead resolves these issues through layered optimization.
- **vs. HairCUP**: Also a layered method but uses SDS for bald image acquisition (lower quality, color bias) and unstructured Gaussians (incompatible with physics simulation). PhysHead's VLM-based approach is more robust, and its strand representation is better suited for simulation.
- **vs. Data-driven hair dynamics methods**: Learning dynamics from data cannot generalize to unseen motions. PhysHead employs a general-purpose physics engine that generalizes to arbitrary head poses and external forces.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Integrating physics engine-driven strand dynamics with 3DGS head avatars is pioneering work; the VLM-assisted layered scheme is original.
- **Experimental Thoroughness**: ⭐⭐⭐ — Qualitative comparisons effectively demonstrate animation advantages, but quantitative evaluation is limited (only static reconstruction metrics); quantitative measures of animation quality are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated and the method pipeline is well-structured.
- **Value**: ⭐⭐⭐⭐ — Opens a new direction for simulatable head avatars and represents an important step toward highly realistic human avatars.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)
- [\[CVPR 2026\] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physicsgrounded_articulation_for_sim.md)
- [\[CVPR 2026\] ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars](progressiveavatars_progressive_animatable_3d_gaussian_avatars.md)
- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[CVPR 2026\] STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction](stavatar_soft_binding_and_temporal_density_control_for_monocular_3d_head_avatars.md)

<!-- RELATED:END -->

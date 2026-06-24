---
title: >-
  [Paper Note] D3-Human: Dynamic Disentangled Digital Human from Monocular Video
description: >-
  [CVPR 2025][Human Understanding][Digital Human Reconstruction] D3-Human proposes a method to reconstruct disentangled (garment + body) digital human geometry from a monocular video. By defining an homomorphic Signed Distance Field on the human manifold (hmSDF), it achieves accurate garment-body segmentation of visible regions without 3D garment priors, generating a disentangled template in approximately 20 minutes and supporting virtual try-on and animation applications.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Digital Human Reconstruction"
  - "Garment Disentangled"
  - "Monocular Video"
  - "Implicit-Explicit Hybrid Representation"
  - "hmSDF"
date: 2026-05-08
content_hash: 8f873970887fc5ac
---

# D3-Human: Dynamic Disentangled Digital Human from Monocular Video

**Conference**: CVPR 2025  
**arXiv**: [2501.01589](https://arxiv.org/abs/2501.01589)  
**Code**: [https://ustc3dv.github.io/D3Human/](https://ustc3dv.github.io/D3Human/)  
**Area**: Human Understanding / 3D Reconstruction  
**Keywords**: Digital Human Reconstruction, Garment Disentangled, Monocular Video, Implicit-Explicit Hybrid Representation, hmSDF

## TL;DR

D3-Human proposes a method to reconstruct disentangled (garment + body) digital human geometry from a monocular video. By defining an homomorphic Signed Distance Field on the human manifold (hmSDF), it achieves accurate garment-body segmentation of visible regions without 3D garment priors, generating a disentangled template in approximately 20 minutes and supporting virtual try-on and animation applications.

## Background & Motivation

**Background**: Reconstructing 3D geometry of clothed humans from video has always been a hot topic in computer graphics and computer vision, with wide applications in virtual reality, augmented reality, holographic communication, film production, and game development. Reconstruction from monocular video is of greater practical value due to its simple setup.

**Limitations of Prior Work**: Existing methods can be categorized into two main types, both having distinct limitations: (1) Implicit representation methods (such as SDF, NeRF) can reconstruct high-quality geometry of clothed humans but often reconstruct garments and body as an inseparable whole, preventing editing applications like virtual try-on or animation; (2) Explicit representation methods rely on pre-defined templates (such as SMPL) to separate the body and garments, but are constrained by the expressiveness of the templates, thus failing to handle diverse types of garments. A few attempts to disentangle either rely on 3D scanning templates or use UDF representation, which perform poorly under sparse single-view supervision.

**Key Challenge**: The occlusion of the human body by garments leads to a conflict between "reconstructing details in visible regions" and "plausibly completing invisible regions." Concurrently, obtaining 3D garment-body segmentation information from 2D single-view images is highly challenging.

**Goal**: Reconstruct high-fidelity, disentangled garment and body geometry directly from a short monocular video without using any 3D garment template priors.

**Key Insight**: The authors observe that the watertight surface of a clothed human can be partitioned into garment and body regions by a closed curve. Consequently, if a continuous segmentation function can be defined on the reconstructed human surface, 3D segmentation can be obtained using only 2D human parsing information.

**Core Idea**: Define a "Human Manifold Signed Distance Field" (hmSDF) on the watertight surface of the complete clothed human to classify surface points as either garment or body. Under the supervision of only 2D parsing masks, combined with an explicit SMPL model to complete occluded body regions, hmSDF achieves complete disentangled reconstruction without 3D priors.

## Method

### Overall Architecture

The input of D3-Human is a short video sequence $\{I_t | t=1,...,N\}$ showing a clothed person in motion. The method consists of two stages: (1) **Template Generation Stage**—reconstructing disentangled garment and body templates in the canonical space using hmSDF, supervised only by LBS deformation and 2D masks; (2) **Detailed Deformation Stage**—introducing a non-rigid deformation field and perceptual normal loss to optimize per-frame garment and body details, using two independent MLPs to model the non-rigid deformation of garments and the body respectively.

### Key Designs

1. **Human Manifold Signed Distance Field (hmSDF)**:

    - **Function**: Distinguish garment regions from body regions on the reconstructed watertight clothed human surface.
    - **Mechanism**: While traditional SDF is defined in 3D space, hmSDF is defined on the human surface manifold. Specifically, the complete clothed human is modeled using a hybrid representation of DMTet (Deep Marching Tetrahedra)—a tetrahedral grid $(V_T, T)$ combined with a neural implicit SDF function $s_\eta(x)$. On this surface $S_\eta = \{x | s_\eta(x) = 0\}$, a mapping $\nu: S_\eta \to \mathbb{R}$ is defined, where $\nu(x) < 0$ denotes the body region, $\nu(x) > 0$ denotes the garment region, and $\nu(x) = 0$ represents the boundary line. Unlike the mSDF in GShell, hmSDF considers points on both sides of the boundary (handling both garment and body simultaneously).
    - **Design Motivation**: Directly using implicit UDF to represent garments performs poorly under sparse single-view supervision—UDF is non-differentiable at the zero level set and is highly sensitive to noise. hmSDF leverages the existing watertight surface as a "carrier," reducing 3D segmentation to a classification problem on the surface, which significantly lowers the difficulty.

2. **Region Aggregation**:

    - **Function**: Correct errors in hmSDF segmentation caused by inconsistent 2D human parsing.
    - **Mechanism**: Since human parsing masks for different frames might be noisy and inconsistent across frames, the segmentation results of hmSDF can be fragmented—small pieces of garment fragments might be mixed into correct body regions, and vice versa. The authors propose a correction strategy based on connected component analysis: first, find all connected subgraphs of each category using depth-first search, then determine their true categories based on the number of vertices in each subgraph. Small fragments (connected components with few vertices) are identified as mis-segmentation and merged into the opposite category. This ensures that the final garment and body regions are connected and free of holes.
    - **Design Motivation**: This is a practical engineering design in response to the "weak supervision signal" (only 2D masks), which simply but effectively solves the multi-view inconsistency problem.

3. **Disentangled Non-Rigid Deformation Field**:

    - **Function**: Separately model the subtle deformations of the garment and the body across different frames.
    - **Mechanism**: After template generation, two independent MLPs are used to model the non-rigid deformations of the garment and the body. For frame $t$, the deformation is defined as $x_t = D(x, h_t, E(x); \phi)$, where $x$ is a point in the canonical space, $h_t$ is the latent code for frame $t$, and $E(x)$ is the positional encoding. Then, they are transformed to the observation space via shared LBS (Linear Blend Skinning). Crucially, the non-rigid deformations for garment and body are learned separately—garments have independent wrinkle dynamics, while the body has its own muscle deformation.
    - **Design Motivation**: Garments and the human body follow different motion laws; body motion is driven by the skeleton, whereas garment motion is also affected by gravity, inertia, and collisions. Separated modeling makes their respective deformations more accurate.

### Loss & Training

The training is divided into two stages, with the loss functions structured as follows:

**Template Generation Stage**:
- RGB Loss $\mathcal{L}_{color}$: L1 distance between rendered RGB and input, calculated separately for garment and body pixels.
- Mask Loss $\mathcal{L}_{mask}$: MSE between rendered masks and GT masks extracted by SAM2.
- Eikonal Loss $\mathcal{L}_{eik}$: Ensures the gradient norm of SDF is close to 1.
- Encourage Hole Opening $\mathcal{L}_{hole}$: Encourages hmSDF to form openings at locations like cuffs.
- Regularize Holes $\mathcal{L}_{reg}$: Prevents the openings from becoming too large.

**Detailed Deformation Stage** (additional additions):
- Perceptual Normal Loss $\mathcal{L}_{per}$: Takes the normal map predicted by Sapiens as GT, computing a multi-level perceptual loss based on MobileNetV2.
- Collision Penalty $\mathcal{L}_{collision}$: Prevents garments from penetrating the human body.
- Geometric Regularizations $\mathcal{L}_{n\_consist} + \mathcal{L}_{laplacian}$: Normal consistency + Laplacian smoothing.

### Occlusion-Aware Differentiable Rendering

An important technical detail: when rendering from the same viewpoint, garments might occlude parts of the human body. If garment masks are rendered separately, they would yield results inconsistent with 2D GT (since the GT mask accounts for occlusion while rendering separately does not). D3-Human renders the garment and body meshes labeled with category tags jointly, automatically handling occlusion relationships via rasterization to generate occlusion-aware 2D labels.

## Key Experimental Results

### Main Results (Quantitative Evaluation on Synthetic Data, Chamfer Distance $\times 10^{-3}$)

| Method | Female1 All | Female3 All | Male1 All | Male2 All | Disentangled |
|------|-----------|-----------|---------|---------|--------|
| REC-MV | 1.789 | 1.461 | 1.945 | 1.201 | Garment only |
| BCNet | 5.561 | 5.681 | 4.802 | 2.853 | ✓ |
| DELTA | 1.388 | — | 1.702 | 1.132 | ✓ |
| SelfRecon | 3.420 | 2.249 | 1.310 | 1.454 | ✗ |
| GoMAvatar | 7.319 | 5.058 | 2.382 | 3.163 | ✗ |
| **D3-Human** | **Best** | **Best** | **Best** | **Best** | ✓ |

### Ablation Study

| Configuration | Visual Effect | Explanation |
|------|---------|------|
| hmSDF | Precise garment shape | Accurate cuff openings and garment boundaries |
| Replace with Implicit UDF | Numerous holes and defects | Huge holes in the abdominal region, cuffs fail to open correctly |
| Perceptual Normal Loss | Smooth with details preserved | Perceptual loss maintains global consistency |
| MSE Normal Loss | Rough and noisy | Point-wise loss results in lack of smoothness |
| COS Normal Loss | Similar issues to MSE | Angular loss is also not smooth enough |

### Key Findings

- The core advantage of hmSDF over implicit UDF is that: UDF is non-differentiable at the zero level set, surface extraction is constrained by manifold requirements, and it is highly sensitive to noise (such as mask errors caused by occlusion). hmSDF solves these three issues by defining segmentation on a watertight surface.
- Perceptual Normal Loss is more effective than point-wise MSE/Cosine loss because it focuses on feature-level consistency rather than pixel-level accuracy, producing smoother results while preserving details.
- D3-Human generates a disentangled template in about 20 minutes and processes a full sequence in a few hours, showcasing a significant efficiency improvement compared to more than 24 hours required by REC-MV/DressRecon.
- Direct application values of disentangled reconstruction: virtual try-on (swapping garments between different people) and physics-based animation (simulating garment dynamics using HOOD).

## Highlights & Insights

- **Elegant Definition of hmSDF**: Defining a segmentation distance field on an existing watertight surface reduces 3D segmentation to a surface classification problem, avoiding many difficulties of directly modeling open surfaces with UDF. This strategy of "reconstructing the whole first, then segmenting the parts" can also be applied to other scene decomposition tasks (such as segmenting furniture and walls in indoor scenes).
- **3D Segmentation under 2D Supervision Only**: Achieving 3D garment-body segmentation relying solely on 2D human parsing (provided by SAM2) without requiring any 3D garment priors or 3D annotations. This greatly reduces data requirements.
- **Disentangled Deformation Modeling**: Modeling the non-rigid deformations of garments and the body with independent MLPs is physically more reasonable (as garments have independent wrinkle dynamics) and yields more accurate results.

## Limitations & Future Work

- Currently, it only handles short videos of single-person scenes; its performance in multi-person interaction or long video sequences has not been validated.
- The quality of human parsing directly affects the segmentation accuracy of hmSDF, which may degrade under complex occlusions or unconventional garments.
- It relies on accurate estimation of SMPL parameters (using CLIFF). When SMPL fitting fails, the entire reconstruction will fail.
- For loose garments (such as long skirts), large gaps between the garment and the body may cause collision penalties to fail.
- The current method does not model garment textures and materials, reconstructing only the geometry. It still needs to be extended for complete virtual try-on applications.

## Related Work & Insights

- **vs SelfRecon**: SelfRecon reconstructs clothed humans with SDF but cannot disentangle them, and its geometric details are average. D3-Human achieves better geometric quality while ensuring disentanglement.
- **vs REC-MV**: REC-MV can reconstruct garments but does not include the body, and directly using SMPL for the body leads to interpenetration. D3-Human reconstructs and optimizes both the garment and body simultaneously.
- **vs DELTA**: DELTA is based on SCARF and represents garments with NeRF. It can disentangle but yields poor garment geometry (as NeRF struggles to extract smooth geometry). The mesh representation of D3-Human produces cleaner geometry.
- **vs GoMAvatar**: GoMAvatar uses Gaussians-on-Mesh representation, leading to coarse mesh quality and non-disentangled results. D3-Human outperforms it significantly in both aspects.

## Rating

- Novelty: ⭐⭐⭐⭐ The definition of hmSDF is the core innovation, and performing segmentation on a watertight surface is an elegant insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative and qualitative comparisons are comprehensive, and the ablation studies are convincing, although quantitative evaluation on real data is lacking.
- Writing Quality: ⭐⭐⭐⭐ The method description is clear, though some mathematical derivations could be more concise.
- Value: ⭐⭐⭐⭐ Disentangled reconstruction is a solid demand in practical applications; the demonstrations of virtual try-on and animation are highly convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)
- [\[CVPR 2025\] SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance](semgeomo_dynamic_contextual_human_motion_generation_with_semantic_and_geometric_.md)
- [\[ICLR 2026\] Disentangled Hierarchical VAE for 3D Human-Human Interaction Generation](../../ICLR2026/human_understanding/disentangled_hierarchical_vae_for_3d_human-human_interaction_generation.md)
- [\[ACL 2025\] TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments](../../ACL2025/human_understanding/transbench_breaking_barriers_for_transferable_graphical_user_interface_agents_in.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Human Hair Reconstruction with Strand-Aligned 3D Gaussians
description: >-
  [ECCV 2024][3D Vision][Hair Reconstruction] This paper proposes Gaussian Haircut, which introduces a dual representation of classic hair strands (polylines) and strand-aligned 3D Gaussian primitives. By integrating 3D orientation field lifting and a coarse-to-fine strand fitting optimization strategy, high-fidelity strand-level hairstyles can be reconstructed from multi-view images. The reconstructed hairstyles can be directly used for editing, rendering…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Hair Reconstruction"
  - "3D Gaussian Splatting"
  - "Strand-level Modeling"
  - "Dual Representation"
  - "Coarse-to-fine Optimization"
date: 2026-05-08
content_hash: 4fa95a9e414bf8d7
---

# Human Hair Reconstruction with Strand-Aligned 3D Gaussians

**Conference**: ECCV 2024  
**arXiv**: [2409.14778](https://arxiv.org/abs/2409.14778)  
**Code**: [https://eth-ait.github.io/GaussianHaircut](https://eth-ait.github.io/GaussianHaircut)  
**Area**: 3D Vision / Digital Humans  
**Keywords**: Hair Reconstruction, 3D Gaussian Splatting, Strand-level Modeling, Dual Representation, Coarse-to-fine Optimization  

## TL;DR

This paper proposes Gaussian Haircut, which introduces a dual representation of classic hair strands (polylines) and strand-aligned 3D Gaussian primitives. By integrating 3D orientation field lifting and a coarse-to-fine strand fitting optimization strategy, high-fidelity strand-level hairstyles can be reconstructed from multi-view images. The reconstructed hairstyles can be directly used for editing, rendering, and physical simulation in graphics engines, achieving a speedup of over 10× compared to previous methods.

## Background & Motivation

**Background**: High-fidelity 3D human hair reconstruction is one of the most challenging sub-problems in digital human modeling. The industry standard is a strand-based representation—describing the geometry of each hair strand using 3D polylines, which can be directly imported into graphics engines like Unreal Engine for rendering and physical simulation. Although recent methods based on 3D Gaussian Splatting (3DGS) have made breakthrough progress in human head reconstruction, these methods model hair as a renderable visual surface using unstructured Gaussian primitives, failing to extract strand structures suitable for simulation.

**Limitations of Prior Work**: (1) **The unstructured-structured gap**—While 3DGS Gaussian primitives render well, they are essentially an unorganized point cloud that cannot represent the internal topological structure of the hair (e.g., which primitives belong to the same strand, the growth direction of the strand). (2) **Noisy orientation maps**—Image-based hair modeling relies on 2D orientation maps to infer 3D strand directions. Conventional methods use Gabor filters to calculate orientation maps from RGB images, but these maps are inherently noisy, severely affecting the accuracy of subsequent strand fitting. (3) **Visible parts vs. internal structures**—Images only observe the outer surface of the hair, but the internal structure of the real hairstyle (how strands grow from the scalp and intertwine internally) is crucial for physical simulation. Although previous methods (such as Neural Haircut) introduce diffusion priors to infer the internal structure, their reconstruction speed is extremely slow.

**Key Challenge**: Strand-level reconstruction requires a structured representation (polylines) to support downstream applications, but structured representations lack differentiable rendering capability to exploit photometric constraints. Conversely, 3DGS possesses excellent differentiable rendering capabilities but lacks structural organization. Combining the advantages of both is the core challenge.

**Goal**: Design a unified dual representation scheme combining hair polylines and 3D Gaussian primitives, enabling hair reconstruction to leverage photometric constraints from differentiable rendering for geometric accuracy while outputting structured hair strands ready for graphics engines.

**Key Insight**: The authors present a key observation: when learning elongated structures (such as hair), the Gaussian primitives in 3DGS naturally stretch along the strand direction—the maximum variance direction of the covariance matrix aligns with the hair strand direction. This implies that 3DGS implicitly learns a 3D orientation field. Explicitly extracting this orientation field allows "lifting" unstructured Gaussians into a structured hair representation.

**Core Idea**: Bind Gaussian primitives to hair strand segments to form strand-aligned Gaussians, retaining the structured representation of hair strands while acquiring differentiable rendering capabilities, thus achieving the best of both worlds.

## Method

### Overall Architecture

The method consists of two stages. **First Stage (3D Line Lifting)**: A modified version of 3DGS is used to reconstruct the scene from multi-view images while performing camera parameter optimization. The covariance matrices of 3DGS are used to extract the 3D orientation field and generate high-quality orientation maps. **Second Stage (Hair Strands Fitting)**: Based on the orientation field and rendering results obtained from the first stage, a coarse-to-fine optimization strategy is applied to fit strand-level hairstyles. The coarse stage optimizes the global structure of the hairstyle in a latent space, while the fine stage optimizes the fine geometry of individual strands in explicit coordinate space. Throughout this process, strand-aligned 3D Gaussians serve as a bridge for differentiable rendering.

### Key Designs

1. **3D Line Lifting with Unstructured Gaussians**:

    - Function: Reconstructs the 3D geometry of the scene from multi-view images and extracts a denoised hair orientation field.
    - Mechanism: Introduces two key extensions to standard 3DGS. First, BARF-style 6-DoF camera parameter optimization is introduced as a residual to the SfM initial estimates, resolving SfM inaccuracy issues in hair scenes. Second, additional learnable attributes are assigned to each Gaussian primitive: a hair segmentation label $l$ and an orientation confidence $\tau$. The maximum variance direction of the covariance matrix $\Sigma = RSS^TR^T$ is utilized as the 3D strand direction $\beta_i$. The orientation supervision loss is designed as $\mathcal{L}_{dir} = \sum_p \tau_p \min\{d(\beta_p, \hat{\beta}_p), d(\beta_p, \hat{\beta}_p) \pm \pi\} - \log \tau_p$, where $d$ is the angular difference and $\hat{\beta}_p$ is the ground-truth orientation calculated by Gabor filters. Introducing the confidence $\tau_p$ allows the model to automatically downweight areas with high orientation uncertainty (e.g., hair boundaries, tangled regions), providing more robustness than directly enforcing alignment with noisy Gabor directions.
    - Design Motivation: Conventional methods directly employ Gabor filter orientation maps for 3D lifting, which are highly noisy. The covariance matrix of 3DGS naturally encodes local geometric orientations; utilizing this property to lift orientation maps reduces the average angular error by approximately 1° compared to Gabor filters and significantly improves performance in poorly-lit and tangled hair regions.

2. **Strand-Aligned 3D Gaussians**:

    - Function: Unifies hair polylines and 3D Gaussian primitives into a structured, differentiably renderable representation.
    - Mechanism: For each hair strand $S_k = \{p_l^k\}$, a 3D Gaussian is placed on each line segment formed by adjacent control points $p_l^k$ and $p_{l+1}^k$. The parameters of this Gaussian are fully determined by the line segment it resides on: the scaling vector $s_l^k = \{\frac{1}{2}\|p_{l+1}^k - p_l^k\|_2, \epsilon, \epsilon\}$ (stretched along the line segment with orthogonal directions set to a minimal value $\epsilon$), the rotation quaternion aligning the x-axis with the line segment direction, and opacity set to 1. Each Gaussian also contains trainable spherical harmonics coefficients $f_l^k$ for color modeling. Consequently, gradients on hair strand coordinates can backpropagate through the Gaussian rendering process—modifying hair control points changes the position, scaling, and orientation of the Gaussians, which in turn changes the rendering result, forming a complete differentiable path from pixels to hair geometry.
    - Design Motivation: Previous methods (such as Neural Haircut) render hair strands as meshes for differentiable rendering, but this loses high-frequency details and is slow. Strand-aligned Gaussians leverage the highly efficient rasterization pipeline of 3DGS, resulting in extremely high rendering efficiency, and the parameter degrees of freedom of individual Gaussians are sufficient to capture variations in hair strand thickness.

3. **Coarse-to-Fine Strands Fitting**:

    - Function: Progressive hairstyle reconstruction from initialization to fine geometry.
    - Mechanism: The hairstyle is represented as a scalp texture map $H$, where each texel stores the 3D polyline of a hair strand. Due to the extremely high degrees of freedom, direct optimization is prone to collapse. **Coarse Stage**: A pre-trained encoder $\mathcal{E}$ and decoder $\mathcal{G}$ are used to map the hairstyle into a low-dimensional latent space $Z = \mathcal{E}(H)$, which is optimized. Due to memory constraints, only 1,000 guide strands $H'$ are decoded at each step, which are then upsampled to 10,000 strands via K-Nearest Neighbor (KNN) interpolation for rendering. This upsampling trick is key to making the photometric loss effective during the coarse stage. **Fine Stage**: 30,000 complete strands are decoded from the latent map, the decoder is frozen, and the control points of each strand are optimized directly in the 3D coordinate space. In both stages, diffusion-model-based SDS regularization is used to ensure the realism of internal hair strand structures.
    - Design Motivation: Latent space constraints provide a strong hairstyle prior, preventing illogical hair shapes in early optimization stages. Explicit coordinate optimization pursues fine geometric details building upon the prior guidance. The two stages complement each other.

### Loss & Training

First Stage: $\mathcal{L}_{gaussian} = \mathcal{L}_{rgb} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{dir}\mathcal{L}_{dir}$, encompassing reconstruction of color, segmentation masks, and orientation supervision.

Second Stage: $\mathcal{L}_{strand} = \mathcal{L}_{rgb} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{dir}\mathcal{L}_{dir} + \lambda_{sds}\mathcal{L}_{sds}$, with the addition of the SDS loss based on the diffusion prior. The loss weights are set to $\lambda_{seg} = \lambda_{dir} = 10^{-1}$ and $\lambda_{sds} = 10^{-2}$. The total training time is approximately 6 hours (RTX 4090).

## Key Experimental Results

### Main Results

**Qualitative comparison on real-world scenes** (compared with Neural Haircut):

| Metric | Neural Haircut | Gaussian Haircut (Ours) |
|------|---------------|------------------------|
| Reconstruction Quality | Visible surface mostly accurate, internal structure coarse | Both inner and outer structures are more precise, with tangled areas significantly improved |
| Optimization Time | ~60 hours | **~6 hours** |
| Speedup | 1× | **>10×** |
| Simulatability | Feasible, but dynamics lack realism | Feasible with more realistic dynamics |

**Quantitative comparison on synthetic scenes** (orientation map error):

| Method | Avg. Angular Error ↓ |
|------|-----------|
| Gabor filter | 8° |
| Ours (3D Lifting) | **7°** |

### Ablation Study

| Configuration | Reconstruction Quality | Description |
|------|---------|------|
| Full model | Best | Complete method |
| w/o fine fitting | Significant degradation | Lacks fine geometric optimization; hair profiles are rough |
| w/o synthetic renders | Degradation | Accuracy drops under poor lighting and in tangled regions |
| w/o strands upsampling | Severe degradation | Photometric loss in the coarse stage fails; fine stage fails to converge |
| w/o $\mathcal{L}_{dir}$ | Degradation | Missing orientation constraints lead to chaotic hair orientations |
| w/o $\mathcal{L}_{sds}$ | Internal structure degradation | Diffusion prior is crucial for the structure of invisible regions |
| w/o $\mathcal{L}_{rgb}$ | Appearance mismatch | Color loss is the primary driver for outer-surface accuracy |

### Key Findings

- **The strand upsampling trick is crucial**—without it, the fine stage completely fails to converge (especially in dense hair regions), because the 1,000 guide strands in the coarse stage cannot cover enough image area to calculate an effective photometric loss.
- **The transition from the coarse stage to the fine stage** is key to the success of the method—using only the coarse stage (latent space optimization) yields a generally reasonable hairstyle but lacks edges and details; using only the fine stage (direct coordinate optimization) cannot converge from a random initialization.
- **Orientation loss and SDS loss are complementary**—$\mathcal{L}_{dir}$ primarily constrains the hair direction on visible surfaces, while $\mathcal{L}_{sds}$ primarily constrains the hair distribution in invisible internal areas.
- **Camera optimization is particularly important for hair scenes**—Standard SfM (COLMAP) suffers from large localization errors in hair regions, but rendering quality is significantly improved after adding BARF-style camera optimization.
- Reconstruction results can be directly imported into Unreal Engine for physical simulation, yielding reasonable dynamics thanks to the hair strands being attached to the FLAME head model and having realistic internal structures.

## Highlights & Insights

- **The clever unification of the dual representation** is the most core contribution of this paper—strand-aligned Gaussians maintain the structural integrity of hair (each Gaussian is strictly bound to a single segment of a hair strand) while remaining completely compatible with the highly efficient differentiable rendering pipeline of 3DGS. This "structured unstructured representation" paradigm can be generalized to the reconstruction of other elongated structures, such as pipelines, cables, and plant branches.
- **Exploiting the orientation information implicitly encoded in the Gaussian covariance matrix** is an ingenious observation—while others use 3DGS to render color, this paper extracts the orientation field as a "free" geometric byproduct.
- **A 10× speedup** is achieved without sacrificing quality, making strand-level reconstruction scalable from a research tool to actual production.

## Limitations & Future Work

- **Difficulty in modeling curly hair**—The hair strand prior is based on a root-to-tip growth model, which has insufficient expressive power for highly curly hairstyles (e.g., afros, dreadlocks). A more flexible hair prior model is needed.
- **No support for complex braided structures**—Hairstyles with intertwined topologies, such as braids or buns, are out of scope for the current method.
- **Unstructured lighting conditions**—While the method does not require studio lighting, performance has not been fully evaluated under extreme backlighting or specular conditions.
- The 6-hour reconstruction time is still relatively long for production; introducing a feed-forward initialization method could be considered to accelerate convergence.
- Currently, only static hairstyle reconstruction is supported; dynamic hairstyles (e.g., hair blowing in the wind) require additional temporal modeling.

## Related Work & Insights

- **vs Neural Haircut**: Prior work to this paper, which uses NeuS + Chamfer loss for hair strand reconstruction. This paper replaces two key components (orientation map lifting and differentiable rendering) with Strand-aligned Gaussians, achieving a 10× speedup and improved quality.
- **vs GaussianHair (Luo et al.)**: Concurrent work that also uses 3DGS for hair reconstruction but requires a controlled studio capture environment. This paper supports unconstrained capture and introduces a camera optimization scheme.
- **vs Neural Strands (Rosu et al.)**: Provides the hair encoder-decoder prior used in this paper. The contribution of this paper lies in combining it with 3DGS to improve reconstruction accuracy and speed.
- **Insights**: The design paradigm of strand-aligned Gaussians—binding Gaussian primitives to structured geometric primitives—can be extended to scenarios requiring structured representations, such as tree modeling and pipeline reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Strand-aligned Gaussians is a highly original representation method that cleverly unifies two seemingly incompatible paradigms.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on both real and synthetic scenes is comprehensive with ablations covering all components, but quantitative metrics are limited (mainly angular error of orientation).
- Writing Quality: ⭐⭐⭐⭐⭐ Precise problem definition, clear methodological intuition, and extremely high-quality figures.
- Value: ⭐⭐⭐⭐⭐ First to achieve high-quality hair reconstruction directly usable in graphics engines, holding direct value for digital human and film industries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CGHair: Compact Gaussian Hair Reconstruction with Card Clustering](../../CVPR2026/3d_vision/cghair_compact_gaussian_hair_reconstruction_with_card_clustering.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)
- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)
- [\[ECCV 2024\] Spring-Gaus: Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians](reconstruction_and_simulation_of_elastic_objects_with_spring-mass_3d_gaussians.md)

</div>

<!-- RELATED:END -->

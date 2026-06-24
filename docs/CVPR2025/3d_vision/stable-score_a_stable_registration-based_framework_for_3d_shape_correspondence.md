---
title: >-
  [Paper Note] Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence
description: >-
  [CVPR 2025][3D Vision][3D Shape Correspondence] Stable-SCore revisits the "registration-correspondence" paradigm by leveraging 2D foundation models (Stable Diffusion + DINO) to establish robust 2D character correspondences. It proposes a semantic-flow-guided registration method (based on Neural Jacobian Fields) to bridge 2D correspondence and 3D deformation via differentiable rendering, significantly outperforming functional map-based methods on non-isometric character shape…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Shape Correspondence"
  - "Registration Methods"
  - "Neural Jacobian Fields"
  - "2D Foundation Models"
  - "Non-isometric Shapes"
date: 2026-05-08
content_hash: 5d1799dd318db588
---

# Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence

**Conference**: CVPR 2025  
**arXiv**: [2503.21766](https://arxiv.org/abs/2503.21766)  
**Code**: [https://haolinliu97.github.io/Stable-Score](https://haolinliu97.github.io/Stable-Score)  
**Area**: 3D Vision  
**Keywords**: 3D Shape Correspondence, Registration Methods, Neural Jacobian Fields, 2D Foundation Models, Non-isometric Shapes

## TL;DR

Stable-SCore revisits the "registration-correspondence" paradigm by leveraging 2D foundation models (Stable Diffusion + DINO) to establish robust 2D character correspondences. It proposes a semantic-flow-guided registration method (based on Neural Jacobian Fields) to bridge 2D correspondence and 3D deformation via differentiable rendering, significantly outperforming functional map-based methods on non-isometric character shape correspondence tasks.

## Background & Motivation

**Background**: 3D shape correspondence aims to establish point-to-point mappings between different shapes. It is a fundamental task in computer vision and computer graphics, directly supporting applications such as re-topology, texture transfer, skeleton transfer, and shape interpolation. Currently, there are two major paradigms: **functional map-based methods** and **registration-correspondence methods**. Functional map methods (e.g., FMNet, ULRSSM) have dominated in recent years, formulating point mapping as a mapping in the functional space.

**Limitations of Prior Work**: Functional map methods perform excellently in "controlled scenarios" (with minor shape and topological differences) but experience severe performance deterioration in real-world, non-isometric scenarios (such as between artist-created characters or AI-generated 3D models). The root cause is that functional mapping relies on strictly aligned low-rank basis functions, an assumption violated by non-isometric deformations. On the other hand, although traditional registration methods are better suited for non-isometric scenarios, they face two fatal challenges: (1) unstable deformation processes (distortion and artifacts); (2) the requirement of high-quality initial 3D correspondences or meticulous pre-alignment, which are extremely difficult to obtain in non-isometric scenarios.

**Key Challenge**: Under non-isometric settings, the mathematical assumptions of functional maps do not hold, while registration methods lack stable deformation models and reliable initial correspondences. Both paradigms suffer from structural defects.

**Goal**: To fix the two core deficiencies of the registration-correspondence paradigm—replacing unstable deformation models with Neural Jacobian Fields and replacing unreliable 3D initial correspondences with 2D foundation models—thereby revitalizing the potential of this paradigm in non-isometric scenarios.

**Key Insight**: It is observed that 2D vision foundation models (such as Stable Diffusion and DINOv2) exhibit remarkable generalization capabilities in image-level dense correspondence. This allows for "cross-modal transfer" to 3D tasks: first establishing correspondences on 2D rendered images and then propagating the 2D correspondence guidance to 3D deformation via differentiable rendering.

**Core Idea**: (1) Train a lightweight 2D correspondence model using Stable Diffusion + DINO features; (2) use Neural Jacobian Fields as a stable deformation engine; (3) guide 3D registration using 2D semantic flows as supervision via differentiable rendering to iteratively optimize the deformation from the source mesh to the target mesh.

## Method

### Overall Architecture

1. **Multi-view Image Rendering**: Render the source and target meshes into multi-view RGB or normal maps under fixed camera poses.
2. **2D Correspondence Estimation**: Use a Stable Diffusion + DINO feature extractor combined with a lightweight adapter network to establish 2D semantic flow maps between source and target images under each view.
3. **Semantic Flow Guided Registration**: Iteratively optimize the per-face Jacobian transformation matrix via Neural Jacobian Fields, rendering the deformed source mesh into a forward flow map through differentiable rendering, and employing 2D semantic flows as supervision.

### Key Designs

1. **2D Character Correspondence Model**:

    - **Function**: Establish robust 2D pixel-wise correspondences between the rendered source and target images.
    - **Mechanism**: Input source and target images into Stable Diffusion (extracting intermediate UNet features of size $120\times120$) and DINOv2 (outputting $60\times60$ features), concatenate them, and map them to a shared embedding space via a lightweight adapter network. Establish 2D correspondences via nearest neighbor search to generate semantic flow maps. The training data includes 3D correspondence datasets (3DBiCar, SURREAL) and 2D correspondence datasets (SPair-71K).
    - **Design Motivation**: 2D foundation models pre-trained on billions of data samples exhibit generalization capabilities far exceeding 3D methods. Reducing the 3D problem to 2D via rendering bypasses the scarcity issue of large-scale 3D datasets.

2. **Geometry-Grounded Negative Loss**:

    - **Function**: Address self-similarity issues in 2D correspondences (e.g., similar features on left and right hands).
    - **Mechanism**: Utilize the pre-computed geodesic distance matrix $\mathbb{G}$ on parametric models (RaBit, SMPL) to penalize feature similarity for point pairs $(p,q)$ whose geodesic distance exceeds a threshold: $\mathcal{L}_{neg} = \sum_{(p,q), \mathbb{G}(p,q) > th} \|\mathcal{X}_{src}(\Pi(p,C_s)) \cdot \mathcal{X}_{tgt}(\Pi(q,C_t))\|_2$. The total training loss is $L_{2D} = L_{con} + \lambda_{neg} L_{neg}$.
    - **Design Motivation**: Standard contrastive loss cannot distinguish between geometrically remote but semantically similar parts. Utilizing the geodesic priors of 3D parametric models to guide "hard negative samples" effectively resolves self-similarity ambiguities.

3. **Semantic Flow Guided Registration**:

    - **Function**: Supervise 3D mesh deformation with 2D semantic flows to achieve stable registration.
    - **Mechanism**: Use Neural Jacobian Fields (NJF) as the deformation model, optimizing the per-face transformation matrix $\tilde{J}_i \in \mathbb{R}^{3 \times 3}$. NJF obtains face transformations $J_i = \tilde{J}_i \mathcal{B}_i$ by projecting onto the tangent space of the Jacobian, and then solves Poisson's equation to obtain the deformed vertex positions $\Phi^* = L^{-1} \mathcal{A} \nabla^T J$. Projecting the deformed vertices to 2D yields the 2D displacements, which are rendered as the forward flow map $\tilde{S}^i$ as color, supervised by the semantic flow $S^i$: $\mathcal{L}_{flow} = \sum_i \|\tilde{S}^i - S^i\|_1$.
    - **Design Motivation**: NJF parameterizes deformation in the tangent space, which is more stable than directly optimizing vertex displacements. Bridging 2D supervision and 3D deformation via differentiable rendering avoids the reliance of traditional methods on high-quality initial 3D correspondences.

### Loss & Training

The total loss during the registration stage is:
$$\mathcal{L} = \lambda_{flow}\mathcal{L}_{flow} + \lambda_{cd}\mathcal{L}_{cd} + \lambda_{normal}\mathcal{L}_{normal} + \lambda_{identity}\mathcal{L}_{identity} + \lambda_{shear}\mathcal{L}_{shear}$$

- $\mathcal{L}_{flow}$: Semantic flow guidance loss (weight 10.0)
- $\mathcal{L}_{cd}$: Chamfer Distance geometric alignment loss (weight 1.0)
- $\mathcal{L}_{normal}$: Normal consistency loss (weight 0.1)
- $\mathcal{L}_{identity}$: Identity-preserving term $\|\tilde{J}_i - I_3\|_F$, with weight linearly decaying from 0.01 to 0.0001
- $\mathcal{L}_{shear}$: Shear-resistant term $\|\tilde{J}_i - \tilde{J}_i^{rot}\|_F$, where the rotation component is extracted via polar decomposition (weight 0.1), encouraging rigid transformations and suppressing shear deformations

Optimization takes 5000 iterations, taking approximately 2 minutes for a 10K-face mesh and about 4 minutes for a 40K-face mesh.

## Key Experimental Results

### Main Results

**Cross-domain settings (trained on 3DBiCar+SURREAL, tested on other datasets), geodesic error $\times 100 \downarrow$:**

| Method | Supervision Type | FAUST | CharW | DT4D-H std | DT4D-H hard |
|------|---------|-------|-------|------------|-------------|
| ULRSSM | Unsupervised | 2.09 | 32.6 | 28.2 | 32.0 |
| Hybrid ULRSSM | Unsupervised | 1.55 | 33.5 | 15.5 | 22.1 |
| Diff3f | Zero-shot | 12.0 | 12.5 | 24.0 | 22.7 |
| SmoothShell | Zero-shot | 2.93 | 11.6 | 13.6 | 12.4 |
| Ours (Zero-shot) | Zero-shot | 5.60 | 3.48 | 19.9 | 14.1 |
| **Ours (Normal)** | **Supervised** | **1.83** | **2.61** | **4.23** | **4.12** |

### Ablation Study

| Configuration | FAUST | CharW (RGB) | DT4D-H hard | Explanation |
|------|-------|-------------|-------------|------|
| Baseline (Vertex displacement + Laplacian smoothing) | 2.88 | 3.44 | 6.04 | No NJF or special losses |
| + Neural Jacobian Field | 2.32 | 2.69 | 4.58 | NJF contributes the most |
| + shear-resistant loss | 2.07 | 2.59 | 4.50 | Shear-resistant loss continuously improves results |
| + geo-grounded neg loss (full) | **1.83** | **2.57** | **4.12** | Geometric negative loss yields further gains |

**Necessity of the feature adapter in the 2D correspondence model:**

| Configuration | FAUST | CharW | DT4D-H hard |
|------|-------|-------|-------------|
| Diff3f (Zero-shot features) | 12.0 | 12.5 | 22.7 |
| Ours (Zero-shot, without adapter) | 5.60 | 3.48 | 14.1 |
| Ours (With adapter) | **1.83** | **2.61** | **4.12** |

### Key Findings

- **Functional map methods collapse completely in non-isometric scenarios**: ULRSSM scores only 2.09 on FAUST (isometric) but surges to 32.6 on CharW (non-isometric). Stable-SCore secures 2.61 on CharW, achieving a $>10\times$ gap.
- **NJF is the most critical component**: From baseline to +NJF, the DT4D-H hard error decreases from 6.04 to 4.58, representing the largest drop.
- **Training the adapter is crucial**: Under the zero-shot mode, the DT4D-H hard error is 14.1, which decreases to 4.12 with the adapter—an improvement of nearly 4 times.
- **Semantic flow supervision outperforms 3D correspondence supervision**: Direct supervision with 3D correspondences (Diff3f style) yields an error of 4.56, while 2D semantic flow supervision yields 4.12. 2D supervision preserves more semantic details.
- **Subtle utility of the shear-resistant loss**: Too large of an identity-preserving term hinders large pose variations, whereas too small of a term causes non-smoothness. The shear-resistant term encourages rotation to suppress shear, maintaining deformation degrees of freedom while enhancing stability.

## Highlights & Insights

- **The bridge design of "2D foundation model $\rightarrow$ 3D task"** is exceptionally elegant: leveraging the generalization ability of foundation models by reducing 3D to 2D via rendering, and then back-propagating 2D supervision to 3D via differentiable rendering to form a closed loop. This serves as an elegant paradigm for utilizing 2D pre-trained knowledge to resolve the scarcity of 3D data.
- **Geometry-grounded negative loss** addresses a long-standing core problem in the correspondence field—self-similarity (e.g., left vs. right hands, front vs. hind legs)—by exploiting the geodesic priors of parametric models to provide geometry-aware hard negatives.
- The introduction of the **CharW benchmark** is also highly valuable—it provides the first wild scenario test set containing both artist-created and AI-generated characters, propelling research in non-isometric correspondence.

## Limitations & Future Work

- **Coarse rotational alignment required**: The source and target meshes must share a rough orientation alignment, making it unable to handle extreme angular rotations.
- **Mesh face limit**: The optimization time of NJF scales linearly with the number of mesh faces (~2 minutes for 10K faces, ~4 minutes for 40K faces), which might be too slow for high-resolution meshes.
- **Occlusion issue in 2D correspondence**: Multi-view rendering alleviates but does not fully resolve occlusion—completely occluded areas remain invisible in all views.
- **Dependence on parametric models for training**: The geometric negative loss requires a pre-computed geodesic distance matrix, limiting its applicability to categories with available parametric models (such as humans and animals).
- Future directions: Exploring 3D generative priors (e.g., 3D diffusion) to replace the 2D rendering pipeline, and self-supervised training without parametric models.

## Related Work & Insights

- **vs Diff3f**: Diff3f back-projects 2D foundation features onto the 3D surface and then solves for continuous mapping using functional maps. Stable-SCore argues that the low-rank projection of functional maps loses semantic information; thus, it bypasses functional maps and directly uses semantic flows to guide registration.
- **vs SmoothShell**: SmoothShell is a traditional registration method, yielding a zero-shot CharW error of 11.6 vs. 2.61 for Ours. The core difference lies in Stable-SCore's leverage of semantic guidance provided by 2D foundation models.
- **vs Functional Map Series (ULRSSM, GeoFMap)**: These methods are powerful in isometric scenarios but structurally fail in non-isometric settings. Exploring the complementarity of these two paradigms remains an interesting avenue.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The pipeline design of "2D foundation model $\rightarrow$ differentiable rendering $\rightarrow$ 3D registration" is highly original. The shear-resistant loss and geometry-grounded negative loss constitute subtle technical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across cross-domain and in-domain settings, multiple benchmarks, rich ablation studies, and downstream application demonstrations.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, though it contains many mathematical formulations that require certain background knowledge.
- Value: ⭐⭐⭐⭐⭐ Breakthrough results on non-isometric shape correspondence, holding direct value for downstream applications such as re-topology and rig transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Stable Score Distillation](../../ICCV2025/3d_vision/stable_score_distillation.md)
- [\[CVPR 2025\] SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images](spar3d_stable_point-aware_reconstruction_of_3d_objects_from_single_images.md)
- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)
- [\[CVPR 2025\] A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)
- [\[CVPR 2025\] PrEditor3D: Fast and Precise 3D Shape Editing](preditor3d_fast_and_precise_3d_shape_editing.md)

</div>

<!-- RELATED:END -->

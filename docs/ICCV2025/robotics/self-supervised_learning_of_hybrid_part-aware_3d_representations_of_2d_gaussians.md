---
title: >-
  [Paper Note] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics
description: >-
  [ICCV 2025][Robotics][Part-aware reconstruction] This paper proposes PartGS, a self-supervised part-aware 3D reconstruction framework that hybridly couples 2D Gaussian Splatting with superquadrics. Through parameter sharing and multiple regularization terms, PartGS achieves simultaneous high-quality geometric decomposition and texture reconstruction, outperforming state-of-the-art methods by 75.9% in reconstruction accuracy and 16.13 dB in PSNR on DTU, ShapeNet…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Part-aware reconstruction"
  - "2D Gaussian splatting"
  - "superquadrics"
  - "self-supervised"
  - "shape decomposition"
date: 2026-05-08
content_hash: 099ce41c42a644d9
---

# Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics

**Conference**: ICCV 2025
**arXiv**: [2408.10789](https://arxiv.org/abs/2408.10789)  
**Code**: [zhirui-gao/PartGS](https://zhirui-gao.github.io/PartGS)  
**Area**: Robotics
**Keywords**: Part-aware reconstruction, 2D Gaussian splatting, superquadrics, self-supervised, shape decomposition

## TL;DR

This paper proposes PartGS, a self-supervised part-aware 3D reconstruction framework that hybridly couples 2D Gaussian Splatting with superquadrics. Through parameter sharing and multiple regularization terms, PartGS achieves simultaneous high-quality geometric decomposition and texture reconstruction, outperforming state-of-the-art methods by 75.9% in reconstruction accuracy and 16.13 dB in PSNR on DTU, ShapeNet, and real-world scenes.

## Background & Motivation

**Part-aware 3D reconstruction** aims to decompose objects or scenes into meaningful structured parts, as opposed to low-level representations such as point clouds or meshes. This aligns with the cognitive-science observation that humans perceive 3D environments as compositions of meaningful parts. Structured geometric decomposition enhances scene interpretability and benefits downstream tasks such as physical simulation, editing, and content generation.

Existing methods suffer from three core limitations:

**Reliance on 3D supervision**: Methods such as EMS and MonteBoxFinder require 3D point cloud or voxel inputs and cannot operate directly on multi-view images, limiting practical applicability.

**Conflict between geometry and appearance**: PartNeRF models parts with multiple NeRFs, but the complex composition of implicit fields leads to suboptimal rendering quality and inefficient decomposition. DBW employs superquadrics with UV texture maps for decomposition; while the decomposition is reasonable, the geometric and appearance reconstruction accuracy is insufficient to capture fine details.

**Speed bottleneck**: PartNeRF requires approximately 8 hours per scene, and the concurrent work GaussianBlock also takes several hours.

**Core insight**: Superquadrics are well suited for representing a broad family of primitive shapes (a continuous parametric family encompassing spheres, cuboids, cylinders, etc.), making them appropriate for part-level decomposition; meanwhile, 2D Gaussian Splatting excels at high-fidelity texture and geometric detail reconstruction. Coupling the two—by distributing Gaussians over superquadric surfaces and sharing pose parameters—simultaneously yields reasonable part decomposition and high-quality rendering.

## Method

### Overall Architecture

PartGS adopts a **two-stage optimization strategy**:

- **Block-level stage**: A hybrid superquadric–Gaussian representation is used to decompose the scene into primitive shape blocks.
- **Point-level stage**: The coupling constraint between Gaussians and superquadrics is relaxed, allowing Gaussians to freely offset and refine geometry.

### Parameterization of the Hybrid Representation

The scene $\mathcal{S}$ is decomposed into $M$ hybrid blocks: $\mathcal{S} = \mathcal{B}_1 \cup \ldots \cup \mathcal{B}_M$

Each hybrid block $\mathcal{B}_i$ consists of a **superquadric** and **2D Gaussians** distributed on its surface. The parameters include:

**Shape and scale parameters**: A superquadric is controlled by two shape parameters $\epsilon_1, \epsilon_2$ and three scale parameters $s_1, s_2, s_3$, with vertex coordinates:
$$\mathbf{v} = [s_1 \cos^{\epsilon_1}(\theta) \cos^{\epsilon_2}(\varphi); \; s_2 \sin^{\epsilon_1}(\theta); \; s_3 \cos^{\epsilon_1}(\theta) \sin^{\epsilon_2}(\varphi)]$$

**Pose parameters**: Rotation $\mathbf{R}_i$ and translation $\mathbf{t}_i$, with the transformation: $\hat{\mathbf{v}}_i^j = \mathbf{R}_i \mathbf{v}_i^j + \mathbf{t}_i$

**Key coupling design**: The centers of 2D Gaussians are uniformly sampled on the triangular faces of the superquadric. Their rotation matrix $\mathrm{R}_v = [r_1, r_2, r_3]$ and scaling $\mathrm{S}_v$ are determined by the face vertex positions (following GaMeS), without independently learning geometric parameters. $r_1$ is aligned with the face normal, $r_2$ points from the centroid to $v_1$, and $r_3$ is obtained via Gram–Schmidt orthogonalization.

**Opacity parameters**: Each block has a learnable opacity $\tau_i$; blocks falling below a threshold during training are removed, enabling adaptive part count.

**Texture parameters**: Spherical harmonic coefficients of the 2D Gaussians control view-dependent appearance.

### Block-level Decomposition: Optimization and Regularization

Using only a rendering loss leads to unstable block localization; therefore four regularization terms are introduced:

**Rendering loss** (standard 3DGS loss):
$$\mathcal{L}_{\text{ren}} = (1 - \lambda) L_1 + \lambda L_{\text{D-SSIM}}$$

**Coverage loss**: Ensures that the hybrid blocks cover the object region without extending beyond its boundary. Based on the inside–outside function $D_i(x) = \Psi_i(x) - 1$ of the superquadric, the interaction between a ray and a block is defined as:
$$\mathcal{L}_{\text{cov}} = \sum_{r \in \mathcal{R}} l_r L_{\text{cross}}(r) + (1 - l_r) L_{\text{non-cross}}(r)$$

**Overlap loss**: Penalizes sampled points that lie inside multiple blocks simultaneously via Monte Carlo estimation:
$$\mathcal{L}_{\text{over}} = \frac{1}{N} \sum_{x \in \Omega} \text{ReLU}(\sum_{i \in \mathcal{M}} \mathcal{O}_i^x - k)$$
where the soft occupancy function is $\mathcal{O}_i^x = \tau_i \cdot \text{sigmoid}(-D_i(x) / \gamma)$.

**Parsimony loss**: Penalizes block opacities to encourage the use of the minimum number of blocks: $\mathcal{L}_{\text{par}} = \frac{1}{M} \sum_{i} \sqrt{\tau_i}$

**Opacity entropy loss**: Pushes block opacities toward binary values (0 or 1):
$$\mathcal{L}_{\text{opa}} = \frac{1}{|\mathcal{R}|} \sum_{r} L_{ce}(\max_{i} \tau_i(x^r), l_r)$$

The total loss is a weighted sum: $\mathcal{L} = \mathcal{L}_{\text{ren}} + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}} + \lambda_{\text{over}} \mathcal{L}_{\text{over}} + \lambda_{\text{par}} \mathcal{L}_{\text{par}} + \lambda_{\text{opa}} \mathcal{L}_{\text{opa}}$

**Adaptive block count**: Blocks whose opacity falls below threshold $t$ are removed; DBSCAN clustering is applied to uncovered initial point cloud regions, and new blocks are introduced for each identified cluster.

### Point-level Refinement

The coupling constraint between Gaussians and superquadrics is relaxed to allow independent optimization. An entry constraint is added to prevent Gaussians belonging to one block from penetrating into adjacent blocks:

$$\mathcal{L}_{\text{enter}} = \frac{1}{N} \sum_{x \in \Omega} \sum_{m \in \mathcal{M} \setminus \{\delta\}} \text{ReLU}(-D_m(x))$$

## Experiments

### Main Results I: Quantitative Comparison on DTU Dataset

| Method | Input | Renderable | Part-aware | Mean CD↓ | PSNR↑ | Time↓ |
|--------|-------|------------|------------|----------|-------|-------|
| EMS | 3D GT | ✗ | ✓ | 4.65 | - | - |
| MBF | 3D GT | ✗ | ✓ | 2.50 | - | - |
| PartNeRF | Image | ✓ | ✓ | 8.54 | 17.97 | ~8h |
| DBW | Image | ✓ | ✓ | 4.76 | 16.44 | ~2h |
| **PartGS (Block)** | Image | ✓ | ✓ | **4.19** | **19.84** | **~30m** |
| **PartGS (Point)** | Image | ✓ | ✓ | **0.98** | **35.04** | **~40m** |
| 2DGS (non-part) | Image | ✓ | ✗ | 0.81 | 34.07 | ~10m |

PartGS at the point level (CD = 0.98) approaches the non-part method 2DGS (0.81) while retaining part decomposition capability. Compared to DBW (the SOTA part-aware method), PartGS achieves a 79% improvement in CD, an 18.6 dB gain in PSNR, and runs three times faster.

### Main Results II: Comparison on ShapeNet Dataset

| Method | Input | Airplane CD | Table CD | Chair CD | Gun CD | Mean CD |
|--------|-------|-------------|----------|----------|--------|---------|
| EMS | 3D GT | 3.40 | 6.92 | 19.0 | 2.02 | - |
| PartGS (Block) | Image | - | - | - | - | 4.19 |
| PartGS (Point) | Image | - | - | - | - | 0.98 |

PartGS demonstrates significant reconstruction accuracy advantages on ShapeNet as well, handling diverse shapes across different categories.

### Ablation Study

| Strategy | Effect |
|----------|--------|
| w/o coverage loss | Blocks fail to fully cover the object, leaving uncovered regions |
| w/o overlap loss | Severe overlap between blocks, degrading decomposition quality |
| w/o parsimony loss | Excessive redundant blocks are used |
| w/o DBSCAN addition | Newly appearing regions of complex objects cannot be covered |
| w/o entry constraint (point-level) | Gaussians cross block boundaries, breaking decomposition continuity |

## Highlights & Insights

1. **Elegant hybrid coupling design**: Gaussians share the pose of the superquadric, making the representation more compact and efficient (no need to independently learn Gaussian geometry).
2. **Self-supervised part decomposition**: Without 3D annotations or segmentation supervision, the method automatically discovers object parts solely through multi-view rendering losses and regularization constraints.
3. **Progressive two-stage strategy**: The block level guarantees decomposition quality while the point level ensures reconstruction accuracy; the two stages are decoupled yet complementary.
4. While maintaining part-aware capability, reconstruction quality approaches or even surpasses non-part methods (PSNR 35.04 vs. 2DGS 34.07).

## Limitations & Future Work

1. The expressive capacity of superquadrics is limited: for highly irregular shapes such as trees or animal fur, the primitive-shape assumption may not hold.
2. The initial block count $M$ is a hyperparameter; although an adaptive mechanism exists, it still requires empirical specification.
3. Validation is primarily conducted at the object level; decomposition of larger-scale indoor or outdoor scenes has not been explored.
4. Comparisons are limited to part-aware methods up to 2024, without comprehensive evaluation against the most recent concurrent work.

## Related Work & Insights

- **Shape decomposition / abstraction**: Blocks World, EMS (probabilistic superquadric recovery), MonteBoxFinder (cuboids + MCTS)
- **Image-driven structured 3D**: PartNeRF (ellipsoids + NeRF), DBW (superquadrics + UV textures), ISCO
- **Mesh–Gaussian hybrid**: SuGaR, GaMeS (Gaussians bound to triangular faces)

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Technical Depth | 5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 4 |
| Overall | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Weakly-Supervised Learning of Dense Functional Correspondences](weakly-supervised_learning_of_dense_functional_correspondences.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[ICLR 2026\] PA3FF: Part-Aware Dense 3D Feature Fields for Generalizable Articulated Object Manipulation](../../ICLR2026/robotics/pa3fflearning_part-aware_dense_3d_feature_field_for_generalizable_articulated_ob.md)
- [\[CVPR 2026\] Learning to Control Physically-simulated 3D Characters via Generating and Mimicking 2D Motions](../../CVPR2026/robotics/learning_to_control_physically-simulated_3d_characters_via_generating_and_mimick.md)
- [\[ICCV 2025\] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)

</div>

<!-- RELATED:END -->

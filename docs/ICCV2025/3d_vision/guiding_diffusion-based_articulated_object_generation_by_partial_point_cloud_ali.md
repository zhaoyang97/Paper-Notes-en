---
title: >-
  [Paper Note] Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints
description: >-
  [ICCV 2025][3D Vision][Articulated object generation] This paper proposes PhysNAP, which guides the reverse diffusion process of the pretrained diffusion model NAP via point cloud alignment loss and SDF-based physical pl…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Articulated object generation"
  - "diffusion models"
  - "point cloud alignment"
  - "physical constraints"
  - "SDF"
date: 2026-05-08
content_hash: b1c96c4724122eef
---

# Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints

**Conference**: ICCV 2025
**arXiv**: [2508.00558](https://arxiv.org/abs/2508.00558)  
**Code**: Not released  
**Area**: 3D Vision
**Keywords**: Articulated object generation, diffusion models, point cloud alignment, physical constraints, SDF

## TL;DR

This paper proposes PhysNAP, which guides the reverse diffusion process of the pretrained diffusion model NAP via point cloud alignment loss and SDF-based physical plausibility constraints (part penetration and joint mobility), enabling category-aware articulated object generation with significant improvements in alignment accuracy and physical plausibility over the unguided baseline.

## Background & Motivation

Articulated objects (e.g., drawers, appliances, laptops) are ubiquitous in everyday environments, and their digital twin generation and reconstruction are critical for virtual reality and robotics applications.

**Limitations of Prior Work**:

**NAP** (baseline): Uses DDPM to generate articulation graphs encoding part shapes (SDF), joint parameters, and graph structure, but operates in an **unconditional, unguided** manner and cannot align with observed data.

**CAGE/SINGAPO**: Require the articulation graph structure (node count, connectivity) as input priors, limiting generality.

**MIDGaRD**: Generates graph structure and shape in separate stages, preventing gradient backpropagation from shape to graph structure.

**PhysPart**: The closest prior work, but assumes a known complete base point cloud and only considers collision between the base and a single part.

**Innovation of PhysNAP**:
- **No assumption on known articulation graph structure** (part count and connectivity are unknown)
- **Uses partial point clouds** (rather than complete point clouds) as training-free guidance
- **Evaluates penetration across all part pairs** (not just base–single-part)
- **First articulated object diffusion model with simultaneous category-awareness, point cloud alignment, and physical plausibility guidance**

## Method

### Overall Architecture

PhysNAP extends NAP with three core components:
1. Category-conditioned articulation graph diffusion model (training phase)
2. Loss-guided reverse diffusion sampling (inference phase)
3. Three guidance loss designs (point cloud alignment + penetration + mobility)

### Articulation Graph Representation

Each articulated object is represented as a graph with up to $K=8$ nodes:
- **Node attributes**: existence indicator $o_i$, pose $\boldsymbol{T}_{gi} \in SE(3)$, bounding box $\boldsymbol{b}_i \in \mathbb{R}^3$, shape latent code $\boldsymbol{s}_i \in \mathbb{R}^{128}$
- **Edge attributes**: existence indicator $c_{i,j} \in \{-1,0,1\}$, articulation axis in Plücker coordinates, joint range

Shape is represented via a pretrained neural SDF decoder, from which meshes are extracted using Marching Cubes.

### Category-Aware Generation

Learnable category embeddings are added to node and edge embeddings within NAP's AGNN (Attention Graph Neural Network), and the model is retrained on category-annotated data.

### Loss-Guided Diffusion

Using the loss-guided diffusion framework, guidance gradients are applied during the last $n_g=500$ steps of the reverse diffusion process:

$$\ell_{\boldsymbol{P}}(\hat{\boldsymbol{x}}_0) = w_{\text{pc}}\ell_{\text{pc}} + w_{\text{pen}}\ell_{\text{pen}} + w_{\text{mob}}\ell_{\text{mob}}$$

#### Point Cloud Alignment Loss $\ell_{\text{pc}}$

Given a partial point cloud $\boldsymbol{P} \in \mathbb{R}^{n_p \times 3}$, the distance from each point to the nearest part is evaluated via the predicted SDF. Since point-to-part correspondences are unknown, soft correspondences (analogous to EM) are used:

$$\alpha_{i,j} = \frac{\exp(-\tau \, d(\boldsymbol{P}_j, i)^2 / (\tilde{o}_i + \epsilon))}{\sum_k \exp(-\tau \, d(\boldsymbol{P}_j, k)^2 / (\tilde{o}_k + \epsilon))}$$

The final loss is $\ell_{\text{pc}} = \sum_j \sum_i \alpha_{i,j} d(\boldsymbol{P}_j, i)^2$, with temperature parameter $\tau=1000$.

#### Penetration Loss $\ell_{\text{pen}}$

Bounding box intersections are computed for all part pairs. 3D grid points are sampled within intersection regions (target $N^*=1000$), and the SDF-based penetration error is evaluated:

$$\psi(\boldsymbol{q}_j, i, i') = \frac{1}{2}\min(0, -(d(\boldsymbol{q}_j, i) + d(\boldsymbol{q}_j, i')))^2$$

#### Mobility Loss $\ell_{\text{mob}}$

Articulation states are randomly sampled within the predicted joint limits. Relative poses between parts are computed via screw transformations, and penetration errors are evaluated under articulated configurations. Only collisions between adjacent parts are assessed.

## Experiments

### Experimental Setup

- **Dataset**: PartNet-Mobility, covering multiple categories of articulated objects
- **Evaluation metrics**: point cloud alignment ($E_{\text{pc}}$, $D_{\text{pc}}$), penetration ($E_{\text{pen}}$), mobility ($E_{\text{mob}}$), generation quality (MMD, 1-NNA)
- **Test set**: 30 randomly selected object models, 1,000 sampled points each

### Main Results

| Category-Aware | Guidance Variant | $E_{\text{pc}}$ | $D_{\text{pc}}$ | $E_{\text{pen}}$ | $E_{\text{mob}}$ | MMD | 1-NNA |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | pc+pen+mob | 0.0024 | 0.0705 | **0.0000** | **0.0003** | 0.1435 | 0.9547 |
| ✗ | pc | **0.0006** | **0.0435** | 0.0018 | 0.0031 | 0.1540 | 0.9666 |
| ✗ | uncond (NAP) | 0.0564 | 0.2063 | 0.0035 | 0.0033 | **0.0915** | **0.9440** |
| ✓ | pc+pen+mob | 0.0012 | 0.0483 | **0.0000** | **0.0003** | 0.1970 | 0.9774 |
| ✓ | pc | **0.0004** | **0.0301** | 0.0079 | 0.0059 | 0.2162 | 0.9817 |
| ✓ | uncond | 0.0076 | 0.0974 | 0.0019 | 0.0027 | 0.1687 | 0.9709 |

### Ablation Study

| Ablation Setting | Key Findings |
|:---:|:---|
| Guidance steps $n_g$ | 500 steps and 1000 steps yield similar performance; 500 steps are more computationally efficient |
| Weight $w_{\text{pen}}$ | Larger weight reduces penetration but degrades generation quality, indicating a trade-off |
| pc only vs. pc+pen+mob | Adding physical constraints slightly reduces alignment but substantially improves physical plausibility |
| Category-aware vs. none | Category information improves alignment accuracy but reduces diversity (higher 1-NNA) |

### Key Findings

1. **Guidance effectiveness**: All guided variants outperform the unguided baseline on corresponding metrics ($E_{\text{pc}}$ drops from 0.0564 to 0.0006).
2. **Trade-off exists**: Competing guidance objectives create tensions—point cloud guidance degrades physical plausibility, while penetration and mobility guidance degrades alignment.
3. **Category conditioning**: Providing category information improves alignment at the cost of generation diversity.
4. **Runtime**: Full guidance requires approximately 2 minutes per sample on an Nvidia A40.

## Highlights & Insights

1. **Elegant application of training-free guidance**: Gradient guidance is injected solely at sampling time without modifying the pretrained model, enabling flexible conditional generation.
2. **Dual use of SDF**: The SDF serves both as a shape representation and as the basis for physical constraint computation (penetration detection), yielding a compact design.
3. **Soft correspondence mechanism**: Differentiable soft assignment resolves unknown point-to-part correspondences, making the loss differentiable with respect to node existence as well.

## Limitations & Future Work

1. The guidance process increases inference time (~2 minutes per sample).
2. The trade-off between point cloud alignment and physical plausibility is difficult to fully eliminate.
3. The category-aware model reduces generation diversity.
4. Validation is limited to the PartNet-Mobility dataset; generalization remains unexplored.

## Related Work & Insights

- **Articulated object generation**: NAP, CAGE, SINGAPO, MIDGaRD, PhysPart
- **Loss-guided diffusion**: DPS, Loss-Guided Diffusion
- **SDF representation**: DeepSDF, Articulated SDF (A-SDF)

## Rating

- Novelty: ⭐⭐⭐⭐ — First work to integrate physical plausibility constraints into articulated object diffusion generation
- Technical depth: ⭐⭐⭐⭐ — SDF-based penetration detection and soft correspondence design are well-crafted
- Experimental Thoroughness: ⭐⭐⭐ — Ablations are comprehensive, but direct comparisons with competing methods are lacking
- Value: ⭐⭐⭐ — Application scenarios are clear (robotics, VR/AR), but inference speed requires further optimization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Blended Point Cloud Diffusion for Localized Text-guided Shape Editing](blended_point_cloud_diffusion_for_localized_textguided_shape.md)
- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->

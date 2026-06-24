---
title: >-
  [Paper Note] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation
description: >-
  [3D Vision] This paper proposes the Gaussian Atlas representation, which maps unordered 3D Gaussians onto a sphere via optimal transport and then flattens them into a structured 2D grid, enabling direct fine-tuning of pretrained 2D Latent Diffusion models for high-quality text-to-3D generation.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: a723f4ffac7aead9
---

# Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2503.15877](https://arxiv.org/abs/2503.15877)
- **Code**: [Project Page](https://cs.stanford.edu/~xtiange/projects/gaussianatlas)
- **Area**: 3D Generation · Diffusion Models
- **Keywords**: 3D Gaussian, 2D diffusion model transfer, Gaussian Atlas, large-scale dataset, text-to-3D

## TL;DR
This paper proposes the Gaussian Atlas representation, which maps unordered 3D Gaussians onto a sphere via optimal transport and then flattens them into a structured 2D grid, enabling direct fine-tuning of pretrained 2D Latent Diffusion models for high-quality text-to-3D generation.

## Background & Motivation

The development of 3D diffusion models is constrained by the scarcity of high-quality 3D data, leaving their performance far behind that of 2D counterparts. Existing methods attempt to leverage 2D diffusion model priors for 3D generation, but face the following limitations:

**Data bottleneck**: Creating and annotating high-quality 3D models is extremely costly, and the data scale is far smaller than that of 2D images (billions of samples).

**Indirect utilization**: Methods such as Score Distillation Sampling (SDS) use frozen 2D model weights only indirectly, resulting in low efficiency and limited quality.

**Representation gap**: The unordered point-set structure of 3D Gaussians cannot be directly fed into 2D networks; a structured 2D representation is required to enable transfer learning.

**Mechanism**: The paper designs a method (Gaussian Atlas) that maps 3D Gaussians into a structured 2D grid, allowing pretrained 2D diffusion models to be directly fine-tuned for 3D generation.

## Method

### Overall Architecture

The pipeline consists of two stages:
1. **3DGS pre-fitting stage**: Pre-fitting high-quality 3D Gaussians for a large collection of 3D objects to construct the GaussianVerse dataset.
2. **Diffusion model training stage**: Converting 3D Gaussians into the Gaussian Atlas 2D representation and fine-tuning the UNet of a pretrained Latent Diffusion model.

### GaussianVerse Dataset

A large-scale dataset containing **205,737 high-quality 3DGS fits** is constructed based on Scaffold-GS. Key improvements include:

- **Visibility-ranked pruning strategy**: Rather than enforcing a fixed count constraint, an upper bound of $\tau=36,864$ (i.e., $192 \times 192$) is set; Gaussians are ranked by their opacity under random camera viewpoints, and those with the lowest visibility are pruned.
- **Perceptual loss augmentation**: LPIPS perceptual loss is incorporated into the fitting objective to achieve higher rendering fidelity with fewer Gaussians.

The fitting loss is:

$$\lambda'_{rgb}\mathcal{L}_{rgb} + \lambda'_{ssim}\mathcal{L}_{ssim} + \lambda'_{lpips}\mathcal{L}_{lpips} + \lambda'_{reg}\mathcal{R}$$

Each object requires approximately 10 minutes of fitting time on an A100 GPU, totaling over 3.8 A100 GPU-years.

### Gaussian Atlas: Mapping from 3D to 2D

The conversion of unordered 3D Gaussians into a structured 2D grid proceeds in three steps:

**Step 1: Sphere Offsetting**

A unit sphere $\mathcal{S}$ is assumed, with $N$ points $\{s_i \in \mathbb{R}^3\}$ uniformly distributed on its surface. **Optimal Transport (OT)** is used to map the positions of 3D Gaussians onto the sphere surface. Unlike GaussianCube, this method maps to a spherical surface rather than the interior of a volumetric grid.

**Step 2: Equirectangular Projection**

The 3D Gaussians on the sphere are flattened onto a 2D plane via the equirectangular projection $\mathcal{M}$, yielding 2D coordinates $\{p_i \in \mathbb{R}^2\}$. Since $\mathcal{M}$ is a deterministic function, the projection is consistent across all objects.

**Step 3: Plane Offsetting**

OT is applied again to map the flattened 2D coordinates to the vertices of a $\sqrt{N} \times \sqrt{N}$ regular grid $\{q_i \in \mathbb{R}^2\}$. Due to the deterministic nature of the projection, this OT step needs to be computed only once and its index can be reused.

The resulting Gaussian Atlas has shape $\sqrt{N} \times \sqrt{N} \times C$, where $C = ||\mathbf{x}-\mathbf{s}|| + ||\mathbf{c}|| + ||\mathbf{o}|| + ||\mathbf{s}|| + ||\mathbf{r}||$, encoding all Gaussian attributes.

### Fine-tuning the Latent Diffusion Model

- **VAE bypassed**: The distribution of Gaussian attributes differs too greatly from that of natural images to be directly processed by a VAE.
- **Normalization alignment**: Per-pixel mean and standard deviation computed over the full GaussianVerse dataset are used to normalize the Atlas, aligning its distribution with VAE-encoded image latents.
- **Channel adaptation**: Each 3-channel attribute is concatenated with a 1-channel opacity to form a 4-channel input; the UNet input layer is repeated four times to accommodate the $128 \times 128 \times 16$ input.

The training loss is:

$$\lambda_{diff}\mathcal{L}_{diff} + \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{lpips}\mathcal{L}_{lpips}$$

where $\mathcal{L}_{diff}$ is the v-parameterized diffusion loss, $\mathcal{L}_{rgb}$ and $\mathcal{L}_{mask}$ are rendering L1 losses, and $\mathcal{L}_{lpips}$ is the perceptual loss.

### Inference

Starting from random 2D noise, reverse diffusion is performed using DPMsolver++ with a guidance scale of 3.5. Generating and rendering a single 3DGS sample takes **less than 5 seconds**.

## Key Experimental Results

### Main Results: Text-to-3D Generation Comparison

| Method | CLIP score ↑ | VQA score ↑ | # Gaussians ↓ |
|--------|-------------|-------------|--------------|
| DreamGaussian | 20.52 | 0.37 | 40K |
| LGM | 20.28 | 0.35 | 66K |
| TriplaneGaussian | 21.10 | 0.46 | 16K |
| GaussianCube | 22.31 | 0.52 | 33K |
| **GaussianAtlas (Ours)** | **23.20** | **0.61** | **16K** |

The proposed method outperforms GaussianCube by 0.9 in CLIP score and by 17% in VQA score, while using only half the number of Gaussians and training steps.

### Ablation Study: Pretrained 2D Model vs. Training from Scratch

| Method | Training Steps | CLIP score ↑ | VQA score ↑ |
|--------|---------------|-------------|-------------|
| From scratch | 500K | 19.33 | 0.23 |
| Transfer 2D LD | 500K | 21.61 | 0.49 |
| From scratch | 1M | 20.85 | 0.40 |
| Transfer 2D LD | 1M | **23.20** | **0.61** |

The pretrained 2D model significantly outperforms training from scratch under equivalent training steps, validating the transferability of 2D knowledge.

### User Study

Based on 2,500+ valid responses:
- vs. GaussianCube: **65%** of users prefer the proposed method.
- vs. TriplaneGaussian: **88%** of users prefer the proposed method.

### Key Findings

1. **Deterministic mapping is critical**: Optimization-based flattening approaches (learning per-object UV mappings independently) produce inconsistent visual patterns, resulting in noise-only outputs after fine-tuning.
2. **Minimal weight deviation**: The deviation between fine-tuned and pretrained UNet weights is extremely small (even the most-changed layer deviates 8× less than random initialization), indicating that pretrained weights serve as an excellent initialization for 3D generation.
3. **VAE not required**: Training the UNet directly in the normalized Atlas space avoids the mismatch introduced by applying a VAE designed for natural images.

## Highlights & Insights

1. **Unifying 2D and 3D generation**: This work is the first to demonstrate that pretrained text-to-image diffusion models can be directly fine-tuned for 3D Gaussian generation without complex intermediate steps.
2. **Elegant 2D representation design**: The three-step pipeline—spherical OT → equirectangular projection → planar OT—preserves 3D topological continuity while ensuring consistent cross-object mappings.
3. **Large-scale dataset contribution**: GaussianVerse (205K high-quality fits) provides important infrastructure for the research community.
4. **Fast inference**: Generation and rendering in under 5 seconds, far superior to SDS-based optimization methods (which require minutes).

## Limitations & Future Work

- Generation resolution is constrained by the Atlas grid size ($128 \times 128 = 16\text{K}$ Gaussians), leading to insufficient detail for complex objects.
- Dataset construction cost is extremely high (3.8+ A100 GPU-years).
- Equirectangular projection introduces inherent area distortion near the poles.
- Only single-object generation is supported; extension to scene-level generation has not been explored.
- Direct comparison with recent methods such as TRELLIS is absent.

## Related Work & Insights

- **2D diffusion model transfer**: Marigold (depth prediction) and GeoWizard (geometry prediction) inspired the transfer to 3D tasks.
- **3D Gaussian generation**: GaussianCube (3D grid OT), GVGen (voxel offsetting), DiffGS (triplane mapping).
- **2D representations of 3D**: Triplane-based methods (NFD, CRM, InstantMesh), Omages (UV mapping), DiffSplat (multi-view latents).
- **Datasets**: ShapeSplat, Objaverse series.

**Insight**: A deterministic and consistent 3D→2D mapping is the key to successful transfer—learned mappings offer flexibility but destroy cross-sample pattern consistency.

## Rating
- **Novelty**: ★★★★★ — Introduces the Gaussian Atlas representation and pioneers the direct fine-tuning of 2D models for 3D generation.
- **Technical Depth**: ★★★★☆ — The OT mapping combined with projection is clean and elegant, though the diffusion model training itself is relatively standard.
- **Practicality**: ★★★★☆ — Inference is extremely fast, but dataset construction requires substantial computational resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

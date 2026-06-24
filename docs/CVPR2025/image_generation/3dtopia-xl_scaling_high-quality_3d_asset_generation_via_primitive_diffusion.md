---
title: >-
  [Paper Note] 3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion
description: >-
  [CVPR 2025][Image Generation][3D Generation] This paper proposes 3DTopia-XL, a native 3D generation model based on a novel primitive representation PrimX and a Diffusion Transformer. It generates high-quality 3D assets with high-resolution geometry, texture, and PBR materials from text or image inputs, significantly outperforming existing methods in both quality and efficiency.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "3D Generation"
  - "PBR Materials"
  - "Primitive Representation"
  - "Diffusion Transformer"
  - "Textured Mesh"
date: 2026-05-08
content_hash: 40c33a68885e6dd4
---

# 3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2409.12957](https://arxiv.org/abs/2409.12957)  
**Code**: [https://github.com/3DTopia/3DTopia-XL](https://github.com/3DTopia/3DTopia-XL)  
**Area**: 3D Generation / Image Generation  
**Keywords**: 3D Generation, PBR Materials, Primitive Representation, Diffusion Transformer, Textured Mesh

## TL;DR
This paper proposes 3DTopia-XL, a native 3D generation model based on a novel primitive representation PrimX and a Diffusion Transformer. It generates high-quality 3D assets with high-resolution geometry, texture, and PBR materials from text or image inputs, significantly outperforming existing methods in both quality and efficiency.

## Background & Motivation

High-quality 3D assets are in high demand across film, gaming, and virtual reality, but manual creation is extremely costly. Existing automatic 3D generation methods fall into three main categories:

**SDS Methods** (e.g., DreamFusion): Distill 2D diffusion priors into 3D via per-scene optimization, but the optimization is time-consuming, geometry quality is poor, and multi-face inconsistency occurs.

**Sparse-view Reconstruction** (e.g., LRM, InstantMesh): Regress 3D from sparse views using large models. However, most rely on triplane-NeRF representations, which suffer from limited resolution due to low parameter efficiency, and lack diversity as deterministic methods.

**Native 3D Diffusion Models** (e.g., Shap-E, 3DTopia): Model the 3D distribution to directly generate 3D objects, but almost none can generate PBR assets containing geometry, texture, and materials.

**Key Challenge**: Existing 3D representations are either parameter-inefficient (triplane), unable to encode PBR materials, or slow to tensorize, making it difficult to train high-quality 3D diffusion models on large-scale datasets.

**Key Insight**: This paper designs a new primitive-based 3D representation, PrimX, which simultaneously encodes shape, color, and materials into a compact $N \times D$ tensor, and then performs diffusion generation using DiT. **Core Idea: Efficiently represent textured meshes with a set of surface-anchored small voxel primitives, and model the global relationships among these primitives using a Transformer.**

## Method

### Overall Architecture
The input is a text prompt or a distinct image, which is processed through a two-stage pipeline to output a GLB mesh with PBR materials:
1. **PrimX Representation**: Encodes the 3D textured mesh into a compact $N \times D$ tensor.
2. **Primitive Patch Compression**: Compresses the payload of each primitive into the latent space using a 3D VAE.
3. **Latent Primitive Diffusion**: Performs diffusion generation on the latent primitive set using DiT.
4. **PBR Asset Extraction**: Decodes the generated PrimX back into a high-quality textured mesh (GLB file).

### Key Designs

1. **PrimX Representation**:

    - **Function**: Unifies the encoding of a textured mesh's shape, color, and materials into a set of surface-anchored voxel primitives.
    - **Mechanism**: Sample $N$ anchor points on the mesh surface. Each primitive $\mathcal{V}_k = \{\mathbf{t}_k, s_k, \mathbf{X}_k\}$ contains a position, a scale, and an $a^3 \times 6$ payload (1-channel SDF + 3-channel RGB + 2-channel material). The attribute of any point in space is obtained via weighted interpolation:
    $F_{\mathcal{V}}(\mathbf{x}) = \sum_{k=1}^N w_k(\mathbf{x}) \cdot \mathcal{I}(\mathbf{X}_k, (\mathbf{x}-\mathbf{t}_k)/s_k)$
   The weights $w_k$ are normalized based on L∞ distance to ensure local support. The entire mesh can be represented as an $N \times D$ tensor ($D = 3+1+a^3 \times 6$).
    - **Design Motivation**: Compared to triplanes (parameter-inefficient, limited resolution) and MLPs (slow), PrimX achieves the highest fitting accuracy with a 10x speedup under the same parameter budget. Surface anchoring ensures that parameters are concentrated in physically meaningful regions.

2. **Primitive Patch Compression (3D VAE)**:

    - **Function**: Compresses the high-dimensional payload of each primitive into a low-dimensional latent representation.
    - **Mechanism**: Build a VAE using 3D convolutions to compress each primitive independently. The encoder compresses $\mathbf{X}_k \in \mathbb{R}^{a^3 \times 6}$ into $\hat{\mathbf{X}}_k \in \mathbb{R}^{(a/2)^3 \times 1}$, achieving a 48x compression ratio. The training objective is reconstruction loss + KL regularization:
    $$\mathcal{L}_{\text{ae}} = \mathbb{E}[\|\mathbf{X}_k - D(E(\mathbf{X}_k))\|_2 + \lambda_{\text{kl}} \mathcal{L}_{\text{kl}}]$$
    - **Design Motivation**: Global compression (such as applying a VAE to the entire triplane) would make the latent space overly complex. Segmenting into local patches and compressing them independently is simple and efficient, leaving the global semantics to be modeled by the subsequent diffusion model.

3. **Latent Primitive Diffusion (DiT)**:

    - **Function**: Models the distribution over the latent primitive set to achieve conditional 3D generation.
    - **Mechanism**: Treat each primitive as a token and use a 28-layer DiT to model the global correlations between primitives. This includes cross-attention to incorporate conditioning signals (text/image embeddings), self-attention to model relationships between primitives, and AdaLN to inject timesteps. Training uses v-prediction + CFG (classifier-free guidance) + cosine schedule. The permutation invariance of PrimX naturally matches the Transformer without requiring positional encodings.
    - **Design Motivation**: Thanks to the compactness of PrimX, training can be performed directly at high resolution without requiring super-resolution post-processing, resulting in a clean and unified framework.

4. **PBR Asset Extraction**:

    - **Function**: Losslessly reconstructs the textured mesh (GLB format) from PrimX.
    - **Mechanism**: Extract geometry on the SDF zero-isosurface using Marching Cubes; sample color and material values in a $1024 \times 1024$ UV space; apply dilation and nearest-neighbor interpolation to the UV map for anti-aliasing.
    - **Design Motivation**: Most 3D generation methods use vertex coloring when exporting meshes, which heavily degrades quality. The high-quality SDF surface in PrimX supports high-resolution UV sampling, preventing quality loss.

### Loss & Training
- **PrimX Fitting**: Two-stage fine-tuning—first optimize SDF ($\lambda_{\text{SDF}}=10, \lambda=0$, 1k iterations), then optimize color and materials ($\lambda_{\text{SDF}}=0, \lambda=1$, 1k iterations), taking about 1.5 minutes per sample in total.
- **VAE**: Reconstruction loss + KL regularization.
- **DiT**: v-prediction objective + CFG (10% probability of dropping conditions), cosine noise schedule, 1000 steps.

## Key Experimental Results

### Main Results: Comparison of Representation Quality (Same 1.05M Parameter Budget)

| Method | Fitting Time | CD ×10⁻⁴ ↓ | PSNR-SDF ↑ | PSNR-RGB ↑ | PSNR-Mat ↑ |
|---------|---------|-----------|-----------|-----------|-----------|
| MLP | 14 min | 4.502 | 40.73 | 21.19 | 13.99 |
| MLP w/ PE | 14 min | 4.638 | 40.82 | 21.78 | 12.75 |
| Triplane | 16 min | 9.678 | 39.88 | 18.28 | 16.46 |
| Dense Voxels | 10 min | 7.012 | 41.70 | 20.01 | 15.98 |
| **PrimX** | **1.5 min** | **1.310** | **41.74** | **21.86** | **16.50** |

### Ablation Study: Number of Primitives and Resolution

| N (Num. Primitives) | a³ (Resolution) | Params | PSNR-SDF ↑ | PSNR-RGB ↑ |
|-----------|-----------|-------|-----------|-----------|
| 64 | 32³ | 2.10M | 61.05 | 22.18 |
| 256 | 16³ | 1.05M | 59.05 | 23.50 |
| 2048 | 8³ | 1.05M | **62.52** | **24.23** |

### Key Findings
- **More Small Primitives Outperform Fewer Large Primitives**: Under the same parameter budget, the N=2048/a=8 configuration significantly outperforms N=64/a=32 in both geometry and texture.
- PrimX's fitting speed is **more than 10 times faster** than triplane (1.5 min vs. 16 min), with significantly better geometric quality.
- The generated PBR assets can be directly imported into graphics engines like Blender, exhibiting realistic specular and gloss effects.
- The CLIP Score for Text-to-3D is superior to both Shap-E and 3DTopia.

## Highlights & Insights
- Elegant design of the PrimX representation: Surface anchoring combined with local voxels balances parameter efficiency and expressiveness.
- The divide-and-conquer strategy of local compression plus global diffusion enables high-resolution 3D generation without requiring a super-resolution module.
- The first among native 3D diffusion models to support complete PBR materials (geometry + texture + metallicity/roughness).
- Applications like 3D inpainting and interpolation demonstrate the unique advantages of native 3D diffusion models over reconstruction methods.

## Limitations & Future Work
- Generation quality is still bottlenecked by the scale and diversity of the training data.
- The number of PrimX primitives is fixed and cannot adapt to objects of varying complexity.
- Currently, only single-object generation is supported; scene-level generation remains to be explored.
- UV unwrapping might be unstable under extreme topologies.

## Related Work & Insights
- M-SDF (Yariv et al.) proposed the mosaic SDF concept but only encoded shape; PrimX extends this to shape, color, and materials.
- The parameter efficiency bottleneck of triplane representations in LRM-like models is the primary driver for this work.
- The success of DiT in 2D image generation inspired its extension to 3D primitive sets.
- Core Insight: **The bottleneck of 3D generation lies in the representation rather than the model**—a well-designed representation allows standard diffusion frameworks to work directly.

## Rating
- Novelty: ⭐⭐⭐⭐ The PrimX representation design is innovative, but the overall pipeline follows the standard VAE + DiT paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The representation comparisons, generation comparisons, and extensive ablation studies are very thorough.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the mathematical derivations are comprehensive.
- Value: ⭐⭐⭐⭐ The first high-quality native 3D generation model supporting PBR materials, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[ICLR 2026\] Scaling Group Inference for Diverse and High-Quality Generation](../../ICLR2026/image_generation/scaling_group_inference_for_diverse_and_high-quality_generation.md)
- [\[CVPR 2025\] EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)
- [\[CVPR 2025\] ArtiFade: Learning to Generate High-quality Subject from Blemished Images](artifade_learning_to_generate_high-quality_subject_from_blemished_images.md)
- [\[CVPR 2025\] StableAnimator: High-Quality Identity-Preserving Human Image Animation](stableanimator_high-quality_identity-preserving_human_image_animation.md)

</div>

<!-- RELATED:END -->

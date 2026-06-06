---
title: >-
  [Paper Note] LiTo: Surface Light Field Tokenization
description: >-
  [ICLR 2026][3D Vision][surface light field] LiTo encodes surface light fields into compact sets of latent vectors to jointly model 3D geometry and view-dependent appearance: random subsampling of light field observations…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "surface light field"
  - "3D latent representation"
  - "view-dependent appearance"
  - "Gaussian Splatting"
  - "flow matching"
date: 2026-05-08
content_hash: 23b2a909d29a60a9
---

# LiTo: Surface Light Field Tokenization

**Conference**: ICLR 2026
**arXiv**: [2603.11047](https://arxiv.org/abs/2603.11047)  
**Code**: Unavailable (Apple internal)  
**Area**: 3D Vision / Generation
**Keywords**: surface light field, 3D latent representation, view-dependent appearance, Gaussian Splatting, flow matching

## TL;DR
LiTo encodes surface light fields into compact sets of latent vectors to jointly model 3D geometry and view-dependent appearance: random subsampling of light field observations from RGB-D multi-view images → Perceiver IO encoder (3D local attention supporting 1M token input) + flow-matching geometry decoder + higher-order spherical harmonic Gaussian decoder → achieves reconstruction and single-image-to-3D generation surpassing TRELLIS, and for the first time models view-dependent effects such as specular highlights and Fresnel reflectance within a latent 3D representation.

## Background & Motivation

**Background**: 3D latent representation methods fall into two categories — geometry-only representations (3DShape2VecSet / TripoSG / ShapeTokens) model shape without appearance; geometry-plus-appearance representations (TRELLIS / 3DTopia-XL) incorporate appearance but support only view-independent diffuse color.

**Limitations of Prior Work**: (1) Geometry-only methods cannot render photorealistic 3D content due to the absence of color, material, and lighting effects. (2) TRELLIS includes appearance but relies on mean pooling of DINOv2 features, discarding viewing direction information and thus failing to model view-dependent effects such as specular highlights and Fresnel reflectance. (3) 3DTopia-XL models PBR materials but requires a preprocessing step to optimize primitive representations from meshes.

**Key Challenge**: Real-world object appearance depends strongly on the viewing angle (metallic reflections, Fresnel effects, etc.), yet existing 3D latent representations discard directional information. Modeling view-dependent effects requires encoding the surface light field — not only surface position and color, but also the viewing direction.

**Key Insight**: RGB-D multi-view images constitute discrete samples of a surface light field — each pixel provides a tuple of (surface point position, viewing direction, color). The method randomly subsamples these light field observations as input, uses an encoder to interpolate, and outputs via a third-order spherical harmonic Gaussian decoder.

**Core Idea**: Randomly subsampled surface light field observations are encoded into compact latent tokens, and a dual-branch decoder (flow-matching geometry + spherical harmonic Gaussian appearance) enables a unified 3D representation capturing both geometry and view-dependent appearance.

## Method

### Overall Architecture
Input: 150 multi-view RGB-D renderings → extract ~160M surface light field samples → randomly select 1M as encoder input. Output: $k=8192$ latent tokens of dimension $d=32$.

Three components: (1) Perceiver IO encoder (supporting 1M tokens), (2) flow-matching geometry decoder, (3) view-dependent Gaussian decoder.

### Key Designs

1. **Surface Light Field Sampling and Encoding**:

    - Function: Transforms RGB-D multi-view images into surface light field samples $\{(\mathbf{x}_i, \hat{\mathbf{d}}_i, \mathbf{c}_i)\}$ and encodes them into latent representations.
    - Mechanism: Surface points $\mathbf{x}$ are obtained by back-projecting RGB-D images; viewing directions $\hat{\mathbf{d}}$ are derived from the pinhole camera model; pixel colors yield $\mathbf{c}$. $N=2^{20}$ samples are randomly selected as encoder input. The Perceiver IO encoder outputs 8192 latent tokens of dimension 32.
    - Design Motivation: The full surface light field contains enormous information (~160M samples) with substantial redundancy. Random subsampling encourages the encoder to learn interpolation and generalize to the complete light field. Including direction $\hat{\mathbf{d}}$ in each sample is critical for capturing view-dependent effects.

2. **3D Local Attention for Million-Scale Input**:

    - Function: Enables Perceiver IO to efficiently process 1M token inputs.
    - Mechanism: A 3D patchification scheme is designed — input samples are assigned to spatial patches corresponding to 8192 queries via K-NN, with each query attending only to samples within its patch (analogous to ViT's 16×16 patches but generalized to 3D surfaces). Self-attention uses voxel-based windowed attention with a half-voxel shift per layer.
    - Design Motivation: Standard Perceiver IO cross-attention over 1M tokens is computationally prohibitive. 3D patchification approximates surface locality using L2 distance (rather than geodesic distance), offering a favorable speed–accuracy trade-off.

3. **Dual-Branch Decoder (Geometry + View-Dependent Appearance)**:

    - Function: Simultaneously recovers 3D geometry and view-dependent appearance from latent tokens.
    - Geometry Decoder: Models the 3D surface distribution $p(\mathbf{x}|\mathcal{S}) \approx \delta(\mathbf{x} \in \partial\Omega)$ via flow matching. Loss: $\mathcal{L}_{geo} = \mathbb{E}_{t,\mathbf{x}} \|V(\mathbf{x}_t; t) - (\mathbf{x} - \epsilon)\|^2$. Point clouds can be sampled at inference time.
    - **Gaussian Decoder**: Outputs 3D Gaussians with degree-3 spherical harmonics (SH degree 3) for view-dependent rendering. A sparse occupancy grid serves as queries; cross-attention to latent tokens followed by an MLP outputs 64 Gaussians per occupied voxel. Loss: $\mathcal{L}_{radiance} = \|I_{est} - I_{gt}\|^2 + 0.2 \cdot \text{LPIPS}$.
    - Design Motivation: The geometry decoder requires no mesh/occupancy/SDF preprocessing — it learns directly from point clouds. Degree-3 spherical harmonics capture higher-frequency view-dependent effects compared to TRELLIS's view-independent color.

4. **Single-Stage Generation (vs. TRELLIS's Two-Stage Pipeline)**:

    - Function: Latent tokens directly encode complete object information, enabling single-stage generation.
    - Mechanism: A DiT flow-matching model (623M parameters) is trained, with DINOv2-encoded input images conditioning latent generation. During training, the world coordinate system is rotated so that the input-view camera corresponds to the identity transform → the output is automatically aligned with the input view.
    - Design Motivation: TRELLIS requires first generating a coarse occupancy and then generating SLAT (two stages). LiTo's latent tokens already encode complete information, making single-stage generation more straightforward. The coordinate rotation strategy ensures the generated object is aligned with the input view (TRELLIS generates in canonical coordinates and requires post-processing).

### Loss & Training
- Encoder + Decoder: batch size 256, 64 GPUs, 90K iterations, 9 days.
- Generative model (DiT): batch size 256, 128 H100 GPUs, 600K iterations, 20 days.
- Data: 500K object subset of Objaverse-XL (same source as TRELLIS); 3 lighting conditions × 150 views per object.

## Key Experimental Results

### Main Results: Reconstruction Quality (Toys4k)

| Method | PSNR↑ (simple) | SSIM↑ | LPIPS↓ | PSNR↑ (hard) | SSIM↑ | LPIPS↓ |
|--------|---------------|-------|--------|-------------|-------|--------|
| TRELLIS | 31.12 | 0.974 | 0.034 | 27.57 | 0.941 | 0.090 |
| **LiTo** | **34.16** | **0.985** | **0.023** | **32.36** | **0.967** | **0.055** |

### Ablation Study: Generation Quality (Toys4k)

| Method | CLIP↑ | Conditioning View FID↓ | KID↓ | Novel View FID↓ |
|--------|-------|----------------------|------|----------------|
| TRELLIS | 0.899 | 12.84 | 0.088 | 7.600 |
| **LiTo** | **0.905** | **6.219** | **0.009** | **6.216** |

### Key Findings
- **3 dB PSNR gain in reconstruction**: PSNR on the hard setting (close-range cameras) improves from 27.57 to 32.36, demonstrating that view-dependent modeling is particularly important at close viewing distances.
- **Geometry quality improves alongside appearance**: Modeling appearance does not compromise geometric accuracy — among methods that do not use GT coarse geometry, LiTo achieves the best geometry (lowest Chamfer distance).
- **Large gain in generation input fidelity**: Conditioning view FID drops from 12.84 to 6.219, validating the coordinate rotation strategy.
- **Different SH degrees capture distinct features**: degree 0 = diffuse, degree 1 = broad directionality, degrees 2–3 = specular highlights / Fresnel effects.
- **Compact latent space**: 8192×32 = 262K parameters, larger than TRELLIS (20K×11=220K) and TripoSG (2048×64=131K), but without requiring GT coarse geometry.

## Highlights & Insights
- **Surface light fields as a unified framework for 3D representation**: A surface light field can theoretically reconstruct images from arbitrary camera poses — it is the most complete 3D appearance representation. Tokenizing it is a natural yet previously underexplored direction.
- The **random subsampling + encoder interpolation** training strategy is elegant: it does not require the complete surface light field (which is unattainable), only a random subset from which the encoder learns to generalize.
- **3D patchification** elegantly addresses the efficiency challenge of million-scale token input, and is conceptually consistent with ViT's patch design, making it intuitive to understand.
- **Single-stage generation + coordinate rotation** is simpler than TRELLIS's two-stage pipeline and naturally resolves the output alignment problem.

## Limitations & Future Work
- Training data requires RGB-D multi-view renderings (150 views), which incurs substantial acquisition cost.
- 3D patchification approximates surface locality via L2 distance, potentially attending across surfaces when multiple surfaces are in close proximity.
- Transparent and semi-transparent objects are not modeled (depth maps assume the first intersection point).
- Computational cost is high: encoder + decoder training requires 64 GPUs for 9 days; DiT training requires 128 H100s for 20 days.

## Related Work & Insights
- **vs. TRELLIS**: TRELLIS uses DINOv2 mean pooling, discarding directional information and yielding only diffuse color. LiTo encodes directional information to achieve view-dependent appearance. Furthermore, TRELLIS requires two-stage generation in canonical coordinates, whereas LiTo uses single-stage generation with input alignment.
- **vs. TripoSG**: A geometry-only method without appearance modeling. LiTo additionally models view-dependent appearance while also achieving superior geometry quality (without GT coarse geometry).
- **vs. 3DTopia-XL**: PrimX requires a mesh-to-primitive optimization preprocessing step. LiTo constructs inputs directly from RGB-D renderings, making it more scalable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First method to achieve view-dependent appearance modeling within a 3D latent representation; the surface light field tokenization concept is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Detailed comparisons for both reconstruction and generation, validated across multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ — Method description is clear; comparisons with TRELLIS and related methods are explicit.
- Value: ⭐⭐⭐⭐⭐ — Represents a significant advance in 3D generative representation methods, addressing a critical blind spot in view-dependent modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Station2Radar: Query-Conditioned Gaussian Splatting for Precipitation Field](station2radar_query_conditioned_gaussian_splatting_for_precipitation_field.md)
- [\[ICLR 2026\] Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps](joint_shadow_generation_and_relighting_via_light-geometry_interaction_maps.md)
- [\[NeurIPS 2025\] From Pixels to Views: Learning Angular-Aware and Physics-Consistent Representations for Light Field Microscopy](../../NeurIPS2025/3d_vision/from_pixels_to_views_learning_angular-aware_and_physics-consistent_representatio.md)
- [\[CVPR 2026\] Neural Field-Based 3D Surface Reconstruction of Microstructures from Multi-Detector Signals in Scanning Electron Microscopy](../../CVPR2026/3d_vision/neural_field-based_3d_surface_reconstruction_of_microstructures_from_multi-detec.md)
- [\[ICLR 2026\] Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting](augmented_radiance_field_a_general_framework_for_enhanced_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

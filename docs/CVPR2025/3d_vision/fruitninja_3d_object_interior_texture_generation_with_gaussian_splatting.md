---
title: >-
  [Paper Note] FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] FruitNinja proposes the first method for generating internal textures for 3DGS objects. Through progressive cross-section inpainting, voxel smoothing, and the OpaqueAtom GS strategy, it achieves real-time rendering after cutting without additional optimization, significantly outperforming baselines in semantic alignment and texture consistency.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "interior texture generation"
  - "cross-section inpainting"
  - "diffusion model guidance"
  - "real-time rendering"
date: 2026-05-08
content_hash: 01bc001f96e27bbe
---

# FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2411.12089](https://arxiv.org/abs/2411.12089)  
**Code**: None (planned for release)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, interior texture generation, cross-section inpainting, diffusion model guidance, real-time rendering

## TL;DR
FruitNinja proposes the first method for generating internal textures for 3DGS objects. Through progressive cross-section inpainting, voxel smoothing, and the OpaqueAtom GS strategy, it achieves real-time rendering after cutting without additional optimization, significantly outperforming baselines in semantic alignment and texture consistency.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become an efficient novel view synthesis method and is widely used for 3D editing tasks (stylization, deformation, object removal, inpainting, etc.). In interactive 3D applications, users often need to perform geometric operations such as cutting and tearing on objects.

**Limitations of Prior Work**: Current editing methods for 3DGS focus solely on editing the object's outer surface. When an object is cut or sliced, the exposed internal texture is untrained, presenting unrealistic and random noisy colors. Existing solutions either simply fill the same color from the surface inward (PhysGaussian, assuming internal and external texture consistency), or perform 2D inpainting step-by-step after each edit (VR-GS/Infusion, which takes ~30 seconds and is inconsistent across multiple edits).

**Key Challenge**: Acquiring data for the complete internal structure of an object is extremely difficult (requiring CT scans or multiple destructive cuts), yet the internal textures of real objects are often completely different from their outer surfaces (e.g., the green rind and red flesh of a watermelon). Existing methods either assume internal-external consistency or fail to guarantee multi-view consistency.

**Goal**: Synthesize realistic internal textures for 3DGS objects without complete internal data, and support real-time rendering of cuts from arbitrary angles.

**Key Insight**: Leverage the cross-sectional symmetry of common objects—cross-sections in different positions of the same direction often exhibit similar texture patterns. With only a few typical cross-section views as references, combined with diffusion model generation guidance, the model can generalize to the entire internal space.

**Core Idea**: Use a pretrained diffusion model to progressively inpaint a small number of cross-section reference views via SDS loss, coupled with OpaqueAtom GS constraints to achieve stable training and real-time rendering of fine internal textures.

## Method

### Overall Architecture
The input is a reconstructed 3DGS object model. First, the OpaqueAtom GS strategy is used to modify the Gaussian primitives (constraining size and setting full opacity), and original primitives are filled into the empty internal regions of the object. Then, for each user-defined cutting angle, a cross-section reference image is generated through SDS optimization. Next, these reference images, combined with the outer surface views, are used to train the 3DGS, while progressively refining the reference images and periodically performing voxel smoothing. Finally, a complete 3D model with consistent internal and external textures is obtained.

### Key Designs

1. **OpaqueAtom GS (Opaque Atom Gaussian Strategy)**:

    - Function: Ensures the stability and precision of internal texture training, addressing two key flaws of standard 3DGS.
    - Mechanism: Two constraints—(a) Atomic Clipping: limits the max scale of each Gaussian primitive to 1/3000 of the object size, preventing large primitives from crossing multiple cross-section regions and causing training conflicts; (b) Uniform Opacification: sets all Gaussian primitives to be fully opaque (opacity=1), ensuring that front primitives completely block the rear ones to avoid inter-color blending.
    - Design Motivation: Standard 3DGS tends to optimize large Gaussians to cover more area, but large Gaussians cannot represent fine textures and cannot be handled during cutting. Semi-transparent blending creates color artifacts (e.g., blending of green rind and red flesh) after geometric editing.

2. **Conditioned Cross-section Inpainting**:

    - Function: Generates high-quality internal cross-section reference images for each cutting angle.
    - Mechanism: A two-stage optimization—the first stage independently performs SDS optimization on each cross-section, using depth-conditioned Stable Diffusion with angle-specific text prompts (e.g., "horizontal cross-section of a watermelon") to generate initial reference images; the second stage uses the reference images as targets to jointly train 3DGS parameters via $\mathcal{L}_{recon} = \alpha \mathcal{L}_{MSE} + (1-\alpha)\mathcal{L}_{SSIM}$. DreamBooth can optionally be used to fine-tune the diffusion model to adapt to the scarce domain of cross-sectional images.
    - Design Motivation: Directly performing SDS optimization on untrained internal Gaussians is highly inefficient (no features render initially). The two-stage strategy first generates reliable 2D references and then guides the 3D optimization.

3. **Voxel Smoothing & Progressive Refinement**:

    - Function: Resolves spatial inconsistency between different cross-section views, and smooths areas not covered by any cross-section.
    - Mechanism: Texture Refinement—re-renders the current cross-section after each iteration, and then applies a few steps of SDS update on the reference image until the reconstruction loss of all cross-sections converges below a threshold. Voxel Smoothing—constructs a 512³ voxel grid, and assigns colors to untrained Gaussians every 30-40 iterations using distance-weighted averaging $C = \sum_i w_i C_i / \sum_i w_i$.
    - Design Motivation: Cross-sections from different directions may produce conflicting signals in overlapping regions (e.g., a vertical cut shows seeds but the horizontal cut at the same location only shows flesh). Iterative refinement allows the 3DGS and the diffusion model to progressively reach consistency.

### Loss & Training
- SDS loss is used for cross-section reference image generation: $\mathcal{L}_{SDS} = \mathbb{E}_{t,\epsilon}[w(t)\|\epsilon - \epsilon_\theta(\mathbf{I}_{label}^p + \sigma_t \epsilon, t, e, d)\|^2]$
- Reconstruction loss is used for 3DGS training: $\mathcal{L}_{recon} = \alpha \mathcal{L}_{MSE} + (1-\alpha)\mathcal{L}_{SSIM}$
- 20 random outer-surface views are joint-trained in each iteration to prevent appearance degradation.
- Training takes 120-200 iterations, with an initial 20-step SDS for each reference view + 3-4 steps of refinement per iteration.

## Key Experimental Results

### Main Results

| Method | CLIP Score↑ | FID↓ | KID↓(×10⁻³) |
|------|-------------|------|-------------|
| **FruitNinja** | **33.1** | 209.2 | 323.7 |
| 2D Inpainting (fine-tuned) | 32.3 | **176.2** | **224.5** |
| 2D Inpainting | 25.1 | 314.2 | 536.3 |
| PhysGaussian | 24.6 | 520.1 | 816.4 |

FruitNinja achieves the highest score in CLIP Score (semantic alignment), and its KID/FID is ~60% better than PhysGaussian, which is comparable to the fine-tuned 2D Inpainting, but the latter requires ~30 seconds of frame-by-frame optimization.

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| w/o Progressive Refinement | Conflicting textures appear between cross-sections (inconsistent seed locations), with obvious noise |
| w/o Atomic Clipping | The 3D model struggles to converge and fails to generate realistic textures aligned with the reference |
| w/o Uniform Opacity | Fails to accurately express sharp color transitions (blurry and mixed green rind-white pith-red flesh boundary) |
| Full OpaqueAtomGS | Stable convergence, clear texture transitions |

| Method | CLIP Score↑ | Cosine Similarity↑ |
|------|-------------|---------------------|
| **FruitNinja** | **29.1** | **0.96** |
| 2D Inpainting | 27.8 | 0.87 |
| PhysGaussian | 23.9 | 0.89 |

In consistency tests of 120 random-angle cuts, FruitNinja achieves a cosine similarity of 0.96, far exceeding the 0.87 of 2D Inpainting.

### Key Findings
- Progressive refinement is key to resolving conflicts between cross-sections—without it, cross-sections from different directions provide contradictory training signals in overlapping regions.
- Atomic Clipping is crucial for training stability—large Gaussian primitives spanning multiple regions prevent optimization from converging.
- The two components of OpaqueAtom (atomic clipping + uniform opacity) solve different issues: the former ensures precision and stability, while the latter ensures sharp color transitions.

## Highlights & Insights
- **Cross-section symmetry assumption**: Leverages the natural symmetry of common objects (similar horizontal/vertical cross-sections) to generalize to the entire interior stage with only a few cut angles—this observation is simple yet effective.
- **OpaqueAtom design**: The analysis of 3DGS flaws is highly accurate—the tendency of large primitives and semi-transparent blending cause different issues, respectively, and two simple constraints solve them individually.
- **Real-time rendering with zero extra optimization**: Internal textures are already embedded in the 3DGS during training, enabling direct rendering for arbitrary cuts during inference, which is orders of magnitude faster than frame-by-frame inpainting.

## Limitations & Future Work
- Cutting angles and text prompts must be manually specified, limiting automation.
- Only 6 common objects (fruits/cakes/breads) were verified; performance on objects with more complex structures (mechanical parts, biological tissues) remains unknown.
- DreamBooth fine-tuning requires 1-4 real cross-section images, which still incurs data collection costs.
- Iterative training with SDS optimization and voxel smoothing still takes a long time (120-200 iterations), and more efficient guidance strategies can be explored.

## Related Work & Insights
- **vs PhysGaussian**: PhysGaussian simply copies surface colors to the interior (assuming internal-external consistency), yielding blurry and unnatural results. FruitNinja generates semantically reasonable internal textures via diffusion models.
- **vs 2D Inpainting (VR-GS/Infusion)**: Frame-by-frame inpainting requires ~30s/frame and is inconsistent across multiple edits. FruitNinja pre-generates the internal textures, making them available in real time.
- **vs AtomGS**: FruitNinja borrows the small Gaussian densification concept from AtomGS for internal modeling and adds the opacity constraint.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formulates and solves the 3DGS internal texture generation problem for the first time; the problem definition is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐ Only 6 objects are used, the dataset is relatively small, and comparisons with more methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and figures are intuitive.
- Value: ⭐⭐⭐⭐ Directly valuable for VR/gaming interactive scenes, opening up a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)
- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mitigating Ambiguities in 3D Classification with Gaussian Splatting](mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

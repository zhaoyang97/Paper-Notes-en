---
title: >-
  [Paper Note] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model
description: >-
  [CVPR 2025][3D Vision][Single-image 3D Reconstruction] GS-RGBN proposes a hybrid Voxel-Gaussian representation to provide 3D spatial constraints for unstructured Gaussians, and designs a Cross-Volume Fusion (CVF) module to fuse RGB semantic information and normal geometric information at the feature level. It generates high-fidelity 3D objects from a single image within seconds, achieving a PSNR improvement of 5.59dB over the second-best method on the GSO dataset.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Single-image 3D Reconstruction"
  - "2D Gaussian Splatting"
  - "Hybrid Voxel-Gaussian"
  - "Normal Fusion"
  - "Cross-Volume Attention"
  - "Feed-forward 3D Generation"
date: 2026-05-08
content_hash: 9a83b32059fab09a
---

# High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model

**Conference**: CVPR 2025  
**arXiv**: [2504.01512](https://arxiv.org/abs/2504.01512)  
**Code**: None  
**Area**: 3D Vision / Single-image 3D Generation  
**Keywords**: Single-image 3D Reconstruction, 2D Gaussian Splatting, Hybrid Voxel-Gaussian, Normal Fusion, Cross-Volume Attention, Feed-forward 3D Generation

## TL;DR

GS-RGBN proposes a hybrid Voxel-Gaussian representation to provide 3D spatial constraints for unstructured Gaussians, and designs a Cross-Volume Fusion (CVF) module to fuse RGB semantic information and normal geometric information at the feature level. It generates high-fidelity 3D objects from a single image within seconds, achieving a PSNR improvement of 5.59dB over the second-best method on the GSO dataset.

## Background & Motivation

**Background**: Single-image 3D generation is a core demand in the VR/AR/gaming fields. Current mainstream methods fall into three categories: (1) Optimization-based (e.g., DreamGaussian): employing SDS loss for optimization, but suffering from distortions due to inconsistencies in multi-view diffusion images; (2) Fine-tuning-based (e.g., Zero-1-to-3): fine-tuning multi-view diffusion models to improve consistency, which remains insufficient; (3) Feed-forward-based (e.g., LGM, TriplaneGaussian): directly predicting 3DGS from multi-view images using neural networks, but lacking spatial structure in 3DGS, leading to geometric distortion and blurry textures.

**Limitations of Prior Work**: (1) **Geometric ambiguity**: The projection from 2D images to 3D has inherent ambiguity, where RGB information alone is insufficient to recover accurate geometry; (2) **Unstructured 3DGS**: Gaussian primitives are distributed freely in 3D space without constraints, making them prone to collapsing into degenerate solutions when learning from inconsistent 2D images; (3) **Lack of spatial structure in feed-forward methods**: Methods like LGM encode image features using 2D convolutions and map them to Gaussian attributes, failing to effectively capture 3D neighborhood correlations.

**Key Challenge**: Multi-view diffusion models generate images with view inconsistencies $\rightarrow$ directly learning unstructured 3DGS is prone to distortion $\rightarrow$ requiring spatially structured 3D representations and explicit geometric information.

**Key Insight**: (1) Constrain Gaussians with a 3D voxel grid $\rightarrow$ structured 3D coordinates; (2) Utilize normal maps to provide explicit geometric information $\rightarrow$ fuse RGB and normal features to resolve geometric ambiguity.

## Method

### Overall Architecture

GS-RGBN pipeline: Input single image $\rightarrow$ Wonder3D generates multi-view RGB and normal images $\rightarrow$ ViT DINO extracts features + Plücker ray embeddings inject camera information $\rightarrow$ back-projection constructs RGB and normal 3D feature volumes $\rightarrow$ CVF module fuses them into an RGBN volume $\rightarrow$ MLP decodes 2D Gaussian attributes of each voxel $\rightarrow$ rendering.

### Key Designs

1. **Hybrid Voxel-Gaussian**:

    - **Function**: Provides structured 3D spatial constraints for unstructured Gaussian primitives.
    - **Mechanism**: Uses ViT DINO to extract feature maps from multi-view RGB/normal images, injects Plücker ray embeddings using Adaptive Layer Normalization (AdaLN) to encode camera poses, and then back-projects the fused features along rays into a $W \times W \times W$ 3D voxel grid, averaging multi-view features at the same location. Each voxel uses an MLP to decode attributes of a 2D Gaussian (offsets $\Delta x_i \in [-1,1]^3$, scale, rotation, opacity, and SH coefficients).
    - **Design Motivation**: The voxel grid establishes an explicit correspondence between 3D positions and 2D projected features, enabling pagan 3D convolutions to effectively capture correlations between neighboring Gaussians. The ablation study shows that removing voxels (Image-Gaussian mode) drops the PSNR by 4.2dB.

2. **Cross-Volume Fusion (CVF)**:

    - **Function**: Fuses RGB semantic information and normal geometric information at the feature level.
    - **Mechanism**: Four Voxel Residual Blocks (VRB, channels 512 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 32) downsample two volume features. Then, two cross-attention blocks—RGB-guided $CA_s$ (RGB $\rightarrow$ Query, Normal $\rightarrow$ KV) and normal-guided $CA_g$ (Normal $\rightarrow$ Query, RGB $\rightarrow$ KV)—individually generate fused volumes. Finally, after concatenation, Self-Attention (SA) balances semantic and geometric weights to output the final RGBN volume. To reduce memory, the $32^3$ volume is unfolded into 16 groups to perform attention individually.
    - **Design Motivation**: RGB captures semantics/textures while normals capture geometric details. Bidirectional cross-attention balances the two types of information dynamically better than simple concatenation.

3. **2D Gaussian Splatting**:

    - **Function**: Ensures geometrically consistent surface modeling and precise depth calculation.
    - **Mechanism**: Employs 2DGS instead of 3DGS, where each Gaussian is a flat elliptical disk rather than a 3D ellipsoid. The key advantage lies in depth calculation: the depth of 3DGS is the alpha-blending of central z-values, which is inaccurate due to depth changes as rays pass though the ellipsoid; 2DGS accurately calculates the depth of each pixel through ray-disk intersections.
    - **Design Motivation**: The intrinsic surface modeling of 2DGS makes depth and normal losses meaningful, fundamentally guaranteeing geometric consistency.

### Loss & Training

- Total loss: $\mathcal{L}_{total} = \mathcal{L}_c + \lambda_d \mathcal{L}_d + \lambda_{reg} \mathcal{L}_{reg}$
- Color loss: $\mathcal{L}_c = L1(RGB) + L1(\alpha) + 0.5 \times LPIPS(RGB)$
- Depth loss: $\mathcal{L}_d = L1(D, \hat{D})$
- Regularization loss $\mathcal{L}_{reg}$: self-supervised distortion loss + normal consistency loss
- AdamW optimizer with initial lr = 1e-5 + cosine annealing
- Trained on 4 $\times$ A100 (40G) for ~6.5 days, batch 4/GPU, bfloat16
- Training data: Objaverse-LVIS (~40K objects), Evaluation: GSO ~200 objects

## Key Experimental Results

### Main Results

GSO Dataset Novel View Synthesis:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ (×10⁻³) | Time(r) |
|------|-------|-------|--------|-------------|---------|
| DreamGaussian | 17.43 | 0.810 | 0.265 | 205.23 | 28.32s |
| LGM | 17.13 | 0.808 | 0.199 | 104.71 | 0.33s |
| TriplaneGaussian | 16.73 | 0.793 | 0.259 | 58.74 | 0.11s |
| **GS-RGBN** | **23.02** | **0.873** | **0.135** | **27.49** | 0.20s |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Image-Gaussian (w/o voxels) | 18.82 | 0.831 | 0.209 |
| w/o normal input | 20.15 | 0.848 | 0.172 |
| w/o CVF | 19.27 | 0.843 | 0.198 |
| w/o $CA_g$ (normal-guided attention) | 21.32 | 0.853 | 0.163 |
| w/o $CA_s$ (RGB-guided attention) | 21.08 | 0.852 | 0.166 |
| **Full model** | **23.02** | **0.873** | **0.135** |

Views ablation: 4 views $\rightarrow$ 20.06 PSNR, 6 views $\rightarrow$ 22.70, 8 views $\rightarrow$ **23.02**

### Key Findings

- The PSNR improves by 5.59dB compared to the second-best method (DreamGaussian: 17.43), and Chamfer Distance drops from 58.74 to 27.49, indicating significantly enhanced geometric quality.
- Removing the hybrid Voxel-Gaussian representation (reverting to Image-Gaussian) leads to a 4.2dB decrease in PSNR, which is the most impactful component among all ablations.
- Removing normal input drops the PSNR by 2.87dB, and removing CVF drops it by 3.75dB, demonstrating that both the normal information and the fusion strategy are indispensable.
- The normal-guided attention $CA_g$ contributes slightly more than the RGB-guided attention $CA_s$ (PSNR drops by 1.70dB vs 1.94dB when removed, respectively), indicating that geometric information is slightly more critical.
- Even with only 4 views, it still outperforms all baseline methods, demonstrating strong robustness.

## Highlights & Insights

1. **Structuring is Key**: The 4.2dB improvement from Image-Gaussian to Voxel-Gaussian proves that "introducing spatial constraints to unstructured 3DGS" is key to learning from inconsistent multi-view images.
2. **Value of Normal Information**: RGB and normal information are complementary: RGB provides semantics/textures, whereas normals provide geometry. The ablation shows that normal-guided cross-attention is more crucial than RGB-guided cross-attention, indicating that geometric priors are scarcer in 3D reconstruction.
3. **Justification for Replacing 3DGS with 2DGS**: The precision of depth calculation is the key difference; the ray-disk intersection depth of 2DGS makes depth loss and normal consistency loss meaningful.

## Limitations & Future Work

- Heavily relies on the generation quality of the Wonder3D multi-view diffusion model; performance degrades when the generated multi-view images have greater inconsistencies.
- The voxel resolution ($32^3$) limits the representation of geometric details; larger scenes require sparse data structures like octrees.
- Currently, it only supports object-level 3D generation. Large-scale scene generation is impracticable due to the high memory overhead of voxels.
- Although the rendering speed (0.20s) is faster than optimization-based methods, it is slower than TriplaneGaussian (0.11s).

## Related Work & Insights

- Comparison with the LRM series (e.g., LGM, InstantMesh): The core difference of GS-RGBN is the introduction of a 3D-native structure (voxels) instead of pure 2D processing.
- The concept of normal fusion can be extended to other tasks requiring geometric clues, such as texture generation and relighting.
- The hybrid Voxel-Gaussian representation opens up a new direction for the application of 3DGS in structured learning.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: Both the hybrid Voxel-Gaussian and the RGBN cross-volume fusion are rational and effective designs.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Main results plus comprehensive ablation studies (representations/losses/fusion strategies/number of views), with clear qualitative comparisons.
- **Writing Quality** ⭐⭐⭐⭐⭐: The methodology description is clear and the flowchart is intuitive.
- **Value** ⭐⭐⭐⭐: High-quality 3D objects are generated within seconds, making it industrially viable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Floating No More: Object-Ground Reconstruction from a Single Image](floating_no_more_object-ground_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)

</div>

<!-- RELATED:END -->

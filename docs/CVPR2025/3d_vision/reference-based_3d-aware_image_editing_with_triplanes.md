---
title: >-
  [Paper Note] Reference-Based 3D-Aware Image Editing with Triplanes
description: >-
  [CVPR 2025 (Highlight)][3D Vision][Triplane representation] Based on the EG3D triplane representation space, a reference-image-guided 3D-aware editing framework is proposed, integrating four modules: encoder, automatic localization, spatial disentanglement, and fusion learning. It achieves editing results superior to existing 2D/3D GAN and diffusion methods across diverse domains including human faces, 360-degree heads, animals, cartoons, and full-body clothing.
tags:
  - "CVPR 2025 (Highlight)"
  - "3D Vision"
  - "Triplane representation"
  - "Reference-based image editing"
  - "3D-aware"
  - "EG3D"
  - "Spatial disentanglement"
  - "Fusion learning"
date: 2026-05-08
content_hash: f92f4d2ad69c5579
---

# Reference-Based 3D-Aware Image Editing with Triplanes

**Conference**: CVPR 2025 (Highlight)  
**arXiv**: [2404.03632](https://arxiv.org/abs/2404.03632)  
**Code**: None (to be released)  
**Area**: 3D Vision / Image Editing  
**Keywords**: Triplane representation, Reference-based image editing, 3D-aware, EG3D, Spatial disentanglement, Fusion learning

## TL;DR

Based on the EG3D triplane representation space, a reference-image-guided 3D-aware editing framework is proposed, integrating four modules: encoder, automatic localization, spatial disentanglement, and fusion learning. It achieves editing results superior to existing 2D/3D GAN and diffusion methods across diverse domains including human faces, 360-degree heads, animals, cartoons, and full-body clothing.

## Background & Motivation

**Background**: GANs have become a powerful tool for high-quality image generation and editing, enabling various editing effects by manipulating their latent spaces. 3D-aware GANs, such as EG3D, introduce an efficient triplane architecture capable of reconstructing 3D geometry from a single image. However, most existing works focus on text-guided or latent-direction-guided editing, leaving a lack of systematic frameworks for reference-image-guided 3D-aware editing.

**Limitations of Prior Work**: (1) Text-guided editing (e.g., InstructPix2Pix) is limited in its ability to precisely control local attribute changes; (2) Latent-direction editing (e.g., InterfaceGAN) can only vary along predefined semantic axes, failing to achieve attribute transfer from arbitrary reference images; (3) Existing 2D editing methods lack 3D consistency, resulting in inconsistencies when viewed from different perspectives after editing; (4) There is a lack of a unified framework capable of handling different types of reference-image-guided editing (e.g., changing hairstyles, facial expressions, outfits).

**Key Challenge**: Reference-image-guided editing requires precisely locating "what to edit" and "what to edit it into" while keeping unedited regions completely unchanged and 3D-consistent — which is difficult to achieve in 2D latent spaces due to highly entangled attributes.

**Goal**: To design a unified reference-image-guided editing framework based on the triplane space, achieving precise, 3D-consistent, multi-domain reference-image-guided editing.

**Key Insight**: The triplane representation naturally possesses spatial disentanglement properties — the XY, XZ, and YZ planes correspond to different spatial dimensions, respectively, providing a natural basis for spatial separation in local editing.

**Core Idea**: To conduct automatic localization and spatially disentangled editing of reference image features in the triplane space — finding the corresponding regions of the source and reference images on the triplanes, disentangling the spatial regions that need editing, and then seamlessly fusing the reference attributes into the source image's triplane representation via fusion learning.

## Method

### Overall Architecture

Given a source image and a reference image, the framework operates in four steps: (1) an encoder maps both images into the EG3D triplane space; (2) an automatic localization module identifies the triplane regions requiring editing; (3) a spatial disentanglement module separates the edited and preserved regions; and (4) a fusion learning module integrates the target attributes of the reference image into the source image's triplane representation. Finally, the edited 3D-aware image is synthesized via EG3D's neural renderer.

### Key Designs

1. **Triplane Encoder**:

    - **Function**: Maps real images into the EG3D triplane latent space.
    - **Mechanism**: Using a pre-trained EG3D generator, the source and reference images are encoded as triplane features via GAN inversion. The triplane representation $T \in \mathbb{R}^{3 \times H \times W \times C}$ comprises three orthogonal planes (XY, XZ, YZ), each encoding feature information of the corresponding spatial dimension. The encoder must guarantee reconstruction quality while preserving the semantic structure of the triplane feature space.
    - **Design Motivation**: The triplane space possesses better spatial disentanglement compared to the conventional $W+$ latent space, making it a key component for enabling precise local editing.

2. **Automatic Localization & Spatial Disentanglement**:

    - **Function**: Automatically identifies the corresponding edited regions between the source and reference images, and separates the edited and preserved regions in the triplane features.
    - **Mechanism**: By comparing the feature differences of the source and reference images on the triplanes, the spatial regions requiring editing are automatically located. Utilizing the spatial structure of the triplanes, region segmentation is performed on the XY, XZ, and YZ planes individually — locating the frontal region on the XY plane, while the XZ and YZ planes provide profile and depth information. After disentanglement, the triplane features of the edited region are derived from the reference image, whereas those of the preserved region come from the source image.
    - **Design Motivation**: Manually specifying edited regions is time-consuming and inaccurate. The three orthogonal planes of the triplane provide a natural multi-view spatial segmentation basis, rendering automatic localization feasible.

3. **Fusion Learning**:

    - **Function**: Seamlessly fuses the target attribute features of the reference image into the source image's triplane representation.
    - **Mechanism**: Rather than performing simple feature concatenation or replacement, a fusion network is learned to handle the boundary transitions between the edited and preserved regions. The fusion network learns within the triplane space: (a) which feature channels need to be transferred from the reference image; (b) how to smooth transitions to avoid artifacts at editing boundaries; and (c) how to maintain consistency across the three planes. The final fused triplane is fed into the EG3D renderer to generate the edited image.
    - **Design Motivation**: Simple concatenation produces boundary artifacts and potentially causes inconsistency among the triplanes. Fusion learning ensures natural editing results and 3D consistency.

### Loss & Training

Training utilizes a combination of multiple loss functions: (1) reconstruction loss to guarantee encoder quality; (2) perceptual loss (LPIPS) to ensure visual quality; (3) identity-preservation loss to keep non-edited regions unchanged; and (4) adversarial loss to maintain the realism of synthesized images. In terms of training strategy, a progressive scheme is adopted, training the encoder first, followed by the localization and fusion modules.

## Key Experimental Results

### Main Results: Multi-Domain Reference-Based Image Editing Quality Comparison

| Method | Type | FID ↓ | ID Preservation ↑ | Editing Accuracy ↑ | 3D Consistency |
|------|------|-------|----------|----------|---------|
| InterfaceGAN | GAN/Latent Direction | High | Medium | Low | Yes |
| StyleCLIP | GAN/Text | Medium | Medium | Medium | No |
| InstructPix2Pix | Diffusion/Text | Medium | High | Medium | No |
| 3D-aware Diffusion | Diffusion/3D | High | High | Medium | Yes |
| **Ours** | **GAN/Reference Image** | **Lowest** | **Highest** | **Highest** | **Yes** |

### Ablation Study: Contribution of Each Module

| Configuration | FID ↓ | ID Preservation | Editing Quality | Description |
|------|-------|--------|---------|------|
| Encoder Only + Direct Replacement | High | Low | Low | No disentanglement leads to global changes |
| + Automatic Localization | Medium | Medium | Medium | More precise edited regions |
| + Spatial Disentanglement | Low | High | High | Separation of edited-preserved regions |
| **+ Fusion Learning (Full Model)** | **Lowest** | **Highest** | **Highest** | Natural boundaries and 3D consistent |

### Key Findings

- The triplane space is more suitable for local editing than the traditional $W+$ latent space; spatial disentanglement capability is a key advantage.
- The method is effective across 6 diverse domains (human faces, 360-degree heads, animal faces, cartoons, full-body clothing, and class-agnostic samples), demonstrating the versatility of the framework.
- The fusion learning module is crucial for boundary quality — significant artifacts appear at editing boundaries when omitted.
- The automatic localization module eliminates the need for manual annotation of edited regions, significantly improving practicality.
- It significantly outperforms 2D editing methods in 3D consistency, yielding consistent results when observed from different viewpoints after editing.

## Highlights & Insights

- **Editing Potential of Triplane Space**: This work is the first to systematically explore the application of EG3D's triplane space in reference-image-guided editing. The three orthogonal planes of the triplane provide a natural basis for spatial disentanglement, an insight that can be generalized to other triplane-based 3D generative models.
- **Unified Multi-Domain Framework**: A single framework covers multiple scenarios, such as facial editing, animal editing, and clothing editing, without requiring independent model training for each domain.
- **3D-Consistent Reference Editing**: Performing edits within the triplane space rather than the 2D latent space naturally guarantees multi-view 3D consistency, which is unachievable with 2D editing methods.

## Limitations & Future Work

- It relies on the generation quality and domain coverage of EG3D; performance may be limited on scene types not covered by EG3D.
- The reconstruction accuracy of GAN inversion is a bottleneck — if the encoder fails to precisely reconstruct the source image, the editing results will be affected accordingly.
- The generation resolution is constrained by EG3D's output resolution (typically $512 \times 512$), rendering it inapplicable to high-resolution scenarios directly.
- Comparison with recent 3D-aware diffusion methods (e.g., Zero-1-to-3, Wonder3D) is limited.
- Temporal consistency under video editing scenarios has not been explored.

## Related Work & Insights

- **vs EG3D**: EG3D is a generative model, while this work develops editing capabilities based on its triplane space.
- **vs InterfaceGAN/StyleCLIP**: Latent-direction/text-guided editing is less flexible than reference-guided editing and lacks precise local control.
- **vs InstructPix2Pix/ControlNet**: Diffusion-based editing methods perform well in 2D but lack 3D consistency.
- **vs PTI/HyperStyle**: While both involve GAN inversion-based editing, this work operates in the triplane space to achieve superior spatial disentanglement.
- The methodology of triplane-space editing can be extended to the next generation of 3D generative models based on 3DGS or NeRF.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically utilize the triplane space for reference-image-guided editing; the spatial disentanglement concept is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across multiple domains with thorough qualitative and quantitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the framework and well-defined design motivations for each module.
- Value: ⭐⭐⭐⭐ Provides a practical framework for 3D-aware reference-image-guided editing, fully deserving its CVPR Highlight status.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion](../../NeurIPS2025/3d_vision/geocomplete_geometry-aware_diffusion_for_reference-driven_image_completion.md)
- [\[CVPR 2026\] ObjectMorpher: 3D-Aware Image Editing via Deformable 3DGS](../../CVPR2026/3d_vision/objectmorpher_3d-aware_image_editing_via_deformable_3dgs.md)
- [\[ICML 2025\] FlowDrag: 3D-aware Drag-based Image Editing with Mesh-guided Deformation Vector Flow Fields](../../ICML2025/3d_vision/flowdrag_3d-aware_drag-based_image_editing_with_mesh-guided_deformation_vector_f.md)
- [\[CVPR 2025\] Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion](ouroboros3d_image-to-3d_generation_via_3d-aware_recursive_diffusion.md)
- [\[CVPR 2025\] PrEditor3D: Fast and Precise 3D Shape Editing](preditor3d_fast_and_precise_3d_shape_editing.md)

</div>

<!-- RELATED:END -->

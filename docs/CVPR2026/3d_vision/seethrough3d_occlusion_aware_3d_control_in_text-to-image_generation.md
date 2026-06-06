---
title: >-
  [Paper Note] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation
description: >-
  [CVPR2026][3D Vision][3D layout control] This paper proposes SeeThrough3D, which conditions the FLUX model on an Occlusion-aware Scene Control Representation (OSCR) rendered from semi-transparent 3D bounding boxes…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "3D layout control"
  - "occlusion awareness"
  - "text-to-image generation"
  - "DiT"
  - "FLUX"
  - "attention mask"
  - "LoRA"
date: 2026-05-08
content_hash: 4e690cf8f87e6d3e
---

# SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation

**Conference**: CVPR2026  
**arXiv**: [2602.23359](https://arxiv.org/abs/2602.23359)  
**Code**: [Project Page](https://seethrough3d.github.io)  
**Area**: 3D Vision / Image Generation  
**Keywords**: 3D layout control, occlusion awareness, text-to-image generation, DiT, FLUX, attention mask, LoRA

## TL;DR

This paper proposes SeeThrough3D, which conditions the FLUX model on an Occlusion-aware Scene Control Representation (OSCR) rendered from semi-transparent 3D bounding boxes, enabling precise 3D layout control and occlusion-consistent text-to-image generation.

## Background & Motivation

**Limitations of 2D control**: Most existing controllable generation methods rely on 2D spatial controls (bounding boxes, segmentation maps) and cannot govern 3D scene properties such as object arrangement and camera viewpoint, failing to meet the demands of design, gaming, and architectural visualization.

**Neglected occlusion reasoning**: Occlusion reasoning is central to 3D-aware generation, yet existing 3D layout methods (e.g., LooseControl, Build-A-Scene) condition on depth maps, which cannot represent occluded objects and frequently fail in multi-object overlapping scenes.

**Insufficient 2D layer decomposition**: Methods such as LaRender and VODiff decompose scenes into 2D object layers to approximate occlusion, but this planar representation discards true 3D geometry, causing occlusion relationships that violate 3D perspective rules.

**Missing semantic binding**: Spatial conditioning cannot associate 3D bounding boxes with their corresponding textual descriptions, leading to attribute confusion and positional errors.

**Inadequate orientation control**: Depth maps encode orientation information only within a 180° range and cannot provide full 3D orientation control.

**Generalization challenges with synthetic data**: Models trained on synthetic data tend to overfit to synthetic backgrounds, necessitating effective data augmentation strategies to ensure generalization to real scenes.

## Method

### Overall Architecture

SeeThrough3D is built upon the pretrained FLUX (DiT architecture) text-to-image model, conditioning the generation process by introducing visual tokens derived from the OSCR representation. The pipeline proceeds as follows: the user places semi-transparent 3D bounding boxes in a virtual environment and sets the camera viewpoint → Blender renders the OSCR image → the image is encoded into OSCR tokens via VAE → the tokens are concatenated with text tokens and noisy image tokens and jointly processed by mmDiT blocks.

### Key Design 1: OSCR Occlusion-Aware 3D Scene Representation

- Each object is represented by a **semi-transparent 3D bounding box**; the transparency makes partially occluded objects visible, providing the model with explicit cues for occlusion reasoning.
- Each face of the bounding box adopts a **predefined canonical color mapping**, where different faces correspond to different colors, directly encoding 3D orientation information in image space.
- The entire scene is rendered from a specified **camera viewpoint**, naturally embedding camera pose information into the rendered image for precise viewpoint control.
- Occlusion may alter the apparent color of certain faces, but the relative color difference between faces remains distinguishable, preserving the reliability of orientation cues.

### Key Design 2: Attention Mask Object Binding

- An attention mask is applied to the self-attention in mmDiT blocks: tokens within OSCR that belong to bounding box $b_i$ **can only attend to** the corresponding object noun token $\mathbf{p}_i$, thereby binding spatial OSCR tokens to their corresponding object semantics.
- The spatial extent is obtained by rendering the amodal segmentation mask $\mathbf{s}_i$ for each bounding box in Blender.
- **Handling overlapping regions**: When the rendered regions of two bounding boxes overlap, OSCR tokens in the intersection attend to multiple object tokens. Experiments show that the model can maintain separable object features in latent space — the attention maps themselves reveal occlusion boundaries.
- Attention from OSCR tokens $\mathbf{z}$ to image tokens $\mathbf{x}_t$ is also blocked to preserve the base model prior.

### Key Design 3: Personalization Extension

- Given a reference object image $v$, it is encoded via VAE into "appearance tokens" $\mathbf{v}$ and concatenated into the token sequence.
- The attention mask strategy is reused so that OSCR tokens within bounding box $b_i$ attend to the appearance tokens, enabling layout-aware personalized object generation.

### Loss & Training

- Only **LoRA** (rank=128) on the projection matrices corresponding to OSCR tokens is trained, preserving the base model's text-to-image prior.
- Learning rate $10^{-4}$, trained for 30K steps.

### Dataset Construction

- 3D assets are procedurally placed in Blender with controlled object positions and camera parameters to produce strong occlusion; paired real images and OSCR representations are rendered.
- **Data augmentation**: Depth is extracted from rendered images, and FLUX.1-Depth-dev is used to generate diverse photorealistic augmented images; CLIP filters samples inconsistent with the original layout.
- **Hard sample filtering**: Scenes with minimal object overlap or excessively low object visibility are discarded; this filtering is critical for occlusion consistency.
- Final dataset: 25K rendered images + 25K augmented images.

## Key Experimental Results

### Main Results (3DOc-Bench, 500 samples)

| Method | Depth Order↑ | Object Score↑ | Angle Error↓ | Text Align↑ | KID(×10⁻³)↓ |
|--------|-------------|---------------|--------------|-------------|--------------|
| VODiff | 0.68 | 19.70 | 92.73 | 29.51 | 15.40 |
| LooseControl | 0.82 | 20.02 | 89.88 | 28.43 | 14.32 |
| Build-A-Scene | 0.89 | 21.0 | 91.62 | 28.05 | 20.12 |
| LaRender | 1.02 | 21.83 | 89.63 | 30.20 | 13.46 |
| **SeeThrough3D** | **1.46** | **22.86** | **47.92** | **31.87** | **5.43** |

SeeThrough3D outperforms all existing methods by a substantial margin across all five metrics. Angle error drops from ~90° to 48°, and KID from 13+ to 5.43.

### Ablation Study

| Variant | Depth Order↑ | Object Score↑ | Angle Error↓ | Text Align↑ | KID(×10⁻³)↓ |
|---------|-------------|---------------|--------------|-------------|--------------|
| w/o transparency | 1.20 | 21.67 | **46.15** | 31.39 | 5.90 |
| w/o color encoding | 1.36 | 22.23 | 88.77 | 31.57 | 5.93 |
| w/o binding mask | 0.98 | 20.45 | 57.44 | 31.61 | 6.35 |
| w/o hard data | 1.24 | 21.89 | 49.73 | 31.32 | 6.34 |
| **Full model** | **1.46** | **22.86** | 47.92 | **31.87** | **5.43** |

### Key Findings

- **Transparency** is the core design of OSCR: removing it drops depth order from 1.46 to 1.20, demonstrating its importance for occlusion reasoning. Opaque bounding boxes yield marginally better orientation accuracy (cleaner color signals) but sacrifice occlusion modeling.
- **Color encoding** is critical for orientation control: removing it causes angle error to surge from 48° to 89°, nearly reverting to baseline levels.
- **Attention binding** is indispensable for layout compliance: removing it drops object score from 22.86 to 20.45, with objects appearing at incorrect positions.
- **Hard sample filtering** effectively improves model performance in complex occlusion scenarios.
- In a user study with 60 participants, SeeThrough3D achieves high preference rates on image realism, layout compliance, and text alignment.
- Despite training only on synthetic scenes with at most 4 objects, the model generalizes to multi-object scenes, unseen categories (musical instruments, electronic devices, transparent objects, etc.), diverse poses (sitting, riding), and natural object interactions.

## Highlights & Insights

- **Elegant OSCR representation**: The semi-transparent, color-encoded 3D bounding boxes are both concise and expressive, simultaneously encoding occlusion, orientation, and camera viewpoint.
- **Attention mask binding** elegantly resolves the spatial-semantic association problem without being restricted to a fixed set of object categories.
- **Strong generalization**: Trained on only 50K synthetic samples (including augmentation), the model generalizes to unseen object categories, complex layouts, and diverse backgrounds.
- **Preservation of base model prior**: The LoRA fine-tuning combined with attention blocking allows the model to retain original capabilities such as transparent object rendering and text generation.
- The proposed **3DOc-Bench** benchmark fills a gap in the evaluation of occlusion-aware 3D layout control.

## Limitations & Future Work

- **Image consistency is not preserved** across layout changes; modifying the layout produces a completely different image, lacking editing continuity.
- Training data covers only **rigid objects in canonical poses**, potentially limiting control over non-rigid objects and complex poses.
- The pipeline depends on Blender to render OSCR images and segmentation masks, resulting in a **lengthy user interaction workflow**.
- No single metric captures 3D layout compliance; evaluation relies on a combination of three proxy metrics: depth order, object score, and angle error.

## Related Work & Insights

- **3D layout control**: LooseControl conditions on depth maps but cannot represent occluded objects; Build-A-Scene incrementally adds objects via multi-round generation-inversion but introduces artifacts; LACONIC and similar methods take bounding boxes as set inputs but are limited to a single scene domain.
- **Occlusion control**: LaRender and VODiff rely on 2D layer decomposition and lack 3D awareness; CObL performs unordered layer decomposition but still provides no 3D layout control.
- **Orientation control**: Compass Control and ORIGEN provide object orientation control but do not support 3D positional placement.
- **3D-aware editing**: Diffusion Handles, 3D-FixUp, and similar methods leverage depth for 3D editing but are limited to single objects.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of OSCR representation and attention mask binding is novel, explicitly modeling occlusion reasoning within the scene representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five metrics + ablation study + user study + personalization extension, with the self-constructed 3DOc-Bench benchmark.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures, and in-depth attention visualization analysis.
- Value: ⭐⭐⭐⭐ — Fills the gap in occlusion-aware 3D layout control with a concise and practical design; generalization capability is impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](text-image_conditioned_3d_generation.md)
- [\[NeurIPS 2025\] GOATex: Geometry & Occlusion-Aware Texturing](../../NeurIPS2025/3d_vision/goatex_geometry_occlusion-aware_texturing.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance](gaussiangrow_geometry-aware_gaussian_growing_from_3d_point_clouds_with_text_guid.md)
- [\[CVPR 2026\] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation
description: >-
  [CVPR 2026][3D Vision][3D garment generation] SwiftTailor is a lightweight two-stage framework that combines PatternMaker for sewing pattern prediction with GarmentSewer for converting patterns into a Garment Geometry Im…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D garment generation"
  - "geometry image"
  - "sewing patterns"
  - "VLM"
  - "Dense Prediction Transformer"
date: 2026-05-08
content_hash: 892f8aa4fb6aab2f
---

# SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation

**Conference**: CVPR 2026
**arXiv**: [2603.19053](https://arxiv.org/abs/2603.19053)  
**Authors**: Phuc Pham, Uy Dieu Tran, Binh-Son Hua, Phong Nguyen
**Code**: To be confirmed  
**Area**: 3D Vision / Garment Generation
**Keywords**: 3D garment generation, geometry image, sewing patterns, VLM, Dense Prediction Transformer

## TL;DR

SwiftTailor is a lightweight two-stage framework that combines PatternMaker for sewing pattern prediction with GarmentSewer for converting patterns into a Garment Geometry Image (GGI) in a unified UV space. Via inverse mapping and dynamic stitching, the framework directly assembles 3D garment meshes, achieving SOTA quality while running orders of magnitude faster than existing methods.

## Background & Motivation

3D garment generation is a long-standing challenge in computer vision and digital fashion. Typical pipelines rely on large vision-language models (VLMs) to produce serialized representations of 2D sewing patterns, which are then converted into simulatable 3D meshes via frameworks such as GarmentCode. Although these approaches yield high-quality results, they suffer from notable bottlenecks:

**Low inference efficiency**: Dependence on physics simulation engines (e.g., GarmentCode) to convert 2D patterns into 3D meshes results in 30–60 seconds per garment, making real-time or large-scale generation impractical.

**VLM redundancy**: Employing large-scale VLMs for sewing pattern prediction introduces unnecessary parameter overhead; lightweight specialized models are sufficient for this task.

**Lack of unified representation**: The conversion from 2D patterns to 3D meshes relies on complex simulation pipelines with multiple non-differentiable intermediate steps, hampering end-to-end optimization.

**Core Problem**: How can inference efficiency in 3D garment generation be substantially improved without sacrificing generation quality?

## Method

### Overall Architecture

SwiftTailor adopts a cascaded two-stage design:

- **Stage 1 — PatternMaker**: A lightweight VLM that predicts sewing pattern parameters from multimodal inputs (text descriptions, reference images, etc.).
- **Stage 2 — GarmentSewer**: An efficient Dense Prediction Transformer that converts sewing pattern parameters into a Garment Geometry Image (GGI), encoding the 3D surface of all panels into a unified UV space.
- **Post-processing**: The final 3D garment mesh is assembled via inverse mapping, remeshing, and dynamic stitching.

The core idea is to replace conventional physics simulation with a learned geometry image representation, amortizing the expensive simulation cost to the training phase.

### Key Design 1: PatternMaker — Efficient Vision-Language Pattern Prediction

PatternMaker is a lightweight vision-language model designed specifically for sewing pattern prediction:

- **Multimodal input**: Supports text descriptions, reference images, and other modalities for flexible user interaction.
- **Efficient architecture**: Substantially reduces model scale compared to large VLMs (e.g., GPT-4V-class models) used in prior work, retaining only the capabilities necessary for pattern prediction.
- **Structured output**: Directly predicts a parameterized representation of sewing patterns, including panel shapes, dimensions, and stitching relationships, without complex sequence decoding.
- **Multimodal training**: Trained on the Multimodal GarmentCodeData dataset, learning joint mappings from visual and textual inputs to pattern parameters.

The key insight is that sewing pattern prediction is fundamentally a structured prediction task that does not require the full capacity of a large general-purpose VLM; a lightweight, task-specific model achieves a better efficiency–accuracy trade-off.

### Key Design 2: GarmentSewer — Garment Geometry Image Generation

GarmentSewer introduces the Garment Geometry Image (GGI), the core contribution of this framework:

- **GGI representation**: Encodes the 3D surface information of all garment panels into a unified 2D UV space, where each pixel stores a 3D coordinate $(x, y, z)$. This reformulates the irregular 3D mesh prediction problem as a regular 2D image prediction problem.
- **Dense Prediction Transformer**: An efficient DPT architecture conditioned on sewing pattern parameters directly predicts the GGI. The global attention mechanism of the Transformer facilitates capturing spatial relationships across different panels.
- **UV space design**: A carefully designed UV mapping arranges panels of varying shapes and sizes compactly in a unified image space, maximizing information density while maintaining geometric consistency across panels.

### Key Design 3: Inverse Mapping and Dynamic Stitching

Reconstructing the final 3D mesh from the GGI involves three key steps:

- **Inverse mapping**: Maps the 3D coordinates of each valid GGI pixel back to the original panel space, recovering the 3D geometry of each panel.
- **Remeshing**: Applies adaptive remeshing to the recovered panel geometry to produce high-quality triangle meshes suitable for downstream applications.
- **Dynamic stitching**: Automatically seams corresponding panel edges according to the stitching relationships defined in the sewing pattern, assembling the complete garment. This algorithm handles practical issues such as mismatched edge lengths between panels.

This pipeline fully replaces conventional physics simulation, reducing per-garment assembly time from tens of seconds to sub-second latency.

## Key Experimental Results

Experiments are conducted on the Multimodal GarmentCodeData dataset.

### Table 1: Quantitative Comparison with Existing Methods

| Method | Pattern Accuracy | 3D Geometry Error ↓ | Visual Fidelity ↑ | Inference Time |
|--------|-----------------|--------------------|--------------------|----------------|
| GarmentCode + Large VLM | High | Low | High | 30–60 s |
| Serialization-based methods | Moderate | Moderate | Moderate | ~30 s |
| **SwiftTailor** | **Highest** | **Lowest** | **Highest** | **< few seconds** |

SwiftTailor achieves SOTA accuracy while improving inference speed by more than an order of magnitude.

### Table 2: Ablation Study

| Configuration | Geometry Error ↓ | Inference Time | Notes |
|---------------|-----------------|----------------|-------|
| Full SwiftTailor | Lowest | Fastest | Complete two-stage framework |
| w/o GGI (physics simulation) | Comparable | 30–60 s | Validates GGI as a simulation substitute |
| w/o Dynamic Stitching | Higher | Fast | Degraded stitching quality |
| w/o Remeshing | Moderate | Fastest | Reduced mesh quality |
| Large VLM replacing PatternMaker | Comparable | Slower | Validates the lightweight VLM design |

Ablation results confirm the necessity of GGI, dynamic stitching, and remeshing individually.

## Key Findings

1. **Geometry images are an efficient representation for 3D garments**: GGI unifies irregular 3D meshes into a regular 2D image space, enabling standard image prediction architectures to be directly applied to garment generation.
2. **Physics simulation can be replaced by learning**: By amortizing simulation costs to the training phase, inference requires no physics engine, substantially reducing latency.
3. **Lightweight VLMs suffice for pattern prediction**: Sewing pattern prediction is a sufficiently structured task that does not require very large-scale VLMs.
4. **Efficiency and quality are not mutually exclusive**: SwiftTailor demonstrates that significant speed improvements need not come at the cost of generation quality in 3D garment generation.

## Highlights & Insights

- **Representation innovation**: The Garment Geometry Image is an inspiring representation design. Encoding all garment panels into a unified 2D image space is a principle that can be generalized to the generation of other multi-component 3D objects.
- **Amortized optimization**: Shifting physics simulation costs from inference to training is a general acceleration strategy, with analogous ideas appearing in neural physics, neural rendering, and related fields.
- **Modular design**: The decoupled two-stage design allows PatternMaker and GarmentSewer to be optimized and replaced independently, offering high flexibility.
- **Deployment practicality**: The over 10× speedup makes the method viable for real-world applications such as real-time 3D virtual try-on and game character dressing.
- **Interpretability**: Retaining sewing patterns as an intermediate representation allows users to inspect and edit pattern parameters, providing a well-defined human–computer interaction interface.

## Limitations & Future Work

1. **Dataset dependency**: Evaluation is limited to the Multimodal GarmentCodeData dataset, whose diversity may be insufficient to cover all real-world garment types (e.g., highly complex gowns, ethnic attire).
2. **GGI resolution constraint**: The resolution of the geometry image imposes an upper bound on mesh detail; fine structures such as pleats and embroidery may not be adequately captured.
3. **Topological constraints**: GGI assumes that garment panels can be unfolded into a 2D UV space, which may be problematic for garments with complex topology (e.g., perforations, multi-layer overlapping).
4. **Physical plausibility**: Whether the learned geometry fully complies with physical laws (e.g., gravity-induced draping, fabric thickness) remains to be verified.
5. **Generalization**: The ability to generalize to novel garment types outside the training distribution warrants further investigation.

## Related Work & Insights

- **GarmentCode**: A procedural garment modeling framework that generates 3D meshes from sewing patterns; the primary alternative replaced by SwiftTailor.
- **SewFormer / DressCode**: Transformer-based sewing pattern prediction methods using serialized representations; inference is relatively slow.
- **Geometry Images (Gu et al., 2002)**: A classical method for encoding 3D meshes as 2D images; SwiftTailor extends this idea to the multi-panel garment setting.
- **Neural Garment Rendering**: NeRF/Gaussian-based methods for garment rendering that do not directly produce manipulable 3D meshes.
- **DPT (Dense Prediction Transformer)**: A Transformer architecture for dense prediction tasks, used as the backbone for GGI generation in GarmentSewer.

**Insights**: The GGI representation can be extended to the generation of other multi-part 3D objects (e.g., furniture, mechanical components). The "amortized simulation" paradigm offers a transferable reference for all 3D content generation pipelines that depend on physics engines.

## Rating

- **Novelty**: 8/10 — Both the GGI representation and the two-stage amortized framework are innovative, cleverly adapting the classical geometry image idea to garment generation.
- **Experimental Thoroughness**: 7/10 — Achieves SOTA on a standard dataset with ablation studies, but lacks cross-dataset generalization and real-world deployment validation.
- **Writing Quality**: 8/10 — The framework is described clearly, the two-stage design logic flows naturally, and the motivation is well articulated.
- **Value**: 8/10 — The 10× speedup has clear application value, and the GGI representation offers meaningful contributions to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] NI-Tex: Non-isometric Image-based Garment Texture Generation](ni-tex_non-isometric_image-based_garment_texture_generation.md)
- [\[CVPR 2026\] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](sgi_structured_2d_gaussians_for_efficient_and_compact_large_image_representation.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](text-image_conditioned_3d_generation.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)

</div>

<!-- RELATED:END -->

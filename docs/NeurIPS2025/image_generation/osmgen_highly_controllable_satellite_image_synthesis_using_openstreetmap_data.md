---
title: >-
  [Paper Note] OSMGen: Highly Controllable Satellite Image Synthesis using OpenStreetMap Data
description: >-
  [NeurIPS 2025 (Workshop: UrbanAI 2025)][Image Generation][Satellite image synthesis] OSMGen synthesizes high-fidelity satellite images directly from OSM JSON data (vector geometry, semantic tags, location…
tags:
  - "NeurIPS 2025 (Workshop: UrbanAI 2025)"
  - "Image Generation"
  - "Satellite image synthesis"
  - "OpenStreetMap"
  - "ControlNet"
  - "DDIM inversion"
  - "change detection"
date: 2026-05-08
content_hash: c8fb40746d37da56
---

# OSMGen: Highly Controllable Satellite Image Synthesis using OpenStreetMap Data

**Conference**: NeurIPS 2025 (Workshop: UrbanAI 2025)  
**arXiv**: [2511.00345](https://arxiv.org/abs/2511.00345)  
**Code**: Available ([https://github.com/amir-zsh/OSMGen](https://github.com/amir-zsh/OSMGen))  
**Area**: Image Generation / Remote Sensing  
**Keywords**: Satellite image synthesis, OpenStreetMap, ControlNet, DDIM inversion, change detection

## TL;DR

OSMGen synthesizes high-fidelity satellite images directly from OSM JSON data (vector geometry, semantic tags, location, and temporal information), and generates temporally consistent before-after image pairs via DDIM inversion, enabling urban change simulation and data augmentation.

## Background & Motivation

Accurate and timely geospatial data is critical for urban planning, infrastructure monitoring, and environmental management. However, automated urban monitoring faces the challenges of **scarce annotated data** and **class imbalance**.

**Limitations of Prior Work**:
- Most methods use rendered **raster map tiles** as conditions, discarding the rich structural information in OSM data (precise vector geometry, semantic tags, etc.)
- Unable to generate temporally consistent before-after image pairs
- Spatiotemporal information (location, season) is largely unexploited

**Motivation for OSMGen**:
1. Exploit the full richness of OSM JSON (beyond image tiles) for fine-grained controllable synthesis
2. Generate consistent before-after image pairs for change detection training data via DDIM inversion
3. Preview urban planning scenarios by editing map data

## Method

### Overall Architecture

OSMGen is an end-to-end conditional generation framework with three core components:

1. **Multi-modal condition extraction**: Geometric masks, spatial encodings, temporal encodings, and text descriptions are extracted from OSM JSON data
2. **ControlNet-augmented diffusion model**: A ControlNet branch is trained on top of a frozen Stable Diffusion U-Net
3. **DDIM inversion editing**: Enables consistent scene manipulation (adding/removing/modifying elements)

### Key Designs

#### 1. Data Collection and Preprocessing

- Approximately **20,000 geographic points** are sampled from the FMoW (Functional Map of the World) benchmark, covering urban, suburban, and rural areas
- For each point, 256×256 satellite image tiles and corresponding OSM JSON are retrieved
- Two zoom levels are supported: z=18 (fine-grained structure) and z=15 (broader context)

#### 2. Multi-modal Condition Extraction

Four types of conditions are extracted from OSM JSON:

| Condition Type | Source | Role |
|---|---|---|
| **Generic masks** | Vector geometry | Coarse-category segmentation: roads, water bodies, vegetation, buildings, etc. |
| **Specific masks** | POI subtypes | Fine-grained types: lakes, rivers, oil tanks, solar farms, etc. |
| **Spatial encoding** | SatCLIP | Continuous embedding of geographic coordinates |
| **Temporal encoding** | Date2Vec | Continuous embedding capturing date (seasonal variation) |
| **Text description** | CLIP encoding | High-level semantic guidance from salient tile categories |

#### 3. Generation Framework

- **Frozen Stable Diffusion U-Net** + **trainable ControlNet branch**
- Generic and specific masks are fused via convolutional layers and fed into ControlNet to enforce geometric fidelity
- Spatial and temporal embeddings are **added to the diffusion timestep embedding** via linear projection
- Text embeddings are injected via **cross-attention**

The training objective is the standard diffusion loss:

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{x_0, t, \epsilon} \left\| \epsilon - \epsilon_\theta(x_t, t | M, \mathbf{e}_{\text{loc}}, \mathbf{e}_{\text{time}}, \mathbf{e}_{\text{text}}) \right\|_2^2$$

#### 4. Controlled Change Generation (DDIM Inversion)

The pipeline for generating before-after image pairs:
1. Given an original image and reference condition $c_{\text{ref}}$, invert to depth $t^*$ via the DDIM forward process to obtain latent code $x_{t^*}$
2. Modify the condition to $c_{\text{new}}$ (e.g., edit masks to add/remove buildings)
3. Run DDIM denoising starting from $x_{t^*}$ conditioned on $c_{\text{new}}$
4. $t^*$ controls editing intensity: smaller values preserve more of the original image (weak edit); larger values permit stronger modifications

Rationale for choosing DDIM inversion:
- Cross-attention editing methods are inapplicable (cannot handle non-text conditions)
- Simple to implement and agnostic to model architecture
- Strong spatial conditions (masks) allow lower CFG scale, mitigating the instability of DDIM inversion at high CFG

### Loss & Training

- **Loss**: Standard diffusion denoising loss (MSE)
- **Training**: 500 epochs, batch size 2048
- **Trainable parameters**: ControlNet branch, mask fusion layers, linear projections for spatial/temporal conditions
- **Frozen parameters**: Stable Diffusion U-Net backbone
- **Hardware**: Single NVIDIA A100 GPU

## Key Experimental Results

### Main Results: Qualitative Generation Evaluation

Evaluated on approximately 2,000 FMoW test locations, generating 256×256 pixel tiles per location:

| Evaluation Dimension | Generic Mask Contribution | Specific Mask Contribution |
|---|---|---|
| Large-scale structure | Accurate reconstruction of road networks and building outlines | — |
| Fine-grained POI | — | Correct shape and context rendering of rare categories (stadiums, oil tanks, etc.) |
| Edit consistency | Original image features preserved outside edited regions | Precise rendering of local additions/removals/modifications |

### Editing Operation Examples

| Edit Operation | Effect Description |
|---|---|
| Add stadium | Stadium generated in specified area; surrounding context unchanged |
| Add building | New building added; existing roads and vegetation unaffected |
| Remove partial buildings | Buildings disappear; vacant area naturally filled |
| Remove oil tanks | Tank area replaced by appropriate background |
| Lake → grassland | Water body converted to green vegetation with natural boundary transition |
| Farmland → solar farm | Crop texture replaced by solar panel arrangement |

### Key Findings

1. **OSM JSON vs. raster tiles**: Directly using JSON data provides richer conditioning information through vector geometry and semantic tags than rendered raster images
2. **Effectiveness of DDIM inversion**: Strong spatial conditions allow reduced CFG scale, resolving the instability of conventional DDIM inversion at high CFG
3. **Role of spatiotemporal conditions**: Location encoding helps the model understand regional characteristics (tropical vs. temperate); temporal encoding captures seasonal variation (greenness, snow cover, etc.)
4. **Closed-loop potential**: Generated (JSON, image) pairs can be used to train models that automatically detect changes in satellite images and update OSM JSON accordingly

## Highlights & Insights

- **Leveraging raw OSM JSON**: Retaining precise geometry and rich semantics over raster tiles is an important direction in this field
- **Temporally consistent editing**: The combination of DDIM inversion and condition modification is concise and effective, avoiding complex image editing techniques
- **Dual-mask design**: Generic masks capture high-level concepts (roads/water/buildings); specific masks distinguish fine-grained POI subtypes
- **Clear application value**: Data augmentation (addressing annotation scarcity) and urban planning visualization (previewing map edits)

## Limitations & Future Work

1. **Lack of quantitative evaluation metrics**: The paper relies primarily on qualitative visualization, without standard generation quality metrics such as FID or IS
2. **Resolution constraint**: 256×256 resolution may be insufficient for fine-grained urban planning applications
3. **Use of FMoW data only**: Geographic coverage can be expanded
4. **Limited diversity in change generation**: DDIM inversion is deterministic, producing only a single result per edit
5. **Workshop paper**: Experimental scale and depth warrant extension, particularly for downstream task validation

## Related Work & Insights

- **ControlNet + diffusion models**: Validates the effectiveness of ControlNet in the remote sensing domain
- **SatCLIP / Date2Vec**: Off-the-shelf spatiotemporal encoding modules that can be directly reused
- **DiffusionSat**: Pioneering work on foundation models for satellite image generation
- **ChangeDiff**: Related work on change detection data generation, driven by text prompts

## Rating

- Novelty: ⭐⭐⭐⭐ — First work to directly exploit the full information of OSM JSON for satellite image synthesis
- Theoretical Contribution: ⭐⭐⭐ — Primarily an engineering combination innovation
- Experimental Thoroughness: ⭐⭐⭐ — Qualitative results are convincing, but quantitative metrics are absent
- Practical Value: ⭐⭐⭐⭐⭐ — Strong real-world applicability for data augmentation and urban planning visualization
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)
- [\[CVPR 2026\] DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data](../../CVPR2026/image_generation/dynavid_learning_to_generate_highly_dynamic_videos_using_synthetic_motion_data.md)
- [\[NeurIPS 2025\] Boosting Generative Image Modeling via Joint Image-Feature Synthesis](boosting_generative_image_modeling_via_joint_imagefeature_sy.md)
- [\[CVPR 2026\] AHS: Adaptive Head Synthesis via Synthetic Data Augmentations](../../CVPR2026/image_generation/ahs_adaptive_head_synthesis.md)
- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](../../ICCV2025/image_generation/polaranything_diffusion-based_polarimetric_image_synthesis.md)

</div>

<!-- RELATED:END -->

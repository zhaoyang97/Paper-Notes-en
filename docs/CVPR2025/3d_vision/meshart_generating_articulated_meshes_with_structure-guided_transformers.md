---
title: >-
  [Paper Note] MeshArt: Generating Articulated Meshes with Structure-Guided Transformers
description: >-
  [CVPR 2025][3D Vision][Articulated Object Generation] MeshArt proposes a hierarchical Transformer framework that decomposes articulated object generation into two stages: high-level joint structures and low-level part meshes. It autoregressively generates compact and sharp triangle mesh articulated objects, improving structural coverage by 57.1% and mesh FID by over 209 points.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Articulated Object Generation"
  - "Triangle Mesh Generation"
  - "Hierarchical Transformer"
  - "Structure-Guided"
  - "Part Joint"
date: 2026-05-08
content_hash: f9af5652a4d7b6f0
---

# MeshArt: Generating Articulated Meshes with Structure-Guided Transformers

**Conference**: CVPR 2025  
**arXiv**: [2412.11596](https://arxiv.org/abs/2412.11596)  
**Code**: [https://daoyig.github.io/Mesh_Art/](https://daoyig.github.io/Mesh_Art/)  
**Area**: 3D Vision  
**Keywords**: Articulated Object Generation, Triangle Mesh Generation, Hierarchical Transformer, Structure-Guided, Part Joint

## TL;DR
MeshArt proposes a hierarchical Transformer framework that decomposes articulated object generation into two stages: high-level joint structures and low-level part meshes. It autoregressively generates compact and sharp triangle mesh articulated objects, improving structural coverage by 57.1% and mesh FID by over 209 points.

## Background & Motivation

1. **Background**: In the 3D mesh generation field, existing methods such as MeshGPT and PolyGen generate static triangle meshes, while recent works like MeshAnythingV2 and MeshXL have further improved the generation quality. However, these methods only generate static objects. Articulated objects (e.g., cabinets with openable doors, rotating office chair wheels) are ubiquitous in real-world environments.

2. **Limitations of Prior Work**: (1) Generating articulated objects requires simultaneously modeling functional part movements and sharp geometry, which existing methods struggle to balance; (2) Methods like NAP use implicit field decoding to generate geometry, leading to over-smoothed results; (3) CAGE relies on part retrieval, which causes geometric inconsistency; (4) Existing articulated object datasets (such as PartNet-Mobility) are extremely small in scale.

3. **Key Challenge**: The main difficulty in articulated object generation lies in decoupling and handling the relationship between the structure (which parts can move and how they move) and the geometry (detailed triangle mesh of each part), while ensuring geometric coherence between parts.

4. **Goal**: To generate articulated 3D objects with sharp details and correct joint attributes in the form of compact triangle meshes.

5. **Key Insight**: To decompose the articulated mesh into high-level structure (bounding boxes + joint attributes) and low-level geometry (part triangle meshes). Both are unified into triangle sequence representations to achieve consistent hierarchical generation.

6. **Core Idea**: Representing bounding boxes as triangle meshes allows both structure and geometry to be unified into a triangle sequence prediction task, utilizing a two-level Transformer to hierarchically generate articulated objects.

## Method

### Overall Architecture
MeshArt consists of two stages. In the first stage (structure generation), a structure VQ-VAE encodes the bounding boxes and joint information into a quantized token sequence, which a structure Transformer autoregressively predicts to generate the object structure. In the second stage (part mesh generation), for each part, a geometry VQ-VAE encodes its triangle mesh into a token sequence. A geometry Transformer then autoregressively predicts the triangle sequence of each part, conditioned on the structure and junction faces. Finally, all part meshes are combined.

### Key Designs

1. **Structure Encoding with Unified Triangle Sequence Representation**:

    - **Function**: Encodes the object structure (bounding boxes, joint types/positions/directions, semantic labels, latent geometry) into an autoregressively predictable sequence.
    - **Mechanism**: Triangulates the AABB bounding box of each part into 12 triangular faces. Bounding boxes of all parts are sorted from lowest to highest and concatenated into a single triangle sequence. A Graph Convolutional encoder encodes triangle coordinates, joint attributes, CLIP semantic vectors, and latent geometry into feature vectors, which are then mapped to a structure codebook via Residual Vector Quantization (RQ, depth $D=6$). The decoder reconstructs the bounding box coordinates, joint types/existence/positions using cross-entropy loss, and regresses joint directions and semantic/geometric features using $L2$ loss.
    - **Design Motivation**: Representing bounding boxes as triangle meshes is a key innovation. It unifies the representation of structure and geometry, making the two-stage generation framework more consistent. Compared to directly using vertex coordinates or min/max corners, the triangle representation improves the COV metric from 35.0/36.3 to 39.1.

2. **Junction Face Mechanism**:

    - **Function**: Ensures smooth geometric transitions between parts during part-by-part generation.
    - **Mechanism**: A prediction head is added to the geometry VQ-VAE to predict the probability $p_k^i$ of each triangular face being a junction face (a face connecting to other parts). Ground truth labels are generated by calculating the distance from each triangular face to adjacent parts. During generation, junction face tokens of already generated parts are cached and injected at the beginning of the sequence as conditioning for generating the next part.
    - **Design Motivation**: Part boundaries are prone to inconsistencies during part-by-part generation. Junction faces provide crucial clues for local geometric connections—for example, the junction face of a chair wheel strongly hints at the shape and connection point of the base.

3. **Structure-Guided Part Mesh Generation**:

    - **Function**: Generates the triangle mesh of each part conditioned on the global object structure and local junction faces.
    - **Mechanism**: The input sequence of the geometry Transformer consists of three components: (1) tokens of the current part in the global structure, which interact with the global structure via cross-attention and are placed at the front of the sequence; (2) junction face tokens from adjacent, already generated parts; (3) triangle tokens of the current part. Loss is only computed on the triangle tokens. Crucially, a **flexible position encoding** is adopted. Since the condition sequence length is variable (due to the varying number of junction faces), position encodings are allocated starting fixedly from the triangle token sequence, while the condition sequence uses constant position encoding vectors.
    - **Design Motivation**: Variations in the condition sequence length lead to inconsistent starting position encodings for triangle tokens, which hinders long-range dependency learning. The flexible position encoding resolves this issue.

### Loss & Training
- The VQ-VAE uses cross-entropy loss to predict quantized triangle coordinates (in a $128^3$ discretized space) with an RQ depth of $D=6$.
- The Transformer uses cross-entropy loss for next-token prediction.
- Training includes data augmentation: random offsets, random scaling, and decimation. Parts with under 700 faces are used for training.
- Pre-training is performed across categories first, followed by single-category fine-tuning.
- Joint information for the PartNet dataset was additionally annotated (requiring over 150 hours of workload), increasing the number of articulated objects by more than six-fold.

## Key Experimental Results

### Main Results

Structure generation quality (AID):

| Category | NAP COV↑ | NAP MMD↓ | CAGE COV↑ | CAGE MMD↓ | MeshArt COV↑ | MeshArt MMD↓ |
|------|---------|---------|----------|----------|-------------|-------------|
| Chair | 28.3 | 3.7 | 32.9 | 3.9 | **43.3** | **3.6** |
| Table | 21.1 | 3.0 | 25.9 | 3.9 | **40.2** | **2.3** |
| Storage | 30.6 | 2.6 | 33.4 | 4.7 | **39.1** | **2.1** |

Mesh generation quality (FID/KID):

| Category | NAP FID↓ | MeshArt FID↓ | NAP KID↓ | MeshArt KID↓ |
|------|---------|-------------|---------|-------------|
| Chair | 267.7 | **40.8** | 0.263 | **0.008** |
| Table | 252.6 | **14.3** | 0.238 | **0.002** |
| Storage | 170.6 | **8.1** | 0.167 | **0.002** |

### Ablation Study

| Configuration | AID COV↑ | AID MMD↓ |
|------|---------|---------|
| Min/Max Bounds | 36.3 | 4.6 |
| Bbox Corners | 35.0 | 4.4 |
| **MeshArt (Triangle)** | **39.1** | **2.1** |

| Configuration Change | Impact |
|---------|------|
| Without junction face conditioning | Parts connect poorly, FID rises significantly |
| Without global structure conditioning | Poor shape consistency |
| Without flexible position encoding | Model can only generate simple shapes |

### Key Findings
- **Triangle representation is far superior to other bounding box parameterizations**: Using a triangle mesh to represent bounding boxes yields 4.1 points higher in COV compared to min/max corners.
- **Huge improvements in FID/KID**: Compared to NAP, the average FID is improved by over 209 points, because NAP uses Marching Cubes which leads to over-tessellation and over-smoothing.
- **Significant dataset expansion efficacy**: The newly annotated PartNet is over 6x larger than PartNet-Mobility, directly supporting the training of data-driven methods.
- **Importance of flexible position encoding**: Without it, the model degenerates to generating only simple shapes.
- **Support for conditional generation**: Enables generating articulated objects dynamically from point clouds or sketch conditions.

## Highlights & Insights
- **Ingenuity of the Unified Triangle Representation**: Triangulating bounding boxes allows both structure and geometry to share the same sequence representation, greatly simplifying the design of the two-stage framework. This "unified representation" concept can be transferred to other hierarchical generation tasks.
- **Elegance of the Junction Face Mechanism**: Without requiring auxiliary alignment networks or post-processing, smooth part-to-part transitions are naturally achieved simply by injecting junction face tokens as conditions. This is much more end-to-end than retrieval-and-assembly approaches.
- **Dataset Contribution**: Releasing joint annotations for PartNet, which took over 150 hours of manual labeling, scales up the available articulated objects by more than six-fold, establishing a much-needed dataset foundation for this field.

## Limitations & Future Work
- Part face counts are limited to $<700$ (due to the Transformer context window bottleneck), making it unable to generate extremely high-fidelity detailed meshes.
- It only covers 3 categories (chairs, tables, storage furniture) and does not extend to more complex articulated objects (such as robots or vehicles).
- Generation speed is bottlenecked by the part-by-part autoregression, resulting in lower efficiency when the number of parts is large.
- Joint types only support fixed, revolute, and prismatic, failing to cover more complex motion patterns.
- Part ordering relies on heuristic rules (from lowest to highest), which may impact the generation quality of certain structures.

## Related Work & Insights
- **vs NAP**: NAP generates geometry with implicit fields and Marching Cubes, which leads to over-smoothing and over-tessellation. MeshArt directly generates triangle meshes, obtaining compact and sharp results (with over 200 points gap in FID).
- **vs CAGE**: CAGE relies on part retrieval, which causes geometric inconsistencies. MeshArt generates both structure and geometry end-to-end, avoiding mismatching issues introduced by retrieval.
- **vs MeshGPT**: MeshGPT only generates static meshes, while MeshArt extends generation to articulated objects, introducing a hierarchical structure-geometry framework and the junction face mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first hierarchical method to generate articulated objects directly in triangle mesh format, with a highly ingenious unified representation design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison, ablation studies, novelty analysis, and conditional generation validation; includes a significant dataset contribution.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, high-quality figures, and well-organized text.
- Value: ⭐⭐⭐⭐⭐ Dual contribution in both the methodology and dataset, presenting a pioneering advancement for articulated object generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Structure from Collision](structure_from_collision.md)
- [\[CVPR 2026\] ArtLLM: Generating Articulated Assets via 3D LLM](../../CVPR2026/3d_vision/artllm_generating_articulated_assets_via_3d_llm.md)
- [\[CVPR 2025\] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes](volumetric_surfaces_representing_fuzzy_geometries_with_layered_meshes.md)
- [\[CVPR 2026\] Sculpt4D: Generating 4D Shapes via Sparse-Attention Diffusion Transformers](../../CVPR2026/3d_vision/sculpt4d_generating_4d_shapes_via_sparse-attention_diffusion_transformers.md)
- [\[CVPR 2026\] Image-Guided Geometric Stylization of 3D Meshes](../../CVPR2026/3d_vision/image-guided_geometric_stylization_of_3d_meshes.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Generating 3D House Wireframes with Semantics
description: >-
  [ECCV 2024][Image Generation][3D Wireframe Generation] A 3D house wireframe generation method based on autoregressive models is proposed. It employs a unified wire representation instead of traditional separate vertex-edge modeling, generating semantically rich wireframe structures via semantically aware BFS sequence ordering and a two-stage coarse-to-fine Transformer decoder, which can be automatically segmented into semantic components like walls, roofs, and rooms.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "3D Wireframe Generation"
  - "Autoregressive Models"
  - "Semantic Grouping"
  - "Residual Quantization"
  - "Transformer"
date: 2026-05-08
content_hash: 3f09f477358aa398
---

# Generating 3D House Wireframes with Semantics

**Conference**: ECCV 2024  
**arXiv**: [2407.12267](https://arxiv.org/abs/2407.12267)  
**Code**: [Available](https://vcc.tech/research/2024/3DWire)  
**Area**: Image Generation / 3D Generation  
**Keywords**: 3D Wireframe Generation, Autoregressive Models, Semantic Grouping, Residual Quantization, Transformer

## TL;DR

A 3D house wireframe generation method based on autoregressive models is proposed. It employs a unified wire representation instead of traditional separate vertex-edge modeling, generating semantically rich wireframe structures via semantically aware BFS sequence ordering and a two-stage coarse-to-fine Transformer decoder, which can be automatically segmented into semantic components like walls, roofs, and rooms.

## Background & Motivation

### Problem Introduction

3D wireframes are crucial data structures in computer vision and graphics, providing a concise abstraction of object shapes through combinations of vertices and edges. They are particularly suited for representing polyhedra (such as buildings, furniture, and mechanical parts). However, automatically generating 3D wireframes remains a complex and challenging process, requiring precise abstraction of object geometry into line segment sequences.

### Limitations of Prior Work

**Wireframe reconstruction methods** (from images/point clouds): Can only reconstruct from input data and lack the capability to generate novel wireframes.

**Autoregressive methods like PolyGen / SolidGen**: Model different types of primitives (vertices, edges, faces) separately as independent sequences.
   - Errors in vertex generation propagate to subsequent edge and face generation (error accumulation).
   - Coordinate-sorting-based sequences lack high-level semantic associations between primitives.

**MeshGPT**: Learns quantized embeddings of triangular faces to generate meshes, but similarly ignores semantic relationships.

**2D floorplan generation methods** (HouseGAN, HouseDiffusion, etc.): Only generate 2D layouts, requiring additional post-processing to convert to 3D.

### Key Insight

**Unified wire representation + semantic sequence alignment**: Viewing wireframes as graph structures (wires as nodes, connectivity as edges) and grouping wires of the same semantic component (e.g., outer walls, roofs, rooms) together using BFS traversal for generation. This reduces error propagation among primitives and endows the generated results with natural semantic segmentability.

## Method

### Overall Architecture

Two-stage framework:

1. **Stage 1: Learning the Geometric Vocabulary (Quantized Wire Embeddings)**
    - Graph convolutional encoder $E_G$: Captures local topological features of each wire.
    - Attention message exchanger $E_A$: Conducts global information exchange across disconnected subgraphs.
    - Residual Lookup-Free Quantization (Residual LFQ): Quantizes wire features into discrete tokens within a geometric codebook.

2. **Stage 2: Autoregressive Wireframe Generation**
    - Coarse Transformer: Predicts wire-level embedding sequences.
    - Fine Transformer: Refines each wire embedding into vertex-level embeddings.
    - Decodes into 3D coordinates to construct the final wireframe.

### Key Designs

#### 1. Wire Feature Representation and Encoding

The features of each wire $l_i \in \mathbb{R}^{n_{\text{in}}}$ include: endpoint coordinates, wire length, orientation, angles with adjacent wires, and midpoint coordinates. All features are quantized into $[0, 128)$ integers and embedded as 196-dimensional vectors.

**Graph Convolutional Encoder $E_G$**: Uses SAGEConv layers to project wires (as graph nodes) into a 384-dimensional latent space, capturing local geometric features. Since wireframes may contain disconnected subgraphs, graph convolution alone is insufficient for global information exchange.

**Attention Message Exchanger $E_A$**: A Local Multi-Head Attention (LMH Attention) layer that allows global information exchange across disconnected subgraphs, making features both topologically informative and context-rich.

#### 2. Residual Lookup-Free Quantization (Residual LFQ)

After assigning wire features to endpoints, Residual Quantization (RQ) with depth $D=2$ is used to quantize each vertex feature, where each wire is represented by $2 \times D = 4$ embeddings. The codebook size is 8192, and each embedding dimension is $\log_2 8192 = 13$.

Key advantage: LFQ quantizes features by treating them as Cartesian products of single-dimensional variables without traditional codebook lookup steps, significantly reducing computational complexity. It is trained using cross-entropy loss + commitment loss + entropy penalty.

#### 3. Semantically Aware Sequence Construction

**Key Innovation**: Unlike PolyGen/MeshGPT, which sort based on coordinates, this work arranges wire sequences based on **semantic relationships**:
- First, sort wireframe nodes hierarchically by z-y-x.
- View the wireframe as a graph structure (wires = nodes, intersections = edges).
- Perform **BFS traversal** on each disconnected subgraph, ensuring that wires of the same physical component (walls/roofs/rooms) are generated continuously.

#### 4. Coarse-to-Fine Transformer Decoding

- **Coarse Transformer**: Reshapes and merges the vertex code sequence $C_v$ (length $2 \cdot D \cdot N$) into the wire code sequence $C_l$ (length $N$), autoregressively predicting wire embeddings.
- **Fine Transformer**: Predicts vertex embeddings along the depth dimension based on each predicted wire embedding.
- 12+2 layer decoder-only architecture learning three types of encodings: discrete positional encoding, vertex positional encoding, and quantization hierarchical encoding.

### Loss & Training

- Stage 1: Cross-entropy loss on 3D coordinates + commitment loss + entropy penalty (to enhance codebook usage).
- Stage 2: Cross-entropy loss on codebook indices.

## Key Experimental Results

### Main Results

**Dataset**: House layouts extracted from RPLAN + straight skeleton constructed roofs $\to$ ~78,000 3D wireframes (training on a subset of <400 wires).

**Unconditional Wireframe Generation (3D House Dataset)**:

| Model | COV(CD)↑ | COV(EMD)↑ | MMD(CD)↓ | MMD(EMD)↓ | 1-NN(CD) | 1-NN(EMD) | 2L-CVP↑ | 3L-CVP↑ | KLD↓ |
|------|----------|-----------|----------|-----------|----------|-------------|---------|---------|------|
| PolyGen | 38.67 | 47.95 | 8.67 | 6.43 | 74.43 | 67.65 | 81.47 | 75.80 | 12.75 |
| MeshGPT | 54.78 | 54.29 | 9.13 | 6.27 | 64.61 | 61.70 | 80.91 | 70.77 | 8.98 |
| **Ours** | **56.15** | **58.64** | **8.11** | **5.75** | **55.21** | **51.35** | **99.53** | **99.26** | **0.73** |

**User Study (60 participants, 24 comparison groups)**:

| Comparison | PolyGen | MeshGPT | Ours | Ground Truth |
|------|---------|---------|------|-------------|
| Ours Rating | 0.84 | 0.75 | — | -0.13 |
| Selection Rate | 92% | 87% | — | — |

**Generalization Test on ABC Dataset**:

| Model | COV↑ | MMD↓ | 1-NN↓ |
|------|------|------|-------|
| PolyGen | 39.94 | 25.53 | 75.87 |
| MeshGPT | 42.38 | 24.62 | 67.65 |
| **Ours** | **44.10** | **22.12** | **62.96** |

### Ablation Study

**Ablation of Design Choices (3D House Dataset, CD-based)**:

| Configuration | COV↑ | MMD↓ | 1-NN | 2L-CVP↑ | 3L-CVP↑ |
|------|------|------|------|---------|---------|
| w/o Encoder LMH Attention | 49.07 | 10.98 | 68.72 | 64.97 | 59.20 |
| w/o Residual LFQ | 50.02 | 9.24 | 69.87 | 69.17 | 63.47 |
| w/o Coarse-to-Fine | 52.56 | 9.08 | 66.20 | 73.97 | 68.77 |
| w/o Semantic Order | 51.33 | 9.07 | 67.27 | 72.42 | 67.69 |
| **Full Model** | **56.15** | **8.11** | **55.21** | **99.53** | **99.26** |

### Key Findings

1. **Huge discrepancy in structural validity metrics**: The 2L-CVP/3L-CVP of the full model reaches up to 99.53%/99.26%, whereas removing any component leads to a sharp drop down to 60-75%, indicating that each design is crucial for maintaining wireframe connectivity.
2. **LMH Attention has the most significant impact**: Removing it drops COV from 56.15 to 49.07 and introduces intersecting wires, owing to the loss of spatial relation encoding across disconnected subgraphs.
3. **Semantic ordering vs. Coordinate ordering**: Using semantic BFS ordering yields significant improvements over z-y-x coordinate ordering in both COV (+4.82) and structural validity (+27pp).
4. **Significant KLD margin**: The full model achieves KLD=0.73 compared to MeshGPT (8.98) and PolyGen (12.75), indicating that the generated wireframes' connected components distribution is closest to the ground truth.
5. **Novelty analysis**: The generated 4096 wireframes cover similar samples from the training set while demonstrating structural differences as CD increases, proving the model's ability to generate novel wireframes.

## Highlights & Insights

- **Unified wire representation** avoids cascading error propagation from vertices $\to$ edges $\to$ faces, which is the fundamental advantage over PolyGen/MeshGPT.
- **Semantic BFS ordering** is an extremely ingenious design—implicitly encoding semantic information via the topological connectivity of the graph structure without requiring additional semantic labels.
- Generated results are naturally segmentable into components like walls, roofs, and rooms, directly applicable to downstream CAD applications.
- Supports **conditional generation** (text incorporated via cross-attention) and **wireframe completion** (multiple possible completions of partial wireframes).
- Residual LFQ eliminates codebook lookup overhead, making training with large vocabularies feasible.

## Limitations & Future Work

- Limited to polyhedral structures like houses; curved objects are not suitable for wireframe representations.
- The training set is limited to wireframes with <400 wires; more complex buildings require an extended context window.
- The effectiveness of text-conditional generation has not been thoroughly evaluated quantitatively.
- Ground truth construction relies on the straight skeleton algorithm, introducing specific style biases.
- Computational resources for training are non-trivial: 8$\times$ RTX 3090, requiring approx. 2 days for the encoder-decoder and 5 days for the Transformer.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first semantically-aware 3D wireframe generation method; the unified wire representation and BFS semantic ordering are highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage across quantitative, qualitative, user studies, ablation, and novelty analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — The methodology is explained clearly, and the diagrams are intuitive.
- **Value**: ⭐⭐⭐⭐ — Opens up a new direction for 3D wireframe generation, leading with overwhelmingly superior structural validity metrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [\[ECCV 2024\] Infinite-ID: Identity-preserved Personalization via ID-semantics Decoupling Paradigm](infinite-id_identity-preserved_personalization_via_id-semantics_decoupling_parad.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ECCV 2024\] Generating Human Interaction Motions in Scenes with Text Control](generating_human_interaction_motions_in_scenes_with_text_control.md)

</div>

<!-- RELATED:END -->

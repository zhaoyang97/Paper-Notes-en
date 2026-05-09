---
title: >-
  [Paper Note] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing
description: >-
  [3D Vision] MeshPad decomposes sketch-driven 3D mesh creation and editing into two sub-tasks—addition and deletion—based on a triangle sequence representation and Transformer autoregressive generation. It further proposes a vertex-aligned speculative decoder achieving a 2.2× speedup, enabling interactive mesh editing within seconds.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: a4958992a134d4a8
---

# MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.01425](https://arxiv.org/abs/2503.01425)
- **Code**: [Project Page](https://derkleineli.github.io/meshpad/)
- **Area**: 3D Vision
- **Keywords**: triangle mesh generation, sketch-conditioned editing, autoregressive Transformer, speculative decoding acceleration, interactive 3D modeling

## TL;DR
MeshPad decomposes sketch-driven 3D mesh creation and editing into two sub-tasks—addition and deletion—based on a triangle sequence representation and Transformer autoregressive generation. It further proposes a vertex-aligned speculative decoder achieving a 2.2× speedup, enabling interactive mesh editing within seconds.

## Background & Motivation

Triangle meshes are the dominant representation in 3D production pipelines, yet existing mesh generation methods suffer from several critical shortcomings:

**Non-editability**: Methods such as MeshGPT and MeshAnything can only generate complete shapes and do not support local editing.

**Iterative creation demands**: Artistic content creation is an iterative process requiring rapid interactive editing capabilities.

**Coarse editing granularity**: 3D editing methods based on SDS or implicit representations (SDF/NeRF) modify unintended regions and cannot produce artist-reminiscent meshes.

**Slow inference**: Autoregressive mesh generation predicts tokens one at a time, making interactive speeds difficult to achieve.

The core insight of MeshPad is that decomposing editing into addition and deletion (a) enables automatic training data generation from existing 3D data without real editing sequences, and (b) restricts modifications to target regions while preserving unedited parts.

## Method

### Overall Architecture

MeshPad takes a sketch image $\mathcal{I}$ as input—comprising a black keep region $\mathcal{I}_k$ and a red edit region $\mathcal{I}_r$—and outputs the corresponding 3D triangle mesh. The framework consists of three core components:

1. **Sketch-conditioned mesh deletion network**: A classification network that labels each vertex for deletion.
2. **Sketch-conditioned mesh addition network**: An autoregressive generation network that generates new mesh triangle sequences.
3. **Vertex-aligned speculative decoder**: Accelerates autoregressive generation.

### Mesh Representation and Edit Decomposition

A mesh $\mathcal{M}$ consists of a set of triangles, where each triangle $\mathcal{F} = \{v_1, v_2, v_3\}$. An editing operation partitions the mesh into two disjoint parts:
- $\mathcal{M}_k$: the preserved part (corresponding to black sketch strokes)
- $\mathcal{M}_r$: the edited part (corresponding to red sketch strokes)

**Deletion**: A binary label is predicted for each vertex in the mesh sequence; triangles whose vertices fall in the red region are removed:

$$\mathcal{M}_r' = \{\mathcal{F} \in \mathcal{M} | \exists v \in \mathcal{F}: v \in \mathcal{V}_r'\}; \quad \mathcal{M}_k' = \mathcal{M} \setminus \mathcal{M}_r'$$

**Addition**: Conditioned on the existing mesh $\mathcal{M}_k$ and the new sketch, a new triangle sequence $\mathcal{M}_r'$ is autoregressively generated and merged with $\mathcal{M}_k$.

### Network Architecture

- **Backbone**: Based on OPT (Open Pre-trained Transformer), initialized with pre-trained weights from MeshAnythingV2.
- **Image encoder**: A frozen RADIO 2.5-h model encodes sketch inputs.
- **Tokenization**: Adopts the MeshAnythingV2 tokenizer, representing meshes as ordered sequences of control tokens (`<split>`, `<start>`, `<end>`) and vertex coordinate tokens.
- **Addition network**: Causal attention + autoregressive generation, producing only the token sequence for $\mathcal{M}_r'$.
- **Deletion network**: Bidirectional attention + classification head, outputting a delete/keep label per vertex.

The core formula for autoregressive generation:

$$P(S_r'^{(i+1)} | S_k, \mathcal{I}, S_r'^{(1...i)}) = \text{OPT}(S_k, \mathcal{I}, S_r'^{(1...i)})$$

### Vertex-Aligned Speculative Decoder

The key innovation exploits the structured property of mesh sequences—each vertex is represented by exactly 3 tokens ($V_x, V_y, V_z$). The speculator receives the Transformer hidden state corresponding to $V_x$ and predicts $V_y$ and $V_z$ in a single step:

$$P(V'_{\{y,z\}}) = \text{Speculator}(E_x, V'_x)$$

Key design choices:
- **Joint training**: The speculator is trained jointly with OPT (rather than independently after freezing OPT), ensuring hidden states carry sufficient contextual information.
- **Vertex alignment**: Only $V_y$ and $V_z$ are predicted, reducing the complexity of the prediction head.
- **OPT loss supervises only** $V_x$ and control tokens.

### Training Data Generation

Training pairs are generated automatically from ShapeNet 3D data in a self-supervised manner, without real editing sequences:

1. Two volumes $\mathcal{L}_a$ and $\mathcal{L}_b$ are randomly sampled from a complete mesh $\mathcal{M}_c$.
2. Cutting yields $\mathcal{M}_k$ and $\mathcal{M}_r$.
3. Synthetic sketches are generated via Canny edge detection on normal and depth renders.
4. Random data augmentation bridges the domain gap between synthetic and hand-drawn sketches.

### Loss & Training

- **Addition network**: Cross-entropy loss supervising token sequence prediction, plus an independent cross-entropy loss for the speculator.
- **Deletion network**: Binary cross-entropy loss supervising the delete/keep label for each vertex.

## Key Experimental Results

### Main Results: Sketch-Conditioned Mesh Generation

| Dataset | Method | CD↓ | LPIPS↓ | CLIP↑ | FID↓ |
|---------|--------|-----|--------|-------|------|
| ShapeNet | LAS | 15.02 | 0.2742 | 94.61 | 46.90 |
| ShapeNet | LAS-MA | 22.06 | 0.2963 | 93.63 | 18.52 |
| ShapeNet | SENS | 8.95 | 0.2753 | 93.36 | 81.88 |
| ShapeNet | SENS-MA | 29.43 | 0.3348 | 91.88 | 42.93 |
| ShapeNet | **MeshPad** | **6.20** | **0.1790** | **95.85** | **9.38** |
| IKEA | LAS | 16.27 | 0.2970 | 94.83 | 69.54 |
| IKEA | **MeshPad** | **6.78** | **0.1837** | **96.67** | **29.67** |

MeshPad improves CD by **22%+** over the best baseline SENS, with substantially larger gains in FID.

### Ablation Study: Speculative Decoder Design

| Method | CD↓ | LPIPS↓ | CLIP↑ | FID↓ | T/s↑ |
|--------|-----|--------|-------|------|------|
| w/o speculator | 7.66 | 0.1765 | 95.59 | 32.59 | 60.7 |
| w/o vert-alignment | 9.00 | 0.1992 | 94.43 | 35.65 | 138.9 |
| w/o joint training | 57.13 | 0.5134 | 84.52 | 211.46 | 130.8 |
| **MeshPad (full)** | **6.78** | **0.1837** | **96.67** | **29.67** | **131.1** |

### Key Findings

1. **Joint training is critical**: Freezing OPT and training the speculator independently causes performance collapse (CD rises from 6.78 to 57.13).
2. **Vertex alignment preserves quality**: Compared to a generic MLP speculator, vertex alignment maintains superior generation quality at comparable speed.
3. **2.16× speedup**: Speculative decoding increases throughput from 60.7 T/s to 131.1 T/s.
4. **Dominant user preference**: Over 90% of participants preferred MeshPad results in pairwise comparisons.
5. **Interactive speed**: Editing operations (approximately 1/5 of a complete shape) complete within 1–5 seconds.

## Highlights & Insights

1. **Edit decomposition paradigm**: Decomposing complex mesh editing into addition and deletion both simplifies the problem and enables automatic training data generation.
2. **Partial generation**: The addition network generates only tokens for the edited region, naturally preserving unedited parts—a core advantage over SDS-based methods.
3. **Speculative decoding innovation**: The xyz triplet structure of mesh sequences is exploited for a more efficient speculative decoder compared to general NLP approaches.
4. **Sketch UI**: A browser-based interactive interface built on Gradio and Three.js lowers the barrier to 3D modeling.
5. **Novel shape generation**: Iterative editing enables the creation of shape categories not seen during training.

## Limitations & Future Work

1. **Context length constraint**: Due to Transformer limitations, generated meshes have a limited face count (<768 faces), far below the million-scale requirements of modern CG applications.
2. **Trained only on ShapeNet**: Coverage of 55 categories is limited; generalization to open-world objects remains to be validated.
3. **Non-exact sketch-mesh alignment**: Pixel-level alignment between generated meshes and input sketches is not guaranteed.
4. **Generation from scratch vs. editing**: Generating complex shapes from scratch is less effective than iterative editing.

## Related Work & Insights

- **MeshGPT/MeshAnything series**: Established the foundation for autoregressive triangle mesh generation but do not support editing.
- **SENS/LAS**: Convert sketches to SDF/Occupancy Grids followed by Marching Cubes, producing non-artist-reminiscent meshes.
- **Speculative decoding**: Imported from NLP and cleverly adapted using domain-specific structure (vertex triplets).
- **Insight**: Decomposing complex tasks into smaller, controllable sub-tasks is a general and effective strategy.

## Rating

⭐⭐⭐⭐ (4/5)

The strengths lie in the elegant edit decomposition design, the well-motivated speculative decoder, and comprehensive user studies. Limitations include significant face count constraints and limited dataset coverage. Overall, this work represents an important step toward interactive editing in the direct mesh generation paradigm.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text](rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing
description: >-
  [CVPR 2025][3D Vision][Mesh Generation] Proposes TreeMeshGPT, which serializes meshes via dynamic tree structure traversal based on triangle adjacency relations. This achieves an efficient representation of only 2 tokens per face (~22% compression rate), scales the artistic mesh generation capability to 5,500 faces, and significantly reduces normal flipping issues.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Mesh Generation"
  - "Autoregressive Transformer"
  - "Tree-structured Serialization"
  - "Point Cloud Conditioning"
  - "Artistic Meshes"
date: 2026-05-08
content_hash: 1b987dd653463aa5
---

# TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing

**Conference**: CVPR 2025  
**arXiv**: [2503.11629](https://arxiv.org/abs/2503.11629)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Mesh Generation, Autoregressive Transformer, Tree-structured Serialization, Point Cloud Conditioning, Artistic Meshes

## TL;DR

Proposes TreeMeshGPT, which serializes meshes via dynamic tree structure traversal based on triangle adjacency relations. This achieves an efficient representation of only 2 tokens per face (~22% compression rate), scales the artistic mesh generation capability to 5,500 faces, and significantly reduces normal flipping issues.

## Background & Motivation

Post 3D generation, Marching Cubes is typically utilized to extract meshes. However, the resulting meshes are overly dense with chaotic wireframes, making them unsuitable for real-time rendering applications like games and VR. Compact, structurally orderly, and semantically aligned meshes are manually created by artists, but this process is extremely time-consuming.

Autoregressive mesh generation methods are emerging. MeshGPT pioneered the face-ordering + VQ-VAE + Transformer paradigm, but MeshAnything requires 9 tokens per face and is limited to 800 faces. MeshAnythingV2 and EdgeRunner utilize triangle adjacency to shorten sequences, extending representation to 1,600 and 4,000 faces, respectively. However, practical applications require higher face counts to accurately represent complex surfaces, and existing methods suffer from artifacts such as gaps, missing faces, and normal flipping.

**Core Problem**: How to further compress token sequences, lower training difficulty, and improve generation quality while resolving normal consistency issues? A paradigm shift from "next token prediction" to "next token retrieval from a dynamic tree."

## Method

### Overall Architecture

TreeMeshGPT employs a 24-layer Transformer decoder (1,024-dimensional hidden layer, 16 attention heads). The input consists of a point cloud condition (8,192 points, encoded into 2,048 latent codes $\mathbf{Z}$ via cross-attention) concatenated with the autoregressive sequence. In each step of the sequence, the input is a directed mesh edge $(v_1^n, v_2^n)$, and the output is the opposite vertex $v_3^n$ or a [STOP] token. A half-edge data structure combined with Depth-First Search (DFS) traversal is used to construct input-output sequence pairs. Using 7-bit quantization, it supports up to 5,500 faces.

### Key Designs

#### Key Design 1: Autoregressive Tree Sequencing

**Function**: Serializes triangular meshes into Transformer-compatible sequences with an efficiency of only 2 tokens per face.

**Mechanism**: Converts a triangular mesh into an equivalent tree structure, where each node represents a directed edge $(v_i, v_j)$. A dynamic stack is used to manage DFS traversal: (1) Initialize the stack with the edge at the lowest spatial position and its twin edge; (2) In each step, pop an edge from the top of the stack as input $I_n = (v_1^n, v_2^n)$; (3) If an opposite vertex $v_3^n$ exists, output this vertex and push the two new edges $(v_3^n, v_2^n)$ and $(v_1^n, v_3^n)$ onto the stack in counter-clockwise order; (4) If it is a boundary or has already been visited, output [STOP].

**Design Motivation**: Traditional next-token prediction treats each step's output as the next step's input, making the sequential order weakly correlated with mesh topology. Tree traversal ensures that each generation step is a local extension—expanding directly from the most recently generated triangular face, which significantly reduces training difficulty. The counter-clockwise constraint of the half-edge data structure naturally guarantees normal consistency, fundamentally preventing normal flipping issues.

#### Key Design 2: Hierarchical MLP Head Vertex Prediction

**Function**: Predicts complete 3D coordinates within a single sequence step.

**Mechanism**: Unlike prior methods that treat each coordinate as an independent token, this work predicts quantized $x$, $y$, and $z$ coordinates in a single step using a hierarchical MLP head. It first predicts $x$, whose embedding participates in predicting $y$, and then the embedding of $y$ is used for $z$ prediction, maintaining sequential dependencies between coordinates. The loss is the sum of cross-entropy losses across the three axes: $\mathcal{L} = \mathcal{L}_{CE}(\mathbf{O}_x, \hat{\mathbf{O}}_x) + \mathcal{L}_{CE}(\mathbf{O}_y, \hat{\mathbf{O}}_y) + \mathcal{L}_{CE}(\mathbf{O}_z, \hat{\mathbf{O}}_z)$.

**Design Motivation**: Splitting 3D coordinates into three independent tokens increases the sequence length by threefold. Hierarchical prediction requires only one sequence step while preserving dependencies among coordinates. Ablation studies demonstrate that hierarchical prediction is easier to sample than simultaneous coordinate prediction.

#### Key Design 3: Inference-time Strategy Adjustments

**Function**: Improves generation quality and stability.

**Mechanism**: (1) Duplicated face detection—tracks generated faces and automatically replaces the predicted vertex with [STOP] if it would form a duplicate triangle; (2) Adaptive EOS enhancement—since [EOS] is hard to sample in long sequences (even if within the top-5 logits), an increasing factor is applied to the [EOS] logit, and it is selected only when it becomes the highest logit; (3) Dynamic temperature—the temperature is set to 1.0 when stack length is <10, 0.4 when <30, and 0.2 otherwise.

**Design Motivation**: Accumulated errors in autoregressive generation degrade the quality of long sequences. Duplicated face detection avoids topological errors; adaptive EOS addresses the issue where the model does not know when to stop; dynamic temperature encourages diversity in the initial stage and tightens distribution in the later stage to guarantee quality.

### Loss & Training

Sum of independent cross-entropy losses for the three coordinate axes. [STOP] and [EOS] tokens are treated as extra classes for the z-axis. Trained using teacher-forcing.

## Key Experimental Results

### Quantitative Results (Objaverse Validation Set, 200 Samples)

| Model | CD↓ | NC↑ | |NC|↑ |
|------|------|------|-------|
| MeshAnything | 0.0115 | 0.223 | 0.853 |
| MeshAnythingV2 | 0.0102 | 0.167 | 0.843 |
| **TreeMeshGPT** | **Better** | **Better** | **Better** |

### Face Capacity Comparison

| Method | Max Faces | Tokens per Face |
|------|---------|-------------|
| MeshAnything | 800 | 9 |
| MeshAnythingV2 | 1,600 | ~4.5 |
| EdgeRunner | 4,000 | ~4.5 |
| **TreeMeshGPT** | **5,500** | **~2** |

### Key Findings

1. **Significant Increase in Face Capacity**: Scales from 800 to 5,500 faces (a 6.9x gain over MeshAnything) and from 4,000 to 5,500 faces (a 37.5% gain over EdgeRunner).
2. **Significantly Improved Normal Consistency**: NC metrics are drastically superior to MeshAnything (0.223) and MeshAnythingV2 (0.167), because the half-edge structure naturally enforces a consistent normal direction.
3. **Compression Rate of ~22%**: Requires ~2 tokens per face vs. 9 tokens in naive methods, representing a compression rate approximately twice that of MeshAnythingV2 and EdgeRunner.
4. **Generalization to Real Scans**: Demonstrates good qualitative results on the GSO dataset.
5. **Lower Difficulty through Localized Prediction**: Dynamic stack management enables the Transformer to make only local predictions at each step, facilitating easier training convergence.

## Highlights & Insights

- **Paradigm Breakthrough**: Shifts from "predicting the next token" to "retrieving the next token from a dynamic tree," altering the fundamental paradigm of autoregressive mesh generation.
- **Data Structure First**: Leverages topology/orientation constraints of the half-edge data structure to naturally solve the normal flipping issue, presenting a more elegant solution than post-processing.
- **Locality of DFS**: DFS traversal ensures that each step of generation is adjacent to the most recently generated faces instead of distant faces on the mesh, reducing the training difficulty caused by "long-distance jumps."

## Limitations & Future Work

- **5,500 Faces is Still Limited**: It may not suffice for extremely complex models, but already covers a vast range of practical scenarios.
- **Manifold Assumption**: Training meshes must be manifold and free of flipped normals, which constrains the volume of usable training data.
- **Uniqueness of DFS Order**: A single mesh can have multiple valid DFS traversal orders; starting from the lowest spatial position may not be globally optimal.
- Future work could explore BFS traversal, larger search spaces, and tighter integration with 3D generation pipelines.

## Related Work & Insights

- **MeshGPT/MeshAnything**: Pioneered the face-ordering + autoregressive generation paradigm. TreeMeshGPT doubles the compression rate via tree traversal.
- **EdgeRunner**: Optimizes sequences using triangle adjacency, yet still relies on standard next-token prediction.
- **Insight**: In autoregressive generation, the choice of data structure (such as half-edge and tree traversal) can be more critical than architectural improvements.

## Rating

⭐⭐⭐⭐ — Clever tree serialization design that pushes face capacity from 800 to 5,500 while resolving normal flipping issues. Data-structure-driven innovation is more profound than mere architectural modifications. The ~22% compression rate establishes a new benchmark in this domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Scaling Mesh Generation via Compressive Tokenization](scaling_mesh_generation_via_compressive_tokenization.md)
- [\[CVPR 2025\] StageDesigner: Artistic Stage Generation for Scenography via Theater Scripts](stagedesigner_artistic_stage_generation_for_scenography_via_theater_scripts.md)
- [\[NeurIPS 2025\] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](../../NeurIPS2025/3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [\[ICML 2025\] FreeMesh: Boosting Mesh Generation with Coordinates Merging](../../ICML2025/3d_vision/freemesh_boosting_mesh_generation_with_coordinates_merging.md)
- [\[CVPR 2026\] FACE: A Face-based Autoregressive Representation for High-Fidelity and Efficient Mesh Generation](../../CVPR2026/3d_vision/face_a_face-based_autoregressive_representation_for_high-fidelity_and_efficient_.md)

</div>

<!-- RELATED:END -->

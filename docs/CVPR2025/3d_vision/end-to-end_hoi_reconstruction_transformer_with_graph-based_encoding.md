---
title: >-
  [Paper Note] End-to-End HOI Reconstruction Transformer with Graph-based Encoding
description: >-
  [CVPR 2025][3D Vision][Human-Object Interaction Reconstruction] Proposes the HOI-TG framework, which implicitly learns human-object interaction relationships using the self-attention mechanism of Transformers and embeds graph residual blocks in the encoder to enhance topological structure modeling for the human body and objects, respectively, achieving SOTA 3D HOI reconstruction on the BEHAVE and InterCap datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human-Object Interaction Reconstruction"
  - "Transformer"
  - "Graph Convolution"
  - "Implicit Interaction Modeling"
  - "Mesh Reconstruction"
date: 2026-05-08
content_hash: 0ed2060e742e3169
---

# End-to-End HOI Reconstruction Transformer with Graph-based Encoding

**Conference**: CVPR 2025  
**arXiv**: [2503.06012](https://arxiv.org/abs/2503.06012)  
**Code**: [https://hoi-tg.github.io/](https://hoi-tg.github.io/)  
**Area**: 3D Vision  
**Keywords**: Human-Object Interaction Reconstruction, Transformer, Graph Convolution, Implicit Interaction Modeling, Mesh Reconstruction

## TL;DR
Proposes the HOI-TG framework, which implicitly learns human-object interaction relationships using the self-attention mechanism of Transformers and embeds graph residual blocks in the encoder to enhance topological structure modeling for the human body and objects, respectively, achieving SOTA 3D HOI reconstruction on the BEHAVE and InterCap datasets.

## Background & Motivation

**Background**: Reconstructing 3D meshes of Human-Object Interaction (HOI) from a single image is a critical task for AR/VR and robotic manipulation. Existing methods, such as StackFLOW, CHORE, and CONTHO, typically model contact constraints (e.g., offsets, contact maps) explicitly between humans and objects to guide joint reconstruction.

**Limitations of Prior Work**: Explicit interaction modeling suffers from an inherent conflict between global and local scales. Mesh reconstruction focuses on the holistic relative positioning of the human and the object, whereas contact constraints (offsets, contact maps) focus on localized regions. Simultaneously optimizing both is highly challenging; for instance, StackFLOW must rely on time-consuming post-optimization processes to obtain reasonable results.

**Key Challenge**: There is a trade-off between global structural reconstruction and local contact accuracy, making it difficult for explicit modeling methods to balance both. Additionally, directly transferring human Transformer methods (such as METRO) to HOI tasks faces three challenges: all 3D points sharing the same features leads to insufficient discriminativeness, learning interaction poses from static templates is difficult, and self-attention tends to confuse the independent topological boundaries of humans and objects.

**Goal**: How to implicitly model human-object interactions using Transformers without relying on explicit contact constraints, while simultaneously maintaining the integrity of the respective topological structures of the human body and the object.

**Key Insight**: The authors argue that the self-attention of Transformers is inherently suited for capturing global interactions, while Graph Convolutional Networks (GCNs) excel at modeling local topologies. Combining the two—letting self-attention handle global interactions and graph convolutions process their respective local structures—enables the implicit and natural learning of HOIs.

**Core Idea**: Implicitly model global human-object interactions using Transformer self-attention, and utilize graph residual blocks embedded in the encoder to maintain the local topological structures of the human body and objects, respectively.

## Method

### Overall Architecture
The input is an RGB image containing HOI and the segmentation masks for the human/object, and the output is the human 3D mesh (6,890 vertices) and the 6D pose (rotation + translation) of the object. The overall pipeline consists of three steps: (1) extracting image features through a pre-trained ResNet50 and generating initial human/object meshes; (2) constructing 3D queries (joint queries, human vertex queries, object vertex queries) by projecting the initial vertices to perform grid sampling on the image features and concatenating them with 3D coordinates; (3) feeding all queries into a three-layer HOI reconstruction Transformer encoder for joint reconstruction, and finally recovering the full-resolution human mesh via an upsampling matrix, and solving for the object's pose via rigid body transformation.

### Key Designs

1. **3D Query Feature Construction (Grid Sampling + Initial Mesh Coordinates)**:

    - **Function**: Providing highly discriminative input queries for the Transformer.
    - **Mechanism**: A pre-trained backbone is first used to generate rough SMPLH parameters and the initial object pose, obtaining the initial mesh vertices. Then, each 3D vertex is projected back to 2D image coordinates, and the corresponding features are extracted from the feature map via grid sampling, which are then concatenated with the 3D coordinates of the vertices. The dimension of each final query is $(2048+3)$.
    - **Design Motivation**: Addressing the problem where "all 3D points share the same feature"—grid sampling equips each vertex with unique visual features. Using the initial mesh instead of a static template as the starting point reduces the difficulty of directly learning complex interactions from a fixed template. Ablation studies demonstrate that this approach is significantly superior to using global pooling features + a static template.

2. **Human Graph Residual Block**:

    - **Function**: Enhancing the local topological relationship modeling of human vertices inside the Transformer encoder.
    - **Mechanism**: After the multi-head attention in each Transformer encoder block, an additional graph convolution is performed on the human vertex features: $Q'_{hv} = \sigma(\bar{A} Q^{mid}_{hv} W_G)$, where $\bar{A}$ is the predefined adjacency matrix of the human mesh, preserving the topological structure of the SMPLH model. Residual connections are adopted.
    - **Design Motivation**: The self-attention of the Transformer is global, which can confuse the independent topological boundaries of humans and objects. Graph convolutions use a predefined adjacency matrix to fuse information within local neighborhoods, helping the model distinguish vertices belonging to the human body and maintain human topological integrity.

3. **Object Graph Residual Block**:

    - **Function**: Building object-specific graph structures for different object templates to enhance local object modeling.
    - **Mechanism**: Similar in structure to the Human Graph Residual Block, but the adjacency matrix $\bar{A}$ is dynamically constructed using the KNN graph algorithm ($K=10$) based on different object templates. Different objects have different topologies (e.g., chair vs. umbrella), requiring distinct graph structures.
    - **Design Motivation**: The topologies of different objects vary significantly, and using only self-attention makes it hard to accurately model the local relationships of symmetric objects or complex shapes. The KNN graph adaptively conforms to various object topologies, aiding in the accurate prediction of object poses.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{human} + \mathcal{L}_{object} + \mathcal{L}_{hbox}$:
- $\mathcal{L}_{human}$: includes multi-scale vertex L1 loss (across three scales: 431→1723→6890), joint L1 loss (initial + refined 3D/2D coordinates), edge-length consistency loss, and SMPLH parameter L1 loss.
- $\mathcal{L}_{object}$: object vertex L1 loss + rotation-translation L1 loss.
- $\mathcal{L}_{hbox}$: hand bounding box L1 loss.
Trained end-to-end, the hidden dimensions of the three-layer Transformer encoder are 1024, 512, and 256, decreasing layer by layer.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (CONTHO) | Gain |
|--------|------|--------|-------------------|------|
| BEHAVE | CD_human ↓ | **4.59** | 4.99 | 8.0% |
| BEHAVE | CD_object ↓ | **8.00** | 8.42 | 5.0% |
| BEHAVE | Contact_p ↑ | **0.662** | 0.628 | +3.4% |
| BEHAVE | Contact_r ↑ | **0.554** | 0.496 | +5.8% |
| InterCap | CD_human ↓ | **5.43** | 5.96 | 8.9% |
| InterCap | CD_object ↓ | **8.68** | 9.50 | 8.6% |
| InterCap | Contact_p ↑ | **0.700** | 0.661 | +3.9% |
| InterCap | Contact_r ↑ | **0.473** | 0.432 | +4.1% |

### Ablation Study

| Configuration | CD_human ↓ | CD_object ↓ | Contact_p ↑ | Contact_r ↑ |
|------|-----------|------------|------------|------------|
| Transformer only | 4.73 | 8.55 | 0.606 | 0.559 |
| +Human GRB | 4.61 | 8.11 | 0.651 | 0.539 |
| +Human GRB + Object GRB | **4.59** | **8.00** | **0.662** | **0.554** |
| Static query (Global features + Template) | 4.95 | 8.90 | 0.632 | 0.472 |
| Initial query (Grid sampling) | **4.59** | **8.00** | **0.662** | **0.554** |

### Key Findings
- Graph residual blocks make a significant contribution: adding only the Human GRB reduces CD_object from 8.55 to 8.11, showing that modeling human topology also indirectly assists object reconstruction.
- Great difference between initial meshes and static templates: queries using grid sampling + initial mesh outperform global features + static template substantially across all metrics, with Contact_r improving from 0.472 to 0.554 in particular.
- $K=10$ is the optimal number of KNN neighbors; too few cannot fully model adjacency relationships, while too many introduce redundancy, leading to performance degradation.
- Attention visualization indicates: in simple interaction scenarios, the model only focuses on local body parts; in complex interactions, the model successfully attends to non-local body parts to infer the object's position.

## Highlights & Insights
- **Implicit interaction modeling naturally replacing explicit constraints**: Learning human-object interactions inherently via Transformer self-attention without manually designed contact maps or offset constraints reduces engineering complexity while improving performance. This insight shows that many "explicit constraints" can be implicitly covered by a sufficiently powerful attention mechanism.
- **Paradigm of embedding GCN in Transformer**: Acting as a component of the Transformer block, the graph residual block balances global attention and local topology. This hybrid architecture design can be transferred to any 3D task that requires processing both global relationships and local structures simultaneously.
- **Adaptive graph structures for different objects**: Using KNN to dynamically construct object adjacency matrices allows the framework to generalize to objects with different topologies.

## Limitations & Future Work
- The authors acknowledge poor performance on lying poses and completely symmetric objects.
- The method relies on prior knowledge of 3D object templates and cannot handle unseen objects.
- Inference requires segmentation masks of the human and object as input, necessitating an additional segmentation model in practical applications.
- Only validated on indoor datasets; generalization ability under complex outdoor backgrounds remains unknown.
- Incorporating multi-frame temporal information can be considered to handle more complex dynamic interaction scenarios.

## Related Work & Insights
- **vs CONTHO**: Shares the same initial mesh, but while CONTHO refines with explicit contact map constraints, HOI-TG uses implicit self-attention + graph residual blocks. HOI-TG performs better on all metrics, proving that the implicit approach is more appropriate.
- **vs Graphormer**: Graphormer integrates GCN into Transformer for single human body reconstruction. HOI-TG extends this idea to the HOI dual-subject scenario and designs different graph residual blocks for humans and objects.
- **vs StackFLOW**: StackFLOW relies on a time-consuming post-optimization process, whereas HOI-TG achieves higher end-to-end inference efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear concept of implicit interaction modeling, though the hybrid Transformer+GCN architecture is not completely pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison on two datasets + multiple ablation groups + attention visualization.
- Writing Quality: ⭐⭐⭐⭐ Motivation clearly analyzed and methods well-structured.
- Value: ⭐⭐⭐⭐ Practical improvement to HOI reconstruction, with an inspiring implicit modeling perspective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] End-to-End Implicit Neural Representations for Classification](end-to-end_implicit_neural_representations_for_classification.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[ICLR 2026\] Mesh Splatting for End-to-end Multiview Surface Reconstruction](../../ICLR2026/3d_vision/mesh_splatting_for_end-to-end_multiview_surface_reconstruction.md)
- [\[CVPR 2025\] ESCAPE: Equivariant Shape Completion via Anchor Point Encoding](escape_equivariant_shape_completion_via_anchor_point_encoding.md)
- [\[ICML 2026\] EPS3D: End-to-End Feed-Forward 3D Panoptic Segmentation](../../ICML2026/3d_vision/eps3d_end-to-end_feed-forward_3d_panoptic_segmentation.md)

</div>

<!-- RELATED:END -->

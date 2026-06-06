---
title: >-
  [Paper Note] Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection
description: >-
  [CVPR2026][Segmentation][CAD Generation] This paper proposes a pointer-based command sequence representation that explicitly incorporates B-Rep geometric entities (edges/faces) into autoregressive CAD generation…
tags:
  - "CVPR2026"
  - "Segmentation"
  - "CAD Generation"
  - "B-Rep"
  - "Pointer Network"
  - "Command Sequence"
  - "LLM"
  - "Graph Neural Network"
  - "Chamfer/Fillet"
date: 2026-05-08
content_hash: f8f0c857492c69d5
---

# Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection

**Conference**: CVPR2026
**arXiv**: [2603.04337](https://arxiv.org/abs/2603.04337)  
**Code**: [Snitro/Pointer-CAD](https://github.com/Snitro/Pointer-CAD)  
**Area**: 3D CAD Generation
**Keywords**: CAD Generation, B-Rep, Pointer Network, Command Sequence, LLM, Graph Neural Network, Chamfer/Fillet

## TL;DR

This paper proposes a pointer-based command sequence representation that explicitly incorporates B-Rep geometric entities (edges/faces) into autoregressive CAD generation, enabling chamfer/fillet operations in command sequence methods for the first time while substantially reducing topology errors caused by quantization errors.

## Background & Motivation

**Time-consuming CAD modeling**: Traditional CAD workflows (2D sketch → 3D modeling → B-Rep storage) are heavily manual, particularly for complex designs.

**Limitations of command sequence methods**: Existing command sequence methods (DeepCAD, Text2CAD, etc.) encode CAD operations as token sequences, enabling fast generation but failing to support operations requiring entity selection (e.g., chamfer, fillet), which must explicitly reference existing geometric entities (edges or faces).

**Quantization error**: In LLM-based sequence generation, discretization of continuous parameters introduces quantization errors, causing newly drawn curves to misalign with existing edges and sketch planes to mismatch target faces, thereby disrupting topological connectivity.

**Efficiency bottleneck of code representations**: Code generation methods (CadQuery/FreeCAD), while flexible, produce token sequences approximately four times longer than command sequences (424 vs. 110 tokens), resulting in significantly longer inference times.

**Inadequacy of general-purpose LLMs**: General-purpose LLMs such as Claude Opus 4, Gemini 2.5 Pro, and GPT-5.2 exhibit low success rates and poor geometric consistency when directly generating CadQuery code (IR as high as 24–50%).

**Ambiguity in entity selection**: Prior attempts to achieve entity selection via face labels and face intersection curves are problematic, as edges derived from face intersections may not be unique, introducing selection ambiguity.

## Method

### Overall Architecture

Pointer-CAD adopts a **multi-step generation strategy**, decomposing CAD model construction into sequential steps, each conditioned on the textual description and the B-Rep generated in preceding steps:

- **Multimodal fusion module**: Fuses tokenized text with B-Rep geometric information; text is tokenized once and reused across all steps, while the B-Rep is incrementally updated after each operation.
- **LLM backbone**: Qwen2.5 (0.5B/1.5B) with LoRA fine-tuning; the final hidden states are fed into two independent fully connected layers to predict Label/Value Tokens and Pointers, respectively.
- **Vector translation module**: Converts predicted command sequences into executable B-Rep geometry.

### Pointer-based Command Sequence Representation

Each token belongs to one of three categories:

| Type | Description |
|------|-------------|
| **Label Token** | Semantic label indicating operation type or structural boundary (e.g., `<ss>` for sketch start, `<sc>` for chamfer start) |
| **Value Token** | Numerical parameter (coordinates, angles, etc.); continuous parameters are quantized to $q$-bit integers with $2^q$ levels |
| **Pointer** | References a face or edge in the B-Rep; the LLM outputs a 128-dimensional vector, and cosine similarity with candidate entities is computed for selection |

Three fundamental operation types:

- **Sketch-Extrude step**: Sketch plane selection uses a Pointer to choose from candidate faces (replacing traditional 6-parameter regression), followed by drawing Line/Arc/Circle primitives on the plane and extruding with $E:(e_p, e_n, b)$.
- **Chamfer step**: $C:(\mathbf{p}, c)$, where a Pointer set $\mathbf{p}$ selects target edges and $c$ is the uniform chamfer distance.
- **Fillet step**: $F:(\mathbf{p}, f)$, similarly using Pointers to select edges, with $f$ as the uniform fillet radius.

### B-Rep Encoder and GNN

- The B-Rep is modeled as a **face adjacency graph** $\mathcal{G}(V, E)$, where nodes represent faces and edges represent shared boundaries.
- Face features: 3D coordinates, normals, Gaussian curvature, and visibility flags sampled on a $32 \times 32$ UV grid → average pooling → 128-dimensional representation.
- Edge features: 3D coordinates, tangents, and derivatives sampled at 32 equidistant points → average pooling → 128-dimensional representation.
- $K$-layer GNN propagation: node updates aggregate neighbor messages (GIN-like mechanism); edge updates employ multi-head attention (MHA) to extract information from global face features.

### Loss & Training

$$\mathcal{L} = \lambda_v \cdot \mathcal{L}_v + \lambda_p \cdot \mathcal{L}_p$$

- **Label/Value prediction**: Cross-entropy classification loss $\mathcal{L}_v$ with label smoothing.
- **Pointer prediction**: Contrastive regression loss $\mathcal{L}_p$ accommodating multiple valid candidates (equivalent entities on coplanar/colinear geometries); positive examples maximize cosine similarity, negative examples minimize it, with a learnable temperature $\tau$.

## Key Experimental Results

### Main Results: Text-to-CAD Generation

**Recap-DeepCAD dataset** (176K models, no chamfer/fillet):

| Model | Line F1↑ | Arc F1↑ | Circle F1↑ | CD mean↓ | CD median↓ | SegE↓ | FluxEE↓ |
|-------|----------|---------|------------|----------|-----------|-------|---------|
| DeepCAD | 80.14 | 31.41 | 79.04 | 37.47 | 12.56 | 0.53 | 25.85 |
| Text2CAD | 88.12 | 45.19 | 87.03 | 17.48 | 3.38 | 0.44 | 17.75 |
| CADmium-7B | 85.13 | 25.68 | 74.94 | 10.53 | 0.44 | 1.21 | 32.22 |
| **Pointer-CAD-0.5B** | **97.70** | **85.70** | **98.27** | 3.81 | 0.54 | **0.13** | **2.14** |
| **Pointer-CAD-1.5B** | 98.73 | 95.14 | 98.66 | **2.58** | **0.30** | 0.11 | 2.97 |

**Recap-OmniCAD+ dataset** (575K models, with chamfer/fillet):

| Model | Chamfer F1↑ | Fillet F1↑ | CD mean↓ | SegE↓ | FluxEE↓ |
|-------|------------|-----------|----------|-------|---------|
| Other methods | Not supported | Not supported | 11.60–27.48 | 0.51–1.39 | 26.36–42.59 |
| **Pointer-CAD-0.5B** | **89.74** | **82.54** | 5.49 | **0.15** | **3.51** |
| **Pointer-CAD-1.5B** | 94.32 | 89.85 | **2.86** | 0.17 | 3.44 |

### Ablation Study: Effectiveness of GNN

| Setting | IR↓ | Arc F1↑ | CD mean↓ |
|---------|-----|---------|----------|
| Pointer-CAD w/o GNN (MLP) | 22.73 | 67.14 | 5.13 |
| **Pointer-CAD w/ GNN** | **15.02** | **85.70** | **3.81** |
| Text2CAD w/o GNN | 30.16 | 45.19 | 17.48 |
| Text2CAD w/ GNN | 27.17 | 51.85 | 14.33 |

The GNN yields particularly pronounced gains in arc (Arc) structure modeling (F1: 67.14 → 85.70).

### Key Findings

- **0.5B model outperforms 7B**: Pointer-CAD-0.5B surpasses CADmium-7B on nearly all metrics, demonstrating that representation design matters more than model scale.
- **SegE reduced by an order of magnitude**: The Pointer mechanism effectively eliminates topology fragmentation caused by quantization errors via snap-to-entity alignment (SegE: 0.44–1.21 → 0.11–0.13).
- **Substantial FluxEE improvement**: Entity watertightness improves from 17–38 to 2–3, yielding generated models closer to watertight solids.
- **Comparison with general-purpose LLMs**: GPT-5.2 achieves an IR of 23.9% (nearly one-quarter of models fail to generate), while Pointer-CAD-0.5B achieves only 14.79%.

## Highlights & Insights

- The **Pointer mechanism** innovatively introduces the Pointer Network concept into CAD command sequences, **enabling chamfer/fillet operations in command sequence methods for the first time**.
- The **multi-step conditional generation** architecture is elegant: each step generates based on cumulative B-Rep and text conditions, faithfully simulating real engineering workflows.
- **Data engineering** is rigorous: a 575K-scale dataset is constructed with multi-view descriptions automatically annotated via Qwen2.5-VL, preserving real parameters rather than normalized values.
- Exceptionally high token efficiency: an average of only 110 tokens per model with 2.13-second inference, far superior to code-based representations.

## Limitations & Future Work

- Evaluation is currently limited to the **text-conditioned** setting, without extension to multimodal inputs such as images or point clouds.
- Only **single-part modeling** is supported; assembly-level constraint relationships (mating constraints, hierarchical dependencies) are not addressed.
- Complex models (≥4 non-sketch operations) occasionally exhibit local positioning deviations.
- Pointer selection relies on cosine matching of B-Rep geometric features; its ability to disambiguate highly similar faces/edges in highly symmetric models remains to be validated.

## Related Work & Insights

- **B-Rep generation**: Methods such as ComplexGen and CMT directly generate B-Rep hierarchical structures, but modeling complex topological relationships remains challenging.
- **CSG representation**: Combines primitives via Boolean operations but struggles to represent curved surfaces (e.g., fillets), and CSG representations are non-unique.
- **Command sequences**: DeepCAD → SkexGen → Text2CAD → CAD-MLLM → CADFusion progressively incorporate more modalities and operations, yet none support entity selection. Fan et al. attempted face-label-based entity selection, but edges derived from face intersection curves remain ambiguous.
- **Code representations**: CADmium (JSON), CadQuery/FreeCAD API approaches are flexible but produce long token sequences with slow inference.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introducing the Pointer mechanism into CAD sequence generation is a key innovation, enabling entity referencing in command sequences for the first time
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset evaluation, comparison with general-purpose LLMs, GNN ablation, and qualitative analysis, though multimodal input experiments are absent
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, well-motivated problem formulation
- Value: ⭐⭐⭐⭐⭐ — Addresses a core limitation of command sequence methods; the 0.5B model surpassing a 7B model carries strong practical significance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting](fov-net_rotation-invariant_cad_b-rep_learning_via_field-of-view_ray_casting.md)
- [\[CVPR 2026\] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)
- [\[ICML 2026\] Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion](../../ICML2026/segmentation/refining_context-entangled_content_segmentation_via_curriculum_selection_and_ant.md)
- [\[CVPR 2026\] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)
- [\[CVPR 2026\] Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels](towards_high-quality_image_segmentation_improving_topology_accuracy_by_penalizin.md)

</div>

<!-- RELATED:END -->

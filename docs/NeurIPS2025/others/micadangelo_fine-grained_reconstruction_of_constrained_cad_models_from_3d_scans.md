---
title: >-
  [Paper Note] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans
description: >-
  [NeurIPS 2025][CAD reverse engineering] MiCADangelo emulates the reverse engineering workflow of human CAD designers: it extracts 2D patterns via multi-plane cross-section analysis, predicts constrained parametric sketches, and optimizes extrusion parameters, achieving for the first time complete parametric model reconstruction with sketch constraints in 3D CAD reverse engineering.
tags:
  - NeurIPS 2025
  - CAD reverse engineering
  - sketch constraints
  - cross-sectional slicing
  - differentiable extrusion optimization
  - parametric modeling
date: 2026-05-08
content_hash: 4a5a7a5a72317bfd
---

# MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans

**Conference**: NeurIPS 2025
**arXiv**: [2510.23429](https://arxiv.org/abs/2510.23429)
**Code**: None
**Area**: Other
**Keywords**: CAD reverse engineering, sketch constraints, cross-sectional slicing, differentiable extrusion optimization, parametric modeling

## TL;DR

MiCADangelo emulates the reverse engineering workflow of human CAD designers: it extracts 2D patterns via multi-plane cross-section analysis, predicts constrained parametric sketches, and optimizes extrusion parameters, achieving for the first time complete parametric model reconstruction with sketch constraints in 3D CAD reverse engineering.

## Background & Motivation

CAD reverse engineering—converting 3D scans (meshes) into parametric CAD models—is a critical step in manufacturing and product development. Existing deep learning methods fall into two categories, each with notable shortcomings:

**Bottom-up methods** (e.g., Point2Cyl): Starting from geometry to predict individual extrusion solids, preserving local detail, but producing outputs that are not fully parametric and cannot be seamlessly integrated into CAD software.

**Top-down methods** (e.g., CAD-SIGNet): Directly predicting sketch-extrusion sequence parameters from point clouds, yielding fully parametric outputs, but often dominated by large-scale structures at the expense of fine-grained geometric detail.

More critically, **existing methods entirely ignore sketch constraints** (e.g., parallel, perpendicular, tangent), which are central to CAD modeling—they encode design intent and govern how a model responds to edits.

The authors observe that human designers perform reverse engineering by: extracting 2D cross-sections → reconstructing profiles with parametric curves and applying constraints → extruding into 3D solids. MiCADangelo automates this process.

## Method

### Overall Architecture

Given an input 3D mesh $\mathbf{M}$, MiCADangelo executes three sequential stages:
1. **Sketch plane detection**: Uniformly sampled cross-sectional planes along the x/y/z axes to identify key cross-sections.
2. **Constrained sketch parametrization**: Predicting parametric sketch primitives and constraints for closed loops within key cross-sections.
3. **Differentiable extrusion optimization**: Optimizing extrusion parameters per sketch to best fit the input mesh.

### Key Designs

1. **Sketch plane detection network**: $N$ equally spaced planes are sampled along each of the three axes (default: 40 per axis), yielding cross-sectional slices $\{\mathcal{S}_i\}_{i=1}^N$. Each slice is projected into a binary image $\mathbf{X}_i \in \{0,1\}^{128 \times 128}$, encoded via a ResNet34 backbone, augmented with three context embeddings (position index, axis identity, normalized parameters), and passed through a 4-layer 4-head Transformer encoder for global interaction. A binary classifier then predicts whether each slice constitutes a key sketch plane.

2. **Constrained sketch parametrization network**: Each closed loop in a key cross-section is rendered as a binary image, encoded by ResNet34, and passed into a Transformer encoder-decoder that predicts embeddings for $n_p$ sketch primitives. Two prediction heads follow:

   - **Parametrization head**: Predicts primitive type (line segment / circle / arc) and geometric parameters (endpoints, center, radius, etc.).
   - **Constraint prediction head**: Predicts 13 constraint types between primitive pairs (coincident, concentric, equal, horizontal, vertical, parallel, perpendicular, tangent, etc.).

   The network operates in a quantized space, pre-trained on the SketchGraphs dataset and fine-tuned on synthetic noisy closed-loop images.

3. **Differentiable extrusion optimization**: For each sketch $\mathcal{K}_j$, anchor points $\{\mathbf{r}_k\}$ are sampled along the loop boundary, and a shared learnable extrusion length $h_j$ generates extrusion vectors $\rho_k = \mathbf{r}_k + h_j \mathbf{v}_j$. Points $\mathcal{Q}$ sampled on the mesh minimize the distance to the nearest extrusion vector:
$$\mathcal{L}_{extr} = \frac{1}{n_M} \sum_{l=1}^{n_M} d(\mathbf{q}_l, \rho_{min})^2 + \lambda \sum_{i,j} h_j^2$$
   Extrusion type (add/cut) is determined by the nesting level of the loop (outermost = add, alternating thereafter).

### Loss & Training

- Sketch plane detection: Binary cross-entropy loss, trained for 20 epochs on the DeepCAD training set, learning rate $10^{-4}$.
- Sketch parametrization: Pre-trained on SketchGraphs, fine-tuned for 50 epochs on synthetic noisy closed-loop images.
- Extrusion optimization: 200 gradient descent iterations, learning rate $2 \times 10^{-4}$.
- Both networks share a ResNet34 encoder, first trained with the parametrization network, then fine-tuned during plane detection training.
- Optimizer: AdamW.

## Key Experimental Results

### Main Results (DeepCAD & Fusion360)

| Method | DeepCAD CD↓ | DeepCAD IoU↑ | DeepCAD IR↓ | DeepCAD ECD↓ | Fusion360 CD↓ | Fusion360 IoU↑ | Fusion360 IR↓ | Fusion360 ECD↓ |
|------|------------|-------------|------------|-------------|--------------|---------------|--------------|---------------|
| DeepCAD | 9.64 | 46.7 | 7.1 | — | 89.2 | 39.9 | 25.2 | — |
| Point2Cyl | 4.27 | 73.8 | 3.9 | — | 4.18 | 67.5 | 3.2 | — |
| CAD-Diffuser | 3.02 | 74.3 | 1.5 | — | 3.85 | 63.2 | 1.7 | — |
| CAD-SIGNet | 0.28 | 77.6 | 0.9 | 0.74 | 0.56 | 65.6 | 1.6 | 4.14 |
| **MiCADangelo** | **0.20** | **80.6** | 2.6 | **0.46** | **0.48** | **68.7** | 3.2 | **2.66** |

### Complex Models and Constraint Robustness

| Test Scenario | Method | CD↓ | IoU↑ | IR↓ | ECD↓ |
|---------|------|-----|------|-----|------|
| Models with ≥4 loops | CAD-SIGNet | 1.34 | 49.2 | 3.2 | 4.75 |
| Models with ≥4 loops | **MiCADangelo** | **0.37** | **68.3** | 4.1 | **2.04** |
| Models with >2 extrusions | CAD-SIGNet | 3.95 | 40.6 | 5.4 | 9.81 |
| Models with >2 extrusions | **MiCADangelo** | **0.46** | **64.8** | **3.0** | **2.27** |
| Constraint deformation robustness | CAD-SIGNet | 2.89 | 57.4 | 3.5 | 20.43 |
| Constraint deformation robustness | **MiCADangelo** | **0.38** | **81.1** | 4.3 | **1.29** |
| CC3D real scans | CAD-SIGNet | 2.90 | 42.6 | 4.4 | 8.68 |
| CC3D real scans | **MiCADangelo** | **1.69** | **50.8** | **2.2** | **5.93** |

### Ablation Study

| Experiment | Key Result | Remarks |
|------|---------|------|
| Context embeddings for plane detection | F1: 0.296→0.870 | Position/axis/normalization embeddings are critical |
| Cross-dataset plane detection | DeepCAD F1=0.870, Fusion360=0.820, CC3D=0.777 | Good generalization |
| Sketch parametrization (SCD) | Davinci: 0.827, MiCADangelo: 0.283 | Attributed to noisy closed-loop fine-tuning |
| Number of extrusion vectors | Performance saturates at 8 vectors | Minimal impact on inference time |

### Key Findings

1. **Comprehensively surpasses SOTA on core metrics**: MiCADangelo achieves a median CD of 0.20 on DeepCAD (vs. 0.28 for CAD-SIGNet) and IoU of 80.6% (vs. 77.6%).
2. **Advantage is more pronounced on complex models**: IoU improves by 19.1 percentage points on models with 4+ loops (68.3 vs. 49.2) and by 24.2 points on multi-extrusion models.
3. **Core value of constraints**: Constrained models maintain structural consistency after sketch editing (ECD 1.29 vs. 20.43), demonstrating that constraints are essential for editability.
4. **Robustness to real scans**: MiCADangelo outperforms CAD-SIGNet on CC3D real-world scans, improving IoU by 8.2 percentage points.

## Highlights & Insights

- **Human-inspired design workflow**: Rather than pursuing end-to-end black-box reconstruction, MiCADangelo emulates the cross-section analysis → sketch reconstruction → extrusion workflow of human CAD designers, achieving both fine-grained accuracy and full parametrization.
- **First introduction of sketch constraints into 3D CAD reverse engineering**: Constraints not only enhance editability but also implicitly restrict the solution space, facilitating more accurate geometric reconstruction.
- **Effective exploitation of cross-section views**: By extracting geometric information from 2D cross-sections, the complex 3D problem is decomposed into multiple 2D image understanding tasks.
- **Differentiable extrusion optimization**: Casting extrusion parameter inference as a continuous optimization problem avoids the difficulty of discrete search.

## Limitations & Future Work

- Only extrusion operations are supported (consistent with prior work); more complex CAD operations such as revolve and sweep are not handled.
- Extrusion direction is based on the sketch plane normal, limiting performance on models with non-axis-aligned extrusions.
- Complex sketch primitives such as B-splines are not supported.
- The invalid rate (IR) metric is slightly higher than that of CAD-SIGNet, as the latter benefits from test-time sampling of multiple candidates and selecting the best.
- Plane detection relies on uniform sampling (40 per axis), which may miss key cross-sections that are not axis-aligned.

## Related Work & Insights

- Compared to CAD-SIGNet (top-down) and Point2Cyl (bottom-up), MiCADangelo combines the strengths of both paradigms.
- Sketch constraint prediction builds on Davinci (2D sketch parametrization) and improves robustness through noisy fine-tuning.
- The SketchGraphs dataset is leveraged to compensate for the absence of constraint annotations in DeepCAD and Fusion360.
- Insight: Decomposing complex 3D reconstruction into multiple 2D subproblems reduces learning difficulty and enables independent optimization of each module.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First constrained CAD reverse engineering; cross-section analysis is a novel perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, diverse scenarios, constraint robustness, real scans, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, rigorous formal definitions, intuitive comparison with human workflow)
- Value: ⭐⭐⭐⭐⭐ (A breakthrough contribution to the CAD reverse engineering field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](../../ICCV2025/others/generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[AAAI 2026\] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement](../../AAAI2026/others/mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)
- [\[ICLR 2026\] cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning](../../ICLR2026/others/cadrille_multi-modal_cad_reconstruction_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)

</div>

<!-- RELATED:END -->

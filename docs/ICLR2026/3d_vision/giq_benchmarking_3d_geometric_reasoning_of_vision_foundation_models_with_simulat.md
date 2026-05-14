---
title: >-
  [Paper Note] GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra
description: >-
  [ICLR 2026][3D Vision][geometric reasoning] This work introduces the GIQ benchmark, comprising 224 synthetic and real polyhedra, and systematically evaluates the geometric reasoning capabilities of vision foundation mode…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "geometric reasoning"
  - "benchmark"
  - "polyhedra"
  - "vision foundation models"
  - "VLM evaluation"
date: 2026-05-08
content_hash: ded975abded5120d
---

# GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra

**Conference**: ICLR 2026
**arXiv**: [2506.08194](https://arxiv.org/abs/2506.08194)
**Code**: [Available](https://toomanymatts.github.io/giq-benchmark/)
**Area**: 3D Vision
**Keywords**: geometric reasoning, benchmark, polyhedra, vision foundation models, VLM evaluation

## TL;DR

This work introduces the GIQ benchmark, comprising 224 synthetic and real polyhedra, and systematically evaluates the geometric reasoning capabilities of vision foundation models across four tasks—monocular 3D reconstruction, symmetry detection, mental rotation testing, and zero-shot classification—revealing significant deficiencies in the geometric understanding of current models.

## Background & Motivation

Modern vision models achieve strong performance on standard benchmarks, yet growing evidence suggests they lack genuine 3D geometric understanding:

**VLMs perform poorly on spatial tasks such as depth ordering**

**Monocular reconstruction algorithms struggle with shapes outside their training distribution**

**Existing 3D evaluation datasets** (e.g., Objaverse) **lack precise geometric property annotations**

Polyhedra serve as ideal evaluation objects: they possess well-defined categorical definitions (Platonic, Archimedean, Johnson solids, etc.), exact symmetry groups, and a hierarchical geometric complexity ranging from simple to highly complex.

## Method

### Overall Architecture

GIQ is a systematic benchmarking study organized around four evaluation dimensions:

1. **Monocular 3D Reconstruction**: recovering 3D geometry from a single image
2. **3D Symmetry Detection**: assessing whether visual encoders implicitly capture symmetry information
3. **Mental Rotation Test (MRT)**: judging shape equivalence across viewpoints
4. **Zero-Shot Polyhedron Classification**: evaluating whether frontier VLMs can recognize basic geometric shapes

### Key Designs

**(1) Dataset Construction**

224 unique polyhedra:

- **Platonic solids** (5): tetrahedron, cube, octahedron, dodecahedron, icosahedron
- **Archimedean solids** (13): convex polyhedra with regular polygon faces but non-identical vertices
- **Catalan solids** (13): duals of the Archimedean solids
- **Johnson solids** (92): faces are regular polygons but vertices lack uniformity
- **Stellation forms** (48), Kepler–Poinsot polyhedra (4), compounds (10), nonconvex uniform polyhedra (53)

Synthetic images: rendered with the Mitsuba physically-based renderer, 20 viewpoints per shape, at 256×256 resolution.
Real images: paper models photographed with a Nikon D3500 (6000×4000), approximately 20 images each in indoor and outdoor settings.

**(2) Monocular 3D Reconstruction Evaluation**

Three methods are evaluated: Shap-E, Stable Fast 3D, and OpenLRM. Even models trained on millions of 3D assets fail to reliably recover basic properties of a cube.

**(3) 3D Symmetry Detection**

Twelve encoders are tested (DINOv2, SigLIP, CLIP, DINO, MAE, VGGT, DUSt3R, MASt3R, etc.) with linear and nonlinear probes to detect central-point reflection, 4-fold, and 5-fold rotational symmetry. A weighted BCE loss addresses class imbalance, and 5-fold cross-validation is applied.

**(4) Mental Rotation Test**

The task is to determine whether two images (synthetic vs. real) depict the same polyhedron. Absolute differences of encoder embeddings are fed into a nonlinear probe classifier. The hard split contains geometrically similar polyhedron pairs. A user study with 42 participants establishes a human baseline.

**(5) Zero-Shot Polyhedron Classification**

Models evaluated include Claude 3.7, Gemini 2.5 Pro, ChatGPT o3/o4-mini-high, and 3D-native models LLaVA-3D, ShapeLLM, and PointBind.

### Loss & Training

- Symmetry detection: weighted BCE with weight $w_c = (N - n_c) / n_c$
- Mental rotation: standard classification loss
- 5-fold cross-validation to ensure shape-level generalization

## Key Experimental Results

### 3D Symmetry Detection (Synthetic Train → Real Test)

| Encoder | Central Reflection | 4-fold Rotation | 5-fold Rotation |
|---|---|---|---|
| DINOv2 | ~85% | **~93%** | ~80% |
| SigLIP | ~82% | ~88% | ~78% |
| MAE | ~65% | ~70% | ~60% |

### Mental Rotation Test (Hard Split, Syn-Wild)

| Model | Accuracy |
|---|---|
| SigLIP (nonlinear probe) | **~69%** |
| DINOv2 | ~67% |
| Human average | 68.05% |
| Human best | 90% |
| Most models | <60% |

### Zero-Shot Classification

| Model | Platonic | Archimedean | Catalan | Johnson | Nonconvex |
|---|---|---|---|---|---|
| ChatGPT o3 | **100%** | ~50% | <20% | <20% | <20% |
| Gemini 2.5 Pro | ~80% | ~60% | <20% | <20% | <20% |
| Claude 3.7 | ~80% | ~40% | <20% | <20% | <20% |
| 3D-native models | — | — | — | — | Do not surpass 2D VLMs |

### Ablation Study

- **Chain-of-thought prompting**: yields negligible gains; models frequently hallucinate in intermediate reasoning steps
- **Multi-view input**: provides only marginal improvement for low-symmetry Johnson solids
- **Linear vs. nonlinear probes**: performance is comparable for symmetry detection

### Key Findings

1. **Reconstruction failure**: all state-of-the-art reconstruction methods fail to reliably reconstruct even a cube
2. **Symmetry is detectable**: encoders such as DINOv2 implicitly capture 3D symmetry information (up to 93% for 4-fold rotation)
3. **Insufficient fine-grained discrimination**: most models approach chance-level performance on the hard mental rotation split
4. **Systematic geometric reasoning deficiencies in VLMs**: models confuse convex and nonconvex shapes, misidentify face types, and conflate compounds with stellations
5. **3D-native models do not outperform 2D VLMs**: even when provided with precise point clouds, they fail to surpass general-purpose VLMs
6. **Clear human advantage**: 68% of human participants outperform the best model

## Highlights & Insights

- **Polyhedra as a geometric litmus test**: mathematically well-defined objects enable rigorous evaluation
- **Separation of implicit vs. explicit geometric understanding**: encoders can detect symmetry via probing, yet fail on tasks requiring explicit geometric reasoning
- **Inclusion of a human baseline**: the 42-participant user study provides a meaningful comparative anchor
- **Exposing training data bias**: reconstruction models have learned noisy surface priors rather than mathematically precise geometry

## Limitations & Future Work

1. **Restricted to polyhedra**: generalization to arbitrary organic shapes remains to be investigated
2. **Limited dataset scale**: 224 shapes, with sparse coverage in certain categories
3. **No remediation proposed**: the work is purely diagnostic and does not propose training strategies to enhance geometric reasoning
4. **Evaluation focuses on zero-shot settings**: few-shot and fine-tuning scenarios are not explored

## Related Work & Insights

- **Probing 3D Awareness** (El Banani et al., 2024): GIQ extends probing to symmetry detection
- **Mental Rotation Test** (Shepard & Metzler, 1971): a classical paradigm borrowed from cognitive science
- **Insights**: existing models learn texture and statistical patterns rather than geometric essence; future work should incorporate mathematically generated geometric data

## Rating

- Novelty: 4/5 — first systematic benchmark for polyhedron geometric reasoning
- Technical depth: 3/5 — primarily an evaluation study with limited methodological innovation
- Experimental Thoroughness: 5/5 — four-dimensional evaluation, multi-model comparison, and human baseline
- Value: 4/5 — provides clear directions for improving geometric understanding in vision models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)
- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](../../AAAI2026/3d_vision/parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](../../AAAI2026/3d_vision/vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[ICCV 2025\] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](../../ICCV2025/3d_vision/vit-split_unleashing_the_power_of_vision_foundation_models_via_efficient_splitti.md)
- [\[CVPR 2026\] Foundry: Distilling 3D Foundation Models for the Edge](../../CVPR2026/3d_vision/foundry_distilling_3d_foundation_models_for_the_edge.md)

</div>

<!-- RELATED:END -->

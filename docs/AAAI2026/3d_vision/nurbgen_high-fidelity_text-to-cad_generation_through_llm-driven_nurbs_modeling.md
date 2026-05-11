---
title: >-
  [Paper Note] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling
description: >-
  [AAAI 2026][3D Vision][Text-to-CAD] NURBGen is the first text-to-CAD generation framework based on NURBS surface representation. By fine-tuning an LLM…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Text-to-CAD"
  - "NURBS"
  - "LLM"
  - "BRep"
  - "3D Generation"
date: 2026-05-08
content_hash: e3326020942e0e24
---

# NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling

**Conference**: AAAI 2026
**arXiv**: [2511.06194](https://arxiv.org/abs/2511.06194)
**Code**: Coming soon
**Area**: 3D Vision
**Keywords**: Text-to-CAD, NURBS, LLM, BRep, 3D Generation

## TL;DR

NURBGen is the first text-to-CAD generation framework based on NURBS surface representation. By fine-tuning an LLM, it maps natural language descriptions to structured NURBS parameter JSONs. A hybrid representation (untrimmed NURBS + analytic primitives) and a large-scale partABC dataset are introduced, achieving significant improvements over existing methods in geometric fidelity and dimensional accuracy.

## Background & Motivation

### Problem Definition

CAD modeling is essential in modern engineering and product design, yet creating detailed CAD models typically requires expertise in professional software (e.g., Onshape, AutoCAD) and is highly time-consuming. Text-to-CAD technology aims to enable designers to describe 3D objects in natural language without requiring professional modeling skills.

### Limitations of Prior Work

**Design-history dependency**: Nearly all existing methods (e.g., DeepCAD, Text2CAD, CAD-LLaMA) rely on design-history-based representations, where shapes are constructed via sequences of parametric operations (extrusion, 2D sketches). Although intuitive and editable, the training datasets (e.g., DeepCAD-170k) are small in scale and low in complexity (mostly cuboids and cylinders), limiting generalization capability.

**Underutilization of the ABC dataset**: The ABC dataset contains over one million 3D CAD models, but has two key limitations: (a) models are stored in BRep (Boundary Representation) format without design history; (b) high-quality text descriptions are absent.

**Difficulty of NURBS modeling**: Analytic surfaces in BRep are most commonly represented as NURBS, yet NURBS have rarely been explored in deep generative research due to challenges including efficient representation, non-differentiability of knot vectors, high parametric variability, and trimming complexity.

### Paper Goals

- Treat NURBS surfaces as language-aligned objects, encoding each surface as a JSON token sequence containing control points, degrees, weights, and knot vectors.
- Reformulate text-to-CAD as a language modeling task.
- Leverage the large scale and geometric diversity of the ABC dataset.

## Method

### Overall Architecture

The NURBGen pipeline proceeds as follows: (1) extract part-level CAD models from the ABC dataset → (2) encode each part in a hybrid format (NURBS + analytic primitives) and generate high-quality descriptions using a VLM → (3) fine-tune Qwen3-4B to map text descriptions to structured hybrid CAD representations → (4) output JSON is directly convertible to BRep models.

### Key Designs

#### 1. **CAD Representation (NURBS Parameter Extraction)**

- **Normalization**: Geometry is normalized into a $2 \times 2 \times 2$ bounding box centered at the origin.
- **NURBS conversion**: Each face is converted to an untrimmed NURBS representation using PythonOCC's `BRepBuilderAPI_NurbsConvert`, unifying all underlying surfaces.
- **Parameter extraction**: For each face, the following are extracted: control points (poles), knot vectors in both parametric directions, knot multiplicities, degrees in the u and v directions, rational weights, and periodicity flags.
- **Exact reconstruction**: Given these parameters, the original surface can be exactly reconstructed via the `Geom_BSplineSurface` constructor.

The mathematical definition of a NURBS surface is:

$$\mathbf{S}(u,v) = \frac{\sum_{i=0}^{n}\sum_{j=0}^{m}N_{i,p}(u)M_{j,q}(v)w_{ij}\mathbf{P}_{ij}}{\sum_{i=0}^{n}\sum_{j=0}^{m}N_{i,p}(u)M_{j,q}(v)w_{ij}}$$

#### 2. **Hybrid Representation**

This is one of the core innovations. Not all surfaces can be robustly represented as untrimmed NURBS — thin regions near holes or fillets in particular frequently introduce geometric artifacts.

- **Degeneracy detection**: The Chamfer Distance between each reconstructed surface $f_n$ and the ground-truth surface $f_{gt}$ is compared: $CD(f_n, f_{gt}) \leq \epsilon$, with threshold $\epsilon = 6 \times 10^{-4}$.
- **Fallback strategy**: When NURBS approximation is unacceptable, the original analytic primitive is retained (lines, circles, B-splines, ellipses, parabolas, hyperbolas).
- **Statistics**: In practice, approximately 70% of faces are modeled with NURBS, while 30% fall back to analytic primitives.
- **Advantages**: More expressive and compact than pure NURBS format, reducing parameter count and producing shorter, more token-efficient inputs.

#### 3. **Automatic Annotation Pipeline**

This addresses the lack of text descriptions in the ABC dataset:

- **Multi-view rendering**: Each BRep is first converted to a triangular mesh, then rendered in Blender from 6 viewpoints at 512×512 resolution with the Freestyle renderer enabled to overlay contour edges.
- **Metadata-guided annotation**: Geometric metadata inaccessible to VLMs is extracted — bounding box dimensions, surface area, volume, and topological hole count (genus computed via the Euler–Poincaré formula: $g = 0.5 \times (2 - \chi)$).
- **Description generation**: InternVL3-13B, a multi-view VLM, takes 6 rendered views and metadata-augmented prompts as input to generate shape-centric descriptions.
- **Quality validation**: GPT-4o validates a random sample of 1,000 instances, achieving approximately 85% accuracy.

### Dataset Construction (partABC)

- Part-level substructures are extracted from 200k models in ABC, yielding 3 million part-level CAD instances.
- **Complexity filtering**: A weighted scoring function is applied: $w(B) = l_1 \times \text{token\_count} + l_2 \times \text{through\_holes} + l_3 \times \frac{\text{surface\_area}}{\text{volume}} + l_4 \times \text{bbox\_diag}$
- Parts are categorized into three tiers: simple (≤0.12), moderate (0.12–0.23), and complex (>0.23).
- A subset of 10% simple + 50% moderate + 40% complex instances is retained, yielding approximately 300k high-quality samples.

### Loss & Training

- **Base model**: Qwen3-4B
- **Optimizer**: AdamW, learning rate $5 \times 10^{-5}$, linear warmup
- **LoRA**: rank=64, $\alpha=128$
- **Training**: 180k steps, batch size=1, 4×H200 GPUs, 3 days
- **Context window**: 8192 for training, 14k for inference
- **Temperature**: 0.3
- **Generation speed**: ~800 tokens/s on RTX 3090
- **Data processing**: Control point coordinates are retained to 6 decimal places; weights are compressed using (value, frequency) encoding.

## Key Experimental Results

### Main Results

| Model | User Pref. (1k)↑ | GPT Pref.↑ | IR↓ | CD↓ | HD↓ | JSD↓ | MMD↓ |
|-------|-----------------|-----------|-----|-----|-----|------|------|
| GPT-4o | 1.5 | 1.9 | 0.17 | 7.2 | 0.36 | 72.87 | 4.17 |
| DeepCAD | 5.6 | 6.1 | 0.32 | 10.28 | 0.45 | 89.77 | 4.43 |
| Text2CAD | 26.1 | 27.2 | 0.05 | 9.66 | 0.42 | 85.27 | 4.54 |
| **NURBGen** | **64.1** | **61.6** | **0.018** | **4.43** | **0.25** | **57.94** | **2.14** |

Note: CD, JSD, and MMD are multiplied by $10^2$. NURBGen substantially outperforms all baselines across every metric.

### Ablation Study

| Configuration | Human Pref.↑ | GPT-4o Pref.↑ | Notes |
|--------------|-------------|--------------|-------|
| NURBS-only | 28% | 21% | Untrimmed NURBS only, no analytic primitive fallback |
| **Hybrid (full)** | **72%** | **79%** | NURBS + analytic primitives |

The NURBS-only model exhibits pronounced geometric artifacts and reconstruction errors near holes, sharp transitions, and regions where NURBS fitting is imprecise.

### Key Findings

1. NURBGen achieves a CD of 4.43 (×$10^2$) on 7,500 test samples, 54% lower than the second-best method Text2CAD (9.66).
2. A top-1 human preference rate of 64.1% is achieved, far exceeding Text2CAD's 26.1%.
3. An invalidity rate of only 0.018 demonstrates strong geometric correctness of the generated BRep models.
4. The hybrid representation improves human preference by 44 percentage points over pure NURBS.

## Highlights & Insights

1. **NURBS as language**: Serializing NURBS surface parameters as JSON tokens elegantly reformulates CAD generation as a language modeling task — a significant paradigm shift.
2. **Practical utility of hybrid representation**: The 70% NURBS + 30% analytic primitive strategy achieves a favorable balance between robustness and token efficiency.
3. **Bottom-up data engineering**: The complete pipeline — from part extraction to complexity filtering to automatic annotation — enables effective utilization of the large-scale, unannotated ABC dataset.
4. **Extremely low invalidity rate**: An IR of 0.018 indicates strong geometric consistency in the structured parameters generated by the LLM.

## Limitations & Future Work

1. **Complex prompts**: For prompts involving complex descriptions (e.g., "a two-story house with a gable roof"), NURBGen struggles to capture fine-grained structural details.
2. **Geometric artifacts**: Self-intersections or topological inconsistencies arise in a minority of cases.
3. **Engraved text**: Prompts involving engraved text cannot be reconstructed.
4. **Context window limitation**: Current training is limited to 8,192 tokens; future work may explore long-context training to handle more complex assemblies.
5. Only 200k of the 1 million models in the ABC dataset have been processed; scaling to the full dataset remains as future work.

## Related Work & Insights

- **Distinction from NeuroNURBS**: NeuroNURBS employs a non-autoregressive transformer VAE to learn latent encodings of untrimmed NURBS, but does not support language-conditioned generation and cannot handle trimming.
- **Comparison with LLaMA-Mesh**: LLaMA-Mesh fine-tunes LLaMA to generate mesh vertices and faces as plain text, whereas NURBGen generates structured, editable NURBS parameters.
- **Insight**: Structured symbolic representations (vs. latent encodings) may be a more promising direction for LLM-driven 3D generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First NURBS-based text-to-CAD framework with an elegantly designed hybrid representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-metric evaluation combined with human assessment, though ablation studies are limited in scope.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete technical details.
- Value: ⭐⭐⭐⭐⭐ — Establishes NURBS as a viable alternative to design-history-based methods; the partABC dataset represents a substantial contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](../../CVPR2026/3d_vision/text-image_conditioned_3d_generation.md)
- [\[AAAI 2026\] MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation](mr-cosmo_visual-text_memory_recall_and_direct_cross-modal_alignment_method_for_q.md)
- [\[AAAI 2026\] AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation](anchords_anchoring_dynamic_sources_for_semantically_consiste.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](../../ICCV2025/3d_vision/segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling
description: >-
  [AAAI 2026][3D Vision][Text-to-CAD] This paper proposes NURBGen, the first text-to-CAD generation framework based on NURBS surface representation. By fine-tuning an LLM to convert natural language descriptions into structured NURBS parameter JSONs, and introducing a hybrid representation (untrimmed NURBS + analytical primitives) alongside the large-scale partABC dataset, it significantly outperforms existing methods in geometric fidelity and dimensional accuracy.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Text-to-CAD"
  - "NURBS"
  - "LLM"
  - "BRep"
  - "3D Generation"
date: 2026-05-08
content_hash: 8944c35ec00811d3
---

# NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling

**Conference**: AAAI 2026  
**arXiv**: [2511.06194](https://arxiv.org/abs/2511.06194)  
**Code**: Coming soon  
**Area**: 3D Vision  
**Keywords**: Text-to-CAD, NURBS, LLM, BRep, 3D Generation

## TL;DR

This paper proposes NURBGen, the first text-to-CAD generation framework based on NURBS surface representation. By fine-tuning an LLM to convert natural language descriptions into structured NURBS parameter JSONs, and introducing a hybrid representation (untrimmed NURBS + analytical primitives) alongside the large-scale partABC dataset, it significantly outperforms existing methods in geometric fidelity and dimensional accuracy.

## Background & Motivation

### Problem Definition

CAD modeling is essential in modern engineering and product design, but creating detailed CAD models typically requires expertise in professional software (such as Onshape, AutoCAD) and is highly time-consuming. Text-to-CAD technology aims to enable designers to describe 3D objects using natural language without requiring professional modeling skills.

### Limitations of Prior Work

**Design-History Dependence**: Almost all existing methods (such as DeepCAD, Text2CAD, CAD-LLaMA) rely on design-history-based representations, where shapes are constructed through a sequence of parametric operations (extrusion, 2D sketches). Although intuitive and editable, the training datasets (such as DeepCAD-170k) are small in scale and low in complexity (mostly cuboids and cylinders), limiting generalization capability.

**Underutilization of the ABC Dataset**: The ABC dataset contains over 1 million 3D CAD models, but faces two key limitations: (a) it stores models in BRep (Boundary Representation) format, lacking design history; (b) it lacks high-quality text descriptions.

**Difficulty in NURBS Modeling**: Analytical surfaces in BRep are most commonly represented by NURBS, but NURBS have been rarely explored in deep generation research due to challenges in efficient representation, the non-differentiability of knot vectors, high parameter variability, and trimming complexity.

### Design Motivation

- Treating NURBS surfaces as language-aligned objects, encoding each surface into a JSON token sequence containing control points, degrees, weights, and knot vectors.
- Formulating the text-to-CAD problem as a language modeling task.
- Leveraging the large scale and geometric diversity of the ABC dataset.

## Method

### Overall Architecture

The overall pipeline of NURBGen: (1) Extract part-level CAD models from the ABC dataset $\rightarrow$ (2) Encode each part into a hybrid format (untrimmed NURBS + analytical primitives) and generate high-quality descriptions using a VLM $\rightarrow$ (3) Fine-tune Qwen3-4B to map text descriptions to structured hybrid CAD representations $\rightarrow$ (4) Convert the output JSON directly into a BRep model.

### Key Designs

#### 1. **CAD Representation (NURBS Parameter Extraction)**

- **Normalization**: Normalize the geometry into a $2 \times 2 \times 2$ bounding box centered at the origin.
- **NURBS Conversion**: Use PythonOCC's `BRepBuilderAPI_NurbsConvert` to convert each face into an untrimmed NURBS representation, unifying all underlying surfaces.
- **Parameter Extraction**: Extract control points (poles), knot vectors in both parametric directions, knot multiplicities, degrees in $u$ and $v$ directions, rational weights, and periodicity flags for each face.
- **Precise Reconstruction**: With these parameters, the original surface can be precisely reconstructed using the `Geom_BSplineSurface` constructor.

The mathematical definition of a NURBS surface is:

$$\mathbf{S}(u,v) = \frac{\sum_{i=0}^{n}\sum_{j=0}^{m}N_{i,p}(u)M_{j,q}(v)w_{ij}\mathbf{P}_{ij}}{\sum_{i=0}^{n}\sum_{j=0}^{m}N_{i,p}(u)M_{j,q}(v)w_{ij}}$$

#### 2. **Hybrid Representation**

One of the core innovations. Not all surfaces can be robustly represented by untrimmed NURBS—especially thin regions near holes or fillets, which often introduce geometric artifacts.

- **Degeneration Detection**: Compare the Chamfer Distance between each reconstructed surface $f_n$ and the ground-truth surface $f_gt$: $CD(f_n, f_gt) \leq \epsilon$, with a threshold of $\epsilon = 6 \times 10^{-4}$.
- **Fallback Strategy**: When the NURBS approximation is unacceptable, retain the original analytical primitives (lines, circles, B-splines, ellipses, parabolas, hyperbolas).
- **Statistics**: In practice, approximately 70% of the faces are modeled with NURBS, while 30% fall back to analytical primitives.
- **Advantages**: More expressive and compact than the pure NURBS format, reducing the number of parameters and producing shorter, more token-efficient inputs.

#### 3. **Automatic Annotation Pipeline**

Resolving the lack of text descriptions in the ABC dataset:

- **Multi-view Rendering**: Each BRep is first converted into a triangular mesh and rendered from 6 views at a resolution of 512×512 in Blender, with the Freestyle renderer enabled to overlay feature edges.
- **Metadata Guidance**: Extract geometric metadata that VLMs cannot directly perceive—dimensions (length, width, height), surface area, volume, and the number of topological holes (calculated via the Euler-Poincaré formula for genus $g = 0.5 \times (2 - \chi)$).
- **Description Generation**: Use the InternVL3-13B multi-view VLM, inputting the 6 rendered views and metadata-augmented annotation prompts to generate shape-centric descriptions.
- **Quality Verification**: GPT-4o validates an accuracy of approximately 85% on 1000 random samples.

### Dataset Construction (partABC)

- Extract part-level substructures from 200k models in ABC, obtaining 3 million part-level CAD instances.
- **Complexity Filtering**: Use a weighted scoring function $w(B) = l_1 \times \text{token\_count} + l_2 \times \text{through\_holes} + l_3 \times \frac{\text{surface\_area}}{\text{volume}} + l_4 \times \text{bbox\_diag}$.
- Categorize parts into three levels: simple ($\le 0.12$), moderate ($0.12\text{--}0.23$), and complex ($> 0.23$).
- Retain 10% simple + 50% moderate + 40% complex, yielding approximately 300k high-quality samples in the end.

### Loss & Training

- **Base Model**: Qwen3-4B
- **Optimizer**: AdamW, learning rate $5 \times 10^{-5}$, linear warmup
- **LoRA**: rank=64, $\alpha=128$
- **Training**: 180k steps, batch size=1, 4×H200 GPUs, 3 days
- **Context Window**: 8192 for training, 14k for inference
- **Temperature**: 0.3
- **Generation Speed**: Approximately 800 tokens/s on RTX 3090
- **Data Processing**: Control point coordinates are kept to 6 decimal places; weights are compressed using (value, frequency)

## Key Experimental Results

### Main Results

| Model | User Preference (1k)↑ | GPT Preference↑ | Invalidity Rate IR↓ | CD↓ | HD↓ | JSD↓ | MMD↓ |
|------|---------|--------|-------|------|------|------|------|
| GPT-4o | 1.5 | 1.9 | 0.17 | 7.2 | 0.36 | 72.87 | 4.17 |
| DeepCAD | 5.6 | 6.1 | 0.32 | 10.28 | 0.45 | 89.77 | 4.43 |
| Text2CAD | 26.1 | 27.2 | 0.05 | 9.66 | 0.42 | 85.27 | 4.54 |
| **NURBGen** | **64.1** | **61.6** | **0.018** | **4.43** | **0.25** | **57.94** | **2.14** |

Note: CD, JSD, and MMD are scaled by $10^2$. NURBGen leads significantly across all metrics.

### Ablation Study

| Configuration | Human Preference↑ | GPT-4o Preference↑ | Note |
|------|----------|-----------|------|
| NURBS-only | 28% | 21% | Only untrimmed NURBS, no analytical primitive fallback |
| **Hybrid (Full)** | **72%** | **79%** | Hybrid NURBS + analytical primitives |

The NURBS-only model exhibits obvious geometric artifacts and reconstruction errors in areas with holes, sharp transitions, and imprecise NURBS fitting.

### Key Findings

1. NURBGen achieves a CD of only 4.43 ($\times 10^2$) on 7500 test samples, which is 54% lower than the runner-up Text2CAD (9.66).
2. A top-1 preference rate of 64.1% in human evaluation, far exceeding Text2CAD's 26.1%.
3. An invalidity rate of only 0.018, indicating high geometric correctness of the generated BReps.
4. The hybrid representation yields a 44 percentage point improvement over pure NURBS in human evaluation.

## Highlights & Insights

1. **NURBS as Language**: Serializing NURBS surface parameters into JSON tokens elegantly converts CAD generation into a language modeling task, which represents a major paradigm shift.
2. **Utility of Hybrid Representation**: The hybrid strategy of 70% NURBS + 30% analytical primitives balances robustness and token efficiency effectively.
3. **Bottom-up Data Engineering**: The complete pipeline design, spanning part extraction $\rightarrow$ complexity filtering $\rightarrow$ automatic annotation, enables the utilization of the large-scale, unlabeled ABC dataset.
4. **Extremely Low Invalidity Rate**: An invalidity rate of 0.018 demonstrates that structured parameters generated by the LLM maintain strong geometric consistency.

## Limitations & Future Work

1. **Complex Prompts**: For complex descriptions (e.g., "a two-story house with a gabled roof"), NURBGen struggles to capture fine structures.
2. **Geometric Artifacts**: Self-intersections or topological inconsistencies occur in a minority of cases.
3. **Text Engraving**: Incapable of reconstructing prompts containing engraved text.
4. **Context Window Limitations**: Current training is limited to 8192 tokens. Future research could investigate long-context training to process more complex assemblies.
5. Only 200k models (from a total of 1 million) in the ABC dataset were processed; this can be scaled up in the future.

## Related Work & Insights

- **Differences from NeuroNURBS**: NeuroNURBS uses a non-autoregressive transformer VAE to learn latent codes of untrimmed NURBS, but it does not support language-conditional generation and cannot handle trimming issues.
- **Comparison with LLaMA-Mesh**: LLaMA-Mesh fine-tunes LLaMA to generate plain text representing mesh vertices and faces, whereas NURBGen generates structured, editable NURBS parameters.
- **Insights**: Structured symbolic representations (vs. latent codes) may represent a more promising direction in LLM-driven 3D generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first NURBS-based text-to-CAD framework, featuring an ingeniously designed hybrid representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across multiple metrics along with human evaluation, though ablation studies are relatively limited.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and highly detailed technical presentation.
- Value: ⭐⭐⭐⭐⭐ — Pioneers a new direction using NURBS as an alternative to design-history-based approaches; the partABC dataset is highly valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LATTICE: Democratize High-Fidelity 3D Generation at Scale](../../CVPR2026/3d_vision/lattice_democratize_high-fidelity_3d_generation_at_scale.md)
- [\[ICML 2026\] RelaxFlow: Text-Driven Amodal 3D Generation](../../ICML2026/3d_vision/relaxflow_text-driven_amodal_3d_generation.md)
- [\[CVPR 2026\] CoSMo3D: Open-World Promptable 3D Semantic Segmentation through LLM-Guided Canonical Spatial Modeling](../../CVPR2026/3d_vision/cosmo3d_open-world_promptable_3d_semantic_segmentation_through_llm-guided_canoni.md)
- [\[ICLR 2026\] Learning Hierarchical and Geometry-Aware Graph Representations for Text-to-CAD](../../ICLR2026/3d_vision/learning_hierarchical_and_geometry-aware_graph_representations_for_text-to-cad.md)
- [\[CVPR 2026\] Text-Driven 3D Hand Motion Generation from Sign Language Data](../../CVPR2026/3d_vision/text-driven_3d_hand_motion_generation_from_sign_language_data.md)

</div>

<!-- RELATED:END -->

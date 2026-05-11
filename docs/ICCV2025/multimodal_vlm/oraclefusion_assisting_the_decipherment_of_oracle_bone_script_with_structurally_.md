---
title: >-
  [Paper Note] OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography
description: >-
  [ICCV 2025][Multimodal VLM][oracle bone script] This paper proposes OracleFusion, a two-stage semantic typography framework. Stage 1 employs MLLM-enhanced Spatial Awareness Reasoning (SAR) to analyze the glyph structure…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "oracle bone script"
  - "semantic typography"
  - "MLLM"
  - "vector graphics generation"
  - "spatial awareness reasoning"
  - "score distillation"
date: 2026-05-08
content_hash: 4d5c72d8b2310ebd
---

# OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography

**Conference**: ICCV 2025
**arXiv**: [2506.21101](https://arxiv.org/abs/2506.21101)
**Code**: [GitHub](https://github.com/lcs0215/OracleFusion)
**Area**: Multimodal VLM / Oracle Bone Script Decipherment / Semantic Typography
**Keywords**: oracle bone script, semantic typography, MLLM, vector graphics generation, spatial awareness reasoning, score distillation

## TL;DR
This paper proposes OracleFusion, a two-stage semantic typography framework. Stage 1 employs MLLM-enhanced Spatial Awareness Reasoning (SAR) to analyze the glyph structure of oracle bone script (OBS) and localize key components. Stage 2 introduces Structural Oracle Vector Fusion (SOVF), which generates semantically enriched vector glyphs through glyph structure constraints and skeleton-preserving losses, conveying semantic meaning while preserving original glyph integrity to assist expert decipherment of undeciphered OBS characters.

## Background & Motivation

**Background**: Oracle bone script (OBS) is one of the earliest writing systems, dating back approximately 3,000 years to the Shang Dynasty. Of the roughly 4,500 discovered OBS characters, only about 1,600 have been successfully deciphered, while nearly 3,000 remain undecoded.

**Complexity of Decipherment**: OBS decipherment requires integrating (1) compositional glyph analysis—reconstructing the scene represented by the ancient symbol, (2) contextual reasoning—inferring meaning from context, and (3) evolutionary tracing—identifying the corresponding modern Chinese character.

**Limitations of Prior Work**:
- **OBS Decipher (OBSD)**: Uses conditional diffusion models to directly translate OBS into modern Chinese characters, but lacks an explicit semantic analysis process and offers poor interpretability.
- **GenOV**: Uses ControlNet to generate photorealistic images but fails to preserve glyph structural information.
- **Word-As-Image**: A semantic typography approach that constrains only outer contours, performing poorly on complex OBS structures with multiple radicals.

**Key Challenge**:
- **(1)** How to systematically analyze OBS glyph structure and interpret the meaning of each component?
- **(2)** How to reconstruct a semantic scene while preserving the original glyph structure?

**Key Insight**: The paper recasts OBS decipherment as a "structurally constrained semantic typography" problem—analyzing glyph radicals → localizing key components → deforming radicals into semantically relevant graphics while preserving the overall glyph structure.

## Method

### Overall Architecture
A two-stage pipeline:
- **Stage 1 (OBSUG)**: An MLLM analyzes OBS glyphs and outputs key components, spatial relationships, semantic descriptions, and layout bounding boxes.
- **Stage 2 (SOVF)**: SDS loss drives SVG parameter optimization while glyph structure constraints (GSDS) and skeleton-preserving constraints (SKST) are applied, producing semantically rich vector glyphs that remain faithful to the original glyph structure.

### Key Design 1: Oracle Glyph Vectorization (OGV)

Converts raster OBS images into differentiable SVG vector representations:
1. The Zhang-Suen skeletonization algorithm extracts $n$ single-pixel-wide skeleton paths $P^s = \{p_i^s\}_{i=1}^n$.
2. Direction vectors $\mathbf{v}_j = C_{j+1}^s - C_j^s$ are computed; rotating by 90° yields normal vectors $\mu_j$.
3. Normal vectors are smoothed via a sliding window: $\tilde{\mu}_j = \frac{1}{k} \sum_{i=j-k/2}^{j+k/2} \mu_i$.
4. Left and right contour points are generated based on stroke width $w$ and fitted with cubic splines.
5. Skeleton control points are embedded in the SVG to provide structural constraints for the subsequent SKST loss.

### Key Design 2: OBSUG (MLLM-based OBS Analysis)

Leveraging QWEN-VL's multi-turn dialogue capability, OBS data is processed in three sub-stages:

**(1) Key Structural Component Identification**:

$$K_i = \Psi(o_i; \theta_{key})$$

The MLLM identifies radicals/key elements in the OBS character, such as "bird," "mountain," or "person."

**(2) Spatial Awareness Reasoning (SAR)**: The model is guided to output spatial relationships between components, constructing a directed acyclic graph (DAG):

$$G_i = (V, E), \quad V = K_i, \quad E = \{(k_a, k_b, r) \mid k_a, k_b \in V\}$$

$$R_i = \Psi(o_i, K_i; \theta_{\text{spa}})$$

SAR enhances the MLLM's understanding of spatial structure and is a critical step in glyph analysis.

**(3) Fine-Grained Semantic Generation**: Components and spatial relations are integrated to produce a holistic semantic description:

$$T_i = \Psi(o_i, K_i, R_i; \theta_{cap})$$

**(4) Visual Grounding**: Localizes the spatial position (bounding box) of each component within the glyph:

$$L_i = \Psi(o_i, K_i, R_i, T_i; \theta_{loc})$$

### Key Design 3: Structural Oracle Vector Fusion (SOVF)

**LSDS Loss**: Standard Latent Score Distillation Sampling, driving SVG parameter optimization to align generated results with the semantic text prompt.

**GSDS Loss (Glyph Structure Constraint)**: Regional constraints applied to the cross-attention layers of Stable Diffusion:
- **In-Region Constraint**: Maximizes cross-attention responses within the designated region: $\mathcal{L}_{IR} = 1 - \frac{1}{P}\sum \text{TopK}(A_j^t \cdot M_j, P)$
- **Out-of-Region Constraint**: Minimizes responses outside the designated region: $\mathcal{L}_{OR} = \frac{1}{P}\sum \text{TopK}(A_j^t \cdot (1-M_j), P)$

This ensures that "bird" is generated in the bird region and "mountain" in the mountain region.

**SKST Loss (Skeleton Structure Preservation)**: Delaunay triangulation associates skeleton points with contour points, constraining angular consistency between the generated shape and the original glyph:

$$\mathcal{L}_{SKST}(P_m, \hat{P}_m) = \frac{1}{N} \sum_{i,j,k} \text{ReLU}(-\cos\theta_{i,j,k})$$

$$\cos\theta_{i,j,k} = \frac{\vec{\alpha_{i,j,k}} \cdot \hat{\vec{\alpha_{i,j,k}}}}{\|\vec{\alpha_{i,j,k}}\| \|\hat{\vec{\alpha_{i,j,k}}}\|}$$

**Overall Optimization Objective**:

$$\min_P \nabla_P \mathcal{L}_{\text{LSDS}} + w \cdot \nabla_P \mathcal{L}_{\text{GSDS}} + \beta \cdot \mathcal{L}_{SKST} + \gamma_t \cdot \mathcal{L}_{tone}$$

where $w$ is a learnable weight and $\beta = 0.5$.

### RMOBS Dataset
A multimodal dataset comprising 900 deciphered OBS characters and over 20K samples is constructed, with each sample containing a glyph image, semantic concept, key component annotations, and bounding box layout.

## Key Experimental Results

### Main Results

| Method | CLIPScore ↑ | Distance ↓ | SR ↑ | VA ↑ | GM ↑ |
|--------|------------|-----------|------|------|------|
| ClipDraw | 27.78 | 1.05 | 3.15 | 3.19 | 3.17 |
| Word-As-Image | 27.28 | 0.92 | 3.42 | 3.48 | 3.44 |
| **OracleFusion** | **28.30** | **0.86** | **3.97** | **3.90** | **3.95** |

OracleFusion achieves the best results across all three evaluation dimensions: CLIPScore (semantic relevance), Distance (glyph fidelity), and user study scores. The user study involved 70 participants familiar with Chinese character composition principles, rating 28 randomly selected OBS characters on a 1–5 scale.

### Ablation Study

**OBSUG Ablation**:

| Method | Acc (%) ↑ | BLEU-4 ↑ | Radical mIoU ↑ | Holistic mIoU ↑ |
|--------|----------|----------|---------------|----------------|
| End-to-End | 81.03 | 0.843 | 72.56 | 87.60 |
| Multi-Turn | 81.65 | 0.865 | 72.03 | 91.00 |
| **+ SAR** | **82.02** | **0.876** | **73.94** | **92.10** |

SAR improves performance across all metrics, with the largest gains in mIoU, demonstrating that spatial awareness reasoning is critical for accurate structural modeling.

**GSDS Loss Ablation**: Without the GSDS loss, generated results fail to convey complete concepts. For example, for the character "disaster" (depicting a house on fire), omitting GSDS causes the model to incorrectly deform fire into a burning house, whereas including GSDS correctly generates "fire" and "house" in their respective designated regions.

**SKST Loss Ablation**:
- $\beta = 0$: Degenerates to the ACAP loss similar to Word-As-Image, failing to preserve complex OBS structures.
- $\beta = 1$: Over-constrains the optimization, producing output nearly identical to the original input and losing semantic expressiveness.
- $\beta = 0.5$: Achieves the optimal balance, preserving glyph contours while conveying semantic meaning.

**Decipherment Demonstration on Undeciphered OBS**: OracleFusion generates plausible semantic interpretations for **undeciphered OBS characters** (e.g., "drooping crops," "thorned seeds"), providing valuable clues for expert analysis. For **deciphered OBS characters**, it accurately reproduces their structure and semantics.

## Highlights & Insights
1. **Creative Problem Formulation**: Recasting OBS decipherment as a "semantic typography" problem enables AI to reconstruct the scenes depicted by ancient characters rather than merely translating them—more closely mirroring the actual workflow of archaeologists.
2. **Elegant Integration of MLLM and SDS**: The MLLM handles "understanding" (structural analysis, semantic reasoning, component localization), while SDS handles "generation" (translating understanding into visual expression), with clear division of responsibilities.
3. **Unique Advantages of Vector Representation**: The SVG format supports lossless scaling and subsequent stylization (color/texture); the black-and-white design focuses attention on semantic expression.
4. **Contribution of the RMOBS Dataset**: With 20K+ samples and 900 annotated characters, the dataset substantially surpasses the prior GenOV benchmark of 364 characters, providing critical infrastructure for the field.
5. **Design Rationale of SAR**: Spatial awareness reasoning guides the MLLM to output relative positional relationships between radicals, analogous to the analytical approach of human experts.

## Limitations & Future Work
1. Generation quality is bounded by Stable Diffusion's prior capacity—for highly abstract OBS concepts, SD may fail to produce semantically aligned images.
2. The OGV method relies on the quality of the skeletonization algorithm and may underperform on noisy OBS rubbings.
3. The MLLM (QWEN-VL) has limited inherent understanding of OBS; the fine-tuning dataset covers only 900 characters.
4. The user study is limited in scale (70 participants, 28 samples), and statistical significance requires validation at larger scale.
5. The paper focuses on "assisting decipherment" rather than "automatic decipherment" and cannot directly produce the modern Chinese character corresponding to an OBS character.

## Related Work & Insights
- **OBS Processing**: OBSD (diffusion-based OBS translation), GenOV (VLM-based semantic expansion).
- **Semantic Typography**: Word-As-Image (SDS + contour constraints), DS-Fusion (adversarial learning).
- **Vector Graphics Generation**: VectorFusion, DiffSketcher (SDS → SVG).
- **MLLM-based Layout Generation**: LayoutGPT, GLIGEN.

## Rating
- **Novelty**: 5/5 (pioneering problem formulation that deeply integrates epigraphy with modern AI)
- **Technical Depth**: 4/5 (rich technical stack spanning OGV skeletonization, SAR spatial reasoning, GSDS regional constraints, and SKST skeleton preservation)
- **Experimental Thoroughness**: 3/5 (lacks large-scale quantitative evaluation; relies primarily on user studies and qualitative analysis)
- **Writing Quality**: 4/5 (narrative is compelling, though some formulas and notations are slightly redundant)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Background Invariance Testing According to Semantic Proximity](background_invariance_testing_according_to_semantic_proximity.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](../../CVPR2026/multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](../../AAAI2026/multimodal_vlm/aligning_the_true_semantics_constrained_decoupling_and_distr.md)
- [\[AAAI 2026\] UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment](../../AAAI2026/multimodal_vlm/unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)

</div>

<!-- RELATED:END -->

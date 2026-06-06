---
title: >-
  [Paper Note] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models
description: >-
  [ICML 2026][Multimodal VLM][Positional Encoding] Circle-RoPE is proposed to map the 2D coordinates of image tokens onto a torus orthogonal to the text position axis…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Positional Encoding"
  - "RoPE"
  - "Vision-Language Models"
  - "Cross-modal Decoupling"
  - "Spatial Reasoning"
date: 2026-05-08
content_hash: adbdf0f3bf34cc25
---

# Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2505.16416](https://arxiv.org/abs/2505.16416)  
**Code**: https://github.com/lose4578/CircleRoPE  
**Area**: Multimodal VLM  
**Keywords**: Positional Encoding, RoPE, Vision-Language Models, Cross-modal Decoupling, Spatial Reasoning  

## TL;DR
Circle-RoPE is proposed to map the 2D coordinates of image tokens onto a torus orthogonal to the text position axis, forming a cone-like geometry. This ensures that each text token is equidistant to all image tokens in the RoPE space (PTD=0), eliminating cross-modal pseudo-positional bias while preserving internal spatial structures through Alternating Geometric Encoding (AGE).

## Background & Motivation

**Background**: Rotary Positional Embedding (RoPE) is widely adopted in Large Language Models. When extended to Vision-Language Models (VLMs), mainstream approaches include: (1) flattening image tokens into a 1D sequence concatenated with text (LLaVA/InternLM-VL); (2) assigning the same position index to all image tokens (mPLUG-Owl3); (3) maintaining 2D spatial indices concatenated with text (M-RoPE in Qwen2-VL).

**Limitations of Prior Work**: Existing methods embed image and text tokens in a shared index space, causing cross-modal relative positions to be determined by concatenation order rather than semantic relevance. For instance, in VQA, "high on the clock tower" should align with the top region of the tower, but due to index ordering, the closest patches may be irrelevant, leading to semantic misalignment and inconsistent multi-token distances.

**Key Challenge**: Schemes (1) and (3) preserve image spatial information but introduce cross-modal coupling bias; scheme (2) eliminates bias but loses internal spatial structures. No solution simultaneously achieves cross-modal decoupling and preserved image spatial relationships.

**Goal**: Design a positional embedding where each text token is equidistant to all image tokens (eliminating cross-modal bias) while maintaining the relative spatial structure among image tokens.

**Key Insight**: From geometric first principles—if text tokens are viewed as "observers" and image tokens form a 2D plane, the observer should be positioned along the normal direction of the plane rather than being coplanar to avoid "perspective distortion."

**Core Idea**: Map image token coordinates to a torus orthogonal to the text position axis, forming a geometry similar to a right circular cone. By placing text tokens at the apex, they become equidistant to all points on the torus, achieving PTD=0 in the RoPE index space.

## Method

### Overall Architecture
Circle-RoPE applies a geometric transformation to image token $(w, h)$ indices before they are processed by RoPE rotation, building upon M-RoPE. The input consists of 2D grid coordinates for image tokens and 1D position indices for text tokens. The output is transformed 3D coordinates for images and original 1D indices for text. The methodology comprises two modules: Circular Image-token Projection (CIP) and Alternating Geometric Encoding (AGE).

### Key Designs

1.  **Per-Token Distance (PTD) Metric and Theoretical Guarantee**:
    - **Function**: Quantifies the degree of cross-modal positional decoupling to provide a formal objective for design.
    - **Mechanism**: For each text token $t$, the variance of its RoPE index distance to all image tokens is calculated. Defining $\bar{D}_t = \frac{1}{N_{\text{image}}}\sum_{i \in I} d(t, i)$, then $\text{PTD} = \frac{1}{N_{\text{image}} N_{\text{text}}} \sum_{t \in T} \sum_{i \in I} |d(t,i) - \bar{D}_t|$. PTD=0 implies that every text token is equidistant to all image tokens, meaning the RoPE-induced attention bias is consistent across all image patches. The authors prove that the bias in RoPE attention logits is bounded by PTD.
    - **Design Motivation**: Prior methods lacked quantitative metrics for cross-modal coupling. Hard embedding has a PTD of 2.22, Spatial embedding is 0.64, and Unordered is 0 but loses spatial info. PTD provides a clear optimization target: pursue PTD=0 while preserving spatial structure.

2.  **Circular Image Token Projection (CIP)**:
    - **Function**: Transforms 2D grid coordinates to a torus orthogonal to the text axis to achieve PTD=0 and preserve spatial structure.
    - **Mechanism**: Composed of three steps. **Coordinate Centralization**: Shifts image coordinates so the geometric center is at the origin. **Mixed-Angle Circular Mapping**: Projects centralized coordinates to a 2D torus where the angle $\theta^{\text{mix}}_{ij} = \alpha \cdot \theta^{\text{SA}}_{ij} + (1-\alpha) \cdot \theta^{\text{GA}}_{ij}$ is a weighted mixture of Spatial极角 (SA) and Grid Index Angle (GA), with radius $R$ controlling the scale. SA preserves spatial structure while GA ensures discriminability. **Target Plane Rotation**: Extends the 2D torus to 3D and rotates it so its normal vector aligns with the text position direction $V_{\text{text}}$, forming a cone. Optimal parameters are $\alpha=0.5$ and $R=10$.
    - **Design Motivation**: Traditional flattening or index sharing fails both decoupling and spatial preservation. A torus naturally satisfies the "equidistant to any point on the normal" property.

3.  **Alternating Geometric Encoding (AGE)**:
    - **Function**: Interleaves Circle-RoPE and M-RoPE layer-by-layer to fuse complementary geometric priors.
    - **Mechanism**: Defines a layer schedule $s(\ell) \in \{\text{Circle-RoPE}, \text{M-RoPE}\}$. Odd layers use Circle-RoPE for unbiased cross-modal alignment, while even layers use M-RoPE for grid-like local spatial awareness.
    - **Design Motivation**: Pure Circle-RoPE relaxes the 2D locality prior, which may hinder chart reading or fine-grained layout understanding. AGE assigns specific roles to different layers—decoupling for semantic localization and spatial layers for visual feature extraction.

## Key Experimental Results

### Main Results

Evaluated on Qwen2.5-VL-3B with SFT (MAmmoTH-VL-Sub 1M data), replacing only the positional encoding:

| Dataset | Qwen2.5-VL (SFT) | Circle-RoPE | Gain |
| :--- | :---: | :---: | :---: |
| MMMU (val) | 51.56 | **52.11** | +0.55 |
| MMMU-Pro | 28.01 | **28.44** | +0.43 |
| MathVista (mini) | 62.40 | **63.40** | +1.00 |
| AI2D | 79.22 | **81.80** | +2.58 |
| RealWorldQA | 66.10 | **66.54** | +0.44 |
| Average | 57.46 | **58.46** | +1.00 |

On the TAM benchmark, Func-IoU increased by +3.45 (71.19→74.64), validating the effectiveness of cross-modal decoupling.

### Ablation Study

| Configuration | MMMU | MMMU-Pro | MathVista | Average |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (M-RoPE) | 50.22 | 27.92 | 62.40 | 46.85 |
| CIP α=0, R=auto | 52.38 | 28.12 | 61.70 | 47.40 |
| CIP α=0.5, R=10 (Best) | 52.11 | 28.44 | 63.40 | 47.98 |
| CIP α=0.5, R=auto | 50.04 | 26.64 | 62.20 | 46.29 |
| Unordered (PTD=0) | 48.55 | 25.50 | 59.50 | — |
| Circle-RoPE (PTD=0) | 51.11 | 27.94 | 62.40 | — |

### Key Findings
- **PTD=0 $\neq$ Optimal**: Unordered embedding satisfies PTD=0 but results in a performance drop (avg. 54.67 vs M-RoPE 56.76), proving that internal image geometry is essential.
- **AGE Outperforms Single Encoding**: Full Circle-RoPE yields 58.33, while AGE reaches 58.46, suggesting geometric priors are complementary.
- **Cross-Architecture Generalization**: Hyperparameters (α=0.5, R=10) from Qwen2.5-VL transferred directly to LLaVA-0.5B without tuning, outperforming 1D-RoPE and M-RoPE.
- **Fixed Radius vs. Adaptive**: $R=10$ significantly outperforms $R=\text{auto}$, likely because adaptive radii introduce excessive variance across different image resolutions.

## Highlights & Insights
- **Driven by Geometric First Principles**: Derives cone-like geometry from the intuition of "observer-canvas orthogonality," supported by the formal PTD verification framework. This approach of designing positional encoding based on geometric invariance is transferable to other modalities like Audio-Text or 3D-Text.
- **Precision in Mixed-Angle Mapping**: The 50/50 mixture of SA (spatial structure) and GA (discriminability) balances complementary signals, a paradigm widely applicable in feature engineering.
- **AGE as a Geometric Regularizer**: Instead of forcing one geometry to serve all needs, AGE allows natural specialization across layers, similar to how different experts handle various patterns in MoE.

## Limitations & Future Work
- Validated only at 3B and 0.5B scales; performance on 7B+ models or during large-scale pre-training remains unknown.
- Improvement margins are relatively modest (+1.0 average), with some benchmarks like RealWorldQA showing only +0.44 gains.
- Adaptation to video understanding (spatio-temporal) is not yet explored. Extending Circle-RoPE to 3D + time dimensions is a valuable direction.
- Selection of $\alpha$ and $R$ relies on ablation; an adaptive learning mechanism is currently lacking.

## Related Work & Insights
- **M-RoPE (Qwen2-VL)**: Maintains 2D indices but suffers from cross-modal coupling—Circle-RoPE decouples this via orthogonal projection.
- **mPLUG-Owl3**: Achieves PTD=0 via shared indices but loses spatial info—this work proves that both PTD=0 and spatial preservation are mandatory.
- **Insight**: Positional encoding design should distinguish between "intra-modal" and "inter-modal" requirements, using distinct geometric priors for each rather than a one-size-fits-all approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](../../CVPR2026/multimodal_vlm/sope_spherical_positional_encoding_3d_lvlm.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[ICML 2026\] Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models](active_exploring_like_a_pigeon_reinforcing_spatial_reasoning_via_agentic_vision-.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](../../CVPR2026/multimodal_vlm/modix_positional_index_scaling.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing
description: >-
  [ICLR 2026][Knowledge Editing][Object Tracking] GOT-Edit integrates 3D geometric information from VGGT into a 2D generic object tracker via null-space-constrained online model editing, enhancing geometric awareness while preserving semantic discriminability, and achieving significant tracking improvements in occlusion and cluttered-background scenarios.
tags:
  - ICLR 2026
  - Knowledge Editing
  - Object Tracking
  - 3D Geometry
  - Null-Space Editing
  - Online Model Updating
  - VGGT
date: 2026-05-08
content_hash: 3ec1ac3c60ae6453
---

# GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing

**Conference**: ICLR 2026
**arXiv**: [2602.08550](https://arxiv.org/abs/2602.08550)
**Code**: [https://github.com/chenshihfang/GOT](https://github.com/chenshihfang/GOT)
**Area**: Knowledge Editing
**Keywords**: Object Tracking, 3D Geometry, Null-Space Editing, Online Model Updating, VGGT

## TL;DR
GOT-Edit integrates 3D geometric information from VGGT into a 2D generic object tracker via null-space-constrained online model editing, enhancing geometric awareness while preserving semantic discriminability, and achieving significant tracking improvements in occlusion and cluttered-background scenarios.

## Background & Motivation

**Background**: 2D generic object tracking (GOT) primarily relies on appearance features (e.g., DINOv2) and achieves strong performance in standard scenarios, but lacks 3D spatial understanding.

**Limitations of Prior Work**: Under challenging conditions such as occlusion, background clutter, and appearance variation, purely 2D features struggle to distinguish targets from distractors. Existing 3D fusion methods require RGB-D inputs or point cloud data, limiting their generality.

**Key Challenge**: Naively fusing geometric features with semantic features (e.g., concatenation or weighted summation) disrupts the learned semantic discriminability — experiments show that naive fusion degrades performance under fast motion and illumination change.

**Goal**: How can 3D geometric information be injected into a model without compromising the discriminability of its semantic features?

**Key Insight**: Drawing on knowledge editing for large language models (AlphaEdit), the geometric perturbation is projected into the null space of the semantic feature weights, ensuring no interference with existing semantics.

**Core Idea**: Project the output of the geometry-aware module into the null space of the semantic model weights, enabling lossless injection of 3D geometric knowledge.

## Method

### Overall Architecture
Given reference and current frames as RGB inputs, semantic features are extracted via DINOv2 and geometric features via VGGT. A gated fusion module combined with a null-space-projected model editor injects geometric information into the localization weights of the tracker, ultimately producing a classification map and bounding box for the target.

### Key Designs

1. **Null-Space-Constrained Knowledge Editing**:

    - Function: Constrains geometric perturbations to the null space of semantic features when updating tracker weights.
    - Mechanism: The FFN is treated as a linear associative memory $V = WK$. A semantic predictor yields weights $W_{\text{sem}}$, and a geometry predictor yields a perturbation $\Delta$. The null-space projection matrix $P_{\text{null}}$ is computed via SVD decomposition of the semantic features, and the final weight is $W_{\text{sem}} + P_{\text{null}} \cdot \Delta$.
    - Design Motivation: The null-space constraint guarantees $(W_{\text{sem}} + \Delta') \cdot K_{\text{semantic}} = W_{\text{sem}} \cdot K_{\text{semantic}}$, meaning geometric perturbations do not alter responses to semantic features and add geometric information only in the orthogonal direction. This is the key mechanism for avoiding naive-fusion degradation.

2. **Gated Feature Fusion**:

    - Function: Adaptively controls the contribution of geometric features.
    - Mechanism: A lightweight convolution followed by sigmoid produces a gating mask $m$; the fused feature is $F = v_s + m \cdot \text{Align}(v_g)$. The gate value varies across spatial positions, allowing the model to learn when and where geometric information is needed.
    - Design Motivation: Not all positions benefit from geometric information. The gating mechanism enables the model to amplify geometric cues where they help (e.g., occluded regions) and suppress them where they may be harmful (e.g., under illumination change).

3. **Dual-Stream Feature Extraction**:

    - Function: Extracts semantic and geometric features from DINOv2 and VGGT respectively.
    - Mechanism: DINOv2-L extracts semantic features; the DPT head of VGGT extracts geometric features (depth, normals, and other 3D attributes). Both backbones are frozen. An alignment layer maps geometric features to the same dimensionality as semantic features.
    - Design Motivation: As a recent visual geometry Transformer, VGGT can infer rich 3D properties from monocular RGB images, eliminating the need for RGB-D inputs.

### Loss & Training
A weighted sum of compound hinge loss (for classification) and GIoU loss (for bounding box regression).

## Key Experimental Results

### Main Results

| Dataset | Metric | GOT-Edit | ToMP-378 | PiVOT-378 | LoRAT-378 |
|--------|------|----------|----------|-----------|-----------|
| AVisT | SUC | 63.7% | 62.0% | 62.2% | 62.0% |
| NfS | SUC | 69.9% | 69.0% | 68.2% | 66.7% |
| GOT-10k | AO | 85.2% | 77.5% | 76.9% | 77.5% |
| LaSOT | SR75 | 83.2% | 75.8% | 75.5% | 78.1% |
| TrackingNet | Pr | 90.6% | 80.8% | 82.1% | 82.0% |

### Ablation Study

| Configuration | AVisT | NfS | LaSOT |
|------|-------|-----|-------|
| Baseline (semantic only) | 59.2% | 68.5% | 70.7% |
| + Geometry (naive fusion) | 59.9% | 67.5% | 70.9% |
| + Null-space projection | 61.5% | 69.3% | 72.7% |
| + Regularization (Full) | **62.0%** | **70.2%** | **73.8%** |

### Key Findings
- Naive fusion of geometric features degrades performance on NfS (69.0% → 67.5%), whereas null-space editing improves it to 70.2%.
- The most significant gain is observed under occlusion: partial occlusion yields +7.28% (64.32% → 71.60%).
- Null-space projection is the core component, contributing 2–3% absolute improvement.
- GOT-Edit outperforms state-of-the-art methods across all 8 tracking benchmarks.

## Highlights & Insights
- **Null-space editing paradigm**: Transferring the knowledge editing methodology from LLMs to visual tracking is highly innovative. The key insight is that multi-source feature fusion should operate in orthogonal subspaces rather than via naive superposition, and this principle is transferable to any multi-modal or multi-source feature fusion setting.
- **No 3D input required**: By leveraging VGGT to infer geometric information from monocular RGB, the method preserves the convenience of generic trackers that require only RGB input.
- **Adaptive gating**: The gating mechanism allows the model to automatically learn under which conditions geometric information is beneficial, avoiding hand-crafted fusion strategies.

## Limitations & Future Work
- VGGT introduces additional forward-pass overhead, which may affect real-time applicability.
- Null-space computation requires SVD decomposition, incurring extra computational cost.
- Validation is limited to the DINOv2 + VGGT combination; generalizability to other backbone combinations remains unexplored.
- The gating mask currently operates at the pixel level; coarser-grained (e.g., object-level) gating may yield more robust behavior.

## Related Work & Insights
- **vs. ToMP (De Haan et al.)**: The semantic baseline for GOT-Edit; this work extends it with geometry-awareness.
- **vs. AlphaEdit (knowledge editing)**: Originally designed for LLMs; this paper is the first to introduce null-space editing into visual tracking.
- **vs. VGGT**: The upstream model providing geometric features, demonstrating its generality for downstream tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Transferring null-space editing to visual tracking is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 tracking benchmarks + detailed ablations + attribute-level analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear method description with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides a general null-space methodology for multi-source feature fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Residual Distribution in Locate-then-Edit Model Editing](../../NeurIPS2025/knowledge_editing/rethinking_residual_distribution_in_locate-then-edit_model_editing.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ICLR 2026\] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)

</div>

<!-- RELATED:END -->

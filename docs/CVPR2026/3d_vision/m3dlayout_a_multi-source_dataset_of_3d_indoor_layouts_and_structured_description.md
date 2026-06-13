---
title: >-
  [Paper Note] M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation
description: >-
  [CVPR 2026][3D Vision][3D indoor layout] This paper constructs M3DLayout, a large-scale multi-source 3D indoor layout dataset comprising 21…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D indoor layout"
  - "dataset"
  - "text-driven scene generation"
  - "diffusion model"
  - "autoregressive model"
date: 2026-05-08
content_hash: b51bb59c19595aa0
---

# M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation

**Conference**: CVPR 2026
**arXiv**: [2509.23728](https://arxiv.org/abs/2509.23728)  
**Code**: [GitHub](https://github.com/Graphic-Kiliani/M3DLayout-code)  
**Area**: 3D Vision
**Keywords**: 3D indoor layout, dataset, text-driven scene generation, diffusion model, autoregressive model

## TL;DR

This paper constructs M3DLayout, a large-scale multi-source 3D indoor layout dataset comprising 21,367 layouts and over 433k object instances. It integrates three complementary sources—real-world scans, professionally designed scenes, and procedurally generated environments—paired with structured textual descriptions, providing a high-quality training foundation for text-driven 3D scene generation.

## Background & Motivation

In text-driven 3D scene generation, object layout serves as a critical intermediate representation bridging language instructions and geometric outputs, providing structural blueprints, enabling semantic controllability, and supporting interactive editing. However, existing datasets suffer from severe bottlenecks:

- **ScanNet, Matterport3D** (real-world scans): large geometric noise, incomplete object coverage, and lack of fine-grained annotations required for generative tasks.
- **3D-FRONT, Structured3D** (professionally designed): clean layout structures but limited object variety, with almost no small objects (3D-FRONT contains only 0.2% small objects).
- **All existing datasets**: lack scene-level textual annotations (global descriptions + large furniture relationships + small object details), making them unsuitable for conditional or multimodal generation tasks.
- Scene feasibility issues (overlaps, displacements, semantic violations) are rarely validated, resulting in noisy training data.

The core need is for a large-scale, diverse 3D layout dataset with structured textual annotations that covers both large furniture and small objects.

## Method

### Overall Architecture

The construction of M3DLayout proceeds in three stages: (1) multi-source data collection and cleaning; (2) structured textual description annotation; and (3) benchmark establishment based on diffusion and autoregressive models. The primary contribution lies at the data level rather than in model innovation.

### Key Designs

1. **Three-Source Data Fusion Strategy**:

    - **Real-world scans (Matterport3D, 1,684 scenes)**: reflect authentic cluttered spatial layouts, averaging 12.6 objects/scene with 39.4% small objects. Cleaning procedures include merging low-frequency categories and filtering scenes with fewer than 2 objects.
    - **Professionally designed scenes (3D-FRONT, 5,754 scenes)**: structurally clean and semantically well-defined, averaging 6.9 objects/scene, but with only 0.2% small objects. Uncommon configurations and unnatural proportions are filtered.
    - **Procedurally generated scenes (Inf3DLayout, 13,929 scenes)**: generated using the Infinigen generator for five common room types (bedroom/living room/dining room/kitchen/bathroom), averaging 26.8 objects/scene with 68.5% small objects. Room segmentation and outlier filtering are applied.

   **Design Motivation**: The three sources are complementary—real-world data provides layout authenticity, professional designs offer structural regularity, and procedural generation greatly enhances object diversity and fine-grained coverage.

2. **Structured Textual Annotation System**: Each layout is paired with a three-level structured description:

    - **Global scene description**: room type, style attributes, geometric characteristics, functional zones, and symmetry patterns.
    - **Large furniture description**: absolute positioning of major furniture ("bookshelf against the opposite wall") and relative spatial relationships ("coffee table beside the sofa").
    - **Small object description**: placement of decorative/functional small objects (on tabletops/shelves) and distribution patterns (uniform/symmetric).

   **Annotation pipeline**: For Matterport3D and Inf3DLayout, top-down views, side views, and close-ups of small objects are rendered and fed into GPT-4o to generate textual descriptions. For 3D-FRONT, a rule-based template approach is used due to its regular layout structure. Sampled human verification is conducted at the end.

3. **Benchmark Model Design**:

    - **Diffusion model (DIFF-M3DLayout)**: based on the DiffuScene architecture, each object is parameterized as $o_i = (c_i, x_i, y_i, z_i, w_i, h_i, d_i, \theta_i)$, with a fixed sequence length of $N=120$, a UNet denoiser, a BERT text encoder, and cross-attention injection. Training objectives: noise prediction loss $\mathcal{L}_{\text{DM}}$ + IoU collision penalty $L_{\text{IoU}}$.
    - **Autoregressive model (AR-M3DLayout)**: a Transformer encoder that takes text tokens and previously generated object embeddings as unified input, autoregressively predicting $p_\theta(x \mid c^{\text{text}}) = \prod_{i=1}^N p_\theta(o_i \mid o_{<i}, c^{\text{text}})$.

### Loss & Training

- Diffusion model: scene loss $L_{\text{sce}}$ (noise prediction error) + IoU regularization $L_{\text{IoU}}$ (penalizing object overlap).
- Autoregressive model: negative log-likelihood loss.
- Both models use BERT to encode text conditions, trained for 30k epochs with the Adam optimizer.
- Learning rate: $2 \times 10^{-4}$ for the diffusion model, $1 \times 10^{-4}$ for the autoregressive model.
- Training set: 12,062 layouts; validation set: 3,018 layouts.

## Key Experimental Results

### Main Results

| Method | FID↓ (3D-FRONT) | FID↓ (Matterport) | FID↓ (Inf3DLayout) | CLIP-Score↑ |
|------|------------------|-------------------|---------------------|-------------|
| DiffuScene | 29.47 | 98.03 | 102.12 | 0.1982 |
| InstructScene | 68.58 | 100.54 | 159.27 | 0.1944 |
| DIFF-M3DLayout (Ours) | 57.64 | **87.89** | **70.85** | 0.2001 |
| AR-M3DLayout (Ours) | 87.98 | 107.58 | **57.90** | **0.2026** |

Note: The proposed methods yield higher FID on 3D-FRONT because generated scenes typically contain more than 12 objects, whereas 3D-FRONT scenes contain only 5–12, leading to distribution mismatch.

### Ablation Study

| Training Data | FID↓ (3D-FRONT) | FID↓ (Matterport) | FID↓ (Inf3DLayout) | Notes |
|----------|------------------|-------------------|---------------------|------|
| 3D-FRONT only | 27.33 | 83.88 | 110.98 | Overfits simple layouts |
| Matterport3D only | - | - | - | Poor generalization |
| Inf3DLayout only | - | - | - | Poor generalization |
| Full M3DLayout | 57.64 | 87.89 | 70.85 | Optimal cross-source balance |

### Key Findings

- The Inf3DLayout subset is critical for generating richly detailed scenes: AR-M3DLayout achieves a 44% FID improvement on the Inf3DLayout reference set compared to InstructScene.
- CLIP-Score consistently outperforms baselines, demonstrating stronger text controllability.
- A user study (42 participants, 15 scenes) shows the most significant advantage on the Scene Richness metric.
- Models can finely control object density through text (from minimal to rich), exhibiting layout granularity controllability.

## Highlights & Insights

1. The **multi-source complementarity** paradigm is noteworthy: real-world scans (authenticity) + professional designs (structural regularity) + procedural generation (diversity) together address data bottlenecks.
2. The three-level structured textual annotation (global → large furniture → small objects) is elegantly designed, providing hierarchical semantic supervision for fine-grained text-to-layout control.
3. The Inf3DLayout subset makes a standout contribution: its 68.5% small object ratio fills the substantial gap in existing datasets regarding decorative and functional small objects.
4. The dataset scale (21k layouts, 433k objects) far exceeds the largest existing datasets, and it is the only 3D layout dataset providing structured textual descriptions.

## Limitations & Future Work

1. **Limited model innovation**: the benchmark models are directly adapted from DiffuScene without proposing new architectures tailored to multi-source data characteristics.
2. FID on the 3D-FRONT reference set is worse because generated scenes are overly complex—a contradiction between evaluation metrics and generation objectives.
3. Inf3DLayout relies on Infinigen's procedural generation, whose layout plausibility depends on hand-crafted rules and may produce unnatural arrangements.
4. Textual annotations rely on GPT-4o, which may introduce hallucinations or inaccurate spatial descriptions; human verification is conducted only through sampling.
5. Final scene quality after object retrieval and actual 3D asset placement has not been validated.
6. Extension to outdoor scenes, multi-floor environments, and dynamic layout changes remains for future work.

## Related Work & Insights

- **DiffuScene/InstructScene**: the former generates regular layouts but lacks small objects and text controllability; the latter exhibits inaccurate spatial relationship modeling. M3DLayout compensates through data quality and diversity.
- **LayoutGPT/HoloDeck**: LLM-based planning methods suffer from imprecise spatial reasoning and require large-scale physically plausible layout data as a foundation.
- **AutoPartGen/OmniPart**: inspire the generality of 3D bounding boxes as an intermediate representation connecting part generation and scene generation.
- **Insight**: In 3D generation, improvements in data quality and diversity often yield greater gains than architectural innovations.

## Rating

- Novelty: ⭐⭐⭐ The core contribution is the dataset rather than methodological innovation; the multi-source fusion and structured annotation strategy offer moderate novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparisons, ablation studies, user studies, and density controllability validation provide rich evaluation dimensions.
- Writing Quality: ⭐⭐⭐⭐ Dataset characteristics are thoroughly analyzed; tables and statistical figures clearly illustrate the complementarity of multi-source data.
- Value: ⭐⭐⭐⭐ Addresses the data bottleneck in text-driven 3D scene generation; the open-source dataset and code carry lasting impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization](customtex_high-fidelity_indoor_scene_texturing_via_multi-reference_customization.md)
- [\[CVPR 2026\] VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection](vggt-det_mining_vggt_internal_priors_for_sensor-geometry-free_multi-view_indoor_.md)
- [\[CVPR 2026\] ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph](forgedreamer_industrial_text-to-3d_generation_with_multi-expert_lora_and_cross-v.md)
- [\[CVPR 2026\] Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments](few-shot_incremental_3d_object_detection_in_dynamic_indoor_environments.md)
- [\[CVPR 2026\] Unified Primitive Proxies for Structured Shape Completion](unified_primitive_proxies_for_structured_shape_completion.md)

</div>

<!-- RELATED:END -->

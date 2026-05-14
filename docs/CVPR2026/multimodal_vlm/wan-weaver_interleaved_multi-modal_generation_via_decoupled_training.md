---
title: >-
  [Paper Note] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training
description: >-
  [CVPR 2026][Multimodal VLM][Interleaved multi-modal generation] Wan-Weaver proposes a decoupled architecture consisting of a planner (VLM) and a visualizer (DiT). By training the planner on large-scale textual-proxy data…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Interleaved multi-modal generation"
  - "decoupled training"
  - "textual-proxy data"
  - "visual consistency"
  - "planning-visualization"
date: 2026-05-08
content_hash: f714b9091e7bcfa5
---

# Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training

**Conference**: CVPR 2026
**arXiv**: [2603.25706](https://arxiv.org/abs/2603.25706)
**Code**: [https://doubiiu.github.io/projects/WanWeaver](https://doubiiu.github.io/projects/WanWeaver)
**Area**: Multimodal VLM
**Keywords**: Interleaved multi-modal generation, decoupled training, textual-proxy data, visual consistency, planning-visualization

## TL;DR

Wan-Weaver proposes a decoupled architecture consisting of a planner (VLM) and a visualizer (DiT). By training the planner on large-scale textual-proxy data instead of real interleaved data, it achieves an Overall score of 8.67 on OpenING—approaching Nano Banana's 8.85—while maintaining strong comprehension capability (MMMU 74.9) and state-of-the-art interleaved text-image generation.

## Background & Motivation

1. **Background**: Interleaved multi-modal generation requires models to produce coherent content with interleaved text and images (e.g., illustrated tutorials, story visualization) following user instructions. GPT-4o+DALL-E3 leads via a pipeline approach, while open-source alternatives (Anole, Emu3) lag significantly behind.
2. **Limitations of Prior Work**: (1) High-quality real interleaved data is extremely scarce—web-crawled image-text data suffers from low quality and copyright risks; (2) Joint training of text understanding and image generation causes mutual interference, with generation training degrading comprehension; (3) Visual consistency across long sequences is difficult to maintain—characters generated earlier tend to change appearance later.
3. **Key Challenge**: Interleaved generation simultaneously demands *planning capability* (deciding when to insert images and formulating their descriptions) and *visual consistency* (maintaining character/style coherence across multiple images), yet the two require fundamentally different training signals and data.
4. **Goal**: Independently optimize planning and visualization through decoupled training, and substitute scarce real interleaved data with synthetic textual-proxy data.
5. **Key Insight**: Decompose interleaved generation into two independently trainable sub-tasks—the planner only needs to learn "where to insert images and what to describe," which can be trained on purely textual proxy data; the visualizer only needs to learn "how to generate consistent images given a description and reference images."
6. **Core Idea**: Decoupled Training + textual-proxy data + Dense Prompt Context Window (DPCW) attention mechanism.

## Method

### Overall Architecture

User instruction → Planner (QWen2.5-VL-32B-Think) generates text with `<imagine>` tags and dense image descriptions → Visualizer (Twin DiT) generates images conditioned on dense descriptions and preceding visual references → DPCW attention enforces visual consistency → Output interleaved text-image content.

### Key Designs

1. **Decoupled Training Strategy**

    - Function: Independently optimize planning and visualization to avoid training conflicts.
    - Mechanism: Three stages—(1) freeze the planner and train only the visualizer (three consistency modes: T2I, single-image reference SI2I, multi-image reference MI2I); (2) freeze the visualizer and fine-tune only the planner (using textual-proxy data, with images replaced by dense descriptions); (3) DPCW fine-tuning to adapt the visualizer to context-window conditioning. Total training: 9.6T tokens (visualizer) + 35.72G tokens (planner).
    - Design Motivation: In joint training, visual and text losses interfere with each other—ablations show that the visual loss curve under decoupled training is smoother (decreasing from ~0.25 to 0.15), compared to oscillations observed under joint training.

2. **Textual-Proxy Data**

    - Function: Simulate interleaved data using pure text to train the planner.
    - Mechanism: Images in target interleaved data are replaced with VLM-generated dense descriptions wrapped in `<imagine>` tags. Three data sources are used: LLM-generated user query pairs, VLM-generated query pairs grounded in a database of images, and multi-image narratives (refined after SigLIP-based clustering). The generation-to-understanding data ratio is 5:1.
    - Design Motivation: High-quality real interleaved data is unavailable, whereas textual-proxy data can be generated at scale. The planner only needs to learn "when to insert images and what to describe," without ever observing actual images.

3. **Dense Prompt Context Window (DPCW)**

    - Function: Enable the visualizer to attend to visual references in context during denoising.
    - Mechanism: Self-attention windows are constructed around dense prompt positions; an attention mask strategy allows the currently generated image to attend to visual reference features from prior steps. 3D RoPE encoding is used for temporal position encoding.
    - Design Motivation: Standard diffusion generation conditions only on the current prompt and cannot leverage visual information from preceding images to maintain consistency.

### Loss & Training

Visualizer: flow-matching loss. Planner: standard autoregressive cross-entropy. The visualizer is trained progressively across three stages (T2I → +SI2I → +MI2I).

## Key Experimental Results

### Main Results

| Method | OpenING Overall ↑ | WeaverBench Overall ↑ | MMMU (Comprehension) ↑ | GenEval (T2I) ↑ | DPG (T2I) ↑ |
|---|---|---|---|---|---|
| Anole | 5.75 | 3.74 | - | - | - |
| Emu3 | 5.76 | - | - | - | - |
| Gemini+Flux | 7.23 | - | - | - | - |
| GPT-4o+DALL-E3 | 8.20 | - | - | - | - |
| Nano Banana | 8.85 | 8.38 | - | - | - |
| Bagel | - | - | 55.3 | 0.88 | 85.07 |
| **Wan-Weaver** | **8.67** | **8.43** | **74.9** | **0.89** | **87.21** |

### Ablation Study

| Configuration | Result | Note |
|---|---|---|
| Decoupled vs. joint training | Visual loss 0.15 vs. 0.25 | Decoupled training is more stable |
| Data ratio 0g1u | Token acc ~0% | Pure understanding data yields no generation capability |
| Data ratio 5g1u | Optimal | Generation-dominant with auxiliary understanding |
| T2I only | Basic text-image alignment | No reference capability |
| +SI2I | Appearance preservation | Single-image reference |
| +MI2I | Long-range visual consistency | Full capability |

### Key Findings

- Wan-Weaver retains comprehension capability close to its base model QWen2.5-VL-32B (MMMU 74.9 vs. 75.1), demonstrating that decoupled training effectively prevents generation from degrading understanding.
- An OpenING score of 8.67 approaches or surpasses GPT-4o+DALL-E3 (8.20) on certain metrics, indicating that open-source solutions are now close to the closed-source ceiling.
- Image editing performance (ImgEdit 4.31) substantially surpasses the dedicated editing model Step1X-Edit (3.06).

## Highlights & Insights

- **Elegant design of textual-proxy data**: Training the planner with dense descriptions in place of real images completely circumvents the scarcity of interleaved data—an elegant "data dimensionality reduction" strategy.
- **Engineering value of decoupled training**: The planner and visualizer can be independently iterated and upgraded without requiring re-joint training, substantially reducing system maintenance costs.
- **Unified understanding, generation, and editing**: A single model achieves state-of-the-art performance across understanding (MMMU 74.9), generation (GenEval 0.89), and editing (ImgEdit 4.31).

## Limitations & Future Work

- Users must pre-specify the resolution and aspect ratio of generated images; the model cannot adaptively determine these based on content.
- Sequential generation bottleneck—all previously generated content must be fed back into the model, causing GPU memory consumption to grow linearly with sequence length.
- Improvements in generation capability do not reciprocally benefit comprehension—bidirectional mutual enhancement remains an open problem.
- Occasional structural collapse (e.g., grid layouts appearing in place of expected individual images); geometric reasoning and symbolic grounding remain deficient.

## Related Work & Insights

- **vs. GPT-4o+DALL-E3**: The closed-source pipeline achieves OpenING 8.20; Wan-Weaver achieves 8.67—the primary advantages lie in multi-step consistency (8.56 vs. 8.38) and content completeness (9.41 vs. 8.66).
- **vs. Bagel/UniWorld**: These unified models suffer comprehension degradation from joint training (MMMU 55–59); Wan-Weaver preserves 74.9 through decoupled training.
- **vs. Emu3**: The pure discrete-token approach achieves only OpenING 5.76; the gap with Wan-Weaver originates from differences in visual quality and consistency.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Decoupled training combined with textual-proxy data represents an important paradigm innovation for interleaved generation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on OpenING + WeaverBench + unimodal benchmarks + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, though training details are dense.
- Value: ⭐⭐⭐⭐⭐ A milestone work bringing open-source interleaved generation to near closed-source performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning](narrative_weaver_towards_controllable_long-range_visual_consistency_with_multi-m.md)
- [\[NeurIPS 2025\] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification](../../NeurIPS2025/multimodal_vlm/mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions](dsert_roll_robust_multi_modal_perception_for_diverse_driving_conditions.md)

</div>

<!-- RELATED:END -->

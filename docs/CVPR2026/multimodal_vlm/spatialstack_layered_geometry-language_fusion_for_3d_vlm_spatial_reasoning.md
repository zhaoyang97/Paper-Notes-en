---
title: >-
  [Paper Note] SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning
description: >-
  [CVPR2026][Multimodal VLM][3D spatial reasoning] This paper proposes SpatialStack, a framework that injects multi-level geometric features from a multi-view geometry encoder (VGGT) into different layers of an LLM decoder (rather than fusing only the final layer), achieving open-source SOTA on multiple 3D spatial reasoning benchmarks through hierarchical alignment where shallow layers handle fine-grained spatial perception and deep layers support high-level semantic reasoning.
tags:
  - CVPR2026
  - Multimodal VLM
  - 3D spatial reasoning
  - geometry-language fusion
  - hierarchical feature fusion
  - VLM
  - VGGT
date: 2026-05-08
content_hash: 53730a6cd3f278ba
---

# SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning

**Conference**: CVPR2026
**arXiv**: [2603.27437](https://arxiv.org/abs/2603.27437)
**Code**: [https://spatial-stack.github.io/](https://spatial-stack.github.io/)
**Area**: Multimodal VLM
**Keywords**: 3D spatial reasoning, geometry-language fusion, hierarchical feature fusion, VLM, VGGT

## TL;DR
This paper proposes SpatialStack, a framework that injects multi-level geometric features from a multi-view geometry encoder (VGGT) into different layers of an LLM decoder (rather than fusing only the final layer), achieving open-source SOTA on multiple 3D spatial reasoning benchmarks through hierarchical alignment where shallow layers handle fine-grained spatial perception and deep layers support high-level semantic reasoning.

## Background & Motivation
Large vision-language models (VLMs) exhibit notable deficiencies in 3D spatial reasoning—they cannot reliably encode 3D geometric structures and spatial relationships. Existing approaches such as Spatial-MLLM, VG-LLM, and VLM-3R integrate end-to-end geometry encoders (DUST3R/VGGT, etc.) into VLMs, but **fuse only the final-layer features** of the geometry encoder with visual encoder features.

**Key Challenge**: Geometry encoders such as VGGT adopt a DPT architecture that explicitly extracts multi-level representations from different Transformer layers to recover detailed geometric information. Taking only the last layer discards rich hierarchical geometric cues from intermediate layers—shallow layers retain sharp local structures and geometric boundaries, while deep layers produce overly homogenized activations. This finding is empirically validated: injecting shallow geometric features benefits low-level perception tasks (depth estimation, distance comparison), while injecting deep features benefits high-level reasoning tasks (cross-view relational reasoning).

**Key Findings**: Naively concatenating multi-layer geometric features and injecting them into the visual pathway (naive multi-layer fusion) causes **feature interference rather than synergy**, yielding worse performance than single-layer fusion. This reveals that the true challenge lies in the **fusion strategy**, not merely in extracting multi-level features.

**Key Insight**: Relocate geometric feature fusion from the visual encoder side to the LLM decoder side, enabling progressive alignment of shallow geometry → shallow LLM layers and deep geometry → deep LLM layers.

## Method

### Overall Architecture
SpatialStack is a general hierarchical fusion framework. The core idea is to project outputs from multiple layers of the geometry encoder (VGGT) through independent projectors and inject them additively as residuals into the corresponding layers of the LLM decoder. The overall pipeline is as follows:
1. **Visual Encoder**: Processes $K$ input frames; produces visual tokens $\tilde{\mathbf{V}}$ via a spatial merger.
2. **Geometry Encoder** (VGGT, frozen): Extracts multi-level geometric features from the same set of images via a multi-view geometry Transformer.
3. **Hierarchical Fusion**: Patch tokens are extracted from VGGT layers 11/17/23, projected by independent projectors, and injected into LLM layers 0/1/2 respectively.
4. **LLM Decoder**: Processes the fused multimodal sequence and generates the answer.

### Key Designs

1. **Layered Geometry-Language Fusion**: Patch tokens $\mathbf{Z}_{l_i} \in \mathbb{R}^{(KN) \times D_{\text{geo}}}$ are extracted from VGGT layer $l_i$ ($l_i \in \{11, 17, 23\}$) and projected via layer-specific geometry token mergers:
    $\mathbf{G}_{l_i} = \mathcal{M}_{\text{geo}}^{(l_i)}(\mathbf{Z}_{l_i}), \quad \mathbf{G}_{l_i} \in \mathbb{R}^{N' \times D_{\text{lang}}}$
   The projected features are then injected into the corresponding LLM layer as additive residuals:
    $\mathbf{H}^{(j)'} = \mathbf{H}^{(j)} + \mathbf{G}_{l_j}, \quad j \in \{0, 1, 2\}$

   **Design Motivation**: Shallow geometric features preserve fine local structures and are injected into shallow LLM layers to enhance low-level perception; deep geometric features encode global semantics and are injected into deep LLM layers to support high-level reasoning. This alignment is more effective than mixing all layer features and injecting them into the visual pathway.

2. **Geometry Token Merger**: Each injection layer has an independent projector $\mathcal{M}_{\text{geo}}^{(l_i)}$ that aligns the spatial resolution and embedding dimensionality of geometric features with the LLM hidden state. Similar to the visual encoder's spatial merger, it groups each $2\times2$ neighboring patches before projection. The layer-independent design prevents interference between features at different abstraction levels.

3. **Training Strategy**: The visual encoder and VGGT geometry encoder are frozen; only the geometry token mergers and LLM decoder are trained. Standard next-token cross-entropy loss is used with no auxiliary objectives. Spatial priors emerge naturally through unified instruction fine-tuning.

### Loss & Training
- Loss: standard cross-entropy $\mathcal{L}_{\text{ce}} = -\sum_{i=1}^{|o|} \log P_\theta(o^{(i)} | o^{(<i)}, q, \mathcal{C})$
- Base models: Qwen2.5-VL / Qwen3.5; geometry encoder: VGGT
- Batch size 64, learning rate $1 \times 10^{-5}$, AdamW, warmup ratio 0.03, cosine schedule
- Training data: SPAR, LLaVA-Hound, ScanNet, VSI-590K subset

## Key Experimental Results

### Main Results (VSI-Bench)

| Method | Rank | Avg | Obj.Count | Abs.Dist | Rel.Dist | Rel.Dir | Route Plan | Appr.Order |
|--------|------|-----|-----------|----------|----------|---------|------------|------------|
| GPT-4o | - | 34.0 | 46.2 | 5.3 | 37.0 | 41.3 | 31.5 | 28.5 |
| Gemini-2.5 Pro | - | 51.5 | 43.8 | 34.9 | 61.1 | 47.8 | 45.9 | 71.3 |
| SpatialStack-4B (Qwen2.5) | 2 | 60.9 | 69.2 | 45.4 | 57.9 | 68.4 | 40.2 | 79.6 |
| **SpatialStack-5B (Qwen3.5)** | **1** | **67.5** | 71.0 | **55.6** | **67.3** | **84.1** | 41.2 | **83.5** |
| Cambrian-S-3B | 3 | 57.3 | 70.7 | 40.6 | 64.8 | 61.9 | 27.3 | 78.8 |
| VG-LLM-4B | 5 | 47.3 | 66.0 | 37.8 | 44.6 | 45.6 | 33.5 | 36.4 |

### Cross-Benchmark Comparison

| Method | VSI-Bench | SPAR-Bench | BLINK-Spatial | CV-Bench | Overall |
|--------|-----------|------------|---------------|----------|---------|
| Qwen3.5 (fine-tuned) | 64.76 | 68.75 | **56.10** | 84.49 | 68.52 |
| GVF-L23 (VG-LLM) | 66.36 | 70.83 | 51.91 | 84.64 | 68.43 |
| GVF-L11/17/23 (naive multi) | 65.15 | 71.20 | 51.28 | 84.33 | 67.99 |
| **SpatialStack** | **67.52** | **71.39** | 52.12 | **85.53** | **69.14** |

### Ablation Study

| Configuration | Low-Level Task Avg | High-Level Task Avg | Notes |
|---------------|--------------------|---------------------|-------|
| Single-layer injection L11 | **66.11** | 64.48 | Shallow layer most beneficial for low-level perception |
| Single-layer injection L23 | 64.33 | **66.36** | Deep layer most beneficial for high-level reasoning |
| Naive multi-layer fusion (visual side) | 64.69 | 65.15 | Feature interference; suboptimal for both |
| SpatialStack (LLM-side hierarchical fusion) | 65.89* | 67.52 | Balances both task types |

### Fusion Order Ablation

| Method | VSI-Bench | SPAR-Bench | BLINK-Spatial | CV-Bench | Overall |
|--------|-----------|------------|---------------|----------|---------|
| SpatialStack (forward order) | **67.52** | 71.39 | 52.12 | **85.53** | **69.14** |
| SpatialStack (reverse order) | 67.22 | **71.97** | 50.08 | 84.82 | 68.52 |
| Vision Fusion | 64.27 | 69.68 | **56.45** | 83.11 | 68.38 |

### Key Findings
- **Hierarchical correspondence**: Shallow VGGT layers capture fine-grained local geometry; deep layers encode global semantic structure—naturally corresponding to the hierarchical functional organization of LLM decoder layers.
- **Naive multi-layer fusion fails**: Mixing multi-layer geometric features and injecting them into the visual pathway causes feature interference, underperforming single-layer fusion (the core motivation of this paper).
- **SpatialStack substantially outperforms methods with the same base model**: SpatialStack-4B (60.9) vs. VG-LLM-4B (47.3) vs. Cambrian-S-3B (57.3) on Qwen2.5.
- Fusion order matters: forward order (shallow-to-shallow) outperforms reverse order, validating the hierarchical alignment hypothesis.
- General capabilities are preserved: performance on MMBench and Video-MME is on par with the base model, with no catastrophic forgetting.
- **Zero-shot generalization on Route Planning**: No route planning data is included in training, yet SpatialStack-5B achieves 84.1 on this task (surpassing all open-source models), demonstrating strong zero-shot transfer.

## Highlights & Insights
- **"Where to fuse" matters more than "what to fuse"**: The paper systematically demonstrates the necessity of relocating geometric feature fusion from the visual encoder to the LLM decoder side—a finding with broad implications for multimodal architecture design.
- **Empirical analysis of hierarchical correspondence**: Through both qualitative (similarity heatmaps) and quantitative (low-/high-level task performance) analyses, the paper establishes the optimal correspondence between geometry encoder layers and LLM decoder layers.
- **Simplicity of additive residual injection**: The fusion operation is simply $H' = H + G$, requiring no cross-attention or gating mechanisms—highly elegant and effective.
- **Model-agnostic framework**: SpatialStack is applicable to arbitrary open-source VLMs, as validated on both Qwen2.5 and Qwen3.5.
- **Inspiration from DeepStack**: The idea of stacking visual tokens across LLM layers in DeepStack is elegantly transferred to geometric tokens—a natural and principled cross-pollination.

## Limitations & Future Work
- SpatialStack underperforms the base model Qwen3.5 on BLINK-Spatial (52.12 vs. 56.10), suggesting that geometric injection may introduce interference in certain fine-grained visual perception tasks.
- Only three VGGT layers (11/17/23) are selected; finer-grained layer selection strategies (e.g., learnable gating) remain unexplored.
- Compatibility with other geometry encoders such as DUST3R and CUT3R has not been tested.
- Additive residual fusion may not be optimal—adaptive weighting or cross-attention fusion could yield further improvements.
- Training data is predominantly from indoor scenes; generalization to outdoor and dynamic scenes remains to be verified.

## Related Work & Insights
- **vs. VG-LLM**: VG-LLM fuses only the final VGGT layer into the visual side; SpatialStack's hierarchical LLM-side fusion improves VSI-Bench from 47.3 to 60.9 under the same base model.
- **vs. Cambrian-S**: Cambrian-S introduces additional self-supervised spatial learning; SpatialStack surpasses it through architectural innovation alone without extra training paradigms (60.9 vs. 57.3).
- **vs. DeepStack**: SpatialStack extends DeepStack's visual token stacking idea to geometric tokens, representing a natural generalization.
- **vs. Spatial-MLLM**: Employs a dual-encoder architecture but performs only single-layer fusion; SpatialStack's hierarchical fusion yields substantial gains.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Layered geometry-language fusion is a first; the core insight (where to fuse matters more than what to fuse) is substantive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four spatial reasoning benchmarks + general capability evaluation + multi-dimensional ablations (layer selection, fusion order, visual vs. language side fusion).
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from qualitative analysis → quantitative validation → method design → experimental verification is complete and coherent.
- **Value**: ⭐⭐⭐⭐ Establishes a new paradigm for visual-language-geometry fusion with important reference value for 3D spatial reasoning and broader modality fusion research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](../../NeurIPS2025/multimodal_vlm/learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_visionlanguage_models_via.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](../../NeurIPS2025/multimodal_vlm/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps
description: >-
  [NeurIPS 2025][Object Detection][Layout-to-Image] OverLayBench introduces the first Layout-to-Image benchmark focused on dense overlap scenarios (4,052 samples + OverLayScore difficulty metric), revealing that SOTA methods suffer severe degradation in mIoU from 71% to 54% under complex overlaps, and proposes Amodal Mask supervision that achieves a 15.9% improvement in overlap IoU.
tags:
  - NeurIPS 2025
  - Object Detection
  - Layout-to-Image
  - Overlapping Layouts
  - Amodal Mask
  - Diffusion Models
  - Benchmark Evaluation
date: 2026-05-08
content_hash: c488fdde3b5debc4
---

# OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps

**Conference**: NeurIPS 2025  
**arXiv**: [2509.19282](https://arxiv.org/abs/2509.19282)  
**Code**: [https://mlpc-ucsd.github.io/OverLayBench](https://mlpc-ucsd.github.io/OverLayBench)  
**Area**: Object Detection  
**Keywords**: Layout-to-Image, Overlapping Layouts, Amodal Mask, Diffusion Models, Benchmark Evaluation

## TL;DR
OverLayBench introduces the first Layout-to-Image benchmark focused on dense overlap scenarios (4,052 samples + OverLayScore difficulty metric), revealing that SOTA methods suffer severe degradation in mIoU from 71% to 54% under complex overlaps, and proposes Amodal Mask supervision that achieves a 15.9% improvement in overlap IoU.

## Background & Motivation

### State of the Field

**Background**: L2I methods (GLIGEN, InstanceDiffusion, CreatiLayout) perform well on simple layouts, but over 80% of existing benchmarks focus on low-overlap scenarios.

**Limitations of Prior Work**: When multiple objects overlap extensively with similar semantics (e.g., two cats of the same color), models tend to merge or lose instances, yet systematic evaluation of this phenomenon is lacking.

**Key Challenge**: Dense overlap is commonplace in real-world scenes, yet existing methods and benchmarks consistently avoid this challenging setting.

**Goal**: Quantify overlap difficulty + construct a stratified evaluation benchmark + explore methods to improve generation under overlapping conditions.

**Key Insight**: $\text{OverLayScore} = \sum \text{IoU}(B_i,B_j) \cdot \cos(p_i,p_j)$ (spatial overlap × semantic similarity), using Amodal Masks to provide complete object contour supervision.

**Core Idea**: OverLayScore quantifies overlap difficulty + stratified benchmark + Amodal Mask supervision improves generation quality under overlapping conditions.

## Method

### Overall Architecture
4,052 samples (2,052 easy + 1,000 normal + 1,000 complex), with instance descriptions and relationships annotated by Qwen2.5-VL-32B. CreatiLayout-AM extends CreatiLayout with amodal mask token/pixel-level loss.

### Key Designs

1. **OverLayScore Metric**: $\text{Score} = \sum \text{IoU}(B_i,B_j) \cdot \cos(p_i,p_j)$ — spatial overlap × semantic similarity
2. **Amodal Mask Supervision**: $\mathcal{L} = \mathcal{L}_{LDM} + \lambda\mathcal{L}_{token} + \beta\mathcal{L}_{pixel}$ — supervision using complete object masks (including occluded regions)
3. **O-mIoU Metric**: IoU computed exclusively within overlapping regions for more precise evaluation of occlusion handling

### Loss & Training
- Built on FLUX DiT architecture with additional amodal mask annotations

## Key Experimental Results

### Main Results

| Method | Easy mIoU | Complex mIoU | O-mIoU |
|--------|-----------|--------------|--------|
| CreatiLayout-FLUX | **71.17%** | 54% | 49.80% |
| **CreatiLayout-AM** | — | — | **65.70%** (+15.9%) |

### Key Findings
- All methods suffer severe degradation under complex overlaps (71% → 54%)
- Amodal mask supervision significantly improves generation in overlapping regions (+15.9% O-mIoU)
- DiT-based models consistently outperform U-Net-based models

## Highlights & Insights
- **OverLayScore quantifies generation difficulty**: The product of spatial and semantic terms captures the essence of "hard overlaps"
- **Amodal Mask is a natural solution**: It informs the model that occluded parts should also be present

## Limitations & Future Work
- Overlap evaluation is limited to bounding-box level
- Dataset scale is relatively small (4,052 samples)
- Improvements on complex overlaps remain limited

## Related Work & Insights
- **vs GLIGEN**: Fixed attention masks fail in overlapping scenarios
- **vs InstanceDiffusion**: Supports instance-controlled generation but lacks overlap evaluation

## Rating
- Novelty: ⭐⭐⭐⭐ First L2I benchmark focused on overlaps
- Experimental Thoroughness: ⭐⭐⭐⭐ 8+ methods evaluated + amodal ablation
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear
- Value: ⭐⭐⭐⭐ Exposes a critical weakness in L2I generation
- Novelty: ⭐⭐⭐⭐ New problem + new metric + new benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model comparison
- Writing Quality: ⭐⭐⭐⭐ Clear
- Value: ⭐⭐⭐⭐ Important complement to L2I evaluation

### In-Depth Analysis
- Dense overlap represents a systematic blind spot in existing layout generation methods — OverLayScore fills the evaluation gap
- OverLayScore correlates strongly with human perception; CreatiLayout-AM significantly outperforms the baseline in dense overlap scenarios
- The core innovation lies in the simplicity and effectiveness of the design approach
- Experimental results thoroughly validate the central hypothesis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention](instanceassemble_layoutaware_image_generation_via_instance_a.md)
- [\[NeurIPS 2025\] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)
- [\[ICCV 2025\] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation](../../ICCV2025/object_detection/yolo-count_differentiable_object_counting_for_text-to-image_generation.md)
- [\[AAAI 2026\] SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation](../../AAAI2026/object_detection/saga_learning_signal-aligned_distributions_for_improved_text-to-image_generation.md)
- [\[NeurIPS 2025\] Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy](lr_yolo_lipschitz_continuity_image_restoration_object_detection.md)

</div>

<!-- RELATED:END -->

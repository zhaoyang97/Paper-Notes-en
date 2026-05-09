---
title: >-
  [Paper Note] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime
description: >-
  [CVPR 2026][Segmentation][Scene graph generation] This paper proposes DSFlash, a low-latency panoptic scene graph generation model that achieves real-time inference at 56 FPS on an RTX 3090 while maintaining state-of-the-art performance (mR@50=30.9), through a unified backbone, bidirectional relation prediction, and mask-guided dynamic pruning.
tags:
  - CVPR 2026
  - Segmentation
  - Scene graph generation
  - panoptic segmentation
  - real-time inference
  - bidirectional relation prediction
  - dynamic pruning
date: 2026-05-08
content_hash: a7e78c68ee121420
---

# DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

**Conference**: CVPR 2026
**arXiv**: [2603.10538](https://arxiv.org/abs/2603.10538)
**Code**: To be released upon acceptance
**Area**: Segmentation
**Keywords**: Scene graph generation, panoptic segmentation, real-time inference, bidirectional relation prediction, dynamic pruning

## TL;DR

This paper proposes DSFlash, a low-latency panoptic scene graph generation model that achieves real-time inference at 56 FPS on an RTX 3090 while maintaining state-of-the-art performance (mR@50=30.9), through a unified backbone, bidirectional relation prediction, and mask-guided dynamic pruning.

## Background & Motivation

Scene graph generation (SGG) aims to extract structured node-edge graph representations from images, where nodes represent instances and edges represent relations (e.g., "person sitting on chair"), providing value for downstream tasks such as VQA, image captioning, and embodied reasoning. Panoptic scene graph generation (PSGG) further employs segmentation masks instead of bounding boxes for instance localization.

**Key Challenge**: Existing PSGG methods prioritize performance at the expense of efficiency. DSFormer achieves state-of-the-art mR@50=30.7 but incurs an inference latency of 458ms; while REACT achieves 19ms, its PSGG performance remains limited. More critically, existing methods typically predict only a subset of relations rather than the complete scene graph.

**Key Insight**: DSFlash builds upon DSFormer and systematically replaces its inefficient components—merging two backbones into one, designing a bidirectional prediction head to halve inference passes, and leveraging mask information for dynamic token pruning—achieving scene graph generation that is simultaneously fast, comprehensive, and accurate.

## Method

### Overall Architecture

DSFlash adopts a two-stage architecture. The first stage employs a frozen EoMT backbone for panoptic segmentation and feature extraction. The second stage appends mask embeddings to each pair of mask combinations, processes them through a transformer neck, and outputs bidirectional relations via a relation prediction head. Ground-truth masks are used during training, while predicted masks are used during inference.

### Key Designs

1. **Merged Backbones**: DSFormer employs two separate backbones (one for segmentation, one for relation prediction). DSFlash directly extracts intermediate feature tensors from the segmentation model EoMT (blocks 2/5/8/11) and concatenates them into a $768 \times 40 \times 40$ feature map, eliminating the overhead of a second backbone. EoMT remains frozen throughout; only the relation prediction components are updated during training, substantially reducing training cost (under 24 hours on a single GTX 1080). EoMT is preferred over MaskDINO due to its encoder-only design, which is faster and easier to integrate.

2. **Bidirectional Predictions**: DSFormer requires two forward passes for each mask pair $(S_0, S_1)$ (forward/reverse directions). DSFlash introduces a gating mechanism that simultaneously outputs relations in both directions within a single forward pass:

    - Compute $g = \sigma(\text{gate}_{mlp}(x))$
    - Forward features $t^{\rightarrow} = g \odot x$, reverse features $t^{\leftarrow} = (1-g) \odot x$
    - A shared MLP predicts $z^{\rightarrow}$ and $z^{\leftarrow}$ respectively

   To prevent the model from exploiting label distribution imbalances between forward and reverse directions as a shortcut, mask order is swapped during training for a second forward pass, and a feature consistency loss is applied: $\text{Consistency} = \frac{1}{D}\sum_{i}(t_i^{\rightarrow} - t_i^{\prime\leftarrow})^2 + (t_i^{\leftarrow} - t_i^{\prime\rightarrow})^2$. Only a single forward pass is required at inference time.

3. **Dynamic Mask Patch Pruning**: The overlap ratio information already computed during mask embedding is leveraged to identify and discard patch tokens that have no overlap with either subject or object masks. Such patches, being distant from the subject-object pair, contribute minimally to relation classification, and pruning incurs negligible additional computational cost. Since the final prediction relies solely on the CLS token, the model naturally supports variable token counts.

4. **Token Merging**: ToMe-SD is applied before each attention layer in the backbone to merge similar tokens, followed by unmerging after attention, reducing attention computation while preserving segmentation capability.

### Loss & Training

- Relation prediction: BCE loss
- Feature consistency: MSE loss
- Data augmentation: DeiT III style (random flipping, color jitter, one of grayscale/exposure/Gaussian blur)
- Ground-truth masks used during training; backbone frozen throughout

## Key Experimental Results

### Main Results

| Method | mR@50 ↑ | Latency (ms) ↓ | Params |
|--------|---------|----------------|--------|
| DSFormer | 30.70 | 458 | 330M |
| HiLo-L | 19.08 | 427 | 230M |
| REACT | 19.00 | 19 | 43M |
| **DSFlash-L** | **30.90** | **50** | 340M |
| DSFlash-B* | 28.50 | 23 | 116M |
| DSFlash-S* | 25.05 | **18** | **40M** |

### Ablation Study (Incremental Optimization)

| Optimization | mR@50 | Latency (ms) | RPS ↑ | Notes |
|--------------|-------|--------------|-------|-------|
| Baseline (DSFormer) | 30.7 | 445 | 435 | Starting point |
| + Merged Backbones | 25.0 | 41 (-91%) | 5,745 | Largest source of speedup |
| + Efficient Mask Encoding | 25.0 | 37 (-10%) | 7,132 | Reduced data copying |
| + Gated Bidirectional Prediction | 28.8 | 29 (-22%) | 11,491 | Halved inference passes + improved performance via additional supervision |
| + Skip Segmentation Upsampling | 28.5 | 23 (-21%) | 12,928 | No upsampling to original resolution required |

### Key Findings

- Segmentation model quality directly determines scene graph performance (mR@50 and mR@inf are strongly correlated); future improvements to segmentation models will directly benefit DSFlash.
- Bidirectional prediction not only halves inference passes but also improves mR@50 (25.0→28.8) due to the additional supervision signal.
- DSFlash-S, with only 40M parameters and 18ms latency, still outperforms all methods except DSFormer.

## Highlights & Insights

- **Engineering meets design**: Each optimization is motivated by clear rationale and quantitative analysis, rather than a superficial stacking of techniques.
- **Complete scene graph**: Unlike REACT, which predicts only partial relations, DSFlash predicts relations for all mask pairs and achieves this more efficiently.
- **Insight behind consistency loss**: The authors identify that skewed forward/reverse label distributions in the training set induce shortcut learning, which is elegantly addressed through consistency regularization.
- **Deployability**: Training completes within 24 hours on a single GTX 1080; inference runs at 56 FPS on an RTX 3090.

## Limitations & Future Work

- mR@50 is highly dependent on segmentation quality; poor segmentation on certain categories directly degrades scene graph quality.
- The PSG dataset covers only 56 predicate categories, which falls short of the relational richness in real-world scenes.
- Downstream task validation (e.g., VQA, embodied reasoning) has not been explored.
- Freezing the backbone may limit adaptation to specific datasets.

## Related Work & Insights

- EoMT's encoder-only design is a key infrastructure choice enabling low latency.
- The gating mechanism for bidirectional prediction draws inspiration from GRU-style gating, yielding a concise and effective solution.
- DSFormer's mask embedding module is well-designed and directly reused.
- Token Merging (ToMe) reduces computation while preserving representational quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The bidirectional gated prediction and dynamic mask pruning are novel contributions; the overall work represents a systematic engineering optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablation studies and latency analyses clearly attribute the contribution of each component.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with sufficiently motivated design choices.
- **Value**: ⭐⭐⭐⭐⭐ Significant practical impact for real-time scene graph generation, lowering the hardware barrier for SGG research.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)
- [\[CVPR 2026\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)
- [\[CVPR 2026\] GenMask: Adapting DiT for Segmentation via Direct Mask Generation](genmask_adapting_dit_for_segmentation_via_direct_mask_generation.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model
description: >-
  [CVPR2026][Segmentation][encoder-only] This paper proposes VidEoMT, an encoder-only video segmentation architecture that unifies segmentation and temporal association within a single ViT encoder via query propagation and…
tags:
  - "CVPR2026"
  - "Segmentation"
  - "encoder-only"
  - "ViT"
  - "video instance segmentation"
  - "video panoptic segmentation"
  - "query propagation"
  - "query fusion"
  - "DINOv2"
date: 2026-05-08
content_hash: fd570e19224a6ddf
---

# VidEoMT: Your ViT is Secretly Also a Video Segmentation Model

**Conference**: CVPR2026
**arXiv**: [2602.17807](https://arxiv.org/abs/2602.17807)  
**Code**: [tue-mps.org/videomt](https://www.tue-mps.org/videomt/)  
**Area**: Video Segmentation
**Keywords**: encoder-only, ViT, video instance segmentation, video panoptic segmentation, query propagation, query fusion, DINOv2

## TL;DR

This paper proposes VidEoMT, an encoder-only video segmentation architecture that unifies segmentation and temporal association within a single ViT encoder via query propagation and query fusion, achieving 5×–10× speedup (160 FPS with ViT-L) while maintaining accuracy comparable to state-of-the-art methods.

## Background & Motivation

1. **Excessive complexity of existing methods**: Current state-of-the-art online video segmentation models (e.g., CAVIS, DVIS++, DVIS-DAQ) decouple the pipeline into a segmenter and a tracker, each containing numerous specialized components (ViT-Adapter, Mask2Former pixel decoder, Transformer decoder, context-aware features, re-identification layers, etc.), resulting in bloated architectures and slow inference.
2. **Underutilized potential of VFM pretraining**: Vision foundation models such as DINOv2 have learned view-consistent feature representations through large-scale pretraining, which theoretically suffice for both instance-level segmentation and temporal tracking; yet existing methods still stack substantial redundant components on top.
3. **EoMT validates encoder-only feasibility for image segmentation**: EoMT demonstrates that injecting learnable queries into the last few layers of a pretrained ViT achieves state-of-the-art image segmentation without a dedicated decoder, directly inspiring simplification in the video domain.
4. **DINO-style pretraining facilitates tracking**: The DINO/DINOv2 training objective encourages consistent features for the same object across different viewpoints—precisely the capability required for tracking—making ViT encoders naturally suited to video segmentation.
5. **Inference speed is critical for video applications**: Online video processing demands real-time or faster inference, whereas existing state-of-the-art methods (e.g., CAVIS at only 15 FPS) fall far short of practical deployment requirements.
6. **Core research question**: Can all dedicated tracking modules be removed, allowing a sufficiently large pretrained ViT encoder to simultaneously perform segmentation and temporal association?

## Method

### Overall Architecture

VidEoMT builds upon EoMT's encoder-only paradigm, directly injecting learnable queries into the last $L_2$ layers of the ViT encoder for joint processing with patch tokens. The key innovation is the introduction of two lightweight mechanisms—**query propagation** and **query fusion**—that enable temporal modeling within the encoder itself, eliminating the need for a separate tracker.

### Progressive Simplification from CAVIS to VidEoMT

1. **Replace the segmenter**: Substituting EoMT for ViT-Adapter + Mask2Former → FPS increases from 15 to 42, AP drops only 0.8.
2. **Remove context-aware features**: Removing Laplacian edge extraction and average-pooling → FPS increases to 72, AP improves slightly.
3. **Remove re-identification layers**: Removing the contrastive MLP → FPS increases to 74, AP largely unchanged.
4. **Remove the tracker**: Degenerates to per-frame EoMT → FPS jumps to 162 but AP drops sharply by 7.6.

### Query Propagation

- **First frame $t=0$**: Standard EoMT procedure; learnable queries $\mathbf{Q}^{\text{lrn}}$ are injected into the last $L_2$ layers of the ViT, producing object queries $\mathbf{Q}_0^{\mathcal{S}}$ and segmentation predictions.
- **Subsequent frames $t>0$**: The output queries $\mathbf{Q}_{t-1}^{\mathcal{S}}$ from the previous frame replace the learnable queries as input, with no additional computational overhead.
- **Limitation**: As the number of frames increases, the influence of the learnable queries diminishes, gradually impairing the model's ability to detect newly appearing objects.

### Query Fusion

A lightweight strategy to address the above limitation:

$$\mathbf{Q}_t^{\mathcal{F}} = \text{Linear}(\mathbf{Q}_{t-1}^{\mathcal{S}}) + \mathbf{Q}^{\text{lrn}}$$

- The previous frame's queries are linearly transformed and element-wise added to the learnable queries.
- This balances temporal continuity (from propagated queries) with the capacity to perceive new objects (from learnable queries).
- A temporally consistent supervision strategy ensures query ordering is consistent across frames, making the element-wise addition well-defined.

### Loss & Training

- Same loss functions as Mask2Former: cross-entropy for classification, BCE + Dice loss for segmentation.
- Temporally consistent GT matching strategy (from DVIS++): Hungarian matching is applied only at the first appearance of each object; query correspondences are maintained in subsequent frames.
- AdamW optimizer, lr=1e-4, layer-wise learning rate decay (LLRD) factor 0.6, polynomial lr decay (power=0.9).

## Key Experimental Results

### Main Results: YouTube-VIS 2019 val (VIS)

| Method | Backbone | AP | GFLOPs | FPS |
|--------|----------|----|--------|-----|
| CAVIS | ViT-L/DINOv2 | **68.9** | 838 | 15 |
| DVIS-DAQ | ViT-L/DINOv2 | 68.3 | 851 | 10 |
| DVIS++ | ViT-L/DINOv2 | 67.7 | 846 | 18 |
| **VidEoMT** | ViT-L/DINOv2 | 68.6 | **566** | **160** |

### Cross-Task Generalization

| Task/Dataset | VidEoMT Metric | CAVIS Metric | VidEoMT FPS | CAVIS FPS |
|--------------|---------------|-------------|-------------|-----------|
| VPS / VIPSeg | VPQ=55.2 | VPQ=56.9 | **75** | 10 |
| VSS / VSPW | mIoU=**64.9** | — | **73** | — |
| VIS / OVIS | AP=52.5 | AP=**53.2** | **115** | 15 |
| VIS / YT-VIS 2022 | AP=**42.6** | AP=39.5 | **161** | 15 |

### Ablation Study: Progressive Module Removal

| Step | Change | AP | FPS |
|------|--------|----|-----|
| (0) CAVIS baseline | — | 68.9 | 15 |
| (1) Replace segmenter with EoMT | ↓0.8 | 42 |
| (2) Remove context-aware features | 68.4 | 72 |
| (3) Remove re-identification layers | 68.0 | 74 |
| (4) Remove tracker = EoMT | 61.3 | 162 |
| (5) + Query Propagation | 63.9 | 162 |
| (6) + Query Fusion = VidEoMT | **68.6** | **160** |

### Effect of Pretraining and Model Scale

- **Pretraining**: Under DINOv2/DINOv3, the gap between VidEoMT and CAVIS is only 0.3 AP; under IN1K pretraining, the gap widens to 2.7 AP → large-scale pretraining is essential.
- **Model scale**: The gap is 0.3 AP at ViT-L, 1.3 AP at ViT-B, and 2.7 AP at ViT-S; nevertheless, VidEoMT with ViT-L (160 FPS) is 8× faster than CAVIS with ViT-S (19 FPS) while achieving 13+ AP higher.

## Highlights & Insights

1. **Radical simplicity**: The approach reduces video segmentation from a complex multi-module segmenter+tracker pipeline to a single ViT encoder with lightweight query fusion, adding only 2M parameters.
2. **Order-of-magnitude speedup**: ViT-L achieves 160 FPS, more than 10× faster than CAVIS, thanks to the pure-Transformer design that fully exploits hardware/software optimizations such as FlashAttention and `torch.compile`.
3. **Progressive hypothesis validation**: Six-step ablation clearly demonstrates the redundancy of each specialized component, producing a convincing experimental narrative.
4. **Cross-task generality**: Strong performance across all three tasks (VIS, VPS, VSS) and six benchmarks, with state-of-the-art results on VSPW for video semantic segmentation.

## Limitations & Future Work

1. **Dependence on large-scale pretraining**: Accuracy degrades noticeably with smaller-scale pretraining such as IN1K (2.7 AP gap vs. CAVIS), indicating a strong reliance on vision foundation models.
2. **Performance degradation on smaller models**: The gap reaches 2.7 AP at ViT-S, suggesting that the encoder-only paradigm offers diminishing advantages at smaller model scales.
3. **Gap on VIPSeg**: VPQ trails CAVIS by 1.7 and DVIS-DAQ by 2.2 on the VPS task, indicating room for improvement in tracking under panoptic settings.
4. **Challenging OVIS scenarios**: The method lags DVIS-DAQ by 1.8 AP on the heavily occluded OVIS dataset, suggesting that pure query propagation may be insufficient under extreme occlusion.
5. **Online-only mode**: The paper does not explore offline or semi-online modes that could exploit future frame information.
6. **Single-frame history in query fusion**: Only the previous frame's queries are propagated; multi-frame aggregation or memory mechanisms are not explored.

## Related Work & Insights

- **EoMT**: The direct predecessor of VidEoMT, supporting image segmentation only; VidEoMT extends it to the video domain via query propagation and fusion, recovering AP from 61.3 to 68.6.
- **CAVIS**: Current state-of-the-art for VIS, containing ViT-Adapter, Mask2Former decoder, context-aware features, re-identification layers, and a Transformer tracker; VidEoMT removes all these components while maintaining comparable accuracy.
- **DVIS / DVIS++ / DVIS-DAQ**: All adopt a decoupled segmentation+tracking paradigm; VidEoMT achieves comparable or superior accuracy on most benchmarks at 5×–14× higher speed.
- **MinVIS**: Also pursues simplicity and efficiency, but still employs Swin-L + Mask2Former decoder; VidEoMT is simpler, faster, and more accurate.
- **TrackFormer**: Uses query propagation for detection and tracking; VidEoMT transfers this idea to an encoder-only segmentation framework and further improves it with query fusion.

## Rating

- Novelty: ⭐⭐⭐⭐ — The encoder-only video segmentation idea is novel; the progressive module-removal validation strategy is highly convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Six benchmarks, three tasks, detailed ablations, and analysis of pretraining and model scale; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logical flow with a well-structured hypothesis–validation–conclusion narrative and excellent figures.
- Value: ⭐⭐⭐⭐ — The practical significance of 10× speedup is substantial, providing a viable solution for real-time video segmentation deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TRACE: Your Diffusion Model is Secretly an Instance Edge Detector](../../ICLR2026/segmentation/trace_your_diffusion_model_is_secretly_an_instance_edge_detector.md)
- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[CVPR 2026\] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation](rs-ssm_refining_forgotten_specifics_in_state_space_model_for_video_semantic_segm.md)
- [\[CVPR 2026\] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)

</div>

<!-- RELATED:END -->

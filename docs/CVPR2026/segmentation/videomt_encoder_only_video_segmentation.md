---
title: >-
  [Paper Note] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model
description: >-
  [CVPR 2026][Segmentation][video segmentation] This paper proposes VidEoMT, an encoder-only video segmentation model that unifies segmentation and temporal association within a single ViT encoder via query propagation and query fusion, eliminating all dedicated tracking modules. It achieves 160 FPS on YouTube-VIS 2019 (10×+ faster than CAVIS) with only a 0.3 AP drop.
tags:
  - CVPR 2026
  - Segmentation
  - video segmentation
  - encoder-only
  - query propagation
  - ViT
  - DINOv2
  - real-time
date: 2026-05-08
content_hash: ac194732e3234f0a
---

# VidEoMT: Your ViT is Secretly Also a Video Segmentation Model

**Conference**: CVPR 2026
**arXiv**: [2602.17807](https://arxiv.org/abs/2602.17807)
**Code**: [https://www.tue-mps.org/videomt/](https://www.tue-mps.org/videomt/)
**Area**: Semantic Segmentation / Video Understanding / Efficient Models
**Keywords**: video segmentation, encoder-only, query propagation, ViT, DINOv2, real-time

## TL;DR
This paper proposes VidEoMT, an encoder-only video segmentation model that unifies segmentation and temporal association within a single ViT encoder via query propagation and query fusion, eliminating all dedicated tracking modules. It achieves 160 FPS on YouTube-VIS 2019 (10×+ faster than CAVIS) with only a 0.3 AP drop.

## Background & Motivation

### Limitations of Prior Work

**Background**: Existing online video segmentation methods (CAVIS, DVIS++, DVIS-DAQ) follow a decoupled "segmenter + tracker" paradigm: the segmenter consists of ViT + ViT-Adapter + Mask2Former pixel decoder + Transformer decoder, while the tracker comprises context-aware features + re-identification layers + Transformer tracking blocks. Although accurate, this architecture is extremely complex and slow (CAVIS achieves only 15 FPS). The EoMT paper demonstrated that image segmentation can be performed in an encoder-only fashion (without decoder or pixel decoder). The question is whether video segmentation can follow suit — with temporal tracking being the key additional challenge.

### Root Cause

**Goal**: Can a single, simple encoder-only ViT simultaneously perform video segmentation and temporal association, achieving near-SOTA accuracy at an order-of-magnitude speed improvement?

## Method

### Overall Architecture
VidEoMT builds on EoMT: $N$ learnable queries are injected into the last $L_2$ layers of a DINOv2 ViT and processed jointly with patch tokens, with query outputs directly predicting categories and masks. On top of this, VidEoMT introduces two lightweight mechanisms for temporal modeling: (1) **Query Propagation** — the output queries from the previous frame serve as input queries for the current frame; (2) **Query Fusion** — the previous-frame queries are linearly transformed and element-wise added to the learnable queries. No tracking modules are used.

### Key Designs
1. **Progressive Module Removal Study**: Starting from CAVIS, each component is incrementally validated: replacing the segmenter with EoMT (AP −0.8, speed 3×↑) → removing context-aware features (AP +0.3, speed 1.7×↑ to 72 FPS) → removing re-identification layers (AP −0.4, speed↑ to 74 FPS) → removing the tracker (AP −7.6, speed↑ to 162 FPS). Key finding: context-aware features and re-identification layers are redundant under DINOv2 pretraining — DINOv2 features already contain sufficient instance-discriminative information.

2. **Query Propagation**: At $t=0$, learnable queries $\mathbf{Q}^{lrn}$ are used for initialization; at $t>0$, the output queries from the previous frame $\mathbf{Q}_{t-1}^S$ replace them and are injected into the last $L_2$ layers of the ViT. This enables cross-frame information transfer with zero additional computation. However, pure propagation gradually diminishes the influence of learnable queries, compromising the detection of newly appearing objects.

3. **Query Fusion**: $\mathbf{Q}_t^F = \text{Linear}(\mathbf{Q}_{t-1}^S) + \mathbf{Q}^{lrn}$, where previous-frame queries undergo a single linear transformation and are element-wise added to the learnable queries. This ensures the model retains both temporal context from the previous frame and the capacity to detect new objects. Only one linear layer is introduced, with negligible overhead.

### Loss & Training
Standard Mask2Former losses are used (CE for classification + BCE/Dice for segmentation). Training follows a two-stage procedure: Stage 1 trains on COCO plus target video datasets for image segmentation; Stage 2 introduces temporal modeling for fine-tuning. Unlike CAVIS and similar methods that can freeze the backbone, VidEoMT requires fine-tuning the ViT encoder due to the encoder-only design. The model uses 200 queries, $D=1024$, and is trained on H100 GPUs.

## Key Experimental Results

| Method | Backbone | YT-VIS 2019 AP | FPS | GFLOPs |
|--------|----------|----------------|-----|--------|
| CAVIS | ViT-L | 68.9 | 15 | 838 |
| DVIS-DAQ | ViT-L | 68.3 | 10 | 851 |
| DVIS++ | ViT-L | 67.7 | 18 | 846 |
| EoMT+CAVIS | ViT-L | 68.1 | 42 | 699 |
| **VidEoMT** | **ViT-L** | **68.6** | **160** | **566** |

Video Semantic Segmentation (VSPW): VidEoMT achieves mIoU 64.9 (+2.1 over DVIS++ at 62.8), mVC16 95.0, at 73 FPS (vs. DVIS++ at 13 FPS).
Video Panoptic Segmentation (VIPSeg): VidEoMT VPQ 55.2 vs. CAVIS 56.9, at 75 vs. 10 FPS (7.5× speedup).

### Ablation Study
- **Query fusion is critical**: No propagation 61.3 AP → query propagation 63.9 (+2.6) → query fusion 68.6 (+4.7), with virtually no change in speed.
- **Effect of model size**: The gap is only 0.3 AP with ViT-L (vs. CAVIS), but 2.7 AP with ViT-S, underscoring the importance of large pretrained models.
- **Pretraining quality is decisive**: Gap of 0.3 AP under DINOv2, 1.4 AP under IN21K, and 2.7 AP under IN1K.
- **VidEoMT vs. EoMT+tracker**: VidEoMT (68.6 AP, 160 FPS) outperforms EoMT+CAVIS (68.1 AP, 42 FPS) — unified is both better and faster than decoupled.
- **Query fusion vs. TrackFormer**: Fusion (68.6 AP, 160 FPS) outperforms TrackFormer (67.7 AP, 117 FPS) — simpler, faster, and more accurate.

## Highlights & Insights
- A 10×+ speedup is game-changing — 160 FPS makes real-time video segmentation practically viable.
- The insight that "a ViT pretrained with VFM objectives has implicitly learned to track" is profound — the DINO training objective promotes cross-view consistency, which is precisely what tracking requires.
- The progressive module removal study is highly convincing — each step quantitatively validates which components are redundant.
- The query fusion design is extremely simple (one linear layer + element-wise addition), embodying the principle that simplicity is power.
- The dramatic FPS improvement stems primarily not from reduced FLOPs (only a 32% reduction), but from the pure ViT architecture's superior compatibility with hardware optimizations such as FlashAttention and `torch.compile`.

## Limitations & Future Work
- Performance gaps are larger for smaller ViTs (ViT-S/B), indicating that the method's effectiveness is heavily dependent on large models and strong pretraining.
- On OVIS (severe occlusion scenarios), VidEoMT lags behind CAVIS by approximately 1.6 AP, suggesting that extreme occlusion may require stronger tracking capacity.
- Query fusion only incorporates information from the immediately preceding frame, limiting long-term temporal modeling capability.
- Training requires fine-tuning the entire ViT encoder (unlike CAVIS, which can freeze it), resulting in higher memory costs.
- Only online inference is supported; the method is not suitable for offline settings that require global temporal reasoning.

## Related Work & Insights
- **vs. CAVIS (ICCV 2025)**: CAVIS is the current SOTA but achieves only 15 FPS; VidEoMT runs at 160 FPS (10.7× faster) with only 0.3 AP loss — a large speed gain at minimal accuracy cost.
- **vs. MinVIS**: MinVIS also pursues simplicity and efficiency but employs Swin-L + Mask2Former decoder; VidEoMT is fully encoder-only, faster (160 vs. 29 FPS), and more accurate (68.6 vs. 61.6 AP).
- **vs. EoMT (CVPR 2025, image segmentation)**: VidEoMT extends EoMT to video via query propagation and fusion, achieving a 7.3 AP improvement (68.6 vs. 61.3).

---

- The argument that "strong pretraining can eliminate task-specific downstream components" is being validated across an increasing number of tasks — from image segmentation (EoMT) to video segmentation (VidEoMT), with 3D perception and video generation as natural next steps.
- The query propagation + fusion paradigm for temporal modeling is transferable to other tasks requiring inter-frame association, such as video object detection and action detection.
- The implications for real-time applications such as autonomous driving are significant: 160 FPS is sufficient for virtually any deployment scenario.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First encoder-only video segmentation model; the 10×+ speedup represents a qualitative leap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks (4 VIS + 1 VPS + 1 VSS), progressive ablation, multi-pretraining/model-size comparisons, and alternative design comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear logical flow (hypothesis → validation → design → experiment); the progressive derivation from CAVIS to VidEoMT is elegantly structured.
- Value: ⭐⭐⭐⭐⭐ 160 FPS video segmentation has substantial industrial value and demonstrates that "less is more" holds in the era of VFMs.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[CVPR 2026\] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation](rs-ssm_refining_forgotten_specifics_in_state_space_model_for_video_semantic_segm.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2026\] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] A Mixed Diet Makes DINO An Omnivorous Vision Encoder](mixed_diet_dino_omnivorous_encoder.md)

<!-- RELATED:END -->

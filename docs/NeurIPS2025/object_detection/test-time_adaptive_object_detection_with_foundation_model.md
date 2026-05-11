---
title: >-
  [Paper Note] Test-Time Adaptive Object Detection with Foundation Model
description: >-
  [NeurIPS 2025][Object Detection][test-time adaptation] This paper proposes TTAOD, a source-free open-vocabulary test-time adaptive object detection framework that combines multimodal Prompt Tuning, Mean-Teacher…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "test-time adaptation"
  - "open-vocabulary detection"
  - "Mean-Teacher"
  - "prompt tuning"
  - "dynamic memory"
date: 2026-05-08
content_hash: ed351c49afb823c8
---

# Test-Time Adaptive Object Detection with Foundation Model

**Conference**: NeurIPS 2025
**arXiv**: [2510.25175](https://arxiv.org/abs/2510.25175)
**Code**: [https://github.com/gaoyingjay/ttaod_foundation](https://github.com/gaoyingjay/ttaod_foundation)
**Area**: Object Detection / Domain Adaptation
**Keywords**: test-time adaptation, open-vocabulary detection, Mean-Teacher, prompt tuning, dynamic memory

## TL;DR
This paper proposes TTAOD, a source-free open-vocabulary test-time adaptive object detection framework that combines multimodal Prompt Tuning, Mean-Teacher, an Instance Dynamic Memory (IDM) module, and memory augmentation/hallucination strategies. It achieves 56.2% AP50 on Pascal-C (+11.0 vs. SOTA) and demonstrates consistent gains across 13 cross-domain datasets.

## Background & Motivation

**Background**: Test-time adaptation (TTA) has been well studied for classification, but TTA for object detection remains underexplored. Existing methods such as STFAR require source-domain statistics (mean/variance) and assume a closed-set category space.

**Limitations of Prior Work**: (a) Reliance on source-domain data or statistics is impractical; (b) closed-set assumptions restrict applicability to open-world scenarios; (c) pseudo-label quality degrades under corruption/domain shift, causing Mean-Teacher to collapse.

**Key Challenge**: Open-vocabulary detectors (e.g., GroundingDINO) suffer performance drops under domain shift, yet TTA must operate without source data and without restricting the category space.

**Goal**: Achieve test-time domain adaptation under a source-free, open-vocabulary setting by tuning only prompt parameters.

**Key Insight**: Vision-language foundation models (GroundingDINO) already possess strong zero-shot capabilities; adapting to the target domain requires tuning only a small number of prompt parameters. A memory module is used to accumulate high-quality instances to guide adaptation.

**Core Idea**: Freeze GroundingDINO and introduce learnable text/visual prompts → update via Mean-Teacher EMA → accumulate high-quality pseudo-labels in an Instance Dynamic Memory (IDM) using DINOv2 features → refine detection scores via memory augmentation → address negative samples via memory hallucination.

## Method

### Overall Architecture
Test image → GroundingDINO (frozen) + learnable prompts → Teacher (EMA) generates pseudo-labels → Student learns → **IDM** stores high-quality detection instances (DINOv2 features) → **Memory Augmentation** refines detection scores using class prototypes → **Memory Hallucination** synthesizes training signals for negative samples.

### Key Designs

1. **Multimodal Prompt Tuning + Warm-Start**:

    - **Function**: Insert learnable prompts into both text and visual encoders.
    - **Mechanism**: Text prompt $\tilde{E}_T = E_T + P_T$; visual prompt inserts learnable tokens $P_{I,i}$ at each layer. Key innovation — **Test-Time Warm-Start (TTWS)**: initializes visual prompts using the average token features of the first test sample, i.e., $P_{I,i} = \text{AvgPool}(E_{I,i})$.
    - **Design Motivation**: Ablation studies show TTWS is critical (+8.5% AP50), as randomly initialized visual prompts produce extremely poor pseudo-labels in early iterations, causing Mean-Teacher to collapse.

2. **Instance Dynamic Memory (IDM)**:

    - **Function**: Maintains a queue of up to 20 high-quality detected instances per class.
    - **Mechanism**: After each detection, high-confidence instances are enqueued with their DINOv2 features and confidence scores. The class prototype $v_c$ is computed as the mean feature. Low-quality instances are replaced by newer ones.
    - **Design Motivation**: Dynamically accumulates visual knowledge of the target domain to support subsequent memory augmentation and hallucination strategies.

3. **Memory Augmentation + Memory Hallucination**:

    - **Function**: Augmentation — refine detection scores using prototypes; Hallucination — synthesize training signals for images with no detections.
    - **Mechanism**: Augmentation — extract DINOv2 features for each detected box, compute cosine similarity with the class prototype, derive a refined score $s' = \alpha \exp(-\beta(1 - \text{sim}))$, and fuse with the original score. Hallucination — for images without pseudo-labels, randomly sample instances from IDM and paste them onto the image (Beta mixing $\lambda \sim \text{Beta}(1,1)$, IoU < 0.2 to prevent overlap), up to 3 instances per image.
    - **Design Motivation**: Augmentation addresses per-frame detection uncertainty; Hallucination addresses Mean-Teacher degradation caused by an abundance of negative samples — without positive samples, the teacher cannot provide effective supervision.

### Loss & Training
- $L_{total} = L_{cls} + L_{loc}$ (contrastive classification loss + localization loss)
- Mean-Teacher EMA with $\gamma = 0.999$
- Only prompt parameters are updated (~0.1% of total parameters)

## Key Experimental Results

### Main Results

| Dataset | Method | AP50 |
|--------|------|------|
| Pascal-C | Direct Test | 44.8% |
| Pascal-C | STFAR | 45.2% |
| Pascal-C | Mean-Teacher | 51.5% |
| Pascal-C | **TTAOD (Ours)** | **56.2%** |
| COCO-C | **TTAOD** | **26.0 mAP** |
| ODinW-13 | **TTAOD** | **54.2 mAP** |

### Ablation Study

| Component | AP50 | Notes |
|------|------|------|
| Baseline | 44.8% | No adaptation |
| + TPT only | 45.4% | Text prompt tuning |
| + VPT only | 41.4% | Visual prompt (no TTWS) — degrades performance |
| + TTWS | **53.4%** | Warm-start is critical |
| + All components | **56.2%** | Best performance |

### Key Findings
- TTWS is the single most critical component (+8.5%), highlighting the importance of prompt initialization in TTA.
- Visual prompts without TTWS are harmful (−3.4%), underscoring the cold-start problem.
- TTAOD achieves the best performance on 14 out of 15 corruption types.
- On ODinW-13 cross-domain benchmark: improvement on 11/13 datasets, average gain of +1.4%.

## Highlights & Insights
- **TTWS addresses the cold-start problem in TTA**: initializing prompts with features from the first test sample is simple yet highly effective, and the idea is transferable to other TTA settings.
- **Memory Hallucination is a novel solution for handling negative samples**: many images in detection may contain no objects, causing Mean-Teacher to degrade on such frames; synthesizing positive samples is an elegant remedy.
- **Source-free + open-vocabulary is a more practical setting**: eliminating dependence on source-domain data and closed-set category assumptions.

## Limitations & Future Work
- Memory quality depends on pseudo-label selection; errors may accumulate under continual domain shift.
- GroundingDINO inference is itself heavy, and additional DINOv2 feature extraction introduces further overhead.
- Applicability to streaming or real-time scenarios is not discussed.
- The memory hallucination synthesis strategy is relatively simple (direct paste); more sophisticated augmentation methods may yield better results.

## Related Work & Insights
- **vs. STFAR**: Requires source-domain statistics and a closed-set assumption; the proposed method is source-free and open-vocabulary.
- **vs. Tent/MEMO**: General-purpose TTA methods designed for classification that do not address detection-specific pseudo-label challenges.
- **vs. TPT**: TPT performs visual prompt tuning for classification; this work extends the idea to detection and introduces TTWS.

## Rating
- **Novelty**: ⭐⭐⭐⭐ TTWS and memory hallucination are novel and well-motivated designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, 15 corruption types, 13 cross-domain datasets, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is presented clearly.
- **Value**: ⭐⭐⭐⭐ Advances the practical applicability of test-time adaptation for object detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](../../CVPR2026/object_detection/cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)
- [\[CVPR 2026\] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](../../CVPR2026/object_detection/foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](../../ICLR2026/object_detection/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](../../ICCV2025/object_detection/yoloe_realtime_seeing_anything.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](../../ICCV2025/object_detection/automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)

</div>

<!-- RELATED:END -->

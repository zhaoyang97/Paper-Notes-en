---
title: >-
  [Paper Note] YOLOE: Real-Time Seeing Anything
description: >-
  [ICCV 2025][Object Detection][open-vocabulary detection] This paper proposes YOLOE, which unifies text prompt, visual prompt, and prompt-free open-scenario detection and segmentation within the YOLO architecture. Through…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "open-vocabulary detection"
  - "YOLO"
  - "text/visual/prompt-free"
  - "re-parameterization"
  - "real-time"
date: 2026-05-08
content_hash: 4f45907704a7265a
---

# YOLOE: Real-Time Seeing Anything

**Conference**: ICCV 2025
**arXiv**: [2503.07465](https://arxiv.org/abs/2503.07465)  
**Code**: [https://github.com/THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)  
**Area**: Object Detection / Instance Segmentation / Open-Vocabulary
**Keywords**: open-vocabulary detection, YOLO, text/visual/prompt-free, re-parameterization, real-time

## TL;DR
This paper proposes YOLOE, which unifies text prompt, visual prompt, and prompt-free open-scenario detection and segmentation within the YOLO architecture. Through three key designs — RepRTA (Re-parameterizable Region-Text Alignment), SAVPE (Semantic-Activated Visual Prompt Encoder), and LRPC (Lazy Region-Prompt Contrast) — YOLOE achieves high efficiency and strong performance, surpassing YOLO-World v2 on LVIS with 3× lower training cost.

## Background & Motivation
Although the YOLO series is efficient and accurate, it is confined to predefined categories and cannot generalize to open scenarios. Existing open-set methods (e.g., GLIP, Grounding DINO, DINO-X) support multiple prompt types but suffer from serious efficiency issues: text-prompt methods require costly cross-modal fusion, visual-prompt methods rely on additional visual encoders or transformers, and prompt-free methods depend on large language models to generate category names. No single model currently supports all three prompt modes efficiently.

## Core Problem
How can a model support text-prompt, visual-prompt, and prompt-free detection and segmentation across open scenarios while maintaining YOLO-level real-time efficiency and deployment simplicity?

## Method

### Overall Architecture
YOLOE builds on the standard YOLO architecture (backbone + PAN + regression head + segmentation head), replacing the classification head with an object embedding head whose outputs are compared against prompt embeddings to produce category labels. The three prompt types are handled respectively by RepRTA, SAVPE, and LRPC, each producing normalized prompt embeddings that are dot-product-contrasted with object embeddings at anchor points.

### Key Designs
1. **RepRTA (Re-parameterizable Region-Text Alignment)**: The text-prompt module. Text embeddings from a CLIP text encoder are pre-cached (no text encoder needed at inference). During training, a lightweight auxiliary network (a single SwiGLU FFN block) refines these embeddings to improve visual-semantic alignment. At inference, the auxiliary network is re-parameterized into the last convolutional layer of the object embedding head via matrix multiplication, restoring a structure identical to the standard YOLO classification head with zero inference overhead.

2. **SAVPE (Semantic-Activated Visual Prompt Encoder)**: The visual-prompt module. Regions of interest are represented as masks and processed through two decoupled branches: (a) a semantic branch that extracts prompt-agnostic high-dimensional semantic features from PAN multi-scale features; and (b) an activation branch that fuses visual prompt masks with image features in a low-dimensional space to generate grouped, prompt-aware weights. The two branches are combined via grouped aggregation to produce the final prompt embedding, encoding visual cues with negligible computational overhead.

3. **LRPC (Lazy Region-Prompt Contrast)**: The prompt-free module. A dedicated prompt embedding is trained to detect "whether an object exists" (category-agnostic), after which category retrieval is performed only on anchor points identified as objects (rather than all 8,400), matching against a built-in vocabulary of 4,585 class names. Compared to language-model-based name generation (e.g., GenerateU uses 250M-parameter FlanT5), this retrieval paradigm eliminates the language model dependency and achieves a 53× inference speedup.

### Loss & Training
- Classification: BCE loss; regression: IoU + DFL loss; segmentation: mask BCE loss.
- Staged training: text prompt for 30 epochs → freeze and train SAVPE for 2 epochs → train prompt-free embedding for 1 epoch.
- Segmentation masks are generated as pseudo-labels by SAM-2.1 using GT bounding boxes as prompts.
- A global negative sample dictionary strategy replaces empty-string negatives, yielding a +0.9 AP improvement.

## Key Experimental Results

| Model | Prompt | LVIS AP | APr | Training Time | FPS (T4/iPhone 12) |
|------|--------|---------|-----|----------|------------|
| YOLO-Worldv2-S | T | 24.4 | 17.1 | 41.7h | 216/49 |
| **YOLOE-v8-S** | T/V | **27.9/26.2** | **22.3/21.3** | **12.0h** | **306/64** |
| YOLO-Worldv2-L | T | 35.5 | 25.6 | 80.0h | 80/22 |
| **YOLOE-v8-L** | T/V | **35.9/34.2** | **33.2/33.2** | **22.5h** | **103/27** |
| T-Rex2 | V | 37.4 | 29.9 | - | - |
| GenerateU (Swin-T) | Free | 26.8 | 20.0 | - | 0.48 |
| **YOLOE-v8-L** | Free | **27.2** | **23.5** | - | **25.3** |

- COCO transfer: YOLOE-v8-L full-tune for 80 epochs achieves 53.0 APb / 42.7 APm, surpassing YOLOv8-L trained from scratch for 300 epochs (52.4/42.3).
- Zero-shot segmentation on LVIS: YOLOE-v8-L achieves 23.5 APm, outperforming fine-tuned YOLO-Worldv2-L at 19.8 APm.

### Ablation Study
- RepRTA contributes the most: +2.3 AP with zero inference overhead.
- Removing cross-modal fusion reduces AP by 1.9 but yields a 1.28× speedup; replacing with a stronger MobileCLIP recovers 1.5 AP.
- SAVPE vs. simple mask pooling: +1.5 AP.
- LRPC achieves a 1.7× speedup (v8-S) with no performance degradation.

## Highlights & Insights
- **Elegant application of re-parameterization**: an auxiliary network enhances alignment during training and is seamlessly merged into the standard YOLO structure at inference — a true "free lunch."
- **Three prompt modes unified in one model**: text, visual, and prompt-free modes share the same backbone and heads, differing only in prompt encoding.
- **Highly economical training**: only 33 epochs total (30+2+1), completable on 8× RTX 4090 GPUs — 3× faster than YOLO-World.
- **Strong practicality**: supports TensorRT and CoreML deployment with real-time performance on mobile, highly suitable for industrial applications.
- **Retrieval over generation** for prompt-free paradigm: eliminates language model dependency with a 53× speedup.

## Limitations & Future Work
- Joint multi-task (detection + segmentation) training causes a 0.9 AP drop on frequent categories (APf).
- Visual prompts are trained for only 2 epochs with other parameters frozen, potentially limiting the upper bound of visual prompt performance.
- The prompt-free mode relies on a predefined vocabulary of 4,585 classes and cannot discover entirely novel categories.
- Pseudo-masks generated by SAM-2.1 may contain noise, affecting segmentation accuracy.

## Related Work & Insights
- **vs. YOLO-World v2**: YOLOE removes costly cross-modal fusion, replacing it with RepRTA re-parameterization — 3× faster training, 1.4× faster inference, +3.5 AP (S model).
- **vs. DINO-X**: DINO-X also supports multiple prompt types but is computationally prohibitive for edge deployment; YOLOE maintains YOLO-level efficiency.
- **vs. T-Rex2**: YOLOE-v8-L achieves +3.3 APr on rare categories with less training data and fewer resources.
- **vs. GenerateU**: prompt-free with 6.3× fewer parameters and 53× faster inference.

The re-parameterization paradigm is transferable to other settings requiring zero inference overhead. The retrieval-based prompt-free detection paradigm is more efficient than generative approaches and warrants exploration in other open-world tasks. The lightweight visual prompt encoder design offers a useful reference for few-shot detection and segmentation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ RepRTA represents a novel application of re-parameterization in open-vocabulary detection; the unified three-prompt framework is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on LVIS/COCO with detailed ablations for all three prompt modes and downstream transfer experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured; a roadmap table clearly illustrates the contribution of each step from baseline to the final model.
- **Value**: ⭐⭐⭐⭐⭐ A highly practical open-vocabulary detection solution with strong potential to become the preferred open-world YOLO framework in industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[AAAI 2026\] An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice](../../AAAI2026/object_detection/an_overall_real-time_mechanism_for_classification_and_quality_evaluation_of_rice.md)
- [\[ICCV 2025\] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)

</div>

<!-- RELATED:END -->

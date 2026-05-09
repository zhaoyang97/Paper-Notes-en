---
title: >-
  [Paper Note] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications
description: >-
  [ICCV 2025][Medical Imaging][Video Foundation Models] This paper introduces SciVid, a benchmark comprising five interdisciplinary scientific video tasks—including animal behavior classification, tissue tracking, and weather forecasting—that systematically evaluates six categories of Video Foundation Models (ViFMs). The study finds that adapting a frozen ViFM backbone with a simple trainable readout suffices to achieve state-of-the-art performance on multiple scientific tasks, providing the first systematic evidence of the transferability of general-purpose ViFMs to scientific domains.
tags:
  - ICCV 2025
  - Medical Imaging
  - Video Foundation Models
  - Cross-Domain Evaluation
  - Scientific Applications
  - Benchmark
  - Spatiotemporal Modeling
date: 2026-05-08
content_hash: 4add54e08dddbc74
---

# SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications

**Conference**: ICCV 2025
**arXiv**: [2507.03578](https://arxiv.org/abs/2507.03578)
**Code**: [github.com/google-deepmind/scivid](https://github.com/google-deepmind/scivid)
**Area**: Medical Imaging
**Keywords**: Video Foundation Models, Cross-Domain Evaluation, Scientific Applications, Benchmark, Spatiotemporal Modeling

## TL;DR

This paper introduces SciVid, a benchmark comprising five interdisciplinary scientific video tasks—including animal behavior classification, tissue tracking, and weather forecasting—that systematically evaluates six categories of Video Foundation Models (ViFMs). The study finds that adapting a frozen ViFM backbone with a simple trainable readout suffices to achieve state-of-the-art performance on multiple scientific tasks, providing the first systematic evidence of the transferability of general-purpose ViFMs to scientific domains.

## Background & Motivation

### Core Problem

Video Foundation Models have achieved remarkable progress in natural video understanding (action recognition, video question answering, etc.), yet their application to scientific domains remains limited. Each scientific field (medicine, animal behavior, meteorology) typically develops its own specialized models and evaluates them exclusively within that domain.

The central question is: **Can general-purpose ViFMs transfer effectively across diverse scientific domains? Can a single pretrained ViFM compete with domain-specific baselines?**

### Limitations of Prior Work

**Domain-specific models**: Models such as Endo-FM (endoscopy) and EchoCLIP (echocardiography) target specific domains, and their cross-domain generalizability is unknown.

**Lack of unified evaluation**: Different fields employ different metrics, data formats, and evaluation protocols, making it impossible to compare the scientific applicability of different ViFMs in a principled manner.

**Limited ViFM context**: Existing ViFM benchmarks (e.g., SSv2, Kinetics) primarily assess natural video understanding and do not cover scientific applications such as medicine or meteorology, leaving the performance of ViFMs under large domain shifts unclear.

### Root Cause

**Key insight**: Many scientific tasks can be formulated as video modeling problems—surgical tissue tracking is point tracking, weather forecasting is spatiotemporal prediction, and animal behavior analysis is video classification. If general-purpose ViFMs can transfer effectively to these highly disparate domains, the barrier to developing specialized models for scientific applications would be substantially lowered. SciVid is designed to provide a unified evaluation framework to answer this question.

## Method

### Overall Architecture

SciVid's evaluation paradigm follows a unified representation learning pipeline:
1. A pretrained ViFM serves as a frozen backbone for video feature extraction.
2. A lightweight task-specific readout module is trained on top of the extracted features.
3. Optionally, the backbone may be fine-tuned.

This design ensures fair comparison across ViFMs—the sole variable is the representational quality of the backbone.

### Key Designs

#### 1. **Task Design Principles and Five Benchmark Tasks**

- **Function**: Construct five video tasks spanning three scientific domains with diverse output types.
- **Mechanism**:

  Task selection follows four principles: (1) broad coverage of scientific application challenges; (2) inclusion of diverse domains and distribution shifts; (3) emphasis on temporal understanding; and (4) a mixture of well-established and emerging tasks.

  **Animal Behavior Classification**:
  - FlyVsFly: Classification of fruit fly social behaviors (7 classes), grayscale video, $144 \times 144$, 16-frame input.
  - CalMS21: Classification of mouse social behaviors (4 classes), grayscale video, $285 \times 512$, 16-frame input.

  **Surgical Tissue Tracking (STIR)**:
  - Tracking tissue surface motion during surgery, RGB video, $1024 \times 1280$, 7–19,419 frames.
  - Task: given a query point in the first frame, track it to the last frame.

  **Weather Forecasting (WeatherBench 2)**:
  - Medium-range weather prediction; input: 8 days (16 frames), output: 8 days.
  - Predicts Z500 (geopotential height), T850 (temperature), and Q700 (specific humidity).

  **Typhoon Pressure Prediction (Digital Typhoon)**:
  - Infrared satellite imagery; input: 12 frames; predict central pressure 12 hours ahead.
  - Time-series regression task.

- **Design Motivation**: The five tasks differ substantially in input modality (grayscale/RGB/infrared/meteorological variables), output format (classification/point tracking/dense prediction/scalar regression), and dataset scale (60 to 1M samples), enabling a comprehensive assessment of ViFM generality.

#### 2. **Backbone Selection and Evaluation**

- **Function**: Systematically evaluate the representational quality of six categories of ViFM backbones.
- **Mechanism**:

  Evaluated backbones include:
  - **Image model**: DINOv2 (ViT-L/g)—self-distillation trained purely on images, augmented with learnable temporal positional encodings.
  - **Video models**:
    - VideoPrism (B/g): Two-stage training—video-text contrastive learning followed by masked autoencoding, leveraging language supervision.
    - VideoMAE / VideoMAEv2 (B/L/H/g): Masked autoencoding in pixel space.
    - V-JEPA (L/H): Masked prediction in latent space (JEPA paradigm).
    - 4DS (L/e): Masked autoencoding in pixel space, with 300M–4B parameters.
  - **Resize baseline**: The input video is resized to low resolution as a naive feature (verifying that backbones extract meaningful representations).

  A key design choice is that all backbones receive standard 3-channel spatiotemporal clips, ensuring a consistent evaluation protocol.

- **Design Motivation**: The selection covers the three dominant ViFM training paradigms—contrastive learning (VideoPrism), pixel-level masked reconstruction (VideoMAE/4DS), and latent-space prediction (V-JEPA)—as well as a pure image baseline (DINOv2), enabling systematic analysis of how different pretraining strategies affect scientific applicability.

#### 3. **Task Readout Design**

- **Function**: Adapt general-purpose ViFM features to task-specific outputs.
- **Mechanism**:

  **Classification / Pressure Prediction**: Cross-attention readout—a single learnable query aggregates backbone features via cross-attention to produce class logits or pressure predictions. Loss: sigmoid cross-entropy (classification) or L2 (regression).

  **Tissue Tracking (STIR)**: Cross-attention readout—queries are provided by positional encodings of query points; keys and values come from backbone features. The model predicts positions, visibility, and uncertainty for all target points. Loss: Huber loss + BCE.

  **Weather Forecasting**: DPT readout—a series of trainable convolutional and reorganization layers upsamples features to per-pixel predictions. Loss: area-weighted L1.

  All readouts are trained from scratch with the backbone frozen. The full suite of experiments runs in under one day on a single H100 GPU.

- **Design Motivation**: Readout designs are kept deliberately simple so that performance differences can be attributed to backbone representational quality rather than task adaptation complexity. Cross-attention significantly outperforms simple linear projection, confirming that the spatial information in features is informative.

### Loss & Training

- **Classification**: Sigmoid cross-entropy loss.
- **Tracking**: Huber loss (position) + BCE (visibility/uncertainty).
- **Weather Forecasting**: Area-weighted, channel-weighted L1 loss.
- **Pressure Prediction**: L2 loss on pressure offsets.

All tasks are trained for a unified 40k steps (some tasks require 400k steps for optimal performance), with the backbone frozen.

## Key Experimental Results

### Main Results

**Comparison with SOTA** (readout training with frozen backbone):

| Task | Domain-Specific SOTA | Best ViFM (Frozen) | ViFM Reaches SOTA? |
|------|---------------------|--------------------|--------------------|
| CalMS21 | VideoPrism-g 91.5 mAP | V-JEPA-H **92.4 mAP** | ✅ Surpassed |
| FlyVsFly | VideoPrism-g 92.0 mAP | VideoPrism-g **92.5 mAP** | ✅ Surpassed |
| STIR | MFT 68.5%/77.6% acc | 4DS-e 51.3%/57.8% (frozen) → 61.2%/69.2% (fine-tuned) | ❌ Significant gap |
| Digital Typhoon | Kitamoto 11.71 RMSE | 4DS-L **3.88 RMSE** (val) | ✅ Large margin |
| WeatherBench 2 | GenCast ~best | 4DS-e/VideoMAEv2-g moderate | ❌ Significant gap |

### Ablation Study

**Frozen feature performance across 5 tasks for different backbones**:

| Backbone | Params (M) | CalMS21 mAP↑ | FlyVsFly mAP↑ | STIR Acc↑ | DT RMSE↓ | WB2 Z500↓ |
|----------|-----------|-------------|--------------|----------|---------|----------|
| 4DS-e | 3811 | 0.817 | 0.894 | **0.513** | 4.23 | **601** |
| DINOv2-g | 1135 | **0.866** | 0.866 | 0.215 | 6.33 | 627 |
| VideoMAEv2-g | 1013 | 0.862 | 0.887 | 0.344 | 4.53 | 594 |
| V-JEPA-H | 635 | 0.828 | **0.901** | 0.443 | **4.16** | 619 |
| VideoPrism-g | 1113 | 0.855 | 0.839 | 0.351 | 5.01 | 635 |
| Resize | 0 | 0.122 | 0.095 | 0.280 | 10.0 | 642 |

**Readout architecture ablation**:

| Task | Linear Readout | Cross-Attention Readout |
|------|---------------|------------------------|
| FlyVsFly mAP↑ | 0.568 | **0.894** |
| CalMS21 mAP↑ | 0.525 | **0.817** |
| Digital Typhoon RMSE↓ | 7.45 | **4.23** |

### Key Findings

1. **No single best backbone**: 4DS-e performs best on tracking and weather forecasting; V-JEPA-H performs best on FlyVsFly; DINOv2 performs best on CalMS21. Optimal backbone selection is task-dependent.
2. **Video models generally outperform image models**: DINOv2 substantially underperforms video models on tasks requiring strong temporal modeling (STIR: 0.215 vs. 4DS-e: 0.513).
3. **Pixel-level masked autoencoding models excel at spatiotemporal prediction**: VideoMAE and 4DS consistently outperform other paradigms on WeatherBench 2.
4. **ViFMs achieve SOTA on 3 of 5 tasks**: They surpass domain-specific methods on animal behavior classification and typhoon pressure prediction, but a significant gap remains on tissue tracking and weather forecasting.
5. **Temporal modeling genuinely matters**: Frame shuffling experiments show a substantial performance drop on tracking tasks, while the impact on classification tasks is comparatively minor.
6. **Model scale does not always help**: 4DS-L (300M) outperforms 4DS-e (4B) on Digital Typhoon, and VideoMAE-B approaches VideoMAE-L on STIR.

## Highlights & Insights

1. **First interdisciplinary scientific ViFM benchmark**: SciVid unifies three entirely distinct domains—medicine, animal behavior, and meteorology—within a single evaluation framework, addressing an important gap in the literature.
2. **Fairness of experimental design**: A unified readout architecture and training protocol ensure comparability across backbones; the full experiment suite completes in under one day on a single H100.
3. **Positive finding**: General-purpose ViFMs can surpass domain-specific methods on multiple scientific tasks, demonstrating the cross-domain transferability of pretrained knowledge.
4. **Practical guidance**: Scientists need not train specialized models from scratch; selecting a strong ViFM backbone combined with a simple readout is sufficient to achieve competitive results.

## Limitations & Future Work

1. **Limited task coverage**: Only 5 tasks across 3 domains; microscopy, satellite time series, and underwater video are not included.
2. **Predominantly short-clip evaluation**: Except for STIR, all tasks use short clips (8–16 frames), leaving long-video understanding unexplored.
3. **Data-efficient adaptation not fully explored**: Low-data regimes are only briefly examined (appendix); few-shot adaptation strategies are not investigated in depth.
4. **Large gap on weather forecasting**: ViFMs fall substantially short of GraphCast/GenCast on WeatherBench 2, likely requiring better pretraining or adaptation methods.
5. **Weak STIR tracking**: Simple readouts lack key components for tracking tasks (feature pyramids, correlation volumes, iterative refinement).

## Related Work & Insights

- **Relation to VideoEval**: VideoEval assesses ViFMs on challenging tasks, but SciVid covers more scientific domains and provides careful comparison against domain-specific SOTA.
- **Relation to ClimaX**: ClimaX is a foundation model for weather and climate but is confined to a single domain. SciVid investigates cross-domain performance of general-purpose ViFMs.
- **Insight**: Future ViFM pretraining could deliberately incorporate scientific domain data (medical video, meteorological data, etc.) to enhance cross-domain transfer capability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of an interdisciplinary benchmark is novel, though the core technical approach (frozen backbone + readout) is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Full matrix evaluation across 6 backbone categories × 5 tasks, with temporal, readout, and scale ablations all covered.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, rich tables and figures, with key conclusions immediately apparent.
- **Value**: ⭐⭐⭐⭐ — Provides an important reference for applying ViFMs in scientific domains, though the primary contribution is a benchmark rather than a novel method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](../../ICLR2026/medical_imaging/adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)
- [\[NeurIPS 2025\] EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](../../NeurIPS2025/medical_imaging/endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[ICCV 2025\] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users](progait_a_multi-purpose_video_dataset_and_benchmark_for_transfemoral_prosthesis_.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](../../CVPR2026/medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)

</div>

<!-- RELATED:END -->

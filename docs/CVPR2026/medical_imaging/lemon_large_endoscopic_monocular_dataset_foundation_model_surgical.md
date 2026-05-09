---
title: >-
  [Paper Note] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings
description: >-
  [CVPR 2026][Medical Imaging][surgical dataset] This paper introduces LEMON, the largest open surgical video dataset to date (4,194 videos, 938 hours, 35 procedure types), and proposes LemonFM, a foundation model based on augmented knowledge distillation, which comprehensively outperforms existing methods across four downstream tasks: surgical phase recognition, tool detection, action recognition, and semantic segmentation.
tags:
  - CVPR 2026
  - Medical Imaging
  - surgical dataset
  - foundation model
  - self-supervised learning
  - data curation pipeline
  - knowledge distillation
date: 2026-05-08
content_hash: 5722c7f32745f532
---

# LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings

**Conference**: CVPR 2026
**arXiv**: [2503.19740](https://arxiv.org/abs/2503.19740)
**Code**: [https://github.com/visurg-ai/LEMON](https://github.com/visurg-ai/LEMON)
**Area**: Medical Imaging / Surgical Vision
**Keywords**: surgical dataset, foundation model, self-supervised learning, data curation pipeline, knowledge distillation

## TL;DR

This paper introduces LEMON, the largest open surgical video dataset to date (4,194 videos, 938 hours, 35 procedure types), and proposes LemonFM, a foundation model based on augmented knowledge distillation, which comprehensively outperforms existing methods across four downstream tasks: surgical phase recognition, tool detection, action recognition, and semantic segmentation.

## Background & Motivation

1. **Background**: Conventional surgical datasets typically contain fewer than 100 videos and 30 hours of footage, leading to poor model generalization. Although self-supervised learning reduces reliance on annotated data, large-scale high-quality surgical data remains scarce.
2. **Limitations of Prior Work**: While GenSurgery and SurgeNetXL have expanded dataset scale, they lack data curation steps and include non-surgical content (conference talks, patient testimonials, etc.), introducing noisy features.
3. **Key Challenge**: Surgical data is constrained by privacy regulations and annotation costs, raising the challenge of constructing high-quality large-scale datasets from publicly available online videos.
4. **Goal**: Design a systematic data curation pipeline to curate high-quality surgical datasets from YouTube videos.
5. **Key Insight**: Multi-stage automated curation (classification → trimming → preprocessing → annotation) combined with manual quality control.
6. **Core Idea**: An augmented knowledge distillation approach that exploits cross-patient appearance similarity and inter-frame motion invariance to learn superior surgical visual representations.

## Method

### Overall Architecture

On the data side: YouTube video collection → storyboard classification → frame-level classification and trimming → non-surgical content removal → annotation validation. On the model side: augmented knowledge distillation based on the DINO framework with a ConvNeXt-L backbone.

### Key Designs

1. **Multi-Stage Data Curation Pipeline**:

   - **Function**: Curates 4,194 high-quality surgical videos from 18K raw videos.
   - **Mechanism**: (1) Video-level: a storyboard classifier (ResNet18) filters out non-surgical videos; (2) Frame-level: a frame classifier is trained to localize the start and end of surgical content, trimming non-surgical segments and discarding videos where surgical frames account for less than 90%; (3) Region-level: YOLOv8 is trained to detect and mask non-surgical regions within frames (UI elements, logos, etc.).
   - **Design Motivation**: Each filtering stage employs an independent model validated by human annotators, achieving 100% video-level precision and >99.9% frame-level precision.

2. **Augmented Knowledge Distillation**:

   - **Function**: Learns surgical representations invariant to subtle motion and cross-patient appearance variation.
   - **Mechanism**: Additional supervisory signals $W_i$ are introduced into the DINO student-teacher framework. Each $W_i$ consists of two images, preferentially retrieved as appearance-similar frames from other videos of the same procedure type (cosine distance < 3× the distance of adjacent frames), supplemented by temporally adjacent frames when necessary. The loss function is $\mathcal{L} = -\sum_i \sum_{u \in U_i} \sum_{v \in V_i \cup W_i} P_t(z|u) \log P_s(z|v)$.
   - **Design Motivation**: Standard DINO learns invariance only across different augmentations of the same image; augmented distillation additionally introduces cross-patient and cross-frame invariance.

3. **Video Classification Model LemonFM-Vid**:

   - **Function**: Performs video-level procedure type classification using LemonFM frame embeddings.
   - **Mechanism**: Frame embeddings are aggregated via typicality-weighted pooling (typicality defined as the inverse of K-NN distance) to obtain a video embedding $v_e = \sum_j \omega_j \phi_j$, which is then classified by a single-layer MLP.
   - **Design Motivation**: Surgical procedures are localized to specific anatomical regions, and characteristic scenes can be associated with procedure types.

### Loss & Training

DINO cross-entropy loss with augmented data pairs. Trained for 60 epochs on 8 × V100 GPUs.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | LemonFM | Prev. SOTA (SurgeNetXL) | Gain |
|---|---|---|---|---|
| AutoLaparo Phase Recognition | Jaccard | 64.8 | 55.1 | +9.7pp |
| Cholec80 Phase Recognition | Jaccard | 85.1 | 72.0 | +13.1pp |
| Cholec80 Tool Detection | mAP | 93.7 | 86.5 | +7.2pp |
| GraSP Tool Detection | mAP | 94.4 | 83.8 | +10.6pp |
| CholecT50 Action Recognition | mAP | 61.9 | 57.5 | +4.4pp |
| CholecSeg8k Semantic Segmentation | mDice | 81.3 | 69.0 | +12.3pp |

### Ablation Study

| Configuration | AutoLaparo (F1) | CholecSeg8k (mDice) | Notes |
|---|---|---|---|
| ImageNet Pretraining | 53.0 | 64.4 | General-purpose pretraining |
| Cholec80 (51h) | 46.9 | 64.1 | Small-scale surgical data |
| LEMON (uncurated) | 61.4 | 67.4 | Without curation pipeline |
| LEMON (curated) | 65.9 | 68.7 | Curation gain: +4.5pp |
| LEMON + Augmented Distillation | 66.9 | 71.9 | Full model |

### Key Findings

- The data curation pipeline contributes substantially: +4.5pp F1 and +1.3pp mDice.
- ConvNeXt outperforms ViT, particularly on segmentation (+10.7pp mDice), as convolutional inductive biases better preserve fine-grained surgical details.
- LemonFM fine-tuned with only 50% of labeled data still surpasses all baselines trained with 100% of data.
- Discriminative pretraining (DINO-family) substantially outperforms generative pretraining (MAE-family).

## Highlights & Insights

- **Quality Assurance at Scale**: A rigorous curation pipeline reducing 18K raw videos to 4,194 high-quality videos achieves an unprecedented balance between scale and data quality.
- **Cross-Patient Augmented Distillation**: Leveraging videos from different patients undergoing the same procedure to learn appearance invariance makes ingenious use of procedure-level annotations.
- **Comprehensive Benchmarking**: Full evaluation across 6 datasets and 4 tasks establishes a standardized benchmark for surgical vision foundation models.

## Limitations & Future Work

- The data source consists of publicly available YouTube videos; while ethical considerations have been addressed, potential controversy remains.
- Procedure classification mAP reaches only 57.8%, with significant confusion among anatomically adjacent procedure types.
- Future work will focus on developing surgical-specific video foundation models.

## Related Work & Insights

- **vs. Endo-FM**: Endo-FM is trained on private data, limiting reproducibility; LEMON is fully open-source.
- **vs. EndoViT**: EndoViT aggregates multiple small public datasets, achieving far inferior scale and performance compared to LEMON.
- **vs. SurgeNetXL**: SurgeNetXL lacks data curation; LEMON's curation pipeline yields substantial performance gains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The augmented distillation design is novel and the data curation pipeline is systematic.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6 datasets, 4 tasks, low-data experiments, cross-validation, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure with sufficient detail.
- **Value**: ⭐⭐⭐⭐⭐ — Both the dataset and model will serve as important foundational resources for the surgical vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [\[CVPR 2026\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->

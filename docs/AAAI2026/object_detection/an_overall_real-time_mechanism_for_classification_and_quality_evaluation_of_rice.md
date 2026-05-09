---
title: >-
  [Paper Note] An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice
description: >-
  [AAAI2026][Object Detection][rice quality evaluation] This paper proposes a real-time overall mechanism for rice quality evaluation, integrating three modules: an improved YOLO-v5 (variety detection), an improved ConvNeXt-Tiny (intactness grading), and K-means (chalkiness region quantification). The system achieves 99.14% mAP and 97.89% detection accuracy on a self-constructed dataset of 20,000 images spanning six rice varieties.
tags:
  - AAAI2026
  - Object Detection
  - rice quality evaluation
  - YOLO-v5
  - ConvNeXt-Tiny
  - K-means
  - SimAM
  - ECA
date: 2026-05-08
content_hash: 2a7b1fd86e1ac1e7
---

# An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice

**Conference**: AAAI2026
**arXiv**: [2502.13764](https://arxiv.org/abs/2502.13764)
**Code**: [Dataset](https://huggingface.co/datasets/xwk25/RiceCC)
**Area**: Object Detection
**Keywords**: rice quality evaluation, object detection, YOLO-v5, ConvNeXt-Tiny, K-means, SimAM, ECA

## TL;DR

This paper proposes a real-time overall mechanism for rice quality evaluation, integrating three modules: an improved YOLO-v5 (variety detection), an improved ConvNeXt-Tiny (intactness grading), and K-means (chalkiness region quantification). The system achieves 99.14% mAP and 97.89% detection accuracy on a self-constructed dataset of 20,000 images spanning six rice varieties.

## Background & Motivation

Rice is one of the most important staple crops worldwide, and its quality directly affects dietary health and market value. Traditional rice quality assessment relies primarily on manual sensory inspection, where experts evaluate variety, intactness, and chalkiness through visual observation. This approach has notable drawbacks:

- **Low efficiency**: Manual inspection is slow and cannot meet large-scale rapid assessment demands.
- **High subjectivity**: Results are susceptible to lighting conditions, eye fatigue, and human factors.
- **Lack of quantitative standards**: Most existing computer vision-based studies provide only qualitative grading and do not conduct quantitative evaluation according to national standards (e.g., GB/T 1354-2018).

Prior work has achieved promising results on individual tasks (e.g., EfficientNet-B0 reaching 98.37% classification accuracy, RiceNet achieving 94% accuracy for five-variety classification), but no comprehensive mechanism capable of simultaneously performing variety recognition, intactness grading, and chalkiness quantification has been established.

## Core Problem

How to construct an end-to-end real-time rice quality evaluation system that simultaneously addresses three key tasks:

1. **Variety recognition**: Accurately detecting and classifying different rice varieties in mixed samples.
2. **Intactness grading**: Classifying grains within the same variety into whole, large broken, and small broken categories.
3. **Chalkiness quantification**: Precisely measuring the area ratio of chalky regions within the rice endosperm.

All evaluations must conform to the Chinese national standard GB/T 1354-2018.

## Method

The overall framework consists of three core modules:

### Module 1: Improved YOLO-v5 (Variety Detection)

In the backbone of standard YOLO-v5, the third C3 layer is replaced with the SimAM (Simple Attention Module). Key advantages of SimAM include:

- **No additional parameters**: Neuron importance is assessed by optimizing an energy function without introducing extra parameters.
- **Self-induction**: An analytical solution for feature weighting is directly derived, avoiding the complex computations of conventional attention mechanisms.
- Implementation: Spatial mean and variance are computed, and attention weights are generated via the inverse of the energy function, then multiplied element-wise with input features.

### Module 2: Improved ConvNeXt-Tiny (Intactness Evaluation)

An ECA (Efficient Channel Attention) module is inserted between the last ConvNeXt Block and the global average pooling layer of standard ConvNeXt-Tiny:

- ECA computes attention weights for each channel independently, without considering spatial relationships between pixels.
- It efficiently captures key channel-wise information while avoiding costly pairwise interaction computations.
- ImageNet-1K pre-trained weights are used for initialization to reduce training cost.
- Intactness is classified into three levels: whole, large broken, and tiny broken; the data ratio is approximately 10:1 (whole:broken).

### Module 3: K-means Clustering (Chalkiness Evaluation)

K-means clustering is applied to grayscale rice images to segment chalky from non-chalky regions:

- $K=1$ is set, as chalky regions typically concentrate in the center of the grain as a single continuous opaque area.
- After clustering, the total number of chalky pixels is extracted and converted to actual area using image resolution calibration parameters.
- A geometric polygon fitting method is used to estimate the segmented area.
- Images are captured under five different lighting intensities; based on comparison with agricultural expert assessments, brightness level 1 is selected as the standard illumination.

### Dataset

- Approximately 20,000 Chinese rice images are collected manually, covering six major cultivated varieties.
- Six varieties: Guangdong Simiao (GD, indica), Northeast Glutinous Rice (NM, glutinous), Wuchang Rice (WC, indica), Panjin Crab Field Rice (PJX, japonica), Wannian Tribute Rice (WN, indica), and Yanbian Rice (YB, japonica).
- Images are captured with an industrial camera (Sony CMOS, 1920×1080P) against a monochrome background.
- An additional 300 images (10 grains per variety × 5 lighting levels) are collected for chalkiness experiments.

## Key Experimental Results

### Object Detection (Variety Recognition)

| Model | Test Accuracy |
|-------|--------------|
| Faster-RCNN (two-stage) | 87.33% |
| Tridentnet (two-stage) | 93.69% |
| YOLO-v5 (one-stage) | 95.05% |
| **Improved YOLO-v5 + SimAM** | **97.89%** |

- Validation set mAP reaches 99.14% (vs. 98.76% for the baseline, +0.38%).
- GD, NM, and YB achieve precision of 1.0 on the validation set.

### Intactness Classification

| Model | Average Accuracy |
|-------|----------------|
| Decision Tree | 93.72% |
| Random Forest | 94.27% |
| AlexNet | 94.90% |
| ConvNeXt-Tiny (baseline) | 95.58% |
| **ConvNeXt-Tiny + ECA** | **97.61%** |

- Average accuracy improves by approximately 2% over the baseline ConvNeXt-Tiny.
- The most significant gain is observed on GD: 88.65% → 98.80% (+10.15%).
- YB achieves the highest accuracy: 99.68%.

### Chalkiness Recognition

- Automated measurements under brightness level 1 are highly consistent with visual assessments by agricultural experts.
- Compared to the manual visual inspection method specified in GB/T 1354-2018, the automated approach is more precise and substantially faster.

## Highlights & Insights

- **Three-stage comprehensive mechanism**: This work is among the first to integrate variety detection, intactness grading, and chalkiness quantification into a unified real-time evaluation pipeline, representing a rare end-to-end system in this domain.
- **Compliance with national standards**: Using GB/T 1354-2018 as the evaluation benchmark gives the results practical industrial applicability.
- **High-quality self-constructed dataset**: 20,000 images covering six major Chinese cultivated varieties, publicly released on HuggingFace.
- **Well-motivated attention module selection**: Both SimAM (parameter-free) and ECA (channel-focused) are lightweight attention mechanisms well-suited for real-time detection scenarios.
- **Clever design of $K=1$ in K-means**: The spatial clustering property of chalky regions is leveraged to simplify the segmentation problem.

## Limitations & Future Work

- **Limited variety coverage**: Only six Chinese rice varieties are included; generalizability to international varieties (e.g., Thai jasmine rice, Indian Basmati) is not validated.
- **Outdated detection backbone**: The system is based on YOLO-v5 and is not compared against more recent versions such as YOLO-v7/v8/v9.
- **Small-scale chalkiness experiment**: Only 300 images (50 per variety) are used, limiting statistical persuasiveness.
- **Overly strong $K=1$ assumption**: The approach may fail for abnormal grains with unevenly distributed or multiple chalky regions.
- **Lack of end-to-end integration**: The three modules are independent; cascaded deployment efficiency and error accumulation are not evaluated.
- **No deployment analysis**: Inference speed and resource consumption on embedded devices or production lines are not reported.

## Related Work & Insights

| Aspect | Ours | Prior Work |
|--------|------|-----------|
| Task completeness | Variety + intactness + chalkiness (unified) | Typically focuses on a single task |
| Evaluation standard | Follows GB/T 1354-2018 | Most lack quantitative national standard alignment |
| Detection model | Improved YOLO-v5 (97.89%) | Moses et al. EfficientNet-B0 (98.37%, classification only) |
| Classification model | ConvNeXt-Tiny + ECA (97.61%) | RiceNet (94%), Lin et al. CNN (95.5%) |
| Data scale | 20,000 images / 6 varieties | RiceNet: 4,700 images / 5 varieties |

Compared to single-task methods, the proposed system's strength lies in its systematic integration and standardization, though it does not match the depth of dedicated studies in terms of architectural innovation for individual tasks.

- **Paradigm for agricultural quality inspection automation**: The three-stage pipeline (detection → grading → quantification) can be generalized to quality inspection of other agricultural products (e.g., wheat, corn, fruit).
- **Engineering value of lightweight attention**: Parameter-free or low-parameter attention mechanisms such as SimAM and ECA have practical value for resource-constrained industrial deployment.
- **Importance of aligning with national standards**: Research that references national or international standards significantly enhances its translational value.
- **Chalkiness quantification approach**: The simple combination of K-means and geometric fitting remains effective for low-complexity image segmentation tasks; not every problem requires a deep learning solution.

## Rating

- Novelty: ⭐⭐ (Methodological contributions are incremental; SimAM and ECA are existing modules directly embedded into standard architectures.)
- Experimental Thoroughness: ⭐⭐⭐ (Detection and classification experiments are reasonably comprehensive, but the chalkiness experiment is limited in scale.)
- Writing Quality: ⭐⭐⭐ (Structure is clear and national standard citations are properly used, though some experimental analyses lack depth.)
- Value: ⭐⭐⭐ (High engineering application value; academic contribution is limited.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[AAAI 2026\] Real-Time 3D Object Detection with Inference-Aligned Learning](real-time_3d_object_detection_with_inference-aligned_learning.md)
- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](../../ICCV2025/object_detection/yoloe_realtime_seeing_anything.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](../../CVPR2026/object_detection/paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)

</div>

<!-- RELATED:END -->

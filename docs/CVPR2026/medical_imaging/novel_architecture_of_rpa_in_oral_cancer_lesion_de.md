---
title: >-
  [Paper Note] Novel Architecture of RPA in Oral Cancer Lesion Detection
description: >-
  [CVPR 2026][Medical Imaging][Oral cancer detection] This paper compares low-code RPA platforms (UiPath, Automation Anywhere) against a Python-based design pattern approach (Singleton + Batch Processing) for oral cancer detection automation. The proposed OC-RPAv2 reduces per-image inference time from 2.5 seconds to 0.06 seconds, achieving a 60–100× speedup.
tags:
  - CVPR 2026
  - Medical Imaging
  - Oral cancer detection
  - RPA
  - EfficientNetV2B1
  - Design patterns
  - UiPath
date: 2026-05-08
content_hash: d1f4342177d9fc2f
---

# Novel Architecture of RPA in Oral Cancer Lesion Detection

**Conference**: CVPR 2026
**arXiv**: [2603.10928](https://arxiv.org/abs/2603.10928)
**Code**: None
**Area**: Medical Imaging / Automated Workflows
**Keywords**: Oral cancer detection, RPA, EfficientNetV2B1, Design patterns, UiPath

## TL;DR

This paper compares low-code RPA platforms (UiPath, Automation Anywhere) against a Python-based design pattern approach (Singleton + Batch Processing) for oral cancer detection automation. The proposed OC-RPAv2 reduces per-image inference time from 2.5 seconds to 0.06 seconds, achieving a 60–100× speedup.

## Background & Motivation

**State of the Field**: Early and accurate detection of oral cancer is critical for improving patient survival rates. Robotic Process Automation (RPA) has been applied to medical workflow automation, including image processing and laboratory data management. Low-code platforms such as UiPath provide accessible automation interfaces.

**Limitations of Prior Work**: Low-code RPA platforms are inefficient when handling deep learning inference—approximately 78% of execution time is spent on model reloading, activity switching, and data serialization overhead, with only 22% devoted to actual inference computation. Processing 31 oral lesion images with UiPath requires approximately 80 seconds.

**Root Cause**: A fundamental tension exists between the ease-of-use of RPA platforms and the efficient execution of computationally intensive AI inference. Sequential execution paradigms and repeated model loading in low-code environments introduce severe bottlenecks.

**Paper Goals**: To optimize a Python-based inference pipeline using software design patterns, thereby eliminating the execution overhead inherent to RPA platforms.

**Starting Point**: Combining the Singleton pattern (model loaded only once) with the Batch Strategy pattern (batch image processing) within a Python inference pipeline.

**Core Idea**: Applying Singleton to eliminate redundant model loading and Batch Processing to enable batched inference, reducing per-image inference time from 2.5 s (RPA platform) to 0.06 s for oral cancer detection.

## Method

### Overall Architecture

An EfficientNetV2B1 model trained on approximately 3,000 oral clinical images is used for 16-class oral lesion multi-class classification. Four deployment configurations are compared: UiPath RPA, Automation Anywhere RPA, OC-RPAv1 (Python, single-image processing), and OC-RPAv2 (Python + Singleton + Batch Processing).

### Key Designs

1. **EfficientNetV2B1 Classification Model**:

    - **Function**: Classifies oral clinical images into 4 major categories (Healthy, Benign, OPMD, Oral Cancer) with 16 subcategories.
    - **Mechanism**: Uses ImageNet-pretrained EfficientNetV2B1 as the backbone with $224\times224\times3$ input. Stage one freezes the base layers for 15 epochs ($lr=1\times10^{-3}$); stage two unfreezes deeper layers for fine-tuning over 10 epochs ($lr=1\times10^{-5}$). Training employs Adam with categorical cross-entropy, alongside early stopping, ReduceLROnPlateau, and checkpoint saving.
    - **Design Motivation**: EfficientNetV2 offers a favorable balance between compactness and accuracy, making it suitable for real-world clinical deployment.

2. **Singleton + Batch Processing Design Patterns (OC-RPAv2)**:

    - **Function**: Optimizes the Python inference pipeline by eliminating repeated model loading and single-image sequential bottlenecks.
    - **Mechanism**: The Singleton pattern ensures that the EfficientNetV2B1 model is loaded once and remains resident in memory throughout inference. The Batch Strategy pattern organizes all pending images into batches for parallel GPU inference. Processed files are moved to a dedicated directory to ensure data integrity.
    - **Design Motivation**: RPA platforms (UiPath/AA) reload the model on every prediction call, which accounts for 78% of the overhead. Singleton eliminates this bottleneck, while batching further exploits GPU parallelism.

3. **RPA–Python Hybrid Workflow**:

    - **Function**: RPA handles workflow orchestration (file management, logging, error handling), while Python handles computationally intensive tasks.
    - **Mechanism**: UiPath manages the automation pipeline and invokes Python functions for model inference. Try-Catch blocks handle runtime exceptions, and processing is performed on a local secure workstation to protect patient privacy.
    - **Design Motivation**: Combines the process standardization, error tracing, and deployment convenience of RPA with the optimized computation and parallel processing capabilities of Python.

### Loss & Training

The classification model uses categorical cross-entropy. Data augmentation includes flipping, affine transformations, rotation, and color adjustments, with 5 augmentations per sample. Classes with fewer than 200 samples undergo random oversampling. Hierarchical augmentation addresses inter-class imbalance.

## Key Experimental Results

### Main Results

| Configuration | Total Time (31 images) | Avg. Time per Image | vs. Ours |
|---|---|---|---|
| UiPath | 80 s | 2.58 s | 43× slower |
| Automation Anywhere | 75 s | 2.42 s | 40× slower |
| OC-RPAv1 (Python, sequential) | 8.65 s | 0.28 s | 4.7× slower |
| **OC-RPAv2 (Singleton+Batch)** | **1.96 s** | **0.06 s** | **Baseline** |

### Ablation Study

| Optimization | Effect | Notes |
|---|---|---|
| Model reloading eliminated only (Singleton) | ~0.28 s/image | OC-RPAv1; primary overhead removed |
| + Batch parallelism | ~0.06 s/image | OC-RPAv2; further GPU utilization gains |
| RPA overhead analysis | 78% overhead | Model loading, activity switching, data serialization |

### Key Findings
- RPA platforms spend 78% of execution time on non-inference overhead; actual model inference accounts for only 22%.
- The Singleton pattern is the most critical optimization, reducing per-image time from ~2.5 s to ~0.28 s and eliminating approximately 90% of overhead.
- Batch Processing yields an additional ~4.7× speedup through GPU parallelism.
- For a 2,500-image scenario, UiPath requires ~1.8 hours, whereas OC-RPAv2 completes the task in under 3 minutes.

## Highlights & Insights
- **Practical problem relevance**: The efficiency conflict between RPA platforms and deep learning inference is a widespread challenge in real-world medical AI deployment.
- **Elegant solution**: Singleton + Batch Processing are classical design patterns; identifying their value in RPA–AI hybrid systems is a meaningful practical contribution.

## Limitations & Future Work
- The technical contribution is primarily engineering-oriented; the application of design patterns is not a novel methodology, and the work reads more as a system optimization report.
- Evaluation is limited to 31 test images, which is insufficient for statistically reliable conclusions.
- No classification performance metrics (e.g., F1, AUC) are reported; the focus is exclusively on execution efficiency.
- No comparison is made against other efficient inference solutions (e.g., ONNX Runtime, TensorRT).
- Security and privacy compliance considerations are inadequately discussed.

## Related Work & Insights
- **vs. Native UiPath/Automation Anywhere**: Low-code RPA platforms are accessible to non-programmers but computationally inefficient; a hybrid approach represents a better trade-off.
- **vs. CLASEG Framework**: The EfficientNetV2B1 pipeline employed in this paper builds upon the CLASEG framework for 16-class oral lesion classification.
- **vs. LMV-RPA**: Abdellaif et al. also explore RPA+Python hybrid automation; the present work focuses on quantifying the effect of specific design patterns.

## Rating
- **Novelty**: ⭐⭐ Application of design patterns is not a novel contribution; value lies primarily in engineering practice.
- **Experimental Thoroughness**: ⭐⭐ Test set of only 31 images; classification accuracy evaluation is absent.
- **Writing Quality**: ⭐⭐ Contains redundant paragraphs; structure could be more refined.
- **Value**: ⭐⭐⭐ Offers practical reference value for addressing RPA efficiency bottlenecks in medical AI deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[CVPR 2026\] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortali.md)
- [\[CVPR 2026\] The Invisible Gorilla Effect in Out-of-distribution Detection](the_invisible_gorilla_effect_in_out-of-distribution_detection.md)
- [\[CVPR 2026\] Event-Level Detection of Surgical Instrument Handovers in Videos](event_level_detection_of_surgical_instrument_handovers_in_videos.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)

<!-- RELATED:END -->

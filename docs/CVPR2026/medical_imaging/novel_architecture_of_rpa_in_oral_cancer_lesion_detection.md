---
title: >-
  [Paper Note] Novel Architecture of RPA In Oral Cancer Lesion Detection
description: >-
  [CVPR 2026][Medical Imaging][Oral cancer detection] This work integrates software design patterns (Singleton + Batch Processing) into an EfficientNetV2B1-based oral cancer lesion detection Python pipeline…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Oral cancer detection"
  - "RPA automation"
  - "EfficientNetV2"
  - "design patterns"
  - "CNN classification"
date: 2026-05-08
content_hash: 0648eb69c3f3c8ad
---

# Novel Architecture of RPA In Oral Cancer Lesion Detection

**Conference**: CVPR 2026
**arXiv**: [2603.10928](https://arxiv.org/abs/2603.10928)
**Code**: None
**Area**: Medical Imaging / Oral Cancer Detection
**Keywords**: Oral cancer detection, RPA automation, EfficientNetV2, design patterns, CNN classification

## TL;DR

This work integrates software design patterns (Singleton + Batch Processing) into an EfficientNetV2B1-based oral cancer lesion detection Python pipeline, achieving a 60–100× inference speedup over conventional RPA platforms (UiPath/Automation Anywhere) — 0.06 s per image vs. 2.58 s — while maintaining diagnostic accuracy.

## Background & Motivation

**Background**: Early detection of oral cancer is critical to patient survival. Robotic Process Automation (RPA) has been introduced into healthcare to automate repetitive workflows such as image processing, laboratory data management, and patient data analysis. Low-code RPA platforms such as UiPath and Automation Anywhere provide accessible workflow orchestration capabilities.

**Limitations of Prior Work**: (1) Conventional RPA platforms are highly inefficient for computationally intensive AI inference — approximately 78% of processing time is consumed by repeated model loading, activity switching, and data serialization, with only 22% devoted to actual inference. (2) Low-code environments inherently lack support for GPU batch processing and model caching, and serial image processing creates severe throughput bottlenecks. (3) Poor computational resource utilization renders such systems unacceptable in terms of cost and latency for high-throughput clinical scenarios.

**Key Challenge**: A fundamental tension exists between the workflow orchestration strengths of RPA platforms and their computational inefficiency — automated process management must be preserved while substantially improving inference throughput.

**Goal**: To optimize a Python inference pipeline through software engineering design patterns, achieving high-efficiency inference while retaining the workflow orchestration advantages of RPA.

**Key Insight**: Introducing the Singleton (single model load) and Batch Processing (batched inference) design patterns into AI clinical deployment pipelines.

**Core Idea**: Singleton eliminates repeated model-loading overhead + Batch Processing exploits GPU parallelism = 60–100× speedup.

## Method

### Overall Architecture

The system comprises two parallel pipelines: OC-RPAv1 (a basic Python pipeline performing per-image processing) and OC-RPAv2 (an optimized pipeline incorporating Singleton + Batch Processing). UiPath manages the automation workflow and invokes Python functions for inference. Both pipelines share the same CNN model.

### Key Designs

1. **Singleton Design Pattern (Eliminating Repeated Model Loading)**:

    - **Function**: Ensures the CNN model is loaded only once and remains resident in memory throughout the entire lifecycle.
    - **Mechanism**: In conventional RPA pipelines, the model is re-instantiated on every prediction call. The Singleton pattern decouples model loading from inference and centralizes model lifecycle management.
    - **Design Motivation**: Model loading and data serialization account for approximately 78% of total processing time in traditional RPA pipelines, representing the dominant performance bottleneck. Eliminating redundant loading is the single most impactful optimization.

2. **Batch Processing Design Pattern (GPU Parallel Inference)**:

    - **Function**: Groups multiple images into a batch and performs inference in a single forward pass.
    - **Mechanism**: Leverages GPU parallel computation to reduce per-image kernel launch and memory transfer overhead. Upon completion, results for each image are automatically logged and the image is moved to a dedicated directory to ensure data integrity.
    - **Design Motivation**: Further compresses inference time beyond the Singleton baseline (from 0.28 s/image in OC-RPAv1 to 0.06 s/image in OC-RPAv2, an additional 4.7× speedup).

3. **CNN Classification Model (EfficientNetV2B1)**:

    - **Function**: 16-class classification of oral lesions.
    - **Mechanism**: An ImageNet-pretrained EfficientNetV2B1 serves as the backbone with 224×224×3 input. The final layer is replaced with a softmax fully connected layer. Two-stage training is employed: the backbone is frozen for 15 epochs (lr = 1e-3), followed by partial unfreezing for fine-tuning over 10 epochs (lr = 1e-5).
    - **Dataset**: Approximately 3,000 clinical oral images spanning 4 macro-categories (Healthy / Benign / OPMD / Oral Cancer) with 16 sub-classes. Five augmentation strategies are applied via Albumentations; classes with fewer than 200 samples undergo random oversampling.

### Loss & Training

- **Loss function**: Categorical cross-entropy
- **Optimizer**: Adam, batch size = 32
- **Data split**: Stratified sampling at 70% / 15% / 15%
- **Training techniques**: Early stopping, model checkpointing (best validation accuracy), and ReduceLROnPlateau (halving the learning rate upon loss plateau)

## Key Experimental Results

### Main Results (Inference Efficiency Comparison on 31 Test Images)

| Platform / Method | Total Time (31 images) | Avg. Time per Image | Relative Speedup |
|---|---|---|---|
| UiPath | 80 s | 2.58 s | 1× (baseline) |
| Automation Anywhere | 75 s | 2.42 s | 1.07× |
| OC-RPAv1 (Basic Python) | 8.65 s | 0.28 s | 9.2× |
| **OC-RPAv2 (Python + Design Patterns)** | **1.96 s** | **0.06 s** | **43×** |

### Ablation Study / Efficiency Analysis

| Analysis Dimension | Key Data | Remarks |
|---|---|---|
| RPA platform overhead analysis | ~78% spent on non-inference operations | Model loading / data serialization are the primary bottleneck |
| Singleton contribution | v1 (0.28 s) vs. RPA (2.58 s) | Eliminating repeated loading yields 9.2× speedup |
| Batch Processing contribution | v2 (0.06 s) vs. v1 (0.28 s) | GPU parallelism delivers a further 4.7× speedup |
| Scalability estimate | 2,500 images: UiPath ≈ 1.8 h, v2 < 3 min | 40× operational efficiency improvement |

### Key Findings

- RPA platforms are severely inefficient for computationally intensive tasks, with the majority of time consumed by non-inference overhead.
- The Singleton pattern's elimination of repeated model loading is the largest single source of performance gain (~9×).
- The introduction of design patterns does not affect diagnostic accuracy; only execution efficiency is improved.
- A hybrid approach combining Python computation with RPA workflow orchestration represents the recommended best practice.

## Highlights & Insights

- This work provides the first systematic quantification of the efficiency bottleneck of conventional RPA platforms in AI inference scenarios (78% overhead attributable to non-inference operations).
- Singleton and Batch Processing design patterns are introduced into RPA-based medical image analysis pipelines.
- A reusable hybrid RPA + Python automation pattern is demonstrated.
- The conclusion, while straightforward, carries practical significance: in clinical AI deployment, engineering optimization may contribute as much value as algorithmic improvement.

## Limitations & Future Work

- **Extremely small test set**: Only 31 test images are used, severely undermining statistical credibility.
- **Absent accuracy comparison**: No classification accuracy, precision, or recall metrics are reported, and no diagnostic performance comparison across methods is provided.
- **No model-level contribution**: EfficientNetV2B1 is adopted directly without architectural modification or domain-specific adaptation for oral lesions.
- **Writing quality**: The paper is loosely structured, contains repeated passages, and some citations are insufficiently rigorous.
- **Limited clinical depth**: The work focuses exclusively on inference speed without addressing interpretability, uncertainty quantification, or other clinically critical requirements.
- Future work could explore the integration of additional design patterns such as Factory, Adapter, and Observer.

## Related Work & Insights

- **vs. Abdellaif et al. (LMV-RPA)**: LMV-RPA similarly explored the idea of augmenting RPA with Python; this paper further quantifies the acceleration attributable to design patterns (60–100×).
- **vs. CLASEG Framework**: CLASEG provides a deep learning baseline for multi-class oral lesion classification and segmentation; this paper directly reuses its model architecture.
- **Essential Positioning**: This work constitutes applied software engineering research (design patterns in AI deployment), rather than algorithmic innovation.
- **Insight**: The practical adoption of clinical AI systems demands not only model accuracy but also engineering-level deployment efficiency — Singleton and Batch Processing represent the most fundamental yet most impactful optimizations available.

## Rating

⭐⭐ (2/5)

- **Novelty** ⭐⭐: Application of established design patterns to an RPA pipeline; no algorithmic innovation.
- **Experimental Thoroughness** ⭐⭐: Test scale is extremely small (31 images); accuracy metric comparisons are absent.
- **Writing Quality** ⭐⭐: Loosely structured with repeated passages and inconsistent citations.
- **Value** ⭐⭐⭐: Offers practical reference for engineering-level clinical AI deployment, but academic contribution is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ov.md)
- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[CVPR 2026\] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortali.md)
- [\[CVPR 2026\] The Invisible Gorilla Effect in Out-of-distribution Detection](the_invisible_gorilla_effect_in_out-of-distribution_detection.md)
- [\[CVPR 2026\] Event-Level Detection of Surgical Instrument Handovers in Videos](event_level_detection_of_surgical_instrument_handovers_in_videos.md)

</div>

<!-- RELATED:END -->

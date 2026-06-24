---
title: >-
  [Paper Note] Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels
description: >-
  [CVPR 2025][Multimodal VLM][Visual grounding] Ground-V is constructed as a dataset containing 500,000 instruction-segmentation pairs to systematically address five major challenges in real-world referring expression segmentation (hallucinated references, multi-object targeting, reasoning, multi-granularity, and part-level references). After training, the VLM achieves an N-Acc improvement of over 20% compared to the previous SOTA on gRefCOCO.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Visual grounding"
  - "referring expression segmentation"
  - "dataset construction"
  - "complex instructions"
  - "pixel-level grounding"
date: 2026-05-08
content_hash: ba9fb98b80d583a9
---

# Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels

**Conference**: CVPR 2025  
**arXiv**: [2505.13788](https://arxiv.org/abs/2505.13788)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Visual grounding, referring expression segmentation, dataset construction, complex instructions, pixel-level grounding

## TL;DR

Ground-V is constructed as a dataset containing 500,000 instruction-segmentation pairs to systematically address five major challenges in real-world referring expression segmentation (hallucinated references, multi-object targeting, reasoning, multi-granularity, and part-level references). After training, the VLM achieves an N-Acc improvement of over 20% compared to the previous SOTA on gRefCOCO.

## Background & Motivation

**Background**:
Large Vision-Language Models (VLMs) have demonstrated powerful capabilities in general multimodal tasks. Recent works (such as LISA and PSALM) have begun to achieve reasoning-driven segmentation by learning special grounding tokens within VLMs and integrating them with SAM.

**Limitations of Prior Work**:
1. Current VLM-based segmentation models remain highly unreliable under complex instructions. For instance, in an image containing apples of various colors, a model instructed to segment a red apple might incorrectly include other colors.
2. As instructions become more complex (e.g., "segment the bitten red apple next to the ceramic bowl"), models tend to overlook contextual details.
3. The root cause lies in training data: most visual grounding datasets only contain simple and direct referring expressions, creating a significant gap with diverse human natural language descriptions.

**Key Challenge**:
Although VLMs possess strong multimodal understanding capabilities, they fail to translate this understanding into precise pixel-level localization under complex instructions due to the simplicity of existing training data.

**Goal**:
To bridge the gap between VLM understanding capabilities and grounding precision by constructing large-scale instruction-segmentation data tailored for complex scenarios.

**Key Insight**:
Systematically identifying five key challenges in real-world referring expression segmentation and designing automated data generation pipelines for each dimension.

**Core Idea**:
Leveraging knowledge distillation (using Claude as a teacher VLM) to automatically generate high-quality instruction-segmentation datasets covering five key challenge dimensions. Directly integrating this data into existing model training significantly boosts performance.

## Method

### Overall Architecture

The core of Ground-V is an automated data generation workflow: (1) identifying five real-world challenges; (2) designing few-shot prompts for each dimension; (3) utilizing Claude 3 Sonnet to generate instruction-response pairs and linking them to existing pixel-level annotations (COCO 2017); and (4) human validation to verify the test set. In total, 50K images and approximately 480,000 instruction-segmentation pairs are generated.

### Key Designs

1. **五大挑战维度的数据设计**:

    - Function: Systematically cover key difficulties in real-world referring expression segmentation.
    - Mechanism:
      - **Multi-granularity**: The same object can be described at different levels of abstraction (e.g., "Corgi" $\rightarrow$ "dog" $\rightarrow$ "pet" $\rightarrow$ "animal").
      - **Multi-object**: Instructions referring to more than 5 objects simultaneously.
      - **Hallucinated Reference**: Instructions describing objects, attributes, or relations not present in the image, where the model should refuse to segment.
      - **Reasoning**: Abstract instructions requiring common-sense reasoning (e.g., "the fruit rich in antioxidants").
      - **Part Reference**: Target components that are parts of an object (e.g., "the button of the microwave").
    - Design Motivation: To cover the complete instruction spectrum from simple to complex, and from concrete to abstract.

2. **自动化数据生成管道**:

    - Function: Generate large-scale instruction-response pairs with pixel-level annotations.
    - Mechanism: Handcrafting 3-shot examples for each dimension, using Claude 3 Sonnet to generate new instruction-response pairs based on the query image, and associating them with existing COCO segmentation annotations. For the test set, Claude 3.5 Sonnet is used for secondary validation, followed by human audit.
    - Design Motivation: Automating data generation utilizing the powerful language capabilities of teacher VLMs to minimize manual annotation requirements.

3. **无缝集成到现有模型**:

    - Function: Plug-and-play integration of Ground-V as extra training data.
    - Mechanism: Keeping the original training hyperparameters and evaluation settings of LISA/PSALM unchanged, and simply incorporating Ground-V into the training data.
    - Design Motivation: To validate the generality and effectiveness of the data rather than proposing a new architecture.

### Loss & Training

- No changes are made to the original training strategies and loss functions of LISA/PSALM.
- Ground-V instruction-segmentation pairs are simply added to the training data.
- LISA is based on the LLaVA + Vicuna-7B + CLIP + SAM architecture.
- PSALM is based on the Phi-1.5 + Swin Transformer + Mask2Former architecture.

## Key Experimental Results

### Main Results

**RefCOCO/RefCOCO+/RefCOCOg (cIoU)**

| Method | RefCOCO val | RefCOCO+ val | RefCOCOg val | Average |
|------|------------|-------------|-------------|------|
| LISA | 70.2 | 59.2 | 63.2 | 64.4 |
| LISA+G5 | **73.9** (+3.7) | **63.1** (+3.9) | **64.9** (+1.7) | **66.6** (+2.2) |
| PSALM | 83.6 | 72.9 | 73.8 | 77.1 |
| PSALM+G5 | **83.9** | **73.1** | **74.8** | **77.3** |

**gRefCOCO (Multi-object referring expression segmentation, gIoU / N-Acc)**

| Method | val gIoU | val N-Acc | testA gIoU | testB gIoU | Average gIoU | Average N-Acc |
|------|----------|----------|------------|------------|----------|-----------|
| LISA | 32.2 | 2.7 | 48.5 | 39.7 | 40.1 | 4.7 |
| LISA+G5 | **46.7** | **36.4** | **63.2** | **51.3** | **53.7** | **40.1** |
| PSALM | 43.3 | 27.7 | 54.5 | 52.5 | 50.1 | 24.5 |
| PSALM+G5 | **64.6** | **83.3** | **74.5** | **72.7** | **70.6** | **83.7** |

### Ablation Study

The five dimensions of Ground-V provide clear contributions to different test subsets, and removing any single dimension leads to performance degradation on the corresponding subset. Hallucination mitigation data is particularly crucial for improving overall robustness.

### Key Findings

1. **Data is Core**: Without modifying the model architectures, simply adding Ground-V training data yields an average improvement of 4.4% gIoU for LISA and 7.9% gIoU for PSALM.
2. **Breakthrough Improvements on gRefCOCO**: PSALM+G5's N-Acc surges from 24.5% to 83.7%, outperforming the Prev. SOTA by over 20%. This demonstrates that the main bottleneck of previous models lies in the data rather than the architecture.
3. **Significantly Enhanced Hallucination Handling**: By introducing negative training samples across three categories of hallucinations (object, attribute, and relation), the model learns to "refuse to segment" non-existent targets.
4. **Efficient and Scalable Data Generation Pipeline**: Relying on COCO annotations and automated generation via Claude, the process requires no additional pixel-level annotations.

## Highlights & Insights

1. **Clear Problem Definition**: Systematically decomposes complex instruction segmentation into five orthogonal dimensions, with distinct data generation strategies for each.
2. **Data-Driven Methodology**: Demonstrates that high-quality training data is far more vital than model architecture innovations for VLM grounding tasks.
3. **Introduction of the Hallucination Dimension**: Systematically introduces negative samples of hallucinated references into visual grounding training for the first time, teaching the model to "say no".
4. **Substantial Boost in N-Acc**: The leap from 24.5% to 83.7% indicates that previous models failed almost entirely in multi-object rejection scenarios, purely due to the absence of corresponding training data.
5. **Human-Annotated Test Set**: Consists of 5,000 images and 57,000 instructions, independently verified by two annotators, with 23.1% of the data discarded due to sub-standard quality.

## Limitations & Future Work

1. Data generation relies on Claude 3 Sonnet, which may exhibit unstable generation quality in certain complex scenarios.
2. The image source is restricted to COCO 2017, limiting scene diversity.
3. Automatically verifying correctness for the reasoning dimension is difficult, potentially introducing noise.
4. Part-level reference data relies on the PACO dataset, limiting the coverage of object categories.
5. Scaling up the data further (e.g., to millions of samples) to explore potential gains remains unexplored.

## Related Work & Insights

- **LISA/PSALM**: These represent two typical architectures for VLM-based segmentation. LISA relies on learning a single segmentation token + SAM decoding, while PSALM is based on Mask2Former multi-token output.
- **ReasonSeg**: The first to propose the concept of reasoning-driven segmentation, but it only supports single objects and provides 1.2K data samples.
- **MUSE**: Supports reasoning and multi-object scenarios but lacks hallucination, part-level, and multi-granularity dimensions, providing 214K data samples.
- **Insight**: In the era of VLMs, data diversity and coverage might be more critical than absolute data scale.

## Rating

- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Aria-UI: Visual Grounding for GUI Instructions](../../ACL2025/multimodal_vlm/aria-ui_visual_grounding_for_gui_instructions.md)
- [\[ICLR 2026\] Teaching VLMs to Admit Uncertainty in OCR from Lossy Visual Inputs](../../ICLR2026/multimodal_vlm/teaching_vlms_to_admit_uncertainty_in_ocr_from_lossy_visual_inputs.md)
- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)
- [\[CVPR 2025\] Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](teaching_large_language_models_to_regress_accurate_image_quality_scores_using_sc.md)
- [\[ICLR 2026\] Talking Points: Describing and Localizing Pixels](../../ICLR2026/multimodal_vlm/talking_points_describing_and_localizing_pixels.md)

</div>

<!-- RELATED:END -->

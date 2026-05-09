---
title: >-
  [Paper Note] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset
description: >-
  [CVPR 2026][Medical Imaging][Chest X-ray] This paper introduces instruction-guided lesion segmentation (ILS) for chest X-rays, constructs the first large-scale automatically generated instruction-answer dataset MIMIC-ILS (1.1M samples, 192K images, 91K masks), and trains the ROSALIA model to achieve gIoU of 71.2% and null-target accuracy of 91.8%, substantially outperforming existing general-purpose and medical segmentation models.
tags:
  - CVPR 2026
  - Medical Imaging
  - Chest X-ray
  - lesion segmentation
  - instruction-guided
  - automatic dataset construction
  - vision-language model
date: 2026-05-08
content_hash: 1585925cba88a52c
---

# Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset

**Conference**: CVPR 2026  
**arXiv**: [2511.15186](https://arxiv.org/abs/2511.15186)  
**Code**: [GitHub](https://github.com/checkoneee/ROSALIA)  
**Area**: Medical Imaging  
**Keywords**: Chest X-ray, lesion segmentation, instruction-guided, automatic dataset construction, vision-language model

## TL;DR

This paper introduces instruction-guided lesion segmentation (ILS) for chest X-rays, constructs the first large-scale automatically generated instruction-answer dataset MIMIC-ILS (1.1M samples, 192K images, 91K masks), and trains the ROSALIA model to achieve gIoU of 71.2% and null-target accuracy of 91.8%, substantially outperforming existing general-purpose and medical segmentation models.

## Background & Motivation

Chest X-ray (CXR) is one of the most common medical imaging examinations, and lesion localization and boundary delineation represent core tasks for radiologists — yet these are labor-intensive and demand high levels of clinical expertise.

Existing CXR lesion segmentation faces two major bottlenecks:
1. **Limited annotation scale**: Existing datasets (VinDr-CXR with 15K images, SIIM-ACR with 13K images) rely on expert manual annotation, restricting scale, and most provide only bounding boxes or masks for a single lesion type.
2. **High barrier for user input**: Existing text-guided segmentation methods require users to provide expert-level detailed descriptions (e.g., "bilateral pulmonary infection with two infected regions…"), rendering them inaccessible to non-expert users.

**Root Cause**: How to generate high-quality lesion masks and instruction-answer pairs at scale without manual annotation, while supporting simple and accessible user instructions?

**Starting Point**: The paper leverages existing image-report paired data in MIMIC-CXR, extracting spatial and textual information from images and reports through a multimodal automated pipeline to produce a fully automatically annotated large-scale ILS dataset.

## Method

### Overall Architecture

The system consists of two major stages:
1. **Lesion mask generation**: Automatically generating lesion segmentation masks from CXR images and radiology reports.
2. **Instruction-answer pair generation**: Constructing diverse training samples based on information extracted in the previous stage.

The ROSALIA model is built upon the LISA architecture, integrating a VLM (LLaVA) and SAM for end-to-end training.

### Key Designs

1. **Multimodal Automatic Mask Generation Pipeline**:
    - **Function**: Automatically generates high-quality lesion segmentation masks from raw, unannotated CXR images.
    - **Mechanism**: A four-step cascaded pipeline —
      1. **Report structuring**: An LLM converts abnormality descriptions in radiology reports into sextuple representations (entity, sentence index, existence, certainty, location, lesion type), with locations mapped to standard anatomical labels.
      2. **Spatial information extraction**: Three visual models are applied in parallel — RadEdit (a diffusion model generating an anomaly map $\mathcal{A}$ by differencing the original image against a synthesized "lesion-free" image), CXAS (an anatomical segmentation model providing anatomical masks $\{\mathcal{M}_i\}$), and YOLO (a lesion detector producing bounding boxes $\{\mathcal{B}_j\}$).
      3. **Mask generation**: High-quality candidate boxes are selected via four-condition filtering (anatomical overlap c1 + detection confidence c2 + anomaly signal ratio c3 + minimum size c4); connected components intersecting with selected boxes are then extracted and refined.
      4. **Location verification**: Confirms whether the generated mask successfully localizes the region described in the report, while flagging blank locations for negative sample generation.
    - **Design Motivation**: Multimodal cross-validation ensures mask quality — textual information specifies "where," the visual anomaly map indicates "what is abnormal," and the detection model provides "boundaries." The four-condition filtering effectively eliminates false positives.

2. **Instruction-Answer Pair Generation System**:
    - **Function**: Automatically constructs diverse training samples based on localization information.
    - **Mechanism**: Three instruction types are supported —
        - **Basic instructions**: Specify lesion type and location (e.g., "Segment the pneumonia in the right lung"), generated only when the mask is successfully localized.
        - **Global instructions**: Specify only the lesion type (e.g., "Segment the opacity"), generated only when the localized position fully matches the position reported in the report.
        - **Lesion reasoning instructions**: Require the model to predict the specific subtype of opacity, replacing pneumonia/atelectasis/edema with "opacity."
    - **Negative sample generation strategy**: Lesion types not mentioned or explicitly negated in the report, as well as blank locations substituted for positive-sample locations, are used to construct negative samples.
    - **Design Motivation**: The three instruction types cover a spectrum of user needs ranging from specific to general. The dynamic generation strategy ensures that only valid instruction-answer pairs are produced, avoiding inconsistencies.

3. **ROSALIA Model Architecture**:
    - **Function**: Generates lesion segmentation masks and textual descriptions conditioned on user instructions.
    - **Mechanism**: Built upon LISA-7B, the VLM (LLaVA) processes image and instruction inputs to produce a special [SEG] token and textual description. The hidden embedding of the [SEG] token is passed to SAM-H's mask decoder to generate the final segmentation mask.
    - **Training strategy**: LoRA fine-tuning of the VLM (rank=128, alpha=256); full fine-tuning of the mask decoder. Trained for 15 epochs with AdamW, batch size 256, and a 1:1 positive-to-negative sample ratio.

### Loss & Training

$$\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + \lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice}$$

- $\mathcal{L}_{txt}$: Autoregressive cross-entropy loss (text generation), $\lambda_{txt}=0.5$
- $\mathcal{L}_{bce}$: Binary cross-entropy loss (segmentation), $\lambda_{bce}=5$
- $\mathcal{L}_{dice}$: Dice loss (segmentation, computed on positive samples only), $\lambda_{dice}=1$

## Key Experimental Results

### Main Results

| Model | gIoU | cIoU | N-Acc. | Note |
|-------|------|------|--------|------|
| LISA-7B | 8.3% | 12.8% | 0.7% | General domain |
| LISA-13B | 8.9% | 12.2% | 0.0% | General domain |
| Text4Seg | 6.1% | 10.3% | 20.6% | General domain |
| BiomedParse | 23.8% | 18.5% | 0.6% | Medical domain |
| RecLMIS | 22.4% | 19.5% | 0.0% | Medical domain |
| **ROSALIA (Ours)** | **71.2%** | **75.6%** | **91.8%** | Trained on MIMIC-ILS |

### Per-Lesion Performance

| Lesion Type | gIoU | cIoU | N-Acc. |
|-------------|------|------|--------|
| Cardiomegaly | 89.0% | 89.0% | 85.8% |
| Pneumonia | 57.2% | 60.4% | 97.1% |
| Atelectasis | 60.2% | 58.7% | 91.7% |
| Opacity | 60.5% | 64.2% | 85.0% |
| Consolidation | 61.9% | 65.6% | 91.2% |
| Edema | 64.8% | 66.6% | 92.2% |
| Pleural Effusion | 60.3% | 59.6% | 90.4% |

### Ablation Study — Dataset Quality Evaluation

| Expert | Overall Acceptance | Positive Acceptance | Negative Acceptance |
|--------|--------------------|---------------------|---------------------|
| Expert A | 96.1% | 95.6% | 96.5% |
| Expert B | 97.2% | 96.0% | 98.3% |
| Expert C | 98.7% | 99.8% | 97.8% |
| Expert D | 97.6% | 96.9% | 98.2% |
| **Overall** | **96.4%** | 90.1% | 97.7% |

### Key Findings

- Existing general-purpose and medical-domain segmentation models systematically fail on the ILS task, achieving gIoU below 24% and near-zero null-target accuracy (N-Acc ≈ 0%).
- The fully automatically generated dataset received an overall acceptance rate of 96.4% from four radiation oncology experts.
- Text response accuracy is 94.4%, with basic instructions achieving the highest score of 96.8%; lesion reasoning instructions reach 84.8%, leaving room for improvement.
- Cardiomegaly achieves the best segmentation (gIoU 89.0%) due to the use of cardiac masks as annotations; focal lesions such as pneumonia score somewhat lower.

## Highlights & Insights

- The fully automatic dataset construction pipeline is the core contribution, achieving annotation quality comparable to manual labeling through multimodal cross-validation.
- The ILS task definition is clinically practical: it supports simple user instructions rather than expert-level descriptions and handles null-target detection ("no lesion found").
- The dataset scale (1.1M samples) is 10–100× larger than existing CXR segmentation datasets.
- The use of RadEdit to generate anomaly maps is elegant — a diffusion model synthesizes a "normal" image, and differencing localizes abnormal regions.

## Limitations & Future Work

- **Annotation quality**: The positive-sample acceptance rate of 90.1% falls below the 97.7% for negative samples, indicating that the precision of positive-sample annotations requires further improvement.
- Only 7 major lesion types are covered; CXR encompasses a broader range of fine-grained abnormalities.
- Accuracy on lesion reasoning tasks (opacity → specific subtype) is relatively low at 75.1%.
- The pipeline depends on three pretrained models (RadEdit, CXAS, YOLO); failure of any single model may degrade overall quality.
- Validation is conducted solely on MIMIC-CXR; CXR style variation across institutions may limit generalizability.

## Related Work & Insights

- **vs. BiomedParse**: Although a medical-domain model, it only supports class-label prompts and cannot handle instruction-level inputs or null-target detection.
- **vs. RecLMIS**: Requires users to provide expert-level descriptions (e.g., "bilateral pulmonary infection…"), imposing a high barrier to use.
- **vs. LISA**: ROSALIA is built on the LISA architecture, but fine-tuning on MIMIC-ILS boosts performance from 8.3% to 71.2%, demonstrating the critical importance of task-specific data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Both the fully automatic dataset construction pipeline and the ILS task formulation are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple baselines, per-lesion evaluation, and expert quality validation are provided, though cross-dataset generalization experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, the pipeline is described in detail, and figures and tables are comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ The dataset scale and the public release of code and data offer high practical value to the community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening](xseg_a_large-scale_x-ray_contraband_segmentation_benchmark_for_real-world_securi.md)
- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](../../ICLR2026/medical_imaging/afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](../../NeurIPS2025/medical_imaging/cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)

<!-- RELATED:END -->

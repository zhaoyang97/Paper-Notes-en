---
title: >-
  [Paper Note] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis
description: >-
  [NeurIPS 2025][Medical Imaging][Rheumatoid Arthritis] RAM-W600 is the first publicly available multi-task wrist conventional radiograph dataset, comprising 1…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Rheumatoid Arthritis"
  - "Carpal Bone Segmentation"
  - "Bone Erosion Scoring"
  - "Dataset"
  - "Instance Segmentation"
date: 2026-05-08
content_hash: 73c506986b697632
---

# RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis

**Conference**: NeurIPS 2025
**arXiv**: [2507.05193](https://arxiv.org/abs/2507.05193)  
**Code**: [GitHub](https://github.com/YSongxiao/RAM-W600)  
**Area**: Medical Imaging
**Keywords**: Rheumatoid Arthritis, Carpal Bone Segmentation, Bone Erosion Scoring, Dataset, Instance Segmentation

## TL;DR

RAM-W600 is the first publicly available multi-task wrist conventional radiograph dataset, comprising 1,048 images and supporting two clinically relevant tasks: carpal bone instance segmentation and SvdH bone erosion (BE) scoring, accompanied by comprehensive benchmarking.

## Background & Motivation

Rheumatoid arthritis (RA) is a prevalent autoimmune disease in which the wrist joint serves as a central region for diagnosis. Conventional radiography (CR) is widely used in clinical practice for disease screening and assessment due to its low cost and accessibility. However, **computer-aided diagnosis (CAD) research for carpal bones remains severely constrained**, with the primary bottleneck being the extreme difficulty of obtaining high-quality instance-level annotations:

**Anatomical complexity**: The wrist comprises multiple small bones with narrow joint spaces, complex overlapping structures, and frequent occlusions, requiring substantial anatomical expertise for accurate annotation.

**Pathological interference**: RA progression introduces osteophytes, bone erosions (BE), and even bony ankylosis, which alter bone morphology and further increase annotation difficulty.

**Limitations of existing datasets**: Publicly available hand radiograph datasets either lack pixel-level segmentation annotations for the wrist or provide incomplete BE scores, rendering them insufficient for RA research.

Existing work has predominantly focused on carpal bone segmentation in CT/MRI modalities, with small-scale datasets (mostly private, ranging from tens to hundreds of images). CR-based carpal bone segmentation research is scarce, particularly lacking publicly available datasets suited to complex pathological conditions.

The paper's **Key Insight** is to construct a multi-task, multi-center carpal bone CR benchmark dataset that simultaneously covers two clinically critical tasks—instance segmentation and SvdH BE scoring—thereby lowering the barrier to entry for RA wrist research.

## Method

### Overall Architecture

RAM-W600 does not propose a new model; rather, it contributes a **dataset and benchmark**. The dataset design encompasses a complete pipeline: data collection → image preprocessing → expert annotation → data splitting → benchmark evaluation.

### Key Designs

1. **Dataset Composition and Annotation Framework**

   The dataset includes 1,048 wrist CR images from 388 patients across 6 medical centers. Among these, 618 images provide pixel-level instance segmentation annotations covering 14 carpal bone categories, and 800 images include SvdH BE scores across 6 joint surfaces (4,800 scores in total).

   The annotation framework consists of three levels:
   - **Anatomical structure annotation**: Precise contour delineation of 14 carpal bones (MC1–5, Tr, Tz, Sca, Lu, Cap, Ham, Tri, Radius, Ulna), with each bone independently annotated using a multi-label strategy.
   - **Bone location annotation**: ROI annotation for 6 joint regions of interest defined by the SvdH scoring system.
   - **SvdH BE scoring annotation**: BE severity scoring for 6 key joint surfaces.

2. **Data Diversity and Quality Control**

   Data sources include 3 Japanese medical institutions (HMCRD, SCGH, HU) and 3 public datasets (DHA, BTXRD, FA), covering 207 RA patients and 181 non-RA patients. Image parameters are managed via the DICOM standard at a resolution of 0.15 mm/pixel (internal cohort).

   The annotation pipeline was supervised by senior radiologists and employed rigorous review procedures to ensure quality. The dataset design emphasizes stratified splitting of BE and non-BE cases to ensure representativeness across training and test sets.

3. **Benchmark Evaluation Protocol**

    - **Segmentation task**: Evaluation of 13 supervised models (UNet, DeepLabV3+, TransUNet, SwinUMamba, etc.) and 3 foundation models (SAM, MedSAM), using DSC, NSD, VOE, MSD, and RAVD metrics.
    - **BE classification task**: Evaluation of 7 classification models (MobileViT, ResNet, MedMamba, etc.), using BACC, F1, DOR, ACC, SEN, SPC, and PRE metrics.

### Loss & Training

All benchmark experiments adopt the AdamW optimizer (weight decay = 1e-2) with cosine annealing learning rate scheduling. For the segmentation task: initial learning rate 1e-4, 100 epochs, batch size 8; for the classification task: initial learning rate 1e-6, 100 epochs, batch size 16. All experiments were repeated 5 times with fixed seeds on an RTX 4090.

## Key Experimental Results

### Main Results — Segmentation

| Model | DSC (%) ↑ | NSD (%) ↑ | VOE (%) ↓ | MSD (pix) ↓ | Params |
|-------|-----------|-----------|-----------|-------------|--------|
| SwinUMamba | **97.75** | **90.71** | **4.35** | **1.06** | 59.89M |
| TransUNet | 97.62 | 89.48 | 4.60 | 1.05 | 105.91M |
| UMambaEnc | 97.56 | 89.10 | 4.71 | 1.11 | 4.58M |
| Unet++ | 97.33 | 86.99 | 5.15 | 1.36 | 2.41M |
| SAM (box) | 88.74 | 64.40 | 18.45 | 4.25 | 641.09M |
| MedSAM (box) | 85.07 | 38.81 | 25.15 | 5.97 | 93.74M |

### Main Results — BE Classification

| Model | BACC (%) | F1 (%) | DOR | SEN (%) | SPC (%) |
|-------|----------|--------|-----|---------|---------|
| MobileViT | **52.64** | 11.85 | 1.82 | 21.06 | 84.23 |
| EfficientFormer | 50.63 | **12.40** | 1.06 | **27.90** | 73.37 |
| MedMamba | 50.83 | 6.91 | **5.89** | 8.94 | 92.73 |
| ConvKAN | 49.26 | 3.49 | 0.44 | 3.82 | **94.70** |

### Key Findings

1. **Segmentation task**: Leading models achieve strong DSC performance (up to 97.75%), yet NSD scores remain relatively low (up to 90.71%), indicating that **precise boundary delineation remains the primary bottleneck**. Significant DSC differences between BE and non-BE groups (p < 0.05–0.001) confirm that bone erosion negatively affects segmentation performance.
2. **Classification task**: All models yield low BACC and F1 scores (maximum 52.64% and 12.40%, respectively), reflecting the extreme difficulty of the task. Models exhibit a pronounced bias toward the non-BE class (high specificity / low sensitivity), with **severe class imbalance being the core challenge**.
3. **Foundation model gap**: SAM and MedSAM substantially underperform supervised models on carpal bone segmentation, demonstrating that general-purpose foundation models still have considerable room for improvement in fine-grained medical image segmentation.

## Highlights & Insights

- RAM-W600 is the first publicly available carpal bone instance segmentation dataset, covering both segmentation and BE scoring tasks, thereby filling a critical data gap in RA wrist CAD research.
- Multi-center data sourcing (6 institutions) enhances data diversity and the reliability of generalization evaluation.
- Benchmark results clearly reveal the shortcomings of current models with respect to bone overlap, blurred boundaries, and erosion-induced deformation, providing concrete directions for future research.
- Mamba-based architectures (SwinUMamba, UMambaEnc) demonstrate a favorable balance between performance and parameter efficiency.

## Limitations & Future Work

- RA cases are predominantly sourced from a single geographic region in Japan, resulting in relatively high demographic homogeneity that may limit model generalizability across different ethnicities and regions.
- The SvdH BE score distribution is severely imbalanced, with high-score cases being extremely scarce (scores of 3 and 5 are nearly absent), which impairs the training and evaluation of fine-grained scoring models.
- Only binary classification (presence vs. absence of BE) is evaluated; multi-grade regression scoring is not explored.
- Downstream tasks aligned with clinical workflows (e.g., JSN progression quantification, longitudinal monitoring) are not investigated.

## Related Work & Insights

- This dataset can inspire multi-task learning frameworks that jointly address carpal bone segmentation and BE detection.
- The dataset construction methodology is extensible to RA assessment of other joints (fingers, feet).
- The extreme class imbalance in the BE classification task provides a valuable testbed for few-shot and imbalanced learning research in medical imaging.

## Rating

- Novelty: ⭐⭐⭐⭐ First publicly available dataset of its kind, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad benchmark coverage, though multi-grade scoring and downstream task validation are absent.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with thorough statistical analysis.
- Value: ⭐⭐⭐⭐⭐ High practical utility for CAD research in the RA domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[ICCV 2025\] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users](../../ICCV2025/medical_imaging/progait_a_multi-purpose_video_dataset_and_benchmark_for_transfemoral_prosthesis_.md)
- [\[NeurIPS 2025\] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders](dermacon-in_a_multi-concept_annotated_dermatological_image_dataset_of_indian_ski.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)

</div>

<!-- RELATED:END -->

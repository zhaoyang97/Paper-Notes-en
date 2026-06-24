---
title: >-
  [Paper Note] Novel Architecture of RPA In Oral Cancer Lesion Detection
description: >-
  [CVPR2025][Medical Imaging][Oral cancer detection] This paper integrates Singleton and Batch Processing design patterns into a Python-based RPA automation pipeline, combining them with the EfficientNetV2B1 model for oral cancer lesion detection, achieving a 60-100× inference speedup compared to traditional RPA platforms such as UiPath and Automation Anywhere.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Oral cancer detection"
  - "RPA"
  - "EfficientNetV2B1"
  - "Design patterns"
  - "Batch processing"
date: 2026-05-08
content_hash: 06ecaa12a330d0dd
---

# Novel Architecture of RPA In Oral Cancer Lesion Detection

**Conference**: CVPR2025  
**arXiv**: [2603.10928](https://arxiv.org/abs/2603.10928)  
**Code**: To be confirmed  
**Area**: Medical Imaging  
**Keywords**: Oral cancer detection, RPA, EfficientNetV2B1, Design patterns, Batch processing

## TL;DR

This paper integrates Singleton and Batch Processing design patterns into a Python-based RPA automation pipeline, combining them with the EfficientNetV2B1 model for oral cancer lesion detection, achieving a 60-100× inference speedup compared to traditional RPA platforms such as UiPath and Automation Anywhere.

## Background & Motivation

- Early and accurate detection of oral cancer is critical for improving patient survival rates, but clinical workflows are still plagued by subjective human judgment, delays, and inconsistent decision-making.
- Robotic Process Automation (RPA) has been utilized in healthcare to automate tasks such as image processing, laboratory data management, and patient data analysis.
- Although existing RPA platforms (e.g., UiPath, Automation Anywhere) are user-friendly, they are highly inefficient for computationally intensive tasks: approximately 78% of processing time is spent on overhead (model reloading, activity transitions, data serialization), while only 22% is allocated to actual inference.
- There is a need for a hybrid solution that combines the workflow orchestration benefits of RPA with the high-performance computing capabilities of Python.

## Method

### Dataset and Preprocessing
- A dataset of approximately 3000 clinical oral images is classified into 4 major categories (Healthy, Benign, OPMD, Oral Cancer) and 16 subcategories.
- Data split: 70% training, 15% validation, and 15% testing (stratified sampling).
- Preprocessing: Pixel normalization to $[0, 1]$ and ImageNet mean/std standardization.
- Data augmentation: Using the Albumentations library, 5 transformations are applied per training sample (flipping, rotation, brightness/contrast adjustment, random cropping), and random duplication is performed for classes with fewer than 200 samples.

### Model Architecture
- ImageNet pre-trained EfficientNetV2B1 is adopted as the feature extractor.
- The input size is $224 \times 224 \times 3$, with a fully connected Dense + Softmax layer appended at the top.
- Two-stage training:
  1. Feature extraction phase: Freezing base layers, 15 epochs, learning rate $1\text{e-}3$.
  2. Fine-tuning phase: Partially unfreezing deep layers, 10 epochs, learning rate $1\text{e-}5$.
- Adam optimizer + categorical cross-entropy, with a batch size of 32.
- Early stopping + ReduceLROnPlateau + checkpointing based on the best validation accuracy.

### RPA Implementation Comparison
1. **OC-RPAv1**: Python-based sequential RPA-style processing, loading the model to predict one image at a time.
2. **OC-RPAv2**: Integrates Singleton and Batch Processing design patterns.
    - Singleton: The model is loaded only once and retained in memory, avoiding repeated reloading overhead.
    - Batch Processing: Batches images to leverage GPU parallel inference.
    - UiPath manages the automation pipeline and invokes Python functions to execute inference.

### Workflow Synchronization and Security
- Image batches are processed sequentially, and the next batch is only processed after each file is classified and logged, avoiding data collisions.
- Try-Catch exception handling ensures workflow continuity.
- Processing is conducted on local secure workstations with anonymized file paths and restricted access controls.
- Processed files are moved to a separate directory to ensure data integrity.

### Related Work & Insights
- Based on the CLASEG framework by Al-Ali et al., which integrates multi-class classification and segmentation for differential diagnosis of oral lesions.
- Follows the LMV-RPA approach of Abdellaif et al., supplementing standard RPA with enhanced Python automation.
- Kim et al. previously demonstrated the speedup effects of a hybrid RPA+Python architecture in computer-aided cancer detection on pathological images.
- This paper further introduces design patterns (Singleton + Batch) into this hybrid architecture and quantifies the acceleration ratio.

## Key Experimental Results

### Inference Speed Comparison (31 Test Images)

| Platform | Total Time | Average Time per Image |
|------|--------|------------|
| UiPath | 80 s | 2.58 s |
| Automation Anywhere | 75 s | 2.42 s |
| OC-RPAv1 (Python) | 8.65 s | 0.28 s |
| OC-RPAv2 (Python+DP) | 1.96 s | **0.06 s** |

- OC-RPAv2 is **~43×** faster than UiPath and **~40×** faster than Automation Anywhere.
- OC-RPAv2 is **~4.4×** faster than OC-RPAv1.
- The introduction of design patterns compresses the execution time of the Python pipeline from 8.65 s to 1.96 s.

### Scalability Estimation
- For 2500 images: UiPath requires 1.8 hours, while OC-RPAv2 takes less than 3 minutes.

## Highlights & Insights

1. **High Engineering Practicality**: The combination of Singleton and Batch Processing design patterns is simple yet effective, lowering the barrier to deployment.
2. **Substantial Acceleration**: Achieving 60-100× acceleration compared to standard RPA platforms, showing clear value for clinical deployment.
3. **Cost Reduction**: Reducing hardware idle time and RPA licensing costs, with the paper claiming a 40× cost reduction.
4. **Hybrid Architecture Approach**: Leverages the strengths of both worlds, with RPA handling workflow orchestration and Python taking charge of computationally intensive inference.
5. **16-Class Oral Lesion Classification**: Covers 4 major categories (Healthy, Benign, OPMD, Oral Cancer) and several subcategories, providing fine-grained classification.

## Limitations & Future Work

1. **Extremely Small Test Set**: Only 31 test images are used, indicating very low statistical reliability, which makes it impossible to draw robust speed benchmarking conclusions.
2. **Lack of Classification Accuracy Metrics**: The paper focuses disproportionately on speed comparison, failing to report critical classification metrics such as accuracy, precision, and recall on the test set.
3. **Limited Technical Contribution**: Singleton and Batch Processing are fundamental software engineering design patterns; the core "innovation" is closer to engineering optimization rather than novel academic contributions.
4. **Unfair Comparison**: The overhead of RPA platforms mainly stems from GUI automation and activity transitions; hence, comparing them directly to raw Python pipelines is inherently a comparison of different paradigms.
5. **Poor Writing Quality**: Contains redundant paragraphs, non-standard formatting, and incomplete references.
6. **Absence of Clinical Validation**: The system has not been deployed or validated in a real-world clinical setting, leaving scalability claims unsupported.
7. **Significant Gap with CVPR Standard**: The overall work resembles an engineering report rather than a top-tier conference paper, lacking academic depth.
8. **No Comparison with State-of-the-Art Methods**: Lacks comparative analysis of detection accuracy against mainstream CNN or ViT-based oral cancer detection methods.
9. **Insufficient Dataset Details**: Fails to report key dataset characteristics such as sample distribution among subcategories and precise sample counts post-augmentation.
10. **No Ablation Study**: Fails to isolate and validate the individual contributions of Singleton and Batch Processing.

## Rating

- Novelty: ⭐⭐ — Singleton and Batch Processing are foundational design patterns, lacking methodological innovation.
- Experimental Thoroughness: ⭐ — Only 31 test images and no classification performance metrics, showing highly insufficient experimental design.
- Writing Quality: ⭐⭐ — Redundant and repetitive text, chaotic formatting, with multiple duplicated paragraphs.
- Value: ⭐⭐ — While the engineering workflow has practical reference value, the academic contribution is insufficient for top-tier publication.
- Overall: ⭐⭐ — More suitable as an engineering technical report than as an academic paper reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortality_in_lung_cancer_screening_co.md)
- [\[NeurIPS 2025\] A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](../../NeurIPS2025/medical_imaging/a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)
- [\[CVPR 2025\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)
- [\[CVPR 2025\] SALIENT: Frequency-Aware Paired Diffusion for Controllable Long-Tail CT Detection](salient_frequency-aware_paired_diffusion_for_controllable_long-tail_ct_detection.md)
- [\[CVPR 2025\] OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection](openmibood_open_medical_imaging_benchmarks_for_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->

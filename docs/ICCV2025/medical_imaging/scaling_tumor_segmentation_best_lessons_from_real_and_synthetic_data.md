---
title: >-
  [Paper Note] Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data
description: >-
  [ICCV 2025][Medical Imaging][Data scaling laws] Through a systematic study of data scaling laws on a large-scale private dataset, this work demonstrates that synthetic tumors can substantially reduce the need for real an…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Data scaling laws"
  - "tumor segmentation"
  - "synthetic data"
  - "AbdomenAtlas"
  - "CT segmentation"
date: 2026-05-08
content_hash: bce159f358abe464
---

# Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data

**Conference**: ICCV 2025
**arXiv**: [2510.14831](https://arxiv.org/abs/2510.14831)  
**Code**: [https://github.com/BodyMaps/AbdomenAtlas2.0](https://github.com/BodyMaps/AbdomenAtlas2.0)  
**Area**: Medical Imaging / Tumor Segmentation
**Keywords**: Data scaling laws, tumor segmentation, synthetic data, AbdomenAtlas, CT segmentation

## TL;DR

Through a systematic study of data scaling laws on a large-scale private dataset, this work demonstrates that synthetic tumors can substantially reduce the need for real annotations (from 1,500 to 500 cases). Building on these findings, the authors construct AbdomenAtlas 2.0—the first large-scale manually annotated CT dataset with over 10,000 scans covering six organ tumor types—achieving significant improvements on both in-distribution and out-of-distribution benchmarks.

## Background & Motivation

Tumor segmentation AI is bottlenecked by the scarcity of large-scale voxel-level annotations. The central question is: **how much annotated data is truly needed to train effective tumor segmentation models, and can synthetic data reduce this requirement?**

Key findings from the authors' study on the private JHH dataset (3,000 pancreatic tumor annotated CTs):

**In-distribution performance saturates at approximately 1,500 cases**—additional in-distribution real data yields diminishing returns.

**With 3× synthetic data augmentation, only 500 real cases are needed to match the same performance**—reducing annotation requirements by 70%.

**Out-of-distribution generalization continues to improve**—performance has not saturated even at 3,000 cases, suggesting data diversity matters more than sheer quantity.

Based on these findings, the authors argue that 500–1,500 annotated cases per tumor type suffice to build effective AI models, motivating the creation of AbdomenAtlas 2.0.

## Method

### Overall Architecture

Two core contributions:
1. **AbdomenAtlas 2.0 Dataset**: 10,135 CT scans, six organ tumor types, annotated by 23 radiologists.
2. **Data Scaling Law Study**: A systematic investigation of how real and synthetic data scale for tumor segmentation.

### Key Designs

1. **SMART-Annotator Annotation Pipeline**:

    - **Core Idea**: Annotating missed tumors from scratch is far more time-consuming than removing AI-generated false positives; therefore, the pipeline is designed to prioritize maximizing sensitivity.
    - **Stage 1 – Model Preparation**: A dedicated segmentation model $f(\cdot)$ is trained for each tumor type.
    - **Stage 2 – FROC Curve Analysis**: A threshold $\theta^*$ is selected to achieve sensitivity >90% while maintaining an acceptable false positive rate.
    - **Stage 3 – Candidate Generation**: AI generates candidate segmentations; senior radiologists confirm true positives and reject false positives (averaging 1.2–2.4 false positives per scan).
    - **Stage 4 – Annotation Refinement**: Junior radiologists refine boundaries and correct omissions; senior radiologists perform final review.
    - **Efficiency Gain**: Per-case annotation time is reduced from 5 minutes to 5 seconds, saving approximately 49,826 minutes (83 working days).

2. **Dataset Construction (AbdomenAtlas 2.0)**:

    - **Scale**: 10,135 CT scans, 4.7 million slices, 15,130 tumor instances.
    - **Coverage**: Six tumor types—liver, pancreas, kidney, colon, esophagus, and uterus.
    - **Sources**: 89 hospitals across 17 countries.
    - **First-of-its-kind**: The first publicly available dataset providing voxel-level annotations for esophageal and uterine tumors.
    - **Rich early-stage tumor coverage** (<20 mm): liver 5,709 cases, pancreas 850 cases, kidney 4,638 cases.

3. **Synthetic Tumor Data Augmentation (DiffTumor)**:

    - Synthetic tumors are generated using DiffTumor at a small:medium:large ratio of 4:2:1.
    - Synthetic data volume is 3× that of the real data.
    - Synthetic tumors are automatically paired with voxel-level labels at generation time.
    - Tumors can be injected into normal CT scans from any source without additional manual annotation.

### Loss & Training

- nnU-Net framework; isotropic resampling to 1.5×1.5×1.5 mm³.
- Intensity clipping to [−175, 250], linearly normalized to [0, 1].
- Random cropping of 96×96×96 patches; SGD optimizer with learning rate 0.01.
- 1,000 training epochs, 250 iterations per epoch, batch size = 2.
- Test-time augmentation and sliding window inference with 50% overlap.

## Key Experimental Results

### Main Results — MSD Leaderboard

| Method | Liver Tumor DSC | Liver Tumor NSD | Pancreas Tumor DSC | Pancreas Tumor NSD |
|---|---|---|---|---|
| nnU-Net | 76.0 | 90.7 | 52.8 | 71.5 |
| Swin UNETR | 75.7 | 91.6 | 58.2 | 79.1 |
| Universal Model | 79.4 | 93.4 | 62.3 | 82.9 |
| **AbdomenAtlas 2.0** | **82.6** | **96.9** | **67.2** | **86.0** |
| Δ | +3.2 | +3.5 | +4.9 | +3.1 |

AbdomenAtlas 2.0 achieves **#1** on the MSD leaderboard.

### Out-of-Distribution Generalization

| External Dataset | Best Baseline DSC | AbdomenAtlas 2.0 DSC | Δ |
|---|---|---|---|
| 3D-IRCADb (Liver) | 67.1 (STU-Net) | **81.1** | **+14.0** |
| PANORAMA (Pancreas) | 43.0 (SegResNet) | **55.3** | **+12.3** |
| Kipa (Kidney) | 76.4 (ResEncM) | **83.6** | **+7.2** |
| JHH (Pancreas) | 39.5 (SegResNet) | **45.1** | **+5.6** |

Out-of-distribution generalization is substantially superior across all benchmarks, with DSC and NSD gains of 14.0% and 17.0% respectively on 3D-IRCADb.

### Ablation Study — Data Scaling

In-distribution saturation experiment (JHH private dataset):

| Real CT Count | DSC (Real Only) | DSC (Real + Synthetic) |
|---|---|---|
| 60 | 40.2 | 48.2 |
| 278 | 52.7 | 58.1 |
| 500 | ~54 | ~59 (≈ Real-only at 1,500) |
| 1500 | 59.3 | 59.2 |
| 3159 | 59.7 | 59.3 |

Key finding: **500 real cases + 3× synthetic data ≈ 1,500 real-only cases in performance**.

### Key Findings

- **Three core scaling laws**:
  1. In-distribution performance saturates at approximately 1,500 cases.
  2. Synthetic tumors reduce real data requirements by 70% (1,500 → 500).
  3. Out-of-distribution generalization continues to benefit from data diversity without saturation.
- Synthetic data accelerates in-distribution convergence (40%–60% of real data suffices to reach saturation).
- Per-tumor improvements: liver +4.9%, pancreas +8.8%, kidney +3.1%, colon +3.6%, esophagus +7.3%, uterus +1.4%.
- Synthetic data consistently contributes to performance gains even on out-of-distribution evaluations.

## Highlights & Insights

- **Scaling law perspective**: This is the first systematic study of data scaling laws for tumor segmentation, revealing the saturation point and the accelerating effect of synthetic data.
- **Practical annotation pipeline**: SMART-Annotator reduces annotation time by 60×, making it a viable solution for large-scale medical annotation.
- **High dataset value**: 10,135 CT scans covering six tumor types far exceed the combined scale of existing public datasets.
- **Deeper value of synthetic data**: Beyond in-distribution efficiency gains, synthetic tumors injected into diverse normal CTs also improve out-of-distribution generalization.
- **Open-source commitment**: Code, models, and data are all publicly released.

## Limitations & Future Work

- The ~1,500-case saturation point is validated only for pancreatic tumors; whether it generalizes to other organ types remains unconfirmed.
- The anatomical realism of synthetic tumors—particularly for infiltrative, necrotic, or early-stage lesions—has not been formally validated by clinical experts.
- Scaling experiments rely exclusively on the ResEncM architecture; different architectures may exhibit different saturation points.
- Coverage is limited to abdominal CT; generalizability to other modalities and anatomical regions remains to be verified.
- The annotation pipeline depends on the quality of the initial AI model, which may require additional adaptation for rare tumor types.

## Related Work & Insights

- This work extends the scaling law framework of Kaplan et al. from language models to medical imaging.
- DiffTumor's synthetic tumor generation establishes a new paradigm for data augmentation.
- Universal Model and SuPreM serve as strong baseline comparisons.
- Insight: In data-scarce medical domains, a strategy combining synthetic data with a small number of high-quality annotations may represent the optimal solution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The scaling law perspective is rarely explored in medical imaging; the finding that synthetic data accelerates convergence is genuinely insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale dataset, multi-baseline comparisons, in- and out-of-distribution evaluation, and detailed scaling experiments—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Logically structured argumentation, clearly articulated findings, and abundant figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ The dataset offers substantial value to the medical imaging community; the scaling law findings provide actionable guidance for future dataset construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)
- [\[ICCV 2025\] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities](mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](../../NeurIPS2025/medical_imaging/mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](radgpt_constructing_3d_image-text_tumor_datasets.md)
- [\[CVPR 2026\] RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation](../../CVPR2026/medical_imaging/rdface_a_benchmark_dataset_for_rare_disease_facial_image_analysis_under_extreme_.md)

</div>

<!-- RELATED:END -->

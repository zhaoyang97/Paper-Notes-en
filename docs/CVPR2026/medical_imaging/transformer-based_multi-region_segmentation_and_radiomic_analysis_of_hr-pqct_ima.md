---
title: >-
  [Paper Note] Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification
description: >-
  [CVPR2026][Medical Imaging][HR-pQCT] This paper proposes a fully automatic multi-region HR-pQCT segmentation framework based on SegFormer, combined with radiomic features and machine learning for binary osteoporosis classification. The key finding is that soft tissue (tendon/fat) features demonstrate greater diagnostic value than traditional bone features.
tags:
  - CVPR2026
  - Medical Imaging
  - HR-pQCT
  - osteoporosis classification
  - SegFormer
  - semantic segmentation
  - radiomics
  - machine learning
date: 2026-05-08
content_hash: 96df4c6c954fb212
---

# Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification

**Conference**: CVPR2026  
**arXiv**: [2603.09137](https://arxiv.org/abs/2603.09137)  
**Code**: Not available  
**Area**: Medical Imaging  
**Keywords**: HR-pQCT, osteoporosis classification, SegFormer, semantic segmentation, radiomics, machine learning

## TL;DR

This paper proposes a fully automatic multi-region HR-pQCT segmentation framework based on SegFormer, combined with radiomic features and machine learning for binary osteoporosis classification. The key finding is that soft tissue (tendon/fat) features demonstrate greater diagnostic value than traditional bone features.

## Background & Motivation

1. **Limitations of osteoporosis diagnosis**: The clinical gold standard DXA measures only areal bone mineral density (aBMD), which cannot assess bone microarchitecture, three-dimensional morphology, or surrounding soft tissue quality, leading to a high false-negative rate.
2. **Advantages of HR-pQCT**: HR-pQCT provides three-dimensional peripheral bone microstructure imaging at 60.7 µm isotropic voxel resolution with extremely low radiation (<5 µSv); however, existing analysis pipelines focus solely on mineralized bone regions, leaving large amounts of acquired data unexploited.
3. **Association between soft tissue and osteoporosis**: Sarcopenia and osteoporosis are highly comorbid conditions, and muscle mass indices (e.g., psoas muscle index) correlate significantly with bone mineral density; yet existing studies largely neglect the diagnostic contribution of soft tissue.
4. **Need for automated segmentation**: The current gold standard for HR-pQCT segmentation relies on semi-automatic methods requiring manual correction, which is time-consuming and subject to inter-operator variability. The only fully automatic method (U-Net) segments only two regions—cortical and trabecular bone.
5. **Limitations of CNNs in modeling long-range dependencies**: CNNs such as U-Net struggle to model global spatial relationships, resulting in insufficient segmentation accuracy for small targets (e.g., the fibula) in HR-pQCT images.
6. **Multi-region potential of radiomics**: Existing studies have demonstrated the effectiveness of radiomics for osteoporosis detection, but all focus on a single anatomical region or tissue type; the comparative diagnostic value of different regions (cortical bone, trabecular bone, soft tissue) remains unclear.

## Method

### Overall Architecture

The complete end-to-end pipeline consists of three stages: (1) five-class semantic segmentation via SegFormer → (2) post-processing and soft tissue subdivision into seven anatomical regions → (3) radiomic feature extraction from each region → machine learning classifier for binary osteoporosis classification.

### Segmentation Network Design

- **Architecture**: SegFormer-B3, with transfer learning from ImageNet/Cityscapes pre-trained weights.
- **Input adaptation**: The RGB three-channel weights are averaged to initialize the patch embedding layer for single-channel grayscale input.
- **Encoder**: A four-stage hierarchical Transformer encoder producing feature maps of sizes 200×200×64, 100×100×128, 50×50×320, and 25×25×512; early stages capture high-resolution local details while later stages capture global semantics.
- **Decoder**: A lightweight MLP decoder that upsamples all four feature maps to 200×200×768, concatenates them, and produces a final output of 200×200×6 (five classes + background).
- **Preprocessing**: Images are cropped to 1600×1600, HU values are clipped to [−4000, 6000] and normalized to [0, 1], then bicubically downsampled to 800×800.
- **Five annotation classes**: tibial cortical bone, tibial trabecular bone, fibular cortical bone, fibular trabecular bone, and soft tissue.

### Loss & Training

$$L_{Total} = L_{CE} + L_{Dice}$$

An equal-weight combination of cross-entropy loss (pixel-wise classification accuracy) and Dice loss (overlap-based segmentation performance) is employed to balance pixel-level precision with overall region coverage.

### Post-Processing and Soft Tissue Subdivision

- **Morphological constraints**: The largest connected component is retained for each class; cortical continuity is restored using convex hull detection combined with morphological closing operations.
- **Soft tissue subdivision**: The outer 2 mm boundary is designated as skin; seed-growing is applied based on HU thresholds (tendons: 100–600 HU; fat: −600 to −200 HU); unassigned pixels are classified using a −50 HU threshold.

### Radiomic Feature Extraction and Selection

- **939 features** are extracted per region: 7 feature families (first-order statistics, 2D shape, GLCM, GLSZM, GLRLM, NGTDM, GLDM) × 9 filter types (original + LoG + Wavelet + square + square root + logarithm + exponential + gradient + LBP).
- Three-stage dimensionality reduction: variance thresholding (0.02) → correlation filtering (Pearson |r| > 0.9) → LASSO regression, retaining 3–14 features per region.

## Key Experimental Results

### Dataset

| Dataset | Purpose | Scale | Source |
|--------|------|------|------|
| Segmentation set | SegFormer training/evaluation | 6,720 images / 40 scans / 22 subjects | CUIMC + ICMH (dual-center) |
| Classification set | Osteoporosis prediction | 20,496 images / 122 scans / 122 subjects (61 osteoporosis + 61 controls) | ICMH (single-center) |

### Segmentation Performance (Test Set, Mean ± SD)

| Model | Soft Tissue IoU | Tibial Cortical IoU | Fibular Trabecular IoU | Mean F1 |
|------|-----------|-------------|-------------|---------|
| U-Net | 98.0±7.5 | 86.1±6.4 | 74.4±23.4 | — |
| Attention U-Net | 98.2±6.2 | 87.0±4.6 | 72.1±24.7 | — |
| **SegFormer** | **99.2±0.2** | 86.5±3.6 | **89.6±5.6** | **95.36%** |

SegFormer achieves a **+20.43%** IoU improvement on fibular trabecular bone (a small target), with the lowest overall variance (IoU SD 3.74% vs. 11.5% for U-Net).

### Image-Level Osteoporosis Classification (Logistic Regression, Test Set)

| Anatomical Region | Accuracy | F1 | AUROC |
|----------|----------|-----|-------|
| Tibial cortical | 76.69% | 0.734 | 0.777 |
| Tibial trabecular | 78.55% | 0.759 | 0.799 |
| Fibular cortical | 78.99% | 0.762 | 0.847 |
| **Tendon tissue** | **80.08%** | **0.787** | 0.850 |
| Adipose tissue | 77.73% | 0.760 | **0.857** |

**Key finding**: Soft tissue features (tendon/fat) consistently outperform bone regions across all classification metrics.

### Patient-Level Classification (Logistic Regression, Test Set: 24 subjects)

| Model | Accuracy | Sensitivity | AUROC |
|------|----------|-------------|-------|
| Non-radiomics (clinical + DXA + HR-pQCT) | 0.792 | 0.667 | 0.792 |
| Radiomics – tibial | 0.792 | 0.833 | 0.826 |
| **Radiomics – soft tissue** | **0.875** | **0.917** | **0.875** |

### Ablation Study: Effect of Soft Tissue Distance

At varying radii centered on the tibial outer surface, the 10 mm region achieves the best XGBoost AUROC of **0.875**, indicating that soft tissue proximal to bone carries stronger osteoporosis-associated signals.

## Highlights & Insights

- **First application of Transformers to multi-region HR-pQCT segmentation**: SegFormer simultaneously segments four bone classes and soft tissue in a fully automatic end-to-end pipeline, achieving a 20% IoU improvement on small targets (fibula).
- **Counter-intuitive finding that soft tissue outperforms bone**: Tendon/fat radiomic features surpass traditional bone features for osteoporosis classification, improving AUROC from 0.792 to 0.875.
- **Complete seven-class segmentation**: Deep learning produces five classes, which are further subdivided by post-processing into skin/tendon/fat, representing the most fine-grained fully automatic segmentation scheme for HR-pQCT to date.
- **Systematic multi-region comparison**: The first systematic comparison of radiomic diagnostic value across cortical bone, trabecular bone, tendon, and adipose tissue regions.
- **First annotated HR-pQCT segmentation dataset**: 6,720 images with five-class pixel-level annotations, promised to be released upon publication.

## Limitations & Future Work

1. **Small patient-level sample size**: Only 122 subjects (24 in the test set), limiting statistical power and generalizability.
2. **Single-center classification data**: The classification set originates from ICMH only; cross-center generalizability has not been validated.
3. **2D slice-based processing**: HR-pQCT is inherently a 3D volumetric modality; slice-by-slice 2D processing discards inter-slice continuity information.
4. **Limited clinical accessibility of HR-pQCT**: The technology is far less available than DXA, constraining near-term clinical translation.
5. **Binary classification only**: The framework does not distinguish between osteopenia, osteoporosis, and normal bone density in a multi-class setting.
6. **No 3D segmentation comparison**: No comparison is made against volumetric segmentation methods such as 3D U-Net or nnU-Net.

## Related Work & Insights

| Method | Imaging Modality | Segmentation | Analysis Region | Advantage of This Work |
|------|---------|---------|---------|---------|
| Neeteson et al. (U-Net) | HR-pQCT | Automatic / 2 classes | Cortical + trabecular | Extended to 5+2=7 classes including soft tissue |
| Wang et al. (Radiomics) | Dual-energy CT | Manual ROI | Vertebral body | Multi-region automatic segmentation + systematic comparison |
| Huang et al. | Abdominal CT | Manual ROI | Psoas muscle | Fully automatic segmentation + multiple soft tissue types |
| Kim et al. (Deep Radiomics) | Hip X-ray | — | Femur | HR-pQCT high resolution + multi-region |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of Transformers to multi-region HR-pQCT segmentation; the finding that soft tissue outperforms bone offers clinically inspiring insights.
- **Experimental Thoroughness**: ⭐⭐⭐ — Segmentation evaluation is thorough, but the classification dataset is small (122 subjects / 24 test subjects) and multi-center validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with detailed methodological descriptions and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Challenges the "osteoporosis diagnosis relies solely on bone" paradigm and highlights the potential diagnostic role of soft tissue in metabolic bone disease.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)
- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)
- [\[AAAI 2026\] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks](../../AAAI2026/medical_imaging/sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)
- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](../../ACL2026/medical_imaging/region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[AAAI 2026\] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation](../../AAAI2026/medical_imaging/denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)

</div>

<!-- RELATED:END -->

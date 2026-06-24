---
title: >-
  [Paper Note] Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification
description: >-
  [CVPR2025][Medical Imaging][HR-pQCT] This paper first applies SegFormer to automatic multi-region (bone + soft tissue) segmentation and radiomic analysis of HR-pQCT imaging, finding that tendon tissue characteristics outperform traditional bone metrics in osteoporosis classification.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "HR-pQCT"
  - "SegFormer"
  - "radiomics"
  - "osteoporosis"
  - "multi-region segmentation"
date: 2026-05-08
content_hash: 37c29c05ce5d6b7c
---

# Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification

**Conference**: CVPR2025  
**arXiv**: [2603.09137](https://arxiv.org/abs/2603.09137)  
**Code**: To be confirmed  
**Area**: Medical Image  
**Keywords**: HR-pQCT, SegFormer, radiomics, osteoporosis, multi-region segmentation

## TL;DR

This paper first applies SegFormer to automatic multi-region (bone + soft tissue) segmentation and radiomic analysis of HR-pQCT imaging, finding that tendon tissue characteristics outperform traditional bone metrics in osteoporosis classification.

## Background & Motivation

Osteoporosis is the most common skeletal disease globally, primarily diagnosed clinically via DXA (dual-energy X-ray absorptiometry) based on bone mineral density T-score ($\le -2.5$ for osteoporosis diagnosis). However, DXA cannot capture three-dimensional microstructural information and surrounding soft tissue changes. HR-pQCT (high-resolution peripheral quantitative CT) provides 3D imaging of bone microstructure at $< 60.7\,\mu\text{m}$ resolution and very low radiation ($< 5\,\mu\text{Sv}$), but existing analysis pipelines mainly focus on mineralized bone regions, leaving a large amount of soft tissue information unutilized.

Existing HR-pQCT segmentation methods are dominated by traditional image processing and CNNs (such as U-Net). However, CNNs struggle to model long-range dependencies, and the current gold-standard protocol remains semi-automatic, requiring manual correction. Furthermore, osteoporosis often co-occurs with sarcopenia, and muscle-related metrics are significantly correlated with bone density, suggesting that soft tissue radiomics may have diagnostic value. This hypothesis is the core research motivation of this paper.

## Method

### Multi-Region Segmentation
The proposed method adopts **SegFormer-B3** (pretrained on Cityscapes), modifying the input layer to accept single-channel HR-pQCT grayscale images (initialized by averaging pretrained RGB channel weights). The $1600 \times 1600$ pixel images are downsampled to $800 \times 800$ inputs to output five-class segmentations: tibia cortical bone, tibia trabecular bone, fibula cortical bone, fibula trabecular bone, and soft tissue.

The SegFormer encoder generates four-level features (from $200 \times 200 \times 64$ to $25 \times 25 \times 512$), and the decoder projects and upsamples each layer to $200 \times 200 \times 768$ before concatenation, ultimately outputting $200 \times 200 \times 6$ (five classes + background). During inference, nearest-neighbor interpolation is used to upsample back to $1600 \times 1600$.

Training details: Adam optimizer, learning rate $1\text{e}-4$, batch size 2, 20 epochs, joint Cross-Entropy + Dice loss. A ReduceLROnPlateau scheduler is used (learning rate $\times 0.1$ after validation Dice plateaus for 3 epochs). Training is conducted on an NVIDIA RTX A5500 GPU.

### Post-processing
- Morphological constraints ensure that only the largest connected component is retained for each class, and five-class fragments are reassigned based on majority neighborhood labels.
- Cortical bone continuity is corrected through convex hull area comparison and morphological closing operations.
- Soft tissue is further segmented into: skin (outer boundary 2mm band), tendon tissue ($100\text{--}600\text{ HU}$), and adipose tissue ($-600$ to $-200\text{ HU}$), achieved through seeded region growing.

### Radiomics Analysis
From each anatomical region except the skin, 939 radiomic features are extracted:
- **7 feature classes**: First-order statistics (18), 2D shape (9), GLCM (24), GLSZM (16), GLRLM (16), NGTDM (5), GLDM (14)
- **8 filter types**: LoG ($\sigma=2$), Wavelet, Square, Square Root, Logarithm, Exponential, Gradient, LBP
- Dimensionality reduction is done via a three-step process: variance thresholding ($0.02$), Pearson correlation analysis (excluding $|r| > 0.9$), and LASSO regression, ultimately retaining 3–14 features per region.
- Binary osteoporosis classification is performed using 6 machine learning classifiers (LR, SVM, RF, XGBoost, KNN, NB) and 5-fold cross-validation.

### Patient-Level Prediction
A total of 43 non-radiomic features, including clinical metrics (5 features, such as age and BMI), functional assessments (7 features, such as gait speed and grip strength), DXA parameters (4 features), and standard HR-pQCT parameters (28 features), are integrated. These are compared with radiomic features (averaged over 168 slices per patient) in group-wise multivariate logistic regression models to evaluate predictive performance.

## Key Experimental Results

**Segmentation Performance** (test set, 1,344 images):

| Model | Soft Tissue IoU | Tibia Cortical IoU | Tibia Trabecular IoU | Fibula Cortical IoU | Fibula Trabecular IoU |
|------|------------|-------------|-------------|-------------|-------------|
| U-Net | 98.0±7.5 | 86.1±6.4 | 98.2±2.1 | 76.7±18.1 | 74.4±23.4 |
| Attn U-Net | 98.2±6.2 | 87.0±4.6 | 98.3±1.3 | 79.8±12.5 | 72.1±24.7 |
| **SegFormer** | **99.2±0.2** | 86.5±3.6 | 98.2±0.6 | **83.9±8.7** | **89.6±5.6** |

SegFormer achieves an IoU of $89.6\%$ on fibula trabecular bone, which is a $20.43\%$ improvement over U-Net. It also has the lowest standard deviation of IoU across all regions ($3.74\%$ vs. U-Net's $11.5\%$), demonstrating more stable performance across different images. The average F1-score reaches $95.36\%$. Qualitative comparison shows that U-Net exhibits significant under-segmentation and misclassification in the fibula trabecular bone (especially when degraded trabecular bone has a similar intensity to soft tissue), whereas SegFormer performs better.

**Feature Selection Results**: Among the 45 ultimately retained features, 25 are first-order statistical features and 10 are GLCM texture features. The absolute correlation coefficients of the mean features for each region range from $0.358$ to $0.526$.

**Image-Level Osteoporosis Classification** (20,496 images, 122 scans):

| Anatomical Region | Best Classifier | Acc (%) | AUROC |
|----------|----------|---------|-------|
| Tibia Cortical Bone | NB | 77.48 | 0.792 |
| Tibia Trabecular Bone | LR | 78.55 | 0.799 |
| Fibula Cortical Bone | LR | 78.99 | 0.847 |
| Fibula Trabecular Bone | NB | 67.86 | 0.709 |
| **Tendon Tissue** | **LR** | **80.08** | **0.850** |
| Adipose Tissue | SVM | 78.50 | 0.833 |

Tendon tissue yields the highest AUROC among all regions, surpassing all bone regions. Currently, the performance of the SVM on adipose tissue is also close to the optimal value of the bone metrics.

**Patient-Level Prediction**: Replacing standard clinical + DXA + HR-pQCT parameters with soft tissue radiomics improves the AUROC from $0.792$ to $0.875$. This finding challenges the traditional paradigm of "focusing only on bone for osteoporosis" and suggests that microstructural changes in muscle and adipose tissue may carry crucial diagnostic information. This aligns with clinical observations where sarcopenia, intramuscular fat infiltration, and osteoporosis frequently co-occur.

The **Hosmer-Lemeshow test** confirms that the logistic regression model is well-calibrated, and the Youden index is used to determine the optimal classification threshold.

## Highlights & Insights

1. **First Transformer for Multi-Region HR-pQCT Segmentation**: SegFormer improves IoU on minority classes (fibula) by over $20\%$, showing a clear advantage in global context modeling and the lowest cross-image variability.
2. **Key Finding—Soft Tissue Outperforms Bone Metrics**: The AUROC of $0.850$ for tendon tissue radiomics features exceeds that of all bone regions, challenging the traditional paradigm of "focusing only on bone for osteoporosis," which is consistent with the clinical evidence of sarcopenia-osteoporosis comorbidity.
3. **End-to-End Analysis Pipeline**: An end-to-end framework spanning segmentation $\rightarrow$ radiomics $\rightarrow$ classification.
    - Evaluated at both image and patient levels.
    - Demonstrates strong potential for clinical translation.
4. **Dataset to be Released**: A dataset of 6,720 HR-pQCT images with five-class pixel-level annotations will be released, facilitating community replication and extension.
5. **In-depth Feature Analysis**: Detailed reporting of the three-step feature selection process, correlation heatmaps, and feature type distributions.

## Limitations & Future Work

1. The segmentation dataset is small (40 scans / 22 subjects) and the classification dataset also contains only 122 cases, which limits statistical power.
2. The annotation process of segmentation involves manual corrections and interpolation every five slices, leading to potential uncertainties in annotation quality.
3. HR-pQCT equipment is expensive (second-generation XtremeCT II) and its application is restricted to peripheral bones, limiting clinical adoption.
4. 2D slice-level analysis ignores 3D spatial information, which contradicts the 3D imaging advantage of HR-pQCT.
5. Only the distal tibia site is used; other commonly used sites (e.g., distal radius) are not evaluated.
6. No comparison is made with 3D volume-level segmentation methods (e.g., nnU-Net).
7. Osteoporosis classification relies solely on a DXA T-score $\le -2.5$ definition, potentially missing continuous spectrum information of osteopenic patients.
8. Data are collected from two centers (CUIMC and ICMH), so multi-center generalization capability requires further validation.
9. Soft tissue segmentation relies on fixed HU thresholds, which may perform unstably across different scanners or patient cohorts.

## Rating
- **Novelty**: 3/5 — First to apply SegFormer to HR-pQCT, but the methodological aspect mainly combines existing tools.
- **Experimental Thoroughness**: 4/5 — Comprehensive analysis covering segmentation + radiomics + classification + patient-level analysis, with sufficient multi-classifier comparisons and comprehensive statistical tests.
- **Writing Quality**: 4/5 — The paper is clearly structured, with detailed methodology descriptions, high-quality figures, and strong reproducibility.
- **Value**: 4/5 — The finding that soft tissue outperforms bone metrics has significant clinical value. The dataset release will facilitate further research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniBrainBench: A Comprehensive Multimodal Benchmark for Brain Imaging Analysis Across Multi-stage Clinical Tasks](../../CVPR2026/medical_imaging/omnibrainbench_a_comprehensive_multimodal_benchmark_for_brain_imaging_analysis_a.md)
- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[NeurIPS 2025\] PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation](../../NeurIPS2025/medical_imaging/physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)

</div>

<!-- RELATED:END -->

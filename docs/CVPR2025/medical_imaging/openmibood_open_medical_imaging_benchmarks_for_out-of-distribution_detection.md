---
title: >-
  [Paper Note] OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection
description: >-
  [CVPR 2025][Medical Imaging][Out-of-Distribution Detection] This paper proposes OpenMIBOOD, a comprehensive benchmark framework for OOD detection specifically designed for medical imaging. It contains 14 datasets from three medical domains (histopathology, endoscopy, and brain MRI), evaluates 24 post-hoc methods, and reveals that findings from natural image OOD benchmarks cannot be directly transferred to medical scenarios.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Out-of-Distribution Detection"
  - "OOD Benchmark"
  - "Post-hoc Methods"
  - "Trustworthy AI"
date: 2026-05-08
content_hash: e93fa81f4a3aeb74
---

# OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.16247](https://arxiv.org/abs/2503.16247)  
**Code**: [https://github.com/remic-othr/OpenMIBOOD](https://github.com/remic-othr/OpenMIBOOD)  
**Area**: Medical Imaging  
**Keywords**: Out-of-Distribution Detection, Medical Imaging, OOD Benchmark, Post-hoc Methods, Trustworthy AI  

## TL;DR
This paper proposes OpenMIBOOD, a comprehensive benchmark framework for OOD detection specifically designed for medical imaging. It contains 14 datasets from three medical domains (histopathology, endoscopy, and brain MRI), evaluates 24 post-hoc methods, and reveals that findings from natural image OOD benchmarks cannot be directly transferred to medical scenarios.

## Background & Motivation

**Background**: Out-of-Distribution (OOD) detection is a critical link in ensuring the trustworthiness of AI systems. Since 2016, a vast number of OOD detection methods have emerged. Although the OpenOOD framework provides standardized evaluation criteria, these benchmarks mainly target natural images (e.g., ImageNet).

**Limitations of Prior Work**: The medical imaging field lacks systematic OOD evaluation benchmarks. The few existing medical OOD studies have obvious limitations, such as a restricted number of evaluated methods, incomprehensive dataset selection, and a lack of systematic evaluation on covariate-shifted ID (cs-ID) data. For instance, the work by Cao et al. only covers 8 post-hoc methods and uses natural images for some evaluation scenarios, which is disconnected from medical practices.

**Key Challenge**: OOD methods that perform exceptionally well on natural image benchmarks may not be equally effective in medical imaging. Medical images exhibit unique characteristics such as low variance and specific semantic shifts (e.g., different scanners, different staining protocols). Consequently, classification-probability-based methods (logits/softmax) are far less effective in medical contexts than feature-space-based methods.

**Goal**: To build a standardized benchmark covering multiple medical domains with fine-grained OOD taxonomy (cs-ID, near-OOD, far-OOD), evaluate a large number of post-hoc OOD detection methods, and provide a reliable reference for OOD detection research in medical applications.

**Key Insight**: To adopt OpenOOD's taxonomy but introduce a key modification: isolating cs-ID instead of merging it with ID, as distinguishing between ID and cs-ID (e.g., images acquired by different scanners) is equally critical in medical settings.

**Core Idea**: To perform a standardized and comprehensive evaluation of 24 post-hoc methods using 14 datasets across three medical domains (histopathology - MIDOG, endoscopy - PhaKIR, and brain MRI - OASIS3).

## Method

### Overall Architecture
OpenMIBOOD does not propose a new OOD detection algorithm; instead, it is a benchmarking framework. Its core pipeline is as follows: (1) construct three medical benchmarks, each containing four levels of data: ID, cs-ID, near-OOD, and far-OOD; (2) train a classifier for each benchmark; (3) run 24 post-hoc OOD detection methods on the classifiers and compare their performance.

### Key Designs

1. **Three-Tier OOD Taxonomy**:

    - **Function**: Classifies domain shifts into cs-ID, near-OOD, and far-OOD based on severity.
    - **Mechanism**: cs-ID refers to cases where the labels remain unchanged while the input feature distribution shifts (e.g., different scanners); near-OOD indicates cases with similar semantics but significant differences (e.g., different types of surgical instruments); far-OOD refers to completely different medical applications (e.g., applying a model trained on endoscopy to ophthalmic images).
    - **Design Motivation**: The detection difficulty and clinical significance of different shift levels vary in medical scenarios, requiring separate evaluations to guide model deployment accurately.

2. **Three Complementary Medical Benchmarks**:

    - **Function**: Covers three domains: histopathology (MIDOG), endoscopy (PhaKIR), and brain MRI (OASIS3).
    - **Mechanism**: MIDOG consists of mitotic figure classification across 10 domains with domain shifts coming from different scanners, staining protocols, and species (human/canine); PhaKIR contains surgical instrument classification in cholecystectomy, where domain shifts stem from smoke occlusion, different surgeries, and different surgery types; OASIS3 involves cognitive normal vs. Alzheimer's disease classification, with domain shifts arising from different modalities (T1w $\to$ T2w), scanners, and anatomical regions.
    - **Design Motivation**: Image features in different medical domains vary drastically (2D pathology vs. 2D endoscopy vs. 3D MRI). Thus, multi-domain evaluation is necessary to draw reliable conclusions.

3. **Standardized Evaluation Protocols**:

    - **Function**: Unifies method implementation, hyperparameter tuning, and evaluation metrics.
    - **Mechanism**: All 24 methods are implemented based on the OpenOOD codebase. Hyperparameter selection is done using the near-OOD validation set. Metrics reported include AUROC, FPR@95, and the harmonic mean of AUPRIN/AUPROUT. Methods are grouped into three categories based on their information source: classification-based (blue), feature-based (orange), and hybrid (green).
    - **Design Motivation**: Standardizing experimental conditions is a prerequisite for fair comparison, and using the harmonic mean avoids the overestimation of single metrics under data imbalance.

### Loss & Training
The classifiers are trained using a weighted cross-entropy loss function (to handle class imbalance) combined with a OneCycle learning rate scheduler. MIDOG and PhaKIR use ImageNet-1k pre-trained weights, while OASIS3 uses an R(2+1)D model pre-trained on Kinetics400.

## Key Experimental Results

### Main Results

| Method | MIDOG nOOD | PhaKIR nOOD | OASIS3 nOOD | Average nOOD AUROC | Type |
|------|-----------|-------------|-------------|----------------|------|
| MDSEns | 91.84 | 97.11 | 99.46 | 96.14 | Feature |
| ViM | 62.67 | 81.14 | 98.40 | 80.74 | Hybrid |
| Residual | 65.78 | 76.99 | 96.70 | 79.82 | Feature |
| MDS | 63.21 | 76.48 | 96.15 | 78.61 | Feature |
| KNN | 61.63 | 55.44 | 97.66 | 71.58 | Feature |
| MSP | 55.90 | 50.16 | 53.50 | 51.19 | Classification |
| EBO | 56.85 | 40.18 | 49.39 | 48.81 | Classification |

### Method Type Performance Comparison

| Method Type | MIDOG Avg | PhaKIR Avg | OASIS3 Avg |
|---------|-----------|-------------|-------------|
| Feature-based | 66.08 | 70.68 | 92.08 |
| Hybrid | 57.18 (-13%) | 50.71 (-28%) | 69.86 (-24%) |
| Classification-based | 55.81 (-16%) | 49.45 (-30%) | 52.13 (-43%) |

### Key Findings
- **Feature-based methods completely outperform classification-based methods**: Across all three medical benchmarks, the average AUROC of feature-space-based methods is significantly higher than that of logits/softmax-based methods. This is likely due to the low variance of medical images (the standard deviation of average pixel intensity for MIDOG/PhaKIR is only 0.148/0.149, much lower than ImageNet's 0.226), which leads to a more compact feature space, making distance-based OOD detection more suitable.
- **The high performance of MDSEns stems from covariate shift detection**: MDSEns utilizes the Mahalanobis distances of all intermediate layers of the network, where shallower layers are better at capturing variations in low-level visual features (such as edges and colors). On MIDOG domain 5 (which only has semantic shift without covariate shift), the performance of MDSEns drops sharply to 71.50%, but it reaches 98.95% on domain 6a (which contains both semantic and covariate shifts).
- **The best method on natural image benchmarks $\neq$ the best in medical scenarios**: Comparing the ranking of methods on ImageNet-1k and OpenMIBOOD reveals no clear positive correlation between them. Some classification-based methods that rank near the top on ImageNet perform the worst in medical settings.

## Highlights & Insights
- **The design of independent cs-ID evaluation is valuable**: In medical scenarios, identical classes of images acquired from different scanners (cs-ID) can cause model failures. Merging them with the ID dataset would mask this risk. This design insight can be transferred to other safety-critical AI deployment scenarios.
- **Discovery of the classifier overconfidence phenomenon**: In the PhaKIR benchmark, OOD instruments from EndoSeg18 are classified as "Grasper" with high confidence because OOD samples cluster near the Grasper class in the feature space. This indicates that the limitations of classification-based OOD methods in medical scenarios inherently stem from classifier overconfidence.
- **Refined selection of evaluation metrics**: Replacing a single metric with the harmonic mean of AUPRIN and AUPROUT avoids biases caused by data imbalance. This practice is applicable to all OOD evaluation scenarios involving class imbalance.

## Limitations & Future Work
- **Focus solely on classification tasks**: The benchmark only covers classification scenarios, leaving out segmentation tasks—which are more common and critical in medical imaging.
- **Evaluation limited to post-hoc methods**: It does not cover methods requiring additional training steps (such as outlier exposure), although the authors cite OpenOOD's conclusion that post-hoc methods are competitive with training-based methods.
- **Fixed classifier architectures**: Each benchmark uses only a single classifier architecture (ResNet50 / ResNet18 / R(2+1)D), leaving the influence of architectural choices on OOD detection method rankings unexplored.
- **Unexplored potential of foundation models**: With the development of medical imaging foundation models (e.g., BiomedCLIP), OOD detection methods leverageable from these models might show superior performance.

## Related Work & Insights
- **vs. OpenOOD**: OpenOOD is geared towards natural images and evaluates cs-ID merged with ID. This work tailor-makes an independent cs-ID evaluation for medical contexts, making it highly suitable for safety-critical clinical deployment.
- **vs. MOOD Challenge**: MOOD uses synthetic image corruption as OOD, whereas this work utilizes real-world medical domain shifts, aligning closer with clinical reality.
- **vs. Cao et al.**: Cao et al. only covers 8 methods and 3 medical scenarios, with partial reliance on natural images. This work is much larger in scale (24 methods, 14 datasets), offering more reliable conclusions.

## Rating
- Novelty: ⭐⭐⭐ The core contribution is the benchmark construction rather than a new algorithm, though the design of evaluating cs-ID independently introduces fresh insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 24 methods, 14 datasets, and multiple metrics, establishing the most comprehensive evaluation of medical OOD detection to date.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the dataset descriptions in the appendix are highly detailed, though there are several LaTeX macro rendering issues in the main body.
- Value: ⭐⭐⭐⭐ It reveals key findings that natural image benchmark conclusions cannot be directly transferred, offering valuable guidance for clinical AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging](../../NeurIPS2025/medical_imaging/dison_decentralized_isolation_networks_for_out-of-distribution_detection_in_medi.md)
- [\[CVPR 2026\] The Invisible Gorilla Effect in Out-of-distribution Detection](../../CVPR2026/medical_imaging/the_invisible_gorilla_effect_in_out-of-distribution_detection.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[CVPR 2025\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

</div>

<!-- RELATED:END -->

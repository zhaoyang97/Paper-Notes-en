---
title: >-
  [Paper Note] Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging
description: >-
  [CVPR 2026][Medical Imaging][scleral imaging] This paper proposes ScleraGluNet, a multi-view deep learning framework that combines five-directional scleral vessel imaging with multi-branch CNN feature extraction, MRFO-based feature refinement, and Transformer-based cross-view fusion, achieving 93.8% accuracy on three-class metabolic state classification and an MAE of 6.42 mg/dL for continuous fasting plasma glucose (FPG) estimation.
tags:
  - CVPR 2026
  - Medical Imaging
  - scleral imaging
  - noninvasive glucose estimation
  - multiview learning
  - transformer fusion
  - MRFO
date: 2026-05-08
content_hash: 2173e80a1b676d49
---

# Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging

**Conference**: CVPR 2026
**arXiv**: [2603.12715](https://arxiv.org/abs/2603.12715)
**Code**: N/A
**Area**: Medical Image Analysis / Noninvasive Detection
**Keywords**: scleral imaging, noninvasive glucose estimation, multiview learning, transformer fusion, MRFO

## TL;DR
This paper proposes ScleraGluNet, a multi-view deep learning framework that combines five-directional scleral vessel imaging with multi-branch CNN feature extraction, MRFO-based feature refinement, and Transformer-based cross-view fusion, achieving 93.8% accuracy on three-class metabolic state classification and an MAE of 6.42 mg/dL for continuous fasting plasma glucose (FPG) estimation.

## Background & Motivation

**Limitations of Prior Work in Diabetes Monitoring**: The global diabetic population is projected to reach 783 million by 2045. Existing gold-standard tests (FPG/OGTT/HbA1c) all require blood sampling, making frequent monitoring inconvenient and carrying infection risks. While CGM reduces the need for fingerstick measurements, it still requires subcutaneous sensor implantation and involves high costs. There is an urgent need for noninvasive monitoring solutions.

**Unique Advantages of Scleral Vasculature**: Unlike retinal imaging, which demands specialized equipment, scleral and conjunctival microvasculature can be directly visualized using low-cost anterior-segment cameras. Chronic hyperglycemia induces scleral microvascular remodeling—including changes in vessel caliber, increased tortuosity, altered branching patterns, and reduced perfusion density—changes that have been validated in OCTA and related studies.

**Limitations of Prior Work**: (1) Single-view acquisition covers only a limited scleral region, discarding region-specific vascular abnormality information; (2) microvascular remodeling across different quadrants is spatially heterogeneous, requiring multi-directional imaging for comprehensive capture; (3) existing approaches lack multi-task architectures capable of simultaneously handling classification and regression.

## Method

### Overall Architecture
Standardized multi-directional scleral image acquisition (5 directions × 5 images per subject) → ROI extraction + CLAHE + Frangi vessel enhancement preprocessing → 5 independent parameter-sharing CNN branches for feature extraction → MRFO-based feature refinement to eliminate redundancy → Transformer cross-view self-attention fusion → classification head (3-class softmax) + regression head (FPG estimation).

### Key Designs

1. **Multi-Directional Standardized Acquisition Protocol**:
    - Function: Scleral photographs are acquired from five gaze directions—straight, superior, inferior, nasal, and temporal.
    - Mechanism: Different quadrants exhibit distinct diabetes-related microvascular changes—temporal and nasal regions may show asymmetric remodeling, while superior and inferior regions display different perfusion characteristics. Multi-directional acquisition preserves region-specific diagnostic details.
    - Design Motivation: Prior studies have demonstrated that diabetes-induced conjunctival/scleral microangiopathy is spatially heterogeneous. Single-view acquisition inevitably discards critical diagnostic information.

2. **MRFO Feature Refinement + Transformer Cross-View Fusion**:
    - Function: Features concatenated from the five CNN branches are first processed by the Manta Ray Foraging Optimization (MRFO) algorithm to select an optimal feature subset and eliminate redundancy, followed by Transformer self-attention to model long-range vascular associations across quadrants.
    - Mechanism: MRFO is a bio-inspired feature selection algorithm that employs swarm intelligence to identify relevant and non-redundant feature subsets in high-dimensional spaces. Transformer self-attention captures subtle inter-quadrant vascular pattern associations, such as cross-regional asymmetric remodeling and cross-quadrant consistent degeneration.
    - Design Motivation: Concatenating five branches yields high-dimensional redundant features; feeding these directly into a Transformer is inefficient. The cascaded design of MRFO-then-Transformer can be understood as "denoising before association."

3. **Dual-Head Output for Multi-Task Learning**:
    - Function: The model simultaneously performs three-class classification (normal / controlled diabetes / hyperglycemic diabetes) and continuous FPG regression.
    - Mechanism: The classification head outputs three-class probabilities via softmax, while the regression head directly estimates FPG values (mg/dL). Both tasks share the underlying feature representation, and complementary learning signals improve overall performance.
    - Design Motivation: Classification and regression exploit vascular features from different perspectives—classification focuses on inter-class boundary features, while regression captures continuous variation trends. Joint training leverages multi-task regularization to prevent overfitting.

### Loss & Training
- Composite loss: Cross-Entropy (classification) + MSE (glucose regression), with empirically tuned weights.
- Subject-wise 5-fold cross-validation: all five images from the same subject are strictly assigned to the same fold to prevent data leakage.
- Adam optimizer; hyperparameters tuned on the validation set of each fold.
- Bootstrap resampling (1,000 iterations) for 95% confidence interval estimation.
- Exclusion criteria: ocular surface disease, active infection, recent surgery, etc., to ensure scleral imaging quality.

## Key Experimental Results

### Main Results

| Metric | Classification | Regression |
|--------|---------------|------------|
| Overall Accuracy | 93.8% (CI: 91.8–95.4%) | — |
| Normal Group Accuracy | 94.0% (141/150) | — |
| Controlled Group Accuracy | 92.1% (129/140) | — |
| Hyperglycemic Group Accuracy | 93.5% (145/155) | — |
| AUC (Normal / Controlled / Hyperglycemic) | 0.971 / 0.956 / 0.982 | — |
| MAE | — | 6.42 mg/dL |
| RMSE | — | 7.91 mg/dL |
| Pearson r | — | 0.983 |
| R² | — | 0.966 |
| Bland–Altman Bias | — | +1.45 mg/dL (LOA: −8.33 to +11.23) |

Dataset size: 445 subjects × 5 images = 2,225 scleral images (normal: 150 / controlled: 140 / hyperglycemic: 155).

### Ablation Study

| Configuration | Classification Accuracy | MAE (mg/dL) |
|---------------|------------------------|-------------|
| Single-view CNN | Lower | Higher |
| Multi-view CNN (w/o MRFO / Transformer) | Improved | Reduced |
| + MRFO Feature Refinement | Further improved | Further reduced |
| + Transformer Fusion (Full model) | **93.8%** | **6.42** |

### Key Findings
- Multi-directional acquisition yields significant improvements in both classification and regression, validating the hypothesis of regional heterogeneity in scleral vascular abnormalities.
- Misclassifications primarily occur between adjacent metabolic states (normal ↔ controlled), consistent with the continuous-spectrum nature of blood glucose levels.
- Per-fold accuracy across 5-fold CV ranges from 92.8% to 94.6%, indicating good stability.
- Bland–Altman analysis demonstrates satisfactory agreement with laboratory measurements, with 95% of points within ±11 mg/dL.

## Highlights & Insights
- **Novel Noninvasive Detection Pathway**: The use of scleral microvasculature (rather than the retina) for metabolic state assessment imposes lower imaging equipment requirements and holds potential for telemedicine applications.
- The multi-directional acquisition protocol has a sound physiological basis—scleral vessels in different quadrants genuinely respond differently to chronic hyperglycemia.
- The cascaded MRFO + Transformer design—"redundancy elimination followed by association modeling"—offers a valuable reference for multi-view fusion scenarios.

## Limitations & Future Work
- Single-center dataset (Aier Eye Hospital, Changsha); the sample size of 445 subjects is limited, raising concerns about generalizability.
- Patients with ocular surface diseases were excluded, yet such conditions frequently co-occur with diabetes in clinical practice.
- No direct comparison with existing noninvasive methods (e.g., PPG, thermal imaging) was conducted.
- Acquisition standardization relies on manual ROI extraction, limiting automation.
- Three-class granularity is relatively coarse; finer-grained glucose stratification may offer greater clinical utility.

## Related Work & Insights
- **AI for Diabetes via Retinal Imaging**: Extensive work has predicted cardiovascular-metabolic status and HbA1c from fundus images; this paper extends analogous ideas to more accessible scleral vasculature.
- **MRFO Algorithm**: Applications of Manta Ray Foraging Optimization in biomedical feature selection are increasing; this paper validates its effectiveness in multi-view feature redundancy elimination.
- Insight: Anterior-segment imaging combined with AI may open a new low-cost, noninvasive, point-of-care avenue for metabolic monitoring.

## Rating
- Novelty: ⭐⭐⭐ The concept of multi-directional scleral vessel imaging for glucose estimation is novel, though architectural innovation is limited.
- Experimental Thoroughness: ⭐⭐⭐ Detailed classification, regression, and Bland–Altman analyses are provided, but the single-center small sample and absence of comparisons with other noninvasive methods are notable weaknesses.
- Writing Quality: ⭐⭐⭐ Clinical background is well articulated, though the method description is somewhat verbose.
- Value: ⭐⭐⭐ Currently at a proof-of-concept stage; multi-center validation and automation improvements are required before clinical translation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [\[CVPR 2026\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning
description: >-
  [CVPR 2026][Medical Imaging][Third Molar] This paper compares three learning paradigms — local learning (LL), federated learning (FL), and centralized learning (CL) — for binary classification of third molar–mandibular canal overlap on panoramic radiographs. Centralized learning achieves the best performance (AUC 0.831), federated learning serves as a competitive privacy-preserving alternative (AUC 0.757), and both substantially outperform local learning (mean AUC 0.672).
tags:
  - CVPR 2026
  - Medical Imaging
  - Third Molar
  - Mandibular Canal
  - Panoramic Radiograph
  - Federated Learning
  - Deep Learning Classification
date: 2026-05-08
content_hash: 3150ac5017c4687d
---

# Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning

**Conference**: CVPR 2026
**arXiv**: [2603.11850](https://arxiv.org/abs/2603.11850)
**Authors**: Johan Andreas Balle Rubak, Sara Haghighat, Sanyam Jain, Mostafa Aldesoki, Akhilanand Chaurasia, Sarah Sadat Ehsani, Faezeh Dehghan Ghanatkaman, Ahmad Badruddin Ghazali, Julien Issa, Basel Khalil, Rishi Ramani, Ruben Pauwels
**Area**: Medical Imaging
**Keywords**: Third Molar, Mandibular Canal, Panoramic Radiograph, Federated Learning, Deep Learning Classification

## TL;DR

This paper compares three learning paradigms — local learning (LL), federated learning (FL), and centralized learning (CL) — for binary classification of third molar–mandibular canal overlap on panoramic radiographs. Centralized learning achieves the best performance (AUC 0.831), federated learning serves as a competitive privacy-preserving alternative (AUC 0.757), and both substantially outperform local learning (mean AUC 0.672).

## Background & Motivation

### Clinical Problem
Impacted mandibular third molars (wisdom teeth) represent one of the most common problems in oral surgery. When a third molar is in close proximity to the mandibular canal — which houses the inferior alveolar nerve — extraction carries a risk of nerve injury, potentially causing permanent paresthesia of the lower lip and mandibular region.

### Existing Diagnostic Workflow
- **Panoramic radiographs (OPG)** serve as the standard screening tool for assessing the spatial relationship between the third molar and the mandibular canal.
- When overlap is suspected on panoramic imaging, **CBCT (cone-beam computed tomography)** is typically ordered to confirm the three-dimensional relationship.
- However, a substantial proportion of CBCT referrals are unnecessary, increasing healthcare costs and patient radiation exposure.

### Need for Automated Classification
- Manual interpretation of panoramic radiographs is subject to inter-observer variability and is time-consuming.
- Deep learning can automate the binary classification of overlap vs. no-overlap, supporting clinical triage decisions.
- Medical imaging data are distributed across multiple institutions, and data sharing is subject to strict privacy regulations (e.g., GDPR, HIPAA).

### Motivation for Federated Learning
- **Federated learning (FL)** enables multi-center collaborative model training without sharing raw patient data.
- This paper aims to systematically compare the three learning paradigms on this clinical task, providing guidance for real-world deployment.

## Method

### Data and Annotation
- Panoramic radiograph dataset, cropped to the region of interest encompassing the third molar and mandibular canal.
- The annotation task is **binary classification**: overlap vs. no-overlap.
- Data are distributed across **8 independent annotators**, simulating 8 clinical centers/clients.
- Each client holds an independently annotated data subset.

### Model Architecture
- A **pretrained ResNet-34** is adopted as the backbone network.
- The model is fine-tuned from ImageNet pretrained weights.
- The output layer performs binary classification (sigmoid) to predict the probability of overlap.

### Three Learning Paradigms

**1. Local Learning (LL)**
- Each client trains an independent model solely on its local data.
- No cross-client model exchange or data sharing occurs.
- This yields 8 independent local models.

**2. Centralized Learning (CL)**
- Data from all 8 clients are pooled together.
- A single model is trained on the entire aggregated dataset.
- This represents the theoretical upper bound, ignoring privacy constraints.

**3. Federated Learning (FL)**
- Each client trains locally and uploads only model parameters to a central server.
- The server aggregates the parameters and distributes the updated model back to clients.
- This process iterates over multiple rounds until convergence, with no raw data ever shared.

### Evaluation Strategy
- **Per-client evaluation**: For each client, the optimal classification threshold is determined on its local validation set to assess local performance.
- **Aggregated test evaluation**: A global threshold is applied to the pooled test set to assess overall generalization.
- **Evaluation metrics**: AUC (area under the ROC curve), accuracy, sensitivity, specificity, and other threshold-dependent metrics.
- **Grad-CAM visualization**: Model attention is analyzed to verify that learned features correspond to clinically relevant anatomical regions.
- **Training dynamics analysis**: Training/validation curves are monitored to detect overfitting.
- **Server-side aggregation monitoring**: Aggregation signals on the FL server are tracked throughout training.

## Key Experimental Results

### Table 1: Overall Performance of the Three Learning Paradigms on the Aggregated Test Set

| Paradigm | AUC | Accuracy | Notes |
|---|---|---|---|
| Centralized Learning (CL) | **0.831** | **0.782** | Best performance; centralized training |
| Federated Learning (FL) | 0.757 | 0.703 | Privacy-preserving; second best |
| Local Learning (LL) Mean | 0.672 | — | Averaged across clients |
| Local Learning (LL) Range | 0.619–0.734 | — | Substantial inter-client variability |

**Key Observations**:
- CL outperforms FL by **7.4 AUC percentage points** and LL (mean) by **15.9 percentage points**.
- FL outperforms LL (mean) by **8.5 AUC percentage points**, demonstrating the value of cross-client parameter aggregation.
- The LL AUC range (0.619–0.734) spans 11.5%, reflecting the impact of data heterogeneity on local models.

### Table 2: Training Characteristics and Model Behavior Across Learning Paradigms

| Dimension | CL | FL | LL |
|---|---|---|---|
| Overfitting | Mild | Moderate | Severe |
| Grad-CAM focus | Anatomically concentrated | Anatomically concentrated | Dispersed / inconsistent |
| Cross-client generalization | Strongest | Strong | Weakest |
| Data privacy | Not protected (requires sharing) | Protected (parameters only) | Protected (fully isolated) |
| Training data volume | Full dataset | Equivalent full (via aggregation) | Local subset only |
| Clinical deployability | Requires data centralization; limited | Multi-center deployable | Single-center only |

**Key Observations**:
- LL models exhibit the most severe overfitting, as limited per-client data make models prone to memorizing training samples.
- Grad-CAM heatmaps for both CL and FL consistently focus on the third molar apex and mandibular canal trajectory, indicating that clinically meaningful features are learned.
- Grad-CAM heatmaps for LL are dispersed and inconsistent, suggesting that models may have learned spurious correlations.

## Highlights & Insights

- **Systematic paradigm comparison**: This work provides the first systematic comparison of LL, FL, and CL on the task of third molar–mandibular canal relationship classification from dental panoramic radiographs, offering direct empirical evidence for multi-center collaboration in this domain.
- **Practical value of federated learning**: Without sharing any raw data, FL raises mean AUC by 12.6% relative to LL (0.672 → 0.757), demonstrating the feasibility of federated learning in oral medical imaging.
- **Grad-CAM validation of clinical plausibility**: Interpretability analysis confirms that CL and FL models attend to the correct anatomical structures, enhancing clinical trustworthiness.
- **Quantification of overfitting risk**: The paper explicitly reveals the overfitting risk associated with small-sample local training, providing a practical reference for model training strategy selection in data-scarce scenarios.
- **Reduction of unnecessary CBCT referrals**: Automated classification can support clinical triage, reducing unnecessary CBCT examinations and thereby lowering radiation exposure and healthcare costs.

## Limitations & Future Work

- **Binary classification only**: The coarse-grained overlap/no-overlap dichotomy may be insufficient to reflect the more nuanced clinical risk stratification required in practice (e.g., contact, perforation).
- **Data scale and diversity**: The distribution across 8 annotators may not adequately represent the data heterogeneity encountered in real multi-center settings (e.g., differences in equipment and patient populations).
- **Single backbone**: Only ResNet-34 is evaluated; alternative architectures (e.g., DenseNet, EfficientNet, Vision Transformer) are not explored.
- **Single FL aggregation strategy**: Different federated aggregation algorithms (e.g., FedProx, FedBN, SCAFFOLD) are not compared, and superior FL strategies may exist.
- **Absence of clinician baseline**: Radiologist diagnostic performance is not reported as a human reference baseline.
- **Limited discussion of threshold robustness**: The discrepancy between global and locally optimized thresholds has practical implications for deployment, but threshold robustness is not discussed in detail.

## Related Work & Insights

- **Dental AI**: Recent applications of deep learning in dental imaging span caries detection, periodontal disease grading, and implant identification; however, automated assessment of the third molar–mandibular canal relationship remains relatively underexplored.
- **Federated learning in medical imaging**: Methods such as FedAvg have been validated on tasks including chest radiography and histopathology, but their application to oral medical imaging remains at an early stage.
- **Panoramic radiograph analysis**: Conventional approaches rely predominantly on handcrafted features or simple CNNs; this paper employs a pretrained ResNet-34 with a systematic paradigm comparison, offering a more methodologically complete treatment.
- **Interpretability analysis**: Grad-CAM is widely used in medical imaging to verify the anatomical plausibility of model attention; here it serves as an auxiliary tool for cross-paradigm comparison.

## Rating

- Novelty: ⭐⭐⭐ — The methodology (ResNet-34 + FedAvg) is well-established; the primary contribution lies in the clinical application and paradigm comparison.
- Experimental Thoroughness: ⭐⭐⭐ — Quantitative comparison across three paradigms and Grad-CAM analysis are provided, but multi-architecture comparisons and a human baseline are absent.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, the problem is well-defined, and the experimental design is sound.
- Value: ⭐⭐⭐⭐ — Provides valuable empirical evidence for privacy-preserving multi-center collaboration in dental imaging.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[CVPR 2026\] Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)

<!-- RELATED:END -->

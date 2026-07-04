---
title: >-
  [Paper Note] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning
description: >-
  [CVPR 2025][Medical Imaging][third molar] This paper compares the performance of three learning paradigms—Local Learning (LL), Federated Learning (FL), and Centralized Learning (CL)—in automatically classifying the overlapping relationship between the third molar and the mandibular canal on panoramic radiographs. Utilizing a pre-trained ResNet-34 as the backbone, the study finds that centralized training achieves the best performance (AUC 0.831)…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "third molar"
  - "mandibular canal"
  - "federated learning"
  - "panoramic radiograph"
  - "deep learning classification"
date: 2026-05-08
content_hash: 14e11a7b27f51de8
---

# Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning

**Conference**: CVPR 2025  
**arXiv**: [2603.11850](https://arxiv.org/abs/2603.11850)  
**Code**: None  
**Area**: Medical Image  
**Keywords**: third molar, mandibular canal, federated learning, panoramic radiograph, deep learning classification

## TL;DR

This paper compares the performance of three learning paradigms—Local Learning (LL), Federated Learning (FL), and Centralized Learning (CL)—in automatically classifying the overlapping relationship between the third molar and the mandibular canal on panoramic radiographs. Utilizing a pre-trained ResNet-34 as the backbone, the study finds that centralized training achieves the best performance (AUC 0.831), while FL significantly outperforms purely local training under privacy-preserving assumptions.

## Background & Motivation

**Background**: When the mandibular third molar (wisdom tooth) is impacted and close to the mandibular canal, extraction surgery may cause inferior alveolar nerve injury. Clinically, panoramic radiographs are routinely used to evaluate the spatial relationship between the third molar and the mandibular canal. However, this evaluation process depends on the subjective judgment of clinicians, resulting in low inter-annotator agreement. While CBCT provides three-dimensional information, it is not required for all cases, and automated binary classification (overlap vs. no-overlap) helps clinical triage.

**Limitations of Prior Work**: Traditional centralized deep learning requires aggregating data from multiple medical institutions into a single location for training, which has become unfeasible under increasingly strict privacy regulations (such as GDPR and HIPAA). Local models trained independently by individual medical centers often suffer from poor generalization due to limited data volume and annotation bias.

**Key Challenge**: Medical imaging AI requires a large amount of diverse data to achieve good generalization, but data privacy regulations prohibit cross-institutional data sharing, constituting a fundamental conflict between performance and privacy.

**Goal**: To systematically compare the performance of three learning paradigms—Local Learning (LL), Federated Learning (FL), and Centralized Learning (CL)—on a multi-annotator panoramic radiograph dataset, and quantitatively evaluate the feasibility of FL as a privacy-preserving alternative.

**Key Insight**: Treat the data from 8 independent annotators as 8 independent clients/centers to simulate three training strategies in a multi-center scenario, and comprehensively evaluate the performance differences of each paradigm using per-client metrics and global threshold metrics.

**Core Idea**: Through rigorous comparative experiments, this study demonstrates that federated learning can significantly outperform local models trained independently at individual centers without sharing raw data. Although slightly inferior to centralized training, it provides an effective privacy-performance trade-off.

## Method

### Overall Architecture

The input is a cropped panoramic radiograph (centered on the third molar region), from which features are extracted using a pre-trained ResNet-34 to output a binary classification of whether the molar and the mandibular canal overlap. The three training paradigms utilize the same network architecture and hyperparameters, differing only in data organization and parameter aggregation strategies.

### Key Designs

1. **Data Partitioning and Multi-Client Simulation**:

    - **Function**: Partition the panoramic radiograph dataset according to annotators into 8 independent clients to simulate a realistic multi-center scenario.
    - **Mechanism**: Eight independent annotators annotated the overlapping relationship (binary classification: overlap vs. no-overlap) between the third molar and the mandibular canal on a batch of panoramic radiographs. The data of each annotator is treated as the local dataset of a "client". Within each client, the data is further split into training and test sets.
    - **Design Motivation**: This partitioning method not only simulates the real-world scenario where multiple clinical centers have different annotated data, but also naturally introduces inter-annotator bias, making the experiment closer to practical application scenarios.

2. **Unified Comparison of Three Learning Paradigms**:

    - **Function**: Fairly compare the three paradigms (LL, FL, CL) under the same network architecture (ResNet-34) and hyperparameters.
    - **Mechanism**: **Local Learning (LL)** — each client independently fine-tunes ResNet-34 using local data; **Federated Learning (FL)** — utilizes the FedAvg algorithm, where each client trains locally and uploads model parameters to the server for aggregation, then the aggregated parameters are sent back to each client for the next round of training; **Centralized Learning (CL)** — merges all 8 client datasets into a single large dataset for unified training. All three paradigms use ImageNet pre-trained ResNet-34.
    - **Design Motivation**: Using the same backbone network eliminates the impact of architectural differences, allowing a pure evaluation of the impacts of data organization and aggregation strategies on performance.

3. **Dual Evaluation Strategy**:

    - **Function**: Evaluate models using per-client local optimal thresholds and a global unified threshold, respectively.
    - **Mechanism**: For the test data of each client, the optimal client-specific threshold is first identified via the ROC curve for per-client evaluation; then, a unified threshold is used on the pooled test set of all clients for global evaluation. Evaluation metrics include AUC, accuracy, sensitivity, specificity, etc.
    - **Design Motivation**: Per-client evaluation reflects the model's adaptability in individual independent centers, while global threshold evaluation reflects the model's cross-center generalizability.

### Loss & Training

A standard binary cross-entropy loss function is used to train ResNet-34. FL training adopts the FedAvg strategy, where each client trains locally for several epochs per round before uploading parameters. During training, Grad-CAM visualization is used to monitor whether the model's attention focuses on anatomically relevant structures.

## Key Experimental Results

### Main Results

| Learning Paradigm | AUC (pooled) | Accuracy (pooled) | Remarks |
|---|---|---|---|
| Centralized Learning (CL) | **0.831** | **0.782** | Best performance |
| Federated Learning (FL) | 0.757 | 0.703 | Second best under privacy protection |
| Local Learning (LL) | 0.619–0.734 (mean 0.672) | — | High variability among clients |

### Per-Client AUC Comparison

| Client | LL (AUC) | FL (AUC) | CL (AUC) |
|---|---|---|---|
| Best Client | 0.734 | — | — |
| Worst Client | 0.619 | — | — |
| Mean | 0.672 | 0.757 | 0.831 |

### Key Findings

- **CL dominates overall**: The AUC of centralized training is about 0.074 higher than that of FL, and about 0.159 higher than the mean of LL, showing the significant positive impact of data volume and diversity on model performance.
- **FL effectively bridges data silos**: Without sharing data, the AUC of FL is 0.023 higher than even the best LL client, and significantly superior to most LL models, proving the effectiveness of federated aggregation.
- **LL exhibits obvious overfitting**: Training curves show that LL models overfit most severely, particularly on clients with smaller data sizes, resulting in poor generalization.
- **Grad-CAM Analysis**: The attention of CL and FL models focuses more on the anatomical region of the molar-mandibular canal junction, while the attention of LL models is often more scattered, indicating that more data helps the model learn more meaningful features.

## Highlights & Insights

- **Systemic three-paradigm comparison framework**: In the privacy-sensitive field of medical imaging, this paper provides a complete comparison baseline of LL/FL/CL, offering a clear reference for subsequent researchers to select training strategies. This experimental design mindset can be migrated to other medical image classification tasks.
- **Clever design of using annotators as clients**: Using the natural grouping of multiple annotators as client partitions in FL not only simulates realistic multi-center scenarios but also avoids bias introduced by artificial data partitioning.
- **Explainability analysis with Grad-CAM**: Instead of just reporting digital metrics, it explains why CL/FL outperforms LL via attention visualization—because they learn more anatomically meaningful feature representations.

## Limitations & Future Work

- **Limited dataset scale**: Drawn from only 8 annotators; real multi-center scenarios may involve dozens or even hundreds of institutions, with stronger data heterogeneity.
- **Relatively simple task**: Binary classification (overlap/no-overlap) is a relatively coarse-grained evaluation. Clinically, regression or multi-class classification may be more needed to evaluate the level of overlap and risk grades.
- **Lack of cross-dataset validation**: All data came from the same source; the generalization of the model on radiographs captured by completely different devices or institutions was not verified.
- **Insufficient quantitative analysis of FL privacy guarantees**: The introduction of stronger privacy guarantee mechanisms such as differential privacy is not discussed, nor is the risk of privacy leakage analyzed.
- **Future directions**: FedAvg combined with Differential Privacy (DP) can be explored to study performance trade-offs under stronger privacy guarantees; the method can also be extended to multi-class classification or detection tasks.

## Related Work & Insights

- **vs Traditional Centralized Deep Learning**: Traditional methods require data centralization, and the CL results in this paper validate its upper bound. However, FL provides a viable alternative that does not require data sharing.
- **vs Pure Federated Learning Methods**: This paper utilizes the most basic FedAvg, without involving more advanced FL algorithms like FedProx or SCAFFOLD, leaving space for future improvement.
- **vs Local Independent Training**: Experiments clearly demonstrate that local models suffer from severe overfitting and insufficient generalization, highlighting the necessity of cross-center collaboration.

## Rating

- **Novelty**: ⭐⭐⭐ No novel contributions in methodology; primarily an empirical comparison of existing technologies applied to specific medical scenarios.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes multiple evaluation metrics and Grad-CAM analysis, but the dataset scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure and detailed experimental analysis.
- **Value**: ⭐⭐⭐⭐ Provides an empirical reference of FL in dental imaging AI, holding certain clinical guidance significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)
- [\[CVPR 2025\] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)
- [\[CVPR 2025\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)
- [\[CVPR 2025\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [\[CVPR 2026\] OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis](../../CVPR2026/medical_imaging/oralgpt-plus_learning_to_use_visual_tools_via_reinforcement_learning_for_panoram.md)

</div>

<!-- RELATED:END -->

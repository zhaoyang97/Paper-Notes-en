---
title: >-
  [Paper Note] DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection
description: >-
  [CVPR 2025][Video Understanding][Multimodal OOD Detection] The **Dynamic Prototype Updating (DPU)** framework is proposed, which establishes a robust representation space via **Cohesive-Separate Contrastive Training**, dynamically updates class centers through **Dynamic Prototype Approximation**, and adjusts the intensification intensity of multimodal prediction discrepancy based on sample-to-prototype distance using **Pro-ratio Discrepancy Intensification**. Serving as a plu…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Multimodal OOD Detection"
  - "Dynamic Prototype Updating"
  - "Intra-class Variation"
  - "Contrastive Learning"
  - "Discrepancy Intensification"
date: 2026-05-08
content_hash: 67e62d3ff8e171ca
---

# DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection

**Conference**: CVPR 2025  
**arXiv**: [2411.08227](https://arxiv.org/abs/2411.08227)  
**Code**: [https://github.com/lili0415/DPU-OOD-Detection](https://github.com/lili0415/DPU-OOD-Detection)  
**Area**: Video Understanding  
**Keywords**: Multimodal OOD Detection, Dynamic Prototype Updating, Intra-class Variation, Contrastive Learning, Discrepancy Intensification

## TL;DR

The **Dynamic Prototype Updating (DPU)** framework is proposed, which establishes a robust representation space via **Cohesive-Separate Contrastive Training**, dynamically updates class centers through **Dynamic Prototype Approximation**, and adjusts the intensification intensity of multimodal prediction discrepancy based on sample-to-prototype distance using **Pro-ratio Discrepancy Intensification**. Serving as a plug-and-play module, it comprehensively improves performance across 5 datasets and 9 base OOD methods, with up to an **80%** performance gain in Far-OOD detection.

## Background & Motivation

OOD detection aims to identify samples that deviate from the training distribution, which is crucial for safety-critical scenarios such as autonomous driving, medical imaging, and robotics. While traditional OOD detection mainly targets single modalities (e.g., images), prediction discrepancies across different modalities in multimodal data (video + optical flow + audio) can serve as a strong signal to distinguish ID from OOD.

**Limitations of Prior Work**:
- Dong et al. found that the **prediction discrepancy** between modalities is a key signal for distinguishing ID vs OOD (ID samples exhibit consistent predictions across modalities, whereas OOD samples do not), and improved detection performance by uniformly amplifying this discrepancy.
- However, this method assumes that **all samples of the same class exhibit perfect cohesion**, ignoring the **intra-class variation** in real-world scenarios.
- Forcing discrepancy amplification on samples near the class center (which inherently feature high cross-modal prediction consistency) **disrupts their consistency**, confusing the model and degrading ID accuracy.

**Key Challenge**: It is necessary to amplify the cross-modal discrepancy of OOD samples to boost detection, but uniform amplification harms high-consistency in-distribution samples. How can the discrepancy amplification intensity be **adaptively** determined based on the sample's position within the class?

**Core Idea**: Dynamically maintain a prototype for each class. The further a sample is from its prototype, the stronger the discrepancy amplification (implying potential boundary or outlier samples), whereas samples closer to the prototype retain lower discrepancy (core samples should not be disturbed).

## Method

### Overall Architecture

DPU consists of three sequential components (see Figure 2): (1) **CSCT** (Step 1): establishing an intra-class cohesive, inter-class separate representation space via robust marginal contrastive learning and variance regularization; (2) **DPA** (Step 2): dynamically updating class prototypes based on batch-wise variance to minimize the impact of outliers; (3) **PDI** (Step 3): adaptively adjusting the intensity of discrepancy amplification according to sample-to-prototype similarity, and further enhancing discriminative ability via adaptive outlier synthesis. DPU is a model-agnostic, plug-and-play framework that can be integrated with any OOD detection algorithm.

### Key Designs

1. **Cohesive-Separate Contrastive Training (CSCT)**
    - **Function**: Establish an intra-class compact (cohesive) and inter-class separate representation space while quantifying intra-class variance.
    - **Mechanism**:
     - **Robust Marginal Contrastive Learning** $\mathcal{L}_{rmcl}$: Based on the InfoNCE loss, arc-cosine distance with an angular margin $m$ is used to enhance inter-class discrimination sensitivity.
     - **Invariant Representation Minimization** $\mathcal{L}_{irm}$: Minimize the variance of loss values within each positive set (samples of the same class) to ensure consistent representations for intra-class samples.
     - Total Loss: $\mathcal{L}_{csct} = \mathcal{L}_{rmcl} + \lambda \cdot \mathcal{L}_{irm}$
    - **Design Motivation**: Based on the Invariant Risk Minimization (IRM) paradigm, contrastive learning provides inter-class separation, while the variance constraint provides intra-class stability. Together, they lay the foundation for subsequent dynamic prototype learning.

2. **Dynamic Prototype Approximation (DPA)**
    - **Function**: Dynamically and adaptively update prototype representations for each class, ensuring the prototypes truly represent class centers.
    - **Mechanism**: Compute the average embedding $H_{av_k}^y$ of same-class samples in each batch, and update the prototypes using a variance-aware moving average:
     $$P_{ty_k}^y = \beta P_{ty_k}^y + (1-\beta) \cdot \frac{1}{\gamma + \text{Var}(\mathcal{L}^j) N^y} \cdot (H_{av_k}^y - P_{ty_k}^y)$$
     The update rate is inversely proportional to the variance—updating faster when variance is low (high sample consistency) and slower when variance is high (containing outliers).
    - **Design Motivation**: Traditional methods weight all samples equally when computing class centers, causing outliers to skew the prototype. Variance-aware dynamic updating maintains prototype stability when encountering noisy batches.

3. **Pro-ratio Discrepancy Intensification (PDI) + Adaptive Outlier Synthesis**
    - **Function**: Adaptively adjust the cross-modal discrepancy amplification intensity based on the distance between the sample and its prototype.
    - **Mechanism**:
     - Intensification Rate = $\mu \cdot (1 - \text{Sigmoid}(F_i^v \cdot (P_{ty_v}^y)^T))$, style: the further the sample is from the prototype (lower similarity), the higher the amplification rate.
     - Discrepancy is measured using the Hellinger distance, calculated over the prediction distributions of two modalities: $\mathcal{L}_{pdi} = -\text{IntensificationRate} \cdot \text{Discr}(\hat{p}_i^{k_1}, \hat{p}_i^{k_2})$
     - **Adaptive Outlier Synthesis**: Fuse prototypes from different classes to generate synthetic OOD samples $\bar{P}_{fuse} = \eta \bar{P}_{y_1} + (1-\eta) \bar{P}_{y_2}$, maximizing the discrepancy and entropy of the synthetic samples.
    - **Loss & Training**: $\mathcal{L}_{aos} = -(\text{Discr}(\bar{P}_{y_1}^{fuse}, \bar{P}_{y_2}^{fuse}) + E(\bar{P}_{y_1}^{fuse}) + E(\bar{P}_{y_2}^{fuse}))$
    - **Design Motivation**: Core samples inherently have consistent cross-modal predictions, and forced amplification would confuse the classifier; boundary samples are more likely to exhibit modal inconsistency, making discrepancy amplification more meaningful. Outlier synthesis provides additional training signals.

## Key Experimental Results

### Near-OOD Detection (HMDB51 as ID, 25/26 Class Split)

| Method | FPR95↓ | AUROC↑ | ID ACC↑ |
|------|--------|--------|---------|
| MSP | 44.66 | 87.74 | 89.32 |
| MSP+A2D | 38.78 | 88.37 | 90.64 |
| **MSP+DPU** | **34.20** | **89.15** | **92.16** |
| Energy | 43.36 | 87.46 | 89.32 |
| **Energy+DPU** | **35.07** | **89.52** | **92.16** |
| Mahalanobis | 40.31 | 85.28 | 89.32 |
| **Mahalanobis+DPU** | **36.17** | **89.53** | **92.16** |

### Far-OOD Detection (HMDB51 vs Kinetics600)

DPU achieves significant improvements across all 9 base OOD methods, reducing FPR95 by up to **80%** (as shown in Figure 1).

### Cross-Dataset Validation

On UCF101 (50/51 split), Kinetics600 (129/110 split), and EPIC-Kitchens, DPU achieves consistent improvements across all base methods:
- UCF101: Average FPR95 reduction of approximately 1-2 percentage points.
- Kinetics600: Average FPR95 reduction of approximately 2-3 percentage points.
- EPIC-Kitchens: Average FPR95 reduction of approximately 3-5 percentage points.

### Ablation Study

- Removing CSCT: AUROC drops by 2.1%
- Removing DPA (using fixed prototypes): AUROC drops by 1.5%
- Removing PDI (using uniform amplification): FPR95 increases by 4.2%
- Removing adaptive outlier synthesis: AUROC drops by 0.8%
- All three components are indispensable, with PDI contributing the most.

### Key Findings

- Uniform discrepancy amplification (A2D) can **decrease** ID accuracy while improving OOD detection under certain configurations, whereas DPU **consistently improves** ID accuracy along with OOD detection.
- In near-OOD detection, all metrics improve by approximately 10% on average; the improvements are even more pronounced in far-OOD detection.
- Variance-aware prototype updating makes prototypes more stable, outperforming simple moving average by approximately 1% in AUROC.

## Highlights & Insights

- **First to address intra-class variation in multimodal OOD detection**: uncovers the fundamental flaws of uniform discrepancy amplification and provides a more fine-grained solution.
- **Model-agnostic, plug-and-play design**: consistently effective across 9 mainstream OOD methods (such as MSP, Energy, Maxlogit, Mahalanobis), demonstrating strong generalizability.
- **Dynamic prototype updating mechanism**: The variance-aware adaptive update rate is a clever design that automatically slows down updates in noisy batches and quickly adapts in consistent ones.
- The improvement in ID accuracy indicates that DPU not only facilitates OOD detection but also enhances ID classification, as improved intra-class compactness and inter-class separation naturally benefit overall classification.

## Limitations & Future Work

- Multimodal inputs are limited to video + optical flow + audio; the effectiveness has not been validated on other modality combinations (such as text, depth, etc.).
- Hyperparameters $\beta$ and $\gamma$ in DPA need to be tuned for each dataset.
- The high dimensionality of the prototype space (equal to feature dimensions) may lead to non-negligible storage and update overheads when dealing with a large number of classes.
- Evaluated only on classification model backbones (e.g., SlowFast, I3D); applicability to VLM-based OOD detection remains to be explored.

## Related Work & Insights

- **Dong et al.** first proposed the multimodal OOD benchmark and observed cross-modal prediction discrepancies; DPU builds upon this by addressing the limitations of uniform amplification.
- **Prototype Learning** (Prototypical Networks) is widely used in few-shot learning; DPU introduces it to OOD detection scenarios with an added dynamic updating mechanism.
- The core concept of **Invariant Risk Minimization (IRM)** inspired the design of the variance constraint in CSCT.
- Insight: In other scenarios requiring "adaptive intensity" (e.g., data augmentation, adversarial training), the adaptive mechanism based on prototype distance is worth referencing.

## Rating

⭐⭐⭐⭐ — Acute and insightful observation of the intra-class variation issue in multimodal OOD detection. The three-step framework is logically clear and cohesive. Its generalizability is thoroughly validated through large-scale experiments across 5 datasets and 9 base methods, showing an impressive improvement in Far-OOD. The plug-and-play nature provides significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](../../CVPR2026/video_understanding/worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)
- [\[CVPR 2025\] Localizing Events in Videos with Multimodal Queries](localizing_events_in_videos_with_multimodal_queries.md)
- [\[CVPR 2025\] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding](stop_integrated_spatial-temporal_dynamic_prompting_for_video_understanding.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](../../ICCV2025/video_understanding/distime_distribution-based_time_representation_for_video_large_language_models.md)

</div>

<!-- RELATED:END -->

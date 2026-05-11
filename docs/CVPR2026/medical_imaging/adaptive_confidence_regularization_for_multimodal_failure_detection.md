---
title: >-
  [Paper Note] Adaptive Confidence Regularization for Multimodal Failure Detection
description: >-
  [CVPR 2026][Medical Imaging][multimodal failure detection] This paper proposes the ACR framework, which addresses multimodal misclassification detection for the first time in a systematic manner through two complementary…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "multimodal failure detection"
  - "confidence degradation"
  - "adaptive confidence regularization"
  - "feature swapping"
  - "misclassification detection"
  - "selective prediction"
date: 2026-05-08
content_hash: 1401d02bc546a195
---

# Adaptive Confidence Regularization for Multimodal Failure Detection

**Conference**: CVPR 2026
**arXiv**: [2603.02200](https://arxiv.org/abs/2603.02200)
**Code**: [mona4399/ACR](https://github.com/mona4399/ACR)
**Area**: Medical Imaging / Multimodal Reliability
**Keywords**: multimodal failure detection, confidence degradation, adaptive confidence regularization, feature swapping, misclassification detection, selective prediction

## TL;DR

This paper proposes the ACR framework, which addresses multimodal misclassification detection for the first time in a systematic manner through two complementary modules: an Adaptive Confidence Loss (ACL) that penalizes "confidence degradation" where multimodal fusion confidence falls below that of individual unimodal branches, and Multimodal Feature Swapping (MFS) that synthesizes failure-aware outlier samples in the feature space. ACR consistently outperforms existing methods across four datasets.

## Background & Motivation

1. **High-stakes deployment requirements**: Multimodal models are widely deployed in safety-critical applications such as autonomous driving and medical diagnosis, where high accuracy alone is insufficient — reliably detecting untrustworthy predictions (failure detection, FD) is equally essential.
2. **Inapplicability of unimodal FD methods**: Existing FD methods are primarily designed for unimodal settings; they cannot exploit cross-modal complementary information or handle multimodal-specific failure modes such as signal conflicts and alignment failures.
3. **Failure of OOD detection methods on FD**: Experiments show that OOD methods such as Energy, Entropy, and MaxLogit perform worse than the simple MSP baseline on FD tasks, indicating that directly transferring OOD techniques is ineffective.
4. **Multimodal signals as FD cues**: Even a simple fusion of video and optical flow substantially improves FD performance, demonstrating the significant potential of multimodal inputs for FD — yet a dedicated framework to exploit this potential is lacking.
5. **Confidence degradation phenomenon**: The authors find that, among misclassified samples, the proportion of cases where fusion confidence falls below that of at least one unimodal branch is substantially higher than among correctly classified samples (32.4% higher on HMDB51, 52.4% higher on HAC). This "confidence degradation" phenomenon serves as a strong indicator of failure.
6. **Lack of real failure training samples**: Traditional Outlier Exposure relies on large-scale external datasets and cannot synthesize multimodal-specific failure modes such as cross-modal conflicts. Unimodal methods such as OpenMix are similarly inapplicable.

## Method

### Overall Architecture

ACR (Adaptive Confidence Regularization) consists of two complementary modules:

- **Adaptive Confidence Loss (ACL)**: Penalizes fusion confidence that falls below unimodal confidence, i.e., the "confidence degradation" phenomenon.
- **Multimodal Feature Swapping (MFS)**: Synthesizes failure-aware outlier samples by swapping feature dimensions across modalities in the feature space.

Architecturally, each of the $M$ modality branches has an encoder $g_k(\cdot)$ that extracts embedding $\mathbf{E}^k$; the concatenated embeddings are fed into a fusion classifier $h(\cdot)$ to produce the multimodal prediction $\hat{p}$. Each modality also has an independent classifier $h_k(\cdot)$ yielding unimodal prediction $\hat{p}^k$.

### Adaptive Confidence Loss (ACL)

Define fusion confidence as $\text{conf} = \max_y \hat{p}$ and unimodal confidence as $\text{conf}_k = \max_y \hat{p}^k$. For the two-modality case, ACL is defined as:

$$\mathcal{L}_{\text{acl}} = \frac{1}{2}\left(\max(0, \text{conf}_1 - \text{conf}) + \max(0, \text{conf}_2 - \text{conf})\right)$$

- No penalty is applied when fusion confidence exceeds that of all unimodal branches; a linear penalty is applied whenever it falls below any single modality.
- This encourages the fusion mechanism to fully integrate complementary information while suppressing overconfidence in individual modalities.

### Multimodal Feature Swapping (MFS)

- A contiguous block of $n_{\text{swap}} \sim \mathcal{U}(n_{\min}, n_{\max})$ dimensions is randomly selected from each modality embedding and swapped across modalities to produce perturbed features $\mathbf{E}_o$.
- Soft labels are constructed by interpolating between the ground-truth label and an outlier class label: $\mathbf{y}_{\text{swapped}} = (1-\lambda)\mathbf{y}_{\text{true}} + \lambda\mathbf{y}_{\text{outlier}}$, where $\lambda = n_{\text{swap}} / n_{\max}$.
- Small swap sizes yield hard negatives close to the in-distribution manifold; large swap sizes yield clear outliers far from the distribution — offering fine-grained controllability.
- No external data is required; the method operates entirely in the feature space and is computationally efficient.

### Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{outlier}} + \lambda_{\text{acl}} \mathcal{L}_{\text{acl}}$$

At inference, MSP scoring is applied only over the original $C$ classes, incurring no additional computational overhead.

## Key Experimental Results

### Main Results (Video + Optical Flow, Table 1)

| Dataset | Method | AURC↓ | AUROC↑ | FPR95↓ | ACC↑ |
|---|---|---|---|---|---|
| HMDB51 | MSP | 29.56 | 88.28 | 52.07 | 86.20 |
| HMDB51 | **ACR** | **19.97** | **92.02** | **41.96** | **87.23** |
| HAC | MSP | 42.90 | 89.27 | 66.67 | 82.11 |
| HAC | **ACR** | **27.41** | **91.48** | **39.39** | **84.86** |
| Kinetics-600 | MSP | 46.29 | 87.33 | 61.29 | 81.24 |
| Kinetics-600 | **ACR** | **41.85** | **88.99** | **55.89** | **81.45** |
| EPIC-Kitchens | Best baseline (RegMixup) | 105.25 | 79.26 | 78.19 | 74.53 |
| EPIC-Kitchens | **ACR** | **103.25** | **79.27** | **71.58** | **75.20** |

ACR achieves the best performance on all datasets, with a maximum AURC improvement of 9.58%, a maximum FPR95 improvement of 15.45%, and concurrent gains in classification accuracy.

### Ablation Study (HMDB51, Table 2)

| Configuration | AURC↓ | AUROC↑ | FPR95↓ | ACC↑ |
|---|---|---|---|---|
| MSP baseline | 29.56 | 88.28 | 52.07 | 86.20 |
| + ACL only | 24.48 | 90.32 | 43.97 | 86.77 |
| + MFS only | 25.11 | 90.55 | 46.22 | 86.43 |
| **ACL + MFS** | **19.97** | **92.02** | **41.96** | **87.23** |

Both modules are individually effective, and their combination yields the best overall performance, demonstrating complementarity.

### Additional Evaluations

- **Generalization across modality combinations** (HAC; video+audio / optical flow+audio / three-modality): Average AURC improvement of 8.39% and FPR95 improvement of 10.65%.
- **Robustness to distribution shift**: ACR maintains consistent advantages under five video corruption types (defocus blur, frost, brightness, pixelation, JPEG compression).
- **Different backbone networks** (I3D, TSN): ACR comprehensively outperforms all baselines.
- **OOD detection**: ACR also achieves strong performance on the MultiOOD benchmark (AUROC 96.82 vs. 95.35 for the second-best method).

## Highlights & Insights

- **First systematic study of multimodal FD**: The paper identifies the confidence degradation phenomenon, quantifies its strong correlation with misclassification, and provides a theoretical foundation for this research direction.
- **Outlier synthesis without external data**: MFS operates in the feature space, making it computationally efficient, modality-agnostic, and highly controllable.
- **Simultaneous improvement in FD and classification accuracy**: The regularization effect of ACL improves classification performance concurrently — an uncommon property among FD methods.
- **Extensive evaluation**: Four datasets, three modalities, multiple settings (distribution shift, different backbones, OOD detection), and thorough ablation studies.

## Limitations & Future Work

- Experiments are limited to the action recognition domain (video + optical flow/audio); applicability to other multimodal tasks such as medical imaging and remote sensing remains unverified.
- Only two- and three-modality fusion settings are evaluated; scalability to four or more modalities is unknown.
- The feature dimension swapping in MFS is uniformly random and does not account for the varying semantic importance of different modality dimensions.
- At inference, only MSP scoring is used; leveraging the difference between unimodal and multimodal confidences as a stronger FD signal has not been explored.
- A theoretical analysis of the distribution of samples generated by MFS in high-dimensional embedding spaces is absent.

## Related Work & Insights

| Method | Type | Multimodal | Requires External Data | FD Performance |
|---|---|---|---|---|
| MSP / MaxLogit / Energy | Scoring function | ✗ | ✗ | Baseline level |
| DOCTOR | Confidence learning | ✗ | ✗ | Marginal improvement |
| OpenMix | Outlier synthesis | ✗ | ✓ | Moderate |
| CRL | Confidence regularization | ✗ | ✗ | Moderate |
| A2D | Multimodal OOD | ✓ | ✗ | Moderate (OOD-oriented) |
| **ACR** | **Multimodal FD-specific** | **✓** | **✗** | **Best** |

## Rating

- Novelty: ⭐⭐⭐⭐ — The discovery of the confidence degradation phenomenon and the design of ACL+MFS are original contributions; the paper is the first to systematically study multimodal FD.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four datasets, three modalities, and diverse evaluation settings (distribution shift, different backbones, OOD detection) with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated, and the logical flow from observation to method is coherent.
- Value: ⭐⭐⭐⭐ — Fills a gap in multimodal FD research; the framework is broadly applicable and has practical value in real-world safety-critical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] SVC 2026: The Second Multimodal Deception Detection Challenge and the First Domain Generalized Remote Physiological Measurement Challenge](svc_2026_the_second_multimodal_deception_detection_challenge_and_the_first_domai.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[AAAI 2026\] SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection](../../AAAI2026/medical_imaging/spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)
- [\[CVPR 2026\] The Invisible Gorilla Effect in Out-of-distribution Detection](the_invisible_gorilla_effect_in_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->

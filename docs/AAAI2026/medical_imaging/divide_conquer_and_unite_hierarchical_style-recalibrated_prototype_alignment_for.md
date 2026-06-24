---
title: >-
  [Paper Note] Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation
description: >-
  [AAAI 2026][Medical Imaging][Federated Learning] To address the two key challenges in federated medical image segmentation — *layerwise style bias accumulation* and *incomplete contextual representation* — this paper proposes FedBCS: a framework that constructs domain-invariant prototypes via Frequency-domain adaptive Style Recalibration (FSR) and designs Context-aware Dual-level Prototype Alignment (CDPA) to fuse multi-level semantics from both encoder and decoder. FedBCS ac…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Federated Learning"
  - "Medical Image Segmentation"
  - "Prototype Alignment"
  - "Frequency-domain Style Calibration"
  - "Feature Heterogeneity"
date: 2026-05-08
content_hash: de4c1b41268a8d27
---

# Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation

**Conference**: AAAI 2026
**arXiv**: [2511.10945](https://arxiv.org/abs/2511.10945)  
**Code**: [https://github.com/zxy1234321/FedBCS](https://github.com/zxy1234321/FedBCS)  
**Area**: Medical Imaging / Federated Learning
**Keywords**: Federated Learning, Medical Image Segmentation, Prototype Alignment, Frequency-domain Style Calibration, Feature Heterogeneity

## TL;DR

To address the two key challenges in federated medical image segmentation — *layerwise style bias accumulation* and *incomplete contextual representation* — this paper proposes FedBCS: a framework that constructs domain-invariant prototypes via Frequency-domain adaptive Style Recalibration (FSR) and designs Context-aware Dual-level Prototype Alignment (CDPA) to fuse multi-level semantics from both encoder and decoder. FedBCS achieves state-of-the-art performance on nuclei segmentation and prostate MRI segmentation tasks.

## Background & Motivation

Federated learning (FL) enables multiple medical institutions to collaboratively train a global model without sharing data, making it an important paradigm for medical image analysis. However, differences in scanning equipment and imaging protocols across institutions introduce severe feature heterogeneity — that is, significant divergence in the conditional feature distribution $P(x|y)$ across clients under the same label distribution.

Existing prototype-based federated learning methods attempt cross-client alignment via class-mean feature vectors, but suffer from two critical limitations:

**Layerwise Style Bias Accumulation**: Style variations introduced by different medical protocols accumulate progressively across intermediate network layers. Existing methods apply style normalization only at the input level or align only the final-layer features, neglecting intermediate-layer style bias. This causes representations of the same anatomical structure to diverge increasingly across institutions.

**Incomplete Contextual Representation Learning**: Existing methods construct prototypes using only the last encoder layer, overlooking local texture and edge details captured in shallow layers and abstract structural information in deep layers, resulting in incomplete semantic understanding.

- **Core Idea**: Introduce content-style disentanglement into the prototype construction process (via frequency-domain operations), and achieve finer-grained cross-client semantic alignment through multi-level prototype fusion across the encoder-decoder hierarchy.

## Method

### Overall Architecture

FedBCS is built upon the standard FL pipeline (FedAvg framework), where clients share a UNet architecture and upload prototypes to the server after each round of local training. The core innovations lie in the local prototype construction and alignment stages: the FSR module first removes domain-specific style bias from features, and CDPA then extracts and fuses prototypes from multiple levels of the encoder and decoder. On the server side, FINCH clustering aggregates multi-client prototypes, which are then redistributed to guide local training.

### Key Designs

1. **Frequency-domain Style Recalibration (FSR)**:

    - **Function**: Removes domain-specific style variations from features prior to prototype construction, preserving semantic content.
    - **Mechanism**: A 2D Fourier transform is applied to encoder features, decomposing them into an amplitude spectrum (encoding style) and a phase spectrum (preserving semantic content). Learnable parameters adaptively weight the normalized and original amplitude spectra: $\hat{z}_{enc} = IFT(\lambda_s^{norm}\chi_{norm} + \lambda_s^{org}\chi, \gamma)$. The weights are generated via global average pooling, a learnable linear mapping, and sigmoid activation.
    - **Design Motivation**: Style information is naturally encoded in the amplitude spectrum of the frequency domain, while semantic content resides in the phase spectrum. Although instance normalization of the amplitude can remove style, it may discard useful information. The learnable parameters allow the model to adaptively determine how much original style to retain.

2. **Context-aware Dual-level Prototype Alignment (CDPA)**:

    - **Function**: Extracts domain-invariant prototypes from multiple levels of the encoder and decoder, and performs inter-level fusion and cross-client alignment.
    - **Mechanism**: At each level $k$, class prototypes are extracted from both the encoder and decoder respectively. Prototypes from adjacent levels (shallow + deep) are concatenated and compressed via a lightweight $1\times1$ convolution fusion module to yield compact multi-scale prototypes. On the server side, FINCH clustering groups same-class prototypes from different clients and computes mean prototypes.
    - **Design Motivation**: Shallow layers encode local tissue texture and edges, while deep layers encode global anatomical structures. Knowledge of the overall organ shape (deep-level information) facilitates more accurate tissue boundary delineation in ambiguous regions (shallow-level information).

3. **Joint Contrastive and Consistency Loss**:

    - **Function**: Aligns local features with global prototypes.
    - **Mechanism**: The contrastive loss pulls features toward same-class prototypes and pushes them away from different-class prototypes; the consistency loss encourages encoder and decoder features to align with their respective global mean prototypes.
    - **Design Motivation**: The contrastive loss captures inter-class discriminability, while the consistency loss ensures cross-institutional feature coherence.

### Loss & Training

The total loss is $L_{total} = L_{MP} + L_{dice}$, where $L_{MP} = L_{contra} + L_{consis}$. $L_{contra}$ is an InfoNCE-style contrastive loss, and $L_{consis}$ is a mean squared error consistency loss.

Training spans 400 communication rounds with 1 local epoch per round. SGD (lr=0.01) is used for nuclei segmentation and Adam (lr=1e-4) for prostate MRI segmentation, with weight decay=1e-4 and batch size=6.

## Key Experimental Results

### Main Results

| Method | Nuclei Seg. AVG (Dice) | △ | Prostate MRI AVG | △ |
|--------|----------------------|-----|-----------------|-----|
| FedAvg | 69.50 | - | 78.80 | - |
| FedProx | 69.00 | -0.50 | 78.10 | -0.70 |
| HarmoFL | 71.90 | +2.40 | 81.20 | +2.40 |
| FPL | 71.66 | +2.16 | 77.40 | -1.40 |
| FedPLVM | 71.00 | +1.50 | 77.50 | -1.30 |
| FedUV | 70.00 | +0.50 | 78.50 | -0.30 |
| **FedBCS** | **74.10** | **+4.60** | **82.60** | **+3.80** |

### Ablation Study

| Configuration | Nuclei AVG | Prostate AVG | Notes |
|--------------|-----------|-------------|-------|
| Baseline (FedAvg) | 69.50 | 78.80 | No FSR, no CDPA |
| +CDPA only | 72.20 | 81.90 | Multi-level alignment only |
| +FSR only | 72.30 | 80.10 | Frequency-domain style calibration only |
| +CDPA+FSR (Full) | **74.10** | **82.60** | Complementary combination |

### Key Findings

- FSR and CDPA each independently yield ~2–3% gains; their combination achieves larger improvements (4.6%/3.8%), confirming that the two modules address complementary problems.
- High communication efficiency: FedBCS uploads only 4 prototypes per client per round (2 classes × 2 levels), compared to 212–365 prototypes on average for FedPLVM, yet substantially outperforms it (+5.1%).
- FSR's layerwise style bias handling outperforms input-level amplitude normalization alone (Fig. 4b), validating the necessity of frequency-domain recalibration at the feature level.
- The optimal temperature parameter $\tau$ differs across datasets (0.005 for nuclei vs. 0.4 for prostate MRI), yet FedBCS consistently surpasses second-best methods across a broad range of values.

## Highlights & Insights

- Introducing frequency-domain style disentanglement into federated prototype learning is an elegant combination, leveraging the natural correspondence between amplitude/phase spectra and style/content in the Fourier domain.
- Theoretical convergence guarantees (Theorems 1 and 2) are provided, enhancing the credibility of the proposed framework within the federated learning literature.
- The communication efficiency advantage is notable: multi-level prototype fusion still yields only 2 prototypes per class for upload, representing a 50–90× reduction compared to FedPLVM.

## Limitations & Future Work

- Validation is limited to the UNet architecture; applicability to more complex architectures (e.g., ViT-based) remains unexplored.
- Experiments focus on binary segmentation tasks (foreground/background); scalability to multi-organ, multi-class scenarios has not been verified.
- The learnable style parameters may overfit when the number of participating clients is very small.
- The theoretical analysis relies on relatively strong assumptions (e.g., bounded gradient variance) that may not fully hold in practice.

## Related Work & Insights

- Compared to HarmoFL (input-level frequency-domain operations), FSR operates at the feature level and offers greater flexibility.
- The dual-path encoder-decoder design of CDPA can be generalized to other scenarios requiring multi-level feature alignment.
- FINCH parameter-free clustering avoids the hyperparameter of pre-specifying the number of prototypes, making it a practical choice for federated prototype learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ (The combination of frequency-domain style calibration and multi-level prototype alignment is relatively novel)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Comprehensive ablation, communication efficiency analysis, convergence analysis, and hyperparameter sensitivity)
- **Writing Quality**: ⭐⭐⭐⭐
- **Value**: ⭐⭐⭐⭐ (Offers substantive improvements for feature heterogeneity in federated medical segmentation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Divide, Conquer, and Aggregate: Asymmetric Experts for Class-Imbalanced Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/divide_conquer_and_aggregate_asymmetric_experts_for_class-imbalanced_semi-superv.md)
- [\[AAAI 2026\] MPA: Multimodal Prototype Augmentation for Few-Shot Learning](mpa_multimodal_prototype_augmentation_for_few-shot_learning.md)
- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[AAAI 2026\] FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation](funkan_functional_kolmogorov-arnold_network_for_medical_image_enhancement_and_se.md)
- [\[ICML 2026\] Shift-Dependent Asymmetry: Orthogonal Inverse Low-Rank Adaptation for Federated Medical Segmentation](../../ICML2026/medical_imaging/shift-dependent_asymmetry_orthogonal_inverse_low-rank_adaptation_for_federated_m.md)

</div>

<!-- RELATED:END -->

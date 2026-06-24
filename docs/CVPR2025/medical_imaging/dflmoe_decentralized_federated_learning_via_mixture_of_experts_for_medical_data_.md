---
title: >-
  [Paper Note] DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data
description: >-
  [CVPR 2025][Medical Imaging][federated learning] DFLMoE is proposed to handle medical data heterogeneity in decentralized federated learning using a Mixture of Experts (MoE) mechanism, enabling collaborative training without a central server while preserving privacy.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "federated learning"
  - "mixture of experts"
  - "decentralized"
  - "medical data"
  - "privacy"
date: 2026-05-08
content_hash: 7789a014a61ba4dc
---

# DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data

**Conference**: CVPR 2025  
**arXiv**: [2503.10412](https://arxiv.org/abs/2503.10412)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: federated learning, mixture of experts, decentralized, medical data, privacy

## TL;DR
DFLMoE is proposed to handle medical data heterogeneity in decentralized federated learning using a Mixture of Experts (MoE) mechanism, enabling collaborative training without a central server while preserving privacy.

## Background & Motivation

### Background

**Background**: Significant progress has been made in the direction of DFLMoE in recent years, but key challenges still remain.

**Limitations of Prior Work**: Existing methods exhibit deficiencies in generalizability, efficiency, or robustness, which limits their practical application. Specifically, most methods operate under specific assumptions, making it difficult to cope with real-world diversity.

**Key Challenge**: The trade-off between performance and efficiency/generalizability acts as the core challenge. There is a need to enhance the practical utility of the models while maintaining high performance.

**Goal**: To design a more efficient, robust, and generalizable solution to overcome the aforementioned limitations.

**Key Insight**: Each client maintains a local expert network and selects relevant experts to process different types of medical data through a dynamic routing mechanism.

**Core Idea**: DFLMoE is proposed to handle medical data heterogeneity in decentralized federated learning using a Mixture of Experts (MoE) mechanism.

## Method

### Overall Architecture
Each client maintains a local expert network and selects relevant experts to process different types of medical data through a dynamic routing mechanism. Decentralized communication only exchanges parameters of the selected experts rather than the entire model.

### Key Designs

1. **Core Module**

    - Function: Implements the core functionality of the method
    - Mechanism: Each client maintains a local expert network and selects relevant experts to handle different types of medical data via a dynamic routing mechanism
    - Design Motivation: Addressing the core limitations of existing methods

2. **Auxiliary Module**

    - Function: Enhancing the effectiveness of the core module
    - Mechanism: Boosting performance through additional constraints or information
    - Design Motivation: Compensating for the shortcomings when the core module is used in isolation


3. **Optimization Strategy**

    - Function: Improving training stability and convergence speed
    - Mechanism: Adopting appropriate learning rate scheduling, gradient clipping, and regularization strategies
    - Design Motivation: Ensuring the training efficiency of the model on large-scale data

### Implementation Details
- The framework is implemented based on PyTorch.
- Standard data augmentation strategies are employed to enhance generalizability.
- Both training and inference are executed efficiently on GPUs.

### Loss & Training
- A loss function that integrates multiple objectives is formulated to balance performance across various aspects.

## Key Experimental Results

### Main Results

| Method | Key Metrics | Explanation |
|------|---------|------|
| Baseline Methods | Lower | Limitations exist |
| **Ours** | **Higher** | Outperforms centralized federated learning methods such as FedAvg and FedProx on multiple medical imaging datasets |

### Ablation Study

| Component | Effect |
|------|------|
| Core Module | Main contribution |
| Auxiliary Module | Additional improvement |
| Full | Best |

### Key Findings
- Outperforms centralized federated learning methods such as FedAvg and FedProx on multiple medical imaging datasets.
- The components are complementary and indispensable to one another.

## Highlights & Insights
- The design concept of proposing DFLMoE to handle medical data heterogeneity using a Mixture of Experts (MoE) mechanism in decentralized federated learning is highly novel.
- Displays strong application potential in real-world scenarios.
- The methodology framework is generalizable and can be extended to related tasks.

## Limitations & Future Work
- Validation on more datasets and scenarios.
- Computational efficiency can be further optimized.
- The complementarity with other methods is worth exploring.

## Related Work & Insights
- Compared with existing representative methods, the proposed method demonstrates significant advantages in key metrics.
- The proposed concepts can inspire research in related fields.

## Rating
- Novelty: ⭐⭐⭐⭐ The core idea is innovative
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear structure
- Value: ⭐⭐⭐⭐ Promising prospects for practical application

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)
- [\[ICML 2026\] EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](../../ICML2026/medical_imaging/eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)
- [\[ICLR 2026\] Mixture of Mini Experts: Overcoming the Linear Layer Bottleneck in Multiple Instance Learning](../../ICLR2026/medical_imaging/mixture_of_mini_experts_overcoming_the_linear_layer_bottleneck_in_multiple_insta.md)
- [\[AAAI 2026\] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](../../AAAI2026/medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)
- [\[ICML 2025\] I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts](../../ICML2025/medical_imaging/i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts.md)

</div>

<!-- RELATED:END -->

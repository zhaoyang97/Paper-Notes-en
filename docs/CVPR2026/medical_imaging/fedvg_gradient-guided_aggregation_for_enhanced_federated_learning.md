---
title: >-
  [Paper Note] FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning
description: >-
  [CVPR2026][Medical Imaging][Federated Learning] FedVG proposes to score each client using layer-wise gradient norms computed on a global validation set…
tags:
  - "CVPR2026"
  - "Medical Imaging"
  - "Federated Learning"
  - "Gradient Aggregation"
  - "Data Heterogeneity"
  - "Validation Gradient"
  - "Fisher Information Matrix"
  - "Medical Image Classification"
date: 2026-05-08
content_hash: 48f893dac89203ed
---

# FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning

**Conference**: CVPR2026
**arXiv**: [2602.21399](https://arxiv.org/abs/2602.21399)  
**Code**: [Project Page](https://machine-intelligence-lab-wvu.github.io/fedvg/)  
**Area**: Medical Imaging / Federated Learning
**Keywords**: Federated Learning, Gradient Aggregation, Data Heterogeneity, Validation Gradient, Fisher Information Matrix, Medical Image Classification

## TL;DR

FedVG proposes to score each client using layer-wise gradient norms computed on a global validation set, assigning higher aggregation weights to clients whose gradients are flatter (i.e., smaller in norm), thereby substantially improving generalization performance of federated learning under high data heterogeneity.

## Background & Motivation

1. **Client drift in federated learning**: Standard FedAvg aggregates by data volume, ignoring model drift caused by heterogeneous client data distributions (client drift), which severely degrades global model performance in non-IID settings.
2. **Data quantity ≠ model quality**: Existing methods assume that clients with more data produce more reliable models; however, under heterogeneous distributions, clients with large amounts of skewed data may instead harm the global model.
3. **Over-emphasis on poorly performing clients**: Some methods assign excessively high weights to underperforming clients as a form of "compensation," which in turn amplifies aggregation bias.
4. **Bias in local gradients**: Gradients computed solely from local client data are inherently biased toward local distributions and cannot objectively reflect a client model's generalization ability.
5. **Ignored layer-wise behavioral differences**: Different layers exhibit distinct convergence behaviors and degrees of drift under non-IID conditions—deeper layers in particular are more susceptible to local bias—yet existing methods rarely conduct layer-wise analysis.
6. **Urgent demand in healthcare**: Medical imaging data is naturally distributed across institutions and cannot be centrally shared due to privacy regulations, necessitating robust federated learning solutions for training high-quality diagnostic models.

## Method

### Overall Architecture

FedVG introduces a **global validation set** $D_{\text{val}}$ (which can be constructed from publicly available datasets) at the server side of the standard federated learning pipeline to evaluate each client model's generalization ability:

1. Each client completes local training and uploads model parameters $\theta_k$ to the server.
2. The server performs forward and backward passes on $D_{\text{val}}$ for each client model to compute validation gradients.
3. Client scores $s_k$ are computed based on layer-wise gradient norms, and weighted aggregation is performed.

### Key Designs: Validation Gradient Scoring

**Layer-wise gradient norm computation**: The client model $\theta_k$ is decomposed into $L$ layers; the gradient norm of each layer on the validation set is computed and averaged:

$$\bar{G}_k = \frac{1}{L} \sum_{\ell=1}^{L} \left\| \nabla_{\theta_k^{(\ell)}} \mathcal{L}_{\text{val}} \right\|$$

**Inverse-proportional scoring**: A smaller gradient norm (flatter loss landscape) indicates stronger generalization and thus a higher aggregation weight:

$$s_k = \frac{1/(\bar{G}_k + \epsilon)}{\sum_{j=1}^{K} 1/(\bar{G}_j + \epsilon)}$$

**Theoretical grounding**: The cross-entropy gradient is the score function of the negative log-likelihood, and its norm is proportional to the diagonal approximation of the Fisher information matrix (Joint Fisher). Small Fisher information corresponds to flat minima regions and better generalization, providing an information-theoretic foundation for FedVG.

### Modular Integration

FedVG does not modify the local training procedure on the client side; it only replaces the server-side aggregation weights $s_k$. It can be seamlessly integrated with FedAvg, FedProx, Scaffold, FedDyn, FedAvgM, Elastic, and other algorithms (by averaging or directly substituting weights), serving as a plug-and-play module.

### Loss & Training

Clients perform local training using the original task loss (e.g., cross-entropy). FedVG itself **introduces no additional loss terms**; it adjusts aggregation weights solely via validation gradient norms, leaving client-side computational overhead unchanged.

## Key Experimental Results

### Main Results: Performance Under Varying Heterogeneity Levels

Experiments are conducted on CIFAR-10 (ResNet-18), OrganAMNIST, and COVID19 (ResNet-50), with heterogeneity controlled via Dirichlet $\alpha \in \{100, 10, 1, 0.1, 0.05\}$:

| Method | CIFAR-10 (α=0.05) | OrganAMNIST (all α) | COVID19 (α=0.05) |
|--------|-------------------|---------------------|------------------|
| FedAvg | Significantly below FedVG | Consistently below FedVG | Below FedVG |
| FedProx | Below FedVG | Below FedVG | Below FedVG |
| Scaffold | Moderate | Below FedVG | Close but below FedVG |
| FedDyn | Significantly below FedVG (p<0.05) | Below FedVG | Below FedVG |
| **FedVG** | **Highest / near-highest** | **Best across all α** | **Best at α=0.05** |

- Wilcoxon test: FedVG significantly outperforms FedDyn (p<0.05) across all α levels; no baseline significantly outperforms FedVG at any α level.
- ViT experiments (ViT-S/16, ViT-B/16): FedVG also achieves the best performance under high heterogeneity, confirming generalizability to non-CNN architectures.

### External Validation Set Experiments

Using STL-10 and CIFAR-100 as external validation sets (distribution-shifted relative to training data) at α=0.1/0.05:

| Validation Set | α=0.1 | α=0.05 |
|----------------|-------|--------|
| Original (CIFAR-10 subset) | 61.06% | 53.58% |
| STL-10 | 59.32% | 53.85% |
| CIFAR-100 | 58.83% | 52.62% |

Even when distribution shift exists in the validation set, FedVG maintains performance superior to baselines.

### Ablation Study

- **Class imbalance in validation set**: As the imbalance ratio ρ→0, FedVG consistently outperforms FedAvg, confirming robustness to imbalanced validation sets.
- **Norm type**: Both L1 and L2 norms correctly identify high-quality clients and assign them higher weights; spectral norm and delta norm perform worse. L1 (70.36%) and L2 (70.43%) are comparable, both outperforming spectral norm (68.50%).
- **Aggregation granularity**: Model-level aggregation (default) performs best on CIFAR-10/OrganAMNIST; layer-level/block-level aggregation shows marginal advantages on COVID19/DermaMNIST (ResNet-50). The optimal granularity depends on architecture and data characteristics.

## Highlights & Insights

- **Simple yet effective**: The core idea is clear—using validation gradient flatness to measure generalization—without requiring complex regularization or control variates.
- **Solid theoretical foundation**: An explicit connection to the Fisher information matrix is established, providing an information-theoretic interpretation of gradient norm as a generalization indicator.
- **Plug-and-play**: Only the server-side aggregation weights are modified; the client training pipeline remains unchanged, enabling seamless integration with six mainstream FL algorithms.
- **Comprehensive evaluation**: 5 datasets × 3 architectures (ResNet-18/50, ViT) × 5 heterogeneity levels × multiple ablations constitute a rigorous experimental design.
- **Privacy-friendly**: The validation set consists of public data, and all gradient computations are performed server-side, imposing no additional computational burden on clients.

## Limitations & Future Work

- **Validation set construction assumption**: A publicly available dataset relevant to the target task is required as the validation set, which may not be readily accessible in specialized domains such as healthcare.
- **Risk of overlap between validation and client data**: If the validation set shares domain similarity or samples with certain clients, unfair bias may be introduced.
- **Additional server-side overhead**: Each round requires full forward and backward passes over all participating client models, which can be computationally expensive when the number of clients or model size is large.
- **Aggregation granularity selection**: The relative performance of model-level, layer-level, and block-level aggregation varies across scenarios, and no adaptive selection mechanism is provided.
- **More complex FL settings not covered**: Scenarios such as heterogeneous model architectures, varying local training epochs, and dynamic client participation are not addressed.

## Related Work & Insights

| Method | Mechanism | Difference from FedVG |
|--------|-----------|----------------------|
| FedAvg | Weighted aggregation by data volume | Ignores model quality; poor performance under high heterogeneity |
| FedProx | Proximal regularization to constrain local drift | Modifies client side; FedVG only modifies server side |
| Scaffold | Control variates to correct gradients | Requires additional communication of control variates; FedVG does not |
| FedDyn | Dynamic regularization to align stable points | Statistically significantly worse than FedVG |
| Elastic | Interpolation aggregation based on parameter sensitivity | Strong baseline; FedVG+Elastic can yield further improvement |
| FedMD/FedDF | Knowledge distillation using public data | Public data used for training/distillation; FedVG uses it only for validation gradient computation |
| FedNCL/FedMA | Layer-wise aggregation | Focuses on layer-wise alignment but does not leverage validation gradient scoring |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of using validation gradient norms as a client generalization indicator is novel, and the theoretical connection to Fisher information is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five datasets, multiple architectures, multiple heterogeneity levels, external validation sets, and four types of ablations provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, theoretical derivations are complete, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — The method is concise, practical, and plug-and-play, with direct applicability to federated learning in privacy-sensitive domains such as healthcare.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Incentives in Federated Learning with Heterogeneous Agents](../../ICLR2026/medical_imaging/incentives_in_federated_learning_with_heterogeneous_agents.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[AAAI 2026\] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification](../../AAAI2026/medical_imaging/federated_clip_for_resource-efficient_heterogeneous_medical_image_classification.md)
- [\[CVPR 2026\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learningbased_assessment_of_the_relation_betw.md)
- [\[CVPR 2026\] Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](better_than_average_spatially-aware_aggregation_of_segmentation_uncertainty_impr.md)

</div>

<!-- RELATED:END -->

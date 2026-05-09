---
title: >-
  [Paper Note] Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation
description: >-
  [ICCV2025][LLM Safety][federated unlearning] This paper proposes FUCRT, a federated unlearning method based on class-aware representation transformation. Rather than directly erasing the representations of forget classes, FUCRT transforms them toward the semantically nearest retain classes, and employs dual contrastive learning to align transformation consistency across clients. The method guarantees 100% unlearning on four datasets while maintaining or even improving performance on retain classes.
tags:
  - ICCV2025
  - LLM Safety
  - federated unlearning
  - class-aware representation
  - contrastive learning
  - transformation alignment
  - Non-IID
date: 2026-05-08
content_hash: a9241139fdff1e26
---

# Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation

**Conference**: ICCV2025
**arXiv**: [2410.06848](https://arxiv.org/abs/2410.06848)
**Code**: To be confirmed
**Area**: Federated Learning / Machine Unlearning / Privacy Protection
**Keywords**: federated unlearning, class-aware representation, contrastive learning, transformation alignment, Non-IID

## TL;DR

This paper proposes FUCRT, a federated unlearning method based on class-aware representation transformation. Rather than directly erasing the representations of forget classes, FUCRT transforms them toward the semantically nearest retain classes, and employs dual contrastive learning to align transformation consistency across clients. The method guarantees 100% unlearning on four datasets while maintaining or even improving performance on retain classes.

## Background & Motivation

Federated Learning (FL) enables multiple clients to collaboratively train a global model without sharing raw data. With the emergence of privacy regulations such as GDPR and CCPA, clients are entitled to the "right to be forgotten," requiring the removal of specific class data from already-trained models. Federated Unlearning (FU) is proposed to address this need.

### Limitations of Prior Work

Existing FU methods fall into three main categories:

**Gradient Ascent** (e.g., Halimi et al.): Tightens generalization bounds via reverse gradient ascent, but frequently leads to severe model degradation (retain class accuracy drops to ~10%).

**Channel Pruning** (FUDP): Leverages TF-IDF to prune channels most correlated with the forget class, but causes excessive pruning when multiple classes are forgotten, degrading retain class performance.

**Momentum Degradation** (FUMD): Erases forget class information using a homomorphically randomly initialized degraded model, yet retain class performance remains suboptimal.

All of these methods treat forget data as an adversary to be eliminated, overlooking the potential representational relationships between forget and retain data.

### Key Observations

Through t-SNE visualization of the representation space on CIFAR-10, the authors identify two important phenomena:

- **Observation 1**: In the representation space of a model retrained from scratch, forget class data (e.g., "automobile") loses separability and intermingles with retain class data, while retain class data maintains compact and separable cluster structures.
- **Observation 2**: The distribution of forget class data exhibits a non-random pattern — it concentrates within the representation regions of specific retain classes (e.g., "automobile" migrates toward "truck" and "airplane"), suggesting that the unlearning process preferentially "forgets toward" semantically similar retain classes.

These observations motivate the insight that forgetting and retaining are not opposites, but can be realized through transformation in the representation space.

## Method

### Overall Architecture: FUCRT

FUCRT (Federated Unlearning via Class-aware Representation Transformation) preserves the standard client-server FL architecture and introduces class-aware representation transformation to perform unlearning. The framework consists of two core components:

1. **Transformation Class Selection** — determines which retain class each forget sample should be "transformed toward."
2. **Transformation Alignment** — ensures consistency of transformations across clients.

### Transformation Class Selection

#### Generating the Global Transformation Class Set

Intuitively, in the representation space, the transformation class for a forget class is typically the class most easily confused with it. The procedure is as follows:

1. For each client, compute the model's output probability vectors on correctly classified forget class samples.
2. Extract the **second-highest probability output** and its corresponding class for each sample (leveraging the high performance of the original model).
3. Aggregate second-highest probabilities by class and select classes whose aggregated values exceed a threshold to form the local transformation class set.
4. Local sets are only generated when the number of forget class samples on a client exceeds a threshold, to avoid bias from data sparsity.
5. The server aggregates local sets from all clients, determines the set size via majority voting, and selects specific classes via frequency voting to produce the global transformation class set.

#### Assigning a Transformation Class to Each Forget Sample

During local unlearning training, for each forget sample, the class with the highest probability output within the global transformation class set is selected as the transformation target. This strategy respects global consensus while utilizing the model's current knowledge of each individual sample.

### Transformation Alignment: Dual Class-Aware Contrastive Learning

Replacing forget data labels with transformation class labels and fine-tuning with cross-entropy loss achieves basic representation transformation. However, Non-IID data distributions cause representational discrepancies across clients, necessitating resolution of transformation inconsistency.

**Local Class-Aware Contrastive Loss**: Based on in-batch sample representations, pulls together samples of the same class and pushes apart samples of different classes, optimizing local representation space structure.

**Global Class-Aware Contrastive Loss**: Incorporates server-aggregated global class prototypes to pull each sample's representation toward the corresponding global prototype, achieving local-global alignment.

**Overall Local Unlearning Training Loss**: Cross-entropy loss + local contrastive loss + global contrastive loss, with hyperparameters controlling the weight of each term.

### Unlearning Procedure

1. Each client computes its local transformation class set based on the original global model and uploads it to the server.
2. The server aggregates the local sets to produce the global transformation class set.
3. Iterative unlearning over $R$ rounds: the server distributes the global model and global class prototypes → clients perform label transformation and contrastive learning training → clients upload models and local prototypes → the server performs FedAvg aggregation.
4. Clients holding only retain class data perform transformation alignment only (without label transformation).

## Key Experimental Results

### Experimental Setup

- **Datasets**: CIFAR-10, CIFAR-100, FMNIST, EuroSAT
- **Architectures**: ResNet-18 (CIFAR-10/FMNIST/EuroSAT), ResNet-50 (CIFAR-100)
- **Baselines**: Fine-tune, Gradient Ascent, FUDP, FUMD, From-scratch (upper bound)
- **Forget class ratio**: 10%

### Main Results (Forget Class ACC / Retain Class ACC)

| Dataset | Method | Forget ACC (IID) | Retain ACC (IID) | Retain ACC (Non-IID) |
|---------|--------|-----------------|-----------------|----------------------|
| CIFAR-10 | Gradient Ascent | 0.00% | 10.94% | 11.15% |
| CIFAR-10 | FUDP | 0.00% | 84.85% | 85.68% |
| CIFAR-10 | FUMD | 0.00% | 82.38% | 82.60% |
| CIFAR-10 | **FUCRT** | **0.00%** | **90.06%** | **89.82%** |
| CIFAR-10 | From-scratch | 0.00% | 87.99% | 85.44% |
| CIFAR-100 | FUDP | 0.03% | 72.25% | 69.71% |
| CIFAR-100 | **FUCRT** | **0.00%** | **75.25%** | **74.88%** |
| FMNIST | **FUCRT** | **0.00%** | **90.82%** | **90.95%** |
| EuroSAT | **FUCRT** | **0.00%** | **94.92%** | - |

Key findings:

1. **100% unlearning guarantee**: FUCRT achieves forget class ACC = 0% across all datasets.
2. **Best retain class performance**: Significantly outperforms all baselines; on CIFAR-10, even surpasses retraining from scratch (90.06% vs. 87.99%).
3. **Robust across settings**: Consistently achieves best results under both IID and Non-IID conditions.
4. **Catastrophic degradation of gradient ascent**: Retain class accuracy drops to ~10%, demonstrating the fragility of adversarial approaches.

## Highlights & Insights

- This work is the first to examine federated unlearning from a representation space perspective, uncovering the transformational relationship between forget and retain classes.
- The "transformation" strategy is more elegant than "erasure," simultaneously guaranteeing unlearning and preserving retain class performance.
- Dual contrastive learning effectively addresses transformation consistency under Non-IID conditions.
- Retain class performance exceeding retraining from scratch suggests that representation transformation effectively acts as a regularizer.

## Limitations & Future Work

- Additional communication overhead is incurred by transmitting the global transformation class set and class prototypes.
- The method assumes all servers and clients are benign, without considering malicious participants.
- Transformation class selection depends on the quality of the original model's probability outputs.
- The transmission of global class prototypes poses a potential privacy leakage risk (prototypes may encode information about data distributions); future work could consider differential privacy protection.
- The scalability of this framework to scenarios with many forget classes warrants further investigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](../../CVPR2026/llm_safety/damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[AAAI 2026\] Beyond Superficial Forgetting: Thorough Unlearning through Knowledge Density Estimation and Block Re-insertion](../../AAAI2026/llm_safety/beyond_superficial_forgetting_thorough_unlearning_through_knowledge_density_esti.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)
- [\[ICCV 2025\] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)
- [\[NeurIPS 2025\] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation](../../NeurIPS2025/llm_safety/geo-sign_hyperbolic_contrastive_regularisation_for_geometrically_aware_sign_lang.md)

</div>

<!-- RELATED:END -->

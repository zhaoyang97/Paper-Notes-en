---
title: >-
  [Paper Note] Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images
description: >-
  [ECCV 2024][Model Compression][Machine Unlearning] Proposes SCAR (Selective-distillation for Class and Architecture-agnostic unleaRning), a retain-set-free approximate unlearning algorithm that guides the feature vectors of forgotten samples toward the nearest incorrect class distribution via Mahalanobis distance, and utilizes OOD image distillation to maintain model performance.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Machine Unlearning"
  - "Knowledge Distillation"
  - "Mahalanobis Distance"
  - "Retain-Set-Free"
  - "OOD Data"
date: 2026-05-08
content_hash: 78ab2344bc4a9cd2
---

# Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images

**Conference**: ECCV 2024  
**arXiv**: [2404.12922](https://arxiv.org/abs/2404.12922)  
**Code**: [Yes](https://github.com/jbonato1/SCAR)  
**Area**: Model Compression / Machine Unlearning  
**Keywords**: Machine Unlearning, Knowledge Distillation, Mahalanobis Distance, Retain-Set-Free, OOD Data

## TL;DR

Proposes SCAR (Selective-distillation for Class and Architecture-agnostic unleaRning), a retain-set-free approximate unlearning algorithm that guides the feature vectors of forgotten samples toward the nearest incorrect class distribution via Mahalanobis distance, and utilizes OOD image distillation to maintain model performance.

## Background & Motivation

Machine Unlearning aims to remove the information of specific data from a trained model while maintaining its performance on the remaining data. This demand is driven by privacy regulations such as GDPR and CCPA, where users have the right to request the deletion of the impact of their data on the model.

The core dilemma of existing approximate unlearning methods lies in their dependence on the **retain set** ($\mathcal{D}_r$):

**Privacy Restrictions**: In certain scenarios, only the forget set $\mathcal{D}_f$ is accessible due to privacy concerns, and the retain set is completely unavailable.

**Efficiency Issues**: When the forget set constitutes only a tiny fraction of a massive dataset like ImageNet, utilizing a huge retain set to recover model performance is highly inefficient.

**Extreme Scenarios**: In class removal, even the forget set may be unavailable, with only the target class IDs provided.

The authors observe a key phenomenon: DNNs also produce highly confident predictions for out-of-distribution (OOD) data, and the feature vectors of OOD data cluster in the same regions of the feature space as the training data. This implies that OOD data can replace the retain set to preserve model knowledge—which constitutes the core motivation of SCAR.

Compared to DUCK (which uses centroids) and Boundary Unlearning (which uses nearest incorrect class labels), SCAR leverages the complete class feature distribution information via Mahalanobis distance rather than focusing solely on centroids or labels.

## Method

### Overall Architecture

SCAR decomposes the DNN $\Phi_\theta$ into a backbone $\Phi_\psi$ and a final fully connected layer $\Phi_\pi$. The unlearning process is driven by two cooperative mechanisms: metric learning for unlearning and a distillation trick for maintaining performance. It supports two scenarios:
- **Class Removal (CR)**: Removing knowledge of an entire class.
- **Homogeneous Removal (HR)**: Removing specific samples distributed across multiple classes.

### Key Designs

1. **Metric Learning Unlearning based on Mahalanobis Distance**

   During the training phase, the feature vector distribution $Q_i$ for each class $i$ is recorded, characterized by its mean $\mu_i$ and covariance matrix $\hat{S}_Q$. For a forgotten sample $(x_j, y_j=k)$, the nearest non-target class distribution is identified:

    $$Q_j^* = \operatorname{argmin}_{Q_i, i \neq k} d_M(\Phi_\psi^U(x_j), Q_i)$$

   The Mahalanobis distance is defined as:

    $$d_M(\Phi_\psi^U(x_j), Q_i) = \sqrt{(\Phi_\psi^U(x_j) - \mu_Q)^T \hat{S}_Q^{-1} (\Phi_\psi^U(x_j) - \mu_Q)}$$

   The unlearning loss minimizes the distance of the forgotten samples to the nearest incorrect class distribution:

    $$\mathcal{L}_M = \frac{1}{N_{f,\text{batch}}} \sum_{j=0}^{N_{f,\text{batch}}-1} d_M(\Phi_\psi^U(x_j), Q_j^*)$$

   Key tricks include: covariance matrix correlation normalization (eliminating scale differences across classes), covariance shrinkage (addressing singularity issues when the number of samples < feature dimensions), and Tukey normalization (making features approximately Gaussian).

2. **Distillation-Trick**

   Core insight: The highly confident outputs produced by DNNs on OOD data can serve as carriers for knowledge retention. Using an external OOD dataset $\mathcal{D}^{\text{sur}}$ (such as a subset of ImageNet), the knowledge of the original model (teacher) is distilled into the unlearned model via Jensen-Shannon divergence:

    $$\mathcal{L}_{TD} = \frac{1}{N_{r,\text{batch}}} \sum_j d_{JS}(\Phi_\theta^U(x_j) \| \Phi_\theta(x_j))$$

   where the JS divergence is:

    $$d_{JS}(\Phi_\theta^U \| \Phi_\theta / T) = \frac{1}{2} D_{KL}(\Phi_\theta^U \| \Phi_\theta / T) + \frac{1}{2} D_{KL}(\Phi_\theta / T \| \Phi_\theta^U)$$

   where $T$ is the temperature parameter. This enables the model to maintain its classification capability on retain classes while executing unlearning.

3. **SCAR Self-Forget (Forget-Set-Free Version)**

   For extreme scenarios where only the ID of the class to be removed is known: the original model is used to classify OOD data. OOD samples predicted as the targeted class are used as a surrogate for the forget set $\mathcal{D}_f^{\text{sur}}$, while the rest serve as a surrogate for the retain set $\mathcal{D}_r^{\text{sur}}$. This represents the first method in CR scenarios that requires neither the retain set nor the forget set.

### Loss & Training

The total loss is: $\mathcal{L} = \lambda_1 \mathcal{L}_M + \lambda_2 \mathcal{L}_{TD}$

An adaptive stopping strategy is adopted:
- CR scenario: stops when the forget set accuracy $\mathcal{A}_f \leq 0$.
- HR scenario: stops when the forget set accuracy is close to the test set accuracy.
- If conditions are not met, training stops at a maximum number of epochs.

## Key Experimental Results

### Main Results

Class Removal (CR) scenario, ResNet18 backbone, average of 10 runs:

| Method | Requires $\mathcal{D}_r$ | CIFAR100 $\mathcal{A}_r^t$↑ | CIFAR100 $\mathcal{A}_f^t$↓ | CIFAR100 AUS↑ | TinyImgNet AUS↑ |
|------|:---:|------|------|------|------|
| Retrained | Yes | 77.97 | 0.00 | 1.004 | 0.993 |
| SCRUB | Yes | 77.29 | 2.00 | 0.977 | 0.986 |
| DUCK | Yes | 71.57 | 1.00 | 0.931 | 0.927 |
| Neg. Grad. | No | 62.84 | 0.50 | 0.849 | 0.911 |
| Rand. Lab. | No | 55.31 | 0.40 | 0.774 | 0.740 |
| **SCAR** | **No** | **72.93** | **2.00** | **0.935** | **0.940** |
| SCAR Self-Forget | No (+ No $\mathcal{D}_f$) | 71.09 | 0.70 | 0.929 | 0.917 |

Homogeneous Removal (HR) scenario:

| Method | Requires $\mathcal{D}_r$ | CIFAR100 $\mathcal{A}^t$↑ | CIFAR100 AUS↑ | TinyImgNet AUS↑ |
|------|:---:|------|------|------|
| DUCK | Yes | 74.74 | 0.965 | 0.916 |
| Fine Tuning | Yes | 72.06 | 0.918 | 0.937 |
| Neg. Grad. | No | 60.83 | 0.718 | 0.770 |
| **SCAR** | **No** | **73.23** | **0.934** | **0.886** |

### Ablation Study

Ablation on loss components (CIFAR100 CR scenario):

| $\mathcal{L}_{TD}$ | $\mathcal{L}_M$ | $\mathcal{A}_r^t$ | $\mathcal{A}_f^t$ | AUS | Description |
|:---:|:---:|------|------|------|------|
| ✗ | ✗ | 77.55 | 77.50 | 0.563 | Original model, no unlearning |
| ✓ | ✗ | 72.09 | 40.00 | 0.675 | Distillation only, insufficient unlearning |
| ✗ | ✓ | 66.90 | 2.60 | 0.871 | Metric learning only, retain accuracy drops significantly |
| ✓ | ✓ | 72.93 | 2.00 | **0.935** | Full SCAR, both components are indispensable |

Comparison of distance metrics (CIFAR100 CR):

| Metric | AUS (CIFAR100) | AUS (TinyImgNet) |
|------|------|------|
| Cosine Similarity | 0.933 | 0.912 |
| L2 Distance | 0.924 | 0.881 |
| **Mahalanobis** | **0.935** | **0.940** |

### Key Findings

- Under retain-set-free conditions, the AUS scores of SCAR are close to or even exceed those of DUCK and other methods that utilize the retain set.
- The Self-Forget variant still achieves performance comparable to standard SCAR without any access to the training data.
- Surrogate datasets (ImageNet subsets, COCO, natural images) are all effective, whereas pure Gaussian noise fails, indicating that semantic information is crucial for distillation.
- Utilizing distribution information via Mahalanobis distance outperforms centroid-only methods and is 22% faster than Cosine similarity.
- SCAR is effective across various architectures (AllCNN, ResNet18/34/50, ViT-B16), confirming its architecture-agnostic property.

## Highlights & Insights

- **Valuable Problem Setting**: Challenges the assumption that "the retain set is mandatory for unlearning," which is more practical in privacy-restricted scenarios.
- **Clever OOD Distillation Trick**: Leverages the "overconfidence" of DNNs on OOD data as an advantage.
- **Introduction of Mahalanobis Distance to Leverage Distribution Information**: Provides more precise feature-space guidance compared to centroid-based methods.
- **Pioneering Self-Forget**: Achieves the first class removal method that requires neither the retain set nor the forget set.

## Limitations & Future Work

- There is a slight drop in retain set accuracy after unlearning (e.g., from 77.55 to 72.93 on CIFAR100), showing a performance gap compared to retain-set-based methods.
- It requires storing the mean and covariance matrix for each class, which may incur storage overhead in scenarios with an extremely large number of classes.
- Mathematical guarantees of certifiability are not provided.
- Self-Forget is not applicable in the HR scenario because specific forgotten instances must be identified.
- Validation on larger-scale datasets (such as the full ImageNet) is not presented.

## Related Work & Insights

- Difference from DUCK: DUCK uses centroid distance, while SCAR uses distribution distance (Mahalanobis).
- Difference from Boundary Unlearning: The latter modifies labels, whereas SCAR operates in the feature space.
- The concept of OOD distillation can be extended to other scenarios requiring model knowledge retention in the absence of original data (e.g., federated unlearning, model editing).
- The use of covariance shrinkage and Tukey normalization provides a practical reference for distance computation in high-dimensional feature spaces.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of retain-set-free unlearning and OOD distillation is novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablation study and validation across 3 datasets, 2 scenarios, and multiple architectures.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and systematic description of the methodology.
- **Value**: ⭐⭐⭐⭐ — Addresses practical unlearning demands in privacy-restricted scenarios, with the Self-Forget variant being highly forward-looking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Uni-LoRA: One Vector is All You Need](../../NeurIPS2025/model_compression/uni-lora_one_vector_is_all_you_need.md)
- [\[NeurIPS 2025\] All You Need is One: Capsule Prompt Tuning with a Single Vector](../../NeurIPS2025/model_compression/all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [\[ECCV 2024\] SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)
- [\[ECCV 2024\] Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)
- [\[ECCV 2024\] Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)

</div>

<!-- RELATED:END -->

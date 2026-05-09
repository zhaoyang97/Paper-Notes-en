---
title: >-
  [Paper Note] FRET: Feature Redundancy Elimination for Test Time Adaptation
description: >-
  [ICCV 2025][AI Safety][Test-time adaptation] This paper proposes Feature Redundancy Elimination (FRET) as a novel perspective for test-time adaptation (TTA), observing that embedding feature redundancy increases significantly under distribution shift. Two methods are designed: S-FRET (direct minimization of the redundancy score) and G-FRET (GCN-based attention-redundancy decomposition with bi-level optimization). G-FRET achieves state-of-the-art performance across multiple architectures and datasets.
tags:
  - ICCV 2025
  - AI Safety
  - Test-time adaptation
  - feature redundancy elimination
  - distribution shift
  - graph convolutional network
  - contrastive learning
date: 2026-05-08
content_hash: 51a1e495bf21160d
---

# FRET: Feature Redundancy Elimination for Test Time Adaptation

**Conference**: ICCV 2025
**arXiv**: [2505.10641](https://arxiv.org/abs/2505.10641)
**Code**: [GitHub](https://anonymous.4open.science/r/fret-21BD)
**Area**: AI Safety
**Keywords**: Test-time adaptation, feature redundancy elimination, distribution shift, graph convolutional network, contrastive learning

## TL;DR
This paper proposes Feature Redundancy Elimination (FRET) as a novel perspective for test-time adaptation (TTA), observing that embedding feature redundancy increases significantly under distribution shift. Two methods are designed: S-FRET (direct minimization of the redundancy score) and G-FRET (GCN-based attention-redundancy decomposition with bi-level optimization). G-FRET achieves state-of-the-art performance across multiple architectures and datasets.

## Background & Motivation
Deep neural networks perform well under the i.i.d. assumption, but frequently encounter distribution shift in real-world deployment. TTA requires access only to a pre-trained model and unlabeled test data, making it particularly suitable for privacy-sensitive scenarios.

**Taxonomy of existing methods**:
- **BN calibration methods**: Replace source-domain BN statistics with target-domain statistics
- **Pseudo-label methods**: Select reliable pseudo-labels via thresholding or entropy filtering
- **Consistency training methods**: Maintain prediction stability under input perturbations
- **Clustering methods**: Reduce prediction uncertainty in the target domain via clustering

**Core observation**: On CIFAR10-C with ResNet-18, the authors find that as distribution shift intensifies, the redundancy of second-order feature relationship graphs (covariance matrices) increases markedly—a redder covariance heatmap indicates higher inter-feature correlation. Quantitative analysis shows that the redundancy score $R_e = \|\tilde{Z}^T\tilde{Z} - I_d\|_1$ positively correlates with corruption severity.

**Root Cause**: Existing TTA methods overlook the increase in feature redundancy caused by distribution shift, whereas redundant features directly undermine the model's ability to adapt to new data.

**Starting Point**: Directly eliminate embedding feature redundancy at test time, addressing TTA from a novel redundancy-elimination perspective.

## Method

### Overall Architecture
The FRET framework operates at two levels:
1. **S-FRET**: Directly adopts the redundancy score $R_e$ as the optimization objective—simple and efficient.
2. **G-FRET**: Introduces a GCN to decompose feature relationships into an attention component and a redundancy component, eliminating redundancy and enhancing discriminability at both the representation layer and the prediction layer.

### Key Designs
1. **Feature Redundancy Score**:

    - Function: Quantifies the degree of redundancy in embedding features.
    - Mechanism: The embedding matrix $Z$ is column-normalized to obtain $\tilde{Z}$; the redundancy score is then computed as $R_e = \|\tilde{Z}^T\tilde{Z} - I_d\|_1$. Ideally, non-redundant features should yield a covariance matrix close to the identity.
    - Design Motivation: Off-diagonal entries of the covariance matrix capture linear correlations between features; minimizing these entries eliminates redundancy.

2. **Attention-Redundancy Decomposition**:

    - Function: Decomposes the feature relationship graph into a useful attention component and a redundancy component to be eliminated.
    - Mechanism: A second-order feature relationship graph $G_F = Z^TZ$ is constructed, then decomposed via a mask matrix $M_M = I_d$ into an attention graph $G_A = G_F \odot I_d$ (retaining only the diagonal) and a redundancy graph $G_R = G_F - G_A$. GCN propagation then yields attention and redundancy representations:
    $R_A = Z D_A^{-1/2} G_A D_A^{-1/2}, \quad P_A = R_A \theta^h$
    $R_R = Z D_R^{-1/2} G_R D_R^{-1/2}, \quad P_R = R_R \theta^h$
    - Design Motivation: S-FRET, which directly minimizes the redundancy score, is label-distribution-agnostic and thus cannot handle label shift. By fusing data information with feature relationship information through the GCN, G-FRET can address both covariate shift and label shift.

3. **Representation-Layer Redundancy Elimination**:

    - Function: Uses contrastive learning to make attention representations class-discriminative while pushing them away from redundancy representations.
    - Mechanism: A contrastive loss $\mathcal{L}_R$ is defined, where the positive sample for attention representation $R_{A_i}$ is its corresponding class centroid $c_o$, and negatives include other class centroids $\{c_j\}$ and the redundancy representation $R_{R_i}$:
    $\mathcal{L}_R = -\sum_{i=1}^{n_t} \log \frac{\exp(\text{sim}(R_{A_i}, c_o))}{\sum_{j=1}^{C} \exp(\text{sim}(R_{A_i}, c_j)) + \exp(\text{sim}(R_{A_i}, R_{R_i}))}$
   Class centroids are computed via pseudo-label clustering.
    - Design Motivation: Redundancy elimination alone is insufficient; the discriminability of useful features must also be enhanced to handle label shift.

4. **Prediction-Layer Redundancy Elimination**:

    - Function: Increases the confidence of attention predictions while suppressing redundancy predictions at the prediction layer.
    - Mechanism: Combines entropy minimization with negative learning:
    $\mathcal{L}_P = -\sum_{i=1}^{N} \sigma(P_{A_i}) \log \sigma(P_{A_i}) - \sum_{i=1}^{N} \sigma(P_{R_i}) \log \sigma(1 - P_{A_i})$
   The first term minimizes the entropy of attention predictions (sharpening predictions); the second penalizes redundancy predictions via negative learning.
    - Design Motivation: Bi-level optimization (representation layer + prediction layer) is more effective than operating at a single layer.

### Loss & Training
- **S-FRET loss**: $\mathcal{L}_{SFRET} = \|\tilde{Z}^T\tilde{Z} - I_d\|_1$
- **G-FRET total loss**: $\mathcal{L}_{GFRET} = \mathcal{L}_R + \lambda \mathcal{L}_P$
- Online adaptation: Upon receiving test data, predictions are generated using the model parameters from the previous step, followed by a single gradient descent update.
- Only BN layer parameters are updated; all other parameters remain frozen.

## Key Experimental Results

### Main Results (Domain Generalization TTA — PACS + OfficeHome)

| Method | Backbone | PACS Avg | OfficeHome Avg |
|--------|----------|----------|----------------|
| Source | ResNet-18 | 81.84 | 62.01 |
| BN | ResNet-18 | 82.66 | 62.03 |
| TENT | ResNet-18 | 85.60 | 63.24 |
| TSD | ResNet-18 | 88.13 | 62.55 |
| TEA | ResNet-18 | 87.98 | 63.06 |
| TIPI | ResNet-18 | 87.23 | 63.29 |
| **G-FRET** | **ResNet-18** | **88.51** | **63.81** |
| TSD | ResNet-50 | 89.97 | 68.74 |
| TEA | ResNet-50 | 88.72 | 68.95 |
| **G-FRET** | **ResNet-50** | **91.28** | **69.96** |

### Ablation Study

| Configuration | PACS Avg | Note |
|---------------|---------|------|
| S-FRET (redundancy minimization only) | 86.20 | Simple and effective, but vulnerable to label shift |
| G-FRET w/o $\mathcal{L}_R$ | 87.53 | Without representation-layer contrastive learning |
| G-FRET w/o $\mathcal{L}_P$ | 87.89 | Without prediction-layer negative learning |
| G-FRET (full) | 88.51 | Bi-level optimization yields best results |
| $\lambda = 0.1$ | 87.92 | Balancing parameter too small |
| $\lambda = 1.0$ | 88.51 | Optimal balance |
| $\lambda = 10$ | 88.05 | Prediction-layer loss weight too large |

### Key Findings
- Feature redundancy positively correlates with the degree of distribution shift; this observation holds across multiple architectures and datasets.
- Despite its simplicity, S-FRET is already effective under covariate shift.
- Through attention-redundancy decomposition and bi-level optimization, G-FRET substantially outperforms S-FRET under label shift.
- Feature visualizations of G-FRET outputs demonstrate markedly reduced redundancy and enhanced discriminability.

## Highlights & Insights
- **Novel perspective**: This is the first work to introduce feature redundancy elimination into TTA, offering a direction orthogonal to BN calibration, pseudo-labeling, and consistency training.
- **Progressive method design**: S-FRET is concise and elegant (a single formula); G-FRET builds upon it by incrementally incorporating GCN, contrastive learning, and negative learning, with clear logical progression.
- **Effective use of GCN**: Graph propagation in the GCN is leveraged to model inter-feature relationships, enabling explicit separation of attention and redundancy relationships at the feature level.
- **Handling label shift**: By incorporating class-centroid-aware contrastive learning, G-FRET compensates for the limitation of pure redundancy minimization approaches.

## Limitations & Future Work
- G-FRET introduces GCN and contrastive learning, increasing computational overhead at test time (graph construction and propagation are required per batch).
- The mask matrix $M_M$ is fixed as the identity matrix, which may not be optimal for all scenarios.
- Performance gains under extreme distribution shift (e.g., corruption level 5) are limited.
- Class centroid computation relies on pseudo-label quality and may become unstable under high noise.
- Combinations with other TTA methods remain unexplored.

## Related Work & Insights
- The approach bears theoretical connections to the SOFT method in feature selection (sharing the mask matrix concept).
- The redundancy elimination idea may extend to other settings such as continual learning and domain adaptation.
- The bi-level framework of contrastive learning combined with negative learning may be applicable to other tasks requiring feature de-redundancy.
- The paper provides a new diagnostic tool for TTA method design (the redundancy score curve).

## Rating
- Novelty: ⭐⭐⭐⭐ The feature redundancy perspective is a novel contribution to TTA, though the individual technical components (GCN, contrastive learning) are not new in themselves.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple architectures (ResNet-18/50/ViT), multiple datasets (PACS/OfficeHome/CIFAR-C), and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is illustrated intuitively, and the method is described clearly, though the mathematical notation is somewhat dense.
- Value: ⭐⭐⭐⭐ Provides a practical new TTA method and novel understanding of feature redundancy, offering meaningful inspiration for subsequent research.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_doubleblind_federated_adaptation_of_foundati.md)
- [\[ICCV 2025\] Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation](controllable_feature_whitening_for_hyperparameter-free_bias_mitigation.md)
- [\[ICCV 2025\] Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning](active_membership_inference_test_amint_enhancing_model_auditability_with_multi-t.md)
- [\[NeurIPS 2025\] Incentivizing Time-Aware Fairness in Data Sharing](../../NeurIPS2025/ai_safety/incentivizing_time-aware_fairness_in_data_sharing.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](../../NeurIPS2025/ai_safety/nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)

<!-- RELATED:END -->

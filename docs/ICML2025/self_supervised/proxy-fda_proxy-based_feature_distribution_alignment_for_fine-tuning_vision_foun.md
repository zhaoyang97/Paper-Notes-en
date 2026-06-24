---
title: >-
  [Paper Note] Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting
description: >-
  [ICML 2025][Self-Supervised Learning][robust fine-tuning] This paper proposes a structure-level feature regularization method termed Proxy-FDA. By transferring the nearest neighbor graph from the pre-trained feature space to the fine-tuned feature space, and employing a lightweight proxy generator to synthesize novel features to enhance distribution coverage, Proxy-FDA achieves forward transfer across all fine-tuning tasks without sacrificing downstream accuracy.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "robust fine-tuning"
  - "concept forgetting"
  - "feature distribution alignment"
  - "proxy learning"
  - "vision foundation model"
date: 2026-05-08
content_hash: f4b185fa85e89b70
---

# Proxy-FDA: Proxy-based Feature Distribution Alignment for Fine-tuning Vision Foundation Models without Forgetting

**Conference**: ICML 2025  
**arXiv**: [2505.24088](https://arxiv.org/abs/2505.24088)  
**Code**: To be confirmed  
**Area**: Self-Supervised Learning  
**Keywords**: robust fine-tuning, concept forgetting, feature distribution alignment, proxy learning, vision foundation model

## TL;DR

This paper proposes a structure-level feature regularization method termed Proxy-FDA. By transferring the nearest neighbor graph from the pre-trained feature space to the fine-tuned feature space, and employing a lightweight proxy generator to synthesize novel features to enhance distribution coverage, Proxy-FDA achieves forward transfer across all fine-tuning tasks without sacrificing downstream accuracy.

## Background & Motivation

**Background**: Vision foundation models such as CLIP and DINOv2 learn rich real-world concept representations during pre-training. However, after fine-tuning on downstream tasks, they often lose the ability to recognize concepts from other tasks—a phenomenon known as concept forgetting.

**Limitations of Prior Work**: L2SP applies L2 regularization in the weight space, while LDIFS performs point-wise matching in the feature space. However, point-wise constraints are **overly strict and blind**—they do not account for the feature neighborhood structure. The local neighborhood of features encodes rich knowledge beyond class labels (e.g., the shared "white" attribute between two dog breeds), which point-wise matching fails to preserve.

**Key Challenge**: Point-wise constraints are both too restrictive (limiting the freedom of feature movement) and insufficient (failing to protect the fine-grained knowledge encoded by neighborhood structures), leading to inadequate knowledge preservation.

**Goal**: (1) Design structure-level feature regularization to preserve the local topology of the feature distribution; (2) Address the issue of insufficient FDA coverage caused by limited downstream data.

**Key Insight**: The authors found that OTDD (Optimal Transport Dataset Distance, which considers local structure) correlates much more strongly with concept forgetting than L2 feature distance—theoretically implying that preserving distribution structures is more effective than point-wise matching.

**Core Idea**: Transfer the nearest neighbor graph to preserve the topological structure of feature neighborhoods, and enhance data diversity via proxy feature generation to achieve structure-level defense against forgetting.

## Method

### Overall Architecture

In addition to the standard fine-tuning loss $\mathcal{L}_{\text{task}}$, an FDA regularization term is incorporated: $\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}(\mathcal{L}_{\text{task}}^i + \lambda\mathcal{L}_{\text{FDA}}^i)$. FDA is achieved by transferring the kNN graph from the pre-trained feature space to the fine-tuned feature space. Proxy-FDA further introduces a proxy generator to enhance FDA coverage with synthetic features.

### Key Designs

1. **Feature Distribution Alignment (FDA)**:

    - **Function**: Preserve the local neighborhood structure in the pre-trained feature space.
    - **Mechanism**: For each pre-trained feature point $\hat{x}_i$, a $K$-nearest neighbor set $R_i$ and cosine similarities $\hat{w}_{ij}$ are constructed within the batch. These neighborhood indices and similarities are directly transferred to the fine-tuned feature space. A Sigmoid contrastive loss (derived from SigLIP) is employed: $\mathcal{L}_{\text{FDA}}^i = \frac{1}{|X|-1}\sum_{j\neq i}\log(1+e^{w_{ij}(-\cos(x_i,x_j)/\tau+b)})$, where $w_{ij}$ takes positive pre-trained similarities for neighbors and negative values for non-neighbors.
    - **Design Motivation**: FDA preserves **relations** rather than **absolute positions**—allowing features to move during fine-tuning as long as the neighborhood topology remains unchanged. Cross-class neighborhoods convey knowledge that transcends class labels.

2. **Batch Construction and Hard Class Mining**:

    - **Function**: Ensure sufficient neighborhood richness within each batch.
    - **Mechanism**: Class-balanced sampling ($m=16$ classes $\times$ $n=4$ samples/class = 64 batch size) is coupled with hard class mining to prioritize nearby classes in the feature space. Setting $K > n$ ensures each neighborhood contains multiple classes.
    - **Design Motivation**: $K > n$ guarantees that neighborhoods cross class boundaries, allowing FDA to transfer cross-class knowledge (e.g., "white" across different white dog breeds). If cross-class similarity is low, FDA automatically degrades to class-semantic alignment.

3. **Proxy Generator**:

    - **Function**: Enhance data diversity for FDA in data-limited scenarios.
    - **Mechanism**: A lightweight network (1 attention layer + 2 convolutional layers, with only 23.6K parameters) is conditioned on the neighbor set $X_i^+$ and non-neighbor set $X_i^-$. It generates two sets of proxy features $P_i^+$ and $P_i^-$ along with their estimated similarities via adaptive pooling. The proxy learning loss consists of a contrastive term restricting proxies to the real feature manifold and a variance loss $\mathcal{L}_{\text{var}}$ encouraging diversity.
    - **Design Motivation**: When downstream data is scarce, real features in a batch are insufficient to describe complex distributions. The proxy synthesizes unseen data points—including **unseen class concepts**—providing fine-grained regularization at neighborhood boundaries. Online joint training ensures the proxy adapts to the evolving feature distribution.

### Loss & Training

Proxy-FDA extends FDA by appending the proxies to the real features and similarities: $\mathcal{L}_{\text{Proxy-FDA}}^i = \mathcal{L}_{\text{FDA}}^i(\{[X_i^+, P_i^+], [X_i^-, P_i^-]\}, \{[\hat{w}_i^+, \hat{w}_i^{p+}], [\hat{w}_i^-, \hat{w}_i^{p-}]\})$. The gradients of the proxy generator are derived from $\mathcal{L}_{\text{proxy}} = \mathcal{L}_{P_i^+} + \mathcal{L}_{P_i^-}$ (each containing contrastive and variance terms, with weight $\alpha$).

## Key Experimental Results

### End-to-end Fine-tuning (CLIP ViT-B/32, 10 Classification Datasets, Table 1)

| Method | Average $\mathcal{A}_{\text{LP}}$↑ | Average $\Delta_{\text{LP}}$↑ |
|------|------|------|
| Naive FT | 91.90 | -4.37 |
| LP-FT | 91.55 | -2.59 |
| L2SP (Weight Regularization) | 90.69 | +0.29 |
| LDIFS (Point-wise Feature Regularization) | 91.66 | +0.86 |
| FDA (Structural Regularization) | 91.86 | +1.39 |
| **Proxy-FDA** | **91.82** | **+1.54** |

### Few-shot Prompt Tuning (CLIP ViT-B/16, 11 Datasets, 16-shot, Table 2)

| Prompt Method | +Proxy-FDA | $\mathcal{A}_{\text{Base}}$ | $\mathcal{A}_{\text{New}}$ | $\Delta_{\text{New}}$↑ | $\mathcal{A}_H$ |
|------------|-----------|------|------|------|------|
| CoOp | ✗ | 82.69 | 63.22 | -10.99 | 71.66 |
| CoOp | ✓ | 83.16 | 73.67 | -0.55 | 78.13 |
| PromptSRC | ✗ | 84.26 | 76.10 | +1.88 | 79.97 |
| PromptSRC | ✓ | 84.47 | 77.45 | +3.23 | 80.81 |

### Key Findings

1. Proxy-FDA achieves forward transfer ($\Delta_{\text{LP}} > 0$) across all 10 fine-tuning tasks, whereas Naive FT and LP-FT yield negative results across the board.
2. Structure-level regularization (Proxy-FDA: +1.54) significantly outperforms point-wise regularization (LDIFS: +0.86).
3. Proxy-FDA improves the $\mathcal{A}_{\text{New}}$ of CoOp by 10.45 percentage points (63.22 $\rightarrow$ 73.67).
4. OTDD exhibits a stronger correlation with forgetting than L2 distance—Proxy-FDA sometimes yields a larger L2 distance but a smaller OTDD, translating to less forgetting.
5. Architecturally robust: Forward transfer is maintained consistently across CLIP, FLAVA, DINOv2, and MAE.
6. Controllable computational overhead: FDA adds +7-9% and Proxy-FDA adds +17-21% to fine-tuning time, with zero overhead during inference.

## Highlights & Insights

- **Qualitative shift from point-level to structure-level regularization**: Upgrading from preserving "feature positions" to preserving "feature relations" brings consistent experimental improvements.
- **Proxies serve as distribution enhancement tools rather than sample substitutes**: Unlike proxies (class prototypes) in metric learning, proxies here are instance-level synthetic features designed to increase data density at neighborhood boundaries.
- **OTDD correlation analysis** provides a solid theoretical foundation for structure-alignment methods.
- **High universality**: Demonstrates efficacy across various settings, including end-to-end, few-shot, continual learning, captioning, and VQA.

## Limitations & Future Work

- The quality of the kNN graph depends on the batch construction strategy and may degrade in extreme class imbalance scenarios.
- Proxies are generated solely from the current batch without leveraging external data or memory banks.
- The neighborhood size $K$ and class number $m$ require hyperparameter tuning for different datasets.
- Verification on ultra-large-scale models (e.g., ViT-L/ViT-G) is currently lacking.

## Related Work & Insights

- LDIFS (Mukhoti et al., 2024): Direct baseline for point-wise feature regularization.
- Relational Knowledge Distillation (Park et al., 2019, RKD): Proxy-FDA can be viewed as relational distillation tailored for robust fine-tuning.
- SigLIP (Zhai et al., 2023): The origin of the FDA loss function, where the Sigmoid formulation accommodates a variable number of positive and negative samples.
- OTDD (Alvarez-Melis & Fusi, 2020): Used as a forgetting diagnostic metric; its strong correlation with forgetting supports the rationality of structural alignment.

## Rating

⭐⭐⭐⭐ — The idea of structure-level feature alignment is novel and effective, with a clear and elegant conceptual upgrade from point-wise constraints to neighborhood structures. The OTDD correlation analysis is insightful, and the proxy generator is lightweight and practical. The experiments are highly convincing, covering end-to-end, few-shot, continual, and multimodal settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](../../NeurIPS2025/self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[ICML 2025\] Test-Time Canonicalization by Foundation Models for Robust Perception](test-time_canonicalization_by_foundation_models_for_robust_perception.md)
- [\[ICML 2025\] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)

</div>

<!-- RELATED:END -->

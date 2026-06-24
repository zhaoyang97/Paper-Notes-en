---
title: >-
  [Paper Note] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning
description: >-
  [ICML 2025][Object Detection][Self-Supervised Learning] This paper proposes the Self-Organizing Prototypes (SOP) strategy, which replaces the single prototype in traditional self-supervised learning (SSL) with multiple semantically similar support embeddings to represent local regions of the feature space. It also introduces a non-parametric masked image modeling (MIM) task, achieving state-of-the-art performance on downstream tasks such as retrieval, detection…
tags:
  - "ICML 2025"
  - "Object Detection"
  - "Self-Supervised Learning"
  - "Non-Parametric Prototypes"
  - "Support Embeddings"
  - "Representation Learning"
  - "Masked Image Modeling"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 6d9c692869bedc6a
---

# Self-Organizing Visual Prototypes for Non-Parametric Representation Learning

**Conference**: ICML 2025  
**arXiv**: [2505.21533](https://arxiv.org/abs/2505.21533)  
**Area**: Object Detection  
**Keywords**: Self-Supervised Learning, Non-Parametric Prototypes, Support Embeddings, Representation Learning, Masked Image Modeling, Vision Transformer

## TL;DR

This paper proposes the Self-Organizing Prototypes (SOP) strategy, which replaces the single prototype in traditional self-supervised learning (SSL) with multiple semantically similar support embeddings to represent local regions of the feature space. It also introduces a non-parametric masked image modeling (MIM) task, achieving state-of-the-art performance on downstream tasks such as retrieval, detection, and segmentation.

## Background & Motivation

Existing prototype-based self-supervised learning methods (such as DINO and iBOT) rely on a large number of learnable prototypes to represent the latent clusters of data. However, two core issues exist:

**Over-clustering issue**: To cover the feature space, $K \gg C$ prototypes are required ($C$ being the ground-truth number of classes), which leads to too few samples per prototype and encourages the learned features to favor simplicity.

**Under-representation issue**: A single prototype struggles to encode all critical features of a cluster region, weakening the interaction between views and prototypes and failing to effectively guide representation learning.

**Dependence on regularization**: Existing methods require techniques like centering, sharpening, and Sinkhorn-Knopp to prevent collapse; removing these regularizations causes the training to fail.

The authors hypothesize that augmenting the feature set of a prototype by introducing multiple semantically similar support embeddings in local regions can better represent the feature space and improve training stability.

## Method

### Overall Architecture

The SOP framework consists of two pre-training tasks:
- **Global-level SOP loss** $\mathcal{L}_{\text{[CLS]}}$: Clustering assignment prediction based on the [CLS] token.
- **Local-level SOP-MIM loss** $\mathcal{L}_{\text{patch}}$: Masked image modeling based on patch tokens.

The final loss is a linear combination of both: $\mathcal{L}_{\text{SOP}} = \lambda_1 \mathcal{L}_{\text{[CLS]}} + \lambda_2 \mathcal{L}_{\text{patch}}$, with a default setting of $\lambda_1 = \lambda_2 = 1$.

### Key Designs: Self-Organizing Prototypes

**Support Embedding Selection Mechanism**:
1. Maintain two memory banks $\mathbf{E}^C \in \mathbb{R}^{N_C \times d}$ ([CLS]-level) and $\mathbf{E}^P \in \mathbb{R}^{N_p \times d}$ (patch-level).
2. Randomly sample $K$ anchor points $\mathbf{A} = \{a_i\}_{i=0}^K$ from the memory banks.
3. For each anchor point, perform a spherical $k$-NN search to find $k$ nearest neighbors as support embeddings (SEs).
4. Each SOP consists of 1 anchor point + $k$ SEs, forming a directed acyclic graph structure.

**Soft Contribution Weights**: The contribution of SEs to the SOP is weighted by their cosine similarity with the anchor point:

$$P^{\text{[CLS]}}(\mathbf{u}) = \sigma(\langle \mathbf{u}, \mathbf{D}^T \rangle) \mathbf{Y}$$

where $\mathbf{Y}$ encodes the soft contribution weights of each SE, and $\mathbf{D}$ contains the embeddings of all SOPs.

### SOP-MIM Task

Reconstruct masked patch tokens, but replace the learnable discrete tokenizer with non-parametric local SOPs:

$$\mathcal{L}_{\text{patch}} = -\sum_{l=1}^{L} m_l P^{\text{patch}}(\mathbf{z}_l^1)^T \log(P^{\text{patch}}(\hat{\mathbf{z}}_l^1))$$

### Loss & Training

Global-level loss (non-parametric version):

$$\mathcal{L}_{\text{[CLS]}} = -\sum_{\mathbf{x} \sim \mathbf{X}} P^{\text{[CLS]}}(\mathbf{z}_0^1)^T \log(P^{\text{[CLS]}}(\mathbf{z}_0^2))$$

Default configuration: $K=4096$ anchor points, $k=8$ SEs, memory bank sizes $N_C=65536$ and $N_p=8192$, and feature dimension $d=256$.

## Key Experimental Results

### Main Results: ImageNet Linear Evaluation

| Method | Architecture | k-NN | Linear Probe | 1% Fine-tuning | 10% Fine-tuning | 100% Fine-tuning |
|------|------|------|---------|---------|---------|-----------|
| DINO | ViT-S/16 | - | - | - | - | - |
| iBOT | ViT-S/16 | - | - | - | - | - |
| **SOP** | **ViT-S/16** | - | - | - | - | - |
| iBOT | ViT-B/16 | - | - | - | - | - |
| **SOP** | **ViT-B/16** | - | - | - | - | - |
| iBOT | ViT-L/16 | - | - | - | - | - |
| **SOP** | **ViT-L/16** | **79.2** | - | - | - | - |

**Key Findings**: The SOP $k$-NN accuracy on ViT-L reaches 79.2%, which is +1.2% higher than iBOT, comparable to I-JEPA ViT-H (79.3%) but with only half the parameter count.

### Object Detection & Segmentation

| Method | COCO AP^b | COCO AP^m | ADE20K Lin. mIoU | ADE20K UPerNet mIoU |
|------|-----------|-----------|-------------------|---------------------|
| Supervised | 49.8 | 43.2 | 35.4 | 46.6 |
| DINO | 50.1 | 43.4 | 34.5 | 46.8 |
| iBOT | 51.2 | 44.2 | 38.3 | 50.0 |
| **SOP** | **51.4** | **44.3** | **38.7** | **50.6** |

**Key Findings**: SOP improves COCO detection by AP^b +0.2 and ADE20K semantic segmentation by mIoU +0.6, significantly exceeding the supervised baseline by up to +5.0 mIoU.

### Ablation Study: Number of Support Embeddings

| Number of SEs (CLS) | 1 | 2 | 4 | **8** | 16 |
|----------------|---|---|---|-------|-----|
| k-NN | - | - | - | **70.0** | - |

**Key Findings**: The global task performs best with 8 SEs, while the local MIM task runs effectively with a single SE.

### Image Retrieval (Oxford & Paris)

On the 'Hard' protocol, SOP achieves an mAP improvement of up to **+3.2** compared to iBOT. In robustness evaluations, SOP achieves the best performance in 6 out of 7 background variations on ImageNet-9.

## Highlights & Insights

1. **Feasibility of Non-Parametric SSL**: Demonstrates that high-quality SSL pre-training does not require learning prototype parameters and can be achieved solely using data features from a memory bank.
2. **Dynamic Region Coverage**: SOPs utilize random sampling to avoid training collapse caused by fixed anchor points (Table C.7 confirms that fixed anchor points lead to immediate collapse).
3. **Alleviating Over-Clustering**: With only 1024 SOPs, the method achieves performance comparable to iBOT using 8192 prototypes (Table 10), with a gap of only 0.5%.
4. **Amplified Gain with Model Scaling**: The performance gap between SOP and competing methods scales up continuously from ViT-S to ViT-L.

## Limitations & Future Work

1. Specific numerical values are missing in some tables (the authors used conditional formatting in LaTeX, so actual values must be found in the PDF).
2. Computational overhead is similar to iBOT (193.5h vs 193.4h), failing to achieve efficiency improvements.
3. Only pre-trained on ImageNet-1M, lacking verification on larger-scale datasets.
4. Combination with methods like DINOv2, which use more data and regularization tricks, has not been explored.

## Related Work & Insights

- **Relationship to iBOT**: SOP can be viewed as a non-parametric alternative to iBOT, eliminating the need for learnable prototypes and Sinkhorn-Knopp regularization.
- **Difference from NNCLR**: NNCLR uses the nearest neighbor as a positive sample to optimize InfoNCE, whereas SOP uses multiple nearest neighbors as voters to optimize region-level matching.
- **Insights**: The concept of support embeddings can be generalized to other tasks that require the local structure of the representation space (e.g., few-shot learning, open-vocabulary detection).

## Rating

- **Novelty**: ★★★★☆ — Non-parametric SOP + support embedding voting mechanism is an interesting paradigm shift in self-supervised learning.
- **Value**: ★★★★☆ — High practical value as it requires no extra regularization and does not rely on over-clustering.
- **Experimental Thoroughness**: ★★★★★ — Extremely comprehensive, covering retrieval, detection, segmentation, transfer, robustness, and ablation studies.
- **Writing Quality**: ★★★★☆ — Clearly structured and mathematically rigorous.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Understanding the Emergence of Multimodal Representation Alignment](understanding_the_emergence_of_multimodal_representation_alignment.md)
- [\[NeurIPS 2025\] Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](../../NeurIPS2025/object_detection/automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)
- [\[AAAI 2026\] CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection](../../AAAI2026/object_detection/casl_curvature-augmented_self-supervised_learning_for_3d_anomaly_detection.md)
- [\[NeurIPS 2025\] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](../../NeurIPS2025/object_detection/detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](../../ICLR2026/object_detection/paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)

</div>

<!-- RELATED:END -->

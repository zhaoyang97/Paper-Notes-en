---
title: >-
  [Paper Note] Graph Your Own Prompt
description: >-
  [NeurIPS 2025][Model Compression][Graph regularization] This paper proposes a Graph Consistency Regularization (GCR) framework that inserts parameter-free Graph Consistency Layers (GCL) at arbitrary network depths. GCL aligns the relational graph of intermediate features with a class-aware semantic graph derived from model predictions, promoting semantically consistent feature learning in a self-prompting manner—improving classification generalization without modifying the ar…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Graph regularization"
  - "feature alignment"
  - "semantic consistency"
  - "classification"
  - "parameter-free module"
date: 2026-05-08
content_hash: e7f9c5d5ac3b3530
---

# Graph Your Own Prompt

**Conference**: NeurIPS 2025
**arXiv**: [2509.23373](https://arxiv.org/abs/2509.23373)  
**Code**: To be confirmed (the paper mentions a Project website and Code link)  
**Area**: Model Compression
**Keywords**: Graph regularization, feature alignment, semantic consistency, classification, parameter-free module

## TL;DR

This paper proposes a Graph Consistency Regularization (GCR) framework that inserts parameter-free Graph Consistency Layers (GCL) at arbitrary network depths. GCL aligns the relational graph of intermediate features with a class-aware semantic graph derived from model predictions, promoting semantically consistent feature learning in a self-prompting manner—improving classification generalization without modifying the architecture or introducing additional parameters.

## Background & Motivation

1. **Lack of semantic alignment in deep features**: Although deep networks learn rich representations, intermediate features often capture noisy inter-class similarities that contradict the model's own predictions, causing samples from different classes to cluster closely in feature space.
2. **Explicit sampling requirements in contrastive learning**: Existing contrastive learning and graph regularization methods require positive/negative sampling strategies or carefully designed data augmentation, and typically operate at a single network layer.
3. **Underutilization of prediction information**: The network's own softmax predictions encode rich semantic relational structure (predictions of same-class samples should be similar), yet existing methods rarely feed this structure back into feature learning.
4. **Lack of multi-layer structural supervision**: Existing auxiliary losses for intermediate layers typically supervise each layer independently, rarely exploiting structured model outputs to regularize feature learning.
5. **Lightweight deployment requirements**: In practice, regularization methods are expected to introduce no extra parameters, require no architectural modifications, and leave the training pipeline unchanged.
6. **Naturalness of graph structure**: Pairwise sample relationships are naturally modeled as graphs; aligning a feature similarity graph with a prediction similarity graph provides an elegant cross-space structural constraint.

## Method

### Graph Consistency Layer (GCL)

**Feature relational graph construction**: Given the feature matrix $X^{(l)} \in \mathbb{R}^{n \times d}$ at layer $l$ (where $n$ is the batch size), a pairwise relational graph is constructed using ReLU-gated cosine similarity:

$$F_{ij}^{(l)} = \text{ReLU}(\cos(x_i^{(l)}, x_j^{(l)}))$$

**Masked prediction relational graph construction**: A prediction similarity matrix $S$ is computed from the softmax of the final-layer logits $Z \in \mathbb{R}^{n \times C}$, then filtered by a class-aware binary mask $M$ (1 for same-class pairs, 0 otherwise):

$$P_{ij} = M_{ij} \odot S_{ij}$$

The mask serves two purposes: (1) filtering unreliable cross-class prediction similarities in early training; (2) focusing on intra-class semantic relationships to avoid interference from visually similar but semantically distinct classes.

### Graph Consistency Regularization (GCR)

**Per-layer alignment loss**: The strict upper-triangular portions of the feature graph and prediction graph (eliminating self-connections and double-counting) are extracted, and the Frobenius norm is computed:

$$\mathcal{L}_{\text{GCR}}^{(l)} = \|\text{triu}(F^{(l)}) - \text{triu}(P)\|_F^2$$

**Multi-layer aggregation**: Alignment losses are collected at $K$ insertion points and aggregated via a weighted sum:

$$\mathcal{L}_{\text{GCR}} = \sum_{l=1}^{K} w_l \cdot \|\text{triu}(F^{(l)}) - \text{triu}(P)\|_F^2$$

**Adaptive weighting**: Softmax weights based on the degree of graph inconsistency assign higher weights to layers with greater misalignment:

$$w_l = \frac{\exp(-\|\text{triu}(F^{(l)}) - \text{triu}(P)\|_F^2)}{\sum_{j=1}^{K} \exp(-\|\text{triu}(F^{(j)}) - \text{triu}(P)\|_F^2)}$$

**Overall training objective**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{GCR}}$, with $\lambda = 1$ in all experiments.

### GCL Insertion Strategy

The network is divided into three stages—Early (E), Mid (M), and Late (L)—and seven configurations are evaluated: E, M, L, E+M, M+L, E+L, and Full. Experiments show that Late GCL generally achieves the best results, as later-layer features are semantically richer and closer to the decision boundary.

### Theoretical Analysis

1. **Generalization bound**: Using covering numbers and Dudley's entropy integral, GCR constraints are shown to reduce the effective hypothesis class complexity, thereby lowering generalization error.
2. **Spectral alignment**: It is proven that as the Frobenius distance between the feature graph and prediction graph decreases, the spectra of their normalized Laplacians also converge, guaranteeing consistent clustering structure.
3. **PAC-Bayes perspective**: The GCR constraint is equivalent to imposing a structural prior over the function space, with the KL divergence upper bound proportional to $\sum_l \|F^{(l)} - P\|_F^2$.

## Key Experimental Results

### CIFAR-10 Classification Accuracy (%, averaged over 11 architectures)

| Configuration | MAE | MobileNet | ShuffleNet | GoogLeNet | ResNet-50 | DenseNet-121 | Mean |
|---|---|---|---|---|---|---|---|
| Baseline | 88.95 | 90.23 | 91.21 | 94.10 | 95.03 | 95.01 | 93.32 |
| Late GCL | 89.70 | 91.40 | 92.36 | 94.88 | 95.66 | 95.72 | **94.07** |
| Gain | +0.75 | +1.17 | +1.15 | +0.78 | +0.63 | +0.71 | **+0.75** |

### CIFAR-100 Classification Accuracy (%, averaged over 9 architectures)

| Configuration | MAE | MobileNet | ResNeXt-50 | ResNet-50 | DenseNet-121 | Mean |
|---|---|---|---|---|---|---|
| Baseline | 64.29 | 65.95 | 77.75 | 77.31 | 77.09 | 72.95 |
| Late GCL | **65.54** | **68.32** | **79.54** | **79.42** | **79.69** | **74.74** |
| Gain | +1.25 | +2.37 | +1.79 | +2.11 | +2.60 | **+1.79** |

### ImageNet-1K Classification Accuracy (%, Transformer architectures)

| Method | iFormer-S | iFormer-B | ViT-B/16 | ViG-B |
|---|---|---|---|---|
| Baseline | 83.4 | 84.6 | 74.3 | 82.3 |
| Late GCL | **84.5** | **86.1** | **75.8** | **84.0** |
| Gain | +1.1 | **+1.5** | **+1.5** | +1.7 |

## Highlights & Insights

1. **Zero parameter overhead**: GCL is entirely parameter-free, requires no architectural modification, and leaves the training pipeline unchanged, adding only lightweight matrix operations.
2. **Strong generalizability**: Effective across lightweight CNNs (MobileNet/ShuffleNet), deep CNNs (ResNet/DenseNet), and Transformers (ViT/Swin/iFormer).
3. **Elegant self-prompting design**: The model's own prediction structure serves as a reference signal for feature learning, forming a self-prompting mechanism that requires no external supervision.
4. **Improved interpretability**: GCL-enhanced feature maps focus more on class-discriminative regions (e.g., eyes and ears for cats, mouths and noses for dogs), with accuracy improving from 98.1% to 99.8%.
5. **Comprehensive theoretical grounding**: Generalization guarantees are established from three perspectives: covering numbers, spectral graph theory, and PAC-Bayes analysis.

## Limitations & Future Work

1. **Validated on classification only**: The method is evaluated solely on image classification and has not been extended to tasks such as segmentation, retrieval, or object detection.
2. **Mask construction requires labels**: The mask $M$ relies on ground-truth labels, making it inapplicable to self-supervised or label-free settings.
3. **Intra-batch relational scope**: Graph construction is limited to samples within the current batch, which may reduce effectiveness under class imbalance or insufficient class coverage per batch.
4. **Fixed $\lambda = 1$**: While this simplifies hyperparameter tuning, different tasks and architectures may require different regularization strengths.
5. **Late GCL is generally optimal but not universal**: The best insertion position varies by architecture, and no automatic selection mechanism is provided.
6. **Relationship to knowledge distillation remains unclear**: Using the prediction graph to guide feature learning resembles self-distillation, but no comparison with such methods is conducted.

## Related Work & Insights

- **vs. Contrastive learning (SimCLR/SupCon)**: Contrastive learning requires positive/negative sampling and data augmentation and typically operates at a single layer; GCR requires no sampling and supports multi-layer alignment.
- **vs. Graph neural networks**: GNNs require maintaining global graphs or additional message-passing modules; GCR constructs graphs dynamically within each batch with no architectural overhead.
- **vs. Center Loss/Triplet Loss**: These methods enforce inter-class distances or intra-class compactness but operate in a single representation space; GCR performs structural alignment across both feature and prediction spaces.
- **vs. Knowledge distillation**: Distillation requires a teacher model; GCR uses the model's own predictions, constituting a form of self-distillation that is considerably more lightweight.
- **vs. Attention mechanisms**: The feature focusing effect of GCL resembles attention, but is entirely parameter-free and achieved through graph alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ The self-prompting idea of aligning feature graphs via prediction graphs is novel; the multi-layer adaptive weighting design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 datasets and 16+ architectures, with visualizations, ablations, weighting scheme comparisons, and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete theoretical exposition, and rich intuitive visualizations.
- Value: ⭐⭐⭐⭐ A zero-parameter plug-and-play regularization method with strong practical utility; broader task coverage is needed to fully establish generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning](graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)
- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[NeurIPS 2025\] Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization](find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)
- [\[NeurIPS 2025\] All You Need is One: Capsule Prompt Tuning with a Single Vector](all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [\[ICML 2026\] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](../../ICML2026/model_compression/mind_your_margin_and_boundary_are_your_distilled_datasets_truly_robust.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Training the Untrainable: Introducing Inductive Bias via Representational Alignment
description: >-
  [NeurIPS 2025][Inductive Bias] This paper proposes Guidance, a method that transfers the architectural inductive bias of one network (the guide) to another otherwise "untrainable" network (the target) via layer-wise representational alignment (CKA), enabling FCNs to perform image classification and RNNs to approach Transformer-level language modeling performance.
tags:
  - NeurIPS 2025
  - Inductive Bias
  - Representational Alignment
  - CKA
  - Architectural Prior
  - Knowledge Distillation
date: 2026-05-08
content_hash: ea15fa3425dd63b7
---

# Training the Untrainable: Introducing Inductive Bias via Representational Alignment

**Conference**: NeurIPS 2025
**arXiv**: [2410.20035](https://arxiv.org/abs/2410.20035)
**Code**: [GitHub](https://untrainable-networks.github.io)
**Area**: Deep Learning / Architectural Inductive Bias
**Keywords**: Inductive Bias, Representational Alignment, CKA, Architectural Prior, Knowledge Distillation

## TL;DR

This paper proposes Guidance, a method that transfers the architectural inductive bias of one network (the guide) to another otherwise "untrainable" network (the target) via layer-wise representational alignment (CKA), enabling FCNs to perform image classification and RNNs to approach Transformer-level language modeling performance.

## Background & Motivation

Architectural choices are critical in neural networks — CNNs revolutionized vision, and Transformers reshaped NLP. Yet **architecture design remains largely a "dark art"**: the inductive biases encoded by different architectures are rarely well understood. The precise role of residual connections, for instance, remains contested.

This leads to several practical challenges:
- **FCNs immediately overfit on image classification**: lacking spatial priors such as local receptive fields
- **Deep CNNs without residual connections suffer from vanishing gradients**: making effective training infeasible
- **RNNs saturate on long-sequence tasks**: constrained by vanishing gradients and limited context integration
- **Transformers fail on formal language tasks requiring full-sequence reasoning**: e.g., parity judgment

The conventional answer is to "switch to a better architecture," but this requires a deep understanding of inductive biases. The authors pose a more fundamental question: **can the inductive bias of one architecture be "injected" into another without changing the target architecture?**

Recent work has shown that networks of different architectures exhibit surprisingly similar internal representations (Han et al.), suggesting that bias transfer at the representational level may be feasible.

## Method

### Overall Architecture

Guidance augments the target network's original loss with a layer-wise representational alignment term, encouraging the intermediate activations of the target to match those of a fixed guide network. The guide may be pretrained (transferring both architectural priors and learned knowledge) or randomly initialized (transferring architectural priors only).

The total loss is:

$$\mathcal{L}(\theta^T) = \mathcal{L}_T(\theta^T) + \sum_{i \in I} \bar{\mathcal{M}}(\mathbf{A}_{i^T}^T(\theta^T), \mathbf{A}_{i^G}^G(\theta^G))$$

where $\bar{\mathcal{M}}$ is a representational dissimilarity measure (the complement of CKA), $I$ denotes the guide–target layer correspondence, and $\theta^G$ is kept frozen throughout training.

### Key Designs

1. **CKA as the representational distance function**: Linear Centered Kernel Alignment (CKA) is adopted, as it operates on second-order statistics (pairwise sample distance matrices) and captures architecture-specific structural information. For example, the local receptive fields of CNNs induce correlations among units corresponding to neighboring pixels; these correlations are reflected in the distance matrix and can be transferred via CKA to FCN layers that lack such local structure. Weight sharing (i.e., applying the same convolutional kernel across all spatial positions) is similarly encoded and transferred.

2. **Layer mapping strategy**: Guide layers are distributed uniformly across target layers. For example, ResNet-18 has 18 convolutional layers; a 50-layer FCN assigns one ResNet layer to every 2–3 linear layers. For stacked RNNs and Transformers, features are extracted from intermediate layers of the stack. Experiments show that using more layer correspondences yields stronger alignment signals.

3. **The guide need not be trained**: Perhaps the most striking finding is that a randomly initialized guide — one that cannot perform the task at all — still yields significant performance improvements, demonstrating that **the architecture itself encodes useful inductive biases independent of learned parameters**.

### Loss & Training

- All networks use cross-entropy loss with Adam/AdamW optimizers
- Fixed batch size of 256 (CKA computation is sensitive to sample count)
- Learning rates tuned via a sweep over 5 values, selecting the lowest validation loss
- Each configuration trained for 100 epochs; error bars computed over 5 random seeds

## Key Experimental Results

### Main Results: Sequence Modeling

| Experiment | Copy-Paste Acc.↑ | Parity Acc.↑ | LM (Small) PPL↓ | LM (Large) PPL↓ |
|---|---|---|---|---|
| RNN (baseline) | 14.35±0.01 | 100 | 69.19±1.89 | 89.13±2.00 |
| Transformer (baseline) | 96.98 | 71.98±3.16 | 34.15 | 33.10 |
| Transformer→RNN | **23.27±1.02** | — | **40.01±1.54** | **36.91±1.04** |
| Untrained Transformer→RNN | **42.56±1.51** | — | 59.61±2.33 | 47.17±2.50 |
| RNN→Transformer | — | **78.49±2.16** | — | — |

### Main Results: Image Classification (ImageNet Top-5)

| Experiment | Validation Acc.↑ |
|---|---|
| Deep FCN (baseline) | 1.65±1.21 |
| ResNet-18→Deep FCN | 7.50±1.51 |
| Untrained ResNet-18→Deep FCN | **13.10±0.72** |
| Wide FCN (baseline) | 34.09±0.91 |
| ResNet-18→Wide FCN | **43.01±0.92** |
| Deep ConvNet (baseline) | 70.02±1.52 |
| ResNet-50→Deep ConvNet | **78.91±2.16** |

### Ablation Study

| Ablation Configuration | Key Metric | Notes |
|---|---|---|
| Guidance used only for initialization (300 steps) | Performance comparable to continuous guidance | Suggests better initialization schemes for FCNs may exist |
| Guidance vs. distillation | Guidance consistently outperforms distillation | Intermediate-layer alignment is far more effective than output-level alignment |
| Trained guide vs. random guide | Random guide superior in most settings | Architectural priors matter more than learned knowledge |
| Error consistency analysis | Guided FCN inherits guide's decision patterns | The error consistency relationship between ResNet-guided and ViT-guided FCNs mirrors that between ResNet and ViT themselves |

### Key Findings

- **Untrained guides frequently outperform trained guides**: On Copy-Paste and Deep FCN tasks, randomly initialized guides perform better, confirming that the architecture itself — not the learned weights — is the primary source of improvement
- **RNN + Guidance substantially narrows the gap with Transformers on language modeling**: perplexity drops from ~70 to 35–40, approaching the Transformer's 34
- **Guidance-only initialization suffices to prevent overfitting**: aligning to a random ResNet for only 300 steps, followed by standard training, completely eliminates overfitting in the FCN
- **Deep ConvNets (without residual connections) only benefit from a trained ResNet guide**: suggesting that residual connections must first be trained before they influence the representational space

## Highlights & Insights

- **Architectural priors can be decoupled from training priors**: comparing trained vs. untrained guides provides an empirical tool for studying what biases different architectures encode
- **Guidance is fundamentally an aid to credit assignment**: layer-wise alignment helps gradient descent better adjust early-layer weights in deep networks
- **The error consistency experiments are particularly compelling**: guided FCNs do not merely improve generically — they inherit the specific decision-making style of the guide
- This work challenges the conventional wisdom that certain architectures are inherently ill-suited for certain tasks

## Limitations & Future Work

- **Breadth is prioritized over peak performance on any single task**: no state-of-the-art results are achieved; extended hyperparameter tuning and longer training may yield further gains
- **Only CKA is evaluated as the distance function**: alternative representational distances (e.g., CCA, RSA) may yield different outcomes
- **The layer correspondence strategy is simple**: uniform distribution mapping may be suboptimal; more complex network pairs may require finer-grained mappings
- **Computational overhead**: each minibatch requires a forward pass through both the guide and target, plus CKA computation, increasing training cost
- **FCNs, while no longer overfitting, remain weak in absolute terms**: substantial additional work is needed before FCNs become practically viable for image classification

## Related Work & Insights

- **Distinction from knowledge distillation**: distillation matches output logits, whereas Guidance aligns intermediate representations; distillation requires a trained teacher, while Guidance can use a random guide
- **Complementary to NAS/architecture search**: Guidance offers a "soft" mechanism for injecting architectural biases without hard-coding them
- **Potential implications for understanding large models**: including architectural choices in inference-time scaling

## Rating

- Novelty: ⭐⭐⭐⭐ Using representational alignment with random networks to transfer architectural priors is a novel and counterintuitive finding
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers image classification and sequence modeling across 4 tasks and multiple architectural combinations, though no single task is explored in depth
- Writing Quality: ⭐⭐⭐⭐ Experimental design is systematic, but the breadth of coverage limits the depth of analysis for individual experiments
- Value: ⭐⭐⭐⭐ Provides a powerful empirical tool for studying architectural inductive biases, though the practical utility of guided FCNs remains to be validated

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Graph Alignment via Birkhoff Relaxation](graph_alignment_via_birkhoff_relaxation.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[NeurIPS 2025\] DPA: A One-Stop Metric to Measure Bias Amplification in Classification Datasets](dpa_a_one-stop_metric_to_measure_bias_amplification_in_classification_datasets.md)
- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)
- [\[NeurIPS 2025\] FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)

<!-- RELATED:END -->

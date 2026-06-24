---
title: >-
  [Paper Note] WAVE: Weight Templates for Adaptive Initialization of Variable-sized Models
description: >-
  [CVPR 2025][Model Compression][Weight Templates] WAVE is proposed to reformulate the initialization of variable-sized models as a multi-task learning problem. By utilizing shared size-agnostic weight templates and lightweight size-specific weight scalers (via Kronecker products), it achieves efficient initialization. It requires only 3.3% of pre-trained parameters to outperform models trained for 150 epochs within just 10 epochs.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Weight Templates"
  - "Model Initialization"
  - "Learngene"
  - "Kronecker Product"
  - "Variable-sized Models"
date: 2026-05-08
content_hash: 606f1508264816c4
---

# WAVE: Weight Templates for Adaptive Initialization of Variable-sized Models

**Conference**: CVPR 2025  
**arXiv**: [2406.17503](https://arxiv.org/abs/2406.17503)  
**Code**: [GitHub](https://github.com/fu-feng/WAVE)  
**Area**: Model Compression / Model Initialization  
**Keywords**: Weight Templates, Model Initialization, Learngene, Kronecker Product, Variable-sized Models

## TL;DR

WAVE is proposed to reformulate the initialization of variable-sized models as a multi-task learning problem. By utilizing shared size-agnostic weight templates and lightweight size-specific weight scalers (via Kronecker products), it achieves efficient initialization. It requires only 3.3% of pre-trained parameters to outperform models trained for 150 epochs within just 10 epochs.

## Background & Motivation

As the parameters of neural networks scale exponentially, training from scratch is becoming prohibitively expensive. Although fine-tuning pre-trained models has become the mainstream paradigm, practical deployment often requires models of **different sizes** due to constraints in memory, computation, and latency. Existing pre-trained models typically offer only a few fixed sizes (e.g., 12-layer ViT-B). When target models mismatch the pre-trained dimensions, retraining from scratch is often unavoidable.

Existing methods (such as Weight Selection and LiGO) initialize models of various sizes by selecting or transforming pre-trained weight matrices. However, they frequently break the structured knowledge within the original model or introduce excessive random parameters, leading to limited performance.

The core insight of WAVE is to analogize **variable-sized model initialization** to **multi-task learning**: the initialization of each specific size represents an individual "task," which requires a shared, task-agnostic backbone (weight templates) paired with a small amount of task-specific adaptation parameters (weight scalers). Under the Learngene framework, weight templates encapsulate size-agnostic pre-trained knowledge. By combining them with size-specific scalers via Kronecker products, models of arbitrary depth and width can be initialized flexibly.

## Method

### Overall Architecture

WAVE operates in two stages: (1) **Knowledge Consolidation**—distilling knowledge from pre-trained models into structured weight templates (a one-time process taking 150 epochs on ImageNet); (2) **Model Initialization**—freezing the weight templates and training only the lightweight weight scalers to determine how the templates are combined to fit target model sizes (requiring only a small amount of data).

### Key Designs

1. **Weight Templates + Kronecker Reconstruction**:
    - Function: Represents the weight matrices as a weighted combination of shared templates.
    - Mechanism: For a weight matrix $W_\star^{(l)} \in \mathbb{R}^{m_1 \times m_2}$ ($\star \in \{qkv, o, in, out\}$) of the $l$-th layer in ViT, it is reconstructed using $N_\star$ shared weight templates $T_\star^{(t)} \in \mathbb{R}^{w_1 \times w_2}$ and corresponding scalers $S_\star^{(l,t)} \in \mathbb{R}^{\frac{m_1}{w_1} \times \frac{m_2}{w_2}}$ via Kronecker product: $W_\star^{(l)} = \sum_{t=1}^{N_\star} T_\star^{(t)} \otimes S_\star^{(l,t)}$. The scalers contain only a few thousand parameters, making them extremely lightweight.
    - Design Motivation: Prior evidence suggests significant structural relations among different layers in pre-trained ViTs (e.g., Mimetic Init, TLEG). Weight templates can capture this shared structure.

2. **Integrating Pre-trained Knowledge via Distillation**:
    - Function: Injects knowledge from pre-trained models into the weight templates.
    - Mechanism: Construct an auxiliary model $f_{aux}$, whose parameters $\theta_{aux} = \mathcal{T} \otimes \mathcal{S}_{aux}$ are fully generated via the Kronecker product of weight templates and auxiliary scalers. Training is conducted using KL divergence distillation + cross-entropy loss $\mathcal{L} = \text{KL}(z_{pre} \| z_{aux}) + \text{CE}(z_{aux}, y)$, propagating gradients back to both templates and scalers. The auxiliary model serves as a medium for knowledge transfer and a bottleneck to filter unstructured knowledge.
    - Design Motivation: Weight templates cannot learn directly from data on their own and require an auxiliary model as a bridge. The Kronecker product constraint ensures that the templates maintain high structural integrity.

3. **Lightweight Scaler Adaptation**:
    - Function: Employs minimal parameters to achieve initialization for arbitrary model sizes.
    - Mechanism: Freeze the weight templates $\mathcal{T}$, construct corresponding scalers $\mathcal{S}_{tar}$ based on target model dimensions, and train the scalers on a small dataset to learn the combination rules for the templates. The dimensions of the scalers are determined by the target weight matrix and template size: $S_{\star,tar}^{(l,t)} \in \mathbb{R}^{\frac{m_1}{w_1} \times \frac{m_2}{w_2}}$. When the target model's depth or width changes, only the scale/layers of the scalers change, while the templates remain shared.
    - Design Motivation: Similar to visual prompt tuning / LoRA—employing shared large representations (templates) + minimal trainable parameters (scalers) for efficient adaptation.

### Loss & Training

- **Knowledge Consolidation Stage**: KL distillation + CE classification loss, trained for 150 epochs on ImageNet-1K.
- **Initialization Stage**: Trains only the scalers (templates are frozen) using a small fraction of data.
- **Teacher Model**: Pre-trained ViT-B or other models acting as the ancestry model.
- **Auxiliary Model**: Parameters are structurally constructed via Kronecker product of templates and scalers.

## Key Experimental Results

### Main Results

Top-1 accuracy on ImageNet-1K for models of different sizes initialized and trained for 10 epochs:

| Method | Parameter transfer ratio | 6-layer ViT | 8-layer ViT | 12-layer ViT | Cross-width |
|------|-----------|--------|--------|---------|-------|
| Train from scratch (150 epochs) | — | X% | Y% | Z% | — |
| Weight Selection | 100% | — | — | — | Small $\to$ Large only |
| LiGO | 100% | — | — | — | Small $\to$ Large only |
| TLEG | ~50% | — | — | — | Depth only |
| **WAVE (10 epochs)** | **3.3%** | **Outperforms** | **Outperforms** | **Outperforms** | **Supports Depth + Width** |

Computational savings: WAVE reduces computation by $15n\times$ for $n$ models of different sizes.

### Ablation Study

| Configuration | Description |
|------|------|
| Number of Templates | More templates increase capacity, but yield diminishing returns. |
| No Distillation (Random Templates) | Significant performance drop, proving the necessity of knowledge consolidation. |
| Fixed Scaler (No Training) | Poor initialization quality, demonstrating the importance of scalers learning combination rules. |
| KL-Only Loss | Slightly worse than KL+CE. |
| Cross-Dataset Transfer | Template knowledge is task-agnostic, transferring well to multiple downstream datasets. |

### Key Findings

- By transferring only 3.3% of the pre-trained model parameters, WAVE initialization combined with 10 epochs of training outperforms training from scratch for 150 epochs.
- WAVE is the first Learngene method that simultaneously supports changes in both **depth and width**.
- Visualizations of the weight templates reveal highly structured knowledge patterns that align with patterns observed in pre-trained models.
- Weight templates possess task-agnostic properties, enabling direct transfer to multiple downstream datasets like CIFAR-100 and Flowers.

## Highlights & Insights

- **Creative application of the multi-task learning perspective**: Analogizing "initializing models of varying sizes" to "adapting to different tasks" naturally leads to a shared backbone (templates) + light adapter (scalers) architecture. The paradigm is simple and elegant.
- **Elegant Kronecker product template combination mechanism**: The Kronecker product naturally scales local structures of templates up to any target size. It can be mathematically shown that previous layer-wise Learngene approaches (e.g., Heur-LG, Auto-LG, TLEG) are special/restricted cases of WAVE.

## Limitations & Future Work

- The one-time construction of weight templates still requires 150 epochs of training, which can be computationally intensive for extremely large pre-trained models.
- Currently validated only on ViT architectures; suitability for other architectures like CNNs or MLP-Mixers remains unexplored.
- The structural constraints of Kronecker product might limit representation capability for completely irregular weight patterns.
- Scaler initialization strategies affect final performance; simple heuristics are currently used.

## Related Work & Insights

- **vs Weight Selection**: Extracts weights from large models to initialize smaller ones. While it transfers parameters directly, it might disrupt original structures. Meanwhile, WAVE retains structure via mathematical templates, using only 3.3% transfer parameter size.
- **vs LiGO**: Linearly scales smaller models into larger counterparts, leading to unidirectional adaptation (small $\to$ large). WAVE supports bi-directional adaptation.
- **vs TLEG**: TLEG represents each layer as a linear combination of two base layers, which is a special case of WAVE. WAVE supports much more flexible multi-template combination and width adaptation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Multi-task perspective + Kronecker product template combination, unifying multiple Learngene methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple sizes $\times$ multiple datasets $\times$ multiple baselines, establishing the first comprehensive Learngene benchmark.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Highly practical with 3.3% parameter transfer securing a $15n\times$ reduction in computational cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics](../../ICML2025/model_compression/beyond_zero_initialization_investigating_the_impact_of_non-zero_initialization_o.md)
- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](../../ICML2025/model_compression/random_initialization_of_gated_sparse_adapters.md)
- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](sampling_innovation-based_adaptive_compressive_sensing.md)
- [\[ICCV 2025\] Scheduling Weight Transitions for Quantization-Aware Training](../../ICCV2025/model_compression/scheduling_weight_transitions_for_quantization-aware_training.md)

</div>

<!-- RELATED:END -->

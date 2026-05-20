---
title: >-
  [Paper Note] Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models
description: >-
  [NeurIPS 2025][Interpretability][foundation model] This paper proposes ComBo, a lightweight probing-based adapter that compresses multi-layer activations from multiple frozen foundation models via affine projection…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "foundation model"
  - "probing"
  - "multi-backbone"
  - "feature combination"
  - "model selection"
date: 2026-05-08
content_hash: ac93c9195acaa44c
---

# Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2512.01405](https://arxiv.org/abs/2512.01405)  
**Code**: Available (bramtoula.github.io/combo)  
**Area**: Interpretability
**Keywords**: foundation model, probing, multi-backbone, feature combination, model selection

## TL;DR

This paper proposes ComBo, a lightweight probing-based adapter that compresses multi-layer activations from multiple frozen foundation models via affine projection, then fuses them with a small transformer—without backpropagation through any backbone. ComBo efficiently integrates complementary representations across models, surpassing prior probing methods and matching distillation-based methods on VTAB-1k.

## Background & Motivation

Different foundation models (CLIP, DINOv2, MAE, SAM, etc.) learn representations with distinct strengths and weaknesses due to differences in training objectives and data. The optimal model—and even the optimal layer—may vary across downstream tasks (intermediate layers sometimes outperform the final layer). Prior approaches suffer from three limitations:

**PEFT methods** (e.g., LoRA, Adapter) target a single model and require backpropagation through the backbone, making multi-model combination computationally prohibitive.

**Distillation methods** (e.g., RADIO) require distilling multiple foundation models into a student model, which is costly and prone to "mode switching" issues.

**Existing probing methods** (Head2Toe, SMP) require dataset-specific hyperparameter tuning, discard spatial information via average pooling of feature maps, and scale poorly.

**Core Problem**: How can complementary representations from multiple foundation models be efficiently combined without backpropagating through any large model?

## Method

### Overall Architecture

ComBo operates in three stages:

1. **Feature Map Extraction**: Feature maps $\mathbf{F}_{k,l} \in \mathbb{R}^{T_k \times D_k}$ are extracted from each layer of $K$ frozen models, token counts are unified to $T$ via bilinear interpolation, and mean-standard-deviation normalization is applied.
2. **Layer Embedding Compression**: At each spatial position $i$, features from all layers of all models are concatenated into $\mathbf{S}_i \in \mathbb{R}^D$ (where $D = \sum_k \sum_l D_k$), and a shared affine projection $\Lambda = \{\mathbf{W}, \mathbf{b}\}$ compresses this to $D' \ll D$.
3. **Transformer Processing**: The compressed tokens, together with a learnable cls token, are passed through a 6-layer lightweight transformer (128-dim, 2 heads, 1.7M parameters); the cls output is fed into a linear classification head.

### Key Designs

**Spatial Information Preservation**: Unlike Head2Toe/SMP, which apply average pooling over feature maps, ComBo compresses along the layer dimension independently at each token position, fully preserving spatial layout. This is critical for Structured tasks (e.g., object counting, distance estimation).

**Layer Selection via Affine Projection**: The projection matrix $\mathbf{W}$ performs feature selection along the layer dimension—learning to retain task-relevant layer features from each model while discarding redundant ones.

**Model Task-Relevance Scoring**: With L2 regularization on $\mathbf{W}$, each backbone $M_k$'s importance score $s_k$ is computed as the L2 norm of its corresponding columns. The loss is:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \sum_{k=1}^K s_k$$

After training, $s_k$ directly reflects each model's task relevance, guiding model subset selection. The selected subset is then used to retrain without regularization.

### Loss & Training

- **Optimizer**: AdamW, lr=0.001, weight decay=0.0001
- **Training**: 100 epochs, 10-epoch linear warmup + cosine schedule
- **Input**: 224×224, no data augmentation
- **Identical hyperparameters across all datasets**—no per-task tuning required (vs. Head2Toe/SMP which require dataset-specific hyperparameters)
- Regularization coefficient λ=0.01 (used only when evaluating model relevance)

## Key Experimental Results

### Main Results: Single-Model Probing (VTAB-1k, ViT-B/16 ImageNet-21K)

| Method | Type | Natural | Specialised | Structured | Overall |
|--------|------|---------|-------------|------------|---------|
| Adapter+ | PEFT | 83.3 | 86.2 | 63.3 | **77.6** |
| Full Fine-tuning | Tuning | 78.6 | 86.3 | 57.8 | 74.2 |
| **ComBo (ours)** | Probing | 79.7 | 84.5 | **59.5** | **74.6** |
| SMP | Probing | 80.7 | 84.8 | 55.4 | 73.6 |
| Head2Toe | Probing | 80.2 | 84.7 | 47.7 | 70.9 |
| Linear Probing | Probing | 73.9 | 79.5 | 29.6 | 61.0 |

ComBo achieves the best performance among probing methods, with a substantial advantage on Structured tasks (59.5 vs. 55.4), surpassing full fine-tuning.

### Main Results: Multi-Model Probing

| Method | Natural | Specialised | Structured | Overall |
|--------|---------|-------------|------------|---------|
| ComBo Top-2 Models | 84.0 | 86.3 | **65.3** | **78.6** |
| ComBo All 4 Models | 83.3 | 86.5 | 64.8 | 78.2 |
| RADIOv2.5 + Adapter+ (distillation) | 83.8 | 86.6 | — | ~78 |
| Best Single Model (DINOv2) | 82.6 | 85.9 | 65.0 | 77.9 |

ComBo Top-2 (78.6) outperforms RADIOv2.5+Adapter+, which requires expensive distillation, without backpropagating through any large model.

### Ablation Study

**Validity of Model Relevance Scoring**: Importance scores derived from regularized training accurately identify the most task-relevant models. Selecting the Top-2 models (78.6) outperforms using all four models (78.2), indicating that removing irrelevant models reduces noise and improves performance.

**Criticality of Spatial Information Preservation**: ComBo substantially outperforms Head2Toe/SMP on Structured tasks (59.5 vs. 47.7/55.4), primarily because it retains the full spatial feature map rather than applying pooling.

### Key Findings

1. Complementarity across foundation models is real: DINOv2 excels on Natural/Structured tasks, while CLIP/SigLIP are stronger on certain Specialised tasks.
2. Intermediate layers can be more informative than the final layer; multi-layer probing is essential.
3. ComBo with fixed hyperparameters achieves stable performance across all 19 diverse tasks, demonstrating strong generalization.
4. For the first time, a probing method matches or exceeds distillation-based fusion approaches.

## Highlights & Insights

- **Minimalist yet effective**: Only a 1.7M-parameter transformer combined with affine projection, requiring no backbone gradients.
- **Computation-friendly**: Multiple large models require only forward inference for feature extraction; training is confined to the small adapter.
- **Unified framework**: Applicable to both single-model and multi-model settings without per-task hyperparameter tuning.
- **Built-in model selection**: Task relevance of each backbone is automatically assessed via the L2 norm of probing weights, eliminating the need for exhaustive trial-and-error.

## Limitations & Future Work

1. **Only validated at ViT-B scale**: Scalability to larger models (ViT-L/G) remains to be verified.
2. **Classification tasks only**: VTAB-1k consists entirely of classification problems; dense prediction tasks such as detection and segmentation have not been evaluated.
3. **Forward inference required for all models**: Although backpropagation is not needed, storing multi-layer feature maps from multiple models still incurs GPU memory overhead.
4. **DINOv2 alone is already strong**: The gain from multi-model combination is modest (77.9→78.6), and the cost-benefit trade-off warrants consideration.
5. Combining ComBo with PEFT methods (e.g., using ComBo for model selection followed by Adapter+ on the best model) is a promising direction.

## Related Work & Insights

- **Head2Toe / SMP**: Pioneers of multi-layer probing, but require hyperparameter tuning and pooling—both issues addressed by ComBo.
- **RADIO / SAM-CLIP**: The distillation-fusion paradigm; ComBo offers a more lightweight alternative.
- **Platonic Representation**: Do different foundation models converge? Experiments in this paper suggest that current-generation models still exhibit significant differences.
- Insight: Multi-model fusion does not necessarily require distillation or joint training; frozen probing alone can effectively exploit complementarity.

## Rating

- Novelty: ★★★★☆ (First probing framework to achieve efficient multi-model fusion with built-in model selection)
- Technical Depth: ★★★☆☆ (Method is elegant; core components are projection + small transformer; theoretical analysis is limited)
- Experimental Thoroughness: ★★★★☆ (Comprehensive evaluation on all 19 VTAB-1k tasks, with complete multi-model combination and ablation studies)
- Practical Value: ★★★★★ (Computationally friendly, no large-model backpropagation required, fixed hyperparameters out of the box)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference](dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ICLR 2026\] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](../../ICLR2026/interpretability/semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)
- [\[NeurIPS 2025\] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)

</div>

<!-- RELATED:END -->

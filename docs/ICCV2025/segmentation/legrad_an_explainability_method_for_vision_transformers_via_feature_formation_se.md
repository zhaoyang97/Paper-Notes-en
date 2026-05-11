---
title: >-
  [Paper Note] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity
description: >-
  [ICCV 2025][Segmentation][Explainability] This paper proposes LeGrad, a layer-wise explainability method designed specifically for ViTs. It computes the gradient of the activation with respect to the attention map at eac…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Explainability"
  - "Vision Transformer"
  - "Attention Gradient"
  - "Open-Vocabulary Segmentation"
  - "CLIP"
date: 2026-05-08
content_hash: 79bd815531650552
---

# LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity

**Conference**: ICCV 2025
**arXiv**: [2404.03214](https://arxiv.org/abs/2404.03214)
**Code**: None
**Area**: Image Segmentation
**Keywords**: Explainability, Vision Transformer, Attention Gradient, Open-Vocabulary Segmentation, CLIP

## TL;DR

This paper proposes LeGrad, a layer-wise explainability method designed specifically for ViTs. It computes the gradient of the activation with respect to the attention map at each layer as the explanation signal, aggregates these signals across layers to produce high-quality spatial saliency maps, and demonstrates superior spatial fidelity in segmentation, perturbation, and open-vocabulary settings.

## Background & Motivation

Vision Transformers (ViTs) have become the standard architecture in computer vision, yet their interpretability remains challenging due to the long-range dependency modeling of the self-attention mechanism. Existing explainability methods suffer from the following issues:

**Traditional methods are inapplicable**: GradCAM relies on convolutional layers, and LRP requires layer-specific propagation rules, neither of which can be directly applied to ViTs.

**Limitations of attention-based methods**: Raw Attention and Rollout ignore nonlinear interactions and the distinction between positive and negative contributions, potentially producing misleading explanations.

**Problems with CheferCAM**: It uses gradient-weighted attention maps, incurring high computational cost and lacking flexibility for architectural variations.

**Insufficient open-vocabulary support**: Existing methods suffer severe performance degradation on large-scale open-vocabulary datasets (e.g., OpenImagesV7 with 5,827 categories).

**Poor scalability to large models**: Most methods struggle to scale effectively to very large models such as ViT-BigG.

**Core Insight**: Feature formation in ViTs proceeds iteratively across layers; explainability methods should capture each layer's contribution to the final representation rather than relying solely on the final output. LeGrad uses the gradient itself (rather than gradient-weighted attention) as the explanation signal, enabling cross-layer additivity.

## Method

### Overall Architecture

The core idea of LeGrad is straightforward: compute the gradient of the target class activation with respect to the attention map at each ViT layer, clip via ReLU, average across heads and patch dimensions, and aggregate across layers.

### Key Designs

1. **Per-layer explanation map computation**:

   - For a given layer $l$, compute the activation $s^l = \bar{y}^l_{[\hat{c}]}$ using the mean of the intermediate token representations $Z^l$, denoted $\bar{z}^l$, passed through the classifier/text embedding $\mathcal{C}$.
   - Compute the gradient of $s^l$ with respect to the attention map $\mathbf{A}^l \in \mathbb{R}^{h \times (n+1) \times (n+1)}$: $\nabla\mathbf{A}^l = \frac{\partial s^l}{\partial \mathbf{A}^l}$
   - Key step: clip negative gradients via ReLU $(\nabla\mathbf{A}^l_{h,i,.})^+$ to prevent negative gradients from corrupting positive activations.
   - Average across heads and patch dimensions to obtain $\hat{E}^l(s^l) = \frac{1}{h \cdot (n+1)}\sum_h\sum_i(\nabla\mathbf{A}^l_{h,i,.})^+$
   - **Design Motivation**: Gradients directly reflect the sensitivity of attention maps to the prediction, which is more direct than gradient-weighted attention; per-layer gradients are naturally additive without requiring additional normalization.

2. **Multi-layer aggregation**:

   - Average the per-layer explanation maps: $\bar{\mathbf{E}} = \frac{1}{L}\sum_l \hat{E}^l(s^l)_{1:}$
   - Remove the CLS token column, reshape to 2D, and apply min-max normalization: $\mathbf{E} = \text{norm}(\text{reshape}(\bar{\mathbf{E}}))$
   - **Design Motivation**: Information aggregation in ViTs is distributed across multiple layers, especially in larger models; using only the final layer discards rich intermediate-layer information.

3. **Attentional Pooler adaptation** (e.g., SigLIP):

   - For ViTs using attentional pooling, process the intermediate representations $Z^l$ at each layer with the Attentional Pooler to obtain pooled queries $q^l$.
   - Replace the self-attention map with the Pooler's attention map $\mathbf{A}_{pool} \in \mathbb{R}^{h \times 1 \times n}$ for gradient computation.
   - **Design Motivation**: This adaptation allows LeGrad to support different feature aggregation strategies beyond CLS token–based architectures.

### Loss & Training

LeGrad is a **training-free** post-hoc explainability method and involves no loss function or training procedure. Explanation maps are generated with a single forward and backward pass.

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | LeGrad | CheferCAM | TextSpan | Rollout | Gain |
|---|---|---|---|---|---|---|
| ImageNet-S Segmentation (ViT-B/16) | mIoU↑ | **58.66** | 47.47 | 40.26 | 40.64 | +11.19 |
| ImageNet-S Segmentation (ViT-B/16) | Pixel Acc↑ | **77.52** | 69.21 | 73.01 | 60.63 | +4.51 |
| OpenImagesV7 OV (ViT-B/16) | p-mIoU↑ | **48.38** | 5.87 | 9.44 | 8.75 | +38.94 |
| OpenImagesV7 OV (ViT-L/14) | p-mIoU↑ | **47.69** | 2.51 | 21.73 | 6.85 | +25.96 |
| OpenImagesV7 OV (ViT-H/14) | p-mIoU↑ | **46.51** | 9.49 | 23.74 | 5.82 | +22.77 |
| SigLIP-B/16 OV | p-mIoU↑ | **25.40** | 1.94 | - | 0.07 | +23.46 |
| ADE20K Sound Seg. | mIoU↑ | **38.9** | - | - | - | +14.7 (vs DenseAV) |

### Ablation Study

| Configuration | Description | Key Findings |
|---|---|---|
| Layer count (ViT-B/16) | Varying number of layers used | Smaller models require only a few layers; larger models benefit from more. |
| Layer count (ViT-L/14) | Using more layers | Information aggregation is more distributed, requiring broader layer coverage. |
| Layer count (ViT-H/14) | Increasing layer count | Layer contributions are more uniform in large models, underscoring the necessity of multi-layer aggregation. |
| Per-layer visualization | Layer-wise heatmap analysis | Localization signals are distributed across multiple layers rather than concentrated in a single layer. |

### Key Findings

- **Dominant advantage in open-vocabulary settings**: LeGrad achieves 48.38 p-mIoU on OpenImagesV7, more than 5× higher than the second-best method, TextSpan (9.44).
- **High inference speed**: 96 FPS (ViT-B/16), close to GradCAM (108 FPS) and far faster than CheferCAM (21 FPS) and TextSpan (3.8 FPS).
- **Strong scalability to large models**: Remains effective on ViT-BigG/14 (2.5 billion parameters).
- **Gradient distribution analysis**: Layer importance distributions differ substantially across pre-training regimes (Laion400M vs. OpenAI vs. MetaCLIP), suggesting utility as a model "fingerprint."

## Highlights & Insights

- **Conceptual simplicity**: The core idea is minimal — using the gradient itself rather than gradient-weighted attention as the explanation signal, enabling cross-layer additivity.
- **Strong generality**: Compatible with both CLS token aggregation and Attentional Pooler architectures, and applicable to audio-visual models (e.g., ImageBind).
- **Open-vocabulary effectiveness**: A 5× improvement over the state of the art on OpenImagesV7 (5,827 categories) demonstrates LeGrad's advantage in fine-grained recognition.
- **Importance of ReLU clipping**: A simple negative-gradient clipping operation effectively removes the uniform noise activations commonly observed in ViTs.
- **Model "fingerprint" discovery**: Layer-wise gradient distributions can serve as diagnostic tools for characterizing different pre-training strategies.

## Limitations & Future Work

- For very high-resolution images, patch granularity constrains spatial precision due to the ViT patch size.
- No post-processing (e.g., CRF) is applied to further refine segmentation boundaries.
- Multi-layer aggregation uses simple averaging; adaptive weighting strategies remain unexplored.
- Generalization to video understanding or 3D scene understanding has not been validated.
- ReLU clipping may be overly aggressive, potentially discarding meaningful negative-gradient information.

## Related Work & Insights

- GradCAM is the landmark explainability method of the convolutional network era; LeGrad can be viewed as its ViT-era counterpart.
- CheferCAM propagates gradient-weighted attention via matrix multiplication, incurring high cost; LeGrad replaces this with direct gradient summation, achieving $O(L)$ complexity without matrix multiplications.
- TextSpan operates without gradients but is slow (3.8 FPS) and less stable on large models.
- LeGrad can inspire direct application to zero-shot segmentation with CLIP by generating pixel-level explanations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The approach is simple yet effective, replacing gradient-weighted attention with raw gradients as the explanation signal.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers segmentation, perturbation, open-vocabulary, audio-visual, speed, and large-model evaluations comprehensively.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and experiments are well-organized.
- **Value**: ⭐⭐⭐⭐ Strong contribution to open-vocabulary explainability with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)
- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](../../ICLR2026/segmentation/revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](../../CVPR2026/segmentation/mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[ICCV 2025\] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond](partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)
- [\[ICLR 2026\] Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers](../../ICLR2026/segmentation/thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers.md)

</div>

<!-- RELATED:END -->

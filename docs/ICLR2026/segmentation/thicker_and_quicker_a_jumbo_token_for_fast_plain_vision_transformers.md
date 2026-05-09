---
title: >-
  [Paper Note] Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers
description: >-
  [ICLR 2026][Segmentation][Vision Transformer] This paper proposes Jumbo: a method that expands the ViT CLS token to $J$ times its original width, splits it into $J$ patch-width tokens before attention, and reassembles them after attention for processing by a dedicated wide FFN. With negligible computational overhead, Jumbo substantially increases global modeling capacity, enabling plain ViT to surpass dedicated efficient architectures (EfficientViT, SHViT, MobileNetV4) in high-throughput inference regimes while preserving all architectural advantages of the plain ViT.
tags:
  - ICLR 2026
  - Segmentation
  - Vision Transformer
  - CLS Token
  - Efficient Architecture
  - Registers
  - Time Series
  - ImageNet
date: 2026-05-08
content_hash: 2e28faa0acd1dd7d
---

# Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers

**Conference**: ICLR 2026  
**arXiv**: [2502.15021](https://arxiv.org/abs/2502.15021)  
**Code**: None  
**Area**: Image Segmentation  
**Keywords**: Vision Transformer, CLS Token, Efficient Architecture, Registers, Time Series, ImageNet

## TL;DR

This paper proposes Jumbo: a method that expands the ViT CLS token to $J$ times its original width, splits it into $J$ patch-width tokens before attention, and reassembles them after attention for processing by a dedicated wide FFN. With negligible computational overhead, Jumbo substantially increases global modeling capacity, enabling plain ViT to surpass dedicated efficient architectures (EfficientViT, SHViT, MobileNetV4) in high-throughput inference regimes while preserving all architectural advantages of the plain ViT.

## Background & Motivation

Vision Transformers offer simplicity, flexibility, and efficiency: they support token dropping (critical for SOTA self-supervised learning algorithms), readily accommodate multimodal data, and directly benefit from kernel optimizations such as FlashAttention. However, in high-throughput inference settings (tiny/nano model sizes), plain ViT lags far behind dedicated efficient architectures such as EfficientViT and SHViT.

**Root Cause**: Under the standard setting (224×224 images, 16×16 patches), 196 patch tokens and 1 CLS token imply that only 1/197 of the model capacity is devoted to global information aggregation—clearly insufficient. Darcet et al. (2024) found that ViT tends to "hijack" certain patch tokens as pseudo-CLS tokens and proposed Registers as additional global tokens to alleviate this issue.

However, Registers have a fundamental limitation: global tokens interact with each other only through attention, which is essentially an information-routing mechanism (weighted linear combination) with limited expressive power. What is missing is the nonlinear function modeling capability provided by FFNs.

**Key Insight**: Concatenating global tokens and passing them through a wide FFN enables nonlinear processing of global features at virtually negligible cost—since only a single token is processed.

## Method

### Overall Architecture

Jumbo introduces minimal modifications to the standard ViT: (1) the CLS token width is expanded to $J \cdot D$; (2) it is split before attention and reassembled after attention; (3) a separate wide FFN processes the Jumbo CLS token. The method preserves the attention-only, non-hierarchical plain ViT structure throughout.

### Key Designs

1. **Creation and Processing of the Jumbo CLS Token**:

    - A learnable CLS token of width $J \cdot D$, denoted $\mathbf{x}_{\text{Jumbo}} \in \mathbb{R}^{J \cdot D}$, is initialized.
    - **Before attention**: it is split along the feature dimension into $J$ tokens of width $D$, which are concatenated with patch tokens to form a sequence of length $(N+J)$.
    - **During attention**: standard multi-head self-attention processes all $(N+J)$ equal-width tokens.
    - **After attention**: the $J$ Jumbo slices are extracted from the sequence and concatenated along the channel dimension to recover $\mathbb{R}^{1 \times J \cdot D}$.
    - **Dedicated FFN**: an independent FFN of width $J \cdot D$ processes the reassembled Jumbo token; patch tokens are processed by a shared standard FFN.
    - The patch FFN in the final layer is discarded, as the classification head projects directly from the Jumbo token.

2. **Why Computational Overhead Is Negligible (Design Motivation)**:

    - The computational cost of a ViT layer is almost entirely determined by the number of patch tokens $N$ and the patch width $D$.
    - Adding $J=6$ extra tokens has a negligible impact on attention FLOPs ($(N+J)$ vs. $N$, with $N=196$).
    - Although the wide FFN has more parameters, it processes only a single token—its FLOP contribution is negligible.

3. **Two Core Hypotheses**:

    - **Hypothesis 1**: The smaller the patch width (i.e., the smaller the model), the larger the gain from Jumbo—because narrower networks suffer more severely from insufficient global capacity.
    - **Hypothesis 2**: The higher the output dimensionality (i.e., the more complex the task), the larger the gain from Jumbo—more width is needed to store and reason over a greater number of concepts.

4. **Transfer from Vision to Time Series**:

    - Jumbo is applied to the PatchTST architecture by adding a Jumbo CLS token to patchified 1D time series.
    - No architectural modifications are required, demonstrating the generality of the plain transformer paradigm.

### Loss & Training

- **ImageNet-1K**: function matching (knowledge distillation), trained at 128×128 px for 400 epochs followed by fine-tuning at 224×224 px for 20 epochs.
- Teachers: DeiT-III base (85.7%) and large (87.0%).
- AdamW optimizer, learning rate $\in \{1\text{e-}3, 3\text{e-}3\}$, batch size 1024.
- Data augmentation: mixup $\alpha=0.8$, CutMix $\alpha=1$, 3-Augment / AutoAugment.
- **ImageNet-21K**: trained directly for 50 epochs with token dropping (linearly reduced from 90% to 10%) to reduce training cost.
- **Time Series**: PatchTST framework with a grid search over 12 hyperparameter combinations (4 learning rates × 3 dropout values).

## Key Experimental Results

### Main Results — ImageNet-1K High-Throughput Regime

| Model | Size | Throughput (imgs/s) | ImageNet-Val Top-1 | ImageNet-v2 Top-1 |
|------|------|-----------|----------------|-------------|
| ViT+Registers | nano (D=128) | 105.9K | 53.6 | 42.4 |
| ViT+Jumbo | nano (D=128) | 101.7K | **68.8** (+15.2) | **55.1** |
| ViT+Registers | tiny (D=192) | 52.2K | 68.8 | 55.9 |
| ViT+Jumbo | tiny (D=192) | 56.5K | **73.0** (+4.2) | **59.4** |
| EfficientViT-B1 | — | 38.7K | 72.8 | 60.4 |
| SHViT-S3 | — | 70.4K | 71.2 | 58.6 |
| MobileNetV4-conv-medium | — | 54.5K | 73.3 | 60.6 |

* Jumbo nano achieves higher throughput than Registers tiny with comparable accuracy.

### ImageNet-21K (10,450 Classes)

| Model | Size | Throughput | Top-1 |
|------|------|-------|-------|
| ViT+Registers | small | 8.4K | 41.48 |
| **ViT+Jumbo** | small | 8.0K | **44.95** (+3.4) |
| ViT+Registers | base | 3.6K | 46.31 |
| **ViT+Jumbo** | base | 3.2K | **47.28** (+1.0) |

### Time Series Classification (PatchTST Framework)

| Variant | Univariate Best Rank | Multivariate Best Rank |
|------|-----------|-----------|
| PatchTST | 2.0 | 2.1 |
| PatchTST+Registers | 2.5 | 2.1 |
| **PatchTST+Jumbo** | **1.5** | **1.6** |

### Ablation Study

| Patch Width | Jumbo $J$ | Inner FFN mult | Throughput (128px) | IN-Val Top-1 |
|------------|-------|-------------|-----------|-------------|
| 192 | 2 | 2 | 71.6K | 70.0 |
| 192 | 4 | 4 | 64.9K | 72.2 |
| 192 | 6 | 4 | 56.5K | **73.0** |
| 384 | 6 | 4 | 19.5K | **78.3** |

### Key Findings

- **Hypothesis 1 confirmed**: Jumbo gains increase as patch width decreases—nano (+13.5%) > tiny (+3.2%) > small (~0%).
- **Hypothesis 2 confirmed**: On ImageNet-21K (10,450 classes), even ViT-small gains +3.4%, whereas no gain is observed for small models on ImageNet-1K (1,000 classes).
- **Parameter count and mitigation**: With $J=6$, ViT-base Jumbo parameters increase from 25.7M to 55M; cross-layer weight sharing with LoRA reduces this to 88.3M→88.8M with negligible accuracy loss.
- **Plain ViT first matches/surpasses dedicated efficient architectures in high-throughput settings**: ViT+Jumbo is the first attention-only, non-hierarchical architecture to be competitive in this regime.

## Highlights & Insights

- Remarkably simple modifications yield substantial gains—two splits, two concatenations, and one independent FFN, amounting to fewer than 10 lines of additional code.
- All advantages of plain ViT are preserved: token dropping (compatible with MAE, I-JEPA, and other SSL methods), multimodal support, and FlashAttention compatibility.
- The insight that "insufficient width is the core bottleneck of narrow ViTs" is highly valuable—Registers only add a linear component to global capacity, whereas Jumbo supplies the missing nonlinear component.
- The experimental design merits recognition for its fairness: all models are trained under identical protocols (same teacher, same hyperparameter grid), avoiding confounding between recipe differences and architectural differences.
- The seamless transfer from image to time series demonstrates that Jumbo is a general-purpose method rather than a vision-specific solution.

## Limitations & Future Work

- ViT-small shows no gain on ImageNet-1K—the method is not universally applicable.
- Parameter count increases substantially ($\approx$2× at $J=6$); although cross-layer sharing with LoRA provides a solution, it adds implementation complexity.
- For ViT-base with $J=6$ on ImageNet-21K, memory constraints require reducing to $J=3$.
- Large-scale pretraining (CLIP, DINOv2, etc.) and downstream tasks such as object detection and segmentation have not been evaluated.
- Training is conducted exclusively via knowledge distillation; the effectiveness of Jumbo under from-scratch training (without a teacher) remains unverified.
- Direct comparisons with hierarchical architectures such as Hiera are absent (though the design philosophies differ).

## Related Work & Insights

- Darcet et al. (ICLR 2024) ViT+Registers is the direct predecessor—Jumbo can be viewed as a "super-Register with nonlinearity."
- Hiera (Ryali et al., 2023) simplifies hierarchical ViT from a different angle by replacing convolutions with pooling.
- PatchTST (Nie et al., 2023) demonstrates the competitiveness of plain transformers for time series.
- Insight: ViT encoders in vision-language frameworks such as CLIP must model vocabularies of 50K+ tokens; Jumbo is expected to yield larger gains in such high-output-dimensionality settings.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](../../CVPR2026/segmentation/mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[ICLR 2026\] Locality-Attending Vision Transformer](locality-attending_vision_transformer.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)
- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](../../ICCV2025/segmentation/legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Locality-Attending Vision Transformer
description: >-
  [ICLR 2026][Segmentation][Vision Transformer] This paper proposes LocAt, a modular plug-in comprising GAug and PRR, which biases attention toward local neighborhoods via learnable Gaussian kernels and refines patch repre…
tags:
  - "ICLR 2026"
  - "Segmentation"
  - "Vision Transformer"
  - "Local Attention"
  - "Gaussian Kernel"
  - "Patch Representation Refinement"
  - "Dense Prediction"
  - "Segmentation Improvement"
date: 2026-05-08
content_hash: 693215af781ade26
---

# Locality-Attending Vision Transformer

**Conference**: ICLR 2026
**arXiv**: [2603.04892](https://arxiv.org/abs/2603.04892)  
**Code**: [GitHub](https://github.com/sinahmr/LocAtViT/)  
**Area**: Segmentation
**Keywords**: Vision Transformer, Local Attention, Gaussian Kernel, Patch Representation Refinement, Dense Prediction, Segmentation Improvement

## TL;DR

This paper proposes LocAt, a modular plug-in comprising GAug and PRR, which biases attention toward local neighborhoods via learnable Gaussian kernels and refines patch representations. Without modifying the training objective, it improves ViT segmentation performance on ADE20K by over 6% while simultaneously boosting classification accuracy.

## Background & Motivation

1. **Global attention in ViT favors classification but hurts segmentation**: The global self-attention mechanism in ViT excels at capturing long-range dependencies and performs well on classification, but is ill-suited for dense prediction tasks (e.g., semantic segmentation) that require precise localization and fine-grained spatial detail. Global attention dilutes local cues.

2. **Classification training neglects patch-level representation quality**: Standard ViT classification computes the loss solely from the [CLS] token output; patch-position outputs receive no direct supervision. This leads to degradation of spatial token representations in the final layer—patch tokens progressively align with the [CLS] token and lose distinctive local structural information.

3. **Existing improvements compromise the original ViT architecture**: Hierarchical ViTs (e.g., Swin) introduce window attention and multi-stage designs; convolution-hybrid schemes append additional convolutional modules. Although these approaches improve dense prediction, they alter the ViT architecture and reduce compatibility with foundation models (e.g., CLIP).

4. **Limitations of GAP as a classification head**: Global Average Pooling (GAP) imposes uniform gradients on all patches, forcing background-region representations to align with the classification target—a property that is detrimental to segmentation (degrading segmentation performance on Base models).

## Method

### Overall Architecture: LocAt = GAug + PRR

- **Function**: Adds lightweight, modular plug-ins to a standard ViT to enhance segmentation capability while preserving classification performance.
- **Design Motivation**: ViT is widely adopted in foundation models and preferred for its architectural simplicity. Rather than replacing ViT, LocAt augments its dense prediction capability with minimal modifications.
- **Mechanism**: (1) GAug adds a learnable Gaussian kernel bias to self-attention logits, guiding tokens to focus on local neighborhoods; (2) PRR inserts a parameter-free self-attention operation before the classification head, ensuring patch positions receive effective gradients. The two components are complementary—GAug improves feature interaction within the backbone, while PRR ensures gradient backpropagation to GAug parameters at the output stage.

### Key Design 1: Gaussian-Augmented Attention (GAug)

In each self-attention layer, the standard attention formulation is modified as:

$$\mathbf{Z} = \text{softmax}\left(\frac{\mathbf{q}\mathbf{k}^\top}{\sqrt{d}} + \mathbf{S}\right)\mathbf{v}$$

where the supplementary matrix $\mathbf{S}$ is constructed from a Gaussian kernel, providing a local attention bias for each patch:

1. **Adaptive variance prediction**: Per-patch 2D Gaussian variances are predicted from the spatial query matrix as $\mathbf{\Sigma} = f(\mathbf{q}_{sp}\mathbf{W}^\sigma) \in \mathbb{R}_+^{hw \times 2}$; small variance yields sharp local focus, large variance approximates uniform (global) attention.
2. **Gaussian kernel computation**: $\mathbf{G}_{pt} = \exp\left(-\frac{1}{2}\sum_{m=1}^{2}\frac{\mathbf{D}_{ptm}}{\mathbf{\Sigma}_{pm}}\right)$, where $\mathbf{D}$ denotes coordinate-wise squared distances between patches.
3. **Adaptive scaling**: $\bm{\alpha} = \text{softplus}(\mathbf{q}_{sp}\mathbf{W}^\alpha)$ predicts a per-query scaling coefficient that balances the original logits against the Gaussian bias.

GAug constitutes a soft, data-dependent locality mechanism—the network learns when and where to apply local focus and when to retain global attention. The [CLS] token does not participate in the Gaussian bias (it has no spatial coordinates); its corresponding rows and columns are set to zero.

### Key Design 2: Patch Representation Refinement (PRR)

Standard ViT classification computes the loss solely from the [CLS] output. The patch-position outputs receive no supervision, causing spatial representations to degrade. PRR inserts a **parameter-free** multi-head self-attention operation before the classification head:

$$\mathbf{x}_i^+ = \text{softmax}\left(\frac{\mathbf{x}_i \mathbf{x}_i^\top}{\sqrt{d}}\right)\mathbf{x}_i$$

The classification head then receives $\mathbf{x}_0^+$ (the [CLS] position). This operation causes the [CLS] token to assign **non-uniform** attention weights across patches, thereby propagating classification gradients **non-uniformly** to patch positions and encouraging each patch to maintain a distinctive, discriminative representation.

PRR can be viewed as an alternative to GAP: whereas GAP imposes uniform gradients on all patches, PRR adaptively allocates gradients based on content. Crucially, PRR also routes gradients to the GAug parameters in the final block, enabling effective learning.

### Additional Parameter Overhead

Only two small weight matrices per layer are introduced—$\mathbf{W}^\sigma \in \mathbb{R}^{d \times 2}$ and $\mathbf{W}^\alpha \in \mathbb{R}^{d \times 1}$—while PRR is entirely parameter-free. For the Base model, the additional parameters number only 2,340 (a 0.003% increase), and FLOPs increase negligibly (17.58G → 17.64G).

## Key Experimental Results

### Experimental Setup

- **Pre-training**: ImageNet-1K classification for 300 epochs, AdamW optimizer, batch size 1024.
- **Segmentation evaluation**: Backbone frozen; only a 1-layer MLP decoder is trained (20K iterations); evaluated on ADE20K, PASCAL Context, and COCO Stuff.
- **Backbone sizes**: Tiny (6M parameters), Small, Base (86M parameters).
- **Baselines**: ViT, Swin, RegViT (ViT + registers), RoPEViT, Jumbo.

### Main Results: Segmentation and Classification Performance

| Method | ADE mIoU | P-Context mIoU | C-Stuff mIoU | ImageNet Top-1 | Params (M) |
|---|---|---|---|---|---|
| ViT-Tiny | 17.30 | 33.71 | 20.29 | 72.39 | 6 |
| **LocAtViT-Tiny** | **23.47 (+6.17)** | **38.57 (+4.86)** | **26.15 (+5.86)** | **73.94 (+1.55)** | 6 |
| Swin-Tiny | 25.58 | 36.78 | 28.34 | 81.18 | 28 |
| + LocAt | 26.52 (+0.94) | 37.65 (+0.87) | 29.09 (+0.75) | 81.43 (+0.25) | 28 |
| RegViT-Tiny | 15.98 | 33.45 | 19.58 | 72.90 | 6 |
| + LocAt | 24.39 (+8.41) | 39.90 (+6.45) | 27.38 (+7.80) | 74.08 (+1.18) | 6 |
| ViT-Base | 28.40 | 43.10 | 30.43 | 80.99 | 86 |
| **LocAtViT-Base** | **32.64 (+4.24)** | **45.35 (+2.25)** | **33.62 (+3.19)** | **82.31 (+1.32)** | 86 |
| RegViT-Base | 27.93 | 41.81 | 28.99 | 80.71 | 86 |
| + LocAt | 32.71 (+4.78) | 46.14 (+4.33) | 34.12 (+5.13) | 82.19 (+1.18) | 86 |

### Ablation Study: Component Contribution Analysis

| Configuration | ADE (Tiny) | ADE (Base) | ImageNet (Tiny) | ImageNet (Base) |
|---|---|---|---|---|
| ViT baseline | 17.30 | 28.40 | 72.39 | 80.99 |
| + GAug | 18.98 (+1.68) | 30.26 (+1.87) | 73.16 (+0.77) | 82.00 (+1.01) |
| + PRR | 21.60 (+4.30) | 29.89 (+1.49) | 73.71 (+1.32) | 82.19 (+1.20) |
| + GAug + PRR (LocAt) | **23.47 (+6.17)** | **32.64 (+4.24)** | **73.94 (+1.55)** | **82.31 (+1.32)** |
| ViT + GAP | 19.65 | 27.99 | 72.50 | 81.84 |
| ViT − positional encoding | 15.13 | 24.59 | 69.36 | 79.39 |
| LocAtViT − positional encoding | 22.69 | 29.73 | 73.10 | 82.17 |

### Key Findings

1. **Substantial and universal segmentation gains**: LocAt yields segmentation improvements across all five baselines (ViT, Swin, RegViT, RoPEViT, Jumbo) and all three model sizes, with a maximum gain of +8.41% (RegViT-Tiny on ADE).
2. **Classification accuracy improves rather than degrades**: All LocAt-augmented models achieve equal or higher ImageNet Top-1 accuracy (up to +1.55%), demonstrating that locality bias does not conflict with global modeling.
3. **GAug and PRR are complementary and synergistic**: Each component is individually effective (GAug +1.68, PRR +4.30), and their combination yields a further improvement to +6.17, confirming that gradient routing via PRR is critical for effective learning of GAug parameters.
4. **LocAt encodes positional information and beyond**: After removing positional encodings, LocAtViT still outperforms ViT with positional encodings (ADE 22.69 vs. 17.30), indicating that the Gaussian kernel captures not only positional information but also richer spatial structure.
5. **Effectiveness in self-supervised settings**: Replacing ViT-S with LocAtViT-S in the DINO framework yields a +2.13% improvement in linear probing and +2.27% in k-NN evaluation.

## Highlights & Insights

- The design is minimal yet highly effective: only three weight matrices of negligible size ($\mathbf{W}^\sigma$, $\mathbf{W}^\alpha$) are added per layer; PRR is entirely parameter-free; FLOPs increase is negligible.
- Fully modular and plug-and-play: LocAt can be directly applied to any ViT or its variants without modifying the training objective or data augmentation strategy.
- The analysis of gradient flow provides an insightful diagnosis of why ViT underperforms on segmentation: the absence of supervision on patch outputs leads to representational degradation.
- Gaussian kernel variances are predicted from queries, enabling data-adaptive local/global balance that is more flexible than fixed-window attention.

## Limitations & Future Work

- Segmentation evaluation relies solely on a frozen backbone with a 1-layer MLP decoder; compatibility with full segmentation heads (e.g., UPerNet) is not assessed.
- Validation is limited to natural images; applicability to domains such as medical imaging and remote sensing remains unexplored.
- The approach has not been validated on large-scale foundation models (e.g., CLIP); computational constraints precluded exploration of this important direction.
- The anisotropic 2D Gaussian kernel operates independently along row and column axes, without accounting for more flexible spatial structures such as rotations.

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [\[ICLR 2026\] Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers](thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers.md)
- [\[AAAI 2026\] Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation](../../AAAI2026/segmentation/adaptive_morph-patch_transformer_for_aortic_vessel_segmentat.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](../../CVPR2026/segmentation/mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[NeurIPS 2025\] STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking](../../NeurIPS2025/segmentation/step_a_unified_spiking_transformer_evaluation_platform_for_fair_and_reproducible.md)

</div>

<!-- RELATED:END -->

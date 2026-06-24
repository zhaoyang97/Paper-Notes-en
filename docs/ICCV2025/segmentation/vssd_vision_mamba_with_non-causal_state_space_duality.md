---
title: >-
  [Paper Note] VSSD: Vision Mamba with Non-Causal State Space Duality
description: >-
  [ICCV 2025][Segmentation][State Space Models] This paper proposes Non-Causal State Space Duality (NC-SSD), which transforms the SSD formulation of Mamba2 into a non-causal form by retaining the relative weights of token contributions in lieu of the cumulative decay of hidden states. Built upon NC-SSD, the VSSD visual backbone surpasses existing SSM-based models across classification, detection, and segmentation benchmarks while achieving 20%–50% faster training speed.
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "State Space Models"
  - "Mamba2"
  - "Non-Causal SSD"
  - "Visual Backbone"
  - "Linear Complexity"
date: 2026-05-08
content_hash: ea0b90465d79cf46
---

# VSSD: Vision Mamba with Non-Causal State Space Duality

**Conference**: ICCV 2025
**arXiv**: [2407.18559](https://arxiv.org/abs/2407.18559)  
**Code**: [GitHub](https://github.com/YuHengsss/VSSD)  
**Area**: Image Segmentation
**Keywords**: State Space Models, Mamba2, Non-Causal SSD, Visual Backbone, Linear Complexity

## TL;DR

This paper proposes Non-Causal State Space Duality (NC-SSD), which transforms the SSD formulation of Mamba2 into a non-causal form by retaining the relative weights of token contributions in lieu of the cumulative decay of hidden states. Built upon NC-SSD, the VSSD visual backbone surpasses existing SSM-based models across classification, detection, and segmentation benchmarks while achieving 20%–50% faster training speed.

## Background & Motivation

Vision Transformers have achieved remarkable success in computer vision owing to their global receptive fields and powerful modeling capacity, yet the quadratic complexity of self-attention limits their applicability to long sequences (i.e., high-resolution images). State Space Models (SSMs) offer an efficient alternative with linear complexity, and Mamba2 further introduces State Space Duality (SSD), which simplifies the transition matrix $\mathbf{A}$ to a scalar to improve both performance and efficiency.

Nevertheless, SSD/SSM faces two core challenges when applied to visual tasks:

**Causality Constraint** (Challenge 1): Each token can only attend to preceding tokens, precluding the integration of information from subsequent tokens — a fundamental mismatch for inherently non-causal image data.

**Disruption of Structural Relationships** (Challenge 2): Flattening 2D feature maps into 1D sequences causes spatially adjacent tokens to become distant in the sequence, destroying the inherent structural information.

Existing solutions such as VMamba mitigate these issues by introducing multiple scanning paths, but do not fundamentally resolve the causality constraint. The core problem is: **can a more effective and efficient approach than multi-scan be found to adapt SSD to non-causal visual data?**

## Method

### Overall Architecture

VSSD is a four-stage hierarchical visual backbone. The first three stages employ VSSD Blocks (NC-SSD + FFN + LPU), while the final stage uses standard Multi-Head Self-Attention (MSA). Overlapping convolutional downsampling layers are adopted to introduce inductive biases.

### Key Design 1: Non-Causal State Space Duality (NC-SSD)

The linear recurrence form of standard SSD is:

$$h(t) = A_t h(t-1) + \mathbf{B}_t x(t), \quad y(t) = \mathbf{C}_t h(t)$$

where the scalar $A_t$ controls the retention ratio of the previous hidden state. The authors propose a key insight: **discard the absolute magnitude between the hidden state and the current input, retaining only the relative weights**, yielding:

$$h(t) = h(t-1) + \frac{1}{A_t}\mathbf{B}_t x(t) = \sum_{i=1}^{t} \frac{1}{A_i}\mathbf{B}_i x(i)$$

Under this formulation, each token's contribution is determined solely by its own weight $\frac{1}{A_i}$, with no dependence on preceding tokens.

Incorporating bidirectional scanning, the hidden state of the $i$-th token becomes:

$$\mathbf{H}_i = \sum_{j=1}^{L} \frac{1}{A_j}\mathbf{Z}_j + \frac{1}{A_i}\mathbf{Z}_i$$

Neglecting the offset term $\frac{1}{A_i}\mathbf{Z}_i$, all tokens share a single global hidden state $\mathbf{H} = \sum_{j=1}^{L}\frac{1}{A_j}\mathbf{Z}_j$. This implies:
- **Causal masking is naturally eliminated**, removing the need for specific scanning path designs (resolving Challenge 1).
- Token contributions are **independent of spatial distance**, avoiding structural degradation from flattening (resolving Challenge 2).
- The global hidden state can be **computed in parallel**, improving training and inference speed.

The final NC-SSD simplifies to:

$$\mathbf{Y} = \mathbf{C}(\mathbf{B}^T(\mathbf{X} \cdot \mathbf{m}))$$

where $\mathbf{m} \in \mathbb{R}^L$ is a weighting vector derived from the learned $A$. Visualizations show that $\mathbf{m}$ predominantly attends to foreground features.

### Key Design 2: Hybrid Integration with Self-Attention

Standard MSA replaces NC-SSD only in the final stage, leveraging the advantages of self-attention for high-level features. Compared to uniformly interleaving attention across all layers as in Mamba2, this strategic hybridization is more efficient.

### Key Design 3: Overlapping Downsampling

Overlapping convolutions replace conventional non-overlapping downsampling (following MLLA), introducing beneficial inductive biases. Model depth is adjusted accordingly to maintain comparable parameter counts and computational costs.

### Model Variants

| Variant | Blocks | Channels | Params | FLOPs |
|---------|--------|----------|--------|-------|
| VSSD-M | [2,2,18,4] | [48,196,192,384] | 14M | 2.3G |
| VSSD-T | [2,4,18,4] | [64,128,256,512] | 24M | 4.5G |
| VSSD-S | [3,4,18,5] | [64,128,256,512] | 40M | 7.4G |
| VSSD-B | [3,4,18,5] | [96,192,384,768] | 89M | 16.1G |

## Key Experimental Results

### ImageNet-1K Classification

| Model | Type | Params | FLOPs | Top-1 (%) |
|-------|------|--------|-------|-----------|
| Swin-T | Attn | 29M | 4.5G | 81.3 |
| ConvNeXt-T | Conv | 29M | 4.5G | 82.1 |
| VMambaV9-T | SSM | 31M | 4.9G | 82.5 |
| **VSSD-T** | **SSD** | **24M** | **4.5G** | **83.7** |
| Swin-S | Attn | 50M | 8.7G | 83.0 |
| VMamba-S | SSM | 44M | 11.2G | 83.5 |
| **VSSD-S** | **SSD** | **40M** | **7.4G** | **84.1** |
| VMambaV9-B | SSM | 89M | 15.4G | 83.9 |
| **VSSD-B** | **SSD** | **89M** | **16.1G** | **84.7** |

VSSD-T achieves 83.7% Top-1 accuracy with only 24M parameters, surpassing VMambaV9-T by 1.2%; with MESA, this further improves to 84.1%.

### COCO Object Detection and Instance Segmentation (Mask R-CNN 1×)

| Backbone | AP^box | AP^mask | Params | FLOPs |
|----------|:------:|:-------:|--------|-------|
| Swin-T | 42.7 | 39.3 | 48M | 267G |
| ConvNeXt-T | 44.2 | 40.1 | 48M | 262G |
| VMamba-T | 46.5 | 42.1 | 42M | 286G |
| **VSSD-T** | **46.9** | **42.6** | 44M | 265G |

VSSD-T outperforms Swin-T by +4.2 box AP and +3.3 mask AP.

### Ablation Study: Efficiency of NC-SSD

| Scanning Strategy | Training Speed Gain |
|-------------------|:-------------------:|
| NC-SSD vs. vanilla SSD | ~20% ↑ |
| NC-SSD vs. Bi-SSD (bidirectional scan) | ~50% ↑ |

### Key Findings

1. NC-SSD surpasses multi-scan SSD variants (e.g., Bi-SSD) in accuracy while achieving faster training.
2. Effective Receptive Field (ERF) analysis demonstrates that VSSD maintains a global receptive field after training, whereas VMamba exhibits cross-shaped attenuation.
3. Visualization of the $\mathbf{m}$ vector confirms that NC-SSD adaptively focuses on foreground features.
4. Hybridizing self-attention exclusively in the final stage outperforms global uniform interleaving.

## Highlights & Insights

- **Elegant Theoretical Derivation**: Starting from the linear recurrence form of SSD, a non-causal formulation is naturally derived through the concise transformation of "retaining relative weights while discarding absolute magnitudes."
- **Solving Two Challenges Simultaneously**: NC-SSD addresses both the causality constraint and the disruption of structural relationships in a unified manner, with a formulation closely related to linear attention.
- **Efficiency and Accuracy**: Global information is captured without multi-scan paths, yielding significant training speed improvements alongside competitive accuracy.

## Limitations & Future Work

- NC-SSD is equivalent to a special form of linear attention, potentially limiting its expressive power compared to standard attention.
- Standard self-attention is still required in the final stage to compensate, preventing a fully linear-complexity architecture.
- Scalability to ultra-high-resolution scenarios (e.g., remote sensing) remains to be validated.

## Related Work & Insights

- **SSM-based vision models**: ViM, VMamba, LocalVMamba, and others apply Mamba to visual tasks via multi-scan paths.
- **Linear attention**: NC-SSD has theoretical connections to linear attention methods such as MLLA.
- **Hierarchical visual backbones**: Classic design paradigms including Swin and ConvNeXt.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The derivation from SSD to NC-SSD is concise and compelling, with prominent theoretical contributions.
- **Technical Depth**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and architectural designs are well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage across classification, detection, and segmentation, with intuitive ERF visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — The framing of two challenges and their resolution follows a clear and logical structure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DefMamba: Deformable Visual State Space Model](../../CVPR2025/segmentation/defmamba_deformable_visual_state_space_model.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](../../CVPR2025/segmentation/exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2025\] GroupMamba: Efficient Group-Based Visual State Space Model](../../CVPR2025/segmentation/groupmamba_efficient_group-based_visual_state_space_model.md)
- [\[CVPR 2025\] MambaOut: Do We Really Need Mamba for Vision?](../../CVPR2025/segmentation/mambaout_do_we_really_need_mamba_for_vision.md)

</div>

<!-- RELATED:END -->

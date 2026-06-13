---
title: >-
  [Paper Note] AnyUp: Universal Feature Upsampling
description: >-
  AnyUp proposes the first **encoder-agnostic** learnable feature upsampling method. Through feature-agnostic convolutional layers and window attention mechanisms…
tags:

date: 2026-05-08
content_hash: 71fa384cc087ff44
---

# AnyUp: Universal Feature Upsampling

- **Conference**: ICLR 2026
- **arXiv**: [2510.12764](https://arxiv.org/abs/2510.12764)
- **Code**: [GitHub](https://github.com/wimmerth/anyup)
- **Area**: Computer Vision / Feature Upsampling
- **Keywords**: feature upsampling, encoder-agnostic, DINO, CLIP, attention, universal

## TL;DR

AnyUp proposes the first **encoder-agnostic** learnable feature upsampling method. Through feature-agnostic convolutional layers and window attention mechanisms, it requires only a single training pass to perform high-quality upsampling of arbitrary visual features across arbitrary resolutions, achieving state-of-the-art performance on semantic segmentation, depth estimation, and related tasks.

## Background & Motivation

**Background**: Feature maps produced by pretrained visual encoders (DINO, CLIP, SigLIP, etc.) are resolution-constrained by the number of Transformer tokens, typically $h \times w \ll H \times W$. Recent methods such as FeatUp, LoftUp, and JAFAR propose learnable feature upsampling to obtain high-resolution features.

**Limitations of Prior Work**: Existing learnable upsamplers **do not generalize across encoders** — an upsampler trained on DINOv2 cannot be directly applied to CLIP or SigLIP, and retraining is required whenever the encoder changes. For large-scale vision models (e.g., DINOv2-G), retraining is computationally prohibitive or entirely infeasible.

**Key Challenge**: The layers in an upsampling network that process low-resolution features are coupled to the specific dimensionality and distribution of a particular encoder, preventing transfer to new feature types at inference time.

**Goal**: Design a **train-once, use-anywhere** feature upsampler capable of upsampling features of arbitrary dimensionality from arbitrary encoders across arbitrary resolutions.

**Key Insight**: The fundamental bottleneck in existing attention-based upsamplers lies in the dimensional coupling of feature processing layers. If a processing layer can be designed to be invariant to the number of input channels, encoder-agnostic upsampling becomes feasible.

**Core Idea**: Design **feature-agnostic convolutional layers** — each input channel is independently convolved with a set of learned filter bases, normalized via softmax, and then averaged across all channels, yielding an output whose dimensionality is independent of the number of input channels.

## Method

### Overall Architecture

AnyUp builds upon an attention-based upsampling architecture (inherited from JAFAR). The pipeline proceeds as follows: given a high-resolution image $I_{hr}$ and low-resolution features $p = e(I_{hr})$, the feature-agnostic layer processes $p$, which is then fed together with image features into a window attention module to produce high-resolution features $q \in \mathbb{R}^{H \times W \times c}$. During training, randomly cropped image patches serve as reference supervision, avoiding the need for expensive full-image high-resolution features.

### Key Designs

#### 1. Feature-Agnostic Convolutional Layer

**Function**: Maps input features of arbitrary dimensionality $N$ to canonical features of fixed dimensionality $M$, enabling encoder-agnostic processing.

**Mechanism**: A set of $M$ filter bases $\{\psi_j \in \mathbb{R}^{k \times k}\}_{j=1}^M$ is learned. Each input channel $p_i$ is independently convolved with each basis, followed by softmax normalization and averaging across all channels:

$$f_j = \frac{1}{N} \sum_{i=1}^{N} \frac{\exp(p_i * \psi_j)}{\sum_{j'=1}^{M} \exp(p_i * \psi_{j'})}$$

The resulting $M$-dimensional output is **entirely independent** of the input dimensionality $N$, allowing the same model to process DINOv2's 384-dimensional features, CLIP's 768-dimensional features, and outputs from any other encoder.

**Design Motivation**: The primary role of the attention upsampler is to capture local structural variations in the input feature map (boundaries, textures, etc.) rather than to reconstruct specific feature values (which are conveyed directly through attention values). Channel-wise independent convolution followed by cross-channel averaging is specifically designed to capture **only** structural information.

#### 2. Local Window Attention

**Function**: Restricts global attention to a local window around each query point, simplifying the upsampling problem and improving efficiency.

**Mechanism**: Analysis of JAFAR's global attention patterns reveals anomalous cases in which pixel queries attend to entirely irrelevant distant regions. Restricting attention to local windows: (a) ensures that high-resolution features are formed as linear combinations of nearby coarse features, simplifying the optimization objective; and (b) reduces computational cost.

**Design Motivation**: Feature upsampling is inherently a **local** operation — the high-resolution feature at a given pixel should be determined primarily by the coarse features of nearby patches. Long-range attention in global schemes introduces noise rather than useful information.

#### 3. Crop-Based Training Strategy

**Function**: Uses randomly cropped local image patches as reference signals, replacing the expensive computation of full-image high-resolution features.

**Mechanism**: Given a high-resolution image $I$, a local crop $I'$ is sampled randomly. Features $p = e(I)$ and $\hat{q} = e(I')$ are extracted independently. After upsampling $p$, the loss is computed only in the region corresponding to $I'$ against $\hat{q}$. This strategy is more effective than JAFAR's low-resolution full-image training and more lightweight than LoftUp's segmentation-mask-based training.

### Loss & Training

The primary loss is a cosine-MSE composite with consistency regularization:

$$L_{\text{cos-mse}}(q', \hat{q}) = 1 - \cos(q', \hat{q}) + L^2(q', \hat{q})$$

This is supplemented by a self-consistency regularizer (for robustness) and an input-consistency regularizer (applying $L_{\text{cos-mse}}$ between the input features and the downsampled output features, preventing feature distribution drift).

## Key Experimental Results

### Main Results: Semantic Segmentation Linear Probing (DINOv2 ViT-S, 448×448 → 14× upsampling)

| Method | COCO mIoU↑ | COCO Acc↑ | PASCAL mIoU↑ | ADE20k mIoU↑ |
|--------|-----------|----------|-------------|-------------|
| Bilinear | 59.48 | 79.32 | 81.43 | 40.54 |
| FeatUp | — | — | 83.37 | — |
| JAFAR | 61.82 | 81.07 | 84.36 | — |
| LoftUp | — | — | — | 42.02 |
| **AnyUp** | **62.16** | **81.37** | **—** | **42.43** |

> AnyUp achieves state-of-the-art performance on most benchmarks, and **requires only a single training run** to generalize across all encoders — competing methods require **separate training for each encoder**.

### Ablation Study: Impact of Design Choices (COCO Semantic Segmentation mIoU)

| Configuration | mIoU↑ |
|---------------|------|
| Global attention (JAFAR-style) | 61.82 |
| + Feature-agnostic layer | 61.95 |
| + Window attention | 62.01 |
| + Crop training + consistency regularization | **62.16** |

**Depth/Normal Estimation** (NYUv2):

| Method | Normal RMSE↓ | Depth RMSE (abs)↓ | Depth δ₁↑ |
|--------|-------------|------------------|----------|
| Bilinear | 32.70 | 0.4925 | 0.8081 |
| LoftUp | 33.94 | — | 0.9166 |
| **AnyUp** | **31.17** | **0.4755** | **0.8216** |

### Key Findings

1. **Encoder Generalization**: AnyUp, trained on DINOv2, directly transfers to CLIP, SigLIP, and MAE outputs while maintaining high-quality upsampling; FeatUp, JAFAR, and LoftUp all require retraining.
2. **Feature Space Preservation**: The input-consistency regularizer effectively prevents feature distribution drift induced by upsampling (PCA visualizations confirm superior semantic consistency).
3. **Local Window Attention Outperforms Global**: Eliminating anomalous long-range attention patterns improves both quality and efficiency.

## Highlights & Insights

- The "train once, generalize to all encoders" paradigm offers substantial practical value by lowering the barrier to adopting learnable feature upsampling.
- The feature-agnostic layer is an elegant design: channel-wise independent convolution followed by softmax normalization and averaging achieves dimension-invariance through a remarkably simple mechanism.
- The crop-based training strategy strikes a favorable balance between quality and efficiency, requiring neither high-resolution reference features nor an auxiliary segmentation model.
- This work is the first in the feature upsampling domain to achieve full combinatorial generalization across "any encoder × any resolution × any task."

## Limitations & Future Work

- The averaging operation in the feature-agnostic layer discards inter-channel correlation information, which may limit performance on tasks requiring precise channel interactions.
- The window size in local attention requires tuning; overly small windows may discard necessary non-local information.
- Evaluation is conducted primarily on ViT-based encoders; applicability to CNN-based encoders warrants further investigation.
- Training still requires ImageNet-scale data and features from multiple encoders, entailing non-trivial computational cost.

## Related Work & Insights

- **FeatUp** (Fu et al., 2024): Trains upsamplers via multi-view equivariance, but is encoder-specific and supports only fixed upsampling ratios.
- **JAFAR** (Couairon et al., 2025): Attention-based upsampling supporting arbitrary resolutions; AnyUp extends this framework with encoder-agnostic capability.
- **LoftUp** (Huang et al., 2025): Stacked attention with segmentation-mask-based training achieves high quality but incurs significant training cost and requires an auxiliary segmentation model.
- **Insights**: The "channel-independent processing + aggregation" paradigm of the feature-agnostic layer may generalize to other multimodal feature alignment scenarios.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Encoder-agnostic upsampling represents a new direction; the feature-agnostic layer is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers segmentation, depth, normals, multiple resolutions, and multiple encoders with thorough ablations.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses the engineering pain point of retraining upon encoder change; open-sourced weights enable plug-and-play use.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear figures and tables; the comparative method taxonomy table is particularly informative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Distributionally Robust Feature Selection](../../NeurIPS2025/others/distributionally_robust_feature_selection.md)
- [\[NeurIPS 2025\] Manipulating Feature Visualizations with Gradient Slingshots](../../NeurIPS2025/others/manipulating_feature_visualizations_with_gradient_slingshots.md)
- [\[ICML 2026\] Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization](../../ICML2026/others/over-alignment_vs_over-fitting_the_role_of_feature_learning_strength_in_generali.md)
- [\[NeurIPS 2025\] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model](../../NeurIPS2025/others/neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)
- [\[ICLR 2026\] A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization](a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)

</div>

<!-- RELATED:END -->

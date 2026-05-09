---
title: >-
  [Paper Note] Axis-Aligned Document Dewarping
description: >-
  [AAAI 2026][LLM Evaluation][document dewarping] This paper proposes to exploit the inherent axis-aligned geometric property of planar documents, systematically incorporating axis-alignment constraints across training, inference, and evaluation stages, achieving state-of-the-art document rectification performance and introducing a new evaluation metric, AAD.
tags:
  - AAAI 2026
  - LLM Evaluation
  - document dewarping
  - geometric constraint
  - image rectification
date: 2026-05-08
content_hash: e5a6015c2fbf4836
---

# Axis-Aligned Document Dewarping

**Conference**: AAAI 2026
**arXiv**: [2507.15000](https://arxiv.org/abs/2507.15000)
**Code**: [https://github.com/chaoyunwang/AADD](https://github.com/chaoyunwang/AADD)
**Area**: LLM Evaluation
**Keywords**: document dewarping, geometric constraint, image rectification

## TL;DR

This paper proposes to exploit the inherent axis-aligned geometric property of planar documents, systematically incorporating axis-alignment constraints across training, inference, and evaluation stages, achieving state-of-the-art document rectification performance and introducing a new evaluation metric, AAD.

## Background & Motivation

Document dewarping aims to restore distorted document images captured by mobile phones or cameras into flat, rectangular documents, serving as a critical preprocessing step for downstream tasks such as OCR. Existing methods suffer from the following limitations:

**Traditional methods rely on low-level feature detection**: Early approaches model deformation through low-level features such as text lines and document boundaries, but low-level feature detection is unstable on severely distorted images and generalizes poorly.

**Deep learning methods depend on strong supervision signals**: Mainstream methods train networks using additional supervision signals such as control points, segmentation masks, and text line layouts. However, these signals either lack geometric semantics (e.g., control points) or are difficult to extract and generalize poorly (e.g., text lines).

**The intrinsic geometric properties of documents are overlooked**: A fundamental characteristic of planar documents is that, after rectification, feature lines (text lines, table lines, etc.) should be aligned with the coordinate axes. This geometric prior has not been sufficiently exploited in prior work.

The core insight of this paper is intuitive: **a well-rectified document is one whose feature lines are aligned with the horizontal and vertical axes**. The authors term this the "axis-aligned property" and systematically leverage this single principle across the training, inference, and evaluation stages of the deep learning pipeline.

## Method

### Overall Architecture

This paper adopts the fully convolutional network architecture of UVDoc as its backbone, where the network simultaneously predicts the 3D mesh and 2D unwarped mesh of the document (dual-task framework). The core innovations consist of three complementary modules organized around the axis-aligned property:

- **Training stage**: Axis-Aligned Geometric Constraint Loss
- **Inference stage**: Axis Alignment Preprocessing
- **Evaluation stage**: A new metric, AAD (Axis-Aligned Distortion)

### Key Design 1: Axis-Aligned Geometric Constraint (Training)

This is the central contribution of the method. The intuition is that in UV space, an ideal planar document corresponds to a uniform grid where every row shares the same $v$-coordinate and every column shares the same $u$-coordinate. Axis-alignment error can therefore be measured by computing the variance of row/column coordinates in UV space.

The detailed procedure is as follows:

1. The network predicts the 2D unwarped grid $P = \{p_{i,j}\}$, where each point $p_{i,j} = (x_{i,j}, y_{i,j})$.
2. An interpolation function maps the predicted grid from image space to UV space: $Q = \{q_{i,j}\}$, where $q_{i,j} = f(p_{i,j}) = (u_{i,j}, v_{i,j})$.
3. Alignment errors in both directions are computed in UV space:
    - **Horizontal error**: the sum of variances of $v$ values within each row $\mathcal{L}_{hor} = \sum_{j=1}^{h} \text{Var}(\{v_{1,j}, \ldots, v_{w,j}\})$
    - **Vertical error**: the sum of variances of $u$ values within each column $\mathcal{L}_{ver} = \sum_{i=1}^{w} \text{Var}(\{u_{i,1}, \ldots, u_{i,h}\})$
4. The axis-alignment constraint loss is: $\mathcal{L}_{AL} = \mathcal{L}_{hor} + \mathcal{L}_{ver}$

The elegance of this design lies in avoiding direct computation of alignment error in image space (which is difficult given that predictions reside there), and instead mapping to UV space where the ground truth constitutes a uniform grid, making measurement straightforward.

### Key Design 2: Axis Alignment Preprocessing (Inference)

At inference time, prior methods employ external segmentation models to crop the document region and reduce rectification difficulty. This paper proposes a self-contained preprocessing strategy:

1. A single forward pass is performed on the input image to obtain a coarse 2D unwarped grid.
2. A minimum-area rotated bounding rectangle is computed from the positional information of this grid.
3. The image is rotated to align the principal document axis with the coordinate axes, and the target region is cropped.
4. The preprocessed image is fed into the network again to obtain a refined rectification result.

This process can be applied iteratively (once for DocUNet benchmark, twice for DIR300). Compared to approaches that rely on external models, this strategy is more efficient and directly leverages the network's own predictions.

### Key Design 3: AAD Evaluation Metric

Existing evaluation metrics (e.g., MS-SSIM, LD, AD) cannot effectively capture the axis-alignment quality of document feature lines. The core idea of the AAD metric is to measure the axis-alignment of feature lines in the rectified result using gradient-weighted optical flow deviation.

The computation procedure is as follows:
1. The SIFT-flow algorithm computes the optical flow field $(v_x, v_y)$ from the ground-truth image to the rectified result.
2. Directional gradients of the ground-truth image are extracted using the Sobel operator, normalized, and used as weights.
3. Gradient-weighted mean deviations of optical flow are computed for each row and column.
4. Row and column deviations are combined into per-pixel deviations, and the global mean yields the AAD value.

The AAD metric offers the following advantages: its heatmap carries clear geometric semantics (bright regions directly correspond to distorted feature lines), aligns with human visual perception, and provides better discriminative power when performance gaps between methods are small.

### Loss & Training

The total loss function comprises four components:

$$\mathcal{L}_{all} = \alpha \mathcal{L}_{2D} + \beta \mathcal{L}_{3D} + \gamma \mathcal{L}_{AL} + \lambda \mathcal{L}_{SSIM}$$

- $\mathcal{L}_{2D}$: L1 loss on the 2D mesh
- $\mathcal{L}_{3D}$: L1 loss on the 3D mesh
- $\mathcal{L}_{AL}$: axis-aligned geometric constraint loss
- $\mathcal{L}_{SSIM}$: structural similarity loss (to avoid optimization instability caused by pixel-level MSE)
- Hyperparameters: $\alpha = \beta = 1, \gamma = 0.2, \lambda = 0.05$

## Key Experimental Results

### Table 1: DocUNet Benchmark Results

| Method | MS-SSIM↑ | LD↓ | AD↓ | AAD↓ | ED↓ | CER↓ |
|--------|----------|------|------|------|------|------|
| DewarpNet | 0.474 | 8.362 | 0.398 | 0.164 | 824.5 | 0.225 |
| DocTr | 0.509 | 7.773 | 0.369 | 0.151 | 708.6 | 0.185 |
| LADoc | 0.525 | 6.706 | 0.300 | 0.121 | 689.8 | 0.180 |
| UVDoc | 0.545 | 6.827 | 0.316 | 0.125 | 754.2 | 0.193 |
| **Ours (Full)** | **0.543** | **6.249** | **0.278** | **0.099** | **603.1** | **0.150** |
| Gain | - | 6.8% | 7.3% | **18.2%** | 12.4% | 14.8% |

### Table 2: DIR300 Benchmark Results

| Method | MS-SSIM↑ | LD↓ | AD↓ | AAD↓ | ED↓ | CER↓ |
|--------|----------|------|------|------|------|------|
| DewarpNet | 0.492 | 13.944 | 0.332 | 0.147 | 1076.8 | 0.336 |
| DocTr | 0.616 | 7.189 | 0.255 | 0.107 | 698.4 | 0.211 |
| LADoc | 0.652 | 5.702 | 0.195 | 0.087 | 495.4 | 0.173 |
| UVDoc | 0.621 | 7.730 | 0.219 | 0.101 | 614.0 | 0.237 |
| **Ours (Full)** | **0.702** | **4.261** | **0.131** | **0.057** | **405.8** | **0.132** |
| Gain | 7.7% | 25.3% | 32.8% | **34.5%** | 9.3% | 23.7% |

Ablation studies show that on DocUNet (where the document occupies a large proportion of the image), the axis-alignment constraint contributes more; on DIR300 (where the document occupies a smaller proportion), the preprocessing strategy contributes more. Their combination achieves optimal performance, demonstrating complementarity.

## Highlights & Insights

1. **Principle-driven methodology**: The entire paper is organized around a single, concise geometric insight — "good rectification equals axis alignment" — and this principle is consistently applied across training, inference, and evaluation, resulting in a clear and elegant framework.
2. **Elegant measurement in UV space**: Directly measuring axis-alignment error in image space is difficult; transforming to UV space reduces the problem to a simple variance computation, illustrating a broadly applicable strategy of leveraging parameterized spaces to simplify geometric problems.
3. **Self-contained inference preprocessing**: No additional segmentation or detection models are required; the method directly uses the network's own coarse predictions for document localization and rotation correction, making it simple and efficient.
4. **Practical utility of the AAD metric**: The heatmap of the AD metric is uninterpretable and its numerical values can contradict human perception; AAD addresses this limitation by incorporating gradient weighting and axis-alignment semantics, and provides better discriminative power as the performance gap between state-of-the-art methods narrows.
5. **Lightweight improvement**: The core contributions do not involve architectural changes; significant gains are achieved solely through loss function design and inference pipeline improvement, making the approach plug-and-play and generalizable to other document rectification networks.

## Limitations & Future Work

1. **Iterative inference increases latency**: The axis-alignment preprocessing requires at least two forward passes (three for DIR300); the increased inference time is not discussed in the paper.
2. **Applicability to non-rectangular documents**: The axis-alignment assumption is premised on standard rectangular documents and may not apply to irregularly shaped documents (e.g., folded or torn documents).
3. **Limitations of SIFT-flow in AAD**: The AAD metric relies on SIFT-flow for optical flow estimation, which may be inaccurate under severe distortion, potentially compromising the reliability of the metric.
4. **Training on synthetic data only**: The model is trained on the Doc3D and UVDoc synthetic datasets; although evaluated on real-world benchmarks, the synthetic-to-real domain gap may still limit generalization.
5. **Marginal MS-SSIM drop on DocUNet**: The full model achieves MS-SSIM of 0.543 on DocUNet, slightly below UVDoc (0.545) and the variant with only the AL loss (0.549), suggesting that preprocessing may introduce slight side effects in large-target scenarios.

## Related Work & Insights

- **UVDoc (Verhoeven et al., ECCV 2023)**: The backbone architecture of this work, providing pseudo-photorealistic training data and the dual-task prediction framework. This paper adds geometric constraints on top of it.
- **LADoc (Li et al., 2023)**: A layout-aware method that leverages document layout information to assist rectification; it was the strongest baseline on DIR300.
- **DocGeoNet (Feng et al., 2022)**: Employs geometric representation learning for rectification and introduces the DIR300 benchmark and the AD metric.
- **PaperEdge (Ma et al., 2022)**: Introduces the AD metric and external segmentation-based preprocessing; the self-contained preprocessing proposed in this paper can be seen as an improvement over this approach.
- **Mesh regularization (Jiang et al., CVPR 2022)**: Combines deep learning-based text line detection with geometric constraint optimization, but incurs long optimization time.

**Insights**: This paper demonstrates the substantial value of exploiting intrinsic geometric priors within a domain — significant improvements are achievable without architectural changes, solely through proper inductive biases (loss functions) and simple inference strategies. This principle-driven, rather than architecture-driven, research paradigm is particularly important as a field matures and architectural gains diminish, and is worth extending to other image rectification tasks such as illumination correction and perspective correction.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The core insight is concise and compelling; the systematic design that applies a single geometric principle across three stages is novel.
- **Experimental Thoroughness** ⭐⭐⭐⭐: The method comprehensively surpasses state-of-the-art on two mainstream benchmarks, ablation studies are thorough, and the comparative analysis of the AAD metric is convincing.
- **Writing Quality** ⭐⭐⭐⭐⭐: The paper is clearly structured, the logical chain from motivation to method is coherent, and the illustrations are intuitive.
- **Value** ⭐⭐⭐: The method is practical but the field is relatively niche; the AAD metric may have lasting impact if adopted by the community.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] HybriDLA: Hybrid Generation for Document Layout Analysis](hybridla_hybrid_generation_for_document_layout_analysis.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/llm_evaluation/document_summarization_with_conformal_importance_guarantees.md)
- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](../../ICCV2025/llm_evaluation/forcennet_foreground-centric_network_for_document_image_rectification.md)
- [\[AAAI 2026\] MindVote: When AI Meets the Wild West of Social Media Opinion](mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)
- [\[AAAI 2026\] Towards a Common Framework for Autoformalization](towards_a_common_framework_for_autoformalization.md)

<!-- RELATED:END -->

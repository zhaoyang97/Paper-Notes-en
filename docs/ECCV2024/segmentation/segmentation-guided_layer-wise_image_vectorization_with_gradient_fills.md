---
title: >-
  [Paper Note] Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills
description: >-
  [ECCV 2024][Segmentation][Image Vectorization] Proposes a segmentation-guided vectorization framework that guides the initialization and optimization of Bézier paths through a gradient-aware segmentation subroutine. It supports radial gradient fills in a layer-wise vectorization method with layered topology for the first time, achieving higher visual quality with fewer paths.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Image Vectorization"
  - "Segmentation Guidance"
  - "Gradient Fill"
  - "Differentiable Rendering"
  - "Layered Topology"
date: 2026-05-08
content_hash: 3fc27b6585c5133b
---

# Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills

**Conference**: ECCV 2024  
**arXiv**: [2408.15741](https://arxiv.org/abs/2408.15741)  
**Code**: None (implemented based on DiffVG and PyTorch)  
**Area**: Segmentation (Image Vectorization)  
**Keywords**: Image Vectorization, Segmentation Guidance, Gradient Fill, Differentiable Rendering, Layered Topology

## TL;DR

Proposes a segmentation-guided vectorization framework that guides the initialization and optimization of Bézier paths through a gradient-aware segmentation subroutine. It supports radial gradient fills in a layer-wise vectorization method with layered topology for the first time, achieving higher visual quality with fewer paths.

## Background & Motivation

Vector graphics (SVGs) are highly popular in digital design (arbitrarily scalable, easy to edit). Rasterization-to-vectorization is an important way to generate vector graphics.

Taxonomy and limitations of existing vectorization methods:

| Category | Representative Methods | Advantages | Limitations |
|------|---------|------|------|
| Mesh Methods | Triangular/Rectangular Meshes | Near photo-realistic | Complex primitives, loses hierarchical structure |
| Curve Methods | Diffusion curves | High fidelity | Non-intuitive, hard to edit |
| Learning Methods | SketchRNN, Im2Vec | Preserve topology | Depend on training data, poor generalization |
| **LIVE** | DiffVG Optimization | Layer-wise topology, model-free | **Does not support gradient fills** |

**LIVE** is the most closely related prior work—a layer-wise vectorization framework built on the differentiable renderer DiffVG, but it only supports solid color fills. For images with gradients, LIVE must add a massive number of redundant paths to approximate the gradients. Furthermore, **simply replacing solid color parameters with gradient parameters is non-trivial**—it requires an effective way to determine which pixels contribute to the gradient fill of a path.

## Method

### Overall Architecture

Progressive vectorization: One or more Bézier paths are added and optimized step-by-step in each epoch.

At each epoch $i$:
1. Compute the difference between the input image and the current output $I_{i-1}$
2. Use gradient-aware segmentation to determine $n_i$ regions
3. Initialize new paths (Bézier curves + radial gradient)
4. Optimize geometric parameters and gradient parameters of all paths

### Key Designs

**Gradient-aware Segmentation**:

Core Insight: Colors change smoothly within the same gradient fill, while sharp color changes occur at boundaries. Therefore, the second-order spatial gradient (Laplacian filter) is used to detect gradient boundaries:

$$S_0 = \text{correlate}((I - \hat{I})\mathbf{1}_{\|\hat{I}-I\|_2 > \epsilon}, L)$$

where $L$ is the discrete Laplacian filter $\begin{bmatrix}1&1&1\\1&-8&1\\1&1&1\end{bmatrix}$.

Subsequent steps:
- Summing the absolute values of the RGB channels $\rightarrow$ grayscale image
- Otsu adaptive threshold binarization
- Morphological closing + watershed algorithm $\rightarrow$ final segmentation

**Key Advantage**: As paths are progressively added, the overall reconstruction error decreases, causing the Otsu adaptive threshold to drop accordingly. Consequently, large gradient regions are fitted first, while tiny detail regions (such as cheek blushes) are automatically segmented and resolved later.

**Segmentation-Guided Initialization**:

Select the segmentation region with the highest accumulated squared error to initialize new paths:

$$w_i = \sum_{p \in \tilde{S}[i]} \|I_p - \hat{I}_p\|^2$$

Prioritize larger regions (encouraging hierarchical ordering) and avoid already well-fitted regions.

Paths are initialized as circles consisting of four cubic Bézier curves, filled with radial gradients:
- Center: Centroid of the region
- Diameter: Geometric mean of the region's bounding box width and height (clipped to [0.2, 1.0])
- Two color stops (0% and 100%) initialized as the input image color at the centroid

### Loss & Training

**Segmentation-Guided Loss (SG Loss)**:

LIVE uses UDF (Unsigned Distance Field) weights to focus optimization on path boundary pixels. However, for gradients, correct boundary color does not guarantee correct interior color. This work extends the UDF weights to all unoccluded pixels covered by the path:

$$w_{\text{SG}}(i) = \begin{cases}\max(d_i', \alpha_s), & i \in F \\ d_i'(1-\alpha_s), & \text{otherwise}\end{cases}$$

where $F$ is the set of all "focused pixels" (the intersection of the path-covered region and the segmented region), and $\alpha_s = 0.6$ balances the UDF weights and the segmentation weights.

$$\mathcal{L}_{\text{SG}} = \frac{1}{3}\sum_{i=1}^{w \times h} w_{\text{SG}}(i)\sum_{c=1}^{3}(I_{c,i} - \hat{I}_{c,i})^2$$

**Xing Loss**: Prevents self-intersection of Bézier curves by penalizing cases where the angle between control vectors $\vec{AB}$ and $\vec{CD}$ exceeds 180°.

**Total Loss**:

$$\mathcal{L} = \mathcal{L}_{\text{SG}} + \lambda\mathcal{L}_{\text{Xing}}, \quad \lambda = 0.05$$

Optimizer: Adam, learning rate $10^{-2}$ (gradient parameters) and 1 (path control points).

## Key Experimental Results

### Main Results

User study results (user preference percentage):

| Dataset | Num. Paths | LIVE Preference | Our Preference |
|--------|--------|----------|---------|
| Noto Emoji | Overall | 40.4% | **59.6%** |
| Fluent Emoji | Overall | 34.7% | **65.3%** |
| Iconfont | Overall | 42.1% | **57.9%** |

The advantage is most pronounced on Fluent Emoji with rich gradients (65.3% vs. 34.7%), and becomes more significant with fewer paths.

### Ablation Study

PSNR comparison on Noto Emoji (higher is better):

| Configuration | 8 Paths | 16 Paths | 32 Paths | 64 Paths |
|------|-------|--------|--------|--------|
| LIVE (No gradient, no segmentation guidance) | Baseline | Baseline | Baseline | Baseline |
| + Gradient (No guidance) | Drop | Drop | Close | Close |
| + Gradient + Segmentation Guidance | **Highest** | **Highest** | **Highest** | Close |

The proposed method achieves significantly higher PSNR when fewer paths are used, and the performance of both methods converges as the number of paths becomes sufficient.

### Key Findings

1. **Huge advantage with fewer paths**: Only 8 paths are needed to vectorize the primary elements of emojis (eyes, mouth, etc.), whereas LIVE requires significantly more paths.
2. **Segmentation guidance is crucial**: Direct optimization of gradient parameters without segmentation guidance leads to performance degradation.
3. **Advantage of adaptive thresholding**: Otsu dynamic thresholding frees the framework from needing pre-assumed input types, avoiding hyperparameter tuning.
4. Across three datasets (Noto Emoji, Fluent Emoji, Iconfont), users generally prefer the proposed method.
5. The framework is model-free and does not depend on any training data.

## Highlights & Insights

1. **Gradient detection $\approx$ segmentation challenge**: Modeling "determining which pixels contribute to the gradient" as a segmentation task is a clever insight connecting two seemingly unrelated domains.
2. **Laplacian for gradient boundary detection**: Second-order derivatives are near-zero inside gradients but spike sharply at boundaries—a simple and effective physical intuition.
3. **Model-free design**: The entire pipeline is built upon classical image processing (Otsu, morphological operations, watershed) + differentiable rendering optimization, requiring no training data.
4. **Practicality of hierarchical decomposition**: The ordered paths added sequentially naturally form editable layer structures, supporting downstream editing operations like recoloring.

## Limitations & Future Work

1. Only radial gradients are supported; linear and conic gradients are not yet covered.
2. For highly complex real-world photos, a significant number of Bézier paths may be needed to achieve satisfactory results.
3. The segmentation method is based on classical algorithms, which may lack precision on certain complex inputs.
4. The optimization process is iterative, making it less efficient than learning-based methods with one-step forward inference.
5. Integrating deep learning segmentation methods (such as SAM) to replace classical segmentation algorithms remains to be explored.

## Related Work & Insights

- **LIVE**: The most direct predecessor; this work adds gradient support upon it.
- **DiffVG**: The differentiable renderer serving as the technical foundation for both this work and LIVE.
- **Im2Vec**: A learning-based vectorization method that supports DiffVG backpropagation but relies on training data.
- **Insight**: Introducing traditional image processing methods as "guidance signals" in a differentiable rendering optimization framework can effectively compensate for the shortcomings of pure optimization methods.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 3 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 4 |
| **Overall** | **3.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)
- [\[ECCV 2024\] VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement](visage_video_instance_segmentation_with_appearance-guided_enhancement.md)
- [\[CVPR 2026\] FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching](../../CVPR2026/segmentation/flowdis_language-guided_dichotomous_image_segmentation_with_flow_matching.md)
- [\[ICCV 2025\] LayerAnimate: Layer-level Control for Animation](../../ICCV2025/segmentation/layeranimate_layer-level_control_for_animation.md)
- [\[ECCV 2024\] ReMamber: Referring Image Segmentation with Mamba Twister](remamber_referring_image_segmentation_with_mamba_twister.md)

</div>

<!-- RELATED:END -->

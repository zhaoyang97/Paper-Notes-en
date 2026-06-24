---
title: >-
  [Paper Note] Layered Image Vectorization via Semantic Simplification
description: >-
  [CVPR 2025][Model Compression][Image Vectorization] This paper proposes a progressive image vectorization method that utilizes the feature-average effect of Score Distillation Sampling (SDS) to generate a sequence of step-by-step simplified images. This sequence guides the layered reconstruction of vectors from macro semantic structures to fine details, outperforming existing methods significantly in visual fidelity, semantic alignment, and compact layered representation.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Image Vectorization"
  - "Semantic Simplification"
  - "SDS Distillation"
  - "Layered Representation"
  - "SVG Generation"
date: 2026-05-08
content_hash: f88c95ad98b3b40c
---

# Layered Image Vectorization via Semantic Simplification

**Conference**: CVPR 2025  
**arXiv**: [2406.05404](https://arxiv.org/abs/2406.05404)  
**Code**: None (implemented based on PyTorch + DiffVG)  
**Area**: Image Generation / Model Compression  
**Keywords**: Image Vectorization, Semantic Simplification, SDS Distillation, Layered Representation, SVG Generation

## TL;DR
This paper proposes a progressive image vectorization method that utilizes the feature-average effect of Score Distillation Sampling (SDS) to generate a sequence of step-by-step simplified images. This sequence guides the layered reconstruction of vectors from macro semantic structures to fine details, outperforming existing methods significantly in visual fidelity, semantic alignment, and compact layered representation.

## Background & Motivation

1. **Background**: Image vectorization (converting raster images to vector formats like SVG) is a classic problem in computer graphics. Recently, differentiable rendering-based methods (such as LIVE and DiffVG) have achieved promising results by iteratively optimizing Bézier curves to approximate the target image.

2. **Limitations of Prior Work**: Existing methods utilize a single target image as the optimization objective, directly adding vector primitives to regions with the largest pixel discrepancies. This leads to two issues: (a) the generated vector primitives are overly complex and lack semantic structure, making them difficult to edit and manage; (b) they fail to capture implicit semantic objects obscured by occlusions, texture variations, and other factors (such as a complete human body contour interrupted by fine details).

3. **Key Challenge**: Vectorization must simultaneously achieve **visual fidelity** and **structural manageability**, whereas methods starting directly from details fail to establish meaningful semantic hierarchies.

4. **Goal**: How to generate a compact, layered vector representation organized by semantic hierarchy—constructed step-by-step from global contours to local details?

5. **Key Insight**: The authors discovered that the "feature-average effect" in SDS can be leveraged for image simplification. When the conditional noise in SDS is eliminated, iterative optimization causes the image to progressively lose details while retaining macro-structures. This provides a natural "fine-to-coarse" simplification sequence.

6. **Core Idea**: To utilize the feature-average effect of SDS to generate progressively simplified image sequences as intermediate optimization targets, guiding the layered reconstruction of vectors from macro semantics to fine details.

## Method

### Overall Architecture
Taking the target image as input, the pipeline consists of three steps: (1) **Progressive Image Simplification**: By modifying the CFG of SDS (setting the conditional text to empty or the CFG scale to 0), a simplified image is generated every 20 iterations using the feature-average effect, forming a sequence from the original image to coarse contours (default 5 levels); (2) **Structural Construction (Stage I)**: Semantic segmentation is performed on each image in the simplified sequence to extract masks, which are ordered from back to front based on overlapping relationships. Closed Bézier curves are initialized for each mask and optimized via structural loss; (3) **Visual Refinement (Stage II)**: Colors are fitted and frozen for the structural vectors, and refinement vectors are added in regions with high visual discrepancies to optimize the visual fidelity loss.

### Key Designs

1. **SDS-based Progressive Image Simplification**:

    - **Function**: Generate a sequence of simplified images from fine to coarse to serve as intermediate targets for vectorization.
    - **Mechanism**: In the SDS update direction $(\epsilon_\phi(\mathbf{x}_t, t, y) - \epsilon)$, the pre-trained DDPM is sensitive to inputs, and the predicted noise exhibits feature inconsistency, causing pixels to be updated in inconsistent directions. This generates a "feature-average" effect (blurring details while preserving macro-structures). To control the simplification degree and prevent severe shape distortions, the conditional text in CFG is set to an empty string " " (equivalent to eliminating the guidance of conditional noise). Thus, SDS relies solely on unconditional noise prediction to update the image. A simplified result is saved every $N=20$ steps, yielding a 5-level simplified sequence. Compared with traditional simplification methods like bilateral/Gaussian filtering or superpixels, the SDS-based method intelligently removes non-structural elements (e.g., trees in front of a house) while maintaining clear semantic boundaries (e.g., the circular contour of a ladybug).
    - **Design Motivation**: Blurred boundaries generated by traditional simplification are unsuitable for vectorization, whereas SDS-based simplification naturally yields semantic hierarchies with smooth boundaries, which are highly compatible with vector graphics.

2. **Layered Structure Construction (with Layered Mask Ordering and Structural Loss)**:

    - **Function**: Extract semantic masks from the simplified sequence and optimize them into a layered vector structure.
    - **Mechanism**: Semantic segmentation (using SAM) is applied to each level of the simplified images, and masks are added sequentially from the coarsest to the finest. Each mask is placed into layers ordered from back to front, where masks in the same layer do not overlap. Mask boundaries are simplified using the Douglas-Peucker algorithm and initialized as closed Bézier curves. During optimization, a **layered structure loss** is employed: $\mathcal{L}_{\text{structure}} = w_1 \mathcal{L}_{\text{mse}} + w_2 \mathcal{L}_{\text{overlap}}$. The MSE term measures the discrepancy between each layer's mask image and the rendered vector graphic, while the overlap term penalizes the overlap of vectors in the same layer (applying a ReLU penalty to pixels in overlapping regions whose transparency exceeds a threshold). During optimization, each mask-vector pair is assigned the same random color, focusing solely on shape alignment.
    - **Design Motivation**: This method can discover implicit semantic structures that cannot be captured by segmenting a single image—such as the simplified "entire robot" or a "face without holes," which are interrupted by textures/occlusions in the original image.

3. **Visual Refinement (Color Fitting + Visual-wise Vector Optimization)**:

    - **Function**: Add refinement vectors while keeping the structural vectors frozen to improve visual fidelity.
    - **Mechanism**: First, colors are fitted for the structural vectors by taking the dominant color of the visible pixels covered by the vector, or through MSE minimization fitting. Then, the structural vectors are frozen, and the pixel discrepancy between the rendered image and the target image is calculated. Refinement vectors are initialized in the Top-K largest discrepancy connected regions (similar to the strategy in LIVE) to optimize the visual fidelity loss: $\mathcal{L}_{\text{fidelity}} = \|I_{\text{target}} - I_{\text{vector}}\|_2^2$. During optimization, vector pruning is periodically performed (merging redundancies and deleting useless vectors).
    - **Design Motivation**: The separation into two stages ensures that structural integrity is not compromised by color/detail optimization, while the refinement stage compensates for the visual loss of structural vectors.

### Loss & Training
- Stage I structural loss: $\mathcal{L}_{\text{structure}} = \mathcal{L}_{\text{mse}} + 10^{-8} \cdot \mathcal{L}_{\text{overlap}}$
- Stage II visual fidelity loss: $\mathcal{L}_{\text{fidelity}} = \|I_{\text{target}} - I_{\text{vector}}\|_2^2$
- Adam optimizer, point coordinate learning rate 1.0, color learning rate 0.01
- 5 levels in the simplified sequence, with an interval of 20 SDS iterations per level

## Key Experimental Results

### Main Results

| Method | MSE ↓ | LPIPS ↓| VeC (%) ↑ | Description |
|------|-------|---------|-----------|------|
| DiffVG | - | - | 41.9 | Basic differentiable rendering |
| LIVE | - | - | 43.4 | Progressive addition |
| O&R | - | - | 39.9 | Optimization + pruning |
| SGLIVE | - | - | 65.9 | Gradient-aware segmentation |
| **Ours** | **Lowest** | **Lowest** | **73.8** | Semantic layering |

VeC (Vector Compactness) measures the proportion of vector primitives that are highly contained (>85% area overlap) inside semantic masks. The proposed method achieves a VeC of 73.8% with the smallest standard deviation (11.9), significantly outperforming all baselines. Across 100 test images, the MSE and LPIPS of this method are optimal under all vector count settings.

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Full model | Best structural + visual quality | Includes SDS simplified sequence guidance |
| w/o Simplified sequence | Lacks implicit semantic structures | e.g., "whole Captain America" and "grass" cannot be captured |
| Replacing SDS with Bilateral filter | Blurred boundaries, structural degradation | Ladybug's circular boundary destroyed |
| Replacing SDS with Gaussian filter | More blurred boundaries | Loss of semantic information |
| Replacing SDS with Superpixel | Over-fragmentation | Unable to recover macro structures |

### Key Findings
- The core advantage of SDS simplification is **semantic intelligence**: it can automatically remove non-structural occlusions (like trees in front of a house) to recover the complete occluded semantic objects (like the front wall of the house).
- CLIP semantic similarity shows that when the number of vectors is small (coarse stage), this method's semantic fidelity is vastly superior to other methods.
- The descriptive text generated by the Florence-2 model for the coarse vector layers aligns highly with the original image content, validating the semantic effectiveness of the macro structures.
- The layered vector representation greatly facilitates downstream editing (such as selecting upper-layer primitives for recoloring based on the underlying structure).

## Highlights & Insights
- **Clever exploitation of SDS "defects"**: The feature-average effect of SDS is typically considered a quality degradation issue (leading to over-smoothing). This work does the opposite, using it as an image simplification tool to turn a bug into a feature.
- **Two-stage separated optimization strategy** decouples shape and color, ensuring structural integrity while simplifying the optimization process. This concept can be transferred to other tasks requiring hierarchical generation.
- **Coarse-to-fine vectorization strategy** mirrors the human drawing process of sketching outlines before filling in details, producing representations that are more intuitive and suitable for human editing.
- **Proposal of the VeC metric** provides a new evaluation dimension for vectorization quality (semantic compactness).

## Limitations & Future Work
- SDS simplification relies on priors from pre-trained diffusion models: its effectiveness can be unstable on out-of-distribution images (such as paintings in specialized styles).
- The quality of the semantic segmentation model (SAM) directly affects the accuracy of mask hierarchical partitioning.
- The number of levels and intervals in the simplified sequence (5 levels, 20 steps) are manually set hyperparameters, which may not scale adaptively to images of varying complexity.
- When processing photorealistic images, the number of vectors remains relatively large (compared to clipart/emoji-style images).
- Richer vector primitives, such as gradient fills, have not yet been explored.

## Related Work & Insights
- **vs LIVE**: LIVE progressively adds vectors in regions with the largest pixel discrepancies, but it relies entirely on low-level pixel analysis and cannot perceive semantic structures. This work introduces semantic simplified sequences as intermediate guidance to build from macro semantics to fine details.
- **vs O&R (Optimize & Reduce)**: O&R initializes and prunes vectors via pixel clustering, which also lacks a semantic hierarchy. This work's layering strategy yields more compact and editable results.
- **vs SGLIVE**: SGLIVE introduces gradient-aware segmentation to improve vector layout, achieving a VeC of 65.9%. This work further improves it to 73.8% through SDS-simplification guidance, achieving noticeably higher semantic alignment quality.
- The concept of utilizing SDS for image simplification can be adapted for video vectorization to generate temporally consistent simplified sequences.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The insight of utilizing the SDS feature-average effect for image simplification is highly ingenious, opening up a new direction for vectorization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparisons on 100 images and ablations are comprehensive, and the CLIP semantic evaluation is novel; however, a large-scale user study is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent illustrations; the pipeline of simplification -> layering -> refinement is clearly and intuitively explained.
- Value: ⭐⭐⭐⭐ High practical value for the design community, as compact layered SVGs facilitate easy editing and recoloring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Understanding Multi-layered Transmission Matrices](understanding_multi-layered_transmission_matrices.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](../../NeurIPS2025/model_compression/one-step_diffusion-based_image_compression_with_semantic_distillation.md)
- [\[CVPR 2025\] CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)
- [\[NeurIPS 2025\] ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization](../../NeurIPS2025/model_compression/replaceme_network_simplification_via_depth_pruning_and_transformer_block_lineari.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)

</div>

<!-- RELATED:END -->

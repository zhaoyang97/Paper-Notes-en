---
title: >-
  [Paper Note] Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration
description: >-
  [CVPR 2025][3D Vision][Open-vocabulary 3D understanding] This paper proposes Dr. Splat, which bypasses the rendering process and directly registers language-aligned CLIP embeddings onto 3D Gaussians. Combined with Product Quantization (PQ) pre-trained on large-scale image data, it achieves a 6.25% embedding compression rate. Without requiring any scene-specific optimization (~10 minutes vs. 1–24 hours in prior work), it significantly outperforms existing methods in open-vocab…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Open-vocabulary 3D understanding"
  - "3D Gaussian Splatting"
  - "CLIP embeddings"
  - "Product Quantization"
  - "Feature registration"
  - "Semantic segmentation"
  - "3D grounding"
date: 2026-05-08
content_hash: ceb60381b47ac7bc
---

# Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration

**Conference**: CVPR 2025  
**arXiv**: [2502.16652](https://arxiv.org/abs/2502.16652)  
**Code**: [https://drsplat.github.io/](https://drsplat.github.io/)  
**Area**: 3D Vision / Scene Understanding  
**Keywords**: Open-vocabulary 3D understanding, 3D Gaussian Splatting, CLIP embeddings, Product Quantization, Feature registration, Semantic segmentation, 3D grounding

## TL;DR

This paper proposes Dr. Splat, which bypasses the rendering process and directly registers language-aligned CLIP embeddings onto 3D Gaussians. Combined with Product Quantization (PQ) pre-trained on large-scale image data, it achieves a 6.25% embedding compression rate. Without requiring any scene-specific optimization (~10 minutes vs. 1–24 hours in prior work), it significantly outperforms existing methods in open-vocabulary 3D semantic segmentation, 3D object grounding, and 3D object selection.

## Background & Motivation

**Background**: Open-vocabulary 3D scene understanding is a core challenge in associating natural language with 3D space. Existing 3DGS-based language embedding methods (such as LangSplat, LEGaussians, etc.) follow the "rendering-distillation" paradigm—attaching language features to 3D Gaussians and optimizing them by volume-rendering 2D feature maps and aligning them with CLIP embeddings.

**Limitations of Prior Work**:
- **Rendering Causes Embedding Distortion**: The rendering process performs alpha blending on the embeddings of multiple Gaussians. The resulting blended features are no longer valid vectors in the original CLIP space, leading to a systematic misalignment between the embeddings on the 3D Gaussians and CLIP text features.
- **Low Inference Efficiency**: To localize objects in 3D space, prior methods must pre-render 2D feature maps from a large number of viewpoints before retrieval, which is highly complex and requires additional 2D-to-3D back-projection mechanisms.
- **Heavy Scene-Specific Optimization Overhead**: Existing methods require training for 1–24 hours per new scene to optimize embeddings or train scene-level encoders-decoders/codebooks.

**Key Challenge**: The weighted summation of language embeddings via the volume rendering equation inherently destroys the semantic structure of the embeddings. This is an intrinsic flaw of the rendering-distillation paradigm that cannot be resolved through optimization.

**Goal**: How to directly associate high-fidelity CLIP embeddings with 3D Gaussians without rendering, while simultaneously achieving efficient storage and retrieval?

## Method

### Overall Architecture

Dr. Splat adopts a new paradigm of "registration instead of distillation," which consists of three steps: (1) Pre-processing: Train standard 3DGS and construct a universal PQ codebook (one-time cost); (2) Feature registration: Extract SAM masks and CLIP embeddings for each training image, assign the embedding of each pixel directly to the top-k Gaussians that contribute most to that pixel, aggregate the multi-view embeddings, and store them with PQ compression; (3) Inference: Directly calculate the cosine similarity and LeRF relevancy score of text queries on the 3D Gaussians. The entire process takes approximately 10 minutes and requires no gradient optimization.

### Key Designs

1. **Direct Feature Registration (Without Rendering-Distillation)**:
    - **Function**: Preserving the original semantic structure of CLIP embeddings and avoiding distortion caused by rendering-based blending.
    - **Mechanism**: Iterating through the training images, for each pixel $\mathbf{r}$, the method finds the top-k Gaussians with the highest contribution weights along the ray (i.e., $w_i(\mathbf{I}, \mathbf{r}) = T_i \cdot \tilde{\alpha_i}$), and accumulates the corresponding CLIP embedding of that pixel $\mathbf{f}_j^{map}$ onto these Gaussians with weight $w_{ij}$. Finally, the aggregated feature of each Gaussian $i$ is formulated as $\dot{\mathbf{f}}_i = \mathbf{f}_i / \|\mathbf{f}_i\|_2$, where $\mathbf{f}_i = \sum_j \frac{w_{ij}}{\sum_k w_{ik}} \mathbf{f}_j^{map}$.
    - **Design Motivation**: This can be viewed as an inverse volume rendering process, but without gradient optimization. Direct aggregation and normalization maintain the semantic consistency of the embeddings in the CLIP space.

2. **Universal Product Quantization (PQ)**:
    - **Function**: Efficiently compressing and storing high-dimensional CLIP embeddings without requiring scene-specific training.
    - **Mechanism**: Splitting the 512-dimensional CLIP vector into $L$ sub-vectors, where each sub-vector finds its nearest centroid in a pre-trained codebook and is replaced by an 8-bit index. PQ is trained once on the CLIP embeddings of the LVIS dataset (1.2M instances) and generalizes to any scene. Each Gaussian only stores the PQ indices instead of the full embedding, achieving a compression ratio of up to 6.25% (512×32-bit $\rightarrow$ L×8-bit). During retrieval, by utilizing a pre-computed query-to-centroid distance lookup table, the retrieval complexity per sample is reduced from $O(D)$ to $O(1)$.
    - **Design Motivation**: Although OpenGaussian also employs a codebook, it requires scene-specific construction; the universality of PQ is its core advantage—trained once, applicable everywhere.

3. **Volume-Aware 3D Evaluation Protocol**:
    - **Function**: Resolving the issue where traditional point cloud IoU is inapplicable for 3D Gaussian grounding evaluation.
    - **Mechanism**: Setting an importance score for each Gaussian as $d_i = s_{ix} \cdot s_{iy} \cdot s_{iz} \cdot \alpha_i$ (ellipsoid volume multiplied by opacity), and using weighted IoU to replace conventional equal-weight IoU. This avoids the irrationality of treating large-volume and small-volume Gaussians equally.
    - **Design Motivation**: The volume of 3D Gaussians can vary significantly (by several orders of magnitude), and traditional point cloud IoU severely underestimates or overestimates the contribution of specific Gaussians.

### Loss & Training

The feature registration process of Dr. Splat is **optimization-free**, involving no gradient computation or backpropagation. The 3DGS itself is trained using standard rendering loss, and the PQ codebook is pre-trained via k-means on LVIS CLIP embeddings.

## Key Experimental Results

### 3D Object Selection (LeRF-OVS Dataset)

| Method | mIoU↑ | mAcc@0.25↑ | Scene-Specific Optimization | Time |
|------|-------|------------|-----------|------|
| LangSplat-m | 9.83 | 15.94 | Required (~4h) | Slow |
| OpenGaussian | 43.06 | 59.61 | Required (~1h) | Medium |
| **Dr. Splat (Top-40)** | **43.58** | 63.87 | **Not Required** | **~10min** |

Dr. Splat outperforms OpenGaussian without requiring scene-specific optimization.

### 3D Object Grounding (ScanNet)

| Method | mIoU↑ | IoU>0.15 | IoU>0.3 | IoU>0.45 |
|------|-------|----------|---------|----------|
| LangSplat-m | 8.0 | 17.1 | 7.8 | 2.9 |
| LEGaussians-m | 9.5 | 19.1 | 8.9 | 7.3 |
| OpenGaussian | 25.2 | 59.5 | 38.0 | 18.3 |
| **Dr. Splat (Top-40)** | **25.4** | **60.7** | **40.3** | **25.6** |

At the high threshold of IoU>0.45, Dr. Splat outperforms OpenGaussian by 7.3 percentage points (25.6 vs. 18.3), demonstrating more precise localization.

### Key Findings

- Rendering-distillation methods (LangSplat-m) almost completely fail on 3D tasks (mIoU of only 8–10), validating the catastrophic impact of rendering-based blending on embeddings (clearly illustrated in the visualization of Fig. 2).
- PQ maintains near-lossless performance even with 64 sub-vectors (1/32 compression) and is almost lossless with 128 sub-vectors.
- As Top-k increases, mIoU improves consistently but memory usage also grows; Top-40 achieves a good trade-off between accuracy and efficiency.
- Dr. Splat also achieves closely competitive performance in semantic segmentation (19-class mIoU of 29.6 and mAcc of 47.7 vs. OpenGaussian's 30.1 and 46.5), despite not being specifically designed for segmentation.

## Highlights & Insights

- **"Registration over Distillation" Paradigm Shift**: It fundamentally solves the embedding distortion issue caused by rendering-based blending. By interpreting feature assignment through the lens of inverse rendering, the weighted aggregation operation is liberated from the differentiable rendering pipeline.
- **Universal PQ Eliminating Scene-Specific Training**: Training the codebook once on LVIS allows compressing CLIP embeddings from any scene. This is the key to truly realizing "out-of-the-box" 3D scene understanding.
- **Volume-Aware IoU Evaluation Metric**: This points out a fundamental issue in the evaluation of 3D Gaussian grounding—different Gaussians can vary in volume by several orders of magnitude, making traditional IoU completely unsuitable. The proposed weighted IoU is expected to become a standard evaluation protocol for future works.
- The entire pipeline takes about 10 minutes to complete, which is 6 to 150 times faster than prior methods requiring scene-specific optimization.

## Limitations & Future Work

- The Top-k mechanism is fundamentally a heuristic selection and may fail to allocate sufficient embeddings to heavily occluded small objects.
- PQ compression inevitably introduces quantization errors, which may affect fine-grained distinction for classes with ambiguous semantic boundaries (e.g., "fabric" vs. "cloth").
- The quality of feature registration depends on the quality of SAM segmentation—over-segmentation or under-segmentation by SAM directly propagates to 3D embeddings.
- In extremely large-scale scenes (millions of Gaussians), storage and retrieval remain challenging despite PQ compression.
- Currently, only cosine similarity and LeRF relevancy scores are utilized, with more complex retrieval mechanisms left unexplored.

## Related Work & Insights

- **vs. LangSplat**: LangSplat distributes embeddings via rendering-distillation, requiring 4 hours of training per scene and suffering from embedding distortion; Dr. Splat completes registration directly in 10 minutes, maintaining high-fidelity embeddings.
- **vs. OpenGaussian**: This is the most direct baseline—OpenGaussian also operates directly in the 3D space but requires scene-specific codebook training; Dr. Splat uses universal PQ to eliminate scene-specific training while achieving comparable or superior performance.
- **vs. LEGaussians**: LEGaussians also uses a codebook but remains within the rendering-distillation paradigm, leading to weak 3D performance.
- **Insight**: When the paradigm of "optimizing to 2D feature maps and then back-projecting to 3D" proves ineffective, directly establishing an explicit 2D-to-3D mapping (even a simple weighted aggregation) may be a superior alternative. This concept can be generalized to other tasks requiring the transfer of 2D foundation model features to 3D spaces.

## Rating

- Novelty: ⭐⭐⭐⭐ Proposing a direct registration paradigm to replace rendering-distillation, with universal PQ eliminating scene-specific training.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks across two datasets, with ablations covering critical parameters of PQ and Top-k.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, standard mathematical notation, and visualizations that intuitively demonstrate embedding distortion.
- Value: ⭐⭐⭐⭐ Significantly lowers the barrier to entry for open-vocabulary 3D scene understanding (requiring zero scene-specific training and completing in 10 minutes).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)
- [\[ICML 2025\] ReferSplat: Referring Segmentation in 3D Gaussian Splatting](../../ICML2025/3d_vision/refersplat_referring_segmentation_in_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization](morpheus_text-driven_3d_gaussian_splat_shape_and_color_stylization.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](../../NeurIPS2025/3d_vision/segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)
- [\[CVPR 2026\] ST4R-Splat: Spatio-Temporal Referring Segmentation in 4D Gaussian Splatting](../../CVPR2026/3d_vision/st4r-splat_spatio-temporal_referring_segmentation_in_4d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->

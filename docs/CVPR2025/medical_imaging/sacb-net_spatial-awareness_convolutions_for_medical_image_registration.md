---
title: >-
  [Paper Note] SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration
description: >-
  [CVPR 2025][Medical Imaging][Medical image registration] This paper proposes a 3D Spatial-Awareness Convolutional Block (SACB) that performs unsupervised clustering on feature maps and generates adaptive convolution kernels for different spatial regions. Combined with a pyramid flow estimator to achieve multi-scale deformation field composition, this method outperforms existing state-of-the-art (SOTA) methods on brain and abdomen CT registration tasks.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Medical image registration"
  - "spatial-adaptive convolutions"
  - "pyramid flow estimation"
  - "deformation field"
  - "3D registration"
date: 2026-05-08
content_hash: dafcd894a3ea6d13
---

# SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration

**Conference**: CVPR 2025  
**arXiv**: [2503.19592](https://arxiv.org/abs/2503.19592)  
**Code**: [https://github.com/x-xc/SACB_Net](https://github.com/x-xc/SACB_Net)  
**Area**: Medical Image  
**Keywords**: Medical image registration, spatial-adaptive convolutions, pyramid flow estimation, deformation field, 3D registration

## TL;DR

This paper proposes a 3D Spatial-Awareness Convolutional Block (SACB) that performs unsupervised clustering on feature maps and generates adaptive convolution kernels for different spatial regions. Combined with a pyramid flow estimator to achieve multi-scale deformation field composition, this method outperforms existing state-of-the-art (SOTA) methods on brain and abdomen CT registration tasks.

## Background & Motivation

**Background**: Deep learning-based medical image registration methods have made significant progress. Mainstream methods include U-Net-based VoxelMorph series, model-driven Fourier-Net/B-Spline methods, and cascaded/pyramid multi-scale methods such as LapIRN, ModeT, etc.

**Limitations of Prior Work**: Almost all existing methods rely on spatially shared convolution kernels—that is, the same convolution weights are used for all spatial positions within the same layer. However, in medical images, the deformation characteristics of different anatomical regions (such as gray matter, white matter, and cerebrospinal fluid) vary significantly. Shared convolution kernels cannot capture non-local spatially varying information, resulting in imprecise deformation field estimation.

**Key Challenge**: There is a fundamental conflict between the spatial invariance of standard convolutions and the spatial heterogeneity of deformations in medical image registration—tissue morphology is tightly coupled with deformation, and voxels/features in different regions require different levels of attention.

**Goal**: (1) Design a convolutional mechanism capable of perceiving spatial-regional differences; (2) Integrate this mechanism into a pyramid framework to handle large deformations.

**Key Insight**: Inspired by content-adaptive convolutions in 2D image pan-sharpening, the authors extend this concept to 3D medical registration. The key observation is that if voxels can be partitioned into different spatial clusters based on feature similarity, and then a dedicated convolution kernel is generated for each cluster, spatially aware feature extraction can be achieved.

**Core Idea**: Unsupervised K-Means clustering is utilized to partition the feature map into semantically similar spatial regions. Cluster centers are then passed through MLPs to generate region-specific adaptive convolution kernel weights and biases, thereby replacing traditional spatially shared convolutions.

## Method

### Overall Architecture

SACB-Net adopts a "shared encoder + pyramid flow estimator" architecture. Given a pair of moving and fixed images, the shared encoder extracts multi-resolution feature pyramids across five scales. Then, from the coarsest scale (scale 5) to the finest scale (scale 1), step-by-step SACB feature enhancement and similarity matching calculations are performed, gradually generating the final deformation field through flow composition.

### Key Designs

1. **3D Spatial-Awareness Convolutional Block (SACB)**:

    - **Function**: Enhances the spatial representation capability of feature maps, enabling different anatomical regions to receive distinct feature processing.
    - **Mechanism**: SACB consists of three components—a spatial context estimation module, an adaptive kernel generator, and a residual connection. First, the feature map $\mathbf{F}$ is unfolded into local patches to compute spatial means, which are then reshaped and clustered using GPU K-Means to obtain a cluster index matrix $S$ and cluster centers $S_n^c$ for each cluster. Subsequently, two MLPs generate spatial weights $\mathcal{F}_w(S_n^c)$ and biases $\mathcal{F}_b(S_n^c)$ from the centers $S_n^c$. The weights are element-wise multiplied with the globally learnable convolution kernel $\mathbf{W}$ to obtain region-specific kernels $\mathbf{W}_n = \mathcal{F}_w(S_n^c) \odot \mathbf{W}$. Finally, the enhanced features are output via a residual connection: $\hat{\mathbf{F}} = \mathbf{F} + \sigma(SAC(\mathbf{F}))$.
    - **Design Motivation**: Clustering without label information is preferred because annotations between moving and fixed images may be inconsistent in the feature space, and labels are generally scarce. Generating weights via MLPs instead of directly learning multiple sets of convolution kernels maintains parameter efficiency while achieving adaptability.

2. **Pyramid Flow Estimator**:

    - **Function**: Progressively estimates the deformation field across multiple scales to handle large deformations.
    - **Mechanism**: Starting from the coarsest scale, each stage first enhances the moving and fixed image features using SACB, and then computes similarity matching scores $M_{sim}$. The matching scores are obtained by taking the dot product of the fixed features and unfolded moving features followed by a Softmax operation, which is then multiplied by a 3D position grid $G$ to obtain a sub-deformation flow. The flow from the previous stage is upsampled to warp the moving features of the current stage before calculating the residual flow and combining it with the upsampled flow. The final stage (scale 1) directly estimates the flow using two convolutional layers without matching.
    - **Design Motivation**: The coarse-to-fine strategy allows low-resolution levels to handle large displacements while high-resolution levels deal with fine alignment. A search window of $k=3$ corresponds to large physical displacements at the coarsest scale but only 1 voxel at the finest scale, so matching is omitted at scale 1 to save computation.

3. **Shared Encoder**:

    - **Function**: Extracts multi-scale feature pyramids for both moving and fixed images.
    - **Mechanism**: It consists of five layers of 3D convolutions with four average pooling downsamplings. Each layer contains a 3D convolution, InstanceNorm, and LeakyReLU (0.1). The moving and fixed images share the same encoder weights.
    - **Design Motivation**: Weight sharing guarantees a consistent representation space for both feature branches, which facilitates subsequent similarity matching.

### Loss & Training

The overall loss is formulated as $\mathcal{L} = \mathcal{L}_{sim}(I_m \circ (\phi + \text{Id}), I_f) + \lambda \mathcal{L}_{reg}$, where the similarity term utilizes Normalized Cross-Correlation (NCC) and the regularization term is the L2 norm of the deformation field gradient $\|\nabla\phi\|_2^2$ to encourage smooth deformations. The model is trained using the Adam optimizer with a learning rate of $10^{-4}$ and a batch size of 1 on an A100 40GB GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | SACB-Net | ModeT (2nd) | LKU | TransMorph | Params |
|--------|------|----------|-------------|-----|------------|-------|
| IXI (30 ROIs) | Dice↑ | **0.769** | 0.758 | 0.765 | 0.754 | 1.11M |
| IXI | \|J\|<0%↓ | 0.083 | 0.114 | 0.109 | 1.579 | — |
| LPBA (54 ROIs) | Dice↑ | **0.731** | 0.721 | 0.706 | 0.695 | — |
| LPBA | HD95↓ | **5.862** | 5.969 | 6.452 | 6.564 | — |
| Abdomen CT | Dice↑ | **0.588** | 0.550 | 0.423 | 0.444 | — |
| Abdomen CT | HD95↓ | **18.253** | 20.351 | 24.252 | 24.187 | — |

### Ablation Study

| Configuration | IXI Dice | LPBA Dice | Abdomen Dice |
|------|----------|-----------|--------------|
| w/o SACB (baseline) | 0.7643 | 0.7141 | 0.5374 |
| SACB @ scale5 only | 0.7668 | 0.7217 | 0.5444 |
| SACB @ scale5-4 | 0.7679 | 0.7241 | 0.5498 |
| SACB @ scale5-3 | 0.7671 | 0.7266 | 0.5685 |
| SACB @ scale5-2 (N=5) | 0.7683 | 0.7294 | 0.5849 |
| SACB @ scale5-2 (N=7) | **0.7691** | **0.7309** | **0.5881** |
| SACB @ scale5-2 (N=11) | 0.7684 | 0.7300 | 0.5875 |

### Key Findings

- The effect of SACB is cumulative across all scales, yielding the most significant improvement in large deformation scenarios such as abdomen CT (Dice improved from 0.537 to 0.588, a gain of approximately 5%).
- The number of clusters $N=7$ is the optimal choice. Too many clusters ($N=11$) lead to a slight performance drop, possibly because over-segmentation results in less meaningful clusters.
- Clustering based on spatial-dimension patch means outperforms channel-dimension means, suggesting that spatial locality is crucial for registration.
- The advantage is most pronounced in the large-deformation abdomen CT task (Dice is 3.8% higher than the runner-up ModeT), validating the complementary effect of the pyramid structure and spatial adaptability.

## Highlights & Insights

- **Replacing label-driven regional division with unsupervised clustering**: This cleverly avoids the issues of label inconsistency and scarcity in registration, while enabling the model to automatically discover meaningful spatial partitions. This idea can be transferred to other 3D tasks requiring spatial-adaptive processing.
- **Adaptive convolution kernel = global kernel × regional weight**: Instead of learning complete convolution kernels for each region independently, the model generates modulation weights via MLPs to adjust the global kernel, which is highly parameter-efficient.
- **Small model, large impact**: With only 1.11M parameters, it outperforms TransMorph with 46.77M parameters, demonstrating that inductive bias (spatial adaptability) is more important than model capacity in registration tasks.

## Limitations & Future Work

- K-Means clustering is non-differentiable, making it impossible to train the clustering process end-to-end; thus, clustering quality depends heavily on the quality of feature representations.
- The number of clusters $N$ needs to be manually adjusted, and the optimal number of clusters may vary across different anatomical regions.
- Diffeomorphic constraints are not considered; although $|J|<0\%$ is low, it is not zero.
- Using simple convolutions instead of matching at scale 1 may compromise fine alignment capabilities; more efficient full-resolution matching schemes could be explored.

## Related Work & Insights

- **vs LKU**: LKU models long-range dependencies using large-kernel convolutions but still relies on spatially shared kernels; SACB-Net achieves spatial-variation awareness with adaptive kernels, yielding higher Dice scores while using half the parameters.
- **vs ModeT**: ModeT uses multi-head attention to fuse multi-scale flows in a weighted manner. SACB-Net adaptively enhances features using spatial clustering before matching, demonstrating a more pronounced advantage in large deformation scenarios.
- **vs TransMorph**: Although Transformers can capture long-range dependencies, they have a massive number of parameters (46.77M vs 1.11M) and lack explicit spatial adaptability.

## Rating

- Novelty: ⭐⭐⭐⭐ Extending 2D content-adaptive convolution to 3D medical registration is a pioneering effort with a clear pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across three datasets, comparison with 14 methods, and multi-dimensional ablation studies are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear and the illustrations are intuitive, though formula notations are occasionally inconsistent.
- Value: ⭐⭐⭐⭐ The concept of spatial-adaptive convolution has broad applicability, high parameter efficiency, and strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CARL: A Framework for Equivariant Image Registration](carl_a_framework_for_equivariant_image_registration.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[ECCV 2024\] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](../../ECCV2024/medical_imaging/textttnephi_neural_deformation_fields_for_approximately_diff.md)
- [\[ICLR 2026\] HFSTI-Net: Hierarchical Frequency-spatial-temporal Interactions for Video Polyp Segmentation](../../ICLR2026/medical_imaging/hfsti-net_hierarchical_frequency-spatial-temporal_interactions_for_video_polyp_s.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)

</div>

<!-- RELATED:END -->

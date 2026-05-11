---
title: >-
  [Paper Note] Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation
description: >-
  [AAAI 2026][Segmentation][Aortic segmentation] This paper proposes the Morph-Patch Transformer (MPT), which generates morphology-aware patches via a velocity-field-based adaptive patch partitioning strategy to preserve vascular topological integrity, and introduces Semantic Clustering Attention (SCA) to dynamically aggregate features from semantically similar patches. The method achieves state-of-the-art performance on three aortic segmentation benchmarks: AVT, AortaSeg24, and TBAD.
tags:
  - AAAI 2026
  - Segmentation
  - Aortic segmentation
  - morphology-aware patch
  - semantic clustering attention
  - diffeomorphic deformation
  - velocity field
date: 2026-05-08
content_hash: 89688e52a51be7c8
---

# Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation

**Conference**: AAAI 2026
**arXiv**: [2511.06897](https://arxiv.org/abs/2511.06897)
**Code**: [https://github.com/iCherishxixixi/MPTransformer](https://github.com/iCherishxixixi/MPTransformer)
**Area**: Image Segmentation
**Keywords**: Aortic segmentation, morphology-aware patch, semantic clustering attention, diffeomorphic deformation, velocity field

## TL;DR

This paper proposes the Morph-Patch Transformer (MPT), which generates morphology-aware patches via a velocity-field-based adaptive patch partitioning strategy to preserve vascular topological integrity, and introduces Semantic Clustering Attention (SCA) to dynamically aggregate features from semantically similar patches. The method achieves state-of-the-art performance on three aortic segmentation benchmarks: AVT, AortaSeg24, and TBAD.

## Background & Motivation

Aortic vessel segmentation is critical for the diagnosis and treatment of cardiovascular diseases, directly affecting the reliability of computational fluid modeling, surgical planning, and disease progression monitoring. While Transformers have become the dominant paradigm in this domain, two fundamental challenges remain:

**Fixed rectangular patches disrupt vascular integrity**: Conventional Transformers partition images into fixed-size rectangular patches, which are ill-suited for the elongated, curved, and morphologically complex structure of blood vessels. Rectangular windows often truncate thin vessels, fragmenting semantic information. Even DPT (Deformable Patch Transformer), which introduces learnable deformation, remains constrained to rectangular patches and cannot adapt to vascular morphology.

**Lack of cross-scale semantic similarity modeling**: The hierarchical window attention of Swin Transformer enables multi-scale feature extraction, but fixed windows still fail to model semantic similarity across patches at different scales. Existing methods—including those leveraging dynamic Snake convolutions—enhance feature extraction for elongated structures but lack a semantic clustering mechanism.

Core insight: Diffeomorphic deformation driven by a velocity field naturally generates morphology-aware patches with topological continuity; Soft K-means clustering enables dynamic aggregation of semantically similar patches via SCA.

## Method

### Overall Architecture

The model follows a 3D UNet-based encoder-decoder structure with two core innovations: (1) the Morph Partition Block replaces fixed patch partitioning; and (2) a Spatial + Semantic Transformer Block fuses spatial relationships (window attention) with semantic relationships (clustering attention). Three variants are provided: MPT (pure 3D ViT), MPT-UNETR (hybrid 3D ViT-CNN), and MPTUNet (lightweight 2D).

### Key Designs

1. **Morph Partition Block (morphology-aware patch partitioning)**:

    - A CNN predicts a velocity field $\upsilon$, which is integrated via the scaling-and-squaring method to obtain a diffeomorphic deformation field $\phi^{(1)}$.
    - Core formula: $y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n + \phi(p_0 + p_n))$
    - Unlike conventional deformable convolutions that directly predict offset fields, velocity field integration guarantees **smooth, invertible, and topology-preserving** transformations. Each point in the deformation field represents a coordinate offset; deformed features are sampled from the original input via bilinear interpolation.
    - Recurrence relation: $\phi^{(1/2^{n-1})} = \phi^{(1/2^n)} \circ \phi^{(1/2^n)}$, yielding $\phi^{(1)}$ after $n$ iterations.

2. **Semantic Clustering Attention (SCA)**:

    - Differentiable Soft K-means is employed to extract core semantic features $F_{core}$, clustering patch features according to softmax-weighted distances to the core features.
    - Update rule: $f_{newcore}^s = \sum_{i=1}^m g_s(f^i) \cdot (f^i - f_{core}^s)$, where $g_s$ is a differentiable membership function.
    - $\lambda$, $\mu$, and $F_{core}$ are all learned by the network; $g_s$ is parameterized as $e^{\lambda^s f^i + \mu^s}$ to ensure differentiability.
    - SCA is computed as: $\text{SCA} = \text{softmax}(QK^T/\sqrt{d}) \cdot V$, where Q comes from patch tokens and K/V come from the updated semantic centers.

3. **Fusion strategy**: Morphology-aware patches from the Morph Partition Block, window attention from Swin Transformer (spatial relationships), and SCA (semantic relationships) are jointly integrated within the Transformer Block.

### Loss & Training

Dice loss is used. The Adam optimizer is employed with a learning rate of 5e-5. The number of clusters is set to 32. Models are trained for 1000 epochs within the nnU-Net framework on an NVIDIA RTX 3090.

## Key Experimental Results

### Main Results: AVT and TBAD Datasets

| Model | Backbone Type | AVT Dice | AVT mIoU | AVT clDice | TBAD Dice | TBAD mIoU | TBAD clDice |
|------|---------|----------|----------|-----------|-----------|-----------|------------|
| MedNeXt | CNN | 0.809 | 0.718 | 0.724 | 0.926 | 0.871 | 0.880 |
| SegMamba | Mamba | 0.829 | 0.730 | 0.711 | 0.932 | 0.881 | 0.918 |
| nnFormer | 3D ViT | 0.835 | 0.743 | 0.732 | 0.926 | 0.871 | 0.895 |
| DPT | 2D ViT-CNN | 0.886 | 0.800 | 0.825 | 0.924 | 0.868 | 0.917 |
| MambaVision | Mamba | 0.882 | 0.795 | 0.795 | 0.929 | 0.874 | 0.914 |
| TransFuse | 2D ViT-CNN | 0.880 | 0.794 | 0.796 | 0.927 | 0.872 | 0.895 |
| **MPT** | **3D ViT** | **0.856** | **0.762** | 0.757 | **0.933** | **0.881** | 0.915 |
| **MPTUNet** | **2D ViT-CNN** | **0.896** | **0.815** | **0.839** | **0.930** | **0.877** | **0.920** |

### AortaSeg24 Dataset (23-class Fine-grained Segmentation)

| Model | Dice | mIoU | clDice |
|------|------|------|--------|
| 3DUXNet | 0.784 | 0.666 | 0.964 |
| nnFormer | 0.779 | 0.666 | 0.923 |
| SwinUNETR | 0.781 | 0.664 | 0.937 |
| DSCViT | 0.788 | 0.673 | 0.965 |
| DPT | 0.778 | 0.662 | 0.959 |
| MambaVision | 0.795 | 0.682 | 0.960 |
| **MPT** | **0.804** | **0.690** | 0.926 |
| **MPTUNETR** | **0.809** | **0.695** | **0.955** |
| **MPTUNet** | **0.796** | **0.686** | **0.966** |

### Key Findings

- **MPTUNet achieves Dice 0.896 on AVT**, surpassing all baselines (including DPT 0.886, MambaVision 0.882, TransFuse 0.880), with the lowest variance (0.046), indicating strong stability.
- **AortaSeg24's 23-class fine-grained segmentation** is highly challenging; MPTUNETR achieves Dice 0.809 / clDice 0.955, outperforming the second-best MambaVision (0.795/0.960).
- **The clDice metric**, specifically designed to measure vascular topological integrity, confirms the effectiveness of diffeomorphic deformation: MPTUNet achieves the highest clDice of 0.920 on TBAD.
- **Model efficiency**: The MPT family is comparable to DPT in FLOPs and parameter count, while delivering substantially better performance.

## Highlights & Insights

- **Velocity field → diffeomorphic deformation → topology-preserving patch partitioning**: Unlike DCN, which directly predicts offsets, deformations obtained via ODE integration are inherently smooth and invertible—particularly well-suited for vascular structures that require topological continuity.
- **Differentiable semantic clustering via Soft K-means**: Hard assignments in standard K-means are softened through an exponential kernel $e^{-\beta \|f-f_{core}\|^2}$, with $\lambda$, $\mu$, and $F_{core}$ fully learnable, yielding greater flexibility than fixed-window attention.
- **Three variants covering diverse requirements**: MPT (pure ViT for accuracy), MPT-UNETR (hybrid architecture balancing efficiency), and MPTUNet (lightweight 2D design).

## Limitations & Future Work

- The CNN for velocity field prediction introduces additional computational overhead; inference speed is not thoroughly analyzed.
- The number of clusters is fixed at 32; adaptive clustering strategies are not explored.
- Validation is limited to aorta-related datasets; generalization to other vascular segmentation tasks (e.g., retinal vessels, coronary arteries) remains untested.
- The trade-off between deformation precision and computational cost with respect to the number of scaling-and-squaring steps is insufficiently discussed.

## Related Work & Insights

| Method Category | Representative | Patch Strategy | Semantic Modeling | Topology Preservation |
|----------|------|-----------|---------|---------|
| Standard Transformer | UNETR, SwinUNETR | Fixed rectangular | Window attention | None |
| Deformable Transformer | DPT | Deformable rectangular | Standard attention | Weak |
| Snake convolution hybrid | TTCNet, DAU-Net | Fixed + Snake | Convolutional features | Implicit |
| **MPT (Ours)** | **MPT/UNETR/UNet** | **Diffeomorphic deformation** | **SCA clustering attention** | **Strong (topology-preserving)** |

## Rating

- Novelty: ⭐⭐⭐⭐ Velocity-field-driven morphology patches + differentiable semantic clustering attention
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + 17 comparison methods + three model variants
- Writing Quality: ⭐⭐⭐⭐ Clear methodological derivation and well-motivated problem formulation
- Value: ⭐⭐⭐⭐ Broadly inspiring for segmentation of morphologically complex structures in medical imaging

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Locality-Attending Vision Transformer](../../ICLR2026/segmentation/locality-attending_vision_transformer.md)
- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](../../ICLR2026/segmentation/revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](../../CVPR2026/segmentation/seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)

</div>

<!-- RELATED:END -->

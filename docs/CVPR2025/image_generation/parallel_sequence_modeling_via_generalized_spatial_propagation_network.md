---
title: >-
  [Paper Note] Parallel Sequence Modeling via Generalized Spatial Propagation Network
description: >-
  [CVPR 2025][Image Generation][Attention Mechanism] GSPN proposes the Generalized Spatial Propagation Network, which achieves a natively 2D spatially-aware sub-quadratic attention mechanism through 2D linear propagation of row/column line scans under a stability-context condition. This reduces the effective sequence length to $\sqrt{N}$ and accelerates SD-XL by up to 84x in 16K image generation.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Attention Mechanism"
  - "Spatial Propagation Network"
  - "Linear Complexity"
  - "2D Structure"
  - "High-Resolution Generation"
date: 2026-05-08
content_hash: e657d6508b6d39f1
---

# Parallel Sequence Modeling via Generalized Spatial Propagation Network

**Conference**: CVPR 2025  
**arXiv**: [2501.12381](https://arxiv.org/abs/2501.12381)  
**Code**: [Project Page](https://whj363636.github.io/GSPN/)  
**Area**: Image Generation / Architecture Design  
**Keywords**: Attention Mechanism, Spatial Propagation Network, Linear Complexity, 2D Structure, High-Resolution Generation

## TL;DR

GSPN proposes the Generalized Spatial Propagation Network, which achieves a natively 2D spatially-aware sub-quadratic attention mechanism through 2D linear propagation of row/column line scans under a stability-context condition. This reduces the effective sequence length to $\sqrt{N}$ and accelerates SD-XL by up to 84x in 16K image generation.

## Background & Motivation

While Transformers have achieved great success in vision tasks, they face two core limitations:
- **Quadratic Computational Complexity**: The computational cost is massive when processing high-resolution images, especially for ultra-high-resolution tasks such as 16K.
- **Neglect of Spatial Structure**: Flattening a 2D image into a 1D sequence loses spatial coherence.
- Although linear attention methods ($Q(K^\top V)$) reduce complexity, they similarly ignore spatial structures.
- State Space Models (SSMs, e.g., Mamba) employ 1D raster scanning to handle 2D data, sacrificing inherent spatial structures.
- The 2D linear propagation of the core challenge: successive multiplication of weight matrices—eigenvalues that are too large cause exponential growth (instability), while those that are too small lead to signal decay (vanishing information).
- There is a critical need to simultaneously ensure stability and model long-range dependencies.

## Method

### Overall Architecture

GSPN achieves 2D linear propagation via row/column line scanning, where each pixel connects to 3 adjacent pixels in the previous row/column (forming a tridiagonal matrix). After propagating from 4 directions (left-to-right, top-to-bottom, and their reverses), they are fused. GSPN offers both global and local variants, allowing seamless replacement of existing attention modules in current architectures.

### Key Designs

#### Key Design 1: Stability-Context Condition

**Function**: Ensures that 2D propagation is both stable over long distances and maintains effective contextual information.

**Mechanism**: The 2D propagation formula is $h_i^c = w_i^c h_{i-1}^c + \lambda_i^c \odot x_i^c$, with the accumulated weight $W_{ij} = \prod_{\tau=j+1}^i w_\tau$. To guarantee that $h_i$ is a weighted average of all previous $x'_j$, two conditions must be met: (1) $W_{ij}$ is a dense matrix, and (2) $\sum_{j=0}^{n-1} W_{ij} = 1$. **Theorem 1**: If all $w_\tau$ are row-stochastic matrices (non-negative with a row sum of 1), then $\sum W_{ij} = 1$ holds. **Theorem 2**: The row-stochastic constraint simultaneously guarantees propagation stability. Implementation: Apply sigmoid to non-zero elements of each row, followed by row normalization.

**Design Motivation**: To simultaneously achieve stable propagation and capture long-range dependencies without introducing decay factors—traditional methods must compromise between the two.

#### Key Design 2: Tridiagonal Matrix + 4-Direction Line Scan

**Function**: Establishes dense pairwise connections among all pixels in a parameter-efficient manner.

**Mechanism**: Each pixel only connects to 3 adjacent pixels in the previous row/column (top-left, top-middle, top-right), meaning $w_\tau$ is a tridiagonal matrix. Key mathematical property: **the product of multiple tridiagonal matrices is a dense matrix**, so long-range connections are naturally established after propagating through multiple rows. Propagation is conducted from 4 directions, which are then aggregated using a learnable merger. Using a custom CUDA kernel for parallelization: inter-row propagation is sequential, while inter-column, inter-channel, and inter-batch calculations are parallelized, resulting in an effective sequence length of only $\sqrt{N}$.

**Design Motivation**: Directly learning an $n \times n$ fully-connected matrix requires too many parameters; the combination of tridiagonal connections and cumulative multiplication achieves equivalent full connectivity with $O(3n)$ parameters.

#### Key Design 3: Global/Local Variants and Task Adaptation

**Function**: Flexibly selects global or local propagation scopes according to task requirements.

**Mechanism**: Local GSPN divides a spatial dimension into $g$ non-overlapping groups, with propagation occurring independently within each group, reducing complexity by $g$ times (in the extreme case of $g=n$, complexity is $O(1)$). Classification tasks: local layers are used at lower levels, and global layers are used at higher levels (requiring semantic understanding). Generation tasks: local layers are primarily used (requiring spatial details and local consistency). T2I Generation: directly replaces the self-attention layers in SD-XL, initializing GSPN parameters using pretrained Q/K/V weights (exploiting the mathematical relationship between GSPN and linear attention).

**Design Motivation**: Different vision tasks have disparate demands for global vs. local information, and flexible switching maximizes efficiency. Eliminating position embeddings (as scanning inherently implies positional information) avoids common aliasing problems.

### Loss & Training

Varies with tasks: cross-entropy for classification, diffusion loss for DiT, and standard SD loss for T2I.

## Key Experimental Results

### Main Results: ImageNet Classification

| Model | Type | Params (M) | MACs (G) | Top-1 Acc |
|------|------|---------|--------|----------|
| **GSPN-T** | Line scan | 30 | 5.3 | **83.0** |
| VMamba-T | Raster | 22 | 5.6 | 82.2 |
| Swin-T | Transformer | 29 | 4.5 | 81.3 |
| ConvNeXT-T | ConvNet | 29 | 4.5 | 82.1 |
| LocalVMamba-T | Raster | 26 | 5.7 | 82.7 |

### Ablation Study: Inference Speed Comparison (SD-XL 16K Generation)

| Attention Type | 16K Inference Time | Acceleration Ratio |
|-----------|------------|--------|
| Softmax Attention | Extremely slow | 1× |
| **GSPN (Local)** | Extremely fast | **84×** |
| GSPN (Global) | Fast | Medium |

### Key Findings

- GSPN-T (83.0%) outperforms all Mamba/Transformer/ConvNet models of similar size on ImageNet classification.
- In DiT class-conditional generation, GSPN outperforms SOTA Diffusion Transformers using only 65.6% of the parameters.
- Replacing self-attention in SD-XL achieves an 84x speedup in 16K image generation while maintaining original performance.
- The theoretical guarantees of the stability-context condition are validated experimentally—demonstrating effective modeling of long-range dependencies.

## Highlights & Insights

- **Mathematical Elegance**: Solves both stability and long-range dependency simultaneously through properties of row-stochastic matrices, supported by clear theoretical guarantees.
- **High Practical Appeal**: The 84x speedup makes ultra-high-resolution generation feasible.
- **No Need for Position Embeddings**: Scanning order implicitly encodes position information, avoiding extrapolation and aliasing issues.

## Limitations & Future Work

- The computational overhead introduced by the 4-direction scanning is a constant factor but non-negligible.
- The sparsity of tridiagonal connections might be suboptimal for certain tasks that require exact global correspondences.
- Currently validated mainly on 2D images; extending this to 3D (e.g., video) and multi-modal settings remains to be explored.
- Future work can explore adaptive numbers of directions and connection schemes.

## Related Work & Insights

- Connection to SPN: GSPN promotes a single-layer, module-level SPN to a stackable backbone architecture and solves the issue of long-range propagation.
- Comparison with Mamba: GSPN natively preserves 2D structures instead of flattening them into 1D.
- An effective sequence length of $\sqrt{N}$ may inspire other efficient architecture designs aimed at processing 2D data.

## Rating

⭐⭐⭐⭐ — A novel attention mechanism featuring theoretical elegance and thorough experimental evaluation. The mathematical derivation of the stability-context condition is convincing, and the 84x speedup provides immense practical value. It exhibits competitive performance across three distinct tasks: classification, conditional generation, and T2I generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](../../NeurIPS2025/image_generation/gspn-2_efficient_parallel_sequence_modeling.md)
- [\[CVPR 2025\] HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [\[CVPR 2025\] Generation of Maximal Snake Polyominoes Using a Deep Neural Network](generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [\[CVPR 2025\] PCM: Picard Consistency Model for Fast Parallel Sampling of Diffusion Models](pcm_picard_consistency_model_for_fast_parallel_sampling_of_diffusion_models.md)

</div>

<!-- RELATED:END -->

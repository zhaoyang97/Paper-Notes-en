---
title: >-
  [Paper Note] Hilbert-Guided Sparse Local Attention
description: >-
  [ICLR 2026][Hilbert curve] By reordering 2D image tokens into a 1D sequence via Hilbert space-filling curves—which preserve spatial locality—this work substantially increases the empty-block ratio in local attention (fro…
tags:
  - "ICLR 2026"
  - "Hilbert curve"
  - "local attention"
  - "block sparsity"
  - "FlexAttention"
  - "Vision Transformer"
date: 2026-05-08
content_hash: ed72649a26efcb8c
---

# Hilbert-Guided Sparse Local Attention

**Conference**: ICLR 2026
**arXiv**: [2511.05832](https://arxiv.org/abs/2511.05832)
**Code**: [GitHub](https://github.com/Yunge6666/Hilbert-Local-Attention)
**Area**: Efficient Transformers / Attention Mechanisms
**Keywords**: Hilbert curve, local attention, block sparsity, FlexAttention, Vision Transformer

## TL;DR
By reordering 2D image tokens into a 1D sequence via Hilbert space-filling curves—which preserve spatial locality—this work substantially increases the empty-block ratio in local attention (from 87.5% to 96.9%), enabling 4× speedup for window attention and 18× for sliding-window attention via FlexAttention, with negligible accuracy loss.

## Background & Motivation

**Background**: The global self-attention mechanism of Vision Transformers incurs $O(N^2)$ complexity, limiting their applicability to high-resolution images. Local attention variants (e.g., Swin's window attention, NAT's neighborhood attention) reduce this complexity and represent the mainstream approach.

**Limitations of Prior Work**: Although local attention reduces computation in theory, practical GPU efficiency depends critically on the underlying kernel implementation. The conventional row-major token ordering causes tokens within a 2D local window to be non-contiguous in 1D, resulting in numerous partial blocks under block-sparse attention frameworks such as FlexAttention, which require additional masking overhead and limit effective sparse acceleration.

**Key Challenge**: Block-sparse kernels skip empty blocks to accelerate computation, yet row-major window attention yields a low empty-block ratio (~87.5%) and many partial blocks, constraining achievable speedups.

**Key Insight**: The Hilbert curve possesses strong locality-preserving properties—points that are spatially adjacent in 2D remain proximate along the 1D curve. Reordering tokens by the Hilbert traversal makes tokens within a local window contiguous in 1D, dramatically increasing the empty-block ratio and reducing partial blocks.

**Core Idea**: Hilbert reordering compacts 2D local attention patterns into the 1D sequence, yielding more empty blocks and enabling more efficient block-sparse kernel execution.

## Method

### Overall Architecture
Input image tokens are reordered according to the Hilbert curve traversal path; local windows or neighborhoods are then constructed on the resulting 1D sequence; FlexAttention's block-sparse kernel computes attention; the high empty-block ratio translates directly into wall-clock speedup. The Hilbert path is computed once at model initialization and cached for reuse.

### Key Designs

1. **Hilbert Window Attention (HWA)**:

    - **Function**: Partitions the Hilbert-reordered 1D sequence into contiguous windows.
    - **Mechanism**: A 2×2 window under row-major ordering touches tokens (1, 2, 5, 6), producing four partial blocks; the same window under Hilbert ordering touches tokens (1, 2, 3, 4), yielding two full blocks and two empty blocks. The empty-block ratio rises from 0% to 50%.
    - **Design Motivation**: Contiguous windows generate more empty blocks, allowing FlexAttention to skip more computation.

2. **Hilbert Sliding/Neighborhood Attention (HSA/HNA)**:

    - **Function**: Applies sliding-window or neighborhood attention on the Hilbert-reordered sequence.
    - **Mechanism**: The locality-preserving property of the Hilbert curve ensures that 1D neighbors $\approx$ 2D spatial neighbors, making 2D neighborhood attention equivalent to 1D neighborhood attention.
    - **Design Motivation**: This formulation avoids the $N^2$ intermediate storage overhead associated with 2D neighborhood attention.

3. **Hilbert Window Transformer (HWT)**:

    - **Function**: Replaces the window self-attention (WSA) in Swin Transformer with HWA.
    - **Mechanism**: HWA and Hilbert Shifted Window Attention (HSWA) are used in alternating layers. Window shifting is performed in 1D by offsetting a fixed number of tokens. A global relative position bias (RPB) replaces the per-window RPB, since Hilbert windows have irregular shapes.
    - **Design Motivation**: FlexAttention's `mask_mod` and `score_mod` interfaces enable integration without modifying the model architecture or training pipeline.

4. **Speedup Analysis**:

    - Total runtime is approximated as $T \approx \frac{\sum_{i=1}^{M}(\alpha + \beta \cdot r_i)}{P_{\text{eff}}}$
    - More empty blocks → fewer CTA launches and K/V loads → shorter runtime.

### Loss & Training
- Standard ImageNet classification training protocol is used throughout.
- The Hilbert path is computed and cached at model initialization.
- The method is compatible with multiple attention kernels including FlexAttention, FlashAttention, xFormers, and NATTEN.

## Key Experimental Results

### Attention Kernel Efficiency

| Attention Type | Input | Window | Empty-Block Ratio | Forward Time | Speedup |
|---|---|---|---|---|---|
| WSA (Flex) | 96×96 | 16×16 | 83.3% | 2.63 ms | 1× |
| **HWA (Flex)** | 96×96 | 16×16 | **97.2%** | **0.40 ms** | **6.6×** |
| SA (Flex) | 64×64 | 7×7 | ~low | slow | 1× |
| **HSA (Flex)** | 64×64 | 7×7 | ~high | fast | **~18×** |

### End-to-End Models (ImageNet)

| Model | Params | Top-1 Acc | Throughput | Notes |
|---|---|---|---|---|
| Swin-T | 28M | 81.3% | baseline | original Swin |
| **HWT-T** | 28M | 81.0% | **higher** | −0.3% accuracy |
| NAT-Mini | — | 81.8% | baseline | original NAT |
| **HNT-Mini** | — | 81.5% | **higher** | −0.3% accuracy |

### Key Findings
- Hilbert reordering raises the empty-block ratio from 83–91% to 96–98%, approaching the theoretical upper bound.
- Larger windows or longer sequences yield greater speedups, as the gap in empty-block ratios becomes more pronounced.
- Accuracy degradation is only 0.2–0.3%, indicating that Hilbert windows, despite their irregular shapes, preserve spatial proximity well.
- HSA achieves the most substantial speedup (~18×), since sliding-window attention is particularly fragmented under row-major ordering.

## Highlights & Insights
- **Elegant application of space-filling curves**: The locality-preserving property of the Hilbert curve aligns perfectly with the requirements of block-sparse optimization, representing a principled combination of mathematical structure and systems-level efficiency.
- **Zero-intrusion token reordering**: Inserting Hilbert reordering does not alter the logical semantics of attention (the same spatial neighborhoods are attended to); it only changes the physical memory layout to suit block-sparse kernels, requiring no modification to existing model architectures.
- **Killer application for FlexAttention**: HWA, HSA, and HNA can all be expressed via FlexAttention's `mask_mod`/`score_mod` interfaces without custom kernel development, demonstrating the power of programmable sparse attention frameworks.

## Limitations & Future Work
- The global RPB introduces more parameters than per-window RPB, potentially causing overfitting in smaller models.
- The Hilbert curve is defined only for power-of-two resolutions; non-square or non-$2^n$ input sizes require padding.
- Shifted windows on the Hilbert sequence may group non-adjacent tokens into the same window, necessitating additional masking.
- Speedups are contingent on FlexAttention and similar frameworks; performance may vary across hardware platforms and CUDA versions.

## Related Work & Insights
- **vs. Swin Transformer**: HWT accelerates Swin's WSA via Hilbert reordering with less than 0.3% accuracy loss.
- **vs. NAT/NATTEN**: HNT reformulates 2D neighborhood attention as a 1D operation, enabling direct use of 1D-optimized kernels.
- **vs. FlashAttention**: FlashAttention targets dense attention; HWA/HSA exploit sparsity for additional acceleration beyond what dense-optimized kernels can achieve.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of Hilbert curves and block-sparse attention is both novel and practically motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple attention types, kernel backends, and end-to-end models.
- **Writing Quality**: ⭐⭐⭐⭐ Figures are clear and the speedup mechanism is explained intuitively.
- **Value**: ⭐⭐⭐⭐ Offers direct practical benefit for high-resolution Vision Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compositional Diffusion with Guided Search for Long-Horizon Planning](compositional_diffusion_long_horizon_planning.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](../../AAAI2026/others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](../../AAAI2026/others/approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)

</div>

<!-- RELATED:END -->

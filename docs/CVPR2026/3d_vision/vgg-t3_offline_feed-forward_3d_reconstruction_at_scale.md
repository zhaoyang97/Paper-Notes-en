---
title: >-
  [Paper Note] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale
description: >-
  [CVPR 2026][3D Vision][3D Reconstruction] This paper proposes VGG-T3, which compresses the variable-length KV representations in VGGT's global attention layers into fixed-size MLP weights via **test-time training (TTT)**…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Reconstruction"
  - "Test-Time Training"
  - "Linear Complexity"
  - "KV Compression"
  - "Visual Localization"
date: 2026-05-08
content_hash: bf37ef806ebae6e1
---

# VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale

**Conference**: CVPR 2026
**arXiv**: [2602.23361](https://arxiv.org/abs/2602.23361)
**Code**: N/A
**Area**: 3D Vision / 3D Reconstruction
**Keywords**: 3D Reconstruction, Test-Time Training, Linear Complexity, KV Compression, Visual Localization

## TL;DR

This paper proposes VGG-T3, which compresses the variable-length KV representations in VGGT's global attention layers into fixed-size MLP weights via **test-time training (TTT)**, reducing the computational complexity of offline feed-forward 3D reconstruction from $O(n^2)$ to $O(n)$, enabling large-scale scene reconstruction at the thousand-image level (1k images in only 58 seconds).

## Background & Motivation

**Background**: Feed-forward multi-view 3D reconstruction methods (e.g., VGGT, Fast3R) leverage Transformer global self-attention for multi-view reasoning, achieving accuracy comparable to the classical COLMAP pipeline with greater robustness under challenging conditions.

**Limitations of Prior Work**: The computational complexity and memory requirements of these methods scale **quadratically** with the number of input images $n$, with the core bottleneck being that global softmax attention must query a variable-length KV space spanning all image tokens. VGGT requires over 11 minutes to process 1k images.

**Key Challenge**: Existing acceleration methods (e.g., token merging in FastVGGT, sparse attention in SparseVGGT) reduce constant factors but leave the **asymptotic complexity quadratic**: $O(n^2) \to O(n/r)^2$. Online methods (e.g., CUT3R, Must3R) use fixed-size implicit memories but suffer from limited accuracy and drift.

**Goal**: Reduce complexity to linear $O(n)$ while preserving the accuracy advantage of global offline reconstruction, supporting reconstruction from arbitrarily large image collections.

**Key Insight**: Inspired by DeepSDF — compressing variable-length representations into fixed-size optimizable parameters. The variable-length KV space of VGGT's global attention layers is distilled into the weights of a fixed-size MLP via TTT.

**Core Idea**: TTT is used to learn an MLP $T_\theta$ that captures the Key-to-Value mapping ($\arg\min_\theta \sum_i L_t(T_\theta(k_i) - v_i)$); at inference time, querying this MLP yields the output with complexity linear in sequence length.

## Method

### Overall Architecture

VGG-T3 retains VGGT's image tokenizer and prediction heads, replacing only all **global attention layers** with TTT layers. The process consists of two stages:
- **Update stage**: Input tokens are projected into QKV; TTT compresses the KV mapping into fixed-size MLP weights $\theta$.
- **Apply stage**: The optimized MLP is applied to query $q$ to obtain output tokens, which are passed to the next layer.

### Key Designs

#### 1. Linearizing the Pretrained Model

- **Function**: Initialized from VGGT pretrained weights, retaining the $W_q, W_k, W_v$ projection matrices.
- **Mechanism**: VGGT's QK projections use LayerNorm ($q_i = \text{LN}_q(W_q x_i)$), but the learnable parameters of LN **distort the input space** during TTT optimization, causing extremely slow convergence. Replacing LN with $L_2$ normalization unlocks fast convergence.
- **Design Motivation**: Post-training linearization strategies have proven successful in LLMs and can significantly reduce training costs.

#### 2. ShortConv2D Nonlinear Spatial Mixing

- **Function**: Applies 2D convolution in the Value space to break the linear dependency in the K→V mapping.
- **Mechanism**: Since both $K = W_k x$ and $V = W_v x$ are linear projections of $x$, theoretically $V = W_v W_k^{-1} K$, and the TTT objective may yield a trivial solution. After applying ShortConv2D, the target becomes learning $K \to V'$, where $V'$ encodes local spatial context:
    - Reshape the 1D token sequence into a 2D image grid of shape $(N, H/p, W/p, d)$
    - Apply a single-layer 2D convolution to aggregate local neighborhood information
    - Flatten back to a 1D sequence
- **Design Motivation**: Forces the MLP to predict targets containing neighborhood information from single-token features, yielding a more robust geometric scene representation.

#### 3. Test-Time Scaling

- **Function**: Handles large-scale image collections beyond the training distribution.
- **Key Findings**: Training typically requires only 1 optimization step, but processing 1k images necessitates increasing the number of steps. Simply increasing to 2 steps achieves near-constant generalization across sequence lengths.
- **Design Motivation**: A fixed number of optimization steps is insufficient to compress significantly larger scenes into a fixed-dimensional MLP.

### Loss & Training

Dot product loss is adopted for TTT optimization:
$$L_t(T_\theta(k_i), v_i) = T_\theta(k_i)^T v_i$$

The Muon optimizer is used with a SwiGLU MLP as the fast weight network. All original VGGT parameters are frozen; only the global attention layers are fine-tuned for 100k steps (approximately 12% of the cost of training VGGT from scratch).

## Key Experimental Results

### Main Results: Standard Benchmarks

| Method | Complexity | DTU CD↓ | ETH3D CD↓ | NRGBD-D CD↓ | 7scenes-D NC↑ |
|------|--------|---------|-----------|-------------|---------------|
| VGGT | $O(n^2)$ | 1.537 | 0.279 | 0.014 | 0.668 |
| SparseVGGT | $O(n^2)$ | 1.541 | 0.327 | 0.018 | 0.665 |
| TTT3R | $O(n)$ | 5.708 | 0.885 | 0.071 | 0.666 |
| **VGG-T3** | $O(n)$ | **1.654** | **0.480** | **0.029** | **0.679** |

- Pointmap estimation: **substantially outperforms** the only $O(n)$ baseline TTT3R across all datasets (DTU error reduced by 2–2.5×), remaining competitive with $O(n^2)$ methods.
- Video depth estimation: $\delta<1.25$ of 0.967 on KITTI, on par with $O(n^2)$ methods.

### Large-Scale Reconstruction Performance

| No. of Images | VGG-T3 | VGGT | FastVGGT | TTT3R |
|----------|--------|------|----------|-------|
| 1k | **58s** | 11min (11.6× slower) | 4min (4.3× slower) | ~60s |
| 2k (4 GPUs) | **48.5s** | 1590s | N/A | N/A |

### Ablation Study

| Design | DTU CD↓ | ETH3D CD↓ |
|------|---------|-----------|
| w/o ShortConv2D | Significant degradation | Significant degradation |
| LayerNorm instead of L2 Norm | Extremely slow convergence | — |
| 1-step TTT (1k images) | Error increases ~5× | — |
| 2-step TTT (1k images) | Approaches small-scale accuracy | Stable |

### Key Findings

1. The quality gap between VGG-T3 and $O(n^2)$ methods **narrows as the number of images increases**.
2. Supports single-GPU processing of arbitrarily large image collections (via minibatch offloading to CPU) as well as multi-GPU distributed inference.
3. Visual localization: freezing the TTT-MLP enables feed-forward localization, achieving $e_r=6.71°, e_t=0.15\text{m}$ on 7scenes.

## Highlights & Insights

1. **Elegant core insight**: Treating the KV space in attention as a "variable-length scene representation" and compressing it into a "fixed-size scene representation" via TTT — a natural and profound analogy to DeepSDF.
2. **Practical large-scale solution**: The additivity of the TTT objective (gradients can be accumulated in minibatches) naturally supports distributed inference and CPU offloading, which softmax attention cannot achieve.
3. **Unified reconstruction and localization**: The same model and TTT-MLP support both scene mapping and localization, opening a new path toward unified end-to-end solutions.
4. **Low fine-tuning cost**: The majority of VGGT parameters are frozen; only the new parameters in global attention layers are trained, at approximately 12% of the cost of training from scratch.

## Limitations & Future Work

1. **Weak camera pose estimation**: The TTT-linearized model performs poorly on pose estimation, likely related to the heterogeneous design of camera tokens in VGGT; this remains a key issue for future work.
2. **Remaining gap with softmax attention**: Especially in wide-baseline settings, the fixed capacity of the MLP limits scene representation ability.
3. **Non-trivial training cost**: Although only 12% of VGGT's cost, training still requires 8×A100-80GB GPUs for 100k steps.
4. **Limited visual localization validation**: Demonstrated only on 7scenes and Wayspots, with a notable gap compared to dedicated localization pipelines (e.g., Reloc3R).

## Related Work & Insights

- **VGGT**: The base architecture for this work; global softmax attention enables multi-view reasoning with high accuracy but $O(n^2)$ complexity.
- **FastVGGT / SparseVGGT**: Accelerate via token merging / block-sparse attention; asymptotic complexity remains unchanged.
- **TTT3R**: A concurrent work; an autoregressive TTT model based on CUT3R; $O(n)$ but lower accuracy and does not support unordered inputs.
- **CUT3R / Must3R / Point3R**: Online methods using fixed-size implicit/spatial memories; linear complexity but poor global consistency.
- **LaCT (Sun et al.)**: The originator of the TTT framework adopted here; VGG-T3 uses its SwiGLU MLP and Muon optimizer.
- **DeepSDF**: A classic implicit representation work; the central idea of "encoding instance geometry in a fixed-size network" directly underpins VGG-T3.

## Rating

- Novelty: ⭐⭐⭐⭐ — Elegantly transfers post-training linearization and TTT from the LLM domain to 3D reconstruction; the ShortConv2D design is targeted and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers pointmap estimation, depth, pose, and localization; includes large-scale evaluation and distributed inference; ablations are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Logically clear with progressively developed motivation; figures and tables are highly informative.
- Value: ⭐⭐⭐⭐⭐ — Addresses the scalability bottleneck of feed-forward 3D reconstruction; achieves 11.6× speedup with minimal accuracy loss; has direct practical value for large-scale scene reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [\[CVPR 2026\] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)
- [\[CVPR 2026\] Particulate: Feed-Forward 3D Object Articulation](particulate_feed-forward_3d_object_articulation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Speed3R: Sparse Feed-forward 3D Reconstruction Models
description: >-
  [CVPR 2026][3D Vision][3D Reconstruction] Speed3R introduces a trainable dual-branch Global Sparse Attention (GSA) mechanism for feed-forward 3D reconstruction models. A compression branch provides coarse-grained scene s…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Reconstruction"
  - "Sparse Attention"
  - "Feed-forward"
  - "Inference Acceleration"
  - "Structure-from-Motion"
date: 2026-05-08
content_hash: 54b65a7bb34430d9
---

# Speed3R: Sparse Feed-forward 3D Reconstruction Models

**Conference**: CVPR 2026
**arXiv**: [2603.08055](https://arxiv.org/abs/2603.08055)
**Code**: [https://visual-ai.github.io/speed3r/](https://visual-ai.github.io/speed3r/)
**Area**: 3D Vision
**Keywords**: 3D Reconstruction, Sparse Attention, Feed-forward, Inference Acceleration, Structure-from-Motion

## TL;DR

Speed3R introduces a trainable dual-branch Global Sparse Attention (GSA) mechanism for feed-forward 3D reconstruction models. A compression branch provides coarse-grained scene summaries while a selection branch focuses fine-grained attention on critical tokens, achieving **12.4× inference speedup** on 1000-view sequences with only marginal accuracy degradation.

## Background & Motivation

**Background**: Recent feed-forward 3D reconstruction models (VGGT, $\pi^3$) can jointly infer dense geometry and camera poses in a single forward pass, bypassing the multi-stage pipelines of classical SfM/MVS.

**Limitations of Prior Work**: These models rely on dense global attention, whose computational cost scales as $O(n^2)$ with the number of tokens. Inference speed becomes a severe bottleneck at large view counts or high resolutions — for example, $\pi^3$ requires **202 seconds** to process 1024 images.

**Key Challenge**: Training-free approaches such as FastVGGT (token merge-unmerge) and Block-Sparse VGGT (top-k attention) cannot be optimized end-to-end, and aggressive pruning leads to significant accuracy degradation.

**Core Insight**: The classical SfM intuition — that sparse keypoints suffice for robust pose estimation — has not yet been fully exploited by feed-forward methods.

**Goal**: Motivated by both SfM and sparse attention in LLMs (NSA, MOBA), this work designs end-to-end trainable sparse attention and transfers dense model performance via knowledge distillation.

## Method

### Overall Architecture

Speed3R adopts a three-stage architecture:

1. **Per-frame Feature Encoder**: $N$ images $\{I_i\}_{i=1}^N$ are independently processed by a visual encoder (e.g., DINOv2) to extract patch-level feature tokens.
2. **Alternating Attention Transformer**: Multiple Transformer blocks alternate between local intra-frame attention (Frame Attention) and Global Sparse Attention (**GSA**, the core contribution), replacing the dense global attention in the original models.
3. **Task-specific Prediction Heads**: Refined tokens are fed into downstream heads to predict per-view camera parameters $\{\hat{C_i}\}$, depth maps $\{\hat{D_i}\}$, and associated uncertainties $\{\hat{\alpha_i}\}$.

### Key Designs: Global Sparse Attention (GSA)

The core idea of GSA is **coarse-to-fine**: first build a global scene understanding from low-resolution representations, then guide the model to attend only to the most informative token subset in high-resolution space.

**Input Decomposition**: The GSA input $X \in \mathbb{R}^{M \times C}$ is formed by concatenating special tokens $X_{\text{spec}}$ and image tokens $X_{\text{img}}$. Projections $W_Q, W_K, W_V$ generate Q/K/V, which are split by token type:

$$Q = \begin{bmatrix} Q_{\text{spec}} \\ Q_{\text{img}} \end{bmatrix}, \quad K = \begin{bmatrix} K_{\text{spec}} \\ K_{\text{img}} \end{bmatrix}, \quad V = \begin{bmatrix} V_{\text{spec}} \\ V_{\text{img}} \end{bmatrix}$$

**Full Attention for Special Tokens**: Special tokens (e.g., pose tokens) serve as global information bottlenecks for critical tasks such as pose estimation, and attend to all tokens via standard dense self-attention:

$$O_{\text{spec}} = \text{softmax}\left(\frac{Q_{\text{spec}} K^T}{\sqrt{d_k}}\right) V$$

Since $M_{\text{spec}}$ is small, this step incurs negligible overhead.

**Dual-branch Sparse Attention for Image Tokens**: The large number of image tokens is handled via a dual-branch strategy.

#### Compression Branch

Provides an efficient coarse-grained global scene summary. $Q_{\text{img}}, K_{\text{img}}, V_{\text{img}}$ are spatially downsampled using $s \times s$ non-overlapping average pooling, yielding compressed tensors $Q_{\text{comp}}, K_{\text{comp}}, V_{\text{comp}} \in \mathbb{R}^{M'_{\text{img}} \times d}$ where $M'_{\text{img}} = M_{\text{img}} / s^2$.

Attention is computed in the compressed space:

$$O'_{\text{comp}} = \text{Attention}(Q_{\text{comp}}, K_{\text{comp}}, V_{\text{comp}})$$

A guidance score matrix is also computed for use by the selection branch:

$$S_{\text{guide}} = Q_{\text{comp}} K_{\text{comp}}^T \in \mathbb{R}^{M'_{\text{img}} \times M'_{\text{img}}}$$

The coarse output is upsampled back to the original resolution via nearest-neighbor interpolation: $O_{\text{comp}} = \text{Upsample}(O'_{\text{comp}})$.

#### Selection Branch

Recovers fine-grained attention. Using the guidance scores $S_{\text{guide}}$, $\text{TopKSelect}(\cdot)$ identifies the most relevant coarse-region indices for each query. The corresponding $K_{\text{sel}}, V_{\text{sel}}$ are then retrieved from the full-resolution $K_{\text{img}}, V_{\text{img}}$ (queries within the same compression window share the same KV pairs):

$$O_{\text{sel}} = \text{Attention}(Q_{\text{img}}, K_{\text{sel}}, V_{\text{sel}})$$

Each query attends to only $k \ll M_{\text{img}}$ tokens, making this step highly efficient.

#### Gated Aggregation

The outputs of both branches are dynamically fused via a learnable gating mechanism:

$$g = \sigma(W_g Q_{\text{img}}), \quad O_{\text{img}} = g \odot O_{\text{comp}} + (1 - g) \odot O_{\text{sel}}$$

where $\sigma$ denotes sigmoid and $W_g$ is a learned projection matrix. The model **adaptively determines** for each token whether to emphasize the global summary or local detail.

### Efficient Triton Kernel Implementation

A naïve implementation would materialize the full $S_{\text{guide}}$ score matrix, incurring excessive memory usage. A fused GSA Triton kernel is developed that integrates a streaming Top-K algorithm into the FlashAttention workflow: score matrix tiles are computed on-chip in SRAM while simultaneously maintaining running top-k index sets, completing region selection and compressed output computation in a single pass without materializing the full score matrix.

### Speed3R-VGGT Adaptation

VGGT treats the first frame as a global reference and employs dedicated camera tokens. To preserve reference-frame information, the selection branch attention set consists of two components:

- **Fixed global context**: all tokens from the reference frame plus tokens from every 100th frame
- **Dynamic Top-K tokens**: key token windows from non-reference frames determined by the standard selection procedure

### Speed3R-$\pi^3$ Adaptation

$\pi^3$ has no reference frame or camera token dependencies, so GSA can be applied directly. Experiments show that the register tokens in $\pi^3$ can be removed in the sparse variant without affecting performance, further simplifying the model.

### Loss & Training

- **Knowledge Distillation**: A pretrained dense model serves as the teacher; its depth and pose predictions are used as pseudo-labels to train the student (sparse model).
- **Total Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{depth}} + \lambda \mathcal{L}_{\text{camera}}$
- **Data**: A mixture of 7 datasets (ArkitScene, Scannet++, DL3DV, CO3D, Hypersim, WildRGBD, VirtualKitti2)
- **Training Configuration**: 80 epochs (800 steps per epoch), 8× NVIDIA H20 GPUs for approximately 7 days, learning rate $1 \times 10^{-5}$, gradient accumulation factor=4 (effective batch size 32)

## Key Experimental Results

### Main Results: Multi-view Pose Estimation (RE10K / CO3Dv2)

| Method | Sparsity (%) | RE10K AUC@30↑ | CO3Dv2 AUC@30↑ |
|--------|--------------|---------------|----------------|
| VGGT (dense) | 0 | 74.17 | 88.33 |
| Block Sparse-VGGT | 75 | 63.82 | 79.92 |
| FastVGGT | 82 | 69.99 | 84.03 |
| **Speed3R-VGGT** | **84** | **74.81** | **87.71** |
| $\pi^3$ (dense) | 0 | 87.37 | 89.67 |
| Block Sparse-$\pi^3$ | 75 | 75.39 | 80.72 |
| FastVGGT-$\pi^3$ | 90 | 86.04 | 86.39 |
| **Speed3R-$\pi^3$** | **94** | **87.17** | **89.41** |

**Key Findings**:

- Speed3R-VGGT at 84% sparsity **surpasses the dense VGGT baseline** on RE10K (74.81 vs. 74.17)
- Speed3R-$\pi^3$ at 94% sparsity nearly matches the performance of dense $\pi^3$
- Consistently outperforms training-free competitors at all sparsity levels

### Long-sequence Pose Estimation (Tanks & Temples, avg. 300 images/scene)

| Method | RRA@5↑ | RTA@5↑ | AUC@30↑ | Time (s)↓ |
|--------|--------|--------|---------|-----------|
| VGGT (dense) | 70.29 | 79.30 | 77.67 | 34.51 |
| Block Sparse-VGGT | 66.83 | 71.29 | 74.15 | 10.79 |
| FastVGGT | 69.28 | 77.98 | 76.29 | 15.98 |
| **Speed3R-VGGT** | **69.51** | **77.81** | **76.57** | **6.55** |
| $\pi^3$ (dense) | 72.14 | 81.26 | 79.63 | 22.32 |
| Block Sparse-$\pi^3$ | 67.85 | 78.91 | 76.64 | 8.16 |
| FastVGGT-$\pi^3$ | 69.78 | 79.51 | 77.76 | 11.96 |
| **Speed3R-$\pi^3$** | **70.72** | **80.72** | **79.77** | **4.19** |

**Key Findings**: Speed3R-$\pi^3$ achieves the best results among sparse methods on all metrics while being the fastest (4.19s), **5.3×** faster than dense $\pi^3$.

### Ablation Study (Speed3R-$\pi^3$, T&T Dataset)

| Configuration | RE10K AUC@30↑ | T&T AUC@30↑ | Time (s)↓ |
|---------------|---------------|-------------|-----------|
| Base (4×4 window, top-32) | 86.35 | 78.69 | 4.19 |
| (1) Remove compression branch Value | 86.29 | 77.90 | 3.99 |
| (2) Remove selection branch | 83.44 | 76.84 | 3.56 |
| (4) Top-8 | 85.37 | 78.17 | 3.72 |
| (5) Top-16 | 85.98 | 78.55 | 3.92 |
| (6) Top-64 | 86.42 | 78.90 | 4.64 |
| (7) 8×8 window | 86.49 | 78.71 | 5.27 |
| (8) Without knowledge distillation | 85.18 | 77.81 | 4.19 |

**Key Findings**:

- **Selection branch is essential**: Removing it causes large drops on both datasets (RE10K −2.91, T&T −1.85)
- **Compression branch matters for long sequences**: Removing its Value has negligible effect on short sequences but hurts long sequences (T&T −0.79)
- **Knowledge distillation is critical**: Removing it reduces RE10K by 1.17 and T&T by 0.88, demonstrating its effectiveness in mitigating noisy labels in real-world datasets
- **4×4 window + top-32 is the optimal trade-off**: top-8/16 are insufficient in accuracy; top-64 and 8×8 windows offer marginal gains at higher cost

### Inference Latency Comparison

| Sequence Length | 32 | 64 | 128 | 256 | 512 | 1024 |
|-----------------|-----|------|------|------|------|-------|
| Full Attn. ($\pi^3$) | 0.50s | 1.31s | 3.97s | 13.41s | 50.01s | 202.39s |
| Block Sparse | 0.46s | 0.85s | 1.69s | 3.77s | 9.64s | 29.58s |
| FastVGGT | 0.44s | 0.88s | 1.96s | 4.95s | 14.13s | 45.49s |
| **Speed3R** | **0.37s** | **0.71s** | **1.44s** | **3.06s** | **6.83s** | **16.38s** |

At 1024 images, Speed3R requires only 16.38s vs. 202.39s for the dense model, yielding a **12.4× speedup**.

### Test-time Adaptation (Tanks & Temples)

Training uses top-32; increasing top-k at inference continuously improves long-sequence performance. At top-128, RTA@5 reaches 82.00, **surpassing the dense model** (81.26), and AUC@30 reaches 80.33, also exceeding the dense model (79.63), with a runtime of only 6.07s.

## Highlights & Insights

- **Bridging Classical and Modern Methods**: Combines the SfM insight that sparse keypoints suffice for robust pose estimation with LLM sparse attention techniques, yielding a trainable sparse attention mechanism tailored to 3D reconstruction.
- **Coarse-to-fine Dual-branch Design**: The compression branch establishes global understanding, which then guides the selection branch to focus on critical regions, balancing global coverage with local precision.
- **End-to-end Trainability**: Offers a significant advantage over training-free methods such as FastVGGT and Block-Sparse through joint optimization.
- **General Plug-and-play Applicability**: Successfully adapted to both VGGT and $\pi^3$ architectures, demonstrating strong generalizability.
- **Custom Triton Kernel**: Fuses Top-K with FlashAttention for efficient memory access, avoiding materialization of the full score matrix.

## Limitations & Future Work

1. **Short-sequence Accuracy Gap**: A performance gap relative to dense models remains under strict thresholds (AUC@5), as high-precision pose regression is particularly challenging for sparse methods.
2. **Memory Overhead**: The dual-branch GSA architecture incurs a **15% memory overhead** compared to full attention, limiting a single 80GB GPU to at most 1024 images.
3. **Dependence on Pretrained Dense Models**: The knowledge distillation strategy requires a high-quality dense teacher, increasing training pipeline complexity.
4. **Pose Regression vs. Generative Tasks**: The high numerical precision demanded by pose regression makes 3D reconstruction less amenable to sparse attention than text or image generation.

## Rating

⭐⭐⭐⭐ — The first trainable sparse attention method targeting feed-forward 3D reconstruction, with a practically significant 12.4× speedup, an elegant dual-branch design, and thorough ablations. However, accuracy gaps remain under strict short-sequence metrics, and the approach is inherently constrained by the high-precision demands of pose regression in 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [\[CVPR 2026\] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)
- [\[CVPR 2026\] Particulate: Feed-Forward 3D Object Articulation](particulate_feed-forward_3d_object_articulation.md)

</div>

<!-- RELATED:END -->

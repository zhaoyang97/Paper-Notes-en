---
title: >-
  [Paper Note] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention
description: >-
  [ICCV 2025][Image Restoration][ToF depth denoising] GIGA-ToF proposes a ToF depth denoising network that fuses motion-invariant graph structures across frames. Through cross-frame graph attention and algorithm unrolling…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "ToF depth denoising"
  - "graph signal processing"
  - "geometric attention"
  - "algorithm unrolling"
  - "temporal consistency"
date: 2026-05-08
content_hash: 6e0f1908f4ab8658
---

# Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention

**Conference**: ICCV 2025
**arXiv**: [2506.23542](https://arxiv.org/abs/2506.23542)
**Code**: [github.com/davidweidawang/GIGA-ToF](https://github.com/davidweidawang/GIGA-ToF)
**Area**: Depth Map Denoising / Computational Imaging
**Keywords**: ToF depth denoising, graph signal processing, geometric attention, algorithm unrolling, temporal consistency

## TL;DR

GIGA-ToF proposes a ToF depth denoising network that fuses motion-invariant graph structures across frames. Through cross-frame graph attention and algorithm unrolling of a MAP problem, the method simultaneously improves temporal stability and spatial sharpness, demonstrating strong generalization on both synthetic and real data.

## Background & Motivation

Continuous-wave Time-of-Flight (ToF) sensors are widely used in robotics, 3D reconstruction, and augmented reality due to their real-time response and low power consumption. However, ToF depth maps suffer from severe noise in regions with long range, low reflectivity, or specular surfaces.

Existing denoising methods exhibit three key limitations:

**Single-frame methods ignore temporal correlations**: Most DNN-based methods (e.g., ToFNet, GLRUN) process only a single frame, failing to exploit inter-frame information, which leads to temporal jitter and insufficient denoising.

**Spatial blurring in multi-frame methods**: Existing multi-frame methods (e.g., MTDNet, DVSR) fuse depth features from corresponding pixels by estimating scene flow or inter-frame correlations. However, due to camera motion, the depth values of the same object differ across frames, and directly fusing depth features causes spatial blurring and loss of detail.

**Poor generalization to real data**: Purely data-driven DNN approaches trained on synthetic data suffer significant performance degradation when applied to real noise, as ground-truth annotations for real data are difficult to obtain.

The core insight of this paper is: **although depth values change with motion, the graph structure among neighboring pixels (i.e., the correlation pattern) exhibits temporal self-similarity — it encodes object shape rather than absolute depth.** Fusing motion-invariant graph structures instead of depth features therefore simultaneously addresses spatial blurring and temporal inconsistency.

## Method

### Overall Architecture

The GIGA-ToF network consists of three components:
1. **Feature extraction network** (blue): An encoder-decoder structure that extracts multi-scale geometric features $\mathbf{F}^t$, estimates initial prior weights, and computes intra-frame graph adjacency matrices.
2. **Graph-Informed Geometric Attention (GIGA) module** (yellow): Learns graph edge weights via cross-frame attention to enable motion-invariant graph structure fusion.
3. **Unrolled GLR module** (green): Unrolls the iterative solution of the MAP optimization problem into learnable filtering layers.

### Key Designs

1. **Intra-frame graph modeling and cross-frame graph fusion**:

    - An 8-connected undirected graph $\mathcal{G}^t$ is constructed for each ToF frame, with adjacency matrix $\mathbf{W}^t$ encoding inter-pixel correlations.
    - An inter-frame graph $\mathbf{W}^{t,t-1}$ is constructed to map the graph structure of the reference frame $t-1$ to the current frame $t$ via 2-hop and 3-hop paths.
    - The mapped graph is computed as: $\hat{\mathbf{W}}^{t-1} = \mathbf{W}^{t,t-1}(\mathbf{W}^{t-1} + \mathbf{I})(\mathbf{W}^{t,t-1})^\top$
    - A confidence matrix $\boldsymbol{\Phi}^{t,t-1}$ is used for weighted fusion to obtain the final graph: $\widetilde{\mathbf{W}}^t = \boldsymbol{\Phi}^{t,t-1}\hat{\mathbf{W}}^{t-1} + \mathbf{W}^t$
    - Design motivation: Graph structure reflects object shape rather than absolute depth, and is therefore motion-invariant, avoiding the spatial blurring caused by depth feature fusion.

2. **MAP problem formulation**:

    - **Data fidelity term**: The likelihood function is derived from the ToF depth noise distribution: $\ln P(\mathbf{n}_d^t) \approx -\frac{1}{2\sigma^2}\|(\mathbf{X}_a^t)^{-1}(\mathbf{x}_q^t \odot \mathbf{y}_i^t - \mathbf{x}_i^t \odot \mathbf{y}_q^t)\|_2^2$
    - **Graph smoothness prior**: Graph Laplacian regularization (GLR) is imposed on the fused graph to constrain the smoothness of the denoised output over the graph.
    - Design motivation: Embedding the ToF imaging model (noise distribution) and signal prior (graph smoothness) into the network design enhances interpretability and cross-dataset generalization.

3. **Algorithm unrolling and graph learning**:

    - The alternating optimization of the MAP problem is unrolled into differentiable iterative filtering layers.
    - At each iteration, the current estimate is convolved via the fused graph adjacency matrix and weighted-fused with the input.
    - Graph edge weights are learned end-to-end via the GIGA attention mechanism: intra-frame graphs are estimated via single-layer convolution, and inter-frame graphs are computed via query-key attention.
    - Design motivation: Algorithm unrolling explicitly parameterizes the filter kernel $\widetilde{\mathbf{W}}$ as a learnable quantity, retaining the low-pass interpretability of graph spectral filtering while adaptively learning optimal parameters via DNN.

### Loss & Training

An L1 loss is applied to supervise the denoising results for both the in-phase and quadrature components:

$$L = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \sum_{\theta \in \{i,q\}} |\mathbf{x}_\theta^{t,*}(v) - \mathbf{x}_\theta^{t,\text{gt}}(v)|$$

Training configuration: Adam optimizer (initial lr=1e-3, decayed by 0.7 at epochs 15/30/45), trained for 60 epochs, with T=3 frames and R=2 unrolling iterations.

## Key Experimental Results

### Main Results — Denoising Accuracy on DVToF Synthetic Dataset

| Method | Type | MAE (m) ↓ | AbsRel ↓ | δ₁ ↑ | TEPE (m) ↓ |
|--------|------|-----------|----------|------|-----------|
| libfreenect2 | Single-frame / Traditional | 0.1044 | 0.0283 | 0.9746 | 0.1023 |
| GLRUN | Single-frame / DNN | 0.0357 | 0.0107 | 0.9929 | 0.0734 |
| WMF | Multi-frame / Traditional | 0.0311 | 0.0116 | 0.9955 | 0.0751 |
| MTDNet | Multi-frame / DNN | 0.0566 | 0.0642 | 0.9816 | 0.1046 |
| DVSR | Multi-frame / DNN | 0.0718 | 0.0844 | 0.9777 | 0.1176 |
| **GIGA-ToF** | **Multi-frame / DNN** | **0.0193** | **0.0060** | **0.9974** | **0.0637** |

GIGA-ToF outperforms the second-best method by at least **37.9%** in MAE and at least **13.2%** in TEPE.

### Ablation Study — Contribution of Each Component

| GLR | Fusion Strategy | Attention | MAE (m) ↓ | TEPE (m) ↓ |
|-----|----------------|-----------|-----------|-----------|
| ✗ | ✗ | ✗ | 0.0409 | 0.0793 |
| Unrolled | ✗ | ✗ | 0.0357 | 0.0734 |
| Unrolled | Feature fusion | ✗ | 0.0238 | 0.0718 |
| Unrolled | Feature fusion | ✓ | 0.0214 | 0.0713 |
| Unrolled | **Graph fusion** | ✗ | 0.0219 | 0.0702 |
| Unrolled | **Graph fusion** | **✓** | **0.0193** | **0.0637** |

Each component contributes clearly: unrolled GLR preserves detail; graph structure fusion outperforms feature fusion and resolves spatial blurring; the attention mechanism ensures accurate inter-frame correspondence.

### Key Findings

1. **Graph structure fusion > Feature fusion**: Graph fusion outperforms feature fusion on all metrics; qualitative results also show sharper edges and less blurring.
2. **Strong cross-dataset generalization**: A model trained on synthetic DVToF data applies directly to real Kinect v2 data, still producing accurate and smooth depth maps, while the purely data-driven MTDNet fails completely.
3. **Robustness to inter-frame stride**: Even at large temporal strides of Δt=8, multi-frame processing outperforms single-frame processing, validating the temporal self-similarity of graph structures.

## Highlights & Insights

- The insight that "graph structure is motion-invariant" is particularly elegant — shifting focus from pixel values to topological relationships fundamentally resolves the spatial blurring problem in multi-frame depth fusion.
- Combining the ToF physical imaging model (noise distribution) with graph signal processing priors via algorithm unrolling yields a network that is both high-performing and interpretable.
- The computational efficiency is moderate (0.027s/frame), far superior to traditional WMF (24.3s/frame), making the method practically deployable.

## Limitations & Future Work

- Only the previous frame is considered as the reference frame; information from earlier frames is not fully exploited.
- The neighborhood size for inter-frame graphs is fixed at q=7, which may not be suitable for all motion magnitudes.
- Quantitative evaluation is conducted only on synthetic data; real-data evaluation remains qualitative only.

## Related Work & Insights

- Relationship to GLRUN: The proposed method inherits its algorithm unrolling framework but extends it from single-frame to multi-frame processing and introduces the core graph fusion mechanism.
- Implications for other video denoising tasks: The observation of motion-invariant graph structure may generalize to other sensor modalities (e.g., event cameras, radar).
- The interpretability of graph spectral filtering provides theoretical grounding for understanding network behavior.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The core insight of motion-invariant graph fusion is highly elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Synthetic + real data, with thorough ablation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous mathematical derivations, clear figures)
- Value: ⭐⭐⭐⭐ (Solid contribution to the ToF depth imaging and graph signal processing communities)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[ACL 2026\] Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness](../../ACL2026/image_restoration/purging_the_gray_zone_latent-geometric_denoising_for_precise_knowledge_boundary_.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICCV 2025\] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](generic_event_boundary_detection_via_denoising_diffusion.md)
- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)

</div>

<!-- RELATED:END -->

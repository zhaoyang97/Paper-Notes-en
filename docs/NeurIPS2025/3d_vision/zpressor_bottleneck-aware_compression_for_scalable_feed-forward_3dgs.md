---
title: >-
  [Paper Note] ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] Grounded in the Information Bottleneck (IB) principle, this paper analyzes the capacity bottleneck of feed-forward 3DGS and proposes ZPressor, a lightweight, architecture-agnostic module that compresses multi-view inputs into a compact anchor-view representation, enabling existing models to scale to 100+ input views (480P, 80GB GPU) with consistent performance gains on DL3DV-10K and RealEstate10K.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Feed-Forward 3DGS
  - Information Bottleneck
  - Multi-View Compression
  - Novel View Synthesis
date: 2026-05-08
content_hash: 20cff4a6ac6cab8f
---

# ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS

**Conference**: NeurIPS 2025
**arXiv**: [2505.23734](https://arxiv.org/abs/2505.23734)
**Code**: [Project Page](https://lhmd.top/zpressor)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Feed-Forward 3DGS, Information Bottleneck, Multi-View Compression, Novel View Synthesis

## TL;DR
Grounded in the Information Bottleneck (IB) principle, this paper analyzes the capacity bottleneck of feed-forward 3DGS and proposes ZPressor, a lightweight, architecture-agnostic module that compresses multi-view inputs into a compact anchor-view representation, enabling existing models to scale to 100+ input views (480P, 80GB GPU) with consistent performance gains on DL3DV-10K and RealEstate10K.

## Background & Motivation
Feed-forward 3DGS methods (e.g., pixelSplat, MVSplat, DepthSplat) represent a significant advance in novel view synthesis. Unlike traditional 3DGS that requires per-scene optimization, feed-forward approaches predict 3DGS parameters in a single forward pass through an encoder, substantially improving practicality.

However, existing feed-forward 3DGS models face a fundamental **scalability bottleneck**: as the number of input views increases, performance degrades rather than improves, while memory consumption grows rapidly. For example, DepthSplat achieves only 19.23 PSNR with 36-view input (compared to 23.32 with 12 views), and pixelSplat runs out of memory beyond 8 views. This issue is not purely an engineering problem — even memory-efficient attention or activation checkpointing cannot resolve the performance degradation.

The authors provide an information-theoretic explanation: the joint entropy of multi-view features $H(\mathbf{F}_1, \mathbf{F}_2, ..., \mathbf{F}_K)$ is not equal to the sum $\sum H(\mathbf{F}_i)$, so a large number of views introduces substantial **information redundancy**. In existing pixel-aligned designs, the number of 3D Gaussians grows linearly with input views, causing representational overload.

The core idea is to apply the IB principle to guide compression: learning a compact latent representation $\mathcal{Z}$ that retains task-relevant information (for NVS) while discarding redundancy. Concretely, input views are divided into anchor views and support views; cross-attention aggregates information from support views into anchor views, producing a compressed latent state $\mathcal{Z}$.

## Method

### Overall Architecture
ZPressor is an architecture-agnostic module that can be inserted after the encoder of any feed-forward 3DGS model. Given features $\mathcal{X} = \{\mathbf{F}_i\}_{i=1}^K$ from $K$ input views, ZPressor compresses them into $\mathcal{Z} = \text{ZPressor}(\mathcal{X})$, which is then fed into the pixel-aligned Gaussian prediction network $\Psi_{pred}$.

### Key Designs
1. **Anchor View Selection**:

    - Farthest Point Sampling (FPS) is applied to select $N$ anchor views from $K$ camera positions.
    - Formula: $\mathbf{T}_{a_{i+1}} = \arg\max_{\mathbf{T}_j \in \mathcal{T} \setminus \mathcal{S}}(\min_{\mathbf{T}_k \in \mathcal{S}} d(\mathbf{T}_j, \mathbf{T}_k))$
    - Design Motivation: Maximizing spatial coverage — anchors must span diverse viewpoints of the scene within a limited count, directly affecting the informational completeness of the compressed representation.

2. **Support-to-Anchor Assignment**:

    - Each support view is assigned to its nearest anchor: $\mathcal{C}_i = \{f(\mathbf{T}) \in \mathcal{X}_{support} \mid \|\mathbf{T} - \mathbf{T}_{a_i}\| \leq \|\mathbf{T} - \mathbf{T}_{a_j}\|, \forall j \neq i\}$
    - Design Motivation: Spatially proximate views contain the most complementary information; assigning them to the same anchor maximizes fusion effectiveness.

3. **Views Information Fusion**:

    - Fusion is implemented via cross-attention: $\mathcal{Z} = \text{Attention}(Q, K, V)$, where $Q \leftarrow \mathcal{X}_{anchor}$, $K, V \leftarrow \mathcal{X}_{support}$.
    - Anchor features serve as queries while support view features provide keys and values, enabling effective injection of support information into anchors.
    - Additional self-attention layers enhance intra-cluster information flow; stacking multiple blocks further improves fusion.
    - Design Motivation: Cross-attention satisfies two key properties: (1) it integrates support information with anchors as the primary subject; (2) it captures inter-group correlations while maintaining compactness and avoiding redundancy. Gradients also flow from the prediction side back to both anchor and support views.

### Loss & Training
- IB-principled training objective: $\mathcal{L} = \mathbb{E}_{\mathcal{Z} \sim p_\theta(\mathcal{Z}|\mathcal{X})}[-\log q_\phi(\mathcal{Y}|\mathcal{Z})] + \beta \mathbb{E}_\mathcal{X}[\text{KL}[p_\theta(\mathcal{Z}|\mathcal{X}), r(\mathcal{Z})]]$
- The prediction term $-\log q_\phi(\mathcal{Y}|\mathcal{Z})$ is modeled by rendering loss (MSE + LPIPS).
- The compression term is controlled by setting the anchor count $N$ to an acceptable training value.
- All baseline training configurations (learning rate, AdamW optimizer, and other hyperparameters) are strictly followed; no additional data or regularization is introduced.
- Training context views: 6 (DepthSplat/MVSplat) or 4 (pixelSplat); anchor count set to 6.

## Key Experimental Results

### Main Results (DL3DV-10K)

| Views | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|--------|-------|-------|--------|
| 36 | DepthSplat | 19.23 | 0.666 | 0.286 |
| 36 | DepthSplat + ZPressor | **23.88 (+4.65)** | **0.815** | **0.150** |
| 24 | DepthSplat | 20.38 | 0.711 | 0.253 |
| 24 | DepthSplat + ZPressor | **24.26 (+3.88)** | **0.820** | **0.147** |
| 12 | DepthSplat | 23.32 | 0.807 | 0.162 |
| 12 | DepthSplat + ZPressor | **24.30 (+0.97)** | **0.821** | **0.146** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Inference Time (s) | Peak Memory (GB) |
|---------------|-------|-------|--------|--------------------|------------------|
| DepthSplat Baseline | 23.32 | 0.808 | 0.162 | 0.260 | 6.80 |
| + ZPressor (Full) | **24.30** | **0.821** | **0.146** | **0.184** | **3.80** |
| + ZPressor (w/o multi-block) | 24.18 | 0.817 | 0.149 | 0.140 | 3.79 |
| + ZPressor (w/o self-attention) | 23.85 | 0.810 | 0.156 | 0.183 | 3.80 |

### Key Findings
- ZPressor's gains are more pronounced with denser input views: +4.65 dB PSNR at 36 views vs. +0.97 dB at 12 views.
- pixelSplat runs out of memory beyond 8 views; with ZPressor it scales to 36 views (PSNR 26.59).
- ZPressor not only improves quality but also **reduces** inference time (0.260s → 0.184s) and peak memory (6.80GB → 3.80GB).
- Information fusion analysis confirms that removing fusion (w/o fusion: PSNR 23.80) or replacing support views with repeated anchors (fuse anchors: PSNR 24.23) both underperform the default (24.30), validating the contribution of support view information.
- Bottleneck constraint analysis: small scene coverage (CG50) is sufficiently handled with 7 anchors; more anchors introduce redundancy. Larger scene coverage (CG100) requires more anchors.

## Highlights & Insights
- Applying the IB principle to analyze capacity issues in feed-forward 3DGS provides a theoretical foundation rather than a purely engineering optimization. This principled methodology enables ZPressor to be adapted across different architectures in an architecture-agnostic manner.
- The result of simultaneously reducing latency, memory, and improving quality is highly compelling — compressing information redundancy not only saves resources but also eliminates the interference of redundant information, improving representation quality.
- The bottleneck constraint analysis revealing the relationship between scene information density and the optimal anchor count is particularly insightful: 7 anchors suffice for small scenes, and adding more introduces redundancy — a perfect demonstration of the compression–preservation trade-off in the IB principle.
- Consistent improvements across three baseline models with distinct architectures strongly validate the architecture-agnostic claim.
- With ZPressor, DepthSplat's peak memory drops from 6.80GB to 3.80GB (−44%) and inference time from 0.260s to 0.184s (−29%), representing significant efficiency gains.

## Limitations & Future Work
- Effectiveness is limited under extremely dense views (e.g., 1000 views), as compression to 50 views still poses computational challenges. Combining with 3D Gaussian merging or memory-efficient rendering warrants exploration.
- The anchor count $N$ is currently a manually set hyperparameter and does not adapt to scene information density. A policy network to predict the optimal anchor count could be considered.
- Evaluation is limited to static scene datasets; applicability to dynamic scenes (4D scene reconstruction) remains unexplored.
- Compression is performed only along the view dimension; spatial resolution compression is not addressed. Combining both dimensions could further improve scalability.
- Training still uses a small number of context views (6); whether training with more views is beneficial deserves investigation.
- Detailed comparison with concurrent work such as FreeSplat is limited.

## Related Work & Insights
- **vs. FreeSplat/GGN**: These methods reduce redundancy by merging Gaussians via cross-view projection inspection, but lack a principled framework. ZPressor provides a more systematic solution grounded in IB theory.
- **vs. StreamGS**: StreamGS merges redundant Gaussians in 3D space; ZPressor performs information compression at the encoder input stage. The two approaches address redundancy at different levels and are in principle complementary.
- **vs. Engineering Optimizations (memory-efficient attention, etc.)**: Engineering optimizations can only alleviate memory issues; they cannot resolve performance degradation. ZPressor addresses the root cause at the level of representation learning.

## Rating
- Novelty: ⭐⭐⭐⭐ Analyzing feed-forward 3DGS through the IB principle is a novel perspective, though cross-attention-based compression itself is not particularly new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three baseline models, two large-scale datasets, and comprehensive ablation and efficiency analyses make this highly complete.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical analysis is clear; the logical chain from principle to design to experiments is coherent and complete.
- Value: ⭐⭐⭐⭐⭐ Addresses the core scalability problem of feed-forward 3DGS; the plug-and-play module offers exceptionally high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](../../CVPR2026/3d_vision/embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting](learning_efficient_fuse-and-refine_for_feed-forward_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting](plana3r_zero-shot_metric_planar_3d_reconstruction_via_feed-forward_planar_splatt.md)
- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](../../ICLR2026/3d_vision/splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)

</div>

<!-- RELATED:END -->

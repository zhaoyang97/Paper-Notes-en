---
title: >-
  [Paper Note] GGPT: Geometry-Grounded Point Transformer
description: >-
  [CVPR 2026][3D Vision][sparse-view 3D reconstruction] This paper proposes the GGPT framework, which first obtains geometrically consistent sparse point clouds via an improved lightweight SfM pipeline (dense matching + sp…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "sparse-view 3D reconstruction"
  - "Point Transformer"
  - "SfM"
  - "feed-forward"
  - "multi-view geometry"
date: 2026-05-08
content_hash: 14e081b5479b09ce
---

# GGPT: Geometry-Grounded Point Transformer

**Conference**: CVPR 2026
**arXiv**: [2603.11174](https://arxiv.org/abs/2603.11174)
**Code**: [Available](https://chenyutongthu.github.io/research/ggpt)
**Area**: 3D Vision / 3D Reconstruction
**Keywords**: sparse-view 3D reconstruction, Point Transformer, SfM, feed-forward, multi-view geometry

## TL;DR

This paper proposes the GGPT framework, which first obtains geometrically consistent sparse point clouds via an improved lightweight SfM pipeline (dense matching + sparse BA + DLT triangulation), then employs Point Transformer V3 to jointly process sparse geometric guidance and feed-forward dense predictions directly in 3D space for residual refinement. Trained exclusively on ScanNet++, GGPT significantly improves multiple feed-forward 3D reconstruction models across architectures and datasets without any fine-tuning.

## Background & Motivation

**Background**: Feed-forward 3D reconstruction networks (DUSt3R → MASt3R → VGGT) predict dense point maps and camera parameters in a single forward pass, offering fast inference and visually appealing results, yet they lack explicit multi-view constraints, leading to geometric inconsistencies—especially severe in out-of-distribution scenarios (medical, surgical, and human-body scenes).

**Limitations of Prior Work**: (1) SfM is geometrically consistent but fragile under wide-baseline or sparse-view settings and recovers only sparse points; (2) prior geometry-guided methods rely on pseudo-GT SfM points or dense video sequences, which are unavailable in genuinely sparse settings; (3) existing refinement approaches operate in 2D image space (depth completion / image Transformers), inherently failing to achieve true cross-view consistency.

**Key Challenge**: Feed-forward methods produce dense but inconsistent reconstructions, while SfM yields consistent but sparse results. Existing approaches either depend on impractical GT guidance or perform 2D-space refinement that cannot guarantee 3D consistency.

**Goal**: Organically combine the geometric precision of SfM with the dense completeness of feed-forward networks in 3D space, enabling sparse-view 3D reconstruction refinement that generalizes across architectures without fine-tuning.

**Key Insight**: A two-stage design—first obtaining genuine sparse geometry from input RGB via an improved SfM pipeline, then performing attention and residual correction directly in point-cloud space using a 3D Point Transformer.

**Core Idea**: Performing geometric fusion and refinement in 3D space rather than in 2D image space is the key to cross-domain generalization.

## Method

### Overall Architecture

The framework consists of two stages. **(1) Improved SfM**: feed-forward model initialization → dense matcher (RoMa + UFM) for global correspondences → cycle-consistency filtering → sparse BA with high-confidence matches (2,048 points/view) → DLT triangulation with lower-threshold matches to obtain $\mathbf{X}_s$. **(2) GGPT**: Point Transformer V3 (53M parameters) jointly processes dense predictions $\mathbf{X}_d$ and sparse guidance $\mathbf{X}_s$ in a global 3D coordinate frame, predicting residual displacements $\boldsymbol{\delta}$ and confidence scores $c$ to produce refined dense reconstructions $\hat{\mathbf{X}}_d$.

### Key Designs

1. **Improved SfM Pipeline**: A feed-forward model (VGGT) initializes cameras and points; dense feature matching (RoMa + UFM) yields a full correspondence tensor $\mathbf{T} \in \mathbb{R}^{N \times N \times W \times H \times 2}$; cycle-consistency filtering (Eq. 3) is applied; two confidence thresholds $\epsilon_{BA} > \epsilon_{DLT}$ select separate match subsets for BA and DLT, respectively; **sparse BA** (high-confidence, few points) optimizes cameras only; **DLT triangulation** (lower threshold, denser matches) linearly reconstructs a large number of 3D points. The core design separates nonlinear optimization (BA) from linear triangulation (DLT): BA requires only a small number of high-confidence points for accurate camera estimation, while DLT leverages denser matches for efficient triangulation.

2. **Geometry-Guided Encoding**: The embedding of a dense point $\mathbf{x}_d$ comprises its own positional encoding $\text{PE}(\mathbf{x}_d)$, a type token $\mathbf{e}_{type(d)}$, the positional encoding of its corresponding sparse guidance point $\text{PE}(\mathbf{x}_{d \to s})$, and the offset $\Delta_{d \to s} = \mathbf{x}_{d \to s} - \mathbf{x}_s$. This encoding allows the network to explicitly perceive the discrepancy between dense predictions and geometric priors—$\Delta_{d \to s}$ directly encodes the signal of "how much correction is needed."

3. **Direct 3D-Space Attention (PTv3)**: Point Transformer V3 (8 layers, 53M parameters, far smaller than a 2D ViT's ~300M) performs patch-wise self-attention over 3D neighbors. Spatial proximity, rather than pixel coordinates, defines the receptive field, naturally ensuring multi-view consistency. Patch-based processing divides the scene into overlapping cubic blocks (radius = 0.2 × scene radius), each processed independently (up to 400K points), with overlapping regions averaged.

### Loss & Training

- **Confidence-weighted regression**: $\mathcal{L}_{conf} = \sum c \|\hat{\mathbf{x}} - \mathbf{x}_{GT}\| - \alpha \log c$; the heteroscedastic formulation automatically down-weights uncertain regions.
- **Identity consistency**: $\mathcal{L}_{id} = \sum \|\hat{\mathbf{x}} - \mathbf{x}_{d \to s}\|$; encourages dense points with correspondences to align toward geometric guidance.
- Total loss: $\mathcal{L} = \mathcal{L}_{conf} + \lambda_{id} \mathcal{L}_{id}$, with $\lambda_{id}=1$, $\alpha=0.2$.
- Training: 20K sequences from ScanNet++; trained for one day on 8× GH200 GPUs.

## Key Experimental Results

### Main Results (AUC@5/10 cm ↑, 8 views)

| Method | ScanNet++ | ETH3D | T&T |
|---|---|---|---|
| VGGT | 19/32 | 23/36 | 25/39 |
| VGGT + Ours | **45/60** | **47/61** | **42/57** |
| Pi3 | 56/71 | 25/41 | 26/42 |
| Pi3 + Ours | **56/72** | **36/53** | **32/50** |
| MapAnything | 38/57 | 7/15 | 9/20 |
| MapAnything + Ours | **48/64** | **33/45** | **40/55** |

### Ablation Study

| Ablation | ScanNet++ 4v | ETH3D 4v | Notes |
|---|---|---|---|
| Full GGPT | 38/53 | 41/55 | Baseline |
| w/o $\mathbf{X}_s$ guidance | Learnable in-domain, collapses OOD | Large drop | Guidance indispensable |
| w/o correspondence encoding $\Delta_{d \to s}$ | Significant drop | Significant drop | Most critical component |
| 2D Transformer replacing PTv3 | Small in-domain gap | Large OOD gap | 3D attention generalizes better |
| Patch r=0.1 vs 0.2 vs 0.5 | r=0.2 optimal | — | Smaller patches enhance generalization |

### Key Findings

- **Strong out-of-distribution generalization**: Trained solely on ScanNet++, GGPT improves five methods across five datasets with no fine-tuning.
- **Largest gain on VGGT**: AUC@5 improves from 19→45 (+137%) on ScanNet++ and from 23→47 (+104%) on ETH3D.
- **Striking OOD results**: On 4D-DRESS, VGGT AUC@1/5cm goes from 10/45 to 66/77 (+Ours); on MV-dVRK, from 8/33 to 45/61.
- **3D vs. 2D refinement**: PTv3 substantially outperforms 2D Transformer alternatives on cross-domain data, representing a fundamental improvement.
- **SfM ablation**: Dense matchers >> sparse matchers (MASt3R); DLT is hundreds of times faster than RANSAC triangulation at comparable accuracy; sparse BA with as few as 512 points suffices.

## Highlights & Insights

- Performing geometric fusion in 3D space rather than 2D image space constitutes a fundamental improvement—yielding substantial cross-domain generalization advantages.
- The design philosophy of "train one configuration, improve multiple feed-forward methods without fine-tuning" offers significant practical value.
- The separation of sparse BA and DLT is concise and efficient: nonlinear optimization is reserved for high-confidence sparse points, while triangulation employs a linear method.
- The geometry-guided encoding is elegantly designed: $\Delta_{d \to s}$ directly encodes the "correction magnitude," providing the network with the most direct supervisory signal.

## Limitations & Future Work

- **SfM error propagation**: SfM and GGPT are executed sequentially; if SfM fails (e.g., in low-texture scenes), refinement cannot compensate.
- **Patch-boundary artifacts**: Block-wise processing may introduce boundary discontinuities; overlap averaging mitigates but does not fully eliminate this issue.
- **Indoor-only training**: Performance on large-scale outdoor scenes and settings with more than 16 views remains unvalidated.
- **Computational overhead**: Running the dense matcher and BA incurs additional inference time.

## Related Work & Insights

- **vs. DUSt3R/VGGT**: Feed-forward methods are fast but geometrically inconsistent; GGPT serves as a universal post-processing module to complement geometric consistency.
- **vs. COLMAP**: Traditional incremental SfM is fragile under sparse views; the global SfM with dense matching proposed here is more robust and efficient.
- **vs. 2D depth-completion methods**: 2D image-space refinement is inherently view-dependent; 3D-space attention fundamentally resolves cross-view consistency.
- **Insight**: The paradigm of 3D-space processing over 2D image-space processing deserves validation across broader tasks (e.g., multi-view fusion for semantic segmentation and object detection).

## Rating

⭐⭐⭐⭐⭐ (5/5)

**Rationale**: The method is elegantly designed with well-motivated foundations (3D- vs. 2D-space refinement). Experiments are comprehensive (5 feed-forward methods × 5 datasets, including OOD medical and human-body data), generalization is remarkable (a single training configuration universally improves diverse methods), and ablation studies are thorough with clear conclusions. Both the improved SfM pipeline and the 3D Point Transformer design carry independent value. This is a high-quality contribution to the sparse-view 3D reconstruction field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation](mvggt_multimodal_visual_geometry_grounded_transformer_for_multiview_3d_referring.md)
- [\[ICLR 2026\] Quantized Visual Geometry Grounded Transformer](../../ICLR2026/3d_vision/quantized_visual_geometry_grounded_transformer.md)
- [\[CVPR 2026\] LitePT: Lighter Yet Stronger Point Transformer](litept_lighter_yet_stronger_point_transformer.md)
- [\[CVPR 2026\] RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations](rng_a_unified_transformer_for_complete_3d_modeling_from_partial_observations.md)
- [\[CVPR 2026\] GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance](gaussiangrow_geometry-aware_gaussian_growing_from_3d_point_clouds_with_text_guid.md)

</div>

<!-- RELATED:END -->

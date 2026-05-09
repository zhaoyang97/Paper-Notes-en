---
title: >-
  [Paper Note] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception
description: >-
  [CVPR 2026][3D Vision][Cooperative Perception] Long-SCOPE proposes a fully sparse long-range cooperative 3D perception framework that achieves state-of-the-art performance in 100–150 m long-range scenarios through geometry-guided query generation and a context-aware association module, while maintaining efficient computation and communication costs.
tags:
  - CVPR 2026
  - 3D Vision
  - Cooperative Perception
  - Sparse Architecture
  - Long-Range 3D Detection
  - Query Association
  - V2X
date: 2026-05-08
content_hash: b7e7375311b9a373
---

# Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception

**Conference**: CVPR 2026
**arXiv**: [2604.09206](https://arxiv.org/abs/2604.09206)
**Code**: N/A
**Area**: 3D Vision
**Keywords**: Cooperative Perception, Sparse Architecture, Long-Range 3D Detection, Query Association, V2X

## TL;DR

Long-SCOPE proposes a fully sparse long-range cooperative 3D perception framework that achieves state-of-the-art performance in 100–150 m long-range scenarios through geometry-guided query generation and a context-aware association module, while maintaining efficient computation and communication costs.

## Background & Motivation

**Background**: Cooperative perception extends the perceptual range of autonomous driving and addresses occlusion issues via V2X communication, but mainstream methods rely on dense BEV features whose computation and communication costs scale quadratically with the perception range.

**Limitations of Prior Work**: (1) Dense BEV representations incur exploding computational costs in long-range scenarios; (2) observation errors and alignment errors for distant objects grow substantially, making existing feature association mechanisms based on fixed distance thresholds brittle.

**Key Challenge**: Efficient sparse communication requires accurate query association, yet positional noise at long range causes rigid threshold-based methods to fail, incorrectly filtering out valid collaborative queries.

**Goal**: Design a fully sparse architecture that simultaneously addresses computational efficiency and robust association in long-range scenarios.

**Key Insight**: Completely abandon BEV features, extract object queries directly from image features, and replace rule-based matching with a learnable attention mechanism.

**Core Idea**: Dynamically generate high-quality 3D queries using geometric priors to handle observation errors, and perform robust query matching via context-aware attention to handle alignment errors.

## Method

### Overall Architecture

Long-SCOPE is a query-centric fully sparse framework: each agent generates object queries (static anchors + dynamic GQG queries) → multi-layer Transformer decoder refinement → collaborative query projection aligned to the ego coordinate system → CAA module for robust matching → fusion refinement → output 3D detection results.

### Key Designs

1. **Geometry-guided Query Generation (GQG)**:

    - **Function**: Dynamically generate high-quality 3D queries for small, distant objects.
    - **Mechanism**: For elevated agents (roadside units/UAVs), the model predicts the global height $\hat{z}_{Q_{glb}}$ of the target rather than directly regressing depth, since height distributions are concentrated while depth distributions are highly dispersed. The depth is then recovered via the similar-triangle geometric relationship: $\hat{z}_{Q_{cam}} = \frac{\hat{z}_{Q_{glb}} - z_{C_{glb}}}{(T_{cam2glb} \cdot K_{cam}^{-1} \cdot P_{img})_z}$. For ground vehicles, the model falls back to direct depth regression.
    - **Design Motivation**: Static anchor sets have low hit rates for distant targets; dynamic queries substantially improve initial detection quality.

2. **Context-Aware Association (CAA) Module**:

    - **Function**: Robustly match collaborative queries under severe positional noise.
    - **Mechanism**: A multi-layer Transformer architecture concatenates queries from all $N$ agents and applies global self-attention. The design follows four principles: injective matching (one-to-one), asymmetric visibility (supporting unmatched queries), spatial consistency (exploiting local neighborhood topology rather than absolute coordinates), and scalability (not limited to pairwise matching).
    - **Design Motivation**: Fixed distance thresholds fail at long range; learnable matching based on content and context can leverage semantic similarity and spatial topology for robust association.

3. **Height-to-Depth Conversion for Elevated Viewpoints**:

    - **Function**: Address depth estimation accuracy for elevated cameras.
    - **Mechanism**: For high-angle views, the notably non-zero $z_{P_{virt}}$ is exploited to avoid numerical instability; for ground-level views, $z_{P_{virt}} \approx 0$ near the horizon causes numerical instability, so the model falls back to direct depth regression.
    - **Design Motivation**: Apply the most suitable depth estimation strategy per scenario rather than a one-size-fits-all approach.

### Loss & Training

The model is trained end-to-end. The 2D detection and depth estimation heads in GQG adopt lightweight structures, and the matching results from the CAA module are used to generate supervision signals.

## Key Experimental Results

### Main Results

| Dataset / Range | Metric | Long-SCOPE | Prev. SOTA | Gain |
|----------------|--------|------------|------------|------|
| V2X-Seq Long-Range | AP | SOTA | — | Significant improvement |
| Griffin-25m 100–150 m | AP | SOTA | — | Breakthrough improvement |
| Griffin-25m Overall | AP | SOTA | — | Superior in both efficiency and accuracy |

### Ablation Study

| Configuration | Key Metric | Description |
|--------------|-----------|-------------|
| w/o GQG | AP drop | Degraded detection of small distant objects |
| w/o CAA | AP drop | Association failure causes duplicate detections |
| Fixed 30 m threshold | Lower AP | Brittle association from the baseline method |
| Full Long-SCOPE | Best | Complementary synergy of both modules |

### Key Findings

- The most significant gains appear at extreme ranges of 100–150 m, validating the necessity of long-range-specific design.
- The height-prediction strategy in GQG is highly effective for elevated agents, whereas direct depth regression should be used for ground vehicles.
- The global attention-based association in CAA substantially outperforms heuristic methods such as fixed distance thresholds and the Hungarian algorithm.

## Highlights & Insights

- **Advantages of the fully sparse architecture**: BEV features are entirely abandoned; communication cost scales linearly with the number of objects rather than quadratically with the perception range.
- **Viewpoint-specific depth estimation**: Using height-to-depth conversion for elevated agents and direct depth regression for ground agents fully exploits the geometric characteristics of different viewpoints.
- **SfM-inspired multi-agent matching**: Borrowing the concatenation and global attention strategy from multi-view matching in SfM naturally extends to $N$-agent scenarios.

## Limitations & Future Work

- The computational cost of global self-attention scales quadratically with the total number of queries; although the number of targets is typically fewer than 100, this may become a bottleneck in extremely dense scenes.
- Practical deployment issues such as communication latency and packet loss are not addressed.
- Evaluation is limited to 3D object detection and has not been extended to tasks such as semantic segmentation.

## Related Work & Insights

- **vs. SparseCoop**: Long-SCOPE replaces the most fragile query generation and association modules built upon SparseCoop.
- **vs. Far3D**: GQG draws on Far3D's 2D detection + depth estimation scheme but introduces height-based depth inference as a new contribution.

## Rating

- Novelty: ⭐⭐⭐⭐ Both GQG and CAA feature targeted innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two datasets, V2X-Seq and Griffin.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and design principles are well-articulated.
- Value: ⭐⭐⭐⭐ Provides a practical solution for long-range deployment in cooperative perception.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates](ltgs_long-term_gaussian_scene_chronology_from_sparse_view_updates.md)
- [\[CVPR 2026\] Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?](can_natural_image_autoencoders_compactly_tokenize_fmri_volumes_for_long-range_dy.md)
- [\[CVPR 2026\] MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification](morel_long-range_flicker-free_4d_motion_modeling_via_anchor_relay-based_bidirect.md)
- [\[CVPR 2026\] LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](longstream_long-sequence_streaming_autoregressive_visual_geometry.md)
- [\[CVPR 2026\] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)

<!-- RELATED:END -->

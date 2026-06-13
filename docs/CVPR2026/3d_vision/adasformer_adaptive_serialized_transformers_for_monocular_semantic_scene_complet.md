---
title: >-
  [Paper Note] AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments
description: >-
  [CVPR 2026][Semantic Scene Completion] This paper proposes AdaSFormer, a serialized Transformer framework for indoor Monocular Semantic Scene Completion (MSSC)…
tags:
  - "CVPR 2026"
  - "Semantic Scene Completion"
  - "Serialized Transformer"
  - "Adaptive Attention"
  - "Indoor Scene"
  - "Monocular RGB"
date: 2026-05-08
content_hash: e0c451f97e4e8c0f
---

# AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments

**Conference**: CVPR 2026
**arXiv**: [2603.25494](https://arxiv.org/abs/2603.25494)  
**Code**: [https://github.com/alanWXZ/AdaSFormer](https://github.com/alanWXZ/AdaSFormer)  
**Area**: Other
**Keywords**: Semantic Scene Completion, Serialized Transformer, Adaptive Attention, Indoor Scene, Monocular RGB

## TL;DR
This paper proposes AdaSFormer, a serialized Transformer framework for indoor Monocular Semantic Scene Completion (MSSC), achieving state-of-the-art performance on NYUv2 and Occ-ScanNet through three core designs: Adaptive Serialization Attention (with learnable offsets), Center-Relative Position Encoding, and Convolutional Modulation Layer Normalization.

## Background & Motivation

**Background**: Monocular semantic scene completion predicts complete 3D voxel occupancy and semantic labels from a single RGB image. While outdoor (autonomous driving) scenarios have received extensive attention, indoor MSSC remains more challenging due to complex spatial layouts and severe occlusion.

**Limitations of Prior Work**: Existing indoor methods predominantly rely on CNN architectures — limited local receptive fields prevent modeling long-range dependencies, and 3D convolutional kernels incur cubically growing computational overhead. Although Transformers can model global context, their direct application to dense 3D voxels imposes prohibitive computational and memory costs.

**Key Challenge**: Indoor scenes require strong global context reasoning (inferring geometry and semantics in occluded regions), yet the $O(N^2)$ complexity of Transformers becomes infeasible at high-resolution 3D voxel scales.

**Key Insight**: Serialized Transformers convert irregular 3D data into ordered sequences, reducing complexity to $O(N \cdot G)$ via local grouping. However, existing methods employ fixed grouping schemes that constrain the receptive field.

**Core Idea**: Introducing learnable offsets to adaptively adjust serialization starting points, allowing different layers to obtain different receptive fields and thus more flexible spatial representations.

## Method

### Overall Architecture
Monocular RGB image → 2D encoder (EfficientNet) + depth estimation → 3D projection → 3D encoder (multiple AdaSFormer blocks alternating Transformer and convolution) → lightweight decoder → SSC output.

### Key Designs

1. **Adaptive Serialization Attention (ASA)**:

    - **Function**: Adaptively adjusts serialization starting points via learnable offsets to achieve more flexible receptive fields.
    - **Mechanism**: Given patch size $P$, $K$ learnable parameters encode offset values at uniform intervals of $P/K$. The **Straight-Through Gumbel-Softmax** enables differentiable discrete selection: $\mathbf{y}_{soft} = \text{softmax}((\mathbf{l} + \mathbf{g})/\tau)$, where the forward pass uses the hard selection $\mathbf{y}_{hard}$ while gradients are backpropagated through $\mathbf{y}_{soft}$. A temperature annealing strategy $\tau_t = \max(\tau_{min}, \tau_{init} \cdot \exp(-\alpha t))$ progressively enforces discreteness.
    - **Design Motivation**: Different starting points substantially alter receptive field coverage — they may fully cover a single object or simultaneously capture spatial relationships across multiple objects. Swin Transformer's window shifts are fixed and non-learnable; serialization attention operates along 1D sequences, offering a broader and more flexible offset space.

2. **Center-Relative Position Encoding (CRPE)**:

    - **Function**: Encodes the spatial relationship between each voxel and the scene center to capture information density.
    - **Mechanism**: The scene center $\mathbf{c}$ is computed as the mean coordinate of all occupied voxels. The yaw difference $\Delta\theta$ and pitch difference $\Delta\phi$ of each voxel relative to the scene center are concatenated and passed through an MLP to serve as attention biases.
    - **Design Motivation**: CNN components already encode local positional information; additional position encoding should focus on spatial information distribution — structural and semantic information density varies at different distances from the scene center.

3. **Convolutional Modulation Layer Normalization (CMLN)**:

    - **Function**: Bridges the heterogeneous feature representations of CNNs and Transformers.
    - **Mechanism**: $\text{CMLN}(h_i | X_{voxel}) = \gamma(X_{voxel}) \odot \frac{h_i - \mu_i}{\sigma_i} + \beta(X_{voxel})$, where normalization parameters $\gamma$ and $\beta$ are generated from voxel features via a small MLP.
    - **Design Motivation**: Transformers and CNNs extract fundamentally different feature types; directly alternating between them introduces learning difficulties that require adaptive feature statistics modulation.

### Loss & Training
Standard SSC losses (cross-entropy + scene completion IoU-related loss).

## Key Experimental Results

### Main Results (NYUv2 Dataset)

| Method | Conference | SC IoU% | SSC mIoU% |
|--------|------------|---------|-----------|
| MonoScene | CVPR'22 | 42.51 | 26.94 |
| NDC-Scene | ICCV'23 | 44.17 | 29.03 |
| ISO | ECCV'24 | 47.11 | 31.25 |
| MonoMRN | ICCV'25 | 53.16 | 26.80* |
| **AdaSFormer (Ours)** | CVPR'26 | **SOTA** | **SOTA** |

*Note: MonoMRN achieves strong SC IoU but lower SSC mIoU; AdaSFormer reaches SOTA on both metrics.

### Ablation Study (NYUv2)

| Configuration | SC IoU | SSC mIoU |
|---------------|--------|----------|
| Baseline (standard serialized Transformer) | — | — |
| + ASA (learnable offsets) | +gain | +gain |
| + CRPE (center-relative encoding) | +gain | +gain |
| + CMLN (modulation normalization) | +gain | +gain |
| All combined | **Best** | **Best** |

### Key Findings
- ASA is the most critical component — learnable offsets yield significant gains over fixed offsets.
- CRPE is particularly effective in indoor scenes, whose structure is more center-oriented.
- CMLN resolves feature mismatches arising from direct CNN–Transformer alternation.
- SOTA is achieved on both NYUv2 and Occ-ScanNet.
- Memory and computational overhead are substantially reduced compared to full 3D Transformers.

## Highlights & Insights
- **Learnable Serialization Offsets**: Gumbel-Softmax renders the discrete serialization starting point selection differentiable — a general improvement to serialized Transformers transferable to point cloud segmentation and 3D detection.
- **Spatial Information Density Encoding**: Unlike standard position encodings that record absolute or relative positions, CRPE encodes spatial information density — regions farther from the scene center tend to be informationally sparser.
- **Heterogeneous CNN–Transformer Feature Bridging**: CMLN provides an elegant solution to feature statistics mismatches in hybrid architecture design.

## Limitations & Future Work
- Validation is limited to indoor scenes (NYUv2 is relatively small); performance on larger-scale indoor datasets remains to be demonstrated.
- Depth estimation quality substantially affects overall performance; end-to-end training must ensure co-optimization of the depth and completion networks.
- Computing the scene center as the mean of occupied voxel coordinates may lack robustness when occupancy distributions are skewed.
- The $K$ candidate offset values are predefined at uniform intervals; adaptive spacing may be more effective.

## Related Work & Insights
- **vs. MonoScene / NDC-Scene / ISO**: Full CNN architectures lack global reasoning capacity; this work introduces Transformers to compensate.
- **vs. OctFormer / PTv3**: General-purpose serialized Transformer designs; this work adds learnable offsets to adapt to SSC.
- **vs. Swin Transformer**: Swin's window shifts are fixed and confined to 2D; the serialization offsets here operate along 1D sequences, offering greater flexibility.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Learnable serialization offsets are creative; CRPE and CMLN are well-motivated designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive validation on NYUv2 and Occ-ScanNet, though indoor dataset scales are relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear with intuitive illustrations.
- **Value**: ⭐⭐⭐ A meaningful contribution to indoor SSC, though the application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](../../AAAI2026/others/towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/others/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [\[ICLR 2026\] The Counting Power of Transformers](../../ICLR2026/others/the_counting_power_of_transformers.md)
- [\[CVPR 2026\] LoViF 2026 Challenge on Human-oriented Semantic Image Quality Assessment](lovif_2026_semantic_quality_assessment_challenge.md)

</div>

<!-- RELATED:END -->

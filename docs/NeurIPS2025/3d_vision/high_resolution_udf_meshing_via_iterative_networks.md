---
title: >-
  [Paper Note] High Resolution UDF Meshing via Iterative Networks
description: >-
  [NeurIPS 2025][3D Vision][Unsigned Distance Field] This paper proposes the first iterative meshing method for Unsigned Distance Fields (UDFs)…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Unsigned Distance Field"
  - "Meshing"
  - "Iterative Network"
  - "Pseudo-sign Prediction"
  - "High-Resolution Surface Reconstruction"
date: 2026-05-08
content_hash: f67c915235e405e9
---

# High Resolution UDF Meshing via Iterative Networks

**Conference**: NeurIPS 2025
**arXiv**: [2509.17212](https://arxiv.org/abs/2509.17212)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: Unsigned Distance Field, Meshing, Iterative Network, Pseudo-sign Prediction, High-Resolution Surface Reconstruction

## TL;DR
This paper proposes the first iterative meshing method for Unsigned Distance Fields (UDFs), which progressively propagates neighborhood information into local voxel pseudo-sign predictions through multiple forward passes. The approach effectively resolves surface holes and discontinuities caused by noisy neural UDFs at high resolutions, significantly outperforming existing single-pass methods across multiple datasets.

## Background & Motivation

**Background**: In implicit neural representations, Signed Distance Fields (SDFs) localize surfaces via sign changes, enabling efficient triangulation through classical algorithms such as Marching Cubes. Unsigned Distance Fields (UDFs) can represent open surfaces and are thus more general than SDFs, but their triangulation is considerably more challenging — UDFs reach zero at the surface with no exploitable sign changes.

**Limitations of Prior Work**: Existing UDF meshing methods (MeshUDF, NSD-UDF, DCUDF, DualMesh-UDF, etc.) operate independently within individual voxels, recovering surfaces by predicting pseudo-signs or dual-contour vertices. However, neural UDFs are inherently noisy — UDF values may not reach zero precisely at the surface, and gradient directions may also be inaccurate.

**Key Challenge**: Counterintuitively, **increasing meshing resolution exacerbates the problem**, since smaller voxels at higher resolutions are more susceptible to UDF noise. Single-pass, per-voxel methods lack sufficient context to make correct decisions in noisy regions, resulting in surface holes and discontinuities.

**Goal**: How to robustly recover complete and accurate triangle meshes from noisy UDFs at high resolutions?

**Key Insight**: The key observation is that while UDF values and gradients near the surface may be inaccurate, gradients farther from the surface remain reliable, and correctly reconstructed surface regions contain information valuable for resolving neighboring ambiguous regions. The method therefore leverages information from already-extracted surface elements to assist in resolving adjacent uncertain regions.

**Core Idea**: Transform UDF meshing from a single-pass independent operation into a multi-pass iterative process, where each round uses the previous round's output (neighboring pseudo-sign configurations) as additional input to propagate spatial information.

## Method

### Overall Architecture
Given a neural UDF $U_\mathcal{S}$, UDF values and gradients are queried on a regular grid. After grouping by voxel, each voxel is fed into a per-voxel fully connected network $f_\theta$. The network's task is to predict pseudo-sign configurations for the 8 corners of a voxel (a 128-class classification problem). The key innovation is that the network takes as input not only UDF values and gradients, but also the current pseudo-sign configurations of the target voxel and its neighbors. Through multiple iterations, the output of each round serves as input for the next, progressively propagating spatial information. The final pseudo-signs are triangulated using Marching Cubes or DualMesh-UDF.

### Key Designs

1. **Iterative Pseudo-sign Prediction Network**:

    - Function: Predicts pseudo-sign configurations per voxel and propagates neighborhood information through iteration.
    - Mechanism: The network input consists of UDF values, gradients, and the current pseudo-sign configurations of the target voxel and its face-adjacent neighbors (6 neighbors + itself). The output at the $i$-th iteration is:
      $\mathbf{y}_{\mathcal{S},c}^{(i)} = f_\theta(U_\mathcal{S}(c), \nabla U_\mathcal{S}(c), \sigma(\mathbf{y}_{\mathcal{S},N_c}^{(i-1)}))$
      where $\mathbf{y}^{(0)} = [0,0,...,0]$ (all zeros indicating no prior information) and $\sigma$ denotes sigmoid activation.
    - Design Motivation: Single-pass methods consider only the UDF information within individual voxels and are prone to errors in high-noise regions. Iteration allows correctly reconstructed regions to "propagate" information to neighboring ambiguous regions. For instance, if a voxel's UDF does not reach zero but its neighbors have confirmed surface passage, the network can make a more informed decision.

2. **Randomized Iteration Count Training**:

    - Function: During training, the number of iterations is randomly sampled from 1 to 6, and cross-entropy loss is computed at each iteration.
    - Mechanism: The loss function is $\mathcal{L}_\theta = \sum_{i=1}^{r} \sum_{c \in \mathcal{S}} CE(softmax(\mathbf{y}_{\mathcal{S},c}^{(i)}), GT_\mathcal{S}(c))$, where $r$ is randomly sampled from $[1,6]$.
    - Design Motivation: Training with a fixed number of iterations may cause the network to overfit to patterns at a specific depth rather than learning a robust propagation strategy. Randomization forces the network to produce reasonable predictions at any iteration depth, while accumulating loss across multiple iterations ensures stable gradient propagation through long chains.

3. **Noise Augmentation and Efficient Filtering**:

    - Function: Enhances robustness to UDF noise and accelerates inference.
    - Mechanism: During training, multiplicative Gaussian noise is applied to UDF values and gradients: $U(c) \leftarrow U(c) \cdot (1 + \mathcal{N}(0, \sigma_\mathcal{N}))$. During inference, voxels with UDF values ≥ a truncation threshold (~85%) are first filtered out, and high-confidence voxels (>0.999) are further filtered in subsequent iterations.
    - Design Motivation: Noise augmentation prevents the network from overfitting to precise UDF values. The filtering strategy reduces computational cost to the same order as single-pass methods (reducing inference at $256^3$ resolution from 7 minutes to 30 seconds) without sacrificing accuracy.

### Loss & Training
Cross-entropy loss is summed over all voxels at all iterations. UDF values are normalized by resolution (divided by voxel size) to enable generalization across different resolutions. Key finding: sigmoid activation and randomized iteration counts are critical for convergence.

## Key Experimental Results

### Main Results — Comparison with Marching Cubes Methods (Auto-decoder UDF, Resolution 512)

| Dataset | Method | CD(×10⁻⁵)↓ | F1↑ | IC↑ |
|--------|------|-------------|-----|-----|
| ShapeNet cars | MeshUDF | 82.7 | 57.0 | 81.7 |
| ShapeNet cars | NSD-UDF+MC | 56.9 | 58.8 | 83.8 |
| ShapeNet cars | **Ours+MC** | **8.84** | **65.6** | **88.9** |
| ShapeNet chairs | MeshUDF | 378 | 61.5 | 65.7 |
| ShapeNet chairs | NSD-UDF+MC | 295 | 64.7 | 75.7 |
| ShapeNet chairs | **Ours+MC** | **8.76** | **74.5** | **87.2** |
| ShapeNet planes | MeshUDF | 12.6 | 88.1 | 84.6 |
| ShapeNet planes | NSD-UDF+MC | 10.0 | 89.4 | 85.1 |
| ShapeNet planes | **Ours+MC** | **2.37** | **90.9** | **87.1** |

At resolution 512, the proposed method substantially outperforms all baselines across all datasets. On ShapeNet cars, CD decreases from 56.9 (NSD-UDF) to 8.84 (a 6.4× improvement); on chairs, from 295 to 8.76 (a 33.7× improvement).

### Performance Trend Across Resolutions

| Resolution | NSD-UDF+MC (cars CD) | Ours+MC (cars CD) | Note |
|--------|---------------------|-------------------|------|
| 128 | 6.79 | 5.64 | Smaller gap at low resolution |
| 256 | 10.2 | 5.23 | Gap begins to widen |
| 512 | 56.9 | 8.84 | Competing methods degrade severely at high resolution; proposed method remains stable |

### Key Findings
- **Advantage grows with resolution**: Competing methods exhibit performance degradation at high resolutions due to amplified noise effects, while the iterative method remains stable and continues to improve. This constitutes the core contribution.
- The filtering strategy reduces inference time at $256^3$ from 7 minutes to 30 seconds and at $512^3$ from 1 hour to 2.5 minutes, achieving computational costs comparable to baseline methods.
- Convergence is typically achieved within 6 iterations, with each round processing a progressively smaller set of uncertain voxels.

## Highlights & Insights
- **Precise identification of a counterintuitive insight**: The observation that high-resolution UDF meshing is harder than low-resolution is critical and directly informs the methodological design. The authors frame this as an information propagation task requiring a larger receptive field, making the solution arise naturally.
- **Elegant iterative mechanism**: Using the previous round's pseudo-sign configuration as "prior knowledge" is conceptually simple yet empirically powerful. This is analogous to CRF post-processing in image segmentation, but here realized as an end-to-end trainable module.
- **Cross-task transfer potential**: The paradigm of "propagating information from resolved neighboring regions to unresolved ones" is transferable to any mesh processing task where local decisions require global context.

## Limitations & Future Work
- The method improves only the meshing step and does not directly enhance the accuracy of the UDF itself. If the UDF is entirely incorrect over a large region (rather than locally noisy), iterative propagation may also fail.
- The number of iterations (up to 6) is empirically determined and may be insufficient for particularly complex topologies.
- Training requires ground-truth SDFs to generate pseudo-sign labels, which limits direct application to in-the-wild data.
- Each iteration still performs independent per-voxel inference, encoding neighborhood information only through inputs. A stronger alternative would be to introduce graph networks for direct message passing between voxels.

## Related Work & Insights
- **vs. MeshUDF**: MeshUDF also attempts to incorporate neighborhood information (via a defined voxel traversal order) but remains a single-pass heuristic primarily targeting simple shapes such as garments. The multi-pass iteration proposed here is more systematic.
- **vs. NSD-UDF**: NSD-UDF uses a neural network to predict pseudo-signs from local UDF values and gradients; the proposed method builds on this by introducing iteration and neighborhood information, constituting a strict improvement.
- **vs. DCUDF**: Although DCUDF includes an optimization refinement step, its mesh extraction remains single-pass and requires considerable manual parameter tuning and cutting operations. The proposed method is more end-to-end and robust.

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce iterative refinement into UDF meshing, addressing the core limitation at high resolutions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 dataset categories, 4 UDF architectures, and 2 triangulation methods (MC and DC)
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated, though notation is occasionally inconsistent
- Value: ⭐⭐⭐⭐ Meaningfully advances UDF surface reconstruction with practical implications for resolution scaling

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](../../ICCV2025/3d_vision/alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[NeurIPS 2025\] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks](instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)
- [\[CVPR 2026\] Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](../../CVPR2026/3d_vision/modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)
- [\[AAAI 2026\] SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images](../../AAAI2026/3d_vision/smartsplat_feature-smart_gaussians_for_scalable_compression_of_ultra-high-resolu.md)

</div>

<!-- RELATED:END -->

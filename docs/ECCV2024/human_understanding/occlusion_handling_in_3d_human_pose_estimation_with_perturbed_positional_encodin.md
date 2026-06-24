---
title: >-
  [Paper Note] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding
description: >-
  [ECCV 2024][Human Understanding][3D Human Pose Estimation] To address the issue where human joint occlusion leads to missing edges in 2D skeleton graphs, rendering traditional graph Laplacian positional encodings ineffective, this paper proposes PerturbPE. Leveraging the Rayleigh-Schrödinger Perturbation Theory, the method repeatedly applies random perturbations and computes the average to extract the consistent part of the graph Laplacian eigenspace as the positional encodin…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "3D Human Pose Estimation"
  - "Graph Convolutional Networks"
  - "Positional Encoding"
  - "Occlusion Handling"
  - "Rayleigh-Schrödinger Perturbation Theory"
date: 2026-05-08
content_hash: ec56539503ca3f3c
---

# Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding

**Conference**: ECCV 2024  
**arXiv**: [2405.17397](https://arxiv.org/abs/2405.17397)  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: 3D Human Pose Estimation, Graph Convolutional Networks, Positional Encoding, Occlusion Handling, Rayleigh-Schrödinger Perturbation Theory

## TL;DR

To address the issue where human joint occlusion leads to missing edges in 2D skeleton graphs, rendering traditional graph Laplacian positional encodings ineffective, this paper proposes PerturbPE. Leveraging the Rayleigh-Schrödinger Perturbation Theory, the method repeatedly applies random perturbations and computes the average to extract the consistent part of the graph Laplacian eigenspace as the positional encoding. This approach outperforms MöbiusGCN on complete skeletons and achieves up to a 12% performance improvement in scenarios with missing edges.

## Background & Motivation

The objective of 3D human pose estimation is to lift 2D joint positions to 3D coordinates. Graph Convolutional Networks (GCNs) have become a mainstream solution due to their small parameter footprint and high accuracy, with MöbiusGCN in particular achieving outstanding results using only 0.042M parameters. To enhance the expressiveness of GCNs, utilizing the eigenvectors of the graph Laplacian as positional encodings is an effective approach.

**Limitations of Prior Work**: In real-world scenarios, the human body is frequently occluded (self-occlusion or object occlusion), causing 2D pose estimators to miss certain joints, which manifests as missing edges in the input graph. This alters the graph Laplacian matrix, making its eigenvectors unsuitable as positional encodings. Existing positional encoding methods (such as Laplacian PE, SignNet/BasisNet) assume a complete graph input and fail to handle cases with missing edges. From a theoretical perspective, subgraph matching after partial edge loss is an NP-complete problem.

**Key Insight**: The eigenspace of the graph Laplacian can be decomposed into a "consistent part" and an "irregular part". The consistency of a network is reflected in the stability of its structural features before and after the random removal of a small number of links. Therefore, the consistent and robust parts of the eigenspace can be extracted by applying multiple random perturbations and taking their average.

**Core Idea**: Rayleigh-Schrödinger Perturbation Theory (RSPT) is employed to efficiently compute the perturbed eigenvectors without recomputing the entire eigenspace. By independently and randomly removing edges multiple times, applying perturbations, and averaging the results, consistent positional encodings (PerturbPE) are obtained.

## Method

### Overall Architecture

PerturbPE is built upon MöbiusGCN. The overall pipeline is as follows: (1) Given a 2D human skeleton graph that may have missing edges, its graph Laplacian matrix is computed; (2) $\kappa$ independent perturbations are applied—each time randomly removing some edges and using RSPT to compute the perturbed eigenvectors; (3) The eigenvectors from the $\kappa$ perturbations are averaged to obtain the consistent positional encoding $\mathbf{P}$; (4) $\mathbf{P}$ is fused into the node features of each layer via an MLP, which is then fed into MöbiusGCN for 3D pose prediction.

### Key Designs

1. **Rayleigh-Schrödinger Perturbation Theory (RSPT) for Perturbed Eigenvectors**:

    - **Function**: Given the original graph Laplacian $\mathbf{A}_0$ and the perturbation matrix $\mathbf{A}_1$ (composed of the removed edges), efficiently compute the perturbed eigenpairs.
    - **Mechanism**: With $\mathbf{A}(\epsilon) = \mathbf{A}_0 + \epsilon \mathbf{A}_1$, a series expansion $\mathbf{v}_i(\epsilon) = \sum_{k=0}^{\infty} \epsilon^k \mathbf{v}_i^{(k)}$ is used for step-by-step approximation. In experiments, $\epsilon=1, k=1$ (first-order perturbation) is chosen, solved efficiently using Moore-Penrose pseudo-inverse and QR decomposition. Degenerate (repeated eigenvalues) and non-degenerate cases are handled separately.
    - **Design Motivation**: The core advantage of RSPT is that it avoids recomputing the entire eigendecomposition, requiring only the computation of pseudo-inverse vector products. For small graphs like the human skeleton (17 nodes), the computational overhead is negligible (inference time only increases from 0.009s to 0.010s).

2. **Averaging Multiple Perturbations to Extract Consistent Positional Encoding (PerturbPE)**:

    - **Function**: Independently perform $\kappa$ RSPT perturbations, each time randomly removing a different set of edges, and average the resulting perturbed eigenvectors.
    - **Mechanism**:
      $$\mathbf{p} = \frac{\sum_{i=1}^{\kappa} \mathbf{v}_i}{\kappa}$$
      where $\mathbf{v}_i$ is the eigenvector after the $i$-th perturbation. The averaging operation filters out the irregular components introduced by random perturbations, preserving the consistent structural information of the graph.
    - **Design Motivation**: This is based on the theory that the graph Laplacian eigenspace can be decomposed into consistent and irregular parts. The consistent part reflects the structural features unaffected by minor edge changes, which is precisely the needed positional encoding. Multiple averaging serves as a simple Monte Carlo approximation.

3. **Positional Feature Fusion and Masked Condition Training Strategy**:

    - **Function**: Fuse the PerturbPE encoding with node features by summation followed by an MLP, embedding it into each layer of MöbiusGCN.
    - **Mechanism**: $\mathbf{X}^{\ell} = \sigma(f(\mathbf{Z}^{\ell} + \mathbf{P}))$, where $f$ is an MLP. The complete MöbiusGCN block becomes:
      $$\mathbf{Z}^{\ell+1} = \sigma(2\Re\{\mathbf{U} \operatorname{Möbius}(\Lambda) \mathbf{U}^\top \sigma(f(\mathbf{Z}^{\ell} + \mathbf{P})) \mathbf{W}^{\ell+1}\} + \mathbf{b})$$
      During training, 0-2 edges are randomly removed per sample (Masked Condition Strategy) to enable the model to learn to handle various missing patterns.
    - **Design Motivation**: A single network adapted to all missing scenarios is more practical than training separate models for each specific missing pattern.

### Loss & Training

The standard MSE loss is used:
$$\mathcal{L}(\mathcal{Y}, \hat{\mathcal{Y}}) = \sum_{i=1}^{k}(\mathcal{Y}_i - \hat{\mathcal{Y}}_i)^2$$

Training details: Adam optimizer, initial learning rate of 0.001, and batch size of 64. 8 MöbiusGCN blocks are used, with intermediate channel sizes of 128 (0.16M parameters) or 192 (0.66M parameters). 2D inputs are normalized to $[-1,1]$, and scale is calibrated via bone length normalization during inference.

## Key Experimental Results

### Main Results

**Complete Skeleton (No Occlusion)**:

| Dataset | Metric | PerturbPE | MöbiusGCN | Gain |
|--------|------|-----------|-----------|------|
| Human3.6M (GT Input) | MPJPE(mm) | 32.7 | 34.1 | -1.4mm |
| MPI-INF-3DHP | PCK | 82.0 | 80.0 | +2.0 |

**Missing 1 Edge**:

| Configuration | MPJPE(mm) | Description |
|------|-----------|------|
| Eigenvector Sign (baseline) | 55.0 | Standard Laplacian PE |
| + Resolving Degeneracy | 51.4 | 6.5% Gain |
| + 1-Edge Perturbation | 49.0 | 10.9% Gain |
| + 2-Edge Perturbation | 48.0 | 12.7% Gain |

**Missing 2 Edges**:

| Configuration | MPJPE(mm) | Description |
|------|-----------|------|
| MöbiusGCN | 60.0 | No Positional Encoding |
| PerturbPE | 54.0 | 10% Reduction |

### Ablation Study

| Configuration | MPJPE | Description |
|------|-------|------|
| Training data reduced to 3 subjects | 42.9 vs 44.7 | Still outperforms MöbiusGCN |
| Training data reduced to 2 subjects | 48.9 vs 50.9 | Advantage maintained with less data |
| Training data reduced to 1 subject | 66.4 vs 67.4 | Marginal but consistent gain |

### Key Findings

- In occlusion scenarios (missing edges), the advantage of PerturbPE is significantly amplified, yielding up to a 12% improvement with 1 missing edge.
- Training a single network can handle all combinations of missing edges, outperforming GFPose's strategy of training separate models for specific missing patterns.
- Subject-specific occlusion experiments show that missing legs (52.4) outperforms missing arms (58.6), likely because leg structures are highly regular and easier to infer from other parts.
- PerturbPE does not introduce additional model parameters (only slight preprocessing overhead), resulting in almost unchanged inference time.

## Highlights & Insights

- Introducing the Rayleigh-Schrödinger Perturbation Theory from quantum mechanics into graph neural network positional encodings represents a highly innovative interdisciplinary connection.
- Understanding the occlusion issue through the lens of consistent/irregular decomposition of the graph Laplacian eigenspace provides solid theoretical support.
- The proposed method is extremely lightweight—introducing no additional parameters, requiring no structural changes, keeping inference time virtually constant, and improving performance purely through better positional encodings.

## Limitations & Future Work

- It assumes a known number of joints (only edges are missing), whereas real-world occlusions may render the joints themselves invisible.
- Increasing the number of perturbations $\kappa$ linearly increases the preprocessing time, and the paper does not discuss the optimal strategy for selecting $\kappa$.
- Validation is limited primarily to MöbiusGCN; its generalizability to other GCN architectures (aside from a preliminary validation on SemGCN) is not fully explored.
- Only Hourglass and Ground Truth (GT) are used as 2D inputs, without systematic testing on more modern 2D detectors (like HRNet).

## Related Work & Insights

- MöbiusGCN replaces standard graph convolutional filters with Möbius transformations and is currently the most lightweight 3D HPE method; this work adopts it as the base architecture.
- SignNet/BasisNet resolve the sign ambiguity and repeated eigenvalue issues of graph Laplacian positional encodings, but do not consider missing edges.
- GFPose trains separate networks for each occlusion pattern, whereas the proposed single-network PerturbPE handles all cases more elegantly.
- The underlying idea is highly generalizable—whenever graph structures have uncertain or missing edges, PerturbPE can be adopted to obtain robust positional encodings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introducing RSPT to GCN positional encoding offers a completely fresh perspective with a tight combination of theory and practice.
- Experimental Thoroughness: ⭐⭐⭐⭐ Experiments cover both complete skeletons and various missing edge scenarios, though 2D input sources are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivation is complete, but it is mathematically heavy, raising the barrier to entry.
- Value: ⭐⭐⭐⭐ Solves an important and often overlooked real-world problem with highly transferable ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[ECCV 2024\] UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)
- [\[ECCV 2024\] 3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)

</div>

<!-- RELATED:END -->

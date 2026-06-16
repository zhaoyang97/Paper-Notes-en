---
title: >-
  [Paper Note] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details
description: >-
  [AAAI 2026][Neural Signed Distance Function] This paper proposes a dual-branch architecture (generalization branch + overfitting branch) to learn a compact latent space over multiple neural SDFs. By combining a shared sp…
tags:
  - "AAAI 2026"
  - "Neural Signed Distance Function"
  - "Implicit Representation"
  - "Compact Latent Space"
  - "Volumetric Grid"
  - "Geometric Detail"
date: 2026-05-08
content_hash: 78abe2bae8a42ab8
---

# Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details

**Conference**: AAAI 2026
**arXiv**: [2511.14539](https://arxiv.org/abs/2511.14539)  
**Code**: [GitHub](https://github.com/eoozbq/Compact-SDF)  
**Area**: Other
**Keywords**: Neural Signed Distance Function, Implicit Representation, Compact Latent Space, Volumetric Grid, Geometric Detail

## TL;DR

This paper proposes a dual-branch architecture (generalization branch + overfitting branch) to learn a compact latent space over multiple neural SDFs. By combining a shared spatial feature grid with a novel bandwidth-based sampling strategy, the method recovers high-fidelity geometric details while maintaining compact latent codes, achieving state-of-the-art performance on Stanford Models, ShapeNet, and D-FAUST.

## Background & Motivation

Neural Signed Distance Functions (Neural SDFs) are a core approach to 3D shape representation, parameterizing a continuous implicit function via a neural network that can be queried at arbitrary spatial coordinates for the signed distance to a surface. However, existing methods face severe bottlenecks when simultaneously representing **multiple** SDFs while preserving high-fidelity geometric details.

**Two dominant paradigms and their limitations**:

**Generalization-based methods** (e.g., DeepSDF): Multiple shapes are encoded into a shared global latent space and decoded by an MLP. While capable of generalizing to novel shapes, these methods are limited by the network's preference for low-frequency signals (spectral bias) and fail to recover high-frequency geometric details such as sharp edges and fine holes.

**Overfitting-based methods** (e.g., Instant-NGP, MosaicSDF): Volumetric grids or multi-resolution hash tables store spatial features, enabling the recovery of fine-grained details. However, these methods typically overfit to a single shape and lack a compact latent space for representing multiple shapes. When multiple shapes share a feature grid, unbalanced sampling across shapes causes mutual interference and introduces artifacts.

**Key Challenge**: Generalization capability and high-frequency detail recovery are fundamentally at odds. Generalization methods offer compact encoding but sacrifice detail; overfitting methods preserve detail but lack a shared representational space.

**Key Insight**: The paper combines the strengths of both paradigms — the generalization branch handles regions far from the surface (where reasonable SDF values can be produced without dense sampling), while the overfitting branch handles near-surface regions (with dense sampling to recover high-frequency details). Both branches share a single compact shape code $\mathbf{z}$.

## Method

### Overall Architecture

The method consists of two parallel branches sharing a learnable latent code $\mathbf{z}$:

- **Generalization Branch**: Takes positional encoding $PE(\mathbf{x})$ and shape code $\mathbf{z}$ as input, and outputs a coarse SDF value $s_g$. The network's generalization capacity enables reasonable predictions even in sparsely sampled regions far from the surface.
- **Overfitting Branch**: Maintains a shared spatial feature grid ($128^3$ resolution, 128-dimensional features per vertex). Local features $\mathbf{c}$ are retrieved via trilinear interpolation and, together with $\mathbf{x}$ and $\mathbf{z}$, decoded into a fine SDF value $s_o$. The grid's high-frequency fitting capacity recovers surface geometric details.

At inference, a **signed distance fusion** strategy combines outputs from both branches: the generalization branch first produces a coarse reconstruction to identify the surface bandwidth $\mathbb{B}$; the overfitting branch is used within $\mathbb{B}$, while the generalization branch is used outside.

### Key Designs

1. **Dual-Branch Architecture and Signed Distance Fusion**:

    - Function: Partitions the SDF query space into near-surface and far-surface regions, handled by respective branches.
    - Mechanism: The fusion is formulated as $s = s_o$ if $\mathbf{x} \in \mathbb{B}$, otherwise $s = s_g$, where $\mathbb{B}$ is a bandwidth region extending $n$ voxel layers on each side of the surface.
    - Design Motivation: The generalization branch suppresses artifacts caused by unbalanced sampling far from the surface (interference between shape features), while the overfitting branch recovers sharp edges and fine structures near the surface. The two branches are complementary.

2. **Bandwidth-Constrained Sampling Strategy**:

    - Function: A novel training sampling scheme designed for the overfitting branch.
    - Mechanism: Sparse samples are first drawn uniformly across the full space at low resolution ($128^3$); dense samples are then drawn within the surface bandwidth $\mathbb{B}$ at high resolution ($512^3$); duplicates are removed after merging.
    - Design Motivation: Enforcing balanced per-voxel sampling across all shapes at high resolution would incur an $O(N \times R^3)$ cost, which is computationally prohibitive. The proposed strategy balances efficiency and quality by sampling densely near the surface and sparsely elsewhere, with the generalization branch compensating for sparse coverage in far-surface regions.

3. **Shared Spatial Feature Grid**:

    - Function: Multiple shapes share a single $128^3 \times 128$-dimensional feature grid, distinguished by their respective latent codes $\mathbf{z}$.
    - Mechanism: For a query point $\mathbf{x}$, features $\mathbf{c}$ are retrieved via trilinear interpolation from the grid and decoded together with $\mathbf{z}$.
    - Design Motivation: Unlike hash grids (Instant-NGP), the explicit shared grid avoids hash collisions and, in conjunction with shape codes, better disambiguates different shapes.

### Loss & Training

The total loss consists of four terms:

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{ovf}} + \lambda_1 \mathcal{L}_z + \lambda_2 \mathcal{L}_c$$

- $\mathcal{L}_{\text{gen}}$: MSE loss for the generalization branch.
- $\mathcal{L}_{\text{ovf}}$: MSE loss for the overfitting branch.
- $\mathcal{L}_z$: Regularization on shape code $\mathbf{z}$ (prevents overfitting and encourages compactness).
- $\mathcal{L}_c$: Regularization on grid features $\mathbf{c}$.

Training details: grid feature learning rate $10^{-1}$, shape code learning rate $10^{-3}$; both networks are 8-layer MLPs with 512 units; latent code dimensionality is 256; total training is 4000 epochs.

## Key Experimental Results

### Main Results (Single Complex Shape Reconstruction — Stanford Models)

| Method | CD(↓) | F-Score(↑) | Precision(↑) | Recall(↑) |
|--------|-------|-----------|-------------|----------|
| ACORN | 6.76e-05 | 0.982 | 0.967 | 0.998 |
| NGLOD | 6.77e-05 | 0.980 | 0.968 | 0.994 |
| Instant-NGP | 7.37e-05 | 0.976 | 0.962 | 0.990 |
| MosaicSDF | 1.30e-03 | 0.902 | 0.846 | 0.972 |
| HyperDiffusion | 1.20e-04 | 0.835 | 0.847 | 0.823 |
| **Ours** | **6.56e-05** | **0.983** | **0.969** | **0.998** |

### Multi-Shape Reconstruction (ShapeNet, CD×10⁻⁴)

| Method | Bench | Chair | Plane | Table | Lamp | Sofa | Avg. |
|--------|-------|-------|-------|-------|------|------|------|
| DeepSDF | 4.890 | 8.630 | 2.660 | 6.330 | 14.63 | 5.040 | 7.030 |
| IF-NET | 1.340 | 1.000 | 0.225 | 0.857 | 0.817 | 1.100 | 0.890 |
| Instant-NGP | 0.881 | 1.210 | 0.664 | 1.030 | 1.880 | 1.100 | 1.128 |
| HyperDiffusion | 1.640 | 1.490 | 2.310 | 1.470 | 6.990 | 2.830 | 2.788 |
| **Ours** | **0.463** | **0.898** | **0.223** | **0.790** | **0.517** | **0.751** | **0.607** |

### Ablation Study

| Configuration | CD(×10⁻⁴) | Notes |
|--------------|-----------|-------|
| Dual-branch (Ours) | 4.01 | Full method |
| Generalization branch only | 4.05 | Loss of high-frequency detail |
| Overfitting branch only | 11.6 | Severe artifacts far from surface |
| Bandwidth $n=1$ | 4.29 | Bandwidth too narrow; insufficient overfitting contribution |
| Bandwidth $n=6$ | 14.6 | Bandwidth too wide; artifacts propagate from overfitting branch |

| Sampling Strategy | CD(×10⁻⁴) | Sample Count |
|------------------|-----------|-------------|
| Uniform $32^3$ | 35.900 | $32^3$ |
| Uniform $128^3$ | 0.890 | $128^3$ |
| Uniform $256^3$ | 0.417 | $256^3$ |
| **Ours** | **0.401** | **≈$220^3$** |

### Key Findings

- In multi-shape reconstruction, the proposed method achieves an average CD of $0.607\times10^{-4}$, which is 32% lower than the second-best method IF-NET ($0.890$).
- Bandwidth parameter $n=3$ yields the best trade-off; performance degrades when $n$ is either too large or too small.
- Latent code performance saturates at dimensionality 64 (CD decreases from 0.443 at $d=8$ to 0.397 at $d=64$).
- The proposed sampling strategy achieves superior results to uniform $256^3$ sampling using only approximately $220^3$ samples.

## Highlights & Insights

- The core insight is precise: generalization and overfitting are fundamentally optimal strategies for different spatial regions of the SDF, and fusing them via spatial division of labor is a clean and effective approach.
- The problem of unbalanced sampling across shapes on a shared feature grid is clearly identified and elegantly resolved through bandwidth restriction combined with generalization branch compensation.
- In shape interpolation experiments, the proposed method produces more pronounced interpolation variation than DeepSDF and is more compact than HyperDiffusion (256-dimensional code vs. full MLP parameters).

## Limitations & Future Work

- The method requires watertight meshes as input and cannot directly process point clouds or noisy scan data.
- Memory consumption of the $128^3$ feature grid grows linearly with the number of shape categories.
- Inference requires two stages (coarse reconstruction to determine bandwidth, then fine-grained fusion), increasing inference time.
- The shape code dimensionality saturates at 64 for 100 shapes; scalability to large-scale datasets with potentially higher required dimensionality remains to be verified.

## Related Work & Insights

- Key distinction from MosaicSDF: MosaicSDF concentrates the grid near the surface but remains a single-shape method; this paper extends the grid to a multi-shape shared representation disambiguated by latent codes.
- Hash collision in Instant-NGP is particularly problematic in multi-shape scenarios, which explains its performance degradation on multi-shape tasks.
- The dual-branch design may generalize to NeRF or other implicit representation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](../../ICLR2026/others/out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)
- [\[ICLR 2026\] A Single Architecture for Representing Invariance Under Any Space Group](../../ICLR2026/others/a_single_architecture_for_representing_invariance_under_any_space_group.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[AAAI 2026\] Formal Abductive Latent Explanations for Prototype-Based Networks](formal_abductive_latent_explanations_for_prototype-based_networks.md)
- [\[ICML 2026\] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](../../ICML2026/others/disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)

</div>

<!-- RELATED:END -->

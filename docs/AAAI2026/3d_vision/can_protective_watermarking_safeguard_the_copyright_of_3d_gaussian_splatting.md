---
title: >-
  [Paper Note] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?
description: >-
  [AAAI2026][3D Vision][3D Gaussian Splatting] This paper presents the first systematic study exposing the vulnerability of 3DGS watermarking frameworks, and proposes GSPure — a purification framework that leverages view-aware Gaussian weight accumulation and geometric feature clustering to precisely isolate and remove watermark-related Gaussian primitives, reducing watermark PSNR by up to 16.34 dB while incurring less than 1 dB loss in scene fidelity.
tags:
  - AAAI2026
  - 3D Vision
  - 3D Gaussian Splatting
  - watermark purification
  - copyright protection
  - HDBSCAN clustering
date: 2026-05-08
content_hash: 6fef089ae99b2715
---

# Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?

**Conference**: AAAI2026  
**arXiv**: [2511.22262](https://arxiv.org/abs/2511.22262)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, watermark purification, copyright protection, HDBSCAN clustering

## TL;DR

This paper presents the first systematic study exposing the vulnerability of 3DGS watermarking frameworks, and proposes GSPure — a purification framework that leverages view-aware Gaussian weight accumulation and geometric feature clustering to precisely isolate and remove watermark-related Gaussian primitives, reducing watermark PSNR by up to 16.34 dB while incurring less than 1 dB loss in scene fidelity.

## Background & Motivation

### State of the Field

3D Gaussian Splatting (3DGS) has emerged as the dominant approach for efficient 3D scene representation and rendering, achieving high-fidelity visual quality and computational efficiency through anisotropic Gaussian primitives. Given the high training cost, 3DGS models constitute valuable digital assets, and a series of watermarking schemes (e.g., GS-Hider, Splats in Splats, SecureGS) have been proposed for ownership verification and provenance tracking.

However, existing work focuses exclusively on **how to embed watermarks**, while systematic adversarial evaluation of watermark robustness remains absent. Traditional 2D watermark removal methods (cropping, rotation, neural network erasure, etc.) cannot be directly transferred to the 3DGS setting, since 3DGS watermarks are embedded into the geometric and photometric attributes of the model rather than the surface of rendered images. This gap motivates the authors to conduct the first systematic security analysis of 3DGS watermarks from an attacker's perspective.

### Root Cause

**Core Problem**: **Are existing 3DGS watermarking schemes truly secure?** The authors find the answer is no. The key observation is that watermark-related Gaussian primitives exhibit significantly different rendering contribution patterns from scene primitives across multiple views — watermark primitives tend to show inconsistent or view-specific behavior, with substantially lower rendering contribution weights than scene primitives. This discrepancy enables precise separation.

## Method

The GSPure framework consists of three core modules:

### 1. View-Aware Gaussian Weight Accumulation

For each Gaussian primitive $\mathcal{G}_k$, its contribution weight is computed across multiple rendered viewpoints. Specifically:

- A ray-Gaussian intersection energy $\mathcal{E}(\mathcal{G}_k, \mathbf{o}_v, \mathbf{r}_v)$ is defined to measure the degree of intersection between a ray and a Gaussian under viewpoint $v$
- Occlusion effects are accounted for via the alpha-blending mechanism, yielding a view-dependent contribution $\omega_{k,v}$ weighted by the accumulated transmittance of preceding Gaussians
- The view-aggregated weight is obtained by averaging over $N$ viewpoints: $\omega_k = \frac{1}{N}\sum_{v=1}^{N}\omega_{k,v}$

Key intuition: due to their inconsistent rendering behavior, watermark primitives tend to have low $\omega_k$ values, whereas scene primitives maintain consistently high contributions across views.

### 2. Geometrically Accurate Feature Clustering

Weight alone cannot perfectly separate watermark primitives (due to viewpoint discontinuities and scene-watermark coupling), so a joint feature representation is constructed:

- Position $\mathbf{p}_k$, opacity $\alpha_k$, and accumulated weight $\omega_k$ are individually normalized and concatenated into a high-dimensional feature vector $\mathbf{F}_k$
- Adaptive density-based clustering via HDBSCAN is applied to automatically discover watermark-related Gaussian clusters and noise points

### 3. Adaptive Pruning

A two-level pruning strategy is designed:

- **Cluster-level pruning**: the mean weight $\widetilde{\Omega}(C_i)$ of each cluster is computed; clusters with mean weight below the threshold $\bar{\omega}/\tau_c$ are removed entirely
- **Noise-point-level pruning**: for points not assigned to any cluster, individual points with weight $\omega_k < \bar{\omega}/\tau_n$ are removed
- The threshold factors $\tau_c$ and $\tau_n$ are dynamically adjusted based on the global mean weight, balancing watermark removal effectiveness and scene fidelity

## Key Experimental Results

Evaluation is conducted on the Mip-NeRF 360 dataset against three mainstream 3DGS watermarking methods:

### Main Results

| Attack Method | Splats in Splats Score | GS-Hider Score | SecureGS Score |
|---------|----------------------|----------------|----------------|
| Random Pruning | -0.88 | 0.68 | -1.59 |
| Feature Scaling | -9.38 | -1.42 | 0.39 |
| Gaussian Noise | -7.15 | 0.11 | -0.55 |
| GOF | 2.24 | -0.93 | 0.64 |
| **GSPure** | **15.21** | **10.16** | **5.03** |

- Score is defined as $\Delta PSNR_{message} - \Delta PSNR_{scene}$; higher is better (more watermark removal, less scene degradation)
- GSPure achieves substantially higher scores than all baselines across all three watermarking frameworks (leading by 12.97, 9.48, and 4.39, respectively)
- Maximum watermark PSNR reduction reaches 16.34 dB, with scene fidelity loss generally below 1 dB

### Ablation Study

Ablation experiments confirm that the combination of weight accumulation, opacity, and clustering achieves the best performance; removing any single component leads to degraded watermark removal or increased scene damage.

## Highlights & Insights

1. **Novel problem formulation**: The first systematic adversarial evaluation of 3DGS watermark security from an attacker's perspective, exposing a largely overlooked vulnerability in the field
2. **Elegant method design**: Exploits the intrinsic difference in multi-view rendering contributions between watermark and scene primitives, requiring no prior knowledge of the watermarking scheme
3. **Strong generalizability**: Effective against three watermarking schemes based on fundamentally different technical designs (SH encryption, decoder hiding, and anchor-based design)
4. **Compelling visualization**: Point cloud cluster visualizations clearly demonstrate the spatial clustering of watermark primitives, intuitively validating the method's reliability

## Limitations & Future Work

1. **Manual threshold tuning**: $\tau_c$ and $\tau_n$ require different settings for different watermarking methods (e.g., (2,3) for SecureGS vs. the default (4,4)), leaving room for further automation
2. **Relatively weaker performance on SecureGS**: The anchor-based watermarking design of Scaffold-GS proves harder to remove, yielding a Score of only 5.03 (vs. 15.21 for Splats in Splats)
3. **Evaluation limited to scene-hiding watermarks**: Rendering image-level watermarks and other protection mechanisms are not considered
4. **Adversarial watermarking not discussed**: If a watermarking scheme specifically hardens against GSPure's clustering strategy, the effectiveness of the current approach may diminish
5. **Computational overhead not analyzed**: The CUDA implementation of 3DGS is modified to compute rendering contribution weights, but no additional runtime cost is reported

## Related Work & Insights

| Method | Type | 3DGS-Specific | Scene Quality Preservation | Watermark Removal |
|------|------|----------|------------|------------|
| Random Pruning | Naive baseline | Partially applicable | Moderate | Poor |
| Feature Scaling | 2D transfer | Partially applicable | Poor | Poor |
| Gaussian Noise Injection | 2D transfer | Partially applicable | Moderate | Poor |
| GOF Surface Extraction | Geometric | Applicable | Very poor | Moderate |
| **GSPure** | **3D-dedicated** | **Purpose-built** | **Good** | **Excellent** |

While GOF can remove some watermarks, it causes irreversible damage to the original scene (scene PSNR drop can exceed 10 dB); GSPure achieves the best trade-off between the two objectives.

- **Security–attack co-evolution**: This work establishes an important security benchmark for 3DGS watermark protection from an attack perspective, driving the development of more robust watermarking schemes
- **View consistency as a signal**: The idea of using multi-view rendering consistency to disentangle information can be generalized to other 3DGS editing and analysis tasks
- **Clustering + pruning paradigm**: The application of HDBSCAN at the Gaussian primitive level demonstrates the potential of density-based clustering on unstructured 3D point clouds
- **Connection to NeRF watermarking research**: The adversarial research paradigm for 3DGS watermarks can draw on adversarial robustness work in the NeRF domain

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First systematic attack on 3DGS watermarks, opening a new research direction)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three watermarking methods × nine scenes × five baselines, complete ablation study, but lacks computational cost analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-motivated, excellent visualizations)
- Value: ⭐⭐⭐⭐⭐ (Carries significant cautionary implications for the field of 3DGS copyright protection)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Where, What, Why: Toward Explainable 3D-GS Watermarking](../../CVPR2026/3d_vision/where_what_why_toward_explainable_3d-gs_watermarking.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[AAAI 2026\] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting](splats_in_splats_robust_and_effective_3d_steganography_towards_gaussian_splattin.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)

<!-- RELATED:END -->

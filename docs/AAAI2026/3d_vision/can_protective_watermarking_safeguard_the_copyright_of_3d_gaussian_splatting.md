---
title: >-
  [Paper Note] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?
description: >-
  [AAAI2026][3D Vision][3D Gaussian Splatting] This work systematically reveals the vulnerability of 3DGS watermarking frameworks for the first time, proposing the GSPure framework. GSPure accurately separates and removes watermark-related Gaussian primitives through view-aware weight accumulation and geometric feature clustering, reducing watermark PSNR by up to 16.34dB while keeping the original scene loss below 1dB.
tags:
  - "AAAI2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "watermark purification"
  - "copyright protection"
  - "HDBSCAN clustering"
date: 2026-05-08
content_hash: 306f762f0a3f9141
---

# Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?

**Conference**: AAAI2026  
**arXiv**: [2511.22262](https://arxiv.org/abs/2511.22262)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, watermark purification, copyright protection, HDBSCAN clustering

## TL;DR

This work systematically reveals the vulnerability of 3DGS watermarking frameworks for the first time, proposing the GSPure framework. GSPure accurately separates and removes watermark-related Gaussian primitives through view-aware weight accumulation and geometric feature clustering, reducing watermark PSNR by up to 16.34dB while keeping the original scene loss below 1dB.

## Background & Motivation

### Background

**Background**: 3D Gaussian Splatting (3DGS) has emerged as a mainstream method for efficient 3D scene representation and rendering, achieving high-fidelity visual quality and computational efficiency via anisotropic Gaussian primitives. Due to high training costs, 3DGS models inherently possess significant digital asset value, prompting the development of various watermarking schemes (e.g., GS-Hider, Splats in Splats, SecureGS) for ownership verification and provenance tracking.

However, existing works focus solely on **how to embed watermarks**, neglecting systematic robustness evaluations against potential attacks. Traditional 2D watermark removal methods (cropping, rotation, neural network erasure, etc.) cannot be directly transferred to 3DGS scenarios because 3DGS watermarks are embedded into the geometric and photometric attributes of the model, rather than just on the rendered image surface. This gap motivates the authors to systematically examine the security of 3DGS watermarks from an attacker's perspective for the first time.

### Key Challenge

**Key Challenge**: **Goal**: **Are existing 3DGS watermarking schemes truly secure?** The authors find the answer is negative. The core observation is that the contribution pattern of watermark-related Gaussian primitives when rendering the original scene differs significantly from that of scene primitives—watermark primitives typically exhibit inconsistent or view-specific behavior during multi-view rendering, and their rendering contribution weights are notably lower than those of scene primitives. This discrepancy enables precise separation.

## Method

The GSPure framework consists of three core modules:

### 1. View-Aware Gaussian Weight Accumulation

For each Gaussian primitive $\mathcal{G}_k$, its contribution weight is computed across multiple rendering views. Specifically:

- Define the ray-Gaussian intersection energy $\mathcal{E}(\mathcal{G}_k, \mathbf{o}_v, \mathbf{r}_v)$, which measures the intersection between the ray and the Gaussian under a given view $v$
- Account for occlusion effects by calculating the view-dependent contribution $\omega_{k,v}$ using the alpha-blending mechanism, which represents the accumulated transmittance through preceding Gaussians
- Average across $N$ views to obtain the view-accumulated weight $\omega_k = \frac{1}{N}\sum_{v=1}^{N}\omega_{k,v}$

Key intuition: Due to their inconsistent rendering behavior, watermark primitives typically exhibit lower $\omega_k$ values, whereas the contributions of scene primitives remain consistent and high across views.

### 2. Geometrically Accurate Feature Clustering

Relying solely on weights is insufficient for perfect separation (due to view discontinuity and scene-watermark coupling issues), hence a joint feature is constructed:

- Normalize and concatenate position $\mathbf{p}_k$, opacity $\alpha_k$, and accumulated weight $\omega_k$ into a high-dimensional feature vector $\mathbf{F}_k$
- Employ the adaptive density clustering algorithm HDBSCAN to perform grouping, automatically identifying watermark-related Gaussian clusters and noise points

### 3. Adaptive Pruning

Design a dual-level pruning strategy:

- **Cluster-level pruning**: Calculate the average weight $\widetilde{\Omega}(C_i)$ for each cluster. If it falls below a threshold $\bar{\omega}/\tau_c$, the entire cluster is removed.
- **Noise-point-level pruning**: For noise points not assigned to any cluster, they are individually removed if their single-point weight meets $\omega_k < \bar{\omega}/\tau_n$.
- Threshold factors $\tau_c$ and $\tau_n$ are dynamically adjusted based on the global average weight, balancing watermark removal efficiency and scene fidelity.

## Key Experimental Results

On the Mip-NeRF 360 dataset, three major 3DGS watermarking methods are evaluated:

### Main Results

| Attack Method | Splats in Splats Score | GS-Hider Score | SecureGS Score |
|---------|----------------------|----------------|----------------|
| Random Pruning | -0.88 | 0.68 | -1.59 |
| Feature Scaling | -9.38 | -1.42 | 0.39 |
| Gaussian Noise | -7.15 | 0.11 | -0.55 |
| GOF | 2.24 | -0.93 | 0.64 |
| **GSPure** | **15.21** | **10.16** | **5.03** |

- Score Definition: $\Delta PSNR_{message} - \Delta PSNR_{scene}$; higher is better (indicates more watermark removal and less scene quality degradation).
- GSPure significantly outperforms the second-best method across all three watermarking frameworks (by 12.97, 9.48, and 4.39, respectively).
- The maximum reduction in watermark PSNR reaches up to 16.34dB, while the loss in scene fidelity is generally <1dB.
- Ablation study confirms that the joint utilization of weight accumulation, opacity, and clustering yields the optimal performance, whereas lacking any of these components degrades performance or increases scene damage.

## Highlights & Insights

1. **Novel Problem Formulation**: Systematically evaluates the security of 3DGS watermarks from an attacker's perspective for the first time, exposing a neglected vulnerability in this domain.
2. **Elegant Method Design**: Utilizes the intrinsic difference in multi-view rendering contributions between watermark primitives and scene primitives, requiring no prior knowledge of the watermarking scheme.
3. **High Versatility**: Demonstrates efficacy against three watermarking approaches with fundamentally different technical routes (SH encryption, decoder-based hiding, and anchor design).
4. **Convincing Visualization**: Point cloud clustering visualizations clearly depict the spatial aggregation characteristics of watermark primitives, intuitively validating the reliability of the proposed method.

## Limitations & Future Work

1. **Manual Threshold Adjustment**: $\tau_c$ and $\tau_n$ require different settings for different watermarking methods (e.g., SecureGS uses (2,3) instead of the default (4,4)), leaving room for higher automation.
2. **Relatively Weaker Effect on SecureGS**: Watermarks based on Scaffold-GS anchor designs are harder to remove, yielding a score of only 5.03 (vs. 15.21 for Splats in Splats).
3. **Only Evaluated on Scene-Hiding Watermarks**: Rendered image-level watermarks or other protective mechanisms are not covered.
4. **Lack of Discussion on Adversarial Watermarks**: If a watermarking scheme is specifically reinforced to counter GSPure's clustering strategy, the effectiveness of the current method may decrease.
5. **Omission of Detailed Computational Overhead Analysis**: Although the CUDA implementation of 3DGS was modified to compute rendering contribution weights, the extra time cost is not reported.

## Related Work & Insights

| Method | Type | Tailored for 3DGS | Scene Quality Preservation | Watermark Removal Effect |
|------|------|----------|------------|------------|
| Random Pruning | Naive Baseline | Partially | Medium | Poor |
| Feature Scaling | 2D Migration | Partially | Poor | Poor |
| Gaussian Noise Injection | 2D Migration | Partially | Medium | Poor |
| GOF Surface Extraction | Geometric Method | Yes | Very Poor | Medium |
| **GSPure** | **3D-Specific** | **Specially Designed** | **Good** | **Excellent** |

While GOF can remove some watermarks, it causes irreversible damage to the original scene (reducing Scene PSNR by up to 10+dB); GSPure achieves the optimal balance between the two.

## Related Work & Insights

- **Co-evolution of Security and Attacks**: This work establishes an important security benchmark for 3DGS watermark protection from an "attack" perspective, driving the development of more robust watermarking schemes.
- **View Consistency as a Signal**: The idea of using multi-view rendering consistency to separate information can be generalized to other 3DGS editing/analysis tasks.
- **Clustering + Pruning Paradigm**: The application of HDBSCAN at the Gaussian primitive level demonstrates the potential of density clustering in unstructured 3D point clouds.
- **Connection to NeRF Watermarking Research**: The research paradigm of 3DGS watermark attacks can draw inspiration from adversarial robustness studies in the NeRF domain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First systematic attack on 3DGS watermarks, pioneering a new direction)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three watermarking methods × nine scenes × five baselines, complete ablation, but lacks computational cost analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-motivated, excellent visualizations)
- Value: ⭐⭐⭐⭐⭐ (Holds significant warning value for the field of 3DGS copyright protection)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CompMarkGS: Robust Watermarking for Compressed 3D Gaussian Splatting](../../ICLR2026/3d_vision/compmarkgs_robust_watermarking_for_compressed_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Robust3DGSW: Toward Robust Watermarking for Quantization-Aware 3D Gaussian Splatting](../../CVPR2026/3d_vision/robust3dgsw_toward_robust_watermarking_for_quantization-aware_3d_gaussian_splatt.md)
- [\[ICLR 2026\] NGS-Marker: Robust Native Watermarking for 3D Gaussian Splatting](../../ICLR2026/3d_vision/ngs-marker_robust_native_watermarking_for_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Where, What, Why: Toward Explainable 3D-GS Watermarking](../../CVPR2026/3d_vision/where_what_why_toward_explainable_3d-gs_watermarking.md)
- [\[ECCV 2024\] Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model](../../ECCV2024/3d_vision/protecting_nerfsapos_copyright_via_plug-and-play_watermarking_base_model.md)

</div>

<!-- RELATED:END -->

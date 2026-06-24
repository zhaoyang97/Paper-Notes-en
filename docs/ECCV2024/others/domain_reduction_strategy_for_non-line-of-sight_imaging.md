---
title: >-
  [Paper Note] Domain Reduction Strategy for Non-Line-of-Sight Imaging
description: >-
  [ECCV 2024][Non-Line-of-Sight Imaging] This paper proposes an optimization-based method for non-line-of-sight (NLOS) imaging. By modeling the transient signal as a superposition of point-wise light propagation functions and designing a coarse-to-fine domain reduction strategy to prune empty regions, it achieves approximately 20x acceleration under general NLOS scenarios while simultaneously reconstructing both albedo and surface normals.
tags:
  - "ECCV 2024"
  - "Non-Line-of-Sight Imaging"
  - "NLOS"
  - "Domain Reduction"
  - "Optimization-based Reconstruction"
  - "Surface Normals"
date: 2026-05-08
content_hash: 016191272ec7b1ff
---

# Domain Reduction Strategy for Non-Line-of-Sight Imaging

**Conference**: ECCV 2024  
**arXiv**: [2308.10269](https://arxiv.org/abs/2308.10269)  
**Code**: [GitHub](https://github.com/hyunbo9/domain-reduction-strategy)  
**Area**: Human Understanding / Computational Imaging  
**Keywords**: Non-Line-of-Sight Imaging, NLOS, Domain Reduction, Optimization-based Reconstruction, Surface Normals

## TL;DR

This paper proposes an optimization-based method for non-line-of-sight (NLOS) imaging. By modeling the transient signal as a superposition of point-wise light propagation functions and designing a coarse-to-fine domain reduction strategy to prune empty regions, it achieves approximately 20x acceleration under general NLOS scenarios while simultaneously reconstructing both albedo and surface normals.

## Background & Motivation

Non-Line-of-Sight (NLOS) imaging aims to reconstruct hidden scenes via indirect reflections from a relay wall, with critical applications in autonomous driving, medical imaging, and search-and-rescue. Existing methods fall into two main categories:

**FFT-based inverse methods** (LCT, FK, Phasor): Fast but require strict assumptions (planar relay walls, confocal scanning, dense raster scanning), and most can only reconstruct albedo without recovering surface geometry.

**Optimization-based methods** (e.g., NeTF): Highly versatile and unconstrained by scanning setups, but suffer from massive computational overhead ($O(N^5)$ or $O(N^6)$ per iteration), wasting significant computation on empty regions.

Key Observation: The visible surface of target objects in NLOS scenarios is extremely sparse, **occupying less than 5% of the hidden space**. This indicates that the vast majority of computation is wasted on empty regions, which represents a massive operational overhead.

## Method

### Overall Architecture

The core concept of the proposed method is:
1. Model the transient measurements as a linear superposition of light propagation functions from individual points in the hidden space.
2. Periodically prune empty regions (domain reduction) during optimization, retaining only active regions.
3. Employ a coarse-to-fine strategy to progressively increase the resolution.

The input consists of transient signals $\tau(t, \mathbf{l}, \mathbf{s})$ measured by time-resolved sensors, and the outputs are the albedo $\rho$ and surface normals $\vec{n}$ of the hidden scene.

### Key Designs

#### 1. Point-wise Light Propagation Model

Decompose the transient signal into independent contributions from each hidden point $\mathbf{p}$:

$$\tau = \int_{\mathbf{p} \in \Omega} \rho(\mathbf{p}) \cdot g_{\mathbf{p}} \, d\mathbf{p}$$

The light propagation function $g_{\mathbf{p}}$ for each point comprises three factors:
- **Cosine term $\Phi_{\mathbf{p}}$**: A view-dependent term based on Lambert's cosine law, modeled using surface normals.
- **Distance attenuation term $\Upsilon_{\mathbf{p}}$**: $1/d_l^2 \cdot 1/d_s^2$, representing the inverse-square distance decay from the laser to the point and from the point to the sensor.
- **Temporal constraint $\delta$**: Specifies that the optical path length equals the measurement time multiplied by the speed of light.

Key Advantage: This model imposes no assumptions on the relay wall geometry or scanning system, inherently supporting non-planar walls and non-confocal configurations.

#### 2. Randomized Sampling-based Reconstruction Scheme

- Divide the hidden space into a grid, where each vertex is assigned a 4D variable (albedo $\rho$ + normal $\vec{n}$).
- Randomly sample a set of points from the grid at each step (one point per cell) and obtain $\rho$ and $\vec{n}$ via trilinear interpolation.
- Compute the light propagation function for each point and perform linear superposition to obtain the predicted transient signal.
- Optimize the variables by minimizing the L2 distance.

Implemented with matrix-free CUDA kernels, enabling GPU-parallelized light propagation computation for all sampled points.

#### 3. Domain Reduction Strategy

The core innovation is divided into three levels:

**Basic Domain Reduction**: Periodically (every 50 steps) check the albedo, pruning regions where $\rho < \epsilon$ (5% of the maximum value) from the sampling domain, and sampling only within active regions.

**Soft Domain Reduction**: To prevent accidental deletion of non-empty regions, a Gaussian kernel low-pass filter is first applied to the albedo voxels before threshold-based pruning. This effectively expands the domain to surrounding areas, granting accidentally pruned regions a second chance.

**Coarse-to-Fine Strategy**:
- Initially utilize a coarse grid for fast computation.
- Once the domain is sufficiently reduced, subdivide the active grid into a higher resolution.
- Initialize the new grid variables using trilinear interpolation.
- Ultimately achieve high-resolution ($128 \times 128$) reconstruction on a single GPU.

#### 4. Noise Regularization

- L1 regularization encourages albedo sparsity.
- Jointly optimize the learnable noise parameter $d(t, \mathbf{l}, \mathbf{s}) = b + \lambda\sigma(z)$ to account for ambient light and dark count effects.
- $b$ controls the minimum noise level, and $\lambda$ controls the maximum.

### Loss & Training

The optimization objective is the L2 distance between predicted and ground-truth transients plus the L1 albedo regularization. The Adam optimizer is utilized with a learning rate of 1 for 1000 steps. Domain reduction is executed every 50 steps.

## Key Experimental Results

### Main Results (Sparse Scanning 32×32 Confocal Conf.)

| Method | Depth MAE(5%)↓ | Depth RMSE(5%)↓ | Normal MAE(5%)↓ | Normal RMSE(5%)↓ |
|------|-------------|--------------|-------------|--------------|
| LCT | 0.1977 | 0.2875 | - | - |
| FK | 0.0719 | 0.1871 | - | - |
| Phasor(BP) | 0.1348 | 0.2049 | - | - |
| NeTF | 0.0679 | 0.1748 | - | - |
| DLCT | 0.3189 | 0.4220 | 0.3796 | 0.4856 |
| **Ours** | **0.0477** | **0.1523** | **0.1147** | **0.2394** |

Qualitative Results: On both synthetic (ZNLOS Bunny/Serapis) and real (Stanford Statue/Dragon) datasets, the proposed method yields the sharpest reconstruction with the richest details. Under $32 \times 32$ sparse scanning, DLCT suffers from distinct artifacts, FK loses details, and NeTF produces blurred and distorted results.

### Ablation Study

| Domain Reduction | Coarse-to-Fine | Time (100 steps) | Time (500 steps) | Time (1000 steps) | Remaining Domain % |
|--------|--------|-----------|-----------|------------|-----------|
| ✗ | ✗ | 109s | 539s | 1087s | 100% |
| ✓ | ✗ | 67s | 103s | 134s | 4% |
| ✓ | ✓ | 4s | 27s | **54s** | 3% |

GPU memory comparison ($N=128$ resolution): LCT 7101MB, FK 6813MB, Ours (w/o DR) 4065MB, **Ours (w/ DR) 1659MB**

### Key Findings

- Domain reduction combined with the coarse-to-fine strategy achieves approximately **20x acceleration** (1087s → 54s).
- A sparse scanning grid of only $32 \times 32$ is sufficient to achieve high-resolution reconstruction, eliminating the need for dense raster scanning.
- Increasing the scanning resolution to $64 \times 64$ yields no significant quality improvement, indicating that $32 \times 32$ is sufficient.
- Noise regularization (L1 + learnable noise) is crucial for robustness on real-world data.
- High-resolution $128 \times 128$ outputs can be reconstructed **within 1 minute**.
- Domain reduction does not sacrifice reconstruction quality—there is no visual difference in reconstruction results with or without domain reduction.

## Highlights & Insights

1. **Flexibility of Point-wise Decomposition**: Decomposing transient signals into point-wise contributions makes domain reduction trivial—requiring sampling only within active regions.
2. **Profound Exploitation of Sparsity**: In NLOS scenarios, objects occupy less than 5% of the space. This "curse" of dimensionality is instead exploited as a key advantage for algorithmic acceleration.
3. **Balancing Versatility and Efficiency**: Inherits the flexibility of optimization-based methods without requiring strict FFT-like assumptions, while mitigating efficiency limitations via domain reduction.
4. **Simultaneous Albedo and Normal Reconstruction**: Accurate modeling of the cosine term allows surface geometry to be recovered at no extra cost.

## Limitations & Future Work

- Assumed that the hidden scene follows a Lambertian reflectance model, ignoring inter-reflections and self-occlusions.
- The domain reduction threshold (5% of the maximum albedo) is manually set; adaptive thresholds could be explored in future work.
- Under non-confocal configurations, the threshold needs to be lowered (to 3%), suggesting further study on parameter sensitivity.
- Combining the method with neural field representations (e.g., NeRF) could further enhance detailed reconstruction capabilities.

## Related Work & Insights

- **LCT/DLCT/FK**: FFT-based inverse methods. Fast but limited by scanning assumptions. DLCT is the only FFT-based method capable of normal reconstruction.
- **NeTF**: Introduces NeRF to NLOS. Versatile but inefficient and yields blurry results.
- **Phasor Field**: An NLOS method from a wave-optics perspective, requiring planar walls and confocal scanning.
- The core idea of this method is similar to the coarse-to-fine strategies and octree acceleration used in 3D reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of point-wise decomposition and domain reduction is novel in NLOS.
- Technical Depth: ⭐⭐⭐⭐⭐ — Robust CUDA implementation, light propagation model, and noise regularization.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Includes synthetic and real-world datasets, multiple scanning setups, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear methodological descriptions and easy-to-understand pseudocode.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Synergy of Sight and Semantics: Visual Intention Understanding with CLIP](synergy_of_sight_and_semantics_visual_intention_understanding_with_clip.md)
- [\[ECCV 2024\] Non-parametric Sensor Noise Modeling and Synthesis](non-parametric_sensor_noise_modeling_and_synthesis.md)
- [\[ACL 2025\] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](../../ACL2025/others/battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)
- [\[ACL 2025\] Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale](../../ACL2025/others/verbosity-aware_rationale_reduction_effective_reduction_of_redundant_rationale_v.md)
- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)

</div>

<!-- RELATED:END -->

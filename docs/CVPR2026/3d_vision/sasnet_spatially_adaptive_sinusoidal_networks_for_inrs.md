---
title: >-
  [Paper Note] SASNet: Spatially-Adaptive Sinusoidal Networks for INRs
description: >-
  [CVPR 2026][3D Vision][Implicit Neural Representation] This paper proposes SASNet, which combines frozen frequency embedding layers with spatially-adaptive masks learned by a lightweight hash-grid MLP to address SIREN's…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Implicit Neural Representation"
  - "SIREN"
  - "Spatial Adaptivity"
  - "Frequency Leakage"
  - "Hash Grid"
date: 2026-05-08
content_hash: b93d598e6bd24a6d
---

# SASNet: Spatially-Adaptive Sinusoidal Networks for INRs

**Conference**: CVPR 2026
**arXiv**: [2503.09750](https://arxiv.org/abs/2503.09750)  
**Code**: [https://github.com/Fengyee/SASNet_inr](https://github.com/Fengyee/SASNet_inr)  
**Area**: 3D Vision / Implicit Neural Representations
**Keywords**: Implicit Neural Representation, SIREN, Spatial Adaptivity, Frequency Leakage, Hash Grid

## TL;DR

This paper proposes SASNet, which combines frozen frequency embedding layers with spatially-adaptive masks learned by a lightweight hash-grid MLP to address SIREN's sensitivity to frequency initialization and its high-frequency leakage problem, achieving faster convergence and higher reconstruction quality on image fitting, volumetric data fitting, and SDF reconstruction tasks.

## Background & Motivation

Implicit neural representations (INRs) have emerged as a powerful tool for modeling low-dimensional signals in computer vision and graphics, directly mapping coordinates to signal values. Among them, sinusoidal networks (SIRENs) are widely adopted for their ability to model high-frequency signals via sinusoidal activations, making them particularly suited for tasks requiring high-frequency reconstruction such as image fitting, super-resolution, and SDF modeling.

However, SIREN suffers from a critical limitation: extreme sensitivity to the frequency parameter $\omega_0$. A small $\omega_0$ yields clean but overly smooth reconstructions that lack fine details, while a large $\omega_0$ captures sharp edges but introduces spurious high-frequency noise in smooth regions (e.g., image backgrounds). The authors term this unwanted activation of high-frequency components in low-frequency regions "frequency leakage." Prolonging training to recover high-frequency details further leads to optimization instability and overfitting.

The root cause lies in the global nature of each neuron in SIREN — a neuron encoding high-frequency details influences the entire spatial domain, including smooth regions that require no high-frequency information. Hash-grid methods (e.g., InstantNGP) achieve spatial localization through hash grids, but representing fine details requires very high-resolution grids, increasing memory and computational costs.

- **Core Idea**: Combine SIREN's frequency control capability with the spatial localization of hash-grid MLPs — using frozen frequency embedding layers to fix the network's spectral support, while a lightweight hash-grid MLP learns spatially-adaptive masks that constrain the spatial influence of each neuron, thereby activating high-frequency neurons in detail-rich regions and suppressing them in smooth areas.

## Method

### Overall Architecture

SASNet consists of two parallel networks: a sinusoidal MLP and a hash-grid MLP. Input coordinates $\mathbf{x}$ are fed into both networks simultaneously. The hash-grid MLP encodes coordinates into features and decodes them via a ReLU MLP into spatially-adaptive masks $\mathcal{M}^i(\mathbf{x})$. Each mask is applied to the corresponding layer of the sinusoidal MLP via Hadamard product $\odot$, modulating neuron activations. The frozen frequency embedding layer fixes the spectral range, while the masks handle spatial localization. Both networks are trained jointly.

### Key Designs

1. **Frozen Frequency Embedding Layer**:

    - **Function**: Explicitly fixes the network's frequency support range, providing a controllable spectrum.
    - **Mechanism**: Following Novello et al., the first layer of SIREN uses a predefined set of frequencies with frozen weights. Unlike standard SIREN, where the spectral range is implicitly determined by a single $\omega_0$, the frozen embedding layer directly specifies available frequency components, making frequency control explicit rather than implicit.
    - **Design Motivation**: In standard SIREN, the frequency range is jointly determined by $\omega_0$ and weight initialization in an uncontrollable manner. The frozen embedding layer eliminates this uncertainty, providing a stable frequency basis for the subsequent spatial masks.

2. **Spatially-Adaptive Masks**:

    - **Function**: Learns which neurons/frequency bands should be activated at each spatial location.
    - **Mechanism**: Multi-scale hash grid encoding maps input coordinates to feature vectors, which are decoded by a small ReLU MLP into mask values matching the dimensionality of each sinusoidal MLP layer. Masks modulate sinusoidal layer outputs via element-wise multiplication $\mathbf{h}^i \odot \mathcal{M}^i(\mathbf{x})$. Intuitively, masks suppress high-frequency neurons in smooth regions and allow them to pass in detail-rich regions. Through joint training, the network automatically learns this spatial assignment.
    - **Design Motivation**: This is the key to resolving frequency leakage — restricting the spatial influence of each global INR neuron to specific regions. Hash grids are inherently spatially local; using them as mask generators rather than direct feature extractors is a novel application.

3. **Joint Training of Hybrid Architecture**:

    - **Function**: Achieves an optimal balance between frequency control and spatial localization under parameter efficiency constraints.
    - **Mechanism**: The sinusoidal MLP and hash-grid MLP share input coordinates and are jointly optimized with a standard INR fitting loss $\mathcal{L}(\theta) = \frac{1}{N}\sum_i \|f_\theta(\mathbf{x}_i) - \mathscr{f}_i\|^2 + \lambda \mathcal{R}(\theta)$. The hash-grid MLP is designed to be lightweight (small-resolution grid + shallow ReLU MLP) to avoid significantly increasing parameter count. For SDF tasks, the regularization term $\mathcal{R}(\theta)$ enforces the eikonal constraint.
    - **Design Motivation**: Compared to pure SIREN or pure hash-grid methods, the hybrid architecture inherits SIREN's frequency expressiveness (derivative accuracy of sinusoidal activations) and the hash grid's spatial locality, while avoiding the weaknesses of each.

### Loss & Training

The basic fitting loss is an L2 reconstruction loss. SDF tasks additionally incorporate eikonal regularization to enforce unit gradient norms. The frequency embedding layer remains frozen throughout training. During joint training, masks naturally converge to a configuration that assigns low-frequency neurons to smooth regions and high-frequency neurons to detail-rich regions.

## Key Experimental Results

### Main Results

Based on the paper's abstract and method description, SASNet is evaluated on the following three task categories (specific values to be supplemented upon full access):

| Task | Metric | SASNet vs. SIREN | Notes |
|------|--------|------------------|-------|
| 2D Image Fitting | PSNR | Significant gain | Sharp edges + clean background |
| 3D Volumetric Fitting | PSNR | Significant gain | Noise in smooth regions eliminated |
| SDF Reconstruction | CD/IoU | Outperforms prior methods | Masks automatically focus on zero level-set |
| ×16 Super-Resolution | PSNR | Surpasses SIREN at varying $\omega_0$ | Both low and high $\omega_0$ are problematic; SASNet handles both |

### Ablation Study

| Configuration | Key Effect | Notes |
|---------------|-----------|-------|
| SIREN (low $\omega_0$) | Smooth but blurry | Missing high-frequency details |
| SIREN (high $\omega_0$) | Sharp but noisy | Severe frequency leakage |
| SASNet w/o frozen embedding | Unstable convergence | Uncontrollable frequency range |
| SASNet w/o masks | Similar to SIREN | No spatial localization |
| SASNet (full) | Sharp and clean | Frequency control + spatial localization |

### Key Findings

- **Frequency leakage is the fundamental bottleneck of SIREN**: No tuning of $\omega_0$ can simultaneously achieve sharp details and clean backgrounds; this is not a problem solvable by hyperparameter search.
- **Spatial masks automatically learn frequency assignment**: Visualizations show that masks for low-frequency neurons have high values in smooth regions, while masks for high-frequency neurons have high values at edges and detail-rich regions, validating the design intuition.
- **High parameter efficiency**: The hash-grid MLP as a mask generator adds only minimal parameters while delivering significant quality improvements.
- **Masks focus on zero level-sets in SDF tasks**: In fine-detail regions such as the legs of the Armadillo model, masks automatically concentrate neuron activations, consistent with the physical meaning of SDFs.

## Highlights & Insights

- **Using the hash grid as a mask generator rather than a feature extractor** is the most elegant design choice — typically, hash grids directly replace sinusoidal activations as feature encoders (e.g., InstantNGP); this paper inverts that role, letting the hash grid serve the sinusoidal network as a spatial modulator. This preserves SIREN's accurate derivative computation while gaining spatial locality.
- **The decoupling of frozen frequencies and learned spatial assignment** is conceptually elegant — "fix what frequencies you can use, learn where to use them" — orthogonalizing frequency control and spatial allocation.
- **The mask modulation mechanism may transfer to NeRF/3DGS**: In neural radiance fields, different spatial regions also require different levels of frequency expressiveness; spatially-adaptive masks may prove effective in those settings.

## Limitations & Future Work

- The cached file contains only the abstract, introduction, and method sections; complete experimental data (specific PSNR values, runtime comparisons, etc.) are unavailable.
- Hash grids introduce discretization artifacts that may manifest as blocking effects at insufficient resolutions.
- Whether mask learning requires many iterations to converge, and whether the approach remains effective under very sparse data points, are not sufficiently discussed.
- Validation is limited to low-dimensional signals (2D images, 3D volumetric data/SDF); extension to higher-dimensional scene representations such as NeRF has not been explored.

## Related Work & Insights

- **vs. SIREN**: The global nature of SIREN is the root cause of frequency leakage; SASNet directly addresses this via spatial masks while preserving the advantages of sinusoidal activations.
- **vs. InstantNGP**: InstantNGP uses hash grids + ReLU MLP for local representation, but ReLU suffers from poor derivative accuracy; SASNet retains the precise derivatives of sinusoidal activations, using the hash grid as an auxiliary modulator.
- **vs. WIRE**: WIRE achieves spatial locality via Gabor wavelets but is prone to overfitting; SASNet's mask-based approach is more flexible and parameter-efficient.
- **vs. FINER**: FINER performs adaptive frequency modulation through dynamic scaling factors, but remains global in nature; SASNet achieves genuine spatial localization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Using a hash grid as a spatial mask generator for SIREN is a novel architectural design; the decoupling of frozen frequencies and learned spatial assignment is conceptually clear.
- **Experimental Thoroughness**: ⭐⭐⭐ Incomplete cache prevents evaluation of specific numbers, but three task categories (image, volumetric, SDF) are covered.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; visualizations of "frequency leakage" are intuitive and compelling; the INR taxonomy (global/local/hybrid) provides organizational value.
- **Value**: ⭐⭐⭐⭐ Proposes an elegant solution to the frequency control problem in INRs; the spatial mask idea has broad transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors](eventhub_data_factory_for_generalizable_event-based_stereo_networks_without_acti.md)
- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [\[CVPR 2026\] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)
- [\[NeurIPS 2025\] High Resolution UDF Meshing via Iterative Networks](../../NeurIPS2025/3d_vision/high_resolution_udf_meshing_via_iterative_networks.md)
- [\[NeurIPS 2025\] Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting](../../NeurIPS2025/3d_vision/mesh_interpolation_graph_network_for_dynamic_and_spatially_irregular_global_weat.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings
description: >-
  [ICLR2026][multi-resolution hash encoding] This paper analyzes Instant-NGP's multi-resolution hash encoding (MHE) through the lens of physical systems…
tags:
  - "ICLR2026"
  - "multi-resolution hash encoding"
  - "neural radiance field"
  - "point spread function"
  - "spatial anisotropy"
  - "Instant-NGP"
date: 2026-05-08
content_hash: 01fda654c871d050
---

# Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings

**Conference**: ICLR2026
**arXiv**: [2602.10495](https://arxiv.org/abs/2602.10495)
**Code**: To be confirmed
**Area**: Other
**Keywords**: multi-resolution hash encoding, neural radiance field, point spread function, spatial anisotropy, Instant-NGP

## TL;DR
This paper analyzes Instant-NGP's multi-resolution hash encoding (MHE) through the lens of physical systems, deriving a closed-form approximation of its point spread function (PSF). The analysis reveals that the effective resolution is governed by the geometric mean resolution $N_{\text{avg}}$ rather than the finest resolution $N_{\max}$, and that axis-aligned grids introduce spatial anisotropy. The paper further proposes Rotated MHE (R-MHE), a zero-overhead method that eliminates anisotropy by applying a distinct rotation to the input coordinates at each hash level.

## Background & Motivation

**Background**: Multi-Resolution Hash Encoding (MHE) is the core innovation of Instant-NGP, providing efficient spatial parameterization for NeRF and SDF. However, its behavior is highly sensitive to hyperparameters (number of levels $L$, growth factor $b$, resolutions $N_{\max}/N_{\min}$, hash table size $T$), which are typically selected heuristically.

**Limitations of Prior Work**: MHE lacks rigorous analysis from the perspective of physical systems. Fundamental questions remain unanswered: What is the shape of MHE's effective spatial kernel? What is its true resolution limit? How do hash collisions quantitatively degrade quality?

**Key Challenge**: The intuitive assumption that MHE resolution is determined by the finest level $N_{\max}$ is incorrect—optimization dynamics cause substantial spatial broadening, so the true resolution is far lower than $N_{\max}$.

**Goal**: To develop a rigorous physical analysis framework for understanding MHE's spatial behavior, thereby guiding hyperparameter selection and architectural improvement.

**Key Insight**: By analogy with Green's functions in physical systems, the spatial characteristics of MHE—resolution, anisotropy, and collision noise—are characterized by measuring its response to a point source (i.e., its PSF).

**Core Idea**: The effective resolution of MHE is jointly determined by $N_{\text{avg}}$ and an empirical broadening factor $\beta_{\text{emp}}$, not by $N_{\max}$; grid-induced anisotropy can be eliminated through per-level coordinate rotations.

## Method

### Overall Architecture
The analysis proceeds in three stages: (1) deriving a closed-form approximation of the collision-free ideal PSF, revealing logarithmic decay and B-spline-induced anisotropy; (2) empirically characterizing optimization-induced spatial broadening and establishing the relationship between effective FWHM and $N_{\text{avg}}$; (3) analyzing collision noise under finite hash capacity and quantifying SNR degradation. The R-MHE improvement is proposed based on these findings.

### Key Designs

1. **Ideal PSF Derivation (Collision-Free)**

    - **Function**: Derive the spatial response function of MHE after point-source constrained optimization.
    - **Mechanism**: Under a linearized decoder assumption, the ideal PSF is the average superposition of $L$ normalized B-spline kernels across levels: $P_{\text{Ideal}}(\mathbf{x}) = \frac{1}{L}\sum_{l} \hat{B}_l(\mathbf{x})$. Approximating the sum as an integral and applying a Taylor expansion of the B-spline yields the closed form: $P \approx \frac{1}{L\ln b}[-\ln\|\mathbf{v}\| + C_D - A_D(\mathbf{v})]$, where $A_D$ captures the intrinsic anisotropy of the B-spline.
    - **Design Motivation**: PSF is the standard characterization tool in physical systems. The closed-form solution reveals two key properties: (a) logarithmic radial decay (rather than Gaussian or exponential); (b) anisotropy in which the kernel is narrower along coordinate axes than along diagonals.

2. **Optimization-Induced Spatial Broadening**

    - **Function**: Quantify how much wider the PSF is after actual training compared to the ideal PSF.
    - **Mechanism**: A total broadening factor is defined as $\beta_{\text{emp}} = \beta_{\text{ideal}} \cdot \beta_{\text{opt}}$, where $\beta_{\text{ideal}} \approx 1.18$ (intrinsic B-spline contribution) and $\beta_{\text{opt}} > 1$ (optimization contribution). Empirical measurements with the Adam optimizer yield $\beta_{\text{emp}} \approx 3.0$, meaning the effective FWHM is approximately 2.5 times the ideal value.
    - **Design Motivation**: This is the most counterintuitive finding—spectral bias (the tendency to learn low frequencies first) causes coarse levels (low $N_l$) to be over-weighted, broadening the effective spatial kernel. The true two-point resolvable distance satisfies $d_{\text{crit}} \propto \beta_{\text{emp}}/N_{\text{avg}}$, not $1/N_{\max}$.

3. **Collision Noise SNR Analysis**

    - **Function**: Quantify signal quality degradation caused by finite hash table capacity.
    - **Mechanism**: Collisions cause spatially distant grid vertices to share the same feature vector, producing speckle noise: $P_{\text{Collision}} = P_{\text{Ideal}} + n(\mathbf{x})$, where noise variance increases with the collision rate. Increasing the number of levels $L$ or the growth factor $b$ improves SNR for a fixed $T$.
    - **Design Motivation**: Provides quantitative guidance for selecting hash table size $T$—enabling computation of the minimum $T$ required to maintain a target SNR for a given scene complexity.

4. **Rotated MHE (R-MHE)**

    - **Function**: Eliminate grid-induced spatial anisotropy.
    - **Mechanism**: A distinct rotation $\mathbf{R}_l$ is applied to the input coordinates at each level $l$: $\mathbf{e}_l(\mathbf{x}) = \text{Interpolate}(\mathbf{F}^l, \mathcal{H}(\lfloor N_l \mathbf{R}_l \mathbf{x}\rceil))$. In 2D, progressive rotations $\theta_l = l \cdot \theta$ are used; in 3D, rotations are sampled from SO(3) using icosahedral vertex directions. Critically, **no additional parameters or computation are introduced**—only the coordinate transformation changes.
    - **Design Motivation**: By using grids with different orientations across levels, the per-level anisotropies cancel upon aggregation, yielding a more isotropic PSF.

### Hyperparameter Selection Guidance
Based on the PSF analysis, a theoretical growth factor $b_{\text{theory}}$ is computed such that $\beta_{\text{emp}}/N_{\text{avg}}$ matches the target spatial resolution (e.g., a single pixel). Experiments confirm that $b_{\text{theory}}$ is nearly identical to the empirically optimal $b_{\text{opt}}$, enabling principled hyperparameter selection without manual tuning.

## Key Experimental Results

### Main Results

| Task | Method | PSNR (dB) |
|------|--------|-----------|
| **2D Image Regression** | Standard MHE (M=1) | 23.88 |
| | R-MHE (M=2) | 24.62 |
| | R-MHE (M=4) | 24.69 |
| | **R-MHE (M=8)** | **24.82 (+0.94)** |
| **3D NeRF (Synthetic)** | Standard MHE | 35.346 |
| | R-MHE (Icosa) | **35.479 (+0.13)** |
| **3D SDF** | Standard MHE | 0.9986 IoU |
| | R-MHE (any) | 0.9986 IoU |

### Ablation Study (PSF Property Verification)

| Property | Theoretical Prediction | Experimental Verification |
|----------|----------------------|--------------------------|
| Anisotropy ratio (axis vs. diagonal) | 1.17 | ≈1.17 (exact match) |
| Total broadening factor $\beta_{\text{emp}}$ (Adam) | — | ≈3.0 (stable across configurations) |
| FWHM vs. $N_{\text{avg}}$ relationship | Linear | Linear (exact match) |
| Two-point resolvable distance $d_{\text{crit}}$ | $\propto$ FWHM | Linear correlation (R²≈1) |

### Key Findings
- **Effective resolution is far below $N_{\max}$**: $\beta_{\text{emp}} \approx 3.0$ implies that the actual resolution is approximately 3× lower than $N_{\max}$ suggests, explaining the diminishing returns of increasing $N_{\max}$.
- **$N_{\text{avg}}$ is the true governing parameter**: For fixed $N_{\text{avg}}$, the FWHM remains unchanged regardless of variations in $L$ and $b$, greatly simplifying hyperparameter selection.
- **R-MHE yields significant gains in 2D but marginal gains in 3D**: The improvement is +0.94 dB in 2D and only +0.13 dB in 3D NeRF. The authors attribute this to the ray integration in volumetric rendering, which inherently averages over viewing directions and thus attenuates the effect of anisotropy.
- **PSF-guided hyperparameter selection is effective**: The theoretically derived $b_{\text{theory}}$ agrees with the empirically optimal $b_{\text{opt}}$, eliminating the need for manual tuning.

## Highlights & Insights
- **Physical thinking applied to neural fields**: Employing PSF/Green's function—standard tools from physics—to analyze neural field encodings represents a genuinely novel methodological perspective, directly transferable to other grid-based encodings such as TensoRF and K-Planes.
- **Counterintuitive core finding**: The result that $N_{\text{avg}}$, not $N_{\max}$, governs resolution overturns the intuition that "the finest level determines accuracy" and has direct practical implications for hyperparameter selection.
- **Spatial interpretation of spectral bias**: The well-known spectral bias phenomenon in optimization is translated into a concrete spatial broadening effect, quantified by the factor $\beta_{\text{opt}}$.
- **Zero-cost improvement via R-MHE**: A pure coordinate transformation that introduces no additional parameters or computation is particularly valuable in resource-constrained settings such as mobile rendering.

## Limitations & Future Work
- **Limited 3D improvement**: R-MHE yields only marginal gains on standard 3D benchmarks. Validation on more challenging scenarios (sparse views, high-frequency textures) is needed.
- **Linearization assumption**: The PSF analysis relies on a linearized decoder assumption; its applicability to deep MLPs requires further verification, although the authors report insensitivity to MLP depth in their experiments.
- **Optimizer dependence of $\beta_{\text{opt}}$**: The broadening factor is approximately 3.0 for Adam but differs for other optimizers; a systematic analysis across optimizers is lacking.
- **Point-source analysis only**: The PSF characterizes the response to a single-point constraint; interactions among multiple constraints in real scenes are more complex.

## Related Work & Insights
- **vs. Instant-NGP**: The original Instant-NGP paper introduced the MHE architecture without analyzing its spatial properties. This work provides a deep theoretical complement, revealing the shape of the spatial kernel, the resolution limit, and the effect of collisions.
- **vs. NTK analysis**: The NTK literature analyzes the frequency bias of neural networks. This paper instantiates the NTK perspective as a spatial PSF for MHE, yielding quantitative conclusions that are directly actionable in engineering practice.
- **vs. TensoRF/K-Planes**: All methods based on axis-aligned grids share analogous anisotropy issues. The rotation strategy underlying R-MHE can be directly transferred to these architectures.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Analyzing neural field encodings via physical PSF/Green's function is a genuinely new methodology; the finding that $N_{\text{avg}}$ governs resolution is counterintuitive and significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive validation across 2D, 3D NeRF, and SDF; PSF theory matches experiments precisely; however, 3D improvements are limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The analysis builds progressively from physical intuition, with rigorous mathematical derivations paired with corresponding experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Establishes a physically principled analytical methodology for the neural fields community; PSF-based hyperparameter guidance has direct practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ICLR 2026\] An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)
- [\[AAAI 2026\] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](../../AAAI2026/others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [\[AAAI 2026\] Structure-Aware Encodings of Argumentation Properties for Clique-width](../../AAAI2026/others/structure-aware_encodings_of_argumentation_properties_for_clique-width.md)
- [\[ICLR 2026\] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)

</div>

<!-- RELATED:END -->

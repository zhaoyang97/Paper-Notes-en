---
title: >-
  [Paper Note] Einstein Fields: A Neural Perspective To Computational General Relativity
description: >-
  [ICLR 2026][3D Vision][Neural Fields] This paper proposes EinFields, the first framework to apply neural implicit representations to the compression of four-dimensional general relativity simulations. By encoding the met…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Neural Fields"
  - "General Relativity"
  - "Tensor Field Compression"
  - "Numerical Relativity"
  - "Automatic Differentiation"
date: 2026-05-08
content_hash: 52c52ab38aabcbe9
---

# Einstein Fields: A Neural Perspective To Computational General Relativity

**Conference**: ICLR 2026
**arXiv**: [2507.11589](https://arxiv.org/abs/2507.11589)  
**Code**: [github.com/AndreiB137/EinFields](https://github.com/AndreiB137/EinFields)  
**Area**: 3D Vision
**Keywords**: Neural Fields, General Relativity, Tensor Field Compression, Numerical Relativity, Automatic Differentiation

## TL;DR
This paper proposes EinFields, the first framework to apply neural implicit representations to the compression of four-dimensional general relativity simulations. By encoding the metric tensor field as compact neural network weights, it achieves 4000× storage compression and 5–7 digits of numerical precision, while tensor derivatives obtained via automatic differentiation are 5 orders of magnitude more accurate than those from finite differences.

## Background & Motivation

**Background**: General relativity (GR) describes gravity as the curvature of four-dimensional spacetime, governed by the Einstein field equations (EFEs). Exact solutions exist only for idealized scenarios, making numerical relativity (NR) indispensable for simulating astrophysical events such as black hole mergers and gravitational waves. NR is among the most computationally intensive domains in scientific computing, requiring petabyte-scale storage and supercomputer-level parallelism.

**Limitations of Prior Work**:
- NR simulations produce **petabyte-scale data**, posing severe storage and distribution challenges.
- Finite difference (FD) methods on adaptive meshes are prone to numerical errors in sensitive regions.
- Higher-order FD stencils improve accuracy but increase communication overhead.
- Discrete representations cannot be queried at arbitrary resolution, and derivative computation is limited by truncation error.

**Key Challenge**: The physics of GR is fully encoded in the metric tensor and its first two derivatives, yet conventional methods discretize continuous tensor fields for storage, incurring enormous storage overhead and loss of derivative accuracy.

**Goal**:
- Compress NR simulation data to manageable storage sizes.
- Provide a continuous, mesh-free representation with unlimited resolution.
- Obtain high-accuracy tensor derivatives (Christoffel symbols, Riemann tensor, etc.) via automatic differentiation.
- Support downstream physical tasks (geodesics, curvature diagnostics, gravitational wave extraction).

**Key Insight**: This work generalizes neural fields from computer vision (NeRF/SDF, etc.) to physical tensor fields, introducing the concept of "neural tensor fields"—MLPs that fit the 10 independent components of the metric tensor.

**Core Idea**: Represent the four-dimensional spacetime metric tensor field with a compact neural implicit network (<2M parameters, ~7 MiB), combined with Sobolev training and automatic differentiation, simultaneously achieving 4000× compression and a 10$^5$× improvement in derivative accuracy.

## Method

### Overall Architecture
**Input**: 4D spacetime coordinates $x = (x^0, x^1, x^2, x^3)$ (from exact solutions or NR simulation data)
**Model**: MLP $\hat{g}_\theta: x \in \mathscr{M} \rightarrow g_{\alpha\beta}(x) \in \text{Sym}^2(T^*_x\mathscr{M})$
**Output**: 10 independent metric tensor components → AD differentiation → Christoffel symbols → Riemann tensor → curvature scalars, etc.
**Downstream**: Geodesic tracing, gravitational wave extraction, black hole rendering

### Key Designs

1. **Distortion Decomposition**

    - **Function**: Decomposes the metric tensor into a flat background plus a distortion term $\Delta_{\alpha\beta} = g_{\alpha\beta} - \eta_{\alpha\beta}$.
    - **Mechanism**: The network learns only the non-trivial curvature contributions, removing the dominant numerical contributions of flat spacetime (e.g., $g_{tt} \sim 1/r$, $g_{\theta\theta} \sim r^2$).
    - **Design Motivation**: Concentrating the network's representational capacity on physically meaningful deviations accelerates convergence and improves scaling behavior.

2. **Sobolev Training (Higher-Order Derivative Supervision)**

    - **Function**: Supervises not only the metric tensor itself but also its Jacobian (40 independent components) and Hessian (100 independent components).
    - **Core Formula**: $\mathcal{L}^g_{\text{Sob}}(\theta) = \mathbb{E}_x[\lambda_0\|g - \hat{g}\|^2 + \lambda_1\|\partial g - \partial\hat{g}\|^2 + \lambda_2\|\partial^2 g - \partial^2\hat{g}\|^2]$
    - **Design Motivation**: Physical quantities in GR (Christoffel symbols, Riemann tensor) are defined by the first and second derivatives of the metric; supervising the metric alone does not guarantee derivative accuracy. The Sobolev loss improves Christoffel symbol accuracy by 2 orders of magnitude.

3. **Automatic Differentiation as a Replacement for Finite Differences**

    - **Function**: Computes differential geometric quantities exactly via JAX's forward-mode AD.
    - **Implementation Pipeline**: $g_{\alpha\beta} \xrightarrow{\texttt{jacfwd}} \Gamma^\gamma_{\alpha\beta} \xrightarrow{\nabla} R^\delta_{\alpha\beta\gamma} \xrightarrow{\text{Tr}_g} R_{\alpha\beta} \xrightarrow{\text{Tr}_g} R$
    - **Design Motivation**: FD under FLOAT32 is constrained by truncation error ($O(h^n)$); AD improves accuracy by up to 5 orders of magnitude in single precision.

4. **Network Architecture Selection**

    - MLP with SiLU activations (best performance under derivative supervision).
    - SOAP optimizer (quasi-Newton method, outperforms Adam).
    - GradNorm for multi-task gradient balancing.
    - Parameter scale: 64×3 to 512×8 layers; total parameters <1.9M.

### Loss & Training
- Sobolev loss: weighted sum of metric, Jacobian, and Hessian terms.
- Training time: 100s (without Sobolev) to 2000s (with Hessian), on NVIDIA H200 GPU.
- Learning rate: cosine schedule.

## Key Experimental Results

### Main Results (Schwarzschild Black Hole Solution)

| Representation | Rel. $\ell_2$ | MAE | Storage | Compression |
|---|---|---|---|---|
| Explicit Grid | - | - | 343 MiB | 1× |
| EinFields | 1.08e-6 | 2.11e-6 | **85 KiB** | **4035×** |
| EinFields (+Jac) | 3.37e-7 | 9.49e-7 | 1.1 MiB | 311× |
| EinFields (+Jac+Hess) | **1.88e-7** | **9.07e-7** | 202 KiB | 1698× |

### Derivative Accuracy Comparison (vs. High-Order FD, FLOAT32)

| Geometric Quantity | FD (h=0.01) MAE | EinFields AD MAE | Accuracy Gain |
|---|---|---|---|
| Christoffel Symbols | 5.37e-6 | **9.98e-7** | 5× |
| Riemann Tensor | 1.78e-2 | **1.25e-6** | **14000×** |
| Ricci Tensor | 4.81e-2 | **9.66e-6** | **5000×** |
| Kretschmann Scalar | 1.33e-2 | **1.07e-5** | **1200×** |

### Ablation Study

| Configuration | Rel. $\ell_2$ | Training Time |
|---|---|---|
| Baseline (Distortion+Jac+Hess+SiLU+SOAP+Cosine) | **1.40e-7** | 1400s |
| Full metric → Distortion | 2.13e-6 | 1407s |
| SOAP → Adam | 4.16e-6 | 1150s |
| SiLU → WIRE | 4.12e-6 | 3045s |
| Remove Hessian supervision | 1.51e-7 | 509s |
| Remove all Sobolev terms | 2.37e-7 | 364s |

### Key Findings
- **Neutron Star NR Simulation Validation**: On a realistic BSSN-evolved neutron star oscillation simulation, EinFields achieves 2121× compression (2.9 GiB → 1.4 MiB) with Rel. $\ell_2$ = 3.60e-5.
- **Distortion decomposition is critical**: Removing it degrades accuracy by 15× (1.40e-7 → 2.13e-6).
- **SOAP optimizer outperforms Adam**: 30× accuracy improvement.
- **Comprehensive validation on geodesics, gravitational waves, and black hole rendering**: Geodesic reconstruction in Schwarzschild/Kerr spacetimes agrees closely with analytic solutions.

## Highlights & Insights
- **Pioneering the "Neural Tensor Field" concept**: EinFields generalizes neural fields from computer vision (SDF/NeRF/radiance fields) to physical tensor fields. A key distinction is that **dynamics emerge naturally**—since the metric encodes spacetime geometry, its derivatives directly yield the equations of motion.
- **AD vs. FD paradigm shift**: Under FLOAT32, AD surpasses 6th-order FD by 5 orders of magnitude for the Riemann tensor. This has broad implications for the scientific computing community—any domain requiring high-order derivatives can benefit from this approach.
- **Neural black hole rendering**: As a proof of concept, the metric represented by EinFields is used directly for ray tracing to render black hole images, demonstrating the framework's compatibility with complex downstream tasks.

## Limitations & Future Work
- Compression is lossy: even with FLOAT64 training, Rel. $\ell_2$ < 1e-9 is currently unattainable.
- EinFields does not yet surpass FD at FLOAT64 precision.
- Query latency is non-trivial (milliseconds for 10$^5$ points), which may become a bottleneck in downstream tasks requiring repeated sequential evaluations.
- Long-term geodesic solver rollouts exhibit error accumulation.
- **Directions for improvement**: Extension to binary black hole mergers, binary neutron star mergers, and other large-scale NR simulations; systematic performance comparison with spectral methods.

## Related Work & Insights
- **vs. SIREN (Sitzmann et al., 2020)**: The classical INR uses sinusoidal activations, but EinFields finds that SiLU performs better under derivative supervision (ablation shows WIRE also underperforms SiLU).
- **vs. PINNs (Raissi et al., 2019)**: PINNs solve PDEs from scratch using physics-informed losses; EinFields **compresses existing solutions** rather than solving them—a more practically applicable use case.
- **vs. Traditional NR (Einstein Toolkit)**: EinFields serves as a downstream complement to the conventional NR pipeline, enhancing rather than replacing it—providing compressed storage and high-accuracy continuous querying.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to introduce neural fields into numerical relativity; the "neural tensor field" concept is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three analytic solutions (Schwarzschild/Kerr/gravitational waves) plus a neutron star NR simulation with good breadth, but lacks the most challenging scenarios such as binary mergers.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Physical background is presented clearly; the method and results are coherently connected; the appendix is highly detailed.
- **Value**: ⭐⭐⭐⭐ Significant reference value for both the numerical relativity and scientific machine learning communities, though practical impact depends on whether the framework can be extended to dynamic binary systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)
- [\[ICLR 2026\] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)
- [\[ICLR 2026\] Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting](augmented_radiance_field_a_general_framework_for_enhanced_gaussian_splatting.md)
- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](../../NeurIPS2025/3d_vision/learning_neural_exposure_fields_for_view_synthesis.md)
- [\[CVPR 2026\] Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields](../../CVPR2026/3d_vision/dynamic_black-hole_emission_tomography_with_physics-informed_neural_fields.md)

</div>

<!-- RELATED:END -->

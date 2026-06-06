---
title: >-
  [Paper Note] Geodesic Flow Matching for Denoising High-Dimensional Structured Representations
description: >-
  [ICML 2026][3D Vision][Geodesic Flow Matching] Addressing Spatial Semantic Pointers (SSPs) in Vector Symbolic Architectures—high-dimensional structured representations embedded in a "Clifford torus within a unit hypersph…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Geodesic Flow Matching"
  - "Spatial Semantic Pointer (SSP)"
  - "Clifford Torus"
  - "Neuro-symbolic Cleanup"
  - "Spiking Neural SLAM"
date: 2026-05-08
content_hash: 90e6da0a3bed82b8
---

# Geodesic Flow Matching for Denoising High-Dimensional Structured Representations

**Conference**: ICML 2026  
**arXiv**: [2606.00248](https://arxiv.org/abs/2606.00248)  
**Code**: https://github.com/kremHabashy/CleanupSSP  
**Area**: Representation Learning / Flow Matching / Neuro-symbolic / Manifold Geometry  
**Keywords**: Geodesic Flow Matching, Spatial Semantic Pointer (SSP), Clifford Torus, Neuro-symbolic Cleanup, Spiking Neural SLAM

## TL;DR
Addressing Spatial Semantic Pointers (SSPs) in Vector Symbolic Architectures—high-dimensional structured representations embedded in a "Clifford torus within a unit hypersphere"—the authors observe that Euclidean straight-line interpolation in standard Flow Matching passes through the sphere's interior, causing amplitude collapse and phase destruction. By constraining the flow to the sphere using Log/Exp mappings, **Geodesic Flow Matching (GFM)** reduces path error by 72% in spiking neural SLAM and enables a 1500-neuron path integrator to match the accuracy of a 2500-neuron baseline.

## Background & Motivation

**Background**: Vector Symbolic Architectures (VSA, focusing here on Plate 95 HRR) encode symbols as high-dimensional vectors, performing compositional reasoning via bundling (addition) and binding (circular convolution). Spatial Semantic Pointers (SSPs) extend this by encoding continuous coordinates $x\in\mathbb{R}^m$ into $d>1000$ dimensional vectors using Fourier phases $\tilde\phi(x)_j=e^{i\langle\theta_j, x\rangle}$, resulting in a "position-to-vector" cognitive map for path integration, SLAM, and hippocampal-entorhinal models. All VSA systems rely on **cleanup**: mapping vectors corrupted by cross-talk, phase drift, or spiking noise back to the valid representation manifold.

**Limitations of Prior Work**: Traditional cleanup follows two paths: (1) Discrete prototypes (Hopfield networks), which do not suit continuous SSPs; (2) Grid lookup + L-BFGS optimization, which either suffers from exponential scale explosion or "snaps to the wrong prototype" under high noise. While diffusion/flow matching can be seen as modern continuous associative memories, directly applying them poses two issues: (a) Diffusion requires many sampling steps, unsuitable for low-latency robotics; (b) Conditional Flow Matching (CFM) compresses this into a deterministic ODE but **assumes Euclidean geometry**.

**Key Challenge**: The valid states of SSPs are not arbitrary points in Euclidean space but are constrained to a Clifford torus $\subset \mathbb{S}^{d-1}$—each Fourier component $e^{i\langle\theta_j,x\rangle}$ must maintain unit magnitude. CFM uses linear interpolation $\phi_t=(1-t)\phi_0+t\phi_1$, which corresponds to a **chord** rather than a geodesic on the sphere. At intermediate times, $\|\phi_t\|<1$, causing amplitude collapse and phase corruption. Experiments show that under high noise, Euclidean CFM produces states that "look like valid SSPs" but have **shifted spatial positions** (Figure 4b) because the phase was compromised.

**Goal**: (i) Provide a cleanup method for high-dimensional structured representations like SSPs; (ii) Use few-step ODE inference instead of diffusion iterations; (iii) Validate the cleanup value in a real-world spiking neural SLAM closed loop.

**Key Insight**: While Chen & Lipman (2024) generalized flow matching to Riemannian Manifolds (RFM), it was only tested on low-dimensional spaces with Gaussian priors. This paper argues that pushing Log/Exp mapping mechanisms to extremely high-dimensional ($d>1000$), non-Gaussian scenarios—specifically grid-based SSP encodings—yields positive gains that stabilize as dimensionality increases.

**Core Idea**: Replace the linear interpolation $\phi_t = (1-t)\phi_0 + t\phi_1$ in CFM with a spherical geodesic $\phi_t = \mathrm{Exp}_{\phi_0}(t\cdot \mathrm{Log}_{\phi_0}(\phi_1))$. The velocity field $v_\theta$ regresses only tangent space vectors, ensuring the sampling trajectory always stays on $\mathbb{S}^{d-1}$, preserving the phase structure.

## Method

### Overall Architecture
GFM treats cleanup as a **generative transport problem from a noise distribution $p_0$ to a legal SSP distribution $p_1$**.

- **Input**: Corrupted vector $\tilde\phi \in \mathbb{R}^d$ (approximated as $p_0$: isotropic Gaussian on the hypersphere, $\phi_0 = z/\|z\|$, $z\sim\mathcal{N}(0,I_d)$).
- **Training**: At each time step $t\sim\mathcal{U}[0,1]$, generate $\phi_t$ via spherical geodesic interpolation and use a residual MLP $v_\theta(\phi_t, t)$ to regress the tangent space velocity.
- **Inference**: Sample $\phi_0$ from $p_0$, solve the ODE $\phi_{k+1} = \mathrm{Exp}_{\phi_k}(\Delta t\, v_k)$ over $K$ steps with periodic normalization for stability, and output $\phi_K$ as the cleaned SSP.
- **Function**: Outputs a valid point on $\mathbb{S}^{d-1}$ that can be correctly decoded into spatial coordinates via unbinding/decoding.

**Downstream Application**: GFM acts as an **online stabilizer** inserted between a spiking path integrator (PI) and a VSA map. It periodically "pulls" the drifting PI state back to the manifold before landmark binding, thereby maintaining SLAM loop closure.

### Key Designs

1.  **Geometric Gap (Diagnosis)**:
    - **Function**: Theoretically identifies why standard CFM fails in the SSP domain.
    - **Mechanism**: Formalizes three noise sources: cross-talk from bundling $\epsilon\sim\mathcal{N}(0, \tfrac{n-1}{d}I_d)$, cumulative phase drift in recurrence ($\mathrm{WrappedNormal}(0, t\sigma^2)$), and spike-based decoding noise.
    - **Design Motivation**: Demonstrated that CFM's target velocity $u_t = \phi_1 - \phi_0$ is a chord. When $t\in(0,1)$, $\|\phi_t\|<1$, destroying the unit magnitude required for Fourier phases. Visualization (Figure 4b) confirms the causality: amplitude collapse $\rightarrow$ phase shift $\rightarrow$ decoding to incorrect coordinates.

2.  **Geodesic Probability Path (Training)**:
    - **Function**: Uses Log/Exp mappings to define training trajectories and target velocities on the sphere.
    - **Mechanism**: The path $\phi_t = \mathrm{Exp}_{\phi_0}(t \cdot \mathrm{Log}_{\phi_0}(\phi_1))$ follows the great-circle arc. Target velocity $u_t$ is the instantaneous tangent along the arc. The **Loss** uses cosine flow loss $\mathcal{L}_{\cos}=1-\frac{v_\theta^\top \dot\phi_t}{\|v_\theta\|\|\dot\phi_t\|}$ instead of MSE to focus on directional alignment.
    - **Design Motivation**: Adopts the Riemannian Flow Matching framework for the environment manifold $\mathbb{S}^{d-1}$. Cosine loss is chosen because SSP semantics are encoded in angles, whereas L2 loss is contaminated by irrelevant magnitude deviations.

3.  **Geodesic Sampling (Inference)**:
    - **Function**: Preserves training geometry during inference via Exp-map ODE steps.
    - **Mechanism**: Starting from $\phi_0\sim p_0$, each step follows $\phi_{k+1} = \mathrm{Exp}_{\phi_k}(\Delta t\, v_\theta(\phi_k, t_k))$, followed by explicit normalization $\phi_{k+1} \leftarrow \phi_{k+1}/\|\phi_{k+1}\|$.
    - **Design Motivation**: Ensures geometric consistency. Unlike Euclidean steps that push states out of the manifold, Exp-map updates allow GFM to act as a **continuous attractor field** in SLAM loops without disrupting integration dynamics.

### Loss & Training
- **Loss**: Cosine flow loss (Eq. 10).
- **Architecture**: $v_\theta$ is a 3-block residual MLP. Each block consists of Linear + GELU + Dropout(0.1) + LayerNorm with a bottleneck schedule ($2d\to d, 4d\to d, 2d\to d$). Time is injected using 32-dimensional sinusoidal embeddings.
- **Training Strategy**: Clean SSPs are generated via Sobol quasi-random sampling for uniform spatial coverage; noise is generated by projecting $\mathcal{N}(0, I_d)$ onto the sphere.

## Key Experimental Results

### Main Results

Core SLAM results (Table 1, RMSE in meters):

| PI Neurons | Method | RMSE (m) | Description |
| :--- | :--- | :--- | :--- |
| 1000 | Grid | 0.586 ± 0.121 | Insufficient resolution |
| 1000 | Euclidean FM | 0.449 ± 0.068 | Straight-line interpolation |
| 1000 | **Geodesic FM** | **0.162 ± 0.055** | 72% better than Grid, 64% over Euclidean |
| 1500 | Grid | 0.249 ± 0.239 | High variance, unstable |
| 1500 | Euclidean FM | 0.204 ± 0.103 | |
| 1500 | **Geodesic FM** | **0.076 ± 0.026** | **72% improvement**, lower variance |
| 2500 | Grid | 0.083 ± 0.017 | Grid only catches up with high neuron counts |
| 2500 | **Geodesic FM** | **0.078 ± 0.009** | |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Geodesic Flow (Full) | RMSE 0.076m (1500n) | Complete model |
| Euclidean Flow | RMSE 0.204m (1500n) | Path error increases ~2.7× |
| Feedforward Regression | Diffuse distribution | Collapses to target mean (looks like bundling) |
| Grid Lookup | Fails in SLAM loops | Discrete snapping causes discontinuous jumps |
| L-BFGS Optimization | Unstable under noise | Stuck in local optima |

### Key Findings
- **Geodesic vs. Linear Interpolation**: Switching Euclidean FM to GFM reduces RMSE from 0.204m to 0.076m. Visualization (Figure 4) proves that Euclidean flow yields states that look like SSPs but are spatially shifted due to phase destruction.
- **Fragility of Discrete Methods in Loops**: Grid lookup works well in static benchmarks but fails in recursive SLAM loops because "snapping to the nearest prototype" creates discontinuities that disrupt velocity integration. GFM provides a continuous attractor field.
- **Value of Geometric Prior**: GFM with 1500 neurons $\approx$ baseline with 2500 neurons, translating to a 40% saving in neural resources.

## Highlights & Insights
- **Diagnosis-Driven Methodology**: The paper identifies three specific noise sources before formalizing why Euclidean geometry fails, making the adoption of GFM a logical necessity rather than a minor tweak.
- **Paradigm Shift in Cleanup**: Redefining cleanup as generative transport shifts the field from discrete embedding searches (Hopfield/Grid) to continuous flows.
- **Efficiency**: Few-step ODE inference with geometric priors makes "manifold-aware associative memory" viable for low-latency robotics.
- **Transferable Trick**: Transitioning to cosine flow loss from MSE is effective for any task where semantics are encoded in direction rather than magnitude (e.g., hyperspherical embeddings).

## Limitations & Future Work
- **Ours**: Currently uses standard MLPs; full deployment on neuromorphic hardware requires conversion to Spiking Neural Networks (SNNs).
- **Novelty**: The framework is tied to hyperspherical topology; other VSA families (e.g., Boolean hypercubes) require different Log/Exp mappings.
- **Internal View**: The SLAM experiments use synthetic 2D data with fixed landmarks. Future work should involve real-world odometry and vision front-ends, as well as an analysis of the latency/accuracy trade-off for varying $K$ steps.

## Related Work & Insights
- **vs. Conditional Flow Matching (Lipman 2022)**: GFM provides the necessary correction for spherical/toroidal representations where Euclidean assumptions fail.
- **vs. Riemannian Flow Matching (Chen & Lipman 2024)**: First to scale RFM to $d>1000$ and non-Gaussian structured targets in a closed-loop system.
- **vs. Grid/L-BFGS (Dumont 2023)**: GFM is more robust to noise and preserves dynamical system continuity.
- **vs. Modern Hopfield (Ramsauer 2020)**: Aligns with the theory that generative denoising acts as a continuous attractor.

## Rating
- **Novelty**: ⭐⭐⭐⭐ High; successfully adapts Riemannian Flow Matching to neuro-symbolic VSA.
- **Experimental Thoroughness**: ⭐⭐⭐ Robust SLAM results, though synthetic; lacks direct numerical tables for qualitative figures.
- **Writing Quality**: ⭐⭐⭐⭐ Excellent causal reasoning.
- **Value**: ⭐⭐⭐⭐ Practical engineering improvement for neuro-symbolic SLAM and normalized embeddings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](../../CVPR2026/3d_vision/geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](../../CVPR2026/3d_vision/hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[ICML 2026\] SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising](simpc_learning_self-induced_mirror-point_consistency_for_unsupervised_point_clou.md)
- [\[ICLR 2026\] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](../../ICLR2026/3d_vision/hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)
- [\[AAAI 2026\] Class-Partitioned VQ-VAE and Latent Flow Matching for Point Cloud Scene Generation](../../AAAI2026/3d_vision/class-partitioned_vq-vae_and_latent_flow_matching_for_point_cloud_scene_generati.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](../../CVPR2026/3d_vision/geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](../../CVPR2026/3d_vision/hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[ICML 2026\] SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising](simpc_learning_self-induced_mirror-point_consistency_for_unsupervised_point_clou.md)
- [\[ICLR 2026\] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](../../ICLR2026/3d_vision/hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)
- [\[AAAI 2026\] Class-Partitioned VQ-VAE and Latent Flow Matching for Point Cloud Scene Generation](../../AAAI2026/3d_vision/class-partitioned_vq-vae_and_latent_flow_matching_for_point_cloud_scene_generati.md)

</div>

<!-- RELATED:END -->

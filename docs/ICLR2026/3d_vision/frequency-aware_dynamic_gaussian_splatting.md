---
title: >-
  [Paper Note] Frequency-Aware Dynamic Gaussian Splatting
description: >-
  [ICLR 2026][3D Vision][Paper Note] This paper reveals the root cause of motion blur in dynamic 3DGS from a frequency perspective—"high-frequency rendering details" and "high-frequency motion" compete for expressive power on fixed Gaussian kernels. It proposes the Frequency-Differentiated Gaussian Kernel (FDGK) and Fourier Deformation Network (FDN) to de
tags:
  - ICLR 2026
  - 3D Vision
date: 2026-05-08
content_hash: e13c09e18c0fcda9
---
# Frequency-Aware Dynamic Gaussian Splatting

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=UZ00ac4eqA](https://openreview.net/forum?id=UZ00ac4eqA)  
**Code**: To be confirmed  
**Area**: 3D Vision / 4D Dynamic Reconstruction  
**Keywords**: Dynamic Gaussian Splatting, 4D Reconstruction, Motion Blur, Frequency-Aware, Deformation Field, Fourier Embedding  

## TL;DR
This paper reveals the root cause of motion blur in dynamic 3DGS from a frequency perspective—"high-frequency rendering details" and "high-frequency motion" compete for expressive power on fixed Gaussian kernels. It proposes the Frequency-Differentiated Gaussian Kernel (FDGK) and Fourier Deformation Network (FDN) to decouple detail expression from motion modeling, significantly reducing blur and achieving a new SOTA on synthetic and real 4D benchmarks.

## Background & Motivation
- **Background**: Dynamic 3DGS methods based on deformation fields (DeformGS, 4D-GS, SC-GS, Grid4D, etc.) can reconstruct 4D scenes in real-time. The mainstream approach is to maintain a set of canonical Gaussians and use an MLP deformation network to predict residual displacements $(\Delta x, \Delta R, \Delta S)$ over time.
- **Limitations of Prior Work**: Dynamic reconstructions generally suffer from severe motion blur in **novel view synthesis**, especially at object boundaries and areas with fast deformation. Previous works focused on "refining the deformation field," overlooking the spectral characteristics of the motion itself.
- **Key Challenge**: Vanilla 3DGS assigns a fixed opacity distribution to each Gaussian (solid at the center, transparent at the edges). In static scenes, high-frequency details are recovered by stacking Gaussians. In dynamic scenes, however, this forces the deformation network to perform two contradictory tasks: organizing dense Gaussian stacks to recover per-frame high-frequency appearance, and coordinating these Gaussians across frames to represent high-frequency motion without collapsing. Consequently, the network tends to settle for a trade-off—"uniform low-frequency motion"—resulting in motion blur.
- **Goal**: Separate the responsibilities of "high-frequency detail expression" and "high-frequency motion modeling" from the deformation network, assigning them to more expressive Gaussian kernels and a frequency-aware deformation network, respectively.
- **Core Idea**: **[Spectral Decoupling]** Allow Gaussian kernels to differentiate into high-frequency kernels (for sharp boundaries) and low-frequency kernels (for smooth regions), offloading detail expression from the deformation network. **[Fourier Motion]** Use high-frequency Fourier embeddings to represent the motion of each point as a superposition of multi-frequency cycles, combined with a frequency-aware gate to amplify truly dynamic points.

## Method

### Overall Architecture
FAGS introduces two upgrades to the Grid4D deformation backbone: on the representation side, it uses the **Frequency-Differentiated Gaussian Kernel (FDGK)** to modify alpha-blending, allowing Gaussians to adaptively partition based on frequency characteristics. On the motion side, the **Fourier Deformation Network (FDN)** fuses low-frequency spatiotemporal features from hash encoding with high-frequency Fourier embeddings. A multi-head decoder predicts deformations regulated by a frequency-aware gate for each Gaussian. Finally, a **Fourier frequency loss** in the spectral domain drives the mechanism toward high-frequency details.

```mermaid
flowchart LR
    A[Canonical Gaussians<br/>λ, β Learnable] --> B[FDGK<br/>Adaptive alpha modulation ψ(g)]
    C["4D Coordinates (x,y,z,t)<br/>Four sets of hash encodings"] --> D[Low-frequency spatiotemporal features]
    E[High-frequency Fourier Embedding<br/>Multi-frequency sin/cos] --> F[Fused High/Low frequency features]
    D --> F
    F --> G[Multi-head Decoder]
    G --> H[Frequency-aware Gating η<br/>Regulate motion intensity]
    B --> I[Frequency-differentiated Gaussians]
    H --> I
    I --> J[Rendering + Fourier Frequency Loss]
```

### Key Designs

**1. Frequency-Differentiated Gaussian Kernel (FDGK): Letting each Gaussian choose its frequency identity.** Re-examining the alpha calculation $\alpha_i = o_i\exp[-\frac{1}{2}(p-\mu_{2D_i})^T\Sigma_{2D_i}^{-1}(p-\mu_{2D_i})]$, the author denotes the exponential term as $g$ and rewrites the opacity as $\alpha_i = \min(o_i\psi(g), 0.99)$ using a learnable piecewise modulation function. $\psi(g)$ is controlled by two learnable parameters: a slope parameter $\lambda\in[0,1]$ (where $r=0.5+\lambda$, $b=0.25-0.5\lambda$) which determines the steepness of the mapping—degenerating to standard Gaussian when $\lambda=0.5$, yielding a smooth low-frequency kernel when $\lambda<0.5$, and a sharp high-frequency kernel when $\lambda>0.5$. The boundary parameter $\beta$ independently scales the endpoints of the differentiation interval $p_l=0.5-\beta d_{g_0}$ and $p_r=0.5+\beta d_{g_0}$. Unlike DRK which only adjusts $r$ and forces Gaussians with the same slope to share a differentiation interval, FDGK decouples "frequency characteristics" from "differentiation span" via independent $\beta$. $\lambda$ and $\beta$ are jointly optimized through backpropagation using closed-form piecewise gradients. This offloads detail recovery to a few sharp Gaussians, significantly reducing the reliance on dense stacking.

**2. Fourier Deformation Network (FDN): Representing motion as multi-frequency periodic superpositions.** Optimizing trajectories for hundreds of thousands of Gaussians per timestep is infeasible. Instead, high-frequency motion is encoded for each point and fused with low-frequency spatial features. 4D coordinates are decomposed into four sets of 3D hash encodings $(x,y,z)$, $(x,y,t)$, $(y,z,t)$, and $(x,z,t)$, processed via MLPs to obtain spatial features $f_{spa}$ and temporal features $f_{tem}$. A **high-frequency Fourier embedding** $f_{fre}=[w_1\sin(\pi\gamma_1 t), w_1\cos(\pi\gamma_1 t),\dots]$ is designed, where frequencies $\gamma_i=2^{\frac{3i-3}{m-1}}$ use dense multi-scale sampling, and amplitudes $[w_1,\dots,w_m]=\text{MLP}(f_{spa})$ are time-invariant but Gaussian-specific. This effectively decomposes motion into periodic signals, using amplitude distributions to characterize intensity across frequencies.

**3. Frequency-Aware Gating (FG): Only moving points that truly move.** Deformation networks often predict updates for all Gaussians indiscriminately, which can erroneously displace static points. The fused high-frequency Fourier features $f_{fre}$ and low-frequency temporal embeddings $f_{tem}$ are fed into a decoder $D_\theta$. In addition to rotation $R_x$, translation $T_x$, and scale/rotation residuals $\Delta r, \Delta s$, a gating score $\eta$ is output to modulate deformation intensity: $\mu'=\eta R_x\mu+\eta T_x$, $S'=S+\eta\Delta s$, and $R'=R+\eta\Delta r$. Gaussians with high-frequency motion receive large $\eta$ values for rapid attribute changes, while low-frequency Gaussians near static areas receive small $\eta$ values for suppression. This adaptive control is smoother and more effective than hard-clamping static boundaries.

**4. Fourier Frequency Loss: Explicitly driving high frequencies in the spectral domain.** To activate the potential of the proposed components, a frequency-domain objective is introduced. FFT is applied to the rendered image $I'$ and target image $I$ to obtain magnitude spectra, defining $L_{fre}=\|I'_{amp}-I_{amp}\|_1$ (comparing only magnitudes as phase encodes structure and geometries are already similar). The total loss $L=\sigma_c L_{L1}+(1-\sigma_c)L_{MISS}+\sigma_r L_r+\sigma_{fre}L_{fre}$ adds this term to the standard Grid4D reconstruction loss, emphasizing hard-to-optimize high-frequency regions.

## Key Experimental Results

### Main Results
Average performance across 7 scenes on the D-NeRF synthetic dataset (compared with Grid4D backbone):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 4D-GS | 36.30 | 0.986 | 0.019 |
| SC-GS | 41.59 | 0.994 | 0.015 |
| Grid4D | 41.99 | 0.993 | 0.008 |
| Grid4D+DRK | 39.43 | 0.990 | 0.015 |
| **Ours (FAGS)** | **42.76** | **0.995** | **0.007** |

Average performance on real-world datasets:

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------|--------|--------|---------|
| Neu3D | Grid4D | 31.63 | 0.937 | 0.149 |
| Neu3D | **Ours** | **32.18** | **0.946** | **0.146** |
| HyperNeRF (Interp.) | Grid4D | 28.59 | 0.844 | 0.199 |
| HyperNeRF (Interp.) | **Ours** | **29.02** | **0.850** | **0.195** |
| HyperNeRF (Rig) | Grid4D | 25.24 | 0.685 | 0.319 |
| HyperNeRF (Rig) | **Ours** | **25.63** | **0.719** | **0.269** |

### Ablation Study
D-NeRF averages, removing components sequentially:

| Configuration | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Full | **42.76** | **0.995** | **0.007** |
| w/o FDGK | 42.11 | 0.993 | 0.009 |
| w/o HFE | 42.38 | 0.994 | 0.009 |
| w/o FG | 42.70 | 0.994 | 0.008 |
| w/o $L_{fre}$ | 42.50 | 0.994 | 0.008 |
| w/o (FG + $L_{fre}$) | 42.43 | 0.994 | 0.008 |
| w/o FDGK.λ | 42.30 | 0.994 | 0.008 |
| w/o FDGK.β | 42.26 | 0.994 | 0.008 |

### Key Findings
- **FDGK contributes most**: Removing it drops PSNR by 0.65 and doubles LPIPS to 0.009, verifying that "Gaussian frequency differentiation offloading detail expression" is the core source of gain. Both $\lambda$ and $\beta$ are necessary, as removing either results in a performance drop.
- **Spontaneous Gaussian Differentiation**: Initially set to $\lambda=\beta=0.5$ (equivalent to standard Gaussians), the population splits into low-frequency and high-frequency clusters (roughly 3:2 ratio) after approximately 5,000 steps, proving that frequency specialization emerges naturally through optimization.
- **Significant gains in real scenes**: On the HyperNeRF Rig subset, SSIM increased from 0.685 to 0.719, and LPIPS decreased from 0.319 to 0.269, qualitatively fixing artifacts such as disappearing sharp edges present in Grid4D.

## Highlights & Insights
- **Insightful Problem Diagnosis**: Attributes motion blur in dynamic 3DGS to "spectral competition between high-frequency detail and high-frequency motion on fixed Gaussian kernels." This is more profound than simply blaming deformation field precision and offers a clear "responsibility decoupling" solution.
- **Learnable Frequency Identity**: Uses decoupled parameters $\lambda$ (frequency) and $\beta$ (span) to let each Gaussian adaptively select its frequency characteristics. This is a lightweight modification to alpha-blending that is easy to integrate into existing 3DGS pipelines.
- **Spectral View of Motion**: Fourier embeddings represent point-wise motion as multi-frequency amplitude superpositions, avoiding the computational bottleneck of per-timestep trajectory optimization. The gating $\eta$ provides an elegant way to suppress static points.

## Limitations & Future Work
- **Strong Backbone Dependency**: The method relies on Grid4D as the deformation backbone and adopts its hyperparameters; generalizability across other deformation frameworks requires further validation.
- **Small Absolute Gains**: Most scenes in D-NeRF are near saturation (PSNR 40+), with an average PSNR gain of ~0.77. The contribution is more evident in qualitative sharp boundaries and novel-view anti-blurring than in numerical margins.
- **Fixed Frequency Hyperparameters**: Parameters like $\sigma_{fre}=0.3$ and Fourier frequency sampling $\gamma_i$ are manually set. Their adaptivity to scenes with different motion scales and whether the 3:2 splitting ratio varies with dynamic range remains to be explored.

## Related Work & Insights
- **Dynamic NeRF**: DyNeRF, Nerfies, and D-NeRF extend NeRF to temporal sequences via deformation fields. This work inherits the "canonical + deformation" paradigm but applies it to Gaussian representations.
- **Dynamic Gaussian Splatting**: Categorized into iterative approaches (e.g., D-3DGS) and deformation-based approaches (e.g., 4D-GS, SC-GS, Grid4D). FAGS belongs to the latter and is the first to introduce spectral characteristics into the Gaussian representation.
- **Gaussian Kernel Enhancement**: DRK also adjusts slope $r$ for better expression, but differentiation boundaries are implicitly tied to $r$. The independent $\beta$ in FAGS is a key improvement for future work on learnable Gaussian shapes/frequencies.
- **Inspiration**: Using "frequency/spectrum" as a unified language to decouple representation and motion may be transferable to other tasks facing "detail vs. dynamics" dilemmas, such as video generation and neural rendering.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Diagnosing motion blur from a spectral competition perspective and solving it via learnable frequency-differentiated kernels is novel and self-consistent; however, it builds on existing components like Grid4D and DRK.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three mainstream benchmarks (D-NeRF, Neu3D, HyperNeRF) with complete ablation studies and $\lambda$ distribution analysis. The only downside is the limited absolute gain and lack of cross-backbone validation.
- **Writing Quality**: ⭐⭐⭐⭐ Motivations (dilemma analysis in Fig. 1) are clear and intuitive. Formulas and diagrams are well-coordinated and the logic is easy to follow.
- **Value**: ⭐⭐⭐⭐ Provides a plug-and-play route for dynamic 3DGS with a focus on novel-view anti-blurring, offering significant reference value for high-fidelity 4D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting](moe-gs_mixture_of_experts_for_dynamic_gaussian_splatting.md)
- [\[CVPR 2026\] TagSplat: Topology-Aware Gaussian Splatting for Dynamic Mesh Modeling and Tracking](../../CVPR2026/3d_vision/tagsplat_topology-aware_gaussian_splatting_for_dynamic_mesh_modeling_and_trackin.md)
- [\[ICLR 2026\] Fracture-GS: Dynamic Fracture Simulation with Physics-Integrated Gaussian Splatting](fracture-gs_dynamic_fracture_simulation_with_physics-integrated_gaussian_splatti.md)
- [\[ICLR 2026\] Gradient-Direction-Aware Density Control for 3D Gaussian Splatting](gradient-direction-aware_density_control_for_3d_gaussian_splatting.md)
- [\[ICLR 2026\] Uncertainty-Aware 3D Reconstruction for Dynamic Underwater Scenes](uncertainty-aware_3d_reconstruction_for_dynamic_underwater_scenes.md)

</div>

<!-- RELATED:END -->

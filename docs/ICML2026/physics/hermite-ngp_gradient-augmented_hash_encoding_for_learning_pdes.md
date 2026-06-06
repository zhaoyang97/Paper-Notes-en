---
title: >-
  [Paper Note] Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs
description: >-
  [ICML 2026][Physics & Scientific Computing][PINN] The paper upgrades the multi-resolution hash table of Instant-NGP to a "gradient-augmented" version—simultaneously storing function values and all mixed partial derivativ…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "PINN"
  - "Hash Encoding"
  - "Hermite Interpolation"
  - "Analytical Differentiation"
  - "Multi-scale Curriculum"
date: 2026-05-08
content_hash: 3a51d8632d1f0ad2
---

# Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs

**Conference**: ICML 2026  
**arXiv**: [2605.24774](https://arxiv.org/abs/2605.24774)  
**Code**: To be confirmed  
**Area**: Scientific Computing / Physics-Informed Neural Networks / Neural Field Representation  
**Keywords**: PINN, Hash Encoding, Hermite Interpolation, Analytical Differentiation, Multi-scale Curriculum

## TL;DR
The paper upgrades the multi-resolution hash table of Instant-NGP to a "gradient-augmented" version—simultaneously storing function values and all mixed partial derivatives at each hash grid point. It uses Hermite interpolation to reconstruct a field that is $C^1$ continuous and analytically second-order differentiable. This allows NGP to be effectively applied to PINNs for solving PDEs for the first time, achieving up to a $20\times$ error reduction compared to SOTA neural PDE solvers across multiple 2D/3D benchmarks, while maintaining a training speed of $2$–$3.5\,\mathrm{ms}$ per epoch.

## Background & Motivation

**Background**: Multi-resolution hash encoding (I-NGP) is a premier representation in NeRF, SDF, and image reconstruction due to $O(1)$ lookup, spatial adaptivity, and instantaneous training. It relies on $d$-linear interpolation to blend $F$-dimensional features from the hash table into a continuous field, which is then fed into a lightweight MLP.

**Limitations of Prior Work**: Directly applying I-NGP to PINNs generally fails. The fundamental reason is that $d$-linear interpolation provides only $C^0$ continuity; the first derivative is piecewise constant within cells and jumps at boundaries, while the second derivative is zero almost everywhere. Consequently, the Laplacian $\nabla^2 u$ in the PDE residual cannot be reliably obtained analytically. Existing workarounds like INGP-FD use finite difference (FD) approximations, requiring $2d+1$ forward passes per Laplacian calculation (5 for 2D, 7 for 3D). However, the $O(\epsilon^2)$ truncation error caps the accuracy at approximately $10^{-5}$, and the FD step size $\epsilon$ requires manual tuning. Another approach involves second-order autodiff, which tends to amplify hash collision noise.

**Key Challenge**: There is an inherent conflict between the "locality + speed" of hash encoding and the requirement for "high-order analytical differentiability" in PINNs. One must either sacrifice sparse hashing for SIREN/Fourier features (accurate but slow) or use hashing with FD (fast but with an accuracy ceiling).

**Goal**: To break this trade-off at the representation level by redesigning hash encoding to natively support analytical second-order derivatives while retaining the locality and instant training of NGP.

**Key Insight**: In computational physics, "gradient-augmented level set" methods (Nave et al., 2010) store both field values and gradients, reconstructing them via Hermite interpolation within grid cells. This provides a blueprint for treating derivatives as "first-class citizens" of the representation. Porting this idea to neural hash tables allows derivatives to be retrieved directly from the hash table rather than being computed post-hoc.

**Core Idea**: The hash table stores not only function values but also all $2^d$ mixed partial derivative coefficients. A tensor-product Hermite basis $H^{(\alpha)}$ is used to reconstruct a $C^1$ continuous field, where the second derivatives are obtained analytically from the basis functions. A single forward pass simultaneously yields $\gamma, \nabla\gamma$, and $\nabla^2\gamma$.

## Method

### Overall Architecture

The training pipeline of Hermite-NGP consists of:

- **Multi-resolution Hash Lookup**: At each resolution $l\in\{0,\dots,L-1\}$, grid vertices are mapped to categorized hash tables using the I-NGP hash function to retrieve Hermite coefficients $\{\theta_{l,h(g)}^{(\alpha)}\}_{\alpha\in\{0,1\}^d}$.
- **Hermite Interpolation Reconstruction**: Tensor-product Hermite bases $H^{(\alpha)}$ blend these coefficients into a local $C^1$ field, simultaneously computing $\nabla\gamma$ and $\nabla^2\gamma$.
- **SIREN MLP + Analytical Chain Rule**: The encoding $\gamma$ is fed into an MLP with $\sin(\omega\cdot)$ activations. Leveraging the SIREN second-order derivative identity $\sigma''=-\omega^2\sigma$, the chain rule is used to propagate $\nabla u$ and $\nabla^2 u$. The entire PDE residual is computed in a single forward pass.
- **Multi-scale Curriculum Training**: Resolution layers are activated from coarse to fine in three stages, mimicking a multigrid V-cycle.

The entire pipeline is summarized in Algorithm 1, with the key chain rules being $\nabla u = \frac{\partial u}{\partial \gamma}\nabla\gamma$ and $\nabla^2 u = \frac{\partial^2 u}{\partial \gamma^2}(\nabla\gamma)^2 + \frac{\partial u}{\partial \gamma}\nabla^2\gamma$.

### Key Designs

1. **Hermite Hash Encoding (Gradient-Augmented Representation)**:
    - **Function**: Upgrades the hash table from "value only" to "value + mixed partial derivatives" to ensure $C^1$ continuity and non-trivial second derivatives.
    - **Mechanism**: Each hash grid point stores $2^d$ coefficients $\{f^{(\alpha)}\}_{\alpha\in\{0,1\}^d}$: $(f, f_x, f_y, f_{xy})$ for 2D, and $(f, f_x, f_y, f_z, f_{xy}, f_{xz}, f_{yz}, f_{xyz})$ for 3D. These are stored in $2^d$ independent hash tables bucketed by derivative order. Reconstruction follows $\gamma^l(\mathbf{x}) = \sum_{g}\sum_{\alpha}\theta_{l,h(g)}^{(\alpha)}H^{(\alpha)}((\mathbf{x}-\mathbf{x}_g)/\Delta x_l)\Delta x_l^{|\alpha|}$. 1D Hermite bases use $h^{(0)}(t)=-2t^3+3t^2$ (value basis) and $h^{(1)}(t)=t^3-t^2$ (derivative basis), with the $d$-dimensional basis constructed via tensor products $H^{(\alpha)}=\Pi_i h^{(\alpha_i)}(x_i)$.
    - **Design Motivation**: Traditional $d$-linear interpolation only uses $2^d$ value coefficients, providing only $C^0$ continuity. Achieving $C^1$ continuity via Hermite interpolation requires expanding the degrees of freedom to $2 \cdot 2^d$ (derivatives at each vertex), which ensures $\nabla^2 u$ is non-zero within the cell. Categorized storage allows fine-grained control over the precision-memory trade-off; for instance, increasing the gradient table size ($2^{14}$) reduces error by 56% because first derivatives are most sensitive to hash collisions.

2. **Analytical Differentiation under SIREN Chain Rule**:
    - **Function**: Propagates $\nabla\gamma$ and $\nabla^2\gamma$ through the MLP to the output $u$, obtaining analytically differentiable $\nabla u$ and $\nabla^2 u$ for the entire network.
    - **Mechanism**: The first and second derivatives of Hermite bases are analytically computed. For the SIREN MLP with $\sigma(x)=\sin(\omega x)$, the single-layer Laplacian is $\nabla^2 u = W_2[-\omega^2 a\odot\sum_i(W_1\gamma_{x_i})^2 + \omega\cos(\omega z)\odot W_1\nabla^2\gamma]$. This second-order chain rule is applied recursively and takes only $3.5\,\mathrm{ms}$ per epoch for a model with $\sim 17$M parameters.
    - **Design Motivation**: INGP-FD requires $2d+1$ forward passes for the Laplacian, which consumes significant memory in 3D and is limited by truncation error. Hermite-NGP uses a single calculation graph, resulting in lower memory usage than INGP-FD (as verified in Table 17). SIREN is chosen because $\sigma''=-\omega^2\sigma$ simplifies the second-order chain rule by reusing intermediate forward pass values.

3. **Multi-resolution Coarse-to-Fine (C2F) Curriculum Training**:
    - **Function**: Mimics a multigrid V-cycle by activating hash resolution levels in stages to prevent early overfitting to all frequencies.
    - **Mechanism**: Three stages: (1) training only coarse layers $l=0,\dots,L_0$ for global structure; (2) progressively activating finer layers via $L_{\text{active}}(t)=\min(L, L_0+\lfloor t/\tau\rfloor)$; (3) joint fine-tuning of all layers.
    - **Design Motivation**: Capturing frequencies from low to high is a classic technique to mitigate spectral bias in PINNs. Multi-resolution hashing naturally provides a hierarchical structure, and C2F explicitly aligns the training dynamics with this hierarchy.

### Loss & Training
The standard PINN loss is used: $\mathcal{L} = \lambda_{\text{res}}\mathcal{L}_{\text{res}} + \lambda_{\text{ic}}\mathcal{L}_{\text{ic}} + \lambda_{\text{bc}}\mathcal{L}_{\text{bc}} + \lambda_{\text{data}}\mathcal{L}_{\text{data}}$. PDE residuals and boundary conditions (including Neumann) are computed using analytical $\nabla u$ and $\nabla^2 u$. The Adam optimizer is used with GradNorm for balancing. SIREN initialization uses $\omega_0=30$, and hash coefficients are initialized near zero.

## Key Experimental Results

### Main Results

| Benchmark | Setting | Hermite-NGP (Ours) | Best Baseline | Gain |
|------|------|--------------------|----------|---------|
| Helmholtz 2D | $a=10$ | **1.81e-5** | PIG 7.04e-4 | $20\times$ |
| Helmholtz 2D | $a=20$ | **7.93e-5** | PirateNet 1.36e-3 | $17\times$ |
| Helmholtz 2D | $a=100$ | **4.59e-2** | All failed | Sole convergence |
| Helmholtz 3D | $a=3$ | **6.09e-5** | PirateNet 8.40e-4 | $14\times$ |
| Helmholtz 3D | $a=10$ | **6.01e-3** | INGP-FD 7.21e-2 | $12\times$ |
| Convection 1+1D | $c=30$ | **8.49e-5** | PirateNet 8.54e-4 | $10\times$ |
| Taylor–Green | $\nu=0.01$ | **7.71e-5** | PIG 7.27e-4 | $9\times$ |
| Flow Mixing | — | **2.35e-4** | PIG 2.67e-4 | $1.1\times$ |

Performance on 3D complex geometries:

| Task | Mesh | Hermite-NGP | Baseline | Gain |
|------|------|--------------------|---------|---------|
| 3D Poisson (L2 ↓) | Armadillo | **0.0055** | PIG 0.0167 | $3.0\times$ |
| 3D Poisson (L2 ↓) | Bunny | **0.0044** | PIG 0.0127 | $2.9\times$ |
| 3D Poisson (L2 ↓) | Fandisk | **0.0031** | PIG 0.0100 | $3.2\times$ |
| SDF (Grad MAE ↓) | Armadillo | **0.0478** | NeuralAngelo 0.1009 | $2.1\times$ |
| SDF (Grad MAE ↓) | Bunny | **0.0416** | NeuralAngelo 0.0887 | $2.1\times$ |
| SDF (Grad MAE ↓) | Dragon | **0.0453** | NeuralAngelo 0.1322 | $2.9\times$ |

### Ablation Study

| Configuration | Helmholtz 2D ($a=10$) L2 | Description |
|------|--------------------------|------|
| Hermite-NGP (Full) | **1.81e-5** | C2F + Hermite Tables |
| No C2F Curriculum | $\sim 8.7$e-5 | Error rises 79.2%; validates multi-scale training |
| Cubic-NGP (No grad storage) | $>0.1$ (fail) | High-order but relies on autodiff; collision noise amplified |
| Bicubic 4×4 NGP | $>0.1$ (fail) | Similar failure to Cubic |
| INGP-FD Comparison | 1.67e-3 | FD derivatives; limited accuracy ceiling |
| Hash Tables $H_1$-$H_2$-$H_3$ = 14-14-10 | **2.26e-5** | Optimal configuration |
| Hash Tables 12-12-12 (Uniform) | 5.13e-5 | 56% degradation with uniform allocation |
| Hash Tables 14-10-14 | 9.98e-5 | First-order table is most sensitive to collisions |
| Full Autodiff 2nd Deriv | $9.5\times$ slower | Analytical encoding derivatives provide main speedup |
| Analytical Encoding + Autograd MLP | $1.2$–$1.5\times$ slower | Encoding derivatives are the primary speedup source |

### Key Findings
- **Hermite storage is essential**: All variants that used higher-order bases without storing derivatives (Cubic, Bicubic, Trilinear) failed (L2 > 0.1). This is because hash collisions inject high-frequency noise into feature channels, which is amplified by autodiff or FD; independent derivative channels absorb and distribute this noise.
- **First-order derivative tables are sensitive**: Shrinking the first-order table size increased error by 5.5×, whereas shrinking the value table only increased it by 4.9×.
- **Computation is extremely cheap**: Training takes 1.8–3.6 ms per epoch with 139–389 MB memory. In contrast, PIG requires 33.5 GB and 5 s/epoch.
- **Stability**: Relative variance across 5 random seeds is $<\sim 15\%$, indicating robustness.

## Highlights & Insights
- **"Derivatives as first-class citizens"**: This concept is transferable to differentiable rendering, SDF normal estimation, and curvature estimation where high-order accuracy is critical.
- **Decoupling hash collision from analytical differentiation**: Defining the hash function as a discrete lookup outside the continuous calculation graph avoids the paradox where high-order interpolation amplifies collision noise.
- **Win-win for storage and efficiency**: Although storing $2^d$ coefficients seems expensive, the elimination of multiple forward passes (required by FD) and associated activation maps results in lower memory usage for 3D tasks.

## Limitations & Future Work
- **High-dimensional scaling**: Storing coefficients scales as $2^d$, making 4D+ spatio-temporal PDEs expensive; quantization or low-rank decomposition may be required.
- **Coupling with SIREN**: While theoretically compatible with other activations like Swish, the implementation currently relies on SIREN for algebraic simplification.
- **PDE Forms**: Testing has been limited to strong-form PDE residuals; applying this to weak forms (Galerkin) or complex moving boundaries remains an open question.
- **Order of Continuity**: $C^1$ cubic Hermite may need expansion to quintic Hermite for fourth-order equations (e.g., plate equations $\nabla^4$).

## Related Work & Insights
- **vs INGP-FD**: Hermite-NGP achieves higher accuracy and lower memory overhead by using analytical derivatives rather than finite differences.
- **vs PirateNet / JAX-PI / PIG**: Hermite-NGP leverages the spatial adaptivity of NGP to outperform dense MLP and Gaussian-based representations by $9$–$20\times$.
- **vs NeuralAngelo**: Provides 2.4× lower gradient MAE and significantly smoother curvature fields by avoiding FD.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First effective application of NGP to PINNs via fundamental innovations in representation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive benchmarks across PDEs and geometries with extensive ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear explanations, rigorous pseudocode, and insightful failure analysis.
- Value: ⭐⭐⭐⭐⭐ Significant advancement in pushing neural PDE solver accuracy to $10^{-5}$ while increasing speed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/physics/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](../../NeurIPS2025/physics/integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](../../ICLR2026/physics/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)
- [\[ICML 2026\] Unbiased and Second-Order-Free Training for High-Dimensional PDEs](unbiased_and_second-order-free_training_for_high-dimensional_pdes.md)
- [\[ICML 2026\] EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](eqgino_equivariant_geometry-informed_fourier_neural_operators_for_3d_pdes.md)

</div>

<!-- RELATED:END -->

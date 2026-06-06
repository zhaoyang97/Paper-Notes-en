---
title: >-
  [Paper Note] EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs
description: >-
  [ICML 2026][Physics & Scientific Computing][Fourier Neural Operator] EqGINO transforms the GNO encoder, FNO backbone, and GNO decoder of GINO into SE(3)-equivariant modules: GNO uses relative distance as a rotation-invar…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Fourier Neural Operator"
  - "SE(3) Equivariance"
  - "Spectral Convolution"
  - "Orbit-based Weight Sharing"
  - "3D PDE Surrogate"
date: 2026-05-08
content_hash: e4d8a2294f873c73
---

# EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs

**Conference**: ICML 2026  
**arXiv**: [2606.03260](https://arxiv.org/abs/2606.03260)  
**Code**: The paper states "available at this URL," but no specific repository is provided.  
**Area**: Scientific Computing / Neural Operators / Equivariant Networks / 3D PDE  
**Keywords**: Fourier Neural Operator, SE(3) Equivariance, Spectral Convolution, Orbit-based Weight Sharing, 3D PDE Surrogate  

## TL;DR
EqGINO transforms the GNO encoder, FNO backbone, and GNO decoder of GINO into SE(3)-equivariant modules: GNO uses relative distance as a rotation-invariant kernel, and FNO employs "orbit-based weight sharing" in the frequency domain to enforce isotropy where $W(R\mathbf k)=W(\mathbf k)$. This preserves the global receptive field of FNO while making the 3D PDE surrogate robust to arbitrary rigid transformations and reducing spectral weight parameters from $\mathcal O(K^3)$ to $\mathcal O(K)$.

## Background & Motivation
**Background**: 3D PDE surrogate models (aerodynamics, structural mechanics, turbulence simulation, etc.) currently follow two main paths: point cloud/mesh-based GNNs (PointNet++, MeshGraphNet, Transolver, GINO) and spectral-based FNO variants. GINO (Li et al., 2023) is a recognized strong baseline because it uses GNO to project irregular point clouds onto regular grids before applying FNO for spectral convolutions, reconciling irregular geometry with a global receptive field.

**Limitations of Prior Work**: Fundamental physical laws are invariant under coordinate transformations (Navier–Stokes remains consistent under rotation/translation), yet nearly all current SOTA models rely on absolute Cartesian coordinates as input features. This causes models to overfit to the "canonical orientation" during training. Performance collapses as soon as test samples are rotated by 90°, 180°, or arbitrary angles (Paper Table 1b: GINO's ShapeNetCar pressure RMSE jumps from 0.166 to 0.563; DeepJEB deformation leaps from 0.111 to 2.319).

**Key Challenge**: Equivariance and a global receptive field are difficult to achieve simultaneously. Existing equivariant GNNs (EGNN, EMNN, T-EMNN) rely on local message passing to ensure SE(3) equivariance, but their local receptive fields cannot capture the long-range interactions essential for PDEs. FNO naturally possesses a global receptive field in the frequency domain, but 3D spectral group convolutions (such as the 2D extension of G-FNO) are computationally too expensive for practical use.

**Goal**: (i) Identify a lightweight mechanism to enforce SE(3) equivariance in the 3D frequency domain; (ii) integrate it seamlessly with a GNO encoder for irregular geometries to obtain an end-to-end equivariant version of GINO; (iii) handle both scalar field (e.g., von Mises stress) and vector field (e.g., deformation $\mathbf u\in\mathbb R^3$) prediction tasks.

**Key Insight**: The authors observe that the Fourier transform satisfies $\widehat{\mathcal T_R f}(\mathbf k)=\hat f(R^{-1}\mathbf k)$, meaning spatial rotation corresponds to a synchronous rotation of the Fourier modes in the frequency domain (Lemma 4.1). FNO breaks equivariance solely because the learnable spectral weights $W(\mathbf k)$ are independent for every $\mathbf k$—by forcing $W(R\mathbf k)=W(\mathbf k)$, equivariance is restored.

**Core Idea**: Forcing spectral weight sharing along "equal-magnitude orbits"—where all frequency modes satisfying $\|\mathbf k\|_2\approx r$ share a single weight $w_r$—ensures rotation equivariance while cutting parameter complexity from $\mathcal O(K^3)$ to $\mathcal O(K)$.

## Method
EqGINO adopts the three-stage skeleton of GINO—an EqGNO encoder lifts point clouds to a regular grid; multi-layer EqFNO performs equivariant global convolutions in the frequency domain; an EqGNO decoder projects grid features back to point clouds to predict physical quantities—but every stage is redesigned for SE(3) equivariance.

### Overall Architecture
- **Input**: Irregular point clouds $\mathcal P=\{y_j\}$ (CFD mesh or vehicle surface); each point carries sparse physical features but **no** absolute coordinates.
- **EqGNO Encoder** $\mathcal E$: For each regular grid point $x^{grid}$, a Riemann summation $v_0(x^{grid})\approx\sum_j\kappa(x^{grid},y_j)\mu_j$ is performed within a local sphere of radius $r$, where the kernel $\kappa$ only takes relative distances as input.
- **EqFNO Layer $\mathcal K_l$**: $L$ layers of equivariant spectral convolutions, where each layer $\mathcal K_l(v_{l-1})=\sigma(S_l v_{l-1}+\mathcal F^{-1}[W(\mathbf k)\cdot\mathcal F v_{l-1}])$ has weights $W$ forced to be isotropic.
- **EqGNO Decoder** $\mathcal D$: For each target point $y^{out}$, kernel integration is performed on neighboring regular grid points to back-project the physical quantity $u(y^{out})$.
- **Overall Pipeline**: $G_\theta=\mathcal D\circ \mathcal K_L\circ\cdots\circ\mathcal K_1\circ\mathcal E$.
- **Output**: Physical fields on each original point (scalars like pressure/von Mises stress; vectors like wall shear stress/deformation).

### Key Designs

1.  **EqGNO Encoder/Decoder with Rotation-Invariant Kernels**:
    - **Function**: Stably maps irregular point clouds to regular grids (encoder) and grid features back to point clouds (decoder), ensuring the entire mapping is equivariant to rigid transformations.
    - **Mechanism**: The original GNO feeds absolute coordinates $(x,y)$ into a kernel MLP, which change upon rotation. EqGNO switches to pure scalar distances: $\kappa(x^{grid},y_j)=\phi_\theta(\|x^{grid}-y_j\|,\|x^{grid}-\bar y\|)$, where the second term is the distance to the point cloud center $\bar y$ to inject global radial context. The decoder dually uses $\|y^{out}-x^{grid}_j\|$ and $\|y^{out}-\overline{x^{grid}}\|$. All quantities are $SE(3)$-invariant scalars, so kernel values remain identical under rotation/translation.
    - **Design Motivation**: The simplest way to achieve equivariance is to restrict the network to invariant inputs. Completely removing coordinate inputs loses global positional information, so adding "distance to center" acts as a weak positional encoding that preserves equivariance without making all points appear identical. This is foundational for handling geometric irregularity while maintaining equivariance.

2.  **EqFNO: Isotropic Spectral Convolution with Orbit-based Weight Sharing**:
    - **Function**: Executes global convolutions in the frequency domain that strictly satisfy $SO(3)$ equivariance, while significantly reducing parameter counts to make Full-FFT affordable.
    - **Mechanism**: Theorems show that for scalar fields, a necessary and sufficient condition for spectral convolution equivariance is $W(R\mathbf k)=W(\mathbf k)$, meaning $W$ depends only on $\|\mathbf k\|_2$. The authors introduce **Orbit-based Weight Sharing**: grouping all frequency modes satisfying $\|\mathbf k\|_2\approx r$ into the same orbit $\mathcal O_r$ to share a single learnable weight $w_r$. This reduces per-layer spectral parameters from $\mathcal O(K^3)$ to $\mathcal O(K)$. Furthermore, Full-FFT is mandatory instead of RFFT, as the Hermitian conjugate symmetry of RFFT would require "anti-linear" operations under rotation, breaking complex-linear convolution. To compensate for the doubled FLOPs of Full-FFT, block-diagonal grouping ($G$ groups) is added to the channel dimension, reducing channel-mixing costs to $d_{out}d_{in}N/G$; at $G=2$, it exactly offsets the extra computation of Full-FFT.
    - **Design Motivation**: 3D spectral group convolutions (3D extensions of G-FNO) are computationally unsustainable. Since PDE physics naturally demands isotropic responses, making weights dependent only on frequency magnitude is both the weakest constraint for equivariance and a strong inductive bias aligned with physical priors. The combination of orbit-sharing, Full-FFT, and channel grouping enables equivariant FNO to run at scale in 3D for the first time.

3.  **SE(3) Equivariant Local Bases for Vector Field Prediction**:
    - **Function**: Allows the naturally "scalar-friendly" orbit-sharing framework to predict 3D vector fields (e.g., deformation $\mathbf u$, wall shear stress $\boldsymbol\tau$).
    - **Mechanism**: Predicting the three components of a vector directly violates equivariance. The authors frame vector prediction as "regressing three projection coefficients $(\alpha,\beta,\gamma)$ on an SE(3)-equivariant local basis $\{\mathbf e_1,\mathbf e_2,\mathbf e_3\}$," and then linearly reconstructing $\mathbf v=\alpha\mathbf e_1+\beta\mathbf e_2+\gamma\mathbf e_3$. Since the coefficients are $SE(3)$-invariant scalars, they can be output directly by EqFNO; the reconstructed vector then rotates with the basis, ensuring global equivariance.
    - **Design Motivation**: This is a "geometric reparameterization" that downgrades vector tasks into scalar tasks, allowing all existing equivariant modules to be reused. The specific construction of local bases varies by dataset (AhmedBody/DeepJEB), but the underlying logic is unified and decouples the framework for general 3D physical quantities.

### Loss & Training
Tasks are regressions using relative $L_2$ error. The paper sets two configurations: EqGINO* ($G=2, K=32$, aligning with GINO's compute budget) and EqGINO ($G=4, K=40$, same parameter count but higher spectral resolution).

## Key Experimental Results

### Main Results: In-Distribution + Zero-shot Discrete Rotation Generalization (Octahedral Group $O$)
3 datasets / 8 physical quantities; relative $L_2$ error (lower is better); "Canonical→Discrete" measures training on canonical orientations and testing on multiples of 90° rotations.

| Dataset / Quantity | GINO (Canon.) | Transolver (Canon.) | **EqGINO (Canon.)** | GINO (Rot.) | Transolver (Rot.) | **EqGINO (Rot.)** |
|---|---|---|---|---|---|---|
| AhmedBody / Wall Shear | 0.199 | 0.129 | **0.196** | 0.624 | 0.795 | **0.196** |
| AhmedBody / Pressure | 0.167 | 0.276 | **0.164** | 0.563 | 0.519 | **0.164** |
| ShapeNetCar / Pressure | 0.161 | 0.119 | **0.177** | 1.495 | 1.663 | **0.177** |
| DeepJEB / Deflection | 0.111 | 0.162 | **0.171** | 2.319 | 4.506 (PointNet) | **0.171** |
| DeepJEB / von Mises Stress | 0.403 | 0.374 | **0.385** | 1.127 | 1.042 | **0.385** |

Key observation: Non-equivariant baselines show errors amplified by 3–20x on rotated test sets, whereas EqGINO's performance remains identical—this is the zero-loss generalization provided by "by design" equivariance.

### Ablation Study: Continuous Rotation Generalization (Training with continuous $SE(3)$ augmentation)

| Model | AhmedBody Press ↓ | ShapeNetCar Press ↓ | DeepJEB Deflection ↓ | DeepJEB Stress ↓ |
|------|-------------------|---------------------|-----------------------|------------------|
| GINO (Non-equivariant) | 0.211 | 0.181 | 0.158 | 0.420 |
| Transolver (Non-equivariant) | 0.422 | 0.335 | 0.217 | 0.366 |
| EGNN (Equivariant GNN) | 0.818 | 0.654 | 0.838 | 0.592 |
| T-EMNN (Equivariant GNN) | 0.620 | 0.180 | 0.305 | 0.424 |
| Transolver* (No-coord version) | 0.642 | 0.927 | 0.401 | 0.512 |
| **EqGINO** | **0.185** | **0.156** | **0.162** | **0.367** |

### Key Findings
- "Equivariance by design" far outperforms "Non-equivariant SOTA + Augmentation": Even with continuous rotation augmentation, GINO/Transolver remain at the 0.21/0.42 level for AhmedBody Pressure, while EqGINO reaches 0.185 without needing any augmentation during training.
- Equivariant GNNs (EGNN/EMNN/T-EMNN) based on local message passing still cannot match EqGINO, confirming that a "global receptive field is a hard requirement for PDEs"—grafting equivariance onto spectral methods is the superior path.
- Transolver* (the variant stripped of coordinates) degrades significantly compared to Transolver, showing that "cheating with absolute coordinates" is not viable for equivariant routes; Ours maintains expressivity through weak positional encoding like "distance to center" in EqGNO.
- Orbit-sharing slashes 3D spectral weights from $\mathcal O(K^3)$ to $\mathcal O(K)$, and combined with block-diagonal grouping, the training cost is comparable to GINO while performance is better or equal—demonstrating that this structural prior saves parameters while improving results.

## Highlights & Insights
- **"The lowest cost of equivariance is isotropy"**: The authors simplify the SO(3) equivariance constraint to $W(\mathbf k)$ depending only on $\|\mathbf k\|$, bypassing expensive 3D spectral group convolutions. This elegant simplification suggests that many equivariant requirements can be met at minimal cost through "isotropy in the dual domain."
- **Computational balancing via Full-FFT + Channel Grouping**: Since RFFT is unusable under equivariance due to Hermitian conjugate breaking complex linearity, the authors use Full-FFT and recover the compute budget through channel blocking. This "correctness first, efficiency later" design is a valuable engineering lesson.
- **Vector Field = Scalar Coefficients + Equivariant Local Bases**: This reparameterization strategy is highly versatile and can be transferred to any scenario where a scalar equivariant backbone needs to handle vector predictions (e.g., force vectors in RL policies, deformation reconstruction).

## Limitations & Future Work
- Strict equivariance is only guaranteed for the Octahedral group $O$ (multiples of 90° on regular grids); it only approximates arbitrary continuous $SE(3)$—generalizing well through structural priors rather than being equivariant by design. Systems with very high symmetry might still require further refinement.
- Datasets are limited to single-step steady-state predictions (external flow, structural static loads); temporal rollout robustness has not been directly validated.
- Orbit discretization (thresholding for $\|\mathbf k\|_2\approx r$) depends on grid resolution $K$; coarse orbit granularity at low resolution may limit expressivity. Adaptive or multi-scale orbits are worth exploring.
- Local basis construction is manually designed per dataset, lacking a unified geometric principle, which complicates adaptation to new tasks.

## Related Work & Insights
- **vs GINO**: Same backbone, but Ours converts every module to be equivariant; it matches canonical tests and leads by orders of magnitude on rotated tests.
- **vs G-FNO (Helwig 2023)**: G-FNO reaches equivariance in 2D using spectral group convolutions but fails in 3D due to compute; EqGINO replaces group convolutions with orbit-sharing to make 3D equivariant FNO practical.
- **vs EGNN / EMNN / T-EMNN**: These rely on local message passing and are limited by receptive fields; EqGINO is naturally global via spectral methods.
- **vs Transolver**: Transolver is strong but heavily dependent on coordinate features; its equivariant variant Transolver* collapses—indicating that Transformer-based operator learning still needs new positional encoding schemes to follow the equivariant path.
- **vs EGNO (Xu 2024)**: EGNO focuses on equivariance in the time dimension (trajectory prediction), while Ours focuses on space-dimension SE(3) equivariance; the two approaches are complementary.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids](../../AAAI2026/physics/phys-liquid_a_physics-informed_dataset_for_estimating_3d_geometry_and_volume_of_.md)
- [\[ICML 2026\] Generative Neural Operators Through Diffusion Last Layer](generative_neural_operators_through_diffusion_last_layer.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/physics/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[ICML 2026\] Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation](iterative_refinement_neural_operators_are_learned_fixed-point_solvers_a_principl.md)
- [\[ICML 2026\] Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs](hermite-ngp_gradient-augmented_hash_encoding_for_learning_pdes.md)

</div>

<!-- RELATED:END -->

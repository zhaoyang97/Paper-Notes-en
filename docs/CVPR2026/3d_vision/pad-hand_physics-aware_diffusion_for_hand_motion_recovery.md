---
title: >-
  [Paper Note] PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery
description: >-
  [CVPR 2026][3D Vision][hand motion recovery] PAD-Hand is a physics-aware conditional diffusion framework that models Euler–Lagrange (EL) dynamics residuals as virtual observations integrated into the diffusion process…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "hand motion recovery"
  - "physics-aware diffusion model"
  - "Euler–Lagrange dynamics"
  - "Laplace approximation"
  - "uncertainty estimation"
date: 2026-05-08
content_hash: 6becb7574fcebf9b
---

# PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery

**Conference**: CVPR 2026
**arXiv**: [2603.26068](https://arxiv.org/abs/2603.26068)  
**Code**: N/A  
**Area**: 3D Vision
**Keywords**: hand motion recovery, physics-aware diffusion model, Euler–Lagrange dynamics, Laplace approximation, uncertainty estimation

## TL;DR

PAD-Hand is a physics-aware conditional diffusion framework that models Euler–Lagrange (EL) dynamics residuals as virtual observations integrated into the diffusion process, while estimating per-joint, per-frame dynamic variance via last-layer Laplace approximation. The method achieves physically plausible and uncertainty-aware hand motion recovery, reducing acceleration error by 50.1% on DexYCB.

## Background & Motivation

1. **Background**: Monocular 3D hand reconstruction has made substantial progress, with large-scale pretrained models (e.g., WiLoR) improving per-frame accuracy. However, temporal inconsistency remains a persistent issue, and existing methods primarily capture kinematic patterns without sensitivity to dynamics.

2. **Limitations of Prior Work**: (1) Image-based estimation lacks temporal consistency, producing inter-frame jitter; (2) existing physics-constrained methods (e.g., enforcing EL residuals to zero) are deterministic—they assume the observed motion can fully satisfy the physical equations, ignoring uncertainty from estimation noise and model approximation; (3) deterministic physical constraints may induce difficult optimization landscapes or suboptimal solutions.

3. **Key Challenge**: 3D hand estimation is inherently noisy and physical models are only approximate; the assumption of zero residuals is inconsistent with reality. A probabilistic approach to physics integration is needed—one that allows the model to reason over the motion data manifold and produce a distributed solution space.

4. **Goal**: (1) Probabilistically integrate physical dynamics into the diffusion model as a replacement for hard constraints; (2) provide interpretable physical consistency metrics (variance) indicating which frames and joints yield unreliable estimates.

5. **Key Insight**: EL dynamics residuals are treated as "virtual observations" sampled from a distribution, whose likelihood is coupled with the visual data term to guide the reverse diffusion process.

6. **Core Idea**: Replace deterministic physical constraints with probabilistic physics—virtual observations combined with Laplace-approximated variance—for diffusion-based hand motion recovery.

## Method

### Overall Architecture

$T$ input frames → off-the-shelf image estimator (e.g., WiLoR) producing per-frame MANO pose $\theta_{1:T}$ and mean shape $\beta_{avg}$ → conditional diffusion model iteratively denoises from $x^N_{1:T}$ to clean motion $x^0_{1:T}$ → simultaneously yields per-step dynamic variance $\text{Var}(\mathcal{F}^n_{1:T})$. The final outputs are the refined pose trajectory $x^0_{1:T}$ and the dynamics residual variance $\text{Var}(\mathcal{F}^0_{1:T})$.

### Key Designs

1. **Euler–Lagrange Hand Dynamics Modeling**:

    - **Function**: Establishes physics-law-based dynamic equations for an articulated hand.
    - **Mechanism**: The EL equation $M\ddot{q} + C + g = \mathcal{F}$ is formulated using generalized coordinates $\mathtt{q} = \{R, t, \theta\}$ (wrist rotation, translation, and 15 joint angles), where $M$ is the generalized mass matrix, $C$ is the Coriolis/centrifugal term, $g$ is the gravity term, and $\mathcal{F}$ is the net generalized force. Mass and inertia tensors for each hand segment are computed by tetrahedralizing the MANO mesh to obtain volume, then multiplying by the density $\rho$ from the literature.
    - **Design Motivation**: Derives hand dynamics from first principles rather than relying on learned implicit physical priors.

2. **Probabilistic Physics Integration via Dynamics Residuals as Virtual Observations**:

    - **Function**: Probabilistically incorporates EL residuals into diffusion training, avoiding deterministic hard constraints.
    - **Mechanism**: The backbone computes the EL residual $Z_t = M_t\ddot{q}_t + C_t + g_t - \hat{\mathcal{F}}_t$ from the denoised motion. The residual is treated as a virtual observation sampled from $\mathcal{N}(Z(x^0_{1:T}), \sigma^2 I)$, and the negative log-likelihood is used as the physics loss $\mathcal{L}_{EL} = \frac{1}{2\sigma_n}\|Z_{1:T}(x^0_{1:T})\|^2$. Crucially, residuals are computed on the reverse-diffusion sample $x^0_{1:T}$ rather than on the directly predicted $\hat{x}_{1:T}$, since Jensen's inequality implies $Z(\mathbb{E}[x^0]) \neq \mathbb{E}[Z(x^0)]$.
    - **Design Motivation**: The probabilistic treatment enables the model to trade off between physical plausibility and visual data, without being misled by the approximate physical model.

3. **Last-Layer Laplace Approximation for Variance Estimation (LLLA)**:

    - **Function**: Estimates per-joint, per-frame dynamic variance as an interpretable physical consistency metric.
    - **Mechanism**: A posterior Laplace approximation is applied to the backbone's last-layer parameters, yielding the predictive posterior $p(\hat{x}_{1:T}|x^n_{1:T},n,D) \approx \mathcal{N}(f_\phi, \text{diag}(\gamma^2_\phi))$. Variance is then propagated through each reverse diffusion step: $\text{Var}(x^{n-1}_{1:T}) = A_n^2\text{Var}(x^n_{1:T}) + B_n^2\text{Var}(\hat{x}_{1:T}) + \Sigma_n^2 + 2A_nB_n\text{Cov}(x^n, \hat{x})$. The force variance is finally obtained via Jacobian linearization: $\text{Var}(\mathcal{F}_{1:T}) \approx J_{\mathcal{F}} \text{Var}(x^0_{1:T}) J^\top_{\mathcal{F}}$.
    - **Design Motivation**: High-variance regions correspond to frames and joints with weak physical consistency, providing a reliability signal for downstream applications.

### Loss & Training

- Total loss: $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{data} + \lambda_2 \mathcal{L}_{EL}$, with $\lambda_1 = 2000$, $\lambda_2 = 500$
- Data loss: $\mathcal{L}_{data} = \mathbb{E}_{n}\|x_{1:T} - f_\phi(x^n_{1:T}, y_{1:T}, n)\|^2$
- Backbone: Transformer encoder-decoder (4 encoder + 4 decoder layers, 8 heads, dim=512); MeshCNN extracts spatial features
- Sequence length $T=16$, diffusion steps $N=4$, Monte Carlo samples $S=20$
- AdamW, lr=$2\times10^{-4}$, decayed by 0.8 every 10 epochs

## Key Experimental Results

### Main Results

DexYCB results (initialized from WiLoR):

| Method | Input | Type | PA-MPJPE↓ | MPJPE↓ | ACCEL↓ |
|--------|-------|------|-----------|--------|--------|
| WiLoR | Image | D | 4.88 | 12.75 | 6.70 |
| Deformer | Sequence | D | 5.22 | 13.64 | 6.77 |
| TCMR | Sequence | D | 6.28 | 16.03 | - |
| MaskHand | Image | P | 5.0 | 11.70 | - |
| **PAD-Hand** | Sequence | P | **4.63** | **10.56** | **3.34** |

HO3D results:

| Method | PA-MPJPE↓ | ACCEL↓ |
|--------|-----------|--------|
| WiLoR | 7.50 | 4.98 |
| Deformer | 9.40 | 6.37 |
| **PAD-Hand** | **7.43** | **2.71** |

### Ablation Study

Component ablation on DexYCB:

| Configuration | PA-MPJPE↓ | MPJPE↓ | ACCEL↓ |
|---------------|-----------|--------|--------|
| WiLoR (baseline) | 4.88 | 12.75 | 6.70 |
| $\mathcal{L}_{data}$ only | 4.65 | 10.62 | 3.36 |
| $\mathcal{L}_{data} + \mathcal{L}^D_{EL}$ (deterministic physics) | 4.66 | 10.61 | 3.35 |
| $\mathcal{L}_{data} + \mathcal{L}_{EL}$ (probabilistic physics, Ours) | **4.63** | **10.56** | **3.34** |

### Key Findings

- **Acceleration error reduced by 50.1%**: from 6.70 to 3.34 mm/frame², demonstrating that physical constraints substantially improve motion smoothness.
- **PA-MPJPE reduced by 5.1%, MPJPE by 17.2%**: physical plausibility is improved without sacrificing—and in fact while enhancing—reconstruction accuracy.
- **Probabilistic physics outperforms deterministic physics**: the probabilistic formulation consistently outperforms the deterministic penalty $\mathcal{L}^D_{EL}$ across all metrics, validating the necessity of modeling residuals as virtual observations.
- **Variance aligns closely with EL residuals**: high-variance intervals coincide with high physical residuals (i.e., physically implausible motion), confirming that the variance estimate is a reliable uncertainty indicator.
- **ACCEL on HO3D reduced from 4.98 to 2.71 (−45.6%)**: consistent improvement across datasets demonstrates strong generalization.

## Highlights & Insights

- **Virtual observation formulation converts physics from a hard constraint to a soft prior**: rather than enforcing zero residuals, the method incorporates residual likelihood into the diffusion objective, allowing coexistence of approximate physical models and observation noise. This idea is transferable to any generative model requiring physical constraints.
- **Variance propagation through the entire diffusion process**: starting from a Dirac delta at the final step, variance is propagated step by step via a closed-form formula, ultimately yielding the variance of $x^0$. Such end-to-end uncertainty propagation within diffusion models is relatively uncommon.
- **MeshCNN + Transformer combines topology-aware spatial features with temporal modeling**: using MeshCNN to extract topological features of the hand mesh before feeding them into a Transformer for temporal modeling is an elegant architectural design.

## Limitations & Future Work

- The physical model does not explicitly account for object geometry or contact forces, limiting accuracy in hand–object interaction scenarios.
- Variance estimation relies on LLLA approximation and Monte Carlo sampling ($S=20$), incurring non-trivial computational overhead.
- Validation is limited to the MANO model; extension to full-body or other articulated structures has not been explored.
- With only 4 diffusion steps, the model is efficient but may have limited capacity for modeling complex motions.

## Related Work & Insights

- **vs. Zhang et al. (2025)**: The prior work also employs diffusion models with physics regularization, but conditions on hand–object proximity and uses deterministic physics. The proposed method adopts probabilistic EL residuals and does not require object information.
- **vs. BioPR**: A deterministic physics-constrained method achieving MPJPE of 12.81 vs. 10.56 for the proposed method—a substantial margin.
- **vs. Rixner et al. / Bastek et al.**: The virtual observation concept is imported from the general framework of physics-informed diffusion models into the 3D hand domain, with the first incorporation of variance estimation.
- The estimated variance can be directly leveraged for active learning (annotating high-variance sequences) or as confidence weights in downstream hand–object interaction modeling.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of probabilistic physics integration and diffusion-based variance propagation is novel, though individual components build on prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on two datasets (DexYCB and HO3D) with clearly designed ablations and convincing qualitative variance visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous, though the density of equations is high.
- **Value**: ⭐⭐⭐⭐ Introduces a quantifiable physical consistency metric for hand motion recovery with practical utility for AR/VR and embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves](glove2hand_synthesizing_natural_hand-object_interaction_from_multi-modal_sensing.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](../../ICCV2025/3d_vision/hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[CVPR 2026\] DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)

</div>

<!-- RELATED:END -->

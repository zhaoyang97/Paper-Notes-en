---
title: >-
  [Paper Note] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors
description: >-
  [CVPR 2026][Medical Imaging][Tagged MRI] This paper proposes InvTag, a framework that, for the first time, integrates a physics-based MR forward model with a pretrained diffusion generative prior to jointly solve three s…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Tagged MRI"
  - "Inverse Problem"
  - "Diffusion Prior"
  - "Motion Estimation"
  - "Image Super-Resolution"
date: 2026-05-08
content_hash: f70924eb7219a1fb
---

# Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors

**Conference**: CVPR 2026
**arXiv**: [2603.00882](https://arxiv.org/abs/2603.00882)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Tagged MRI, Inverse Problem, Diffusion Prior, Motion Estimation, Image Super-Resolution

## TL;DR

This paper proposes InvTag, a framework that, for the first time, integrates a physics-based MR forward model with a pretrained diffusion generative prior to jointly solve three sub-tasks in 3D Tagged MRI—anatomical recovery, Cine synthesis, and motion estimation—without requiring any additional training data.

## Background & Motivation

Tagged MRI applies periodic tags to tissue to track internal motion, and is widely used in cardiac motion analysis and brain biomechanics research. However, its post-processing faces three major challenges:

**Tag interference**: The presence of tags prevents conventional anatomical segmentation methods from being applied directly.

**Tag Fading**: Due to T1 relaxation, tag contrast degrades sharply over time, violating the brightness constancy assumption of optical flow methods.

**Low resolution**: To accelerate acquisition, the spatial resolution of Tagged MRI is typically lower than that of standard structural MRI.

Traditional methods treat motion tracking, Cine synthesis, and super-resolution as independent tasks. However, these three tasks are inherently coupled: reliable motion tracking requires handling tag fading and spectral overlap, while resolving spectral overlap requires separating anatomical structures from tag patterns. The authors argue that a unified framework for joint estimation is necessary.

## Method

### Overall Architecture

InvTag formulates Tagged MRI analysis as a **nonlinear blind inverse problem**. Given a low-resolution Tagged MRI time series, the framework jointly recovers: (1) a high-resolution anatomical image $a$; (2) a tag-free Cine sequence; (3) 3D diffeomorphic motion fields $\{\phi_t\}$; and (4) the anisotropic point spread function (PSF) of the imaging system. The nonlinearity arises from deformable spatial transformations, and the blindness arises from the unknown PSF and fading parameters.

### Key Designs

1. **Physics-based forward model**: The observed Tagged image is modeled as
   $$g_t^{\Box} = h_\gamma^{\Box} * \phi_{\theta_t}^* [a \cdot f_{\beta_t}(q_\alpha^{\Box})] + n_t^{\Box}$$
   where $a$ is the reference-frame anatomy, $q_\alpha^{\Box}$ is the SPAMM-parameterized sinusoidal tag pattern, $f_{\beta_t}$ is an affine fading model, $h_\gamma^{\Box}$ is an anisotropic Gaussian PSF, and $\phi_{\theta_t}$ is a PINN-based diffeomorphic deformation field (diffeomorphism is guaranteed via the exponential map $\phi_t = \exp\{v_t\}$). All time frames share the same anatomy $a$, with geometric variation produced solely through $\phi_t$, naturally enforcing temporal consistency.

2. **Diffusion generative prior**: A diffusion model pretrained on 80,000+ 1mm isotropic T1w 3D brain volumes is used as an anatomical prior. The data fidelity term is incorporated into the reverse diffusion SDE via DPS (Diffusion Posterior Sampling):
   $$da_\tau = -\eta_\tau \Big[\frac{1}{2}a_\tau + s_\vartheta(a_\tau, \tau) - \rho \nabla_{a_\tau} \mathcal{L}_{\text{rec}}(\hat{a}_0(a_\tau))\Big] d\tau + \sqrt{\eta_\tau} d\bar{w}$$
   The diffusion score $s_\vartheta$ pulls samples toward the anatomical manifold, while the data fidelity term enforces consistency with observations.

3. **Coordinate Descent with Diffusion Prior (CDDP)**: Two steps are alternated: (A) fix the forward model parameters and update anatomy $a$ via diffusion posterior sampling; (B) fix $a$ and estimate forward model parameters via maximum likelihood. Low-dimensional parameters ($\gamma, \alpha, \beta_t$) are optimized with a bounded differential evolution optimizer (to handle the highly non-convex landscape), while high-dimensional motion parameters $\theta_t$ are updated with Adam. The first frame jointly estimates $(a^\star, \alpha^\star, \gamma^\star)$, which are then fixed; subsequent frames update only the fading and motion parameters.

### Loss & Training

- **Data reconstruction loss**: $\mathcal{L}_{\text{rec}}(a) = \sum_t \sum_{\Box} \|g_t^{\Box} - \mathcal{A}_t^{\Box}(a)\|_2^2$
- **Diffusion prior**: Pretrained weights are frozen; 256-step DDIM sampling is used.
- **CDDP iterations**: $L=4$ rounds of coordinate descent; motion is initialized from the previous time step (to avoid periodic tag matching ambiguity).
- No external Tagged or Cine training data is required; no paired supervision or fine-tuning.

## Key Experimental Results

### Main Results

**Tag-to-Cine synthesis** (160 test cases: 20 AIBL + 20 Sleep subjects × 4 imaging configurations):

| Method | PSNR ↑ (t=1) | SSIM ↑ (t=1) | PSNR ↑ (t=6) | SSIM ↑ (t=6) |
|---|---|---|---|---|
| LowpassFuse | 26.43 | 0.62 | 26.68 | 0.66 |
| HARP Demod. | 24.28 | 0.52 | 23.93 | 0.54 |
| **InvTag (Ours)** | **28.38** | **0.83** | **28.41** | **0.84** |

**Motion estimation**:

| Method | EPE ↓ | EPE@95 ↓ | NegDet(%) ↓ |
|---|---|---|---|
| LKUnet | 1.35 | 2.94 | 0.043 |
| DeepTag | 1.27 | 2.97 | 0.060 |
| SyN | 1.06 | 2.41 | <0.001 |
| DRIMET | 0.79 | 1.61 | <0.001 |
| **InvTag (Ours)** | **0.60** | **1.31** | **<0.001** |

### Ablation Study

| Configuration | PSNR ↑ | SSIM ↑ | EPE ↓ | EPE@95 ↓ |
|---|---|---|---|---|
| w/o PSF estimation | 27.27 | 0.69 | 0.62 | 1.41 |
| w/o fading estimation | 28.21 | 0.80 | 0.71 | 1.56 |
| w/o CDDP (joint optimization) | 22.05 | 0.46 | 1.57 | 2.73 |
| **Full model** | **28.40** | **0.83** | **0.60** | **1.31** |

### Key Findings

- Replacing CDDP with joint optimization causes severe degradation (PSNR drops by 6.35), confirming that alternating optimization is critical for blind inverse problem solving.
- PSF estimation contributes significantly to synthesis quality, while fading estimation is more critical for motion tracking.
- On real rotating gel phantom data—where the diffusion prior is trained only on synthetic ellipsoids—the framework still successfully recovers anatomy and motion.
- Variance in estimated PSF/tag parameters across 5 random initializations is negligible, confirming reliable convergence of CDDP.

## Highlights & Insights

- **First unified framework**: Jointly addresses three core tasks in Tagged MRI analysis, leveraging inter-task coupling for mutual reinforcement.
- **Nonlinear blind inverse problem**: Treats MR physics as hard constraints and the diffusion prior as a soft constraint, overcoming the limitations of prior diffusion-based inverse problem solvers that assume linear or known forward operators.
- **Zero-shot generalization**: Requires no Tagged/Cine training data whatsoever, relying solely on a T1w diffusion prior.
- The CDDP strategy demonstrates strong stability and convergence robustness in non-convex optimization.

## Limitations & Future Work

- **Long runtime**: Processing a single frame takes 1.2 hours on a single A40 GPU; repeated diffusion sampling and PINN optimization are the bottlenecks.
- Only sinusoidal tags are assumed; grid tags and higher-order tag patterns are not supported.
- Validation is limited to brain Tagged MRI; extension to broader applications such as cardiac MR tagging has not been explored.
- Uncertainty quantification is absent, which may limit clinical trustworthiness.

## Related Work & Insights

- Compared to classical frequency-domain methods such as HARP and SinMod, InvTag does not require a predefined tag frequency and estimates it automatically.
- Compared to general-purpose diffusion inverse problem solvers such as DPS, InvTag handles the more challenging nonlinear and blind setting.
- The CDDP alternating optimization strategy is generalizable to other medical imaging inverse problems involving unknown forward models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to combine MR physics with a diffusion prior for nonlinear blind inverse problems.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 160 test cases + real data validation + comprehensive ablation, but cardiac data is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear and rigorous; physical modeling is complete.
- Value: ⭐⭐⭐⭐ — Elegant methodology, though runtime efficiency limits practical clinical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT](../../AAAI2026/medical_imaging/unsupervised_multi-parameter_inverse_solving_for_reducing_ring_artifacts_in_3d_x.md)
- [\[CVPR 2026\] cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold](cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and.md)
- [\[AAAI 2026\] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](../../AAAI2026/medical_imaging/pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)
- [\[CVPR 2026\] NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization](neuroseg_meets_dinov3_transferring_2d_self-supervised_visual_priors_to_3d_neuron.md)
- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)

</div>

<!-- RELATED:END -->

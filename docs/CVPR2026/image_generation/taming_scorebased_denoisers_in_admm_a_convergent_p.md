---
title: >-
  [Paper Note] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework
description: >-
  [CVPR 2026][Image Generation][ADMM] This paper proposes ADMM-PnP with an AC-DC denoiser, which integrates diffusion priors into the ADMM primal-dual framework via a three-stage correct-then-denoise procedure (Auto-Correction + Directional Correction + score-based denoising). The method addresses the geometric mismatch between ADMM iterates and the diffusion training manifold, establishes convergence guarantees under two sets of conditions, and consistently outperforms baselines such as DAPS, DPS, and DiffPIR across seven inverse problems.
tags:
  - CVPR 2026
  - Image Generation
  - ADMM
  - Plug-and-Play
  - Diffusion Model Prior
  - Convergence Guarantee
  - Inverse Problems
  - AC-DC Denoiser
date: 2026-05-08
content_hash: 735efb20d4b9ae1d
---

# Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework

**Conference**: CVPR 2026
**arXiv**: [2603.10281](https://arxiv.org/abs/2603.10281)
**Code**: Provided in supplementary material
**Area**: Inverse Problem Solving / Computational Imaging
**Keywords**: ADMM, Plug-and-Play, Diffusion Model Prior, Convergence Guarantee, Inverse Problems, AC-DC Denoiser

## TL;DR
This paper proposes ADMM-PnP with an AC-DC denoiser, which integrates diffusion priors into the ADMM primal-dual framework via a three-stage correct-then-denoise procedure (Auto-Correction + Directional Correction + score-based denoising). The method addresses the geometric mismatch between ADMM iterates and the diffusion training manifold, establishes convergence guarantees under two sets of conditions, and consistently outperforms baselines such as DAPS, DPS, and DiffPIR across seven inverse problems.

## Background & Motivation
Score-based diffusion models have become powerful priors for solving inverse problems. Existing Plug-and-Play (PnP) methods replace the proximal operator of traditional regularization terms with score functions, but face two core challenges: (1) **Manifold mismatch** — score functions are trained on noisy manifolds $\mathcal{M}_{\sigma(t)}$ perturbed by Gaussian noise, whereas the iterates $\mathbf{z}^{(k)}$ of optimization algorithms do not necessarily lie on these manifolds; in particular, the dual variable $\mathbf{u}^{(k)}$ in ADMM further distorts the noise geometry; (2) **Lack of convergence understanding** — the convergence of embedding score-based denoisers into ADMM has not been established, as existing analyses primarily cover primal algorithms rather than primal-dual methods.

## Core Problem
How to effectively embed score-based denoisers into the ADMM framework — simultaneously resolving the geometric mismatch between iterates and the training manifold (especially the additional mismatch induced by the dual variable) while providing theoretical convergence guarantees for this combination.

## Method

### Overall Architecture
The standard ADMM-PnP framework splits the inverse problem $\min_\mathbf{x} \ell(\mathbf{y}\|\mathcal{A}(\mathbf{x})) + \gamma h(\mathbf{z})$ via variable splitting into an $\mathbf{x}$-subproblem (data fidelity) and a $\mathbf{z}$-subproblem (denoising/regularization). The $\mathbf{z}$-subproblem replaces the proximal operator with the AC-DC denoiser. Each iteration proceeds as: (7a) solve the ML subproblem → (7b) apply AC-DC denoising → (7c) update the dual variable $\mathbf{u}$.

### Key Designs
1. **AC-DC Three-Stage Denoiser**:

    - **AC (Auto-Correction)**: Gaussian noise is added to the ADMM iterate $\tilde{\mathbf{z}}^{(k)}$, giving $\mathbf{z}_{ac}^{(k)} = \tilde{\mathbf{z}}^{(k)} + \sigma^{(k)}\mathbf{n}$, pulling it toward the neighborhood of the noisy training manifold $\mathcal{M}_{\sigma^{(k)}}$. However, noise injection alone does not guarantee manifold alignment.
    - **DC (Directional Correction)**: Conditional Langevin dynamics ($J$ steps) refine $\mathbf{z}_{ac}^{(k)}$ onto $\mathcal{M}_{\sigma^{(k)}}$. The gradient of the conditional distribution $p(\mathbf{z}_{\sigma^{(k)}}|\mathbf{z}_{ac}^{(k)})$ is approximated by combining the score function with a Gaussian likelihood term.
    - **Denoising**: The Tweedie formula is applied on the aligned manifold: $\mathbf{z}_{tw}^{(k)} = \mathbf{z}_{dc}^{(k)} + (\sigma^{(k)})^2 \mathbf{s}_\theta(\mathbf{z}_{dc}^{(k)}, \sigma^{(k)})$. An ODE-based sampler may alternatively be used (Ours-ode).

2. **Convergence Guarantee under Weakly Non-Expansive Operators (Theorems 1+2)**: Under the assumptions that $\ell$ is $\mu$-strongly convex and the AC-DC denoiser satisfies a weakly non-expansive condition ($\|R_\sigma(\tilde{\mathbf{z}}_1) - R_\sigma(\tilde{\mathbf{z}}_2)\|^2 \leq \epsilon^2\|\tilde{\mathbf{z}}_1-\tilde{\mathbf{z}}_2\|^2 + \delta^2$), a fixed step size $\rho$ guarantees convergence to a $\delta$-ball around a fixed point. It is further shown that AC-DC satisfies this assumption when $\log p_{data}$ is $M$-smooth and satisfies a coercivity condition.

3. **Adaptive Step-Size Convergence without Convexity (Theorem 3)**: Relaxing the strong convexity assumption, the $\rho$-increasing rule of Chan et al. is adopted. The AC-DC denoiser is shown to be a bounded denoiser under this setting, and convergence to a fixed point with high probability is established under an appropriate schedule with $\sigma^{(k)} \to 0$.

### Loss & Training
- $\sigma^{(k)}$ follows a linear decay schedule in the range $[0.1, 10]$ over a window of $W$ iterations
- DC steps: $J=10$, step size $\eta^{(k)} = 5\times10^{-4}\sigma^{(k)}$
- The ML subproblem (7a) is solved with the Adam optimizer, with up to 1000 steps and convergence detection

## Key Experimental Results
**FFHQ, 100 images (PSNR↑/SSIM↑):**

| Task | Ours-tweedie | DAPS | DPS | DiffPIR | DPIR |
|------|-------------|------|-----|---------|------|
| SR ×4 | **30.44/0.857** | 29.53/0.814 | 24.83/0.705 | 26.77/0.749 | 28.85/0.826 |
| Random Inpainting | **32.84/0.906** | 31.65/0.847 | 29.08/0.828 | 28.56/0.709 | - |
| Motion Deblurring | **30.00/0.854** | 29.05/0.815 | 23.26/0.663 | - | - |
| Gaussian Deblurring | **30.40/0.853** | 29.79/0.813 | 26.11/0.730 | 25.15/0.699 | 28.88/0.833 |
| Phase Retrieval | **27.94/0.793** | 26.71/0.749 | 11.63/0.366 | - | - |
| Box Inpainting | **24.03/0.859** | 23.64/0.815 | 23.49/0.817 | 20.93/0.561 | - |

The proposed method achieves the best or second-best results on nearly all tasks and metrics.

### Ablation Study
- **Removing AC-DC correction (direct denoising)**: PSNR drops substantially across all tasks — SR ×4: 26.92→30.44, phase retrieval: 11.98→27.94, box inpainting: 15.60→24.03. The correction stages are critical to performance gains.
- **Effect of DC steps $J$**: Setting $J=0$ (no DC) still yields severe artifacts in phase retrieval; increasing $J$ progressively improves image quality (Fig. 5 qualitative demonstration).
- **NFE efficiency**: Most tasks saturate within 10 iterations (110 NFE for Ours-tweedie); only challenging tasks such as phase retrieval and nonlinear deblurring require more NFEs.
- **Empirical validation of assumptions**: The Lipschitz ratio of the score function concentrates in the range 50–160, supporting the smoothness assumption; coercivity of the energy function is verified via the linear relationship between $\langle\mathbf{x}, -\nabla\log p(\mathbf{x})\rangle$ and $\|\mathbf{x}\|^2$.
- **Flexibility of additional regularization**: The ADMM framework allows incorporating LPIPS perceptual loss in the ML subproblem, enabling style-transfer inpainting.

## Highlights & Insights
- **Solid theoretical contributions**: Three theorems establish convergence guarantees under (i) weakly non-expansive + strong convexity, (ii) AC-DC-specific conditions, and (iii) no convexity + adaptive step size — constituting the first systematic theoretical analysis of score-based PnP in a primal-dual framework.
- **Principled AC-DC three-stage design**: AC pulls iterates into the manifold neighborhood → DC refines alignment → score-based denoising takes effect; each stage has a clear role and is theoretically interpretable.
- **ADMM flexibility**: The framework supports the incorporation of additional regularizers such as LPIPS — a capability unavailable to pure diffusion sampling methods.
- Consistent advantages across 7 inverse problems, including challenging tasks such as nonlinear deblurring and phase retrieval.

## Limitations & Future Work
- Convergence results are fixed-point convergence rather than stationary-point convergence, limiting theoretical strength — the authors acknowledge this as an open problem in the PnP community.
- Hyperparameters such as the noise schedule $\sigma^{(k)}$ and the number of DC steps $J$ rely on empirical heuristics, lacking theoretically optimal design principles.
- Each AC-DC denoising call requires 11 score function evaluations (Tweedie) or 20 (ODE), resulting in high computational cost.
- Validation is limited to FFHQ/ImageNet at 256×256 resolution; higher resolutions and other domains have not been evaluated.

## Related Work & Insights
- **DiffPIR**: Also a PnP framework using variable splitting rather than ADMM; employs only the AC step (noise injection + Tweedie) without DC refinement, yielding PSNR 3–4 dB lower than the proposed method across all tasks.
- **DAPS**: Uses decoupled noise annealing but remains a sampling rather than optimization method; PSNR is 1–2 dB lower, with no theoretical convergence guarantee.
- **DPS**: Modifies the MCMC process for conditional sampling; PSNR is substantially lower on challenging tasks (phase retrieval/motion deblurring: 11.63 vs. 27.94).
- **RED-diff**: Constructs an explicit regularization term and takes its gradient; generally yields the lowest PSNR among all baselines (16–20 dB range).
- **SNORE**: A primal method with similar noise injection, but without dual variable handling; the proposed work demonstrates that AC-DC can handle the more challenging primal-dual setting.

The AC-DC "correct-then-denoise" paradigm is a general framework for manifold alignment and can be extended to any PnP scenario employing pretrained models. The $\delta$-ball convergence theory for weakly non-expansive operators can be applied to analyze other optimization methods using noisy denoisers. The flexibility of ADMM in incorporating arbitrary regularization terms offers a unique advantage in practical inverse problems with multiple constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ The AC-DC three-stage design offers theoretical novelty; this is the first systematic convergence analysis for ADMM combined with score-based denoisers.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven inverse problems (+2 supplementary), two datasets, 6–8 baselines, NFE efficiency analysis, assumption validation, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and complete theoretical derivations (5 theorems, 6 lemmas); experiments and theory complement each other.
- Value: ⭐⭐⭐⭐⭐ Dual theoretical and practical contributions to the fields of computational imaging and inverse problem solving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_score_function_generalization_diffusion_models.md)
- [\[CVPR 2026\] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution](framer_frequency-aligned_self-distillation_with_adaptive_modulation_leveraging_d.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment](da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)
- [\[CVPR 2026\] Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models](taming_sampling_perturbations_with_variance_expansion_loss_for_latent_diffusion_.md)

</div>

<!-- RELATED:END -->

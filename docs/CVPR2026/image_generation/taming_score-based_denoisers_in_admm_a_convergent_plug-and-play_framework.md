---
title: >-
  [Paper Note] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework
description: >-
  [CVPR 2026][Image Generation][Plug-and-Play] This paper proposes AC-DC, a three-stage denoiser (Auto-Correction + Directional Correction + Score Denoising) that addresses the manifold mismatch between ADMM iterations and…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Plug-and-Play"
  - "ADMM"
  - "Diffusion Models"
  - "Score-based Denoiser"
  - "Inverse Problem Solving"
date: 2026-05-08
content_hash: 5fffa1e29faf7065
---

# Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework

**Conference**: CVPR 2026
**arXiv**: [2603.10281](https://arxiv.org/abs/2603.10281)  
**Code**: Available (provided with supplementary material)  
**Area**: Image Generation
**Keywords**: Plug-and-Play, ADMM, Diffusion Models, Score-based Denoiser, Inverse Problem Solving

## TL;DR

This paper proposes AC-DC, a three-stage denoiser (Auto-Correction + Directional Correction + Score Denoising) that addresses the manifold mismatch between ADMM iterations and the score training manifold. It provides the first convergence guarantee for ADMM-PnP combined with score-based denoisers, achieving state-of-the-art performance across multiple inverse problems.

## Background & Motivation

### 1. State of the Field

Inverse problems (deblurring, super-resolution, inpainting, phase retrieval, etc.) widely adopt the Plug-and-Play (PnP) paradigm, which embeds pretrained denoisers directly into optimization algorithms such as ADMM or HQS as proximal operators in place of regularization terms. In recent years, score functions derived from diffusion models have become a popular choice for PnP denoisers due to their powerful distribution modeling capability.

### 2. Limitations of Prior Work

- **Manifold mismatch**: Score functions are trained on the noisy data manifold $\mathcal{M}_{\sigma(t)}$ induced by Gaussian perturbations, whereas the intermediate variables produced by ADMM iterations, $\tilde{\bm{z}}^{(k)} = \bm{x}^{(k+1)} + \bm{u}^{(k)}$, do not lie on these manifolds. The dual variable $\bm{u}^{(k)}$ further distorts the noise geometry.
- **Lack of convergence theory**: Existing score-based PnP methods predominantly target primal algorithms (e.g., gradient descent); the convergence behavior of primal-dual methods such as ADMM combined with score-based denoisers remains entirely unknown.

### 3. Root Cause

The flexibility of ADMM—enabling it to handle multiple regularization terms and constraints—makes it highly attractive for inverse problems. However, the presence of dual variables causes the inputs to score-based denoisers to deviate further from the training manifold than in primal methods, and simple noise-injection (purification) strategies are insufficient to guarantee alignment.

### 4. Paper Goals

(1) Design a denoiser capable of pulling ADMM iterates back onto the score training manifold; (2) establish rigorous convergence guarantees for the resulting ADMM-PnP framework.

### 5. Starting Point

The approach departs from conditional Langevin dynamics sampling: a coarse correction via noise injection is applied first, followed by a refined correction via conditional Langevin steps to project the input onto $\mathcal{M}_{\sigma^{(k)}}$, after which Tweedie denoising is applied.

### 6. Core Idea

The three-stage AC-DC denoiser = Auto-Correction (noise injection for coarse manifold alignment) + Directional Correction (conditional Langevin for precise alignment) + Score Denoising (Tweedie/ODE denoising).

## Method

### Overall Architecture

The inverse problem is solved under the ADMM variable-splitting framework:

$$\min_{\bm{x},\bm{z}} \ell(\bm{y} \| \mathcal{A}(\bm{x})) + \gamma h(\bm{z}) \quad \text{s.t.} \quad \bm{x} = \bm{z}$$

ADMM three-step iteration:
1. **x-update**: Solve the data-fidelity subproblem via gradient descent with the Adam optimizer (up to 1000 steps).
2. **z-update**: Denoising subproblem → replaced by the AC-DC denoiser as the proximal operator.
3. **u-update**: Dual variable update $\bm{u}^{(k+1)} = \bm{u}^{(k)} + (\bm{x}^{(k+1)} - \bm{z}^{(k+1)})$.

The core contribution lies in the AC-DC denoiser within the z-update.

### Key Designs

#### Design 1: Auto-Correction (AC) — Coarse Correction via Gaussian Noise Injection

- **Function**: Adds Gaussian noise to the ADMM iterate $\tilde{\bm{z}}^{(k)}$.
- **Mechanism**: $\bm{z}_{\text{ac}}^{(k)} = \tilde{\bm{z}}^{(k)} + \sigma^{(k)} \bm{n}$, where $\bm{n} \sim \mathcal{N}(\bm{0}, \bm{I})$.
- **Design Motivation**: The noise distribution of ADMM iterates is unknown and is influenced by the dual variable. Adding Gaussian noise effectively "submerges" the iterate in Gaussian noise, driving it toward some $\mathcal{M}_{\sigma(t)}$. However, AC alone is insufficient to guarantee manifold alignment.

#### Design 2: Directional Correction (DC) — Fine Correction via Conditional Langevin Dynamics

- **Function**: Starting from $\bm{z}_{\text{ac}}^{(k)}$, runs $J$ steps of conditional Langevin dynamics.
- **Mechanism**: The target distribution is $p(\bm{z}_{\sigma^{(k)}} | \bm{z}_{\text{ac}}^{(k)})$, whose support is contained in $\mathcal{M}_{\sigma^{(k)}}$, so samples naturally lie on the score training manifold. The conditional score is decomposed into the unconditional score $\bm{s}_\theta$ plus an approximate Gaussian likelihood gradient.
- **Design Motivation**: AC provides only coarse alignment; DC exploits the score function itself for directional refinement, precisely aligning the iterate to $\mathcal{M}_{\sigma^{(k)}}$ while preserving measurement information. Each update step:

$$\bm{w}^{(k,j+1)} = \bm{w}^{(k,j)} + \eta^{(k)}\left(\frac{1}{\sigma_{\bm{s}^{(k)}}^2}(\bm{z}_{\text{ac}}^{(k)} - \bm{w}^{(k,j)}) + \bm{s}_\theta(\bm{w}^{(k,j)}, \sigma^{(k)})\right) + \sqrt{2\eta^{(k)}}\bm{n}$$

#### Design 3: Score-based Denoising — Final Denoising

- **Function**: Applies Tweedie's formula or an ODE solver to denoise the DC output.
- **Mechanism**: Tweedie: $\bm{z}_{\text{tw}}^{(k)} = \bm{z}_{\text{dc}}^{(k)} + (\sigma^{(k)})^2 \bm{s}_\theta(\bm{z}_{\text{dc}}^{(k)}, \sigma^{(k)})$; ODE: integrate from $\sigma^{(k)}$ to $0$.
- **Design Motivation**: After AC and DC, the input already lies on $\mathcal{M}_{\sigma^{(k)}}$, where the score function achieves its best denoising performance. The paper provides two variants: Tweedie (fast/single-step) and ODE (slow/multi-step/higher fidelity).

### Loss & Training

- **No training required**: Uses a pretrained score model directly (diffusion model from Chung et al., 2023).
- **Noise schedule**: $\sigma^{(k)}$ decays linearly from $10$ to $0.1$ over a window of $W$ steps; DC steps $J=10$; Langevin step size $\eta^{(k)} = 5 \times 10^{-4} \sigma^{(k)}$.
- **x-subproblem**: Adam optimizer, up to 1000 steps, with early stopping if the loss increases for 3 consecutive steps.

## Key Experimental Results

### Main Results

Evaluated on FFHQ 256×256 and ImageNet 256×256, with 100 randomly sampled images per task.

**Table 1: Performance Comparison on FFHQ**

| Task | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------|-------|--------|
| Super-Resolution (4×) | **Ours-tweedie** | **30.44** | **0.857** | 0.178 |
| | Ours-ode | 29.99 | 0.845 | **0.156** |
| | DAPS | 29.53 | 0.814 | 0.167 |
| | DPS | 24.83 | 0.705 | 0.257 |
| Random Inpainting | **Ours-tweedie** | **32.84** | **0.906** | 0.122 |
| | Ours-ode | 32.13 | 0.894 | **0.095** |
| | DAPS | 31.65 | 0.847 | 0.124 |
| Motion Deblurring | **Ours-tweedie** | **30.00** | **0.854** | 0.179 |
| | Ours-ode | 29.65 | 0.841 | **0.154** |
| | DAPS | 29.05 | 0.815 | 0.175 |
| Phase Retrieval | **Ours-tweedie** | **27.94** | **0.793** | 0.209 |
| | Ours-ode | 27.10 | 0.757 | 0.237 |
| | DAPS | 26.71 | 0.749 | 0.230 |

**Table 2: Performance Comparison on ImageNet**

| Task | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------|-------|--------|
| Super-Resolution (4×) | **Ours-tweedie** | **27.32** | **0.717** | 0.280 |
| | DAPS | 26.65 | 0.680 | **0.266** |
| Random Inpainting | **Ours-tweedie** | **29.56** | **0.817** | 0.184 |
| | Ours-ode | 28.73 | 0.795 | **0.148** |
| Gaussian Deblurring | **Ours-tweedie** | **27.20** | **0.705** | 0.281 |
| | Ours-ode | 26.90 | 0.690 | 0.282 |
| | DAPS | 26.89 | 0.678 | **0.260** |

### Ablation Study

**Effect of DC steps $J$** (phase retrieval task):

- $J=0$ (DC disabled): Reconstructed images exhibit severe artifacts.
- Increasing $J$: Images progressively become cleaner.
- This confirms that the DC stage is the critical component for bridging the manifold mismatch.

### Key Findings

1. **Ours-tweedie consistently achieves the best PSNR/SSIM**: It ranks first in PSNR and SSIM on nearly all tasks; Ours-ode tends to achieve the best perceptual metric LPIPS.
2. **Substantial improvement over PnP baselines**: PSNR gains of 2–10 dB over comparable PnP methods such as DiffPIR and RED-diff.
3. **Pronounced gains on challenging tasks**: For the highly non-convex phase retrieval task, Ours-tweedie outperforms DPS by more than 16 dB on FFHQ.
4. **Complementary variants**: The Tweedie variant is faster with better pixel-level metrics; the ODE variant yields superior perceptual quality.

## Highlights & Insights

1. **Precise problem formulation**: The paper explicitly identifies the underappreciated issue that ADMM dual variables exacerbate manifold mismatch, which is the fundamental reason score-based denoisers are rarely combined with primal-dual methods.
2. **Elegant three-stage AC-DC design**: Coarse correction (AC) → fine correction (DC) → denoising follows a logically progressive structure, with clear geometric intuition at each step.
3. **Solid theoretical contributions**:
    - Theorem 1: Under weak non-expansiveness, constant step size converges to a $\delta$-ball.
    - Theorem 2: Proves that the AC-DC denoiser satisfies weak non-expansiveness.
    - Theorem 3: Removes the strong convexity assumption; adaptive step sizes also converge.
4. **Strong generality**: The AC-DC denoiser is not restricted to ADMM and can be embedded into any proximal operator-based optimization framework.

## Limitations & Future Work

1. **High computational cost**: Each ADMM iteration requires multiple score function evaluations (1 for AC + $J$ for DC + Tweedie/ODE), resulting in a high NFE count.
2. **Empirically determined noise schedule**: The schedules for $\sigma^{(k)}$ and $\sigma_{\bm{s}^{(k)}}$ are manually set as linear decay, lacking an adaptive mechanism.
3. **Convergence under constant step size unproven for non-convex objectives**: Experiments show constant step sizes work on non-convex objectives, but the theory only proves convergence for adaptive step sizes.
4. **Weak fixed-point convergence**: Convergence to a fixed point (not a stationary point) is guaranteed, with no quality assurance on the solution.
5. **Idealized DC assumption**: The theory assumes DC reaches its stationary distribution, whereas in practice only a finite number of steps are executed.

## Related Work & Insights

- **DiffPIR (Zhu et al., 2023)**: Also a PnP+score framework, but uses HQS instead of ADMM and corrects the manifold mismatch only through noise injection (equivalent to AC without DC), yielding noticeably inferior results.
- **SNORE (Renaud et al., 2024)**: Constructs an explicit regularizer whose gradient corresponds to the score, following a gradient descent approach that is theoretically cleaner but less flexible than ADMM.
- **DAPS (Zhang et al., 2024)**: A posterior sampling method that occasionally approaches the proposed method on LPIPS, but systematically lags behind on PSNR/SSIM.
- **Ryu et al. (2019)**: Classical convergence analysis for ADMM-PnP, requiring strict contractiveness of the denoiser residual; this paper relaxes the condition to "weak non-expansiveness + $\delta$-ball convergence."
- **Insight**: The idea of using conditional Langevin dynamics for manifold alignment can be generalized to other scenarios requiring score function evaluation, such as conditional generation and image editing.

## Rating

⭐⭐⭐⭐ A rigorous and well-rounded contribution with both strong theory and comprehensive experiments. The AC-DC three-stage denoiser is elegantly designed with clear geometric intuition, and the convergence analysis represents a substantive advance in PnP theory. Experiments cover 7 types of inverse problems and consistently outperform baselines. The primary drawbacks are the high computational cost and the lack of an adaptive noise scheduling strategy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](../../ICCV2025/image_generation/trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_the_score_function_for_generalization_in_diffusion_models.md)
- [\[ICLR 2026\] RNE: plug-and-play diffusion inference-time control and energy-based training](../../ICLR2026/image_generation/rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)
- [\[CVPR 2026\] Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)
- [\[ICML 2026\] DiScoFormer: Plug-In Density and Score Estimation with Transformers](../../ICML2026/image_generation/discoformer_plug-in_density_and_score_estimation_with_transformers.md)

</div>

<!-- RELATED:END -->

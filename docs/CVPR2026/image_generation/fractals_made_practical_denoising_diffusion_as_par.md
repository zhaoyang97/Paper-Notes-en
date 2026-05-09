---
title: >-
  [Paper Note] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems
description: >-
  [CVPR 2026][Image Generation][Fractal Geometry] This paper proves that the DDIM deterministic reverse chain is equivalent to a Partitioned Iterated Function System (PIFS), and derives three computable quantities from fractal geometry—the contraction threshold $L_t^*$, the diagonal expansion function $f_t(\lambda)$, and the global expansion threshold $\lambda^{**}$—providing a unified theoretical explanation for four empirically motivated design choices: cosine schedule offset, resolution logSNR shift, Min-SNR loss weighting, and Align Your Steps sampling schedule.
tags:
  - CVPR 2026
  - Image Generation
  - Fractal Geometry
  - Diffusion Models
  - Partitioned Iterated Function Systems
  - DDIM
  - Noise Schedule
  - Fractal Dimension
date: 2026-05-08
content_hash: f44720eee4b05b60
---

# Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems

**Conference**: CVPR 2026
**arXiv**: [2603.13069](https://arxiv.org/abs/2603.13069)
**Code**: To be confirmed
**Area**: Diffusion Model Theory / Image Generation / Fractal Geometry
**Keywords**: Fractal Geometry, Diffusion Models, Partitioned Iterated Function Systems, DDIM, Noise Schedule, Fractal Dimension

## TL;DR
This paper proves that the DDIM deterministic reverse chain is equivalent to a Partitioned Iterated Function System (PIFS), and derives three computable quantities from fractal geometry—the contraction threshold $L_t^*$, the diagonal expansion function $f_t(\lambda)$, and the global expansion threshold $\lambda^{**}$—providing a unified theoretical explanation for four empirically motivated design choices: cosine schedule offset, resolution logSNR shift, Min-SNR loss weighting, and Align Your Steps sampling schedule.

## Background & Motivation
Modern score-matching diffusion models construct high-quality images through sequential denoising processes, grounded theoretically in continuous-time SDEs or probability-flow ODEs. However, this continuous perspective treats the learned score network as a black box, offering no structural explanation of how the discrete sampling chain assembles global spatial context in early steps and synthesizes local details in later steps, nor why self-attention serves as such an effective generative primitive. Existing scheduling choices (cosine offset $s_{off}=0.008$), Min-SNR weighting ($\gamma=5$), and similar decisions are based on empirical tuning and lack a unified theoretical justification.

## Core Problem
What does the diffusion model actually do when transforming noise into images? What is the mathematical structure of the DDIM reverse chain? Can a unified design language be derived from this structure to explain and guide the selection of noise schedules, training objectives, and sampling strategies?

## Method

### Overall Architecture
The DDIM single-step operator $\Phi_t(x) = \sqrt{\bar\alpha_{t-1}/\bar\alpha_t}\,x + b_t\,\hat\varepsilon_\theta(x,t)$ is treated as an affine map within a PIFS. The spectral structure of its Jacobian is analyzed to derive contraction/expansion properties, and the Moran equation is applied to compute the attractor dimension.

### Key Designs
1. **Contraction Structure Analysis (Section 3)**: Two contraction conditions are derived—Euclidean Contraction (EC), based on a closed-form contraction threshold $L_t^* = (\sqrt{\bar\alpha_{t-1}/\bar\alpha_t}-1)/|b_t|$ (depending solely on the noise schedule, not on data or the network), and Block Max-Norm Contraction (PC), which separates the diagonal term $\kappa_t^{diag}$ from the cross-patch coupling $\delta_t^{cross}$. Score-matching training is shown to be a diffusion analogue of Barnsley collage minimization, establishing an $L_2$-$\mathcal{W}_1$ bridge.
2. **Two-Phase Structure (Section 4)**: In high-noise Regime I (global context assembly), attention is broadcast and a directional "suppression field" $S_{k,t}$ keeps each diagonal block below the expansion threshold. In low-noise Regime II (detail synthesis), attention localizes and suppression is released patch-by-patch in ascending order of patch variance (Stratified Crossover), with the sign of the Spearman rank correlation reversing and then recovering.
3. **Attractor Geometry (Section 5)**: The Kaplan-Yorke dimension is defined via the Lyapunov spectrum. The global expansion threshold $\lambda^{**}$ is the unique solution to the discrete Moran equation $\prod_t f_t(\lambda^{**})=1$. Directions whose patch variance exceeds $\lambda^{**}$ contribute positive Lyapunov exponents. The suppression-field-corrected effective dimension satisfies $d_{KY}^{eff} \leq d_{KY}$.
4. **Three Design Criteria (Section 6)**: (i) Maximize the weakest-link contraction threshold $\min_t L_t^*$; (ii) balance the per-step Lyapunov contribution (equivalent to a constant information-gain condition); (iii) concentrate sampling steps where $L_t^*$ is smallest to equalize contraction workload.

### Loss & Training
The standard DSM loss is shown to be equivalent to the collage error weighted by SNR across timesteps. A PIFS regularization loss is proposed: $\mathcal{L}_{PIFS} = \mathcal{L}(\theta) + \mu_{reg}\sum_{t,k,j\neq k}\|[J_x\hat\varepsilon_\theta]_{kj}\|_F^2$, which directly penalizes cross-patch Jacobian blocks to enhance the (PC) contraction margin. Gradients are computable via $O(M^2 n_k)$ JVP/VJP operations.

## Key Experimental Results

Experiments are conducted on DDPM CIFAR-10 (32×32, 8×8 patches, $M$=16) and CelebA-HQ-256 (64×64, 16×16 patches):

| Prediction | Measured Result | Note |
|------|------|------|
| Regime I diagonal blocks near neutral | $\kappa_t^{diag}\approx 1.000$ (range 0.993–1.007) | Maintained by suppression field |
| Regime II diagonal blocks expand | $\kappa_t^{diag}$ rises from 1.007 to 1.211 | Released in low-noise regime |
| (PC) crossover window | $\sim$40 steps ($t$∈[160,200]) | Violations drop from 87.5% to 0% |
| (MM) Spearman correlation | Peak $\rho$=0.953 ($p$=1.2e-8) at $t$=601 | Suppression increases with patch variance |
| Score deviation $O(\bar\alpha_t)$ | OLS slope 0.95, CI [0.88,1.02], $R^2$=0.994 | Consistent with theoretical prediction |
| CelebA $d_{KY}^{eff}=0$ | $\Lambda_{eff,k}<0$ for all 16 patches | 16/16 sign agreement |

The cosine schedule information-gain coefficient of variation is 0.867 vs. 1.107 for linear; the $L_t^*$ coefficient of variation is 0.341 for linear vs. 0.474 for cosine—indicating the two criteria cannot be simultaneously optimized by a single schedule.

### Ablation Study
- PIFS Moran threshold on the full CIFAR-10 training chain: $\lambda^{**}$=1.0024; all 16 patches have $\lambda_k\in[18.7,44.4]$, far exceeding the threshold, yielding $d_{KY}=n=3072$ (saturated).
- On CelebA-HQ, the strong suppression field yields $\lambda^{***}=500 \gg \lambda_{max}=231.7$, giving $d_{KY}^{eff}=0$—reflecting this model's low-diversity, high-fidelity characteristics.
- The 50-step DDIM sub-sampled chain preserves the same two-phase structure and $L_t^*$ distribution as the 1000-step chain (step-count invariance).
- The cosine offset $s_{off}=0.008$ raises $L_1^*$ from 7.9e-4 to 3.2e-3 (a 4× improvement), precisely corresponding to Criterion 1.

## Highlights & Insights
- The paper offers a novel and highly original theoretical perspective on diffusion models through the lens of fractal geometry.
- The three computable quantities $L_t^*$, $f_t(\lambda)$, and $\lambda^{**}$ fully characterize the denoising dynamics without requiring model evaluation.
- Four independently discovered empirical design choices are unified within a single mathematical framework, yielding strong explanatory power.
- The paper presents 27 theorems/corollaries forming a complete theoretical system, supported by thorough experimental validation.

## Limitations & Future Work
- The theoretical analysis is restricted to the deterministic DDIM case; stochastic DDPM samplers and Flow Matching are only briefly extended in the appendix.
- The block-diagonal covariance assumption under Gaussian data holds only approximately for real data.
- Experimental validation uses relatively simple DDPM models (CIFAR-10/CelebA) and has not been verified on modern architectures such as DiT.
- Quantitative improvements from the PIFS regularizer $\mathcal{L}_{PIFS}$ are not reported with comparative numbers.

## Related Work & Insights
- **Raya & Ambrogioni (NeurIPS 2023)**: Reports spontaneous symmetry breaking in diffusion models; this paper provides a structural proof of the two-phase structure from a PIFS perspective.
- **Nichol & Dhariwal (ICML 2021)**: Empirically proposes the cosine schedule offset $s_{off}=0.008$; this paper proves it is an approximate solution to maximizing $\min_t L_t^*$.
- **Min-SNR (ICCV 2023)**: Proposes loss weighting from a multi-task learning perspective; this paper proves it is equivalent to balancing the Kaplan-Yorke dimension growth rate.
- **Align Your Steps (ICML 2024)**: Optimizes sampling step allocation from a KL divergence perspective; this paper independently derives the same allocation density from a contraction workload perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (PIFS–diffusion model connection, three computable quantities, unification of four empirical design choices)
- Experimental Thoroughness: ⭐⭐⭐⭐ (systematic experimental validation of theoretical predictions across 7 subsections)
- Writing Quality: ⭐⭐⭐⭐ (rigorous mathematical derivations paired with clear intuitive explanations)
- Value: ⭐⭐⭐⭐ (provides a first-principles theoretical framework for diffusion model design)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_score_function_generalization_diffusion_models.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[CVPR 2026\] Diffusion Mental Averages](diffusion_mental_averages.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)

<!-- RELATED:END -->

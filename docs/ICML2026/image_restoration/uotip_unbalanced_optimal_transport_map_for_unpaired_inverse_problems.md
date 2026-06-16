---
title: >-
  [Paper Note] UOTIP：无须配对的反演问题的非平衡最优传输映射
description: >-
  [ICML 2026][Image Restoration][Paper Note] The UOTIP method is proposed—formulating the unpaired image inversion problem as mapping learning from a noisy measurement distribution to a clean signal distribution through an Unbalanced Optimal Transport (UOT) framework, achieving robustness and theoretical guarantees by introducing a likelihood cost function and a
tags:
  - ICML 2026
  - Image Restoration
date: 2026-05-08
content_hash: f736640ffb449558
---
# UOTIP: Unbalanced Optimal Transport Mapping for Unpaired Inversion Problems

**Conference**: ICML 2026  
**arXiv**: [2605.21094](https://arxiv.org/abs/2605.21094)  
**Code**: To be confirmed  
**Area**: Image Restoration / Optimal Transport  
**Keywords**: Unbalanced Optimal Transport, Unpaired Learning, Inverse Problems, Image Deblurring, Super-Resolution

## TL;DR
The UOTIP method is proposed—formulating the unpaired image inversion problem as mapping learning from a noisy measurement distribution to a clean signal distribution through an Unbalanced Optimal Transport (UOT) framework, achieving robustness and theoretical guarantees by introducing a likelihood cost function and a quadratic cost term.

## Background & Motivation

**Background**: Image inversion problems (deblurring, super-resolution, HDR reconstruction, etc.) are typically addressed using a Bayesian framework, solved by learning prior distributions and likelihood functions. Recently, generative models (GANs, VAEs, Diffusion Models, Optimal Transport) have been employed to learn data priors.

**Limitations of Prior Work**: Standard Optimal Transport (OT) methods face three primary issues in unpaired settings: (1) the requirement for strict marginal distribution matching, which is sensitive to data imbalance; (2) the inability to adaptively handle scenarios with multi-level noise; (3) the ill-posed nature of linear inversion problems, which may lead to the non-existence or non-uniqueness of OT maps.

**Key Challenge**: Image inversion must simultaneously satisfy two objectives: ensuring reconstruction results align with the clean signal distribution (prior fidelity) and maximizing the likelihood for specific measurements (data fidelity). While the standard OT framework can guarantee the former, it requires a meticulously designed cost function to integrate the latter; furthermore, the non-convexity and ill-posedness of the Monge problem threaten theoretical existence.

**Goal**: To design an unpaired inversion solver that possesses both theoretical guarantees and practical robustness.

**Key Insight**: Unbalanced Optimal Transport (UOT) naturally handles practical issues such as distribution mismatch and class imbalance by relaxing marginal constraints. Key insight: design the inversion problem's cost function as a combination of a likelihood cost and a quadratic cost, where the likelihood term encodes the data fidelity of MAP estimation, and the quadratic term satisfies the twist condition to ensure the existence and uniqueness of the mapping.

**Core Idea**: Formulate the inversion problem as learning a mapping between distributions using UOT, achieving global unpaired MAP estimation through a hybrid cost function.

## Method

### Overall Architecture

UOTIP addresses unpaired image inversion: given a set of noisy measurements $\mathbf{y} = \mathcal{A}(\mathbf{x}) + \mathbf{n}$ and a set of non-corresponding clean signals, the goal is to recover $\mathbf{x}$. This is reformulated as learning a mapping $T: \mathcal{Y} \to \mathcal{X}$ that aligns the pushed-forward measurement distribution with the clean distribution ($T_\# \mu \approx \nu$), with UOT relaxing this alignment constraint. The core design involves the cost function—the likelihood term handles "fidelity to the current measurement," and the quadratic term ensures "theoretical mapping existence and uniqueness." These are combined and solved via semi-dual adversarial optimization.

### Key Designs

**1. Likelihood Cost Function: Embedding MAP Estimation into OT Cost**

Standard OT quadratic costs only consider pixel distances and lack knowledge of the physical structure of the inversion problem (degradation operator $\mathcal{A}$, noise model). UOTIP leverages the structure of the MAP estimate $\mathbf{x}_{MAP}(\mathbf{y}_0) = \arg\min_{\mathbf{x}}[-\log p(\mathbf{y}_0|\mathbf{x})-\log p(\mathbf{x})]$: under the Gaussian noise assumption, the negative log-likelihood $-\log p(\mathbf{y}|\mathbf{x})$ is precisely $\|\mathcal{A}(\mathbf{x})-\mathbf{y}\|_2^2$. This is directly used as the OT cost $c_l(\mathbf{y},\mathbf{x})=\|\mathcal{A}(\mathbf{x})-\mathbf{y}\|_2^2$. Consequently, minimizing the OT transport cost is equivalent to minimizing the negative log-posterior $-\log p(\mathbf{x}|\mathbf{y})$, while the OT marginal constraint $T_\#\mu=\nu$ automatically incorporates the clean signal prior $p(\mathbf{x})$. The data term is managed by the likelihood cost, and the prior term is managed by distribution alignment, strictly transplanting the MAP estimation into the OT framework.

**2. Quadratic Cost + Unbalanced Relaxation: Restoring Existence and Handling Imbalance**

Likelihood cost alone presents issues: linear inversion is inherently ill-posed, and when $\mathcal{A}$ is non-invertible, the composite cost may not satisfy the twist condition, meaning the OT map is not guaranteed to exist or be unique. UOTIP adds a quadratic term $c_q(\mathbf{y},\mathbf{x})=\|\mathbf{y}-\mathbf{x}\|_2^2$ to form the final cost $c(\mathbf{y},\mathbf{x}) = \tau(c_l + c_q)$. Prop. 3.1 in the paper proves that this quadratic term is sufficient for the composite cost to satisfy the twist condition, restoring the existence and uniqueness of the map. Additionally, UOT replaces hard marginals with f-divergence soft constraints $D_{\Psi_i}(\pi_0\|\mu) + D_{\Psi_i}(\pi_1\|\nu)$. As marginals do not need to match strictly, the model can automatically reweight source samples via scaling factors $\Psi^*_i'$, allowing robust handling of multi-level noise, class imbalance, and heterogeneous distributions.

**3. Semi-dual UOT Solver: Approximating the Optimal Map via Adversarial Iteration**

Directly solving the Monge form is non-convex and difficult to optimize. UOTIP expands the Kantorovich dual of UOT into a semi-dual form, introducing the c-transform $v^c(\mathbf{y})=\inf_{\mathbf{x}}[c(\mathbf{y},\mathbf{x})-v(\mathbf{x})]$ for convexification. It is then parameterized with two networks: a potential function $v_\phi$ acting as the discriminator and a mapping $T_\theta$ acting as the generator, where $T_\theta(\mathbf{y}) \in \arg\inf_{\mathbf{x}}[c(\mathbf{y},\mathbf{x})-v_\phi(\mathbf{x})]$ explicitly satisfies the optimality condition of the c-transform. Through adversarial iteration, $T_\theta$ converges to the optimal UOT mapping.

## Key Experimental Results

### Main Results

| Task | Method | FFHQ PSNR | FFHQ SSIM | FFHQ FID | AFHQ PSNR | AFHQ SSIM | AFHQ FID |
|------|------|-----------|-----------|----------|-----------|-----------|----------|
| Gaussian Deblur | NOT | 20.11 | 0.6035 | 52.901 | 19.99 | 0.5472 | 58.927 |
| | OTUR | 23.82 | 0.7106 | 24.337 | 23.91 | 0.6777 | 30.773 |
| | RCOT | 22.07 | 0.5492 | 123.692 | 22.34 | 0.5365 | 132.465 |
| | **Ours** | **24.06** | **0.7139** | **21.210** | **24.22** | **0.6804** | **12.566** |
| 4× SR | NOT | 20.13 | 0.6257 | 50.066 | 20.14 | 0.5833 | 44.252 |
| | OTUR | 24.09 | 0.7243 | 22.751 | 24.71 | 0.7079 | 19.575 |
| | RCOT | 24.05 | 0.6820 | 118.776 | 25.04 | 0.7137 | 69.072 |
| | **Ours** | **24.35** | **0.7371** | **19.475** | **24.97** | **0.7142** | **15.939** |

### Ablation Study

| Configuration | Gaussian Deblur | HDR Recon | Non-linear Deblur | Description |
|------|----------------|--------|-------------|------|
| Full Model (UOT+$c_l$+$c_q$) | 24.06 | 26.02 | 28.52 | Baseline |
| w/o Quadratic Cost | 22.18 | 24.31 | 26.74 | Ill-posedness worsens without $c_q$ |
| w/o Likelihood Cost | 23.41 | 25.18 | 27.65 | Data fidelity drops without $c_l$ |
| Standard OT (Hard Marginal) | 23.12 | 25.04 | 27.31 | UOT relaxation outperforms strict constraints |

### Key Findings
- **Multi-level Noise Handling**: By training a single model on 3 different noise levels, UOTIP maintains stable performance (PSNR fluctuation <0.5dB).
- **Class Imbalance**: The adaptive marginal matching of UOT allows it to outperform standard OT across various class distribution ratios.
- **Super-Resolution Success**: Although SR breaks the theoretical guarantees of the twist condition, the network's inductive bias allows UOTIP to achieve SOTA results.
- **Texture Preservation**: Qualitative comparisons show that UOTIP preserves details better than OTUR (which over-smooths) and RCOT (which produces artifacts).

## Highlights & Insights
- **Ingenious MAP-OT Bridging**: The likelihood cost function naturally corresponds to the negative log-likelihood of Gaussian noise, implicitly introducing data priors through OT constraints—marking the first time MAP estimation is strictly mapped to the OT framework.
- **Practical Strategies with Theoretical Guarantees**: Proposition 3.1 proves that the quadratic cost $\lambda c_q$ ensures the twist condition, providing unique mappings even for ill-posed problems.
- **Significant Advantages of UOT**: Naturally addresses three types of practical difficulties—multi-level noise, data imbalance, and distribution heterogeneity—by allowing soft marginal matching.

## Limitations & Future Work
- In SR tasks, the modified quadratic cost does not satisfy the theoretical twist condition, leaving existence/uniqueness unguaranteed.
- The cost intensity parameter $\tau$ requires manual tuning.
- Computational costs are higher compared to GANs (requires potential function optimization).
- Future Work: Extending the twist condition applicability to non-L-Lipschitz operators; designing an adaptive $\tau$; integrating diffusion models as a prior supplement.

## Related Work & Insights
- **vs NOT**: Both use a neural OT framework, but NOT uses a standard quadratic cost and lacks specificity for inversion problems; UOTIP introduces a likelihood cost to encode the data term more precisely.
- **vs OTUR**: GAN-based methods tend to over-smooth or distort texture details; UOTIP is more robust through the global optimality of OT and the distribution relaxation of UOT.
- **vs Standard Bayesian Inversion**: Traditional methods rely on handcrafted priors; UOTIP uses data-driven learning for distribution priors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic combination of UOT with inversion problems, introducing likelihood cost and twist condition theoretical guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 inversion tasks across multiple datasets, including ablation and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression starting from MAP principles to gradual introduction of the method.
- Value: ⭐⭐⭐⭐⭐ Unifies the fields of inverse problems and optimal transport, offering both theoretical and applied value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Normalized Energy Models for Linear Inverse Problems](learning_normalized_energy_models_for_linear_inverse_problems.md)
- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[ICML 2026\] Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)
- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] UOTIP: Unbalanced Optimal Transport Mapping for Unpaired Inverse Problems
description: >-
  [ICML 2026][Image Restoration][Unbalanced Optimal Transport] The UOTIP method is proposed, which formulates the unpaired image inverse problem as learning a mapping from the noisy measurement distribution to the clean si…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Unbalanced Optimal Transport"
  - "Unpaired Learning"
  - "Inverse Problems"
  - "Image Deblurring"
  - "Super-Resolution"
date: 2026-05-08
content_hash: 4d55e36a37c803f2
---

# UOTIP: Unbalanced Optimal Transport Mapping for Unpaired Inverse Problems

**Conference**: ICML 2026  
**arXiv**: [2605.21094](https://arxiv.org/abs/2605.21094)  
**Code**: To be confirmed  
**Area**: Image Restoration / Optimal Transport  
**Keywords**: Unbalanced Optimal Transport, Unpaired Learning, Inverse Problems, Image Deblurring, Super-Resolution

## TL;DR
The UOTIP method is proposed, which formulates the unpaired image inverse problem as learning a mapping from the noisy measurement distribution to the clean signal distribution via an Unbalanced Optimal Transport (UOT) framework, achieving robustness and theoretical guarantees by introducing a likelihood cost function and a quadratic cost term.

## Background & Motivation

**Background**: Image inverse problems (deblurring, super-resolution, HDR reconstruction, etc.) are typically solved using a Bayesian framework by learning prior distributions and likelihood functions. Recently, generative models (GANs, VAEs, diffusion models, optimal transport) have been employed to learn data priors.

**Limitations of Prior Work**: Standard Optimal Transport (OT) methods face three major issues in the unpaired setting: (1) requirement for strict marginal distribution matching, making them sensitive to data imbalance; (2) inability to adaptively handle multi-level noise scenarios; (3) ill-posed characteristics in linear inverse problems may lead to the non-existence or non-uniqueness of the OT map.

**Key Challenge**: Image inversion must simultaneously satisfy two objectives: ensuring the reconstruction results conform to the clean signal distribution (prior fidelity) and maximizing the likelihood for specific measurements (data fidelity). While standard OT frameworks can guarantee the former, they require carefully designed cost functions to integrate the latter; furthermore, the non-convexity and ill-posedness of the Monge problem threaten theoretical existence.

**Goal**: To design an unpaired inverse problem solver that possesses both theoretical guarantees and practical robustness.

**Key Insight**: Unbalanced Optimal Transport (UOT) naturally handles practical issues such as distribution mismatch and class imbalance by relaxing marginal constraints. Key insight: The cost function of the inverse problem is designed as a combination of a likelihood cost and a quadratic cost, where the likelihood term encodes the data fidelity of MAP estimation, and the quadratic term satisfies the twist condition to ensure the existence and uniqueness of the mapping.

**Core Idea**: Use UOT to formulate the inverse problem as learning a mapping between distributions, achieving global unpaired MAP estimation through a hybrid cost function.

## Method

### Overall Architecture
Three stages: (1) **Problem Reformulation**: The unpaired inverse problem $\mathbf{y} = \mathcal{A}(\mathbf{x}) + \mathbf{n}$ is rewritten as learning a UOT map $T: \mathcal{Y} \to \mathcal{X}$ such that $T_\# \mu \approx \nu$; (2) **Cost Function Design**: A likelihood cost $c_l(\mathbf{y},\mathbf{x})=\|\mathcal{A}(\mathbf{x})-\mathbf{y}\|_2^2$ is combined with a quadratic cost $c_q(\mathbf{y},\mathbf{x})=\|\mathbf{y}-\mathbf{x}\|_2^2$; (3) **Neural Network Solution**: The problem is solved using the semi-dual UOT formulation with parameterized potential functions, learning the optimal mapping through adversarial optimization.

### Key Designs

1.  **Likelihood Cost Function**:
    - **Function**: Encodes the data fidelity term under the assumption of Gaussian measurement noise, causing the OT map to implicitly maximize the observation likelihood.
    - **Mechanism**: Utilizing the structure of the MAP estimate $\mathbf{x}_{MAP}(\mathbf{y}_0) = \arg\min_{\mathbf{x}}[-\log p(\mathbf{y}_0|\mathbf{x})-\log p(\mathbf{x})]$, $-\log p(\mathbf{y}|\mathbf{x})$ is used as the OT cost. Minimizing the OT transport cost is equivalent to minimizing the negative log-posterior $-\log p(\mathbf{x}|\mathbf{y})$, whereby the OT constraint $T_\#\mu=\nu$ automatically introduces the prior $p(\mathbf{x})$.
    - **Design Motivation**: Compared to the standard quadratic cost, the likelihood cost directly models the data term of the inverse problem and is more sensitive to the problem structure.

2.  **Quadratic Cost + Unbalanced Relaxation**:
    - **Function**: (a) Ensures the existence and uniqueness of the mapping by satisfying the twist condition; (b) handles multi-level noise and class imbalance through the marginal relaxation of UOT.
    - **Mechanism**: The final cost is $c(\mathbf{y},\mathbf{x}) = \tau(c_l + c_q)$. The quadratic term supplements the likelihood term so that the synthetic cost satisfies the twist condition (Prop. 3.1). The f-divergence terms $D_{\Psi_i}(\pi_0\|\mu) + D_{\Psi_i}(\pi_1\|\nu)$ in the UOT objective allow for soft matching of marginal distributions, automatically reweighting source samples through scaling factors $\Psi^*_i'$.
    - **Design Motivation**: Standard OT exhibits unsolvability in ill-posed problems; the hybrid cost ensures existence while achieving robustness to multi-level noise.

3.  **Semi-dual UOT Solver**:
    - **Function**: Parameterizes the potential function $v_\phi$ and the mapping $T_\theta$ with neural networks, learning the optimal UOT mapping through adversarial iterations.
    - **Mechanism**: The Kantorovich dual form of UOT is expanded into a semi-dual formula, introducing the c-transform $v^c(\mathbf{y})=\inf_{\mathbf{x}}[c(\mathbf{y},\mathbf{x})-v(\mathbf{x})]$, which allows $T_\theta(\mathbf{y}) \in \arg\inf_{\mathbf{x}}[c(\mathbf{y},\mathbf{x})-v_\phi(\mathbf{x})]$.
    - **Design Motivation**: The semi-dual form convexifies the solution compared to the original Monge problem; c-transform parameterization ensures that optimality conditions are satisfied.

## Key Experimental Results

### Main Results

| Task | Method | FFHQ PSNR | FFHQ SSIM | FFHQ FID | AFHQ PSNR | AFHQ SSIM | AFHQ FID |
|------|------|-----------|-----------|----------|-----------|-----------|----------|
| Gaussian Deblurring | NOT | 20.11 | 0.6035 | 52.901 | 19.99 | 0.5472 | 58.927 |
| | OTUR | 23.82 | 0.7106 | 24.337 | 23.91 | 0.6777 | 30.773 |
| | RCOT | 22.07 | 0.5492 | 123.692 | 22.34 | 0.5365 | 132.465 |
| | **UOTIP** | **24.06** | **0.7139** | **21.210** | **24.22** | **0.6804** | **12.566** |
| 4× Super-resolution | NOT | 20.13 | 0.6257 | 50.066 | 20.14 | 0.5833 | 44.252 |
| | OTUR | 24.09 | 0.7243 | 22.751 | 24.71 | 0.7079 | 19.575 |
| | RCOT | 24.05 | 0.6820 | 118.776 | 25.04 | 0.7137 | 69.072 |
| | **UOTIP** | **24.35** | **0.7371** | **19.475** | **24.97** | **0.7142** | **15.939** |

### Ablation Study

| Configuration | Gaussian Deblurring | HDR Reconstruction | Nonlinear Deblurring | Description |
|------|----------------|--------|-------------|------|
| Full Model (UOT+$c_l$+$c_q$) | 24.06 | 26.02 | 28.52 | Baseline |
| w/o Quadratic Cost | 22.18 | 24.31 | 26.74 | Ill-posedness worsens after removing $c_q$ |
| w/o Likelihood Cost | 23.41 | 25.18 | 27.65 | Data fidelity decreases after removing $c_l$ |
| Standard OT (Hard Marginal Constraint) | 23.12 | 25.04 | 27.31 | UOT relaxation outperforms strict constraints |

### Key Findings
- Multi-level noise processing—UOTIP maintains stable performance (PSNR fluctuation <0.5dB) when training a single model on three different noise levels.
- Class imbalance—UOT's adaptive marginal matching makes it superior to OT under different class distribution ratios.
- Practical success in super-resolution—Although super-resolution violates the theoretical twist condition guarantee, the network's inductive bias allows UOTIP to still reach Prev. SOTA.
- Texture preservation—Qualitative comparisons show that UOTIP preserves details better than OTUR (which is over-smoothed) and RCOT (which produces artifacts).

## Highlights & Insights
- **Ingenious MAP-OT Bridging**: The likelihood cost function naturally corresponds to the negative log-likelihood of Gaussian noise, implicitly introducing the data prior through OT constraints—mapping MAP estimation strictly into the OT framework for the first time.
- **Theoretically Guaranteed Practical Strategy**: Proposition 3.1 proves that the quadratic cost $\lambda c_q$ ensures the twist condition, resulting in a unique mapping even for ill-posed problems.
- **Significant Advantages of UOT**: By allowing soft marginal matching, it naturally handles three categories of practical difficulties: multi-level noise, data imbalance, and distribution heterogeneity.

## Limitations & Future Work
- In super-resolution tasks, the modified quadratic cost does not satisfy the theoretical twist condition, meaning existence/uniqueness is not guaranteed.
- The cost intensity parameter $\tau$ requires manual adjustment.
- Computational costs are higher compared to GANs (requires optimization of potential functions).
- Future Work: Extend the twist condition's applicability to non-L-Lipschitz operators; design adaptive $\tau$; integrate diffusion models as prior supplements.

## Related Work & Insights
- **vs NOT**: Both use neural OT frameworks, but NOT uses a standard quadratic cost and lacks inverse problem specificity; UOTIP introduces a likelihood cost to more accurately encode the data term.
- **vs OTUR**: Texture details in GAN methods are prone to over-smoothing or distortion; UOTIP is more robust through the global optimality of OT and the distribution relaxation of UOT.
- **vs Standard Bayesian Inversion**: Traditional methods rely on handcrafted priors; UOTIP learns distribution priors through data-driven approaches.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically integrates UOT with inverse problems for the first time, introducing likelihood costs and theoretical guarantees for the twist condition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 types of inverse tasks across multiple datasets, including ablation and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative is clear, proceeding logically from MAP principles.
- Value: ⭐⭐⭐⭐⭐ Unifies the two fields of inverse problems and optimal transport, offering both theoretical and applied merit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Normalized Energy Models for Linear Inverse Problems](learning_normalized_energy_models_for_linear_inverse_problems.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[CVPR 2026\] Variational Garrote for Sparse Inverse Problems](../../CVPR2026/image_restoration/variational_garrote_for_sparse_inverse_problems.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](../../CVPR2026/image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)
- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](../../NeurIPS2025/image_restoration/learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)

</div>

<!-- RELATED:END -->

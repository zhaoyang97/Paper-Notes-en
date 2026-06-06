---
title: >-
  [Paper Note] Rao-Blackwellized Score Matching on Manifolds
description: >-
  [ICML 2026][Image Generation][Denoising score matching] When the data distribution lies on an embedded manifold $M\subset\mathbb{R}^D$, the tangential target learned by Denoising Score Matching (DSM) with ambient Gaussia…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Denoising score matching"
  - "Manifold hypothesis"
  - "Rao-Blackwell"
  - "Riemannian score"
  - "Extrinsic curvature"
date: 2026-05-08
content_hash: eb88b90dfbc2890b
---

# Rao-Blackwellized Score Matching on Manifolds

**Conference**: ICML 2026  
**arXiv**: [2605.25567](https://arxiv.org/abs/2605.25567)  
**Code**: To be confirmed  
**Area**: Diffusion models / Score matching on manifolds / Generative modeling theory  
**Keywords**: Denoising score matching, Manifold hypothesis, Rao-Blackwell, Riemannian score, Extrinsic curvature

## TL;DR
When the data distribution lies on an embedded manifold $M\subset\mathbb{R}^D$, the tangential target learned by Denoising Score Matching (DSM) with ambient Gaussian noise contains a normal noise channel with variance diverging at a rate of $d/\sigma^2$. This paper proves that Rao-Blackwell conditioning on the nearest-point projection $\pi(X)$ cleanly removes this singular channel and precisely expands the remaining target as "Intrinsic Riemannian score + $O(\sigma^2)$ Tweedie correction + $O(\sigma^2)$ Weingarten/Ricci extrinsic curvature correction."

## Background & Motivation

**Background**: Score-based generative models rely on DSM to regress the residual $(Z-X)/\sigma^2$ on noisy samples, characterizing $\nabla\log p_\sigma$ via the Tweedie formula. However, real-world data often follow the "manifold hypothesis"—distributions are concentrated on low-dimensional submanifolds $M$, making them singular with respect to the ambient Lebesgue measure. Consequently, a strict $\nabla\log q$ does not exist, and DSM is well-defined only for $\sigma>0$.

**Limitations of Prior Work**: Existing approaches are unsatisfactory. **Intrinsic methods** (RSGM, Riemannian SDE) use Brownian motion on the manifold but require manifold-specific infrastructure like exponential maps or heat kernel simulations, which are almost unusable for general embedded manifolds. **Ambient methods** continue using Euclidean DSM based on the heuristic that "the projection onto the tangent space as $\sigma\to 0^+$ should converge to the intrinsic score." However, they neither characterize what is actually learned nor explain why generalization bounds collapse as $\sigma$ approaches 0.

**Key Challenge**: The tangential regression target $T_\sigma=P_T(\pi(X))(Z-X)/\sigma^2$ in ambient DSM has a conditional variance that diverges at $d/\sigma^2$ as $\sigma\to 0^+$. This is not a side effect of parameterization but an incompressible noise channel contributed by Gaussian noise on the normal fibers. Any direct regression on $T_\sigma$ is effectively "fed junk" by this diverging variance.

**Goal**: (i) Provide a statistically principled, signal-preserving, and variance-bounded tangential target within the ambient DSM framework; (ii) Expand this target to the $O(\sigma^2)$ order to clarify its deviation from the true intrinsic Riemannian score.

**Key Insight**: Noise on the normal fibers enters observations only through the dimension $X-\pi(X)\in N_{\pi(X)}M$, while the tangential signal is fully preserved by the nearest-point projection $\pi(X)$. This inspires using $\pi(X)$ as a sufficient statistic for Rao-Blackwellization: by conditioning $T_\sigma$ on $\pi(X)$, components depending solely on normal fiber noise are averaged out.

**Core Idea**: Define $r_\sigma(z)=\mathbb{E}[T_\sigma\mid\pi(X)=z]$. In short: "Perform Rao-Blackwellization once using the nearest-point projection to flatten the singular normal noise; the remainder is a principled target that is $O(\sigma^2)$-close to the intrinsic score."

## Method

### Overall Architecture

Setup: $Z\sim q\,d\mathrm{Vol}_M$ lies on a compact $C^5$ embedded submanifold $M\subset\mathbb{R}^D$ (dimension $d$, positive reach); ambient Gaussian noise $X=Z+\sigma\xi$, $\xi\sim\mathcal{N}(0,I_D)$. For small $\sigma$, the event $X\in\mathrm{Tub}_{r_0}(M)$ holds with probability $1-e^{-c/\sigma^2}$, making the nearest-point projection $\pi:\mathrm{Tub}_{r_0}(M)\to M$ well-defined almost everywhere.

The paper follows a three-step derivation chain:

1. **Define the Canonical Target**: The original tangential DSM target is $T_\sigma=P_T(\pi(X))\,(Z-X)/\sigma^2$. The canonical target $r_\sigma(z)=\mathbb{E}[T_\sigma\mid\pi(X)=z]$ is a tangential field on $TM$.
2. **Prove Statistical Optimality**: Among the family of all tangential field predictors that depend only on $\pi(X)$ (fiber-collapsing summaries), $r_\sigma$ is the unique $L^2$ risk minimizer. Meanwhile, the conditional variance of $T_\sigma$ diverges precisely as $d/\sigma^2+O(1)$, providing an unreachable Bayes lower bound under this restricted family.
3. **Compute Small-Noise Expansion**: On any embedded submanifold, $r_\sigma(z)=\nabla_M\log q(z)+\sigma^2[b_q(z)+g_M^{\mathrm{ext}}(z)]+o(\sigma^2)$, where $b_q$ is the intrinsic Tweedie term and $g_M^{\mathrm{ext}}$ is the extrinsic curvature term involving the Weingarten and Ricci operators.

### Key Designs

1. **Rao-Blackwellized Tangential Target $r_\sigma$ (Principledness + $L^2$ Optimality)**:
    - **Function**: Performs conditional expectation of the original DSM tangential target $T_\sigma$ along the "nearest-point projection $\pi(X)$" to obtain a tangential field $r_\sigma:M\to TM$ that depends only on $z\in M$, serving as the true "target to be learned" for ambient DSM on manifold data.
    - **Mechanism**: Since $X-\pi(X)\in N_{\pi(X)}M$, the original $T_\sigma$ can be written as $\sigma^{-2}P_T(\pi(X))(Z-\pi(X))$. Using $L^2$ projection decomposition conditioned on $\pi(X)=z$, the authors prove that for any tangential field $h$, $\mathcal{R}_\sigma(h)=\mathcal{R}_\sigma(r_\sigma)+\mathbb{E}\|r_\sigma(\pi(X))-h(\pi(X))\|^2$, thus $r_\sigma$ is the unique minimizer (Theorem 4.1). More generally, for any fiber-collapsing statistic $S$ "coarser than" $\pi(X)$ (i.e., $\sigma(S)\subseteq\sigma(\pi(X))$), $\pi(X)$ is the "finest collapsing statistic" and $r_\sigma$ is its optimal predictor, corresponding to the classic Rao-Blackwell theorem.
    - **Design Motivation**: Existing ambient methods regress directly on $T_\sigma$, forcing the network to approximate a target with diverging variance—essentially treating normal Gaussian noise as a supervision signal. Rao-Blackwellization provides an analytically principled alternative that is computationally cheaper than intrinsic methods (requiring only $\pi(X)$ rather than exponential maps), combining "signal purity" with "zero-cost infrastructure."

2. **Variance Collapse Theorem + $d/\sigma^2$ Bayes Bound (Theorem 4.2)**:
    - **Function**: Quantifies the gap between the original target $T_\sigma$ and the Rao-Blackwellized target $r_\sigma$, proving that the $d/\sigma^2$ divergence rate is a lower bound that no fiber-collapsing predictor can overcome.
    - **Mechanism**: In tubular coordinates, $T_\sigma$ is decomposed into a tangential signal and normal noise. The normal component is isotropic Gaussian conditioned on $\pi(X)$ and independent of it, leading to $\mathrm{Var}(T_\sigma\mid\pi(X)=z)=d/\sigma^2+O(1)$. By the law of total variance, $\mathrm{Var}(T_\sigma)=\mathrm{Var}(r_\sigma(\pi(X)))+d/\sigma^2+O(1)$. While $r_\sigma(\pi(X))$ has bounded variance ($O(1)$), $T_\sigma$ diverges; thus any $S$-measurable predictor $\eta$ faces an irreducible risk $\inf_\eta\mathbb{E}\|T_\sigma-\eta\|^2\geq d/\sigma^2+O(1)$.
    - **Design Motivation**: This elevates the necessity of Rao-Blackwellization to an information-theoretic level—without this step, any network regardless of capacity or training time incurs an irreducible risk; with it, the variance collapses to $O(1)$, making the signal-to-noise ratio meaningful as $\sigma\to 0$.

3. **Extrinsic $\sigma^2$ Correction Expansion (Theorem 5.2 + Corollary 5.4)**:
    - **Function**: Precisely expands the canonical target $r_\sigma$ to the $\sigma^2$ order, isolating bias from intrinsic Tweedie smoothing and extrinsic curvature bias.
    - **Mechanism**: Using graph coordinates for Bayesian computation, the expansion reveals $r_\sigma(z)=\nabla_M\log q(z)+\sigma^2[b_q(z)+g_M^{\mathrm{ext}}(z)]+o(\sigma^2)$. The intrinsic Tweedie term is $b_q(z)=\tfrac{1}{2}\nabla_M[\Delta_M\log q+\|\nabla_M\log q\|^2](z)$. The extrinsic term is $g_M^{\mathrm{ext}}(z)=(\tfrac{1}{2}W_{H(z)}-\mathrm{Ric}_z^\sharp)\nabla_M\log q(z)$, where $W$ is the Weingarten operator, $H$ is the mean curvature vector, and $\mathrm{Ric}^\sharp$ is the Ricci endomorphism. On $S^d$, the extrinsic coefficient collapses to a scalar $\alpha_d=1-d/2$, where $\alpha_1=+1/2, \alpha_2=0, \alpha_3=-1/2$.
    - **Design Motivation**: This explains (1) exactly what ambient DSM learns—the intrinsic score plus an explicit bias derived from curvature tensors, and (2) why ambient DSM works exceptionally well on $S^2$—since $S^2$ is an Einstein manifold where $\tfrac{1}{2}W_H=\mathrm{Ric}^\sharp=\mathrm{Id}$ causes the extrinsic terms to cancel, leaving only intrinsic bias. This cancellation fails for $S^1, S^3,$ and $S^d (d\geq 3)$.

## Key Experimental Results

### Main Results

| Target | Setup | Prediction | Result |
|---|---|---|---|
| Variance Collapse (Thm 4.2) | $S^2$ + vMF$(\mu,\kappa=2)$ | $\log\mathbb{E}\|T_\sigma\|^2$ vs $\log\sigma$ slope is $-2$ | Slope is $-2$ ($d/\sigma^2$); blue line for $r_\sigma$ remains flat (Fig 1) |
| Extrinsic Coeff (Cor 5.4) | $S^d, T^2, \sigma\in\{0.05, 0.08\}$ | $\alpha_1=+1/2, \alpha_2=0, \alpha_3=-1/2$ | Numerical $\alpha_{\mathrm{ext}}$ matches predictions (Fig 2) |
| Sampling Debias (Cor 5.4) | Langevin drift, $\sigma=0.3$ | Correction via $(1+\sigma^2\alpha_d)\nabla_M\log q$ | Debiased drift (blue) matches intrinsic density; ambient drift (orange) deviates (Fig 3a) |

### Ablation Study

| Target | Manifold | Score MSE Trend | Observation |
|---|---|---|---|
| Regress $T_{\sigma,i}$ (Original) | Einstein Manifolds | High, worsens with $d$ | Contaminated by $d/\sigma^2$ variance |
| Regress $\widehat{r}_{\sigma,i}$ (RB) | Einstein Manifolds | Significantly lower | Validates Theorem 4.2 |
| Flat Case $M=V$ | $\mathbb{R}^d$ embedding | Reduces to low-dim DSM | Both corrections vanish (Prop 5.1) |

### Key Findings
- **Variance collapse slope on $S^2$ is exactly $-2$**: Confirms $d/\sigma^2$ is an asymptotically precise rate rather than a loose bound.
- **$T^2$ as a critical non-spherical control**: The torus is intrinsically flat ($\mathrm{Ric}=0$), yet predicted extrinsic coefficient $+1/2$ is confirmed—proving bias stems from embedding.
- **$S^2$ cancellation is a coincidence**: The zeroing of extrinsic bias on $S^2$ is specific to $d=2$ Einstein manifolds; for $d\neq 2$, systematic bias exists and changes direction based on the sign of $\alpha_d$.

## Highlights & Insights
- **The "Correct" Rao-Blackwell for Manifold DSM**: While traditional Rao-Blackwell requires sufficient statistics for parameters, this paper uses the geometric nearest-point projection $\pi(X)$ as the "finest collapsing statistic" provided by the embedding.
- **Divergence as Inevitability**: The $d/\sigma^2$ divergence is not an algorithmic weakness but a Bayes lower bound for fiber-collapsing predictors.
- **Computable Extrinsic Correction**: The closed-form correction $\sigma^2(\tfrac{1}{2}W_H-\mathrm{Ric}^\sharp)\nabla_M\log q$ offers a middle ground: train using Euclidean pipelines for speed, then subtract the correction for intrinsic-level accuracy.
- **Explaining the $S^2$ Success**: Much of the success of ambient manifold DSM in literature is attributed to the $S^2$ Einstein coincidence; this provides a cautionary tale for generalizing to general surfaces.

## Limitations & Future Work
- **Limitations**: Population-level analysis; finite-sample minimax rates are only touched upon in Appendix G. Requirements for $C^5$ smoothness and positive reach may limit applications to non-compact or low-regularity manifolds. Dependency on curvature estimation for unknown manifolds is not yet addressed in an end-to-end workflow.
- **Future Work**: Extending to manifolds with boundaries (e.g., probability simplices for categorical diffusion) and developing neural local estimators for $r_\sigma$ on unknown manifolds.

## Related Work & Insights
- **vs Riemannian SGM**: RSGM avoids ambient singularities but requires expensive manifold infrastructure; this work achieves $O(\sigma^2)$ proximity via Euclidean pipelines.
- **vs Vincent (2011)**: Exposes the implicit assumption of absolute continuity with respect to Lebesgue measure and provides the manifold-supported counterpart.
- **vs Heuristic Projection**: Provides a statistical foundation (Rao-Blackwell) and precise bias quantification for the "tangential projection" heuristic.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] DiScoFormer: Plug-In Density and Score Estimation with Transformers](discoformer_plug-in_density_and_score_estimation_with_transformers.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](../../ICCV2025/image_generation/balanced_image_stylization_with_style_matching_score.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)
- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](../../ICCV2025/image_generation/balanced_image_stylization_with_style_matching_score.md)
- [\[ICML 2026\] DiScoFormer: Plug-In Density and Score Estimation with Transformers](discoformer_plug-in_density_and_score_estimation_with_transformers.md)
- [\[ICML 2025\] Efficient Diffusion Models for Symmetric Manifolds](../../ICML2025/image_generation/efficient_diffusion_models_for_symmetric_manifolds.md)
- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)

</div>

<!-- RELATED:END -->

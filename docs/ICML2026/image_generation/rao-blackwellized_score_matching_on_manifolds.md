---
title: >-
  [Paper Note] Rao-Blackwellized Score Matching on Manifolds
description: >-
  [ICML 2026][Image Generation][Rao-Blackwell] When the data distribution lies on an embedded manifold $M\subset\mathbb{R}^D$, the tangential target learned by ambient space Gaussian denoising score matching (DSM) contains normal noise channels with variance diverging as $d/\sigma^2$. This paper proves that performing Rao-Blackwell conditioning on the nearest-point
tags:
  - ICML 2026
  - Image Generation
  - Rao-Blackwell
date: 2026-05-08
content_hash: e94950204b2abbc9
---
# Rao-Blackwellized Score Matching on Manifolds

**Conference**: ICML 2026  
**arXiv**: [2605.25567](https://arxiv.org/abs/2605.25567)  
**Code**: TBD  
**Area**: Diffusion models / Score matching on manifolds / Generative modeling theory  
**Keywords**: Denoising score matching, manifold hypothesis, Rao-Blackwell, Riemannian score, extrinsic curvature

## TL;DR
When the data distribution lies on an embedded manifold $M\subset\mathbb{R}^D$, the tangential target learned by ambient space Gaussian denoising score matching (DSM) contains normal noise channels with variance diverging as $d/\sigma^2$. This paper proves that performing Rao-Blackwell conditioning on the nearest-point projection $\pi(X)$ cleanly removes this singular channel and expands the remaining target exactly into "Intrinsic Riemannian score + $\sigma^2$-order Tweedie correction + $\sigma^2$-order Weingarten/Ricci extrinsic curvature correction."

## Background & Motivation

**Background**: Score-based generative models rely on DSM to regress the residual $(Z-X)/\sigma^2$ on noisy samples, using the Tweedie formula to characterize $\nabla\log p_\sigma$. However, real-world data generally follows the "manifold hypothesis"—where the distribution is concentrated on a low-dimensional submanifold $M$, making it singular with respect to the ambient Lebesgue measure. Consequently, the strict $\nabla\log q$ does not exist, and DSM is only well-defined for $\sigma>0$.

**Limitations of Prior Work**: Two existing approaches are unsatisfactory. **Intrinsic methods** (RSGM, Riemannian SDE) switch to Brownian motion on the manifold but require manifold-specific infrastructure like exponential maps or heat kernel simulations, which are almost unusable for general embedded manifolds. **Ambient methods** continue using Euclidean DSM based on the heuristic that "projecting to the tangent space as $\sigma\to 0^+$ should converge to the intrinsic score." However, they neither characterize exactly what is learned nor explain why generalization bounds typically collapse when $\sigma$ is close to 0.

**Key Challenge**: The tangential regression target of ambient DSM, $T_\sigma=P_T(\pi(X))(Z-X)/\sigma^2$, has a conditional variance that diverges as $d/\sigma^2$ when $\sigma\to 0^+$. This is not a side effect of parameterization; it is an incompressible noise channel contributed by the Gaussian noise itself on the normal fibers. Any direct regression of $T_\sigma$ is fed "garbage" by this diverging variance.

**Goal**: (i) Provide a statistically canonical, signal-lossless, and variance-bounded tangential target within the ambient DSM framework; (ii) Expand it to the $\sigma^2$ order to clearly identify its deviation from the true intrinsic Riemannian score.

**Key Insight**: Note that noise on the normal fibers enters the observation only through $X-\pi(X)\in N_{\pi(X)}M$, whereas the tangential signal is fully preserved by the nearest-point projection $\pi(X)$. This inspires using $\pi(X)$ as a sufficient statistic for Rao-Blackwellization: by conditioning $T_\sigma$ on $\pi(X)$, any components depending solely on normal fiber noise are averaged out.

**Core Idea**: Define $r_\sigma(z)=\mathbb{E}[T_\sigma\mid\pi(X)=z]$. In essence, "performing Rao-Blackwell with the nearest-point projection flattens the singular normal noise, leaving a canonical target that is $O(\sigma^2)$ close to the intrinsic score."

## Method

### Overall Architecture

The goal is to solve the problem of "ambient DSM tangential targets being polluted by normal noise" on manifold data. Data $Z\sim q\,d\mathrm{Vol}_M$ lies on a compact, $C^5$, positive-reach embedded submanifold $M\subset\mathbb{R}^D$ (dimension $d$). Noise is added as $X=Z+\sigma\xi$ ($\xi\sim\mathcal{N}(0,I_D)$). For small $\sigma$, the event $X\in\mathrm{Tub}_{r_0}(M)$ holds with probability $1-e^{-c/\sigma^2}$, making the nearest-point projection $\pi:\mathrm{Tub}_{r_0}(M)\to M$ well-defined almost everywhere. This paper does not change the training pipeline but clarifies "what goal should be regressed": by using $\pi(X)$ as a sufficient statistic to perform Rao-Blackwell conditioning, it is proven that the resulting canonical target is statistically optimal and variance-bounded.

### Key Designs

**1. Rao-Blackwellized Canonical Tangential Target $r_\sigma$: Averaging the Singular Noise Channel**

Original ambient methods directly regress the tangential target $T_\sigma=P_T(\pi(X))(Z-X)/\sigma^2$, which is equivalent to forcing a network to approximate a target with diverging variance—essentially learning normal Gaussian noise as a supervision signal. The key observation is that normal noise enters only through $X-\pi(X)\in N_{\pi(X)}M$, while the tangential signal is preserved by $\pi(X)$. Thus, $T_\sigma$ can be rewritten as $\sigma^{-2}P_T(\pi(X))(Z-\pi(X))$, and by taking the conditional expectation along $\pi(X)$, the canonical target $r_\sigma(z)=\mathbb{E}[T_\sigma\mid\pi(X)=z]$ is defined as a tangential field $r_\sigma:M\to TM$. Using an $L^2$ projection decomposition, it is proven that for any tangential field $h$, $\mathcal{R}_\sigma(h)=\mathcal{R}_\sigma(r_\sigma)+\mathbb{E}\|r_\sigma(\pi(X))-h(\pi(X))\|^2$, confirming $r_\sigma$ as the unique risk minimizer (Theorem 4.1). This step achieves the signal purity of intrinsic methods with the zero-infrastructure cost of ambient methods.

**2. Variance Collapse Theorem and the $d/\sigma^2$ Bayes Lower Bound**

To quantify the difference made by Rao-Blackwellization, $T_\sigma$ is decomposed into tangential signal and normal noise components in tubular coordinates. The normal part is isotropic Gaussian and independent of $\pi(X)$, leading to $\mathrm{Var}(T_\sigma\mid\pi(X)=z)=d/\sigma^2+O(1)$. By the law of total variance, $\mathrm{Var}(T_\sigma)=\mathrm{Var}(r_\sigma(\pi(X)))+d/\sigma^2+O(1)$, showing that $r_\sigma(\pi(X))$ has bounded variance ($O(1)$) while $T_\sigma$ diverges as $d/\sigma^2$ (Theorem 4.2). This elevates the necessity of projection from a heuristic to an information-theoretic level: without this step, the irreducible risk is at least $d/\sigma^2$ regardless of network capacity.

**3. Extrinsic $\sigma^2$ Correction Expansion: The Gap from Intrinsic Score**

Finally, a Bayesian calculation in graph coordinates yields a second-order expansion $r_\sigma(z)=\nabla_M\log q(z)+\sigma^2[b_q(z)+g_M^{\mathrm{ext}}(z)]+o(\sigma^2)$ (Theorem 5.2). The leading term is the true intrinsic Riemannian score. The bias consists of two parts: the intrinsic Tweedie term $b_q(z)=\tfrac{1}{2}\nabla_M[\Delta_M\log q+\|\nabla_M\log q\|^2](z)$ and the extrinsic curvature term $g_M^{\mathrm{ext}}(z)=(\tfrac{1}{2}W_{H(z)}-\mathrm{Ric}_z^\sharp)\nabla_M\log q(z)$, where $W_u$ is the Weingarten operator, $H(z)$ is the mean curvature vector, and $\mathrm{Ric}_z^\sharp$ is the Ricci endomorphism. On a sphere $S^d$, the extrinsic coefficient collapses to $\alpha_d=1-d/2$ (Corollary 5.4). This answers why ambient DSM on $S^2$ performs surprisingly well: it is an Einstein manifold where extrinsic terms cancel out.

### Loss & Training

This paper provides population-level identification theorems and does not define a new loss function. The implicit training strategy involves replacing the raw tangential residual $T_\sigma$ with a finite-sample estimate $\widehat{r}_{\sigma,i}$, such as through local linear regression. Figure 3(b) compares "regressing $T_{\sigma,i}$ vs. regressing $\widehat{r}_{\sigma,i}$" on Einstein manifolds, showing that the latter achieves significantly lower score MSE, with the gap widening as $d$ increases, consistent with the $d/\sigma^2$ factor in Theorem 4.2.

## Key Experimental Results

### Main Results

| Verification Goal | Setting | Prediction | Numerical Result |
|---|---|---|---|
| Variance Collapse Rate (Theorem 4.2) | $S^2$ + vMF$(\mu,\kappa=2)$ | $\log\mathbb{E}\|T_\sigma\|^2$ slope vs $\log\sigma$ is $-2$; $r_\sigma$ is flat | Black line slope is $-2$ ($d/\sigma^2$); blue line stays at $\mathbb{E}\|\nabla_M\log q\|^2$ (Fig 1) |
| Extrinsic Coefficient (Corollary 5.4) | $S^1,S^2,S^3,S^4,T^2$, $\sigma\in\{0.05,0.06,0.08\}$ | $\alpha_1=+1/2,\alpha_2=0,\alpha_3=-1/2,\alpha_4=-1$; $T^2$ is $+1/2$ | Numerical $\alpha_{\mathrm{ext}}$ matches predictions (Fig 2) |
| Sampling De-biasing (Corollary 5.4) | Closed-form Langevin drift, $\sigma=0.3$ | Using $(1-\sigma^2\alpha_d)(1+\sigma^2\alpha_d)\nabla_M\log q$ corrects ambient bias | De-biased drift (blue) equilibrium matches intrinsic score (black), ambient (orange) deviates (Fig 3a) |

### Ablation Study

| Target | Manifold | Score MSE Trend | Explanation |
|---|---|---|---|
| Regressing $T_{\sigma,i}$ (Original) | Multiple Einstein manifolds | High, worsens significantly with $d$ | Polluted by $d/\sigma^2$ diverging variance |
| Regressing $\widehat{r}_{\sigma,i}$ (RB) | Same manifolds & budget | Significantly lower, advantage grows with $d$ | Validates Theorem 4.2 |
| Flat Case $M=V$ (Prop 5.1) | $\mathbb{R}^d$ embedding | Strictly degrades to low-dim Gaussian DSM | Corrections vanish, providing a clean baseline |

### Key Findings

- **Variance collapse slope on $S^2$ is exactly $-2$**: Confirms $d/\sigma^2$ is an asymptotic exact rate, providing quantifiable value for the "Rao-Blackwell first, then regress" practice.
- **$T^2$ as a critical non-spherical control**: The torus is intrinsically flat ($\mathrm{Ric}=0$), but the extrinsic coefficient is predicted and confirmed as $+1/2$, proving bias stems from the embedding rather than intrinsic curvature alone.
- **$S^2$ cancellation is an Einstein manifold coincidence**: Only $\tfrac{1}{2}W_H=\mathrm{Ric}^\sharp=\mathrm{Id}$ makes the extrinsic term vanish; for $d\neq 2$, systematic bias exists and its direction depends on the sign of $\alpha_d$.

## Highlights & Insights

- **Correct adaptation of Rao-Blackwell to manifold DSM**: By using the $L^2$ projection decomposition and fiber-collapsing summaries, the paper uses $\pi(X)$ as the "finest collapsing statistic" provided naturally by embedded manifolds, making the theoretical logic very clean.
- **Redefining $\sigma\to 0$ instability**: The divergence is not a weakness of a specific algorithm but a Bayes lower bound $d/\sigma^2$ for any predictor using $\pi(X)$. This provides a rigorous theoretical foundation for engineering practices involving projection.
- **Calculable extrinsic correction**: The term $\sigma^2(\tfrac{1}{2}W_H-\mathrm{Ric}^\sharp)\nabla_M\log q$ suggests an alternative pipeline: train with the cheap Euclidean pipeline and then subtract the closed-form extrinsic correction to approximate the intrinsic score.
- **Explanation for $S^2$ success**: Papers demonstrating manifold DSM on $S^2$ benefit from an Einstein coincidence. The theory predicts systematic "reverse" shifts on $S^3$, which should guide future benchmark designs.

## Limitations & Future Work

- **Ours Limitations**: Results are at the population level and lack complete finite-sample minimax rates. The requirements (compact, $C^5$, positive reach) do not yet cover non-compact or low-regularity manifolds.
- **Practical Limitations**: All results assume $\pi(X)$ is computable. For unknown manifolds, this implies a prior manifold estimation/projection step. Furthermore, the extrinsic correction requires explicit curvature tensors ($W_H, \mathrm{Ric}^\sharp$), for which a closed-loop workflow is not yet provided.
- **Future Directions**: (i) Replace non-parametric estimates with efficient neural local estimators to provide end-to-end training; (ii) Integrate the $\sigma^2$ correction as a "post-processing" module for existing ambient DSM pipelines; (iii) Extend derivations to manifolds with boundaries (e.g., simplices).

## Related Work & Insights

- **vs. Riemannian SGM (Bortoli et al., 2022)**: RSGM avoids ambient singularities via heat kernels. Ours stays in the ambient space but uses Rao-Blackwell to achieve a target $O(\sigma^2)$ close to the intrinsic score without exponential maps.
- **vs. Pidstrigach (2022)**: While prior work characterizes ambient score alignment with normal spaces, this work quantitatively determines exactly what the tangential component learns.
- **vs. Vincent (2011)**: This work exposes the "absolute continuity" assumption in classic DSM and provides the manifold-supported counterpart, where $r_\sigma$ replaces the Tweedie score.
- **Related Insights**: The Rao-Blackwell + fiber-collapsing framework is applicable to any regression setting where the observation contains separable nuisance channels, such as surface normal estimation in noisy point clouds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

- **Riemannian Score-Based Generative Modelling** (Bortoli et al., 2022)
- **Score-based Generative Models for Data Near Manifolds** (Pidstrigach, 2022)
- **A Connection Between Score Matching and Denoising Autoencoders** (Vincent, 2011)

</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](../../ICCV2025/image_generation/balanced_image_stylization_with_style_matching_score.md)
- [\[ICML 2025\] Efficient Diffusion Models for Symmetric Manifolds](../../ICML2025/image_generation/efficient_diffusion_models_for_symmetric_manifolds.md)
- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)

</div>

<!-- RELATED:END -->

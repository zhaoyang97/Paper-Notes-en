---
title: >-
  [Paper Note] Rao-Blackwellized Score Matching on Manifolds
description: >-
  [ICML 2026][Image Generation][Denoising score matching] When the data distribution lies on an embedded manifold $M\subset\mathbb{R}^D$, the tangential target learned by ambient Gaussian Denoising Score Matching (DSM) contains normal noise channels with variance diverging as $d/\sigma^2$. This paper proves that a single Rao-Blackwell conditioning step on the nearest-point projection $\pi(X)$ cleanly removes this singular channel and expands the remaining target precisely as "i…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Denoising score matching"
  - "Manifold hypothesis"
  - "Rao-Blackwell"
  - "Riemannian score"
  - "Extrinsic curvature"
date: 2026-05-08
content_hash: 70ceb0d6671b2a5e
---

# Rao-Blackwellized Score Matching on Manifolds

**Conference**: ICML 2026  
**arXiv**: [2605.25567](https://arxiv.org/abs/2605.25567)  
**Code**: To be confirmed  
**Area**: Diffusion models / Score matching on manifolds / Generative modeling theory  
**Keywords**: Denoising score matching, Manifold hypothesis, Rao-Blackwell, Riemannian score, Extrinsic curvature

## TL;DR
When the data distribution lies on an embedded manifold $M\subset\mathbb{R}^D$, the tangential target learned by ambient Gaussian Denoising Score Matching (DSM) contains normal noise channels with variance diverging as $d/\sigma^2$. This paper proves that a single Rao-Blackwell conditioning step on the nearest-point projection $\pi(X)$ cleanly removes this singular channel and expands the remaining target precisely as "intrinsic Riemannian score + $\sigma^2$-order Tweedie correction + $\sigma^2$-order Weingarten/Ricci extrinsic curvature correction."

## Background & Motivation

**Background**: Score-based generative models rely on DSM to regress residuals $(Z-X)/\sigma^2$ on noisy samples, characterizing $\nabla\log p_\sigma$ via the Tweedie formula. However, real-world data often satisfies the "manifold hypothesis"—distributions are concentrated on low-dimensional submanifolds $M$, which are singular with respect to the ambient Lebesgue measure. Consequently, $\nabla\log q$ does not strictly exist in the ambient space, and DSM is well-defined only when $\sigma>0$.

**Limitations of Prior Work**: Two existing paradigms are unsatisfactory. **Intrinsic methods** (e.g., RSGM, Riemannian SDE) utilize Brownian motion on the manifold but require manifold-specific infrastructure such as exponential maps or heat kernel simulations, which are often unavailable for general embedded manifolds. **Ambient methods** continue to use Euclidean DSM based on the heuristic that "after projection onto the tangent space, the target should converge to the intrinsic score as $\sigma\to 0^+$." However, these methods lack a characterization of what is actually learned and fail to explain why generalization bounds typically collapse as $\sigma$ approaches 0.

**Key Challenge**: The conditional variance of the ambient DSM tangential target $T_\sigma=P_T(\pi(X))(Z-X)/\sigma^2$ diverges as $d/\sigma^2$ when $\sigma\to 0^+$. This is not an artifact of parameterization but an incompressible noise channel contributed by the Gaussian noise on the normal fibers. Any direct regression of $T_\sigma$ is effectively "contaminated" by this diverging variance.

**Goal**: (i) To provide a statistically canonical, signal-lossless, and variance-bounded tangential target within the ambient DSM framework; (ii) to expand this target to order $\sigma^2$ to precisely characterize its deviation from the true intrinsic Riemannian score.

**Key Insight**: The noise on the normal fibers enters observations only through the component $X-\pi(X)\in N_{\pi(X)}M$, while the tangential signal can be fully preserved by the nearest-point projection $\pi(X)$. This motivates using $\pi(X)$ as a sufficient statistic for Rao-Blackwellization: by conditioning $T_\sigma$ on $\pi(X)$, components that depend solely on normal fiber noise are averaged out.

**Core Idea**: Define $r_\sigma(z)=\mathbb{E}[T_\sigma\mid\pi(X)=z]$. In short, "performing one Rao-Blackwell step with the nearest-point projection flattens the singular normal noise, leaving a canonical target that is $O(\sigma^2)$ close to the intrinsic score."

## Method

### Overall Architecture

The goal is to address the issue of the ambient DSM tangential target being contaminated by normal noise for manifold data. Data $Z\sim q\,d\mathrm{Vol}_M$ lies on a compact $C^5$, positive-reach embedded submanifold $M\subset\mathbb{R}^D$ (with dimension $d$). Noisy samples are generated as $X=Z+\sigma\xi$ where $\xi\sim\mathcal{N}(0,I_D)$. For small $\sigma$, the event $X\in\mathrm{Tub}_{r_0}(M)$ occurs with probability $1-e^{-c/\sigma^2}$, making the nearest-point projection $\pi:\mathrm{Tub}_{r_0}(M)\to M$ almost surely well-defined. This work does not change the training pipeline but clarifies the "proper target" for regression by conditioning the original tangential target using $\pi(X)$ as a sufficient statistic. The paper proves this canonical target is statistically optimal and variance-bounded, then provides a $\sigma^2$-order expansion.

### Key Designs

**1. Rao-Blackwellized Canonical Tangential Target $r_\sigma$: Averaging the Singular Noise Channel**

Original ambient methods regress the tangential DSM target $T_\sigma=P_T(\pi(X))(Z-X)/\sigma^2$, which is equivalent to forcing a network to approximate a target with diverging variance—essentially treating normal Gaussian noise as a supervision signal. The key observation here is that normal noise affects the observation only through $X-\pi(X)\in N_{\pi(X)}M$, while the tangential signal is preserved by $\pi(X)$. Thus, $T_\sigma$ can be rewritten as $\sigma^{-2}P_T(\pi(X))(Z-\pi(X))$, and a canonical target can be defined via conditional expectation: $r_\sigma(z)=\mathbb{E}[T_\sigma\mid\pi(X)=z]$, mapping $z\in M$ into a tangential field $r_\sigma:M\to TM$. Using an $L^2$ projection decomposition on $\pi(X)=z$, it is proved that for any tangential field $h$, $\mathcal{R}_\sigma(h)=\mathcal{R}_\sigma(r_\sigma)+\mathbb{E}\|r_\sigma(\pi(X))-h(\pi(X))\|^2$, making $r_\sigma$ the unique risk minimizer (Theorem 4.1). This approach achieves the signal purity of intrinsic methods while maintaining the low infrastructure cost of ambient methods.

**2. Variance Collapse Theorem and the $d/\sigma^2$ Bayes Lower Bound**

To quantify the difference made by Rao-Blackwellization, the paper decomposes $T_\sigma$ into tangential signal and normal noise in tubular coordinates. The normal component is conditionally isotropic Gaussian and independent of $\pi(X)$, leading to $\mathrm{Var}(T_\sigma\mid\pi(X)=z)=d/\sigma^2+O(1)$. By the law of total variance, $\mathrm{Var}(T_\sigma)=\mathrm{Var}(r_\sigma(\pi(X)))+d/\sigma^2+O(1)$, showing that $r_\sigma(\pi(X))$ has bounded variance ($O(1)$) while $T_\sigma$ diverges as $d/\sigma^2$ (Theorem 4.2). This also establishes a lower bound $\inf_\eta\mathbb{E}\|T_\sigma-\eta\|^2\geq d/\sigma^2+O(1)$ for any $S$-measurable predictor $\eta$, where equality holds only if $\sigma(S)=\sigma(\pi(X))$. This justifies the projection step from an information-theoretic perspective.

**3. Extrinsic $\sigma^2$ Correction Expansion: Analyzing the Bias from Intrinsic Score**

The paper performs a Bayesian calculation in graph coordinates to account for induced volume element corrections and mean curvature corrections of the tubular Jacobian. This yields an expansion precise to second order: $r_\sigma(z)=\nabla_M\log q(z)+\sigma^2[b_q(z)+g_M^{\mathrm{ext}}(z)]+o(\sigma^2)$ (Theorem 5.2). The leading term is the true intrinsic Riemannian score. The bias consists of two parts: the intrinsic Tweedie term $b_q(z)=\tfrac{1}{2}\nabla_M[\Delta_M\log q+\|\nabla_M\log q\|^2](z)$, and the extrinsic curvature term $g_M^{\mathrm{ext}}(z)=(\tfrac{1}{2}W_{H(z)}-\mathrm{Ric}_z^\sharp)\nabla_M\log q(z)$, where $W_{u}$ is the Weingarten operator in direction $u$, $H(z)$ is the mean curvature vector, and $\mathrm{Ric}_z^\sharp$ is the Ricci endomorphism. On the sphere $S^d$, the extrinsic coefficient collapses to a scalar $\alpha_d=1-d/2$ (Corollary 5.4). This explains why ambient DSM works exceptionally well on $S^2$ (where $\alpha_2=0$ leads to cancellation), but develops biases on $S^1, S^3,$ or $S^d(d\geq3)$.

### Loss & Training

This work presents a population-level identification theorem rather than a new loss function. The implied strategy is to replace the regression target $T_\sigma$ with a finite-sample estimate $\widehat{r}_{\sigma,i}$, such as one obtained through local linear regression. Figure 3(b) compares regression on $T_{\sigma,i}$ vs. $\widehat{r}_{\sigma,i}$ on several Einstein manifolds, showing that the latter achieves significantly lower score MSE, with the gap widening as $d$ increases, consistent with the $d/\sigma^2$ prediction.

## Key Experimental Results

### Main Results

| Target | Setup | Prediction | Numerical Result |
|---|---|---|---|
| Variance Collapse (Theorem 4.2) | $S^2$ + vMF$(\mu,\kappa=2)$ | Slope of $\log\mathbb{E}\|T_\sigma\|^2$ vs $\log\sigma$ is $-2$; $r_\sigma$ flat | Black line slope $-2$; blue line ($r_\sigma$) flat at $\mathbb{E}\|\nabla_M\log q\|^2$ (Fig 1) |
| Extrinsic Coeff (Corollary 5.4) | $S^1,S^2,S^3,S^4,T^2$, $\sigma\in\{0.05,0.06,0.08\}$ | $\alpha_1=+1/2,\alpha_2=0,\alpha_3=-1/2,\alpha_4=-1$; $T^2$ is $+1/2$ | Numerical $\alpha_{\mathrm{ext}}$ matches predictions (Fig 2) |
| Sampling Debiasing (Corollary 5.4) | Closed-form Langevin drift, $\sigma=0.3$ | Correct bias via $(1-\sigma^2\alpha_d)(1+\sigma^2\alpha_d)\nabla_M\log q$ | Debiased drift (blue) matches intrinsic score; ambient drift (orange) deviates (Fig 3a) |

### Ablation Study

| Target | Manifold | Score MSE Trend | Explanation |
|---|---|---|---|
| Regressing $T_{\sigma,i}$ (Original) | Multiple Einstein manifolds | High, worsens with $d$ | Contaminated by $d/\sigma^2$ diverging variance |
| Regressing $\widehat{r}_{\sigma,i}$ (RB) | Same manifolds & budget | Significantly lower; advantage grows with $d$ | Validates Theorem 4.2 |
| Flat Case $M=V$ (Prop 5.1) | $\mathbb{R}^d$ embedding | Reduces to low-dim Gaussian DSM | Both corrections vanish, providing a clean baseline |

### Key Findings

- **The slope of variance collapse on $S^2$ is exactly $-2$**: Confirms $d/\sigma^2$ as a precise asymptotic rate, quantifying the value of RB before regression.
- **$T^2$ as a critical non-spherical control**: The torus is intrinsically flat ($\mathrm{Ric}=0$), yet the extrinsic coefficient is predicted and found to be $+1/2$, proving that the bias stems from the embedding rather than just intrinsic curvature.
- **The cancellation on $S^2$ is a coincidence of Einstein manifolds**: $\tfrac{1}{2}W_H=\mathrm{Ric}^\sharp=\mathrm{Id}$ only occurs for specific cases; for $d\neq 2$, biases exist and their directions depend on the sign of $\alpha_d$.

## Highlights & Insights

- **Correct version of Rao-Blackwell for manifold DSM**: Formalizes the use of $\pi(X)$ as the "finest fiber-collapsing statistic" to achieve signal purity without requiring manifold-exclusive operations like exponential maps.
- **Divergence at $\sigma\to 0$ redefined as information-theoretic necessity**: The $d/\sigma^2$ term is a Bayes lower bound for any predictor based on $\pi(X)$, providing rigorous theoretical support for projection-based engineering practices.
- **Calculable extrinsic correction**: The expansion $\sigma^2(\tfrac{1}{2}W_H-\mathrm{Ric}^\sharp)\nabla_M\log q$ suggests an alternative: train using the cheap Euclidean pipeline and subtract the closed-form extrinsic bias afterward.
- **Explanation for the $S^2$ benchmark success**: Papers reporting success on $S^2$ are benefiting from an Einstein coincidence; the theory predicts systematic "opposite" drifts on $S^3$, which can guide future benchmark design.

## Limitations & Future Work

- **Ours**: Results are population-level; no complete finite-sample minimax rates are provided (only preliminary non-parametric rates in Appendix G). Requires compact, $C^5$, positive-reach manifolds.
- **General Limitations**: Results rely on the computability of $\pi(X)$, implying a prior step of manifold estimation/projection for unknown manifolds. Extrinsic corrections require curvature tensors ($W_H, \mathrm{Ric}^\sharp$), and the paper lacks end-to-end experiments on high-dimensional data like images.
- **Future Directions**: (i) Replacing non-parametric $\widehat{r}_\sigma$ with neural local estimators for end-to-end training; (ii) applying the $\sigma^2$ correction as a post-processing module for existing models; (iii) extending the derivation to manifolds with boundaries (e.g., simplices).

## Related Work & Insights

- **vs Riemannian SGM (Bortoli et al., 2022)**: RSGM avoids ambient singularity via manifold heat kernels; this work stays in ambient space but achieves a target $O(\sigma^2)$ close to the intrinsic score via Rao-Blackwell.
- **vs Convergence analysis (Pidstrigach, 2022)**: While previous work characterizes ambient score alignment with normal fibers, this paper quantitatively defines what the tangential component actually learns.
- **vs Vincent (2011) DSM+Tweedie**: Extends the classic results to cases where the density is singular with respect to the Lebesgue measure, introducing $r_\sigma$ as the new canonical target.
- **Statistical Foundation for Heuristics**: Provides a rigorous theoretical basis for "project-and-regress" heuristics used in recent literature.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] Riemannian MeanFlow for One-Step Generation on Manifolds](riemannian_meanflow_for_one-step_generation_on_manifolds.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](../../ICCV2025/image_generation/balanced_image_stylization_with_style_matching_score.md)
- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)

</div>

<!-- RELATED:END -->

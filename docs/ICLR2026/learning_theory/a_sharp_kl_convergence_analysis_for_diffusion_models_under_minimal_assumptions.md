---
title: >-
  [Paper Note] A Sharp KL Convergence Analysis for Diffusion Models under Minimal Assumptions
description: >-
  [ICLR 2026][learning_theory][Diffusion Model] This paper provides a sharper KL divergence convergence analysis for diffusion models (DDPM samplers) under the minimal assumption of "only $L^2$ accuracy of score estimation, without assuming any smoothness." By modeling the generation process as "one-step probability flow ODE + one small noise-addition step" and deve
tags:
  - ICLR 2026
  - learning_theory
  - Diffusion Model
date: 2026-05-08
content_hash: 694269addf6ea45a
---
# A Sharp KL Convergence Analysis for Diffusion Models under Minimal Assumptions

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=c8Ft3246KD](https://openreview.net/forum?id=c8Ft3246KD)  
**Code**: None  
**Area**: Learning Theory / Diffusion Model Convergence Analysis  
**Keywords**: Diffusion Models, KL Convergence, Probability Flow ODE, Stochastic Localization, Discretization Error

## TL;DR
This paper provides a sharper KL divergence convergence analysis for diffusion models (DDPM samplers) under the minimal assumption of "only $L^2$ accuracy of score estimation, without assuming any smoothness." By modeling the generation process as "one-step probability flow ODE + one small noise-addition step" and developing new proof techniques to handle the second-order spatial derivatives of the score (Laplacian), the iteration complexity required to achieve $\varepsilon^2$-KL is improved from the previous best $\tilde O(d/\varepsilon^2)$ to $\tilde O(d/\varepsilon)$. This reduces the dependence on accuracy $\varepsilon$ from quadratic to linear while maintaining linear dependence on the dimension $d$.

## Background & Motivation

**Background**: Sampling in diffusion models (score-based generative models) can follow two equivalent paths: simulating a reverse SDE or simulating a probability flow ODE that shares the same marginal distributions at all times. Recent theoretical works focus on analyzing the convergence rate of this generation process, concerning the number of iterations $K$ required for the generated distribution to approximate the true data distribution.

**Limitations of Prior Work**: Early analyses required additional regularity conditions (smooth scores, bounded support of data, etc.). A more realistic line of work shifted to "minimal assumptions"—assuming only that the trained score estimate is accurate in the $L^2$ sense. Under this setting, the current best KL guarantee without any smoothness assumptions (Benton et al. 2024, using reverse SDE) is $\tilde O(d/\varepsilon^2)$ steps to reach an $\epsilon^2$ KL divergence. While the linear dependence on dimension $d$ is satisfactory, the quadratic dependence on $\varepsilon$ appears suboptimal. Another line of work using ODEs (e.g., Li & Yan 2024) achieves a TV distance of $O(d/\varepsilon)$, but since TV is only an upper bound on the square root of KL, its converted KL guarantee is weaker.

**Key Challenge**: Each interval's discretization error in the reverse SDE path is $O(h_k^2)$ ($h_k$ is the step size), which is the root of the quadratic $\varepsilon$ dependence. The probability flow ODE path could theoretically achieve better step size dependence (reducing per-interval error to $O(h_k^3)$), but the error aggregates and explodes when using ODEs directly without smoothing. Previous solutions either relied on Langevin corrector noise for smoothing (Chen et al. 2023b, requiring smoothness assumptions) or additional assumptions on the score's Jacobian/divergence. In other words: **to achieve better $\varepsilon$ dependence, one must use the ODE path, but using the ODE path under "no smoothness assumptions" makes it difficult to keep the dimension $d$ linear.**

**Goal**: Can the dependence of KL on $\varepsilon$ be improved while maintaining linear dependence on $d$, assuming only $L^2$ score estimation accuracy?

**Key Insight**: Borrowing from recent work on second-order/stochastic midpoint discretization (Li & Cai 2024, Li & Jiao 2024, Jain et al. 2025), a generation interval is split into "one step along the reverse ODE + one small noise step in the forward direction." The ODE step controls Wasserstein-type error (without requiring smoothness), and the noise step converts this Wasserstein error into KL divergence with better step size dependence.

**Core Idea**: Replace the reverse SDE/Langevin corrector with an "ODE step + small noise step" to analyze the DDPM sampler. Specifically, develop an analytical framework for this ODE path capable of handling second-order spatial derivatives of the score (Laplacian) to simultaneously achieve $d$-linearity and $\varepsilon$-linearity without smoothness assumptions.

## Method

### Overall Architecture

This is a purely theoretical analysis paper; the "method" refers to the algorithmic setting and proof techniques for convergence, with no new models or training procedures.

Setting: The forward process is a standard OU process $dx(t) = -x(t)dt + \sqrt{2}\,dw_t$, corresponding to $x(t) = e^{-t}y + \sqrt{1-e^{-2t}}\,\epsilon$, where $y\sim p_{\text{data}}$. Reverse generation can use either a reverse SDE or a probability flow ODE $dx(t) = -x(t)dt - s(t,x(t))dt$, where $s(t,\cdot)=\nabla\log p_t$ is the score function, with only an approximation $\hat s$ available.

The object of analysis is the inference algorithm (Algorithm 1): In the interval $[t_{k-1},t_k]$, an Exponential Integrator discretizes the empirical probability flow ODE for "one and a half" steps, followed by adding a small amount of Gaussian noise in the forward direction:
$$\hat x_{k-0.5} = e^{h_k+h_{k-1}}\hat x_k + (e^{h_k+h_{k-1}}-1)\,\hat s(t_k,\hat x_k),\qquad \hat x_{k-1} = e^{-h_{k-1}}\hat x_{k-0.5} + \sqrt{1-e^{-2h_{k-1}}}\,\eta_k.$$
This "ODE step + noise step" combination can be interpreted as an equivalent simulation of the reverse SDE process. The proof chain is: use the chain rule to decompose end-to-end KL into conditional KLs per interval, write each conditional KL as a Wasserstein-type error $\|x_{k-0.5}-\hat x_{k-0.5}\|^2$, and split this error into "discretization error $T_d$ + score estimation error $T_s$." The most difficult part is $T_d$, which involves the total time derivative of the score, leading to the score's second-order spatial derivative (Laplacian)—a term that does not appear in reverse SDE analysis and is the focus of the new techniques.

The final main theorem (Theorem 3.1) provides the bound:
$$\mathrm{KL}(p_{t_1}\,\|\,\hat p_{t_1}) \lesssim (d+m_2)e^{-T} + d^2 c^3 K + T\varepsilon_{\text{score}}^2,$$
where the three terms are initialization error (starting from $N(0,I_d)$), discretization error, and score estimation error. $m_2$ is the second moment of the data, and $c$ is the step size coefficient. With exponentially decaying step sizes $h_k = c\min\{t_k,1\}$, the iteration complexity $K=\Theta\!\big(d\log^{3/2}(1/\delta)/\varepsilon\big)$ compresses the KL to $\tilde O(\varepsilon^2)$.

### Key Designs

**1. ODE Step + Small Noise Step: Superior step size dependence**
Discretization error in reverse SDE paths is typically $\int_{t_{k-1}}^{t_k}\mathbb{E}\|s(t_k,x_k)-s(t,x(t))\|^2 dt$, which, bounded by the score's Jacobian, yields $O(h_k^2)$ dependence—the source of the quadratic $\varepsilon$. Ours switches to the probability flow ODE: controlling Wasserstein error via a deterministic ODE step (requiring no smoothness), then converting it to KL via the forward noise step. The key advantage is that the ODE discretization error is $O(h_k^3)$, superior to the SDE's $O(h_k^2)$. Unlike Chen et al. (2023b), which uses Langevin dynamics as a corrector, Ours simply adds noise in the forward direction, bypassing smoothness requirements for the true or approximate score.

**2. Chain Decomposition + Step-wise Wasserstein-to-KL Conversion**
Using the data processing inequality and the chain rule (Lemma A.2), end-to-end KL is decomposed into an initialization term and the sum of conditional KLs: $\mathrm{KL}(p_{t_1}\|\hat p_{t_1}) \le \mathrm{KL}(p_{t_{K+1}}\|\hat p_{t_{K+1}}) + \sum_k \mathbb{E}\,\mathrm{KL}(p_{t_{k-1}|t_k}\|\hat p_{t_{k-1}|t_k})$. When the true and generative processes start from the same point $x_k$, the single-interval conditional KL is exactly the KL between Gaussians: $\mathrm{KL}(p_{t_{k-1}|t_k}\|\hat p_{t_{k-1}|t_k}) = \frac{e^{-2h_{k-1}}\|x_{k-0.5}-\hat x_{k-0.5}\|_2^2}{2(1-e^{-2h_{k-1}})}$ (Lemma A.1). The problem reduces to bounding the Wasserstein-type quantity $\mathbb{E}\|x_{k-0.5}-\hat x_{k-0.5}\|^2$, further split into $T_d$ and $T_s=(e^{h_k+h_{k-1}}-1)^2\mathbb{E}\|s-\hat s\|^2$.

**3. Fokker-Planck for Converting Time Derivatives to Spatial Derivatives**
To bound $T_d$, the authors use a scale transformation $z(t)=e^t x(t)$ and Taylor residuals to obtain $\mathbb{E}\|z_{k-0.5}-\tilde z_{k-0.5}\|^2 \lesssim (h_k+h_{k-1})^3\int e^{4t}\mathbb{E}\|s_r'(t,z)\|^2 dt$, where $s_r'=\frac{d}{dt}s_r$ is the total derivative. This contains the partial derivative $\partial_t s_r$, which makes the ODE path harder than the SDE. The authors use a score-based Fokker-Planck equation (Lemma A.9) to replace the time derivative: $\partial_t s_r(t,z) = e^{2t}\Delta s_r(t,z) + 2e^{2t}\nabla s_r(t,z)^\top s_r(t,z)$. This yields:
$$\mathbb{E}_{q_t}\|s_r'\|^2 = e^{4t}\,\mathbb{E}_{q_t}\!\big[\|\Delta s_r\|_2^2 + \|\nabla s_r^\top s_r\|_2^2\big] + \mathbb{E}_{q_t}\big[(\Delta s_r)^\top(\nabla s_r^\top s_r)\big].$$
The appearance of the score's Laplacian $\Delta s_r$ (second-order spatial derivative) is a technical hurdle not present in standard SDE or first-order ODE analyses.

**4. Generalizing Stochastic Localization to ODEs for Linear $d$**
Directly applying second-order analysis (e.g., Li & Cai 2024) to DDPM would result in $d^3$ in the integral $\int\mathbb{E}\|s_r'\|^2 dt$, leading to $d^{3/2}$ in the KL—worse than Benton et al. (2024). To recover $d$-linearity, the key observation is that $\mathbb{E}\|s_r\|^2$ is $O(d/(e^{2t}-1))$ while $\mathbb{E}\|\nabla s_r\|_F^2$ is $O(d^2/(e^{2t}-1)^2)$. Benton et al. (2024) achieved linear $d$ in the SDE case by invoking stochastic localization to bound the Jacobian. This paper generalizes those arguments to the ODE path. By proving $\frac{d}{dt}\mathbb{E}_{q_t}[\|s_r\|^2] = -2e^{2t}\mathbb{E}_{q_t}[\|\nabla s_r\|_F^2]$ and deriving generalizations for powers $m$ and new identities for the Laplacian $\mathbb{E}_{q_t}\|\Delta s_r\|^2$, the authors constrain these terms (Lemmas A.11–A.17). This results in $\mathbb{E}_{q_t}\|s_r'\|^2 \lesssim \frac{d^2 e^{4t}}{(e^{2t}-1)^3} - \cdots$, eventually yielding linear iteration complexity in $d$.

### Loss & Training
Ours does not involve training. The only "learning" assumption is the $L^2$ accuracy of the score estimate: $\frac{1}{T}\sum_k h_k\,\mathbb{E}_{x\sim p_{t_k}}\|\hat s(t_k,x)-s(t_k,x)\|^2 \le \varepsilon_{\text{score}}^2$ (Assumption 2.1), plus finite second moment of the data $\mathbb{E}\|x_0\|^2 = m_2 < \infty$ (Assumption 2.2). No assumptions regarding score smoothness, bounded Jacobian, or bounded support are made.

## Key Experimental Results

As a purely theoretical paper, **there are no numerical experiments**. The contribution is the theoretical improvement of the convergence rate. The following table compares iteration complexity to reach target accuracy under "minimal assumptions" ($L^2$ score accuracy only).

### Convergence Rate Comparison (Minimal Assumption Setting)

| Work | Path | Metric | Iteration Complexity | Note |
|------|------|------|-----------|------|
| Benton et al. 2024 | Reverse SDE | KL ($\le\varepsilon^2$) | $\tilde O(d/\varepsilon^2)$ | Previous best KL; $d$ linear, $\varepsilon$ quadratic |
| Li & Yan 2024 | Reverse SDE | TV ($\le\varepsilon$) | $\tilde O(d/\varepsilon)$ | TV metric; weaker when converted to KL |
| Li et al. 2024a | ODE | TV ($\le\varepsilon$) | $O(d/\varepsilon)$ | Requires additional Jacobian assumptions |
| Naive Li & Cai 2024 | ODE + Noise | KL ($\le\varepsilon^2$) | $\tilde O(d^{3/2}/\varepsilon)$ | Improved $\varepsilon$ but $d$ degrades to $3/2$ power |
| **Ours** | ODE + Noise | KL ($\le\varepsilon^2$) | $\tilde O(d/\varepsilon)$ | $d$ linear, $\varepsilon$ linear; New SOTA |

### Key Findings
- **$\varepsilon$ dependence reduced from quadratic to linear**: Compared to Benton et al. (2024), Ours achieves $\tilde O(d/\varepsilon)$, the new optimal bound for KL convergence under minimal assumptions.
- **ODE path does not sacrifice dimension**: Naively applying second-order analysis to DDPM degrades $d$ to $d^{3/2}$; Ours restores linear $d$ via new identities for score spatial derivatives, showing that superior step size dependence and $d$-linearity can coexist.
- **Automatic TV guarantee from KL**: Via Pinsker’s inequality $\mathrm{TV}^2\le\frac{1}{2}\mathrm{KL}$, the KL bound provides a stronger TV convergence guarantee than Li & Yan (2024).

## Highlights & Insights
- **Systematic conversion of time derivatives to spatial derivatives**: Using the Fokker-Planck equation to rewrite $\partial_t s_r$ is the key to handling total derivative terms in the probability flow ODE. This approach is transferable to the analysis of other deterministic samplers like Consistency Models.
- **Generalizing stochastic localization to ODEs**: While Benton et al. (2024) only needed to bound the Jacobian for SDEs, the ODE path generates Laplacian terms. This paper identifies and resolves this through a family of identities for $\frac{d}{dt}\mathbb{E}\|s_r\|^m$, representing a genuine technical advancement.
- **Minimalist yet effective algorithm**: Replacing the Langevin corrector with a "single ODE step + small noise addition" preserves the step size advantages of ODEs while bypassing smoothness requirements—an insightful "noise-for-smoothness" trade-off.

## Limitations & Future Work
- **Theoretical upper bound only**: No numerical experiments verify the $\tilde O(d/\varepsilon)$ rate in actual samplers, and constant factors are not quantified.
- **Early stopping requirement**: The bound is relative to $p_{t_1}$ at time $t_1>\delta>0$ (data perturbed by variance $\delta$ noise), a standard trade-off to avoid data distribution smoothness assumptions.
- **Potential for further $\varepsilon$ optimization**: It remains to be explored whether the "ODE + Noise" framework can further improve step size dependence to achieve even faster convergence.
- **Solver dependency**: Results are tied to the Exponential Integrator and OU forward process; applicability to other noise schedules or solvers is not explored.

## Related Work & Insights
- **vs. Benton et al. (2024)**: Both pursue linear $d$ for KL under minimal assumptions. Benton uses reverse SDE ($O(h^2)$ step dependence, $\varepsilon$ quadratic); Ours uses ODE ($O(h^3)$ step dependence, $\varepsilon$ linear) at the cost of handling Laplacian terms. This is a strict improvement.
- **vs. Li & Yan (2024)**: Li & Yan provide TV distance $O(d/\varepsilon)$. Ours achieves the same rate for the stronger KL divergence.
- **vs. Chen et al. (2023b)**: Chen uses ODE steps + Langevin corrector but requires score smoothness. Ours replaces the corrector with forward noise, achieving the same Wasserstein-to-KL transition without smoothness.
- **vs. Li & Cai (2024) / Jain et al. (2025)**: These works proposed "ODE + Noise" for second-order/midpoint settings with smoothness. Ours adapts this to DDPM under minimal assumptions and introduces techniques to prevent $d^{3/2}$ degradation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to reduce KL convergence dependence on $\varepsilon$ from quadratic to linear under minimal assumptions.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical, but comparisons are clear and well-positioned.
- Writing Quality: ⭐⭐⭐⭐ Motivation is logical, and the proof sketch clearly explains the technical hurdles.
- Value: ⭐⭐⭐⭐ Establishes a new SOTA for KL convergence theory in diffusion models.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Polynomial Convergence of Riemannian Diffusion Models](polynomial_convergence_of_riemannian_diffusion_models.md)
- [\[ICLR 2026\] Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis](best-of-n_through_the_smoothing_lens_kl_divergence_and_regret_analysis.md)
- [\[ICLR 2026\] Finite-Time Convergence Analysis of ODE-based Generative Models for Stochastic Interpolants](finite-time_convergence_analysis_of_ode-based_generative_models_for_stochastic_i.md)
- [\[ICLR 2026\] On the Interpolation Effect of Score Smoothing in Diffusion Models](on_the_interpolation_effect_of_score_smoothing_in_diffusion_models.md)
- [\[ICLR 2026\] Provable Separations between Memorization and Generalization in Diffusion Models](provable_separations_between_memorization_and_generalization_in_diffusion_models.md)

</div>

<!-- RELATED:END -->

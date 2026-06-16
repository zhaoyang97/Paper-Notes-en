---
title: >-
  [Paper Note] Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures
description: >-
  [ICML 2026][Image Generation][Paper Note] The authors upgrade diffusion model inference-time reward guidance from "particle-space SMC + higher-order derivatives" to "path-space SMC + Girsanov likelihood ratio," resulting in the URGE algorithm. Each trajectory only requires a first-order gradient of the guidance $G$ and accumulates a simple Itô term as weight,
tags:
  - ICML 2026
  - Image Generation
date: 2026-05-08
content_hash: 2b82352751c6f4ff
---
# Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures

**Conference**: ICML 2026  
**arXiv**: [2605.17850](https://arxiv.org/abs/2605.17850)  
**Code**: Not released  
**Area**: Diffusion Models / Inference-time Guidance / Sequential Monte Carlo  
**Keywords**: Inference-time scaling, Girsanov Theorem, Path-space SMC, Derivative-free guidance, reward-tilted sampling  

## TL;DR
The authors upgrade diffusion model inference-time reward guidance from "particle-space SMC + higher-order derivatives" to "path-space SMC + Girsanov likelihood ratio," resulting in the URGE algorithm. Each trajectory only requires a first-order gradient of the guidance $G$ and accumulates a simple Itô term as weight, completely eliminating the need for derivatives, Hessians, or score estimation of the reward $r$. It matches or outperforms FK-Corrector, AFDPS, and FK-Steering on Gaussian Mixture Models (GMM), inverse problems, and text-to-image tasks.

## Background & Motivation

**Background**: Diffusion models view generation as an SDE $dX_t = v(X_t,t)dt + V(t)dW_t$. In practice, there is often a need to "tilt" samples toward a specific reward $\mathbf{r}(x)$ without fine-tuning, where the target distribution is the reward-tilted posterior $q(x)\propto p_\text{data}(x)\mathbf{r}(x)$. The mainstream approach is guidance—modifying the drift to $v + V^2\nabla_x G$ to approximate this posterior.

**Limitations of Prior Work**: Standard guidance does not sample from the true $q$ because a rigorous approach requires the Doob $h$-transform $h(x,t)=\mathbb{E}[\mathbf{r}(X_T)\mid X_t=x]$. Computing $h$ requires solving a backward Kolmogorov equation, which is intractable in high dimensions. Recent refinement schemes (FK-Corrector, AFDPS) follow a "particle-space SMC" route, calculating unbiased weights for each particle and then resampling. However, these weights involve higher-order terms like $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x \log p_t$, requiring second derivatives of the reward and score function evaluations. This becomes a bottleneck when using black-box neural reward models (e.g., ImageReward, HPS).

**Key Challenge**: The gap between the desire for "unbiased reward-tilted sampling" and the "actually computable weight terms." Unbiased correction in particle space naturally involves the generator $\mathcal{L}^G$, and applying the generator to $r$ inevitably introduces second derivatives.

**Goal**: Find a weight construction method that retains the unbiasedness of SMC without requiring reward derivatives, allowing black-box neural rewards to be used directly.

**Key Insight**: The authors shift from the framework of "weighting particles at each time step" to "weighting the entire trajectory." Since the path measure ratio between the guided SDE and the reference SDE can be expressed in closed-form via the Girsanov Theorem, SMC can be performed directly in the path space.

**Core Idea**: Use the Girsanov path likelihood ratio $\mathrm{d}\mathbb{P}/\mathrm{d}\mathbb{P}^G$ multiplied by $\mathrm{d}\mathbb{Q}/\mathrm{d}\mathbb{P}=\exp(r(X_t)-r(X_0))$ as the importance weight for the trajectory. The weights only contain $\nabla_x G$ (the gradient of the guidance itself, which is already computed) and the difference in $r$, without any derivatives of $r$.

## Method

### Overall Architecture
URGE aims to solve the problem of unbiased sampling from the reward-tilted posterior $q(x)\propto p_\text{data}(x)e^{\mathbf{r}(x)}$ without fine-tuning a pre-trained diffusion model. Its mechanism is as follows: rather than assigning a weight containing second derivatives to a single particle at each moment (as in FK-Corrector / AFDPS), it calculates the Girsanov path likelihood ratio once for the entire trajectory as the weight. Specifically, given a drift $v(x,t)$, guidance potential $G(x,t)$, and reward $r(x,t)$ (where $r(x,T)=\mathbf{r}(x)$), the algorithm simulates $N$ guided trajectories in parallel. After each Euler-Maruyama (EM) step $\Delta t$, each trajectory is multiplied by a weight $\beta^{(i)}$ that depends only on $\nabla_x G$ and the difference in $r$. Then, Categorical resampling is performed based on normalized $\beta$ to replicate high-weight particles and eliminate low-weight ones. After $K$ steps, the terminal particles $\{X_T^{(i)}\}$ serve as approximate samples from $q$, becoming strictly unbiased as $\Delta t\to 0$ and $N\to\infty$.

### Key Designs

**1. Path-space Girsanov weights: Packaging second derivatives into first-order quantities and incorporating stochastic path information**  

Existing particle-space corrections (FK-Corrector / AFDPS) require $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x\log p_t$. When facing black-box neural rewards like ImageReward or HPS, these methods fail because the Hessian cannot be computed. URGE circumvents this by moving the unbiasedness requirement to the path measure: using the Girsanov Theorem, the closed-form ratio between the reference measure $\mathbb{P}$ (without guidance) and the guided measure $\mathbb{P}^G$ is written as $\mathrm{d}\mathbb{P}/\mathrm{d}\mathbb{P}^G \propto \exp(-\int_0^t V(s)\nabla_x G^\top dW_s - \tfrac{1}{2}\int_0^t V^2\|\nabla_x G\|^2 ds)$. Multiplying this by the density of the reward-tilted measure relative to $\mathbb{P}$ ($\exp(r(X_t)-r(X_0))$) yields the target weight $\mathrm{d}\mathbb{Q}/\mathrm{d}\mathbb{P}^G$. After EM discretization, this becomes:
$$\beta^{(i)}_{s,t}=\exp\!\big(r(X_t)-r(X_s) - V(s)\nabla_x G^\top\sqrt{t-s}\,\xi^{(i)} - \tfrac{1}{2}V(s)^2\|\nabla_x G\|^2(t-s)\big)$$
where $\xi^{(i)}$ is the Gaussian noise already sampled during the EM step, allowing for reuse with nearly zero extra cost. The expression contains only $\nabla_x G$ and numerical differences of $r$, with absolutely no reward derivatives. This allows URGE to be the first method to interface directly with black-box neural scorers. Moreover, the first term $-V(s)\nabla_x G^\top\sqrt{t-s}\,\xi^{(i)}$ is an Itô integral (continuous form $\int_s^t -V(\tau)\nabla_x G^\top dW_\tau$) that reintroduces the noise $\xi^{(i)}$ into the weight. In contrast, AFDPS and FK-Corrector weights are deterministic functions of the endpoint $x$, assuming "any path reaching the same endpoint is equally valid," which discards critical stochasticity from the diffusion process. In URGE, two trajectories reaching the same endpoint—one "drifting with guidance" and another "colliding against noise"—receive different weights, leading to more precise resampling, lower variance, and more stable scaling with the number of particles $N$.

**2. Path-particle equivalence theorem: Proving URGE is the parent of AFDPS rather than another approximation**  

To demonstrate that moving to path space does not introduce further approximations, the authors define an instantaneous intensity $\lambda(x,t):=\lim_{h\to 0}\tfrac{1}{h}\big(\mathbb{E}_{\mathbb{P}^G}[w^\text{URGE}_{t-h,t}\mid X_t=x]-1\big)$. Using the Feynman-Kac backward value function, they derive the marginalized generator $\mathcal{L}^\text{eff}_t = \mathcal{L}^G_t + \lambda(\cdot,t)$. In Theorem 3.3, they prove that $\lambda(x,t)\equiv w_\text{AFDPS}(x,t)$. This implies that taking the conditional expectation of URGE’s path weights at the endpoint exactly recovers all second-order terms of AFDPS. This equivalence ensures that URGE inherits the unbiasedness of AFDPS while highlighting that AFDPS is merely a special case after conditional expectation. Thus, path space offers greater design flexibility for higher-order discretization and sparser time grids.

### Loss & Training
URGE is a pure inference-time algorithm and **requires no additional training**. Hyperparameters include the number of particles $N$, the number of discretization steps $K$, and the guidance strength (typically setting $G=r$ or using CFG terms in text-to-image). The simplest version with EM discretization works effectively, and the weight construction can be replaced with higher-order formats to improve accuracy when $N$ is limited.

## Key Experimental Results

### Main Results

30-dimensional 40-component GMM toy task (reward chosen as a known quadratic function for analytical comparison):

| Method | MMD↓ | SWD↓ | Mean $\ell_2$↓ | Cov Frob↓ |
|------|------|------|----------------|-----------|
| Pure Guidance | 0.17 | 1.68 | 7.14 | 469.09 |
| AFDPS | 0.10 | 1.04 | 5.07 | 335.19 |
| AFDPS+VCG | 0.08 | 0.83 | 4.13 | 246.61 |
| FK-Steering | 0.07 | 0.85 | 4.86 | 198.20 |
| **URGE** | **0.06** | **0.62** | **3.20** | **181.31** |

URGE outperformed other methods across all four metrics, particularly achieving a 26% lower covariance Frobenius error than AFDPS+VCG, without requiring additional learned control drifts.

Four inverse problems on ImageNet-256 (PSNR↑/LPIPS↓):

| Method | Gaussian Deblur PSNR | Motion Deblur LPIPS | Super-Res PSNR | Box Inpaint LPIPS |
|------|----------------------|---------------------|----------------|--------------------|
| SGS-EDM | 22.09 | 0.526 | 15.43 | 0.298 |
| FK-Corrector | 18.36 | 0.601 | 18.58 | 0.714 |
| AFDPS-SDE | 22.43 | 0.520 | 21.03 | 0.307 |
| AFDPS-ODE | 22.57 | 0.503 | 19.60 | **0.275** |
| **URGE** | 22.38 | 0.525 | 21.00 | 0.305 |

URGE performed on par with the strongest AFDPS variants and significantly exceeded FK-Corrector.

### Ablation Study

Text-to-Image (Stable Diffusion v1.5, 50 prompts × 3 seeds):

| Sampler | CLIP-Score↑ | HPS↑ | ImageReward↑ | GenEval↑ |
|--------|-------------|------|--------------|----------|
| Base $N=1$ | 0.273 | 0.262 | 0.214 | 0.640 |
| Grad Guidance $N=1$ | 0.273 | 0.262 | 0.207 | 0.640 |
| FK-Steering $N=4$ | 0.290 | 0.285 | 0.840 | 0.720 |
| Gradient FK $N=4$ | 0.290 | 0.284 | 0.791 | 0.747 |
| **URGE $N=4$** | **0.300** | **0.293** | **0.996** | **0.780** |

ImageReward jumped from a base of 0.21 to 0.996 ($\approx 4.7\times$). Metrics for CLIP, HPS, and GenEval were also leading. The authors noted that SDv1.5 + URGE often matched or surpassed SDXL baselines on dual-object prompts.

### Key Findings
- **Derivative-free maintains accuracy**: After removing $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x \log p_t$, URGE was more accurate in GMM and inverse problems than AFDPS, which retains these terms. This suggests that the stochastic information from the Itô path term compensates for the removal of second-order terms.
- **Monotonic particle scaling**: Results showed ImageReward growing monotonically with $N$, whereas FK-Steering plateaued at larger $N$, indicating more stable weight variance in path space.
- **Small model outperforms large model**: SDv1.5 + URGE ($N=4$) achieved a higher ImageReward than base SDXL, implying that adding SMC resampling may be more cost-effective than using a larger model for the same compute budget.
- **Black-box reward is the killer feature**: URGE is the only SMC solution compatible with black-box neural scorers like ImageReward/HPS.

## Highlights & Insights
- **Measure-theoretic substitution**: Changing from "weighting particles" to "weighting trajectories" seems like a notation change, but the Girsanov Theorem packages all second-order terms from the infinitesimal generator into an Itô integral. This simplifies engineering to "reusing noise from the EM step."
- **Explanatory power of the equivalence theorem**: Theorem 3.3 serves as both a proof of correctness and a justification for design freedom—since AFDPS is a special case of URGE, URGE naturally allows for more options such as higher-order formats or different time grids.
- **Transferable techniques**: The Feynman-Kac duality argument that "path weight $\equiv$ particle weight conditioned on the endpoint" can be applied to any SMC-based diffusion algorithm, such as reward fine-tuning or molecular conformation sampling.

## Limitations & Future Work
- Experiments were limited to $N=4 \sim 16$; it is unclear if performance saturates at $N \approx 100$ due to particle degeneracy.
- The weights still contain $\nabla_x G$, so "derivative-free" specifically refers to "no reward derivatives." If $G$ also becomes a black-box, URGE loses its unbiasedness.
- Discretization step $\Delta t$ must be small enough for Girsanov stability; an adaptive $\Delta t$ scheme was not provided, which could be a bottleneck for long-horizon video diffusion.
- Future work: Coupling URGE with higher-order SDE solvers (Heun / DPM-Solver-2); studying URGE variants for non-smooth rewards (e.g., discrete GenEval metrics); extending URGE to jump diffusion for categorical or discrete diffusion models.

## Related Work & Insights
- **vs FK-Corrector (Skreta et al., 2025) / AFDPS (Chen et al., 2025)**: They modify the generator in particle space using weights containing second-order terms. URGE uses Girsanov weights in path space, which are simpler to implement and support black-box rewards.
- **vs FK-Steering (Singhal et al., 2025)**: FK-Steering uses reward differences but omits the Girsanov path correction term, making it biased.
- **vs Doob $h$-transform methods (DEFT / adjoint matching)**: These typically require training a network to estimate $h$; URGE is training-free.
- **vs VCG (Ren et al., 2025a)**: VCG learns a control drift to reduce variance; URGE achieves lower error in GMM tasks without learning.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[CVPR 2026\] Tiny Inference-Time Scaling with Latent Verifiers](../../CVPR2026/image_generation/tiny_inference-time_scaling_with_latent_verifiers.md)
- [\[ICML 2026\] SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate](surge_approximation_and_training_free_particle_filter_for_diffusion_surrogate.md)
- [\[CVPR 2026\] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation](../../CVPR2026/image_generation/rethinking_prompt_design_for_inference-time_scaling_in_text-to-visual_generation.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](../../ICML2025/image_generation/performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](../../CVPR2026/image_generation/dpcache_denoising_path_planning_diffusion_accel.md)

</div>

<!-- RELATED:END -->

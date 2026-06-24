---
title: >-
  [Paper Note] Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures
description: >-
  [ICML 2026][Image Generation][Inference-time scaling] The authors upgrade inference-time reward guidance for diffusion models from "particle-space SMC + high-order derivatives" to "path-space SMC + Girsanov likelihood ratios," resulting in the URGE algorithm. Each trajectory only requires a first-order gradient of the guidance $G$ and an accumulated simple Itô term as weight, completely eliminating the need for derivatives of the reward $r$, the Hessian…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Inference-time scaling"
  - "Girsanov Theorem"
  - "Path-space SMC"
  - "Derivative-free guidance"
  - "reward-tilted sampling"
date: 2026-05-08
content_hash: 402f503cdf777910
---

# Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures

**Conference**: ICML 2026  
**arXiv**: [2605.17850](https://arxiv.org/abs/2605.17850)  
**Code**: Not released  
**Area**: Diffusion Models / Inference-time Guidance / Sequential Monte Carlo  
**Keywords**: Inference-time scaling, Girsanov Theorem, Path-space SMC, Derivative-free guidance, reward-tilted sampling  

## TL;DR
The authors upgrade inference-time reward guidance for diffusion models from "particle-space SMC + high-order derivatives" to "path-space SMC + Girsanov likelihood ratios," resulting in the URGE algorithm. Each trajectory only requires a first-order gradient of the guidance $G$ and an accumulated simple Itô term as weight, completely eliminating the need for derivatives of the reward $r$, the Hessian, or score estimation. It matches or exceeds FK-Corrector / AFDPS / FK-Steering on GMM, inverse problems, and text-to-image tasks.

## Background & Motivation

**Background**: Diffusion models view generation as an SDE $dX_t = v(X_t,t)dt + V(t)dW_t$. During deployment, it is often necessary to "tilt" samples toward a reward $\mathbf{r}(x)$ without fine-tuning, where the target distribution is the reward-tilted posterior $q(x)\propto p_\text{data}(x)\mathbf{r}(x)$. The standard approach is guidance—modifying the drift to $v + V^2\nabla_x G$ to approximate this posterior.

**Limitations of Prior Work**: Actual guidance does not sample from the true $q$ because the rigorous approach requires a Doob $h$-transform $h(x,t)=\mathbb{E}[\mathbf{r}(X_T)\mid X_t=x]$, where $h$ solves a backward Kolmogorov equation, which is intractable in high dimensions. Recent correction schemes (FK-Corrector, AFDPS) use "particle-space SMC" to compute an unbiased weight for each particle and resample. However, these weights contain high-order terms like $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x \log p_t$, requiring second-order derivatives of the reward and score evaluations, which fail for black-box neural scorers (e.g., ImageReward, HPS).

**Key Challenge**: The gap between "the desire for unbiased reward-tilted sampling" and "the actually computable weight terms": unbiased corrections in particle space naturally involve the generator $\mathcal{L}^G$, and the generator acting on $r$ inevitably introduces second-order derivatives.

**Goal**: Find a weight construction that maintains SMC unbiasedness without requiring reward derivatives, enabling the direct use of black-box neural rewards.

**Key Insight**: Instead of weighting particles at each time step, weight the entire trajectory. Since the path measure ratio between the guided SDE and the reference SDE has a closed-form expression via the Girsanov theorem, SMC can be performed directly in the path space.

**Core Idea**: Use the Girsanov path likelihood ratio $\mathrm{d}\mathbb{P}/\mathrm{d}\mathbb{P}^G$ multiplied by $\mathrm{d}\mathbb{Q}/\mathrm{d}\mathbb{P}=\exp(r(X_t)-r(X_0))$ as the importance weight for trajectories. The weight only involves $\nabla_x G$ (the gradient of the guidance itself, which is already computed) and differences in $r$, without any derivatives of $r$.

## Method

### Overall Architecture
URGE addresses unbiased sampling for reward-tilted posteriors $q(x)\propto p_\text{data}(x)e^{(x)}$ without fine-tuning. Its logic is: instead of assigning high-order derivative weights to particles at each step (like FK-Corrector or AFDPS), it computes a Girsanov path likelihood ratio once per trajectory. Specifically, given drift $v(x,t)$, guidance potential $G(x,t)$, and reward $r(x,t)$ (where $r(x,T)=\mathbf{r}(x)$), the algorithm parallelly simulates $N$ guided trajectories. After each Euler-Maruyama step $\Delta t$, each trajectory is multiplied by a weight $\beta^{(i)}$ that depends only on $\nabla_x G$ and differences in $r$. Particles are then resampled according to normalized $\beta$ using Categorical weights to replicate high-weight particles and prune low-weight ones. After $K$ steps, the final particles $\{X_T^{(i)}\}$ approximate $q$ and are strictly unbiased as $\Delta t\to 0$ and $N\to\infty$.

### Key Designs

**1. Path-space Girsanov weights: Packaging second-order derivatives into first-order terms and incorporating stochastic information**

Prior particle-space corrections (FK-Corrector / AFDPS) require $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x\log p_t$, making them unusable for black-box neural rewards like ImageReward / HPS where the Hessian is unavailable. URGE shifts the unbiasedness requirement to path measures: it uses the Girsanov theorem to write the closed-form ratio between the reference measure $\mathbb{P}$ and the guided measure $\mathbb{P}^G$ as $\mathrm{d}\mathbb{P}/\mathrm{d}\mathbb{P}^G \propto \exp(-\int_0^t V(s)\nabla_x G^\top dW_s - \tfrac{1}{2}\int_0^t V^2\|\nabla_x G\|^2 ds)$. Combined with the reward-tilted density $\exp(r(X_t)-r(X_0))$, the target weight $\mathrm{d}\mathbb{Q}/\mathrm{d}\mathbb{P}^G$ is obtained. After Euler-Maruyama discretization, it becomes $\beta^{(i)}_{s,t}=\exp\!\big(r(X_t)-r(X_s) - V(s)\nabla_x G^\top\sqrt{t-s}\,\xi^{(i)} - \tfrac{1}{2}V(s)^2\|\nabla_x G\|^2(t-s)\big)$, where $\xi^{(i)}$ is the Gaussian noise already sampled for the EM step, resulting in zero extra overhead. The expression contains only $\nabla_x G$ (already used for guidance) and numerical differences of $r$, effectively making the method derivative-free. Furthermore, the term $-V(s)\nabla_x G^\top\sqrt{t-s}\,\xi^{(i)}$ is an Itô integral that incorporates the noise $\xi^{(i)}$ into the weight. Unlike AFDPS / FK-Corrector, which use deterministic functions of the terminal point $x$, URGE distinguishes between two trajectories reaching the same endpoint based on whether they "drifted with guidance" or "forced by noise," leading to lower variance and better scaling with $N$.

**2. Path-particle equivalence theorem: Proving URGE is the parent of AFDPS rather than another approximation**

To show that shifting to path space does not introduce new approximations, the authors define an instantaneous intensity $\lambda(x,t):=\lim_{h\to 0}\tfrac{1}{h}\big(\mathbb{E}_{\mathbb{P}^G}[w^\text{URGE}_{t-h,t}\mid X_t=x]-1\big)$. Using the Feynman-Kac backward value function, they derive the marginalized generator $\mathcal{L}^\text{eff}_t = \mathcal{L}^G_t + \lambda(\cdot,t)$ and prove in Theorem 3.3 that $\lambda(x,t)\equiv w_\text{AFDPS}(x,t)$. This implies that the conditional expectation of URGE path weights given the endpoint exactly recovers all second-order terms in AFDPS. This equivalence ensures URGE inherits the unbiasedness of AFDPS while retaining greater design flexibility (e.g., higher-order discretization or sparser grids).

### Loss & Training
URGE is a pure inference-time algorithm and **requires no additional training**. Hyperparameters include the number of particles $N$, discretization steps $K$, and guidance strength (standardly $G=r$ or CFG terms). The paper demonstrates that the simplest EM discretization (Equation 7) works well, noting that weight construction can be replaced with higher-order schemes if $N$ is limited.

## Key Experimental Results

### Main Results

30-dimensional 40-component GMM toy task (reward chosen as a known quadratic for analytical ground truth):

| Method | MMD↓ | SWD↓ | Mean $\ell_2$↓ | Cov Frob↓ |
|------|------|------|----------------|-----------|
| Pure Guidance | 0.17 | 1.68 | 7.14 | 469.09 |
| AFDPS | 0.10 | 1.04 | 5.07 | 335.19 |
| AFDPS+VCG | 0.08 | 0.83 | 4.13 | 246.61 |
| FK-Steering | 0.07 | 0.85 | 4.86 | 198.20 |
| **URGE** | **0.06** | **0.62** | **3.20** | **181.31** |

URGE outperforms on all four metrics, notably achieving 26% lower Covariance Frobenius error than AFDPS+VCG without requiring the additional drift-control training of VCG.

ImageNet-256 inverse problems (PSNR↑/LPIPS↓):

| Method | Gaussian Deblur PSNR | Motion Deblur LPIPS | Super-Res PSNR | Box Inpaint LPIPS |
|------|----------------------|---------------------|----------------|--------------------|
| SGS-EDM | 22.09 | 0.526 | 15.43 | 0.298 |
| FK-Corrector | 18.36 | 0.601 | 18.58 | 0.714 |
| AFDPS-SDE | 22.43 | 0.520 | 21.03 | 0.307 |
| AFDPS-ODE | 22.57 | 0.503 | 19.60 | **0.275** |
| **URGE** | 22.38 | 0.525 | 21.00 | 0.305 |

URGE matches the strongest AFDPS variants (though $\nabla\log p_t$ as reward still requires a score function, the path weight construction remains derivative-free) and significantly outperforms FK-Corrector.

### Ablation Study

Text-to-image (Stable Diffusion v1.5, 50 prompts × 3 seeds):

| Sampler | CLIP-Score↑ | HPS↑ | ImageReward↑ | GenEval↑ |
|--------|-------------|------|--------------|----------|
| Base $N=1$ | 0.273 | 0.262 | 0.214 | 0.640 |
| Grad-Guidance $N=1$ | 0.273 | 0.262 | 0.207 | 0.640 |
| FK-Steering $N=4$ | 0.290 | 0.285 | 0.840 | 0.720 |
| Grad FK $N=4$ | 0.290 | 0.284 | 0.791 | 0.747 |
| **URGE $N=4$** | **0.300** | **0.293** | **0.996** | **0.780** |

ImageReward jumps from 0.21 (base) to 0.996 ($\times 4.7$), with CLIP / HPS / GenEval also leading. SDv1.5 + URGE often matches or exceeds the SDXL baseline on dual-object prompts.

### Key Findings
- **Derivative-free maintains accuracy**: Removing $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x \log p_t$ results in URGE performing better on GMM and inverse problems than AFDPS, suggesting Itô stochastic information compensates for the lack of second-order terms.
- **Monotonic particle scaling**: Figure 3 shows ImageReward grows monotonically with $N$, while FK-Steering plateaus, indicating more stable weight variance in path space.
- **Small models can surpass large models**: SDv1.5 + URGE ($N=4$) yields higher ImageReward than base SDXL, suggesting that adding SMC resampling might be more cost-effective than using larger models for the same compute budget.
- **Black-box rewards as the killer app**: FK-Corrector / AFDPS are unusable in the text-to-image setting (cannot compute Hessian of neural rewards), making URGE the only SMC solution capable of directly using ImageReward / HPS.

## Highlights & Insights
- **Measure-theoretic transformation**: Replacing particle weighting with path weighting via the Girsanov theorem packages all second-order infinitesimal generator terms into an Itô integral. This simplifies implementation: "just reuse EM noise $\xi^{(i)}$."
- **Equivalence theorem's explanatory power**: Theorem 3.3 serves both as a proof of correctness and a rationale for design flexibility—since AFDPS is a special case of marginalized URGE, URGE naturally opens avenues for sparse schedules and higher-order formats.
- **Transferable techniques**: The "path weight $\equiv$ conditional expectation of particle weight" argument can be applied to any SMC-based diffusion inference, such as reward fine-tuning, molecular conformation sampling, or protein inverse folding.

## Limitations & Future Work
- Experiments only scales up to $N=4 \sim 16$; the degree of particle degeneracy at large $N$ (e.g., $\sim 100$) was not systematically tested.
- Weights still contain $\nabla_x G$, so "derivative-free" strictly means "no reward derivatives." Guidance potential $G$ must remain differentiable (standard in CFG).
- Discrete step size $\Delta t$ must be small for Girsanov stability. No adaptive $\Delta t$ scheme is provided, which might be a bottleneck for long-horizon video diffusion.
- Equivalence holds in the $N\to\infty$ and continuous-time limit; variance bounds for finite $N$ were not explicitly derived.
- Future Work: Coupling URGE path weights with high-order SDE solvers (Heun / DPM-Solver-2); studying URGE variants for non-smooth rewards; extension to jump diffusions for categorical/discrete diffusion.

## Related Work & Insights
- **vs FK-Corrector (Skreta et al., 2025) / AFDPS (Chen et al., 2025)**: These use $\mathcal{L}^G + w_\text{AFDPS}$ in particle space with weights involving $\Delta_x r$, etc. URGE uses Girsanov weights in path space, which are theoretically equivalent but simpler to implement and black-box-reward friendly.
- **vs FK-Steering (Singhal et al., 2025)**: FK-Steering uses $r(X_{t+\Delta t})-r(X_t)$ as weight, omitting the Girsanov path correction term, making it biased. URGE includes the path term for guaranteed unbiasedness and superior performance.
- **vs Doob $h$-transform methods (DEFT / Adjoint matching)**: These require training a network to estimate $h$. URGE is training-free, costing only $N$ times more forward passes at inference.
- **vs VCG (Ren et al., 2025a)**: VCG learns a control drift via weighted least squares to reduce variance. URGE achieves 26% lower error on GMM without any learning, proving path-space stability.

## Rating
- Novelty: ⭐⭐⭐⭐ Path-space SMC + Girsanov is new for diffusion scaling, though Girsanov IS is classic in finance/molecular dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers GMM, inverse problems, and text-to-image, though $N$ range is small and lacks detailed runtime comparison.
- Writing Quality: ⭐⭐⭐⭐ Clearly organizes Girsanov, Feynman-Kac, and Kolmogorov backward; Table 1 effectively differentiates weights.
- Value: ⭐⭐⭐⭐⭐ Derivative-free and black-box reward compatibility means ImageReward / HPS can be plugged in directly for any scenario (video, 3D, etc.). High reuse value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Scaling of Diffusion Models Through Classical Search](../../ICLR2026/image_generation/inference-time_scaling_of_diffusion_models_through_classical_search.md)
- [\[ICML 2026\] SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate](surge_approximation_and_training_free_particle_filter_for_diffusion_surrogate.md)
- [\[ICLR 2026\] Compositional Visual Planning via Inference-Time Diffusion Scaling](../../ICLR2026/image_generation/compositional_visual_planning_via_inference-time_diffusion_scaling.md)
- [\[CVPR 2026\] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation](../../CVPR2026/image_generation/rethinking_prompt_design_for_inference-time_scaling_in_text-to-visual_generation.md)
- [\[ICLR 2026\] One Step Further with Monte-Carlo Sampler to Guide Diffusion Better](../../ICLR2026/image_generation/one_step_further_with_monte-carlo_sampler_to_guide_diffusion_better.md)

</div>

<!-- RELATED:END -->

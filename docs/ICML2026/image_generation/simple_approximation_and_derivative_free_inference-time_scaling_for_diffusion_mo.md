---
title: >-
  [Paper Note] Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures
description: >-
  [ICML 2026][Image Generation][Inference-time Scaling] The authors upgrade inference-time reward guidance for diffusion models from "particle-space SMC + higher-order derivatives" to "path-space SMC + Girsanov likelihood…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Inference-time Scaling"
  - "Girsanov Theorem"
  - "Path-space SMC"
  - "Derivative-free Guidance"
  - "Reward-tilted Sampling"
date: 2026-05-08
content_hash: c4ff8d0789ee2e82
---

# Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures

**Conference**: ICML 2026  
**arXiv**: [2605.17850](https://arxiv.org/abs/2605.17850)  
**Code**: Not released  
**Area**: Diffusion Models / Inference-time Guidance / Sequential Monte Carlo  
**Keywords**: Inference-time Scaling, Girsanov Theorem, Path-space SMC, Derivative-free Guidance, Reward-tilted Sampling  

## TL;DR
The authors upgrade inference-time reward guidance for diffusion models from "particle-space SMC + higher-order derivatives" to "path-space SMC + Girsanov likelihood ratio," resulting in the URGE algorithm. Each trajectory only requires the first-order gradient of the guidance $G$ and the accumulation of a simple Itô term as weight, completely eliminating the need for derivatives/Hessian of the reward $r$ or score estimation. URGE matches or outperforms FK-Corrector / AFDPS / FK-Steering across GMM, inverse problems, and text-to-image tasks.

## Background & Motivation

**Background**: Diffusion models treat generation as an SDE $dX_t = v(X_t,t)dt + V(t)dW_t$. During deployment, it is often necessary to "nudge" samples toward a specific reward $\mathbf{r}(x)$ without fine-tuning, where the target distribution is the reward-tilted posterior $q(x)\propto p_\text{data}(x)\mathbf{r}(x)$. The prevailing approach is guidance—modifying the drift to $v + V^2\nabla_x G$ to approximate this posterior.

**Limitations of Prior Work**: Standard guidance does not actually sample from the true $q$ because a strict approach requires the Doob $h$-transform $h(x,t)=\mathbb{E}[\mathbf{r}(X_T)\mid X_t=x]$, and $h$ requires solving a backward Kolmogorov equation, which is practically impossible in high dimensions. Recent correction schemes (FK-Corrector, AFDPS) follow the "particle-space SMC" route, calculating an unbiased weight for each particle followed by resampling. However, these weights involve higher-order terms like $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x \log p_t$, requiring second-order derivatives of the reward and score function evaluations, which fail when using black-box neural reward models (e.g., ImageReward, HPS).

**Key Challenge**: The gap between the desire for "unbiased reward-tilted sampling" and the "practically computable weight terms." Unbiased corrections in particle space naturally involve the generator $\mathcal{L}^G$, and applying the generator to $r$ inevitably introduces second-order derivatives.

**Goal**: To find a weight construction method that preserves the unbiased nature of SMC without requiring reward derivatives, enabling the direct use of black-box neural rewards.

**Key Insight**: The authors shift from the framework of "weighting particles at each time step" to "weighting the entire trajectory." Since the path measure ratio between the guided SDE and the reference SDE can be written in closed form via the Girsanov theorem, SMC can be performed directly in path space.

**Core Idea**: Use the Girsanov path likelihood ratio $\mathrm{d}\mathbb{P}/\mathrm{d}\mathbb{P}^G$ multiplied by the density $\mathrm{d}\mathbb{Q}/\mathrm{d}\mathbb{P}=\exp(r(X_t)-r(X_0))$ as the importance weight for the trajectory. This weight only involves $\nabla_x G$ (the gradient of the guidance itself, which is already computed) and the difference in $r$, containing no derivatives of $r$.

## Method

### Overall Architecture
Input: Drift $v(x,t)$ corresponding to a pre-trained score model, user-specified guidance potential $G(x,t)$, and reward function $r(x,t)$ (requiring $r(x,T)=\mathbf{r}(x)$). Process:

1. Discretize $[0,T]$ into $K$ steps and simulate $N$ guided trajectories $\{X_{t_k}^{(i),G}\}_{i=1}^N$ in parallel, taking a small step $\Delta t$ using Euler-Maruyama;
2. After each step, calculate a Girsanov-type multiplicative weight $\beta^{(i)}_{t_k,t_{k+1}}$ for each trajectory (dependent only on $\nabla_x G$ and the difference in $r$);
3. Perform Categorical resampling normalized by $\beta$ to duplicate high-weight particles and eliminate low-weight ones;
4. Repeat for $K$ steps; the final states $\{X_T^{(i)}\}_{i=1}^N$ serve as approximate samples from the reward-tilted posterior $q$.

Output: $N$ samples following $q(x,T)\propto p_\text{data}(x)e^{\mathbf{r}(x)}$ (strictly unbiased as $\Delta t\to 0$ and $N\to\infty$).

### Key Designs

1. **Path-space Girsanov Weights (URGE Core)**:
    - **Function**: Assigns an unbiased importance weight to each guided trajectory without requiring reward derivatives.
    - **Mechanism**: Applying the Girsanov theorem to the reference measure $\mathbb{P}$ (no guidance) and the guided measure $\mathbb{P}^G$ yields $\mathrm{d}\mathbb{P}/\mathrm{d}\mathbb{P}^G \propto \exp(-\int_0^t V(s)\nabla_x G^\top dW_s - \tfrac{1}{2}\int_0^t V^2\|\nabla_x G\|^2 ds)$. Multiplying this by the density of the reward-tilted measure $\mathbb{Q}$ relative to $\mathbb{P}$, $\exp(r(X_t)-r(X_0))$, results in the target weight $\mathrm{d}\mathbb{Q}/\mathrm{d}\mathbb{P}^G$. After Euler-Maruyama discretization, this is expressed as $\beta^{(i)}_{s,t}=\exp(r(X_t)-r(X_s) - V(s)\nabla_x G^\top\sqrt{t-s}\,\xi^{(i)} - \tfrac{1}{2}V(s)^2\|\nabla_x G\|^2(t-s))$, where $\xi^{(i)}$ is the Gaussian noise used in the EM step, which can be reused with nearly zero overhead.
    - **Design Motivation**: The weight only involves $\nabla_x G$ and the numerical difference of $r$, completely bypassing the second-order terms ($\Delta_x r$, $\|\nabla_x r\|^2$, $\nabla_x\log p_t$) that FK-Corrector / AFDPS must compute. This allows URGE to be used directly with black-box neural rewards like ImageReward or HPS for the first time.

2. **Itô Term Injection of Stochastic Path Information**:
    - **Function**: Explicitly incorporates Brownian noise $dW_\tau$ into the weights, allowing resampling to distinguish between "lucky" and "unlucky" trajectories reaching the same endpoint.
    - **Mechanism**: The first term of the weight is the Itô integral $\int_s^t -V(\tau)\nabla_x G^\top dW_\tau$, which discretizes exactly to $-V(s)\nabla_x G^\top\sqrt{\Delta t}\,\xi^{(i)}$. By writing the $\xi^{(i)}$ sampled in each EM step back into the weight, it contrasts with methods like AFDPS that perform deterministic second-order expansions for each particle.
    - **Design Motivation**: The authors point out that AFDPS / FK-Corrector weights are purely deterministic functions of the endpoint $x$, losing critical stochastic information about how that endpoint was reached. Explicitly including $dW_\tau$ makes resampling more granular; experiments observe lower variance and better scaling with $N$.

3. **Path-Particle Equivalence Theorem (Theoretical Backbone)**:
    - **Function**: Proves that URGE is not just another approximation but a different implementation of the same process as FK-Corrector / AFDPS at the generator level.
    - **Mechanism**: Defining the instantaneous intensity $\lambda(x,t):=\lim_{h\to 0}\tfrac{1}{h}(\mathbb{E}_{\mathbb{P}^G}[w^\text{URGE}_{t-h,t}\mid X_t=x]-1)$ and deriving the marginalized generator $\mathcal{L}^\text{eff}_t = \mathcal{L}^G_t + \lambda(\cdot,t)$ via Feynman-Kac backward value functions. Theorem 3.3 further proves $\lambda(x,t) \equiv w_\text{AFDPS}(x,t)$, meaning that taking the conditional expectation of path weights given the endpoint exactly recovers all second-order terms of AFDPS.
    - **Design Motivation**: Equivalence guarantees the unbiasedness of URGE (inheriting the theory of AFDPS) while demonstrating that path space offers strictly more flexible design degrees of freedom—AFDPS is a special case of URGE after taking conditional expectations, whereas URGE can conversely choose any higher-order discretization or sparser sampling grids.

### Loss & Training
URGE is a pure inference-time algorithm and **requires no additional training**. The only hyperparameters are the number of particles $N$, the number of discretization steps $K$, and the guidance strength (typically setting $G=r$ or the CFG term in text-to-image). The paper uses the simplest EM discretization (Equation 7) and notes that the weight construction can be swapped for any higher-order format to improve accuracy when $N$ is limited.

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

URGE performs best across all four metrics, particularly reducing covariance Frobenius error by 26% compared to AFDPS+VCG, without requiring extra steps like VCG's learned control drift.

Four inverse problems on ImageNet-256 (PSNR↑/LPIPS↓):

| Method | Gaussian Deblur PSNR | Motion Deblur LPIPS | Super-Res PSNR | Box Inpaint LPIPS |
|------|----------------------|---------------------|----------------|--------------------|
| SGS-EDM | 22.09 | 0.526 | 15.43 | 0.298 |
| FK-Corrector | 18.36 | 0.601 | 18.58 | 0.714 |
| AFDPS-SDE | 22.43 | 0.520 | 21.03 | 0.307 |
| AFDPS-ODE | 22.57 | 0.503 | 19.60 | **0.275** |
| **URGE** | 22.38 | 0.525 | 21.00 | 0.305 |

URGE ties with the strongest AFDPS variants (where $\nabla\log p_t$ is still needed if used as a reward, but the path weight itself remains a derivative-free construction) and significantly outperforms FK-Corrector.

### Ablation Study

Text-to-Image (Stable Diffusion v1.5, 50 prompts × 3 seeds):

| Sampler | CLIP-Score↑ | HPS↑ | ImageReward↑ | GenEval↑ |
|--------|-------------|------|--------------|----------|
| Base $N=1$ | 0.273 | 0.262 | 0.214 | 0.640 |
| Gradient Guidance $N=1$ | 0.273 | 0.262 | 0.207 | 0.640 |
| FK-Steering $N=4$ | 0.290 | 0.285 | 0.840 | 0.720 |
| Gradient FK $N=4$ | 0.290 | 0.284 | 0.791 | 0.747 |
| **URGE $N=4$** | **0.300** | **0.293** | **0.996** | **0.780** |

ImageReward jumps from 0.21 for the base to 0.996 ($\times 4.7$), and CLIP / HPS / GenEval also lead comprehensively. The authors emphasize that SDv1.5 + URGE often matches or exceeds the SDXL baseline on dual-object prompts.

### Key Findings
- **Derivative-free without performance loss**: Even after removing $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x \log p_t$, URGE is more accurate on GMM and inverse problems than AFDPS which keeps those terms—confirming that stochastic information from the Itô path term offsets the loss of second-order terms.
- **Monotonic particle scaling**: Figure 3 shows ImageReward grows monotonically with $N$, while FK-Steering plateaus; indicating more stable weight variance in path space.
- **Small models surpassing large models**: SDv1.5 + URGE ($N=4$) achieves a higher ImageReward than SDXL base, suggesting that "adding SMC resampling" might be more cost-effective than "switching to a larger model" for the same compute budget.
- **Black-box rewards as the killer app**: FK-Corrector / AFDPS are unusable in text-to-image settings (impossible to compute Hessian for neural rewards). URGE is the only SMC scheme that can directly integrate ImageReward / HPS.

## Highlights & Insights
- **Measure-theoretic substitution**: Replacing "per-particle weighting" with "per-trajectory weighting" might seem like a mere notation change, but the Girsanov theorem bundles all second-order terms in the infinitesimal generator into a single Itô integral. This simplifies engineering to "just reuse the EM step $\xi^{(i)}$," a classic case of "using deeper math for simpler code."
- **Explanatory power of the equivalence theorem**: Theorem 3.3 serves as both a "proof of correctness" and an "argument for design freedom"—since AFDPS is a special case of URGE after conditional expectation, URGE clearly has more options (sparser discretization, higher-order schemes, different time grids), opening future directions.
- **Transferable techniques**: This "path weight ≡ particle weight conditioned on endpoint" Feynman-Kac duality argument can be applied to any SMC-based diffusion inference algorithm, such as reward fine-tuning, molecular conformation sampling, or protein inverse folding guidance correction.

## Limitations & Future Work
- Experiments only tested $N=4 \sim 16$; it is not systematically explored whether particle degeneracy occurs when $N$ is large ($\sim 100$), though the saturation of ImageReward in text-to-image might be a precursor.
- Weights still contain $\nabla_x G$, so "derivative-free" strictly refers to "reward derivative-free"—the guidance potential $G$ must remain differentiable (in CFG, $G$ is the classifier-free term, which is differentiable, but if $G$ also becomes a black box, URGE collapses to just $r$ differences and loses unbiasedness).
- The discretization step $\Delta t$ must be small enough to ensure Girsanov discretization stability; the authors do not provide an "adaptive $\Delta t$" scheme, which might be a bottleneck for long-horizon video diffusion models.
- The equivalence theorem only holds as $N\to\infty$ and in the continuous-time limit. The variance gap between URGE and AFDPS for finite $N$ is theoretically guaranteed but lacks a closed-form bound in the paper.
- Future work: Coupling higher-order SDE discretizations (Heun / DPM-Solver-2) with URGE path weights; studying URGE variants for non-smooth rewards (like GenEval discrete surrogate metrics); extending URGE to jump diffusion for categorical / discrete diffusion support.

## Related Work & Insights
- **vs FK-Corrector (Skreta et al., 2025) / AFDPS (Chen et al., 2025)**: These correct the generator in particle space using $\mathcal{L}^G + w_\text{AFDPS}$, where weights contain $\Delta_x r$, $\|\nabla_x r\|^2$, and $\nabla_x \log p_t$. URGE uses Girsanov weights in path space where second-order terms disappear; they are theoretically equivalent (Theorem 3.3) but URGE is simpler and supports black-box rewards.
- **vs FK-Steering (Singhal et al., 2025)**: FK-Steering uses $r(X_{t+\Delta t})-r(X_t)$ as weight, discarding the Girsanov path correction term and thus losing unbiasedness. URGE includes the path information term, ensuring unbiasedness and significantly leading in experiments.
- **vs Doob $h$-transform methods (DEFT / various adjoint matching)**: $h$-transform requires training a network to estimate $h$ or solving the backward Kolmogorov equation. URGE is training-free and only adds an $N$-fold forward pass overhead at inference.
- **vs VCG (Ren et al., 2025a)**: VCG learns a control drift via weighted least squares to reduce variance. URGE learns nothing yet achieves 26% lower covariance error than AFDPS+VCG on GMM, showing path space is inherently more stable.
- **Intellectual lineage**: This work continues the recent thread of "using SMC + diffusion for unbiased inference-time guidance" (Wu et al. 2023; Singhal et al. 2025; Skreta et al. 2025; Chen et al. 2025), pushing it to the concise extreme of "derivative-free + path space." Further back, it stems from classic SMC (Del Moral 2004) and Girsanov importance sampling in financial mathematics.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of path-space SMC + Girsanov weights is new for diffusion inference scaling, though Girsanov importance sampling itself is a classic tool in finance/molecular simulation; this is an effective migration rather than a brand-new invention.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers toy GMM + 4 inverse problems + text-to-image. However, the range of $N$ is small, there is no validation on long-horizon video diffusion, and a detailed runtime comparison table is missing.
- Writing Quality: ⭐⭐⭐⭐ Clearly organizes the trio of Girsanov, Feynman-Kac, and Kolmogorov backward; Table 1 clarifies weight differences at a glance; the term "approximation-free" appears somewhat redundantly.
- Value: ⭐⭐⭐⭐⭐ Derivative-free and black-box reward friendliness means ImageReward / HPS / human preference scorers can be directly plugged in. It is transferable to any scenario using neural rewards for guidance (text-to-image, video, 3D); the combination of theoretical and engineering clarity provides high reuse value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Tiny Inference-Time Scaling with Latent Verifiers](../../CVPR2026/image_generation/tiny_inference-time_scaling_with_latent_verifiers.md)
- [\[ICML 2026\] SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate](surge_approximation_and_training_free_particle_filter_for_diffusion_surrogate.md)
- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](../../NeurIPS2025/image_generation/inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](../../CVPR2026/image_generation/dpcache_denoising_path_planning_diffusion_accel.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)

</div>

<!-- RELATED:END -->

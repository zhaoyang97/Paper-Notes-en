---
title: >-
  [Paper Note] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models
description: >-
  [ICML 2026][Computational Biology][temperature sampling] By multiplying the score output of pre-trained diffusion/flow models by an analytical rescaling factor $r_t$, which depends only on the timestep, variable $k$, and $\sigma$, the sampling distribution can be made "locally" sharper or flatter during the inference stage without any fine-tuning. This method is fully compatible with deterministic samplers such as DDIM.
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "temperature sampling"
  - "score rescaling"
  - "diffusion"
  - "flow matching"
  - "training-free"
date: 2026-05-08
content_hash: fbc98632cdd0918f
---

# Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models

**Conference**: ICML 2026  
**arXiv**: [2510.01184](https://arxiv.org/abs/2510.01184)  
**Code**: https://temporalscorerescaling.github.io  
**Area**: Computational Biology  
**Keywords**: temperature sampling, score rescaling, diffusion, flow matching, training-free  

## TL;DR
By multiplying the score output of pre-trained diffusion/flow models by an analytical rescaling factor $r_t$, which depends only on the timestep, variable $k$, and $\sigma$, the sampling distribution can be made "locally" sharper or flatter during the inference stage without any fine-tuning. This method is fully compatible with deterministic samplers such as DDIM.

## Background & Motivation
**Background**: Diffusion models (DDPM) and Flow Matching models have become general paradigms for tasks like image generation, depth estimation, pose prediction, robot policies, and protein design by learning the score $\nabla\log p_t(\mathbf{x})$ of the noise distribution $p(\mathbf{x}_t)$. However, practical deployment often requires deviating from the training distribution—for example, depth estimation seeks "more likely" predictions, while image generation seeks "more diverse" samples.

**Limitations of Prior Work**: Traditional temperature sampling is difficult to implement in diffusion models. Although Classifier-Free Guidance (CFG) offers a trade-off between diversity and likelihood, it is not essentially equivalent to temperature scaling and only applies to conditional models. Likelihood-weighted fine-tuning (Shih et al., 2023) requires retraining, which is unrealistic for large models. Langevin/MCMC correction (Du et al., 2023) increases inference overhead by more than five times. The most commonly used Constant Noise Scaling (CNS) simply multiplies the noise term in the SDE by a constant $1/\sqrt{k}$, which is incompatible with ODE samplers and leads to mode collapse due to excessive suppression of exploration at high noise steps and insufficient suppression at low noise steps.

**Key Challenge**: Temperature sampling requires "changing mode width without changing mode weight." All existing training-free methods use a constant "exploration intensity" across timesteps, making it impossible to achieve such local scaling precisely.

**Goal**: Design a temperature sampling mechanism that is (a) training-free, (b) compatible with deterministic samplers like DDIM/Euler, (c) does not increase the number of score evaluations, and (d) provides provability for simple distributions.

**Key Insight**: The authors noted that if the original data is an isotropic Gaussian or a mixture thereof, there exists an analytical, time-varying linear ratio between the score of the "temperature-scaled data" after forward noising and the original score. In other words, for a pre-trained score network, multiplying the output by $r_t$ during inference equivalently replaces the sampling target distribution with a version whose variance is scaled by $1/k$.

**Core Idea**: Rescale the score using a time-dependent scalar $r_t(k,\sigma)=\frac{\eta_t\sigma^2+1}{\eta_t\sigma^2/k+1}$ (where $\eta_t=\alpha_t^2/\sigma_t^2$ is the signal-to-noise ratio). This converts the "global temperature" problem into a "local temperature" problem, avoiding mode shift.

## Method

### Overall Architecture
The method, called **Temporal Score Rescaling (TSR)**, addresses the problem of making pre-trained diffusion/flow models sharper (likelihood-biased) or flatter (diversity-biased) during inference without retraining. This approach inserts a single step during inference: take the output of any pre-trained diffusion or flow matching model, convert the noise prediction $\boldsymbol\epsilon_\theta$ or velocity $\boldsymbol v_\theta$ equivalent to the score perspective, multiply it by a time-dependent scalar rescaling factor $r_t(k,\sigma)$, and feed the rescaled score back into the original sampler (DDPM/DDIM/Euler/Heun all work). For users, this only adds two hyperparameters: $k$ (controlling the strength of sharpening/flattening) and $\sigma$ (controlling the step at which intervention begins), which, like CFG, can be reused once tuned.

### Key Designs

**1. Analytical score scaling formula on a single Gaussian: Compressing temperature operations into a scalar**

The power of TSR stems from a simple observation: temperature scaling the data distribution $\Sigma\to\Sigma/k$ can be equivalently applied only to the score. Under the stochastic interpolant framework $\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sigma_t\boldsymbol\epsilon$, if $\mathbf{x}_0\sim\mathcal{N}(\boldsymbol\mu,\sigma^2\mathbf{I})$, the score of the noise distribution has a closed form $\nabla\log p_t(\mathbf{x})=-(\mathbf{x}-\alpha_t\boldsymbol\mu)/(\alpha_t^2\sigma^2+\sigma_t^2)$. After replacing the data variance with $\sigma^2/k$, only the denominator of this expression changes. Thus, the ratio between the scores before and after temperature scaling is an analytical scalar $r_t(k,\sigma)=\frac{\eta_t\sigma^2+1}{\eta_t\sigma^2/k+1}$ (where $\eta_t=\alpha_t^2/\sigma_t^2$), and when $k=1$, $r_t\equiv1$, exactly recovering the original score. Because it is a closed form rather than an additional network or iterative correction, the method is "plug-and-play" for any sampler.

**2. Generalization to Gaussian Mixtures: Retreating from "global temperature" to "local temperature"**

What truly distinguishes TSR from older methods is that it abandons changing mode weights and only changes mode widths. The authors prove that when the data is a "well-separated" mixture of Gaussians with equal variance, the $r_t$ derived for a single Gaussian remains a bounded approximation of the true score. At small $t$, a single component dominates and the error decays exponentially; at large $t$, the distribution approximates pure noise and the error decays polynomially, tending to zero at both ends. Physically, TSR only compresses the variance within each mode without altering the relative weights between modes. Consequently, sampling results uniformly cover all modes rather than collapsing toward the center. This bypasses the side effects of "collapsing to central modes" in CNS and "mode weight imbalance" in CFG. The paper validates this "local" characteristic on 1D Gaussian mixtures, 2D checkerboards, and Swiss-rolls: while CNS loses peripheral modes, TSR preserves them all. Although real data are not explicit GMMs, any sufficiently smooth distribution can be locally approximated by a Gaussian, giving the formula cross-task universality.

**3. Unified adaptation for score / $\epsilon$ / velocity parameterizations: One formula for heterogeneous models**

To allow the same $r_t$ to drive both noise-prediction models (like DDPM/DDIM) and velocity-prediction models (like Flow Matching), the authors convert all parameterizations back to scores. The score has a linear relationship with noise prediction $\mathbf{s}_\theta=-\sigma_t^{-1}\boldsymbol\epsilon_\theta$, so for diffusion models, $\tilde{\boldsymbol\epsilon}_\theta=r_t(k,\sigma)\boldsymbol\epsilon_\theta$ is applied directly. For flow matching models, velocity and score satisfy $\mathbf{s}_\theta=-(\alpha_t\boldsymbol v_\theta-\dot{\alpha}_t\mathbf{x})/[\sigma_t(\dot{\alpha}_t\sigma_t-\alpha_t\dot{\sigma}_t)]$. Substituting this yields $\tilde{\boldsymbol v}_\theta=\alpha_t^{-1}(r_t(k,\sigma)(\alpha_t\boldsymbol v_\theta-\dot{\alpha}_t\mathbf{x})+\dot{\alpha}_t\mathbf{x})$. $x_0$-prediction and $v$-prediction can be substituted similarly. This eliminates the need to rebuild the sampling stack for each model type, allowing TSR to be plug-and-play for heterogeneous models like Stable Diffusion 2/3, Flux.1 dev, FoldingDiff, Marigold, and Pi-0.

### Loss & Training
No new training is required. Selection of hyperparameters $k$ and $\sigma$: first fix $\sigma=1.0$ and use binary search to find the optimal $k$, then fix $k$ and use binary search to find the optimal $\sigma$, iterating once if necessary. This is much more efficient than grid search, and empirically, the same $(k, \sigma)$ pair can transfer across different models for the same task.

## Key Experimental Results

### Main Results

| Task / Dataset | Model | Metrics | Default Sampling | + TSR | Remarks |
|---------------|------|------|----------|-------|------|
| Text-to-Image / LAION Aesth. 5k | SD3 | FID ↓ / CLIP ↑ | 24.77 / 32.82 | **22.81 / 33.05** | $k=0.93, \sigma=3.0$, exceeds CFG Pareto frontier |
| Text-to-Image | SD2 | FID / CLIP | 22.81 / 33.66 | **19.75 / 33.75** | Same $(k,\sigma)$ transferred |
| Text-to-Image | Flux.1 dev | FID / CLIP | 53.99 / 31.97 | **51.79 / 32.14** | Same $(k,\sigma)$ transferred |
| Depth / ETH3D | Marigold(DDIM) | AbsRel ↓ / $\delta_1$ ↑ | 7.1 / 90.4 | **6.68 / 95.7** | Better than CNS (6.82 / 95.6) |
| Depth / NYUv2 | Marigold | AbsRel / $\delta_1$ | 6.0 / 95.9 | **5.84 / 96.0** | — |
| Pose / SYMSOL | $SO(3)$ diffusion | mean err (deg) ↓ | 0.444 | **0.356** ($k=7,\sigma=0.5$) | CNS slightly better (0.350) but only for SDE |
| Robotics / LIBERO-10 | Pi-0 (flow) | Avg Success ↑ | 81.7 | **82.8** ($k=1.25,\sigma=0.25$) | 6 gains, 2 ties out of 10 tasks |
| Protein Gen | FoldingDiff | designability ↑ | 0.22 | Significant improvement | FID better than CNS while preserving diversity |

### Ablation Study

| Configuration | FID ↓ (SD3) | CLIP ↑ | Meaning |
|------|-------------|--------|------|
| Default Euler-ODE | 24.77 | 32.82 | Baseline |
| + CFG Adjusted | Pareto Curve | Pareto Curve | Diversity/alignment trade-off |
| + CNS | N/A for flow ODE | — | Only applicable to SDE samplers |
| + TSR ($k=0.93,\sigma=3.0$) | **22.81** | **33.05** | Exceeds CFG frontier, stacks with CFG |
| TSR $k<1$ (flatter) | More detail but noisy | — | Benefit for generative tasks |
| TSR $k>1$ (sharper) | Smooth, lacks detail | — | Benefit for predictive tasks |

### Key Findings
- The "local temperature" concept is the core differentiator: CNS collapses toward center modes by suppressing noise with a constant, whereas TSR uses a time-dependent factor to only compress intra-mode variance, thereby preserving all modes.
- Creative tasks (image generation) prefer $k<1$ (flatter), while predictive tasks (depth, pose, robotics, protein design) prefer $k>1$ (sharper); the unified framework serves both.
- Optimal $(k, \sigma)$ values transfer stably between different models for the same task, with tuning costs similar to CFG.
- Orthogonal to CFG and can be used in combination; natively compatible with deterministic samplers like DDIM / Euler-ODE with zero extra NFE.

## Highlights & Insights
- Using the simple observation that "a closed-form ratio exists between the score of temperature-scaled data and the original score" compresses a problem previously thought to require retraining or MCMC into "multiplying by a scalar," serving as an elegant case of "analysis replacing algorithms."
- Relaxing temperature from "global" to "local" is a key conceptual leap: giving up on changing mode weights to only change mode widths effectively avoids mode dropping. This trade-off of "self-limitation for universality" is worth reusing in other generative control problems.
- The single $r_t$ formula covering score, $\epsilon$, and velocity parameterizations demonstrates the unified value of the stochastic interpolant perspective—this design philosophy of "finding the least common variable" provides a reference for other inference-time intervention methods in diffusion and flow models.

## Limitations & Future Work
- Restricted to "local" temperature: It cannot change the relative weights of modes in a GMM, limiting effectiveness for tasks intended to "suppress incorrect modes and amplify correct ones" (e.g., suppressing fake objects in conditional sampling).
- Theoretical guarantees are limited to well-separated, equal-variance Gaussian mixtures; empirical effectiveness must be relied upon for real complex distributions, as the error bounds provided are loose for low $\sigma$ data.
- Hyperparameters $(k, \sigma)$ must be tuned for each task/model separately. Although the cost is similar to CFG, it still lacks zero-shot adaptation and a data-driven method for estimating $\sigma$.
- Performance drops at improper $k$ values (e.g., on Libero Task 2/8) suggest that for tasks where the base model already performs poorly, sharpening might amplify errors, necessitating task-level dynamic control.

## Related Work & Insights
- **vs CFG (Ho & Salimans 2022)**: CFG is only effective for conditional models and requires condition dropout during training; TSR works for both unconditional and conditional models and is orthogonal to CFG.
- **vs CNS (Yim et al., 2023; Geffner et al., 2025)**: CNS scales the noise term by a constant $1/\sqrt{k}$, is only compatible with SDE, and loses modes; TSR scales the score with a time-dependent $r_t$, is compatible with ODE, and preserves modes.
- **vs Likelihood-weighted Finetuning (Shih et al., 2023)**: Requires retraining and access to training data; TSR is completely training-free.
- **vs Langevin / MCMC Corrector (Du et al., 2023)**: Multiplies inference overhead; TSR has zero extra NFE.
- **vs Variance-reduced sampling for proteins (Geffner et al., 2025)**: Essentially a special case of CNS; TSR achieves both higher designability and lower FID on FoldingDiff.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach is refined and elegant, though "rescaling scores" is not a brand-new direction; the highlights are the analytical formula and the "local temperature" concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Spans 5 entirely different tasks, covers both ODE and SDE samplers, and systematically compares against CFG/CNS.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations and ample illustrations; provides user-friendly intuitive explanations for hyperparameters.
- Value: ⭐⭐⭐⭐⭐ Zero training cost, zero inference overhead, and immediately applicable to almost all diffusion/flow models; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[ICML 2026\] Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](../../NeurIPS2025/computational_biology/consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[ICLR 2026\] Enhancing Diffusion-Based Sampling with Molecular Collective Variables](../../ICLR2026/computational_biology/enhancing_diffusion-based_sampling_with_molecular_collective_variables.md)

</div>

<!-- RELATED:END -->

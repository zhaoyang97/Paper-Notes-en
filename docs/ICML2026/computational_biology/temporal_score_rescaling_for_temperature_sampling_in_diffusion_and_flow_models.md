---
title: >-
  [Paper Note] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models
description: >-
  [ICML 2026][Computational Biology][temperature sampling] By multiplying the score output of pre-trained diffusion/flow models by an analytical rescaling factor $r_t$—which depends only on the timestep, variable $k$…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "temperature sampling"
  - "score rescaling"
  - "diffusion"
  - "flow matching"
  - "training-free"
date: 2026-05-08
content_hash: 9418d28c075f7313
---

# Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models

**Conference**: ICML 2026  
**arXiv**: [2510.01184](https://arxiv.org/abs/2510.01184)  
**Code**: https://temporalscorerescaling.github.io  
**Area**: Computational Biology  
**Keywords**: temperature sampling, score rescaling, diffusion, flow matching, training-free  

## TL;DR
By multiplying the score output of pre-trained diffusion/flow models by an analytical rescaling factor $r_t$—which depends only on the timestep, variable $k$, and $\sigma$—the sampling distribution can be made "locally" sharper or flatter during inference without any fine-tuning. This method is fully compatible with deterministic samplers such as DDIM.

## Background & Motivation
**Background**: Diffusion models (DDPM) and Flow Matching models have become a universal paradigm for tasks such as image generation, depth estimation, pose prediction, robot policies, and protein design by learning the score $\nabla\log p_t(\mathbf{x})$ of the noise distribution $p(\mathbf{x}_t)$. However, practical deployment often requires deviating from the training distribution—for instance, depth estimation demands "more likely" predictions, while image generation requires "more diverse" samples.

**Limitations of Prior Work**: Traditional temperature sampling is difficult to implement in diffusion models. Although Classifier-Free Guidance (CFG) allows for a trade-off between diversity and likelihood, it is not essentially equivalent to temperature scaling and is limited to conditional models. Likelihood-weighted fine-tuning (Shih et al., 2023) requires retraining, which is impractical for large models. Langevin/MCMC corrections (Du et al., 2023) increase inference overhead by more than five times. The most common method, Constant Noise Scaling (CNS), merely multiplies the noise term in the SDE by a constant $1/\sqrt{k}$, which is incompatible with ODE samplers and causes mode collapse due to excessive suppression of exploration at high noise levels and insufficient suppression at low noise levels.

**Key Challenge**: Temperature sampling requires "altering mode width without changing mode weights," whereas existing training-free methods use a constant "exploration intensity" across all timesteps, failing to achieve precise local scaling.

**Goal**: Design a temperature sampling mechanism that is (a) training-free, (b) compatible with deterministic samplers like DDIM/Euler, (c) does not increase the number of score evaluations, and (d) provides provability on simple distributions.

**Key Insight**: The authors observe that if the original data follows an isotropic Gaussian or a mixture thereof, there exists an analytical, time-varying linear ratio between the score of the "temperature-scaled data" after forward noising and the original score. In other words, for a pre-trained score network, multiplying by $r_t$ during inference is equivalent to replacing the target sampling distribution with a version where variance is scaled by $1/k$.

**Core Idea**: Rescale the score using a time-dependent scalar $r_t(k,\sigma)=\frac{\eta_t\sigma^2+1}{\eta_t\sigma^2/k+1}$ (where $\eta_t=\alpha_t^2/\sigma_t^2$ is the SNR), converting the "global temperature" problem into a "local temperature" problem to avoid mode drift.

## Method

### Overall Architecture
The proposed method is called **Temporal Score Rescaling (TSR)**. The pipeline maintains the training process and only adds an intermediate step during inference: given any pre-trained diffusion or flow matching model $\mathbf{s}_\theta$, $\boldsymbol\epsilon_\theta$, or $\boldsymbol v_\theta$, these are first converted to the score perspective, multiplied by $r_t(k,\sigma)$, and then fed back into the original sampler (DDPM, DDIM, Euler, or Heun). Users only specify two hyperparameters: $k$ (controlling the intensity of narrowing/widening) and $\sigma$ (controlling the intervention starting point), which, like CFG, can be reused once tuned.

### Key Designs

1. **Analytical Derivation: Score Scaling Formula on a single Gaussian**:
    - Function: Converts the "temperature operation" of $ \Sigma \to \Sigma / k $ on the data distribution into an equivalent operation acting only on the score.
    - Mechanism: Under the stochastic interpolant framework $\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sigma_t\boldsymbol\epsilon$, if $\mathbf{x}_0\sim\mathcal{N}(\boldsymbol\mu,\sigma^2\mathbf{I})$, the score of the noise distribution is $\nabla\log p_t(\mathbf{x})=-(\mathbf{x}-\alpha_t\boldsymbol\mu)/(\alpha_t^2\sigma^2+\sigma_t^2)$. Changing the data variance to $\sigma^2/k$ only modifies the denominator of the score. The ratio between the two is the rescaling factor $r_t(k,\sigma)=\frac{\eta_t\sigma^2+1}{\eta_t\sigma^2/k+1}$, which recovers the original score when $k=1$.
    - Design Motivation: Use a closed-form expression to replace any additional neural networks or iterative corrections, making the method "plug-and-play" for any sampler.

2. **Generalization from Single Gaussian to GMM: Local Temperature Sampling**:
    - Function: Avoids side effects like "collapsing to the center mode" (as in CNS) or "mode weight imbalance" (as in CFG) on multimodal data.
    - Mechanism: The authors prove that when the data is a "well-separated" Gaussian Mixture Model (GMM) with equal variance, the above formula remains a bounded approximation of the score. At small $t$, a single component dominates, yielding an exponential error bound; at large $t$, the distribution approximates pure noise, yielding a polynomial error bound. Error at both ends tends toward zero. Physically, TSR only modifies the variance within each mode without altering mode weights, ensuring the sampling results uniformly cover all modes rather than collapsing to the center. Validation on 1D GMM, 2D checkerboard, and Swiss-roll shows that while CNS loses edge modes, TSR preserves them all.
    - Design Motivation: While real data is not an explicit GMM, any sufficiently smooth distribution can be locally approximated by a Gaussian, ensuring the cross-task universality of the formula.

3. **Unified Adaptation for score / $\epsilon$ / velocity Parameterizations**:
    - Function: Allows the same formula to drive noise prediction models (DDPM/DDIM) and velocity prediction models (Flow Matching).
    - Mechanism: The score has a linear relationship with noise prediction: $\mathbf{s}_\theta=-\sigma_t^{-1}\boldsymbol\epsilon_\theta$. Thus, for diffusion models, $\tilde{\boldsymbol\epsilon}_\theta=r_t(k,\sigma)\boldsymbol\epsilon_\theta$. For flow matching, velocity and score satisfy $\mathbf{s}_\theta=-(\alpha_t\boldsymbol v_\theta-\dot{\alpha}_t\mathbf{x})/[\sigma_t(\dot{\alpha}_t\sigma_t-\alpha_t\dot{\sigma}_t)]$. Substitution yields $\tilde{\boldsymbol v}_\theta=\alpha_t^{-1}(r_t(k,\sigma)(\alpha_t\boldsymbol v_\theta-\dot{\alpha}_t\mathbf{x})+\dot{\alpha}_t\mathbf{x})$. $x_0$-prediction and $v$-prediction can be derived similarly.
    - Design Motivation: Avoid rebuilding sampling stacks, allowing TSR to be plug-and-play for heterogeneous models like Stable Diffusion 2/3, Flux.1 dev, FoldingDiff, Marigold, and Pi-0.

### Loss & Training
No new training is required. The selection of hyperparameters $k$ and $\sigma$: first fix $\sigma=1.0$ and use binary search to find the optimal $k$, then fix $k$ and use binary search to find the optimal $\sigma$, iterating once if necessary. This is more efficient than grid search, and empirically, the same $(k, \sigma)$ pair can transfer across different models within the same task.

## Key Experimental Results

### Main Results

| Task / Dataset | Model | Metric | Default Sampling | + TSR | Remarks |
|---------------|------|------|----------|-------|------|
| Text-to-Image / LAION Aesth. 5k | SD3 | FID ↓ / CLIP ↑ | 24.77 / 32.82 | **22.81 / 33.05** | $k=0.93, \sigma=3.0$, exceeds CFG Pareto frontier |
| Text-to-Image | SD2 | FID / CLIP | 22.81 / 33.66 | **19.75 / 33.75** | Transfer same $(k,\sigma)$ |
| Text-to-Image | Flux.1 dev | FID / CLIP | 53.99 / 31.97 | **51.79 / 32.14** | Transfer same $(k,\sigma)$ |
| Depth Est. / ETH3D | Marigold(DDIM) | AbsRel ↓ / $\delta_1$ ↑ | 7.1 / 90.4 | **6.68 / 95.7** | Better than CNS (6.82 / 95.6) |
| Depth Est. / NYUv2 | Marigold | AbsRel / $\delta_1$ | 6.0 / 95.9 | **5.84 / 96.0** | — |
| Pose Est. / SYMSOL | $SO(3)$ diffusion | mean err (deg) ↓ | 0.444 | **0.356** ($k=7,\sigma=0.5$) | CNS slightly better (0.350) but only for SDE |
| Robotics / LIBERO-10 | Pi-0 (flow) | Avg Success ↑ | 81.7 | **82.8** ($k=1.25,\sigma=0.25$) | Improvement in 6/10 tasks |
| Protein Gen. | FoldingDiff | designability ↑ | 0.22 | Significant improv. | FID better than CNS, maintains diversity |

### Ablation Study

| Configuration | FID ↓ (SD3) | CLIP ↑ | Meaning |
|------|-------------|--------|------|
| Default Euler-ODE | 24.77 | 32.82 | Baseline |
| + CFG Adjustment | Pareto Curve | Pareto Curve | Diversity/Alignment trade-off |
| + CNS | N/A under flow ODE | — | Only applicable to SDE samplers |
| + TSR ($k=0.93,\sigma=3.0$) | **22.81** | **33.05** | Beyond CFG frontier, stackable with CFG |
| TSR $k<1$ (flatter) | More detail, closer to noise | — | Gains in generative tasks |
| TSR $k>1$ (sharper) | Smooth, lacking detail | — | Gains in predictive tasks |

### Key Findings
- The "local temperature" concept is the core distinction: while CNS collapses toward the center mode by suppressing noise with a constant, TSR uses a time-dependent factor to only suppress intra-mode variance, thereby preserving all modes.
- Creative tasks (image generation) prefer $k<1$ (flatter), while predictive tasks (depth, pose, robotics, protein design) prefer $k>1$ (sharper); the unified framework serves both.
- Optimal values of $(k, \sigma)$ transfer stably between different models of the same task, with tuning costs similar to CFG.
- Orthogonal to and stackable with CFG; natively compatible with deterministic samplers like DDIM / Euler-ODE with zero extra NFE.

## Highlights & Insights
- Using the simple observation that "a closed-form ratio exists between the score of temperature-scaled data and the original score" reduces a problem previously thought to require retraining or MCMC to a "scalar multiplication." This is an elegant case of "analysis replacing algorithm."
- Relaxing temperature from "global" to "local" is a key conceptual leap: giving up changing mode weights to only change mode width avoids mode dropping. This strategy of "limiting scope for universality" is worth reusing in other generative control problems.
- The use of the same $r_t$ formula across score/$\epsilon$/velocity parameterizations demonstrates the unified value of the stochastic interpolant perspective—this design philosophy of "finding the smallest common variable" is instructive for other inference-time intervention methods in diffusion/flow models.

## Limitations & Future Work
- Restricted to "local" temperature: It cannot change the relative weights of modes in a GMM, meaning its effect is limited for tasks requiring suppression of incorrect modes (e.g., suppressing fake objects in conditional sampling).
- Theoretical guarantees are limited to well-separated equal-variance Gaussian mixtures; empirical validity is relied upon for complex real distributions, and error bounds are loose for low-$\sigma$ data.
- $(k, \sigma)$ needs to be tuned separately for each task/model, and while costs are similar to CFG, it lacks zero-shot adaptation; there is no data-driven method for estimating $\sigma$.
- Performance drops on certain tasks (e.g., LIBERO Task 2/8) suggest that for tasks where the base model already performs poorly, sharpening might amplify errors, necessitating task-level dynamic control.

## Related Work & Insights
- **vs CFG (Ho & Salimans 2022)**: CFG only works for conditional models and requires condition dropout during training; TSR applies to both and is orthogonal to CFG.
- **vs CNS (Yim et al., 2023; Geffner et al., 2025)**: CNS uses a constant $1/\sqrt{k}$ for noise terms, is limited to SDEs, and loses modes; TSR uses time-dependent $r_t$ for scores, is ODE-compatible, and preserves modes.
- **vs Likelihood-weighted Finetuning (Shih et al., 2023)**: Requires retraining and access to training data; TSR is entirely training-free.
- **vs Langevin / MCMC Corrector (Du et al., 2023)**: Multiple times higher inference overhead; TSR has zero extra NFE.
- **vs Variance-reduced sampling for proteins (Geffner et al., 2025)**: Essentially a special case of CNS; TSR achieves higher designability and lower FID on FoldingDiff.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea is refined and elegant; while "score rescaling" is not entirely new, the analytical formula and "local temperature" concept are highlights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 diverse tasks, includes both ODE and SDE samplers, and systematically compares against CFG/CNS.
- Writing Quality: ⭐⭐⭐⭐ Derivations are clear with sufficient illustrations; intuitive explanations for hyperparameters are helpful.
- Value: ⭐⭐⭐⭐⭐ Zero training cost, zero inference overhead, immediately applicable to almost all diffusion/flow models; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[ICML 2026\] Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](../../NeurIPS2025/computational_biology/consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[ICML 2026\] EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)

</div>

<!-- RELATED:END -->

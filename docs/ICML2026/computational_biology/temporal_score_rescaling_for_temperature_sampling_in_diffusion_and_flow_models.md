---
title: >-
  [Paper Note] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models
description: >-
  [ICML 2026][Computational Biology][temperature sampling] By multiplying the score output of pre-trained diffusion/flow models by an analytic rescaling factor $r_t$ that depends only on the timestep, variable $k$, and $\sigma$, the sampling distribution can be made "locally" sharper or flatter during inference without any fine-tuning. This approach is fully compatible with de
tags:
  - ICML 2026
  - Computational Biology
  - temperature sampling
  - score rescaling
  - diffusion
  - flow matching
  - training-free
date: 2026-05-08
content_hash: a4936bef3f3dc445
---
# Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models

**Conference**: ICML 2026  
**arXiv**: [2510.01184](https://arxiv.org/abs/2510.01184)  
**Code**: https://temporalscorerescaling.github.io  
**Area**: Computational Biology  
**Keywords**: temperature sampling, score rescaling, diffusion, flow matching, training-free  

## TL;DR
By multiplying the score output of pre-trained diffusion/flow models by an analytic rescaling factor $r_t$ that depends only on the timestep, variable $k$, and $\sigma$, the sampling distribution can be made "locally" sharper or flatter during inference without any fine-tuning. This approach is fully compatible with deterministic samplers like DDIM.

## Background & Motivation
**Background**: Diffusion models (DDPM) and Flow Matching models have become the universal paradigm for tasks such as image generation, depth estimation, pose prediction, robot policies, and protein design by learning the score $\nabla\log p_t(\mathbf{x})$ of the noise distribution $p(\mathbf{x}_t)$. However, practical deployment often requires deviating from the training distribution—for example, depth estimation desires "more likely" predictions, while image generation seeks "more diverse" samples.

**Limitations of Prior Work**: Traditional temperature sampling is difficult to implement in diffusion models. Although Classifier-Free Guidance (CFG) offers a trade-off between diversity and likelihood, it is not intrinsically equivalent to temperature scaling and is limited to conditional models. Likelihood-weighted fine-tuning (Shih et al., 2023) requires retraining, which is impractical for large models. Langevin/MCMC corrections (Du et al., 2023) increase inference overhead by more than five times. The most commonly used Constant Noise Scaling (CNS) simply multiplies the noise term in the SDE by a constant $1/\sqrt{k}$, which is incompatible with ODE samplers and tends to over-suppress exploration at high noise levels while under-suppressing it at low noise levels, leading to mode collapse.

**Key Challenge**: Temperature sampling requires "changing the mode width without changing the mode weight." All existing training-free methods use a constant "exploration intensity" across timesteps, which naturally fails to achieve precise local scaling.

**Goal**: Design a temperature sampling mechanism that is (a) training-free, (b) compatible with deterministic samplers like DDIM/Euler, (c) does not increase the number of score evaluations, and (d) is provable for simple distributions.

**Key Insight**: The authors observe that if the original data follows an isotropic Gaussian or its mixtures, there exists an analytic, time-varying linear ratio between the score of the "temperature-scaled data" after forward noising and the original score. In other words, for a pre-trained score network, multiplying the score by $r_t$ during inference equivalently shifts the sampling target to a version where the variance is scaled by $1/k$.

**Core Idea**: Rescale the score using a time-dependent scalar $r_t(k,\sigma)=\frac{\eta_t\sigma^2+1}{\eta_t\sigma^2/k+1}$ (where $\eta_t=\alpha_t^2/\sigma_t^2$ is the signal-to-noise ratio). This transforms the "global temperature" problem into a "local temperature" problem, avoiding mode drift.

## Method

### Overall Architecture
The method, termed **Temporal Score Rescaling (TSR)**, addresses how to make pre-trained diffusion or flow models sharper (likelihood-biased) or flatter (diversity-biased) during inference without retraining. The procedure involves a single step during inference: take the output of any pre-trained diffusion or flow matching model, convert it to the score perspective (e.g., from noise prediction $\boldsymbol\epsilon_\theta$ or velocity $\boldsymbol v_\theta$), multiply it by the time-dependent rescaling factor $r_t(k,\sigma)$, and feed the scaled score back into the original sampler (DDPM, DDIM, Euler, or Heun). For users, this introduces only two hyperparameters, $k$ (controlling the intensity of sharpening/flattening) and $\sigma$ (controlling the intervention start point), which are reusable once tuned, similar to CFG.

### Key Designs

**1. Analytic Score Scaling on a Single Gaussian: Compressing Temperature Operations into a Scalar**

The power of TSR stems from a simple observation: temperature scaling the data distribution ($\Sigma\to\Sigma/k$) can be equivalently applied solely to the score. Under the stochastic interpolant framework $\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sigma_t\boldsymbol\epsilon$, if $\mathbf{x}_0\sim\mathcal{N}(\boldsymbol\mu,\sigma^2\mathbf{I})$, the score of the noise distribution has the closed form $\nabla\log p_t(\mathbf{x})=-(\mathbf{x}-\alpha_t\boldsymbol\mu)/(\alpha_t^2\sigma^2+\sigma_t^2)$. Replacing the data variance with $\sigma^2/k$ only modifies the denominator. Thus, the ratio between the scores before and after temperature scaling is an analytic scalar $r_t(k,\sigma)=\frac{\eta_t\sigma^2+1}{\eta_t\sigma^2/k+1}$ (where $\eta_t=\alpha_t^2/\sigma_t^2$). When $k=1$, $r_t\equiv1$, recovering the original score. Because this is a closed-form expression rather than an additional network or iterative correction, the method is "plug-and-play" for any sampler.

**2. Generalization to Gaussian Mixtures: Retreating from "Global" to "Local" Temperature**

TSR distinguishes itself from previous methods by abandoning the attempt to change mode weights, focused instead on changing mode widths. The authors prove that when the data consists of "well-separated" isotropic Gaussian mixtures, the $r_t$ derived for a single Gaussian remains a bounded approximation of the true score. At small $t$, a single component dominates and the error decays exponentially; at large $t$, the distribution approximates pure noise and the error decays polynomially, tending to zero at both ends. Physically, TSR only compresses the variance within each mode without altering the relative weights between modes; consequently, sampling results uniformly cover all modes rather than collapsing to the center. This avoids the side effects of mode collapse seen in CNS and mode weight imbalance seen in CFG. The "local" property was validated on 1D GMMs, 2D checkerboards, and Swiss-roll: while CNS loses peripheral modes, TSR retains them all. Although real-world data are not explicit GMMs, any sufficiently smooth distribution can be locally approximated as Gaussian, making the formula universally applicable.

**3. Unified Adaptation for Score / $\epsilon$ / Velocity Parameterizations: One Formula for Heterogeneous Models**

To enable the same $r_t$ to drive both noise-prediction models (DDPM/DDIM) and velocity-prediction models (Flow Matching), all parameterizations are converted back to the score. Given the linear relationship $\mathbf{s}_\theta=-\sigma_t^{-1}\boldsymbol\epsilon_\theta$, diffusion models use $\tilde{\boldsymbol\epsilon}_\theta=r_t(k,\sigma)\boldsymbol\epsilon_\theta$. For flow matching, where velocity and score satisfy $\mathbf{s}_\theta=-(\alpha_t\boldsymbol v_\theta-\dot{\alpha}_t\mathbf{x})/[\sigma_t(\dot{\alpha}_t\sigma_t-\alpha_t\dot{\sigma}_t)]$, the rescaled velocity is $\tilde{\boldsymbol v}_\theta=\alpha_t^{-1}(r_t(k,\sigma)(\alpha_t\boldsymbol v_\theta-\dot{\alpha}_t\mathbf{x})+\dot{\alpha}_t\mathbf{x})$. Both $x_0$-prediction and $v$-prediction can be integrated similarly. This eliminates the need to rebuild sampling stacks for different models, allowing TSR to be used out-of-the-box for heterogeneous models like Stable Diffusion 2/3, Flux.1 dev, FoldingDiff, Marigold, and Pi-0.

### Loss & Training
No new training is required. Hyperparameters $k$ and $\sigma$ are selected by first fixing $\sigma=1.0$ and using binary search for the optimal $k$, then fixing $k$ and using binary search for the optimal $\sigma$, iterating if necessary. This is significantly more efficient than grid search, and empirically, the same $(k,\sigma)$ pair can transfer across different models for the same task.

## Key Experimental Results

### Main Results

| Task / Dataset | Model | Metric | Default Sampling | + TSR | Remarks |
|----------------|-------|--------|------------------|-------|---------|
| Text-to-Image / LAION Aesth. 5k | SD3 | FID ↓ / CLIP ↑ | 24.77 / 32.82 | **22.81 / 33.05** | $k=0.93, \sigma=3.0$; surpasses CFG Pareto front |
| Text-to-Image | SD2 | FID / CLIP | 22.81 / 33.66 | **19.75 / 33.75** | Transfer of same $(k,\sigma)$ |
| Text-to-Image | Flux.1 dev | FID / CLIP | 53.99 / 31.97 | **51.79 / 32.14** | Transfer of same $(k,\sigma)$ |
| Depth Estimation / ETH3D | Marigold(DDIM) | AbsRel ↓ / $\delta_1$ ↑ | 7.1 / 90.4 | **6.68 / 95.7** | Better than CNS (6.82 / 95.6) |
| Depth Estimation / NYUv2 | Marigold | AbsRel / $\delta_1$ | 6.0 / 95.9 | **5.84 / 96.0** | — |
| Pose Prediction / SYMSOL | $SO(3)$ diffusion | mean err (deg) ↓ | 0.444 | **0.356** ($k=7,\sigma=0.5$) | CNS slightly better (0.350) but SDE-only |
| Robotics / LIBERO-10 | Pi-0 (flow) | Avg Success ↑ | 81.7 | **82.8** ($k=1.25,\sigma=0.25$) | Improvement in 6/10 tasks |
| Protein Gen | FoldingDiff | designability ↑ | 0.22 | Significant improv. | Better FID than CNS; preserves diversity |

### Ablation Study

| Configuration | FID ↓ (SD3) | CLIP ↑ | Meaning |
|---------------|-------------|--------|---------|
| Default Euler-ODE| 24.77 | 32.82 | Baseline |
| + CFG Scaling | Pareto curve | Pareto curve | Diversity/Alignment trade-off |
| + CNS | Not usable in flow ODE | — | Limited to SDE samplers |
| + TSR ($k=0.93,\sigma=3.0$) | **22.81** | **33.05** | Surpasses CFG front; additive with CFG |
| TSR $k<1$ (Flatter) | More detail but noisy | — | Benefit for generative tasks |
| TSR $k>1$ (Sharper) | Smoother, lacks detail | — | Benefit for predictive tasks |

### Key Findings
- The "local temperature" concept is the core differentiator: CNS collapses to a central mode by suppressing noise with a constant, while TSR uses a time-dependent factor to suppress only intra-mode variance, thereby preserving all modes.
- Creative tasks (image generation) prefer $k<1$ (flatter), while predictive tasks (depth, pose, robotics, protein design) prefer $k>1$ (sharper). The unified framework serves both.
- Optimal $(k,\sigma)$ values transfer stably between different models of the same task, with tuning costs comparable to CFG.
- TSR is orthogonal to and can be used alongside CFG. It is natively compatible with deterministic samplers like DDIM/Euler-ODE with zero extra NFE.

## Highlights & Insights
- By using the minimalist observation that "the ratio between the score of temperature-scaled data and the original score has a closed form," the authors reduce a problem previously thought to require retraining or MCMC to a simple scalar multiplication. This is an elegant case of "analysis replacing computation."
- Relaxing temperature from "global" to "local" is a key conceptual shift: by giving up the ability to change mode weights and focusing on mode width, mode dropping is avoided. This trade-off—limiting scope to gain universality—is a strategy worth replicating in other generative control problems.
- The unified $r_t$ formula for score/$\epsilon$/velocity parameterizations demonstrates the value of the stochastic interpolant perspective—finding "minimal common variables" is a design philosophy that other inference-time intervention methods should mirror.

## Limitations & Future Work
- Restricted to "local" temperature: It cannot change the relative weights of modes in a GMM, making it less effective for tasks requiring the suppression of incorrect modes (e.g., suppressing artifacts in conditional sampling).
- Theoretical guarantees are limited to well-separated isotropic Gaussian mixtures; empirical validity is relied upon for complex real-world distributions. The error bounds become loose for data with small $\sigma$.
- $(k,\sigma)$ must be tuned for each task/model. Although the cost is similar to CFG, zero-shot adaptation is missing, and there is a lack of data-driven methods to estimate $\sigma$.
- Performance drops in cases where $k$ is inappropriate (e.g., LIBERO Task 2/8) suggest that for tasks where the base model already performs poorly, sharpening might amplify errors, requiring task-level dynamic control.

## Related Work & Insights
- **vs CFG (Ho & Salimans 2022)**: CFG works only for conditional models and requires training-time condition dropout; TSR works for both and is orthogonal to CFG.
- **vs CNS (Yim et al., 2023; Geffner et al., 2025)**: CNS scales the noise term by $1/\sqrt{k}$, is compatible only with SDEs, and loses modes; TSR scales the score with a time-dependent $r_t$ and is ODE-compatible and mode-preserving.
- **vs Likelihood-weighted Finetuning (Shih et al., 2023)**: Requires retraining and access to training data; TSR is entirely training-free.
- **vs Langevin / MCMC Corrector (Du et al., 2023)**: Multiplies inference overhead; TSR has zero extra NFE.
- **vs Variance-reduced sampling for proteins (Geffner et al., 2025)**: Essentially a special case of CNS; TSR achieves both higher designability and lower FID on FoldingDiff.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegant reasoning, though the general direction of rescaling scores is not a first. The analytic formula and "local temperature" concept are the highlights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Spans 5 distinct tasks, covers ODE and SDE samplers, and systematically compares with CFG/CNS.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, ample illustrations, and intuitive explanation of hyperparameters.
- Value: ⭐⭐⭐⭐⭐ High deployment value due to zero training cost, zero inference overhead, and immediate applicability to nearly all diffusion/flow models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](../../NeurIPS2025/computational_biology/consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[ICML 2026\] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)
- [\[ICML 2026\] Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)
- [\[ICLR 2026\] Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](../../ICLR2026/computational_biology/scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)

</div>

<!-- RELATED:END -->

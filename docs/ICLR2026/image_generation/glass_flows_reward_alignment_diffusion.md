---
title: >-
  [Paper Note] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models
description: >-
  [ICLR 2026 Oral][Image Generation][flow matching] This paper proposes GLASS (Gaussian Latent Sufficient Statistic) Flows — a novel "flow within a flow" sampling paradigm that recasts the stochastic Markov transition $p_{t'|t}(x_{t'} | x_t)$ as an internal ODE problem via Gaussian sufficient statistic reparameterization, reusing the pretrained denoiser without retraining. This enables Feynman-Kac Steering without sacrificing ODE efficiency or SDE stochasticity, consistently surpassing the Best-of-N ODE baseline on the FLUX text-to-image model and achieving a new state of the art in inference-time reward alignment.
tags:
  - ICLR 2026 Oral
  - Image Generation
  - flow matching
  - diffusion models
  - reward alignment
  - Feynman-Kac steering
  - GLASS
  - stochastic transitions
  - inference-time scaling
date: 2026-05-08
content_hash: c601510c9137f8b2
---

# GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models

**Conference**: ICLR 2026 Oral
**OpenReview**: [vH7OAPZ2dR](https://openreview.net/forum?id=vH7OAPZ2dR)
**Code**: Available
**Area**: Image Generation / Diffusion Models
**Keywords**: flow matching, diffusion models, reward alignment, Feynman-Kac steering, GLASS, stochastic transitions, inference-time scaling

## TL;DR
This paper proposes GLASS (Gaussian Latent Sufficient Statistic) Flows — a novel "flow within a flow" sampling paradigm that recasts the stochastic Markov transition $p_{t'|t}(x_{t'} | x_t)$ as an internal ODE problem via Gaussian sufficient statistic reparameterization, reusing the pretrained denoiser without retraining. This enables Feynman-Kac Steering without sacrificing ODE efficiency or SDE stochasticity, consistently surpassing the Best-of-N ODE baseline on the FLUX text-to-image model and achieving a new state of the art in inference-time reward alignment.

## Background & Motivation
**Background**: Flow matching and diffusion models can be enhanced at inference time through reward adaptation algorithms (inference-time scaling). Existing methods such as Sequential Monte Carlo (SMC) and Feynman-Kac Steering (FKS) require stochasticity in the denoising trajectory to explore high-reward regions.

**Limitations of Prior Work**: Stochastic transitions (SDE sampling) are far less efficient than deterministic ODE sampling and suffer severe quality degradation under low step counts. Experiments show that standard FKS with SDE transitions fails to outperform even the simple Best-of-N ODE baseline — revealing a fundamental tension between efficiency and stochasticity.

**Key Challenge**: Methods such as FKS/SMC theoretically require the stochastic branching provided by SDEs to effectively explore the posterior distribution, yet the computational and quality costs of SDEs render them impractical on real-world SOTA models. Best-of-N ODE is efficient but does not exploit intermediate reward signals.

**Goal**: Eliminate the trade-off between efficiency and stochasticity — enabling ODE sampling to produce rich stochastic transitions so that FKS can be genuinely effective.

**Key Insight**: The observation that the Gaussian transition kernel $p_{t'|t}$ can be transformed, via sufficient statistics and time reparameterization, into an internal conditional flow matching ODE driven by the pretrained denoiser.

**Core Idea**: Recast stochastic transitions as "internal flow matching" ODEs; by reusing the pretrained model through sufficient statistics, achieve "ODE speed + SDE diversity."

## Method

### Overall Architecture
Given the velocity field $u_t(x)$ and denoiser $D_t(x)$ of a pretrained flow matching model, GLASS Flows treats the two-step transition $x_t \to x_{t'}$ as a conditional generation problem. An auxiliary variable $\bar{X}_s$ ($s \in [0,1]$) is introduced to construct an internal flow ODE $\frac{d\bar{x}_s}{ds} = \bar{u}_s(\bar{x}_s | x_t, t)$, where $\bar{x}_0 \sim \mathcal{N}(\bar{\gamma} x_t, \bar{\sigma}_0^2 I)$ (the random initial condition provides stochasticity) and $\bar{x}_1 \sim p_{t'|t}(\cdot | x_t)$ (the terminal state follows the target transition distribution).

### Key Designs
1. **GLASS Transition Kernel Construction**:
    - Function: Defines a family of Gaussian Markov transitions $p_{t'|t}^{\text{GLASS}}$ parameterized by correlation parameter $\rho$.
    - Mechanism: Models $(X_t, X_{t'})$ as two "noisy observations" of a latent variable $Z$: $X_t = \alpha_t Z + \sigma_t \epsilon_1$, $X_{t'} = \alpha_{t'} Z + \sigma_{t'} \epsilon_2$, where $\text{Corr}(\epsilon_1, \epsilon_2) = \rho$. The joint distribution is
    $$\begin{pmatrix} X_t \\ X_{t'} \end{pmatrix} = \begin{pmatrix} \alpha_t \\ \alpha_{t'} \end{pmatrix} Z + \begin{pmatrix} \sigma_t \epsilon_1 \\ \sigma_{t'} \epsilon_2 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} \sigma_t^2 & \rho \sigma_t \sigma_{t'} \\ \rho \sigma_t \sigma_{t'} & \sigma_{t'}^2 \end{pmatrix}$$
    - Design Motivation: $\rho$ controls the degree of stochasticity. Setting $\rho = \alpha_t \sigma_{t'} / (\sigma_t \alpha_{t'})$ recovers the DDPM transition; $\rho = 1$ recovers the deterministic ODE. The default $\rho = 0.4$ is empirically optimal.

2. **Sufficient Statistic Reparameterization (Core Contribution)**:
    - Function: Proves that the GLASS denoiser can be expressed directly in terms of the pretrained denoiser $D_t$, requiring no retraining.
    - Mechanism: Defines the sufficient statistic $S(\mathbf{x}) = \frac{\mu^\top \Sigma^{-1}}{\mu^\top \Sigma^{-1} \mu} \begin{pmatrix} x_t \\ \bar{x}_s \end{pmatrix}$, where $\mu = (\alpha_t, \bar{\alpha}_s + \bar{\gamma}\alpha_t)^\top$. The GLASS denoiser is then
    $$D_{\mu, \Sigma}(x_t, \bar{x}_s) = D_{t^\star}(\alpha_{t^\star} S(\mathbf{x}))$$
    where $t^\star = g^{-1}((\mu^\top \Sigma^{-1} \mu)^{-1})$ and $g(t) = \sigma_t^2 / \alpha_t^2$ is the signal-to-noise ratio function. That is: compress two noisy observations into a sufficient statistic → denoise with the pretrained denoiser at equivalent time $t^\star$.
    - Design Motivation: The mathematical structure of Gaussian conjugacy and sufficient statistics guarantees an exact, training-free reparameterization with no additional training required.

3. **Internal Conditional Flow ODE**:
    - Function: Substitutes the GLASS denoiser into the conditional flow matching framework to obtain the internal ODE velocity field $\bar{u}_s(\bar{x}_s | x_t, t)$.
    - Mechanism: Adopts the CondOT schedule $\bar{\alpha}_s = s \bar{\alpha}_1$, $\bar{\sigma}_s = (1-s) \bar{\sigma}_0 + s \bar{\sigma}_1$, with velocity field
    $$\bar{u}_s = w_1(s) \bar{x}_s + w_2(s) D_{\mu(s), \Sigma(s)}(x_t, \bar{x}_s)$$
    where $w_1(s) = \frac{\dot{\bar{\sigma}}_s}{\bar{\sigma}_s}$ and $w_2(s) = \dot{\bar{\alpha}}_s - \bar{\alpha}_s \frac{\dot{\bar{\sigma}}_s}{\bar{\sigma}_s}$. Samples are obtained by integrating with the Euler method over $M$ steps.
    - Design Motivation: Each step requires only one neural network evaluation (identical to SDE), while leveraging the stability of ODE integrators for superior quality.

### Plug-and-Play Applications
- **FKS-GLASS**: Replaces SDE transitions in Feynman-Kac Steering with GLASS transitions, with particle reweighting and resampling.
- **GLASS + Gradient Guidance**: Augments the internal ODE with reward gradient $\nabla_y r(D_{t^\star}(y))\big|_{y=\alpha_{t^\star}S(\mathbf{x})}$.
- **Total NFE = K × M**: $K$ outer steps × $M$ internal ODE steps per transition. Fair comparison with SDE (equal total NFE).

## Key Experimental Results

### Main Results: FLUX Text-to-Image Reward Alignment (GenEval + PartiPrompts)

| Method | CLIP ↑ | PickScore ↑ | HPSv2 ↑ | ImageReward ↑ | GenEval ↑ |
|--------|--------|------------|---------|--------------|-----------|
| ODE (Best-of-8) | Baseline | Baseline | Baseline | Baseline | Baseline |
| FKS + SDE | < Best-of-8 | < Best-of-8 | < Best-of-8 | < Best-of-8 | ≈ |
| FKS + GLASS | **> Best-of-8** | **> Best-of-8** | **> Best-of-8** | **> Best-of-8** | **Improved** |
| FKS + GLASS + Guidance | **Highest** | **Highest** | **Highest** | **Highest** | **Highest** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Best-of-N (ODE) | Baseline | Simple but cannot exploit intermediate reward signals |
| Best-of-N (GLASS) | ≈ Best-of-N (ODE) | Same marginal distribution; terminal quality consistent |
| FKS + SDE | < Best-of-N (ODE) | Poor SDE quality degrades FKS performance |
| $\rho = 0.2, 0.4, 0.6, 0.8, 1.0$ | $\rho = 0.4$ optimal | All $\rho$ values achieve ODE-level sample quality |
| DreamSim Diversity | ODE ≈ SDE ≈ GLASS | All three methods sample from the same marginal distribution |
| SiT-XL ImageNet-256 | GLASS FID ≈ ODE FID | Effective on non-FLUX architectures as well |

### Key Findings
- **FKS + SDE fails on FLUX**: Standard SDE transitions severely degrade quality under FLUX's 50-step ODE configuration (producing residual noise artifacts), underperforming even Best-of-N ODE.
- **GLASS eliminates the efficiency–stochasticity trade-off**: FKS-GLASS consistently outperforms Best-of-N ODE across all 4 reward models × 2 benchmarks, whereas FKS-SDE does not.
- **GLASS stochasticity originates from the initial condition**: $\bar{X}_0 \sim \mathcal{N}(\bar{\gamma} x_t, \bar{\sigma}_0^2 I)$ provides stochastic branching, while the subsequent evolution is a deterministic ODE — in contrast to SDE's stepwise noise injection.
- **GLASS exactly preserves marginal distributions**: Theoretically proven that composing GLASS transitions yields $X_{t_k} \sim p_{t_k}$ for any $\rho$.

## Highlights & Insights
- **Conceptual originality and elegance of "flow within a flow"**: Treating a single stochastic transition as a complete conditional flow matching problem yields a conceptually clean and straightforward implementation.
- **Mathematical elegance of sufficient statistic construction**: Compressing two noisy observations into one via Gaussian conjugacy enables exact reuse of the pretrained denoiser with zero additional training.
- **Addresses a practical bottleneck**: FKS/SMC is unusable on SOTA models due to poor SDE quality; GLASS directly resolves this bottleneck.
- **Practical utility as a drop-in replacement**: No model modification, no retraining, only a change of sampler — any existing method relying on SDE transitions can benefit immediately.
- **Complementary to RL fine-tuning**: GLASS can accelerate SDE sampling in RL training pipelines such as DDPO/Flow-GRPO, and can also be applied at inference time on already fine-tuned models.

## Limitations & Future Work
- Relies on the Gaussian transition kernel assumption; applicability to non-Gaussian architectures is unverified.
- The correlation parameter $\rho$ is currently constant; a time-dependent adaptive schedule $\rho(t, t')$ is theoretically feasible.
- Validation is limited to FLUX and SiT-XL; other architectures (SD3, Stable Cascade) remain untested.
- Increasing the number of internal ODE steps $M$ raises computational cost; optimal $M$ selection is task-dependent.
- Numerical stability is not thoroughly compared against discrete-time diffusion models (e.g., standard discrete schedules of DDPM).

## Related Work & Insights
- **vs. DDPM/SDE sampling**: In the continuous limit, GLASS samples exactly the same transition distribution but replaces SDE integration with an ODE integrator, improving both quality and efficiency.
- **vs. TADA (arXiv:2506.21757)**: TADA employs the same mathematical tools of Gaussian conjugacy and sufficient statistics, but for a different purpose (augmented dynamics for training-free improvement). GLASS uses them to construct stochastic transitions as a substitute for SDEs.
- **vs. Transition Matching (DTM)**: DTM is a special case corresponding to $\rho = 1$, but it constitutes a distinct pretraining paradigm (per-patch approximation) and is not directly comparable.
- **vs. Best-of-N**: Best-of-N uses only terminal rewards with independent samples; GLASS + FKS exploits intermediate reward signals and particle resampling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The "flow within a flow" concept and sufficient statistic construction are highly original and mathematically elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validation across 4 reward models × 2 benchmarks on FLUX 768×1360 is convincing, though comparisons across more architectures are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and clearly presented, building progressively from intuition to formalism; reviewer dnmw rated the paper as having "excellent soundness."
- Value: ⭐⭐⭐⭐⭐ — Directly addresses a practical bottleneck in inference-time reward alignment, with strong utility as a drop-in replacement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[NeurIPS 2025\] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models](../../NeurIPS2025/image_generation/psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)

</div>

<!-- RELATED:END -->

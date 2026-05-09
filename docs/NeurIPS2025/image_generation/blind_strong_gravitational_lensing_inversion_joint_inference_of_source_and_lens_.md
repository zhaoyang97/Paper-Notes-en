---
title: >-
  [Paper Note] Blind Strong Gravitational Lensing Inversion: Joint Inference of Source and Lens Mass with Score-Based Models
description: >-
  [NeurIPS 2025][Image Generation][Strong gravitational lensing] This work presents the first application of score-based generative model priors to blind strong gravitational lensing inversion — jointly inferring the morphology of background source galaxies and lens mass distribution parameters. By extending GibbsDDRM to the continuous-time domain, the method achieves reconstruction residuals consistent with observational noise and unbiased marginal posteriors over lens parameters.
tags:
  - NeurIPS 2025
  - Image Generation
  - Strong gravitational lensing
  - blind inversion
  - score-based models
  - joint inference
  - dark matter
date: 2026-05-08
content_hash: 567d713e1c1751ba
---

# Blind Strong Gravitational Lensing Inversion: Joint Inference of Source and Lens Mass with Score-Based Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.04792](https://arxiv.org/abs/2511.04792)
**Code**: Available
**Area**: Scientific Computing / Astronomy / Generative Models
**Keywords**: Strong gravitational lensing, blind inversion, score-based models, joint inference, dark matter

## TL;DR
This work presents the first application of score-based generative model priors to blind strong gravitational lensing inversion — jointly inferring the morphology of background source galaxies and lens mass distribution parameters. By extending GibbsDDRM to the continuous-time domain, the method achieves reconstruction residuals consistent with observational noise and unbiased marginal posteriors over lens parameters.

## Background & Motivation

**Background**: Score-based models have been successfully applied to astronomical inverse problems, including interferometric imaging, gravitational lensing source reconstruction, and cosmological field inference. In strong gravitational lensing, they serve as data-driven priors for inferring background sources from distorted multiple images.

**Limitations of Prior Work**:
- Previous score-based lensing analyses assume the lens mass distribution (i.e., the forward operator) is known, whereas in practice the lens parameters are also unknown.
- Jointly inferring the source and lens ("blind inversion") is highly challenging — the lens parameter posterior contains multiple local minima, and degeneracies exist between source and lens.
- Prior blind inversion methods rely on analytic priors (Gaussian assumptions or smoothness constraints), limiting the flexibility of source reconstruction.

**Key Challenge**: Score-based priors are highly expressive but require a known forward operator (lens model), whereas in astronomy the forward operator is itself unknown — necessitating simultaneous inference of the data and the operator.

**Goal**: Achieve joint inference of source and lens (blind inversion) within the score-based prior framework.

**Key Insight**: Adapt the discrete-time GibbsDDRM — a diffusion sampler for inverse problems with unknown forward operators — to the continuous-time domain, combined with two likelihood score approximations (CLA and ΠiGDM).

**Core Idea**: Continuous-time GibbsDDRM + score-based source prior + parametric lens model = first score-based blind lensing inversion.

## Method

### Overall Architecture
Joint posterior sampling over $p(\mathbf{x}, \boldsymbol{\ell} | \mathbf{y})$, where $\mathbf{x}$ is the pixelized source and $\boldsymbol{\ell}$ denotes 12 lens parameters (elliptical power-law profile + external shear + multipole moments). Sampling alternates between: (1) fixing $\boldsymbol{\ell}$ and drawing $\mathbf{x}$ via a guidance-based diffusion sampler; and (2) fixing $\mathbf{x}$ and drawing $\boldsymbol{\ell}$ via Langevin dynamics. The forward model $\mathbf{y} = A_{\boldsymbol{\ell}} \mathbf{x} + \boldsymbol{\eta}$ is linear given $\boldsymbol{\ell}$, with Gaussian noise.

### Key Designs

1. **Continuous-Time GibbsDDRM**:

   - **Function**: Extends discrete-time GibbsDDRM to the continuous-time SDE framework.
   - **Mechanism**: Alternately updates the source (via diffusion guidance) and lens parameters (via Langevin steps) within the reverse SDE, using the score-based prior $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ together with an approximate likelihood score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} | \mathbf{x}_t)$.
   - **Design Motivation**: The continuous-time formulation enables flexible noise scheduling via VE-SDE (variance-exploding), avoiding the step-size constraints of discrete-time methods.

2. **Two Likelihood Score Approximations**:

   - **Function**: Approximate the intractable $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{y} | \mathbf{x}_t)$.
   - **CLA (Conditional Likelihood Approximation)**: Assumes $p_t(\mathbf{y} | \mathbf{x}_t)$ is proportional to $p(\mathbf{y} | \hat{\mathbf{x}}_0(\mathbf{x}_t))$.
   - **ΠiGDM**: Uses a projection-based approximation, more accurate but computationally costlier.
   - **Design Motivation**: Exact computation requires marginalizing over all possible $\mathbf{x}_0$; the two approximations offer different accuracy–efficiency trade-offs across scenarios.

3. **Lens Parameter Sampling**:

   - **Function**: Sample the 12 lens macro-parameters conditioned on the reconstructed source.
   - **Mechanism**: Exploits the differentiability of the forward model to perform gradient-based Langevin steps over $\boldsymbol{\ell}$. Following each source sample, 500 conditional lens samples are drawn to adequately explore parameter space.
   - **Design Motivation**: The parameter space contains multiple local minima; MCMC sampling (rather than point optimization) enables proper uncertainty quantification.

### Loss & Training
- The score model is pretrained on simulated unlensed galaxy images.
- Lensing simulations use the Caustics code (EPL + external shear + $m=3$ multipole moments + PSF).
- Joint sampling produces 406 source–lens posterior samples.

## Key Experimental Results

### Main Results
Reconstruction quality on simulated strong lensing systems:

| Metric | Result | Description |
|--------|--------|-------------|
| Residual consistency | Residual / noise ~ N(0,1) | Reconstruction residuals match measurement noise |
| Lens parameter bias | True values within marginal posterior | No systematic bias |
| Source reconstruction fidelity | Highly consistent with ground truth | Fine structure recovered |
| Posterior sample count | 406 joint samples | Each augmented with 500 conditional lens samples |

### Ablation Study: CLA vs. ΠiGDM

| Approximation | Source Reconstruction | Lens Parameter Accuracy | Computational Cost |
|---------------|-----------------------|-------------------------|--------------------|
| CLA | Good | Good | Lower |
| ΠiGDM | Better | Better | Higher |

### Key Findings
- **First successful score-based blind lensing inversion**: Prior work was limited to conditional inversion with a known lens.
- **Residual statistics validation**: The normalized residual distribution is consistent with a unit Gaussian, serving as a gold standard for reconstruction quality.
- **All 12 lens parameter marginal posteriors are unbiased**: True values fall within the credible intervals of the posteriors.
- **Multimodal posteriors are captured**: Joint sampling successfully identifies degeneracy directions between source and lens parameters.
- **Continuous-time formulation is more stable than discrete-time**: VE-SDE noise scheduling is more flexible than DDPM stepping.

## Highlights & Insights
- **Direct transfer of ML blind inverse problem methods (GibbsDDRM) to astronomy** — demonstrating the practical value of cross-domain methodological translation.
- **Joint source+lens inference** unlocks the full applicability of score-based models in gravitational lensing science; the prior assumption of a known lens was a severe practical limitation.
- The method has direct implications for automated analysis pipelines in upcoming Rubin/LSST and Euclid surveys, which are expected to discover ~200,000 strong lensing systems.

## Limitations & Future Work
- Validation is currently limited to simulated data; testing on real HST/JWST observations remains to be done.
- The 12-parameter lens model is parametric; more complex mass distributions (e.g., dark matter substructure) require extensions.
- Computational cost remains substantial: generating 406 joint samples requires significant GPU time.
- Only single-lens systems are tested; multi-plane group lensing configurations are considerably more complex.
- The quality of source reconstruction is directly affected by the score model's training data distribution — out-of-distribution sources may be poorly reconstructed.

## Related Work & Insights
- **vs. traditional MCMC lensing analysis (PyAutoLens)**: PyAutoLens uses parametric sources (Sérsic profiles); this work employs a score-based nonparametric prior.
- **vs. GibbsDDRM (Murata et al.)**: This work is the first to extend GibbsDDRM to continuous time and apply it to a scientific inverse problem.
- **vs. Adam et al. (2022)**: Prior score-based lensing analysis assumed a known lens; this work relaxes that assumption.
- The methodology provides a reference for other scientific blind inversion problems, such as joint calibration and imaging in radio interferometry.

## Rating
- Novelty: ⭐⭐⭐⭐ First score-based blind lensing inversion; high scientific significance.
- Experimental Thoroughness: ⭐⭐⭐ Workshop-level scope, but residual validation is rigorous; real-data testing pending.
- Writing Quality: ⭐⭐⭐⭐ Effective conceptual bridging between astronomy and ML.
- Value: ⭐⭐⭐⭐ Direct contribution to gravitational lensing analysis and large-survey data processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[NeurIPS 2025\] MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction](mge-ldm_joint_latent_diffusion_for_simultaneous_music_generation_and_source_extr.md)
- [\[NeurIPS 2025\] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models](psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)

</div>

<!-- RELATED:END -->

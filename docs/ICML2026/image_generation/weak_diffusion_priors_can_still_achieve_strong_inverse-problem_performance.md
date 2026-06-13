---
title: >-
  [Paper Note] Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance
description: >-
  [ICML 2026][Image Generation][Diffusion model priors] The paper discovers that diffusion model priors with low fidelity or domain mismatch can still achieve strong performance in information-rich inverse problems—explain…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diffusion model priors"
  - "inverse problems"
  - "weak priors"
  - "Bayesian inference"
  - "latent noise optimization"
date: 2026-05-08
content_hash: d1a2462ce4b8ce95
---

# Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2601.22443](https://arxiv.org/abs/2601.22443)  
**Code**: To be confirmed  
**Area**: Image Generation / Diffusion Models / Inverse Problem Solving  
**Keywords**: Diffusion model priors, inverse problems, weak priors, Bayesian inference, latent noise optimization

## TL;DR
The paper discovers that diffusion model priors with low fidelity or domain mismatch can still achieve strong performance in information-rich inverse problems—explaining this seemingly contradictory phenomenon through Bayesian consistency theory and local correlation analysis, and providing explicit conditions for when weak priors are effective.

## Background & Motivation

**Background**: Diffusion models are widely used as priors for solving inverse problems due to their powerful generative capabilities. Standard practice involves using "full-strength" high-fidelity diffusion models (such as 1000-step DDPM) where the training data matches the target task.

**Limitations of Prior Work**: In practical applications, the ideal prior is often unavailable—memory constraints force researchers to use DDIM samplers with only 3-4 steps, and data-scarce fields like medical imaging cannot train domain-specific models. These "weak priors" are theoretically expected to yield limited reconstruction quality.

**Key Challenge**: Experimentally, the performance of weak priors is often comparable to or even better than full-strength priors—Wang et al. achieved a 22-66 dB PSNR gain using a 3-step DDIM for inverse problems, and Jalal et al. reconstructed knee MRIs using a single-mode brain MRI model. Currently, such successes are mostly anecdotal and lack a systematic theoretical explanation.

**Goal**: Decomposition into two questions—(1) Under what conditions are inverse problems robust to the choice of prior? (2) Are weak priors truly as "weak" as their sample quality suggests?

**Key Insight**: In high-dimensional measurement settings, the information content of the data may outweigh the constraints of the prior; although weak priors have poor sample quality, they retain local spatial structures similar to those of strong priors.

**Core Idea**: Use Bayesian consistency theory to characterize how the posterior concentrates when measurement information is rich; demonstrate through local correlation diagnosis that weak priors share similar local statistical structures with strong priors.

## Method

### Overall Architecture
An initial noise optimization framework is used to solve the inverse problem $y = \mathcal{A}(x) + \epsilon$. Treating the generative model $G$ as a black box, the latent variable $z$ is optimized directly to minimize $\arg\min_{z} \|\mathcal{A}(G(z)) - y\|_2^2$. This avoids backpropagation through chains of hundreds of sampling steps, making extremely weak (3-step) generators feasible.

### Key Designs

1. **AdamSphere Optimizer**:

    - **Function**: Constrains the latent variables to be optimized on a Gaussian sphere, preventing them from deviating from the typical high-dimensional Gaussian shell.
    - **Mechanism**: Projects $z$ onto the unit sphere $\|z\|=\sqrt{d}$ at each step, utilizing the natural manifold learned by the diffusion model.
    - **Design Motivation**: Standard Adam allows $z$ to deviate arbitrarily from the sphere, whereas during diffusion training, the mass of $z \sim \mathcal{N}(0, I_d)$ concentrates near $\|z\| \approx \sqrt{d}$. Constraining $z$ to the valid region improves sample quality.

2. **HoldoutTopK Early Stopping**:

    - **Function**: Prevents optimization from overfitting to noisy measurements by tracking loss on held-out measurements and selecting the Top-K best.
    - **Mechanism**: Unlike typical ML which selects a single optimal point, this strategy saves the most recent point among the Top-K; the validation set is a subset of measurements not used for optimization. Setting K > 1 eliminates noise fluctuations.
    - **Design Motivation**: Initial noise optimization is prone to overfitting noisy measurements, leading to reconstruction artifacts; HoldoutTopK improves PSNR by 3-5%.

3. **Bayesian Posterior Consistency Theory**:

    - **Function**: Describes when the posterior concentrates on the true signal consistent with measurements in high-dimensional single-observation inverse problems.
    - **Mechanism**: Models the generative prior as a Gaussian mixture $\pi(x)=\sum_{j=1}^M w_j \varphi(x;\mu_j,\tau^2 I_n)$. Theorem 3.2 proves that when the measurement dimension $m$ is sufficiently large and the optimal matching component has a score gap $\delta_0>0$, the posterior concentrates at an exponential rate of $CM\exp(-\delta_0 m)$. Even if prior weights $w_j$ vary significantly, posteriors from different priors concentrate on the same optimal component.
    - **Design Motivation**: Explains why weak priors remain feasible—under information dominance (high-dimensional measurements), the effect of the prior is overwhelmed by the data. At 70% inpainting, the score gap is 0.22-0.28, which is much larger than 0.

## Key Experimental Results

### Main Results: Cross-Domain Inverse Problem Solving

| Task | Method | Prior Domain | CelebA PSNR | Bedroom PSNR | Church PSNR |
|------|------|--------|------------|----------|----------|
| Inpainting | DPS | CelebA | 31.98 | 27.97 | 24.15 |
| Inpainting | Ours | CelebA (3-step) | 33.78 | 27.78 | 23.56 |
| Inpainting | Ours | Bedroom (3-step) | 32.76 | 28.88 | 24.22 |
| Inpainting | Ours | Church (3-step) | 32.62 | 28.66 | 24.93 |
| Super-Res | DPS | CelebA | 26.82 | 22.95 | 20.28 |
| Super-Res | Ours | CelebA (3-step) | 31.27 | 25.88 | 22.68 |
| Super-Res | Ours | Bedroom (3-step) | 30.34 | 26.59 | 22.86 |

Even in extreme scenarios with complete domain mismatch (e.g., using a Bedroom prior to reconstruct faces), the proposed method still outperforms DPS by 1-4 dB.

### Local Correlation Analysis

| Pixel Distance | CelebA 3-step | CelebA 20-step | Bedroom 3-step | Bedroom 20-step |
|---------|----------|----------|---------|---------|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1 | 0.9558 | 0.9814 | 0.9645 | 0.9615 |
| 4 | 0.8866 | 0.9100 | 0.8786 | 0.8573 |
| 8 | 0.7767 | 0.8108 | 0.7637 | 0.7437 |
| 16 | 0.5595 | 0.6261 | 0.5632 | 0.5618 |

The spatial autocorrelation decay trajectories remain similar regardless of changes in generative steps or training domains, validating the hypothesis of shared local structures.

### Key Findings
- Bayesian consistency + local correlation jointly explain the surprising effectiveness of weak priors.
- Failure modes: In large-area box inpainting and $16\times$ super-resolution, missing regions are too large $\to$ the posterior fails to concentrate $\to$ weak prior performance degrades.
- The combination of AdamSphere + HoldoutTopK enhances optimization stability.

## Highlights & Insights
- **Deep Integration of Theory and Empiricism**: Evolves from asking "why weak priors sometimes work" to defining "under what quantitative conditions they work."
- **Ingenuity of Local Correlation Diagnosis**: Spatial autocorrelation curves provide indirect evidence that "weak priors are not as weak as their sample performance suggests."
- **Precise Characterization of Failure Modes**: Theoretical predictions (small $m$ $\to$ failure of posterior concentration $\to$ strengthened dependence on the prior) correspond perfectly with experimental results.

## Limitations & Future Work
- Weak priors degrade significantly under large-area missingness or extremely high super-resolution factors.
- The tightness of the Gaussian mixture prior assumption in actual diffusion models has not been analyzed in depth.
- Generalizability to real medical data needs further validation.
- Improvements: Hybrid methods (weak priors + parameter-efficient fine-tuning); investigating the moment when posterior concentration conditions fail; adaptive early stopping.

## Related Work & Insights
- **vs DPS**: DPS requires traversing the diffusion chain while injecting measurement information at each step; the proposed initial noise optimization is competitive using just a 3-step generator.
- **vs General Theory of Generative Priors**: This is the first systematic characterization of the posterior concentration phenomenon within an inverse problem framework.
- **vs Medical Imaging Applications**: Provides a scientific basis for the practice of "no data = use a general prior."

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First to systematically explain the effectiveness of weak priors using Bayesian posterior consistency theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  4 types of inverse problems + 3 datasets + various prior strengths + failure mode analysis.
- Writing Quality: ⭐⭐⭐⭐⭐  Clear logical hierarchy (phenomenon $\to$ theory $\to$ diagnosis $\to$ application).
- Value: ⭐⭐⭐⭐⭐  Offers both deep theoretical contributions and practical guiding value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[ICML 2026\] LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)
- [\[ICML 2026\] Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion](beyond_generative_priors_minority_sampling_with_jepa-guided_diffusion.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)

</div>

<!-- RELATED:END -->

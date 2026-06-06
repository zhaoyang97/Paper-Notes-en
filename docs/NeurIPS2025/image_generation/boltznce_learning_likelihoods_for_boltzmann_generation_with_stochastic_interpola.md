---
title: >-
  [Paper Note] BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants
description: >-
  [NeurIPS 2025][Image Generation][Boltzmann distribution] BoltzNCE trains an Energy-Based Model (EBM) via a hybrid Score Matching + InfoNCE objective to approximate the likelihood of a Boltzmann Generator…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Boltzmann distribution"
  - "noise contrastive estimation"
  - "stochastic interpolants"
  - "molecular conformation"
  - "free energy"
date: 2026-05-08
content_hash: cb0e70b6af1381e6
---

# BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants

**Conference**: NeurIPS 2025
**arXiv**: [2507.00846](https://arxiv.org/abs/2507.00846)  
**Code**: Available  
**Area**: Generative Models / Molecular Simulation
**Keywords**: Boltzmann distribution, noise contrastive estimation, stochastic interpolants, molecular conformation, free energy

## TL;DR
BoltzNCE trains an Energy-Based Model (EBM) via a hybrid Score Matching + InfoNCE objective to approximate the likelihood of a Boltzmann Generator, eliminating expensive Jacobian trace computations. On alanine dipeptide conformation generation, it achieves a 100× inference speedup with a free energy error of only 0.02 $k_BT$.

## Background & Motivation
**Background**: Boltzmann Generators (BGs) are deep generative models designed to sample from Boltzmann distributions $p(x) \propto \exp(-E(x)/k_BT)$ defined by an energy function $E(x)$. Normalizing flow- and diffusion-based approaches can generate samples, but computing likelihoods requires expensive Jacobian determinants or trace estimation.

**Limitations of Prior Work**: Exact likelihood computation (e.g., the Hutchinson estimator for trace) is prohibitively slow at inference time — equivariant CNFs require 9.37 hours for alanine dipeptide — creating a bottleneck for applying BGs to free energy calculations and importance sampling.

**Key Challenge**: High sampling quality requires accurate flow/diffusion models, yet likelihood evaluation cost is independent of sampling quality: even when sampling is good, each likelihood evaluation remains expensive.

**Goal**: Decouple sampler quality from likelihood tractability by training a separate EBM to approximate the likelihood, thereby avoiding Jacobian computation entirely.

**Key Insight**: Decompose the BG pipeline into two stages — first train a Boltzmann Emulator via flow matching, then train an EBM to approximate its density using NCE combined with score matching.

**Core Idea**: Use NCE-trained EBMs as fast likelihood surrogates for Boltzmann Generators, achieving a 100× inference speedup.

## Method

### Overall Architecture
The framework consists of two stages: (1) train a Boltzmann Emulator using stochastic interpolants and flow matching to generate samples from the Boltzmann distribution; (2) train an EBM $\hat{U}(x)$ on the generated samples via a hybrid Score Matching + InfoNCE objective, such that $\exp(\hat{U}(x))$ approximates the true density.

### Key Designs

1. **Stochastic Interpolant Framework**:

    - **Function**: Constructs a smooth interpolation path $I_t = \alpha_t x_0 + \beta_t x_1$ between noise $x_0$ and Boltzmann samples $x_1$.
    - **Mechanism**: A vector field $v_t$ is trained to match the probability flow induced by the interpolant, enabling a mapping from noise to the Boltzmann distribution.
    - **Design Motivation**: Stochastic interpolants naturally bridge ODE flows and diffusion processes, and provide a well-defined score function that facilitates subsequent EBM training.

2. **BoltzNCE Hybrid Training**:

    - **Function**: Jointly trains the EBM using Score Matching and InfoNCE.
    - **Score Matching Loss**: $\mathcal{L}_{SM} = \mathbb{E}[|\alpha_t \nabla\hat{U}_t(\tilde{I}_t) + x_0|^2]$ — enforces that the EBM gradient matches the score of the interpolation process.
    - **InfoNCE Loss**: $\mathcal{L}_{InfoNCE} = -\mathbb{E}[\log\frac{\exp(\hat{U}_t(\tilde{I}_t))}{\sum_{t'}\exp(\hat{U}_{t'}(\tilde{I}_t))}]$ — learns density through contrastive learning across time steps.
    - **Combined**: $\mathcal{L}_{BoltzNCE} = \mathcal{L}_{SM} + \mathcal{L}_{InfoNCE}$
    - **Design Motivation**: InfoNCE alone yields good global density estimates but inaccurate gradients; Score Matching alone yields accurate gradients but poor global density calibration — the two objectives are complementary.

3. **Free Energy Estimation via Importance Reweighting**:

    - **Function**: Uses EBM likelihoods for importance sampling to correct sampling bias.
    - **Mechanism**: $\hat{Z} = \sum_i w_i$, $w_i = \exp(-E(x_i)/k_BT) / \hat{\rho}(x_i)$; free energy difference: $\Delta F = -k_BT \ln(\hat{Z}_A / \hat{Z}_B)$.

### Loss & Training
Two-stage training. Stage 1: flow matching with an equivariant vector field. Stage 2: Score Matching + InfoNCE; total training time approximately 12 hours per epoch.

## Key Experimental Results

### Main Results (Alanine Dipeptide)

| Method | $\Delta F / k_BT$ | Error | Inference Time |
|--------|-------------------|-------|----------------|
| Umbrella Sampling (ground truth) | 4.10 ± 0.26 | – | – |
| ECNF (exact likelihood) | 4.09 ± 0.05 | 0.01 | 9.37h |
| GVP Vector Field | 4.38 ± 0.67 | 0.28 | 18.4h |
| **BoltzNCE** | **4.08 ± 0.13** | **0.02** | **0.09h** |

### Ablation Study

| Configuration | KL Divergence (8-mode Gaussian) | Notes |
|---------------|----------------------------------|-------|
| InfoNCE only | 0.2395 | Inaccurate gradients |
| Score Matching only | 0.2199 | Poor global density |
| **BoltzNCE (combined)** | **0.0150** | 15× improvement |

| Configuration | KL (Checkerboard) | Notes |
|---------------|-------------------|-------|
| InfoNCE only | 3.8478 | Difficulty aligning multimodal distributions |
| **BoltzNCE** | **0.1987** | 19× improvement |

### Key Findings
- The hybrid objective improves KL divergence by 15–19× over either individual loss, confirming that the two objectives are genuinely complementary.
- Inference is 100× faster (0.09h vs. 9.37h), as EBM forward passes are substantially cheaper than Jacobian trace computation.
- Free energy error of 0.02 $k_BT$ is comparable to the exact method (0.01 $k_BT$).
- The approach generalizes to 7 dipeptide systems with acceptable error (0.43 $k_BT$) and a 6× speedup.

## Highlights & Insights
- **Decoupling Sampling from Likelihood**: Separating a high-quality sampler from a fast likelihood estimator is an elegant design choice — the generative model need not be likelihood-tractable; an independent EBM serves as a surrogate. This paradigm is broadly applicable to any setting where likelihoods are required but the generative model does not provide them.
- **Complementarity of NCE and SM**: InfoNCE provides global density alignment through contrastive comparison across time steps, while Score Matching ensures local gradient accuracy. Their combination yields a 15× improvement in 2D experiments, offering strong empirical evidence of complementarity.
- **Practical Acceleration for Scientific Computing**: A 100× inference speedup has substantial practical value for molecular simulation, making free energy calculations feasible on a single GPU rather than requiring a compute cluster.

## Limitations & Future Work
- Validation is limited to small molecules (dipeptides); scalability to large proteins and complex systems remains unexplored.
- Generalization to unseen dipeptide systems incurs larger errors (0.43 vs. 0.02 $k_BT$), suggesting that fine-tuning may be necessary.
- EBM training itself requires approximately 12 hours, which, while a one-time cost, is non-negligible.
- Approximation errors in the learned likelihood may be amplified in the extreme tails of the distribution.

## Related Work & Insights
- **vs. ECNF (Köhler et al.)**: ECNF computes exact Jacobians but is extremely slow; BoltzNCE uses an approximate EBM and is 100× faster, with comparable free energy accuracy.
- **vs. Targeted Free Energy Perturbation**: Classical reweighting methods rely on sufficient distributional overlap; BoltzNCE improves overlap by learning the likelihood directly.
- **vs. Flow Matching (Lipman et al., 2023)**: Stage 1 of BoltzNCE employs flow matching to construct the sampler, while Stage 2 — EBM training via the hybrid objective — constitutes the primary novel contribution.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of sampling–likelihood decoupling with a hybrid NCE/SM objective in the context of Boltzmann generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2D synthetic benchmarks + alanine dipeptide + generalization across 7 dipeptide systems.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; 2D experiments intuitively illustrate the complementarity of the two loss terms.
- Value: ⭐⭐⭐⭐ Significant practical impact for molecular simulation and free energy calculation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] FALCON: Few-step Accurate Likelihoods for Continuous Flows](falcon_few-step_accurate_likelihoods_for_continuous_flows.md)
- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[ICML 2026\] Coarse-Grained Boltzmann Generators](../../ICML2026/image_generation/coarse-grained_boltzmann_generators.md)

</div>

<!-- RELATED:END -->

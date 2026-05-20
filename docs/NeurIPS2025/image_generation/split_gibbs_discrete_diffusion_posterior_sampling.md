---
title: >-
  [Paper Note] Split Gibbs Discrete Diffusion Posterior Sampling
description: >-
  [NeurIPS 2025][Image Generation][Discrete Diffusion Models] This paper proposes SGDD (Split Gibbs Discrete Diffusion), a plug-and-play posterior sampling algorithm for discrete diffusion models based on the split Gibbs s…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Discrete Diffusion Models"
  - "Posterior Sampling"
  - "Split Gibbs Sampling"
  - "Inverse Problems"
  - "DNA Sequence Design"
date: 2026-05-08
content_hash: eb3f77fe2477f0e1
---

# Split Gibbs Discrete Diffusion Posterior Sampling

**Conference**: NeurIPS 2025
**arXiv**: [2503.01161](https://arxiv.org/abs/2503.01161)  
**Code**: [GitHub](https://github.com/chuwd19/Split-Gibbs-Discrete-Diffusion-Posterior-Sampling)  
**Area**: Diffusion Models / Discrete Generation
**Keywords**: Discrete Diffusion Models, Posterior Sampling, Split Gibbs Sampling, Inverse Problems, DNA Sequence Design

## TL;DR

This paper proposes SGDD (Split Gibbs Discrete Diffusion), a plug-and-play posterior sampling algorithm for discrete diffusion models based on the split Gibbs sampling principle. By introducing auxiliary variables and a Hamming-distance-based regularization potential, SGDD decomposes posterior sampling into alternating likelihood and prior sampling steps, achieving substantial improvements over baselines on DNA sequence design, discrete image inverse problems, and music infilling tasks.

## Background & Motivation

**Background**: Posterior sampling methods for diffusion models have made significant progress in continuous spaces (e.g., DPS, SMC, variational methods), with broad applications in image restoration and scientific inverse problems. However, these methods rely on gradient information of the form $\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\mathbf{x}_t)$, which cannot be directly extended to discrete state spaces.

**Limitations of Prior Work**: In discrete state spaces, $\log p(\mathbf{y}|\mathbf{x})$ is defined only on a finite support and admits no meaningful gradient. Existing discrete diffusion posterior sampling methods (e.g., SVDD-PM) require approximating value functions via reinforcement learning-style techniques, which are difficult to tune and exhibit limited performance under complex guidance signals.

**Key Insight**: The split Gibbs sampling framework naturally avoids reliance on likelihood gradients. By carefully designing a regularization potential such that the prior step corresponds exactly to a partial denoising process of a discrete diffusion model, this work achieves plug-and-play posterior sampling for discrete diffusion models.

## Method

### Overall Architecture

**Objective**: Sample from the posterior $p(\mathbf{x}|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{x})p(\mathbf{x})$, where $p(\mathbf{x})$ is modeled by a pretrained discrete diffusion model.

**Mechanism**: An auxiliary variable $\mathbf{z}$ is introduced to construct an augmented distribution $\pi(\mathbf{x}, \mathbf{z}; \eta) \propto \exp(-f(\mathbf{z};\mathbf{y}) - g(\mathbf{x}) - D(\mathbf{x}, \mathbf{z}; \eta))$. As the regularization parameter $\eta \to 0$, both marginals $\pi^X(\mathbf{x}; \eta)$ and $\pi^Z(\mathbf{z}; \eta)$ converge to the target posterior. Gibbs sampling is then used to alternate between likelihood and prior steps.

### Key Designs

1. **Hamming-Distance-Based Regularization Potential (Core Innovation)**: To ensure that the prior step exactly corresponds to the denoising process of a discrete diffusion model, a special potential function is designed:
    $$D(\mathbf{x}, \mathbf{z}; \eta) = d(\mathbf{x}, \mathbf{z}) \log \frac{1+(N-1)e^{-\eta}}{(N-1)(1-e^{-\eta})}$$
   where $d(\mathbf{x}, \mathbf{z})$ is the Hamming distance. **Design Motivation**: As $\eta \to 0^+$, $D \to \infty$ (unless $\mathbf{x}=\mathbf{z}$), guaranteeing convergence; simultaneously, the prior sampling step $\mathbf{x}^{(k+1)} \sim \pi(\mathbf{x}, \mathbf{z}=\mathbf{z}^{(k)}; \eta) \propto p_0(\mathbf{x})(\tilde\beta/(1-\tilde\beta))^{d(\mathbf{z}^{(k)}, \mathbf{x})}$ is exactly equivalent to conditional denoising of $\mathbf{z}^{(k)}$ by the discrete diffusion model at noise level $\sigma_t = \eta$.

2. **Likelihood Sampling Step**: At each iteration, the likelihood step samples from $\pi(\mathbf{x}=\mathbf{x}^{(k)}, \mathbf{z}; \eta) \propto \exp(-f(\mathbf{z};\mathbf{y}) - D(\mathbf{x}^{(k)}, \mathbf{z}; \eta))$. Since the unnormalized probability density is directly computable, MCMC methods such as Metropolis-Hastings can be applied efficiently. **Key Advantage**: No gradient of the likelihood function is required, making the approach naturally compatible with discrete spaces.

3. **Annealing Schedule**: An annealing schedule $\{\eta_k\}_{k=0}^{K-1}$ decreasing from large to small values is employed to accelerate Markov chain mixing. Large $\eta$ promotes exploration, while small $\eta$ progressively approaches the posterior.

### Convergence Guarantee (Theorem 1)

The convergence analysis of SGDD leverages generalized Fisher information. Accounting for imperfect score functions and CTMC discretization errors, it is shown that the average relative Fisher divergence converges at a rate of $O(1/K)$:
$$\frac{1}{K}\sum_{k=0}^{K-1} \text{avg Fisher} \leq \frac{2\text{KL}(\pi_0\|\mu_0)}{Kt^*} + \frac{4M\epsilon}{c} + \frac{2MLt^*}{cH}$$
This provides stronger theoretical guarantees than existing methods (e.g., SVDD-PM, which relies on surrogate value functions and infinite-particle assumptions).

## Key Experimental Results

### Main Results 1: Posterior Sampling Accuracy on Synthetic Data

| Dimension D | Method | Hellinger ↓ | TV ↓ |
|-------------|--------|------------|------|
| D=2 | SVDD-PM (M=100) | 0.275 | 0.292 |
| D=2 | SMC | 0.238 | 0.248 |
| D=2 | DPS | 0.182 | 0.176 |
| D=2 | **SGDD** | **0.149** | **0.125** |
| D=10 | SVDD-PM (M=100) | 0.448 | 0.494 |
| D=10 | DPS | 0.410 | 0.453 |
| D=10 | **SGDD** | **0.334** | **0.365** |

### Main Results 2: DNA Enhancer Sequence Design (Reward-Guided Generation)

| Method | Pred-Activity (median) ↑ | Pred-Activity (avg) ↑ | ATAC Acc% ↑ | Log-likelihood ↑ |
|--------|--------------------------|----------------------|-------------|-------------------|
| SVDD-PM | 5.41 | 5.08 | 49.9 | -241 |
| DRAKES w/ KL | 5.61 | 5.24 | 92.5 | -264 |
| CG | 2.90 | 2.76 | 0.0 | -265 |
| **SGDD (β=50)** | **9.14** | **8.96** | **93.0** | -261 |

SGDD achieves a median activity **42% higher** than the previous SOTA (DRAKES).

### Main Results 3: Discrete Image Inverse Problems (MNIST XOR/AND)

| Task | Method | PSNR ↑ | Accuracy% ↑ |
|------|--------|--------|-------------|
| XOR | SVDD-PM | 11.81 | 51.4 |
| XOR | SMC | 10.05 | 27.8 |
| XOR | **SGDD** | **20.17** | **91.2** |
| AND | SVDD-PM | 10.04 | 33.7 |
| AND | **SGDD** | **17.25** | **79.4** |

SGDD improves PSNR by **8.36 dB** on the XOR task.

### Ablation Study

| Configuration | Hellinger (D=5) | Notes |
|---------------|----------------|-------|
| SGDD (full) | 0.214 | With annealing schedule |
| Fixed η (no annealing) | ~0.35 | Annealing is critical |
| H=5 Euler steps | ~0.30 | Insufficient steps |
| H=20 Euler steps | 0.214 | Default setting |

### Key Findings

- As dimensionality increases (D=2→10), the accuracy of all methods degrades, but SGDD degrades most slowly.
- Under sparse observations (e.g., large masked regions), SGDD is capable of sampling from multiple plausible modes (e.g., digits 1, 4, 7, 9), demonstrating strong posterior diversity.
- The guidance strength $\beta$ controls the trade-off between reward and prior: larger $\beta$ yields higher activity but lower log-likelihood.

## Highlights & Insights

- The split Gibbs framework avoids gradient computation, making it the most natural choice for discrete posterior sampling.
- The regularization potential is elegantly designed as a function of Hamming distance, enabling the prior step to be exactly equivalent to diffusion denoising without any approximation.
- The theoretical guarantees are stronger than those of existing methods — they do not rely on surrogate value functions and explicitly account for discretization errors.
- The algorithm is straightforward to implement, requiring only an off-the-shelf discrete diffusion model and a likelihood function, achieving true plug-and-play behavior.

## Limitations & Future Work

- The likelihood sampling step relies on MCMC, which may mix slowly in high-dimensional discrete spaces.
- The choice of the annealing schedule $\{\eta_k\}$ currently depends on empirical tuning, lacking an adaptive strategy.
- The quality of posterior sampling is directly affected by the pretraining quality of the discrete diffusion model.
- Validation on larger-scale discrete generation tasks (e.g., protein sequence design) has not yet been conducted.

## Related Work & Insights

- SGDD is analogous to split Gibbs methods in continuous spaces (PnP-DM, DiffPIR), but generalizes the $\ell_2$ regularization to Hamming distance regularization.
- SGDD is complementary to SVDD-PM (value-function approximation) and SMC methods, as it requires no value function training.
- This work may inspire extensions of the split Gibbs framework to other discrete structures, such as graph structures and combinatorial optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The Hamming distance potential is elegantly designed, though the split Gibbs framework itself has established precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers four settings — synthetic data, DNA, images, and music — with comprehensive baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and theoretical derivations are complete, though notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides a unified and effective solution for discrete diffusion posterior sampling with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Thompson Sampling via Fine-Tuning of LLMs
description: >-
  [Medical Imaging] This paper proposes ToSFiT, which extends Thompson Sampling to large-scale unstructured discrete spaces by fine-tuning large language models to directly parameterize the Probability of Maximality (PoM), thereby circumventing the intractability of acquisition function maximization.
tags:
  - Medical Imaging
date: 2026-05-08
content_hash: 7ec662966881b6ad
---

# Thompson Sampling via Fine-Tuning of LLMs

- **Conference**: ICLR 2026
- **arXiv**: [2510.13328](https://arxiv.org/abs/2510.13328)
- **Code**: [GitHub](https://github.com/IBM/thompson-sampling-via-fine-tuning-of-llms)
- **Area**: Medical Imaging
- **Keywords**: Thompson Sampling, Bayesian Optimization, LLM Fine-Tuning, Probability of Maximality, VBOS

## TL;DR

This paper proposes ToSFiT, which extends Thompson Sampling to large-scale unstructured discrete spaces by fine-tuning large language models to directly parameterize the Probability of Maximality (PoM), thereby circumventing the intractability of acquisition function maximization.

## Background & Motivation

Bayesian optimization faces fundamental challenges in large-scale unstructured discrete spaces (e.g., amino acid sequences, quantum circuit designs): without gradient information, maximizing the acquisition function over combinatorially large discrete domains is computationally infeasible. For instance, the protein sequence space of length 100 over 20 amino acids exceeds the number of atoms in the observable universe.

Thompson Sampling (TS) is a classical Bayesian optimization strategy that balances exploration and exploitation by sampling from the reward posterior and selecting the point that maximizes the drawn sample. The induced sampling distribution is intrinsically the Probability of Maximality (PoM). However, directly sampling from the PoM in large discrete domains still requires enumerating all candidates.

**Mechanism**: Since pre-trained LLMs already encode rich prior knowledge, the paper asks whether the generative distribution of an LLM can directly parameterize the PoM, thereby recasting Thompson Sampling as an LLM fine-tuning problem.

## Method

### Overall Architecture

The core idea of ToSFiT is to treat candidate generation as Thompson Sampling, parameterize the PoM with a pre-trained LLM, and incrementally adapt the LLM to the posterior PoM via the VBOS objective. The overall pipeline is:

1. Generate initial candidates via a prompt-conditioned LLM and collect observations.
2. Fit a Gaussian process reward model.
3. Iterate: generate candidates → estimate VBOS gradient → fine-tune LLM → observe new candidates.

### Variational Bayesian Optimistic Sampling (VBOS)

The PoM can be approximated by maximizing the VBOS objective:

$$\mathcal{V}(\pi) = \mathbb{E}_{x \sim \pi}\left[\mu_x + \sqrt{-2\ln(\pi_x)} \cdot \sigma_x\right]$$

where $\mu_x$ is the posterior mean and $\sigma_x$ is the posterior standard deviation. The term $\sqrt{-2\ln(\pi_x)}$ serves as an adaptive UCB exploration bonus.

### VBOS Gradient Derivation (Proposition 1)

$$\frac{d}{d\theta}\mathcal{V}(\pi^\theta) = \mathbb{E}_{x \sim \pi^\theta}\left[(\mu_x - \xi - v^{-1}(\pi_x^\theta) \cdot \sigma_x) \cdot \frac{d}{d\theta}\ln\pi_x^\theta\right]$$

where $v^{-1}(u) = \sqrt{-2\ln u} - 1/\sqrt{-2\ln u}$. This gradient admits an energy-based model interpretation: when the LLM-implied expected reward $\mu_x^\theta$ underestimates the true $\mu_x$, the generation probability is upweighted.

### Gradient Stabilization

- **RLOO baseline** (Reinforce Leave-One-Out) is applied to reduce variance.
- Normalization is performed via the empirical standard deviation of the advantage function.
- This is mathematically equivalent to GRPO (Group Relative Policy Optimization).

### Scalable Gaussian Process

A feature map $\phi: X \to \mathbb{H}$ converts the kernel to a linear kernel, yielding complexity $\Theta(\dim(\mathbb{H})^2)$ independent of the number of observations.

### Loss & Training

$$\frac{d}{d\theta}\mathcal{V}(\pi^\theta) \approx \frac{1}{B}\sum_i \frac{\hat{\hat{r}}_{x_i}^\theta - \xi_i}{\widehat{\text{advantage std}}} \cdot \frac{d}{d\theta}\ln\pi_{x_i}^\theta$$

## Theoretical Analysis

**Theorem 1 (Core Theoretical Contribution)**: The cumulative regret bound for exact VBOS is improved from $\tilde{\mathcal{O}}(\sqrt{T|X|})$ to $\tilde{\mathcal{O}}(\sqrt{T\gamma^T})$ (where $\gamma^T$ denotes the maximum information gain), and the first regret bound for approximate VBOS is established:

$$\mathbb{E}\left[\sum_{t=1}^T R^* - R_{x^t}\right] \leq \sqrt{C_{\sigma_n} H T \gamma^T} + \mathbb{E}\sum_{t=1}^T D_{\sigma^t}(\pi^t, \tilde{\pi}^t)$$

A key insight is that policy initialization must remain close to the prior (pre-training + context), and fine-tuning must be conservative (small learning rate) to preserve prior knowledge.

## Key Experimental Results

### Three Tasks

| Task | Model | Search Space | Reward |
|------|-------|--------------|--------|
| FAQ Answer Optimization | Qwen3-1.7B/8B | All token sequences | Semantic alignment score |
| Protein Search | ProtGPT2-0.7B | Amino acid sequences | Thermostability index |
| Quantum Circuit Design | Qwen2.5-Coder-1.5B/7B | Qiskit circuit code | Negative energy |

### Main Results

ToSFiT achieves state-of-the-art sample efficiency and computational efficiency across all three tasks, substantially outperforming seven baseline methods including in-context BO, reinforcement learning, and evolutionary search.

### Key Findings

- **Importance of strong priors**: Removing key contextual information from the prompt (e.g., qubit count) significantly degrades performance.
- **Conservative fine-tuning**: Excessively large learning rates cause prior forgetting and performance stagnation.
- **Batch optimization**: Larger batch sizes reduce sample efficiency but improve per-iteration efficiency.
- **Compute–sample efficiency trade-off**: Increasing the number of gradient steps per round further improves sample efficiency.

### Ablation Study

| Ablation | Effect |
|----------|--------|
| Remove prior context | Significant performance degradation |
| Large learning rate | Initial improvement followed by stagnation |
| Increase gradient steps | Improved sample efficiency |
| Increase batch size | Improved iteration efficiency |

## Highlights & Insights

1. Seamless integration of theory and practice: the new regret bound directly informs algorithm design.
2. Clever exploitation of LLM pre-training priors, avoiding acquisition function maximization in discrete spaces.
3. The energy-based model interpretation of the VBOS gradient is elegant and intuitive.
4. Generality is validated across three highly diverse experimental tasks (NLP, protein design, quantum computing).

## Limitations & Future Work

1. A fixed feature map is used without joint learning with the GP.
2. Full-model fine-tuning incurs non-trivial computational and memory overhead.
3. The scalable GP assumes a linear kernel, limiting the expressiveness of the reward model.
4. Evaluation is restricted to sequence generation tasks; other discrete spaces such as graph structures are not explored.

## Related Work & Insights

- **Discrete-domain BO**: Bal et al. (2025) assume a Cartesian product decomposition; Swersky et al. (2020) optimize via local mutation strategies.
- **VAE relaxation**: Kusner et al. (2017) and others relax discrete spaces into continuous latent spaces.
- **Deep kernel learning**: Ranković & Schwaller (2025) learn feature maps online.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Combines Thompson Sampling with LLM fine-tuning, contributing both theoretically and methodologically.
- **Value**: ⭐⭐⭐⭐ — Applicable to practical domains such as protein design and circuit optimization.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous and experimental design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Opens a new direction for integrating LLMs with Bayesian optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)

</div>

<!-- RELATED:END -->

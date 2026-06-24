---
title: >-
  [Paper Note] Generalized Interpolating Discrete Diffusion
description: >-
  [ICML2025][LLM (Other)][Discrete Diffusion] The Generalized Interpolating Discrete Diffusion (GIDD) framework is proposed, which generalizes Masked Diffusion Models (MDM) to a family of diffusion processes supporting arbitrary time-varying mixture distributions. By combining mask and uniform noise, GIDD equips the model with self-correction capabilities and achieves compute-matched SOTA in diffusion language modeling.
tags:
  - "ICML2025"
  - "LLM (Other)"
  - "Discrete Diffusion"
  - "Masked Diffusion"
  - "Uniform Noise"
  - "Self-Correction"
  - "ELBO"
  - "Language Models"
date: 2026-05-08
content_hash: 56f28f58e6c85fcd
---

# Generalized Interpolating Discrete Diffusion

**Conference**: ICML2025  
**arXiv**: [2503.04482](https://arxiv.org/abs/2503.04482)  
**Code**: [dvruette/gidd](https://github.com/dvruette/gidd/)  
**Area**: Discrete Diffusion Models / Language Modeling  
**Keywords**: Discrete Diffusion, Masked Diffusion, Uniform Noise, Self-Correction, ELBO, Language Models

## TL;DR

The Generalized Interpolating Discrete Diffusion (GIDD) framework is proposed, which generalizes Masked Diffusion Models (MDM) to a family of diffusion processes supporting arbitrary time-varying mixture distributions. By combining mask and uniform noise, GIDD equips the model with self-correction capabilities and achieves compute-matched SOTA in diffusion language modeling.

## Background & Motivation

Autoregressive language models (such as GPT) predict tokens sequentially during generation, which entails two fundamental limitations: (1) generating a sequence of length $N$ requires $N$ forward passes; (2) generated tokens cannot be modified, meaning errors propagate irreversibly to subsequent steps once they occur.

Discrete diffusion models generate data by progressively adding noise and then learning to reverse this process, decoupling the number of inference steps from the sequence length. However, the current most popular **Masked Diffusion Models (MDM)** only use the [MASK] token as noise—once a token is filled, it is never changed, essentially reintroducing the "irreversibility" issue of autoregressive generation.

The core motivation of this paper is that if a fraction of **uniform noise** (randomly replacing tokens with other tokens instead of masks) is mixed into the diffusion process, the model must learn to distinguish between "correct" and "incorrect" tokens, thereby acquiring self-correction capabilities. To realize this, a unified theoretical framework is required to support arbitrary noise designs.

## Method

### GIDD Forward Process

GIDD generalizes masked diffusion into a **generalized interpolating** form. Given data $x$, the marginal transition at time $t$ is:

$$q_t(z_t|x) = \mathrm{Cat}(z_t;\;\alpha_t \mathbf{x} + \beta_t \boldsymbol{\pi}_t)$$

where $\alpha_t$ is the signal-to-noise ratio schedule (monotonically decreasing from 1 to 0), and $\boldsymbol{\pi}_t$ is a **time-varying mixture distribution** that can be any probability distribution. When $\boldsymbol{\pi}_t = \mathbf{m}$ (the one-hot representation of the mask), it degenerates to standard MDM.

This work proves the existence of a Markov chain that satisfies the above marginals and derives the closed-form solution for the cumulative transition matrix:

$$Q_t = \alpha_t I + \beta_t \boldsymbol{\pi}_t \mathbf{1}^\top$$

### Mixture Schedule Design

In practice, a mixture of mask and uniform noise is adopted:

$$q_t(z_t|x) = \frac{1}{C_t}\big((1-t)\mathbf{x} + t\mathbf{m} + c_t \mathbf{u}\big)$$

where $\mathbf{u} = \frac{1}{N-1}(\mathbf{1}-\mathbf{m})$ represents the uniform distribution, $c_t = Bt^{\gamma/2}(1-t)^{\gamma/2}$ controls the amount of uniform noise, and $p_u$ is the expected proportion of uniform tokens at $t=0.5$. Setting $p_u=0$ falls back to pure masked diffusion.

### GIDD ELBO

Based on Continuous-Time Markov Chains (CTMC), a general ELBO is derived, which consists of two parts:

1. **KL Divergence Term**: The KL divergence between the predicted distribution $q_t(\cdot|\mathbf{x}_\theta)$ of the model and the true conditional distribution $q_t(\cdot|x)$.
2. **IS Divergence Term**: The pointwise divergence at the sampling state $z_t$.

Both terms vanish simultaneously if and only if the model perfectly matches the ground-truth distribution, ensuring that the ELBO has a global minimum.

### Loss Weight Rescaling

The ELBO weight $w_t(z_t, x)$ grows exponentially as $t\to 0$ and $t\to 1$, leading to optimization instability. The paper proposes two schemes:

- **Clamp**: $\tilde{w}_t^{\mathrm{clamp}} = \min(w_{\max}, w_t)$, a simple truncation with $w_{\max}=1$.
- **Dynamic**: $\tilde{w}_t^{\mathrm{dyn}} = w_{\max}(1 + \delta_{z_t,m} + (\frac{B}{N}e^{-\lambda_t/2}-1)\delta_{z_t,x})$, which preserves the relative weights across different token types.

The combination of Dynamic weight and weight decay (0.02), designated as **GIDD+**, achieves the best performance.

### Self-Correction Sampling

After generation is complete, the entire denoised sequence $Z_{t_0}$ is fed into the model and resampled with temperature $\tau$; the token with the highest model confidence among those differing from the original is selected for replacement. This process is iterated until convergence.

## Key Experimental Results

The 110M (small) and 320M (base) models are trained on OpenWebText, using a DiT architecture with a GPT-2 tokenizer.

### Validation Perplexity (PPL)

| Model | Training Tokens | PPL ↓ |
|------|------------|-------|
| GPT2 | - | 23.40 |
| Llama 110M (re-trained) | 262B | 16.11 |
| MDLM | 262B | 23.21 |
| MDM (re-impl.) | 262B | 23.36 |
| **GIDD+ (p_u=0.0)** | **262B** | **22.29** |

### Ablation: Impact of Weighting Schemes

| Scheme | p_u=0.0 | p_u=0.1 | p_u=0.2 |
|------|---------|---------|---------|
| No Rescaling | 24.36 | 26.88 | 28.22 |
| + clamp | 23.23 | 25.09 | 26.40 |
| + dynamic | 23.24 | 23.90 | 24.64 |
| + weight decay (GIDD+) | **23.05** | **23.67** | **24.38** |

### Self-Correction Performance (Base Model)

After applying self-correction to the $p_u=0.2$ model: the generation PPL drops from 214 to **93.3** (↓56%), and self-accuracy increases from 62.0% to **73.5%**. Conversely, applying self-correction to pure masked models degrades generation quality.

### GPT-4o Quality Rating

| Model | Clarity | Grammar | Factuality | Style | Creativity |
|------|--------|------|--------|------|--------|
| GIDD (p_u=0.0) + Self-Correction | -20.9% | -19.3% | -16.2% | -21.1% | -19.5% |
| GIDD (p_u=0.2) + Self-Correction | **+16.5%** | **+16.6%** | **+8.5%** | **+13.4%** | **+5.5%** |

### Zero-Shot Benchmark

GIDD+ (p_u=0.0) achieves an average accuracy of **39.30** across 7 benchmarks including ARC, BoolQ, Hellaswag, and PIQA, outperforming GPT2-small (38.77) and MDM (38.25).

## Highlights & Insights

1. **Theoretical Elegance**: GIDD is a rigorous generalization of masked diffusion, deriving closed-form cumulative transitions and ELBO, accompanied by a complete proof of global optimality.
2. **Self-Correction** is the true differentiator of discrete diffusion compared to autoregressive models—not merely a post-processing step, but naturally acquired during training via uniform noise.
3. **Loss Weight Rescaling** yields significant gains: dynamic weighting reduces the PPL starting from 28.22 to 24.64 for $p_u=0.2$.
4. **Advantage at Low Inference Steps**: With 32-step denoising, the generation PPL of $p_u=0.1$ (387) is substantially better than that of pure masking (904).
5. Likelihood evaluation (PPL/benchmarks) is not entirely aligned with generation quality—masked models exhibit better likelihood but worse generation quality, suggesting that diffusion language models require more comprehensive evaluation frameworks.

## Limitations & Future Work

1. **Limited Scale**: The maximum model size is 320M parameters; the efficacy of mixture noise at larger scales remains to be verified.
2. **PPL degradation from uniform noise** requires more model capacity to compensate, necessitating further scaling behavior validation.
3. Self-correction relies on iterative inference, which increases generation latency.
4. The evaluation of generation quality relies on "generation PPL" (scored by a larger model), an evaluation metric that itself may have biases.
5. The hyperparameter design for mixture scheduling ($p_u$, $\gamma$, $B$) remains somewhat heuristic, and the search space is not fully explored.
6. Validation is limited to language modeling; its applicability to other discrete data domains such as code generation and protein sequences remains to be explored.

## Related Work & Insights

- **Austin et al. (2023)**: First to introduce the diffusion ELBO to discrete Markov chains.
- **MDLM / MD4 (Sahoo et al., 2024; Shi et al., 2024)**: Simplifying the objective of masked diffusion.
- **BERT (Devlin et al., 2019)**: Pre-training via masking + random replacement inspired the mixture noise design in GIDD.
- **Discrete Flow Matching (Gat et al., 2024)**: Adapting the flow matching paradigm to discrete data.

## Rating

- Novelty: ⭐⭐⭐⭐ — Generalizing masked diffusion into a unified framework and introducing controllable uniform noise is a highly novel direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ — While scaling limitations exist, the ablation studies are comprehensive, presenting evaluations of both likelihood and generation quality.
- Writing Quality: ⭐⭐⭐⭐⭐ — The theoretical derivations are clear and rigorous, with intuitive diagrams.
- Value: ⭐⭐⭐⭐ — Self-correction is an important breakthrough in discrete diffusion, and the framework offers substantial flexibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] COSMIC: Generalized Refusal Direction Identification in LLM Activations](../../ACL2025/llm_nlp/cosmic_generalized_refusal_direction_identification_in_llm_activations.md)
- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](../../ACL2025/llm_nlp/language_codec_bridging_discrete_codec_speech_language_models.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](../../ACL2025/llm_nlp/analyzing_and_mitigating_inconsistency_in_discrete_speech_tokens_for_neural_code.md)
- [\[ICLR 2026\] Teaching Metric Distance to Discrete Autoregressive Language Models](../../ICLR2026/llm_nlp/teaching_metric_distance_to_discrete_autoregressive_language_models.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](../../ACL2025/llm_nlp/segment_level_diffusion.md)

</div>

<!-- RELATED:END -->

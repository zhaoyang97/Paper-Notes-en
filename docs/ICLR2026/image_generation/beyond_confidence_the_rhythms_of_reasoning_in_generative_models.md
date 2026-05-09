---
title: >-
  [Paper Note] Beyond Confidence: The Rhythms of Reasoning in Generative Models
description: >-
  [ICLR 2026][Image Generation][Token Constraint Bound] This paper proposes the Token Constraint Bound ($\delta_{\text{TCB}}$) metric, which quantifies the largest perturbation to an LLM's hidden state that preserves the next-token prediction, measuring local prediction robustness and revealing instabilities that traditional perplexity fails to capture.
tags:
  - ICLR 2026
  - Image Generation
  - Token Constraint Bound
  - prediction robustness
  - hidden state perturbation
  - output embedding geometry
  - prompt engineering
date: 2026-05-08
content_hash: 035f8c031b97e05e
---

# Beyond Confidence: The Rhythms of Reasoning in Generative Models

**Conference**: ICLR 2026
**arXiv**: [2602.10816](https://arxiv.org/abs/2602.10816)
**Code**: None
**Area**: Image Generation
**Keywords**: Token Constraint Bound, prediction robustness, hidden state perturbation, output embedding geometry, prompt engineering

## TL;DR
This paper proposes the Token Constraint Bound ($\delta_{\text{TCB}}$) metric, which quantifies the largest perturbation to an LLM's hidden state that preserves the next-token prediction, measuring local prediction robustness and revealing instabilities that traditional perplexity fails to capture.

## Background & Motivation
**State of the Field**: LLMs are highly sensitive to minor variations in input context—minor formatting changes can cause accuracy to fluctuate by 76%, and reordering in-context examples can shift accuracy from 54% to 93%.

**Limitations of Prior Work**:
   - Accuracy provides only an aggregate view and cannot assess the stability of individual predictions.
   - Perplexity conflates probability distributions and ignores the geometric structure of internal states.
   - Softmax normalization can yield high-probability yet unstable predictions—high probability may stem from relative normalization rather than robust internal states.

**Root Cause**: A high-probability, high-confidence prediction may correspond to an unstable equilibrium in the internal state space—existing metrics cannot distinguish between genuinely stable high confidence and fragile high confidence.

**Paper Goals**: Quantify the robustness of the internal state $\mathbf{h}$ produced by an LLM in a given context to small perturbations.

**Starting Point**: Analyze the first-order sensitivity of softmax outputs to hidden states via the Jacobian matrix.

**Core Idea**: Prediction robustness = the maximum perturbation radius around the hidden state that preserves the output distribution, determined by the geometric dispersion of output embeddings.

## Method

### Overall Architecture
The final-layer hidden state $\mathbf{h} \in \mathbb{R}^d$ is mapped through the output weight matrix $\mathbf{W} \in \mathbb{R}^{\mathcal{V} \times d}$ and softmax to yield probability distribution $\mathbf{o}$. $\delta_{\text{TCB}}$ quantifies the radius of a perturbation ball around $\mathbf{h}$ within which the change in $\mathbf{o}$ does not exceed tolerance $\epsilon$.

### Key Designs

1. **Token Constraint Bound ($\delta_{\text{TCB}}$) Definition**:

    - Function: Measures the robustness of LLM predictions to perturbations of the internal state.
    - Mechanism: Using the first-order linear approximation $\Delta\mathbf{o} \approx \mathbf{J}_\mathbf{W}(\mathbf{h}) \Delta\mathbf{h}$, the condition $\|\Delta\mathbf{o}\|_2 \leq \epsilon$ implies $\|\Delta\mathbf{h}\|_2 \leq \epsilon / \|\mathbf{J}_\mathbf{W}(\mathbf{h})\|_F$, yielding the definition $\delta_{\text{TCB}}(\mathbf{h}) = \epsilon / \|\mathbf{J}_\mathbf{W}(\mathbf{h})\|_F$.
    - Design Motivation: A larger $\delta_{\text{TCB}}$ indicates that the model's prediction remains stable under a wider range of hidden state perturbations.

2. **Precise Connection to Output Embedding Geometry**:

    - Function: Derives an analytic expression for the Jacobian norm.
    - Mechanism: Proves that $\|\mathbf{J}_\mathbf{W}(\mathbf{h})\|_F^2 = \sum_{i=1}^{\mathcal{V}} o_i^2 \|\mathbf{w}_i - \boldsymbol{\mu}_\mathbf{w}(\mathbf{h})\|_2^2$, where $\boldsymbol{\mu}_\mathbf{w}(\mathbf{h}) = \sum_j o_j \mathbf{w}_j$ is the probability-weighted mean embedding.
    - Geometric Interpretation: Sensitivity is determined by the dispersion of token embeddings relative to the weighted centroid, weighted by $o_i^2$—the embedding positions of high-probability tokens exert the greatest influence.

3. **Analysis of Two Prediction Regimes**:

    - **High-confidence regime** (low $\mathcal{V}_{\text{eff}}$): $\boldsymbol{\mu}_\mathbf{w} \to \mathbf{w}_k$ (dominant token), $\delta_{\text{TCB}} \to \infty$. In this regime, $\delta_{\text{TCB}}$ strongly correlates with the top-2 logit margin ($r = 0.62$).
    - **Uncertain regime** (high $\mathcal{V}_{\text{eff}}$): Probability is distributed across multiple tokens; $\delta_{\text{TCB}}$ correlates positively with $\sqrt{\mathcal{V}_{\text{eff}}}$ ($r = 0.95$). Crucially, even with high $\mathcal{V}_{\text{eff}}$, if the embeddings of high-probability tokens are geometrically clustered, $\delta_{\text{TCB}}$ can remain large.

### Loss & Training
- $\delta_{\text{TCB}}$ is an analytical metric and does not involve training.
- Computation requires only a forward pass to obtain $\mathbf{h}$, $\mathbf{o}$, and $\mathbf{W}$, followed by evaluation of the analytic formula.
- $\epsilon = 1.0$ is adopted as the normalization standard.

## Key Experimental Results

### Main Results — Prediction Regime Validation (LLaMA-3.1-8B)

| Dataset | Corr($\delta_{\text{TCB}}, \mathcal{V}_{\text{eff}}$) | Corr($\delta_{\text{TCB}}, z_{top1} - z_{top2}$) |
|--------|------|------|
| Diverse Prompts (N=309) | **0.95** (strong positive) | -0.40 |
| Low-$\mathcal{V}_{\text{eff}}$ Targeted (N=360) | 0.08 (near zero) | **0.62** (strong positive) |

### Ablation Study — Embedding Geometry Validation

| Embedding Operation | Rate at which $\delta_{\text{cluster}} > \delta_{\text{orig}} > \delta_{\text{disperse}}$ holds |
|---------|---|
| Low $\mathcal{V}_{\text{eff}}$ (< 20) | 95% |
| Overall | 90% |

- Holding $\mathbf{o}$ fixed, artificially clustering or dispersing competing token embeddings causes $\delta_{\text{TCB}}$ to increase or decrease accordingly.
- **This confirms that geometric structure influences prediction stability independently of the probability distribution.**

### Key Findings
- **$\delta_{\text{TCB}}$ discriminates prompt quality**: well-constructed prompts yield higher $\delta_{\text{TCB}}$, even when accuracy is identical.
- **Identifies instabilities missed by perplexity**: in text generation, positions with low perplexity but sharp drops in $\delta_{\text{TCB}}$ may correspond to semantic turning points or potential errors.
- **ICL example effects are reflected in $\delta_{\text{TCB}}$**: effective few-shot examples not only improve accuracy but also increase $\delta_{\text{TCB}}$.

## Highlights & Insights
- **High probability ≠ stability**: This core insight is highly valuable—softmax normalization may create a "false sense of security," while $\delta_{\text{TCB}}$ directly probes the true stability of internal states.
- **Dominant role of embedding geometry**: Even with an identical probability distribution, altering the geometric structure of the embedding space changes prediction stability—this offers insights into representation learning in LLMs.
- **Elegant analytic formula**: $\|\mathbf{J}\|_F^2 = \sum o_i^2 \|\mathbf{w}_i - \boldsymbol{\mu}\|^2$ reduces the complex Jacobian norm to an intuitively clear weighted dispersion measure.

## Limitations & Future Work
- The first-order linear approximation may be inaccurate for large perturbations.
- Validation is limited to LLaMA-3.1-8B; experiments across more models and scales are needed.
- The choice of $\epsilon = 1.0$ lacks theoretical justification.
- It remains unexplored how to incorporate $\delta_{\text{TCB}}$ into training objectives to directly improve robustness.
- The Frobenius norm as a sensitivity measure may be overly conservative compared to the spectral norm.

## Related Work & Insights
- **vs. Perplexity**: PPL measures sequence likelihood; $\delta_{\text{TCB}}$ measures local prediction robustness—the two are complementary rather than substitutes.
- **vs. Calibration metrics**: Calibration concerns the alignment between probability and correctness; $\delta_{\text{TCB}}$ concerns the stability of predictions under perturbation—orthogonal dimensions.
- **vs. Adversarial robustness**: Adversarial research seeks worst-case perturbations in input space; $\delta_{\text{TCB}}$ quantifies safety margins in hidden state space.

## Rating
- Novelty: ⭐⭐⭐⭐ Jacobian analysis is not new, but connecting it to output embedding geometry and defining a practically meaningful metric is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical validation + prompt analysis + ICL analysis + text generation analysis, though model diversity is limited.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, though the exposition is somewhat verbose.
- Value: ⭐⭐⭐⭐ Provides a new analytical perspective on LLMs with practical utility for prompt engineering and reliability evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)
- [\[ICLR 2026\] Verifier-Constrained Flow Expansion for Discovery Beyond the Data](verifier-constrained_flow_expansion_for_discovery_beyond_the_data.md)
- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICLR 2026\] QVGen: Pushing the Limit of Quantized Video Generative Models](qvgen_pushing_the_limit_of_quantized_video_generative_models.md)
- [\[ICLR 2026\] NeuralOS: Towards Simulating Operating Systems via Neural Generative Models](neuralos_towards_simulating_operating_systems_via_neural_generative_models.md)

<!-- RELATED:END -->

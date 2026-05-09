---
title: >-
  [Paper Note] Superposition Yields Robust Neural Scaling
description: >-
  [NeurIPS 2025][LLM Pretraining][Neural scaling laws] This paper identifies representational superposition as the core driver of neural scaling laws: in the strong-superposition regime, loss **universally** scales inversely with model dimension ($L \propto 1/m$), independent of the specific form of the data frequency distribution—consistent with empirical scaling behavior in real LLMs.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - Neural scaling laws
  - superposition
  - representation learning
  - LLM theory
  - weight decay
date: 2026-05-08
content_hash: b58f8c8971027b95
---

# Superposition Yields Robust Neural Scaling

**Conference**: NeurIPS 2025
**arXiv**: [2505.10465](https://arxiv.org/abs/2505.10465)
**Code**: [GitHub](https://github.com/liuyz0/SuperpositionScaling)
**Area**: LLM Pre-training
**Keywords**: Neural scaling laws, superposition, representation learning, LLM theory, weight decay

## TL;DR

This paper identifies representational superposition as the core driver of neural scaling laws: in the strong-superposition regime, loss **universally** scales inversely with model dimension ($L \propto 1/m$), independent of the specific form of the data frequency distribution—consistent with empirical scaling behavior in real LLMs.

## Background & Motivation

Neural scaling laws are a central empirical regularity in modern AI: larger models achieve lower loss, following a power-law relationship. Yet their **origin** remains poorly understood.

Limitations of existing explanations:
- Data-manifold/function-fitting theories require the data distribution itself to be a power law in order to produce power-law scaling
- Skill-learning models (Hutter 2021, Michaud et al. 2023) similarly rely on power-law distribution assumptions
- Kernel-method analyses depend on power-law decay of eigenvalues
- These explanations operate in the weak-superposition regime (finite-variance regime), which may not match the regime in which LLMs actually operate

Key observation: LLMs must represent over fifty thousand tokens and numerous abstract concepts within hidden spaces of only a few thousand dimensions. This implies that LLMs necessarily operate in a **superposition** regime—representing far more features than the model dimension.

Core Problem: How does superposition affect neural scaling laws?

## Method

### Overall Architecture

The paper adopts Anthropic's (2022) superposition toy model (an autoencoder) and systematically studies how the degree of superposition affects scaling behavior.

**Input generation**: $x_i = u_i v_i$, where $u_i \sim \text{Bernoulli}(p_i)$, $v_i \sim U(0,2)$
- $p_i$ is the frequency (importance) of feature $i$, sorted by frequency
- Activation density $E = \sum_i p_i$

**Model**: $h = W^T x$ (encoding), $y = \text{ReLU}(Wh + b)$ (decoding)
- $W \in \mathbb{R}^{n \times m}$, $n$ is the number of features, $m$ is the model dimension, $m \ll n$
- Loss $L = \langle \|y - x\|_2^2 \rangle_x$

### Key Designs: Controlling Superposition via Weight Decay

The paper introduces decoupled weight decay (positive or negative) to systematically control the degree of superposition:

$$W_{i,t+1} = \begin{cases} W_{i,t} - \eta_t \gamma W_{i,t}, & \gamma \geq 0 \\ W_{i,t} - \eta_t \gamma W_{i,t}(1/\|W_{i,t}\|_2 - 1), & \gamma < 0 \end{cases}$$

- $\gamma > 0$ (positive weight decay): suppresses superposition → weak-superposition regime
- $\gamma < 0$ (negative weight decay): encourages unit norms → strong-superposition regime

Superposition metric: $\phi_{1/2} = |\{i: \|W_i\|_2 > 1/2\}| / n$
- Weak superposition: $\phi_{1/2} \approx m/n$ (only the $m$ most important features are represented)
- Strong superposition: $\phi_{1/2} \approx 1$ (nearly all features have representations)

### Analysis in the Weak-Superposition Regime

Under ideal non-superposition conditions, the top $\phi_{1/2} n$ most frequent features are perfectly represented and the rest are ignored:

$$L \approx \langle v^2 \rangle \sum_{i > \phi_{1/2} n} p_i$$

When $p_i \propto 1/i^\alpha$, $L \propto m^{-(\alpha-1)}$ (a power law only when $\alpha > 1$).

**Conclusion**: Under weak superposition, the existence and exponent of the scaling law **depend on the specific form of the data frequency distribution**.

### Analysis in the Strong-Superposition Regime

The dominant loss source becomes geometric overlap between representation vectors $(W_i \cdot W_j)^2$.

Key geometric properties:
1. **Random unit vectors**: The mean squared inner product of two random unit vectors in $\mathbb{R}^m$ is $1/m$
2. **Equiangular tight frames (ETF)**: Representations of approximately $m^2/2$ important features approach an ETF structure
3. **Welch bound**: $\max_{i \neq j} |w_i \cdot w_j| \geq \sqrt{\frac{\nu - m}{m(\nu - 1)}} \approx 1/\sqrt{m}$

Consequently, squared overlap typically scales as $1/m$, yielding:

$$L \propto 1/m \quad (\alpha_m = 1)$$

When feature frequencies are more skewed (large $\alpha$), the ETF-like feature contribution becomes negligible, leading to $\alpha_m \approx 2(\alpha - 1)$.

### Loss & Training

AdamW optimizer with warmup and cosine decay learning rate schedule. New data is sampled at each step. $n = 1000$ is fixed; $m$ varies from 10 to 100.

## Key Experimental Results

### Main Results: Toy Model

| Regime | Data exponent $\alpha$ | Model exponent $\alpha_m$ | Power law? | Distribution-dependent? |
|--------|----------------------|--------------------------|-----------|------------------------|
| Weak superposition | $\alpha = 0.5$ | No power law | ✗ | ✓ |
| Weak superposition | $\alpha = 1.0$ | $\approx 0$ | Marginal | ✓ |
| Weak superposition | $\alpha = 2.0$ | $\approx 1.0$ | ✓ | ✓ |
| **Strong superposition** | **$\alpha = 0.5$** | **$\approx 1.0$** | **✓** | **✗** |
| **Strong superposition** | **$\alpha = 1.0$** | **$\approx 1.0$** | **✓** | **✗** |
| **Strong superposition** | **$\alpha = 2.0$** | **$\approx 1.3$** | **✓** | **✗** |

### Validation on Real LLMs

Analysis of four open-source model families (OPT, GPT-2, Qwen, Pythia):

| Observation | Result |
|-------------|--------|
| Mean squared inner product of row-normalized language model head $W$ | Approximately follows $1/m$ scaling |
| Loss vs. model dimension | $L = C_m/m^{\alpha_m} + L_{\backslash m}$, $\alpha_m = 0.91 \pm 0.04$ |
| Inferred from Chinchilla | $\alpha_m = (2.52 \pm 0.03) \times 0.35 = 0.88 \pm 0.06$ |
| Whether LLMs operate in superposition | ✓ Confirmed (supported by row-norm and interference distributions in the language model head) |

### Ablation Study

- **Activation density $E$**: Does not affect scaling behavior (verified in Appendix D.4)
- **Weight decay value $\gamma$**: Systematically controls the degree of superposition; small $\gamma$ → strong superposition, large $\gamma$ → weak superposition
- **Cross-entropy vs. squared-error loss**: Does not affect scaling behavior (demonstrated in Appendix A.2)
- **ETF vs. random vectors**: Representations of important features are closer to ETF (lower variance), but the mean is $1/m$ in both cases

### Key Findings

1. The strong-superposition regime yields **robust** $1/m$ scaling, independent of the specific form of the data frequency distribution
2. Scaling laws in the weak-superposition regime are sensitive to the data distribution—power-law frequencies are necessary to produce power-law scaling
3. Real LLMs operate in the strong-superposition regime; $\alpha_m \approx 1$ is consistent with theoretical predictions
4. Loss decomposes into a model-size-dependent term (representation loss) and a model-size-independent term (intrinsic data uncertainty)

## Highlights & Insights

1. **Unified explanation**: The origin of scaling laws is attributed to geometry—interference between representation vectors scales as $\sim 1/m$, yielding an elegant and intuitive account
2. **Robustness finding**: In the strong-superposition regime, the scaling exponent is approximately 1 regardless of data distribution details—explaining the universality of neural scaling laws
3. **New role of weight decay**: This work is the first to systematically demonstrate that weight decay controls the degree of superposition, with direct implications for practical training
4. **Verifiable prediction**: nGPT (which constrains hidden states to the unit hypersphere) encourages superposition and should therefore be more parameter-efficient—preliminary evidence supports this
5. **Theory–experiment loop**: A complete chain of reasoning is established, from precise analysis of the toy model to empirical validation on real LLMs

## Limitations & Future Work

1. **Lack of rigorous mathematical proof**: Analysis of the strong-superposition regime is primarily based on observation and heuristic reasoning; the model is not solved exactly
2. **Only representation loss is analyzed**: LLM loss also includes processing loss from Transformer layers $f_\ell(\ell)$, which is not studied independently
3. **Data/training-step scaling not analyzed**: Only model-width scaling is studied; data-size scaling is left for future work
4. **Gap between toy model and LLMs**: The toy model lacks Transformer layers, uses a different loss function, and simplifies data structure
5. **Causality not established**: $\alpha_m \approx 1$ in LLMs may have alternative explanations (e.g., depth–width balance)

## Related Work & Insights

- **Relation to Kaplan et al. (2020) Chinchilla scaling laws**: This paper provides a mechanistic explanation for those laws
- **Relation to Anthropic (2022) superposition model**: The framework is directly inherited, but the relationship between superposition and scaling is studied systematically for the first time
- **Relation to Michaud et al. (2023) quantization model**: Weak-superposition results are consistent with prior work; strong-superposition results are entirely new
- **Implication**: Encouraging superposition (e.g., nGPT, optimizers without weight decay) may be an effective means of improving LLM efficiency

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The insight that superposition is the core mechanism underlying scaling laws is entirely original
- **Theoretical Depth**: ⭐⭐⭐⭐ — The geometric argument is intuitive and compelling, though rigorous proofs are absent
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Toy-model experiments are comprehensive and LLM validation is solid, but intervention experiments are lacking
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Well-illustrated, clearly explained, and excellently structured
- **Value**: ⭐⭐⭐⭐ — Offers direct guidance for training strategies (weight decay, architecture choices)
- **Overall**: ⭐⭐⭐⭐⭐ (9/10) — Elegantly connects two major themes in AI research: mechanistic interpretability (superposition) and scaling laws

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)

<!-- RELATED:END -->

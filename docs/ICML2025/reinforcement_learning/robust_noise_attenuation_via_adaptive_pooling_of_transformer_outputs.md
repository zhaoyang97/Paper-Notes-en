---
title: >-
  [Paper Note] Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs
description: >-
  [ICML2025 Spotlight][Reinforcement Learning][Transformer pooling] This paper formalizes the pooling operations of Transformer outputs as a vector quantization problem, demonstrates that AvgPool and MaxPool suffer from performance collapse when the signal-to-noise ratio (SNR) varies, and proposes an adaptive pooling method based on cross-attention (AdaPool). AdaPool is theoretically shown to approximate the signal-optimal quantizer under any SNR and exhibits superior robustnes…
tags:
  - "ICML2025 Spotlight"
  - "Reinforcement Learning"
  - "Transformer pooling"
  - "noise robustness"
  - "adaptive pooling"
  - "vector quantization"
  - "attention mechanism"
  - "vision Transformer"
date: 2026-05-08
content_hash: 210c791141784dc7
---

# Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs

**Conference**: ICML2025 Spotlight  
**arXiv**: [2506.09215](https://arxiv.org/abs/2506.09215)  
**Code**: [agbrothers/pooling](https://github.com/agbrothers/pooling)  
**Area**: Transformer Robustness / Vector Pooling  
**Keywords**: Transformer pooling, noise robustness, adaptive pooling, vector quantization, attention mechanism, reinforcement learning, vision Transformer

## TL;DR

This paper formalizes the pooling operations of Transformer outputs as a vector quantization problem, demonstrates that AvgPool and MaxPool suffer from performance collapse when the signal-to-noise ratio (SNR) varies, and proposes an adaptive pooling method based on cross-attention (AdaPool). AdaPool is theoretically shown to approximate the signal-optimal quantizer under any SNR and exhibits superior robustness across RL, relational reasoning, and vision tasks.

## Background & Motivation

Transformer encoders generate output embeddings equal in number to the input tokens at each inference step. While each output has a clear target in sequence-to-sequence tasks, domains like computer vision or reinforcement learning (RL) require aggregating multiple output embeddings into a single representation for downstream tasks—a process known as **Global Pooling**.

Current mainstream methods include:

- **AvgPool**: Takes the average over all outputs.
- **MaxPool**: Takes the maximum along the feature dimension.
- **ClsToken**: Appends a learnable class token and uses its corresponding output.

However, these methods are often treated as arbitrary design choices and lack theoretical analysis. The core finding of this paper is that **when inputs are mixed with signal and noise vectors, AvgPool and MaxPool are each optimal at opposite ends of the SNR spectrum, while suffering catastrophic performance collapse at the other end**. This is particularly common in real-world RL environments, where an agent must extract task-relevant information from an abundance of sensor inputs, the majority of which are distractors.

## Method

### Problem Formulation

The input set $\mathbf{X} \in \mathbb{R}^{N \times d}$ contains $N$ $d$-dimensional vectors, of which $k$ vectors belong to the signal subset $\mathbf{X}_s$, and the remaining belong to the noise subset $\mathbf{X}_\eta$. The signal-to-noise ratio is defined as:

$$SNR = \frac{k}{N}$$

A vector $\mathbf{x}_i$ belongs to the signal subset if and only if the partial derivative of the learning objective $y$ with respect to it is non-zero: $\mathbf{x}_i \in \mathbf{X}_s \iff \frac{\partial y}{\partial \mathbf{x}_i} \neq 0$.

### Vector Quantization Perspective

Global vector pooling is defined as a degenerate vector quantizer (single cluster):

$$C(\mathbf{X}) = \sum_{i}^{N} \mathbf{w}_i \odot \mathbf{x}_i$$

**Signal Loss** is defined as the MSE between the compressed representation and the signal subset:

$$\mathbb{L}(\mathbf{X}, \mathbf{x}_c) = \frac{1}{k} \sum_{\mathbf{x}_s \in \mathbf{X}_s} (\mathbf{x}_s - \mathbf{x}_c)^2$$

The signal-optimal quantizer $C^*$ is the centroid of the signal subset, whose weights are: $w_i = 1/k$ (for signal vectors) or $w_i = 0$ (for noise vectors).

### Limitations of AvgPool and MaxPool

- **AvgPool** is signal-optimal only under no-noise scenarios ($\mathbf{X}_\eta = \emptyset$) or when signal and noise are identically distributed $\rightarrow$ adding each noise vector increases signal loss.
- **MaxPool** is signal-optimal only under a single signal vector that takes maximum values along each dimension $\rightarrow$ adding each signal vector increases signal loss.
- Their inductive biases are complementary, each being optimal at opposite ends of the SNR spectrum.

### AdaPool: Adaptive Pooling

AdaPool performs pooling using cross-attention with a single query:

$$\text{AdaPool}(\mathbf{X}) = \sum_{i}^{N} w_i \cdot \mathbf{x}_i W_V$$

Weights are given by the softmax over relation scores:

$$w_i = \frac{\exp(r_i)}{\sum_j^N \exp(r_j)}, \quad r_i = \frac{\mathbf{x}_q W_Q W_K^\top \mathbf{x}_i^\top}{\sqrt{d}}$$

**Key Property**: Both AvgPool and MaxPool are special cases of AdaPool.

### Error Bound Theorem (Theorem 3.12)

For any SNR, AdaPool can approximate the signal-optimal quantizer, with the error bound determined by the distribution of relation scores of signals and noises. Define the signal/noise neighborhood widths $\epsilon_s, \epsilon_\eta$ and the minimum margin $M$:

- Signal weight error bound: $L_s \leq w_i^* - w_i \leq U_s$
- Noise weight error bound: $L_\eta \leq w_i^* - w_i \leq U_\eta$

**Core Conclusion**: As the margin $M$ increases and neighborhoods $\epsilon_s, \epsilon_\eta$ shrink, the approximation error approaches zero.

### Query Selection Strategy

The paper suggests selecting the query $\mathbf{x}_q \in \mathbf{X}_s$ from the signal subset, as dot product measures similarity and a signal query yields higher dot products with other signal vectors. Specific selections include:

- **Entity-based RL**: The self-embedding of the controlled agent.
- **Memory Vector**: The current environment state.
- **Vision Tasks**: The center patch of the image (which usually contains the focal content).
- **Default Option**: The mean of all embeddings (Mean query), which offers robust performance.

## Key Experimental Results

### Synthetic Dataset KNN-Centroid Task

On synthetic data with $N=128, d=16$, AdaPool achieves signal loss that is an order of magnitude lower than other methods in the low SNR range (0.03–0.25).

### Multi-Agent RL (MPE)

| Scenario | AvgPool Drop | MaxPool Drop | ClsToken Drop | AdaPool Drop |
|------|-------------|-------------|--------------|-------------|
| Simple Tag + Noise | 77.4% | 60.7% | 70.4% | **50.9%** |

AdaPool achieves the highest final return and the minimum performance degradation across different noise levels.

### BoxWorld Relational Reasoning

- Entity-level observations (8 tokens): MaxPool is the most sample-efficient (exploiting the numerical properties of the white target gem).
- Pixel-level observations (50 tokens, high noise): AdaPool is optimal, while MaxPool experiences the most severe performance collapse.

### CIFAR Image Classification

| Method | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| ClsToken | 84.52±0.21 | 55.56±0.13 |
| AvgPool | 87.15±0.35 | 59.63±0.23 |
| MaxPool | 87.65±0.17 | 60.55±0.28 |
| **Ada-Focal** | **87.98±0.42** | 61.22±0.33 |
| **Ada-Mean** | 87.84±0.30 | **61.23±0.20** |
| Ada-Corner | 87.00±0.30 | 57.08±0.31 |

Focal and Mean queries perform the best, while the Corner query (edge patches) performs the worst, validating the importance of query selection.

## Highlights & Insights

1. **Solid Theoretical Contribution**: Formalizes pooling as vector quantization, provides a rigorous derivation of the failure conditions for AvgPool/MaxPool, and establishes the approximation error bound for AdaPool.
2. **Unified Perspective**: Proves that AvgPool, MaxPool, and ClsToken are all special cases of AdaPool, providing a unified analysis framework for pooling methods.
3. **Comprehensive Experiments**: Progresses from synthetic data $\rightarrow$ RL $\rightarrow$ relational reasoning $\rightarrow$ visual classification, systematically validating theoretical predictions.
4. **Practical Guidance**: The query selection strategy offers clear engineering guidance—the Mean query serves as a safe default.
5. **Connection to Associative Memory**: Establishes a link with Dense Associative Memories / Hopfield Networks, empowering attention pooling with superior interference resistance capacity.

## Limitations & Future Work

1. **Query Selection Relies on Domain Knowledge**: Although the Mean query is a reasonable default, optimal query selection still requires prior knowledge, making it less generalizable to scenarios with completely unknown signal distributions.
2. **Signal/Noise Binary Assumption**: In reality, vectors often partition into partial signal and partial noise; the binary classification framework is a simplification.
3. **Additional Computational Overhead**: AdaPool introduces a layer of cross-attention. Although the authors do not report this overhad, it may impact extremely low-latency scenarios.
4. **Evaluation Limited to Encoder Architectures**: Decoder-only or Encoder-Decoder architectures are not explored.
5. **Small Scale of ViT Experiments**: Only validated on $32 \times 32$ images from CIFAR-10/100, without testing on large-scale datasets like ImageNet.
6. **Multi-head Extension**: AdaPool uses a single query, and theoretical analysis is missing for multi-query extensions (similar to Perceiver).

## Rating

- Novelty: ⭐⭐⭐⭐ — The formal perspective of casting pooling as vector quantization is novel, and the derivation of error bounds is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers synthetic datasets, RL, reasoning, and vision, though large-scale vision experiments are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ — Demonstrates clear theoretical derivations and intuitive illustrations, with an excellent progression from theory to experiments.
- Value: ⭐⭐⭐⭐ — Provides a practical theoretical guidance tool for Transformer pooling design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring](../../ICLR2026/reinforcement_learning/beyond_noisy-tvs_noise-robust_exploration_via_learning_progress_monitoring.md)
- [\[ICML 2025\] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)
- [\[ICLR 2026\] RAMPS: Robust Adaptive Multi-step Predictive Shield](../../ICLR2026/reinforcement_learning/robust_adaptive_multi-step_predictive_shielding.md)
- [\[ICML 2025\] Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer](mastering_massive_multi-task_reinforcement_learning_via_mixture-of-expert_decisi.md)
- [\[ICLR 2026\] Recurrent Action Transformer with Memory](../../ICLR2026/reinforcement_learning/recurrent_action_transformer_with_memory.md)

</div>

<!-- RELATED:END -->

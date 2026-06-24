---
title: >-
  [Paper Note] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models
description: >-
  [Model Compression][Quantization] The OSP (Outlier-Safe Pre-Training) framework proactively prevents outlier formation during the pre-training phase through three key innovations: the Muon optimizer (eliminating privileged basis directions), Single-Scale RMSNorm (preventing channel magnification), and a learnable embedding projection layer (redistributing embedding layer activations). A 1.4B model trained on 1T tokens achieves near-zero excess kurtosis (0.04 vs. 1818.56 in st…
tags:
  - "Model Compression"
  - "Quantization"
  - "Outlier Elimination"
  - "Pre-training"
  - "Muon Optimizer"
  - "Low-Bit Inference"
date: 2026-05-08
content_hash: 2a0e8220c25b8036
---

# Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models

| Conference | Area | arXiv | Code |
|------------|------|-------|------|
| ACL2025 | Model Compression / Quantization | [2506.19697](https://arxiv.org/abs/2506.19697) | [GitHub](https://github.com/dmis-lab/Outlier-Safe-Pre-Training) |

**Keywords**: Quantization, Outlier Elimination, Pre-training, Muon Optimizer, Low-Bit Inference

## TL;DR

The OSP (Outlier-Safe Pre-Training) framework proactively prevents outlier formation during the pre-training phase through three key innovations: the Muon optimizer (eliminating privileged basis directions), Single-Scale RMSNorm (preventing channel magnification), and a learnable embedding projection layer (redistributing embedding layer activations). A 1.4B model trained on 1T tokens achieves near-zero excess kurtosis (0.04 vs. 1818.56 in standard models) and scores an average of 35.7 (compared to 26.5 for Adam) under aggressive 4-bit quantization, with only 2% training overhead.

## Background & Motivation

### Fundamental Obstacle in Quantization: Activation Outliers

LLMs inevitably generate activation outliers during standard training processes. These outliers catastrophically inflate the quantization scaling factor $s$, leading to severe rounding errors and information loss. For $n$-bit quantization:

$$\hat{x}_\text{int} = \text{clip}\left(\left\lfloor \frac{x}{s} \right\rceil + z, 0, 2^n - 1\right)$$

Outliers make $s$ excessively large, which severely compromises the quantization precision of normal values.

### Limitations of Prior Work

- **PTQ (Post-Training Quantization)**: Methods like GPTQ, QuaRot, and SpinQuant act as "remedies" rather than "prevention," treating outliers as an inherent property of LLMs.
- **QAT (Quantization-Aware Training)**: Simulates quantization errors during training but does not eliminate the outliers themselves, and significantly slows down training.
- **Small Scale of Prior Studies**: Most are limited to < 1B parameters or < 100B tokens, without considering production-level scalability.

### Three Hypotheses on the Causes of Outliers

**Channel Scaling Factors in Normalization Layers** (Kovaleva et al., 2021; Wei et al., 2022)

**Attention Sinks** (Bondarenko et al., 2023)

**Adaptive Gradient Scaling of Diagonal Optimizers (Adam/AdaFactor)** (He et al., 2024; Guo et al., 2024)

## Method

### Overall Architecture

The three components of OSP work synergistically to eliminate outliers at the source while maintaining training efficiency and architectural compatibility.

### Component 1: Muon Optimizer

**Principle**: Diagonal optimizers like Adam maintain parameter-wise gradient variance statistics and perform element-wise normalization. This introduces a "privileged basis" where certain channels disproportionately accumulate magnitude.

Muon utilizes the Newton-Schulz algorithm to approximate the orthogonalization of the gradient matrix:

$$G = U\Sigma V^T \mapsto UV^T$$

Key Advantages:
- Maintains only gradient momentum, avoiding element-wise scaling
- Updates parameters through full-rank linear transformations, eliminating the privileged basis
- Achieves **97.9%** of Adam's training throughput while reducing memory by 33%

| Optimizer | Relative Throughput | Memory | Compilation Time |
|-----------|--------------------|--------|------------------|
| Adam | 100% | $O(36LD^2)$ | 2m30s |
| Muon | 97.9% | $O(24LD^2)$ | 3m48s |
| Shampoo | 75.5% | $O(\frac{338}{3}LD^2)$ | 24m24s |

This is the first work to validate the scalability of Muon at a scale of billions of parameters and trillions of tokens.

### Component 2: Single-Scale RMSNorm (SSNorm)

**Problem**: The learnable channel-wise scaling parameters $\gamma_i$ of RMSNorm build an explicit basis alignment, which can lead to the amplification of certain channels.

Simple RMSNorm (removing all learnable parameters and dividing by $\sqrt{d}$) prevents outliers but introduces two issues:
- Early Training: Activation magnitudes are heavily suppressed, leading to slow convergence
- Late Training: Fixed scaling leads to instability

**SSNorm Solution**: Retain a single global scalar scaling parameter $\gamma \in \mathbb{R}$:

$$\text{SSNorm}(x) = \gamma \cdot \frac{x}{\|x\|_2}$$

All dimensions share a single $\gamma$, preserving dynamic scaling capabilities while fundamentally preventing the emergence of privileged coordinates.

### Component 3: Decoupled Embedding Optimization + Learnable Embedding Projection (EmbProj)

**Problem**: The extremely high dimensions of word embedding matrices pose a computational bottleneck for Muon (causing an additional 6% drop in throughput). Moreover, Jordan et al. (2024) demonstrated that decoupled embeddings trained with Adam achieve better convergence. However, this reintroduces outliers.

**Solution**:
- The embedding layer is still trained using Adam (preserving efficiency and convergence)
- A full-rank projection matrix is added after the embedding layer and before the de-embedding layer
- The projection matrix redistributes outliers across different dimensions, preventing their concentration and propagation
- Post-training, the projection matrices can be fused into adjacent embeddings, maintaining identical computation during inference

### Outlier Quantization Metrics

Excess Kurtosis is used to quantify the severity of outliers:

$$\text{Kurt}[X] - 3 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] - 3$$

## Experiments

### Ablation Study (100B tokens)

| Optimizer | SSNorm | EmbProj | Ex. Kurt | 4-4-4 Avg |
|-----------|--------|---------|----------|-----------|
| Adam | ✗ | ✗ | 1818.56 | 26.8 |
| Muon | ✗ | ✗ | 1575.12 | 29.0 |
| Muon | ✓ | ✗ | 66.69 | 36.4 |
| Muon | ✗ | ✓ | 703.23 | 30.4 |
| **Muon (OSP)** | **✓** | **✓** | **0.04** | **38.9** |

Key Findings:
- The three components must be **simultaneously enabled** to maintain kurtosis at a near-zero level
- No single component alone is sufficient to eliminate outliers completely—once formed somewhere in the network, they propagate throughout the entire architecture

### Scaling to 1T Tokens

Comparing 4-bit quantization performance against 12 open-source LLMs of similar scale:

| Model | Params | Tokens | 4-bit Avg |
|-------|--------|--------|-----------|
| Pythia | 1.4B | 0.3T | 26.5 |
| TinyLlama | 1.1B | 2T | 26.4 |
| OLMo | 1.2B | 3T | 27.6 |
| Qwen 2 | 1.5B | 7T | 29.3 |
| LLaMA 3.2 | 1.2B | – | 28.1 |
| Adam (from scratch) | 1.4B | 1T | 26.5 |
| **Muon (OSP)** | **1.4B** | **1T** | **35.7** |

Under 4-bit quantization, most models drop to near-random performance (~25%) on ARC and CSQA, whereas OSP significantly preserves performance.

### Complementarity with PTQ Methods

| Quantization Method | Adam PPL | OSP PPL |
|---------------------|----------|---------|
| RTN | 14475.51 | 45.92 |
| + GPTQ | 3723.46 | 14.29 |
| + QuaRot | 16.62 | **14.38** |
| + SpinQuant | 14.94 | **13.66** |

OSP is complementary to all PTQ methods, leading to further improvements when combined. This is because OSP eliminates the fundamental bottleneck of outliers, providing a much higher-quality starting point for PTQ calibration.

### Attention Sink Analysis

**Key Discovery**: After eliminating outliers, **attention sinks still persist**, but through a different mechanism:
- Adam models: Achieve 'no-op' operations by driving attention logits toward negative infinity, spawning massive activations.
- OSP models: Achieve the same functionality by concentrating positive attention on specific tokens without requiring extreme activations.

This challenges the hypothesis that "attention sinks cause outliers"—attention sinks themselves are not the cause of outliers; rather, models prone to outliers adopt extreme logit strategies.

## Highlights & Insights

1. **Paradigm Shift**: The conclusion that "outliers are not an inherent property of LLMs, but a consequence of training strategies" offers a paradigm-shifting perspective.
2. **Highly Practical**: Introduces only a 2% training overhead, retains standard Transformer architectures, and remains fully compatible with existing inference pipelines.
3. **First Production-Grade Outlier-Free LLM**: Evaluated at 1.4B parameters and 1T tokens, going beyond prior small-scale experiments.
4. **Orthogonal to PTQ**: Does not replace PTQ; instead, it provides a much stronger foundation for quantization.
5. **New Understanding of Attention Sinks**: Demonstrates that attention sinks and activation outliers are separable phenomena.

## Limitations & Future Work

1. **Validated Only up to 1.4B Scale**: Has not been expanded to 3B or 7B scales, which are common targets for on-device deployment.
2. **Lack of Extensive Comparison with Other Second-Order Methods**: Long TPU compilation times limited the ablation studies.
3. **Generality of SSNorm**: Whether single-scalar scaling is effective across all architectures and scales remains to be validated.
4. **Embedding Layer Still Uses Adam**: This is a compromise that might introduce issues in scenarios with extremely large vocabularies.

## Related Work

- **Quantization Methods**: RTN, GPTQ (Hessian-optimized weight rounding), SmoothQuant/AWQ (activation migration), QuaRot/SpinQuant (rotation matrix redistribution)
- **Outlier Sources/Causes**: Privileged basis theory by Elhage et al. (2023), attention sink hypothesis by Bondarenko et al. (2023)
- **Optimizers**: K-FAC (Fisher information approximated Hessian), Shampoo/SOAP (tensor-dimension decoupled preconditioners), Muon (Newton-Schulz orthogonalization, validated at trillion-token scale for the first time in this work)
- **Normalization Layers**: Simple RMSNorm (Qin et al., 2023), normalization-free schemes by He et al. (2024)

## Rating

⭐⭐⭐⭐⭐ (5/5)

This is a highly influential work: it fundamentally proves that activation outliers in LLMs are preventable rather than inevitable, and provides a practical, production-ready solution. The methodology is elegantly designed (with three components each addressing a distinct source of outliers), the experimentation is exceptionally solid (spanning comprehensive validation from 100B to 1T tokens), and the analysis is profound (re-evaluating the relationship between attention sinks and outliers). It holds major significance for the efficient deployment of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [\[ACL 2025\] PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](ptq161_low_bit_quantization.md)
- [\[ACL 2025\] Accurate KV Cache Quantization with Outlier Tokens Tracing](accurate_kv_cache_quantization_with_outlier_tokens_tracing.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)

</div>

<!-- RELATED:END -->

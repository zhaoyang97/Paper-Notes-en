---
title: >-
  [Paper Note] Neural Entropy
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Models] This paper explores the connection between deep learning and information theory through the lens of diffusion models, introducing a "neural entropy" measure to quantify the amount of information stored in neural networks during the diffusion process, revealing that image diffusion models achieve remarkably high compression efficiency on structured data.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion Models
  - Information Theory
  - Entropy
  - Data Compression
  - Neural Networks
date: 2026-05-08
content_hash: 0edb1570520bde5f
---

# Neural Entropy

**Conference**: NeurIPS 2025
**arXiv**: [2409.03817](https://arxiv.org/abs/2409.03817)
**Code**: None
**Area**: Generative Models / Information Theory
**Keywords**: Diffusion Models, Information Theory, Entropy, Data Compression, Neural Networks

## TL;DR

This paper explores the connection between deep learning and information theory through the lens of diffusion models, introducing a "neural entropy" measure to quantify the amount of information stored in neural networks during the diffusion process, revealing that image diffusion models achieve remarkably high compression efficiency on structured data.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Diffusion models work by converting noise into structured data, with the core process being the "recovery" of information erased when data is diffused into noise. This information is stored in the neural network's parameters during training. However, a systematic theoretical framework for quantifying the amount of stored information has been lacking.

Key questions:
1. How much information exists in diffusion models, and how can it be measured?
2. How efficient is a neural network as an information storage medium?
3. How does the diffusion process itself — not merely the data distribution — affect the amount of information?

These questions carry not only theoretical significance but also practical implications for understanding the compression capacity, generalization ability, and training optimization of generative models.

## Method

### Overall Architecture

The authors establish a rigorous correspondence between diffusion models and information theory:

1. **Forward diffusion**: Data → Noise, with information progressively erased
2. **Information storage**: During training, erased information is transferred into the neural network parameters
3. **Reverse generation**: Noise → Data, with the neural network releasing stored information to reconstruct structure

### Key Designs

**Definition of Neural Entropy**:
- Neural entropy is associated with the total entropy produced by the diffusion process
- It is a function not only of the data distribution but also of the diffusion process itself
- Mathematically, neural entropy $S_\text{neural}$ can be computed via the score function of the diffusion process:

$$S_\text{neural} = \int_0^T \mathbb{E}_{x_t} \left[ \| \nabla_{x_t} \log p_t(x_t) \|^2 \right] dt$$

where $T$ is the diffusion time and $p_t$ is the marginal distribution at time $t$.

**Relationship to Classical Entropy**:
- Neural entropy provides a finer-grained information measure than Shannon entropy
- It captures structured information in data rather than mere statistical randomness
- In limiting cases, neural entropy reduces to classical information-theoretic quantities

**Dependence on the Diffusion Process**:
- Different noise schedules give rise to different neural entropies
- This implies that the "encoding" of information is influenced by the choice of diffusion process
- It offers an information-theoretic perspective for hyperparameter selection in diffusion models

### Loss & Training

- Standard diffusion model training via denoising score matching
- Neural entropy is indirectly measured through the convergence behavior of the training loss
- Neural entropy variations are analyzed across different datasets and diffusion configurations

## Key Experimental Results

### Neural Entropy Measurements

Neural entropy measurements across different image datasets:

| Dataset | # Images | Resolution | Neural Entropy (nats/image) | Neural Entropy (nats/pixel) | Compression Ratio |
|--------|---------|--------|---------------------|-------------------------|--------|
| MNIST | 60,000 | 28×28 | 142.3 | 0.182 | 43.8× |
| CIFAR-10 | 50,000 | 32×32 | 1,847.5 | 0.601 | 13.3× |
| CelebA | 202,599 | 64×64 | 5,234.8 | 1.278 | 6.25× |
| LSUN-Bedroom | 3,033,042 | 256×256 | 28,471.2 | 0.434 | 18.4× |

### Effect of Noise Schedule on Neural Entropy

| Noise Schedule | CIFAR-10 Neural Entropy | Training Steps | FID |
|----------|----------------|---------|-----|
| Linear | 1,847.5 | 800K | 3.21 |
| Cosine | 1,692.1 | 800K | 2.94 |
| Sigmoid | 1,731.8 | 800K | 3.08 |
| VP-SDE | 1,804.3 | 800K | 3.15 |

### Ablation Study

| Analysis Dimension | Finding |
|---------|------|
| Dataset size vs. neural entropy | Approximately log-linear relationship: $S \propto \log N$ |
| Resolution vs. neural entropy | Sublinear growth: below $O(d)$ |
| Model capacity vs. neural entropy | A saturation point exists beyond which neural entropy no longer increases significantly |
| Data diversity vs. neural entropy | More data categories correspond to higher neural entropy |

### Key Findings

1. **Exceptionally high compression efficiency**: Diffusion models store far less information per image than the raw pixel data requires
2. **Distinctiveness of structured information**: Highly structured data (e.g., human faces) exhibits lower per-pixel neural entropy
3. **Effect of the diffusion process**: The cosine schedule outperforms the linear schedule in information utilization efficiency
4. **Scaling behavior**: As dataset size grows, the average per-sample neural entropy increases slowly, indicating that the model learns shared structure

## Highlights & Insights

- **Theoretical contribution**: This work establishes, for the first time, a rigorous measure of information content in diffusion models, bridging deep learning and statistical physics
- **Practical implications**: Neural entropy measurements can guide the design and optimization of diffusion models
- **Information compression perspective**: Reveals the essential nature of generative models as "information compressors"
- **Interdisciplinary value**: Connects three fields — information theory, statistical mechanics, and deep learning

## Limitations & Future Work

1. Experiments are conducted primarily on simple image datasets; validation on large-scale, high-resolution models is insufficient
2. Computing neural entropy requires a trained model, incurring high computational cost
3. Theoretical analysis is mainly applicable to continuous diffusion models; extension to discrete diffusion models remains to be explored
4. The quantitative relationship between neural entropy and model generalization is not discussed

## Related Work & Insights

- **Diffusion model theory** (Song et al., 2021; Kingma et al., 2021) → Provides the theoretical foundation of score matching and SDEs
- **Information bottleneck theory** (Tishby et al., 2000) → Measures information in neural networks from a different perspective
- **Minimum description length** (Rissanen, 1978) → Classical theory of data compression and model selection
- **Statistical physics and deep learning** → This work embodies a profound connection between these two fields

## Rating

- **Novelty**: ★★★★★ — First rigorous definition and measurement of information content in diffusion models
- **Theoretical Depth**: ★★★★★ — Establishes an elegant connection between information theory and diffusion models
- **Experimental Thoroughness**: ★★★☆☆ — Experimental scale is limited; primarily a proof of concept
- **Writing Quality**: ★★★★☆ — Theoretical exposition is clear, but requires substantial background in information theory

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] EVODiff: Entropy-aware Variance Optimized Diffusion Inference](evodiff_entropy-aware_variance_optimized_diffusion_inference.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](graph-based_neural_space_weather_forecasting.md)
- [\[NeurIPS 2025\] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery](score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)

<!-- RELATED:END -->

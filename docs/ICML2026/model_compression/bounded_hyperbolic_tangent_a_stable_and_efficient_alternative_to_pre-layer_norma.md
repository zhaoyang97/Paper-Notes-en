---
title: >-
  [Paper Note] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models
description: >-
  [ICML 2026][Model Compression][Normalization alternative] Ours proposes Bounded Hyperbolic Tanh (BHyT), a data-driven input-bounding $\tanh$ transformation…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Normalization alternative"
  - "Transformer stability"
  - "Training efficiency"
  - "Variance propagation"
  - "Depth scaling"
date: 2026-05-08
content_hash: b0e6c6c84245ac2c
---

# Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2601.09719](https://arxiv.org/abs/2601.09719)  
**Code**: https://github.com/MLAI-Yonsei/BHyT  
**Area**: LLM Efficiency  
**Keywords**: Normalization alternative, Transformer stability, Training efficiency, Variance propagation, Depth scaling

## TL;DR

Ours proposes Bounded Hyperbolic Tanh (BHyT), a data-driven input-bounding $\tanh$ transformation, as a plug-and-play alternative to Pre-Layer Normalization. It suppresses depth-wise activation growth while avoiding redundant variance calculations, achieving 1.6% faster training and 1.77% higher generation throughput than RMSNorm, with superior downstream performance.

## Background & Motivation

**Background**: Pre-Layer Normalization (Pre-LN) is the standard design for current LLMs, typically implemented with RMSNorm to stabilize deep network training by normalizing before self-attention and MLP sub-layers.

**Limitations of Prior Work**: Pre-LN faces two core issues. First, the **curse of depth**—the interaction between residual connections and Pre-LN causes hidden state magnitudes and variances to expand rapidly with layer count, causing deep blocks to degenerate into expensive identity maps. Second, **computational overhead**—each block redundantly calculates per-token statistics (mean/variance) at two Pre-LN sites, where cumulative reduction operations become latency bottlenecks for training and inference.

**Key Challenge**: Existing improvements only manage either stability or efficiency. Stability-oriented methods like Peri-LN perform normalization both before and after each sub-layer, effectively suppressing variance growth but doubling computational overhead. Efficiency-oriented normalization-free methods like Dynamic Tanh (DyT) replace normalization with $\tanh(\alpha x)$ using a learnable scalar $\alpha$, eliminating statistic calculations; however, global scalar scaling fails to control depth-wise growth of the residual stream, and large inputs easily enter the $\tanh$ saturation region, causing vanishing gradients.

**Goal**: Design a Pre-LN alternative that addresses both stability and efficiency—avoiding redundant statistic calculations like DyT while providing provable guarantees for depth-wise variance control.

**Key Insight**: The root of DyT's instability is its data-independent learnable global scalar, which cannot adaptively constrain the input to the non-saturated region of $\tanh$. If the input range can be bounded in a data-driven manner, stability can be achieved while maintaining the efficiency of bounded transformations.

**Core Idea**: Replace DyT's global scalar scaling with probabilistic input bounding based on Chebyshev's inequality. This ensures that inputs to the $\tanh$ function fall within a predefined non-saturated interval $[-\lambda, \lambda]$ with high probability, while redundant precise calculations are avoided through block-level variance approximation.

## Method

### Overall Architecture

BHyT serves as a plug-and-play replacement for the two Pre-LN normalization sites in a Transformer block. Input token embeddings $x$ undergo $\text{BHyT}_{\text{Attn}}$ before Attention and $\text{BHyT}_{\text{MLP}}$ before the MLP. The core process involves: (1) calculating the precise input variance $s_x^2$ once at the first site; (2) using a data-driven scaling factor to constrain the input to the $\tanh$ non-saturated region; (3) using a lightweight variance approximation at the second site instead of precise calculation to avoid extra reduction overhead.

### Key Designs

1.  **Probabilistic Input Bounding**:
    -   **Function**: Constrains the input before $\tanh$ to the non-saturated interval $[-\lambda, \lambda]$ with high probability to prevent gradient vanishing caused by deep activation expansion.
    -   **Mechanism**: Based on Chebyshev's inequality, for any distribution with finite variance, $|X - \mu| \leq \kappa s$ holds with probability $\geq 1 - \kappa^{-2}$. Accordingly, the scaling factor is designed as $\alpha = \lambda / (\kappa s_x + |\mu_x|)$, resulting in $\text{BHyT}^*(x) = \gamma \odot \tanh(\alpha x)$. In practice, an RMSNorm-style zero-mean approximation is used, simplifying to $\alpha = \lambda / (\kappa s_x)$. Defaults are set to $p=0.99$ (i.e., $\kappa=10$) and $\lambda=1$.
    -   **Design Motivation**: Unlike DyT’s learnable global scalar $\alpha_{\text{DyT}}$, BHyT’s scaling factor is data-driven—adjusting adaptively based on the statistical properties of the current input. This ensures saturation is avoided regardless of input distribution shifts, providing a core advantage in depth stability.

2.  **Block-Level Variance Approximation**:
    -   **Function**: Avoids redundant precise variance calculations at the second Pre-LN site of each block, reducing reduction operation overhead.
    -   **Mechanism**: After calculating precise $s_x^2$ at the first site, the variance of the input $x' = x + h_{\text{Attn}}$ at the second site is approximated as $\tilde{s}_{x'}^2 = s_x^2 + \tilde{s}_{h_{\text{Attn}}}^2$. The attention output variance is approximated using a model-level constant: $\tilde{s}_{h_{\text{Attn}}}^2 \approx \frac{1}{Td} \|W_V W_O\|_F^2 \cdot \lambda_{\text{Attn}}^2 / \kappa^2$. This term depends only on weight matrices and hyperparameters, allowing it to be cached during inference and updated periodically during training.
    -   **Design Motivation**: Precise variance calculation requires a full reduction operation (traversing the feature dimension), the primary overhead of normalization. This approximation reduces the second site's calculation from $O(d)$ reduction to a constant lookup, which can be executed in parallel with attention forward propagation.

3.  **Finite-Depth Variance Bound**:
    -   **Function**: Provides provable guarantees for depth-wise variance control.
    -   **Mechanism**: When hyperparameters satisfy $\lambda / \kappa < 1/\sqrt{L}$ (where $L$ is network depth), the output variance of BHyT at each layer is strictly smaller than that of LayerNorm Scaling (LNS). For instance, with $\lambda=1, \kappa=10$, this condition holds for any network with $L < 100$, covering most practical models.
    -   **Design Motivation**: Since RMSNorm and DyT lack theoretical guarantees for depthwise variance growth, BHyT is the first normalization alternative to combine efficiency advantages with provable stability bounds.

## Key Experimental Results

### Main Results (Pre-training)

| Model | Method | PT Eval PPL | Avg. 3-shot Acc. | Training Throughput |
| :--- | :--- | :--- | :--- | :--- |
| Llama-374M | RMSNorm | 24.92 | 40.03 | baseline |
| Llama-374M | Peri-LN | 25.02 | 39.71 | — |
| Llama-374M | DyT | 27.09 | 39.12 | — |
| Llama-374M | **Ours (BHyT)** | **24.71** | **40.31** | — |
| Llama-1B | RMSNorm | 17.75 | 41.94 | 81.7K tok/s |
| Llama-1B | Peri-LN | 17.89 | 42.31 | — |
| Llama-1B | LNS | 17.26 | 42.85 | — |
| Llama-1B | DyT | 18.07 | 42.71 | — |
| Llama-1B | **Ours (BHyT)** | **16.47** | **43.42** | **83.0K tok/s (+1.6%)** |

### Ablation Study (Variance Approximation Effect)

| Configuration | PT Eval PPL | Avg. Acc. | Training Steps/s | Note |
| :--- | :--- | :--- | :--- | :--- |
| RMSNorm (Precise Var) | 26.35 | 41.64 | 0.346 | Baseline |
| BHyT* (Precise Var) | 25.89 | **42.71** | 0.335 | Precise version, best performance but slower than RMSNorm |
| BHyT (Approx Var) | 25.87 | 42.53 | **0.381** | Approx version, almost no performance loss, fastest |

### Key Findings
- BHyT achieves the lowest PPL and highest average downstream accuracy across Llama-374M, 1B, and 3B scales, with the advantage widening as model size increases (BHyT is 2.8 percentage points higher than Peri-LN on Llama-3B).
- Regarding generation throughput, BHyT reaches 1199.9 tokens/s on Llama-1B, which is 1.6% faster than RMSNorm and 18.0% faster than Peri-LN. While DyT remains the fastest (1352.0 tokens/s), its depth stability is significantly inferior to BHyT.
- Variance approximation causes almost no performance loss (PPL difference of 0.02) but increases training speed from 0.335 to 0.381 steps/sec, even surpassing RMSNorm's 0.346.
- Layer-wise analysis shows that while activation magnitudes and variances grow significantly with depth in RMSNorm and DyT, BHyT maintains stability across the entire network depth.

## Highlights & Insights
- **Data-driven vs. Learnable Scaling**: BHyT demonstrates that in bounded transformations, replacing learnable global scalars with adaptive scaling based on input statistics yields inherently better depth stability without increasing parameter counts. This idea can generalize to any scenario using bounded activation functions.
- **"Free Lunch" of Variance Approximation**: By exploiting the property that attention output variance depends primarily on weight matrices, the $O(d)$ reduction operation is reduced to a constant lookup. This approximation can be performed in parallel with attention forward propagation. Experiments show high precision in high-dimensional spaces (Pearson correlation $r > 0.99$).
- **Practical Value of Theoretical Bounds**: The condition $\lambda/\kappa < 1/\sqrt{L}$ is very loose (default parameters satisfy it for $L < 100$), meaning BHyT's stability guarantees are broadly applicable to mainstream LLM architectures.

## Limitations & Future Work
- Experimental scale is capped at Llama-3B / 20B tokens; performance at 7B+ or hundreds of billions of tokens has not been verified, particularly regarding the robustness of hyperparameters $\lambda$ and $\kappa$.
- Variance approximation relies on the uniform attention weight assumption (Assumption 3.3), which may require modification for sparse attention or MoE architectures.
- DyT still outperforms BHyT in pure throughput (by ~12%); for shallow models not requiring extreme depth stability, DyT may remain the preferable choice.
- Only verified on decoder-only LLMs; applicability to encoder-decoder or Vision Transformers remains to be confirmed.

## Related Work & Insights
- **Peri-LN** (Kim et al., 2025): A stability scheme using normalization before and after every sub-layer. While stable, the double normalization is costly. BHyT proves that better results can be reached with a more lightweight approach.
- **Dynamic Tanh (DyT)** (Zhu et al., 2025): An efficiency scheme replacing normalization with $\tanh(\alpha x)$. BHyT builds on this by adding data-driven input bounding to solve depth stability issues.
- **LayerNorm Scaling (LNS)** (Sun et al., 2026): Controls variance growth via layer-index scaling. BHyT theoretically proves a tighter variance bound than LNS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models](the_shape_of_addition_geometric_structures_of_arithmetic_in_large_language_model.md)
- [\[ICML 2026\] GradPower: Powering Gradients for Faster Language Model Pre-Training](gradpower_powering_gradients_for_faster_language_model_pre-training.md)

</div>

<!-- RELATED:END -->

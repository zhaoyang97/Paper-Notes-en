---
title: >-
  [Paper Note] Continuous Visual Autoregressive Generation via Score Maximization
description: >-
  [ICML 2025][Image Generation][Continuous Autoregressive] Proposes a continuous visual autoregressive framework—based on the theory of strictly proper scoring rules, using energy score as a likelihood-free training objective to replace vector quantization for continuous token autoregressive image generation, where EAR-H achieves an FID of 1.97 and is approximately 10 times faster in inference than the diffusion-loss method MAR.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Continuous Autoregressive"
  - "Scoring Rules"
  - "Energy Score"
  - "Visual Generation"
  - "Quantization-free"
date: 2026-05-08
content_hash: 66c90175b6bfddd6
---

# Continuous Visual Autoregressive Generation via Score Maximization

**Conference**: ICML 2025  
**arXiv**: [2505.07812](https://arxiv.org/abs/2505.07812)  
**Code**: [GitHub](https://github.com/shaochenze/EAR)  
**Area**: Image Generation  
**Keywords**: Continuous Autoregressive, Scoring Rules, Energy Score, Visual Generation, Quantization-free

## TL;DR
Proposes a continuous visual autoregressive framework—based on the theory of strictly proper scoring rules, using energy score as a likelihood-free training objective to replace vector quantization for continuous token autoregressive image generation, where EAR-H achieves an FID of 1.97 and is approximately 10 times faster in inference than the diffusion-loss method MAR.

## Background & Motivation

**Background**: Visual autoregressive models (VAR) typically require vector quantization (VQ) to discretize continuous visual data into finite vocabulary tokens, followed by training with cross-entropy. The reconstruction FID of the VQ tokenizer is only 5.87, which becomes a bottleneck for generation quality.

**Limitations of Prior Work**: Explicitly predicting likelihood in continuous spaces is usually intractable—GIVT uses Gaussian mixture approximation, yet its expressiveness is limited by predefined distribution families; the diffusion loss in MAR requires multi-step denoising per token, leading to significant inference latency.

**Key Challenge**: Autoregressive generation of continuous tokens requires a training objective that guarantees fidelity without requiring explicit likelihood estimation.

**Goal**: Provide a unified theoretical framework to understand and design training objectives for continuous VAR.

**Key Insight**: Strictly proper scoring rules—a mathematical tool in statistics to evaluate the quality of probabilistic predictions, guaranteeing that the expected score is maximized if and only if the predictive distribution matches the true distribution.

**Core Idea**: Cross-entropy (discrete VAR) is a special case of the logarithmic score; for continuous spaces, the energy score can be used—which requires no likelihood estimation, only sampling from the model distribution.

## Method

### Overall Architecture
1. Use a continuous KL-16 tokenizer to encode images into a sequence of continuous tokens (stride 16)
2. Use a masked autoregressive Transformer to predict unknown tokens
3. Replace the softmax layer with an MLP generator in the output layer, injecting random noise to generate samples
4. Train using the energy score—simultaneously optimizing sample-target proximity and diversity among samples

### Key Designs

1. **Strictly Proper Scoring Rules Unified Framework**:
    - **Function**: Unify training objectives of discrete and continuous VAR into a single theoretical framework
    - **Mechanism**: The scoring rule $S(p,x): \mathcal{P}\times\mathcal{X}\mapsto\bar{\mathbb{R}}$ measures the fitness of the predictive distribution $p$ for an observation $x$; strictly proper means $S(p,q)\leq S(q,q)$ with equality holding if and only if $p=q$. Cross-entropy/GIVT corresponds to the logarithmic score, the diffusion loss corresponds to the Hyvärinen score, and EAR corresponds to the energy score
    - **Design Motivation**: A unified perspective reveals the essential differences between methods—the logarithmic score requires explicit likelihood (limited by parametric assumptions), while the Hyvärinen score requires multi-step denoising (slow inference)

2. **Energy Loss**:
    - **Function**: Train the probability prediction of continuous tokens in a likelihood-free manner
    - **Mechanism**: The energy score $S(p,y) = \mathbb{E}[|x_1-x_2|^\alpha] - 2\mathbb{E}[|x-y|^\alpha]$ ($\alpha\in(0,2)$), where the first term encourages diversity among generated samples, and the second term requires generated samples to be close to the target. An unbiased estimate requires only two independent samples $x_1,x_2\sim p$: $\mathcal{L}(p,y) = |x_1-y|^\alpha + |x_2-y|^\alpha - |x_1-x_2|^\alpha$
    - **Design Motivation**: The key advantage of the energy score is that it only requires sampling capability without requiring an explicit probability density, enabling the output distribution to be any implicit generative model

3. **MLP Generator (Substituting Softmax)**:
    - **Function**: Transform Transformer hidden representations into continuous token distributions (implicitly represented through the sampling process)
    - **Mechanism**: Similar to GAN-based implicit generation—taking random noise $\epsilon\sim U[-0.5,0.5]^{d_{\text{noise}}}$ as input, and gradually injecting noise perturbations into the prediction through residual blocks. The noise modulates hidden representations via adaptive layer normalization (shift/scale/gate)
    - **Design Motivation**: Not restricted by parametric assumptions like Gaussian mixtures, its expressiveness is only constrained by the capacity of the MLP

### Loss & Training
- Main loss: Energy loss with $\alpha=1$ (strictly proper and gradient-stable)
- Training temperature: Standard energy loss for the first 750 epochs, and $\tau_{\text{train}}=0.99$ for the last 50 epochs (reducing the weight of the diversity term to improve quality)
- Inference temperature: $\tau_{\text{infer}}=0.7$, scaling only the shift signal
- The MLP generator uses a 0.25x learning rate (for stable training)
- Classifier-Free Guidance: 10% probability of replacing conditions with a dummy token, with a linearly increasing guidance scale during inference
- Total training of 800 epochs, batch size 2048, AdamW optimizer

## Key Experimental Results

### Main Results (ImageNet 256×256 Conditional Generation)

| Model | Type | Params | FID↓ (w/ CFG) | IS↑ | Precision | Recall |
|------|------|--------|----------------|------|-----------|--------|
| DiT-XL/2 | Diffusion | 675M | 2.27 | 278.2 | 0.83 | 0.57 |
| VAR-d30 | Discrete AR | 2.0B | 1.92 | 323.1 | 0.82 | 0.59 |
| GIVT | Continuous AR | 304M | 3.35 | — | 0.84 | 0.53 |
| MAR | Continuous AR+Diffusion | 943M | 1.55 | 303.7 | 0.81 | 0.62 |
| **EAR-B** | **Continuous AR+Energy** | **205M** | **2.83** | **253.3** | **0.82** | **0.54** |
| **EAR-L** | **Continuous AR+Energy** | **474M** | **2.37** | **273.8** | **0.81** | **0.57** |
| **EAR-H** | **Continuous AR+Energy** | **937M** | **1.97** | **289.6** | **0.81** | **0.59** |

### Ablation Study: Influence of the energy score exponent $\alpha$ (EAR-B, 400 epochs, CFG=3.0)

| $\alpha$ | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 |
|----------|-----|------|-----|------|-----|
| FID↓ | **3.55** | 3.73 | 4.10 | 4.32 | 188.1 |
| IS↑ | **230.3** | 223.1 | 212.1 | 204.2 | 6.4 |

### Key Findings
- EAR-B achieves an FID of 2.83 with only 205M parameters, demonstrating extremely high parameter efficiency.
- Inference speed advantage is significant: EAR takes about 1 second to generate an image, while MAR takes about 10 seconds (Fig. 2 speed/quality trade-off).
- When $\alpha=2$, the energy score degenerates to only matching expected values (proper but not strictly proper), with FID collapsing to 188.1—validating the necessity of strict propriety.
- Training collapses for $\alpha<1$: the denominator $|x_1-x_2|^{2-\alpha}$ approaches zero, causing gradient explosion.
- Continuous tokenizer reconstruction FID of 1.22 vs VQ tokenizer 5.87—showing the inherent advantages of continuous tokens.
- Masked autoregressive (bidirectional attention) is far superior to causal (unidirectional) autoregressive, where the latter only achieves an FID of about 20.

## Highlights & Insights
- The unified scoring rules framework is extremely elegant—cross-entropy, GIVT, diffusion loss, and EAR are all special cases of choosing different strictly proper scores. This theoretical contribution goes beyond the specific method.
- The failure case of $\alpha=2$ precisely validates the theoretical prediction: proper but not strictly proper scoring rules cannot uniquely determine the optimal model, indicating that the mathematical condition of "strictness" is crucial in practice.
- The design of the MLP generator is ingenious—it projects noise into shift, scale, and gate signals via adaptive layer norm (borrowing from DiT), which is both flexible and controllable.

## Limitations & Future Work
- EAR-H (FID 1.97) still lags behind MAR (FID 1.55), indicating that the energy score might be inferior to the diffusion loss in absolute quality.
- Only validated on ImageNet 256×256, leaving higher resolutions and text-to-image scenarios unexplored.
- The energy loss requires two independent samples for estimation—sample efficiency might be limited in high-dimensional spaces.
- The relationship between the expression capacity limit of the MLP generator and the required number/width of residual blocks is not analyzed.
- The fine-tuning of the training temperature (changing $\tau$ during the last 50 epochs) is highly heuristic.

## Related Work & Insights
- **vs GIVT (Tschannen et al., 2023)**: Both are continuous VAR, but GIVT is limited by GMM parametrization, whereas EAR sidesteps this limitation through implicit generation.
- **vs MAR (Li et al., 2024)**: Both belong to the continuous VAR framework (Hyvärinen score vs Energy score); EAR sacrifices a small amount of FID in exchange for a 10x inference speedup.
- **vs VQ-based AR (VQGAN, LlamaGen, etc.)**: The energy score framework fundamentally eliminates the loss of quantization information.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified scoring rules framework is a theoretical breakthrough, connecting three major continuous VAR directions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison on standard ImageNet benchmarks, with ablations validating theoretical predictions.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical coherence across theory, methodology, and experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a unified theoretical foundation and a practical, efficient method for continuous autoregressive generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Randomized Autoregressive Visual Generation](../../ICCV2025/image_generation/randomized_autoregressive_visual_generation.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[ICML 2025\] Continuous Semi-Implicit Models](continuous_semi-implicit_models.md)
- [\[ICLR 2026\] NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale](../../ICLR2026/image_generation/nextstep-1_toward_autoregressive_image_generation_with_continuous_tokens_at_scal.md)
- [\[ICML 2025\] Visual Generation Without Guidance](visual_generation_without_guidance.md)

</div>

<!-- RELATED:END -->

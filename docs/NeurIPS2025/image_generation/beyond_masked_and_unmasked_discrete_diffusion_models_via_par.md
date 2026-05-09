---
title: >-
  [Paper Note] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking
description: >-
  [NeurIPS 2025][Image Generation][discrete diffusion models] Prime (Partial masking scheme) represents each token as a base-$b$ sub-token sequence and independently masks at the sub-token level, introducing intermediate states into masked diffusion models to enable fine-grained denoising. On OpenWebText, it achieves a perplexity of 15.36, becoming the first MDM to surpass ARM (17.54) without relying on an autoregressive formulation.
tags:
  - NeurIPS 2025
  - Image Generation
  - discrete diffusion models
  - masked diffusion
  - partial masking
  - sub-tokens
  - text generation
date: 2026-05-08
content_hash: e5d62ef9881386b5
---

# Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking

**Conference**: NeurIPS 2025
**arXiv**: [2505.18495](https://arxiv.org/abs/2505.18495)
**Code**: None
**Area**: Image Generation / Discrete Diffusion
**Keywords**: discrete diffusion models, masked diffusion, partial masking, sub-tokens, text generation

## TL;DR
Prime (Partial masking scheme) represents each token as a base-$b$ sub-token sequence and independently masks at the sub-token level, introducing intermediate states into masked diffusion models to enable fine-grained denoising. On OpenWebText, it achieves a perplexity of 15.36, becoming the first MDM to surpass ARM (17.54) without relying on an autoregressive formulation.

## Background & Motivation

**State of the Field** Masked diffusion models (MDMs) are powerful generative models for discrete data, generating samples by progressively unmasking tokens. Each token exists in one of only two states: masked or unmasked.

**Limitations of Prior Work** This binary representation leads to significant computational waste. During the reverse diffusion process, a large number of steps produce no change (idle steps), with the model repeatedly processing identical inputs. Experiments show that 37% of steps are ineffective.

**Root Cause** The binary state space of MDMs limits model utilization: a token is either fully masked (no information) or fully revealed (finalized), with no intermediate transitional states to enable gradual information release.

**Paper Goals** To redefine the diffusion process by converting idle steps into informative updates, thereby improving model utilization during generation.

**Starting Point** Each token is decomposed into a sub-token sequence via base-$b$ encoding, and masking is applied independently at the sub-token level, naturally inducing intermediate states.

**Core Idea** Extending the binary masked/unmasked representation to multi-level intermediate states via partial sub-token masking, such that a four-way prediction can be decomposed into multiple sequential binary decisions.

## Method

### Overall Architecture
MDM-Prime consists of three steps: (1) each token $x_0^i \in \mathcal{X}$ is mapped via an invertible function $f$ to a sub-token sequence $\mathbf{y}_0^i \in \mathcal{Y}^\ell$ of length $\ell$ using base-$b$ encoding, where $b = \lceil \sqrt[\ell]{C} \rceil$; (2) a masked diffusion forward process is applied independently at the sub-token level; (3) the reverse diffusion process progressively reveals sub-tokens, enabling fine-grained transitions from fully masked to intermediate states to fully revealed.

### Key Designs

1. **Partial Masking Scheme (Prime)**:
   - **Function**: Introduces intermediate states into discrete diffusion.
   - **Mechanism**: Token $x_0^i$ is encoded as a sub-token sequence $\mathbf{y}_0^i = f(x_0^i)$; independent masking of sub-tokens produces intermediate states. For example, a 4-class token encoded with 2 bits yields intermediate states such as "m0" or "1m", providing partial information. The number of intermediate states is $(b+1)^\ell - (C+1)$, which is always positive.
   - **Design Motivation**: Intermediate states allow the model to make more precise predictions based on partially revealed token information, reducing idle steps. Theoretical analysis proves that idle steps decrease monotonically as $\ell$ increases.

2. **Joint Probability Parameterization**:
   - **Function**: Models dependencies among sub-tokens and prevents the generation of invalid samples.
   - **Mechanism**: The joint distribution $p_\theta(\mathbf{y}_0^i|\mathbf{y}_t)$ is parameterized directly, assigning probability mass only to valid base-$b$ encodings ($\mathbf{y}_0^i \in f(\mathcal{X})$) and explicitly zeroing out logits outside $|\mathcal{V}(\mathbf{y}_t^i)|$. A carry-over constraint is also enforced, keeping already-revealed sub-tokens fixed.
   - **Design Motivation**: Independent parameterization via $\prod_j p_\theta(y_0^{i,j}|\mathbf{y}_t)$ not only introduces an erroneous independence assumption (causing distributional degeneracy) but may also generate invalid sub-token combinations (e.g., under the GPT-2 vocabulary of 50,257 tokens).

3. **Sub-Token Embedding Encoder**:
   - **Function**: Efficiently processes sub-token inputs.
   - **Mechanism**: A separate $D/\ell$-dimensional embedding lookup table is created for each sub-token position; the $\ell$ embeddings are concatenated to produce a $D$-dimensional token embedding. Each table has size $(b+1) \times D/\ell$, far smaller than a full lookup table over $|\tilde{\mathcal{Y}}^\ell|$.
   - **Design Motivation**: The sub-token space $\tilde{\mathcal{Y}}^\ell$ can be much larger than the original token space, making a direct lookup table infeasible. The concatenation strategy maintains compatibility with standard MDM architectures.

### Loss & Training
The variational upper bound loss is:
$$\mathcal{L}_{vb}(\mathbf{y}_0;\theta) = \int_0^1 \frac{\alpha'_t}{1-\alpha_t} \mathbb{E}_{q(\mathbf{y}_t|\mathbf{y}_0)}\!\left[\sum_i \log p_\theta(\mathbf{y}_0^i|\mathbf{y}_t)\right] dt,$$
which is a weighted cross-entropy loss and is theoretically guaranteed to be an upper bound on the negative log-likelihood.

## Key Experimental Results

### Main Results — Text Generation (OpenWebText Perplexity PPL)

| Method | PPL ↓ | Idle Step Ratio |
|--------|-------|-----------------|
| ARM (autoregressive)* | 17.54 | — |
| MDLM | ≤22.98 | 36.77% |
| EDLM-coAR* | ≤17.58 | — |
| MDLM-Prime ($\ell$=2) | ≤17.90 | 13.52% |
| MDLM-Prime ($\ell$=4) | ≤15.62 | 1.83% |
| **MDLM-Prime ($\ell$=6)** | **≤15.36** | **0.25%** |

### Main Results — Image Generation

| Method | CIFAR-10 FID ↓ | ImageNet-32 FID ↓ |
|--------|---------------|-------------------|
| Continuous diffusion SOTA | ~2.5–3.5 | ~6–8 |
| MDM-Prime | **3.26** | **6.98** |

### Ablation Study

| Configuration | OWT PPL | Notes |
|--------------|---------|-------|
| Independent parameterization | Degenerate | Sub-token independence distorts the distribution |
| Joint parameterization | 15.36 | Captures sub-token dependencies |
| Without carry-over | Higher | Carry-over is critical for zero-shot generalization |
| $\ell$=2→8 | 17.90→15.48 | Performance converges at $\ell \geq 4$ |

### Key Findings
- Prime is the first MDM to surpass ARM without relying on an autoregressive formulation (15.36 vs. 17.54).
- The idle step ratio is strongly correlated with PPL — reducing it from 36.77% (MDLM) to 0.25% (Prime $\ell$=6) lowers PPL from 22.98 to 15.36.
- Prime achieves performance comparable to continuous diffusion methods on image generation (CIFAR-10 FID 3.26).
- Performance converges at $\ell \geq 4$; $\ell = 4$ or $6$ is recommended in practice.

## Highlights & Insights
- The core idea of extending binary to multi-level intermediate states is intuitively simple yet remarkably effective — modifying only the embedding layer improves MDLM by 7 PPL points.
- The idle step analysis offers a new perspective for understanding performance bottlenecks in MDMs.
- The combination of joint parameterization and carry-over constraints ensures both theoretical correctness and efficient implementation.

## Limitations & Future Work
- Sub-token encoding increases sequence length to $L \times \ell$, raising the computational cost of the Transformer.
- Validation is currently limited to 130M-parameter models; performance at larger LLM scales remains to be confirmed.
- The base-$b$ encoding is manually designed; more optimal token decomposition strategies may exist.

## Related Work & Insights
- **vs. MDLM**: Prime is a direct enhancement of MDLM, modifying only the embedding layer with full architectural compatibility.
- **vs. SEDD**: SEDD uses absorbing states with score matching, while Prime uses partial masking with a variational upper bound — two complementary perspectives.
- **vs. BD3-LM**: BD3 strengthens MDMs by incorporating an autoregressive formulation, whereas Prime demonstrates that ARM can be surpassed without any AR component.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The partial masking idea is concise and powerful; being the first MDM to surpass ARM is a milestone result.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Cross-modal validation on text and images, seven zero-shot benchmarks, and thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous theoretical derivations and clear illustrations.
- **Value**: ⭐⭐⭐⭐⭐ A significant contribution to the field of discrete diffusion models.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)

<!-- RELATED:END -->

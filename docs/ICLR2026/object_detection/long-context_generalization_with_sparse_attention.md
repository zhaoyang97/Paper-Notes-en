---
title: >-
  [Paper Note] Long-Context Generalization with Sparse Attention
description: >-
  [ICLR 2026][Object Detection][α-entmax] ASEntmax (Adaptive-Scalable Entmax) is proposed, replacing softmax attention with $\alpha$-entmax using learnable temperature. The work theoretically and experimentally proves that sparse attention achieves $1000\times$ length extrapolation, resolving the attention dispersion problem of softmax under long contexts.
tags:
  - ICLR 2026
  - Object Detection
  - α-entmax
  - Transformer
date: 2026-05-08
content_hash: 5af89245bb30870c
---
# Long-Context Generalization with Sparse Attention

**Conference**: ICLR 2026  
**arXiv**: [2506.16640](https://arxiv.org/abs/2506.16640)  
**Code**: [deep-spin/asentmax](https://github.com/deep-spin/asentmax)  
**Area**: Object Detection  
**Keywords**: Sparse Attention, Long-context Generalization, $\alpha$-entmax, Length Extrapolation, Transformer

## TL;DR

ASEntmax (Adaptive-Scalable Entmax) is proposed, replacing softmax attention with $\alpha$-entmax using learnable temperature. The work theoretically and experimentally proves that sparse attention achieves $1000\times$ length extrapolation, resolving the attention dispersion problem of softmax under long contexts.

## Background & Motivation

**Softmax Attention Dispersion Problem**: As context length $n$ increases, softmax distributes probability mass across all tokens, causing the attention weights of relevant tokens to approach zero. Theoretically, as $n \to \infty$, the normalized entropy of softmax approaches 1 (a perfectly uniform distribution), defined as **complete dispersion**.

**Root Cause of Extrapolation Failure**: Attention patterns learned by models on short sequences cannot transfer to long sequences—the weight distribution of softmax in long sequences differs fundamentally from short sequences, leading to a collapse in retrieval and reasoning capabilities.

**Limitations of Prior Work**: Position encoding methods such as RoPE extrapolation and ALiBi only handle positional information and do not resolve the dispersion of the attention distribution itself. Scalable Softmax (SSMax) mitigates this via scaling factors but lacks theoretical guarantees.

**Theoretical Advantages of Sparse Attention**: Sparse transformations like $\alpha$-entmax can precisely zero out irrelevant tokens, naturally avoiding dispersion. However, rigorous theoretical analysis explaining why sparse attention facilitates length extrapolation has been missing.

**Absence of Three Major Theoretical Properties**: Formal proof is required to demonstrate that sparse attention outperforms softmax in: (1) non-vanishing attention; (2) concentration resilience; and (3) representational preservation.

**Requirement for Adaptive Sparsity**: Different attention heads across various layers may require different degrees of sparsity. A fixed $\alpha$ is too rigid, necessitating a learnable adaptive mechanism.

## Method

### Overall Architecture

ASEntmax maintains the overall Transformer architecture, merely replacing the softmax in each attention head with a sparse $\alpha$-entmax, equipped with a temperature that adapts to the context length. The paper proceeds in two steps: first, it theoretically establishes why sparse attention inherently resists length extrapolation failure—$\alpha$-entmax can precisely zero out weights of irrelevant tokens, from which three formal properties (non-vanishing, non-dispersive, and representational preservation) are derived. Second, to address the side effect where fixed sparsity becomes overly sharp on extremely long sequences, Adaptive-Scalable Entmax (ASEntmax) is introduced. This allows the inverse temperature of each head to vary learnably with sequence length $n$ in the form of $\delta + \beta(\log n)^\gamma$, smoothly interpolating between sparse (focusing on fixed patterns) and dense (approaching softmax) states. Consequently, when trained on short sequences and tested on sequences far exceeding the training length, the attention distribution pattern does not qualitatively change, enabling up to $1000\times$ length extrapolation.

### Key Designs

**1. $\alpha$-entmax Sparse Transformation: Precisely Zeroing Irrelevant Token Weights**

The fundamental issue with softmax is that its output is always a strictly positive dense distribution, where every token receives some probability, causing relevant token weights to be diluted by irrelevant ones in long sequences. $\alpha$-entmax is a continuous generalization of softmax; it reduces to softmax when $\alpha = 1$ and becomes sparsemax when $\alpha = 2$. When $\alpha > 1$, its output contains exact zeros, automatically excluding irrelevant tokens with scores below a threshold from the attention support set. This transformation remains differentiable and end-to-end trainable, blocking the leakage of probability mass to irrelevant tokens at the source by simply replacing the softmax operator.

**2. Three Theoretical Properties: Mathematically Proving Why Sparse Attention Extrapolates**

The core theoretical contribution is the formal proof that sparse attention facilitates length extrapolation, summarized as three properties. First, **Non-vanishing Attention**: When $\alpha > 1$, adding irrelevant tokens with scores below the threshold to a sequence leaves the weights of relevant tokens completely unchanged; in contrast, softmax reduces all existing weights proportionally regardless of relevance. Second, **Concentration Resilience (Non-dispersion)**: Using normalized entropy $H(z)/\log n$ to measure dispersion—softmax approaches 1 (complete dispersion to uniform distribution) as $n \to \infty$, while the entropy upper bound for $\alpha$-entmax is $O(\log s)$ (where $s$ is the support size, $s \ll n$), independent of $n$. This keeps normalized entropy below 1 as length increases. Thus, even if a sequence is scaled by $1000\times$, the attention concentration remains stable as long as the number of relevant tokens $s$ is constant. Third, **Representational Preservation**: In an $L$-layer network, the number of gradient paths for softmax is $O(n^L)$, leading to representation collapse and exacerbated over-squashing in deep layers due to path explosion. $\alpha$-entmax restricts this to $O(s^L)$, strengthening the gradient flow for long-range dependencies and maintaining discriminative power in long sequences.

**3. ASEntmax Adaptive-Scalable Temperature: Adjusting Sparsity with Length to Avoid Over-Sharpness**

Fixed $\alpha$ or fixed temperature poses a risk: when many tokens are truly relevant, fixed sparsity may "ignore" too much, making attention overly peaky on long sequences. ASEntmax formulates the scaling factor as a function of sequence length $n$, learnable for each head independently:

$$\text{ASEntmax}(z) = \alpha\text{-entmax}\big((\delta + \beta(\log n)^\gamma)\,z\big)$$

Here, $\delta, \beta, \gamma$ are learnable scalars (inverse temperature coefficients) for each head. $\gamma > 0$ allows the temperature to rise slowly with length, while $\gamma < 0$ causes it to decay, allowing the model to learn a schedule for how sparsity evolves. When $\beta = 0$, it reverts to standard $\alpha$-entmax, enabling a smooth transition between scaling and non-scaling. Using $\log n$ instead of $n$ avoids interference with position encodings. Experimental results show that different heads learn different scheduling coefficients, validating the necessity of this per-head, per-length adaptive mechanism.

### Loss & Training

Training follows the standard language modeling objective using cross-entropy loss for next-token prediction. The inverse temperature coefficients $\delta, \beta, \gamma$ for each head are optimized jointly with model parameters via backpropagation. $\alpha$ is typically fixed at an experimentally verified optimal value of 1.5 (making $\alpha$ learnable is possible but may introduce instability). The key evaluation setting involves training on short sequences (e.g., length 64) and testing directly on sequences far exceeding that length (e.g., 65K) to assess pure extrapolation capability.

## Key Experimental Results

### Main Results

Length extrapolation accuracy on the Associative Recall task (Training length 64):

| Method | 64 | 256 | 1K | 4K | 16K | 65K |
|------|-----|------|------|------|------|------|
| Softmax | 99.8% | 52.1% | 12.3% | 3.1% | 0.8% | 0.2% |
| SSMax | 99.7% | 89.4% | 71.2% | 45.6% | 28.3% | 15.1% |
| Adaptive Temp | 99.6% | 91.2% | 78.5% | 52.3% | 34.7% | 21.4% |
| **ASEntmax** | **99.9%** | **99.5%** | **99.1%** | **98.2%** | **96.8%** | **95.3%** |

### Ablation Study

Impact of $\alpha$ and temperature learnability (Associative Recall, test length 16K):

| Configuration | Accuracy | Description |
|------|--------|------|
| ASEntmax (α=1.5, θ learnable) | **96.8%** | Optimal configuration |
| α-entmax (α=1.5, fixed temp) | 88.4% | Lacks adaptive capability |
| α-entmax (α=2.0, fixed temp) | 82.1% | Over-sparsity leads to info loss |
| ASEntmax (α learnable, θ learnable) | 95.2% | Unstable α learning, slight drop |
| Softmax + Adaptive Temp | 34.7% | Temp cannot solve fundamental softmax dispersion |

### Key Findings

1. **1000× Extrapolation**: From training length 64 to test length 65K, ASEntmax maintains 95.3% accuracy, whereas softmax drops to 0.2%.
2. **Language Modeling Advantage**: In long-context LM evaluation, ASEntmax shows significantly better perplexity trends than softmax and SSMax at 8× the training length.
3. **Retrieval Capability Maintenance**: In "needle-in-a-haystack" tests far exceeding training length, ASEntmax maintains high retrieval success rates.
4. **Sparsity Adaptation**: Different layers and heads learn distinct temperature scheduling coefficients, confirming the necessity of the per-head, per-length adaptive mechanism.

## Highlights & Insights

- **Solid Theoretical Depth**: The formal proofs of the three properties (non-vanishing, concentration resilience, representational preservation) are the paper's greatest contributions, providing a rigorous mathematical foundation for the extrapolation advantages of sparse attention.
- **Introduction of the Dispersion Concept**: Long-context failure in softmax is unified under the concept of "dispersion" and quantified via normalized entropy, providing a clear and convincing conceptual framework.
- **Insight into $O(s^L)$ vs $O(n^L)$**: Revealed the intrinsic advantage of sparse attention in deep networks—the combinatorial explosion of gradient paths is effectively suppressed by sparsity.
- **Simple Implementation**: Replacing softmax with $\alpha$-entmax + learnable temperature is straightforward, requires no architectural modifications, and is engineering-friendly.

## Limitations & Future Work

1. **Computational Efficiency**: Forward/backward passes for $\alpha$-entmax involve sorting operations with $O(n \log n)$ complexity, higher than the $O(n)$ of softmax. While sparse outputs can accelerate subsequent steps, the attention computation itself is slower.
2. **Pre-training Cost**: Requires pre-training from scratch or full fine-tuning; it cannot be applied as a simple drop-in replacement for existing pre-trained models.
3. **Lack of Large-scale Validation**: Experiments were primarily conducted on medium-scale models and have not yet been validated on LLMs with 7B+ parameters.
4. **Compatibility with FlashAttention**: The irregular memory access patterns of sparse attention may conflict with hardware optimization methods like FlashAttention.
5. **Selection of α**: While experiments suggest 1.5 is optimal, there is a lack of theoretical guidance for determining the best $\alpha$.

## Related Work & Insights

- **Scalable Softmax (SSMax)**: Scales softmax logits via $\log n$ bias terms to mitigate dispersion but does not cure it—the theoretical analysis in this paper explains SSMax's limited effectiveness.
- **RoPE / ALiBi / YaRN**: Length extrapolation methods at the positional encoding level. These are orthogonal to ASEntmax and can be combined.
- **Entmax (Peters et al., 2019)**: The original work on $\alpha$-entmax, mainly used for NLP classification and translation. This paper is the first to link it to long-context extrapolation.
- **Sparse Transformer (Child et al., 2019)**: Structured sparse attention, which differs from the data-driven sparsity of $\alpha$-entmax.
- **Gated Attention / Linear Attention**: Alternative schemes to replace softmax, but they lack the theoretical guarantees of $\alpha$-entmax.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The formal proof of three theoretical properties is pioneering, establishing a rigorous mathematical link between sparse attention and length extrapolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Both synthetic tasks and language modeling are covered with impressive $1000\times$ extrapolation results, though large-scale model validation is missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, conceptual hierarchy is well-defined, and the dispersion definition and visualization are intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a theoretically grounded new direction for long-context LLMs, though engineering adoption requires addressing efficiency and compatibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhancing Vision Transformers for Object Detection via Context-Aware Token Selection and Packing](enhancing_vision_transformers_for_object_detection_via_context-aware_token_selec.md)
- [\[ICLR 2026\] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [\[CVPR 2026\] Bridge: Basis-Driven Causal Inference Marries VFMs for Domain Generalization](../../CVPR2026/object_detection/bridge_basis-driven_causal_inference_marries_vfms_for_domain_generalization.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](../../ICCV2025/object_detection/adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/object_detection/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)

</div>

<!-- RELATED:END -->

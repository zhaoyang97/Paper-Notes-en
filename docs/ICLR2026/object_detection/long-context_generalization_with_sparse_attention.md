---
title: >-
  [Paper Note] Long-Context Generalization with Sparse Attention
description: >-
  [ICLR 2026][Object Detection][Sparse Attention] This paper proposes ASEntmax (Adaptive-Scalable Entmax), which replaces softmax attention with α-entmax equipped with a learnable temperature. Through both theoretical anal…
tags:
  - "ICLR 2026"
  - "Object Detection"
  - "Sparse Attention"
  - "Long-Context Generalization"
  - "α-entmax"
  - "Length Extrapolation"
  - "Transformer"
date: 2026-05-08
content_hash: 4d32cd6652989c2e
---

# Long-Context Generalization with Sparse Attention

**Conference**: ICLR 2026
**arXiv**: [2506.16640](https://arxiv.org/abs/2506.16640)  
**Code**: [deep-spin/asentmax](https://github.com/deep-spin/asentmax)  
**Area**: Object Detection
**Keywords**: Sparse Attention, Long-Context Generalization, α-entmax, Length Extrapolation, Transformer

## TL;DR

This paper proposes ASEntmax (Adaptive-Scalable Entmax), which replaces softmax attention with α-entmax equipped with a learnable temperature. Through both theoretical analysis and empirical evaluation, it demonstrates that sparse attention enables up to 1000× length extrapolation, addressing the attention dispersion problem of softmax under long-context settings.

## Background & Motivation

**Attention Dispersion in Softmax**: As context length $n$ grows, softmax distributes probability mass across all tokens, causing the attention weights of relevant tokens to approach zero. Theoretically, as $n \to \infty$, the normalized entropy of softmax tends to 1 (uniform distribution), a phenomenon termed **complete dispersion**.

**Root Cause of Length Extrapolation Failure**: Attention patterns learned on short sequences fail to transfer to longer ones — the weight distribution of softmax differs drastically between short and long sequences, leading to the collapse of retrieval and reasoning capabilities.

**Limitations of Prior Work on Long-Context Modeling**: Positional encoding methods such as RoPE extrapolation and ALiBi address only positional information and do not resolve the intrinsic dispersion of attention distributions. Scalable Softmax (SSMax) mitigates dispersion via a scaling factor but lacks theoretical guarantees.

**Theoretical Advantages of Sparse Attention**: Sparse transformations such as α-entmax can set the attention weights of irrelevant tokens to exactly zero, naturally avoiding dispersion. However, a rigorous theoretical explanation for why sparse attention facilitates length extrapolation has been lacking.

**Absence of Three Key Theoretical Properties**: Formal proofs are needed to establish the superiority of sparse attention over softmax with respect to: (1) non-vanishing attention; (2) concentration resilience; and (3) representational preservation.

**Need for Adaptive Sparsity**: Different attention heads across different layers may require varying degrees of sparsity; a fixed $\alpha$ is overly rigid, motivating a learnable adaptive mechanism.

## Method

### Overall Architecture

ASEntmax replaces the softmax operation in standard Transformer attention with α-entmax using a learnable temperature $\theta$. Specifically, the attention weight computation changes from $\text{softmax}(QK^T/\sqrt{d})$ to $\alpha\text{-entmax}(QK^T/(\sqrt{d} \cdot \theta))$, where $\alpha > 1$ controls the degree of sparsity and $\theta$ is a temperature parameter learned independently for each attention head.

### Key Designs

**1. α-entmax Sparse Transformation**

- A generalization of softmax: reduces to softmax when $\alpha = 1$, and to sparsemax when $\alpha = 2$.
- Core property: produces exact zeros in the output, automatically zeroing out attention weights for irrelevant tokens.
- Differentiable and supports end-to-end training.

**2. Three Theoretical Properties**

- **Non-vanishing Attention**: For $\alpha > 1$, adding irrelevant tokens to the sequence does not reduce the attention weights of relevant tokens. Formally, if the score of a newly added token falls below a threshold, the attention weights of existing tokens remain entirely unchanged. In contrast, softmax always reduces the weights of all existing tokens regardless of the relevance of added tokens.
- **Concentration Resilience**: The entropy of α-entmax attention is upper-bounded by $O(\log s)$, where $s$ is the support size, rather than $O(\log n)$ as in softmax, where $n$ is the sequence length. This means that even when sequence length increases by 1000×, attention concentration remains stable as long as the number of relevant tokens $s$ stays constant.
- **Representational Preservation**: In an $L$-layer Transformer, the number of gradient paths under softmax is $O(n^L)$, causing representational collapse in deep networks; α-entmax reduces this to $O(s^L)$, effectively preserving the distinguishability of different inputs.

**3. Learnable Temperature θ (ASEntmax)**

- Each attention head learns an independent temperature parameter $\theta$.
- Large $\theta$ → greater sparsity; small $\theta$ → closer to dense attention.
- Allows the model to adaptively interpolate between sparse and dense attention, with different heads adopting different strategies.

**4. Non-dispersion Property**

- Complete dispersion of softmax: normalized entropy $H(\text{softmax}(z))/\log n \to 1$ as $n \to \infty$.
- α-entmax maintains concentration: normalized entropy is bounded and does not approach 1 as $n$ grows.
- This serves as the theoretical cornerstone of length extrapolation capability.

### Loss & Training

- Standard language modeling objective (next-token prediction with cross-entropy loss).
- Temperature $\theta$ is jointly optimized with model parameters via backpropagation.
- $\alpha$ is typically fixed at 1.5 (empirically validated as optimal) but can also be made learnable.
- Models are trained on short sequences (e.g., length 64) and directly evaluated on long sequences (e.g., 65K).

## Key Experimental Results

### Main Results

Length extrapolation accuracy on the Associative Recall task (training length 64):

| Method | 64 | 256 | 1K | 4K | 16K | 65K |
|---|---|---|---|---|---|---|
| Softmax | 99.8% | 52.1% | 12.3% | 3.1% | 0.8% | 0.2% |
| SSMax | 99.7% | 89.4% | 71.2% | 45.6% | 28.3% | 15.1% |
| Adaptive Temp | 99.6% | 91.2% | 78.5% | 52.3% | 34.7% | 21.4% |
| **ASEntmax** | **99.9%** | **99.5%** | **99.1%** | **98.2%** | **96.8%** | **95.3%** |

### Ablation Study

Effect of $\alpha$ and temperature learnability (Associative Recall, test length 16K):

| Configuration | Accuracy | Note |
|---|---|---|
| ASEntmax (α=1.5, learnable θ) | **96.8%** | Optimal configuration |
| α-entmax (α=1.5, fixed temp) | 88.4% | Lacks adaptive capacity |
| α-entmax (α=2.0, fixed temp) | 82.1% | Over-sparsity leads to information loss |
| ASEntmax (learnable α, learnable θ) | 95.2% | Unstable α learning causes slight degradation |
| Softmax + Adaptive Temp | 34.7% | Temperature cannot resolve the fundamental dispersion of softmax |

### Key Findings

1. **1000× Extrapolation**: Trained at length 64 and tested at length 65K, ASEntmax maintains 95.3% accuracy while softmax drops to 0.2%.
2. **Language Modeling Advantage**: In long-context LM evaluation, ASEntmax shows significantly better perplexity trends at 8× training length compared to softmax and SSMax.
3. **Retrieval Capability Preservation**: In needle-in-a-haystack tests far exceeding training length, ASEntmax maintains high retrieval success rates.
4. **Adaptive Sparsity**: Different layers and heads learn distinct temperature values, validating the necessity of the adaptive mechanism — lower layers tend toward denser attention while higher layers tend toward sparser attention.

## Highlights & Insights

- **Theoretical Rigor**: The formal proofs of three properties — non-vanishing attention, concentration resilience, and representational preservation — represent the paper's primary contribution, providing a rigorous mathematical foundation for the length extrapolation advantages of sparse attention.
- **The Dispersion Concept**: Attributing long-context failures of softmax to "dispersion" and quantifying it via normalized entropy is conceptually clear and compelling.
- **$O(s^L)$ vs. $O(n^L)$ Insight**: This reveals the fundamental advantage of sparse attention in deep networks — the combinatorial explosion of gradient paths is effectively suppressed by sparsity.
- **Simplicity of Implementation**: Only softmax is replaced with α-entmax plus a learnable temperature, requiring no additional architectural modifications, making engineering adoption straightforward.

## Limitations & Future Work

1. **Computational Efficiency**: The forward and backward passes of α-entmax involve sorting operations with complexity $O(n \log n)$, higher than the $O(n)$ of softmax; although sparse outputs can accelerate downstream computation, the attention computation itself is slower.
2. **Pretraining Cost**: The method requires training from scratch or full fine-tuning and cannot be straightforwardly applied as a drop-in replacement for existing pretrained models.
3. **Insufficient Large-Scale Validation**: Experiments are conducted primarily on medium-scale models; validation on models with 7B+ parameters has not been performed.
4. **Compatibility with FlashAttention**: The irregular memory access patterns of sparse attention may conflict with hardware-optimized methods such as FlashAttention.
5. **Selection of α**: Although experiments suggest 1.5 is preferable, theoretical guidance for determining the optimal $\alpha$ is lacking.

## Related Work & Insights

- **Scalable Softmax (SSMax)**: Mitigates dispersion by scaling softmax logits with a $\log n$ bias term but does not resolve it fundamentally — the theoretical analysis in this paper explains why SSMax has limited effectiveness.
- **RoPE / ALiBi / YaRN**: Length extrapolation methods at the positional encoding level; orthogonal to ASEntmax and can be combined with it.
- **Entmax (Peters et al., 2019)**: The original α-entmax work, primarily applied to NLP classification and translation tasks; this paper is the first to connect it to long-context extrapolation.
- **Sparse Transformer (Child et al., 2019)**: Structured sparse attention, distinct from the data-driven sparsity of α-entmax.
- **Gated Attention / Linear Attention**: Alternative approaches to replacing softmax, but without the theoretical guarantees of α-entmax.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The formal proofs of three theoretical properties are pioneering, establishing a rigorous mathematical connection between sparse attention and length extrapolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Coverage of both synthetic tasks and language modeling is solid, and the 1000× extrapolation results are impressive, though large-scale model validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, conceptual hierarchy is well-structured, and the definition and visualization of dispersion are highly intuitive.
- **Value**: ⭐⭐⭐⭐ — Offers a theoretically grounded new direction for long-context LLMs, though engineering deployment still requires addressing efficiency and compatibility challenges.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [\[ICML 2026\] FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization](../../ICML2026/object_detection/focus_forcing_in-context_object_localization_through_visual_support_constraints_.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/object_detection/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](../../ICCV2025/object_detection/adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[CVPR 2026\] AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos](../../CVPR2026/object_detection/ar2-4fv_anchored_referring_and_re-identification_for_long-term_grounding_in_fixe.md)

</div>

<!-- RELATED:END -->

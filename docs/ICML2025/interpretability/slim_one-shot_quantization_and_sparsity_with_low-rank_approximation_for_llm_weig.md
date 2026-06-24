---
title: >-
  [Paper Note] SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression
description: >-
  [ICML 2025][Interpretability][one-shot compression] Proposes SLiM, a one-shot compression framework that seamlessly integrates hardware-friendly uniform quantization, semi-structured sparsity, and saliency-based low-rank adapters, achieving up to 5.66% accuracy improvement under 4-bit + 2:4 sparsity conditions.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "one-shot compression"
  - "quantization"
  - "sparsity"
  - "low-rank adapter"
  - "SLiM-Quant"
  - "SLiM-LoRA"
date: 2026-05-08
content_hash: 73e2bd007e2f3a86
---

# SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

**Conference**: ICML 2025  
**arXiv**: [2410.09615](https://arxiv.org/abs/2410.09615)  
**Code**: [GitHub](https://github.com/Mohammad-Mozaffari/slim)  
**Area**: Interpretability  
**Keywords**: one-shot compression, quantization, sparsity, low-rank adapter, SLiM-Quant, SLiM-LoRA  

## TL;DR

Proposes SLiM, a one-shot compression framework that seamlessly integrates hardware-friendly uniform quantization, semi-structured sparsity, and saliency-based low-rank adapters, achieving up to 5.66% accuracy improvement under 4-bit + 2:4 sparsity conditions.

## Background & Motivation

- While pruning or quantization alone can effectively reduce inference costs, their **joint application leads to error accumulation**, resulting in severe performance degradation.
- Existing one-shot compression methods struggle to recover the accuracy of dense models under 4-bit + structured sparsity scenarios.
- Low-rank adapters can mitigate compression losses, but typically require **expensive retraining**.
- Existing low-rank methods (such as L2QER) are designed solely for quantization and perform poorly when combined with sparsity.

## Method

### Three-Step Pipeline

#### Step 1: SLiM-Quant (Probabilistic Uniform Quantization)

Reformulates the quantization problem from non-convex optimization into a probabilistic framework:

$$\alpha^* = \arg\min_\alpha \intLock_{-\infty}^{\infty} f(x)|Q^{-1}(Q(x)) - x|^2 dx$$

Decomposed into quantization error + clipping error:

$$E_Q(\alpha) = E_{quant}(\alpha) + E_{clip}(\alpha)$$

Since actual weight distributions do not match any standard distribution (Gaussian, Laplace, etc., are all ruled out), a **numerical integration + multi-grid strategy** is adopted to solve for the optimal $\alpha$: first performing a coarse search over 10 uniformly sampled points, and then refining within the optimal region.

**Activation-Aware Extension (SLiM-Quant^O)**: Defines channel saliency as $|diag(\mathbf{x}) \times \mathcal{W}|$, amplifying weights and scaling down activations for approximately the top 1% most salient channels.

#### Step 2: Sparsification

Applies Wanda on the quantized weights to perform semi-structured (2:4) or unstructured sparsity.

#### Step 3: SLiM-LoRA (Saliency-Based Low-Rank Adapter)

**Core Innovation**: Designs a saliency function $F(\mathcal{W}) = diag(\mathbf{x})\mathcal{W}$ that satisfies **invertibility** and **additivity**:

$$F(A+B) = F(A) + F(B) \quad \text{(additivity)}$$

Leveraging these two properties, the adapter values are solved directly via SVD without iterative optimization:

$$diag(\mathbf{x})\mathcal{L}, \mathcal{R} = -SVD(diag(\mathbf{x})(E_Q + E_S))$$

where $E_Q$ and $E_S$ represent the quantization and sparsity errors, respectively.

### Adapter Quantization

Also applies 4-bit quantization to the low-rank adapters (AbsMax group quantization with a group size of 128), reducing overhead by 4×.

### Optional Post-Compression Fine-Tuning

Freezes the sparse-quantized weights and only fine-tunes the low-rank adapters, utilizing STE (Straight-Through Estimator) to achieve quantization-aware fine-tuning.

## Key Experimental Results

### Main Results: Zero-Shot Accuracy (4-bit + 2:4 Sparsity)

| Method | LLaMA-2-7B | LLaMA-2-13B | OPT-13B |
|------|-----------|-------------|---------|
| Dense | 56.6 | 60.8 | 48.7 |
| Wanda + Group AbsMax | 40.6 | 49.6 | 37.7 |
| SparseGPT + OPTQ | 42.6 | 53.3 | 43.0 |
| JSQ | 44.3 | 53.7 | 42.0 |
| **SLiM** | **46.3** | **57.2** | **43.6** |
| **SLiM + PEFT** | **47.0** | **58.9** | - |

- Average improvement of **5.66%** on LLaMA-2-7B
- Average improvement of **3.89%** on LLaMA-2-13B
- SLiM even **outperforms the dense model** by 0.6% in certain configurations

### Hardware Acceleration

| GPU | Layer-wise Speedup |
|-----|----------|
| RTX 3060 | 4.3× |
| A100 | 3.8× |
| Memory Reduction | 0.23× |

### Ablation Study: Contribution of Each Component

| Configuration | PPL (WikiText2) |
|------|---------------|
| SLiM-Quant only | 6.89 |
| + Wanda | 8.12 |
| + SLiM-LoRA | **7.45** |
| + Adapter Quantization | 7.51 |
| + PEFT | 7.32 |

## Highlights & Insights

- The invertible and additive design of the saliency function is highly elegant, enabling closed-form solutions for the low-rank adapters.
- Uniform quantization + probabilistic optimization eliminates the extra overhead of group quantization.
- Seamless integration of the three compression technologies, with clear mathematical motivation for each step.
- End-to-end one-shot scheme, requiring no large-scale retraining.
- Maintains high accuracy even under extreme compression conditions (4-bit + 2:4 sparsity).

## Limitations & Future Work

- The numerical integration of SLiM-Quant relies on the weight histogram and may not be robust to outlier distributions.
- SLiM-LoRA has limited options for the saliency function (only $diag(\mathbf{x})\mathcal{W}$), which may not be optimal.
- Evaluated only on LLaMA-2 and OPT, lacking experiments on newer models such as LLaMA-3 and Mistral.
- The rank selection for the adapters lacks an adaptive mechanism.
- Although the optional PEFT step increases accuracy, it also adds to the complexity of the pipeline.

## Rating

⭐⭐⭐⭐ — Rigorous theoretical derivation and complete engineering implementation. The joint optimization of the three compression technologies achieves a new SOTA in one-shot compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Escaping Low-Rank Traps: Interpretable Visual Concept Learning via Implicit Vector Quantization](../../ICLR2026/interpretability/escaping_low-rank_traps_interpretable_visual_concept_learning_via_implicit_vecto.md)
- [\[ICLR 2026\] Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](../../ICLR2026/interpretability/towards_understanding_the_nature_of_attention_with_low-rank_sparse_decomposition.md)
- [\[ICLR 2026\] Sequences of Logits Reveal the Low Rank Structure of Language Models](../../ICLR2026/interpretability/sequences_of_logits_reveal_the_low_rank_structure_of_language_models.md)
- [\[ICLR 2026\] Learning is Forgetting: LLM Training As Lossy Compression](../../ICLR2026/interpretability/learning_is_forgetting_llm_training_as_lossy_compression.md)
- [\[ICML 2025\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)

</div>

<!-- RELATED:END -->

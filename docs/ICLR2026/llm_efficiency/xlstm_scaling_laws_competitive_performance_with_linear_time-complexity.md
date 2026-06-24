---
title: >-
  [Paper Note] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity
description: >-
  [ICLR2026][LLM Efficiency][scaling laws] This paper systematically compares the scaling laws of xLSTM and Transformer, demonstrating that xLSTM consistently outperforms Transformers of the same scale in terms of the training loss-compute Pareto frontier, overtraining regime, and inference speed, with the advantage increasing as context length grows.
tags:
  - "ICLR2026"
  - "LLM Efficiency"
  - "scaling laws"
  - "xLSTM"
  - "linear complexity"
  - "Transformer comparison"
  - "inference efficiency"
date: 2026-05-08
content_hash: a661abf5c98fe593
---

# xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity

**Conference**: ICLR2026  
**arXiv**: [2510.02228](https://arxiv.org/abs/2510.02228)  
**Code**: [NX-AI/xlstm_scaling_laws](https://github.com/NX-AI/xlstm_scaling_laws)  
**Area**: LLM Efficiency  
**Keywords**: scaling laws, xLSTM, linear complexity, Transformer comparison, inference efficiency  

## TL;DR
This paper systematically compares the scaling laws of xLSTM and Transformer, demonstrating that xLSTM consistently outperforms Transformers of the same scale in terms of the training loss-compute Pareto frontier, overtraining regime, and inference speed, with the advantage increasing as context length grows.

## Background & Motivation
- Scaling laws are core guiding tools for LLM design (Kaplan 2020, Chinchilla 2022), but existing research focuses almost entirely on the Transformer architecture.
- Linear complexity architectures like xLSTM have shown competitiveness at the billion-parameter scale (xLSTM 7B), but lack systematic scaling comparisons with Transformers.
- Traditional FLOP approximations $C(N,D)=6ND$ ignore the computational overhead of the attention mechanism, failing to provide a fair comparison between linear and quadratic complexity models.
- Systematic analysis of the interaction between inference efficiency (TTFT, step time) and context length is also lacking.

## Core Problem
1. **Training Efficiency**: Given a fixed compute budget, which architecture achieves lower loss, xLSTM or Transformer?
2. **Overtraining Regime**: Does xLSTM maintain a stable power-law exponent under high token-to-parameter ratios?
3. **Context Length**: How does linear vs. quadratic complexity affect the compute-optimal model size?
4. **Inference**: How do both architectures scale with context length regarding TTFT and step time?

## Method

### Overall Architecture

This paper does not propose a new model but builds a scaling law measurement pipeline covering both training and inference to answer "whether and when linear-complexity xLSTM can outperform Transformer in scaling." The pipeline consists of three steps: First, scanning Llama-2 style dense Transformers and xLSTM 7B architectures (pure mLSTM layers + MLP) across 80M–7B parameters and 2B–2T tokens using unified data, tokenizers, and a **precise compute caliber**; second, using **scaling law fitting with free exponents + IsoFLOP extrapolation** to locate compute-optimal configurations for each compute budget and compare power-law exponents; finally, using **roofline-based inference latency modeling** to incorporate the impact of context length on TTFT and step time into a unified comparison framework. The study conducted 672 training runs (292 Transformer + 380 xLSTM) using high-quality filtered web documents from DCLM-Baseline, GPT-NeoX tokenizer, and a default sequence length of 8192, totaling $3.2 \times 10^{23}$ FLOPs.

### Key Designs

**1. Precise FLOP Caliber: Enabling Fair Comparison between Linear and Quadratic Models**

Traditional scaling research uses $C(N,D)=6ND$ to approximate compute, which treats models as purely feed-forward and ignores the quadratic term of attention. This is fatal for comparing Transformer and xLSTM—the former's compute grows quadratically with context length, while the latter's is linear. $6ND$ systematically underestimates the true cost of Transformers. Ours adopts an operator-wise precise FLOP formula, separating the quadratic computation of attention from feed-forward components and separately modeling xLSTM's recursive updates and mLSTM matrix operations. This provides a truly standardized compute axis for subsequent Pareto frontier and compute-optimal conclusions.

**2. Scaling Law Fitting with Free Exponents and IsoFLOP Extrapolation: Locating Optimal Models for Each Budget**

Loss fitting uses $\hat{L}(N,D) = E + (A N^{-\alpha} + B D^{-\beta})^{\gamma}$, where $E$ is irreducible loss, $N$ is parameter count, and $D$ is the number of tokens. Compared to the fixed Chinchilla form, a free exponent $\gamma$ is introduced to improve the fitting quality for both architectures. Based on this, the IsoFLOP method is applied: for a fixed compute budget $H$, combinations of $N$ and $D$ are sampled and fitted with a second-order polynomial to find the optimal $N^*(H)$ and $D^*(H)$, which are then extrapolated to larger compute using power laws $\hat{N}^*(H)=A'\cdot H^{a}$ and $\hat{D}^*(H)=B'\cdot H^{b}$. This process allows for a comparable answer to whether compute should be allocated to a larger model or more data, and allows the power-law exponent in the overtraining regime to be directly read and compared.

**3. Roofline-based Inference Latency Modeling: Incorporating Context Length**

Beyond training scaling, inference latency is modeled as either compute-bound or memory-bound limits: $\tau = \text{FLOPs}_{\text{algo}} / \alpha_{\text{eff}} + \epsilon$ or $\tau = \text{Bytes}_{\text{mem}} / \beta_{\text{eff}} + \epsilon$, where $\alpha_{\text{eff}}$ and $\beta_{\text{eff}}$ are measured effective compute/bandwidth and $\epsilon$ is fixed overhead. Prefill and generation phases are analyzed separately to characterize the structural difference where Transformer's KV cache expands linearly with context while xLSTM's state remains constant. This allows the dependence of TTFT and step time on context length to be theoretically predicted and validated.

## Key Experimental Results

### Training Scaling

| Finding | Details |
|------|------|
| **Pareto Dominance** | xLSTM strictly Pareto-dominates Transformer over nearly 5 orders of magnitude of compute. |
| **Overtraining Exponent** | xLSTM's power-law exponent $\eta$ remains constant in the range of $M=22$ to $M=2200$, consistent with Transformer. |
| **Compute-optimal Size** | At the same compute, the optimal xLSTM model is larger (linear operations are cheaper, allocating more parameters to depth/width). |
| **Context Length Impact** | Transformer's optimal model size drops significantly from 2048 to 16384; xLSTM remains stable. |

### Inference Performance

| Metric | 16k Prefill Results |
|------|-----------------|
| **TTFT** | xLSTM is 30–50% lower than a similarly sized Transformer. |
| **Step Time** | xLSTM is independent of prefill length (constant); Transformer grows linearly. |
| **Extreme Comparison** | At 16k prefill, the step time of the largest xLSTM < the step time of the smallest Transformer. |

### General Laws
- The "loss vs. model size" relationship for compute-optimal models for both xLSTM and Transformer approximately falls on the same line—implying a cross-architectural universal relationship between performance and model size.

## Highlights & Insights
- **Comprehensive Systematicity**: 672 training runs covering 5 orders of compute magnitude, considering training + inference + context length.
- **Precise FLOP Calculation**: Moves beyond $6ND$ approximation to provide a fair benchmark for linear vs. quadratic architectures.
- **Practical Guidance**: Proves xLSTM exponent stability in the overtraining regime, supporting "small model + large data" deployment strategies.
- **Inference Modeling**: Roofline-based theoretical models align highly with empirical measurements.

## Limitations & Future Work
- Only considers cross-entropy loss, without evaluating downstream tasks (reasoning, code, multilingual, etc.).
- Does not involve MoE or hybrid Attention+xLSTM architectures.
- Inference experiments are limited to a single GPU, not considering multi-GPU distributed scenarios.
- Training data is limited to DCLM-Baseline, without verifying the impact of data distribution changes.
- Does not explore the actual quality performance of xLSTM under ultra-long contexts (>16k) (e.g., recall ability).
- No horizontal comparison with other linear architectures like Mamba or RWKV.

## Related Work & Insights
- **Chinchilla (Hoffmann 2022)**: Ours reproduces compute-optimal exponents for Transformer and extends them to xLSTM.
- **Gadre 2024 / Sardana 2024**: Ours aligns with these in overtraining regime analysis but adds a cross-architecture dimension.
- **Shen 2024**: While showing linear models are "on par" with Transformer, ours further proves xLSTM "outperforms" Transformer.
- **Poli 2024**: Shows hybrid architectures outperform pure Transformers; ours proves pure linear architectures can also win.
- **Porian 2024**: Ours reproduces their Transformer power-law exponent $a$.

## Related Work & Insights
- The Pareto dominance of xLSTM implies better pre-trained models can be obtained at the same compute budget, which is particularly valuable for resource-constrained scenarios.
- The impact of context length on compute-optimal model size is a widely ignored dimension that warrants verification in other architectures (Mamba, RWKV, etc.).
- Inference advantages expand with context growth, suggesting massive potential for linear architectures in long-context inference (e.g., CoT, document understanding).
- The cross-architectural "model size vs. loss" universal relationship is a theoretical problem deserving deeper research.
- The precise FLOP calculation methodology can be reused to evaluate the scaling behavior of other linear architectures like Mamba, RWKV, and RetNet.
- The finding of constant exponents in the overtraining regime provides theoretical assurance for "small model, more data" deployment strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic scaling law comparison between linear complexity and Transformer.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 672 training runs, multi-dimensional analysis, theoretical + empirical inference modeling.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, professional charts.
- Value: ⭐⭐⭐⭐ — Provides important scaling guidance for the engineering deployment of linear complexity architectures.
- Comprehensive: ⭐⭐⭐⭐ — Solid experiments, clear conclusions, direct reference value for architecture selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs](scaling_laws_meet_model_architecture_toward_inference-efficient_llms.md)
- [\[ICLR 2026\] Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](towards_greater_leverage_scaling_laws_for_efficient_mixture-of-experts_language_.md)
- [\[ICLR 2026\] Scaling Linear Attention Capacity with Sparse State Expansion](scaling_linear_attention_capacity_with_sparse_state_expansion.md)
- [\[NeurIPS 2025\] Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](../../NeurIPS2025/llm_efficiency/tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)
- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)

</div>

<!-- RELATED:END -->

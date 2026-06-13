---
title: >-
  [Paper Note] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity
description: >-
  [ICLR2026][LLM Efficiency][scaling laws] This paper systematically compares the scaling laws of xLSTM and Transformer, demonstrating that xLSTM strictly dominates Transformer of the same scale on the training loss–comput…
tags:
  - "ICLR2026"
  - "LLM Efficiency"
  - "scaling laws"
  - "xLSTM"
  - "linear complexity"
  - "Transformer comparison"
  - "inference efficiency"
date: 2026-05-08
content_hash: a0184355aab2b9b1
---

# xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity

**Conference**: ICLR2026  
**arXiv**: [2510.02228](https://arxiv.org/abs/2510.02228)  
**Code**: [NX-AI/xlstm_scaling_laws](https://github.com/NX-AI/xlstm_scaling_laws)  
**Area**: LLM Efficiency  
**Keywords**: scaling laws, xLSTM, linear complexity, Transformer comparison, inference efficiency  

## TL;DR
This paper systematically compares the scaling laws of xLSTM and Transformer, demonstrating that xLSTM strictly dominates Transformer of the same scale on the training loss–compute Pareto frontier, in the overtrained regime, and in inference speed, with the advantage growing as context length increases.

## Background & Motivation
- Scaling laws are a core guiding tool for LLM design (Kaplan 2020, Chinchilla 2022), yet existing studies focus almost exclusively on the Transformer architecture.
- Linear-complexity architectures such as xLSTM have demonstrated competitive performance at the billion-parameter scale (xLSTM 7B), but a systematic scaling comparison with Transformers is lacking.
- The conventional FLOP approximation $C(N,D)=6ND$ ignores the compute contribution of the attention mechanism, making it unsuitable for fair comparison between linear- and quadratic-complexity models.
- The interaction between inference efficiency (TTFT, step time) and context length also lacks systematic analysis.

## Core Problem
1. **Training efficiency**: Given a fixed compute budget, which achieves lower loss — xLSTM or Transformer?
2. **Overtrained regime**: Does xLSTM maintain stable power-law exponents at high token-to-parameter ratios?
3. **Context length**: How does linear vs. quadratic complexity affect compute-optimal model size?
4. **Inference**: How does TTFT and step time scale with context length for each architecture?

## Method

### Overall Architecture
- **Models**: Llama-2-style dense multi-head Transformer vs. xLSTM 7B architecture (pure mLSTM layers + MLP).
- **Scale range**: 80M–7B parameters, 2B–2T tokens, 672 training runs in total (292 Transformer + 380 xLSTM).
- **Total compute**: $3.2 \times 10^{23}$ FLOPs.
- **Training data**: DCLM-Baseline (high-quality filtered web documents), GPT-NeoX tokenizer, default sequence length 8192.

### Key Designs

**Exact FLOP Computation**
- The simplified $6ND$ approximation is discarded in favor of exact FLOP formulas that distinguish attention compute (quadratic term) from feed-forward compute.
- Exact FLOP accounting is also applied to xLSTM's recurrent updates and mLSTM matrix operations.

**Scaling Law Fitting**
- **Parametric fit**: $\hat{L}(N,D) = E + (A N^{-\alpha} + B D^{-\beta})^{\gamma}$, where the free parameter $\gamma$ is introduced to improve fit quality.
- **IsoFLOP method**: For a fixed compute budget, $N$ and $D$ are varied; a second-order polynomial is fitted to identify optimal $N^*(H)$ and $D^*(H)$.
- **Power-law extrapolation**: $\hat{N}^*(H) = A' \cdot H^a$, $\hat{D}^*(H) = B' \cdot H^b$.

**Inference Modeling**
- Inference time is modeled as $\tau = \text{FLOPs}_{\text{algo}} / \alpha_{\text{eff}} + \epsilon$ or $\tau = \text{Bytes}_{\text{mem}} / \beta_{\text{eff}} + \epsilon$.
- A roofline model is used to determine whether execution is compute-bound or memory-bound.
- Prefill and generation stages are analyzed separately.

## Key Experimental Results

### Training Scaling

| Finding | Details |
|---------|---------|
| **Pareto dominance** | xLSTM strictly Pareto-dominates Transformer across nearly 5 orders of magnitude in compute. |
| **Overtrained exponent** | xLSTM's power-law exponent $\eta$ remains constant over $M=22$ to $M=2200$, consistent with Transformer. |
| **Compute-optimal size** | At equal compute, xLSTM's optimal model is larger (cheaper linear operations → more parameters allocated to depth/width). |
| **Context length effect** | Transformer's optimal model size drops significantly as context grows from 2048 to 16384; xLSTM remains stable. |

### Inference Performance

| Metric | Results at 16k prefill |
|--------|----------------------|
| **TTFT** | xLSTM is 30–50% lower than same-size Transformer. |
| **Step time** | xLSTM is constant regardless of prefill length; Transformer grows linearly. |
| **Extreme comparison** | At 16k prefill, the largest xLSTM's step time is less than the smallest Transformer's step time. |

### General Findings
- The loss vs. model size relationship for compute-optimal models falls approximately on the same curve for both xLSTM and Transformer, suggesting a universal cross-architecture relationship between performance and model size.

## Highlights & Insights
- **Comprehensive systematicity**: 672 training runs spanning 5 orders of magnitude in compute, jointly examining training, inference, and context length.
- **Exact FLOP computation**: Moving beyond the $6ND$ approximation provides a fair basis for comparing linear and quadratic architectures.
- **Practical guidance**: The stable power-law exponent in the overtrained regime supports "small model + large data" deployment strategies.
- **Inference modeling**: The roofline-based theoretical model closely matches empirical measurements.

## Limitations & Future Work
- Only cross-entropy loss is considered; downstream tasks (reasoning, code, multilingual, etc.) are not evaluated.
- MoE and hybrid Attention+xLSTM architectures are not explored.
- Inference experiments are limited to a single GPU; multi-GPU distributed inference is not considered.
- Training data is restricted to DCLM-Baseline; the effect of distribution shift is not validated.
- Actual quality of xLSTM at very long contexts (>16k), such as recall ability, is not examined.
- No lateral comparison with other linear architectures such as Mamba and RWKV is performed.

## Related Work & Insights
- **Chinchilla (Hoffmann 2022)**: This paper reproduces the compute-optimal exponents for Transformer and extends the analysis to xLSTM.
- **Gadre 2024 / Sardana 2024**: This paper's overtrained regime analysis is consistent with their findings but adds a cross-architecture dimension.
- **Shen 2024**: Demonstrates that linear models are "on par" with Transformers; this paper goes further to show xLSTM is "superior."
- **Poli 2024**: Hybrid architectures outperform pure Transformers; this paper shows that a pure linear architecture can also outperform them.
- **Porian 2024**: This paper reproduces their Transformer power-law exponent $a$.

**Implications**
- xLSTM's Pareto dominance implies better pretrained models at equal compute, which is particularly valuable in resource-constrained settings.
- The effect of context length on compute-optimal model size is a widely neglected dimension and deserves validation in other architectures (Mamba, RWKV, etc.).
- The inference advantage grows with context length, suggesting substantial potential for linear architectures in long-context inference scenarios such as chain-of-thought and document understanding.
- The universal "model size vs. loss" relationship across architectures is a theoretically interesting problem worthy of deeper investigation.
- The exact FLOP computation methodology can be directly reused to evaluate the scaling behavior of other linear architectures such as Mamba, RWKV, and RetNet.
- The finding that overtrained-regime exponents remain constant provides theoretical support for the "small model, large data" deployment strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic scaling law comparison between linear-complexity and Transformer architectures.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 672 training runs, multi-dimensional analysis, both theoretical and empirical inference modeling.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and professional figures.
- Value: ⭐⭐⭐⭐ — Provides important scaling guidance for engineering deployment of linear-complexity architectures.
- Overall: ⭐⭐⭐⭐ — Rigorous experiments and clear conclusions with direct reference value for architecture selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[NeurIPS 2025\] Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](../../NeurIPS2025/llm_efficiency/tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)
- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [\[ICLR 2026\] IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling](iterresearch_rethinking_long-horizon_agents_with_interaction_scaling.md)
- [\[AAAI 2026\] HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](../../AAAI2026/llm_efficiency/hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)

</div>

<!-- RELATED:END -->

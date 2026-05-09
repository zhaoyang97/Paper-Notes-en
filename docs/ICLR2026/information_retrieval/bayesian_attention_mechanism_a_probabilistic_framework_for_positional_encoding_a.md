---
title: >-
  [Paper Note] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation
description: >-
  [ICLR 2026][Bayesian attention] This paper reformulates positional encoding as prior distributions within a Bayesian attention mechanism, unifying NoPE (uniform prior) and ALiBi (Laplacian prior), and proposes a Generalized Gaussian prior (GGD-BAM) that achieves perfect passkey retrieval at 500× the training length by adding only 384 parameters.
tags:
  - ICLR 2026
  - Bayesian attention
  - positional encoding
  - context extrapolation
  - generalized Gaussian distribution
  - long context
date: 2026-05-08
content_hash: 9b0369237c31da23
---

# Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation

**Conference**: ICLR 2026
**arXiv**: [2505.22842](https://arxiv.org/abs/2505.22842)
**Code**: [https://github.com/ArthurSBianchessi/BAM](https://github.com/ArthurSBianchessi/BAM)
**Area**: Information Retrieval
**Keywords**: Bayesian attention, positional encoding, context extrapolation, generalized Gaussian distribution, long context

## TL;DR
This paper reformulates positional encoding as prior distributions within a Bayesian attention mechanism, unifying NoPE (uniform prior) and ALiBi (Laplacian prior), and proposes a Generalized Gaussian prior (GGD-BAM) that achieves perfect passkey retrieval at 500× the training length by adding only 384 parameters.

## Background & Motivation

**Background**: Transformers lack inherent positional information and rely on positional encoding. Existing methods (Sinusoidal, RoPE, ALiBi, NoPE) exhibit varying context length extrapolation performance but lack a unified theoretical understanding.

**Limitations of Prior Work**: (a) PE methods are largely empirically driven with weak theoretical foundations; (b) evaluation over-relies on perplexity, which is insufficient—models can achieve low perplexity via sliding-window attention while failing to retrieve distant information.

**Key Challenge**: Different PE methods behave inconsistently across scenarios, yet no unified framework exists to analyze their behavioral differences and applicable ranges.

**Goal**: (a) Provide a theoretically unified framework for PE; (b) derive novel PE strategies from theory; (c) achieve genuine long-context extrapolation in terms of retrieval rather than perplexity alone.

**Key Insight**: Interpret attention weights $p_{ij}$ as a joint probability distribution over content and position, making PE a natural positional prior.

**Core Idea**: PE serves as a positional prior in attention—by selecting the Generalized Gaussian distribution and permitting "anti-local" attention heads with $\beta < 0$, the model achieves retrieval over ultra-long contexts.

## Method

### Overall Architecture
BAM decomposes attention weights as $p_{ij} = p(f_{\text{cont}}(\mathbf{q}_i, \mathbf{k}_j)) \cdot p(g_{\text{pos}}(i,j)) / Z$. The positional prior $p(g_{\text{pos}})$ governs the positional preference of attention, with different PE methods corresponding to different prior distributions.

### Key Designs

1. **Probabilistic Unification of PE**:

    - NoPE = uniform prior (equal weight for all positions)
    - ALiBi = Laplacian prior ($\beta = 1$, linear decay)
    - BAM-GGD = Generalized Gaussian prior (learnable $\beta$)

2. **Generalized Gaussian Positional Prior**:

    - Each attention head learns two parameters: $\theta_\alpha$ (scale) and $\theta_\beta$ (shape)
    - $\beta > 1$: more locally concentrated than ALiBi
    - $\beta \in (0,1)$: heavy-tailed decay, capable of attending to more distant positions than ALiBi
    - $\beta \leq 0$: **anti-local attention**—suppresses nearby tokens and focuses on distant information, functioning as "retrieval heads"

3. **Automatic Emergence of Three Attention Head Modes**:

    - After training, parameters spontaneously cluster into three groups: $\beta > 0$ (local heads), $-0.6 \leq \beta \leq 0$ (retrieval heads), and $\beta < -0.6$ (aggressive retrieval heads), appearing stably across model scales

### Loss & Training
- Standard language modeling cross-entropy loss
- Trained on 512-token contexts; total additional parameters amount to only 384 ($2 \times \text{Heads} \times \text{Layers}$, with $\mu$ fixed at 0)

## Key Experimental Results

### Main Results (Passkey Retrieval Accuracy)

| Method | Training Length | 1K | 4K | 8K | 16K | 32K | 256K |
|--------|----------------|-----|-----|-----|------|------|-------|
| Sinusoidal | 512 | ~random | 0% | 0% | 0% | 0% | 0% |
| RoPE | 512 | ~random | 0% | 0% | 0% | 0% | 0% |
| ALiBi | 512 | high | declining | low | ~0% | ~0% | 0% |
| **BAM-GGD** | **512** | **100%** | **100%** | **100%** | **100%** | **100%** | **>80%** |

### Ablation Study

| Configuration | Max Passkey Extrapolation | Perplexity | Notes |
|---------------|--------------------------|------------|-------|
| BAM + SSMax | 500× training length | comparable to ALiBi | optimal configuration |
| BAM ($\beta > 0$ only) | limited | slightly better | no retrieval heads |
| BAM ($\beta < 0$ allowed) | 500× | slightly higher | retrieval heads present, weaker local capability |
| ALiBi baseline | ~10× | good | cannot extrapolate far |

### Key Findings
- Attention heads with $\beta \leq 0$ are critical for long-range retrieval—they counterintuitively "ignore" nearby tokens and focus on distant information
- BAM outperforms all baseline PE methods across all tasks on the RULER benchmark
- Perplexity is comparable to ALiBi, yet retrieval capability far exceeds it—demonstrating that perplexity is an insufficient metric for evaluating extrapolation
- The three $\beta$ clustering patterns emerge stably as model scale increases from 120M to larger sizes

## Highlights & Insights
- The **theoretical framework of PE as prior** is notably elegant, providing a unified perspective that explains NoPE, ALiBi, and related methods while naturally deriving novel approaches. This is transferable to any attention design requiring positional sensitivity.
- **Negative-$\beta$ retrieval heads** represent the most striking finding: allowing certain heads to adopt an "anti-local" focus on distant information resembles the functional specialization observed in Mixture of Experts architectures.
- Achieving 500× extrapolation with only 384 additional parameters demonstrates extreme parameter efficiency.

## Limitations & Future Work
- Validation is limited to 120M-parameter models; large-scale (7B+) models remain untested
- Passkey retrieval is a synthetic task; performance on real long-document understanding scenarios has not been verified
- Heads with $\beta < 0$ increase perplexity, necessitating a trade-off between local and retrieval heads

## Related Work & Insights
- **vs. ALiBi**: ALiBi is a special case of BAM at $\beta = 1$; BAM automatically discovers superior positional preferences by learning $\beta$
- **vs. RoPE**: RoPE employs multiplicative PE and falls outside BAM's additive framework; however, RoPE demonstrates significantly weaker extrapolation than BAM in experiments
- **vs. NoPE**: NoPE is a special case of the uniform prior with limited extrapolation capability

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant theoretical framework; negative-$\beta$ retrieval heads constitute a genuinely novel discovery
- Experimental Thoroughness: ⭐⭐⭐⭐ Passkey + RULER + perplexity + visualization, though model scale is limited
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear; visualizations are intuitive
- Value: ⭐⭐⭐⭐⭐ Makes a fundamental contribution to the theoretical understanding of PE

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Embedding-Based Context-Aware Reranker](embedding-based_context-aware_reranker.md)
- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](../../ACL2026/information_retrieval/bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](q_rag_long_context_multi_step_retrieval.md)
- [\[AAAI 2026\] SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](../../AAAI2026/information_retrieval/sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)

</div>

<!-- RELATED:END -->

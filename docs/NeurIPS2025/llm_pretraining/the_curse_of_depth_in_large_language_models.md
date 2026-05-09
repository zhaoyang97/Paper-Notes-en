---
title: >-
  [Paper Note] The Curse of Depth in Large Language Models
description: >-
  [NeurIPS 2025][LLM Pretraining][Pre-Layer Normalization] This paper identifies the root cause of deep-layer degradation in Pre-LN Transformers—exponential growth of output variance causing deep layers to collapse into identity mappings—and proposes a parameter-free LayerNorm Scaling (LNS) strategy that multiplies the LayerNorm output by $1/\sqrt{\ell}$, compressing variance growth from exponential to polynomial. LNS consistently improves perplexity by 5–8% across scales from 130M to 7B parameters.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - Pre-Layer Normalization
  - Curse of Depth
  - Variance Control
  - LayerNorm Scaling
  - Transformer
date: 2026-05-08
content_hash: 6909a690b0c9a163
---

# The Curse of Depth in Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2502.05795](https://arxiv.org/abs/2502.05795)
**Code**: [https://github.com/wenfang-sun/LayerNorm-Scaling](https://github.com/wenfang-sun/LayerNorm-Scaling)
**Area**: LLM Pre-training
**Keywords**: Pre-Layer Normalization, Curse of Depth, Variance Control, LayerNorm Scaling, Transformer

## TL;DR
This paper identifies the root cause of deep-layer degradation in Pre-LN Transformers—exponential growth of output variance causing deep layers to collapse into identity mappings—and proposes a parameter-free LayerNorm Scaling (LNS) strategy that multiplies the LayerNorm output by $1/\sqrt{\ell}$, compressing variance growth from exponential to polynomial. LNS consistently improves perplexity by 5–8% across scales from 130M to 7B parameters.

## Background & Motivation

**Background**: Recent studies have found that nearly half of the deep Transformer blocks in modern LLMs (LLaMA, Mistral, DeepSeek, Qwen) are inefficient—removing deep layers has negligible impact on performance.

**Limitations of Prior Work**: LLM training is extremely costly, and a large number of ineffective layers represents severe waste of computational resources; however, a systematic theoretical explanation for this phenomenon has been lacking.

**Key Challenge**: Although Pre-LN resolves training stability issues (compared to Post-LN), it introduces a new problem: residual connections cause output variance to grow exponentially with depth, progressively diluting the normalization effect of LayerNorm.

**Goal**: ① Why are deep layers in Pre-LN ineffective? ② How can this be characterized mathematically? ③ What is the simplest possible fix?

**Key Insight**: Through analysis of variance propagation and gradient flow, the paper shows that the output variance $\sigma_{x_L}^2$ grows exponentially from $\Theta(L)$ to $\Theta(\exp(L))$, bounding the gradient norm as $\|\partial y_L / \partial x_1\| \leq M$ (a constant), which causes deep layers to degenerate into identity mappings.

**Core Idea**: Multiplying the LayerNorm output by a layer-wise decaying factor $1/\sqrt{\ell}$ compresses variance growth from exponential to polynomial, enabling deep layers to learn effectively again.

## Method

### Overall Architecture
The paper conducts a theoretical analysis to locate the exponential variance growth in Pre-LN, then proposes LayerNorm Scaling (LNS)—a deterministic scaling factor $1/\sqrt{\ell}$ applied to each LayerNorm output—to address the issue.

### Key Designs

1. **Theoretical Diagnosis of Variance Growth (Lemma 3.2 + Theorem 3.3)**:

    - Function: Proves exponential output variance growth in Pre-LN and its consequences.
    - Mechanism: $\sigma_{x_\ell}^2 = \sigma_{x_1}^2 \cdot \Theta(\prod_{k=1}^{\ell-1}(1 + 1/\sigma_{x_k}))$, bounded as $\Theta(L) \leq \sigma_{x_L}^2 \leq \Theta(\exp(L))$. The gradient norm satisfies $\|\partial y_L/\partial x_1\| \leq M$ (constant), rendering deep layers equivalent to identity mappings.
    - Design Motivation: Validated through Jacobian matrix visualization of LLaMA2-7B—deep layers exhibit diagonal dominance with vanishing off-diagonal entries.

2. **LayerNorm Scaling (LNS)**:

    - Function: $\tilde{h}^{(\ell)} = \text{LayerNorm}(h^{(\ell)}) \times \frac{1}{\sqrt{\ell}}$
    - Mechanism: LNS compresses variance growth from exponential to polynomial, satisfying $\Theta(L) \leq \sigma_{x_L}^2 \leq \Theta(L^{2-\epsilon})$. The gradient norm grows from a bounded constant to $\omega(1)$ (increasing with depth), restoring effective learning in deep layers.
    - Design Motivation: $1/\sqrt{\ell}$ is chosen over $1/\ell$—the latter is too aggressive and causes gradient explosion in early layers, while $\sqrt{\ell}$ achieves a sub-linear growth balance.

3. **Relationship with Scaled Initialization**:

    - Function: Analyzes why initialization alone cannot resolve the problem.
    - Mechanism: Scaled Initialization adjusts weights only at initialization, but variance continues to grow exponentially during training. LNS continuously controls variance throughout the entire training process.
    - Design Motivation: Experiments show that combining LNS with Scaled Initialization actually degrades performance, as the two mechanisms for variance control conflict with each other.

### Loss & Training
Standard language modeling loss is used, with zero additional parameters and zero hyperparameters. Only a single line of code modification is required: `output * (1 / sqrt(layer_index))`. It is recommended to remove Scaled Initialization when using LNS.

## Key Experimental Results

### Main Results (Pre-training Perplexity ↓)

| Method | LLaMA-130M | LLaMA-250M | LLaMA-350M | LLaMA-1B |
|--------|-----------|-----------|-----------|---------|
| Post-LN | 26.95 | 1409.79 | 1368.33 | 1390.75 |
| DeepNorm | 27.17 | 22.77 | 1362.59 | 1409.08 |
| Mix-LN | 26.07 | 21.39 | 1363.21 | 1414.78 |
| Pre-LN (baseline) | 26.73 | 21.92 | 19.58 | 17.02 |
| **Pre-LN + LNS** | **25.76** | **20.35** | **18.20** | **15.71** |

DeepNorm and Mix-LN diverge at larger scales, while LNS remains consistently stable.

### Ablation Study (Fine-tuning Downstream Task Accuracy ↑)

| Method | MMLU | BoolQ | ARC-e | PIQA | HellaSwag | Avg. |
|--------|------|-------|-------|------|-----------|------|
| Pre-LN (250M) | 24.93 | 38.35 | 40.15 | 63.55 | 26.34 | 36.93 |
| **LNS (250M)** | **27.08** | **58.17** | **45.24** | **67.38** | **32.81** | **43.14** |
| Pre-LN (1B) | 26.54 | 62.20 | 45.70 | 67.79 | 30.96 | 43.01 |
| **LNS (1B)** | **28.69** | **61.80** | **48.85** | **67.92** | **33.94** | **44.87** |

### Key Findings
- **Substantial Variance Reduction**: Deep-layer variance in Pre-LN reaches 175; LNS constrains it below 25 (a 7× reduction).
- **Deep Layers Restored**: Under LNS, layer-wise pruning incurs uniform performance degradation; Angular Distance increases from near 0 to >0.6.
- **Scale Consistency**: Trends are consistent across scales from 60M to 7B; OLMo-7B loss improves from 2.69 → 2.50 (+7.1%).
- **Pre-training to Fine-tuning Transfer**: Pre-training gains transfer fully to downstream tasks.

## Highlights & Insights
- **Theoretical Depth**: A complete causal chain from variance → gradient → identity mapping is established; Theorem 3.3 and 4.2 rigorously prove both the problem and the solution.
- **Extreme Simplicity**: A single line of code, zero parameters, and zero hyperparameters—a rare improvement that is "too simple to refuse."
- **Practical Applicability**: All LLMs using Pre-LN (virtually all mainstream models) can directly benefit.
- **Jacobian Visualization**: Diagonal dominance in deep layers intuitively reveals the identity mapping behavior.

## Limitations & Future Work
- The choice of $1/\sqrt{\ell}$ lacks a rigorous optimality proof and is validated only heuristically and empirically.
- The conflict with Scaled Initialization is not analyzed in depth.
- The optimal placement of LNS in ViT differs (after Attn/MLP vs. after LayerNorm), limiting its generality.
- The relationship between variance and sequence length in long-context settings is not discussed.

## Related Work & Insights
- **vs. Mix-LN**: Mixing Pre-LN/Post-LN introduces a hyperparameter α and diverges at scales beyond 350M. LNS is simpler and more stable.
- **vs. DeepNorm**: Adjusts residual weights to stabilize training, but diverges at 1B scale (PPL 1409).
- **vs. LayerScale**: Learns per-layer scaling factors, introducing learnable parameters; empirically yields lower performance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First analysis of Pre-LN deep-layer inefficiency from the fundamental perspective of variance growth, with both theoretical and practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across multiple architectures (LLaMA/OLMo/Qwen2.5/ViT) × multiple scales (130M–7B) × full training pipelines.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow from problem → root cause → solution → validation; Jacobian visualization is intuitive.
- Value: ⭐⭐⭐⭐⭐ A one-line code change that benefits all Pre-LN models; should become standard practice in LLM training.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](retrospective_incontext_learning_for_temporal_credit_assignm.md)
- [\[NeurIPS 2025\] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Long-Short Alignment for Effective Long-Context Modeling in LLMs
description: >-
  [ICML 2025][LLM Efficiency][length generalization] This paper proposes a new perspective on length generalization from the angle of model output distributions, termed Long-Short Alignment. It highlights that the consistency of output distributions across inputs of different lengths is a key factor in length generalization. The authors introduce a Long-Short Misalignment metric and utilize it as a training regularization term, which significantly improves long-context modeling…
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "length generalization"
  - "output distribution alignment"
  - "long-short alignment"
  - "regularization"
  - "long-context modeling"
date: 2026-05-08
content_hash: 75e3bc40aa02eb74
---

# Long-Short Alignment for Effective Long-Context Modeling in LLMs

**Conference**: ICML 2025  
**arXiv**: [2506.11769](https://arxiv.org/abs/2506.11769)  
**Code**: [https://github.com/PKU-ML/LongShortAlignment](https://github.com/PKU-ML/LongShortAlignment)  
**Area**: LLM Efficiency  
**Keywords**: length generalization, output distribution alignment, long-short alignment, regularization, long-context modeling

## TL;DR
This paper proposes a new perspective on length generalization from the angle of model output distributions, termed Long-Short Alignment. It highlights that the consistency of output distributions across inputs of different lengths is a key factor in length generalization. The authors introduce a Long-Short Misalignment metric and utilize it as a training regularization term, which significantly improves long-context modeling capabilities on both synthetic and natural language tasks.

## Background & Motivation
**Background**: LLMs are limited by the fixed context window of Transformers, making long-context modeling a core challenge. Expanding the context window brings more in-context learning exemplars and longer reasoning chains.

**Limitations of Prior Work**: Long-context training is extremely time- and memory-consuming, making it crucial to understand and improve length generalization (generalizing from short sequence training to long sequence testing).

**Key Challenge**: Existing work mainly understands length generalization from the input side (positional encoding design) or internal model mechanisms (such as RASP analysis), overlooking a key dimension—the output behavior of the model.

**Goal**: Reveal how the consistency of the model's output distribution across different lengths (long-short alignment) affects length generalization, and propose methods to improve it.

**Key Insight**: Comparative experiments on synthetic tasks reveal a phenomenon—the mean prediction task generalizes well (where the output space is fixed in $[0,1]$), while the length prediction task generalizes poorly (where the output space scales with length). This insight is then generalized to natural language tasks.

**Core Idea**: Generating consistent output distributions across inputs of different lengths is key to length generalization, and this consistency can be explicitly promoted via a regularization term.

## Method

### Overall Architecture
The method consists of three steps: (1) discovering the causal relationship between long-short alignment and length generalization in synthetic tasks; (2) proposing the Long-Short Misalignment metric to quantify the degree of alignment in natural language tasks; (3) incorporating this metric as a regularization term in the training loss. During training, only two forward passes (the original sequence and a truncated sequence) are required to compute the Symmetrical Cross-Entropy (SCE) loss on their overlapping parts.

### Key Designs

1. **Synthetic Task Analysis and Output Reparameterization (OutRep)**:

    - **Function**: Reveals that the stability of the output space determines length generalization capability through a comparison between mean prediction vs. length prediction.
    - **Mechanism**: The output of mean prediction is always within $[0,1]$ regardless of how the input length changes, whereas the output of length prediction scales linearly with length (with support set $\{l\}$), leading to out-of-distribution generalization failure.
    - **Theorem 3.1**: The generalization error of length prediction is $O((l_{\text{test}} - l_{\text{train}})^2)$, while that of mean prediction is $O(1)$.
    - **OutRep**: Uses an invertible function $f(x)$ to map outputs of length prediction (e.g., $f(x)=1/\sqrt{x}$) to make the output distributions across different lengths more consistent, which significantly improves generalization.
    - **Design Motivation**: Unlike input-side methods such as modifying positional encodings, this approach directly addresses the problem from the output side.

2. **Long-Short Misalignment Metric**:

    - **Function**: Quantifies the deviation of the model's output distributions under inputs of different lengths.
    - **Mechanism**: Given a sequence $x$ and its two suffixes $x[-l_1:]$ and $x[-l_2:]$ (of similar but different lengths), compute the Symmetrical Cross-Entropy (SCE) loss between the model's predictive distributions on both.
    - **Formula**: 
    $$\mathcal{L}_{\text{misalign}} = \mathbb{E}_{x, l_1, l_2}[\mathcal{L}_{\text{SCE}}(g(x[-l_1:]), g(x[-l_2:]))]$$
    - $l_1$ and $l_2$ are sampled from $[l_{\text{train}}/2, l_{\text{train}}]$ to avoid excessively large length discrepancies.
    - **Table 1 Key Finding**: The correlation coefficient (absolute value) between $\mathcal{L}_{\text{misalign}}$ and long-context benchmarks is $0.85$, which is much higher than that of training loss ($0.62$).

3. **Misalignment Regularization**:

    - **Function**: Incorporates $\mathcal{L}_{\text{misalign}}$ as a regularization term in the training loss.
    - **Mechanism**: New loss: $\mathcal{L}_{\text{train}}^* = \mathcal{L}_{\text{train}} + \alpha \cdot \mathcal{L}_{\text{misalign}}$.
    - **Efficient Implementation**: Sample a sequence of length $l_{\text{train}} + l_{\text{extra}}$, use the first $l_{\text{train}}$ and last $l_{\text{train}}$ tokens as two inputs, requiring only two forward passes to simultaneously compute both $\mathcal{L}_{\text{train}}$ and $\mathcal{L}_{\text{misalign}}$.
    - It is recommended to set $\alpha$ within the range $[0.1, 0.3]$; excessively large values lead to over-regularization.

### Loss & Training
- Total loss: $\mathcal{L}_{\text{train}}^* = \mathcal{L}_{\text{train}} + \alpha \cdot \mathcal{L}_{\text{misalign}}$, with $\alpha$ recommended to be $0.1 \sim 0.3$.
- Efficient implementation requires only two forward passes, incurring minimal additional overhead.
- **Theorem 4.1**: Generalization error upper bound = $C_1 \cdot \mathcal{L}_{\text{misalign}} + C_2 \cdot \mathcal{L}_{\text{train}} + C_0$, where $C_1/C_2$ increases as the testing length increases, indicating $\mathcal{L}_{\text{misalign}}$ is more critical for long sequence generalization.
- Applicable to various model adaptation strategies: CLEX, LongQLora, EABF.

## Key Experimental Results

### Main Results (Training on 4K, adapted with CLEX based on Llama2-7B)

| Dataset | Method | LongBench-E (200 steps) | PPL (200 steps) |
|--------|------|-------------------|------------|
| RedPajama-Book | $\mathcal{L}_{\text{train}}$ (Baseline) | 24.7 | 6.12 |
| RedPajama-Book | $+0.1 \cdot \mathcal{L}_{\text{misalign}}$ (Ours) | **26.6** | **5.88** |
| RedPajama-Book | $+0.5 \cdot \mathcal{L}_{\text{misalign}}$ (Ours) | 24.7 | 6.54 |
| PG19 | $\mathcal{L}_{\text{train}}$ (Baseline) | 22.5 | 7.45 |
| PG19 | $+0.1 \cdot \mathcal{L}_{\text{misalign}}$ (Ours) | **25.3** | **7.35** |

### Training on 8K (LongQLora + EABF Adaptation)

| Adaptation Method | Method | LongBench-E (200 steps) | PPL (200 steps) |
|---------|------|-------------------|------------|
| LongQLora | Baseline | 23.4 | 5.82 |
| LongQLora | $+0.1 \cdot \mathcal{L}_{\text{misalign}}$ | **25.8** | **5.77** |
| EABF | Baseline | 23.6 | 6.01 |
| EABF | $+0.1 \cdot \mathcal{L}_{\text{misalign}}$ | **24.8** | **5.91** |

### BABILong Experiment (Reasoning-in-a-haystack)

| Training Loss | 4K | 8K | 16K |
|---------|-----|-----|------|
| Baseline | 48.2 | 42.4 | 37.9 |
| $+0.1 \cdot \mathcal{L}_{\text{misalign}}$| **49.1** | **44.4** | **40.1** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $\alpha=0.1$ | LongBench-E 26.6 | Optimal performance |
| $\alpha=0.3$ | LongBench-E 27.1 | Slightly better but riskier |
| $\alpha=0.5$ | LongBench-E 24.7 | Over-regularization |
| $\alpha=1.0$ | LongBench-E 19.9, PPL 12.92 | Severe over-regularization |
| Sampling range $[1, l_{\text{train}}/2]$ | LongBench-E 25.8 | Current strategy |
| Sampling range $[1, l_{\text{train}}]$ | LongBench-E 19.1 | Excessively large differences hurt performance |

### Key Findings
- The correlation coefficient between $\mathcal{L}_{\text{misalign}}$ and long-context performance ($0.85$) is much higher than that of training loss ($0.62$).
- $\alpha=0.1 \sim 0.3$ is the optimal range; excessively large regularization severely hurts performance.
- The effect is most significant in the middle-context of BABILong (fact depth=50%), alleviating the "loss-in-the-middle" phenomenon.
- The method is compatible with various long-context adaptation strategies.

## Highlights & Insights
- Proposes a brand-new perspective: understanding length generalization from output distributions rather than input features, filling a gap in existing literature.
- Comparative experiments on synthetic tasks (mean vs. length prediction) reveal the importance of long-short alignment in a highly intuitive manner.
- Solid theoretical foundation: Theorem 3.1 explains the synthetic task phenomenon, and Theorem 4.1 provides an upper bound on generalization error.
- Extremely simple implementation: requires only two forward passes and one SCE loss term, with almost zero engineering overhead.

## Limitations & Future Work
- Experiments are mainly based on Llama2-7B, lacking validation on larger models and newer architectures.
- $\alpha$ requires hyperparameter tuning and might have different optimal values for different tasks/datasets.
- The sampling range must be carefully designed; excessively large discrepancies degrade performance.
- The regularization term is primarily applied during the fine-tuning phase; its effect on pre-training from scratch remains unexplored.
- Although the theory establishes an upper bound, it is not a tight bound.

## Related Work & Insights
- Orthogonal to positional encoding methods (PI, YaRN, CLEX): while they modify input representations, this work modifies the training objective.
- Difference from methods like RandomPos: RandomPos implicitly promotes long-short alignment through position randomization, while this work explicitly quantifies and optimizes it.
- The long-short misalignment metric itself can serve as an evaluation metric for model selection.
- Insight: The alignment concept can be generalized to other distribution shift problems (e.g., domain generalization).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Brand-new theoretical perspective, first to introduce the concept of long-short alignment in output space)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Dual validation on synthetic + natural language tasks and comprehensive ablations, but limited model scale)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous reasoning, clear logical progression from synthetic to natural language tasks)
- Value: ⭐⭐⭐⭐ (Provides a new dimension of understanding length generalization, with high practical utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)
- [\[ICML 2025\] NExtLong: Toward Effective Long-Context Training without Long Documents](nextlong_toward_effective_long-context_training_without_long_documents.md)
- [\[ICML 2025\] Curse of High Dimensionality Issue in Transformer for Long-context Modeling](curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling.md)
- [\[ICLR 2026\] SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization](../../ICLR2026/llm_efficiency/solopo_unlocking_long-context_capabilities_in_llms_via_short-to-long_preference_.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)

</div>

<!-- RELATED:END -->

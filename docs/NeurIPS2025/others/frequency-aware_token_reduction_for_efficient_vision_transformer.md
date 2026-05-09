---
title: >-
  [Paper Note] Frequency-Aware Token Reduction for Efficient Vision Transformer
description: >-
  [NeurIPS 2025][token reduction] This paper proposes frequency-aware token reduction from a frequency-domain perspective, partitioning tokens into high-frequency (HF) and low-frequency (LF) groups. HF tokens are selectively retained while LF tokens are aggregated into DC tokens, simultaneously alleviating rank collapse and reducing computational cost in ViTs. The method outperforms existing SOTA across multiple models at a 30% token reduction ratio.
tags:
  - NeurIPS 2025
  - token reduction
  - rank collapse
  - over-smoothing
  - frequency analysis
  - vision transformer
date: 2026-05-08
content_hash: 061b6846354aba5f
---

# Frequency-Aware Token Reduction for Efficient Vision Transformer

**Conference**: NeurIPS 2025
**arXiv**: [2511.21477](https://arxiv.org/abs/2511.21477)
**Code**: [GitHub](https://github.com/jhtwosun/frequency-aware-token-pruning)
**Area**: Other
**Keywords**: token reduction, rank collapse, over-smoothing, frequency analysis, vision transformer

## TL;DR
This paper proposes frequency-aware token reduction from a frequency-domain perspective, partitioning tokens into high-frequency (HF) and low-frequency (LF) groups. HF tokens are selectively retained while LF tokens are aggregated into DC tokens, simultaneously alleviating rank collapse and reducing computational cost in ViTs. The method outperforms existing SOTA across multiple models at a 30% token reduction ratio.

## Background & Motivation
**Background**: The quadratic complexity of Vision Transformers has motivated extensive research on token reduction, primarily categorized into merging (fusing similar tokens) and pruning (discarding unimportant tokens), with representative methods including ToME, EViT, and DynamicViT.

**Limitations of Prior Work**: Existing methods overlook the frequency-domain characteristics of self-attention. SA fundamentally operates as a low-pass filter, and stacking SA layers induces rank collapse (token representations converge toward uniformity). Token reduction exacerbates this issue: merging directly averages out high-frequency signals, while pruning accelerates collapse if tokens carrying high-frequency information are removed.

**Key Challenge**: Reducing token count for efficiency versus preserving high-frequency information to maintain ViT expressiveness presents an apparent contradiction.

**Goal**: To design a method that explicitly protects high-frequency information during token reduction, improving efficiency while alleviating rank collapse.

**Key Insight**: The attention matrix is decomposed into a low-frequency component $A^{LP} = \frac{1}{n}\mathbf{11}^T$ and a high-frequency component $A^{HP} = A - A^{LP}$. Each token's contribution to the high-frequency component in $A^{HP}$ determines whether it is retained or aggregated.

**Core Idea**: Retain tokens that contribute most to the high-frequency output component, and aggregate low-frequency tokens into DC tokens to preserve zero-frequency information.

## Method

### Overall Architecture
At each reduction layer: (1) decompose the high-frequency component $A^{HP}$ from the attention matrix; (2) identify HF and LF tokens via column-wise summation; (3) retain the top-$r$ HF tokens and aggregate LF tokens into DC tokens by spatial local groups; (4) adjust subsequent attention weights using learnable parameters $\omega_1, \omega_2$ to mitigate collapse.

### Key Designs

1. **Frequency-Domain Token Sorting**:

    - **Function**: Partition tokens into high-frequency (HF) and low-frequency (LF) groups.
    - **Mechanism**: Compute the high-frequency component $A^{HP} = A - \frac{1}{n}\mathbf{11}^T$ from the multi-head attention matrix, and obtain each token's high-frequency contribution score $\tilde{A}_k$ via column-wise summation. The top-$r$ tokens by score are designated HF tokens; the bottom-$r$ are LF tokens.
    - **Design Motivation**: Only a simple column-averaging operation is required (far more efficient than FFT or cosine similarity computation), directly leveraging the existing attention matrix with zero additional computational overhead.

2. **Local DC Token Aggregation**:

    - **Function**: Aggregate LF tokens into DC tokens across $w^2$ spatial local groups to preserve zero-frequency information.
    - **Mechanism**: $x_{DC}^j = \frac{1}{|N_{LF}^j|} \sum_{i \in N_{LF}^j} x_i$, with DC tokens updated recursively across multiple reduction layers.
    - **Design Motivation**: Directly discarding LF tokens would eliminate the DC signal (Figure 2b confirms that LF tokens dominate the DC component); local DC tokens ($w>1$) in early layers retain residual high-frequency spatial-local information present in LF tokens.

3. **Attention Weight Adjustment**:

    - **Function**: Modify the attention matrix to emphasize HF tokens and compensate for the reduced attention weights of DC tokens.
    - **Mechanism**: $\hat{A} = A^{LP} + (\omega_1+1)A^{HP} + (\omega_2+1)A^{N_{DC}}$, where $\omega_1$ amplifies high-frequency signals and $\omega_2$ compensates for the underestimated attention scores of DC tokens caused by Jensen's inequality.
    - **Design Motivation**: Reducing token count alone is insufficient; actively suppressing rank collapse tendencies in subsequent layers for the remaining tokens is also necessary.

### Theoretical Support
Proposition 3.1 proves that both pruning and merging satisfy $\|H_f[SA(MX)]\|_F \leq \|H_f[SA(X)]\|_F$, meaning token reduction accelerates rank collapse. The proposed method mitigates this trend by selectively retaining HF tokens.

## Key Experimental Results

### Main Results on DeiT Series (ImageNet-1K, 30% token reduction per layer)

| Model | Method | MACs | Accuracy |
|-------|--------|------|----------|
| DeiT-S | Baseline | 4.6G | 79.8% |
| DeiT-S | ToME | 2.9G | 79.5% |
| DeiT-S | EViT | 3.0G | 79.5% |
| DeiT-S | DiffRate | 2.9G | 79.6% |
| DeiT-S | **Ours** | **2.9G** | **80.0%** |
| DeiT-B | Baseline | 17.6G | 81.8% |
| DeiT-B | ToME | 11.2G | 81.7% |
| DeiT-B | **Ours** | **11.5G** | **82.1%** |

### Self-Supervised Models

| Model | Baseline Acc | Ours Acc | MACs Reduction |
|-------|-------------|----------|----------------|
| MAE ViT-B | 83.6% | **83.5%** | ~35% |
| DINO ViT-S | 81.5% | **81.5%** | ~35% |

### Ablation Study

| Configuration | Accuracy |
|---------------|----------|
| HF token retention only | 79.6% |
| + DC token | 79.8% |
| + Local DC | 79.9% |
| + Attention adjustment ($\omega_1, \omega_2$) | **80.0%** |

### Key Findings
- HF tokens demonstrably carry more high-frequency signal (validated by frequency analysis in Figure 2a), while LF tokens dominate the DC component (validated by similarity analysis in Figure 2b).
- Adding noise to HF tokens degrades accuracy far more than adding noise to LF tokens (Figure 2c), confirming that HF tokens are more critical to model performance.
- Existing pruning methods (e.g., EViT) tend to retain HF tokens only in the last few layers, exhibiting inconsistent behavior in intermediate layers; the proposed method explicitly retains HF tokens across all layers.
- Under 30%+ token reduction, accuracy improves over the unreduced baseline on most models, indicating that the positive effect of alleviating rank collapse outweighs the information loss.

## Highlights & Insights
- **Frequency-Domain Understanding of Token Reduction**: This work is the first to connect token reduction with ViT rank collapse/over-smoothing theory, offering a novel design perspective.
- **Efficient HF/LF Sorting**: Column-averaging of the attention matrix alone, with zero additional computation, effectively distinguishes high- and low-frequency tokens.
- **Counter-Intuitive Accuracy Gains**: Accuracy improvements achieved by reducing tokens suggest that rank collapse is a genuine bottleneck in ViTs, rather than insufficient capacity or parameters.

## Limitations & Future Work
- **Classification Tasks Only**: Dense prediction tasks such as detection and segmentation may be more sensitive to low-frequency information.
- **Fixed Reduction Layer Positions** (layers 4/7/10): Currently hard-coded, which may not be optimal for all model architectures.
- **Fine-Tuning Required for $\omega_1, \omega_2$**: A 30-epoch fine-tuning stage is needed, making the method not fully training-free.

## Related Work & Insights
- **vs. ToME (merging)**: ToME merges similar tokens, which is essentially low-pass filtering and accelerates rank collapse; the proposed method retains HF tokens to counteract collapse.
- **vs. EViT (CLS-based pruning)**: EViT selects tokens based on CLS attention but does not explicitly account for frequency characteristics.
- **vs. DiffRate**: DiffRate learns adaptive reduction ratios but similarly neglects frequency-domain properties.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Frequency-domain token reduction is a novel contribution; the connection to rank collapse theory is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple model families (DeiT/ViT/MAE/DINO), multiple training strategies, detailed ablations, and frequency analysis visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical motivation is clear and closely aligned with experimental findings.
- **Value**: ⭐⭐⭐⭐ — A practical ViT acceleration method; the rank collapse perspective offers broader inspiration for ViT efficiency research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] UniFormer: Unified and Efficient Transformer for Reasoning Across General and Custom Computing](uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)
- [\[NeurIPS 2025\] Active Measurement: Efficient Estimation at Scale](active_measurement_efficient_estimation_at_scale.md)
- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[NeurIPS 2025\] AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking](acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)

</div>

<!-- RELATED:END -->

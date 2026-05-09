---
title: >-
  [Paper Note] REOrdering Patches Improves Vision Models
description: >-
  [NeurIPS 2025][Model Compression][patch ordering] This paper reveals that patch ordering significantly affects the performance of long-sequence vision models, and proposes the REOrder framework, which leverages information-theoretic priors and reinforcement learning to automatically discover optimal patch permutations, achieving up to 3.01% improvement on ImageNet-1K and 13.35% on FMoW.
tags:
  - NeurIPS 2025
  - Model Compression
  - patch ordering
  - long-sequence models
  - Mamba
  - reinforcement learning
  - Plackett-Luce
date: 2026-05-08
content_hash: 982407f85ce7cf9e
---

# REOrdering Patches Improves Vision Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.23751](https://arxiv.org/abs/2505.23751)
**Code**: [Project Page](https://arxiv.org/abs/2505.23751)
**Area**: Model Compression
**Keywords**: patch ordering, long-sequence models, Mamba, reinforcement learning, Plackett-Luce

## TL;DR

This paper reveals that patch ordering significantly affects the performance of long-sequence vision models, and proposes the REOrder framework, which leverages information-theoretic priors and reinforcement learning to automatically discover optimal patch permutations, achieving up to 3.01% improvement on ImageNet-1K and 13.35% on FMoW.

## Background & Motivation

Vision Transformers and other sequence models require flattening 2D images into 1D patch sequences. The conventional approach uses raster-scan (row-major) order, based on the assumption that full self-attention is permutation-equivariant and thus patch order is irrelevant.

However, this assumption **does not hold** for modern long-sequence models:

- **Transformer-XL** introduces a memory mechanism based on segment-level recurrence and relative positional encodings, where the relative index $i-j$ makes it sensitive to permutations.
- **Longformer** employs sliding-window local attention with global tokens, and the fixed local mask is not equivariant under permutation.
- **Mamba** uses content-dependent state-space updates (linear recurrence), where the token-by-token causal scan is inherently order-dependent.

These architectures trade the permutation equivariance of full attention for $O(n)$ or $O(nw)$ complexity, at the cost of output sensitivity to patch ordering. Experiments show that simply switching from row-major to column-major or Hilbert curve ordering can lead to substantial accuracy changes. More critically, **no single fixed permutation is optimal across all models and datasets**, motivating a data-driven approach to discover the optimal ordering.

## Method

### Overall Architecture

REOrder is a two-stage framework:

1. **Information-theoretic prior discovery**: Candidate orderings are filtered by evaluating the compressibility of different patch sequences, providing a warm-start initialization.
2. **Reinforcement learning optimization**: A Plackett-Luce permutation policy combined with the REINFORCE algorithm searches the combinatorial space for the optimal permutation.

### Key Designs

1. **Theoretical analysis of permutation equivariance**: The authors formally prove that full self-attention satisfies $\text{Attn}(\mathbf{P}\mathbf{X}) = \mathbf{P}\,\text{Attn}(\mathbf{X})$, and individually analyze why Transformer-XL, Longformer, and Mamba violate this property. This provides a theoretical foundation for the claim that patch order matters. The core insight is that *efficient attention approximations introduce sensitivity to input permutations as a byproduct of reducing complexity*.

2. **Information-theoretic prior (compression rate analysis)**: Image patches are discretized into VQ-VAE tokens and compressed via LZMA after unigram/bigram tokenization. Row-major and Hilbert curve orderings yield higher compression rates (greater local redundancy), while column-major and spiral orderings yield lower rates. Compression rate has a weak, model-dependent correlation with downstream accuracy—lower compression rate is not always better, but serves as a weak prior for initialization.

3. **Plackett-Luce permutation policy**: The permutation distribution is parameterized by a score vector $\mathbf{z} \in \mathbb{R}^n$ (where $n$ is the number of patches). Sampling is parallelized via the Gumbel-top-$k$ trick: $\pi = \text{argsort}_{\text{desc}}(\mathbf{z} + \tau \mathbf{g})$, where $g_i \sim \text{Gumbel}(0,1)$. For 196 patches, the policy requires only 196 parameters with negligible overhead. The log-probability admits a closed-form expression: $\log P(\pi|\mathbf{z}) = \sum_{i=1}^n [z_{\pi_i} - \log\sum_{k=i}^n \exp(z_{\pi_k})]$, computed efficiently via reverse cumulative logsumexp.

### Loss & Training

A three-stage curriculum is adopted:

- **First $N$ epochs**: The classifier is trained with standard row-major ordering to establish a stable feature representation.
- **Epochs $N$ to $N+M$**: The PL policy is activated and trained with REINFORCE. The temperature $\tau$ is linearly annealed from 0 to 0.2 and back to 0, encouraging exploration before convergence.
- **After epoch $N+M$**: The policy is frozen to the deterministic maximum-likelihood permutation $\hat{\pi} = \text{argsort}(\mathbf{z})$, and only the backbone is trained.

The reward is defined as the negative cross-entropy loss $r = -\mathcal{L}_{\text{CE}}$, with a momentum baseline $b_{t+1} = \beta b_t + (1-\beta)r_t$ ($\beta=0.99$) for variance reduction. The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{policy}}$.

## Key Experimental Results

### Main Results — Fixed Permutation Comparison

| Model | Dataset | Row-major | Best Fixed Permutation | Accuracy Range |
|-------|---------|-----------|----------------------|----------------|
| ViT | IN-1K | 37.5% | ~37.5% | ±0% (equivariant) |
| T-XL | IN-1K | baseline | Column-major: +1.92% | −6.43% ~ +1.92% |
| Longformer | IN-1K | baseline | Column-major: +1.83% | −0.5% ~ +1.83% |
| Mamba | IN-1K | baseline | — | general degradation for non-row/column orderings |
| T-XL | FMoW | baseline | Column-major best | smaller variation |

### Main Results — REOrder Improvements

| Model | Dataset | Best REOrder Gain | Permutation Source |
|-------|---------|------------------|--------------------|
| Mamba | IN-1K | +3.01% (±0.23) | Hilbert curve |
| Mamba | FMoW | +13.35% (±0.21) | Diagonal |
| T-XL | IN-1K | +1.50% (±0.21) | Hilbert curve |
| T-XL | FMoW | +1.10% (±0.21) | Column-major |
| Longformer | IN-1K/FMoW | ~0% | No significant gain |

### Ablation Study

| Configuration | Key Observation | Notes |
|---------------|----------------|-------|
| Random permutation vs. REOrder | REOrder significantly outperforms random | Rules out coincidental training dynamics |
| 100 vs. 300 epochs | Gains persist across training lengths | Not an artifact of insufficient training |
| With/without information-theoretic initialization | Initialization helps but is not required | Weak prior provides a reasonable starting point |
| Longformer | Negligible improvement | Approximation is closest to full attention; lowest permutation sensitivity |

### Key Findings

- Mamba is most sensitive to patch ordering; its fixed four-directional causal scan conflicts with non-standard permutations, yet REOrder successfully discovers better orderings through learning.
- Different datasets have different optimal permutations (ImageNet prefers column-major; FMoW prefers diagonal), confirming that no universal optimal permutation exists.
- During policy training, salient patches (e.g., keyboard keys) tend to be placed toward the end of the sequence, reflecting the center bias of the dataset.

## Highlights & Insights

- **Deep theoretical insight**: The widely accepted assumption that "patch order is irrelevant" is precisely characterized as holding only for full self-attention, and shown to fail for all efficient attention approximations.
- **Minimal overhead**: The PL policy requires only 196 parameters (for a 14×14 patch grid) and a single argsort operation at training time.
- **The 13.35% gain on FMoW is remarkable**: It suggests that non-natural images such as satellite imagery may offer larger gains from permutation optimization.

## Limitations & Future Work

- No significant improvement is observed for Longformer, indicating that the method is primarily effective for architectures with strong permutation sensitivity.
- The current approach uses a globally shared permutation (one per batch); sample-adaptive permutations remain unexplored.
- Validation is limited to classification tasks; effectiveness on dense prediction tasks such as detection and segmentation is unknown.
- Mamba's four-directional scan directions are fixed; ideally, scan directions should be jointly optimized with the permutation.

## Related Work & Insights

- Compared to ARM (a Mamba vision variant) that clusters patches within rows at small scale, REOrder searches over a substantially larger permutation space and achieves superior results.
- This work has implications for any setting that converts 2D structured data into 1D sequences, such as point clouds and graph-structured data.
- REOrder can be combined with efficiency methods such as token pruning/merging: permutation optimization followed by pruning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic study revealing and addressing the impact of patch ordering on long-sequence vision models
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across four models, two datasets, six orderings, and multiple ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical analysis, fluent exposition, and well-designed figures
- Value: ⭐⭐⭐⭐ Immediate value for the long-sequence vision model community; extension to more tasks and architectures is needed

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](../../ICCV2025/model_compression/vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/model_compression/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)

<!-- RELATED:END -->

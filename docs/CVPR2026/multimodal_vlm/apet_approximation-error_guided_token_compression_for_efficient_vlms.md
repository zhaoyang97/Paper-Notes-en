---
title: >-
  [Paper Note] ApET: Approximation-Error Guided Token Compression for Efficient VLMs
description: >-
  [CVPR 2026][Multimodal VLM][token compression] From an information-theoretic angle, this paper proposes a visual token importance measure based on linear-approximation reconstruction error. The method requires no attenti…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "token compression"
  - "visual token redundancy"
  - "approximation error"
  - "FlashAttention compatibility"
  - "VLM acceleration"
date: 2026-05-08
content_hash: e3f3bce07ab27c80
---

# ApET: Approximation-Error Guided Token Compression for Efficient VLMs

**Conference**: CVPR 2026  
**arXiv**: [2602.19870](https://arxiv.org/abs/2602.19870)  
**Code**: [MaQianKun0/ApET](https://github.com/MaQianKun0/ApET)  
**Area**: Multimodal VLM  
**Keywords**: token compression, visual token redundancy, approximation error, FlashAttention compatibility, VLM acceleration

## TL;DR

From an information-theoretic angle, this paper proposes a visual token importance measure based on linear-approximation reconstruction error. The method requires no attention weights, is naturally compatible with FlashAttention, and on LLaVA-1.5 retains 95.2% of the original performance after compressing away 88.9% of visual tokens.

## Background & Motivation

**Severe visual token redundancy in VLMs**: Mainstream VLMs (LLaVA, InternVL, etc.) encode an image into hundreds or even thousands of visual tokens before feeding them into the LLM. A 336×336 image yields 576 tokens in LLaVA-1.5, and high-resolution variants exceed 2000. Yet many studies report that the truly task-relevant "key tokens" account for only 10-20% — neighboring patches encode highly overlapping information.

**Compute bottleneck**: LLM self-attention is $O(n^2)$ in the sequence length $n$. Visual tokens dominate the sequence (typically >70%), so reducing them yields near-quadratic compute savings. The issue is even more acute in multi-frame video understanding, where the stacked sequence easily exceeds 10K tokens.

**Two fatal flaws of attention-weight-based methods**:

   **(a) Positional bias**: Mainstream methods (FastV, FitPrune, VisionZip, ...) rate token importance using attention weights from intermediate LLM layers — high-attention tokens are kept, low-attention ones are dropped. The authors find that LLM attention has a strong positional bias: tokens later in the sequence are attended more often due to the causal mask, and they themselves also collect higher attention scores, regardless of how much information they actually carry. Empirically, position alone predicts more than 60% of the attention ranking.

   **(b) Incompatibility with FlashAttention**: FlashAttention is the de-facto inference accelerator for LLMs; it tile-computes attention without materializing the full matrix. Pruning by attention weights, however, requires the full $n \times n$ attention matrix and clashes with FlashAttention's design. Disabling FlashAttention to use such methods may even slow inference down — a deal-breaker for deployment.

**Information-theoretic insight**: If a token can be linearly reconstructed from other tokens with small error, it carries little unique information and is "redundant"; if its reconstruction error is large, it contains unique information that no other token can substitute and is therefore "important". This intuition is concise and powerful, and importance estimation only needs the token feature vectors — no attention computation involved.

**Difference from existing attention-free methods**: A few methods avoid attention (e.g. ToMe merges via similarity, LOOK-M compresses KV cache), but their importance criteria remain heuristic. ApET grounds importance in approximation theory.

## Method

### Overall Architecture

ApET inserts a token compression module after a chosen LLM layer (e.g. layer 2). The flow has three steps:

```
Visual token sequence V ∈ R^{N×d}
    ↓
Step 1: Basis token selection → pick M basis tokens B ∈ R^{M×d}
    ↓
Step 2: Approximation error → compute linear reconstruction error e_i for every non-basis token
    ↓
Step 3: Token merging → keep the K tokens with the largest errors + the basis tokens,
                       merge the rest by weighted average into the nearest retained token
    ↓
Compressed sequence V' ∈ R^{K'×d}, fed into subsequent LLM layers
```

### Key Design 1: Basis Token Selection

Pick $M$ "basis tokens" out of the $N$ visual tokens to span the linear reconstruction basis. The paper explores three strategies:

- **FPS (Farthest Point Sampling)**: Greedy selection of tokens farthest from the chosen set, ensuring spatial diversity. Complexity $O(NM)$.
- **DPC (Density Peak Clustering)**: Picks tokens with high local density and far from any higher-density point, balancing density and diversity.
- **Random sampling**: Used as a baseline.

FPS turns out to be the most stable; DPC is slightly better on some tasks; random sampling still reaches >90% of the FPS performance — the method is not very sensitive to basis choice. FPS is the default.

Choice of $M$: $M = \lfloor \alpha \cdot N \rfloor$ with $\alpha = 0.1$ (i.e. 10% of tokens form the basis), trading off reconstruction quality against compute.

### Key Design 2: Approximation Error

For every non-basis token, the basis tokens are used to perform linear reconstruction; the reconstruction error then serves as the importance score.

Given a non-basis token $v_i$ and basis matrix $B \in \mathbb{R}^{M \times d}$, the optimal linear reconstruction coefficients are:

$$w_i^* = (B^\top B)^{-1} B^\top v_i$$

The reconstruction is $\hat{v}_i = B w_i^*$, and the approximation error is:

$$e_i = \|v_i - \hat{v}_i\|_2 = \|v_i - B(B^\top B)^{-1}B^\top v_i\|_2$$

i.e. the projection length of $v_i$ on the orthogonal complement of the column space of $B$. The larger the error, the more $v_i$'s information cannot be explained by the basis, and the more important it is.

**Compute optimization**: $(B^\top B)^{-1}$ only needs to be computed once ($M \times M$ matrix inverse with $M \ll N$, very fast); the projection is then computed batch-wise over all non-basis tokens. Total complexity is $O(M^2 d + NMd)$, which is only ~1ms at $N=576$, $M=58$, $d=4096$.

### Key Design 3: Token Merging Strategy

After scoring all tokens, compression proceeds:

1. **Keep**: the $M$ basis tokens + the top-$(K - M)$ non-basis tokens by error, $K$ in total.
2. **Merge**: the remaining $(N - K)$ tokens are not simply dropped; they are merged via weighted average into the nearest retained token.

Merging weights use cosine similarity:

$$v_j' = v_j + \sum_{i \in \text{merged\_to\_j}} \frac{\text{sim}(v_i, v_j)}{\sum_{k \in \text{merged\_to\_j}} \text{sim}(v_k, v_j)} \cdot v_i$$

Merging (vs. dropping) preserves part of the discarded tokens' information — a lossy but low-loss compression. Under aggressive compression (keeping <10% tokens), merging beats pure dropping by ~2-3pp.

### Compatibility with FlashAttention

A core advantage of ApET is its native compatibility with FlashAttention:

- The whole compression procedure only needs token feature vectors (hidden states), no attention matrix.
- Compression runs after layer 2 of the LLM (tokens have already gone through a few attention layers, so features are more informative).
- The compressed (much shorter) sequence then feeds the subsequent layers, where FlashAttention works normally and runs even faster thanks to the shorter sequence.

By contrast, methods like FastV must disable FlashAttention to read the attention matrix during compression and only re-enable it later — more complex engineering, with the loss of one accelerated layer.

## Key Experimental Results

### Main Results: LLaVA-1.5-7B Image Understanding (Various Compression Ratios)

| Method | Tokens kept | VQAv2 | GQA | TextVQA | POPE | MM-Vet | Avg. retention |
|---|---|---|---|---|---|---|---|
| Original | 576 | 78.5 | 62.0 | 58.2 | 85.9 | 31.1 | 100% |
| FastV | 192 | 76.8 | 60.5 | 55.1 | 83.2 | 28.7 | 96.3% |
| VisionZip | 192 | 77.1 | 61.0 | 56.3 | 84.5 | 29.4 | 97.8% |
| **ApET** | **192** | **77.3** | **61.2** | **56.8** | **84.7** | **29.8** | **98.0%** |
| FastV | 64 | 72.1 | 56.3 | 48.7 | 78.4 | 24.2 | 88.5% |
| VisionZip | 64 | 73.5 | 57.8 | 50.2 | 80.1 | 25.6 | 91.0% |
| **ApET** | **64** | **74.6** | **58.5** | **51.9** | **81.3** | **26.8** | **92.8%** |

At 192 tokens (33% retention), ApET keeps 98.0% of the performance, leading VisionZip by 0.2pp. Under aggressive compression to 64 tokens (11% retention), the lead grows to 1.8pp.

### Video Understanding (LLaVA-1.5 + Multi-frame)

| Method | Retention | MSVD-QA | MSRVTT-QA | ActivityNet-QA | Avg. retention |
|---|---|---|---|---|---|
| Original | 100% | 70.8 | 58.3 | 47.2 | 100% |
| FastV | 20% | 68.1 | 55.7 | 44.8 | 95.5% |
| ToMe | 20% | 69.0 | 56.2 | 45.3 | 96.5% |
| **ApET** | **20%** | **70.5** | **58.0** | **47.5** | **100.4%** |

On video tasks, ApET is especially striking — performance does not drop but actually rises (100.4%) after compressing to 20% tokens. The reason: removing redundant tokens lets attention focus on key frame content, reducing noise.

### Ablation Study

| Basis selection | Error metric | Merging | VQAv2 (64 tok) | GQA (64 tok) |
|---|---|---|---|---|
| FPS | L2 reconstruction error | weighted merge | **74.6** | **58.5** |
| Random | L2 reconstruction error | weighted merge | 73.2 | 57.1 |
| DPC | L2 reconstruction error | weighted merge | 74.3 | 58.2 |
| FPS | cosine distance | weighted merge | 73.8 | 57.6 |
| FPS | L2 reconstruction error | direct drop | 72.1 | 56.0 |
| FPS | attention weights | weighted merge | 71.5 | 55.8 |

Key takeaways: (1) FPS beats random and DPC by a small margin; (2) L2 reconstruction error clearly outperforms cosine distance and attention weights (+3.1 over attention); (3) weighted merging beats direct dropping by +2.5.

### Key Findings

- **Approximation error vs attention weights**: At equal compression ratio, the importance ranking from approximation error correlates with the attention-weight ranking only at Kendall's $\tau \approx 0.42$ — the two measure different aspects of "importance". Approximation error focuses on information uniqueness, attention focuses on contextual relevance.
- **Empirical positional bias**: For tokens in the top-10% by attention weight, 65% sit in the second half of the sequence; for tokens in the top-10% by approximation error, the positional distribution is roughly uniform — confirming the bias and ApET's immunity.
- **Compression layer placement**: After layer 2 is best; layer 0 (before LLM) is worse (tokens have not yet interacted, features are too raw); too deep a layer (e.g. layer 16) also hurts (deep computation has already been performed and would be wasted).
- **Real speedup with FlashAttention**: On A100, going from 576→64 tokens with FlashAttention gives a 2.1× actual inference speedup; FastV achieves only 1.6× because FA must be disabled at one layer.
- **Sensitivity to $M$**: As $\alpha$ varies from 0.05 to 0.2, VQAv2 fluctuates only ±0.5pp — the method is insensitive.

## Highlights & Insights

- **An elegant information-theoretic design**: Using linear-reconstruction error to score token importance is theoretically clean (cannot be reconstructed = unique information) and easy to implement.
- **Solves the FlashAttention compatibility pain point**: A frequent reason that token compression methods are abandoned in deployment; ApET's attention-free design solves it cleanly.
- **"De-noising" effect on video understanding**: Performance going up after compression is a pleasant surprise and points to a new direction for video VLM token management.
- **Negligible compute overhead**: The compression module itself costs ~1ms — negligible against the hundreds of ms of LLM inference.

## Limitations & Future Work

- Validation is limited to LLaVA-1.5 (7B/13B); larger models (LLaVA-OneVision-72B, InternVL2-76B, ...) are untested.
- The linear-approximation assumption treats token-space redundancy as linear; for highly nonlinear feature relations, some tokens' importance may be underestimated.
- Basis selection (FPS) only uses feature-space distance and ignores spatial-position priors (e.g. neighboring patches are more likely to be redundant).
- Compression layer is fixed at layer 2; layer-adaptive compression (more retention in shallow layers, more compression in deep ones) is not explored.
- Merging is uniform; semantic regions of different granularity (e.g. text regions may need more tokens) are not differentiated.
- Comparison with the latest dynamic-token methods (MatryoshkaKV, PyramidDrop, ...) is missing.

## Related Work & Insights

- **Token pruning**: FastV (attention-weight based), FitPrune (fitting pruning-ratio curves) are direct competitors; ApET surpasses them while staying attention-free.
- **Token merging**: ToMe (similarity-based merging of neighboring tokens) pioneered the merge route; ApET borrows the merging step but is driven by a different importance criterion.
- **VisionZip**: Also avoids attention and uses CLS-token similarity as a proxy, but is still indirectly tied to attention pre-training results.
- **KV cache compression**: LOOK-M, SnapKV, H2O, etc. compress along the KV-cache dimension; orthogonal to and composable with token compression.
- **Information bottleneck**: ApET's approximation-error measure has a theoretical link to the Information Bottleneck framework — keep the smallest sufficient statistic with maximum information.
- **Inspiration**: The approximation-error idea can be generalized to LLM text-token compression — text sequences are also redundant (repeated phrases, structured templates), and the same recipe may apply.

## Rating

- Novelty: ⭐⭐⭐⭐ — Approximation-error-based token importance is novel and theoretically grounded, although linear approximation itself is not a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple benchmarks including video, thorough ablations; missing larger-model validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — The motivation→method→experiment chain is crystal clear; the analysis of attention bias is convincing.
- Value: ⭐⭐⭐⭐ — FlashAttention compatibility is a deployment must-have, the method is highly practical, but the gain over the next baseline is moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](flashcache_frequency_kv_cache_compression.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[CVPR 2026\] Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)

</div>

<!-- RELATED:END -->

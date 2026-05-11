---
title: >-
  [Paper Note] Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][token compression] This paper proposes V2Drop, the first method to approach token importance from the perspective of token variation. By progressively dropping "lazy" vision tokens with minima…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "token compression"
  - "vision token pruning"
  - "LVLM acceleration"
  - "variation-aware"
  - "training-free inference acceleration"
  - "FlashAttention compatibility"
date: 2026-05-08
content_hash: ad7bf68557eb42ae
---

# Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2509.01552](https://arxiv.org/abs/2509.01552)
**Code**: [xuyang-liu16/V2Drop](https://github.com/xuyang-liu16/V2Drop)
**Area**: Multimodal VLM
**Keywords**: token compression, vision token pruning, LVLM acceleration, variation-aware, training-free inference acceleration, FlashAttention compatibility

## TL;DR

This paper proposes V2Drop, the first method to approach token importance from the perspective of token variation. By progressively dropping "lazy" vision tokens with minimal variation inside the LLM, V2Drop achieves training-free, position-bias-free, and efficient-operator-compatible LVLM inference acceleration, retaining 94.0% and 98.6% of original performance on image and video understanding tasks while reducing LLM generation latency by 31.5% and 74.2%, respectively.

## Background & Motivation

1. **Explosion in vision token count**: High-resolution image understanding and long-form video understanding have caused a dramatic increase in the number of vision tokens, imposing quadratic computational complexity on LVLM inference and severely limiting practical deployment efficiency.
2. **Position bias in attention-guided methods**: Existing inner-LLM token compression methods (e.g., FastV, SparseVLM, PDrop) rely on attention weights to assess token importance, systematically favoring tokens at the end of the sequence regardless of semantic content. This leads to the discarding of important tokens and the retention of irrelevant ones, exacerbating multimodal hallucination.
3. **Incompatibility with efficient operators**: Attention-guided methods require explicit computation of attention weights, conflicting with efficient operators such as FlashAttention. Peak memory usage can actually exceed that of uncompressed models (e.g., SparseVLM increases memory by 54.8% on MVBench), undermining the intended acceleration.
4. **Root Cause — external signals vs. intrinsic properties**: Using external signals such as attention to assess token importance is indirect and unreliable. Whether token importance can instead be judged directly from the token's own behavioral patterns within the model remains an unexplored fundamental question.
5. **Training overhead limits scalability**: Some token compression methods require additional training (training-aware), making plug-and-play application across different models difficult and limiting the generality and scalability of such approaches.
6. **Long-sequence bottleneck in video understanding**: VideoLLMs increasingly process long video sequences (e.g., hour-level frame-by-frame understanding). Existing methods either compress insufficiently or, due to position bias, over-retain tokens from later frames while neglecting critical information from earlier frames, necessitating position-agnostic efficient compression.

## Method

### Core Insight: Token Variation Reflects Importance

The authors conduct the first systematic analysis of how vision token representations vary across LLM layers (variation), revealing a key finding: **tokens with large inter-layer variation correspond to task-relevant regions, while tokens with small variation ("lazy tokens") correspond to task-irrelevant regions.** This pattern is task-agnostic—it holds across different questions and spatial positions—and naturally avoids position bias.

### Variation Measurement

Three metrics are used to measure token variation between adjacent layers:

- **L1 distance**: captures sparse changes
- **L2 distance** (default): captures overall magnitude changes, achieving the best performance–efficiency trade-off
- **Cosine similarity**: captures directional changes in representation

Formula: $\text{Var}(\mathbf{f}_i^{(l-1)}, \mathbf{f}_i^{(l)}) = \|\mathbf{f}_i^{(l)} - \mathbf{f}_i^{(l-1)}\|_2$

### Progressive Dropping Strategy (V2Drop)

Pruning is performed at three strategically selected positions in the LLM (shallow, middle, and deep layers). Each pruning layer executes three steps:

1. **Variation computation**: compute the L2 distance between each vision token and its representation in the previous layer
2. **Ranking and selection**: sort tokens by variation in descending order and retain the top-$K$ tokens with the highest variation
3. **Token reorganization**: restructure the selected tokens for use in subsequent layers

Progressive dropping schedule: $M \rightarrow K_a \rightarrow K_b \rightarrow K_c$, gradually reducing token count to avoid the information loss associated with one-shot dropping.

### Theoretical Guarantee

Through a first-order Taylor expansion, the authors prove the **Variation-Impact Theorem**: under smoothness assumptions, token variation $\|\Delta x_j^{(t)}\|$ is proportional to its impact on model output $\|\Delta f_j\|$, i.e., $\|\Delta f_j\| \approx \|J_j\|_{\text{op}} \cdot \|\Delta x_j^{(t)}\|$. Tokens with greater variation exert a larger influence on the final prediction, providing a theoretical foundation for variation-based pruning.

### Negligible Computational Overhead

The three pruning layers incur approximately 21M FLOPs in total, accounting for only 0.002% of the full forward pass. Throughput is nearly identical to random dropping (9.01 vs. 9.08 items/s).

## Key Experimental Results

### Image Understanding: Comparison at Different Compression Ratios on LLaVA-1.5-7B

| Method | Retained Tokens | GQA | SQA | TextVQA | POPE | MME | MMBench | Avg% |
|--------|----------------|-----|-----|---------|------|-----|---------|------|
| Original | 576 (100%) | 61.9 | 69.5 | 58.2 | 85.9 | 1862 | 64.6 | 100% |
| FastV | 192 (↓67%) | 52.7 | 67.3 | 52.5 | 64.8 | 1612 | 61.2 | 88.2% |
| SparseVLM | 192 (↓67%) | 57.6 | 69.1 | 56.1 | 83.6 | 1721 | 62.5 | 95.9% |
| PDrop | 192 (↓67%) | 57.1 | 68.8 | 56.1 | 82.3 | 1766 | 63.2 | 96.0% |
| **V2Drop** | **192 (↓67%)** | **58.5** | **69.3** | **55.6** | **85.1** | **1826** | **63.7** | **97.6%** |
| FastV | 128 (↓78%) | 49.6 | 60.2 | 50.6 | 59.6 | 1490 | 56.1 | 81.7% |
| **V2Drop** | **128 (↓78%)** | **56.3** | **68.8** | **53.8** | **80.9** | **1712** | **61.8** | **94.0%** |

V2Drop retains 97.6% of original performance at 67% compression, outperforming the second-best method (PDrop) by 1.6%; it still achieves 94.0% retention at 78% compression.

### Efficiency Comparison: Inference Latency and Memory (LLaVA-1.5-7B / LLaVA-OV-7B)

| Method | LLM Latency Reduction | Total Latency Reduction | Peak Memory Change | Throughput Gain | Performance Retention |
|--------|-----------------------|------------------------|-------------------|-----------------|----------------------|
| FastV (image) | ↓26.5% | ↓17.6% | ↑3.7% | 1.21× | 86.8% |
| SparseVLM (image) | ↓28.0% | ↓18.6% | **↑23.5%** | 1.23× | 92.9% |
| **V2Drop (image)** | **↓31.5%** | **↓20.8%** | **↓3.3%** | **1.26×** | **95.7%** |
| SparseVLM (video) | ↓34.4% | ↓20.0% | **↑54.8%** | 1.06× | 99.1% |
| **V2Drop (video)** | **↓74.2%** | **↓46.5%** | **↓7.8%** | **1.38×** | **99.1%** |

V2Drop is the only method that simultaneously reduces both latency and memory; SparseVLM achieves comparable performance but at the cost of a 54.8% increase in memory.

## Highlights & Insights

- **Strong originality of perspective**: This is the first work to examine token importance through the lens of inter-layer token variation, establishing a new compression paradigm distinct from attention-guided approaches.
- **Unified theory and experiment**: The Variation-Impact Theorem provides rigorous theoretical justification, complemented by comprehensive empirical validation (6 image benchmarks + 2 video benchmarks + 3 models).
- **Truly plug-and-play**: Requires no training, no architectural modification, and is compatible with FlashAttention; computational overhead is only 0.002%, making it highly engineering-friendly.
- **Fundamental resolution of position bias**: By relying on intrinsic token properties rather than external signals, the method inherently avoids the position bias that afflicts attention-based approaches.
- **Pronounced advantage in video scenarios**: Retaining only 25% of tokens in video understanding achieves 98.6% of original performance, far surpassing comparable methods, with particular advantages for long videos.

## Limitations & Future Work

- Pruning layer positions and token retention counts must be predefined; an adaptive mechanism for dynamically adjusting the compression ratio based on input content is lacking.
- The choice among the three variation metrics (L1/L2/cosine) relies on empirical selection; more sophisticated variation measures have not been explored.
- Validation is limited to 7B-scale models; applicability to 70B+ models and newer architectures (e.g., MoE) remains unknown.
- The theoretical analysis is based on first-order Taylor approximation and smoothness assumptions, which may not hold exactly in extreme layers of deep networks.
- The combination with pre-LLM compression methods has not been explored; complementary gains may exist.

## Related Work & Insights

- **vs. FastV (ECCV'24)**: FastV employs one-shot dropping guided by attention weights, exhibiting severe position bias (POPE: 59.6 vs. V2Drop's 80.9) and increased memory usage; V2Drop's progressive dropping with variation guidance comprehensively outperforms it.
- **vs. SparseVLM (ICML'25)**: SparseVLM also adopts progressive dropping but relies on attention weights and token merging, causing memory to spike by 54.8% in video scenarios; V2Drop achieves comparable performance while actually reducing memory.
- **vs. PDrop (CVPR'25)**: PDrop employs attention-guided progressive dropping; V2Drop outperforms it at all compression ratios and is additionally compatible with FlashAttention.
- **vs. ToMe (ICLR'23)**: ToMe uses a token merging strategy whose performance degrades sharply under aggressive compression (only 69.7% at 64 tokens); V2Drop maintains 86.9% under equivalent compression.
- **vs. Pre-LLM methods (e.g., LLaVA-PruMerge)**: Pre-LLM methods compress before the LLM and may discard contextual information acquired during LLM processing; V2Drop prunes within the LLM, leveraging inter-layer information for more precise compression.

## Rating

- Novelty: ⭐⭐⭐⭐ — The variation perspective is a genuinely novel entry point, though the core operation (L2 distance + Top-K) is itself relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers multiple models, benchmarks, compression ratios, efficiency analyses, visualizations, and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, problem definition is precise, theoretical derivations are rigorous, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Strong practical utility and meaningful inspiration for the community, though the simplicity of the method also implies limited headroom for further gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation-error_guided_token_compression_for_efficient_vlms.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][vision token compression] This work is the first to approach vision token compression from the perspective of inter-layer token variation. It identifies "lazy" vision tokens with small inter-layer variation as having negligible impact on model output, and proposes V2Drop, a progressive dropping scheme that eliminates low-variation tokens. V2Drop retains 94.0% of image understanding performance while reducing generation latency by 31.5%, and retains 98.6% of video understanding performance while reducing latency by 74.2%, with full compatibility with FlashAttention.
tags:
  - CVPR 2026
  - Multimodal VLM
  - vision token compression
  - LVLM inference acceleration
  - token variation
  - training-free acceleration
  - FlashAttention compatibility
date: 2026-05-08
content_hash: 318a683834e237d7
---

# V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2509.01552](https://arxiv.org/abs/2509.01552)
**Code**: [https://github.com/xuyang-liu16/V2Drop](https://github.com/xuyang-liu16/V2Drop)
**Area**: Multimodal VLM / Model Acceleration
**Keywords**: vision token compression, LVLM inference acceleration, token variation, training-free acceleration, FlashAttention compatibility

## TL;DR
This work is the first to approach vision token compression from the perspective of inter-layer token variation. It identifies "lazy" vision tokens with small inter-layer variation as having negligible impact on model output, and proposes V2Drop, a progressive dropping scheme that eliminates low-variation tokens. V2Drop retains 94.0% of image understanding performance while reducing generation latency by 31.5%, and retains 98.6% of video understanding performance while reducing latency by 74.2%, with full compatibility with FlashAttention.

## Background & Motivation

The explosive growth in the number of vision tokens when LVLMs process high-resolution images and long videos has made inference efficiency a critical bottleneck. Existing intra-LLM token compression methods (FastV, SparseVLM, PDrop) rely on attention weights to identify important tokens, but suffer from two fundamental flaws: (1) **positional bias**—attention scores systematically favor tokens at the end of the sequence regardless of their semantic content, causing informative tokens to be discarded and hallucinations to increase; (2) **incompatibility with efficient operators**—computing attention weight matrices conflicts with FlashAttention, causing peak memory to actually exceed that of uncompressed models (SparseVLM increases memory by 54.8% on MVBench).

## Core Problem

Can token importance be assessed directly from intrinsic behavioral patterns rather than relying on attention weights as an indirect signal, thereby enabling LVLM inference acceleration that is free of positional bias and compatible with efficient operators?

## Method

### Overall Architecture

V2Drop is a training-free, plug-and-play approach. During the prefilling stage of LLM inference, three layers (e.g., layers 3/17/22, corresponding to shallow/middle/deep) are selected for progressive vision token dropping. At each dropping layer, the L2 variation of each token between adjacent layers is computed; tokens are ranked by variation magnitude, with high-variation tokens retained and low-variation "lazy" tokens discarded. The sequence is then reassembled for forward propagation without recomputing attention weights.

### Key Designs

1. **Token variation as importance**: The core insight is that vision tokens exhibiting large inter-layer variation are being actively processed and enriched by the network ($\Delta x_j^{(t)} = x_j^{(t+1)} - x_j^{(t)}$ represents the effective update from Attention+FFN within the residual stream), while "lazy" tokens with small variation have negligible effect on the final output. The Variation-Impact Theorem provides a theoretical justification: $\|\Delta f_j\| \approx \|J_j\|_{op} \cdot \|\Delta x_j^{(t)}\|$, i.e., a token's impact on the output is proportional to its variation magnitude. A key advantage is that all three metrics (L1/L2/cosine) accurately localize semantically relevant regions without positional bias.

2. **Progressive three-stage dropping**: Rather than dropping tokens all at once, V2Drop drops tokens progressively at three layers, e.g., $576 \to 288 \to 192 \to \text{target}$. The progressive strategy outperforms one-shot dropping by 109 points on MME and 8% on POPE. This is because shallow-layer dropping is guided by preliminary semantic representations, while deep-layer dropping is guided by more refined ones.

3. **Native compatibility with FlashAttention**: V2Drop only requires computing the L2 distance between adjacent-layer features ($3MD'$ FLOPs), with no need for explicit attention matrices, making it fully compatible with FlashAttention. The computational overhead is only 0.022% of a single attention layer and 0.002% of the full forward pass; measured throughput is nearly identical to random dropping (9.01 vs. 9.08 items/s).

### Loss & Training

V2Drop is entirely training-free and requires no fine-tuning. L2 distance is used as the default variation metric. For LLaVA-1.5-7B, dropping layers are set to layers 3/17/22; when retaining 192 tokens, the three stages drop 50%/70%/100% of vision tokens respectively.

## Key Experimental Results

| Model / Task | Retention Rate | V2Drop Performance | Best Baseline | Speedup | Memory |
|---|---|---|---|---|---|
| LLaVA-1.5-7B Image | 33.3% (192/576) | 97.6% | PDrop 96.0% | 1.26× | −3.3% |
| LLaVA-1.5-7B Image | 22.2% (128/576) | 94.0% | PDrop 93.6% | — | — |
| LLaVA-OV-7B Video | 25% | 98.6% | SparseVLM 98.4% | 1.38× | −7.8% |
| LLaVA-OV-7B Video | 15% | 93.9% | SparseVLM 92.1% | — | — |
| Qwen2-VL-7B Image | 33.3% | 96.0% | DART 95.5% | — | — |
| Qwen2-VL-7B Video | 20% | 93.3% | DART 90.5% | — | — |

Key efficiency comparison (LLaVA-OV-7B video): V2Drop memory 16,298 MB vs. SparseVLM 27,378 MB (SparseVLM increases memory by 54.8%); throughput 0.72 vs. 0.55 items/s.

### Ablation Study

- **Variation metric**: L2 > L1 ≈ Cosine > FastV (attention); all three variation metrics consistently outperform attention-guided selection.
- **Progressive dropping is critical**: MME score 1826 (progressive) vs. 1717 (one-shot); POPE 85.1% vs. 77.1%.
- **Layer selection is robust**: Six different layer combinations yield performance in the range of 96.0%–97.6%, indicating low sensitivity to specific layer choice.
- **Positional bias eliminated**: Visualizations show that tokens retained by V2Drop are spatially uniform, whereas FastV/SparseVLM strongly favor tokens at the end of the sequence.
- **Advantage on long videos**: V2Drop shows larger gains on VideoMME (Long), as attention-based methods' preference for later-frame tokens becomes more severe in longer videos.

## Highlights & Insights

- **Paradigm shift**: "Look at variation, not attention" is a concise yet powerful perspective shift with rigorous theoretical backing.
- Full FlashAttention compatibility is the most practically significant advantage for deployment—existing attention-guided "acceleration" methods may actually increase memory usage.
- The advantage on video understanding is particularly notable (98.6% performance, 74.2% latency reduction), as video sequences contain far more tokens and positional bias is more severe.
- The Variation-Impact Theorem rigorously links variation magnitude to output impact as an a priori derivation rather than a post-hoc explanation.
- The codebase is open-source and the implementation is minimal (a single L2 distance computation followed by TopK selection).

## Limitations & Future Work

- The layer selection and compression ratio schedule for the three-stage dropping still require manual configuration; adaptive strategies may yield further improvements.
- The theoretical analysis assumes a lower bound on the Jacobian norm, providing insufficient characterization for tokens near zero.
- Experiments are limited to 7B-scale models; effectiveness on larger models (70B+) remains to be verified.
- Variation is assessed using only adjacent layers; cross-layer variation patterns may provide richer signals.
- Prefilling-stage acceleration yields limited benefit for short-prompt scenarios; the primary use case is high-resolution images and long videos.

## Related Work & Insights

- **vs. FastV**: FastV performs attention-based dropping, which suffers from positional bias and is incompatible with FlashAttention. At 192 tokens, V2Drop achieves 97.6% vs. FastV's 88.2%—a substantial performance gap.
- **vs. SparseVLM**: SparseVLM uses attention-ranked sorting with token recycling and achieves competitive performance, but increases peak memory by 23.5%–54.8%. V2Drop achieves comparable or better performance while actually reducing memory.
- **vs. PDrop**: PDrop, which uses progressive attention-based dropping, is the closest competitor. V2Drop outperforms it by +1.6% on images (97.6% vs. 96.0%) and +2.6% on video (98.6% vs. 96.0%).
- **vs. DyCoke**: DyCoke is designed specifically for video but achieves only 87.1% on LLaVA-OV at 25% retention, compared to V2Drop's 98.6%.

The "variation as importance" principle is generalizable to NLP (text token pruning) and multimodal fusion (cross-modal token alignment). The diagnosis of positional bias serves as a cautionary signal for all methods that use attention as an importance proxy. V2Drop is complementary to GKD (VFM distillation): GKD addresses model compression while V2Drop addresses inference acceleration, and the two can be combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Highly original perspective; the first to adopt token variation for token compression, with rigorous theoretical grounding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple models (LLaVA/Qwen2-VL) × multiple tasks (image/video) × multiple baselines × comprehensive ablations, with detailed efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is clear, theory and experiments are tightly integrated, and visualizations are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — Highly practical: training-free, FlashAttention-compatible, open-source, and ready for direct deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)
- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token-aware_adaptive_error_reconstruction_with_mixture_of_experts_.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)

</div>

<!-- RELATED:END -->

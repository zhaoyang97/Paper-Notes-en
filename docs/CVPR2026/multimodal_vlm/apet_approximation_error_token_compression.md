---
title: >-
  [Paper Note] ApET: Approximation-Error Guided Token Compression for Efficient VLMs
description: >-
  [CVPR 2026][Multimodal VLM][Approximation error] Grounded in information theory, ApET reconstructs each visual token via linear approximation and measures its informativeness by reconstruction error (larger error = more information = should be retained). The proposed framework is entirely independent of attention weights, achieves 95.2% accuracy retention at 88.9% compression on LLaVA-1.5-7B, even surpasses the baseline at 100.4% on video tasks, and is fully compatible with FlashAttention.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Approximation error
  - information theory
  - linear approximation
  - FlashAttention compatibility
  - attention-free token compression
date: 2026-05-08
content_hash: bf26b38c51b4af36
---

# ApET: Approximation-Error Guided Token Compression for Efficient VLMs

**Conference**: CVPR 2026  
**arXiv**: [2602.19870](https://arxiv.org/abs/2602.19870)  
**Code**: [https://github.com/MaQianKun0/ApET](https://github.com/MaQianKun0/ApET)  
**Area**: Multimodal VLM / Model Acceleration / Token Compression  
**Keywords**: Approximation error, information theory, linear approximation, FlashAttention compatibility, attention-free token compression  

## TL;DR
Grounded in information theory, ApET reconstructs each visual token via linear approximation and measures its informativeness by reconstruction error (larger error = more information = should be retained). The proposed framework is entirely independent of attention weights, achieves 95.2% accuracy retention at 88.9% compression on LLaVA-1.5-7B, even surpasses the baseline at 100.4% on video tasks, and is fully compatible with FlashAttention.

## Background & Motivation
Existing VLM token compression methods (SparseVLM, PyramidDrop, etc.) rely on attention weights to assess token importance, which suffer from two key problems: (1) **Positional bias**—LLM attention systematically assigns higher weights to visual tokens closer to the text (i.e., near the end of the sequence), regardless of actual informativeness; (2) **Incompatibility with FlashAttention**—FlashAttention does not expose the attention matrix, so computing attention weights separately introduces additional overhead. Experiments confirm that on Qwen2.5-VL, attention-guided compression methods are actually slower than the FlashAttention baseline.

## Core Problem
Can attention signals be abandoned entirely, replacing them with the intrinsic informativeness (reconstructibility) of tokens as the compression criterion, while maintaining full compatibility with FlashAttention?

## Method

### Overall Architecture
ApET can be inserted at two positions: after the visual encoder output and at an intermediate LLM layer (e.g., layer 16). At each position, three steps are executed: (1) **Token selection**: sample $M$ basis tokens from all visual tokens using FPS; (2) **Approximation error computation**: linearly approximate each token using the basis tokens $v' \approx \sum \alpha_i b_i$ and compute reconstruction error $\xi = \|v - v'\|_2$; (3) **Token merging**: rank tokens by error, retain high-error tokens, and merge low-error tokens into their closest retained token (average merging).

### Key Designs
1. **Information-theoretic foundation**: Starting from mutual information maximization, $\max_S I(V;S) = H(V) - H(V|S)$. Since $H(V)$ is fixed, the goal is to minimize $H(V|S)$. Shannon's theorem provides a lower bound: $\frac{1}{2\pi e}\exp(\frac{2H(V|S)}{d}) \leq \xi$, meaning minimizing reconstruction MSE is equivalent to minimizing conditional entropy. Therefore, **tokens with larger reconstruction error contain more unique information that cannot be expressed by the subset**, and should be retained.

2. **Linear approximation as a surrogate reconstruction model**: No additional reconstruction network needs to be trained. The linear system $V \approx BA$ is solved directly ($B$ being the basis token set and $A$ the coefficient matrix). Computational overhead is minimal: only $M=10$ basis tokens are needed, with $M \ll N$ (576). FPS sampling ensures diversity among basis tokens; DPC is also applicable but more computationally expensive.

3. **Token merging strategy**: Low-error tokens are not discarded outright but are merged into the most similar high-error token (average merging), reducing information loss. Basis tokens are automatically retained in the preserved set to ensure the approximation basis is not lost.

### Loss & Training
Completely training-free. Two compression stages are applied at the visual encoder output and at LLM layer 16 (LLaVA series) or layer 14 (Qwen2.5-VL). $M=10$ is used as the default value (performance varies by less than 2% for $M \in [5, 20]$). Code is open-sourced.

## Key Experimental Results
**LLaVA-1.5-7B (average over 9 benchmarks)**:

| Retained Tokens | ApET | VisionZip | PDrop | SparseVLM | FastV |
|--------|------|------|----------|------|------|
| 192 (33%) | **98.0%** | 97.8% | 97.2% | 96.1% | 90.4% |
| 128 (22%) | **97.1%** | 96.2% | 96.2% | 93.7% | 85.4% |
| 64 (11%) | **95.2%** | 92.7% | 86.6% | 87.2% | 76.7% |

**Video-LLaVA (256 tokens, 87.5%↓)**: ApET **100.7%** surpasses baseline vs. VisionZip 94.4% vs. FastV 83.9%

**Qwen2.5-VL-7B (20% retention rate)**: ApET 93.3% vs. PDrop 90.3% vs. SparseVLM 89.8%

**Efficiency (LLaVA-1.5-7B, 11.1% retention)**: 1.46× total inference speedup, 1.38× prefill speedup

### Ablation Study
- **FPS is the optimal sampling strategy**: FPS ≈ DPC >> Random (the fact that Random still works demonstrates the validity of approximation error itself)
- **Results are insensitive to $M$**: POPE varies by less than 2% for $M \in [5, 20]$; $M=10$ is optimal
- **Key advantage at extreme compression**: At 64 tokens (11%), ApET outperforms VisionZip by 2.5% and PDrop by 8.6%
- **Strong advantage on video tasks**: Attributed to the elimination of positional bias in attention (which is more severe in long video sequences)
- **Other methods slow down on Qwen2.5-VL**: Attention-based methods require recomputing weights and are slower than the baseline, whereas ApET achieves a 1.19× speedup

## Highlights & Insights
- **Unique information-theoretic perspective**—the derivation chain from Shannon conditional entropy → reconstruction MSE → approximation error is clear and elegant
- Complete independence from attention = full FlashAttention compatibility = genuinely practical on modern VLMs
- Extreme simplicity: only 10 basis tokens for linear approximation + L2 error + FPS sampling; the entire method can be implemented in a few lines of code
- Surpassing the baseline on video understanding (100.7%) further validates the hypothesis that "redundant tokens are harmful"—a denoising effect
- Forms a complementary "triangle of token importance estimation" with V2Drop (variation perspective) and GACD (gradient perspective)

## Limitations & Future Work
- Linear approximation may be insufficient to capture nonlinear feature relationships—stronger approximation methods could yield more accurate estimates
- FPS sampling introduces $O(NM)$ additional computation (though the practical overhead is negligible given small $M$)
- Post-compression token merging uses simple averaging—weighted average or attention-based merging may perform better
- No direct comparison with variation-based methods such as V2Drop—the complementarity of the two signals (approximation error vs. inter-layer variation) warrants further exploration
- Applied at inference only (not during training)—a unified training-and-inference approach similar to DUET-VLM could further improve performance

## Related Work & Insights
- **vs. V2Drop (CVPR'26)**: V2Drop uses inter-layer variation (also attention-free and FlashAttention-compatible); ApET uses approximation error. Both share a similar spirit but differ in signal source—V2Drop measures "how much a token changes across layers," while ApET measures "how well a token can be represented by others"
- **vs. VisionZip (CVPR'25)**: VisionZip selects dominant tokens via CLS attention + global merging; ApET uses approximation error + FPS + local merging, achieving 95.2% vs. 92.7% at 64 tokens
- **vs. DUET-VLM (CVPR'26)**: DUET-VLM employs a two-stage approach (visual-side clustering + language-side attention pruning); ApET applies a single principle (approximation error) at two locations. DUET-VLM is trainable, while ApET is purely inference-time
- **vs. GACD (CVPR'26)**: GACD uses gradients to estimate token contribution for hallucination mitigation; ApET uses approximation error for efficiency compression—different emphases, but both avoid direct attention dependency

## Insights & Connections
- The notions of "linearly reconstructible by other tokens = redundant" (ApET) and "small inter-layer variation = unimportant" (V2Drop) may be **unified into a single framework**: approximation error measures the intrinsic informativeness of a token (static), while inter-layer variation measures the degree to which the network utilizes a token (dynamic); their product may constitute an optimal token importance signal
- The linear approximation idea can be extended to **KV cache compression**—using a small number of basis KVs to linearly approximate the remaining KVs and selecting which to retain based on approximation error

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The information-theoretic + linear approximation error framework is entirely original and orthogonal to all attention-based methods
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 models (LLaVA / LLaVA-NeXT / Video-LLaVA / Qwen2.5-VL), 9+ benchmarks, efficiency analysis, and extensive ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivation is clear; the motivation → theory → method → experiment logic chain is seamless
- Value: ⭐⭐⭐⭐⭐ Novel information-theoretic perspective + FlashAttention compatibility + open-source code = substantial impact on the VLM compression field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](flashcache_frequency_kv_cache_compression.md)
- [\[CVPR 2026\] V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](v2drop_variation_aware_token_dropping.md)
- [\[CVPR 2026\] Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)
- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)

</div>

<!-- RELATED:END -->

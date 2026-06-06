---
title: >-
  [Paper Note] Autoregressive Image Generation with Randomized Parallel Decoding
description: >-
  [ICLR 2026][Image Generation][autoregressive image generation] This paper proposes ARPG, a visual autoregressive model built upon a "guided decoding" framework that decouples positional guidance (query) from content repr…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "autoregressive image generation"
  - "random-order modeling"
  - "parallel decoding"
  - "KV cache"
  - "controllable generation"
date: 2026-05-08
content_hash: a9646fd4e368bc71
---

# Autoregressive Image Generation with Randomized Parallel Decoding

**Conference**: ICLR 2026
**arXiv**: [2503.10568](https://arxiv.org/abs/2503.10568)  
**Code**: [https://github.com/hp-l33/ARPG](https://github.com/hp-l33/ARPG)  
**Area**: Image Generation
**Keywords**: autoregressive image generation, random-order modeling, parallel decoding, KV cache, controllable generation

## TL;DR
This paper proposes ARPG, a visual autoregressive model built upon a "guided decoding" framework that decouples positional guidance (query) from content representation (key-value), enabling fully randomized-order training and generation with efficient parallel decoding. On ImageNet-1K 256×256, ARPG achieves 1.94 FID in 64 steps with over 20× throughput improvement and over 75% memory reduction.

## Background & Motivation
Autoregressive (AR) models have achieved remarkable success in large language models, and this paradigm has been extended to visual generation (e.g., VQGAN, LlamaGen). However, applying next-token prediction to image generation faces two core challenges:

**Fixed-order constraint**: Images have a 2D spatial structure, but AR models require flattening them into a 1D sequence (e.g., raster scan order), making it difficult for the model to handle zero-shot generalization tasks that require non-causal dependencies (e.g., inpainting, outpainting).

**Inference inefficiency**: Token-by-token generation is highly inefficient at high resolutions, where a 256×256 image requires generating hundreds of tokens.

Existing alternatives each have shortcomings: MaskGIT achieves random-order generation via masked modeling but relies on bidirectional attention and cannot use KV caching; RandAR enables random ordering via positional instruction tokens but doubles the sequence length, incurring substantial computational and memory overhead.

**Core Idea**: Embed positional information of the prediction target as queries in the attention mechanism, fully decoupling content representation (KV) from positional guidance (Q), thereby supporting random-order modeling and parallel decoding while preserving causality.

## Method

### Overall Architecture
ARPG adopts a 2-Pass Decoder architecture. The first pass applies standard causal self-attention over known tokens to obtain contextualized representations (serving as global key-value pairs). The second pass applies cross-attention, using target-aware queries (position-embedded [MASK] tokens) to predict tokens at arbitrary positions. The input consists of a class label and an image token sequence; the output is the predicted token at the corresponding position.

### Key Designs
1. **Three Core Insights**:

    - **Insight 1**: Breaking order-specific constraints in AR models requires explicit positional guidance so that the model knows where the next token to predict is located.
    - **Insight 2**: In masked sequence modeling, queries corresponding to unmasked tokens receive no gradients from the loss function and thus play no role during training — meaning the queries can be entirely data-independent.
    - **Insight 3**: [MASK] tokens encode only positional information and contribute nothing to the contextual representation; moreover, they are harmful to causality — and should therefore be removed from the key-value pairs.

2. **Guided Decoding Framework**: Based on the above insights, ARPG redefines the probability distribution for permutation autoregressive modeling. Each query $q_{\tau_i}$ is obtained by applying 2D RoPE positional encoding to a data-independent [MASK] token, while the key-value pairs are composed entirely of data-dependent known tokens. Through causal cross-attention, each target-aware query independently attends to the contextual key-value pairs, guiding the model to predict the token at a specific position.

3. **Parallel Decoding**: Since all tokens to be predicted are mutually independent (their queries do not influence one another), ARPG naturally supports parallel decoding. Multiple queries can be processed simultaneously, sharing a single KV cache. Unlike conventional cross-attention, ARPG swaps the roles of input and condition — known tokens serve as KV, and target positions serve as Q — thereby avoiding attention conflicts among multiple generation targets.

4. **2-Pass Decoder Architecture**: The first-pass (self-attention decoder) processes input tokens to obtain global contextual representations; the second-pass (cross-attention decoder) uses guided decoding to predict target tokens. Experiments show that a symmetric configuration (e.g., 12+12 layers) achieves the best balance between efficiency and quality.

### Loss & Training
- Training uses standard teacher-forcing on randomly permuted sequences.
- Sequences within each batch are independently shuffled, with the class token placed at the start.
- RoPE frequencies are expanded along the batch dimension and shuffled accordingly to maintain alignment.
- AdamW optimizer ($\beta_1=0.99$, $\beta_2=0.95$), initial learning rate 1e-4 per 256 batch size.
- 400 epochs total training, with 100 epochs warmup followed by cosine scheduling to 1e-5.
- Classifier-free guidance (CFG) class embedding dropout rate of 0.1.
- LlamaGen tokenizer (16× downsampling, codebook size 16384).

## Key Experimental Results

### Main Results

| Model | Params | Steps | Throughput | Memory | FID↓ | IS↑ |
|-------|--------|-------|------------|--------|------|-----|
| LlamaGen-XXL | 1.4B | 576 | 1.58 it/s | 26.22 GB | 2.62 | 244.1 |
| VAR-d24 | 1.0B | 10 | 48.90 it/s | 22.43 GB | 2.09 | 312.9 |
| RandAR-XXL | 1.4B | 88 | 10.46 it/s | 21.77 GB | 2.15 | 322.0 |
| RAR-XL | 955M | 256 | 8.00 it/s | 10.55 GB | 1.50 | 306.9 |
| **ARPG-L** | **320M** | **64** | **62.12 it/s** | **2.43 GB** | **2.44** | **287.1** |
| **ARPG-XL** | **719M** | **64** | **35.89 it/s** | **4.48 GB** | **2.10** | **331.0** |
| **ARPG-XXL** | **1.3B** | **64** | **25.39 it/s** | **7.31 GB** | **1.94** | **339.7** |

### Ablation Study

| Configuration | Steps | Throughput | Memory | FID |
|---------------|-------|------------|--------|-----|
| ARPG-L (12+12) baseline | 64 | 62.12 it/s | 2.43 GB | 2.44 |
| Fewer Guided (18+6) | 64 | 50.72 it/s | 3.19 GB | 3.82 |
| More Guided (6+18) | 64 | 66.11 it/s | 1.67 GB | 3.51 |
| w/o Guided (24+0) | 256 | 11.70 it/s | 4.96 GB | 90 |
| Guided Only (0+24) | 64 | 72.26 it/s | 0.91 GB | 4.57 |
| w/o Shared KV | 64 | 48.02 it/s | 3.83 GB | 2.37 |
| Random order | 64 | 62.12 it/s | 2.43 GB | 2.44 |
| Raster order | 256 | - | - | 2.49 |

### Key Findings
- ARPG-XXL achieves 1.94 FID within 64 steps, with over 20× higher throughput than LlamaGen.
- Compared to VAR at similar throughput, ARPG reduces memory consumption by over 75% (7.31 GB vs. 22.43 GB).
- Reducing sampling steps (e.g., from 64 to 32) does not significantly degrade quality (ARPG-XXL: FID=2.08 at 32 steps vs. FID=1.94 at 64 steps).
- Random-order generation, despite the increased modeling difficulty ($n!$ possible permutations), outperforms fixed-order generation.
- Removing the guided decoder degrades the model to a standard AR model (FID spikes to 90), completely losing random-order generation capability.

## Highlights & Insights
- **Theoretical clarity**: Starting from a comparison between masked and autoregressive modeling, the method is derived through three rigorous insights, forming a complete logical chain.
- **Efficiency and quality**: The approach substantially improves inference efficiency while maintaining competitive generation quality, which is highly valuable for practical deployment.
- **Zero-shot generalization**: Random-order modeling enables the model to naturally support inpainting, outpainting, and resolution extrapolation without additional training.
- **Controllable generation**: Simply replacing [MASK] queries with condition tokens (e.g., Canny edges, depth maps) enables controllable generation, achieving state-of-the-art results on ControlVAR and ControlAR.
- **Minimal design**: The method does not rely on additional techniques such as QK normalization, AdaLN, or linear attention.

## Limitations & Future Work
- Due to computational constraints, the method has not been extended to text-to-image generation.
- The 512×512 resolution experiment involves only 50 epochs of fine-tuning rather than training from scratch, leaving high-resolution performance insufficiently validated.
- The 2-pass decoder introduces additional architectural complexity, though the authors partially mitigate the overhead through shared KV caching.
- Random-order training may require more training epochs to achieve the same convergence quality.
- Compared to diffusion models, the FID scores remain behind the very top tier (e.g., DiT-XL/2 achieves a strong 2.27 FID).

## Related Work & Insights
- **Causal sequence modeling**: Raster-order AR models such as VQGAN and LlamaGen, whose efficiency is limited by token-by-token generation.
- **Masked sequence modeling**: The MaskGIT family achieves parallel generation via bidirectional attention but cannot utilize KV caching.
- **RandAR**: Achieves random ordering via positional instruction tokens, but doubling the sequence length introduces significant overhead.
- **RAR**: Specifies the next token position via target-aware positional embeddings, yet still performs best under raster order.
- **Insight**: Redefining the roles of Q, K, and V in the attention mechanism — where Q encodes position and KV encodes content — is an elegant design principle that may inspire other sequence modeling tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)
- [\[ICCV 2025\] Randomized Autoregressive Visual Generation](../../ICCV2025/image_generation/randomized_autoregressive_visual_generation.md)
- [\[ICLR 2026\] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)
- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](../../ICCV2025/image_generation/grouped_speculative_decoding_for_autoregressive_image_generation.md)

</div>

<!-- RELATED:END -->

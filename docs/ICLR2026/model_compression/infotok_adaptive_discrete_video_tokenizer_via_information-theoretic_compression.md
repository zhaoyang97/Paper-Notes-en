---
title: >-
  [Paper Note] InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression
description: >-
  [ICLR 2026][Model Compression][video tokenizer] InfoTok introduces Shannon’s source coding theorem into discrete video tokenization, utilizing ELBO to estimate the information volume of each video for adaptive token allocation. It proves that fixed or data-independent tokenizers are biased and suboptimal in representation length. InfoTok reduces token usage by appro
tags:
  - ICLR 2026
  - Model Compression
  - video tokenizer
  - adaptive compression
  - information theory
  - ELBO
  - discrete tokenization
  - source coding theorem
date: 2026-05-08
content_hash: b2e10a1feb861455
---
# InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression

**Conference**: ICLR 2026 (Oral)  
**OpenReview**: [https://openreview.net/forum?id=JEYWpFGzvn](https://openreview.net/forum?id=JEYWpFGzvn)  
**Code**: [https://research.nvidia.com/labs/dir/infotok/](https://research.nvidia.com/labs/dir/infotok/)  
**Area**: Video discrete tokenizer / adaptive compression  
**Keywords**: video tokenizer, adaptive compression, information theory, ELBO, discrete tokenization, source coding theorem  

## TL;DR
InfoTok introduces Shannon’s source coding theorem into discrete video tokenization, utilizing ELBO to estimate the information volume of each video for adaptive token allocation. It proves that fixed or data-independent tokenizers are biased and suboptimal in representation length. InfoTok reduces token usage by approximately 20%~50% while maintaining the same reconstruction quality, achieving a 2.3× higher compression rate than heuristic adaptive methods (ElasticTok) with 11× less inference overhead.

## Background & Motivation
**Background**: Encoding videos into discrete tokens is a critical step for unifying Multimodal Large Language Models (MLLMs) and interfacing with LLMs. A standard discrete tokenizer consists of an encoder + quantizer (VQ/FSQ/LFQ) + decoder, trained to minimize reconstruction error under token sequence constraints—essentially acting as a "compressor."

**Limitations of Prior Work**: Most tokenizers employ a **fixed compression rate**, partitioning tokens at a constant ratio of $c\cdot THW$ for any video. However, information density varies significantly across spatial content (scenes/objects) and temporal dynamics (motion speed/magnitude). Assigning the same number of tokens to a static dog video as to an intense cat fight video leads to redundancy in simple videos and information deficiency in complex ones. Existing "flexible tokenization" (ElasticTok) allows variable lengths but relies on **data-independent uniform sampling during training + trial-and-error binary search during inference**, which is slow and fails to utilize information content effectively.

**Key Challenge**: The difficulty of adaptive tokenization is not just enabling "variable length," but **allocating tokens based on information volume in a principled manner**. Since the token count $N_x$ is discrete and not directly differentiable, including the expected length in the loss function for end-to-end optimization is infeasible.

**Goal**: To answer "What is a theoretically ideal discrete video tokenizer and how should it be trained in a principled manner."

**Core Idea**: **Replace trial-and-error with information theory**. Shannon’s source coding theorem states that the optimal expected token length is proportional to the negative log-likelihood $-\log p(x)$. Thus, the neural-network-computable **ELBO is used as a proxy for $-\log p(x)$** to determine the token count for each video, bypassing the bottleneck of "directly optimizing discrete length."

## Method

### Overall Architecture
InfoTok does not retrain the tokenizer but "upgrades" any existing fixed-length tokenizer by adding two components: a **router** that estimates video information complexity via ELBO to determine the token count $N_x$, and a **transformer adaptive compressor $M_\psi$** that compresses fixed-length embeddings into a sequence of length $N_x$. After quantization, a symmetric decompressor restores the original length before reconstruction by the decoder. This entire process requires only one extra decoder forward pass (to calculate ELBO), and training remains end-to-end driven by reconstruction loss.

```mermaid
flowchart LR
    X[Video x] --> E[Encoder Eφ]
    E --> H[Fixed length embedding h]
    H --> R["Router rβ: Estimating Nx via ELBO"]
    H --> M["Adaptive Compressor Mψ<br/>(8-layer Transformer)"]
    R -->|Nx| M
    M --> Q[Quantizer FSQ]
    Q --> Z[Discrete tokens of length Nx]
    Z --> DQ[De-quantize]
    DQ --> MI["Decompressor Mψ⁻¹<br/>Restore original length"]
    MI --> D[Decoder Dθ]
    D --> XR[Reconstruction x̂]
```

### Key Designs

**1. Proving "suboptimality" as a theorem via Source Coding Theorem: Fixed/data-independent routers are inherently biased.** The paper first restates Shannon’s source coding theorem in the context of tokenizers (Theorem 2.1): for any tokenizer that reconstructs data perfectly, the expected length is lower-bounded by $\mathbb{E}_{x}[N_x] \ge H_C(\mathcal{D}) \triangleq \mathbb{E}_x[-\log_C p(x)]$, and an adaptive scheme can approach this entropy lower bound. Based on this, it proves (Theorem 2.2) that when the router follows a uniform distribution over $\{1,\dots,N\}$ (as in data-independent training like ElasticTok), there exists a data distribution where the optimal solution satisfies $\mathbb{E}[N_x] \ge \kappa H_C(\mathcal{D})$ for any constant $\kappa>1$—meaning the expected length can be **arbitrarily longer** than the optimal. The intuition is clear: a uniform router requires the model to reconstruct at all lengths without any incentive to shorten expected length, treats data with different likelihoods equally. This section upgrades "adaptive tokenization" from an empirical observation to a provable necessity.

**2. ELBO router: Using a computable lower bound as a proxy for $-\log p(x)$.** Theory indicates that the optimal length $N_x \propto -\log p(x)$, but the log-likelihood of video is not directly computable. InfoTok uses ELBO as a proxy: $\text{ELBO}(x)=\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}[q_\phi(z|x)\|p(z)]$. The router is defined as a deterministic allocation $r_\beta(N_x|x)=\delta\!\left(\beta\cdot\frac{\text{ELBO}(x)}{\mathbb{E}[\text{ELBO}(x)]}\right)$, where $\beta$ is the average compression factor. The paper further proves (Theorem 3.1) that as long as the loss is well-minimized, the inference expected length $\mathbb{E}[N_x]\le H_C(\mathcal{D})+\beta-\mathbb{E}[-\log p(x)]$, reaching optimality within an approximation error. In practice, only one encoder-decoder pass is needed to compute reconstruction error (the main term of negative ELBO), so **only one additional decoder forward pass** is required to determine length, completely eliminating the block-wise binary search of ElasticTok.

**3. INFOTOK-Flex: One model across multiple compression rates.** To avoid training a tokenizer for each target compression rate, several $\beta$ values are integrated into a single compressor. During training, $\beta$ is randomly sampled from a set $\mathcal{B}$ and used as a conditional input. This allows a single model during inference to automatically determine length for each video by specifying a target BPP16, covering a continuous spectrum of compression rates with performance parity to rate-specific models.

**4. Likelihood-weighted token selection: Dropping tokens with lowest information + 5% mask overhead.** After obtaining $N_x$, how should the embedding of length $N$ be compressed to $N_x$? InfoTok's compressor retains the $N_x$ tokens with the highest **per-token log-likelihood (also approximated by ELBO)** and discards the lowest. This likelihood term is already computed by the router, adding no extra forward passes. The end-to-end reconstruction loss naturally trains the compressor to move information from "to-be-discarded" tokens to the retained positions. To inform the decoder of retained positions, a mask is stored as part of the token sequence, incurring approximately 5% length overhead—a worthwhile trade-off for the gains achieved.

## Key Experimental Results

### Main Results
Comparing fixed-length and adaptive tokenizers on TokenBench and DAVIS (both 256×256), measured by BPP16 (bits-per-16-pixels):

| Method | BPP16↓ | TokenBench PSNR↑ | LPIPS↓ | FVD↓ | DAVIS PSNR↑ | FVD↓ |
|------|--------|------------------|--------|------|-------------|------|
| Cosmos-DV4x8x8 (Fixed) | 1.00 | 30.01 | 0.138 | 49 | 25.92 | 404 |
| ElasticTok (Adaptive) | 0.81 | 28.26 | 0.244 | 141 | 24.69 | 754 |
| **INFOTOK-Flex** | 0.81 | 29.86 | 0.148 | 54 | 25.69 | 441 |
| **INFOTOK** | 0.81 | 30.08 | 0.145 | 49 | 25.79 | 408 |
| ElasticTok | 0.56 | 27.34 | 0.276 | 194 | 23.76 | 930 |
| **INFOTOK-Flex** | 0.56 | 29.30 | 0.179 | 71 | 24.84 | 581 |
| **INFOTOK** | 0.56 | 29.27 | 0.176 | 70 | 24.52 | 540 |

InfoTok nearly matches the fixed-rate Cosmos-DV (PSNR 30.08 vs 30.01) at BPP16=0.81 while **saving ~20% of tokens**. Compared to ElasticTok at the same rate, FVD decreases by 40-60%, LPIPS by 25-40%, and PSNR increases by 1.0-2.0. Even InfoTok at BPP16=0.56 outperforms ElasticTok at BPP16=0.81.

### Ablation Study
**(a) Router vs. Brute-force Optimal Search**: Comparing the ELBO router with an "Optimal" upper bound (exhaustive search per video for optimal allocation). The results are nearly identical, validating the effectiveness of ELBO for length estimation.

| BPP16 | Method | TokenBench PSNR↑ | FVD↓ | DAVIS PSNR↑ | FVD↓ |
|-------|------|------------------|------|-------------|------|
| 0.81 | INFOTOK-Flex | 29.86 | 54 | 25.69 | 441 |
| 0.81 | Optimal (Upper Bound) | 29.92 | 54 | 25.79 | 431 |
| 0.56 | INFOTOK-Flex | 29.30 | 71 | 24.84 | 581 |
| 0.56 | Optimal (Upper Bound) | 29.39 | 74 | 24.93 | 601 |

**(b) Inference Efficiency**: ElasticTok requires binary searching each 4096-token block, leading to $\log_2(4096)-1=11$ extra network forward passes. InfoTok requires only 1 decoder forward pass for ELBO, **reducing additional NFEs by 11×**.

### Key Findings
- Adaptive allocation systematically saves 20%~50% of tokens for equivalent quality, and InfoTok shows almost no gap compared to exhaustive optimal search, proving ELBO as a strong proxy for $-\log p(x)$.
- InfoTok-Flex achieves performance parity with rate-specific models while being deployment-friendly.
- Compared to heuristic adaptive methods (ElasticTok), it achieves **2.3×** higher compression and **11×** lower inference overhead.

## Highlights & Insights
- **Upgrading adaptive concepts from empirical intuition to provable theorems**: Theorem 2.2 provides a counterexample where fixed/uniform routers are arbitrarily suboptimal, and Theorem 3.1 provides near-optimality guarantees for the ELBO router.
- **Cleverly bypassing non-differentiable length optimization**: Instead of hard-optimizing $N_x$, it leverages information theory to link $N_x$ to $-\log p(x)$ and uses existing ELBO signals to determine length at near-zero cost.
- **Plug-and-play**: The framework is built on top of any fixed-length tokenizer, reusing its encoder/decoder, allowing it to benefit from future improvements in tokenizers.
- **Dual-use information**: Per-token ELBO is used both for router length determination and compressor token selection, avoiding redundant computations.

## Limitations & Future Work
- **Lack of generative downstream coverage**: The paper focuses on reconstruction/compression. Adaptation of variable-length tokens for autoregressive generation remains an open problem.
- **Mask storage overhead**: Binary masks for retention/discarding occupy ~5% of length, which becomes more significant at high compression rates.
- **ELBO≈log-likelihood assumption**: Theoretical guarantees rely on ELBO being close to the true log-likelihood; approximation errors might amplify for poorly trained or out-of-distribution tokenizers.
- **Evaluation scope**: Primarily evaluated on 256×256 videos. Generalization to higher resolutions/longer sequences is discussed mostly in the appendix.

## Related Work & Insights
- **Fixed-length tokenizers**: VQ-VAE, MAGVIT2, OmniTokenizer, Cosmos Discrete Video Tokenizer—InfoTok treats these as reusable foundations.
- **Flexible/Adaptive tokenization**: ElasticTok (right-to-left random masking + binary search) is the direct baseline. InfoTok proves its training is biased and inference is inefficient, providing a principled alternative.
- **Information Theory and Compression**: Shannon’s source coding theorem is the theoretical cornerstone. Translating the classical conclusion "content-aware compression outperforms content-agnostic compression" to neural discrete tokenizers is the core contribution.
- **Inspiration**: The idea of using ELBO as an "information complexity meter" to drive resource allocation can be transferred to other "computing by information volume" scenarios such as adaptive patching, dynamic token pruning, or KV-cache budget allocation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Strictly grounding video discrete tokenization with source coding theorems and using ELBO to bypass discrete optimization is a rare "theory-guided design" (well-deserved Oral).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid multi-rate evaluation on two datasets, alignment with optimal search, and NFE efficiency; slightly limited by the lack of generative downstream validation and resolution range.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theorem-intuition-counterexample-algorithm flow is very clear.
- **Value**: ⭐⭐⭐⭐ Plug-and-play, saving 20%~50% tokens efficiently, with direct practical value for long-video modeling and unified multimodal tokenization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](../../CVPR2026/model_compression/planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)
- [\[ICLR 2026\] InfoScan: Information-Efficient Visual Scanning via Resource-Adaptive Walks](infoscan_information-efficient_visual_scanning_via_resource-adaptive_walks.md)
- [\[ICLR 2026\] COMI: Coarse-to-fine Context Compression via Marginal Information Gain](comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[ICLR 2026\] Adaptive Nonlinear Compression for Large Foundation Models](adaptive_nonlinear_compression_for_large_foundation_models.md)

</div>

<!-- RELATED:END -->

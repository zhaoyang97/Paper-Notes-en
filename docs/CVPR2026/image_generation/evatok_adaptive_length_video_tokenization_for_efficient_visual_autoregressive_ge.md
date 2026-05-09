---
title: >-
  [Paper Note] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation
description: >-
  [CVPR2026][Image Generation][video tokenizer] This paper proposes EVATok, a four-stage framework that defines optimal token allocation via a proxy reward, trains a lightweight router to predict the optimal token budget for each video segment, and achieves content-adaptive variable-length video tokenization. EVATok attains state-of-the-art generation quality on UCF-101 while saving at least 24.4% of tokens.
tags:
  - CVPR2026
  - Image Generation
  - video tokenizer
  - adaptive length
  - autoregressive generation
  - video quantization
  - content-adaptive
  - proxy reward
  - router
date: 2026-05-08
content_hash: 2258657ecb599928
---

# EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

**Conference**: CVPR2026
**arXiv**: [2603.12267](https://arxiv.org/abs/2603.12267)
**Code**: [Project Page](https://silentview.github.io/EVATok/)
**Area**: Image Generation / Video Generation
**Keywords**: video tokenizer, adaptive length, autoregressive generation, video quantization, content-adaptive, proxy reward, router

## TL;DR

This paper proposes EVATok, a four-stage framework that defines optimal token allocation via a proxy reward, trains a lightweight router to predict the optimal token budget for each video segment, and achieves content-adaptive variable-length video tokenization. EVATok attains state-of-the-art generation quality on UCF-101 while saving at least 24.4% of tokens.

## Background & Motivation

1. **AR video generation depends on token sequence length**: Autoregressive video generation models compress video into discrete token sequences, where sequence length directly governs the trade-off between reconstruction quality and downstream generation cost.
2. **Existing tokenizers use fixed lengths**: Conventional video tokenizers assign the same number of tokens to all videos and all temporal chunks, ignoring differences in content complexity—simple, static, or repetitive segments waste tokens, while dynamic or complex segments receive insufficient tokens.
3. **Spatiotemporal information density is non-uniform**: Information density in video varies not only across samples but also along the temporal dimension of the same video (e.g., static frames vs. fast-motion scenes).
4. **Existing adaptive methods have shortcomings**: ElasticTok uses heuristic threshold search that ignores global quality–cost balance; AdapTok uses mini-batch ILP, coupling single-sample decisions to batch composition and fixed average budget constraints.
5. **Lack of a definition and estimation method for optimal allocation**: Prior work neither formally defines what constitutes an "optimal token allocation" nor provides an efficient method to estimate it.
6. **Train–inference inconsistency**: Previous variable-length tokenizers enumerate all possible allocations during training but use only a few at inference, creating a training–inference gap that degrades performance.

## Method

### Overall Architecture: Four-Stage Pipeline

EVATok consists of four stages:

- **Stage 1 — Train a proxy tokenizer**: A Q-Former-style 1D tokenizer capable of reconstructing video under varying token allocations is trained to serve as a proxy for evaluating allocation quality.
- **Stage 2 — Dataset construction**: The proxy tokenizer is used to enumerate all candidate allocations, compute the proxy reward, select the optimal allocation for each video, and construct a (video, optimal allocation) training set.
- **Stage 3 — Router training**: A lightweight ViT-S-scale classifier (19.9M parameters) is trained on the constructed dataset to predict the optimal token allocation for a video in a single forward pass.
- **Stage 4 — Final adaptive tokenizer training**: An adaptive tokenizer is trained from scratch with token allocations determined by the router, eliminating the training–inference gap.

### Key Designs

**Proxy Reward**:
$$R_{\text{proxy}} = w_q \cdot Q(\mathcal{E}_{\text{proxy}}, x, a) - w_l \cdot L(a)$$
where $Q$ denotes reconstruction quality (normalized LPIPS), $L(a)$ is the normalized token length, and $w_q, w_l$ balance quality against cost. The optimal allocation is $a^* = \arg\max_{a \in A} R_{\text{proxy}}$.

**1D Variable-Length Tokenizer Architecture**:
- Input video undergoes spatio-temporal patchification → 3D embedding (8× spatial downsampling, 4× temporal downsampling).
- 1D queries are initialized according to allocation $a = (k_1, \ldots, k_T)$ from 2D-pooled features of corresponding temporal chunks.
- A Q-Former encoder with vector quantization produces discrete tokens; the decoder initializes 3D queries from the first 1D token for reconstruction.
- Tail-token-dropping is not used, avoiding additional computation and role ambiguity for tail queries.

**Router**: A ViT-like architecture that classifies each video into one of $m^T$ candidate allocations ($m=5$ levels × $T=4$ temporal chunks → 625 candidates), trained with cross-entropy loss.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vqgan}} + \lambda \mathcal{L}_{\text{align}} + \gamma \mathcal{L}_{\text{entropy}}$$

- $\mathcal{L}_{\text{vqgan}}$: L1 reconstruction + perceptual loss + adversarial loss + VQ codebook loss.
- $\mathcal{L}_{\text{align}}$: patch-wise cosine similarity alignment with pretrained V-JEPA2-L ($\lambda=0.7$).
- $\mathcal{L}_{\text{entropy}}$: entropy loss for codebook utilization ($\gamma=0.02$).
- In Stage 4, a frozen VideoMAE-B may optionally be introduced as a semantic discriminator to further improve reconstruction and generation quality.

## Key Experimental Results

### Main Results: System-Level Comparison (UCF-101)

| Method | Tok. Param | #rTokens | rFVD↓ | gFVD↓ | #gTokens |
|--------|-----------|----------|-------|-------|----------|
| LARP-L-Long | 173M | 1024 | 6.2 | 57 | 1024 |
| AdapTok | 195M | 1024 | 11 | 67 | 1024 |
| **EVATok** | **145M** | **774 (−24.4%)** | **4.6** | **48** | **756 (−26.2%)** |

- EVATok achieves state-of-the-art class-to-video generation on UCF-101 (gFVD=48) while saving 26.2% of generation tokens.
- Reconstruction rFVD improves from 6.2 (LARP) to 4.6, with fewer parameters (145M vs. 173M).

### WebVid Validation

| Setting | LPIPS↓ | rFVD↓ | #rTokens |
|---------|--------|-------|----------|
| Uniform (Final) | 0.1056 | 63 | 1024 |
| Router (Final) | 0.1068 | 33 | 721 (−29.6%) |
| Router + VideoMAE Disc. | 0.1144 | 9.2 | 721 (−29.6%) |

- The router-guided tokenizer achieves comparable LPIPS, reduces rFVD from 63 to 33, and saves 29.6% of tokens.

### Ablation Study

**Quality–Cost Curve Comparison**:
- Under equal token budgets, the max-proxy-reward strategy consistently outperforms fixed uniform allocation and heuristic threshold search.
- The router closely approximates the max-proxy-reward curve and generalizes to unseen datasets (UCF).

**Video Semantic Encoder Ablation**:
- Removing both the VideoMAE discriminator and V-JEPA2 alignment degrades gFVD from 98 to 230.
- Both V-JEPA2 representation alignment and the VideoMAE semantic discriminator are essential.

**Training–Inference Gap Ablation**:
- The final tokenizer consistently outperforms the proxy tokenizer under equivalent training iterations, validating the effectiveness of Stage 4 in eliminating the gap.

## Highlights & Insights

- **First formal definition of optimal video token allocation**: The proxy reward formalizes "optimal allocation" as an optimizable objective rather than a heuristic search target.
- **Efficient and generalizable router**: A lightweight router with only 19.9M parameters performs prediction in a single forward pass and generalizes well to unseen datasets.
- **First demonstration that adaptive-length AR generation outperforms fixed-length**: Downstream AR models trained on variable-length token sequences achieve superior generation quality while saving 27.7% of generation tokens.
- **Intuition-consistent allocation**: Segments with heavy motion or complex layouts receive more tokens; repetitive or static segments receive fewer, consistent with human intuition.
- **Systematic resolution of the training–inference gap**: Stage 4 trains the tokenizer from scratch using only router-predicted allocations, fundamentally eliminating the gap present in prior work.

## Limitations & Future Work

- Evaluation is currently limited to short videos at 16 frames and 128×128 resolution; generalization to longer, higher-resolution videos remains unknown.
- Router training relies on substantial precomputation from Stages 1–2 (enumerating 625 allocations across 100k videos), incurring significant upstream overhead.
- The token level hierarchy (32–512) and temporal chunk division ($T=4$) are manually designed, limiting the granularity of adaptation.
- Spatial adaptive allocation is not explored; adaptation is performed only along the temporal dimension.
- The codebook size comparison is not entirely fair (proxy uses 16384; final uses 8192).

## Related Work & Insights

| Method | Adaptive Strategy | Allocation Mechanism | Global Optimality Defined |
|--------|------------------|---------------------|--------------------------|
| ElasticTok | tail-token-dropping | threshold search | ✗ |
| AdapTok | tail-token-dropping | mini-batch ILP | ✗ |
| InfoTok | mask less important tokens | ELBO-based | ✗ |
| Dynamic VQ | region-level adaptation | Gumbel Softmax | ✗ |
| **EVATok** | **1D query initialization** | **proxy reward + router** | **✓** |

The key distinction of EVATok lies in starting from a formal definition of "optimality" (proxy reward) rather than heuristically searching for a "good enough" allocation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The proxy reward combined with the four-stage framework constitutes a systematic new approach to adaptive video tokenization.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset validation, quality–cost curves, and comprehensive ablations are provided, though resolution and video length are limited.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, framework diagrams are intuitive, and experimental presentation is logically coherent.
- Value: ⭐⭐⭐⭐ — First demonstration that adaptive-length AR video generation outperforms fixed-length generation, with practical implications for video generation efficiency.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](depthvar_depth_adaptive_var.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](../../ICCV2025/image_generation/efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[CVPR 2026\] PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](promo_promptable_outfitting_for_efficient_high-fidelity_virtual_try-on.md)
- [\[CVPR 2026\] Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction](physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)
- [\[CVPR 2026\] BiGain: Unified Token Compression for Joint Generation and Classification](bigain_unified_token_compression_for_joint_generation_and_classification.md)

<!-- RELATED:END -->
